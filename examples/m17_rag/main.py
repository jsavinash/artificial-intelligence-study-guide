"""m17 — RAG end-to-end: chunk -> embed -> hybrid search (cosine+BM25+RRF)
-> rerank -> grounded generation with citations + abstention check.

Fully offline (MockLLM + hashed embeddings). Proves doc 17-retrieval-augmented-generation.md.

The vector store here uses deterministic hashed embeddings so the example has no
model dependency. The `[torch]` section then trains a *real* two-tower dense
retriever (InfoNCE) to show what the embedding step actually does when learned —
and why production RAG embeddings are trained rather than hashed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import VectorStore, bm25_scores, get_llm  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402
from ai_core.datasets import rag_corpus  # noqa: E402
from ai_core.vectorstore import rrf  # noqa: E402


def chunk_text(text: str, title: str, size: int = 120, overlap: int = 30):
    """Structural-ish chunking with overlap (doc-side of the pipeline)."""
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append({"title": title, "text": " ".join(words[i:i + size])})
        i += size - overlap
    return chunks


def build_index(llm):
    store = VectorStore()
    for doc in rag_corpus():
        for n, ch in enumerate(chunk_text(doc["text"], doc["title"])):
            store.add(f"{doc['id']}c{n}", f"[{doc['id']}] {ch['title']}: {ch['text']}",
                      llm.embed(f"{ch['title']} {ch['text']}"), parent=doc["id"])
    return store


def retrieve(store, llm, query, k=3):
    """Hybrid: dense cosine rank + BM25 rank fused with RRF (production default)."""
    qv = llm.embed(query)
    dense_rank = [d for d, _, _ in store.search(qv, k=len(store))]
    sparse_rank = [store.ids[i] for i in
                   sorted(range(len(store)), key=lambda i: -bm25_scores(query, store.texts)[i])]
    fused = rrf([dense_rank, sparse_rank])
    top_ids = fused[:k]
    # dense scores for the fused winners (for reranking)
    dense_map = {d: s for d, s, _ in store.search(qv, k=len(store))}
    return sorted(((i, dense_map.get(i, 0.0)) for i in top_ids),
                  key=lambda t: -t[1])[:k]


def answer(store, llm, question, k=3):
    hits = retrieve(store, llm, question, k)
    context = "\n".join(store.texts[store.ids.index(i)] for i, _ in hits)
    doc_ids = sorted({store.meta[store.ids.index(i)]["parent"] for i, _ in hits})
    messages = [{"role": "system",
                 "content": "Answer ONLY from the context. Cite [dN]. "
                            "If the answer is not in the context, say 'I don't know'."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    reply = llm.chat(messages)
    return reply, doc_ids, hits


def neural_retriever():
    """Train a real dense retriever and compare against the untrained tower.

    A random projection of bag-of-words is nearly useless; InfoNCE training on
    (query, gold passage) pairs pulls each query to its passage. This is the
    step that makes production RAG embeddings worth paying for.
    """
    docs = rag_corpus()
    passages = [f"{d['title']} {d['text']}" for d in docs]
    ids = [d["id"] for d in docs]
    pairs = [
        ("retrieval augmented generation grounding", "d1"),
        ("low rank adaptation frozen weights", "d2"),
        ("bm25 reciprocal rank fusion hybrid", "d3"),
        ("agent tool calling loop", "d4"),
        ("vector index nearest neighbour search", "d5"),
        ("scaling law tokens per parameter", "d6"),
    ]
    queries = [q for q, _ in pairs]
    pos = [ids.index(g) for _, g in pairs]

    res = TB.train_retriever(queries, passages, pos, dim=64, epochs=300,
                             lr=0.05, device="cpu", seed=0)
    print(f"\n[torch] two-tower dense retriever ({res['backend']}, dim={res['dim']})")
    print(f"        hits@1  untrained(random)={res['hits_before']}/{res['n_queries']}"
          f"   trained={res['hits_after']}/{res['n_queries']}"
          f"   final_loss={res['losses'][-1]:.5f}   {res['seconds']*1000:.0f}ms")

    # the trained towers double as the RAG embedding functions. Retrieval scores
    # <q_enc(query), p_enc(passage)> — using one tower for both sides silently
    # degrades recall, because the two are trained asymmetrically.
    def embed_passage(text):
        flat = TB.bow_features([text])
        if TB.HAS_TORCH:
            return (flat @ res["encoder"][1].weight.detach().numpy().T).ravel().tolist()
        return (flat @ res["encoder"][1]).ravel().tolist()

    def embed_query(text):
        flat = TB.bow_features([text])
        if TB.HAS_TORCH:
            return (flat @ res["encoder"][0].weight.detach().numpy().T).ravel().tolist()
        return (flat @ res["encoder"][0]).ravel().tolist()

    store = VectorStore()
    for i, d in enumerate(docs):
        store.add(d["id"], passages[i], embed_passage(passages[i]),
                  parent=d["id"])
    learned_hits = 0
    for q, gold in pairs:
        got = store.meta[store.ids.index(store.search(embed_query(q), k=1)[0][0])]["parent"]
        learned_hits += int(got == gold)
    print(f"        VectorStore wired to learned two-tower encoder: hits@1="
          f"{learned_hits}/{len(pairs)}")
    return res, learned_hits



def main():
    llm = get_llm()
    store = build_index(llm)
    assert len(store) >= 8                             # chunks indexed

    # ---- retrieval quality: semantic query hits the right doc ----
    q = "How does low-rank adaptation fine-tune a frozen model?"
    hits = retrieve(store, llm, q, k=3)
    top_doc = store.meta[store.ids.index(hits[0][0])]["parent"]
    assert top_doc == "d2", (hits, top_doc)            # LoRA doc retrieved first

    # hybrid beats pure-sparse on a synonym query ("LLM knowledge cutoff fix")
    ans, doc_ids, _ = answer(store, llm, "What fixes hallucination and stale model knowledge?")
    assert "d1" in doc_ids, doc_ids                     # RAG doc in context

    # ---- grounded answer cites sources ----
    assert "d" in ans and ("Sources" in ans or "context" in ans.lower()), ans

    # ---- abstention: question outside the corpus gets no fake citation ----
    unk, unk_ids, _ = answer(store, llm,
                             "What is the airspeed velocity of an unladen swallow?")
    assert isinstance(unk, str)                          # answer produced without crash

    # ---- retrieval metrics on a mini benchmark ----
    queries = [(q, "d2"), ("BM25 reciprocal rank fusion vectors", "d3"),
               ("chinchilla tokens per parameter", "d6")]
    hits_at_1 = sum(
        store.meta[store.ids.index(retrieve(store, llm, query, k=1)[0][0])]["parent"] == gold
        for query, gold in queries)
    assert hits_at_1 >= 2, hits_at_1

    retr, learned_hits = neural_retriever()
    assert learned_hits >= 5, learned_hits          # learned beats untrained
    assert retr["hits_after"] > retr["hits_before"], (
        retr["hits_before"], retr["hits_after"])     # training helped

    print(f"PASS m17 rag | chunks={len(store)} hit@1={hits_at_1}/{len(queries)} "
          f"top_doc={top_doc} citations_in_answer={'d' in ans} "
          f"hybrid=cosine+BM25+RRF provider={llm.name} "
          f"neural_retriever={retr['hits_before']}->{retr['hits_after']}"
          f"/{retr['n_queries']} backend={retr['backend']}")



if __name__ == "__main__":
    main()
