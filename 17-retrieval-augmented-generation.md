# 17 — Retrieval-Augmented Generation (RAG)

> Back to [index](README.md) · Prev: [16 Prompt Engineering](16-prompt-engineering.md) · Next: [18 Agents](18-ai-agents-and-tool-use.md)

**RAG = retrieve relevant documents at query time → put them in the prompt → generate grounded answers.** Fixes knowledge cutoffs, hallucination, and private-data access **without** retraining ([14](14-large-language-models.md)).

## 1. Why RAG beats the alternatives (for knowledge)

| Approach | Freshness | Cost | Citations | Scale of corpora |
|---|---|---|---|---|
| Stuff everything in context | instant | huge, hits limits | yes | tiny |
| Fine-tune knowledge | slow, brittle | retrain | no | medium, forgets |
| **RAG** | instant (index updates) | index + retrieval | **yes** | millions of docs |

## 2. The RAG pipeline

```
ingest → chunk → embed → index  ‖  query → embed → search → rerank → build prompt → generate → cite
```

### Ingestion & chunking
- Parse: PDF (layout-aware parsers beat naive text extraction), HTML, Markdown, code, tables (structure-preserving chunks).
- **Chunk size:** 200–1000 tokens typical; overlap 10–20%; **split on structure** (headings, functions, paragraphs) before falling back to fixed windows.
- Attach **metadata** (title, date, source, permissions — security-aware retrieval is mandatory, [22](22-ai-safety-security-ethics.md)).
- Parent–child chunking: retrieve small chunks, return surrounding context.

### Embeddings (dense retrieval)
- Encoder maps text → dense vector where **cosine similarity ≈ semantic similarity** ([11](11-transformers-and-foundation-models.md)).
- Models: OpenAI `text-embedding-3`, Voyage, Cohere, open (bge, E5, GTE, Nomic). Dimensions 384–3072 (Matryoshka: truncate to shrink index).
- **Hybrid search:** dense (semantics) + **BM25 sparse** (exact keywords, IDs, rare terms) → fused with **RRF (reciprocal rank fusion)** — near-universal best practice.
- Domain mismatch is real: embed *your* queries/docs, evaluate retrieval on your data.

### Vector databases
FAISS (library), chromadb (local, easy), **pgvector** (Postgres — transactions + filters you already run), Qdrant, Weaviate, Pinecone/Weaviate Cloud, LanceDB, Elasticsearch/OpenSearch (hybrid native).
Choose by: scale, filtering requirements, hybrid support, ops burden, cost.

### Retrieval strategies
- **Semantic + keyword hybrid** (above).
- **Metadata filtering first** (date, department, ACL) then semantic search.
- **Query rewriting:** expand acronyms, fix typos, HyDE (hypothetical doc), multi-query synthesis.
- **Parent-document & multi-vector** (summary index retrieves, full text returned).
- **Step-back prompting** (retrieve on the general question too).
- **Graph RAG:** build entity graph + community summaries; excellent for "connect the dots" questions over corpora (higher ingest cost).
- **Agentic retrieval:** model issues multiple searches, reads, re-queries until satisfied ([18](18-ai-agents-and-tool-use.md)).

### Reranking (the cheapest big win)
First stage = high-recall top-20–100 (cheap); second stage = **cross-encoder reranker** (e.g. bge-reranker, Cohere Rerank) scores query×doc jointly → top-3–10 into context. Typically improves groundedness more than any prompt tweak.

### Generation with retrieved context
- Prompt: instructions + "answer ONLY from this context; cite [1]; say I don't know if absent" + numbered chunks + query.
- Keep chunks ordered/delimited; include titles/dates; watch context budget vs lost-in-the-middle ([16](16-prompt-engineering.md)).

## 3. Advanced RAG patterns

- **Query classification:** short factual → direct answer from one passage; analytical → multi-hop synthesis; greetings → no retrieval.
- **Multi-hop / iterative RAG:** retrieve → reason → new query → retrieve (HotpotQA-style).
- **Self-RAG / CRAG:** model grades retrieved docs for relevance; drops or corrects them; falls back to web search.
- **Table/figure-aware retrieval** and structured-knowledge fusion (RAG + SQL: generate query → execute → ground).
- **Contextual retrieval:** prepend doc summary to each chunk before embedding (big recall lift).
- **Cache:** semantic cache for repeated queries.

## 4. RAG evaluation (do not skip)

- **Retrieval metrics:** recall@k, MRR, nDCG against labeled (query → gold docs).
- **Generation metrics:** **faithfulness/groundedness** (is every claim in context?), answer relevance, context precision, citation correctness.
- **End-to-end:** exact match/F1 for short answers; **LLM-as-judge rubrics** (RAGAS, TruLens, DeepEval frameworks: faithfulness, answer relevancy, context recall/precision).
- Build a golden question set from real users incl. "unanswerable" questions — measure abstention.

## 5. When RAG is the wrong tool

- Needs *procedural* behavior/style change → fine-tune ([15](15-fine-tuning-and-peft.md)).
- Tiny corpus < context window → just inline it.
- Table math/aggregations → SQL tools, not vector search.
- Latency-critical one-word lookups → direct DB call beats LLM+RAG.

## Mastery Checklist

- [ ] Builds the full ingest→chunk→embed→index pipeline with metadata
- [ ] Explains hybrid search + RRF and why BM25 still matters
- [ ] Adds a reranker and measures recall@k / nDCG before & after
- [ ] Designs a grounded prompt with citations and an abstention path
- [ ] Chooses a vector DB for a given scale/filtering/ops constraint
- [ ] Runs a RAG eval suite (retrieval + faithfulness + answer quality)
- [ ] Knows Graph RAG, query rewriting, and agentic retrieval by name and use case
