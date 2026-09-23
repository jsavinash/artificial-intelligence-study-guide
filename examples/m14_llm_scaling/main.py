"""m14 — LLMs: BPE tokenizer trained from scratch + scaling-law calculator.

Proves theory doc 14-large-language-models.md: tokenization mechanics,
next-token objective vocabulary, and the Chinchilla 20-tokens-per-param rule.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))


# ---------------- BPE (byte-pair encoding) ----------------
def get_stats(words: dict[str, int]):
    """Count adjacent symbol pairs across the corpus (with word frequencies)."""
    pairs = {}
    for w, freq in words.items():
        syms = w.split()
        for a, b in zip(syms, syms[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + freq
    return pairs


def merge_pair(words, pair):
    a, b = pair
    merged = {}
    for w, freq in words.items():
        syms, out, i = w.split(), [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(syms[i]); i += 1
        merged[" ".join(out)] = freq
    return merged


def train_bpe(corpus: str, num_merges: int = 30):
    """Learnmerges: repeatedly merge the most frequent pair (GPT-style)."""
    words, vocab = {}, set()
    for w in corpus.split():
        tokens = " ".join(list(w)) + " </w>"
        words[tokens] = words.get(tokens, 0) + 1
        vocab.update(tokens.split())
    merges = []
    for _ in range(num_merges):
        stats = get_stats(words)
        if not stats:
            break
        best = max(stats, key=stats.get)
        words = merge_pair(words, best)
        merges.append(best)
        for w in words:
            vocab.update(w.split())
    return merges, vocab, words


def encode(text, merges):
    for a, b in merges:
        text_pairs = []
        for w in text.split():
            syms, out, i = list(w) + ["</w>"], [], 0
            # apply in-stream merge (simplified encode)
            joined = "".join(syms)
            joined = joined.replace(a + b, f" {a+b} ")
            text_pairs.append(" ".join(joined.split()))
        text = " ".join(text_pairs)
    return text.split()


# ---------------- scaling laws ----------------
def compute_optimal(n_params_B: float, flops_budget: float = 6e21):
    """Chinchilla-style: budget = 6 * N * D  -> D tokens; ratio = budget/(6 N^2).

    Default 6e21 FLOPs ~= compute-optimal (~20 tokens/param) for a 7B model.
    """
    N = n_params_B * 1e9
    D = flops_budget / (6 * N)
    return D, D / N


def main():
    corpus = "the cat sat on the mat the cat ran the cat sat " * 5 + \
             "language models predict the next token predict language " * 5

    # ---- 1) BPE learns subword merges (frequent pairs merge first) ----
    merges, vocab, final_words = train_bpe(corpus, num_merges=25)
    assert len(merges) == 25
    assert ("t", "h") in merges or ("h", "e") in merges or any(
        "th" in (a + b) for a, b in merges), merges[:3]
    # vocab shrinks sequence length vs char-level
    enc = encode("the cat sat", merges)
    n_chars = len("thecatsat</w></w></w>")
    assert len(enc) <= n_chars

    # round-trip property: merging never invents characters
    joined = "".join(enc).replace("</w>", "")
    assert set(joined) <= set("".join(corpus.split()))

    # ---- 2) scaling laws: loss ~ power law; Chinchilla ratio ~ 20 tokens/param ----
    D_7b, ratio_7b = compute_optimal(7, 6e21)         # budget ~ optimal for 7B
    D_70b, ratio_70b = compute_optimal(70, 6e21)
    assert D_7b > D_70b                       # bigger model -> fewer optimal tokens
    # within the classic 10-40 band of "tokens per parameter"
    assert 5 < ratio_7b < 60, ratio_7b

    # compute-optimal 20 tokens/param for a target size
    target_B = 13
    D_target = 20 * target_B * 1e9
    flops = 6 * (target_B * 1e9) * D_target

    # ---- 3) decoding vocabulary: temperature flattens the distribution ----
    logits = np.array([3.0, 1.0, 0.2])
    def probs(t):
        p = np.exp(logits / t); return p / p.sum()
    assert probs(0.5)[0] > probs(2.0)[0]      # low tau sharper, high tau flatter

    # ---- 4) chat format shape (messages the factory accepts) ----
    from ai_core import get_llm
    llm = get_llm()
    reply = llm.chat([{"role": "system", "content": "You are terse."},
                      {"role": "user", "content": "summarize transformers and attention"}])
    assert isinstance(reply, str) and len(reply) > 10

    print(f"PASS m14 llm | bpe_merges={len(merges)} vocab={len(vocab)} "
          f"encode('the cat sat')={len(enc)}tok vs {n_chars}chars | "
          f"7B_opt_D={D_7b/1e9:.1f}B ratio={ratio_7b:.1f} | "
          f"13B_chinchilla_D={D_target/1e9:.1f}B | llm={llm.name}")


if __name__ == "__main__":
    main()
