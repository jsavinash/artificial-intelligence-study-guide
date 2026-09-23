# 27 — Accelerated Computing & PyTorch

> Back to [index](../../README.md) · Prev: [26 ML Interview Prep & Coding](26-ml-interview-prep-and-coding.md)

Every other module shows *what* to compute; this one is about *what runs it*. It is the module that decides whether the code you already understand is fast, slow, or impossible.

## 1. Why frameworks exist

Pure NumPy expresses math beautifully but executes **one operation at a time on the CPU**. Three things break at scale:

1. **Autograd** — you hand-write `backward()` for every model. Fine for an MLP, brutal for a transformer.
2. **Kernel efficiency** — a Python `for` loop over image pixels runs at ~1% of hardware capability.
3. **Accelerators** — GPUs/TPUs want thousands of parallel ops and one big graph, not millions of tiny Python calls.

PyTorch answers all three: a **tape-based autograd engine**, **fused vendor kernels**, and **device abstraction**. The cost is a ~590 MB install and a ~1.2 s import.

## 2. The amortization rule — the central lesson

This is the part most tutorials skip. A framework is **not automatically faster**. Its overhead is paid up front, so the decision is economic:

```
torch_time  =  numpy_time / speedup  +  IMPORT_COST
break-even  =  IMPORT_COST × s / (s − 1)
```

| Speedup `s` | Break-even NumPy time | Verdict at 0.15 s | At 3.0 s |
|---|---|---|---|
| 2× | 2.48 s | NumPy | torch |
| 3× | 1.86 s | NumPy | torch |
| 5× | 1.55 s | NumPy | torch |
| 12× | 1.35 s | NumPy | torch |
| 70× | 1.26 s | NumPy | torch |

Two consequences that surprise people:

- **Break-even approaches the import cost from above, never below.** Even at infinite speedup you must still spend ~1.24 s importing. There is no speedup large enough to justify torch for a 0.1 s workload.
- **A huge speedup does not mean "use it".** The convolution in this repo is ~40–90× faster in torch and *still* shouldn't use torch — the whole workload is 0.05 s.

Measured on this machine (Apple M1, 8 cores, Python 3.14, torch 2.14):

| Workload | NumPy | torch (CPU) | Speedup | Worth it? |
|---|---|---|---|---|
| conv2d 48×48 k=5 ×20 | 0.050 s | 0.001 s | ~40× | No (too small) |
| attention S=512 d=64 ×200 | 0.29 s | 0.19 s | ~1.5–5× | No (too small) |
| MLP train 20000×64, 100 ep | 2.25 s | 1.08 s | ~2–3× | No (below 1.55 s threshold) |
| matmul 100000×512 @ 512×256 | 1.80 s | 1.56 s | ~1× | No — NumPy already calls BLAS |

**The BLAS row is the sleeper finding.** NumPy's `@` is not naive — it dispatches to Accelerate/MKL. For large dense matmul there is *nothing to gain*; torch and NumPy are the same library underneath. Frameworks win on **many small ops in one graph** and **fusion**, not on raw matmul.

## 3. What PyTorch actually gives you

| Capability | Without a framework | With PyTorch |
|---|---|---|
| Gradients | hand-derived `backward()` per model | `loss.backward()` — reverse-mode autodiff |
| Layers/params | manual dicts of arrays | `nn.Module` + `parameters()` |
| Optimizers | hand-coded SGD/Adam | `torch.optim.*` (Adam, AdamW, Lion…) |
| Data pipeline | manual batching/shuffling | `Dataset` + `DataLoader` (workers, pinning) |
| Devices | none | `.to("cpu"\|"mps"\|"cuda")` |
| Precision | float64 default | `autocast` / bfloat16 |
| Fusion/compile | none | `torch.compile`, fused kernels, SDPA/flash-attention |
| Distributed | none | DDP, FSDP, ZeRO |

Autograd is the biggest single win: it makes *new architectures* cheap to try, because you only ever write the forward pass.

## 4. Devices, and the synchronization trap

```python
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model, batch = model.to(dev), batch.to(dev)
```

**GPU work is asynchronous.** The Python call returns before the GPU finishes, so naive timing measures *launch overhead*, not compute:

```python
loss.backward()            # returns immediately
torch.mps.synchronize()    # ← without this, your benchmark is a lie
```

Our measured example: a 4 000×32 MLP for 60 epochs took **0.04 s on CPU vs 0.60 s on MPS** — the Apple GPU was **15× slower** because the batch was too small to hide kernel-launch latency. GPU ≠ faster; GPU = *more parallel*, and it needs enough work to fill.

## 5. Where frameworks win vs where they don't

**Win — small ops fused into one graph (what a training step is):**
- Any model with a real backward pass: autograd eliminates hand-derived gradients
- Convolutions: `im2col` + GEMM beats nested Python loops by 10–100×
- Attention: `scaled_dot_product_attention` fuses the whole QKᵀ→softmax→V chain (flash attention on GPU)
- Long training runs: fixed import cost amortizes to nothing over hours

**Don't win:**
- **Large dense matmul** — NumPy already dispatches to BLAS
- **Small models / short demos** — import cost dominates (this is why 27 examples here stay in NumPy)
- **Data preparation / ETL** — Pandas/Polars/Spark beat torch there
- **Classical ML** — XGBoost/LightGBM/sklearn beat hand-rolled torch for tabular; gradient boosting is not a GPU problem

## 6. GPU economics: what actually limits speed

Four distinct ceilings, in the order you usually hit them:

1. **Launch overhead** — each kernel launch costs ~µs. Thousands of tiny ops = GPU idles.
2. **Memory bandwidth** — most DL ops are memory-bound; the GPU is faster at *moving* data than the problem has data to move. This is why **fusion** matters more than FLOPs.
3. **Compute (FLOPs)** — what marketing quotes. Rarely the binding constraint at small scale.
4. **Utilization (MFU)** — fraction of peak actually achieved. 30–50% MFU is a *good* large-scale training run.

**Consequence:** the first optimization is almost always *bigger batches* and *fewer, larger kernels* — not a bigger GPU.

## 7. Scaling levers, cheapest first

| Lever | What it does | Cost / caveat |
|---|---|---|
| Larger batch | fills the device, fewer launches | may need LR warmup/scale-up |
| `torch.compile` | fuses ops, removes Python overhead | compile time; occasional regressions |
| Mixed precision (bf16/fp16) | ~2× throughput, halves memory | needs loss scaling (fp16) |
| Gradient accumulation | big *effective* batch on small memory | slower wall-clock per step |
| Gradient checkpointing | trades compute for activation memory | ~30% slower, enables much bigger models |
| DDP (data parallel) | replicate model, split batch | must scale LR with world size |
| FSDP / ZeRO | shard params/grads/optimizer state | complexity; comms-bound at small scale |
| Pipeline/tensor parallel | split the model itself | only worth it past single-node memory |

See [21 MLOps](21-mlops-and-production-ml.md) for serving-side optimization (quantization, batching, distillation) and [15 Fine-tuning & PEFT](15-fine-tuning-and-peft.md) for memory tricks specific to LLMs.

## 8. Measurement discipline

Benchmarking is where most "GPU is faster" claims die. Rules:

- **Warm up first** — the first call includes lazy init and kernel selection.
- **Synchronize** — otherwise you time the queue, not the work (§4).
- **Fix the data** — same inputs, same dtype, same seeds for both backends.
- **Compare like for like** — same init distribution, same LR, same epochs, or you are benchmarking luck. A silent bug in this repo's NumPy reference (reusing pre-activations when scoring accuracy) made the two backends appear to differ by 0.35 accuracy; the model was fine, the *evaluation* was the bug.
- **Report variance** — run-to-run noise here turned a 2.7× into a 2.1× measurement.

`make bench` runs the comparison; `benchmarks/bench_backends.py --json` emits machine-readable results.

## 9. Decision playbook

```
Is the workload already fast enough?      → yes: stop. Don't add torch.
Is it a training step with a backward pass? → torch, if runtime > ~1.5 s
Is it large dense matmul?                  → NumPy/BLAS is already optimal
Is it tabular / classical ML?              → sklearn / LightGBM
Is it ETL or data prep?                    → Pandas / Polars / SQL
Is it inference at scale?                  → ONNX Runtime / TensorRT / vLLM
Is it a short demo or teaching example?    → pure NumPy (zero deps, visible math)
```

**This repo's choice:** examples stay NumPy-first so `make run-all` works with zero installs and the math stays visible; torch is an **optional accelerator** (`ai_core.accelerators`) that engages only where it pays. `make setup` reports whether it is present, and every example passes either way.

## Mastery Checklist

- [ ] States the amortization rule and computes a break-even from an import cost and a speedup
- [ ] Explains why a 40× speedup can still be the wrong choice
- [ ] Names the four things PyTorch gives you that NumPy does not
- [ ] Explains why GPU timing without `synchronize()` is meaningless
- [ ] Gives a case where the GPU is *slower* than the CPU and says why
- [ ] Distinguishes memory-bandwidth-bound from compute-bound and picks a fix for each
- [ ] Orders the scaling levers by cost and says when each is appropriate
- [ ] Identifies when NumPy's `@` is already optimal and torch adds nothing
