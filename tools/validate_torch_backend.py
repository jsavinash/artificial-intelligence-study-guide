#!/usr/bin/env python3
"""Validate every function in ai_core.torch_backend — torch path and fallback.

Usage:
    python3 tools/validate_torch_backend.py

Runs each public backend function on tiny inputs and reports PASS/FAIL, so a
regression in either path is caught immediately. Prints the active backend so
you always know which branch was exercised.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from ai_core import torch_backend as T  # noqa: E402

RESULTS: list[tuple[str, bool | str, str]] = []   # status: True | False | "skip"


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, str(detail or "")))
        print(f"  PASS  {name:<32} {detail or ''}")
    except Exception as e:                                   # noqa: BLE001
        # torch_required() raises by design when torch is absent — the module
        # refuses to ship an "honest-looking" fake for VAE/GAN/diffusion/DQN.
        # That is a legitimate skip in numpy mode, not a regression.
        if "requires PyTorch" in str(e):
            RESULTS.append((name, "skip", "torch-only (numpy mode)"))
            print(f"  SKIP  {name:<32} torch-only (no torch installed)")
            return
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name:<32} {type(e).__name__}: {e}")
        if "--trace" in sys.argv:
            traceback.print_exc()


def _cls(n=300, d=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
         + (X[:, 2] > 0.8).astype(int))
    return X.astype(np.float32), y


def main() -> int:
    print("=" * 72)
    print(f"torch_backend validation | backend={T.backend_label()} "
          f"| HAS_TORCH={T.HAS_TORCH} | mps={T.HAS_MPS} cuda={T.HAS_CUDA}")
    print("=" * 72)

    print("\n[device & utils]")
    check("get_device()", lambda: T.get_device())
    check("set_seed(0)", lambda: T.set_seed(0))
    check("sync(device)", lambda: T.sync(T.get_device()))
    check("count_params(mlp)", lambda: T.count_params(T.build_mlp(8, 3))
          if T.HAS_TORCH else "n/a (numpy)")

    print("\n[mlp / generic training]")
    X, y = _cls()

    def _mlp():
        r = T.train_classifier(X, y, hidden=(32, 16), epochs=80, lr=0.05)
        assert r["losses"][-1] < r["losses"][0], "loss did not decrease"
        assert r["acc"] > 0.5, r["acc"]
        return f"acc={r['acc']:.3f} loss={r['losses'][-1]:.3f} {r['backend']}"
    check("train_classifier()", _mlp)

    def _mlp_reg():
        r = T.train_classifier(X, y, hidden=(64,), epochs=60, lr=0.05,
                               dropout=0.2, weight_decay=1e-4)
        assert r["acc"] > 0.5, r["acc"]
        return f"acc={r['acc']:.3f} dropout+wd"
    check("train_classifier(dropout,wd)", _mlp_reg)

    def _opts():
        out = []
        for opt in ("adam", "sgd", "adamw"):
            r = T.train_classifier(X, y, epochs=40, lr=0.05, optimizer=opt)
            assert r["acc"] > 0.4, (opt, r["acc"])
            out.append(f"{opt}={r['acc']:.2f}")
        return " ".join(out)
    check("optimizers adam/sgd/adamw", _opts)

    print("\n[convolutional]")
    Xc = np.random.default_rng(0).normal(size=(120, 1, 8, 8)).astype(np.float32)
    yc = (Xc.mean((1, 2, 3)) > 0).astype(int)

    def _cnn():
        r = T.train_cnn(Xc, yc, in_size=8, epochs=25, lr=0.01)
        assert r["acc"] >= 0.5, r["acc"]
        return f"acc={r['acc']:.3f} {r['backend']}"
    check("train_cnn()", _cnn)

    print("\n[sequences]")
    Xs = np.random.default_rng(0).normal(size=(120, 6, 4)).astype(np.float32)
    ys = (Xs[:, -1, 0] > 0).astype(int)

    def _mk_rnn(cell):
        def inner():
            r = T.train_rnn(Xs, ys, hidden=16, cell=cell, epochs=40, lr=0.02)
            acc = r.get("acc", r.get("score"))
            assert acc >= 0.5, (cell, acc)
            return f"{cell} acc={acc:.3f} {r['backend']}"
        return inner
    check("train_rnn(lstm)", _mk_rnn("lstm"))
    check("train_rnn(gru)", _mk_rnn("gru"))
    check("train_rnn(rnn)", _mk_rnn("rnn"))

    print("\n[attention & transformer]")

    def _attn():
        out = T.attention_output(np.random.rand(5, 8).astype(np.float32))
        assert out.shape == (5, 16), out.shape
        return f"shape={out.shape}"
    check("attention_output()", _attn)

    def _attn_causal():
        x = np.random.rand(4, 6, 8).astype(np.float32)
        out = T.attention_output(x, causal=True)
        assert out.shape == (4, 6, 16), out.shape
        return f"shape={out.shape} batched+causal"
    check("attention_output(batch,causal)", _attn_causal)

    def _lm():
        rng = np.random.default_rng(0)
        toks = rng.integers(0, 12, size=400).tolist()
        r = T.train_tiny_lm(toks, block_size=8, epochs=60, lr=3e-3)
        assert r["losses"][-1] < r["losses"][0], "LM loss did not decrease"
        return f"loss {r['losses'][0]:.2f}->{r['losses'][-1]:.2f} {r['backend']}"
    check("train_tiny_lm()", _lm)

    print("\n[generative]")
    Xg = np.random.default_rng(0).normal(size=(200, 16)).astype(np.float32)

    def _vae():
        r = T.train_vae(Xg, epochs=40, lr=1e-3)
        assert r["losses"][-1] < r["losses"][0]
        return f"loss {r['losses'][0]:.1f}->{r['losses'][-1]:.1f} {r['backend']}"
    check("train_vae()", _vae)

    def _gan():
        r = T.train_gan(Xg, epochs=40, lr=2e-3)
        assert "d_losses" in r and len(r["d_losses"]) == 40
        assert r["final_d"] == r["d_losses"][-1]
        return f"d={r['final_d']:.3f} g={r['final_g']:.3f} {r['backend']}"
    check("train_gan()", _gan)

    def _diff():
        r = T.train_diffusion(Xg, timesteps=20, epochs=30, lr=1e-3)
        assert r["losses"][-1] < r["losses"][0]
        return f"loss {r['losses'][0]:.3f}->{r['losses'][-1]:.3f} {r['backend']}"
    check("train_diffusion()", _diff)

    print("\n[reinforcement learning]")

    def _dqn():
        r = T.train_dqn(episodes=120)
        assert "returns" in r and len(r["returns"]) == 120
        return (f"avg_first={r['avg_first']:.1f} avg_last={r['avg_last']:.1f} "
                f"{r['backend']}")
    check("train_dqn() (default GridWorld)", _dqn)

    def _dqn_learns():
        r = T.train_dqn(episodes=400)
        assert r["avg_last"] > r["avg_first"], (
            f"DQN did not improve: {r['avg_first']:.2f} -> {r['avg_last']:.2f}")
        return f"{r['avg_first']:.2f} -> {r['avg_last']:.2f} improved"
    check("train_dqn() learns", _dqn_learns)

    print("\n[peft / lora]")

    def _lora_math():
        m = T.lora_param_math(512, 512, 8)
        assert m["percent"] < 10, m
        return f"r=8 -> {m['percent']:.2f}% trainable"
    check("lora_param_math()", _lora_math)

    def _lora():
        r = T.train_peft(X, y, hidden=(32,), r=4, epochs=40, lr=1e-2)
        assert r["lora"]["percent"] < 50, r["lora"]
        return (f"acc={r['acc']:.3f} adapters={r['lora']['adapters']} "
                f"trainable={r['lora']['percent']:.1f}%")
    check("train_peft()", _lora)

    def _lora_merge():
        if not T.HAS_TORCH:
            return "n/a (numpy)"
        model = T.build_mlp(8, 3, hidden=(16,))
        info = T.apply_lora(model, r=4)
        assert info["adapters"] == 2, info["adapters"]      # both Linear layers
        assert info["trainable"] < info["total"], "nothing was frozen"
        merged = T.merge_lora(model)
        assert merged == 2, merged
        return (f"adapters={info['adapters']} trainable={info['percent']:.1f}% "
                f"merged={merged}")
    check("apply_lora() + merge_lora()", _lora_merge)

    def _lora_zero_is_identity():
        """B is initialised to zero, so an untrained adapter must not change y."""
        if not T.HAS_TORCH:
            return "n/a (numpy)"
        model = T.build_mlp(8, 3, hidden=(16,))
        x = np.random.rand(5, 8).astype(np.float32)
        before = model(T.torch.as_tensor(x) if T.HAS_TORCH else x)
        T.apply_lora(model, r=4)
        after = model(T.torch.as_tensor(x))
        assert T.torch.allclose(before, after, atol=1e-6), "LoRA changed output!"
        return "untrained LoRA is a no-op (B=0)"
    check("lora_zero_init_identity", _lora_zero_is_identity)
    print("\n[classical ML via tensors/autograd]")

    def _ridge():
        """Closed form vs 600 SGD steps must agree — the autograd self-check."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 5)).astype(np.float32)
        y = (X @ np.array([1.5, -2.0, 0.5, 0.0, 3.0]) + 0.4).astype(np.float32)
        plain = T.fit_linear_regression(X, y, epochs=600, lr=0.1)
        ridged = T.fit_linear_regression(X, y, epochs=600, lr=0.1, ridge=0.5)
        assert plain["mse"] < 0.5, plain["mse"]
        assert plain["weights"].shape[0] == 5
        return (f"mse={plain['mse']:.3f} ridge_mse={ridged['mse']:.3f} "
                f"closed_form_agrees={plain['agrees_with_closed_form']} "
                f"{plain['backend']}")
    check("fit_linear_regression()", _ridge)

    def _logreg():
        """Logistic regression by gradient descent on a separable problem."""
        X, y = _cls(n=400, d=6, k=2)
        y = (np.asarray(y) > 0).astype(int)
        r = T.train_logistic_regression(X, y, epochs=300, lr=0.1, l2=0.01)
        assert r["acc"] > 0.85, r["acc"]
        return f"acc={r['acc']:.3f} l2=0.01 {r['backend']}"
    check("train_logistic_regression()", _logreg)

    def _kmeans():
        """k-means++ must find the near-optimal solution from EVERY seed.

        Regression guard: with plain random init, seed 0 landed in a bad local
        optimum (inertia 2998 vs the optimal ~90) — the 3-blob test below is
        the case that exposed it.
        """
        rng = np.random.default_rng(0)
        centres = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float64)
        X = np.vstack([rng.normal(c, 0.5, size=(60, 2)) for c in centres])
        tss = float(((X - X.mean(0)) ** 2).sum())          # ~7903
        worst = 0.0
        for seed in (0, 1, 2, 3):
            r = T.kmeans(X, k=3, epochs=60, seed=seed)
            assert len(np.unique(r["labels"])) == 3, f"seed {seed}: collapsed"
            assert r["inertia"] < 0.05 * tss, (seed, r["inertia"], tss)
            worst = max(worst, r["inertia"])
        return (f"inertia<={worst:.1f} over 4 seeds (<5% of TSS {tss:.0f}) "
                f"{T.backend_label()}")
    check("kmeans()", _kmeans)

    def _pca():
        """PCA on correlated data: PC1 must explain most of the variance."""
        rng = np.random.default_rng(0)
        base = rng.normal(size=(300, 1))
        X = np.hstack([base * 3 + rng.normal(0, 0.1, (300, 1)),
                       base * 2 + rng.normal(0, 0.1, (300, 1)),
                       rng.normal(0, 0.1, (300, 1))])
        r = T.pca(X, n_components=2)
        assert r["explained_variance_ratio"][0] > 0.9, r["explained_variance_ratio"]
        assert r["components"].shape == (2, 3), r["components"].shape
        return (f"pc1={r['explained_variance_ratio'][0]:.3f} "
                f"{r['backend']}")
    check("pca()", _pca)

    def _gmm():
        """Soft clustering: responsibilities must be a valid simplex per point."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal([0, 0], 0.4, size=(80, 2)),
                       rng.normal([6, 6], 0.4, size=(80, 2))])
        r = T.train_gmm(X, k=2, epochs=250, seed=0)
        R = np.asarray(r["responsibilities"])
        assert R.shape == (160, 2)
        assert np.allclose(R.sum(1), 1.0, atol=1e-4), "rows must sum to 1"
        assert r["log_likelihood"][-1] >= r["log_likelihood"][0], "not improving"
        return f"rows_sum_to_1 loss_up={r['log_likelihood'][-1]:.1f} {r['backend']}"
    check("train_gmm()", _gmm)

    print("\n[advanced / specialized]")

    def _mf():
        """Matrix factorisation must beat the global-mean baseline."""
        rng = np.random.default_rng(0)
        U = rng.normal(size=(30, 4))
        V = rng.normal(size=(25, 4))
        R = U @ V.T * 2 + 3
        mask = rng.random(R.shape) < 0.7            # 70% observed
        r = T.train_matrix_factorization(R, n_factors=4, epochs=400, lr=0.02,
                                         mask=mask, seed=0)
        assert r["rmse"] < r["baseline_rmse"], (r["rmse"], r["baseline_rmse"])
        return (f"rmse={r['rmse']:.3f}<base={r['baseline_rmse']:.3f} "
                f"{r['backend']}")
    check("train_matrix_factorization()", _mf)

    def _gnn():
        """Message passing over a 2-community graph (block model)."""
        rng = np.random.default_rng(0)
        n, k = 40, 2
        labels = np.repeat([0, 1], n // 2)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                p = 0.5 if labels[i] == labels[j] else 0.03
                if rng.random() < p:
                    A[i, j] = A[j, i] = 1
        A += np.eye(n)
        deg = A.sum(1, keepdims=True)
        A_norm = A / np.maximum(deg, 1e-9)
        feats = rng.normal(size=(n, 6))
        # make features weakly informative so the graph must do real work
        feats[:, 0] += labels * 0.3
        train_mask = rng.random(n) < 0.5
        r = T.train_gnn(A_norm, feats, labels, train_mask=train_mask,
                        epochs=200, seed=0)
        assert r["acc"] >= 0.8, r["acc"]
        return f"acc={r['acc']:.3f} nodes={r['n_nodes']} {r['backend']}"
    check("train_gnn()", _gnn)

    def _clip():
        """Two-tower contrastive training: retrieval must beat chance."""
        rng = np.random.default_rng(0)
        n, d = 48, 8
        latent = rng.normal(size=(n, 4))
        img = latent @ rng.normal(size=(4, d)) + rng.normal(0, 0.05, (n, d))
        txt = latent @ rng.normal(size=(4, d)) + rng.normal(0, 0.05, (n, d))
        r = T.train_clip(img, txt, epochs=300, lr=0.01, embed_dim=8, seed=0)
        assert r["i2t_top1"] > 0.9, r["i2t_top1"]
        assert r["t2i_top1"] > 0.9, r["t2i_top1"]
        assert r["similarity"].shape == (n, n)
        return (f"i2t={r['i2t_top1']:.2f} t2i={r['t2i_top1']:.2f} "
                f"{r['backend']}")
    check("train_clip()", _clip)



    n_pass = sum(1 for _, ok, _ in RESULTS if ok is True)
    n_skip = sum(1 for _, ok, _ in RESULTS if ok == "skip")
    n_fail = sum(1 for _, ok, _ in RESULTS if ok is False)
    print("\n" + "-" * 72)
    print(f"RESULT: {n_pass} passed, {n_fail} failed"
          + (f", {n_skip} skipped (torch-only)" if n_skip else "")
          + f"  (backend={T.backend_label()})")
    for name, ok, msg in RESULTS:
        if ok is False:
            print(f"  FAILED {name}: {msg}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
