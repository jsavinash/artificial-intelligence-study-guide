# 07 — Deep Learning Fundamentals

> Back to [index](../../README.md) · Prev: [06 Feature Engineering](06-feature-engineering.md) · Next: [08 Training & Regularization](08-training-and-regularizing-networks.md)

## 1. The neuron (perceptron)

A single unit computes: `z = w·x + b`, `a = f(z)` — weighted sum, bias, then a nonlinearity.

**Why nonlinearity is mandatory:** stacked *linear* layers collapse to one linear layer (`W₂W₁x = W'x`). Without `f`, depth buys nothing. Networks are **universal approximators** (with enough width) *because* of nonlinear activations.

**History:** Rosenblatt's perceptron (1958) could only learn linearly separable patterns (XOR failure, 1969) → winter → backprop + hidden units (1986) → deep learning revolution (2012).

## 2. Multilayer Perceptron (MLP / feedforward net)

```
input → [Dense + activation] → [Dense + activation] → ... → output
```

- **Width × depth** = capacity. Deeper = more compositional features (edges → textures → parts → objects).
- **Output layer conventions:** regression = linear (MSE/MAE); binary = 1 sigmoid + BCE; multiclass = K softmax + categorical cross-entropy.
- Framework anatomy (PyTorch): `nn.Module`, layers as attributes, `forward()` defines the graph, `loss.backward()` computes gradients, `opt.step()` updates weights.

## 3. Loss functions (what "good" means)

| Task | Loss |
|---|---|
| Regression | MSE (Gaussian), MAE (Laplace, robust), Huber (compromise) |
| Binary cls | Binary cross-entropy (Bernoulli MLE) |
| Multiclass | Categorical cross-entropy over softmax |
| Ranking | Margin/ranking loss, contrastive |
| Generation | Next-token cross-entropy ([11](11-transformers-and-foundation-models.md)) |

Cross-entropy **penalizes confident mistakes catastrophically** (`−log p` → ∞ as p→0) — exactly what you want while learning.

## 4. Backpropagation — the engine

Forward pass computes activations and loss; backward pass applies the **chain rule layer by layer**, producing `∂L/∂w` for every weight via reverse-mode autodiff (compute graph + adjoints). Implementation notes:

- **Autograd** (PyTorch `loss.backward()`) builds/uses a dynamic graph; `zero_grad()` before each step — gradients accumulate by default.
- **Vanishing gradients:** sigmoid/tanh derivatives ≤ 0.25 multiplied across layers → early layers stop learning. Fixes: ReLU family, careful init, residual connections, normalization ([08](08-training-and-regularizing-networks.md)).
- **Exploding gradients:** exploding products → NaN. Fixes: gradient clipping, sane init.

## 5. Gradient descent variants

```
full batch:  w ← w − η·∇L(all data)          stable, slow
SGD:         sample 1 point                    noisy, fast
mini-batch:  sample B points (32–4096)         the standard — GPU friendly
```

- **Learning rate η** is the #1 hyperparameter: too high → divergence/oscillation; too low → slow/stuck. Use LR finder (train one epoch sweeping LR; pick steepest loss descent region).
- Mini-batch noise is a *feature*: it escapes sharp minima that generalize poorly.

## 6. Activation functions

| Function | Properties | Use |
|---|---|---|
| ReLU `max(0,z)` | fast, non-saturating | **default hidden** |
| Leaky/PReLU | small negative slope | dying-ReLU mitigation |
| GELU / SiLU(Swish) | smooth, stochastic gating | transformers, modern CNNs |
| Sigmoid `σ` | (0,1) | binary output, gates (vanishes in hidden layers) |
| tanh | (−1,1) | RNN gates, centered |
| Softmax | simplex over K | multiclass output |
| Softplus / ELU | smooth ReLU variants | niche |

## 7. PyTorch mini-primer (the default framework)

```python
import torch, torch.nn as nn
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for Xb, yb in train_loader:            # DataLoader = batching + shuffling
    opt.zero_grad()
    logits = model(Xb)
    loss = loss_fn(logits, yb)         # includes softmax
    loss.backward()                    # backprop
    opt.step()                         # update
```

Tensor mechanics: shapes/dtype/device, `requires_grad`, broadcasting, `.to(device)` for GPU, `DataLoader`/`Dataset`, `no_grad()` for inference, mixed precision (`torch.autocast`) in [08](08-training-and-regularizing-networks.md).

## 8. Embeddings — learning features from categories/sequences

`nn.Embedding(V, d)` = a trainable lookup table mapping token IDs → dense vectors. Replaces one-hot explosion; learned similarity. Central to NLP, recommenders, and everything after ([10](10-rnn-and-sequence-modeling.md), [11](11-transformers-and-foundation-models.md)).

## Mastery Checklist

- [ ] Explains why stacked linear layers without activation collapse to one layer
- [ ] Derives gradient flow through a 2-layer net using the chain rule
- [ ] Matches every task type to its correct output activation + loss
- [ ] Chooses batch vs mini-batch vs SGD and justifies the learning rate range
- [ ] Writes the PyTorch training loop from memory (zero_grad → forward → loss → backward → step)
- [ ] Names 4 activations with where each belongs and why ReLU won
