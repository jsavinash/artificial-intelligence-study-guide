# Matrix Multiply for AI

> Companion to [00 §1](../curriculum/00-mathematical-foundations.md) · [Docs index](../README.md)

Every layer, attention head, and batch runs on this one op.

![Matmul shapes](../figures/00-math/02_matmul_shapes.png)

---

## 1. Core theory

**Analogy:** mixing console. Inputs = mics, knobs = `W`, outputs = speakers. Training tunes the knobs.

**Definition:** `(m×n)·(n×p) → (m×p)`; cell = row·column dot product.

**Why AI:** data = matrices; layer = `Y = WX + b`; one op scores a batch; matmul + ReLU = deep nets.

---

## 2. Formula

    C[i,j] = Σ A[i,k]·B[k,j]; (m×n)·(n×p) → (m×p)
    Y = W·X + b; Attention = softmax(Q·Kt/√d)·V

`m` outputs · `n` features (summed out) · `p` samples · `W(out×in)` · `X(in×batch)` · `b` bias. Row = template, column = sample, dot = match.

---

## 3. Behavior

Weight ↑ → output ↑. Input `0` kills the path. `+` excites, `−` inhibits. Shapes must agree; `A·B ≠ B·A`; cost `O(mnp)`; add ReLU (matmul alone is linear).

---

## 4. Diagram

`W(2×3)·X(3×2) → Y(2×2)`, inner 3 matches:

    W: 2 neurons x 3 features      X: 3 features x 2 samples
    [ 0.5 -1.0  2.0 ] --+          [ 1.0  0.0 ] --+
    [ 1.5  0.5 -0.5 ]   | inner    [ 2.0  1.0 ]   | inner
                         | 3 == 3   [-1.0  3.0 ]   | 3 == 3 OK
      row = template     |  OK                  |
                         +----> col = sample    |
                                                v
                         Y: 2 x 2
                          [ -3.0   5.5 ] <- stay
                          [  2.0  -2.0 ] <- churn
                             A       B

Plot: `imshow(W)`, `scatter(Y[0], Y[1])`, dashed `y=x`. Dead rows = wasted neurons; collapsed points = rank loss.

    y2 (churn)
     3 |     A(-3.0, 2.0) x
     2 |    /
     1 |   /
     0 |--/------> y1 (stay)
    -1 | / direction of W row 1
    -2 |/            x B(5.5, -2.0)
    -3 |
       +---------------------
        -4 -2  0  2  4  6

---

## 5. Worked example: churn layer

3 features → 2 neurons, 2 customers:

    W = [[ 0.5, -1.0,  2.0 ],
         [ 1.5,  0.5, -0.5 ]]   (2x3)
    X = [[ 1.0,  0.0 ],
         [ 2.0,  1.0 ],
         [-1.0,  3.0 ]]         (3x2)
    Goal: Y = W·X + b (2×2)

    Y[0,0] = 0.5·1 + (-1)·2 + 2·(-1) + 0.5 = -3.0
    Y[1,0] = 1.5·1 + 0.5·2 + (-0.5)·(-1) - 1 = 2.0
    Y[0,1] = 0.5·0 + (-1)·1 + 2·3 + 0.5 = 5.5
    Y[1,1] = 1.5·0 + 0.5·1 + (-0.5)·3 - 1 = -2.0
    Y = [[-3.0, 5.5],[2.0, -2.0]]

A → churn (`-3.0` vs `2.0`); B → stay (`5.5` vs `-2.0`). Check: `W @ X + b` in NumPy.
