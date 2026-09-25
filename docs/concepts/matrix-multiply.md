# Matrix Multiply for AI / ML / Deep Learning

> Companion to [00 - Mathematical Foundations](../curriculum/00-mathematical-foundations.md), Sec 1 Linear Algebra. Back to [Documentation Index](../README.md).

Every dataset is a matrix. Every model is matrix operations. This note is the deep-dive for the linear-algebra summary in module 00: theory, formula, behavior, visual, and a fully worked AI calculation.

---

## 1. THE CORE THEORY

**Intuitive analogy - a mixing console.**

Think of 3 input microphones (drums, guitar, voice) as input vector X. Two output speakers (Left, Right) as output vector Y. The knobs are matrix W. Knob W[0,2] controls how much voice goes to the Left speaker: 2.0 amplifies, 0 mutes, -1.0 inverts. Matrix multiply turns all knobs at once. Every output is a weighted mix of every input. A neural layer is a learnable mixing console, and training learns the knob positions.

**Technical definition in plain language.**

Matrix multiply takes table A of size m x n and table B of size n x p and produces table C of size m x p. Cell C[i,j] equals row i of A dotted with column j of B (pairwise multiply, then add). So C is a table of dot products.

**Why this is necessary for AI.**

- Data is matrices: 1000 samples x 50 features is a 1000 x 50 matrix.
- Models are matmuls: a layer is Y = WX + b; attention is softmax(QKt/sqrt(d))V.
- Batching equals speed: W dot X processes a whole mini-batch in one parallel GPU call.
- Depth equals composed mixes: alternate mixing (matmul) with bending (ReLU).

If vectors are the nouns of AI, matrix multiply is the verb.

![Matmul shape flow](../figures/00-math/02_matmul_shapes.png)

---

## 2. THE MATHEMATICAL FORMULA

Foundational formula (A is m x n, B is n x p, C = A dot B is m x p):

    C[i,j] = sum(k=1..n) A[i,k] * B[k,j]

Shape law:

    (m x n) dot (n x p) -> (m x p), inner n must match

AI layer formula:

    Y = W dot X + b

Batched form:

    Y[b,o] = sum(i=1..in) X[b,i] * W[o,i] + b[o]

Attention uses matmul twice:

    Attention(Q,K,V) = softmax(Q dot Kt / sqrt(d)) dot V

Symbol list:

- A, B: input matrices being multiplied.
- C: output, C = A dot B.
- m: rows of A and C, number of outputs.
- n: columns of A and rows of B, input features, summed away.
- p: columns of B and C, number of samples.
- i: row index, which output neuron.
- j: column index, which data sample.
- k: inner index over features, summed out.
- W: weight matrix (out x in).
- X: input data (in x batch), one column per sample.
- b: bias vector (out).
- Y: outputs or logits (out x batch).
- Q, K, V, d: attention queries, keys, values, head dim.

Why row times column? Each output neuron holds a template (a row of W) and asks how much the sample (a column of X) looks like it. The dot product is that alignment score.
---

## 3. CAUSATION AND BEHAVIOR

Matmul is linear, so cause and effect are proportional.

- Weights up implies output up: doubling W[0,1] doubles feature 1 vote in output 0.
- Input near 0 kills the path: W[o,i] * 0 = 0. Zero features block signal.
- Sign flips direction: positive is excitatory, negative is inhibitory.

Edge cases and limits:

- Shape mismatch is illegal: (2x3) dot (2x3) fails, 3 != 2. Transpose one side.
- Not commutative: A dot B != B dot A. Order is meaning.
- A dot 0 = 0 kills info. A dot I = A is a no-op.
- Rank collapse: dependent rows in W waste neurons.
- Cost O(m n p). FP16 overflow needs LayerNorm and 1/sqrt(d) scaling.
- Linearity alone never suffices, so interleave ReLU or GELU.

---

## 4. TEXT-BASED INTERACTIVE PLOT DIAGRAM

Shape flow for W(2x3) dot X(3x2) -> Y(2x2):

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

Output space, boundary y1 = y2 is diagonal:

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

## 5. AI EXAMPLE WITH STEP-BY-STEP CALCULATION

Use case: one forward pass of a churn layer. 3 features (age, spend, tickets) to 2 neurons (stay, churn). Batch is 2 customers.

    W = [[ 0.5, -1.0,  2.0 ],
         [ 1.5,  0.5, -0.5 ]]   (2x3)
    X = [[ 1.0,  0.0 ],
         [ 2.0,  1.0 ],
         [-1.0,  3.0 ]]         (3x2)
    b = [ 0.5, -1.0 ]
    Goal: Y = W dot X + b  ->  (2x2)

Cells:

    Y[0,0] = Row0(W) dot Col0(X) + b0
    Y[1,0] = Row1(W) dot Col0(X) + b1
    Y[0,1] = Row0(W) dot Col1(X) + b0
    Y[1,1] = Row1(W) dot Col1(X) + b1

Arithmetic, no skipped steps:

Y[0,0] (stay, customer A):

    = (0.5*1.0) + (-1.0*2.0) + (2.0*-1.0) + 0.5
    = 0.5 + (-2.0) + (-2.0) + 0.5 = -3.0

Y[1,0] (churn, customer A):

    = (1.5*1.0) + (0.5*2.0) + (-0.5*-1.0) + (-1.0)
    = 1.5 + 1.0 + 0.5 - 1.0 = 2.0

Y[0,1] (stay, customer B):

    = (0.5*0.0) + (-1.0*1.0) + (2.0*3.0) + 0.5
    = 0.0 - 1.0 + 6.0 + 0.5 = 5.5

Y[1,1] (churn, customer B):

    = (1.5*0.0) + (0.5*1.0) + (-0.5*3.0) + (-1.0)
    = 0.0 + 0.5 - 1.5 - 1.0 = -2.0

Result:

    Y = [[-3.0,  5.5],
         [ 2.0, -2.0]]

Customer A: stay -3.0 vs churn 2.0 predicts churn. Customer B: 5.5 vs -2.0 predicts stay. Backprop through this layer is itself matmuls with transposes. Verified with NumPy: (W @ X + b) gives exactly this matrix.

---
Companion to module 00 section 1. Numbers verified by hand and NumPy. Executable version: make run M=00.
