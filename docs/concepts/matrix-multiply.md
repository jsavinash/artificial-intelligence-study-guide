# Matrix Multiply for AI

> Companion to [00 §1](../curriculum/00-mathematical-foundations.md) · [Docs index](../README.md)

Every layer in a neural network uses this one operation to turn inputs into outputs.

![Matmul shapes](../figures/00-math/02_matmul_shapes.png)

---

## 1. Core idea in plain words

**Analogy:** think of a sound mixer. The inputs are microphones. The knobs decide how loud each mic is in each speaker. Training a model just means finding the best knob positions.

**Definition:** we multiply a table `A` with rows `m` and columns `n` by a table `B` with rows `n` and columns `p`. We get a new table `C` with rows `m` and columns `p`. Each box in `C` is one row from `A` multiplied piece by piece with one column from `B`, then added up.

**Why AI uses it:** data comes as tables. A layer is `Y = W times X plus b`. One multiply scores a whole group of examples at once. Stacking this with ReLU builds a deep network.

---

## 2. Formula, decoded

Each output box:

    C[row, col] = add up A[row, k] times B[k, col] for every k

Shape rule:

    (rowsA x colsA) times (rowsB x colsB) works only if colsA equals rowsB.

A neural layer:

    Y = W times X plus b

- `W` holds the knobs: one row per output, one column per input.
- `X` holds the data: one column per example.
- `b` is a small extra push per output.
- Attention uses this twice: first to compare queries with keys, then to mix values.

---

## 3. What changes the answer

- Bigger weight means bigger output. A weight of zero means that input is ignored.
- An input of zero kills that path. Empty or masked inputs add nothing.
- A plus weight pushes the output up. A minus weight pushes it down.
- Both tables must fit: the middle sizes must match, and order matters.
- Multiplying alone can only mix, never bend, so real networks add ReLU after it.

---

## 4. Picture

`W` has 2 rows and 3 columns. `X` has 3 rows and 2 columns. The middle 3 matches, so the answer is 2 by 2:

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

Plot hint: show `W` as a heatmap, plot each answer column as one point, and draw the tie-line `y = x`.

    up = churn score
     3 |     A(-3.0, 2.0) x
     2 |    /
     1 |   /
     0 |--/------> across = stay score
    -1 | / churn grows if you go up, stay grows if you go right
    -2 |/            x B(5.5, -2.0)
       +---------------------
        low ----------> high

Reading the plot: A sits high on churn, B sits far on stay. The diagonal is the tie. Whichever side a point lands on wins.

---

## 5. Worked example: will a customer stay or churn?

Inputs are age, monthly spend, and support tickets. Outputs are stay and churn scores. Two customers:

    W = [[ 0.5, -1.0,  2.0 ],
         [ 1.5,  0.5, -0.5 ]]   (2x3)
    X = [[ 1.0,  0.0 ],
         [ 2.0,  1.0 ],
         [-1.0,  3.0 ]]         (3x2)
    b = [0.5, -1.0]: small extra push for stay and churn.
    Goal: Y = W times X plus b (2x2)

    Y[0,0] = 0.5·1 + (-1)·2 + 2·(-1) + 0.5 = -3.0
    Y[1,0] = 1.5·1 + 0.5·2 + (-0.5)·(-1) - 1 = 2.0
    Y[0,1] = 0.5·0 + (-1)·1 + 2·3 + 0.5 = 5.5
    Y[1,1] = 1.5·0 + 0.5·1 + (-0.5)·3 - 1 = -2.0
    Y = [[-3.0, 5.5],[2.0, -2.0]]

Customer A leans churn because `-3.0` is less than `2.0`. Customer B leans stay because `5.5` beats `-2.0`. Same check in NumPy: `W @ X + b`.
