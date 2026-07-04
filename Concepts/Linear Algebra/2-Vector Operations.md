# Vector Operations - Concise Summary
Once you can represent things as vectors, you need ways to combine and scale them. Every ML model — from linear regression to a transformer — is built almost entirely out of two operations: adding vectors together and stretching/shrinking them.

## Intuition
Think of vector addition as following one direction, then another. If you walk "4 east, 3 north" and then walk "1 east, 2 north", your total displacement is "5 east, 5 north" — you just add the pieces that point the same way.
Scalar multiplication is stretching or shrinking (or flipping) an arrow without changing what direction it points (unless you multiply by a negative number, which flips it 180°).

## Core Operations

**Vector Addition** (componentwise):
$$a + b = (a_1+b_1, a_2+b_2, ...)$$

**Scalar Multiplication** (every component × scalar):
$$c \cdot a = (c \times a_1, c \times a_2, ...)$$

## Examples
- **Addition**: `(4,3) + (1,2) = (5,5)` (follow one displacement, then another)
- **Scaling**: `2 × (4,3) = (8,6)` (doubles length, same direction)
- **Negative scaling**: `-1 × (4,3) = (-4,-3)` (flips direction 180°)

## Key Rules
- Addition requires **same dimension** (can't add ℝ² + ℝ³)
- Scalar can be **any real number**
- Multiplication by 0 → **zero vector**
- Multiplication by negative → **direction flip**


## 5. Mathematical formulation, symbol by symbol

**Vector addition:**
$$a + b = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1+b_1 \\ a_2+b_2 \end{bmatrix}$$

- $a_1, a_2$ — the individual components of vector $a$ (subscripts just mean "1st entry," "2nd entry")
- Addition happens **componentwise** — you add matching positions, nothing more exotic than that.

**Scalar multiplication:**
$$c \cdot a = c \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} = \begin{bmatrix} c\,a_1 \\ c\,a_2 \end{bmatrix}$$

- $c$ — a **scalar**, meaning just a single plain number (as opposed to a vector). The word "scalar" is used specifically because it "scales" the vector's length.

## 6. Geometric interpretation

- **Addition** = combining two displacements into one net displacement (this is literally how physicists add forces or velocities).
- **Scalar multiplication** = resizing a vector along its own line. $c>1$ stretches it, $0<c<1$ shrinks it, $c<0$ flips it, and $c=0$ collapses it to a point (the zero vector).
- Together, these two operations let you build **any** vector that's a "weighted combination" of a few starting vectors — which is exactly the idea behind our next topic, *linear combinations*, so keep this picture in mind.

## 7. Why it matters for AI/ML

- **Gradient descent** updates a model's weights using exactly this operation: $w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla L$ — that's a vector ($w_{\text{old}}$) plus a scaled vector ($-\eta$ times the gradient). Every single training step in every neural network is a vector addition plus a scalar multiplication.
- **Neural network layers**: a layer computes a weighted sum of inputs — this is built from scalar multiplication (weights scale inputs) and addition (summing them up).
- **Embeddings**: averaging word vectors to represent a sentence is repeated vector addition, then scaling (dividing by count).
- **Momentum-based optimizers** (like Adam) combine the current gradient and a running average of past gradients — again, scaled vector additions.

## 8. Common interview questions and pitfalls

- **Pitfall**: thinking vector addition works like multiplication (multiplying components) — it doesn't; addition is always componentwise addition, never componentwise multiplication (that's a different operation, called the Hadamard product, used in things like LSTMs).
- **Pitfall**: forgetting that vectors must have the **same dimension** to be added. You can't add a vector in $\mathbb{R}^2$ to one in $\mathbb{R}^3$.
- **Interview flavor**: "In gradient descent, why do we subtract the gradient instead of adding it?" (Tests whether you understand that scalar multiplication by a negative number flips direction — subtracting the gradient moves you *opposite* to the direction of steepest increase, i.e., downhill.)


---

## Conceptual Questions Answered

**1. Why is scalar multiplication by -1 equivalent to "flipping" a vector, not moving it somewhere else?**

`-1 × (x, y) = (-x, -y)` points in the **exact opposite direction** but starts from the same point. It doesn't change location—it reverses orientation by 180°. If vector is "3 right, 2 up," `-1×` makes it "3 left, 2 down" from the same origin. The arrow stays in the same line but points backward.

---

**2. Why do gradient descent updates involve both addition and scalar multiplication (learning rate η)?**

Gradient descent needs two things:
- **Which direction** to move: the gradient `∇L` tells us (direction of steepest increase)
- **How far** to move: the learning rate `η` scales the step size

The update `w - η·∇L`:
- Scalar multiplication (`η × gradient`) controls step length
- Vector addition (`w + negative_gradient`) applies the move

If you only had addition without scaling, you'd overshoot. If you only had scaling without addition, you'd just stretch the vector in place. Both are needed: **scale determines distance, addition determines displacement.**

---

## Numerical Practice Answered

**1. Given `a = (2, -1, 4)` and `b = (-3, 5, 0)`:**

- **a + b** = `(2-3, -1+5, 4+0)` = `(-1, 4, 4)`
- **3a** = `(3×2, 3×(-1), 3×4)` = `(6, -3, 12)`

**2. If `v = (6, -2)`, find scalar c such that `c·v = (-3, 1)`:**

We need `c×6 = -3` and `c×(-2) = 1`.

From first equation: `c = -3/6 = -0.5`

Check second: `-0.5 × (-2) = 1` ✓

**Answer:** `c = -0.5` (negative scalar flips direction and shrinks to half size)

---

## Interview-Style Answer

**"You're implementing gradient descent and notice that after several updates, your weight vector is oscillating wildly instead of converging. Using only the vector addition/scalar multiplication update rule, what part of the equation would you suspect first, and why?"**

**I'd suspect the learning rate η (the scalar) first.**

The update rule is: `w_new = w_old - η·∇L`

- **η too large**: Each scalar multiplication makes the gradient step too big. The vector jumps past the minimum, then the next gradient points the opposite direction, so you overshoot back. This creates oscillation—the weight vector ping-pongs.

- **The vector addition part** isn't the issue—we need to add the displacement to move. The scalar controls step size, which directly causes oscillation when too aggressive.

**Fix**: Reduce η (scalar) to shrink step sizes, or use adaptive methods like Adam that scale gradients differently per component.

**Key insight**: This tests whether you understand that the scalar controls *magnitude* of change, and excessive magnitude causes overshoot in the vector space.










## 5. Mathematical formulation, symbol by symbol

**Vector addition:**
$$a + b = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1+b_1 \\ a_2+b_2 \end{bmatrix}$$

- $a_1, a_2$ — the individual components of vector $a$ (subscripts just mean "1st entry," "2nd entry")
- Addition happens **componentwise** — you add matching positions, nothing more exotic than that.

**Scalar multiplication:**
$$c \cdot a = c \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} = \begin{bmatrix} c\,a_1 \\ c\,a_2 \end{bmatrix}$$

- $c$ — a **scalar**, meaning just a single plain number (as opposed to a vector). The word "scalar" is used specifically because it "scales" the vector's length.

## 6. Geometric interpretation

- **Addition** = combining two displacements into one net displacement (this is literally how physicists add forces or velocities).
- **Scalar multiplication** = resizing a vector along its own line. $c>1$ stretches it, $0<c<1$ shrinks it, $c<0$ flips it, and $c=0$ collapses it to a point (the zero vector).
- Together, these two operations let you build **any** vector that's a "weighted combination" of a few starting vectors — which is exactly the idea behind our next topic, *linear combinations*, so keep this picture in mind.

## 7. Why it matters for AI/ML

- **Gradient descent** updates a model's weights using exactly this operation: $w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla L$ — that's a vector ($w_{\text{old}}$) plus a scaled vector ($-\eta$ times the gradient). Every single training step in every neural network is a vector addition plus a scalar multiplication.
- **Neural network layers**: a layer computes a weighted sum of inputs — this is built from scalar multiplication (weights scale inputs) and addition (summing them up).
- **Embeddings**: averaging word vectors to represent a sentence is repeated vector addition, then scaling (dividing by count).
- **Momentum-based optimizers** (like Adam) combine the current gradient and a running average of past gradients — again, scaled vector additions.

## 8. Common interview questions and pitfalls

- **Pitfall**: thinking vector addition works like multiplication (multiplying components) — it doesn't; addition is always componentwise addition, never componentwise multiplication (that's a different operation, called the Hadamard product, used in things like LSTMs).
- **Pitfall**: forgetting that vectors must have the **same dimension** to be added. You can't add a vector in $\mathbb{R}^2$ to one in $\mathbb{R}^3$.
- **Interview flavor**: "In gradient descent, why do we subtract the gradient instead of adding it?" (Tests whether you understand that scalar multiplication by a negative number flips direction — subtracting the gradient moves you *opposite* to the direction of steepest increase, i.e., downhill.)
