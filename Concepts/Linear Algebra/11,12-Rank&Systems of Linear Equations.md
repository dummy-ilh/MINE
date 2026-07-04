# Linear Combinations, Span, Basis, and Independence - Concise Summary

## Linear Combinations

A **linear combination** of vectors is a sum of scaled vectors:

$$c_1v_1 + c_2v_2 + ... + c_kv_k$$

where each $c_i$ is a scalar.

**Example**: `3(1,0) + 2(0,1) = (3,2)` is a linear combination of basis vectors.

---

## Span

The **span** of a set of vectors is the set of all possible linear combinations.

**Geometric interpretation**:
- One vector: span = line through origin
- Two non-parallel vectors in ℝ²: span = entire plane
- Two vectors in ℝ³: span = plane through origin
- n linearly independent vectors in ℝⁿ: span = entire ℝⁿ

**Key insight**: Span answers "what space can we reach?"

---

## Linear Independence

A set of vectors is **linearly independent** if no vector can be written as a linear combination of the others.

**Formal definition**: $c_1v_1 + c_2v_2 + ... + c_kv_k = 0$ implies **all** $c_i = 0$.

**Geometric tests**:
- Two vectors: independent if not scalar multiples (not collinear)
- Three vectors: independent if not coplanar
- In general: independent if you can't remove one without losing "coverage"

**Dependent example**: `(1,2)`, `(2,4)`, `(3,6)` all lie on same line → dependent.

---

## Basis

A **basis** is a set of vectors that is:
1. **Linearly independent**
2. **Spans** the entire space

**Standard basis in ℝⁿ**:
- ℝ²: `e₁ = (1,0)`, `e₂ = (0,1)`
- ℝ³: `e₁ = (1,0,0)`, `e₂ = (0,1,0)`, `e₃ = (0,0,1)`

**Key properties**:
- Every vector has a **unique** representation in a given basis
- All bases for a space have the same number of vectors (the dimension)

---

## Dimension

The **dimension** of a vector space = number of vectors in any basis.

**For matrices**:
- Column space dimension = rank
- Nullspace dimension = n - rank

---

## Why for AI/ML

- **Feature engineering**: Choose basis/features that span the right space
- **Dimensionality reduction**: PCA finds new basis (principal components)
- **Neural network capacity**: Weight matrices create linear combinations
- **Regularization**: Penalizes certain linear combinations
- **Embeddings**: Words as linear combinations of latent features

---

## Common Pitfalls

- **❌** Thinking "span" = "all vectors in the set"
- **❌** Assuming any set of n vectors in ℝⁿ is a basis (need independence!)
- **❌** Forgetting that dependent vectors "waste" dimensions
- **❌** Confusing basis with orthogonal basis (basis = independent, not necessarily perpendicular)

---

## Conceptual Questions Answered

**1. If `v₁ = (1,0)` and `v₂ = (2,0)`, what is their span? Are they independent?**

**Span**: All vectors `(c, 0)` where c is any scalar = the x-axis (line through origin).

**Independence**: **No** - `v₂ = 2v₁`, so they're scalar multiples. Only one direction covered.

**Geometric**: Both lie on same line → can't reach y-axis no matter how you combine them.

---

**2. What does it mean for a set of vectors to "span" a space? Give an example of vectors that span ℝ².**

**Spanning**: Every vector in the space can be written as a linear combination of the set.

**Example spanning ℝ²**: `(1,0)` and `(0,1)`
- Any `(x,y)` = `x(1,0) + y(0,1)`

**Counterexample**: `(1,0)` alone does NOT span ℝ² (can't reach `(0,1)`).

---

**3. Is it possible for a set of 3 vectors in ℝ³ to be linearly dependent? If so, what does that mean geometrically?**

**Yes** - three vectors are dependent if they lie in the same plane through origin (coplanar).

**Geometric meaning**: You only need 2 of the 3 to reach any point in the plane. The third is a linear combination of the other two.

**Example**: `(1,0,0)`, `(0,1,0)`, `(1,1,0)` are dependent (third = first + second). All lie in xy-plane.

---

## Numerical Practice Answered

**1. Is `v₁ = (1,0)`, `v₂ = (0,1)` a basis for ℝ²? Why?**

**Yes** because:
- **Independent**: `c₁(1,0) + c₂(0,1) = (0,0)` → `(c₁, c₂) = (0,0)` → c₁ = c₂ = 0 ✓
- **Spans ℝ²**: Any `(x,y) = x(1,0) + y(0,1)` ✓

Two independent vectors in ℝ² = basis.

---

**2. Write `v = (3,5)` as a linear combination of `(1,0)` and `(0,1)`**

`(3,5) = 3(1,0) + 5(0,1)`

Coefficients: c₁ = 3, c₂ = 5.

---

**3. Are `(1,2,3)`, `(2,4,6)`, `(0,1,1)` linearly independent?**

**Check**: `(2,4,6) = 2(1,2,3)` → these are dependent!

Since the second vector is a scalar multiple of the first, the set is dependent regardless of the third vector.

**Test**: c₁(1,2,3) + c₂(2,4,6) + c₃(0,1,1) = (0,0,0)
- Pick c₁ = 2, c₂ = -1, c₃ = 0 → `2(1,2,3) + (-1)(2,4,6) = (0,0,0)` ✓
- Non-zero coefficients exist → dependent.

---

## Interview-Style Answer

**"Your team uses a neural network with a 500-dimensional embedding layer. A colleague suggests that since you only have 200 distinct words, the embedding vectors must be linearly dependent and you're wasting dimensions. Are they correct? Why or why not?"**

### The Short Answer

**They're likely wrong** - embedding dimension and vocabulary size are different concepts.

### The Detailed Explanation

**What they're missing**: Even with 200 words, you can choose embedding vectors that are linearly independent in ℝ⁵⁰⁰.

**Key points**:

1. **Dimension ≠ vocabulary size**: You can have 200 independent vectors in 500-dimensional space (they just live in a 200-dimensional subspace).

2. **Each word gets its own vector**: The vectors are not inherently dependent just because there are fewer words than dimensions.

3. **Training determines dependence**: The optimizer will find some relationship, but with 500D and 200 words, there's room for all to be independent.

4. **Even if dependent, that's not "wasted"**: Independence is just one property. Word embeddings encode semantic relationships - similar words are close together.

**The real question**: Is 500D necessary for 200 words?
- Yes if you need rich semantic representations
- No if you're overparameterizing (could use 200D)
- It depends on the task and data

**Common misconception**: Thinking "number of vectors" = "dimension needed for independence." You can have arbitrary many independent vectors in a fixed dimension.

### Interview Follow-up

**"When would the colleague be right?"**

If embedding matrix has shape `(vocab_size, dim)` with rank < vocab_size (i.e., some words are linear combinations of others). This happens if:
- Embedding initialization has linear dependencies
- Training collapses similar words

**Why it matters**: Low effective rank = less representational capacity.

---

## Additional Linear Algebra Interview Questions

### Q: What is the relationship between linear independence and solving linear systems?

**Independent columns** = unique solution (if any) to `Ax = b`.
**Dependent columns** = infinite solutions or no solution.

**Example**: `x + y = 1` and `2x + 2y = 2` (dependent equations) → infinite solutions.

---

### Q: How do you check if a set of vectors is linearly independent in practice?

**Method 1**: Form matrix with vectors as columns, compute determinant (if square) or rank.
- Full rank → independent
- Rank < number of vectors → dependent

**Method 2**: Solve `c₁v₁ + ... + cₖvₖ = 0` for c_i.
- Only all-zero solution → independent
- Non-zero solution → dependent

**In code**: Use `np.linalg.matrix_rank()` - if rank == number of vectors, they're independent.

---

### Q: Why is PCA finding a basis important?

PCA finds new basis vectors (principal components) that:
1. Capture maximum variance
2. Are orthogonal (perpendicular)
3. Are ordered by importance

**Why it matters**: You can represent data with fewer dimensions by projecting onto top k principal components.

---

### Q: What's the difference between a basis and an orthogonal basis?

**Basis**: Independent + spans space (vectors can be at any angles).

**Orthogonal basis**: Basis where vectors are perpendicular (dot product = 0).

**Orthonormal basis**: Orthogonal + all vectors have length 1.

**Why orthonormal matters**: 
- Normalization is simple (just dot product)
- Projection is trivial: `proj_v(u) = (u·v)v`
- PCA gives orthonormal basis

---

### Q: In linear regression, why is feature independence important?

**Multicollinearity** = features are linearly dependent (correlated).

**Problems**:
- Unstable coefficient estimates
- Hard to interpret individual feature importance
- Can cause numerical issues in matrix inversion

**Fix**: Use PCA, remove redundant features, or use regularization.

---

### Q: What is the "basis" of a neural network's representation?

**Different meanings**:

1. **Input basis**: Standard basis features (raw pixels, word counts)
2. **Hidden basis**: Activation vectors that span the representation space
3. **Concept basis**: Interpretable features discovered by model

**In autoencoders**: Learned basis represents data more efficiently.

---

### Q: If two embeddings are linearly independent, does that mean they're dissimilar?

**Not necessarily** - independence doesn't imply dissimilarity.

**Example**: `(1,0.1)` and `(0.1,1)` are nearly independent but semantically similar.

**What independence tells you**: They capture different "directions" in the feature space. Similarity requires additional metrics like cosine similarity or Euclidean distance.

---

### Q: Why do transformers use 768+ dimensions if many embeddings might be dependent?

**Key insight**: Even if dimensions aren't linearly independent, they can still be useful!
- **Independence isn't everything**: Redundancy helps with robustness
- **Semantic richness**: More dimensions allow fine-grained distinctions
- **Optimization**: Higher dimensions can help gradient flow

**Practical reality**: Most embeddings in large models are effectively rank-deficient (low effective rank) but still valuable.


This is the finalized, consolidated version of our instructions — matches exactly what we've been doing, so I'll keep going chapter by chapter through the full topic list (through matrix calculus). Here are Chapter 11's answers, then straight into Systems of Linear Equations.

**Conceptual Q1:** Why can rank never exceed $\min(m,n)$?
Rank counts linearly independent columns (or rows). You can have at most as many independent columns as there are columns ($n$), and at most as many independent rows as there are rows ($m$) — and since row rank always equals column rank, the shared number is capped by whichever of $m, n$ is smaller.

**Conceptual Q2:** A $5\times5$ matrix has rank 3 — is it invertible?
No. A square matrix is invertible only if it has **full rank** — for a $5\times5$ matrix, that means rank 5. Rank 3 means the matrix is rank-deficient (2 dimensions' worth of information are missing/redundant), so it's not invertible.

**Numerical Q1:** Rank of $\begin{bmatrix}1&2&3\\2&4&6\end{bmatrix}$
Notice row 2 = 2 × row 1 (or equivalently, every column is a multiple of $(1,2)$: column 2 = 2×column 1, column 3 = 3×column 1). All rows/columns are dependent on one direction, so $\text{rank} = 1$.

**Numerical Q2:** Maximum possible rank of a $3\times5$ matrix
$\min(3,5) = 3$, so the maximum possible rank is **3**.

**Interview question:** User-item matrix (millions × thousands) has effective rank ~50 — what does this suggest?
It suggests that despite having millions of users and thousands of items, user preferences aren't truly that varied — they can be explained by roughly 50 independent underlying "taste dimensions" or latent factors (e.g., genre preference, price sensitivity, brand loyalty). Everyone's preferences are effectively a linear combination of these ~50 basis patterns. This directly justifies matrix factorization: instead of storing/learning the full matrix, you can approximate it as the product of two much smaller matrices (users × 50, and 50 × items), massively reducing parameters while capturing almost all the meaningful signal — exactly the same "low intrinsic rank" idea behind LoRA in Chapter 11.

---

## Chapter 12: Systems of Linear Equations

### 1. What problem does this concept solve?

A huge number of real problems boil down to: "I have several conditions that must all hold simultaneously — what values satisfy all of them at once?" A system of linear equations is the formal way to express multiple linear conditions together, and understanding when such a system has no solution, one solution, or infinitely many is essential — this is *exactly* what training a linear model is trying to do.

### 2. Intuition

Imagine two friends telling you where to meet: one says "I'll be somewhere on this street," the other says "I'll be somewhere on that street." If the streets cross at exactly one point, there's exactly one meeting spot (one solution). If the streets are actually the same street just described differently, any point on it works (infinitely many solutions). If the streets are parallel and never cross, there's no way to meet at all (no solution). Systems of linear equations are just this idea generalized to many dimensions.### 3. Small numerical example

Consider:
$$x + y = 5$$
$$x - y = 1$$

Written as a matrix equation:
$$\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 5 \\ 1 \end{bmatrix}$$

Adding the two original equations: $2x = 6 \Rightarrow x=3$. Substituting back: $3+y=5 \Rightarrow y=2$. So the unique solution is $(x,y) = (3,2)$ — this matrix has full rank (2), matching our left-panel picture: two lines crossing at exactly one point.

Now consider a dependent system:
$$x + y = 5$$
$$2x + 2y = 10$$

The second equation is just the first multiplied by 2 — same line, described twice. This has **infinitely many solutions** (any point satisfying $x+y=5$ works). Here the coefficient matrix $\begin{bmatrix}1&1\\2&2\end{bmatrix}$ has rank 1, not 2 — rank deficiency directly signals this collapse into infinite solutions.

### 4. Visual explanation

The diagram above shows all three outcomes side by side: crossing lines (unique solution), the same line twice (infinite solutions), and parallel non-intersecting lines (no solution) — this is the complete picture for any 2-variable, 2-equation system, and it generalizes to higher dimensions with planes and hyperplanes instead of lines.

### 5. Mathematical formulation, symbol by symbol

A system of linear equations is written compactly as:

$$Ax = b$$

- $A$ — the **coefficient matrix** (each row is one equation's coefficients)
- $x$ — the vector of **unknowns** we're solving for
- $b$ — the vector of **right-hand-side constants** (the "targets" each equation must equal)

**How rank determines the outcome** (comparing $\text{rank}(A)$ to $\text{rank}([A|b])$, the "augmented matrix" formed by appending $b$ as an extra column):
- If $\text{rank}(A) = \text{rank}([A|b]) = n$ (number of unknowns) → **exactly one solution**
- If $\text{rank}(A) = \text{rank}([A|b]) < n$ → **infinitely many solutions**
- If $\text{rank}(A) < \text{rank}([A|b])$ → **no solution** (the equations contradict each other)

### 6. Geometric interpretation

Each equation defines a line (in 2D), a plane (in 3D), or a hyperplane (higher dimensions). Solving the system means finding the intersection of all these geometric objects simultaneously. Rank tells you the "effective number of independent constraints" — if you have more equations than the rank supports, some equations are either redundant (infinite solutions) or contradictory (no solution).

### 7. Why it matters for AI/ML

- **Linear regression's normal equations**: solving for the optimal weights $w$ in $X^TXw = X^Ty$ is literally solving a system of linear equations — understanding when it has a unique solution (full rank $X^TX$) versus infinitely many (rank-deficient, meaning multicollinear features) is central to understanding when regression is well-posed.
- **Training a linear model with more parameters than data points**: this is exactly the "infinitely many solutions" case — you get an underdetermined system, and additional techniques (like regularization) are needed to pick one "best" solution among infinitely many.
- **Solving for equilibrium points** in various optimization and control problems reduces to linear systems.
- **Backpropagation** ultimately relies on solving (approximately, iteratively) systems related to gradients being zero at a minimum.

### 8. Common interview questions and pitfalls

- **Pitfall**: assuming "more equations than unknowns" always guarantees a solution — it's the *opposite* concern; more equations increase the risk of contradiction (no solution) unless they're consistent.
- **Pitfall**: assuming "fewer equations than unknowns" always means infinitely many solutions — this is *usually* true (underdetermined systems), but only if the equations are consistent with each other; you can still get no solution in some degenerate cases.
- **Interview flavor**: "In linear regression, if you have more features than data points ($p > n$), what happens to the system $X^TXw=X^Ty$, and how do techniques like Ridge regression address this?" (Answer: $X^TX$ becomes rank-deficient/singular, so infinitely many $w$ satisfy the equations equally well on the training data; Ridge regression adds a penalty term that makes $X^TX + \lambda I$ full rank and invertible, picking out a single unique, well-behaved solution.)

### 9. Summary

A system of linear equations $Ax=b$ asks for values of $x$ that satisfy every equation (every row of $A$) at once, and the answer — no solution, one solution, or infinitely many — is entirely determined by comparing the rank of $A$ to the rank of the augmented matrix $[A|b]$. This framework directly explains when linear regression has a unique, well-defined solution and motivates techniques like regularization when it doesn't.

---

**Conceptual questions:**
1. Why does having more unknowns than independent equations typically lead to infinitely many solutions rather than no solution?
2. In linear regression, what does it mean intuitively for the system $X^TXw = X^Ty$ to have "no exact solution," and why does the least-squares approach handle this?

**Numerical practice:**
1. Solve the system: $2x+y=8$, $x-y=1$.
2. Does the system $x+2y=4$, $2x+4y=9$ have a solution? Why or why not?

**Interview-style question:**
"You're told a linear regression model was trained with a dataset that has perfectly duplicated columns (two features that are exact copies of each other). Explain, using the concept of rank, why the normal equations $X^TXw=X^Ty$ won't have a unique solution, and what practical effect this has on the model's learned coefficients."

Say "next" whenever you're ready for **Chapter 13: Matrix Inverse** — where we'll formally define what it means to "undo" a matrix transformation, directly building on rank and systems of equations.
