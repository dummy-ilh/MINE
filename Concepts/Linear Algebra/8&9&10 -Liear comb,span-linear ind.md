

## Chapter 8: Linear Combinations

### 1. What problem does this concept solve?

We now know how to add vectors and scale them. Linear combinations answer a bigger question: **what is the full set of vectors we can possibly create just by adding and scaling a given set of starting vectors?** This turns out to be one of the most important ideas in all of linear algebra — it's the concept underlying span, basis, and rank, all of which determine what a model *can* and *cannot* represent.

### 2. Intuition

Imagine you have two ingredients: flour and water. By mixing different *amounts* of each (some flour, some water — scaling), and combining them (adding), you can make many different doughs. A linear combination is exactly this: take your "ingredient" vectors, scale each by some amount, and add them together to produce a new vector. The question of "what doughs can I make with these two ingredients" is exactly the question of "span" — coming next chapter.

### 3. Small numerical example

Let $v_1 = (1, 0)$ and $v_2 = (0, 1)$. A linear combination is:

$$3v_1 + 2v_2 = 3(1,0) + 2(0,1) = (3,0) + (0,2) = (3,2)$$

Here, 3 and 2 are called the **coefficients**. Notice: with just these two vectors and different coefficients, you can reach *any* point in $\mathbb{R}^2$ — that's a preview of what "span" means.

A second, less trivial example: let $v_1 = (1,1)$, $v_2 = (2, 0)$. Is $(4, 2)$ a linear combination of these?

We need $c_1, c_2$ such that $c_1(1,1) + c_2(2,0) = (4,2)$. From the second component: $c_1 = 2$. Plug into the first: $2 + 2c_2 = 4 \Rightarrow c_2 = 1$. So yes: $2v_1 + 1v_2 = (4,2)$. ✅### 4. Visual explanation

The diagram above shows exactly the second example: scale $v_1$ by 2, scale $v_2$ by 1, place them tip to tail, and the result lands exactly at $(4,2)$. Every linear combination is this same process — scale each vector, then chain them together.

### 5. Mathematical formulation, symbol by symbol

$$w = c_1v_1 + c_2v_2 + \dots + c_kv_k$$

- $v_1, v_2, \dots, v_k$ — a set of given vectors
- $c_1, c_2, \dots, c_k$ — scalar **coefficients**, one per vector, which can be any real numbers (including 0 or negative)
- $w$ — the resulting vector, called "a linear combination of $v_1, \dots, v_k$"

### 6. Geometric interpretation

A linear combination is a recipe for reaching *some specific point* in space using only your starting vectors as "directions you're allowed to move in," with distances controlled by the coefficients. Different coefficient choices reach different points — and the natural next question (span, next chapter) is: which points can you reach at all, using every possible coefficient combination?

### 7. Why it matters for AI/ML

- **Neural network layer outputs**: every value coming out of a layer's linear transformation is a linear combination of the inputs, weighted by that neuron's row of weights.
- **PCA**: each principal component score for a data point is a linear combination of the original features.
- **Basis representations / embeddings**: representing a word or feature vector as a combination of "latent factors" is fundamentally about expressing it as a linear combination of basis vectors.
- **Feature engineering / interpretability**: understanding a model's prediction as a weighted (linearly combined) sum of input features is the entire basis of linear regression's interpretability.

### 8. Common interview questions and pitfalls

- **Pitfall**: confusing "linear combination" with just "combination" — coefficients can be *any* real number (positive, negative, fractional, zero), not just simple counts or positive weights.
- **Pitfall**: not recognizing that matrix-vector multiplication *is* a linear combination of columns — many candidates only know the "row dot product" view (Chapter 7) and get stuck when asked to reason about which vectors a matrix can produce as outputs.
- **Interview flavor**: "Express $Ax$ as a linear combination — what are you combining, and what are the coefficients?" (Answer: you're combining the *columns* of $A$, and the *entries of $x$* are the coefficients.)

### 9. Summary

A linear combination combines a set of vectors by scaling each one and adding the results together, and it's the same mechanical process as vector addition and scalar multiplication, just applied to more than one vector at once. Crucially, matrix-vector multiplication can itself be viewed as a linear combination of the matrix's columns — a reframing that's essential for understanding span, basis, and rank next.

---

**Conceptual questions:**
1. Why is matrix-vector multiplication $Ax$ considered "a linear combination of the columns of $A$"? What plays the role of the coefficients?
2. Is $(0,0)$ always reachable as a linear combination of any set of vectors? Why?

**Numerical practice:**
1. Is $(5, 1)$ a linear combination of $v_1=(1,1)$ and $v_2=(2,-1)$? Find the coefficients if so.
2. Given $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $x=(2,1)$, write $Ax$ explicitly as a linear combination of $A$'s columns, then compute the result.

**Interview-style question:**
"You're told a certain output vector cannot be produced as any linear combination of a neural network layer's weight columns, no matter what input you provide. What does this tell you about the layer's capacity to represent that output, and what concept (coming up next) would formally describe this limitation?"


**Conceptual Q1:** Why is $Ax$ a linear combination of $A$'s columns?
Matrix-vector multiplication can be rewritten as: $Ax = x_1 \cdot (\text{column 1 of }A) + x_2 \cdot (\text{column 2 of }A) + \dots$. The **columns of $A$** are the vectors being combined, and the **entries of $x$** are the coefficients scaling each column.

**Conceptual Q2:** Is $(0,0)$ always reachable as a linear combination of any set of vectors?
Yes — always. Set every coefficient to 0: $0\cdot v_1 + 0\cdot v_2 + \dots = (0,0)$, regardless of what the vectors are. This is called the "trivial" linear combination, and it's why the zero vector is always in the span of any set of vectors (a fact that becomes important in the next chapter).

**Numerical Q1:** Is $(5,1)$ a linear combination of $v_1=(1,1)$, $v_2=(2,-1)$?
Solve $c_1(1,1)+c_2(2,-1)=(5,1)$:
$c_1+2c_2=5$ and $c_1-c_2=1$
From the second equation: $c_1 = 1+c_2$. Substituting: $1+c_2+2c_2=5 \Rightarrow 3c_2=4 \Rightarrow c_2=\frac{4}{3}$, so $c_1=\frac{7}{3}$.
Check: $\frac{7}{3}(1,1)+\frac{4}{3}(2,-1) = (\frac{7}{3}+\frac{8}{3},\ \frac{7}{3}-\frac{4}{3}) = (5, 1)$ ✅ Yes, it works.

**Numerical Q2:** Write $Ax$ as a linear combination of columns, for $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$, $x=(2,1)$
Columns of $A$: $\text{col}_1=(1,3)$, $\text{col}_2=(2,4)$.
$$Ax = 2\cdot(1,3) + 1\cdot(2,4) = (2,6)+(2,4) = (4,10)$$

**Interview question:** An output vector can't be produced by any linear combination of a layer's weight columns — what does this mean?
It means that output vector lies **outside the span** of the layer's weight columns — no matter what input you feed in, the layer is mathematically incapable of producing that exact output, because its outputs are restricted to whatever combinations its weight columns can form. This limitation is formally described by **span** (and the closely related concept of **rank**) — exactly the topic we're moving to now.

---

## Chapter 9: Span and Basis

### 1. What problem does this concept solve?

We just saw that a layer's possible outputs are limited to linear combinations of its weight columns — not every vector is reachable. Span formalizes this idea precisely: **given a set of vectors, what is the entire collection of points you can reach using every possible linear combination of them?** Basis then asks: **what's the smallest, most efficient set of vectors needed to reach that same collection, with no redundancy?**

### 2. Intuition

Back to the flour-and-water analogy: with only flour and water, you can make a huge variety of doughs — but you'll never make a genuine chocolate cake, no matter how you vary the ratio, because chocolate isn't reachable from those two ingredients. The **span** of {flour, water} is "all doughs of that specific consistency spectrum" — a limited universe of possibilities. A **basis** is the minimum, non-redundant ingredient list: if you had "flour, water, and 2x water" as your three ingredients, that third one is redundant — it adds no new reachable doughs, since it's already a scaled version of something you have.### 3. Small numerical example

Let $v_1 = (1, 0)$ and $v_2 = (0, 1)$. Their **span** is *all* of $\mathbb{R}^2$ — for any target point $(a,b)$, just pick coefficients $c_1=a$, $c_2=b$: $a(1,0)+b(0,1) = (a,b)$. Since $\{v_1, v_2\}$ can reach every point using exactly 2 vectors with no redundancy, they form a **basis** for $\mathbb{R}^2$.

Now let $v_1 = (1,1)$ and $v_2 = (2,2)$. Notice $v_2 = 2v_1$ — it's just a scaled copy. Their span is only the **line** through the origin in the direction $(1,1)$ — you can never leave that line no matter what coefficients you pick, since both vectors point the same way. This pair is *not* a basis for $\mathbb{R}^2$ (it can't reach all of it), even though it has 2 vectors.

### 4. Visual explanation

The diagram above shows this exact contrast: two non-parallel vectors span the entire plane (left), while two parallel vectors — even though there are two of them — only span a single line (right), because the second vector adds no new direction.

### 5. Mathematical formulation, symbol by symbol

**Span:**
$$\text{span}(v_1, \dots, v_k) = \{c_1v_1 + c_2v_2 + \dots + c_kv_k \mid c_1, \dots, c_k \in \mathbb{R}\}$$

- This reads: "the span of $v_1$ through $v_k$ is the set of all possible linear combinations of them, using any real-number coefficients."
- $\mathbb{R}$ — the set of all real numbers (so coefficients can be anything: fractions, negatives, zero).

**Basis**: a set of vectors is a basis for a space if it satisfies **two conditions**:
1. It **spans** the space (can reach every point in it).
2. It is **linearly independent** (no vector in the set is redundant — none can be written as a combination of the others). We'll formally define linear independence next chapter, but informally: no "wasted" vectors like our $(2,2)$ example above.

### 6. Geometric interpretation

- Span answers: "what's the full shape (a point, a line, a plane, all of space, etc.) that these vectors can reach?"
- A basis is the minimal "coordinate system" for that shape — think of the x and y axes as the standard basis for the 2D plane; every point can be described uniquely using just those two directions.
- **Dimension** of a space is defined as the number of vectors in *any* basis for it — this is the formal definition of "how many independent directions" a space has.

### 7. Why it matters for AI/ML

- **Model capacity**: the span of a neural network layer's weight columns determines what outputs it can possibly produce — if the true target function requires reaching outside that span, the model simply cannot represent it, no matter how it's trained (this connects directly back to your Chapter 8 interview question).
- **Embeddings**: a well-trained embedding space should ideally have basis directions that correspond to meaningful, independent semantic concepts — this is the intuition behind "disentangled representations."
- **PCA**: finds a new, smaller basis (the principal components) that still spans (approximately) the meaningful variation in your data, letting you represent data with fewer dimensions.
- **Feature redundancy**: if two features in your dataset are linearly dependent (redundant, like a scaled copy of each other), they don't expand your model's representational span — this is directly related to multicollinearity in regression.

### 8. Common interview questions and pitfalls

- **Pitfall**: assuming "more vectors" always means "a bigger span." Adding a redundant vector (in the span of existing ones) doesn't expand span at all — as shown in our second example.
- **Pitfall**: confusing "span" (a *set* of reachable points) with "basis" (a specific *minimal set of vectors* used to describe that set). Many bases can describe the same span.
- **Interview flavor**: "You have 5 feature vectors describing your dataset, but you're told the effective dimensionality is only 3. What does this tell you about the span of those 5 vectors?" (Answer: the 5 vectors span only a 3-dimensional subspace — meaning at least 2 of them are redundant/linearly dependent on the others.)

### 9. Summary

The span of a set of vectors is the entire collection of points reachable through their linear combinations — it might be a line, a plane, or all of space, depending on whether the vectors add genuinely new directions. A basis is the smallest, non-redundant set of vectors that spans a given space, and the number of vectors in a basis defines that space's dimension. This idea underlies model capacity, feature redundancy, and dimensionality reduction techniques like PCA.

---

**Conceptual questions:**
1. Can 3 vectors in $\mathbb{R}^2$ ever form a basis for $\mathbb{R}^2$? Why or why not?
2. If a set of vectors spans $\mathbb{R}^3$ but contains 5 vectors, what does that tell you about whether it's a basis?

**Numerical practice:**
1. Does $\{(1,2), (3,6)\}$ span all of $\mathbb{R}^2$? Justify your answer.
2. Find a basis for the span of $\{(1,0,0), (0,1,0), (1,1,0)\}$ — how many vectors does it actually need?

**Interview-style question:**
"A colleague claims that adding more neurons to a hidden layer always increases the range of functions the network can represent. Using the concept of span, explain under what condition this claim could be false."

This final version matches exactly what we've been doing — I'll keep going chapter by chapter with this exact structure, covering the full topic list through matrix calculus. Here are Chapter 9's answers, then straight into Linear Independence.

**Conceptual Q1:** Can 3 vectors in $\mathbb{R}^2$ ever form a basis for $\mathbb{R}^2$?
No. A basis for $\mathbb{R}^2$ must contain exactly 2 vectors, because the dimension of $\mathbb{R}^2$ is 2 (the number of vectors in *any* basis equals the space's dimension). Any 3 vectors in a 2D space are automatically linearly dependent — one of them can always be written as a combination of the other two — so they fail the "no redundancy" requirement of a basis, even if together they still span the whole plane.

**Conceptual Q2:** A set of 5 vectors spans $\mathbb{R}^3$ — is it a basis?
No, it can't be a basis. A basis for $\mathbb{R}^3$ needs exactly 3 vectors. Since this set has 5 but only needs 3 to span the space, at least 2 of the 5 are redundant (linearly dependent on the others). The set spans the space, but it's not minimal, so it's not a basis — though it does *contain* a basis as a subset.

**Numerical Q1:** Does $\{(1,2),(3,6)\}$ span all of $\mathbb{R}^2$?
No — notice $(3,6) = 3\times(1,2)$, so the second vector is just a scaled copy of the first. Both point in the same direction, so their span is only the line through the origin in that direction, not the full plane.

**Numerical Q2:** Basis for span of $\{(1,0,0),(0,1,0),(1,1,0)\}$
Notice $(1,1,0) = (1,0,0)+(0,1,0)$ — the third vector is redundant. A basis needs only $\{(1,0,0),(0,1,0)\}$ — just 2 vectors are enough to span this set (which is the entire xy-plane sitting inside $\mathbb{R}^3$).

**Interview question:** When could adding more neurons NOT increase representable functions?
If the new neurons' weight vectors are linear combinations of the existing neurons' weight vectors (i.e., they lie in the span of the current weights), they add no new independent direction to the layer's output space — the span of the (pre-activation) outputs stays exactly the same size despite having more neurons. In a purely linear layer, this means the extra neurons are completely redundant. (With a nonlinear activation, extra redundant-weight neurons can still occasionally help slightly by adding extra nonlinear "bends," but the core linear-algebra argument — no new output span — is the answer being tested here.)

---

## Chapter 10: Linear Independence

### 1. What problem does this concept solve?

We've been informally using the phrase "no redundant vectors" throughout the last two chapters. Linear independence makes that idea precise: it gives us an exact mathematical test for whether a set of vectors each contributes a genuinely new direction, or whether some of them are "wasted" — expressible using the others.

### 2. Intuition

Think of a team where each member is supposed to bring a unique skill. If one team member's contribution can be fully replicated by combining two other members' skills, that person is redundant — the team's *effective* capability doesn't shrink if you remove them. Linear independence checks exactly this for vectors: is every vector in the set truly "irreplaceable," or can at least one be reconstructed from the others?

### 3. Small numerical example

Take $v_1 = (1,0)$, $v_2=(0,1)$. Is there any way to write one as a combination of the other? No — you'd need $c \cdot (1,0) = (0,1)$, which is impossible for any real $c$. These vectors are **linearly independent**.

Now take $v_1=(1,2)$, $v_2=(2,4)$. Notice $v_2 = 2v_1$. This means:
$$2v_1 - v_2 = (2,4)-(2,4) = (0,0)$$

We found a way to combine them (with coefficients not all zero: 2 and $-1$) to get the zero vector. This is exactly the formal definition of **linear dependence**.### 4. Visual explanation

The diagram above shows the test visually: independent vectors point in genuinely different directions (left), while dependent vectors lie along the same line (right) — meaning you can combine them (using nonzero coefficients) to cancel out and land exactly on the zero vector.

### 5. Mathematical formulation, symbol by symbol

A set of vectors $v_1, v_2, \dots, v_k$ is **linearly independent** if the *only* solution to:

$$c_1v_1 + c_2v_2 + \dots + c_kv_k = 0$$

is the **trivial solution** $c_1=c_2=\dots=c_k=0$.

- $0$ here means the **zero vector** (all-zero entries), not the number zero
- If you can find *any other* set of coefficients (not all zero) that also satisfies this equation, the vectors are **linearly dependent**
- This equation is called the "linear dependence relation" — finding a nonzero solution to it is literally the mathematical proof that redundancy exists

### 6. Geometric interpretation

- In 2D: two vectors are dependent if and only if they lie on the same line through the origin.
- In 3D: three vectors are dependent if they all lie in the same plane through the origin (even if no two of them individually lie on the same line).
- Independence means every new vector you add genuinely expands the span into a new dimension; dependence means at least one vector is "trapped" inside the span of the others, adding nothing new.

### 7. Why it matters for AI/ML

- **Multicollinearity in regression**: if two or more input features are linearly dependent (or nearly so — "highly correlated"), the regression's coefficient estimates become unstable and hard to interpret, because the model literally cannot uniquely determine how much credit to assign to each redundant feature.
- **Rank and invertibility** (next chapter): a matrix's columns being linearly independent is exactly the condition needed for that matrix to be invertible — a huge deal for solving linear systems and understanding when models have unique solutions.
- **Neural network weight redundancy**: if two neurons in a layer end up learning linearly dependent weight vectors during training, they contribute no more representational power than one neuron would — a real, empirically observed phenomenon.
- **PCA**: works by finding a smaller *linearly independent* set of directions (principal components) that captures most of the data's variation, discarding redundant dependent directions.

### 8. Common interview questions and pitfalls

- **Pitfall**: assuming you need to check "is one vector exactly equal to a specific other vector" — linear dependence is about combinations, not just direct equality. E.g., $(1,1,0)$, $(0,1,1)$, $(1,2,1)$ are dependent because $(1,2,1) = (1,1,0)+(0,1,1)$, even though none of them is a simple scalar multiple of another.
- **Pitfall**: confusing "linearly independent" with "orthogonal" (perpendicular) — orthogonal vectors are always linearly independent (as long as none is the zero vector), but linearly independent vectors don't have to be orthogonal at all. Orthogonality is a *stronger*, more specific condition, coming a few chapters ahead.
- **Interview flavor**: "Given 3 vectors in $\mathbb{R}^3$, how would you check if they're linearly independent using a matrix operation?" (Answer: form a matrix with these vectors as columns and check if its determinant is nonzero, or equivalently check if its rank is 3 — this is the practical, computational version of the independence test.)

### 9. Summary

A set of vectors is linearly independent if the only way to combine them into the zero vector is by using all-zero coefficients — meaning none of them is redundant or "hiding" inside the span of the others. Geometrically, dependent vectors collapse onto a shared line or plane, while independent vectors each carve out a genuinely new direction. This concept is the precise foundation for rank, invertibility, and understanding when models have stable, well-determined solutions.

---

**Conceptual questions:**
1. If one vector in a set is the zero vector, are the vectors automatically linearly dependent? Why?
2. Why does linear independence matter for whether a matrix can be inverted (a preview of the next few chapters)?

**Numerical practice:**
1. Are $(1,2,3)$, $(2,4,6)$, and $(1,0,0)$ linearly independent? Justify your answer.
2. Are $(1,0)$, $(0,1)$, and $(1,1)$ linearly independent? What does this tell you about the maximum number of independent vectors possible in $\mathbb{R}^2$?

**Interview-style question:**
"During training, you notice two neurons in the same hidden layer have nearly identical weight vectors (differing only by a tiny scaling factor). Using the concept of linear independence, explain why this might indicate wasted model capacity, and what you might do about it."




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
