 **Chapter 4: Norms and Distance**.

## 1. What problem does this concept solve?

Last chapter, we ran into a real limitation: the dot product mixes together "angle" and "length," and we couldn't fully untangle them because we hadn't yet defined how to measure a vector's **length**. Norms solve that — they give us a precise way to answer "how big is this vector?" and, by extension, "how far apart are two points/vectors?"

## 2. Intuition

If a vector is an arrow, its **norm** is simply the length of that arrow — like measuring it with a ruler. And once you can measure length, you can measure **distance**: the distance between two points is just the length of the vector connecting them (found by subtracting one from the other).

This shows up everywhere: how far apart are two users' preference profiles, how "large" are a model's weights, how much did a gradient step move you.

## 3. Small numerical example

Take $a = (3, 4)$. Its length is found using the Pythagorean theorem — literally the same triangle formula you learned in school:

$$\|a\| = \sqrt{3^2 + 4^2} = \sqrt{9+16} = \sqrt{25} = 5$$

Now take two points $p = (1, 2)$ and $q = (4, 6)$. The distance between them is the norm of their difference:

$$q - p = (4-1,\ 6-2) = (3, 4)$$
$$\text{distance} = \|q-p\| = \sqrt{3^2+4^2} = 5$$

Notice this is the exact same triangle as before — distance is just "the norm of the difference vector."

## 5. Mathematical formulation, symbol by symbol

**The most common norm** (called the **L2 norm** or **Euclidean norm**):

$$\|a\| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2} = \sqrt{\sum_{i=1}^n a_i^2}$$

- $\|a\|$ — notation for "the norm (length) of vector $a$" (double vertical bars, to distinguish from $|a|$ which is used for a single number's absolute value, though many people use them interchangeably for norms)
- The subscript "2" in $\|a\|_2$ specifies *which* norm — this matters because there's more than one way to measure "size":

$$\|a\|_1 = |a_1| + |a_2| + \dots + |a_n| \quad \text{(L1 norm, "Manhattan distance")}$$

- $\|a\|_1$ sums up absolute values instead of squaring — think of it as "how far you'd walk along city blocks" instead of "as the crow flies."

**Distance between two vectors/points:**

$$d(p, q) = \|q - p\| = \sqrt{\sum_i (q_i - p_i)^2}$$

This is just: subtract the vectors, then take the norm of the result.

## 6. Geometric interpretation

- **L2 norm**: straight-line ("as the crow flies") distance — the length of the direct arrow, generalized Pythagorean theorem.
- **L1 norm**: distance if you can only move along grid lines (like walking city blocks) — no diagonal shortcuts allowed.
- Norms turn the abstract idea of "vector" into something you can compare with real numbers — bigger norm means longer/larger vector, by whichever definition of "large" you choose.
- Importantly: **the choice of norm changes what "closest" means**. Two points that are "closest" under L2 might not be the closest under L1.

## 7. Why it matters for AI/ML

- **Regularization** (L1/L2 regularization in linear regression, Lasso, Ridge, and neural networks): these penalize the *norm* of the weight vector to keep the model simple and avoid overfitting. L2 regularization (Ridge) shrinks weights smoothly; L1 (Lasso) tends to push some weights exactly to zero, causing automatic feature selection.
- **Gradient descent convergence**: we often check the norm of the gradient vector to decide when to stop training (a very small norm means we're near a minimum, since the "slope" is flat).
- **K-Nearest Neighbors / clustering**: literally defined by measuring distances (norms) between data points.
- **Embeddings similarity**: distance between two embeddings (using L2 norm) is a common alternative to cosine similarity.
- **Batch/Layer normalization** in neural networks: normalizing activations relies on computing norms/magnitudes of vectors.

## 8. Common interview questions and pitfalls

- **Pitfall**: assuming "norm" only ever means the Euclidean (L2) length — interviewers often test whether you know L1 vs. L2 differ and *why* you'd pick one over the other (e.g., L1 regularization for sparse/feature-selecting models, L2 for smooth shrinkage).
- **Pitfall**: forgetting that distance requires *subtracting* vectors first, then taking the norm — not taking norms separately and subtracting those (that gives a completely different, usually meaningless, number).

## 9. Summary

A norm measures the length/magnitude of a vector, generalizing the Pythagorean theorem to any number of dimensions; the L2 norm gives straight-line length, while the L1 norm gives "grid-walking" length. Distance between two points is simply the norm of their difference vector. Norms are the backbone of regularization, convergence checks in optimization, and virtually every distance-based ML method like KNN and clustering.



**Conceptual Q1:** Why does L1 tend to zero out weights while L2 shrinks smoothly?
Geometrically, when you regularize, you're finding the point where the "cost surface" (the loss function) first touches the norm constraint's boundary shape. The L2 constraint region is a smooth circle (or sphere) — there's no reason the touching point should land exactly on an axis, so weights shrink smoothly toward zero but rarely hit it exactly. The L1 constraint region is a diamond (square rotated 45°) with sharp corners sitting exactly on the axes — and those corners are the most likely place for the touching point to land, since corners "stick out" more than the flat edges do. Landing on a corner means one or more coordinates are exactly zero. That's why L1 produces sparse solutions (automatic feature selection) and L2 doesn't.

**Conceptual Q2:** If $\|a\|=0$, what must be true about $a$?
Since a norm sums squares (all non-negative) and takes a square root, the only way the total is zero is if every single component is zero. So $\|a\|=0 \iff a = (0,0,\dots,0)$, the zero vector. This is one of the defining properties of any valid norm ("positive definiteness").

**Numerical Q1:** $\|v\|_2$ for $v=(6,8,0)$
$$\|v\| = \sqrt{6^2+8^2+0^2} = \sqrt{36+64} = \sqrt{100} = 10$$

**Numerical Q2:** L1 and L2 distance between $p=(1,1)$, $q=(4,5)$
Difference: $q-p = (3,4)$
- L2 distance: $\sqrt{3^2+4^2} = 5$
- L1 distance: $|3|+|4| = 7$

Notice L1 distance is always $\geq$ L2 distance for the same pair of points — walking city blocks is never shorter than a straight line.

**Interview question:** A high-activity user has a large-norm embedding; a low-activity user has a small-norm one. Ranking purely by Euclidean distance biases recommendations toward (or away from) items simply based on the *magnitude* of the user's embedding, not the true direction of their preferences — a large-norm user will appear "far" from almost everything in absolute distance, even items they'd genuinely love, simply because their vector is long. Cosine similarity fixes this by dividing out both vectors' norms ($\cos\theta = \frac{a \cdot b}{\|a\|\|b\|}$), comparing only the *direction* — i.e., the pattern of preferences — regardless of how active the user has been.

## Sumamry

## Definition

A **norm** `||v||` measures the **length/magnitude** of a vector.

### Common Norms

**L² norm (Euclidean)**: Straight-line distance
$$||v||_2 = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$$

**L¹ norm (Manhattan)**: City-block distance
$$||v||_1 = |v_1| + |v_2| + ... + |v_n|$$

**Key fact**: `||v||₁ ≥ ||v||₂` (Manhattan is never shorter than straight line)

**L∞ norm (Max)**: Largest component magnitude
$$||v||_\infty = \max(|v_1|, |v_2|, ..., |v_n|)$$

---

## Distance Between Vectors

**Euclidean distance**: `d(a,b) = ||a-b||₂`

**Manhattan distance**: `d(a,b) = ||a-b||₁`

---

## Properties of Norms

1. **Non-negative**: `||v|| ≥ 0`, equality only if `v = 0`
2. **Scalable**: `||cv|| = |c|·||v||`
3. **Triangle inequality**: `||a+b|| ≤ ||a|| + ||b||`

---

## Why for AI/ML

- **Clustering**: K-means uses Euclidean distance
- **Nearest neighbors**: Find closest training examples
- **Regularization**: L1 (Lasso) and L2 (Ridge) penalize weight norms
- **Embedding evaluation**: Distance = dissimilarity
- **Gradient clipping**: Scale gradient if norm exceeds threshold

---

## Common Pitfalls

- **❌** Confusing L1 and L2: L1 is robust to outliers, L2 is sensitive
- **❌** Using raw Euclidean distance with unnormalized features
- **❌** Forgetting that L1 ≥ L2 for same vectors
- **❌** Thinking norm alone gives direction (it doesn't—need normalization)

---

## Conceptual Questions Answered

**1. If ||a|| = 5 and ||b|| = 12, can ||a+b|| be 20? Why or why not?**

**No.** Triangle inequality: `||a+b|| ≤ ||a|| + ||b|| = 5 + 12 = 17`

The maximum possible is 17 (when vectors point same direction). Minimum is 7 (when opposite). Range: `[7, 17]`.

**Geometric**: You can't get total displacement longer than sum of individual displacements.

---

**2. In what scenario is L1 distance equal to L2 distance for two vectors?**

When vectors differ in only **one component**. Example: `a = (0,0)`, `b = (3,0)`
- L1 = 3, L2 = 3 (equal)

In general: equality when the difference vector lies along a single axis (all other components zero).

---

**3. Why is L2 regularization called "weight decay" in neural networks?**

Update rule with L2 penalty:
$$w_{new} = w - \eta(\nabla L + \lambda w) = (1 - \eta\lambda)w - \eta\nabla L$$

The weight **decays** by factor `(1-ηλ)` each step—it's literally shrinking toward zero. L1 doesn't have this smooth decay; it's more like "hard pruning."

---

## Numerical Practice Answered

**1. Compute L1 and L2 norms of `v = (3, -4, 0, 5)`**

**L1**: `|3| + |-4| + |0| + |5| = 3 + 4 + 0 + 5 = 12`

**L2**: `√(3² + (-4)² + 0² + 5²) = √(9 + 16 + 0 + 25) = √50 ≈ 7.07`

**Verify**: L1 (12) ≥ L2 (7.07) ✓

---

**2. Normalize `v = (2, -1, 2)` to unit length**

**Norm**: `||v|| = √(4 + 1 + 4) = √9 = 3`

**Unit vector**: `v/||v|| = (2/3, -1/3, 2/3)`

Check: `√(4/9 + 1/9 + 4/9) = √1 = 1` ✓

---

**3. Find distance (L2) between `a = (1, 2)` and `b = (4, 6)`**

**Difference**: `a-b = (-3, -4)`

**Distance**: `||a-b||₂ = √(9 + 16) = √25 = 5`

---

## Interview-Style Answer

**"A high-activity user has a large-norm embedding; a low-activity user has a small-norm one. Why does ranking purely by Euclidean distance bias recommendations, and how does cosine similarity fix it?"**

### The Bias Problem

Euclidean distance = `||user - item||₂` measures **absolute distance** in vector space.

- **High-activity user** (norm = 100): Most items are relatively far because user vector is long in all directions
- **Low-activity user** (norm = 3): Most items are relatively close

**Result**: Items are ranked **differently based on user magnitude**, not preference direction. An item a high-activity user loves (aligned) might be "far" while an item they hate (opposite) might be closer in absolute distance.

**Analogy**: Two people rating movies 1-5. Person A gives 5s to everything (large magnitude), Person B gives 3s to favorites. Euclidean distance says A is "far" from everything, B is "close" to everything—but that's activity, not preference!

### The Fix: Cosine Similarity

$$\cos(\theta) = \frac{a \cdot b}{||a|| \times ||b||}$$

- Divides out both norms → compares **only direction**
- Range: `[-1, 1]` where 1 = identical preference pattern
- Activity level doesn't affect score—only **relative preferences** matter

**Example**:
- User A (active): `(5, 5, 5)` norm = 8.66
- User B (inactive): `(1, 1, 1)` norm = 1.73
- Item: `(4, 5, 5)`

Euclidean: A is far (2.45), B is close (2.24) → B ranks item higher despite same preference pattern!
Cosine: Both users have cos = 0.94 → same score ✓

---

## Additional Norm FAQ

**Q: Why does L1 regularization produce sparse weights?**
L1's "diamond" shape pushes weights exactly to zero; L2's "circle" just shrinks them. L1 encourages feature selection.

**Q: When should I use L1 vs L2 distance?**
- **L1**: Robust to outliers, high dimensions (less affected by curse)
- **L2**: Smooth, differentiable, Euclidean intuitive

**Q: What's the connection between norm and dot product?**
`||v||² = v·v`. Norm is dot product of vector with itself.

**Q: Why normalize embeddings?**
To remove magnitude bias—compare only patterns/directions. Common in recommendation, retrieval, and similarity search.

**Q: What is the "curse of dimensionality" for norms?**
In high dimensions, most pairwise distances become similar—L1 suffers less than L2. That's why cosine is preferred for high-dimensional embeddings (measures angle, less affected by distance concentration).

---

## Chapter 5: Angles and Cosine Similarity

### 1. What problem does this concept solve?

We just saw the core issue: raw distance and raw dot products are both distorted by vector length/magnitude. Often what we actually care about isn't "how long are these vectors" but "do they point in the same direction" — i.e., pure similarity of pattern, independent of scale. Cosine similarity isolates exactly that.

### 2. Intuition

Imagine two customers who both like "action movies more than romance movies" — one rates action 5/5 and romance 1/5, the other rates action 10/10 and romance 2/10. Their *preferences* (the shape/direction of their taste) are identical, even though their raw numbers (magnitudes) differ. Cosine similarity recognizes these two people as essentially identical in taste, while raw distance would incorrectly see them as different because the numbers themselves aren't equal.### 3. Small numerical example

Let $a = (5, 4)$ (User A's ratings) and $b = (10, 8)$ (User B's ratings — exactly double A's).

**Dot product:**
$$a \cdot b = (5)(10) + (4)(8) = 50 + 32 = 82$$

**Norms:**
$$\|a\| = \sqrt{5^2+4^2} = \sqrt{41} \approx 6.40$$
$$\|b\| = \sqrt{10^2+8^2} = \sqrt{164} \approx 12.81$$

**Cosine similarity:**
$$\cos\theta = \frac{a\cdot b}{\|a\|\|b\|} = \frac{82}{6.40 \times 12.81} \approx \frac{82}{82} = 1$$

A cosine similarity of exactly 1 confirms what we expected: identical direction, meaning identical "taste pattern," regardless of the fact that $b$'s numbers are twice as large.

### 4. Visual explanation

The diagram above shows precisely this: two vectors of very different lengths, but zero angle between them — cosine similarity sees them as perfectly aligned (similarity = 1) because it only cares about direction, having divided out both vectors' magnitudes.

### 5. Mathematical formulation, symbol by symbol

$$\cos\theta = \frac{a \cdot b}{\|a\|\,\|b\|}$$

- $a \cdot b$ — the dot product (from Chapter 3)
- $\|a\|, \|b\|$ — the norms/lengths of each vector (from Chapter 4)
- $\theta$ — the angle between the two vectors
- Cosine similarity ranges from **$-1$ to $1$**: $1$ means identical direction, $0$ means perpendicular (no relationship), $-1$ means exactly opposite direction.

Notice this formula is literally the dot product formula from Chapter 3, rearranged: $a\cdot b = \|a\|\|b\|\cos\theta \Rightarrow \cos\theta = \frac{a\cdot b}{\|a\|\|b\|}$. Nothing new is being invented here — we're just isolating the angle by dividing out the magnitudes.

### 6. Geometric interpretation

Cosine similarity essentially **normalizes** vectors to unit length ($\|a\|=1$) before comparing them — you can think of it as "does it matter how far each user's tastes extend, or just which way they point?" It measures pure directional alignment, stripped of scale.

### 7. Why it matters for AI/ML

- **Embeddings** (words, sentences, images, users, items): cosine similarity is the default way to compare embeddings in search, recommendation systems, and semantic similarity tasks — precisely because embedding *magnitude* often reflects something incidental (like frequency or activity level) rather than meaning.
- **Semantic search / retrieval-augmented generation (RAG)**: finding the most relevant document to a query is typically done by cosine similarity between the query's embedding and each document's embedding.
- **Transformers' attention mechanism**: while raw dot products are used (not normalized cosine), understanding cosine similarity clarifies why attention scores are later scaled and passed through softmax — to control for the "magnitude distortion" problem we've been discussing.
- **Clustering algorithms** (like some variants of K-means) sometimes use cosine distance instead of Euclidean distance when direction matters more than scale (e.g., text data with TF-IDF vectors).

### 8. Common interview questions and pitfalls

- **Pitfall**: confusing cosine *similarity* with cosine *distance*. Cosine distance is usually defined as $1 - \cos\theta$, so a similarity of 1 (identical direction) gives a distance of 0, and a similarity of $-1$ gives a distance of 2.
- **Pitfall**: assuming cosine similarity of 0 means "unrelated" in a strong sense — it just means perpendicular, i.e., no linear directional relationship; it doesn't rule out other kinds of relationships.
- **Interview flavor**: "Why does the Transformer's attention formula divide by $\sqrt{d_k}$ instead of normalizing with the full vector norms like cosine similarity does?" (Answer: dividing by $\sqrt{d_k}$ controls the *variance* of the dot products as dimensionality grows, preventing extremely large values from dominating the softmax, whereas cosine similarity normalizes by *each vector's own length* — different fix for a related magnitude problem.)

### 9. Summary

Cosine similarity measures the angle between two vectors by dividing their dot product by the product of their norms, isolating pure directional similarity while ignoring magnitude. It ranges from $-1$ (opposite) to $1$ (identical direction), with $0$ meaning perpendicular, and it's the standard tool for comparing embeddings in search, recommendations, and semantic similarity tasks.

---

**Conceptual questions:**
1. Two vectors have a dot product of 0. What is their cosine similarity, and why?
2. Why might cosine similarity be preferred over Euclidean distance specifically for comparing text embeddings?

**Numerical practice:**
1. Compute the cosine similarity between $a=(1,0)$ and $b=(1,1)$.
2. If $a$ and $b$ point in exactly opposite directions, what is $\cos\theta$, and what does that imply about $a \cdot b$'s sign?

**Interview-style question:**
"Two product embeddings have a cosine similarity of 0.99 but a large Euclidean distance between them. Explain how this is possible, and which metric you'd trust more for a 'find similar products' feature — and why."

# Angles and Cosine Similarity - Concise Summary

## Definition

**Cosine similarity** measures the angle between two vectors, independent of their lengths:

$$\text{cos}(\theta) = \frac{a \cdot b}{||a|| \times ||b||}$$

**Range**: `[-1, 1]`
- **1**: Same direction (perfectly aligned)
- **0**: Perpendicular (orthogonal, no correlation)
- **-1**: Opposite direction (perfectly anti-aligned)

---

## Connection to Dot Product

From dot product formula: `a·b = ||a|| × ||b|| × cos(θ)`

Therefore:
$$\text{cos}(\theta) = \frac{a \cdot b}{||a|| \times ||b||}$$

**Key insight**: Cosine similarity = **normalized dot product** — removes magnitude influence.

---

## Geometric Interpretation

| cos(θ) | Angle | Meaning |
|--------|-------|---------|
| 1 | 0° | Identical direction |
| 0.5 | 60° | Similar, moderate alignment |
| 0 | 90° | Orthogonal (no relation) |
| -0.5 | 120° | Dissimilar, opposite-ish |
| -1 | 180° | Opposite directions |

---

## Why for AI/ML

- **Text embeddings**: Compare semantic similarity regardless of document length
- **Recommendation systems**: User-item preference alignment
- **Information retrieval**: Rank documents by query relevance
- **Clustering**: Group by direction rather than magnitude
- **Face recognition**: Compare facial features (direction of feature vectors)
- **Word embeddings**: "king" similar to "queen" = high cosine similarity

---

## Common Pitfalls

- **❌** Thinking cosine similarity = correlation (it's angle, not centered covariance)
- **❌** Using cosine similarity when magnitude matters (e.g., sales volume)
- **❌** Forgetting that cosine similarity ignores magnitude entirely (sometimes you need both)
- **❌** Using raw dot product when comparing vectors of different scales

---

## Conceptual Questions Answered

**1. Two vectors have a dot product of 0. What is their cosine similarity, and why?**

**Cosine similarity = 0**

Because: `cos(θ) = (a·b) / (||a|| × ||b||) = 0 / (||a|| × ||b||) = 0`

**Geometric meaning**: Vectors are perpendicular (θ = 90°). They have no alignment—neither similar nor opposite.

**Important**: This holds regardless of vector lengths. Even if `||a|| = 1000` and `||b|| = 0.001`, cos = 0.

---

**2. Why might cosine similarity be preferred over Euclidean distance specifically for comparing text embeddings?**

**Text embeddings vary in magnitude** due to document length, word frequency, and writing style:

- Long document → larger embedding norm (more words)
- Short document → smaller embedding norm
- Active user → larger magnitude
- "The" appears everywhere → large norm but low semantic value

**Cosine similarity solves this** by:
- Normalizing both vectors → compares **only semantic direction**
- A 1000-word document and a 50-word query can be compared fairly
- "Cat" and "kitten" will be close regardless of how often they appear

**Euclidean distance** would say:
- Long document is "far" from everything (large magnitude)
- Short document is "close" to everything
- Activity level dominates, not meaning

**Analogy**: Two people rating movies. One uses scale 1-10, another uses 1-5. Raw Euclidean distance says the 10-scale user is "far" from everything. Cosine similarity says "we both like action movies equally."

---

## Numerical Practice Answered

**1. Compute cosine similarity between `a = (1, 0)` and `b = (1, 1)`**

**Step 1**: Compute dot product
```
a·b = 1×1 + 0×1 = 1
```

**Step 2**: Compute norms
```
||a|| = √(1² + 0²) = √1 = 1
||b|| = √(1² + 1²) = √2
```

**Step 3**: Apply formula
```
cos(θ) = 1 / (1 × √2) = 1/√2 = 0.707
```

**Interpretation**: 45° angle between vectors. They're moderately aligned (about halfway between same and perpendicular).

---

**2. If a and b point in exactly opposite directions, what is cos(θ), and what does that imply about a·b's sign?**

**cos(θ) = -1** (180° angle)

**Dot product sign**: `a·b` will be **negative** (unless one vector is zero).

Because: `a·b = ||a|| × ||b|| × cos(180°) = ||a|| × ||b|| × (-1) < 0`

**Example**: `a = (2, 0)`, `b = (-3, 0)`
- cos(θ) = -1
- a·b = -6 (negative)

**Interpretation**: Perfectly opposite preferences—they dislike exactly what the other likes.

---

## Additional Cosine Similarity FAQ

**Q: Does cosine similarity change if I scale one vector?**
No. Scaling one vector (multiplying by positive scalar) doesn't change the angle. Negative scaling flips direction (cos becomes -cos).

**Q: Is cosine similarity a distance metric?**
No. It's a similarity metric (higher = closer). Distance would be `1 - cos(θ)` but this isn't a proper metric (violates triangle inequality).

**Q: When should I use Euclidean distance instead?**
When magnitude matters: sales volume, word count, price, intensity, or when all vectors are already normalized.

**Q: In high dimensions, are vectors all nearly perpendicular?**
Yes—the "curse of dimensionality." Random vectors in high dimensions tend to be orthogonal. This makes cosine similarity less discriminative.

**Q: What's the relationship to Pearson correlation?**
Pearson correlation is cosine similarity **after centering** (subtracting mean from each vector). Correlation ignores scale and mean; cosine ignores only scale.

**Q: Why do Transformer attention scores use dot product, not cosine similarity?**
They're typically computed in normalized spaces (LayerNorm), making dot product = cosine similarity (up to scaling). The scaling factor `1/√d` acts as soft normalization.
