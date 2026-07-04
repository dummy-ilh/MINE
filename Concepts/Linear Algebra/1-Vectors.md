# Chapter 1: Vectors

## 1. What problem does this concept solve?

Imagine you want to describe something that has both a **size** and a **direction** — wind blowing at 20 km/h toward the northeast, or a customer's preferences across 300 different product features. A plain number (like "20" or "5 stars") can't capture that. You need an object that bundles multiple related numbers together and treats them as one thing.

That's what a vector is for: a single mathematical object that represents a point, a direction, a quantity, or a list of features — all at once.

## 2. Intuition (real-world analogy)

Think of giving someone directions: "walk 4 blocks east, then 3 blocks north." That instruction has two pieces of information, but it's really *one* instruction — one movement. A vector is exactly that: a bundle of numbers describing one movement, one point, or one "thing," where the order and grouping matter.

In AI/ML, a vector is usually a **list of features** describing something: a house (size, bedrooms, age, price), a word (its meaning encoded as 300 numbers), or an image pixel (red, green, blue intensity).

## 3. Small numerical example

The vector in the diagram above is:

$$v = (4, 3)$$

This means: starting at the origin, move 4 units right and 3 units up. That single arrow *is* the vector — not the starting point, not the ending point, but the *displacement* between them.

Another example, in 3D: $u = (2, -1, 5)$ — go 2 right, 1 back, 5 up.

## 4. Visual explanation

The diagram above shows exactly this: the arrow from the origin to (4, 3), with dashed lines showing how the vector "breaks down" into its horizontal piece (4) and vertical piece (3). This breakdown is the key visual intuition you'll reuse constantly — vectors are made of independent components along each axis.

## 5. Mathematical formulation, symbol by symbol

$$v = \begin{bmatrix} 4 \\ 3 \end{bmatrix}$$

- $v$ — the name we give the vector (bold or arrow-topped in textbooks: **v** or $\vec{v}$)
- The numbers inside the brackets are called **components** or **coordinates**
- The **dimension** of a vector is how many components it has. $(4,3)$ is 2-dimensional. A word embedding might be 300-dimensional — same idea, just more numbers, impossible to draw but easy to reason about algebraically.
- We write $v \in \mathbb{R}^2$ to mean "$v$ is a vector living in 2-dimensional real space." $\mathbb{R}^n$ just means "the space of all possible $n$-number lists."

## 6. Geometric interpretation

A vector has **two equally valid interpretations**, and interview questions love testing whether you know both:

1. **A point** in space — the location (4, 3).
2. **An arrow / displacement** — a direction and length, which can be drawn starting anywhere (not just the origin) and still represent the "same" vector, as long as the direction and length match.

Both views are useful: "point" view is natural for data (a row in a spreadsheet), "arrow" view is natural for movement, forces, and gradients.

## 7. Why it matters for AI/ML

- Every data point fed into a model — an image, a sentence, a user profile — is first converted into a vector of numbers. This is the **entire foundation** of ML: models don't understand images or words, they understand vectors.
- **Word embeddings**: the word "king" becomes a vector like $(0.2, -1.4, 0.7, \dots)$, and vectors that are close together represent similar meanings.
- **Model weights** in a neural network layer are also just vectors (and matrices, which are collections of vectors — coming next chapter).
- **Gradients** (used in gradient descent to train models) are vectors — they point in the direction of steepest increase of a function.

## 8. Common interview questions and pitfalls

- **Pitfall**: confusing a vector with a point. A vector has no fixed location — $(4,3)$ starting at the origin and $(4,3)$ starting at $(10,10)$ (ending at $(14,13)$) represent the *same vector*, just drawn in different places.
- **Pitfall**: thinking dimension means "size" in the everyday sense. A 300-dimensional vector isn't "big," it just has 300 independent numbers.
- **Interview flavor**: "What's the difference between a vector and a scalar?" (Answer: a scalar is a single number with magnitude only; a vector has both magnitude and direction — or in ML terms, a scalar is one feature, a vector is a collection of features.)

## 9. Summary

A vector is an ordered list of numbers that represents a point, a direction, or a set of features as a single mathematical object. It can be viewed geometrically as an arrow (direction + length) or as a point in space, and this dual view is what makes vectors the basic building block of literally every ML model — every input, weight, and gradient is a vector.

---

**Conceptual questions:**
1. If two arrows have the same length and direction but start at different points, are they the same vector? Why?
2. Why do we say an image with 784 pixels can be represented as a "vector in $\mathbb{R}^{784}$"?

**Numerical practice:**
1. Draw (mentally or on paper) the vector $(-2, 5)$. Which direction does it point?
2. What is the dimension of the vector $(1, 0, -3, 4)$?

**Interview-style question (Google/Apple flavor):**
"You're told two data points are represented as vectors in $\mathbb{R}^{50}$. What does that tell you about how those data points were generated or featurized, and what does it *not* tell you about their similarity?"

Take your time — let me know your answers, or just say "next" when you're ready to move to **Vector Operations** (addition, scalar multiplication), which builds directly on what we just covered.
