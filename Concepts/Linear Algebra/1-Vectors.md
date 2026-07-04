# Vectors - Concise Summary

A **vector** is an ordered list of numbers representing a point, direction, quantity, or set of features.

## Core Concepts

### Notation & Structure
- **Vector**: `v = [4, 3]` or `v = (4, 3)`
- **Components**: The numbers inside brackets (4 and 3)
- **Dimension**: Number of components. `(4,3)` is 2-dimensional → `v ∈ ℝ²`
- **ℝⁿ**: The space of all possible n-number lists

### Two Views of a Vector
1. **Point view** - location (4,3) in space
2. **Arrow view** - displacement (direction + length), can be drawn anywhere

**Key insight**: Same vector can start anywhere - `(4,3)` from origin OR `(4,3)` from `(10,10)` is the SAME vector.

### Why for AI/ML
- Every data point → vector (images, text, user profiles)
- Word embeddings: "king" → `(0.2, -1.4, 0.7, ...)`
- Neural network weights = vectors
- Gradients = vectors (point to steepest increase)

### Common Pitfalls
- Vector ≠ fixed point (no location)
- Dimension ≠ "size" (just number of independent values)
- Vector vs scalar: scalar = single number; vector = collection with direction

---

## Conceptual Questions Answered

**1. If two arrows have the same length and direction but start at different points, are they the same vector? Why?**

**Yes.** A vector represents displacement, not position. `(4,3)` starting at origin and `(4,3)` starting at `(10,10)` have the same:
- Direction (same angle)
- Magnitude (same length)
- Component values

They differ only in location, which vectors don't encode. We can slide the arrow anywhere; it remains the same vector.

---

**2. Why do we say an image with 784 pixels can be represented as a "vector in ℝ⁷⁸⁴"?**

Each of the 784 pixel values becomes a component in the vector. The image is transformed into a 784-number list:
```
[px₁, px₂, px₃, ..., px₇₈₄]
```
This list is exactly a 784-dimensional vector. It exists in `ℝ⁷⁸⁴` because there are 784 independent numbers (each pixel can vary independently). The "point view" treats this as a single point in 784-dimensional space.

---

## Numerical Practice Answered

**1. Vector `(-2, 5)`** - Points 2 units left, 5 units up (upper-left quadrant, approximately 112° angle from positive x-axis).

**2. Dimension of `(1, 0, -3, 4)`** - 4 dimensions (4 components).

---

## Interview-Style Answer

**"You're told two data points are represented as vectors in ℝ⁵⁰. What does that tell you about how those data points were generated or featurized, and what does it *not* tell you about their similarity?"**

**What it tells you:**
- Each point was converted to exactly 50 numerical features
- The features are ordered consistently across points
- Both live in the same feature space

**What it does NOT tell you:**
- **Feature interpretation** - Which features? (pixels? word counts? embedding dimensions?)
- **Scale** - Are features normalized? One feature range 0-1, another 0-10000?
- **Similarity** - Euclidean distance depends on scale; cosine similarity depends on angles; without knowing feature engineering, we can't judge similarity
- **Distribution** - Are features dense or sparse? Gaussian or power-law?
- **Meaning** - Two vectors far apart might be similar if features weren't scaled appropriately

**The key point**: Dimension tells you the *structure* of the representation, not its *interpretation* or quality.
