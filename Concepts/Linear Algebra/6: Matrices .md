
**Interview question:** Cosine similarity 0.99 but large Euclidean distance — how, and which metric to trust?
This happens when the two embeddings point in nearly the same direction but have very different magnitudes — e.g., $a=(1,1)$ and $b=(100,100)$ have cosine similarity 1 but a huge Euclidean distance. For a "find similar products" feature, you'd typically trust **cosine similarity** more, since the direction (the pattern of features/attributes) usually encodes the actual semantic content, while magnitude often reflects an unrelated factor like popularity, review count, or how "extreme" a product's raw feature values happen to be — not true dissimilarity.

---

## Chapter 6: Matrices

### 1. What problem does this concept solve?

So far we've worked with single vectors. But real data isn't just one vector — it's usually a whole *collection* of vectors: many data points, each with many features, or a transformation that needs to act on every vector in a space at once. A matrix solves this by packing multiple vectors (or a whole linear transformation) into a single organized object.

### 2. Intuition

Think of a spreadsheet: each row could be one data point (say, one house), and each column one feature (size, bedrooms, price). That spreadsheet — rows and columns of numbers — is a matrix. Separately, a matrix can also represent an *action*: a set of instructions like "rotate everything 90°" or "stretch the x-axis by 2" that can be applied to any vector you feed into it. Both views (a table of data, and a transformation machine) are central to ML, and you'll switch between them constantly.### 3. Small numerical example

A 2×3 matrix (2 rows, 3 columns) representing two houses with three features (size, bedrooms, age):

$$A = \begin{bmatrix} 1200 & 3 & 5 \\ 900 & 2 & 12 \end{bmatrix}$$

Here, row 1 = house 1's features, row 2 = house 2's features.

A small square matrix used as a transformation — this one rotates any 2D vector by 90°:

$$R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

We'll see exactly how this "acts" on a vector once we cover matrix multiplication next chapter — for now, just notice a matrix is simply a rectangular grid of numbers, described by how many rows and columns it has.

### 4. Visual explanation

The diagram above captures both interpretations side by side: on the left, a matrix as a spreadsheet where rows are individual data points; on the right, a matrix as a "machine" that takes an input vector and produces some transformed output vector (rotated, stretched, or both).

### 5. Mathematical formulation, symbol by symbol

$$A = \begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix}$$

- $A$ — the name of the matrix (capital letters are conventionally used for matrices, lowercase for vectors and scalars)
- $a_{ij}$ — the entry in **row $i$, column $j$** (row index always comes first — a very common point of confusion)
- We say $A$ is an **$m \times n$ matrix**, meaning $m$ rows and $n$ columns — always read as "rows by columns," in that order
- A matrix can be thought of as **a collection of column vectors** side by side, or equivalently **a collection of row vectors** stacked — both views are valid and useful in different contexts (e.g., "each column is a feature" vs. "each row is a data point").

### 6. Geometric interpretation

- **As data**: a matrix is a cloud of points/vectors, one per row (or column, depending on convention).
- **As a transformation**: a matrix describes how to move every point in space — e.g., rotating, scaling, shearing, or projecting. Multiplying a vector by a matrix (next chapter) applies that transformation to that specific vector. This is the geometric heart of linear algebra: matrices are functions that transform space in a *linear* way (straight lines stay straight, the origin stays fixed).

### 7. Why it matters for AI/ML

- **Datasets**: virtually every dataset used in ML is stored as a matrix — rows as examples, columns as features (this is exactly what a "design matrix" $X$ is in regression).
- **Neural network weights**: each layer's weights form a matrix; the layer's operation (before the activation function) is a matrix-vector multiplication.
- **Images**: a grayscale image is literally a matrix of pixel intensities; a color image is 3 stacked matrices (red, green, blue channels).
- **Transformations in PCA, embeddings, and attention**: all rely on matrices to represent transformations between spaces (e.g., projecting high-dimensional data onto fewer dimensions).

### 8. Common interview questions and pitfalls

- **Pitfall**: mixing up "$m \times n$" — always rows first, then columns. A $3\times 2$ matrix has 3 rows and 2 columns, not the other way around.
- **Pitfall**: assuming a matrix is always square — most ML matrices (like datasets) are rectangular, not square, and many operations (inverses, eigenvalues) only apply to square matrices, which trips people up later.
- **Interview flavor**: "If your dataset has 1000 samples and 50 features, what's the shape of your design matrix, and what does each dimension represent?" (Answer: $1000 \times 50$ — rows are samples, columns are features; this convention matters because weight vectors and matrix multiplications downstream depend on getting this orientation right.)

### 9. Summary

A matrix is a rectangular grid of numbers that can represent either a collection of vectors (like a dataset, where rows are data points and columns are features) or a linear transformation that maps input vectors to output vectors. This dual view — data table vs. transformation machine — underlies everything from storing datasets to defining neural network layers.

---

**Conceptual questions:**
1. In the design matrix for a dataset with $n$ samples and $d$ features, what does each row represent, and what does each column represent?
2. Why is it important to know whether a matrix is being interpreted as "data" versus as a "transformation" when reading ML code or papers?

**Numerical practice:**
1. Write out the shape ($m \times n$) of a matrix representing 500 images, each flattened into a vector of 784 pixel values.
2. Given $A = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}$, identify the entry $a_{21}$.

**Interview-style question:**
"You're told a weight matrix in a neural network layer has shape $(256, 128)$. Without more context, what are the two possible interpretations of which dimension is input size and which is output size, and why does this ambiguity matter when debugging a shape mismatch error?"

Noted — this refined version matches what we've been doing, so I'll keep going exactly this way: full topic list, chapter by chapter, waiting for you each time, intuition before notation. Here are the answers to Chapter 6, then we'll move to Matrix Multiplication.

**Conceptual Q1:** In a design matrix with $n$ samples and $d$ features, what do rows/columns represent?
Each **row** represents one data sample (one observation — e.g., one house, one user, one image). Each **column** represents one feature (e.g., size, age, a specific pixel). So the matrix has shape $n \times d$.

**Conceptual Q2:** Why does it matter whether a matrix is "data" or a "transformation"?
Because the *meaning* of rows vs. columns, and what operations make sense, completely changes depending on interpretation. If a matrix is data, multiplying it by a weight vector produces predictions per row (each row dotted with the weights). If a matrix is a transformation, multiplying it by a vector produces one *new* vector representing where that vector "moved to" in space. Confusing the two leads to shape mismatches and, worse, code that silently runs but computes something conceptually meaningless — a classic real-world debugging trap.

**Numerical Q1:** Shape of a matrix representing 500 images, each flattened to 784 pixels
$$500 \times 784$$
(500 rows = images/samples, 784 columns = pixel features per image.)

**Numerical Q2:** For $A = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}$, find $a_{21}$
Row 2, column 1 → $a_{21} = 1$.

**Interview question:** Weight matrix shape $(256, 128)$ — two interpretations?
By convention there are two common layouts, and frameworks differ:
1. **(output_size, input_size)** = $(256, 128)$ means this layer takes a 128-dim input and produces a 256-dim output (this is PyTorch's `nn.Linear` convention internally: weight shape is `(out_features, in_features)`).
2. **(input_size, output_size)** = $(256, 128)$ means it takes a 256-dim input and produces a 128-dim output (common when you write $y = xW$ with $x$ as a row vector, as in many hand-written NumPy implementations or TensorFlow's `Dense` layer weight convention).

This ambiguity matters enormously when debugging shape mismatches — a `(256,128) @ (128,)` multiplication is valid in one convention and will silently succeed even if you meant the *other* convention, because the arithmetic doesn't always throw an error; it just quietly computes the wrong thing or errors two layers downstream in a confusing place. Always check the specific framework's documented convention rather than assuming.

---

## Chapter 7: Matrix Multiplication

### 1. What problem does this concept solve?

We now know matrices can represent transformations (Chapter 6), but we haven't defined *how* a matrix actually acts on a vector to produce an output. Matrix multiplication is that missing mechanical rule — it's the precise recipe for "applying" a matrix to a vector (or to another matrix, chaining transformations together).

### 2. Intuition

Think of matrix-vector multiplication as a **weighted mixing recipe**: each output number is a weighted sum of the input numbers, where the weights come from one row of the matrix. If you have a neuron that looks at 3 inputs and computes "0.5×input1 + 2×input2 − 1×input3," that's one row of a matrix acting on your input vector. Stack several such "recipes" (rows) together, and you get a full matrix multiplication — it just runs that same recipe-application once per row, all at once.### 3. Small numerical example

Let $A = \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}$ and $x = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$.

As shown in the diagram above: each row of $A$ is dotted with $x$ to produce one entry of the output.

$$Ax = \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} (2)(3)+(1)(4) \\ (0)(3)+(1)(4) \end{bmatrix} = \begin{bmatrix} 10 \\ 4 \end{bmatrix}$$

### 4. Visual explanation

The visual above shows the mechanical rule directly: multiplying a matrix by a vector is just "dot product each row of the matrix with the vector," and stacking the resulting numbers gives you the output vector. This is the entire recipe — no more, no less.

### 5. Mathematical formulation, symbol by symbol

**Matrix-vector multiplication:**
$$(Ax)_i = \sum_{j=1}^n A_{ij}\,x_j$$

- $(Ax)_i$ — the $i$-th entry of the resulting output vector
- $A_{ij}$ — entry in row $i$, column $j$ of matrix $A$
- $x_j$ — the $j$-th entry of vector $x$
- **Critical requirement**: the number of columns in $A$ must equal the number of entries (dimension) in $x$ — this is the "inner dimensions must match" rule that trips up almost everyone at some point.

**Matrix-matrix multiplication** (a natural extension — think of it as multiplying $A$ by each column of $B$ separately, then stacking results as columns):

$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$$

- If $A$ is $m \times n$ and $B$ is $n \times p$, the result $AB$ is $m \times p$
- The **inner dimensions ($n$) must match**; the **outer dimensions ($m$ and $p$) become the new matrix's shape**

### 6. Geometric interpretation

Matrix-vector multiplication **applies** a transformation to a vector — the output is where the input vector "lands" after being rotated, scaled, sheared, or projected by the matrix. Matrix-matrix multiplication **composes transformations** — if $A$ represents "rotate 90°" and $B$ represents "scale by 2," then $AB$ (applied to a vector) means "first scale by 2, then rotate 90°" (order matters — matrix multiplication is applied right-to-left when transforming a vector, i.e., $ABx$ means "apply $B$ first, then $A$").

### 7. Why it matters for AI/ML

- **Every neural network layer** computes $Wx + b$ — a matrix-vector multiplication (weights times input) plus a bias vector. This is *the* core computation of deep learning, repeated at every layer.
- **Batches of data**: in practice we multiply a whole matrix of data $X$ (many rows, one per example) by a weight matrix $W$ at once — $XW$ — computing all predictions in a single matrix multiplication rather than looping over samples.
- **PCA**: projecting data onto principal components is a matrix multiplication between data and the matrix of eigenvectors.
- **Attention in Transformers**: computing Query, Key, and Value vectors from token embeddings is done via matrix multiplications ($Q = XW_Q$, etc.), and the attention scores themselves come from matrix multiplication between $Q$ and $K^T$.
- **Convolutions** in CNNs can be reframed as structured matrix multiplications.

### 8. Common interview questions and pitfalls

- **Pitfall**: matrix multiplication is **not commutative** — $AB \neq BA$ in general (unlike multiplying regular numbers). This is one of the most-tested facts in interviews.
- **Pitfall**: forgetting the dimension-matching rule — trying to multiply an $m\times n$ matrix by a $p \times q$ matrix when $n \neq p$ is simply undefined, not "approximately correct."
- **Pitfall**: confusing matrix multiplication with the element-wise (Hadamard) product — they are completely different operations with different rules and different results.
- **Interview flavor**: "If $A$ is $3\times 4$ and $B$ is $4\times 2$, what's the shape of $AB$? Is $BA$ even defined?" (Answer: $AB$ is $3\times 2$; $BA$ is undefined since $B$'s columns (2) don't match $A$'s rows (3) for that order.)

### 9. Summary

Matrix-vector multiplication applies a transformation to a vector by dotting each matrix row with the vector; matrix-matrix multiplication composes two transformations into one, computed by dotting rows of the first matrix with columns of the second. The inner dimensions must always match, multiplication is not commutative, and this single operation is the computational core of virtually every ML model — from a single neuron's weighted sum to full Transformer attention.

---

**Conceptual questions:**
1. Why is matrix multiplication not commutative, even though multiplying regular numbers is?
2. In $y = Wx + b$ for a neural network layer, what do the shapes of $W$, $x$, and $b$ need to satisfy for this to be valid?

**Numerical practice:**
1. Compute $Ax$ for $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $x = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.
2. If $A$ is $2\times3$ and $B$ is $3\times5$, what is the shape of $AB$? Can you compute $BA$?

**Interview-style question:**
"You're stacking two neural network layers: layer 1 has weight matrix shape $(64, 128)$ and layer 2 has weight matrix shape $(64, 32)$ using the convention $(output, input)$. What's wrong with this design, and how would you fix the shape mismatch?"

