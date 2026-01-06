# k-Nearest Neighbors (KNN)

---

## 1. What is the concept?

**k-Nearest Neighbors (KNN)** is a **non-parametric, instance-based (lazy) learning algorithm** used for **classification and regression**.

Given a query point $x$, KNN:
- Finds the $k$ closest points to $x$ in the training set under a chosen distance metric
- Predicts the output by **aggregating** the labels/values of those neighbors

KNN does **no explicit model fitting**; the training data itself constitutes the model.

---

## 2. Intuition

- Similar inputs should produce similar outputs
- The prediction for a point should resemble the outcomes of its nearest neighbors

Plain intuition:
- “Tell me what usually happens to points like this”

Geometric intuition:
- The feature space is partitioned implicitly into regions determined by proximity to training points

---

## 3. Mathematical formulation

### Distance metric

For points $x, x_i \in \mathbb{R}^d$, common distances include:

- **Euclidean**:
$$
d(x, x_i) = \sqrt{\sum_{j=1}^d (x_j - x_{ij})^2}
$$

- **Manhattan**:
$$
d(x, x_i) = \sum_{j=1}^d |x_j - x_{ij}|
$$

- **Minkowski**:
$$
d(x, x_i) = \left( \sum_{j=1}^d |x_j - x_{ij}|^p \right)^{1/p}
$$

---

### Classification

Let $\mathcal{N}_k(x)$ be the set of $k$ nearest neighbors.

Predicted class:
$$
\hat{y}(x) = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} \mathbf{1}(y_i = c)
$$

Weighted variant:
$$
\hat{y}(x) = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} w_i \mathbf{1}(y_i = c)
$$

where typically $w_i = \frac{1}{d(x, x_i)}$.

---

### Regression

Prediction:
$$
\hat{y}(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
$$

Weighted regression:
$$
\hat{y}(x) = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}
$$

---

## 4. Why the concept matters (theory + practice)

### Theoretical importance

- Canonical example of:
  - **Non-parametric learning**
  - **Local approximation**
- KNN is **universally consistent**:
  - As $n \to \infty$, $k \to \infty$, and $k/n \to 0$, KNN converges to Bayes optimal risk

### Practical importance

- Strong baseline for:
  - Low-dimensional problems
  - Small to medium datasets
- Minimal assumptions about data distribution
- Used as:
  - Benchmark
  - Feature space sanity check

---

## 5. Assumptions and limitations

### Assumptions

- Meaningful notion of **distance**
- Local smoothness of the target function
- Features are on comparable scales

### Limitations

- Suffers heavily from the **curse of dimensionality**
- Memory-intensive (stores full dataset)
- Prediction time grows with dataset size
- Sensitive to irrelevant and noisy features

---

## 6. Common pitfalls and misconceptions

| Pitfall | Explanation |
|---|---|
| Not scaling features | Distance dominated by large-scale features |
| Choosing $k=1$ blindly | Leads to extreme variance |
| Using KNN in high dimensions | Distances become uninformative |
| Ignoring class imbalance | Majority class dominates neighbors |
| Assuming KNN “learns” | It performs no parametric learning |

---

## 7. How to detect issues related to this concept

### Bias–variance diagnostics via $k$

| $k$ value | Behavior |
|---|---|
| Small $k$ | Low bias, high variance |
| Large $k$ | High bias, low variance |

---

### Empirical diagnostics

- Cross-validation error vs $k$
- Sensitivity to feature scaling
- Performance degradation as dimensionality increases
- Visualization of decision boundaries in low dimensions

---

## 8. How to fix or improve those issues

### Choosing $k$

- Use cross-validation to minimize validation error
- Rule of thumb: $k \approx \sqrt{n}$ (only a heuristic)

---

### Distance and weighting

- Use **distance-weighted KNN**
- Select appropriate metric:
  - Cosine for text
  - Mahalanobis for correlated features

---

### Dimensionality reduction

- PCA before KNN
- Feature selection
- Metric learning

---

### Computational improvements

| Method | Purpose |
|---|---|
| KD-tree | Faster neighbor search (low $d$) |
| Ball tree | Better for moderate $d$ |
| Approximate NN (HNSW, FAISS) | Large-scale systems |

---

## 9. Connections to other ML concepts

- **Bias–Variance Tradeoff**:
  - $k$ controls model complexity
- **Kernel methods**:
  - KNN approximates kernel regression with a uniform kernel
- **Manifold learning**:
  - Relies on local neighborhoods
- **Instance-based learning**:
  - Related to case-based reasoning
- **Density estimation**:
  - KNN implicitly estimates local density

---

## 10. Real-world applications

### Recommendation systems
- User-based and item-based collaborative filtering
- Similarity defined over embeddings or interaction vectors

### Computer vision
- Image classification with learned embeddings
- Face recognition via nearest-neighbor search

### NLP
- Semantic similarity using embedding distance
- Retrieval-augmented generation components

### Anomaly detection
- Points with large average neighbor distance flagged as anomalies

### Search and retrieval
- Nearest neighbor lookup in embedding spaces
- Core primitive in vector databases

---

