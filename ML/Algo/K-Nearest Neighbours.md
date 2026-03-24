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


### Distance Metrics

For points $$x, x_i \in \mathbb{R}^d$$, common distance metrics include:

$$
d(x, x_i) = \sqrt{\sum_{j=1}^d (x_j - x_{ij})^2} \quad \text{(Euclidean)}
$$

$$
d(x, x_i) = \sum_{j=1}^d |x_j - x_{ij}| \quad \text{(Manhattan)}
$$

$$
d(x, x_i) = \left( \sum_{j=1}^d |x_j - x_{ij}|^p \right)^{1/p} \quad \text{(Minkowski)}
$$


### Classification


Let `$\mathcal{N}_k(x)$` be the set of `$k$` nearest neighbors.

**Majority vote:**
$$
\hat{y}(x) = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} \mathbf{1}(y_i = c)
$$

**Weighted variant:**
$$
\hat{y}(x) = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} w_i \mathbf{1}(y_i = c)
$$
where typically `$w_i = 1/d(x,x_i)$`.

### Regression


**Simple average:**
$$
\hat{y}(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
$$

**Weighted average:**
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


## 1. Advantages and Disadvantages of KNN

### Advantages

| Advantage | Explanation |
|---|---|
| Non-parametric | Makes no assumptions about data distribution |
| Simple and intuitive | Easy to understand and implement |
| Flexible decision boundaries | Can model highly non-linear patterns |
| Naturally multi-class | No modification needed for multi-class classification |
| Strong baseline | Useful for sanity checks and benchmarking |
| Asymptotically optimal | Universally consistent under mild conditions |

---

### Disadvantages

| Disadvantage | Explanation |
|---|---|
| Curse of dimensionality | Distances lose meaning as dimensionality increases |
| High inference cost | Prediction requires distance computation to all points |
| Memory intensive | Stores entire training dataset |
| Sensitive to noise | Outliers can strongly affect predictions |
| Feature scaling required | Distance metrics depend on scale |
| Poor extrapolation | Only interpolates from existing data |

---

## 2. When to Use KNN and When Not to Use

### When to Use KNN

KNN is appropriate when:

- Dataset size is **small to medium**
- Feature space is **low-dimensional**
- Decision boundary is **complex and nonlinear**
- Interpretability via examples is useful
- Fast training but slower inference is acceptable
- You have a **meaningful distance metric**

Typical use cases:
- Prototype systems
- Recommendation based on similarity
- Image / text embeddings with cosine distance
- Anomaly detection via neighborhood density

---

### When NOT to Use KNN

Avoid KNN when:

- Dataset is very large (millions of points)
- Feature space is high-dimensional and sparse
- Real-time inference with strict latency constraints
- Features are noisy or mostly irrelevant
- Data distribution shifts frequently

In such cases, parametric or representation-learning models are preferred.

---

## 3. Variants and Improvements of KNN

### Distance-weighted KNN

Instead of uniform voting:

$$
w_i = \frac{1}{d(x, x_i)^\alpha}
$$

Closer neighbors have higher influence.

---

### Radius-based Nearest Neighbors

- Uses all neighbors within radius $r$
- Adapts neighborhood size to local density
- Can fail in sparse regions

---

### Edited / Condensed KNN

- Remove noisy or redundant points
- Reduces memory and inference cost
- Improves robustness

---

### Approximate Nearest Neighbors

- Trade accuracy for speed
- Used in large-scale systems

Examples:
- HNSW
- FAISS
- Annoy

---

### Metric Learning + KNN

- Learn distance metric instead of fixing it
- Mahalanobis distance:

$$
d(x, x') = \sqrt{(x - x')^T M (x - x')}
$$

Where $M \succeq 0$ is learned from data.

---

## 4. Working of Nearest Neighbors and Choosing $k$

### Step-by-step Working

1. Store all training data
2. Choose distance metric
3. For a query point:
   - Compute distance to all training points
   - Select $k$ closest points
4. Aggregate outputs:
   - Majority vote (classification)
   - Mean or weighted mean (regression)

---

### Effect of $k$ (Bias–Variance View)

| $k$ | Bias | Variance | Behavior |
|---|---|---|---|
| Small ($k=1$) | Low | High | Overfitting |
| Large | High | Low | Underfitting |

---

### How to Choose $k$

- Cross-validation over candidate values
- Odd $k$ for binary classification (avoid ties)
- Heuristic: $k \approx \sqrt{n}$ (only a starting point)
- Consider class imbalance and noise level

---

### Visualization Insight

- Small $k$: jagged, complex decision boundary
- Large $k$: smooth, coarse decision boundary

---

## 5. scikit-learn KNN Parameters (Detailed Explanation)

### Core Estimator

python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
`

---

### Key Parameters Explained

| Parameter       | Meaning                                         | Impact                              |
| --------------- | ----------------------------------------------- | ----------------------------------- |
| `n_neighbors`   | Number of neighbors ($k$)                       | Controls bias–variance tradeoff     |
| `weights`       | `'uniform'` or `'distance'`                     | Distance weighting reduces variance |
| `algorithm`     | `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'` | Neighbor search strategy            |
| `leaf_size`     | Leaf size for tree-based methods                | Affects speed and memory            |
| `p`             | Power for Minkowski distance                    | $p=2$ Euclidean, $p=1$ Manhattan    |
| `metric`        | Distance metric                                 | Defines similarity                  |
| `metric_params` | Extra params for metric                         | Custom distance tuning              |
| `n_jobs`        | Parallel jobs                                   | Speeds up distance computation      |

---

### Distance Metric Details

Default:
$$
\text{metric} = \text{minkowski}, \quad p = 2
$$

Common choices:

| Metric      | When to Use                 |
| ----------- | --------------------------- |
| Euclidean   | Continuous, scaled features |
| Manhattan   | Robust to outliers          |
| Cosine      | Text / embedding similarity |
| Mahalanobis | Correlated features         |

---

### Algorithm Choices

| Algorithm   | Best Use Case             |
| ----------- | ------------------------- |
| `brute`     | High dimensions           |
| `kd_tree`   | Low dimensions ($d < 20$) |
| `ball_tree` | Moderate dimensions       |
| `auto`      | Lets sklearn choose       |

---

### Practical Notes

* Always scale features before KNN
* `n_jobs=-1` uses all cores
* Tree-based methods degrade in high dimensions
* sklearn KNN is exact, not approximate

---

## Closing Insight

KNN is best understood as a **local averaging method** whose power and limitations come directly from how similarity is defined and how neighborhoods are chosen. Its simplicity makes it a conceptual cornerstone for understanding more advanced ideas like kernels, metric learning, and vector search systems.



# Improving KNN and Understanding KD-Tree & Ball Tree

---

## 1. How to Improve KNN

KNN’s main limitations are **high variance**, **curse of dimensionality**, and **high prediction cost**. Improvements generally focus on **reducing variance**, **handling high-dimensional data**, and **speeding up neighbor search**.

### a) Reduce Variance / Improve Accuracy

| Method | How it Helps |
|---|---|
| **Increase $k$** | Smoother predictions, reduces sensitivity to noise (variance ↓) |
| **Distance weighting** | Closer neighbors count more, reducing influence of outliers |
| **Feature scaling / normalization** | Ensures distance metrics are meaningful |
| **Feature selection / dimensionality reduction** | Remove irrelevant features to reduce variance and improve signal |
| **Metric learning** | Learns an optimal distance metric for the task |

---

### b) Reduce Overfitting / Underfitting

| Issue | Solution |
|---|---|
| Overfitting (small $k$) | Increase $k$, use distance weighting, remove noisy points |
| Underfitting (large $k$) | Reduce $k$, select more informative features |

---

### c) Speeding Up Prediction

- KNN naïvely computes distances to **all training points** → $O(n d)$ per query
- **Tree-based methods** and **approximate nearest neighbor** search can drastically reduce computation

---

## 2. KD-Tree

### What is a KD-Tree?

- KD-Tree = **k-dimensional binary search tree**
- Organizes points in $d$-dimensional space for fast nearest neighbor search
- Each node splits data along **one dimension** (axis-aligned) at a median

#### Structure




     x1 < median
    /         \


left subtree      right subtree



- Root: splits dataset along first axis (e.g., x-axis)
- Children: split recursively along other axes cyclically (x, y, z, …)
- Leaf nodes: contain a small number of points (leaf_size parameter)

---

### Advantages

- Nearest neighbor query: **O(log n)** on average for low dimensions
- Reduces brute-force distance computations
- Works best for **low-dimensional spaces ($d < 20$)**

### Limitations

- In high dimensions, KD-Tree degenerates
- Queries may require visiting many nodes → performance ≈ brute-force

---

## 3. Ball Tree

### What is a Ball Tree?

- Ball Tree = tree where nodes are **hyperspheres (balls)** enclosing subsets of points
- Each node stores:
  - **Center of ball**
  - **Radius**
- Children: partition points recursively into smaller balls
- Better than KD-Tree when data is **moderate-dimensional or clusters unevenly distributed**

#### Structure




    Ball(center, radius)
   /                   \


child ball 1             child ball 2



- During query, balls **outside search radius can be pruned**, reducing distance computations

---

### Comparison: KD-Tree vs Ball Tree

| Feature | KD-Tree | Ball Tree |
|---|---|---|
| Partition | Axis-aligned planes | Hyperspheres |
| Best for | Low dimensions (<20) | Moderate dimensions (20–100) |
| Handles clusters | Poorly (axis-aligned splits may cut clusters) | Better (balls adapt to cluster shapes) |
| Complexity | O(log n) average | O(log n) average |
| Use in sklearn | `algorithm='kd_tree'` | `algorithm='ball_tree'` |

---

## 4. Other Improvements / Modern Techniques

| Technique | Explanation |
|---|---|
| Approximate Nearest Neighbors | Trade exactness for speed in large datasets (e.g., FAISS, HNSW) |
| Condensed / Edited KNN | Remove redundant or noisy points to reduce memory and variance |
| Hybrid / ensemble KNN | Combine multiple KNNs or with other classifiers to reduce variance |
| Feature embedding | Use learned embeddings (word2vec, CNN embeddings) and then apply KNN |
| Parallelization (`n_jobs=-1`) | Speed up computations on large datasets |

---

## 5. Summary

1. **Improving KNN**: Proper $k$, distance weighting, scaling, feature selection, metric learning.
2. **Speeding up KNN**: KD-Tree (axis-aligned splits, low-dim), Ball Tree (spherical splits, moderate-dim), approximate NN.
3. **Tradeoffs**: Trees are fast for low-to-moderate dimensions but degrade in high dimensions; approximate methods balance accuracy and speed.

> KD-Tree and Ball Tree are essentially **data structures** for efficiently organizing the training points so that KNN queries do **not require brute-force distance computation**.




