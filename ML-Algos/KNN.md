## KNN - K Nearest Neighnours

KNN predicts a new point by looking at its `k` closest neighbors in the training data and going with the majority (classification) or the average (regression). No training phase in the usual sense — it just memorizes the data and does the work at prediction time. That's why it's called a **lazy learner**, and it's the thing interviewers most want you to say unprompted.

**Analogy:** guessing someone's taste in restaurants by asking their 5 closest friends what they'd order, and going with the majority.**When to use it**
- Small-to-medium datasets, low dimensionality
- Decision boundary is irregular / non-linear and you don't want to assume a functional form
- You need a fast-to-build baseline (zero training time)
- Data is well-scaled and mostly clean

**When not to use it**
- High-dimensional data (curse of dimensionality — distances stop being meaningful, more below)
- Large datasets (prediction is O(n) per query with brute force — slow at inference, the opposite of most models)
- Features on very different scales without normalization
- Lots of irrelevant/noisy features (they pollute the distance metric)

**Without KNN:** most other classifiers (logistic regression, SVM, trees) learn an explicit decision boundary or function during training and predict in near-constant time. KNN skips that — it defers all the "thinking" to prediction time, trading train speed for inference speed. That tradeoff is worth stating explicitly in an interview.

## The math

| Symbol | Meaning |
|---|---|
| $x_q$ | query point (the one we're predicting) |
| $x_i, y_i$ | a training point and its label |
| $d(x_i, x_q)$ | distance between them |
| $k$ | number of neighbors to consider |
| $N_k(x_q)$ | the set of $k$ nearest training points to $x_q$ |

**Distance metrics** (this is Minkowski distance, generalized):

$$d(x_i, x_q) = \left(\sum_{j=1}^{n} |x_{ij} - x_{qj}|^p\right)^{1/p}$$

- $p=1$ → Manhattan distance (sum of absolute differences, robust to outliers)
- $p=2$ → Euclidean distance (straight-line distance, the default, sensitive to outliers)
- Other options: Cosine distance (for text/high-dim sparse data, ignores magnitude), Hamming distance (for categorical/binary features)

**Classification prediction** — majority vote:

$$\hat{y}_q = \text{mode}\{y_i : x_i \in N_k(x_q)\}$$

Plain language: look at the labels of the k closest points, output whichever label shows up most.

**Regression prediction** — average:

$$\hat{y}_q = \frac{1}{k}\sum_{x_i \in N_k(x_q)} y_i$$

**Distance-weighted variant** (fixes the "all neighbors count equally" issue):

$$\hat{y}_q = \frac{\sum_{i} w_i \, y_i}{\sum_i w_i}, \quad w_i = \frac{1}{d(x_i, x_q)}$$

Closer neighbors get more say. This is what `weights="distance"` does in sklearn — worth mentioning as the fix when an interviewer asks "what if a neighbor is a clear outlier but still in your k-set?"

**Worked example by hand** — predict if a customer churns based on (age, monthly spend). Training data:

| Point | Age | Spend | Churn? |
|---|---|---|---|
| A | 25 | 40 | No |
| B | 45 | 20 | Yes |
| C | 30 | 45 | No |
| D | 50 | 15 | Yes |

Query: age=32, spend=38. Use $k=3$, Euclidean distance:

- $d(A) = \sqrt{(32-25)^2+(38-40)^2} = \sqrt{49+4} = \sqrt{53} \approx 7.28$
- $d(B) = \sqrt{(32-45)^2+(38-20)^2} = \sqrt{169+324} = \sqrt{493} \approx 22.20$
- $d(C) = \sqrt{(32-30)^2+(38-45)^2} = \sqrt{4+49} = \sqrt{53} \approx 7.28$
- $d(D) = \sqrt{(32-50)^2+(38-15)^2} = \sqrt{324+529} = \sqrt{853} \approx 29.21$

3 nearest: A (7.28, No), C (7.28, No), B (22.20, Yes) → majority = **No, churn predicted "No"**.

Notice: without scaling, "spend" (range ~15-45) and "age" (range ~25-50) happen to be similar scales here, but if spend were in dollars in the thousands, it would completely dominate the distance and age would become irrelevant. That's the segue into the single most important practical pitfall — scaling — which the diagnostics below cover.

To see how distance weighting works, let’s run through a concrete example.

Imagine we are using the **$k$-Nearest Neighbors ($k$-NN)** algorithm to predict the price of a house (Regression) based on its distance from a city center. We want to predict the value for a query house ($x_q$), and we set $k = 3$.

Our algorithm finds the 3 closest neighbor houses ($x_1, x_2, x_3$).

---
### Example 2

Nearest neighbors:

| Neighbor | Distance ($d_i$) | Price ($y_i$) |
|----------|------------------:|--------------:|
| 1 | 2 | \$300,000 |
| 2 | 4 | \$200,000 |
| 3 | 10 | \$100,000 |

Weight formula:

$$w_i=\frac{1}{d(x_i,x_q)}$$

Weights:

- $w_1=\frac12=0.5$
- $w_2=\frac14=0.25$
- $w_3=\frac1{10}=0.1$

Total weight:

$$\sum w_i=0.5+0.25+0.1=0.85$$

Weighted sum:

$$
\sum(w_i y_i)=0.5(300000)+0.25(200000)+0.1(100000)=210000
$$

Final prediction:

$$
\hat y_q=\frac{210000}{0.85}\approx \mathbf{\$247,059}
$$

Unweighted average:

$$
\frac{300000+200000+100000}{3}=\mathbf{\$200,000}
$$

> **Takeaway:** Distance weighting gives more influence to closer neighbors, producing a prediction closer to the nearest house.

---

## Distance-Weighted k-NN Classification

Neighbors:

| Neighbor | Distance | Label |
|----------|---------:|-------|
| 1 | 0.1 | Spam |
| 2 | 0.5 | Ham |
| 3 | 0.8 | Ham |

Weights:

- Spam: $\frac1{0.1}=10$
- Ham: $\frac1{0.5}=2$
- Ham: $\frac1{0.8}=1.25$

Weighted votes:

- **Spam:** $10$
- **Ham:** $2+1.25=3.25$

**Prediction:** **Spam**

Without weighting:

- Spam = 1 vote
- Ham = 2 votes → **Ham** ❌

> **Takeaway:** In classification, closer neighbors cast stronger votes, preventing distant neighbors from dominating the prediction.
---

### Why Distance Weighting Fixed a Wrong Prediction Here

If you used the standard **majority vote** formula without distance weights:

* **Spam Count:** 1 vote
* **Ham Count:** 2 votes
* **Unweighted Prediction:** **Ham** ❌ (Incorrectly classified)

**The Takeaway:** Without distance weighting, the query was overwhelmed by Neighbors 2 and 3 simply because there were more of them, even though Neighbor 1 was practically identical to our query document. Distance weighting ensures that absolute proximity trumps raw numbers.

## Code (verified to run)

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline: scaling is mandatory for KNN (it's distance-based)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Tune k, weighting, and distance metric via cross-validation
param_grid = {
    "knn__n_neighbors": range(1, 21, 2),   # odd k avoids ties in binary case
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2]                       # 1=Manhattan, 2=Euclidean
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)     # -> {'n_neighbors': 9, 'p': 1, 'weights': 'uniform'}
print("CV accuracy:", round(grid.best_score_, 3))   # -> 0.986

y_pred = grid.best_estimator_.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 3))  # -> 1.0
```

The two things an interviewer is checking for here: **(1)** did you scale features inside a `Pipeline` (not before the split — that would leak test-set statistics into training), and **(2)** did you tune `k` via cross-validation instead of guessing.

## Diagnostics — the pitfalls interviewers actually probe

**1. Feature scaling (the #1 pitfall, always comes up)**
KNN uses raw distance, so a feature ranging 0–100,000 (e.g. income) will completely swamp a feature ranging 0–1 (e.g. a binary flag), regardless of which one actually matters. Symptom: model ignores features you know are predictive. Fix: `StandardScaler` or `MinMaxScaler`, fit only on the training set, inside a pipeline so cross-validation doesn't leak.

**2. The k tradeoff (bias-variance, shown above)**
Small $k$ (e.g. $k=1$) → the boundary hugs individual training points → low bias, high variance → overfitting (train error near 0, test error high). Large $k$ → prediction is smoothed over a big neighborhood → high bias, low variance → underfitting (both errors rise, train error especially). The chart above is a textbook bias-variance curve — this is often *the* question: "what happens as k→1 vs k→n?" k→n degenerates to always predicting the majority class in the whole dataset.

**3. Curse of dimensionality**
As feature count grows, all points start looking equidistant — "nearest" neighbor stops being meaningfully closer than "farthest" one. Symptom: accuracy degrades as you add features, even relevant ones, especially past ~10-20 dimensions with limited data. Fix: dimensionality reduction (PCA), feature selection, or don't use KNN at all past moderate dimensionality.

**4. Class imbalance**
With unbalanced classes, a query near the boundary will have most of its k neighbors belong to the majority class purely by base rate, not by proximity signal. Symptom: minority class recall is poor even though overall accuracy looks fine. Fix: distance-weighting, resampling (SMOTE/undersampling), or a class-weighted voting scheme.

**5. Inference cost**
Training is O(1) (just store the data) but a brute-force prediction is O(n·d) per query — the opposite of most models. Symptom: fine on a small dataset in a demo, unusably slow at production scale. Fix: KD-trees / Ball-trees (sklearn does this automatically for lower dimensions), or approximate nearest-neighbor libraries (FAISS, Annoy) for large/high-dim data — KD-trees themselves degrade to brute force above roughly 20 dimensions, another curse-of-dimensionality symptom worth naming.

**6. Ties in voting**
With even $k$ in binary classification, a 50/50 split has no natural winner (sklearn breaks ties by class order, which is arbitrary). Fix: use odd $k$ for binary problems, or use distance-weighting so it's rarely an exact tie.

## Practice Q&A

**Q1 (easy).** Is KNN a parametric or non-parametric model? Why?

<br>

*A: Non-parametric — it makes no assumption about the functional form of the decision boundary and the "model" grows with the data (it's just the stored training set). Contrast with logistic regression, which fixes a finite number of parameters regardless of dataset size.*

**Q2 (easy).** Why does KNN have "zero training time" but slow prediction time?

<br>

*A: "Training" is just storing the data — no optimization happens. All the computation (finding nearest neighbors) is deferred to prediction time, making it the reverse of most models where training is expensive and inference is cheap.*

**Q3 (medium).** You scale your features and accuracy jumps from 65% to 89%. Explain what likely happened before scaling.

<br>

*A: One or more features had a much larger numeric range than the others, so Euclidean distance was dominated by that feature — the model was effectively only "looking at" that one dimension. Scaling puts all features on comparable footing so the distance metric reflects all of them.*

**Q4 (medium).** Given the bias-variance chart above, why is k=1 always the training-set-perfect but often the worst real-world choice?

<br>

*A: At k=1, every training point is its own nearest neighbor, so it "predicts" its own label — training error is exactly 0. But that also means predictions are extremely sensitive to noise/outliers in individual points, which is high variance and generalizes poorly, showing up as elevated test error.*

**Q5 (medium — calculation).** With k=5 and neighbor distances [1, 2, 3, 10, 20] having labels [Yes, Yes, No, No, No], what does uniform-vote KNN predict? What about distance-weighted?

<br>

*A: Uniform vote: 3 No vs 2 Yes → predicts No. Distance-weighted: weights = 1/d = [1, 0.5, 0.33, 0.1, 0.05]. Yes weight = 1.5, No weight = 0.48 → predicts Yes. This shows how weighting can flip a decision when far-away neighbors are barely relevant.*

**Q6 (hard).** Your model performs great on 5 features but degrades when you add 50 more (even though some are informative). What's happening, and what's the fix?

<br>

*A: Curse of dimensionality — as dimensions grow, the ratio of nearest-to-farthest distance approaches 1, so "nearest neighbor" carries less and less signal, and irrelevant new features add noise to the distance calculation that swamps the informative ones. Fix: feature selection to drop the uninformative dimensions, or PCA to compress into fewer, denser dimensions before running KNN.*

**Q7 (hard — spot the bug).** A colleague scales the entire dataset with `StandardScaler().fit_transform(X)` before doing `train_test_split`. What's wrong, and why does it matter more for KNN than for, say, a decision tree?

<br>

*A: This is data leakage — the scaler's mean/std are computed using test-set values, so test data has influenced the "training" preprocessing, giving an overly optimistic evaluation. It matters especially for KNN because the entire prediction mechanism is the distance metric itself, which is exactly what scaling touches — a decision tree's splits are threshold-based per feature and less directly corrupted by this leak (though it's still bad practice there too).*

**Q8 (hard).** Interviewer asks: "how would you pick k in production without cross-validation eating your whole budget?" What do you say?

<br>

*A: A reasonable heuristic is $k \approx \sqrt{n}$ as a starting point, then do a coarse cross-validated sweep around that value (e.g. k ∈ {√n/2, √n, 2√n}) rather than a dense grid search over every integer — narrows the search using domain knowledge of the bias-variance curve's shape instead of brute-forcing it.*

---
## Choosing the Value of $k$ in k-NN

The value of **$k$** determines how many nearest neighbors participate in the prediction.

### Effect of Different $k$

- **Small $k$ (e.g., 1, 3):**
  - Captures local patterns.
  - Sensitive to noise (**high variance**).

- **Large $k$ (e.g., 15, 25):**
  - Produces smoother predictions.
  - May ignore local structure (**high bias**).

---

### How is $k$ Chosen?

The best value of $k$ is usually selected using **Cross-Validation**.

Example:

| $k$ | Validation Accuracy |
|----:|--------------------:|
| 1 | 91% |
| 3 | 94% |
| 5 | **96%** ✅ |
| 7 | 95% |
| 11 | 93% |

Choose the value with the **best validation performance** (here, $k=5$).

> **Rule of Thumb:** Start with $k \approx \sqrt{N}$ (where $N$ is the number of training samples), then tune using cross-validation.

---

## KD-Tree

A **KD-Tree (k-dimensional tree)** is a binary tree that recursively splits the feature space along feature axes.

Instead of comparing the query with **every training sample**, it searches only the most promising regions.

### Example

Instead of checking **100,000** points:

```
Query
  ↓
KD-Tree
  ↓
Relevant regions only
  ↓
Nearest neighbors
```

**Advantages**

- Much faster than brute-force search.
- Works well for **low-dimensional data** (typically < 20–30 features).

**Limitation**

Performance degrades in high-dimensional data (Curse of Dimensionality).

---

## Ball-Tree

A **Ball-Tree** groups nearby points into nested **hyperspheres (balls)** instead of splitting by axes.

```
Large Ball
├── Small Ball
└── Small Ball
```

During search, entire balls that cannot contain the nearest neighbor are skipped.

**Advantages**

- Better than KD-Tree for **higher-dimensional** datasets.
- Efficient with non-axis-aligned data.

---

## KD-Tree vs Ball-Tree

| KD-Tree | Ball-Tree |
|----------|-----------|
| Splits by feature axes | Groups points into hyperspheres |
| Faster for low dimensions | Better for medium/high dimensions |
| Simpler structure | More flexible structure |

> **Note:** In `scikit-learn`, `algorithm='auto'` automatically chooses the best search method.

---

## Hyperparameters to Tune (`sklearn.neighbors.KNeighborsClassifier` / `KNeighborsRegressor`)

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `n_neighbors` | Number of neighbors ($k$) | 3, 5, 7, 11, 15 |
| `weights` | Neighbor weighting | `'uniform'`, `'distance'` |
| `metric` | Distance metric | `'minkowski'`, `'euclidean'`, `'manhattan'`, `'cosine'`* |
| `p` | Power parameter for Minkowski | 1 (Manhattan), 2 (Euclidean) |
| `algorithm` | Neighbor search algorithm | `'auto'`, `'kd_tree'`, `'ball_tree'`, `'brute'` |
| `leaf_size` | Tree leaf size (KD/Ball Tree) | 20–50 (default: 30) |

\* `cosine` is supported via the brute-force algorithm.

---

### Common GridSearchCV Example

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["auto", "kd_tree", "ball_tree", "brute"]
}
```

> **Most Important Hyperparameters:** `n_neighbors`, `weights`, and `metric`. The search `algorithm` mainly affects **speed**, not prediction quality.


Both Google and Apple frequently include $k$-Nearest Neighbors ($k$-NN) concepts in their interview loops for Data Scientist, Machine Learning Engineer, and Research Scientist roles. However, they rarely ask you to just define it. Instead, they focus on **architectural scaling**, **mathematical edge cases**, and **production trade-offs**.

Past core conceptual and coding interview questions from both companies include:

---

### 1. The Pure Coding Challenge (Common at Apple)

> **Question:** *"Write a production-ready $k$-NN classifier from scratch using only Python and NumPy. Do not use Scikit-Learn."*

* **What they are looking for:** They want to see clean vectorization using NumPy instead of nested loops.
* **The Trap:** If you use a `for` loop to calculate the distance of every data point one by one, you will fail the efficiency bar. They expect you to compute pairwise distances globally using matrix operations (e.g., Euclidean distance via dot products: $\sqrt{\sum(x-y)^2} = \sqrt{x^2 - 2xy + y^2}$).

---

### 2. The Scaling & Latency Dilemma (Classic Google System Design)

> **Question:** *"You have a billion user-profile vectors, and you want to use $k$-NN to serve real-time recommendations on Google Search. Brute-force $O(N)$ is too slow. How do you scale this system to keep latency under 10 milliseconds?"*

* **What they are looking for:** This tests your knowledge of spatial partitioning trees (KD-Trees, Ball Trees) and **Approximate Nearest Neighbor (ANN)** algorithms.
* **The Ideal Answer:** You should explain that KD-Trees fail in high dimensions due to the "curse of dimensionality." To solve a 1-billion-scale problem at Google, you must trade 100% mathematical accuracy for speed using algorithms like **HNSW (Hierarchical Navigable Small World graphs)** or **LSH (Locality-Sensitive Hashing)** to find *approximate* nearest neighbors in $O(\log N)$ or $O(1)$ time.

---

### 3. The "Curse of Dimensionality" (Apple Hardware & ML Question)

> **Question:** *"When deploying a $k$-NN model on an Apple Watch to detect anomalies in health data, you notice its classification accuracy drops significantly as you add more tracking features (dimensions). Mathematically, why does this happen, and how do you fix it?"*

* **What they are looking for:** This tests your deep geometric understanding of high-dimensional vector spaces.
* **The Ideal Answer:** As the number of dimensions ($D$) grows, the volume of the space grows exponentially, causing the data points to become extremely isolated and sparse. In high dimensions, the distance between the *closest* neighbor and the *farthest* neighbor converges to roughly the same value ($d_{max} \approx d_{min}$). Therefore, Euclidean distance loses its meaning.
* **The Fix:** Mention applying dimensionality reduction (like PCA or Autoencoders) or feature selection before running $k$-NN.

---

### 4. Behavioral & Data Quirks

> **Question:** *"Why is feature scaling (Normalization/Standardization) non-negotiable for $k$-NN? What happens if you skip it?"*

* **The Answer:** Because $k$-NN relies entirely on geometric distance. If Feature A (e.g., Salary) ranges from $\$0$ to $\$200,000$ and Feature B (e.g., Age) ranges from $0$ to $100$, the distance formula will completely ignore Age because the large numerical swings in Salary will dominate the calculations.

> **Question:** *"Why do we almost always choose an odd number for $K$ (like $K=3$ or $K=5$) in a binary classification model?"*

* **The Answer:** To eliminate the risk of mathematical ties. If $K=4$, you could easily end up with exactly 2 neighbors voting for Class A and 2 voting for Class B, requiring an arbitrary tie-breaking rule.



Here is a production-ready, highly optimized $k$-NN classifier written from scratch using only **Python** and **NumPy**.

To make it truly production-ready, this implementation completely avoids slow Python `for` loops during distance calculation. Instead, it leverages a linear algebra trick to compute pairwise distances globally using vectorized matrix operations:


$$\text{Distances} = \sqrt{\sum(X_{\text{train}} - X_{\text{test}})^2} = \sqrt{X_{\text{train}}^2 - 2 \cdot X_{\text{train}} \cdot X_{\text{test}}^T + X_{\text{test}}^2}$$

---

### The Code

```python
import numpy as np

class NumPyKNNClassifier:
    def __init__(self, k: int = 3):
        """
        Production-ready k-NN Classifier using vectorized NumPy operations.
        
        Parameters:
        -----------
        k : int, default=3
            Number of nearest neighbors to consider for classification.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer greater than 0.")
        
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using X as training data and y as target values.
        k-NN is a lazy learner, so this simply stores the data.
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X and y must have the same number of rows/samples.")
            
        if self.k > self.X_train.shape[0]:
            raise ValueError(f"k={self.k} cannot be greater than the number of training samples ({self.X_train.shape[0]}).")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided test data.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Matrix of test samples, shape (n_test_samples, n_features)
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted class labels, shape (n_test_samples,)
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("This NumPyKNNClassifier instance is not fitted yet. Call 'fit' first.")
            
        X_test = np.asarray(X_test, dtype=np.float64)
        
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"Feature dimension mismatch. Expected {self.X_train.shape[1]} features, got {X_test.shape[1]}.")

        # 1. Efficient Pairwise Euclidean Distance Matrix Calculation (No Loops!)
        # Using the algebraic identity: (a - b)^2 = a^2 - 2ab + b^2
        train_squared = np.sum(self.X_train ** 2, axis=1)          # Shape: (n_train,)
        test_squared = np.sum(X_test ** 2, axis=1, keepdims=True)  # Shape: (n_test, 1)
        dot_product = np.dot(X_test, self.X_train.T)               # Shape: (n_test, n_train)
        
        # Combining them throws a broad broadcast operation
        # clip(min=0) safeguards against tiny negative floats due to floating-point imprecision
        dists = np.sqrt(np.clip(test_squared - 2 * dot_product + train_squared, a_min=0, a_max=None))

        # 2. Find the indices of the k smallest distances per test point
        # argpartition is O(N), much faster than sorting the entire array O(N log N)
        k_indices = np.argpartition(dists, self.k, axis=1)[:, :self.k]

        # 3. Retrieve the target labels for those k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]                 # Shape: (n_test, k)

        # 4. Majority Vote Vectorization
        # Since classes can be strings or integers, we iterate row-wise across the test set
        predictions = []
        for row in k_nearest_labels:
            labels, counts = np.unique(row, return_counts=True)
            predictions.append(labels[np.argmax(counts)])

        return np.array(predictions)

# ==========================================
# Example Verification Usage
# ==========================================
if __name__ == "__main__":
    # Mocking standard dataset structure
    X_train_mock = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
    y_train_mock = np.array(["Class_A", "Class_A", "Class_B", "Class_B"])
    
    X_test_mock = np.array([[1.2, 1.9], [5.5, 8.5]])

    # Initialize and fit
    knn = NumPyKNNClassifier(k=2)
    knn.fit(X_train_mock, y_train_mock)
    
    # Predict
    preds = knn.predict(X_test_mock)
    print("Predictions:", preds) 
    # Output expected: ['Class_A' 'Class_B']

```

---

### Production Engineering Details You Need to Highlight to an Interviewer:

1. **Why `np.argpartition` over `np.argsort`?**
* Using `np.argsort()` sorts the entire dataset of distances, costing $O(N \log N)$ complexity.
* `np.argpartition()` only ensures the smallest $k$ items are pushed to the front partition, which runs in a highly optimized **linear time $O(N)$**.


2. **Numerical Stability Guarantee (`np.clip`):**
* Floating-point precision errors during `test_squared - 2 * dot_product + train_squared` can sometimes result in ultra-tiny negative values (e.g., `-1e-15`) for perfectly identical data points. Applying `np.sqrt()` to a negative number outputs `NaN`. `np.clip(..., a_min=0)` strictly neutralizes this risk.


3. **True Memory Efficiency:**
* Passing `dtype=np.float64` explicitly safeguards against arbitrary object types degrading underlying C-level performance speeds in NumPy array math.
