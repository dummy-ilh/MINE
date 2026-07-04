## DS & Algo topics specific to ML / Data Science interviews

These interviews blend LeetCode-style coding with "implement this ML concept from scratch" and numpy/pandas fluency. Here's what actually comes up.

---

### 1. NumPy essentials (you'll be judged on vectorization, not loops)

```python
import numpy as np

a = np.array([1,2,3])
a.shape, a.dtype, a.ndim

np.zeros((3,3)); np.ones((3,3)); np.eye(3)          # identity matrix
np.arange(0,10,2)                                     # like range but array
np.linspace(0,1,5)                                    # 5 evenly spaced points

a + b, a * b, a @ b                                    # elementwise vs matrix mult (@)
a.reshape(3,1)                                         # reshape
a.T                                                     # transpose
a.sum(axis=0)                                           # sum down columns
a.mean(), a.std(), a.var()

a[a > 2]                                                # boolean masking (huge for interviews)
np.where(a > 2, 1, 0)                                   # vectorized if/else
np.argmax(a), np.argsort(a)
np.dot(a, b)                                            # dot product / matrix mult
np.linalg.norm(a)                                       # vector magnitude (used in distance calcs)
```

**Why this matters:** "write it without a for loop" is an implicit expectation. If your k-NN or distance calc has a Python loop over rows, that's a red flag in an ML interview.

---

### 2. Pandas essentials

```python
import pandas as pd

df.head(); df.info(); df.describe()
df.isnull().sum()                     # missing values per column
df.dropna(); df.fillna(0)
df.groupby("col").agg({"x": "mean"})
df.merge(df2, on="key", how="left")
df.sort_values("col", ascending=False)
df.apply(lambda row: ..., axis=1)
df["col"].value_counts()
pd.get_dummies(df["category"])         # one-hot encoding
df.pivot_table(values="x", index="a", columns="b", aggfunc="sum")
```

---

### 3. "Implement from scratch" — the classic ML coding questions

**k-Nearest Neighbors**
```python
def knn_predict(X_train, y_train, x_query, k):
    dists = np.linalg.norm(X_train - x_query, axis=1)
    nearest_idx = np.argsort(dists)[:k]
    nearest_labels = y_train[nearest_idx]
    return Counter(nearest_labels).most_common(1)[0][0]
```

**k-Means clustering**
```python
def kmeans(X, k, iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None] - centroids, axis=2)   # (n_points, k)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids
```

**Linear regression via gradient descent**
```python
def linear_regression(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0
    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        dw = (2/n) * X.T @ error
        db = (2/n) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return w, b
```

**Logistic regression (sigmoid + BCE loss)**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0
    for _ in range(epochs):
        z = X @ w + b
        y_pred = sigmoid(z)
        dw = (1/n) * X.T @ (y_pred - y)
        db = (1/n) * np.sum(y_pred - y)
        w -= lr * dw
        b -= lr * db
    return w, b
```

**Softmax**
```python
def softmax(z):
    z = z - np.max(z)          # numerical stability trick — always subtract max
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)
```

---

### 4. Common distance / similarity metrics (memorize these formulas + code)

```python
euclidean = np.linalg.norm(a - b)
manhattan = np.sum(np.abs(a - b))
cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

### 5. Classic array/stats coding questions specific to DS interviews

- **Compute mean/median/mode/std without libraries** — shows you understand the underlying math, not just `.mean()`.
- **Moving average / rolling window** — sliding window pattern from before, applied to time series.
```python
def moving_average(arr, window):
    return [sum(arr[i:i+window])/window for i in range(len(arr)-window+1)]
```
- **Reservoir sampling** — sample k items from a stream of unknown length, each with equal probability. Comes up a lot for "sample from a huge dataset."
```python
import random
def reservoir_sample(stream, k):
    result = []
    for i, item in enumerate(stream):
        if i < k:
            result.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                result[j] = item
    return result
```
- **Compute correlation coefficient from scratch** (Pearson).
- **Sample from a given probability distribution** using `np.random.choice(values, p=probabilities)`.

---

### 6. Complexity of common ML algorithms (they WILL ask this)

| Algorithm | Training | Inference |
|---|---|---|
| k-NN | O(1) (lazy) | O(n·d) per query |
| k-Means | O(n·k·d·iters) | O(k·d) |
| Linear/Logistic Regression (GD) | O(n·d·epochs) | O(d) |
| Decision Tree | O(n·d·log n) | O(depth) |
| Random Forest | O(trees · tree cost) | O(trees · depth) |

---
## DSA-style data manipulation problems (Python/pandas)

These are the "coding" style Google DS interviews actually test — not backtracking/graphs, but clean data transformation logic. Here's a practice set with patterns.

---

### 1. Merge two dataframes to build complete records
**Problem type:** given addresses in one table, city→state mapping in another, build full addresses.

```python
import pandas as pd

addresses = pd.DataFrame({
    "street": ["123 Main St", "456 Oak Ave"],
    "city": ["Springfield", "Shelbyville"],
    "zip": ["11111", "22222"]
})

city_state = pd.DataFrame({
    "city": ["Springfield", "Shelbyville"],
    "state": ["IL", "IL"]
})

merged = addresses.merge(city_state, on="city", how="left")
merged["full_address"] = (
    merged["street"] + ", " + merged["city"] + ", " +
    merged["state"] + " " + merged["zip"]
)
```
**Watch for:** `how="left"` vs `"inner"` — decide based on whether unmatched rows should be dropped or kept with NaN. Always ask: can a city map to multiple states (dedupe issue)? Check with `city_state["city"].duplicated().any()` before merging.

---

### 2. Get the latest/last row per group (the SQL window-function pattern, in pandas)
```python
df.sort_values("timestamp").groupby("user_id").tail(1)

# or, more explicit / handles ties by keeping first occurrence:
df["rank"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=False)
latest = df[df["rank"] == 1]
```
**Gotcha to state out loud:** what happens with exact tied timestamps? `rank(method="first")` breaks ties by row order — mention this tradeoff, same as the SQL version.

---

### 3. Rolling/moving metrics (rolling average, cumulative sum)
```python
df = df.sort_values("date")
df["rolling_7day_avg"] = df["sales"].rolling(window=7).mean()
df["cumulative_sales"] = df["sales"].cumsum()

# per-group rolling (e.g. per store)
df["rolling_avg"] = df.groupby("store_id")["sales"].transform(lambda x: x.rolling(7).mean())
```

---

### 4. Deduplicate / find duplicates with conditions
```python
df.duplicated(subset=["user_id", "date"])                 # bool mask
df.drop_duplicates(subset=["user_id"], keep="last")         # keep most recent

# find users with duplicate entries on same day
dupes = df[df.duplicated(subset=["user_id", "date"], keep=False)]
```

---

### 5. Pivot / reshape data
```python
# long to wide
df.pivot_table(values="value", index="user_id", columns="metric_name", aggfunc="sum")

# wide to long
df.melt(id_vars=["user_id"], value_vars=["jan", "feb", "mar"],
        var_name="month", value_name="sales")
```

---

### 6. Simulate a distribution (numpy) — the "truncated normal" style question
```python
import numpy as np
from scipy.stats import truncnorm

def sample_truncated_normal(mean, std, lower, upper, n):
    a, b = (lower - mean) / std, (upper - mean) / std   # convert to standard-normal bounds
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=n)

samples = sample_truncated_normal(mean=50, std=10, lower=0, upper=100, n=1000)
```
Without scipy (manual rejection sampling — good to know if you can't import scipy in the interview):
```python
def sample_truncated_normal_manual(mean, std, lower, upper, n):
    samples = []
    while len(samples) < n:
        batch = np.random.normal(mean, std, size=n)
        valid = batch[(batch >= lower) & (batch <= upper)]
        samples.extend(valid.tolist())
    return samples[:n]
```

---

### 7. Group-and-rank (top-k per group — very common)
```python
# top 3 highest-paid employees per department
df["rank"] = df.groupby("department")["salary"].rank(method="dense", ascending=False)
top3 = df[df["rank"] <= 3]
```
**Say out loud:** `method="dense"` vs `"min"` — dense ranking doesn't skip numbers after ties (1,1,2 vs 1,1,3), and this changes whether you get exactly 3 or more than 3 rows per group when there are ties. This is the same "how do you handle ties" question interviewers ask in SQL, just in pandas.

---

### 8. Handle missing data thoughtfully (not just `.dropna()`)
```python
df.isnull().sum()                       # audit first
df["col"].fillna(df["col"].median())     # numeric: median more robust to outliers than mean
df["cat_col"].fillna("Unknown")          # categorical: explicit placeholder
df.groupby("group")["value"].transform(lambda x: x.fillna(x.mean()))  # group-wise fill
```
**Interview point to make:** dropping rows with missing data can bias your dataset if missingness isn't random (MCAR vs MAR vs MNAR) — mention this, it's literally listed as a Google DS question (how would you handle bias from removing missing values).

---

### 9. Detect an anomaly in a time series (common "explain what you'd investigate" question)
```python
mean, std = df["value"].mean(), df["value"].std()
df["z_score"] = (df["value"] - mean) / std
anomalies = df[df["z_score"].abs() > 3]     # >3 std devs = simple outlier flag

# better for time series: rolling z-score (adapts to trend/seasonality)
rolling_mean = df["value"].rolling(30).mean()
rolling_std = df["value"].rolling(30).std()
df["rolling_z"] = (df["value"] - rolling_mean) / rolling_std
```

---
## Major algorithms actually relevant for Data Scientist roles

These are different from generic SWE DSA — they're the algorithms that show up because of what a DS *does*: sampling, working with distributions, feature engineering at scale, and basic ML mechanics. Here's the full list, grouped by why it matters.

---

### 1. Sampling algorithms (very high yield — comes up constantly)

**Reservoir Sampling** — sample k items from a stream of unknown/huge size, uniform probability
```python
import random
def reservoir_sample(stream, k):
    result = []
    for i, item in enumerate(stream):
        if i < k:
            result.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                result[j] = item
    return result
```
Why it matters: "sample from a massive log file without loading it all into memory" is a direct DS interview question.

**Weighted random sampling**
```python
np.random.choice(items, size=n, p=probabilities, replace=False)
```

**Stratified sampling** (preserve group proportions)
```python
df.groupby("group", group_keys=False).apply(lambda x: x.sample(frac=0.1))
```

**Bootstrap resampling** (for confidence intervals without assuming a distribution)
```python
def bootstrap_ci(data, n_iterations=1000, ci=95):
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_iterations)]
    lower = np.percentile(means, (100-ci)/2)
    upper = np.percentile(means, 100 - (100-ci)/2)
    return lower, upper
```

**Rejection sampling / inverse transform sampling** — generate samples from a custom distribution using only `uniform()`.

---

### 2. Hashing-based algorithms (for scale problems)

**Count-distinct at scale** — HyperLogLog concept (probabilistic cardinality estimation). You won't implement it from scratch, but you should be able to explain: instead of storing a full set to count uniques, hash each item and track the longest run of leading zeros in the hash — this approximates cardinality in O(1) space. Comes up as "how would you count unique users across a billion rows without enough memory."

**MinHash** — estimate Jaccard similarity between two large sets without storing them fully. Relevant for de-duplication / similarity-at-scale questions.

**Bloom filter** — probabilistic "have I seen this before" check with false positives but no false negatives, O(1) space per check regardless of set size.

---

### 3. Tree-based algorithms (need to know mechanics, not just call sklearn)

**Decision tree split logic (Gini/Entropy)**
```python
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

def information_gain(y, y_left, y_right):
    p = len(y_left) / len(y)
    return gini(y) - (p * gini(y_left) + (1-p) * gini(y_right))
```
Interviewers ask you to reason through "why does entropy/gini prefer this split" — know the formula cold.

---

### 4. Dimensionality reduction — PCA from scratch (classic "implement from scratch" ask)
```python
def pca(X, n_components):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    return X_centered @ eigvecs[:, idx]
```

---

### 5. Naive Bayes (from scratch — common)
```python
def naive_bayes_predict(X_train, y_train, x_query):
    classes = np.unique(y_train)
    posteriors = {}
    for c in classes:
        X_c = X_train[y_train == c]
        prior = len(X_c) / len(X_train)
        likelihood = np.prod([
            np.exp(-((x_query[i] - X_c[:,i].mean())**2) / (2*X_c[:,i].var())) /
            np.sqrt(2*np.pi*X_c[:,i].var())
            for i in range(len(x_query))
        ])
        posteriors[c] = prior * likelihood
    return max(posteriors, key=posteriors.get)
```

---

### 6. A/B testing math-as-algorithm (statistics that gets coded up live)

**Sample ratio mismatch check** — chi-square test to verify traffic split is as expected:
```python
from scipy.stats import chisquare
observed = [4980, 5020]      # actual counts in A vs B
expected = [5000, 5000]       # expected 50/50 split
stat, p_value = chisquare(observed, expected)
```

**Sample size calculator for A/B tests**
```python
from statsmodels.stats.power import NormalIndPower
analysis = NormalIndPower()
n = analysis.solve_power(effect_size=0.2, alpha=0.05, power=0.8)
```

**Difference-in-differences** (increasingly asked per recent reports) — conceptually: `(treatment_after - treatment_before) - (control_after - control_before)`, coded as simple group-mean arithmetic, but you should be able to explain why it removes time-invariant confounders.

---

### 7. Graph-lite algorithms that show up in DS (less common, but real)

- **Connected components** for clustering/de-duplication of records (e.g. "these 3 user records are actually the same person" — union-find).
- **PageRank-style iterative algorithm** for ranking (if asked to reason about recommendation ranking systems).

---

### Priority order to actually study, given what you're targeting
1. Reservoir sampling + bootstrap CI (near-certain to come up)
2. Pandas groupby/rank/window patterns (already covered above)
3. Gini/entropy split logic + PCA from scratch (common whiteboard asks)
4. A/B test sanity check code (chi-square, sample size) — ties directly to the causal inference shift mentioned in recent reports
5. Bloom filter / HyperLogLog — know the *concept* verbally, rarely asked to code fully

Want worked practice problems for reservoir sampling or bootstrap CI specifically, since those are the highest-yield ones?
