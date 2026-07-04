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
