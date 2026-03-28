# Standardisation vs Normalisation

## What is Normalisation?

Normalisation (also called **Min-Max Scaling**) rescales data to a fixed range, typically **[0, 1]**.

### Formula

$$X' = \frac{X - X_{min}}{X_{max} - X_{min}}$$

- $X$ → original value
- $X_{min}$ → minimum value in the feature
- $X_{max}$ → maximum value in the feature
- $X'$ → scaled value (always between 0 and 1)

### Properties
- Output range is **bounded**: always [0, 1] (or [a, b] if you specify a custom range)
- Sensitive to **outliers** — one extreme value compresses all others into a narrow range
- Does **not** assume any distribution of the data

---

## What is Standardisation?

Standardisation (also called **Z-score Normalisation**) transforms data to have a **mean of 0** and a **standard deviation of 1**.

### Formula

$$X' = \frac{X - \mu}{\sigma}$$

- $X$ → original value
- $\mu$ → mean of the feature
- $\sigma$ → standard deviation of the feature
- $X'$ → standardised value (unbounded, centred at 0)

### Properties
- Output range is **unbounded** — values can be negative or greater than 1
- **Robust to outliers** compared to normalisation
- Assumes (or works best with) data that is **roughly normally distributed**

---

## Side-by-Side Comparison

| Property | Normalisation (Min-Max) | Standardisation (Z-score) |
|---|---|---|
| Formula | (X − min) / (max − min) | (X − μ) / σ |
| Output range | [0, 1] (bounded) | Unbounded (centred at 0) |
| Mean after scaling | Not fixed | 0 |
| Std dev after scaling | Not fixed | 1 |
| Outlier sensitivity | High | Low |
| Distribution assumption | None | Works best with normal dist. |

---

## When to Use What?

### Use **Normalisation** when:
- You need values in a **specific bounded range** (e.g., pixel values, neural net inputs)
- The algorithm requires inputs in [0, 1] — e.g., **KNN, Neural Networks**
- The data does **not** follow a Gaussian distribution
- You're working with **image data** or anything where relative scale matters

### Use **Standardisation** when:
- The algorithm **assumes zero mean / unit variance** — e.g., **PCA, SVM, Linear/Logistic Regression, LDA**
- Your data has **outliers** (standardisation is less distorted by them)
- You don't know the min/max of future data (standardisation generalises better)
- Features have **different units** (e.g., age in years vs. salary in lakhs)

---

## Quick Rule of Thumb

> **"Normalise for neural nets and bounded algorithms.  
> Standardise for everything else — especially when distributions matter."**

---

## Algorithms & Recommended Scaling

| Algorithm | Recommended Scaling |
|---|---|
| Linear Regression | Standardisation |
| Logistic Regression | Standardisation |
| SVM | Standardisation |
| PCA | Standardisation |
| K-Means Clustering | Either (Standardisation preferred) |
| KNN | Normalisation |
| Neural Networks | Normalisation |
| Decision Trees / Random Forest | **Neither needed** |
| Gradient Boosting (XGBoost etc.) | **Neither needed** |

> Tree-based models are **invariant to scaling** — they split on thresholds, so the scale of a feature doesn't affect the result.

---

## Example (Python)

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

X = np.array([[10], [20], [30], [40], [200]])  # has an outlier

# Normalisation
norm = MinMaxScaler()
print(norm.fit_transform(X))
# → [[0.  ], [0.05], [0.1 ], [0.16], [1.  ]]  — outlier squashes everything

# Standardisation
std = StandardScaler()
print(std.fit_transform(X))
# → [[-0.8], [-0.6], [-0.5], [-0.3], [2.3]]  — outlier is just another z-score
```
