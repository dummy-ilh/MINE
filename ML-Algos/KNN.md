Good, it runs cleanly. Let's go through KNN top to bottom.

## The one-liner

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

## Diagnostics — the pitfalls interviewers actually probe**1. Feature scaling (the #1 pitfall, always comes up)**
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

Want me to package this into a saved reference doc (Word or Markdown) so you can review it before the interview?
