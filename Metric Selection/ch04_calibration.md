# Chapter 4: Calibration — Platt Scaling, Isotonic Regression, and Beyond

> *"A model that says 'I'm 90% confident' should be right 90% of the time. Not 60%. Not 99%. 90%. That's calibration — and most models fail it badly out of the box."*

---

## 4.1 What Is Calibration?

A model outputs a probability score. But what does that score actually mean?

**Calibration** is the alignment between a model's predicted probabilities and the true frequencies of outcomes.

```
A well-calibrated model:
  Predicts 0.9 probability → event happens ~90% of the time
  Predicts 0.7 probability → event happens ~70% of the time
  Predicts 0.3 probability → event happens ~30% of the time

A poorly calibrated model:
  Predicts 0.9 probability → event happens ~60% of the time  ← overconfident
  Predicts 0.3 probability → event happens ~45% of the time  ← underconfident
```

### Why Calibration Matters

Calibration matters whenever the **magnitude** of a probability is used — not just its rank order.

| Use Case | Why Calibration Matters |
|---|---|
| Medical diagnosis | "80% chance of cancer" drives biopsy decisions |
| Credit risk | Loan approval thresholds set by expected default probability |
| Weather forecasting | "30% chance of rain" determines if you bring an umbrella |
| Ad bidding | Expected click probability × bid price = actual bid amount |
| Ensembling | Combining models requires comparable probability scales |
| Uncertainty-aware decisions | Downstream systems rely on confidence as a signal |

> **Key distinction:** A model can rank examples perfectly (high AUC) but be terribly calibrated. These are independent properties. AUC measures discriminative ability; calibration measures probabilistic accuracy.

---

## 4.2 Diagnosing Calibration: The Reliability Diagram

The **reliability diagram** (also called a calibration curve) is the primary diagnostic tool.

### How to Build One

1. Collect model predictions and true labels on a held-out set
2. Bin predictions into K buckets (e.g., [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0])
3. For each bucket, compute:
   - Mean predicted probability (x-axis)
   - Fraction of positives (y-axis)
4. Plot these points against the diagonal

```
Perfect calibration = points fall on the diagonal y = x

     1.0 |                              /
         |                           * /
Fraction |                        /   /
of       |                     * /
Positives|                  /
     0.5 |               * /
         |            /
         |         * /
         |      /  ← Model is overconfident
         |   * /      (predicted high, actual lower)
     0.0 +--/------------------------------
         0.0       0.5                  1.0
                Predicted Probability
```

**Reading the diagram:**
- Points **below** the diagonal → model is **overconfident** (predicts higher than reality)
- Points **above** the diagonal → model is **underconfident** (predicts lower than reality)
- S-shaped curve → common in neural networks (overconfident in middle, underconfident at extremes)

---

## 4.3 Calibration Metrics

### Expected Calibration Error (ECE)

The most common scalar calibration metric. Weighted average of the gap between confidence and accuracy across bins.

```
ECE = Σₘ (|Bₘ| / n) × |acc(Bₘ) - conf(Bₘ)|
```

Where:
- M = number of bins
- |Bₘ| = number of samples in bin m
- n = total samples
- acc(Bₘ) = fraction of positives in bin m
- conf(Bₘ) = mean predicted probability in bin m

**Interpretation:** ECE = 0.05 means the model's predictions are off by 5 percentage points on average.

### Maximum Calibration Error (MCE)

The worst-case bin gap:

```
MCE = max_m |acc(Bₘ) - conf(Bₘ)|
```

Use MCE when you care about high-stakes tails — a model might have low ECE on average but be badly miscalibrated in the 0.9–1.0 bin (very high confidence predictions), which matters most for critical decisions.

### Brier Score

A proper scoring rule that jointly measures both **calibration** and **sharpness** (resolution):

```
Brier Score = (1/n) × Σ (pᵢ - yᵢ)²
```

Where pᵢ is predicted probability and yᵢ ∈ {0, 1} is the true label. Lower is better (0 = perfect, 1 = worst).

**Brier Score Decomposition:**

```
Brier Score = Calibration + Resolution - Uncertainty
```

- **Calibration**: How far mean predictions deviate from mean outcomes per bin
- **Resolution**: How much predictions vary across bins (sharpness)
- **Uncertainty**: Inherent noise in the outcome (irreducible)

A model can improve Brier Score by being better calibrated OR by being sharper (more decisive). This decomposition tells you which lever to pull.

### Log-Loss (Cross-Entropy)

```
Log-Loss = -(1/n) × Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

Log-loss is also a proper scoring rule but penalizes confident wrong predictions exponentially. A model that says p=0.999 when y=0 is catastrophically penalized. This makes log-loss very sensitive to outlier miscalibration.

---

## 4.4 Why Models Are Miscalibrated

### Neural Networks: Overconfidence

Modern deep neural networks are systematically overconfident. Guo et al. (2017) showed that while neural networks have gotten more accurate over the years, their calibration has gotten *worse*.

**Why:**
- Large models with many parameters fit training data exactly → extreme logits
- Softmax of large logits → probabilities near 0 or 1
- L2 regularization and dropout help but don't fully fix this
- Models trained with cross-entropy loss are discriminative, not calibrated

### SVMs and Boosting: Overconfidence Near Decision Boundary

SVMs don't output probabilities naturally (they output margin scores). When you convert margins to probabilities naively, the result is poorly calibrated — often overconfident near the decision boundary.

Gradient boosting (XGBoost, LightGBM) also tends to be overconfident, especially on imbalanced datasets.

### Logistic Regression: Well Calibrated (Usually)

Logistic regression is naturally calibrated when the model is well-specified and trained on sufficient data. It's a common baseline and often used as the calibration layer on top of other models.

---

## 4.5 Platt Scaling

The simplest and most widely used post-hoc calibration method.

### The Idea

Train a **logistic regression** on top of the raw model scores, using a separate calibration set.

```
Step 1: Train your model on training data
Step 2: Generate raw scores on a held-out calibration set (never the test set)
Step 3: Fit logistic regression:
         p_calibrated = sigmoid(A × score + B)
         where A, B are learned parameters
Step 4: At inference time, pass raw scores through this sigmoid
```

The parameters A and B stretch and shift the sigmoid to align predictions with true frequencies.

### Why It Works

Platt scaling assumes the model's raw scores are **monotonically related to true probabilities** but miscalibrated in a systematic, sigmoidal way. It corrects:
- Overall bias (B shifts the curve left/right)
- Sharpness (A compresses/expands the curve)

### Limitations

- Only corrects **sigmoidal miscalibration** (symmetric S-shaped curves)
- If the miscalibration is complex and non-monotonic, Platt scaling fails
- Needs enough calibration data to fit two parameters reliably
- Works best when the model's decision function is monotone

### Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# Method 1: Using sklearn's built-in
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)
p_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# Method 2: Manual Platt scaling
raw_scores = base_model.decision_function(X_calib)  # or predict_proba
lr = LogisticRegression()
lr.fit(raw_scores.reshape(-1, 1), y_calib)
p_calibrated = lr.predict_proba(base_model.decision_function(X_test).reshape(-1, 1))[:, 1]
```

---

## 4.6 Isotonic Regression

A more flexible, non-parametric calibration method.

### The Idea

Fit a **monotonically non-decreasing step function** that maps raw scores to calibrated probabilities.

```
Raw scores:        [0.1, 0.4, 0.5, 0.6, 0.8, 0.9]
True labels:       [0,   0,   1,   0,   1,   1  ]

Isotonic regression finds a step function f such that:
  - f is non-decreasing
  - f minimizes Σ (f(scoreᵢ) - yᵢ)²

Result: f(0.1) = 0.0, f(0.4) = 0.33, f(0.6) = 0.33, f(0.8) = 1.0, f(0.9) = 1.0
```

The non-decreasing constraint is enforced by the **Pool Adjacent Violators (PAV) algorithm** — it merges neighboring bins that violate monotonicity by averaging them.

### Platt Scaling vs. Isotonic Regression

| Property | Platt Scaling | Isotonic Regression |
|---|---|---|
| Functional form | Parametric sigmoid | Non-parametric step function |
| Flexibility | Low (2 parameters) | High (can fit any shape) |
| Data needed | Small calibration set works | Needs larger calibration set |
| Overfitting risk | Low | Medium-high on small data |
| Best for | Smooth, sigmoidal miscalibration | Complex, non-monotonic miscalibration |
| Speed | Very fast | Fast |

**Rule of thumb:**
- < 1000 calibration samples → use Platt scaling
- ≥ 1000 calibration samples → try isotonic regression, validate on a separate held-out set

### Implementation

```python
from sklearn.isotonic import IsotonicRegression

# Fit isotonic regression on calibration set
iso = IsotonicRegression(out_of_bounds='clip')
raw_scores_calib = base_model.predict_proba(X_calib)[:, 1]
iso.fit(raw_scores_calib, y_calib)

# Apply at test time
raw_scores_test = base_model.predict_proba(X_test)[:, 1]
p_calibrated = iso.predict(raw_scores_test)
```

---

## 4.7 Temperature Scaling

The dominant calibration method for **neural networks**, introduced by Guo et al. (2017).

### The Idea

Before the final softmax, divide the logits by a single scalar **temperature T**:

```
p_calibrated = softmax(logits / T)
```

- T > 1 → softens the distribution (reduces overconfidence)
- T < 1 → sharpens the distribution (increases confidence)
- T = 1 → no change (original model)

T is fit by minimizing NLL on the calibration set. Only **one parameter** — simpler than Platt's two.

### Why Temperature Scaling Is Preferred for Neural Networks

- **Doesn't change predictions** — only probabilities. Accuracy is preserved.
- **Single global parameter** — can't overfit
- **Multiclass-native** — works naturally for softmax outputs (Platt and isotonic are binary)
- **Computationally trivial** — just divide logits by a scalar

```python
import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

# Optimize temperature on calibration set
optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
nll_criterion = nn.CrossEntropyLoss()

def eval_step():
    optimizer.zero_grad()
    logits = scaler.model(X_calib)
    loss = nll_criterion(logits / scaler.temperature, y_calib)
    loss.backward()
    return loss

optimizer.step(eval_step)
```

---

## 4.8 Calibration in Multiclass and Ranking Settings

### Multiclass Calibration

Calibration extends to multiclass via **confidence calibration**: the predicted class probability for the argmax class should match empirical accuracy at that confidence level.

Temperature scaling handles this naturally. Platt scaling requires one-vs-rest binary calibration.

### Calibration and Ranking Together

A common misconception: *if I'm using a model for ranking (not probability estimation), do I need calibration?*

**Answer: usually yes.** Ranking systems often combine scores from multiple models (e.g., relevance score × freshness score × user preference score). If these scores come from miscalibrated models on different scales, the combination is meaningless.

Calibration ensures scores are on a **comparable probability scale** before multiplication or combination.

---

## 4.9 The Calibration Workflow

```
1. Train model on training set
        │
2. Generate predictions on calibration set
   (held-out from both training and test)
        │
3. Plot reliability diagram
   Diagnose: overconfident? underconfident? complex shape?
        │
4. Choose calibration method:
   - Simple S-shape → Platt Scaling / Temperature Scaling
   - Complex shape, enough data → Isotonic Regression
        │
5. Fit calibrator on calibration set
        │
6. Evaluate calibration improvement:
   - Reliability diagram (after)
   - ECE before vs. after
   - Brier score before vs. after
   - Confirm AUC/accuracy unchanged
        │
7. Apply calibrator at inference time
```

> **Never fit your calibrator on the test set.** Calibration fitting is a form of training. Use a separate calibration split or a held-out fold from cross-validation.

---

## 4.10 Common Mistakes

| Mistake | Why It's Wrong |
|---|---|
| Calibrating on the test set | Data leakage — inflated calibration metrics |
| Assuming high AUC = good calibration | These are independent; a perfectly ranked model can be completely miscalibrated |
| Using ECE with too few bins | Noisy estimates; use adaptive binning or reliability diagram visually |
| Ignoring calibration in ensembles | Ensemble components on different scales → garbage combination |
| Not re-calibrating after retraining | Calibration drifts as models are updated; re-calibrate periodically |
| Calibrating on imbalanced data without care | ECE can be misleading when one class dominates |

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Calibration | Predicted probability should match empirical frequency |
| Reliability diagram | Primary diagnostic: plot confidence vs. fraction of positives |
| ECE | Scalar calibration error; weighted average gap across bins |
| Brier Score | Joint measure of calibration + sharpness |
| Platt Scaling | Sigmoid fit on raw scores; 2 params; good for small calibration sets |
| Isotonic Regression | Non-parametric step function; more flexible; needs more data |
| Temperature Scaling | Divide logits by T; best for neural networks; preserves accuracy |
| AUC ≠ Calibration | Discriminative power and probabilistic accuracy are independent |

---

## Further Reading

- Guo et al. — *On Calibration of Modern Neural Networks* (ICML 2017) — the landmark paper
- Platt, J. — *Probabilistic Outputs for SVMs* (1999) — original Platt scaling
- Niculescu-Mizil & Caruana — *Predicting Good Probabilities with Supervised Learning* (ICML 2005)
- Kull et al. — *Beta Calibration* (AISTATS 2017) — generalization of Platt scaling

---

*Next: Chapter 5 — Business Metric Alignment*
