# Chapter 4: Calibration — Platt Scaling, Isotonic Regression, and Beyond
### Apple MLE Interview Master Notes — Improved & Expanded Edition

> *"A model that says 'I'm 90% confident' should be right 90% of the time. Not 60%. Not 99%. 90%. That's calibration — and most models fail it badly out of the box."*

---

## 4.0 Master Cheat Sheet

### 4.0.1 Calibration Methods at a Glance

| Method | Params | Best For | Key Constraint |
|---|---|---|---|
| Platt Scaling | 2 (A, B) | SVMs, boosting; small calibration sets; S-shaped miscalibration | Assumes sigmoidal correction is sufficient |
| Isotonic Regression | Non-parametric | Complex, non-monotonic miscalibration; ≥ 1,000 calibration samples | Risk of overfitting on small data |
| Temperature Scaling | 1 (T) | Neural networks; multiclass softmax outputs | Only scales logits — cannot fix complex shape |
| Beta Calibration | 3 | Skewed output distributions | More flexible than Platt; less common |

### 4.0.2 Key Facts to Keep at the Front of Your Mind

| # | Fact | Detail |
|---|---|---|
| 1 | Calibration ≠ AUC | A model can rank examples perfectly (high AUC) while being completely miscalibrated — these are independent properties |
| 2 | ECE formula | ECE = Σₘ (|Bₘ| / n) × \|acc(Bₘ) − conf(Bₘ)\| |
| 3 | Temperature T > 1 | Softens predictions → reduces overconfidence |
| 4 | Temperature T < 1 | Sharpens predictions → increases confidence |
| 5 | Platt scaling data threshold | < 1,000 calibration samples → use Platt; ≥ 1,000 → try isotonic |
| 6 | Temperature scaling preserves accuracy | Only divides logits by a scalar; the argmax class never changes |
| 7 | Logistic regression is well calibrated | Naturally calibrated when well-specified; often used as the calibration layer |
| 8 | Neural nets are systematically overconfident | Guo et al. (2017): accuracy improved over years; calibration got worse |
| 9 | Never calibrate on the test set | Calibration fitting is training — use a separate held-out calibration split |
| 10 | Brier score = calibration + resolution − uncertainty | Decomposition tells you which lever to pull to improve it |

---

## 4.1 What Is Calibration?

### 4.1.1 The Core Idea

A model outputs a probability score. But what does that number actually mean? **Calibration** is the alignment between a model's predicted probabilities and the true frequencies of outcomes in the real world.

**Plain-English analogy:** Imagine a weather forecaster who says "70% chance of rain" every day for a month. If it rained on exactly 70% of those days, that forecaster is perfectly calibrated. If it only rained on 40% of those days, they're overconfident. If it rained 90% of the time, they're underconfident.

```
A well-calibrated model:
  Predicts 0.9 → event happens ~90% of the time   ✅
  Predicts 0.7 → event happens ~70% of the time   ✅
  Predicts 0.3 → event happens ~30% of the time   ✅

A poorly calibrated model (overconfident):
  Predicts 0.9 → event happens ~60% of the time   ❌
  Predicts 0.3 → event happens ~45% of the time   ❌
```

### 4.1.2 Why Calibration Matters

Calibration matters whenever the **magnitude** of a probability is used — not just its rank order. If you only care about which example scores higher (e.g., sorting a recommendation feed), calibration is less critical. If you care about what the number actually means (e.g., "should we perform a biopsy?"), calibration is essential.

| Use Case | Why Calibration Matters |
|---|---|
| Medical diagnosis | "80% chance of cancer" drives biopsy decisions — off-by-30% is dangerous |
| Credit risk | Loan approval thresholds are set by expected default probability |
| Weather forecasting | "30% chance of rain" determines behavior — people calibrate actions to this |
| Ad bidding | Expected click probability × bid price = actual dollar bid; wrong probability = money lost |
| Model ensembling | Combining scores from multiple models requires comparable probability scales |
| Uncertainty-aware systems | Downstream systems that act on confidence need it to be trustworthy |

### 4.1.3 The Critical Distinction: Calibration vs. Discrimination

> **A model can rank examples perfectly (high AUC) while being terribly calibrated. These are independent properties.**

- **AUC / ranking** — does the model correctly order examples by risk? (relative)
- **Calibration** — are the predicted probability values themselves accurate? (absolute)

A fraud detector that correctly identifies the top 1% riskiest transactions (high AUC) but assigns them all 0.55 probability instead of 0.95 has excellent discrimination and terrible calibration. For ranking, it works fine. For setting a dollar-value fraud reserve, the miscalibrated probabilities produce the wrong reserve estimate.

---

## 4.2 Diagnosing Calibration: The Reliability Diagram

### 4.2.1 What It Is

The **reliability diagram** (also called a calibration curve) is the primary visual diagnostic tool for calibration. It plots what the model predicted against what actually happened.

### 4.2.2 How to Build One

1. Collect model predictions and true labels on a held-out set
2. Bin predictions into K equal-width buckets (e.g., [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0])
3. For each bucket, compute:
   - **x-axis:** Mean predicted probability in that bucket
   - **y-axis:** Fraction of samples in that bucket that are actually positive
4. Plot these points; draw the diagonal (y = x) as the reference line for perfect calibration

### 4.2.3 How to Read It

| Pattern | What It Means | Common Cause |
|---|---|---|
| Points fall on the diagonal | Perfect calibration | — |
| Points below the diagonal | Model is overconfident (predicts too high) | Neural networks, boosted trees |
| Points above the diagonal | Model is underconfident (predicts too low) | Naive Bayes, some SVMs |
| S-shaped curve | Overconfident in the middle, underconfident at extremes | Neural networks trained with cross-entropy |
| Sigmoid-shaped | Common in SVMs / margin-based models | Needs Platt scaling |

```
     1.0 |                             *
         |                          * /
Fraction |                       /
of       |                    * /
Positives|                 /
     0.5 |              * /
         |           /
         |        * /
         |     /  ← Model is overconfident
         |  * /      (predicts high; actual outcome is lower)
     0.0 +--/-------------------------------
         0.0        0.5                  1.0
                Predicted Probability

Diagonal (/) = perfect calibration
Points (*) below the line = overconfident model
```

> **Apple Production Tip:** Always plot reliability diagrams for any model that outputs probabilities before shipping. A model that looks great on AUC in offline evaluation but is overconfident can produce systematically wrong business decisions (mispriced risk, over-triggered alerts, misleading confidence UIs).

---

## 4.3 Calibration Metrics

### 4.3.1 Metric 1: Expected Calibration Error (ECE)

The most common single-number calibration metric. It measures the weighted average gap between predicted confidence and actual accuracy across all bins.

```
ECE = Σₘ (|Bₘ| / n) × |acc(Bₘ) − conf(Bₘ)|
```

Where:
- M = number of bins
- |Bₘ| = number of samples in bin m (the weight — larger bins matter more)
- n = total number of samples
- acc(Bₘ) = fraction of positives in bin m (actual outcome rate)
- conf(Bₘ) = mean predicted probability in bin m

**Interpretation:** ECE = 0.05 means predicted probabilities are off by 5 percentage points on average. ECE = 0 is perfect calibration.

**Weakness:** ECE depends on bin choice. Too few bins → noisy estimates. Use adaptive binning (equal-sample bins rather than equal-width bins) for more reliable estimates, especially with skewed score distributions.

---

### 4.3.2 Metric 2: Maximum Calibration Error (MCE)

The **worst-case** bin gap — how badly miscalibrated is the most problematic confidence range?

```
MCE = max_m |acc(Bₘ) − conf(Bₘ)|
```

**When to use MCE over ECE:** When you care about high-stakes tails. A model can have a low ECE on average while being badly miscalibrated in the 0.9–1.0 bin (very high confidence predictions). For a medical or financial system where the highest-confidence predictions drive the most consequential decisions, MCE is the relevant metric — the average doesn't tell you about the worst case.

---

### 4.3.3 Metric 3: Brier Score

A **proper scoring rule** that jointly measures both calibration and sharpness (how decisive the model is).

```
Brier Score = (1/n) × Σ (pᵢ − yᵢ)²
```

Where pᵢ is the predicted probability and yᵢ ∈ {0, 1} is the true label. Lower is better (0 = perfect, 1 = worst possible).

**Plain-English analogy:** Brier score is the mean squared error of your probability predictions. A model that says p = 0.9 when y = 1 incurs a loss of (0.9 − 1)² = 0.01. A model that says p = 0.1 when y = 1 incurs (0.1 − 1)² = 0.81 — a much larger penalty.

#### Brier Score Decomposition

```
Brier Score = Calibration + Resolution − Uncertainty
```

| Component | What It Measures | Interpretation |
|---|---|---|
| Calibration | How far mean predictions deviate from mean outcomes per bin | Lower = better-calibrated |
| Resolution | How much predictions vary across bins (sharpness) | Higher = more decisive model |
| Uncertainty | Inherent noise in the outcome (irreducible) | Fixed by the problem — can't change it |

This decomposition is powerful: if your Brier score is high, it tells you *why*. Is the model poorly calibrated, or is it not sharp enough (predicting near 0.5 for everything)? Each problem has a different fix.

---

### 4.3.4 Metric 4: Log-Loss (Cross-Entropy)

```
Log-Loss = −(1/n) × Σ [yᵢ log(pᵢ) + (1 − yᵢ) log(1 − pᵢ)]
```

Log-loss is also a proper scoring rule but penalizes **confident wrong predictions exponentially**. A model that says p = 0.999 when y = 0 is catastrophically penalized (log-loss → ∞ in the limit). This makes log-loss very sensitive to outlier miscalibration and to predictions near 0 or 1.

### 4.3.5 Calibration Metric Summary

| Metric | Measures | Range | Better = | Key Weakness |
|---|---|---|---|---|
| ECE | Average calibration gap | [0, 1] | Lower | Sensitive to bin choice |
| MCE | Worst-case calibration gap | [0, 1] | Lower | Noisy on small datasets |
| Brier Score | Calibration + sharpness jointly | [0, 1] | Lower | Conflates two properties |
| Log-Loss | Calibration, with exponential penalty | [0, ∞) | Lower | Explodes on confident wrong predictions |

---

## 4.4 Why Models Are Miscalibrated

### 4.4.1 Neural Networks: Systematic Overconfidence

Modern deep neural networks are systematically overconfident. Guo et al. (2017) showed that while neural networks have become more accurate over the years, their calibration has actually gotten *worse* — larger, more accurate models tend to be *more* miscalibrated.

**Why this happens — step by step:**

1. Large models with many parameters can fit the training data exactly
2. This drives logits (the pre-softmax values) to become very large in magnitude
3. Softmax of large logits produces probabilities near 0 or 1 (near-certain predictions)
4. Near-certain predictions are rarely correct 100% of the time → overconfidence
5. L2 regularization and dropout help but do not fully fix this

**Root cause:** Cross-entropy loss is a **discriminative** objective — it only cares about getting the ranking right, not about the calibration of the probability values.

### 4.4.2 SVMs and Gradient Boosting: Overconfidence Near Boundaries

SVMs don't output probabilities natively — they output margin scores (how far a point is from the decision boundary). Naively converting margins to probabilities produces poorly calibrated outputs, often overconfident near the boundary.

Gradient boosting (XGBoost, LightGBM) also tends to be overconfident, especially on imbalanced datasets, because the tree-based outputs cluster near the boundaries of each leaf's probability range.

### 4.4.3 Logistic Regression: Naturally Well Calibrated

Logistic regression is naturally well calibrated when the model is correctly specified (the true log-odds are linear in the features) and trained on sufficient data. This is why logistic regression is often used as the **calibration layer** on top of other models. It's not always the best classifier, but it's a trustworthy probability estimator.

### 4.4.4 Miscalibration Source Summary

| Model Type | Typical Miscalibration | Direction |
|---|---|---|
| Deep neural networks | Extreme logits from overparameterization | Overconfident |
| SVMs | Margin scores ≠ probabilities | Overconfident near boundary |
| Gradient boosting (XGBoost, LightGBM) | Leaf probability clustering | Overconfident, especially with imbalance |
| Naive Bayes | Feature independence assumption violated | Underconfident (probabilities too close to 0.5) |
| Logistic regression | Usually well calibrated | Minimal bias when well-specified |
| Random forests | Probability averaging over trees | Slightly underconfident (biased toward 0.5) |

---

## 4.5 Platt Scaling

### 4.5.1 The Core Idea

Platt scaling is the simplest and most widely used post-hoc calibration method. The key insight: after training, take the model's raw output scores and train a **logistic regression** on top of them using a separate held-out calibration set.

**Why logistic regression?** The model's raw scores are assumed to be monotonically related to true probabilities but miscalibrated in a sigmoidal way. Logistic regression learns the correct sigmoid shape to map scores to probabilities.

### 4.5.2 The Four Steps

```
Step 1: Train your base model on training data (do not touch the calibration set)
Step 2: Generate raw scores on a held-out calibration set
Step 3: Fit a logistic regression: p_calibrated = sigmoid(A × score + B)
        where A and B are learned to minimize log-loss on the calibration set
Step 4: At inference time, pass raw scores through this learned sigmoid
```

The parameter **A** controls sharpness (compresses or expands the curve). The parameter **B** controls bias (shifts the curve left or right to correct overall overconfidence or underconfidence).

### 4.5.3 Why It Works

Platt scaling corrects two systematic errors:
- **Bias correction (B):** Shifts the entire probability distribution up or down (e.g., if the model consistently predicts 0.8 when the truth is 0.6)
- **Sharpness correction (A):** Compresses or expands the confidence spread (e.g., if the model uses the range [0.3, 0.9] but should use [0.1, 0.99])

### 4.5.4 Limitations

1. Only corrects **sigmoidal miscalibration** — it assumes the miscalibration has a simple S-shape
2. If miscalibration is complex and non-monotonic (e.g., the model is overconfident in the middle but underconfident at the extremes in a non-sigmoidal way), Platt scaling can't fix it
3. Requires enough calibration data to reliably estimate two parameters — generally at least a few hundred samples
4. Assumes the model's score ordering is monotone (the highest-scoring examples are truly the most likely positives)

### 4.5.5 Implementation

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Method 1: sklearn built-in (recommended)
# cv='prefit' means the base_model is already trained; we're just calibrating
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)
p_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# Method 2: Manual (useful when base model outputs raw scores, not predict_proba)
raw_scores_calib = base_model.decision_function(X_calib)  # SVM margin scores
lr = LogisticRegression()
lr.fit(raw_scores_calib.reshape(-1, 1), y_calib)
raw_scores_test = base_model.decision_function(X_test)
p_calibrated = lr.predict_proba(raw_scores_test.reshape(-1, 1))[:, 1]
```

---

## 4.6 Isotonic Regression

### 4.6.1 The Core Idea

Isotonic regression is a more flexible, **non-parametric** calibration method. Instead of assuming a sigmoid shape, it fits any monotonically non-decreasing step function that maps raw scores to calibrated probabilities.

**Plain-English analogy:** Platt scaling assumes the correction is shaped like an S-curve and fits parameters to that curve. Isotonic regression makes no assumption about the shape — it finds the best-fitting staircase function, with the only rule being that the steps can only go up, never down (monotone constraint).

### 4.6.2 How It Works (PAV Algorithm)

The non-decreasing constraint is enforced by the **Pool Adjacent Violators (PAV) algorithm**:

```
Raw scores (sorted): [0.1, 0.4, 0.5, 0.6, 0.8, 0.9]
True labels:         [ 0,   0,   1,   0,   1,   1  ]

Problem: score 0.6 has label 0 but score 0.5 has label 1
         → This violates monotonicity if we map scores directly to labels

PAV resolution: merge adjacent violating bins and average their labels
  Bin {0.5, 0.6}: labels {1, 0} → average = 0.5 → both get calibrated prob = 0.5

Final mapping:
  f(0.1) = 0.0   f(0.4) = 0.0   f(0.5) = 0.5
  f(0.6) = 0.5   f(0.8) = 1.0   f(0.9) = 1.0
```

The result is a step function that is guaranteed to be monotone and to minimize the sum of squared errors between the calibrated probabilities and the true labels.

### 4.6.3 Platt Scaling vs. Isotonic Regression — Head-to-Head

| Property | Platt Scaling | Isotonic Regression |
|---|---|---|
| Functional form | Parametric sigmoid | Non-parametric step function |
| Flexibility | Low (2 parameters) | High (can fit any monotone shape) |
| Calibration data needed | Small sets work (hundreds) | Needs ≥ 1,000 samples |
| Overfitting risk | Low | Medium-high on small data |
| Best for | Smooth, sigmoidal miscalibration | Complex, non-monotonic miscalibration |
| Speed | Very fast | Fast |
| Handles extrapolation? | Yes (sigmoid is bounded) | Requires `out_of_bounds='clip'` |

**Rule of thumb:**
- Fewer than 1,000 calibration samples → use Platt scaling
- 1,000 or more calibration samples → try isotonic regression; validate on a separate held-out set before committing

### 4.6.4 Implementation

```python
from sklearn.isotonic import IsotonicRegression

# Fit on calibration set
iso = IsotonicRegression(out_of_bounds='clip')  # clip handles test scores outside training range
raw_scores_calib = base_model.predict_proba(X_calib)[:, 1]
iso.fit(raw_scores_calib, y_calib)

# Apply to test set
raw_scores_test = base_model.predict_proba(X_test)[:, 1]
p_calibrated = iso.predict(raw_scores_test)
```

> **Apple Production Tip:** `out_of_bounds='clip'` is critical. Isotonic regression is defined only over the range of calibration scores. At inference time, a test sample may produce a score outside that range (especially after model updates or distribution shift). Without clipping, predictions for out-of-range inputs are undefined.

---

## 4.7 Temperature Scaling

### 4.7.1 The Core Idea

Temperature scaling is the **dominant calibration method for neural networks**, introduced by Guo et al. (2017). It is the simplest possible modification: before applying the final softmax, divide the logits by a single scalar called the **temperature T**.

```
Original:            p = softmax(logits)
Temperature-scaled:  p_calibrated = softmax(logits / T)
```

T is fit by minimizing negative log-likelihood (NLL) on the calibration set. It has only **one parameter** — even simpler than Platt scaling's two.

### 4.7.2 What T Controls

| Value of T | Effect on Probabilities | When to Use |
|---|---|---|
| T > 1 | Softens distribution (spreads probabilities closer to uniform) | Model is overconfident (most neural nets) |
| T < 1 | Sharpens distribution (pushes probabilities toward 0 and 1) | Model is underconfident |
| T = 1 | No change — original model output | Model is already well calibrated |

**Plain-English analogy:** Temperature comes from physics (thermodynamics / statistical mechanics). High temperature makes a physical system's particles spread out into more states uniformly. Low temperature makes them cluster into the lowest-energy state. In ML, high T spreads probability mass away from the most confident class; low T concentrates it.

### 4.7.3 Why Temperature Scaling Is Preferred for Neural Networks

1. **Preserves accuracy** — dividing all logits by the same scalar T does not change their relative ordering. The argmax class (the prediction) is identical. AUC is unchanged.
2. **Single parameter** — cannot overfit the calibration set; works on very small calibration sets.
3. **Multiclass-native** — naturally extends to K-class softmax without any modification (Platt and isotonic require one-vs-rest treatment for multiclass).
4. **Computationally trivial** — one division at inference time; zero latency overhead.

### 4.7.4 Implementation

```python
import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))  # start at T=1 (no scaling)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

# Fit temperature on calibration set using L-BFGS optimizer
scaler = TemperatureScaler(base_model)
optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
nll_criterion = nn.CrossEntropyLoss()

def eval_step():
    optimizer.zero_grad()
    logits = scaler.model(X_calib)
    loss = nll_criterion(logits / scaler.temperature, y_calib)
    loss.backward()
    return loss

optimizer.step(eval_step)
print(f"Optimal temperature: {scaler.temperature.item():.3f}")
# T > 1: model was overconfident and was softened
# T < 1: model was underconfident and was sharpened
```

---

## 4.8 Calibration in Multiclass and Ranking Settings

### 4.8.1 Multiclass Calibration

Calibration extends to multiclass through **confidence calibration**: the predicted probability for the winning (argmax) class should match the empirical accuracy at that confidence level.

- **Temperature scaling** handles this directly and naturally — dividing all K logits by T simultaneously adjusts the whole softmax distribution.
- **Platt scaling** requires a separate one-vs-rest binary calibration for each of the K classes, which is more cumbersome and doesn't account for dependencies between class probabilities.

### 4.8.2 Calibration in Ranking Systems

A common misconception: *"If I'm using a model purely for ranking (not probability estimation), I don't need calibration."*

**This is usually wrong in production.** Here's why:

Ranking systems often combine scores from multiple models — for example: `final_score = relevance_score × freshness_score × personalization_score`. If each sub-model is miscalibrated on a different scale, their product is meaningless. A relevance model outputting [0.85, 0.87, 0.90] and a freshness model outputting [0.1, 0.5, 0.9] cannot be combined until both are on a comparable probability scale.

Calibration ensures scores are **semantically comparable** before multiplication, weighted summing, or any other combination.

---

## 4.9 The Complete Calibration Workflow

```
Step 1: Train model on training set
        (Never use calibration or test data here)
        │
Step 2: Generate predictions on the calibration set
        (Held-out from both training and test — a separate split)
        │
Step 3: Plot reliability diagram
        → Diagnose: overconfident? underconfident? S-shaped? complex?
        │
Step 4: Choose calibration method
        ├── Simple S-shape, small calibration set  → Platt Scaling
        ├── Complex shape, ≥ 1,000 samples         → Isotonic Regression
        └── Neural network with softmax output     → Temperature Scaling
        │
Step 5: Fit calibrator on calibration set
        (Only ever on calibration set — never test)
        │
Step 6: Evaluate improvement
        → Reliability diagram (after calibration)
        → ECE before vs. after
        → Brier score before vs. after
        → Confirm accuracy and AUC are unchanged
        │
Step 7: Apply calibrator at inference time
        (Single additional step after the base model forward pass)
```

> **Critical rule:** Never fit your calibrator on the test set. Calibration fitting is a form of training. Doing it on the test set is data leakage — your reported calibration metrics will be inflated and your model will be miscalibrated in real deployment.

---

## 4.10 Common Mistakes and How to Avoid Them

| # | Mistake | Why It's Wrong | How to Fix It |
|---|---|---|---|
| 1 | Calibrating on the test set | Data leakage — reported ECE will be falsely low; model is miscalibrated in production | Always use a separate calibration split, or held-out fold from cross-validation |
| 2 | Assuming high AUC = good calibration | AUC and calibration are independent; a perfectly ranked model can be completely wrong on probability values | Always plot a reliability diagram in addition to computing AUC |
| 3 | Using ECE with too few bins | Noisy, unreliable estimates — bin gaps are driven by sampling variance | Use adaptive (equal-sample) binning, or increase dataset size |
| 4 | Ignoring calibration in ensemble systems | Components on different probability scales produce a meaningless weighted combination | Calibrate each component independently before combining |
| 5 | Not re-calibrating after model retraining | Calibration drifts as models are updated; T or A,B learned on the old model may not apply | Re-run calibration fitting every time the base model is retrained |
| 6 | Calibrating with imbalanced data without care | ECE is dominated by the majority class; minority-class calibration can be completely wrong | Use stratified calibration splits; evaluate calibration separately per class |

---

## 4.11 Interview Q&A Bank

### Q1: Explain the difference between calibration and discrimination (AUC). Can a model have perfect AUC and terrible calibration simultaneously? Give a concrete example.

**Why interviewers ask this:** This is one of the most common ML evaluation misconceptions. At Apple's scale, deploying a well-discriminating but miscalibrated model can cause systematic downstream failures. This tests whether you truly understand what AUC measures vs. what calibration measures.

**Answer:**

**Discrimination (AUC)** measures whether the model correctly *ranks* examples — specifically, the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative example. AUC = 1.0 means perfect ranking. AUC does not care about the actual values of the probabilities.

**Calibration** measures whether the probability *values themselves are accurate* — i.e., whether a predicted probability of 0.8 corresponds to an event that happens 80% of the time in reality.

**Can a model have perfect AUC and terrible calibration? Yes — here's a concrete example:**

| Sample | True label | Raw probability | Monotone-transformed probability |
|---|---|---|---|
| A | 1 | 0.95 | 0.51 |
| B | 1 | 0.90 | 0.52 |
| C | 0 | 0.20 | 0.49 |
| D | 0 | 0.10 | 0.48 |

Apply any monotone transformation to the probabilities — for example, squish everything into [0.48, 0.52]. The ranking is perfectly preserved (A > B > C > D), so AUC = 1.0. But the calibration is catastrophic: the model says every sample has ~50% probability regardless of whether it's actually positive or negative.

**Practical consequence:** If this model is used to set a loan approval threshold at p > 0.6, it would approve or reject all loans identically (since everything is between 0.48 and 0.52) even though the underlying ranking is perfect.

**When each matters:**

| Situation | What you need |
|---|---|
| Ranking a recommendation feed | AUC / ranking metrics |
| Setting a fraud detection dollar threshold | Calibration |
| Combining scores from multiple models | Both — calibration ensures comparable scales |
| Triaging medical tests by priority | AUC primarily |
| Quoting insurance premiums by risk level | Calibration — wrong probabilities = wrong premiums |

---

### Q2: Derive or explain why Temperature Scaling preserves accuracy (AUC and argmax prediction) while changing calibration. What are its limitations?

**Why interviewers ask this:** This tests whether you understand what temperature scaling actually does mathematically vs. empirically — important for explaining design decisions in production ML systems.

**Answer:**

**Why accuracy is preserved:**

Temperature scaling divides all K logits by the same scalar T before applying softmax:

```
Original logits:      z = [z₁, z₂, ..., zₖ]
Scaled logits:        z/T = [z₁/T, z₂/T, ..., zₖ/T]

The argmax class:
  argmax(z) = argmax(z/T)  for any T > 0
```

Multiplying or dividing all logits by the same positive constant does not change their relative ordering — it's equivalent to scaling the x-axis uniformly. The class with the highest logit stays highest. Therefore:
- The predicted class label never changes → accuracy unchanged
- The ranking of classes never changes → AUC unchanged

**Why calibration changes:**

Softmax is a nonlinear function of logits. Dividing logits by T > 1 makes them smaller in magnitude, which causes softmax to produce a **flatter, more uniform distribution** (less extreme probabilities). For example:

```
Original logits: [3.0, 1.0, 0.5]
softmax → [0.87, 0.12, 0.07]   ← very confident on class 0

After T = 2.0:  [1.5, 0.5, 0.25]
softmax → [0.65, 0.24, 0.17]   ← less confident; better calibrated if true P(y=0) ≈ 0.65
```

**Limitations of Temperature Scaling:**

| Limitation | Detail |
|---|---|
| Global scalar only | Uses a single T for all inputs; cannot fix miscalibration that varies across different types of inputs or confidence levels |
| Cannot fix complex shapes | If miscalibration is non-monotone (the model is overconfident in some ranges and underconfident in others), a single T cannot correct both simultaneously |
| Assumes uniform miscalibration | Works best when the entire distribution is shifted in one direction — doesn't handle asymmetric or multimodal miscalibration |
| Does not fix label distribution shift | If the calibration set has different class proportions than deployment, T will be fit to the wrong prior |

---

### Q3: You have a fraud detection model deployed at Apple Pay. The model achieves 0.97 AUC and 0.04 ECE. A product manager says the model is production-ready. What questions would you ask, and what additional validation would you perform before agreeing?

**Why interviewers ask this:** This is a production ML judgment question. Strong MLE candidates know that headline metrics can hide critical failure modes, especially for high-stakes, imbalanced problems. Apple Pay involves real financial harm — this tests whether you can think beyond benchmark numbers.

**Answer:**

**Questions to ask before agreeing:**

1. **What is the class imbalance ratio?** If fraud is 0.1% of transactions, a model that always predicts "not fraud" has 99.9% accuracy and potentially good ECE on the majority class. ECE can be dominated by the majority class and completely hide minority-class miscalibration.

2. **How was the 0.04 ECE computed?** Equal-width bins? If most fraud scores cluster near 0 and 1, the intermediate bins may have few samples and high variance. Were bins adaptive (equal-sample)? Was the calibration set class-balanced?

3. **What does the reliability diagram look like?** ECE = 0.04 on average could hide MCE = 0.30 in the high-confidence range (0.9–1.0) — exactly the range that drives the largest fraud decisions (chargebacks, account blocks).

4. **Has calibration been evaluated per customer segment?** A globally well-calibrated model may be miscalibrated for specific demographics, device types, or transaction types. At Apple's scale, a 5% miscalibration for one user segment is millions of incorrect probability assessments.

5. **Was the calibration set temporally held out?** If calibration was done on randomly shuffled data, temporal leakage may be inflating metrics. Fraud patterns shift over time; a time-based split better simulates production.

**Additional validation steps:**

| Validation | What It Checks |
|---|---|
| Per-class calibration curve | Calibration for fraud class specifically, not just overall |
| MCE in high-confidence bins (0.9–1.0) | Worst-case calibration where decisions are most consequential |
| Calibration by transaction amount | Is the model more miscalibrated for high-value transactions? |
| Calibration by user segment | Check device type, geography, account age |
| Temporal calibration stability | Does calibration degrade over a 30/60/90-day out-of-time window? |
| False positive cost analysis | At the operating threshold, what dollar loss does miscalibration produce? |
| Brier score decomposition | Is the problem calibration, resolution, or irreducible uncertainty? |

**Key point to convey:** A fraud model is not a research benchmark — it's a financial system. "Production-ready" requires not just accurate metrics but appropriate metrics, evaluated on appropriate data splits, with an explicit analysis of failure modes that carry financial or user harm.

---

### Q4: When would you prefer Isotonic Regression over Temperature Scaling for calibrating a neural network? What risks do you take on?

**Why interviewers ask this:** This tests whether you understand when "more flexible" is actually better, and when it causes problems — a judgment call that matters when choosing calibration approaches in production.

**Answer:**

**Prefer Isotonic Regression over Temperature Scaling when:**

1. **The miscalibration is non-monotone or complex in shape.** Temperature scaling is a single global scalar — it can only uniformly stretch or compress the probability distribution. If the neural network is overconfident in the middle range (0.4–0.6) but underconfident at the extremes (near 0 and 1), a single T cannot fix both simultaneously. Isotonic regression fits a step function and can correct arbitrary monotone miscalibration patterns.

2. **You have a large calibration set (≥ 1,000 samples, ideally 5,000+).** The non-parametric step function that isotonic regression fits has many effective degrees of freedom. With small data, many bins will have very few samples, and the fitted step function will be noisy and will overfit to calibration-set randomness.

3. **The model outputs do not come from a softmax (binary or non-neural).** Temperature scaling is specifically designed for neural networks with logit → softmax pipelines. For a gradient-boosted tree outputting probabilities in [0, 1], there are no logits to scale. Isotonic regression works on any probability output.

**Risks you take on with Isotonic Regression:**

| Risk | Detail | Mitigation |
|---|---|---|
| Overfitting to calibration set | Step function memorizes noise in small samples | Use cross-validated calibration; validate ECE on a separate held-out fold |
| Extrapolation failures | Isotonic regression is undefined outside the training score range | Always use `out_of_bounds='clip'`; monitor for out-of-range scores in production |
| Calibration drift | The step function is fit to a static distribution; it may become wrong after model retraining or distribution shift | Re-fit after every model update; monitor ECE in production |
| Non-smooth outputs | Step function produces discontinuous probability estimates | Smooth with kernel density estimation post-hoc if downstream systems expect smooth probabilities |

**Bottom line for the interview:** Temperature scaling is the default starting point for neural networks — it's simple, fast, and can't overfit. Reach for isotonic regression when you've diagnosed a non-sigmoidal miscalibration pattern and you have enough calibration data to support it.

---

## 4.12 Rapid-Fire Flashcards

| # | Prompt | Answer |
|---|---|---|
| 1 | Calibration definition? | Predicted probabilities match empirical outcome frequencies |
| 2 | Reliability diagram axes? | x: mean predicted probability per bin; y: fraction of positives per bin |
| 3 | Points below the diagonal mean? | Model is overconfident |
| 4 | Points above the diagonal mean? | Model is underconfident |
| 5 | ECE formula? | Σₘ (\|Bₘ\| / n) × \|acc(Bₘ) − conf(Bₘ)\| |
| 6 | MCE vs. ECE — when to prefer MCE? | When high-confidence (tail) miscalibration is most consequential |
| 7 | Brier score decomposition? | Calibration + Resolution − Uncertainty |
| 8 | Platt scaling: how many parameters? | 2 (A and B in the sigmoid) |
| 9 | Isotonic regression: data threshold? | < 1,000 → use Platt; ≥ 1,000 → try isotonic |
| 10 | PAV algorithm does what? | Merges adjacent violating bins to enforce monotonicity in isotonic regression |
| 11 | Temperature T > 1 → ? | Softer probabilities (reduces overconfidence) |
| 12 | Temperature T < 1 → ? | Sharper probabilities (increases confidence) |
| 13 | Does temperature scaling change argmax? | No — only divides logits by a scalar; ranking is unchanged |
| 14 | Why are neural nets overconfident? | Large models drive logits to extreme magnitudes → softmax outputs near 0 or 1 |
| 15 | Which model is naturally well calibrated? | Logistic regression (when well-specified) |
| 16 | Never calibrate on... | The test set (data leakage) |
| 17 | High AUC implies good calibration? | No — AUC and calibration are independent |
| 18 | Temperature scaling is preferred for neural nets because? | One parameter, can't overfit, multiclass-native, preserves accuracy |

---

## 4.13 Summary Table

| Concept | One-line takeaway |
|---|---|
| Calibration | Predicted probability should match empirical outcome frequency |
| Reliability diagram | Primary diagnostic: plot confidence vs. fraction of positives per bin |
| ECE | Weighted average calibration gap across bins; lower is better |
| MCE | Worst-case bin gap; use when tail miscalibration is most consequential |
| Brier Score | Joint measure of calibration + sharpness; decomposes into three components |
| Log-Loss | Proper scoring rule with exponential penalty for confident wrong predictions |
| Platt Scaling | Two-parameter sigmoid fit; good for small calibration sets and S-shaped errors |
| Isotonic Regression | Non-parametric step function; more flexible; needs ≥ 1,000 calibration samples |
| Temperature Scaling | Divide logits by T; best for neural networks; preserves accuracy and AUC |
| AUC ≠ Calibration | Discriminative ranking power and probabilistic accuracy are fully independent |

---

## 4.14 Further Reading

1. Guo et al. — *On Calibration of Modern Neural Networks* (ICML 2017) — the landmark paper establishing neural net overconfidence
2. Platt, J. — *Probabilistic Outputs for Support Vector Machines* (1999) — original Platt scaling derivation
3. Niculescu-Mizil & Caruana — *Predicting Good Probabilities with Supervised Learning* (ICML 2005) — comprehensive empirical comparison of calibration methods
4. Kull et al. — *Beta Calibration* (AISTATS 2017) — generalization of Platt scaling for skewed outputs

---

> **Next:** Chapter 5 — Business Metric Alignment
