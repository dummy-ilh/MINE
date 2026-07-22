Logistic regression is natively designed to output well-calibrated probabilities because it directly minimizes cross-entropy loss (log-loss). However, in real-world engineering, **logistic regression models miscalibrate frequently**.

Here is a comprehensive deep dive into **Model Calibration**, covering why it breaks, how to diagnose it, and step-by-step methods to fix it.

---

# Model Calibration & Diagnostics

## 1. WHY: Calibration vs. Discrimination

A classifier has two distinct properties:

1. **Discrimination (Ranking / AUC):** Can the model correctly rank positive instances higher than negative instances?
2. **Calibration (Reliability):** Do the predicted raw probability numbers reflect real-world event frequencies?

```
                     PERFECT RANKING, TERRIBLE CALIBRATION
                     
   Actual Class:    [  0  ,   0  ,   0  ,   1  ,   1  ,   1  ]
   Model Output:    [0.81 , 0.83 , 0.85 , 0.92 , 0.95 , 0.98]
   
   ROC-AUC Score = 1.0 (Perfect separation)
   Real Event Frequency for top items ≈ 100%, but predictions hover at 95%.
   If business logic spends $100 whenever P(Event) > 0.90, this model wastes money on 0s!

```

> **Definition of Calibration:**
> A model is well-calibrated if, among all instances where it predicts a probability $\hat{p} = 0.80$, exactly $80\%$ of those instances are actually Positive ($Y = 1$).

---

## 2. WHY LOGISTIC REGRESSION MISCALIBRATES

While standard logistic regression on unweighted, balanced data is naturally well-calibrated in the aggregate, it breaks in production due to four primary triggers:

| Trigger | Root Cause | Effect on Calibration Curve |
| --- | --- | --- |
| **1. Class Resampling / Subsampling** | Training on downsampled negatives (e.g., $1:10$ instead of real $1:1000$ fraud ratio) to speed up training. | **Overconfidence:** Baseline probabilities shift upward across the board. |
| **2. Heavy Regularization ($\lambda \gg 0$)** | L1/L2 penalties pull feature weights $\mathbf{w} \to 0$ toward the origin. | **Squeezed Probabilities:** Outputs pull toward the prior $P(Y=1)$, under-predicting extremes ($p \to 0.5$). |
| **3. Non-Linearity / Misspecification** | True decision boundary is non-linear, but model uses raw features without polynomial/spline terms. | **Local Distortion:** Calibration is good on average, but severely distorted in specific feature regions. |
| **4. Feature Drift in Production** | Distribution of input $\mathbf{x}$ shifts over time while weights $\mathbf{w}$ remain static. | **Systematic Bias:** Overall predicted mean drifts away from observed positive rate. |

---

## 3. DIAGNOSTICS: How to Detect Miscalibration

### Metric 1: Expected Calibration Error (ECE)

ECE bins predictions into $M$ equally spaced (or quantile-spaced) probability intervals $B_m$ and computes the weighted average absolute difference between predicted probability and actual positive fraction:

$$\text{ECE} = \sum_{m=1}^{M} \frac{\vert{}B_m\vert{}}{N} \left\vert{} \text{acc}(B_m) - \text{conf}(B_m) \right\vert{}$$

Where:

* $\text{conf}(B_m) = \frac{1}{\vert{}B_m\vert{}} \sum_{i \in B_m} \hat{p}_i$ (Average predicted probability in bin $m$)
* $\text{acc}(B_m) = \frac{1}{\vert{}B_m\vert{}} \sum_{i \in B_m} y_i$ (Actual fraction of positives in bin $m$)

---

### Metric 2: Brier Score

The MSE of probabilistic predictions:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

* Decomposes into: $\text{Brier Score} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$.
* Lower is better ($\text{BS} = 0$ is perfect).

---

## 4. HOW TO FIX IT: Diagnostic & Correction Strategies

```
                             CALIBRATION FIX DECISION TREE
                             
                                 What caused the shift?
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
    Class Subsampling            Over-Regularization / Drift      Non-Linear Features
             │                             │                             │
             ▼                             ▼                             ▼
   Analytical Prior Shift       Platt Scaling / Isotonic       Platt Scaling + Feature
   (Exact Prior Adjustment)            Regression               Splines / Interactions

```

---

### FIX 1: Analytical Prior Shift Adjustment (Subsampling Case)

If you downsampled the negative class during training to handle extreme imbalance, **do not retrain the model**. You can fix the output mathematically.

Let $\beta$ be the fraction of negative instances kept during training ($\beta = \frac{N_{\text{neg, train}}}{N_{\text{neg, real}}}$):

#### Mathematical Correction:

$$\ln\left(\frac{p_{\text{real}}}{1 - p_{\text{real}}}\right) = \ln\left(\frac{p_{\text{train}}}{1 - p_{\text{train}}}\right) + \ln(\beta)$$

$$\text{Adjusted Intercept } b_{\text{real}} = b_{\text{train}} + \ln(\beta)$$

> **Example:**
> If you kept $10\%$ of negatives ($\beta = 0.1$), $\ln(0.1) \approx -2.302$. Simply subtract $2.302$ from your learned bias $b_{\text{train}}$, and raw sigmoid outputs will instantly recalibrate to production base rates!

---

### FIX 2: Platt Scaling (Parametric Post-Processing)

Platt scaling trains a secondary 1D logistic regression model on the uncalibrated validation set logits ($z = \mathbf{w}^T \mathbf{x} + b$).

#### Formula:

$$\hat{p}_{\text{calibrated}} = \frac{1}{1 + e^{-(A \cdot z + B)}}$$

Where scalar parameters $A$ and $B$ are learned via Cross-Entropy Loss on a held-out validation set.

* **When to use:** Small validation datasets ($N < 1000$), or when the uncalibrated probabilities show a smooth sigmoidal distortion.

---

### FIX 3: Isotonic Regression (Non-Parametric Post-Processing)

Isotonic regression fits a non-decreasing, piecewise constant step function to map uncalibrated probabilities to true outcomes.

$$\hat{p}_{\text{calibrated}} = f_{\text{isotonic}}(\hat{p}_{\text{raw}})$$

* **When to use:** Large validation datasets ($N > 1000$).
* **Advantage:** Non-parametric; can fix arbitrary monotonic distortions (curves, multi-step s-shapes).
* **Warning:** Risk of overfitting on small validation samples.

---

## 5. INTERACTIVE CALIBRATION & RE-CALIBRATION SIMULATOR

Test how Class Subsampling distorts probability outputs and evaluate how **Platt Scaling** and **Prior Adjustment** recover ideal calibration curves in real time.

---

## 6. COMPLETE PYTHON IMPLEMENTATION

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# 1. Simulate imbalanced dataset (1% positive class)
np.random.seed(42)
N = 10000
X = np.random.randn(N, 2)
# True log-odds formula
z_true = -4.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1]
p_true = 1 / (1 + np.exp(-z_true))
y = (np.random.rand(N) < p_true).astype(int)

# 2. Subsample negatives for training (Keep 10% negatives: beta = 0.1)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
sampled_neg_idx = np.random.choice(neg_idx, size=int(len(neg_idx) * 0.1), replace=False)

train_idx = np.concatenate([pos_idx, sampled_neg_idx])
X_train, y_train = X[train_idx], y[train_idx]

# 3. Fit uncalibrated model on subsampled data
model = LogisticRegression(C=1.0)
model.fit(X_train, y_train)

# Raw uncalibrated predictions on full production dataset
p_uncalibrated = model.predict_proba(X)[:, 1]
brier_uncalibrated = brier_score_loss(y, p_uncalibrated)

# 4. FIX A: Analytical Prior Adjustment
beta = 0.1
b_train = model.intercept_[0]
model.intercept_[0] = b_train + np.log(beta)  # Shift bias

p_analytical = model.predict_proba(X)[:, 1]
brier_analytical = brier_score_loss(y, p_analytical)

# Reset intercept for Platt scaling demonstration
model.intercept_[0] = b_train

# 5. FIX B: Post-Hoc Platt Scaling via CalibratedClassifierCV
platt_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
platt_model.fit(X_train, y_train)

p_platt = platt_model.predict_proba(X)[:, 1]
brier_platt = brier_score_loss(y, p_platt)

# Evaluation Output
print(f"Uncalibrated Brier Score: {brier_uncalibrated:.5f}")
print(f"Analytical Shift Brier:    {brier_analytical:.5f}")
print(f"Platt Scaling Brier:       {brier_platt:.5f}")

```
