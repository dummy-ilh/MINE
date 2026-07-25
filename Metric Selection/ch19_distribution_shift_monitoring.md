# Chapter 19: Distribution Shift & Monitoring

> *"A model is not a static artifact. It is a living system operating in a changing world. The moment you deploy it, the world begins to drift away from the world it was trained on. Your job doesn't end at deployment — it begins there."*

---

## 19.1 The Deployment Reality

Every ML model is trained on historical data and deployed into a future that differs from that history. This gap — between training distribution and deployment distribution — is the central challenge of production ML.

```
Training time:  Data from Jan 2023 – Dec 2023
Deployment:     Jan 2024 → ongoing

What has changed?
  User behavior:    New patterns, new demographics, new devices
  Product:          UI changes, new features, A/B test effects
  World:            Economic shifts, news events, seasonal patterns
  Data pipeline:    Schema changes, upstream feature changes, bugs
  Competition:      Users' alternatives change their behavior
```

A model that achieves 92% accuracy at deployment might achieve 78% accuracy six months later — not because the model changed, but because the world did.

### The Monitoring Gap

Most teams invest heavily in pre-deployment evaluation and lightly in post-deployment monitoring. This is backwards:

```
Pre-deployment:   Model is evaluated once, carefully, on held-out data
                  → Low stakes; you catch problems before they reach users

Post-deployment:  Model runs 24/7 on real users for months or years
                  → High stakes; undetected drift causes real harm
                  → Monitoring investment is typically 10% of modeling investment
                  → Should be much closer to 50%
```

---

## 19.2 Types of Distribution Shift

### Covariate Shift (Input Drift)

The input distribution P(X) changes, but the conditional distribution P(Y|X) remains the same.

```
Training:   Young urban users; mostly mobile traffic
Production: Expanded to rural demographic; desktop traffic increases

P(X) changed: feature distributions are different
P(Y|X) same: given the same features, the model's relationships hold

Effect: Model may still be correct per-sample, but operates in an
        unfamiliar region of feature space. Calibration may drift.
        Features that were rare in training become common.
```

### Label Shift (Prior Probability Shift)

The marginal distribution of labels P(Y) changes, but P(X|Y) remains the same.

```
Training:   Flu season; high prevalence of flu cases
Production: Summer; low flu prevalence

P(Y) changed: base rate of flu dropped
P(X|Y) same: flu patients still present the same way

Effect: Model's predicted probabilities are miscalibrated.
        A model trained at 20% flu prevalence over-predicts
        flu probability in summer when prevalence is 5%.
```

### Concept Drift (Real Concept Change)

The relationship between inputs and outputs P(Y|X) changes.

```
Fraud detection:
  Training:   Fraudsters use stolen card numbers, large transactions
  Production: Fraudsters now use card-not-present attacks, small test transactions

P(Y|X) changed: the same features no longer predict fraud the same way

This is the hardest to handle: the model's learned function is wrong,
not just the input distribution.
```

### Data Drift Subcategories

```
Feature drift:    Individual feature distribution changes
  → P(age) shifts; younger users dominate

Schema drift:     Feature structure changes
  → New categories added; old ones retired; column renamed

Upstream drift:   Upstream pipeline changes affect features
  → A feature is computed differently after a pipeline update

Feedback loop drift: Model's own decisions change the distribution
  → Recommender shows more action movies → users watch more action
    → training data shifts toward action → model recommends even more action
```

---

## 19.3 Statistical Tests for Distribution Shift

### Population Stability Index (PSI)

The most widely used industry metric for detecting feature drift. Originally from credit risk modeling.

```
PSI = Σᵢ (Actual%ᵢ - Expected%ᵢ) × ln(Actual%ᵢ / Expected%ᵢ)

Where:
  i = bin index
  Expected% = fraction of training data in bin i
  Actual% = fraction of production data in bin i
```

**Interpretation:**

| PSI | Meaning |
|---|---|
| < 0.10 | No significant shift; model stable |
| 0.10 – 0.20 | Moderate shift; investigate |
| > 0.20 | Significant shift; model likely degraded; consider retraining |

```python
import numpy as np

def psi(expected, actual, n_bins=10):
    """
    expected: array of training data values
    actual:   array of production data values
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        n_bins + 1
    )

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   breakpoints)[0] / len(actual)

    # Clip to avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct   = np.clip(actual_pct,   1e-6, None)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value

# Monitor each feature
for feature in features:
    psi_val = psi(X_train[feature], X_prod[feature])
    if psi_val > 0.20:
        print(f"ALERT: {feature} PSI = {psi_val:.3f} — significant drift")
    elif psi_val > 0.10:
        print(f"WARN:  {feature} PSI = {psi_val:.3f} — moderate drift")
```

### Kolmogorov-Smirnov (KS) Test

Non-parametric test comparing two distributions. Tests whether two samples come from the same distribution.

```
KS statistic = max|F₁(x) - F₂(x)|

F₁ = empirical CDF of training distribution
F₂ = empirical CDF of production distribution
```

```python
from scipy.stats import ks_2samp

ks_stat, p_value = ks_2samp(X_train[feature], X_prod[feature])

if p_value < 0.05:
    print(f"KS test: distributions differ significantly (D={ks_stat:.4f}, p={p_value:.4f})")
```

**PSI vs. KS:**
| Property | PSI | KS |
|---|---|---|
| Output | Scalar (interpretable threshold) | Statistic + p-value |
| Sensitivity | Moderate | High |
| Requires binning | Yes | No |
| Null distribution | Assumed | Exact |
| Industry standard | Credit risk, banking | Research, general ML |

### Chi-Squared Test (Categorical Features)

For categorical features, test whether the distribution of categories has changed:

```python
from scipy.stats import chi2_contingency

# Observed counts in training and production
observed = np.array([
    [train_cat_A, train_cat_B, train_cat_C],
    [prod_cat_A,  prod_cat_B,  prod_cat_C]
])

chi2, p_value, dof, expected = chi2_contingency(observed)
if p_value < 0.05:
    print(f"Categorical drift detected (χ²={chi2:.2f}, p={p_value:.4f})")
```

### Maximum Mean Discrepancy (MMD)

A kernel-based test for distribution shift that works on the full multivariate distribution simultaneously — not feature by feature.

```
MMD²(P, Q) = E_{x,x'~P}[k(x,x')] - 2E_{x~P,y~Q}[k(x,y)] + E_{y,y'~Q}[k(y,y')]

k = kernel function (typically RBF: k(x,y) = exp(-||x-y||²/2σ²))
```

MMD = 0 when P = Q. Increasing MMD indicates increasing distributional divergence.

```python
from sklearn.metrics.pairwise import rbf_kernel

def mmd_squared(X_train, X_prod, gamma=1.0):
    KXX = rbf_kernel(X_train, X_train, gamma)
    KYY = rbf_kernel(X_prod,  X_prod,  gamma)
    KXY = rbf_kernel(X_train, X_prod,  gamma)
    return KXX.mean() - 2*KXY.mean() + KYY.mean()

mmd = mmd_squared(X_train_sample, X_prod_sample)
```

Best for detecting subtle multivariate shift that univariate tests miss.

---

## 19.4 Data Drift vs. Concept Drift: Detection

### Detecting Data Drift

Monitor input feature distributions:
- PSI per feature (numerical)
- Chi-squared per feature (categorical)
- Multivariate: MMD or classifier-based drift detection

### Detecting Concept Drift

Harder — requires labels, which are often delayed or unavailable in production.

**With labels (when available):**
Monitor model performance metrics directly:
```
Accuracy over time, rolling 7-day window:
  Week 1: 91.2%
  Week 2: 90.8%
  Week 3: 89.1%   ← declining
  Week 4: 87.3%   ← trigger retrain
```

**Without labels (label-free drift detection):**

**Classifier-based detection (Domain Classifier):**
Train a binary classifier to distinguish training vs. production samples. If it achieves high accuracy, the distributions are distinguishable → drift detected.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Label training data as 0, production data as 1
X_combined = np.vstack([X_train, X_prod])
y_domain   = np.array([0]*len(X_train) + [1]*len(X_prod))

domain_clf = GradientBoostingClassifier()
auc = cross_val_score(domain_clf, X_combined, y_domain,
                      scoring='roc_auc', cv=5).mean()

if auc > 0.7:
    print(f"Drift detected: domain classifier AUC = {auc:.3f}")
    # Feature importances show WHICH features are drifting
    importances = dict(zip(features, domain_clf.feature_importances_))
```

**Prediction distribution monitoring:**
Monitor the distribution of model outputs (scores, predicted probabilities) rather than inputs. If P(ŷ) shifts significantly, something has changed upstream.

```python
# Monitor predicted score distribution
psi_score = psi(train_predicted_scores, prod_predicted_scores)
if psi_score > 0.10:
    print(f"Score distribution drift: PSI = {psi_score:.3f}")
```

### Drift Detection Algorithms

**ADWIN (Adaptive Windowing):**
Maintains a sliding window; detects change points in the data stream by finding sub-windows with significantly different statistics.

**DDM (Drift Detection Method):**
Monitors error rate over time. Triggers warning when error rate exceeds expected range by 1 SD; triggers alert at 2 SD.

**Page-Hinkley Test:**
Detects changes in the mean of a metric. Accumulates a statistic and triggers when it exceeds a threshold.

```python
from river.drift import ADWIN

detector = ADWIN()
for accuracy in accuracy_stream:
    detector.update(accuracy)
    if detector.drift_detected:
        print(f"Drift detected at observation {detector.n_seen}")
```

---

## 19.5 Monitoring Infrastructure

### What to Monitor

```
Layer 1: Data pipeline health
  ├── Null rates per feature
  ├── Feature value ranges (min, max, mean, std)
  ├── Schema compliance (expected types, cardinalities)
  └── Volume (# records per time period)

Layer 2: Feature distribution
  ├── PSI per numerical feature
  ├── Chi-squared per categorical feature
  ├── Feature correlation matrix drift
  └── Domain classifier AUC (multivariate drift)

Layer 3: Model output
  ├── Score/probability distribution (PSI)
  ├── Prediction distribution (class balance)
  ├── Calibration (when labels available)
  └── Confidence distribution

Layer 4: Model performance (when labels available)
  ├── Accuracy, F1, AUC (rolling window)
  ├── Per-segment performance
  ├── Calibration metrics (ECE, Brier Score)
  └── Business metric proxy (revenue, CTR, conversion)

Layer 5: Business impact
  ├── Primary KPI trend
  ├── User satisfaction signals
  └── Escalation / error rates
```

### Monitoring Cadence

| Signal | Cadence | Why |
|---|---|---|
| Pipeline health (nulls, volume) | Real-time / hourly | Catch upstream failures fast |
| Feature distribution (PSI) | Daily | Drift is usually gradual |
| Score distribution | Daily | Fast signal for concept drift |
| Model performance (with labels) | As labels arrive | Weekly for churn, daily for fraud |
| Business metrics | Daily | Lagging but high-signal |
| Full retraining evaluation | Monthly or on-trigger | Expensive; data-dependent |

### Alerting Thresholds

Design alerts with escalating severity:

```
LEVEL 1 (INFO): PSI 0.10-0.20 on a feature
  Action: Log; investigate in next daily review

LEVEL 2 (WARNING): PSI > 0.20 on a feature OR score PSI > 0.10
  Action: Notify ML engineer; investigate within 24h

LEVEL 3 (CRITICAL): Model performance dropped > X% OR revenue impact detected
  Action: Page on-call; incident response; potential rollback

LEVEL 4 (INCIDENT): Complete model failure, data pipeline error, safety issue
  Action: Immediate rollback; incident channel; post-mortem
```

---

## 19.6 Shadow Mode and Challenger Models

### The Challenger Pattern

Always have a retrained model ready to replace the champion:

```
Production:   Champion model (trained Jan 2023)
Shadow mode:  Challenger model (retrained Aug 2023, not yet serving users)

Every request → both models score it
               → champion's score serves the user
               → challenger's score logged for comparison

After N days: Compare champion vs. challenger offline
              If challenger better: A/B test → ship
              If challenger worse: investigate; don't ship
```

This maintains a continuous pipeline of improvements without emergency retrains under pressure.

### Canary Retraining

When data drift is detected, don't immediately retrain on all new data. Test incrementally:

```
Drift detected → Retrain on recent 3 months
              → Shadow mode evaluation (1 week)
              → Canary: 5% traffic (3 days)
              → A/B test: 50/50 (7 days)
              → Full rollout if metrics pass
```

---

## 19.7 Retraining Strategies

When and how to retrain is as important as what to monitor.

### Trigger-Based Retraining

Retrain when a monitored signal crosses a threshold:

```
Trigger: PSI > 0.20 on any top-5 feature
Trigger: Model accuracy drops > 3% on rolling 7-day window
Trigger: Business metric (CTR, revenue) drops > 2% for > 3 days
Trigger: Label distribution shifts > 15%
```

**Advantage:** Efficient — only retrain when needed.
**Risk:** Latency between drift and trigger; damage accumulates.

### Scheduled Retraining

Retrain on a fixed schedule regardless of drift signals:

```
Daily:    High-velocity data (fraud, ads, news)
Weekly:   Medium-velocity data (recommendations, search)
Monthly:  Low-velocity data (credit risk, churn)
```

**Advantage:** Simple; predictable; catches slow drift that falls below alert thresholds.
**Risk:** May retrain unnecessarily when drift is low.

### Continuous Training (Online Learning)

Update the model incrementally as new data arrives:

```python
# Incremental learning with SGD
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss')

for batch in data_stream:
    X_batch, y_batch = batch
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
```

**Advantage:** Always up-to-date; no retraining latency.
**Risk:** Concept drift can corrupt the model rapidly; no rollback checkpoint; hard to debug.

**Practical approach:** Use continuous training for feature extractors and lightweight heads; retain periodic checkpointing for rollback.

### Choosing Training Window

When retraining, how much historical data should you use?

```
Expanding window:  All data from the beginning
  + More data → more stable estimates
  - Old data may be misleading under concept drift

Sliding window:    Only the most recent N months
  + Reflects current distribution
  - Less data; higher variance; throws away useful history

Weighted window:  All data, but recent data weighted more
  + Best of both worlds
  - Requires tuning the weight decay

Hybrid:           Recent data for recalibration; older data for structure
  + Common in practice for deep models (pretrain on large, finetune on recent)
```

---

## 19.8 Label Delay and Delayed Feedback

One of the most underappreciated challenges in production ML: **labels arrive late**.

```
Fraud detection:
  Transaction happens at t=0
  Fraud confirmed (charged back) at t=30 to t=90 days
  → For 30-90 days, you can't evaluate model accuracy on recent data

Churn prediction:
  Prediction made at t=0
  Churn observed at t=30 to t=180 days
  → Model quality is invisible for months

Credit default:
  Loan issued at t=0
  Default occurs at t=6 to t=24 months
  → Cannot measure model quality in near-real-time
```

### Strategies for Delayed Feedback

**Proxy labels:** Use fast-arriving signals as proxies for delayed true labels.
```
True label:  Customer churns (observed in 60 days)
Proxy label: Customer stops logging in for 14 days (observed immediately)

Risk: Proxy may not perfectly reflect true label.
```

**Partial feedback:** Use available labels even if incomplete.
```
At t+7 days: 30% of fraud labels arrived → evaluate on these
At t+30 days: 80% arrived → more complete evaluation
At t+90 days: 99% arrived → near-complete evaluation

Plot metric as a function of label maturity to understand delay distribution.
```

**Waiting period + lag evaluation:** Accept that you're evaluating last month's model on last month's data. Plan your retrain cadence accordingly.

---

## 19.9 The Monitoring Dashboard

A production ML system should have a living dashboard with at minimum:

```
┌─────────────────────────────────────────────────────────┐
│  ML System Health Dashboard                             │
├─────────────────────────────────────────────────────────┤
│  Data Pipeline                                          │
│    Record volume:    12,341 (↓2.1% vs. yesterday) ⚠️   │
│    Null rate:        0.3%   (baseline: 0.3%)     ✓     │
│    Schema errors:    0                            ✓     │
├─────────────────────────────────────────────────────────┤
│  Feature Drift (top 10 features)                        │
│    age:           PSI = 0.04  ✓                        │
│    purchase_amt:  PSI = 0.11  ⚠️                       │
│    device_type:   χ² p=0.21  ✓                        │
│    [Domain clf AUC: 0.58  ✓ (<0.65 = no drift)]       │
├─────────────────────────────────────────────────────────┤
│  Model Output                                           │
│    Score PSI:     0.06   ✓                             │
│    Positive rate: 2.3%   (baseline: 2.1%)  ✓          │
├─────────────────────────────────────────────────────────┤
│  Model Performance (7-day rolling)                      │
│    AUC:           0.873  (baseline: 0.881)  ⚠️        │
│    Precision@top: 0.61   (baseline: 0.64)   ⚠️        │
│    Brier Score:   0.098  (baseline: 0.092)  ⚠️        │
├─────────────────────────────────────────────────────────┤
│  Business Metrics                                       │
│    CTR:           2.14%  (baseline: 2.18%)  ✓         │
│    Conversion:    0.43%  (baseline: 0.42%)  ✓         │
├─────────────────────────────────────────────────────────┤
│  Challenger Model (shadow mode since 2024-01-10)        │
│    AUC vs champion: +0.012  →  Ready for A/B test      │
└─────────────────────────────────────────────────────────┘
```

---

## 19.10 Worked Example: Fraud Detection Monitoring

**System:** Real-time fraud classification. 2M transactions/day. Model trained October 2023, deployed November 2023.

```
January 2024: Monitoring Dashboard Alerts

Week 1 (Jan 8):
  PSI(transaction_amount): 0.13  ⚠️
  PSI(merchant_category):  0.08  ✓
  Score PSI:                0.07  ✓
  AUC (30-day lag):         0.891  ✓ (still good; labels from November)

  → Holiday shopping spike caused transaction amount shift
  → Expected seasonal effect; no action

Week 2 (Jan 15):
  PSI(transaction_amount):  0.09  ✓  (post-holiday normalization)
  PSI(device_type):         0.22  ❌  NEW ALERT
  Score PSI:                0.14  ⚠️
  AUC (30-day lag):         0.891  ✓ (still November labels)

  Domain classifier AUC: 0.74  ❌  (> 0.70 threshold)

  Investigation: New mobile OS update changed device fingerprint encoding.
  Upstream feature pipeline hadn't adapted. device_type = "unknown" increased
  from 2% to 34% of transactions.

  → Pipeline bug fixed (Jan 17).
  → device_type drift resolved by Jan 19.

Week 3 (Jan 22):
  All PSI < 0.10  ✓
  Score PSI: 0.05  ✓
  AUC (30-day lag, now covering Dec labels): 0.847  ❌

  Drop of 0.044 in AUC. December fraud patterns differ from October.
  New fraud vector: account takeover via credential stuffing
  (not well-represented in October training data)

  → Concept drift confirmed.
  → Emergency retrain on Oct-Dec data (2 days)
  → Challenger model: AUC = 0.882 on held-out Jan labels
  → Canary: 5% traffic for 3 days → clean
  → A/B test: 50% for 7 days → AUC improvement confirmed
  → Full rollout: Jan 31
```

**Timeline summary:**
```
Nov 2023: Model deployed, AUC = 0.891
Jan 8:    Holiday seasonal drift detected → no action (expected)
Jan 15:   Pipeline bug detected via PSI → fixed in 2 days
Jan 22:   Concept drift detected via lagged AUC → retrain
Jan 31:   Retrained model deployed, AUC = 0.882
```

**Lessons:**
1. PSI caught the pipeline bug that AUC missed (because labels were lagged)
2. Seasonal drift looked like a problem but wasn't — investigation prevented unnecessary retrain
3. Domain classifier AUC was the first signal for the device_type bug
4. Lagged AUC confirmed concept drift 6 weeks after deployment — the delay is unavoidable
5. Challenger model in shadow mode meant retrain cycle was 9 days, not weeks

---

## 19.11 The Monitoring Maturity Model

Teams evolve through stages:

```
Stage 0: No monitoring
  "We shipped it; it's fine."
  Drift discovered by user complaints or revenue drop.

Stage 1: Ad-hoc monitoring
  Someone checks metrics occasionally.
  Alerts are manual and inconsistent.

Stage 2: Basic automated monitoring
  Pipeline health alerts.
  Scheduled accuracy checks when labels arrive.
  Manual investigation.

Stage 3: Feature drift monitoring
  PSI/KS per feature, automated.
  Score distribution monitoring.
  Alerting with severity levels.

Stage 4: Full ML observability
  All layers monitored automatically.
  Challenger models always running in shadow mode.
  Automated retrain trigger pipeline.
  Business metric integration.

Stage 5: Closed-loop ML
  Drift → automated investigation → triggered retrain → shadow eval
  → canary → A/B → rollout, all automated with human approval gates.
  Continuous training for appropriate components.
  Full experiment audit trail.
```

Most production teams are at Stage 2–3. Stage 4–5 requires significant infrastructure investment but pays dividends at scale.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Covariate shift | P(X) changes; P(Y\|X) stable; recalibrate |
| Label shift | P(Y) changes; recalibrate priors |
| Concept drift | P(Y\|X) changes; must retrain |
| PSI | Industry-standard feature drift detector; threshold at 0.10/0.20 |
| KS test | Non-parametric shift test; good for continuous features |
| Domain classifier | Multivariate drift detection; AUC > 0.70 signals drift |
| Label delay | Performance is invisible until labels arrive; use proxy labels |
| Trigger-based retrain | Retrain on signal; efficient but has latency |
| Scheduled retrain | Predictable; catches slow drift; may be unnecessary |
| Shadow mode | Always have a challenger running; reduces emergency retrain pressure |
| Monitoring layers | Data → features → scores → performance → business; monitor all |
| Maturity model | Most teams at Stage 2–3; target Stage 4 |

---

## Further Reading

- Sculley et al. — *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015) — the foundational paper on production ML challenges
- Klaise et al. — *Alibi Detect: Algorithms for Outlier, Adversarial and Drift Detection* (JMLR, 2022)
- Lu et al. — *Learning Under Concept Drift: A Review* (IEEE TKDE, 2018)
- Deng et al. — *Data Drift Monitor: Detecting and Diagnosing Model Performance Degradation* (2022)
- Gama et al. — *A Survey on Concept Drift Adaptation* (ACM Computing Surveys, 2014)
- Huyen, C. — *Designing Machine Learning Systems* (O'Reilly, 2022) — Chapter 8–9 on monitoring
- Kleppmann, M. — *Designing Data-Intensive Applications* (O'Reilly, 2017) — infrastructure foundations

---

*This concludes the ML Metrics & Evaluation course.*

---

## Course Summary: All 19 Chapters

| Chapter | Topic | Key Takeaway |
|---|---|---|
| 1 | Metric Selection | Satisficing + optimizing; proxy vs. target |
| 2 | Offline vs. Online | Evaluation ladder; never skip levels |
| 3 | Ranking Metrics | NDCG for graded relevance; MRR for single answer |
| 4 | Calibration | Platt → isotonic → temperature scaling |
| 5 | Business Alignment | Causal chain from model to business outcome |
| 6 | Goodhart's Law | Measure becomes a target; ceases to be a measure |
| 7 | Confusion Matrix | Threshold is a business decision |
| 8 | Regression Metrics | MAE robust; RMSE penalizes outliers; Huber blends both |
| 9 | Imbalanced Classes | MCC > F1 > Accuracy; PR-AUC > ROC-AUC |
| 10 | Multi-label | Hamming lenient; subset accuracy strict; macro exposes rare class failures |
| 11 | Probabilistic | Log-loss proper; Brier decomposable; ECE diagnostic |
| 12 | Uncertainty | Conformal prediction gives guaranteed coverage |
| 13 | NLP Evaluation | BLEU for speed; BERTScore for meaning; human for truth |
| 14 | LLM Evaluation | Multi-dimensional; no single number; arena = gold standard |
| 15 | Recommender | Beyond accuracy: diversity, novelty, serendipity, fairness |
| 16 | Fairness | Impossibility theorem; equalized odds vs. predictive parity |
| 17 | Survival | C-Index = survival AUC; IBS = survival calibration |
| 18 | Statistical Significance | Power before running; CUPED for variance reduction |
| 19 | Distribution Shift | Monitor all layers; drift before labels arrive |
