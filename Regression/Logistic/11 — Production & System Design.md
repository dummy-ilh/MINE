
# Module 11 — Production & System Design Angle (L5-Critical)

## 1. WHY

Everything so far has been about building and understanding the model itself. But at the L5/ICT5 level, interviewers don't just want to know you can build a model — they want to know you can **operate one in a real production system**, over time, at scale, with real engineering tradeoffs.

This module is where "I understand logistic regression" becomes "I could be trusted to own a production ML system." This is explicitly flagged as critical in your curriculum, so let's be thorough.

```
+-----------------------------------------------------------------------------------+
|                        PRODUCTION ML LIFE CYCLE ARCHITECTURE                      |
+-------------------+-------------------+-------------------+-----------------------+
|  DATA & FEATURES  | TRAINING PIPELINE | SERVING INFRA     | MONITORING & GOV.     |
+-------------------+-------------------+-------------------+-----------------------+
| • Feature Store   | • Offline Loss    | • Sub-ms Latency  | • Drift (PSI/KS)      |
| • Transforms      | • Class Weights   | • Online Ingestion| • Calibration Decay   |
| • One-Hot / Bins  | • Cross-Val (PR)  | • Microservice    | • Automated Retrain   |
+-------------------+-------------------+-------------------+-----------------------+

```

---

## 2. FEATURE ENGINEERING FOR LOGISTIC REGRESSION

### WHY this needs special attention (vs. tree-based models)

Logistic regression can **only** express a straight-line relationship with log-odds (Module 10). Tree-based models (like gradient boosting) can automatically discover non-linear patterns and interactions on their own through recursive partitioning. Logistic regression can't — **you, the engineer, have to hand it those patterns explicitly**, through feature engineering.

```
       [ Gradient Boosting ]                      [ Logistic Regression ]
       (Automatic Splitting)                      (Requires Hand-Crafting)

     Feature X1                                     Feature X1
      /      \                                          |
   <= 50    > 50                                  [ log(X1) ]  or  [ Splines ]
   /   \    /   \                                       |
 ...   ... ...   ...                              Linear in Log-Odds

```

### Monotonic Transforms

If a feature's true relationship with log-odds is curved but consistently increasing or decreasing (monotonic), applying a functional transformation before feeding it into the model can straighten out that relationship:

$$\text{Logarithmic: } X' = \ln(X + 1) \quad \mid \quad \text{Power / Root: } X' = \sqrt{X} \quad \mid \quad \text{Box-Cox: } X' = \frac{X^\lambda - 1}{\lambda}$$

*Example:* In fintech, `income` or `account_balance` follows a power-law distribution. Using $\ln(\text{income})$ transforms a heavily right-skewed variable into a Gaussian-like distribution, linearizing its relationship with the log-odds of loan repayment.

### Binning & Discretization

Convert a continuous feature into $K$ discrete buckets (e.g., tenure $\to$ "0-6 months", "6-12 months", etc.) and apply one-hot encoding across $K-1$ dummy variables.

* **Advantage:** Allows the model to assign a completely independent coefficient $\beta_k$ to each bucket. This captures arbitrary non-linear and non-monotonic shapes (such as U-shaped churn behavior: high churn for new users, low for mid-tenure, high for long-tenure).
* **Tradeoff:** Destroys continuous distance metrics between adjacent values, introduces arbitrary boundary thresholds, increases feature dimensionality, and requires adequate sample density per bucket to avoid high estimator variance.

### Interaction Terms

Logistic regression assumes additive effects on log-odds: $\eta = \beta_0 + \beta_1 X_1 + \beta_2 X_2$. It cannot detect conditionally dependent risk unless you manually inject multiplicative interaction terms:

$$X_{\text{inter}} = X_1 \times X_2$$

*Example:* A credit model where `high_credit_card_utilization` is only risky when paired with `low_liquid_savings`.

---

## 3. HANDLING CLASS IMBALANCE IN PRODUCTION SYSTEMS

We already established WHY this matters in Module 8 (the accuracy trap with rare positive classes). Here is the practical toolkit for fixing it in production:

```
+---------------------------------------------------------------------------------+
|                        CLASS IMBALANCE STRATEGY MATRIX                          |
+-----------------+-----------------------+-----------------------+---------------+
| Technique       | Mechanism             | Operational Cost      | Risk          |
+-----------------+-----------------------+-----------------------+---------------+
| Class Weights   | Multiplies loss per   | Zero runtime cost     | Calibration   |
|                 | minority row          |                       | shift         |
| Oversampling    | Replicates minority   | Increases offline     | Overfitting   |
|                 | samples (or SMOTE)    | training time         | duplicates    |
| Undersampling   | Drops majority class  | Fast training; loses  | Discards      |
|                 | samples               | real data             | real signals  |
+-----------------+-----------------------+-----------------------+---------------+

```

### Class Weights (Loss Modification)

Adjusts the binary cross-entropy loss function by weighting the contribution of each class inversely proportional to its class frequency:

$$\mathcal{L}_{\text{weighted}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 \cdot y_i \ln(\hat{p}_i) + w_0 \cdot (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

Where standard balanced weights are set as:

$$w_1 = \frac{N}{2 \cdot N_{y=1}}, \quad w_0 = \frac{N}{2 \cdot N_{y=0}}$$

* **Pros:** Zero footprint during runtime/inference. Simple to implement during training.
* **Cons:** Alters raw output probability scale. The raw output $\hat{p}_{\text{weighted}}$ no longer reflects the true prior base rate and requires mathematical recalibration before downstream probability-driven decisions are made.

### Resampling — Oversampling vs. Undersampling

* **Oversampling (SMOTE / Random Duplication):** Generates synthetic minority samples by interpolating between nearest neighbors in feature space.
* *Production Hazard:* Synthetic data points can violate physical domain constraints or generate unrealistic feature combinations (e.g., negative balances or invalid state codes).


* **Undersampling (Random / Tomek Links):** Randomly discards majority-class instances until target ratios are achieved.
* *Production Hazard:* Discarding majority class data removes crucial variance informativeness near decision boundaries, increasing false positive rates in dense majority-class feature regions.



---

## 4. MONITORING A LOGISTIC REGRESSION MODEL IN PRODUCTION

Deploying a model is the beginning, not the end. The real world is non-stationary, and static models silently decay.

```
       [ Production Data Stream ]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  Feature Drift       Target / Label Shift
 (PSI / KS-Test)     (Calibration Decay)
        │                   │
        └─────────┬─────────┘
                  ▼
      Alerting & Auto-Retrain

```

### Feature Drift (Covariate Shift)

Covariate shift occurs when the marginal distribution of inputs $P(X)$ changes over time, while the conditional probability $P(Y \mid X)$ remains fixed.

**Metric: Population Stability Index (PSI)**
PSI quantifies how much a continuous or categorical variable's distribution has shifted between a reference dataset (training) and a target dataset (production):

$$\text{PSI} = \sum_{k=1}^B \left( \% \text{ Actual}_k - \% \text{ Expected}_k \right) \times \ln\left( \frac{\% \text{ Actual}_k}{\% \text{ Expected}_k} \right)$$

*Where $B$ is the number of bins, $\%\text{ Actual}_k$ is the fraction of production observations in bin $k$, and $\%\text{ Expected}_k$ is the baseline training fraction.*

| PSI Threshold | Interpretation | Action Required |
| --- | --- | --- |
| $\text{PSI} < 0.10$ | No significant distribution shift | Monitor normally |
| $0.10 \le \text{PSI} < 0.25$ | Moderate shift detected | Investigate features; schedule retraining |
| $\text{PSI} \ge 0.25$ | Severe distribution shift | High priority: Retrain & re-validate immediately |

---

### Calibration Decay (Prior Probability Shift)

Calibration decay occurs when the model's output probabilities diverge from the true empirical rate, even if the model's relative ranking capability ($\text{ROC-AUC}$) remains intact.

$$\text{Expected Calibration Error (ECE)} = \sum_{m=1}^M \frac{\vert{}B_m\vert{}}{N} \left\vert{} \text{acc}(B_m) - \text{conf}(B_m) \right\vert{}$$

**Why AUC can stay high while Calibration Fails:**
Assume a holiday fraud spike doubles the true base rate of fraud from $1\%$ to $2\%$. The model still correctly ranks fraudulent transactions higher than legitimate ones ($\text{AUC} = 0.85$ stable). However, a transaction that previously had an estimated fraud probability of $0.40$ now has a true risk of $0.80$. If your automated rule auto-blocks transactions above $0.50$, the model will **fail to trigger blocks**, systematically missing fraud despite its unchanged ranking accuracy.

---

## 5. ARCHITECTURAL DECISION FRAMEWORK: LOGISTIC REGRESSION VS. GRADIENT BOOSTING / DEEP LEARNING

```
+-----------------------------------------------------------------------------------+
|                        MODEL CHOICE SYSTEM DESIGN TRADE-OFFS                      |
+----------------------+-----------------------+------------------------------------+
| Dimension            | Logistic Regression   | Gradient Boosted Trees / Neural Net|
+----------------------+-----------------------+------------------------------------+
| Latency              | Sub-millisecond (<1ms)| 10ms - 100ms+                      |
| Computational Cost   | $O(d)$ dot product    | $O(T \cdot D)$ tree traversals     |
| Interpretability     | Direct Odds Ratios    | SHAP / LIME Approximations         |
| Regulatory Compliance| High (Fully Auditable)| Moderate-Low (Black-Box risk)      |
| Cold-Start & Data Size| High performance on   | Requires large $N$ to avoid        |
|                      | small $N$             | overfitting                        |
+----------------------+-----------------------+------------------------------------+

```

### Key Selection Drivers

1. **Interpretability & Regulation:** In credit scoring (FCRA compliance), healthcare, or insurance, you are legally obligated to provide "adverse action notices" specifying the top features driving a rejection. Logistic regression provides exact log-odds contribution per feature without SHAP approximation errors.
2. **Serving Latency & Hardware Costs:** A logistic regression score is an $O(d)$ matrix multiplication ($\mathbf{w}^T \mathbf{x} + b$) followed by a scalar lookup ($\sigma(z)$). It can run in microsecond SLA environments (e.g., ad auctions, high-frequency trading) directly on edge CPUs without requiring GPU clusters.
3. **The Baseline Gold Standard:** Deploying a logistic regression model first gives you a clean performance and operational baseline. Fancier models must justify their added infrastructure costs, latency penalties, and debugging complexity by proving significant lift over this baseline.

---

## 6. A/B TESTING & ROLLOUT STRATEGY FOR MODEL REPLACEMENT

Replacing an existing production model requires careful live validation. Offline validation metrics do not always map linearly to business outcomes.

```
                  [ Live User Traffic ]
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
         [ Control 90% ]           [ Treatment 10% ]
      Legacy Logistic Reg.        New Gradient Boosting
               │                         │
               └────────────┬────────────┘
                            ▼
               Evaluate Metric Divergence:
          • Offline AUC vs Online Conversion
          • Threshold Differences
          • Calibration Errors

```

### Pre-Deployment Checklists

* **Threshold Recalibration:** Never carry over a classification threshold from Model A to Model B. If Model A was calibrated under class weights and Model B uses raw probabilities, identical thresholds will cause massive operational shifts in prediction volumes.
* **Metric Divergence Safeguard:** Verify that offline ranking metrics (PR-AUC) correlate with online KPIs (e.g., net revenue, false rejection friction).
* **Subgroup Disparity Analysis:** Evaluate sliced performance across protected categories, customer tiers, and device types to ensure aggregate gains do not mask localized performance degradations.
* **Canary Deployment Pattern:**
1. **Shadow Mode:** Route $100\%$ of production traffic to both models, but use only the legacy model's outputs. Compare predictions, check latency SLAs, and audit errors asynchronously.
2. **Canary Routing:** Direct $1\% \to 5\% \to 25\%$ of traffic to the new model using a deterministic user-id hash.
3. **Full Rollout with Circuit Breakers:** Monitor real-time error rates and automated fallbacks to immediately revert traffic if feature-drift or latency thresholds breach boundaries.



---

## 7. COMPLETE PRODUCTION PIPELINE IN PYTHON

This production-grade script covers data processing, model training with class weights, threshold tuning based on business cost matrices, population stability index (PSI) monitoring, and probability recalibration.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.isotonic import IsotonicRegression

# -------------------------------------------------------------------
# 1. Generate Imbalanced Synthetic Production Data
# -------------------------------------------------------------------
np.random.seed(42)
n_samples = 10000
fraud_ratio = 0.02  # 2% Imbalance

# Generate features
income = np.random.exponential(scale=50000, size=n_samples) + 10000
tx_amount = np.random.exponential(scale=100, size=n_samples)
risk_score = np.random.normal(loc=50, scale=15, size=n_samples)

# True log odds with non-linear log transform applied to income
log_odds = (
    -4.5 
    + 0.015 * tx_amount 
    - 0.8 * np.log(income / 10000) 
    + 0.05 * risk_score
)
probs = 1 / (1 + np.exp(-log_odds))
y = np.random.binomial(1, probs)

df = pd.DataFrame({
    'income': income,
    'tx_amount': tx_amount,
    'risk_score': risk_score,
    'target': y
})

# Split Train (Reference) and Test Sets
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])

# -------------------------------------------------------------------
# 2. Production Feature Engineering & Training Pipeline
# -------------------------------------------------------------------
def transform_features(data):
    df_out = data.copy()
    # Monotonic log transform to linearize skewed income feature
    df_out['log_income'] = np.log(df_out['income'] + 1)
    return df_out[['log_income', 'tx_amount', 'risk_score']]

X_train = transform_features(train_df)
y_train = train_df['target']

X_test = transform_features(test_df)
y_test = test_df['target']

# Fit Logistic Regression with Class Weights
model = LogisticRegression(class_weight='balanced', solver='lbfgs')
model.fit(X_train, y_train)

# -------------------------------------------------------------------
# 3. Cost-Sensitive Threshold Optimization
# -------------------------------------------------------------------
# Predict raw probabilities on test set
raw_probs_test = model.predict_proba(X_test)[:, 1]

# Business Cost Matrix
cost_false_negative = 500.0  # Missed fraud cost
cost_false_positive = 10.0   # User friction cost per false alarm

precisions, recalls, thresholds = precision_recall_curve(y_test, raw_probs_test)

best_threshold = 0.5
min_total_cost = float('inf')

for thresh in np.linspace(0.01, 0.99, 100):
    preds = (raw_probs_test >= thresh).astype(int)
    fn = np.sum((y_test == 1) & (preds == 0))
    fp = np.sum((y_test == 0) & (preds == 1))
    
    total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
    if total_cost < min_total_cost:
        min_total_cost = total_cost
        best_threshold = thresh

print("=" * 60)
print("1. OPTIMAL THRESHOLD SELECTION")
print("=" * 60)
print(f"Optimal Decision Threshold: {best_threshold:.4f}")
print(f"Minimum Total Operational Cost: ${min_total_cost:,.2f}")

# -------------------------------------------------------------------
# 4. Monitoring: Population Stability Index (PSI) Implementation
# -------------------------------------------------------------------
def calculate_psi(expected, actual, num_bins=10):
    """Calculates PSI between baseline (expected) and production (actual) data."""
    bins = np.linspace(min(expected.min(), actual.min()), 
                       max(expected.max(), actual.max()), 
                       num_bins + 1)
    
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    expected_pct = np.maximum(expected_counts / len(expected), 1e-4)
    actual_pct = np.maximum(actual_counts / len(actual), 1e-4)
    
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_val

# Simulate drifted production feature stream (e.g., inflation spikes transaction amounts)
production_tx_amount = np.random.exponential(scale=180, size=3000)

psi_tx = calculate_psi(train_df['tx_amount'], production_tx_amount)

print("\n" + "=" * 60)
print("2. FEATURE DRIFT MONITORING (PSI)")
print("=" * 60)
print(f"Transaction Amount PSI: {psi_tx:.4f}")
if psi_tx >= 0.25:
    print("ACTION: Severe Drift Detected! Trigger Automated Retraining Pipeline.")
elif psi_tx >= 0.10:
    print("WARNING: Moderate Drift Detected. Monitor closely.")
else:
    print("STATUS: Distribution Stable.")

```

---

## 8. FAANG L5 INTERVIEW PREPARATION

### System Design Scenario & Interview Rubrics

#### Interview Question

*"Design a real-time risk assessment engine for a fintech payment platform handling 50,000 requests per second. How would you build, deploy, monitor, and update a logistic regression model for this system?"*

```
                             [ High-Throughput Request ]
                                   (50,000 QPS)
                                        │
                                        ▼
                                [ Load Balancer ]
                                        │
                                        ▼
                             [ Feature Store Lookup ]
                             (Redis Cluster < 1ms)
                                        │
                                        ▼
                            [ Scoring Microservice ]
                            (Dot Product + Sigmoid)
                                        │
                                        ▼
                            [ Rules & Thresholds ]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                  [ Decision Engine ]           [ Async Kafka ]
                  (Pass / Challenge)             (Logging Stream)
                                                       │
                                                       ▼
                                             [ Monitoring Pipeline ]
                                            (PSI Drift / Recalib)

```

#### Expected L5 Architectural Answer

1. **Inference System Architecture:**
* **Feature Ingestion:** Low-latency online features (e.g., transaction frequency in last 5 minutes) pulled from an in-memory feature store (e.g., Redis / Feast) with $<1\text{ms}$ SLAs. Heavy batch features (e.g., 90-day account history) calculated asynchronously offline and synced to cache.
* **Serving Engine:** Compile logistic regression weights $\mathbf{w}$ into optimized low-level microservices (C++ / Rust / Go) executing matrix operations natively without heavy framework runtime overhead.


2. **Imbalance & Optimization Strategy:**
* Evaluate baseline using Precision-Recall AUC ($\text{PR-AUC}$), avoiding false signals from $\text{ROC-AUC}$ given $0.1\%$ fraud base rates.
* Train using loss class weighting. Calculate downstream operational thresholds by weighting the cost of false negatives (missed fraud) against customer friction from false positives (blocked transactions).


3. **Production Monitoring System:**
* Implement real-time asynchronous logging via Kafka.
* Track feature stability using Population Stability Index ($\text{PSI}$) daily across top input variables.
* Monitor probability calibration continuously via reliability diagrams and Expected Calibration Error ($\text{ECE}$) on short time windows to detect base rate shifts.


4. **Model Lifecycle & Fallbacks:**
* Deploy new model candidates via **Shadow Mode** to validate latency limits and prediction agreement rates asynchronously.
* Execute blue-green/canary traffic shifts over a 7-day window.
* Implement static, rule-based fallback safety nets if feature store lookups time out or if live inference latency exceeds $5\text{ms}$.



---

## 9. CHECK YOUR UNDERSTANDING

### Question 1

You are told your production logistic regression fraud model's AUC has stayed stable at $0.85$ for 6 months, but the fraud team says the model is "missing more fraud lately." What would you investigate first, given what you learned about drift vs. calibration?

> **Answer:** Investigate **calibration decay** caused by a shift in the baseline prior probability $P(Y=1)$ (e.g., seasonal fraud spikes or macro-economic shifts).
> $\text{ROC-AUC}$ measures purely relative ranking ability—whether fraudulent events score higher than non-fraudulent events. It remains completely unchanged if all predicted probabilities shift downward proportionately. However, if the base rate rises, fixed operational thresholds (e.g., auto-flagging predictions above $0.50$) will under-predict true risk, letting fraudulent transactions pass undetected. Recompute the **reliability diagram** and **Expected Calibration Error (ECE)** on recent data to confirm calibration drift, and update the prediction threshold or recalibrate output probabilities.

### Question 2

Your manager asks why you are starting with a logistic regression model instead of jumping straight to a deep neural network for a new binary classification problem. Give a 2-sentence answer that satisfies an L5 engineering bar.

> **Answer:** "Logistic regression provides a low-latency, fully interpretable baseline that quantifies feature signal strength and sets a clear operational benchmark before incurring the serving costs and complexity of deep models. If its performance meets business SLAs, its sub-millisecond inference and auditability make it a superior production choice; if not, it provides a validated performance floor to measure neural network lift against."
