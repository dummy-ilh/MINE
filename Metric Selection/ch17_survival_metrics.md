# Chapter 17: Survival & Time-to-Event Metrics

> *"Standard classification asks: will this event happen? Survival analysis asks: when will it happen — and how do we learn from patients we're still watching? These are different questions, and they demand different mathematics."*

---

## 17.1 The Survival Analysis Problem

Survival analysis was developed in medicine to answer questions like:
- How long do patients survive after a cancer diagnosis?
- When does a machine component fail?
- How long before a customer churns?
- When does a parolee reoffend?

What makes it different from standard regression:

**The censoring problem.** At the time of analysis, many subjects haven't experienced the event yet. A patient is still alive. A machine hasn't failed. A customer hasn't churned. You don't know their true event time — only that it's somewhere in the future.

```
Standard regression:
  Observation: [age=45, treatment=A, survival_time=2.3 years]  ← event observed

Survival analysis:
  Observation: [age=45, treatment=A, survival_time=2.3 years, censored=True]
  ← Patient is alive at 2.3 years; true survival time unknown

  Observation: [age=62, treatment=B, survival_time=1.1 years, censored=False]
  ← Patient died at 1.1 years; event observed
```

Ignoring censored observations (dropping them from analysis) produces **survivorship bias** — you only analyze people who experienced the event, systematically underestimating survival times.

### Types of Censoring

**Right censoring (most common):** Study ends before the subject experiences the event. True event time is greater than the observed time.

**Left censoring:** Subject experienced the event before the study started. Event time is less than the observed time.

**Interval censoring:** Event occurred within a known interval but exact time unknown (e.g., patient tested positive at follow-up visit, but we don't know when seroconversion occurred).

Most survival analysis focuses on right censoring.

### Survival Analysis Applications in ML

| Domain | Event | Time |
|---|---|---|
| Healthcare | Death, disease progression, readmission | Days, months, years |
| Customer analytics | Churn, subscription cancellation | Days, months |
| Manufacturing | Component failure, defect occurrence | Hours, cycles |
| Finance | Default, bankruptcy | Months, years |
| HR | Employee attrition | Months, years |
| Software | Bug occurrence, system failure | Hours, requests |

---

## 17.2 Core Survival Functions

### The Survival Function S(t)

The probability that the event has not occurred by time t:

```
S(t) = P(T > t)

Where T is the (random) time to event.

Properties:
  S(0) = 1   (no one has experienced the event at time 0)
  S(∞) = 0   (eventually everyone experiences the event)
  S(t) is monotonically non-increasing
```

### The Hazard Function h(t)

The instantaneous rate of the event occurring at time t, given survival to time t:

```
h(t) = lim_{Δt→0} P(t ≤ T < t+Δt | T ≥ t) / Δt
```

In plain language: among those who have survived to time t, what is the rate at which events are occurring?

```
Constant hazard:    h(t) = λ         ← exponential distribution
                    "memoryless" — past survival gives no information

Increasing hazard:  h(t) increases   ← Weibull with shape > 1
                    "aging" — the longer you survive, the more likely to fail

Decreasing hazard:  h(t) decreases   ← early failure dominant
                    common in infant mortality / software bugs at launch

Bathtub hazard:     h(t) high → low → high
                    early failures + wear-out failures; common in engineering
```

### The Cumulative Hazard Function H(t)

```
H(t) = ∫₀ᵗ h(u) du
```

### The Relationship Between S(t) and h(t)

```
S(t) = exp(-H(t)) = exp(-∫₀ᵗ h(u) du)
h(t) = -d/dt log S(t)
```

These are equivalent representations of the same underlying distribution. Know one, know all.

---

## 17.3 The Kaplan-Meier Estimator

The non-parametric estimator of the survival function. The foundation of all survival analysis.

### Construction

At each observed event time tⱼ:

```
S(tⱼ) = S(tⱼ₋₁) × (1 - dⱼ/nⱼ)

dⱼ = number of events at time tⱼ
nⱼ = number of subjects at risk just before time tⱼ
     (alive and uncensored)
```

Censored observations reduce nⱼ at their censoring time but do not contribute to dⱼ.

### Worked Example

10 patients. Event times († = died, C = censored):

```
Patient  Time  Status
1        2     †
2        3     C
3        4     †
4        5     †
5        6     C
6        7     †
7        8     C
8        9     †
9        10    C
10       12    †
```

Kaplan-Meier calculation:

| Time | Events (d) | At risk (n) | 1 - d/n | S(t) |
|---|---|---|---|---|
| Start | — | 10 | — | 1.000 |
| t=2 | 1 | 10 | 9/10 | 0.900 |
| t=4 | 1 | 8* | 7/8 | 0.788 |
| t=5 | 1 | 7 | 6/7 | 0.675 |
| t=7 | 1 | 5* | 4/5 | 0.540 |
| t=9 | 1 | 3* | 2/3 | 0.360 |
| t=12 | 1 | 1* | 0/1 | 0.000 |

*n reduced at censored times (t=3: patient 2, t=6: patient 5, t=8: patient 7, t=10: patient 9)

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations=[2,3,4,5,6,7,8,9,10,12],
        event_observed=[1,0,1,1,0,1,0,1,0,1])

kmf.plot_survival_function()
print(kmf.median_survival_time_)  # Time at which S(t) = 0.5
```

### Log-Rank Test

Compares survival curves between two groups. Tests H₀: no difference in survival distributions.

```python
from lifelines.statistics import logrank_test

results = logrank_test(
    durations_A, durations_B,
    event_observed_A=events_A,
    event_observed_B=events_B
)
print(f"p-value: {results.p_value:.4f}")
print(f"test statistic: {results.test_statistic:.4f}")
```

The log-rank test weights all time points equally. Weighted variants (Gehan-Breslow, Tarone-Ware) emphasize early or late differences.

---

## 17.4 The Cox Proportional Hazards Model

The most widely used survival model. Combines the non-parametric baseline hazard with a parametric effect of covariates.

### Model Specification

```
h(t | x) = h₀(t) × exp(β₁x₁ + β₂x₂ + ... + βₚxₚ)
          = h₀(t) × exp(xᵀβ)

h₀(t) = baseline hazard (non-parametric; left unspecified)
exp(xᵀβ) = hazard ratio relative to baseline
```

**Key assumption — Proportional Hazards:** The ratio of hazards between any two individuals is constant over time.

```
h(t | xᵢ) / h(t | xⱼ) = exp((xᵢ - xⱼ)ᵀβ)
```

The hazard ratio does not depend on t. This is a strong assumption that must be tested.

### Hazard Ratios

The exponentiated coefficients are hazard ratios:

```
HR = exp(β)

HR = 1.0: covariate has no effect on hazard
HR > 1.0: covariate increases hazard (shorter survival)
HR < 1.0: covariate decreases hazard (longer survival)

Example:
  β_age = 0.05  →  HR = exp(0.05) = 1.051
  Each additional year of age increases hazard by 5.1%
```

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='survival_time', event_col='event_occurred')
cph.print_summary()
# Shows coefficients, hazard ratios, 95% CI, p-values
```

### Testing the Proportional Hazards Assumption

**Schoenfeld residuals test:**

```python
from lifelines.statistics import proportional_hazard_test

results = proportional_hazard_test(cph, df, time_transform='rank')
print(results.summary)
# p-value < 0.05 for a covariate: PH assumption violated for that covariate
```

**Visual test:** Plot Schoenfeld residuals against time. Should be flat (no trend) if PH assumption holds.

If assumption is violated: use time-varying coefficients, stratified Cox model, or accelerated failure time models.

---

## 17.5 The Concordance Index (C-Index)

The primary evaluation metric for survival models. The survival analog of AUC-ROC.

### Definition

```
C-Index = P(risk_score(i) > risk_score(j) | Tᵢ < Tⱼ, event_i observed)
```

Among all **comparable pairs** (where we know which subject failed first), what fraction does the model correctly rank?

A comparable pair requires:
1. The earlier-failing subject actually had an event (not censored)
2. The earlier-failing subject failed before the other was censored

```
C-Index = 1.0   → Perfect discrimination (always ranks higher-risk first)
C-Index = 0.5   → Random (coin flip)
C-Index = 0.0   → Perfect inverse ranking
```

### Computing C-Index

```python
from lifelines.utils import concordance_index

c_index = concordance_index(
    event_times=y_test['time'],
    predicted_scores=-predicted_risk,   # Negative because higher risk → shorter time
    event_observed=y_test['event']
)
print(f"C-Index: {c_index:.4f}")
```

Or using scikit-survival:

```python
from sksurv.metrics import concordance_index_censored

c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
    event_indicator=y_test['event'].astype(bool),
    event_time=y_test['time'],
    estimate=predicted_risk
)
```

### C-Index vs. AUC-ROC

| Property | C-Index | AUC-ROC |
|---|---|---|
| Handles censoring | Yes | No |
| Measures | Ranking discrimination | Binary classification discrimination |
| Baseline | 0.5 | 0.5 |
| At fixed time t | Time-dependent AUC (see below) | Standard AUC |
| Most common in | Survival analysis | Classification |

### Time-Dependent AUC

C-Index averages discrimination over all time points. Sometimes you want discrimination at a specific time t:

```
AUC(t) = P(risk_score(i) > risk_score(j) | Tᵢ ≤ t, Tⱼ > t)

"At time t, does the model correctly rank cases vs. controls?"
```

```python
from sksurv.metrics import cumulative_dynamic_auc

times = [365, 730, 1095]   # 1, 2, 3 years
auc_scores, mean_auc = cumulative_dynamic_auc(y_train, y_test, predicted_risk, times)

for t, auc in zip(times, auc_scores):
    print(f"AUC at {t} days: {auc:.3f}")
```

Time-dependent AUC is especially important when the model's discrimination ability changes over time (e.g., a biomarker predicts early events well but not late ones).

---

## 17.6 Brier Score Over Time

From Chapter 11, Brier Score measures calibration for probability predictions. For survival, we extend it to measure calibration of survival probability estimates over time.

### Integrated Brier Score (IBS)

At each time point t, compute the Brier Score for S(t):

```
BS(t) = (1/n) × Σᵢ [(S(t|xᵢ) - 𝟙[Tᵢ > t])² × IPCW_weight(i, t)]

Where IPCW = Inverse Probability of Censoring Weighting
             (corrects for the fact that censored subjects are not observed)
```

**Integrate over time:**

```
IBS = (1/τ) × ∫₀^τ BS(t) dt
```

Where τ is the maximum follow-up time.

**Interpretation:**
- IBS = 0: perfect calibration
- IBS = 0.25: random model (analogous to Brier score baseline of p(1-p) ≈ 0.25 for balanced binary)
- Lower is better

```python
from sksurv.metrics import integrated_brier_score

times = np.arange(30, 730, 30)   # Monthly intervals, up to 2 years
ibs = integrated_brier_score(y_train, y_test, survival_probabilities, times)
print(f"Integrated Brier Score: {ibs:.4f}")
```

### Brier Score at Specific Time Points

For clinical decision points (e.g., "what is the 5-year survival probability?"):

```python
from sksurv.metrics import brier_score

times_of_interest = [365, 1825]   # 1 year, 5 years
scores = brier_score(y_train, y_test, survival_probabilities, times_of_interest)
for t, bs in zip(times_of_interest, scores[1]):
    print(f"Brier Score at {t} days: {bs:.4f}")
```

---

## 17.7 Calibration in Survival Models

A well-calibrated survival model's predicted survival probabilities match observed survival rates.

### Calibration Curve for Survival

At a fixed time point t:
1. Divide patients into K risk groups (deciles of predicted survival probability)
2. For each group: plot mean predicted S(t) vs. observed KM survival rate

```
Perfectly calibrated: points on diagonal
Overoptimistic model: points below diagonal (predicts better survival than observed)
Overpessimistic model: points above diagonal
```

```python
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def survival_calibration_plot(y_true, predicted_survival, time_point, n_groups=10):
    predictions_at_t = predicted_survival[:, time_index]  # Survival prob at time t
    groups = pd.qcut(predictions_at_t, q=n_groups, labels=False)

    observed_survival = []
    predicted_mean = []

    for g in range(n_groups):
        mask = groups == g
        kmf = KaplanMeierFitter()
        kmf.fit(y_true['time'][mask], y_true['event'][mask])
        observed_survival.append(kmf.survival_function_at_times([time_point]).values[0])
        predicted_mean.append(predictions_at_t[mask].mean())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(predicted_mean, observed_survival)
    plt.xlabel('Predicted Survival Probability')
    plt.ylabel('Observed Survival (Kaplan-Meier)')
    plt.title(f'Calibration at {time_point} days')
    plt.show()
```

### D-Calibration

Tests whether the predicted survival probabilities are globally well-calibrated across all time points using a chi-squared test:

```python
from sksurv.calibration import integrated_calibration_index

ici = integrated_calibration_index(
    survival_train=y_train,
    survival_test=y_test,
    times=times,
    estimate=predicted_survival
)
print(f"Integrated Calibration Index: {ici:.4f}")
# Lower = better calibrated
```

---

## 17.8 Evaluating Competing Risks

In many real applications, subjects can experience multiple types of events, and one event prevents the others from occurring.

```
Example: Cancer patients
  Event 1: Death from cancer
  Event 2: Death from other causes  ← "competing risk"

  If patient dies from other causes, they can no longer die from cancer.
  Standard survival analysis overestimates cancer mortality risk by ignoring this.
```

### Cause-Specific Hazard

Hazard for event type k, treating other events as censored:

```
hₖ(t) = lim P(t ≤ T < t+Δt, event type k | T ≥ t) / Δt
```

### Cumulative Incidence Function (CIF)

Probability of experiencing event type k by time t, accounting for competing risks:

```
CIF_k(t) = P(T ≤ t, event type = k)
```

Note: CIF_1(t) + CIF_2(t) + ... = 1 - S(t), where S(t) accounts for all events.

**Important:** The sum of cause-specific survival functions (1 - CIF_k) is NOT the overall survival function. This is a common error.

```python
from lifelines import AalenJohansenFitter

ajf = AalenJohansenFitter()
ajf.fit(durations, event_observed, event_col='event_type')
ajf.plot_cumulative_density()   # Plots CIF for each event type
```

### Evaluation for Competing Risks

**Cause-specific C-Index:** C-Index computed for each event type separately.

**Time-dependent AUC for competing risks:**

```python
from sksurv.metrics import cumulative_dynamic_auc

# For competing risk k, use cause-specific survival probability
auc_k, mean_auc_k = cumulative_dynamic_auc(
    y_train_k, y_test_k, risk_scores_k, times
)
```

---

## 17.9 Machine Learning for Survival Analysis

Beyond Cox regression, modern ML methods handle non-linear relationships and high-dimensional data.

### Random Survival Forests

Extends random forests to survival data. Each tree is built using the log-rank test for splitting:

```python
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, random_state=42)
rsf.fit(X_train, y_train)

# Predict risk score (higher = higher hazard = shorter survival)
risk_scores = rsf.predict(X_test)
c_index = concordance_index_censored(y_test['event'], y_test['time'], risk_scores)
```

### DeepSurv and Neural Survival Models

Neural network extension of Cox model:

```python
# DeepSurv architecture
import torch.nn as nn

class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output: log hazard ratio
        )

    def forward(self, x):
        return self.net(x)

# Loss: negative partial log-likelihood (Cox loss)
def cox_loss(log_hazard, times, events):
    # Standard Cox partial likelihood
    ...
```

### DRSA / Transformer-Based Survival Models

Recent approaches use attention mechanisms to model temporal dependencies in survival data, particularly for time-series covariates (longitudinal EHR data).

### Evaluation Comparison Across ML Survival Models

```python
models = {
    'Cox PH':          CoxPHFitter(),
    'RSF':             RandomSurvivalForest(n_estimators=200),
    'DeepSurv':        DeepSurvModel(input_dim=X.shape[1]),
    'Gradient Boosting': GradientBoostingSurvivalAnalysis()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    risk = model.predict(X_test)

    results[name] = {
        'c_index': concordance_index_censored(y_test['event'], y_test['time'], risk)[0],
        'ibs':     integrated_brier_score(y_train, y_test, model.predict_survival_function(X_test), times),
    }

pd.DataFrame(results).T.sort_values('c_index', ascending=False)
```

---

## 17.10 The Full Survival Analysis Evaluation Pipeline

```
1. Inspect censoring
   ├── Censoring rate (should not be > 80%)
   ├── Is censoring random (non-informative)?
   └── Kaplan-Meier of censoring distribution

2. Fit survival models
   ├── Kaplan-Meier (non-parametric baseline)
   ├── Cox PH (interpretable)
   └── ML methods (RSF, DeepSurv)

3. Check Cox PH assumption
   ├── Schoenfeld residuals test
   └── Log-log survival plots

4. Evaluate discrimination
   ├── C-Index (overall)
   ├── Time-dependent AUC (at clinically relevant times)
   └── Log-rank test (group comparison)

5. Evaluate calibration
   ├── Calibration curve at key time points
   ├── Integrated Brier Score (overall)
   └── Brier Score at specific time points

6. Handle competing risks (if applicable)
   ├── Cumulative Incidence Function (CIF)
   └── Cause-specific C-Index per event type

7. Clinical / business validation
   ├── Does stratification by risk score align with clinical knowledge?
   ├── Do high-risk predictions correspond to known risk factors?
   └── Is the model useful at the decision threshold?
```

---

## 17.11 Worked Example: Customer Churn Survival Model

**Problem:** Predict when SaaS customers will churn. 50,000 customers, 23% churned in observation window.

```
Data structure:
  customer_id, months_since_signup, churned (1/0), plan, usage_score, support_tickets

Censoring: 77% of customers haven't churned yet → right-censored at observation end

Step 1: Kaplan-Meier baseline
  Median survival time: 18.3 months
  12-month survival: 82.4%
  24-month survival: 63.1%

Step 2: Cox PH model
  Covariates: plan_type, usage_score, support_tickets, company_size

  Hazard Ratios:
    usage_score:     HR = 0.72  (high usage → 28% lower churn hazard) ✓ (expected)
    support_tickets: HR = 1.34  (each extra ticket → 34% higher churn hazard) ✓
    plan=enterprise: HR = 0.51  (enterprise → 49% lower churn hazard)  ✓

  PH test: all p > 0.05 ✓ (proportional hazards assumption holds)

Step 3: Evaluation
  C-Index:           0.74   (good discrimination)
  IBS:               0.092  (good calibration; random ≈ 0.179)
  AUC at 12 months:  0.79
  AUC at 24 months:  0.76   (slightly lower; discrimination weakens over time)

Step 4: Random Survival Forest
  C-Index:           0.81   (+9.5% vs Cox) ← captures non-linear interactions
  IBS:               0.078  (+15% vs Cox)
  AUC at 12 months:  0.84

Step 5: Calibration
  Calibration curve at 12 months: slight overestimation for low-risk customers
  Applied isotonic regression calibration → IBS improved to 0.072

Step 6: Business validation
  Top decile risk score: 68% churn within 6 months → 3× base rate ✓
  Retention intervention ROI: 
    Cost per intervention: $200
    Revenue saved per prevented churn: $2,400
    C-Index improvement from Cox → RSF: +0.07
    Estimated additional churns caught per 1000 customers: 23
    Net revenue saved: 23 × ($2,400 - $200) = $50,600 per 1,000 customers
```

**Lesson:** C-Index and IBS together reveal that the RSF is both better at ranking (discrimination) and better calibrated. The business value calculation translates metric improvement into revenue — the language stakeholders understand.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Censoring | Incomplete observation; ignoring it causes survivorship bias |
| Survival function S(t) | Probability of surviving past time t |
| Hazard function h(t) | Instantaneous event rate given survival to t |
| Kaplan-Meier | Non-parametric survival curve; handles censoring correctly |
| Cox PH | Semi-parametric regression; interpretable hazard ratios |
| Proportional hazards | Key Cox assumption; must be tested |
| C-Index | Survival AUC; fraction of correctly ranked comparable pairs |
| Time-dependent AUC | Discrimination at specific time points |
| Integrated Brier Score | Calibration integrated over time; lower is better |
| Competing risks | Multiple event types; use CIF, not KM |
| Random Survival Forest | Non-linear survival model; often beats Cox |

---

## Further Reading

- Cox — *Regression Models and Life-Tables* (JRSS-B, 1972) — the original Cox model
- Harrell et al. — *Evaluating the Yield of Medical Tests* (JAMA, 1982) — C-Index
- Graf et al. — *Assessment and Comparison of Prognostic Classification Schemes* (1999) — Brier Score for survival
- Ishwaran et al. — *Random Survival Forests* (Annals of Applied Statistics, 2008)
- Katzman et al. — *DeepSurv: Personalized Treatment Recommender via Cox Model* (BMC Med Research, 2018)
- lifelines documentation — best practical Python survival analysis resource
- scikit-survival documentation — ML-focused survival analysis toolkit

---

*Next: Chapter 18 — Statistical Significance in Experiments*
