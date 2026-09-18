# Chapter 16: Fairness & Bias Metrics

> *"A model that is accurate on average can be deeply unfair to individuals. Fairness is not a constraint you add after building a model — it is a design requirement you build in from the start. And you cannot build it in unless you can measure it."*

---

## 16.1 Why Fairness Metrics Matter

ML systems make consequential decisions at scale: who gets a loan, who gets a job interview, who gets bail, whose medical symptoms are taken seriously. When these systems are biased, they don't just make individual errors — they systematically disadvantage entire groups, often groups that are already marginalized.

```
A recidivism prediction model:
  Accuracy for white defendants:   67%
  Accuracy for Black defendants:   67%    ← same accuracy

  False positive rate for white defendants: 23%
  False positive rate for Black defendants: 44%  ← 2× higher

  Conclusion: equal accuracy, deeply unequal false alarm rates.
  Black defendants are twice as likely to be incorrectly flagged as high risk.
```

This is the COMPAS case (ProPublica, 2016). Equal accuracy is not equal treatment.

### The Stakes of Unfair ML

| Domain | Unfair model | Consequence |
|---|---|---|
| Credit scoring | Disadvantages minority applicants | Denied access to capital |
| Hiring | Penalizes women's resumes | Perpetuates wage and opportunity gaps |
| Healthcare | Lower pain scores for Black patients | Undertreated pain, worse outcomes |
| Criminal justice | Higher risk scores for Black defendants | Unjust incarceration |
| Ad targeting | Targets predatory products at vulnerable groups | Financial harm |
| Face recognition | High error rate for dark-skinned women | Misidentification, false arrests |

---

## 16.2 Protected Attributes and Groups

A **protected attribute** is a characteristic that should not be used as a basis for differential treatment: race, gender, age, religion, national origin, disability status, sexual orientation.

```
Protected attributes (legally defined, varies by jurisdiction):
  Race, color, national origin
  Sex, gender identity
  Age (typically ≥ 40)
  Religion
  Disability
  Pregnancy
  Sexual orientation (in many jurisdictions)
```

**Proxy discrimination:** Even if you don't use a protected attribute directly, other features may be highly correlated with it — zip code as a proxy for race, name as a proxy for gender, purchase history as a proxy for religion. Removing the protected attribute from features does not eliminate bias.

```
Example:
  Removed "race" from credit model.
  Kept "zip code," "school attended," "credit history length."
  These correlate strongly with race due to historical redlining and
  unequal access to financial systems.
  Model remains racially biased.
```

---

## 16.3 The Impossibility Theorem

Before diving into metrics, the most important result: **you cannot simultaneously satisfy all fairness criteria.** This is not a limitation of current techniques — it is mathematically provable.

### Chouldechova's Impossibility Result (2017)

For a binary classifier with unequal base rates between groups:

```
You cannot simultaneously achieve:
  1. Equal False Positive Rates (FPR) across groups
  2. Equal False Negative Rates (FNR) across groups
  3. Equal Positive Predictive Value (Precision) across groups

Unless: base rates are equal OR the classifier is perfect.
```

**Implication:** When base rates differ between groups (which they often do due to historical inequity), every fairness criterion trades off against others. Choosing a fairness criterion is a value judgment — a societal and legal decision, not a technical one.

```
COMPAS example:
  Recidivism base rate (white): ~39%
  Recidivism base rate (Black): ~51%   ← differ due to historical inequity

  Cannot simultaneously have:
    Equal FPR across races (Northpointe's claim)
    Equal PPV across races (also true)
    Equal FNR across races (violated)

  ProPublica and Northpointe were BOTH correct — measuring different criteria.
```

This is why fairness conversations require stakeholder involvement. There is no technically correct answer.

---

## 16.4 Group Fairness Metrics

### Demographic Parity (Statistical Parity)

*The model produces positive outcomes at the same rate for all groups.*

```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)

Disparate Impact Ratio = P(Ŷ = 1 | A = minority) / P(Ŷ = 1 | A = majority)

Legal threshold: Disparate Impact Ratio ≥ 0.8 (the "four-fifths rule")
```

**Example:**
```
Loan approval rate for Group A: 62%
Loan approval rate for Group B: 41%

Disparate Impact = 41/62 = 0.66  ← below 0.80 threshold; legally problematic
```

**When to use:** When the positive outcome is a benefit (loan, job, admission) and equal access is the goal.

**Limitation:** Does not account for genuine qualification differences. Enforcing demographic parity can require approving less-qualified applicants from some groups while rejecting more-qualified applicants from others — which may itself be unfair to individuals.

### Equalized Odds

*The model has equal TPR and FPR across groups.*

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)    [Equal TPR]
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)    [Equal FPR]
```

**Interpretation:** Given the true label, the model's decision is independent of group membership. A qualified person from any group has the same chance of being approved. An unqualified person from any group has the same chance of being incorrectly approved.

**When to use:** Binary decisions where both kinds of errors have consequences for individuals (hiring, lending, parole). Generally considered the most defensible group fairness criterion.

### Equal Opportunity

*The model has equal TPR across groups (relaxes the FPR requirement).*

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)    [Equal TPR only]
```

**Interpretation:** Qualified individuals from all groups have equal chances of being correctly identified. Less strict than equalized odds — allows FPR to differ.

**When to use:** When false negatives are the primary concern (e.g., missing qualified candidates in hiring).

### Predictive Parity (Calibration Across Groups)

*Positive predictions are equally accurate across groups.*

```
P(Y = 1 | Ŷ = 1, A = 0) = P(Y = 1 | Ŷ = 1, A = 1)    [Equal PPV]
P(Y = 0 | Ŷ = 0, A = 0) = P(Y = 0 | Ŷ = 0, A = 1)    [Equal NPV]
```

**Interpretation:** When the model says "high risk," the actual risk is the same regardless of group. A credit score of 700 means the same default probability for everyone.

**When to use:** When the downstream system uses predicted probabilities (not just labels) for decisions. Required for actuarially fair insurance, unbiased medical risk scores.

**Note:** As shown in 16.3, this conflicts with equalized odds when base rates differ.

### Summary Table: Group Fairness Criteria

| Criterion | What Is Equalized | Measures | Requires |
|---|---|---|---|
| Demographic Parity | Positive rate | Equal access to outcomes | Same base rates not needed |
| Equal Opportunity | TPR | Qualified people equally served | Same base rates not needed |
| Equalized Odds | TPR + FPR | Equal error rates | Conflicts with PPV parity |
| Predictive Parity | PPV (+ NPV) | Equal prediction accuracy | Conflicts with equalized odds |
| Individual Fairness | Similar inputs → similar outputs | Consistent treatment | Requires similarity metric |

---

## 16.5 Individual Fairness

Group fairness criteria evaluate average outcomes across groups. Individual fairness asks: are similar individuals treated similarly?

```
Individual Fairness:
  If d(xᵢ, xⱼ) ≤ ε  (individuals are similar)
  Then d(f(xᵢ), f(xⱼ)) ≤ δ  (predictions are similar)
```

Where d(·, ·) is a task-specific similarity metric.

**Challenge:** Defining "similar" requires a similarity metric, and this metric encodes value judgments. Two people with identical qualifications but different zip codes — are they similar? What about different years of experience?

**Counterfactual Fairness:** A special case of individual fairness:

```
Model is counterfactually fair if:
  P(Ŷ = 1 | X = x, A = a) = P(Ŷ = 1 | X = x, A = a')

For all values a, a' of the protected attribute.
```

In plain language: if we changed only the person's protected attribute (and the causally downstream features), would the prediction change? If yes, the model is unfair.

---

## 16.6 Calibration Across Groups

As discussed in Chapter 4, calibration means predicted probabilities match observed frequencies. Calibration can vary across demographic groups — a model can be globally well-calibrated but miscalibrated for a subgroup.

```
Overall calibration: ECE = 0.03  ← looks good

Calibration by group:
  Group A: ECE = 0.02  ✓
  Group B: ECE = 0.11  ✗  ← systematically overconfident for Group B
```

**Why this matters:** If a medical risk model overestimates risk for one demographic group, they receive unnecessary interventions. If it underestimates risk for another, they receive insufficient care.

**How to measure:**

```python
def calibration_by_group(y_true, y_prob, groups, n_bins=10):
    results = {}
    for group in np.unique(groups):
        mask = groups == group
        y_g = y_true[mask]
        p_g = y_prob[mask]

        fraction_pos, mean_pred = calibration_curve(y_g, p_g, n_bins=n_bins)
        bin_counts = np.histogram(p_g, bins=n_bins)[0]
        ece = np.sum(bin_counts * np.abs(mean_pred - fraction_pos)) / len(y_g)
        results[group] = {'ece': ece, 'n': mask.sum()}

    return results
```

---

## 16.7 Bias Sources and Measurement

Understanding where bias comes from helps target measurement correctly.

### Pre-existing Bias (Historical Bias)

The world is biased. Training data reflects historical inequities. A model trained on historical hiring decisions will replicate historical discrimination.

```
Example: Amazon's hiring tool (2018)
  Trained on 10 years of hiring decisions.
  Historical hires were predominantly male in tech roles.
  Model learned to penalize resumes mentioning "women's" (as in women's college).
  Systematic gender bias from historical data.
```

**Measurement:** Compare base rates in training data to true population rates. Large disparities signal pre-existing bias.

### Representation Bias

Some groups are underrepresented in training data. The model performs poorly on them because it has seen fewer examples.

```
Face recognition:
  Training data: 77% male, 83% white (Buolamwini & Gebru, 2018)
  Error rate on white males: 0.8%
  Error rate on dark-skinned females: 34.7%
```

**Measurement:** Compute per-group accuracy, F1, or error rate. Large disparities indicate representation bias.

### Measurement Bias

The features or labels are measured with different accuracy across groups.

```
Healthcare example:
  "Healthcare utilization" used as proxy for "healthcare need."
  Groups with less access to healthcare use it less.
  Model underestimates need for underserved populations.
  Result: resources directed away from those who need them most.
```

**Measurement:** Audit label quality across groups. Are labels (e.g., ground truth diagnoses) equally accurate across groups?

### Aggregation Bias

Using a single model for heterogeneous populations when different sub-populations have different relationships between features and outcomes.

```
Example: HbA1c (diabetes biomarker) levels
  Mean HbA1c levels differ across racial groups for the same disease severity.
  A single threshold used across all groups mismeasures disease severity
  for some groups.
```

**Measurement:** Fit separate models or subgroup-specific calibrations; compare to universal model.

---

## 16.8 Intersectionality

Bias compounds across multiple protected attributes. A model can be fair along each individual dimension while being deeply unfair at their intersection.

```
Face recognition audit:

  Gender-only view:
    Males: 0.8% error
    Females: 3.2% error   ← small gap

  Race-only view:
    Light skin: 0.9% error
    Dark skin: 5.1% error  ← moderate gap

  Intersectional view:
    Light-skinned males: 0.3% error
    Dark-skinned females: 34.7% error  ← 115× gap

Race-only and gender-only analyses completely missed the worst failure.
```

**Measurement:** Always disaggregate metrics by intersectional groups, not just individual protected attributes. Key intersections:
- Race × Gender
- Age × Disability
- Race × Socioeconomic status
- Gender × Geographic region

**Challenge:** Intersectional groups can be small. Standard metrics have high variance on small samples. Use confidence intervals; flag groups where sample size is insufficient for reliable estimates.

---

## 16.9 Fairness Metrics in Practice

### Computing Group Fairness Metrics

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def fairness_metrics(y_true, y_pred, y_prob, sensitive_attr):
    """
    Compute group fairness metrics across all groups.
    y_true: true labels
    y_pred: predicted labels (binary)
    y_prob: predicted probabilities
    sensitive_attr: group membership array
    """
    groups = np.unique(sensitive_attr)
    results = {}

    for g in groups:
        mask = sensitive_attr == g
        y_g, yh_g = y_true[mask], y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(y_g, yh_g).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0   # Recall / Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0   # Fall-out
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0   # Precision
        pos_rate = yh_g.mean()                           # Positive prediction rate

        results[g] = {
            'n': mask.sum(),
            'positive_rate': pos_rate,
            'tpr': tpr,
            'fpr': fpr,
            'ppv': ppv,
            'base_rate': y_g.mean()
        }

    # Compute pairwise ratios (reference = first group)
    ref = groups[0]
    for g in groups[1:]:
        results[f'demographic_parity_ratio_{g}'] = (
            results[g]['positive_rate'] / results[ref]['positive_rate']
        )
        results[f'equal_opportunity_ratio_{g}'] = (
            results[g]['tpr'] / results[ref]['tpr']
        )
        results[f'equalized_odds_tpr_diff_{g}'] = (
            abs(results[g]['tpr'] - results[ref]['tpr'])
        )
        results[f'equalized_odds_fpr_diff_{g}'] = (
            abs(results[g]['fpr'] - results[ref]['fpr'])
        )

    return results
```

### Fairness Libraries

| Library | Features |
|---|---|
| **Fairlearn** (Microsoft) | Fairness metrics + mitigation algorithms (reweighting, threshold optimization) |
| **AI Fairness 360** (IBM) | 70+ fairness metrics; 10+ bias mitigation algorithms |
| **What-If Tool** (Google) | Visual exploration of fairness across groups |
| **Aequitas** (UChicago) | Audit tool; generates bias reports across groups |
| **Themis-ML** | Fairness-aware ML algorithms |

---

## 16.10 Bias Mitigation Strategies and Their Trade-offs

### Pre-processing: Data Intervention

**Reweighting:** Assign higher weights to underrepresented (group, label) combinations.

**Resampling:** Oversample minority group + positive label combinations.

**Disparate impact remover:** Transforms features to reduce correlation with protected attribute while preserving rank-ordering within groups.

### In-processing: Constrained Optimization

Add fairness constraints directly to the training objective:

```
Minimize: Loss(y, ŷ)
Subject to: |TPR_groupA - TPR_groupB| ≤ ε       [Equal opportunity]
            |FPR_groupA - FPR_groupB| ≤ ε       [Equalized odds]
```

**Exponentiated gradient** (Fairlearn): Converts constrained optimization to a sequence of cost-sensitive classification problems. Produces a Pareto frontier of accuracy-fairness trade-offs.

### Post-processing: Threshold Adjustment

After training, set **group-specific thresholds** to equalize the chosen fairness criterion:

```python
from fairlearn.postprocessing import ThresholdOptimizer

postprocess = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",  # or "demographic_parity"
    objective="accuracy_score"
)
postprocess.fit(X_train, y_train, sensitive_features=A_train)
y_pred = postprocess.predict(X_test, sensitive_features=A_test)
```

**Trade-off:** Post-processing cannot improve beyond the Pareto frontier of the base model. If the model has low accuracy for a group, threshold adjustment can equalize error rates but at the cost of reduced accuracy for other groups.

### The Accuracy-Fairness Trade-off

There is almost always a trade-off between overall accuracy and any fairness criterion. The Pareto frontier characterizes this trade-off:

```
Pareto Frontier: Accuracy-Fairness

Accuracy
  |
  |  ×  ← Current model (high accuracy, low fairness)
  |
  |     × ← Constrained model (moderate accuracy, high fairness)
  |
  |       × ← Over-constrained (low accuracy, perfect fairness)
  |
  +----------------------------------
                                  Fairness

Goal: move along the frontier to the socially acceptable trade-off point.
```

The right point on this frontier is a **value judgment** that must involve stakeholders, legal counsel, and affected communities. It cannot be determined purely by technical optimization.

---

## 16.11 Fairness Evaluation Checklist

Before deploying any model that makes consequential decisions about people:

**Data audit:**
- [ ] What are the base rates for each group in the training data?
- [ ] Do base rates in training data reflect true population rates?
- [ ] Are labels measured equally accurately across groups?
- [ ] Are there proxies for protected attributes in the feature set?

**Model audit:**
- [ ] What is the overall accuracy / F1 / AUC?
- [ ] What are per-group accuracy / F1 / AUC?
- [ ] What is the disparate impact ratio?
- [ ] Are TPR and FPR equal across groups (equalized odds)?
- [ ] Is PPV equal across groups (predictive parity)?
- [ ] Is calibration equal across groups?
- [ ] Have intersectional groups been audited?

**Deployment context:**
- [ ] What is the consequential decision being made?
- [ ] Which fairness criterion is most appropriate given the domain?
- [ ] Have affected communities been consulted?
- [ ] Is there a human appeals process for automated decisions?
- [ ] How will the model be monitored for fairness drift post-deployment?

---

## 16.12 Worked Example: Loan Application Scoring

**Task:** Binary classifier for loan approval. Applicants are from Groups A and B with different historical credit access.

```
Dataset: 100,000 applications
  Group A: 60,000 applicants, base rate 35% (would repay)
  Group B: 40,000 applicants, base rate 48% (would repay)

Initial model (accuracy-optimized):
  Overall accuracy: 78.3%

  Group A:
    Approval rate: 31.2%
    TPR: 0.72   (correctly identifies repayers)
    FPR: 0.19   (incorrectly approves non-repayers)
    PPV: 0.81   (when approved, 81% actually repay)

  Group B:
    Approval rate: 51.4%
    TPR: 0.81
    FPR: 0.28
    PPV: 0.76

Disparate Impact: 31.2/51.4 = 0.61  ← below 0.80 threshold
Equalized Odds TPR gap: |0.72 - 0.81| = 0.09
Equalized Odds FPR gap: |0.19 - 0.28| = 0.09
PPV gap: |0.81 - 0.76| = 0.05
```

**Diagnosis:**
- Group A has lower approval rates (31.2% vs 51.4%) — disparate impact
- Group A's true repayers are less likely to be approved (TPR 0.72 vs 0.81) — equal opportunity violated
- Both groups have different FPR — equalized odds violated
- PPV gap is smaller but present

**Mitigation: Post-processing with ThresholdOptimizer (equalized odds constraint):**

```
After threshold optimization:
  Group A threshold: 0.38  (lowered)
  Group B threshold: 0.47  (raised)

  Group A:
    Approval rate: 38.1%
    TPR: 0.78  ← improved
    FPR: 0.24
    PPV: 0.76

  Group B:
    Approval rate: 46.2%
    TPR: 0.78  ← matched
    FPR: 0.24  ← matched
    PPV: 0.76  ← matched

Disparate Impact: 38.1/46.2 = 0.82  ✓ (above 0.80)
Equalized Odds TPR gap: 0.00  ✓
Equalized Odds FPR gap: 0.00  ✓
Overall accuracy: 76.1%  (−2.2 pp; accepted trade-off)
```

**Key outcomes:**
- Equalized odds achieved at cost of 2.2% overall accuracy
- Group A approval rate increased from 31.2% to 38.1%
- PPV decreased slightly for Group A, increased slightly for Group B
- Predictive parity not perfectly achieved (expected under different base rates)

**Lesson:** Fairness interventions involve trade-offs that must be decided by stakeholders, not optimizers. The technical solution shows what's possible; the value judgment determines which solution to deploy.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Protected attributes | Characteristics that should not drive differential treatment |
| Proxy discrimination | Removing protected attributes doesn't remove bias |
| Impossibility theorem | Cannot simultaneously satisfy all fairness criteria |
| Demographic parity | Equal positive rates across groups |
| Equal opportunity | Equal TPR across groups |
| Equalized odds | Equal TPR + FPR across groups |
| Predictive parity | Equal PPV across groups; conflicts with equalized odds |
| Individual fairness | Similar individuals treated similarly |
| Calibration across groups | Probabilities must be accurate for all groups |
| Intersectionality | Bias compounds across multiple protected attributes |
| Accuracy-fairness trade-off | There is always a trade-off; where to sit is a values decision |

---

## Further Reading

- Chouldechova — *Fair Prediction with Disparate Impact* (2017) — impossibility theorem
- Hardt, Price, Srebro — *Equality of Opportunity in Supervised Learning* (NeurIPS 2016) — equalized odds
- Buolamwini & Gebru — *Gender Shades* (FAccT 2018) — intersectional bias in face recognition
- Obermeyer et al. — *Dissecting Racial Bias in an Algorithm Used to Manage Health* (Science 2019)
- Barocas, Hardt, Narayanan — *Fairness and Machine Learning* (fairmlbook.org) — the textbook
- Mehrabi et al. — *A Survey on Bias and Fairness in ML* (ACM Computing Surveys, 2021)

---

*Next: Chapter 17 — Survival & Time-to-Event Metrics*
