# Bias-Variance in XAI — Complete Guide

> Why importance and explainability estimates are wrong, how wrong, and for what reasons. The Rashomon set, overfitting effects, train vs test, sample size — all with numbers.

---

## Table of Contents

1. [The Meta-Problem: XAI Methods Have Their Own Errors](#1-the-meta-problem-xai-methods-have-their-own-errors)
2. [The Bias-Variance Framework Applied to XAI](#2-the-bias-variance-framework-applied-to-xai)
3. [The Rashomon Problem](#3-the-rashomon-problem)
   - 3.1 What It Is
   - 3.2 Numerical Example
   - 3.3 Implications for Feature Importance
   - 3.4 What to Do About It
4. [Train vs Test: The Overfit Bias](#4-train-vs-test-the-overfit-bias)
   - 4.1 What Happens When You Use the Training Set
   - 4.2 Numerical Example
   - 4.3 The Diagnostic Value of the Gap
5. [Underfitting — When All Importances Collapse](#5-underfitting--when-all-importances-collapse)
6. [Sample Size Effects](#6-sample-size-effects)
7. [Bias-Variance Table — Every Method](#7-bias-variance-table--every-method)
8. [Method-Specific Deep Dives](#8-method-specific-deep-dives)
   - 8.1 Permutation Importance
   - 8.2 MDI / Gini
   - 8.3 SHAP (TreeSHAP)
   - 8.4 SHAP (KernelSHAP)
   - 8.5 LIME
   - 8.6 PDP / ALE
   - 8.7 Surrogate Models
9. [The Faithfulness Problem — Explaining an Overfit Model](#9-the-faithfulness-problem--explaining-an-overfit-model)
10. [Confidence Intervals for Importance Estimates](#10-confidence-intervals-for-importance-estimates)
11. [Interview Q&A](#11-interview-qa)
12. [Summary Card](#12-summary-card)

---

## 1. The Meta-Problem: XAI Methods Have Their Own Errors

There is a common assumption hidden in most XAI tutorials: that the importance or explanation returned by a method is *correct*. It is not.

Every XAI method is itself an estimator — it takes data and a model and outputs an importance number or explanation. Like any estimator, it has:

- **Bias** — systematic error that doesn't go away with more data
- **Variance** — random error that changes between runs

When you read a feature importance value of 0.23, you are reading a noisy, potentially biased estimate. The true importance could be quite different.

This guide covers: what causes bias, what causes variance, how large these errors are, and what you can do about them.

---

## 2. The Bias-Variance Framework Applied to XAI

For an importance estimator Î(j) of the true importance I(j):

```
Expected Squared Error = Bias²(Î) + Variance(Î)

Bias(Î)    = E[Î(j)] − I(j)       systematic deviation from truth
Variance(Î) = E[(Î(j) − E[Î(j)])²]  spread across random realisations
```

**Sources differ by method:**

```
Bias sources:
  ├── Using training set instead of test set     (overfitting bias)
  ├── Correlated features                        (substitute / absorption bias)
  ├── Wrong expectation assumption               (marginal vs conditional SHAP)
  ├── Linear approximation error                 (LIME, surrogate models)
  └── Extrapolation into unrealistic regions     (PDP, marginal SHAP)

Variance sources:
  ├── Random permutation shuffles                (permutation importance)
  ├── Random perturbation samples                (LIME, KernelSHAP)
  ├── Bootstrap / feature subsampling            (MDI in RF)
  └── Small dataset                             (all methods)
```

**The key insight:** Bias and variance have very different remedies.

```
To reduce variance:   more repeats, more samples, more trees
To reduce bias:       use test set, fix correlation handling, use better method
```

Increasing n_repeats for permutation importance reduces variance but does nothing for the bias that comes from using the training set.

---

## 3. The Rashomon Problem

### 3.1 What It Is

Named after Kurosawa's 1950 film where four witnesses describe the same event differently, each truthful from their own perspective.

In ML: many different models can achieve the same (or nearly the same) predictive performance on a dataset. This set of equally-good models is called the **Rashomon set**.

```
Rashomon set = {all models f : test_metric(f) ≥ best_metric − ε}
```

The problem: different models in the Rashomon set can attribute very different importances to the same features, even though they perform identically.

### 3.2 Numerical Example

**Dataset:** predict income from [education, zip_code, age, gender, job_type].

Five models, all achieving test AUC ≈ 0.88:

```
Model        education  zip_code  age   gender  job_type
──────────────────────────────────────────────────────────
RF           0.38       0.25      0.20   0.08    0.09
GBM          0.29       0.35      0.18   0.06    0.12
Logistic     0.45       0.20      0.22   0.05    0.08
SVM          0.31       0.31      0.19   0.09    0.10
Neural Net   0.22       0.40      0.21   0.07    0.10

Std across models:
             0.09       0.08      0.01   0.02    0.01
```

Observations:
- **age, gender, job_type**: Low std → all models agree → robust conclusion
- **education, zip_code**: High std → models disagree substantially → unreliable conclusion

For education: RF says 0.38, Neural Net says 0.22 — a 73% relative difference, with identical test performance.

**This is not a failure of any model.** It is an inherent property of the data — when education and zip_code are correlated (they often are), the information can be captured through either feature, and different model architectures learn to use different combinations.

### 3.3 Implications for Feature Importance

The Rashomon problem means:

**Feature importance is a property of the model, not the data.**

When you report "education has importance 0.38," you are reporting what your specific model does with education. A different equally-valid model might give education importance 0.22.

This has practical consequences:

```
1. Don't make strong causal claims based on one model's importance.
   "Education drives income" may just mean "this model uses education."
   Another equally accurate model might attribute more to zip_code.

2. Don't use one model's importance for feature selection without checking
   robustness across model types.

3. Be especially cautious for protected attributes (gender, race, age).
   One model may show gender importance = 0.08 (the model "seems fair"),
   while an equally accurate model shows gender importance = 0.04.
   The apparent fairness is model-dependent, not a data property.
```

### 3.4 What to Do About It

**Report importance ranges, not point estimates:**

```
Compute importance across multiple model types:
  [RF, GBM, Logistic, SVM] all at equal performance

For each feature, report:
  - Mean importance across models
  - Std importance across models
  - Min and Max

Features with low std → robust, trustworthy
Features with high std → model-dependent, Rashomon effect present
```

**Focus on consensus features:**

Features that rank highly across all model types represent genuinely robust signals in the data. Features that rank highly in only some models are picking up model-specific correlations.

**For fairness auditing specifically:**

The Rashomon set matters most here. If protected attribute A has high importance in some models and low in others (all equal performance), the fairness conclusion depends entirely on which model you happen to choose. This is a serious problem for regulatory compliance.

---

## 4. Train vs Test: The Overfit Bias

### 4.1 What Happens When You Use the Training Set

An overfit model has memorised patterns in training data — including noise. On the training set, the model correctly predicts even the noise patterns. When you compute importance on the training set:

- Features that captured real patterns: appear important
- Features that captured noise (memorised training labels): also appear important
- Random noise features that happened to correlate with training labels: appear important

On the test set, noise features don't generalise. Permuting them has no effect on test performance, giving them near-zero test-set importance.

**The training-set importance is biased upward for noise features** — they get credit for memorised patterns that don't exist in the real world.

### 4.2 Numerical Example

**Setup:** Classification problem. N=200 training samples, 50 test samples.

Features: age (real signal), income (real signal), random_noise (pure noise).

Random Forest with max_depth=None (fully grown, severely overfit):

```
Performance:
  Train accuracy: 0.98   (near-perfect — memorised training data)
  Test accuracy:  0.73   (large gap → severe overfitting)

Permutation importance on TRAINING SET:
  age:           0.41
  income:        0.38
  random_noise:  0.19  ← memorised noise → appears important on train

Permutation importance on TEST SET:
  age:           0.16
  income:        0.14
  random_noise:  0.01  ← noise correctly identified as unimportant
```

The training-set importance for random_noise (0.19) is 19× its true importance (0.01).

**For SHAP values specifically:**

```
SHAP (training set) for a training sample:
  age SHAP:    +0.18
  income SHAP: +0.15
  noise SHAP:  +0.12  ← model genuinely uses noise on training data → SHAP is correct
                          but this is describing memorisation, not generalisation

SHAP (test set) for a test sample:
  age SHAP:    +0.14
  income SHAP: +0.12
  noise SHAP:  +0.01  ← noise correctly near zero on new data
```

SHAP is not wrong — it faithfully describes what the model does. The problem is the model itself: it memorised noise. SHAP on training data describes memorisation; SHAP on test data describes generalisation.

### 4.3 The Diagnostic Value of the Gap

Comparing train-set and test-set importance is a **diagnostic tool for overfitting**:

```
Define:  ΔImp(j) = Imp_train(j) − Imp_test(j)

ΔImp >> 0:  Model is overfitting feature j
            Feature j was memorised — spurious training correlations
            Action: investigate this feature; consider regularisation

ΔImp ≈ 0:  Feature j generalises well
            Same signal in training and test

ΔImp << 0: Rare — feature j has a training-specific noise that accidentally
           made it look less important on training data
```

**Practical application:**

```
Feature importance table with both train and test:

Feature    Train imp   Test imp   Delta    Diagnosis
──────────────────────────────────────────────────────────────────
age          0.41       0.16      +0.25   ← Some overfitting
income       0.38       0.14      +0.24   ← Some overfitting
noise        0.19       0.01      +0.18   ← SEVERE overfitting to this feature
zip_code     0.02       0.09      −0.07   ← This feature generalises better
                                            than training suggests
```

The fact that noise has ΔImp=0.18 is a red flag: the model is specifically overfitting to the noise column. Regularise and re-evaluate.

---

## 5. Underfitting — When All Importances Collapse

The flip side of overfitting. An underfit model (too simple, or too heavily regularised) makes predictions that are barely influenced by any feature. It predicts the mean (or majority class) for most inputs.

**Effect on importance:**

```
A severely underfit model:
  Permuting any feature → barely changes predictions → all importances ≈ 0
  SHAP values for all features ≈ 0 (model uses nothing)
  PDP curves are flat (model output doesn't change with feature value)
```

**This looks like all features are unimportant — but it's not the data's fault.**

It's the model's failure to learn anything. Importances from an underfit model are uninformative.

**Practical check before trusting any importance estimate:**

```
Is the model performing better than a baseline (majority class, mean prediction)?
  If NO:  don't compute importance — the model hasn't learned anything
  If YES: proceed, but check train-test gap to calibrate trust
```

---

## 6. Sample Size Effects

All importance methods suffer from high variance with small datasets. The smaller the dataset, the less reliable the importance estimates.

**Permutation importance variance vs sample size:**

```
Variance(Imp(j)) ≈ Var(single-permutation score) / (n_test × n_repeats)

n_test = 50,  n_repeats=10:   high variance — importance estimates unreliable
n_test = 500, n_repeats=10:   moderate variance
n_test = 5000, n_repeats=10:  low variance — stable estimates
```

**Practical effect:**

```
n_test = 50, feature j has true importance = 0.05:
  Estimated importance across 100 random test sets: 0.02 to 0.11
  Range is so wide that you can't reliably distinguish j from noise

n_test = 1000, same feature:
  Estimated importance: 0.04 to 0.06
  Tight range — reliable estimate
```

**Mitigation for small datasets:**

```
1. Increase n_repeats (more permutation shuffles per feature)
2. Bootstrap the test set: resample test set with replacement K times,
   compute importance on each, report mean and CI
3. Use cross-validated permutation importance:
   Compute importance on each CV fold's validation set, average
4. Report confidence intervals, not point estimates
```

**SHAP with small background datasets:**

KernelSHAP and marginal SHAP need representative background samples to correctly compute E[f(X)]. With only 30 background samples, the baseline is poorly estimated and coalition values v(S) are noisy.

Rule: use at least 100 background samples. For critical analyses, use 300+.

---

## 7. Bias-Variance Table — Every Method

```
Method           Primary Bias               Primary Variance            Controlled By
─────────────────────────────────────────────────────────────────────────────────────
Permutation      Training set if overfit    n_repeats, n_test            Test set, n_repeats
(test set)       Correlated features        Sample size                  Grouped permutation

MDI / Gini       Training set always        Bootstrap + feat sampling    Not fixable; use test-set
                 High-cardinality features  Decreases with T             permutation instead
                 Correlated features

TreeSHAP         Marginal expectation       Zero (deterministic)         Use interventional variant
                 (unrealistic combos)                                    for correlated features

KernelSHAP       Coalition sampling         1/sqrt(M_coalitions)         Increase M
                 Background dataset         Background size              Use representative bg

LIME             Linear approximation       Random perturbation          n_samples, kernel width
                 Kernel width σ choice      Replacement values           Run multiple times
                 Tabular unrealistic inputs Lasso path sensitivity

PDP              Extrapolation (correlated  Bin count (few samples       Use ALE for correlated
                 features)                  at extreme values)            features

ALE              Bin width (misses fine     Few samples per bin          Tune B; larger dataset
                 curvature)                 Accumulation amplifies early
                                            variance

Surrogate        1 − fidelity R²            Surrogate training           Increase max_depth,
                 (model behaviour not                                     check fidelity
                 captured = blind spot)

Anchors          Precision threshold τ      KL-LUCB sampling             More samples in beam
                                                                          search, set τ carefully

Counterfactuals  Distance metric choice     Optimisation multiple optima  DiCE for diverse CF
```

---

## 8. Method-Specific Deep Dives

### 8.1 Permutation Importance

**Variance control:**

The variance of permutation importance is approximately:

```
Var(Imp(j)) ≈ Var(single permuted score) / n_repeats

95% confidence interval (approximate):
  Imp(j) ± 1.96 × std_j / sqrt(n_repeats)

For reliable ranking: the gap between adjacent importances should be
                      > 1.96 × std / sqrt(n_repeats)
```

**Bias from training set (overfit model):**

Noise feature with true importance 0 can appear with importance 0.15–0.25 on training set when model overfits. This is systematic — it doesn't decrease with n_repeats. Only using the test set fixes it.

**Bias from correlated features:**

Each feature in a group of r-correlated features has its importance underestimated by approximately `r²`. So at r=0.85, each feature is underestimated by ~72%.

### 8.2 MDI / Gini

**Bias:** Always computed on training data → always contains overfit bias. Additionally:
- High-cardinality bias (covered in `2_Impurity_Importance.md`)
- Correlation bias (covered in `8_Correlated_Features.md`)

**Variance across trees:**

```
Var(MDI_RF(j)) ≈ Var_tree(j) / T

CV = Std/Mean across trees:
  CV > 0.30 → feature ranking is unstable even across this single model's trees
  CV < 0.10 → stable estimate
```

MDI has lower variance than permutation importance but higher bias. It's a biased-but-stable estimator.

### 8.3 SHAP (TreeSHAP)

**Zero variance:** TreeSHAP is exact and deterministic — given the same model and data, it always produces the same values.

**Bias:** From using the marginal expectation (creates unrealistic coalitions for correlated features). Use interventional SHAP to reduce this.

**The overfit effect:**

```
TreeSHAP on training set: large |SHAP| for noise features
                           (model uses them for training predictions)
TreeSHAP on test set:     near-zero SHAP for noise features
                           (noise patterns don't generalise)

Always compute on test set for interpretability that reflects generalisation.
```

**Background dataset bias:**

The baseline E[f(X)] and the marginal distributions used for absent features both depend on the background dataset. Biased background → biased SHAP values (absolute magnitude, not relative).

### 8.4 SHAP (KernelSHAP)

**Variance: Proportional to 1/sqrt(M)**

```
M=100 coalitions:   SHAP value error ≈ 20–40% of true value
M=500:              ≈ 5–15%
M=2000:             ≈ 2–5%
M=5000:             ≈ 1–2%
```

The variance affects individual SHAP values — the efficiency axiom still holds (sum is correct) but individual values can be off.

**Practical rule:** Run KernelSHAP with M=2000 for moderate precision. For publication or consequential decisions, use M≥5000.

### 8.5 LIME

**Variance: Highest of all common methods for tabular data**

```
Coefficient of Variation (CV = std/mean) across runs:
  Tabular LIME: CV often 0.3–0.8
  Text LIME:    CV often 0.1–0.3
  Image LIME:   CV often 0.1–0.2
```

**Bias from linear approximation:**

If the true local decision boundary is nonlinear (e.g., interaction effects), the linear model systematically misrepresents it. This bias doesn't decrease with n_samples.

**Practical stability test:**

```
Run LIME 20 times. Compute:
  sign_consistency(j) = fraction of runs where feature j has the same sign
  cv(j)               = std(coeff_j) / |mean(coeff_j)|

Reliable:   sign_consistency ≥ 0.90 AND cv ≤ 0.30
Unreliable: sign_consistency < 0.80 OR cv > 0.50
```

### 8.6 PDP / ALE

**PDP bias:** Proportional to feature correlations. No variance issue (averaging over all N samples gives a stable estimate).

**ALE variance:** Each bin uses only N/B samples. For B=20 bins and N=500 samples, each bin has ~25 samples — noisy. Early bin noise accumulates through the integral.

```
ALE standard error at value v ≈ sqrt(Σ_{l≤v} (SE_bin_l)²)
  where SE_bin_l = std(local_effects in bin l) / sqrt(n_l)

Confidence bands widen at extreme feature values (fewer samples in tails)
```

**Always plot ALE with confidence bands** — sklearn's implementation provides them.

### 8.7 Surrogate Models

**Bias = 1 − fidelity R²:**

```
If the surrogate captures R²=0.80 of the black-box's variance, then 20%
of the model's behaviour is not captured. Explanations derived from the
surrogate may be systematically wrong for that 20%.

The bias is not random — it's systematic to the cases where the surrogate
differs most from the black box. These are typically the complex, nonlinear
cases near decision boundaries.
```

**Variance:** Low for a decision tree surrogate on sufficient data. High when dataset is small (<200 samples).

---

## 9. The Faithfulness Problem — Explaining an Overfit Model

This is a subtle but important conceptual point.

**SHAP, LIME, and other post-hoc methods explain what the model does, not what is true about the world.**

If your model overfits, the explanations are:
- Internally consistent (they correctly describe the model's behaviour)
- Externally misleading (the model's behaviour doesn't reflect real patterns)

```
Scenario: model learns that zip_code=90210 (Beverly Hills) predicts high income.
          This is true in the training data.
          But the model overfits this specific zip code — it doesn't generalise.

SHAP on training data:
  zip_code SHAP for a Beverly Hills sample: +0.35 (large positive)
  This is correct — the model uses this feature heavily

SHAP on test data from other zip codes:
  zip_code SHAP: +0.02 (near zero)
  The zip_code pattern doesn't generalise to other areas

The training-data SHAP is technically correct about the model.
But reporting it to stakeholders would imply zip code is a major income predictor
— which is only true for this specific overfitted model.
```

**The practical rule:**

Compute all XAI outputs on test data (or cross-validated on held-out folds). Training-data explanations describe memorisation — useful for debugging but not for communicating about the model's general behaviour.

---

## 10. Confidence Intervals for Importance Estimates

Feature importance is an estimate. It should be reported with uncertainty.

**Permutation importance CI:**

```
From n_repeats permutation runs, for feature j:
  mean_imp = mean of importance across runs
  std_imp  = std of importance across runs

95% CI: mean_imp ± 1.96 × std_imp / sqrt(n_repeats)

For comparing two features:
  They are statistically distinguishable if their CIs don't overlap.
  If CIs overlap → cannot reliably rank them → treat as tied.
```

**SHAP CI (KernelSHAP):**

Run KernelSHAP K times with different random seeds. Compute mean and std of SHAP values per feature.

**MDI CI across trees:**

```
importances_per_tree = [tree.feature_importances_ for tree in forest.estimators_]
mean = np.mean(importances_per_tree, axis=0)
std  = np.std(importances_per_tree, axis=0)
CI   = mean ± 1.96 × std / sqrt(n_estimators)
```

**Practical reporting example:**

```
Feature       Importance   95% CI         Stable rank?
──────────────────────────────────────────────────────────
debt_ratio    0.185        [0.171, 0.199]  ✅ Yes
income        0.142        [0.128, 0.156]  ✅ Yes (gap > combined CI width)
age           0.088        [0.071, 0.105]  ⚠️ Maybe (CI overlaps with employment)
employment    0.079        [0.063, 0.095]  ⚠️ Maybe
zip_code      0.031        [0.018, 0.044]  ✅ Yes (bottom tier)
noise_col     0.008        [0.000, 0.019]  ✅ Yes (CI includes 0 → not significant)
```

---

## 11. Interview Q&A

**Q: What is the Rashomon problem in the context of XAI?**

The Rashomon problem is the observation that many different models can achieve the same predictive performance (they form a "Rashomon set"), but attribute very different importances to the same features. Since feature importance is a property of the model, not the data, two equally accurate models can give completely different importance rankings. This means importance estimates are inherently model-dependent. The practical implication: report importances across multiple model types and flag features where models disagree — those are unreliable importance estimates.

---

**Q: Why should you compute feature importance on the test set rather than the training set?**

An overfit model memorises noise patterns in training data and assigns non-zero importance to noise features on the training set — they genuinely affect training-set predictions. On the test set, those memorised noise patterns don't apply, so permuting noise features has no effect. Training-set importance is biased upward for any feature the model overfitted to. Test-set importance reflects generalisation — what features actually matter for new data.

---

**Q: What does the gap between train-set and test-set importance tell you?**

A large positive gap (train >> test) means the model is overfitting to that feature — it memorised spurious training-set correlations. This is actionable: investigate the feature for noise or leakage, and regularise the model. A gap near zero means the feature generalises well. A rare negative gap (test > train) can indicate that the feature's signal is more evident in the test distribution than in the training sample — possible sign of training data sampling bias.

---

**Q: An underfit model shows near-zero importance for all features. What does this mean?**

The model has failed to learn. It predicts roughly the mean or majority class for all inputs, ignoring features. Near-zero importance across all features is a symptom of the model, not a property of the data. Before computing any importance, verify that the model performs significantly better than a naive baseline (mean prediction, majority class). If it doesn't, improve the model before interpreting it.

---

**Q: TreeSHAP has zero variance. Does that mean SHAP values are always reliable?**

Zero variance means TreeSHAP produces the same values every run — it's deterministic. But it can still be biased:
1. Marginal SHAP creates unrealistic coalitions for correlated features — the values are deterministically wrong for those cases.
2. If the model overfits, TreeSHAP on training data correctly describes the overfit model's memorised patterns, not generalisation.
3. The background dataset affects the baseline — a non-representative background produces biased absolute values.

Deterministic ≠ correct. TreeSHAP's zero variance is a property of precision, not accuracy.

---

**Q: Two models both achieve AUC=0.87. Their permutation importance rankings are completely different. What do you conclude?**

This is the Rashomon effect. Both models are valid but use different feature combinations to achieve the same performance. The differences likely come from correlated features — both models found different paths through the correlated feature space. Conclusions: (1) Feature importance for those features is model-dependent, not a data property. (2) For consensus features (same rank in both), the importance is robust. (3) For divergent features (different rank in both), report a range rather than a point estimate. (4) Check VIF for the divergent features — they are likely correlated.

---

**Q: What is the relationship between model fidelity and explainability bias for surrogate models?**

Fidelity R² directly equals 1 − bias fraction. If fidelity = 0.80, then 20% of the black-box's behaviour is not captured by the surrogate. Every explanation derived from the surrogate is systematically wrong for that 20% of cases — and there's no way to know which cases fall in that 20%. Reporting surrogate-based explanations without reporting fidelity is like reporting a regression coefficient without its standard error — the number is meaningless without its uncertainty.

---

## 12. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BIAS-VARIANCE IN XAI — KEY FACTS                                        │
├──────────────────────────────────────────────────────────────────────────┤
│  CORE PRINCIPLE                                                           │
│    Every XAI method is an estimator with bias and variance.              │
│    Bias = systematic error (doesn't shrink with more data)               │
│    Variance = random error (shrinks with more repeats / data)            │
│                                                                           │
│  THE RASHOMON PROBLEM                                                     │
│    Multiple equally accurate models can give different importances.      │
│    Feature importance is a property of the MODEL, not the DATA.         │
│    Fix: report importance across multiple model types; flag disagreement │
│                                                                           │
│  TRAIN VS TEST                                                            │
│    Train-set importance: biased for overfit features (too high)          │
│    Test-set importance: reflects generalisation                          │
│    Gap = train − test: large gap = overfitting to that feature           │
│    Rule: ALWAYS use test set for importance/explainability               │
│                                                                           │
│  UNDERFITTING                                                             │
│    All importances collapse to zero → meaningless                        │
│    Verify model performance before computing importance                  │
│                                                                           │
│  VARIANCE BY METHOD                                                       │
│    LIME (tabular):     very high — run 10+ times, compute CV             │
│    KernelSHAP:         moderate — use M≥2000 coalitions                 │
│    Permutation:        moderate — use n_repeats≥20 on test set          │
│    MDI (RF):           low (deterministic per tree) — decreases with T  │
│    TreeSHAP:           zero (exact, deterministic)                       │
│                                                                           │
│  BIAS BY METHOD                                                           │
│    MDI:         always uses training set + cardinality + correlation     │
│    Permutation: training set if overfit + correlated features            │
│    Marginal SHAP: unrealistic coalitions for correlated features         │
│    LIME:        linear approximation + unrealistic perturbations         │
│    PDP:         extrapolation for correlated features                    │
│    Surrogate:   1 − fidelity R²                                         │
│                                                                           │
│  CONFIDENCE INTERVALS                                                     │
│    Always report: mean ± CI, not just point estimate                    │
│    If two features' CIs overlap: cannot reliably rank them              │
│    Report features with CI covering 0 as "not significantly important"  │
│                                                                           │
│  THE FAITHFULNESS TRAP                                                   │
│    XAI methods explain the model, not reality.                          │
│    A perfectly explained overfit model is still wrong about the world.  │
│    Explanation correctness ≠ model correctness                           │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Fisher, Rudin & Dominici (2019)** — *All Models are Wrong, but Many are Useful.* — Model reliance and the Rashomon set.
- **Semenova & Rudin (2019)** — *A Study in Rashomon Curves and Volumes.* — Formal characterisation.
- **Hooker & Mentch (2019)** — *Please Stop Permuting Features: An Explanation and Alternatives.* — Bias in permutation importance.
- **Alvarez-Melis & Jaakkola (2018)** — *On the Robustness of Interpretability Methods.* — LIME variance analysis.
- **Molnar (2022)** — *Interpretable Machine Learning*, Chapter 8.6 on pitfalls.
- **Companion files:** All files in this folder.
