# Chapter 12: Uncertainty Quantification

> *"A model that doesn't know what it doesn't know is the most dangerous kind. Uncertainty quantification is how we give models the ability to say: I'm not sure — and have that mean something."*

---

## 12.1 Why Uncertainty Quantification?

Standard ML models output a prediction. Well-calibrated models output a probability. But neither tells you the full picture of how uncertain the model is — and why.

Consider these two scenarios:

```
Scenario A: Chest X-ray classification
  Model: p(pneumonia) = 0.85
  Training data had thousands of similar X-rays
  → High confidence is justified

Scenario B: Chest X-ray classification
  Model: p(pneumonia) = 0.85
  This X-ray looks unlike anything in training
  → High confidence is NOT justified
```

Both produce identical outputs. Only uncertainty quantification distinguishes them.

**UQ matters for:**
- **Safety-critical systems**: autonomous vehicles, medical diagnosis, structural engineering
- **Human-in-the-loop**: knowing when to escalate to a human
- **Active learning**: querying labels for samples the model is most uncertain about
- **Distribution shift detection**: model uncertainty increases when input is out-of-distribution
- **Decision making under uncertainty**: risk-aware downstream systems

---

## 12.2 Types of Uncertainty

The field distinguishes two fundamental sources of uncertainty:

### Aleatoric Uncertainty (Data Uncertainty)

Irreducible noise inherent in the data itself. Cannot be reduced by collecting more data or building a better model.

```
Examples:
  - A blurry photo — ambiguity is in the image itself
  - Two patients with identical symptoms but different diagnoses
  - A coin flip — the outcome is genuinely random
  - Sensor noise in measurements
```

Aleatoric uncertainty is the irreducible floor of prediction error. Even a perfect model cannot eliminate it.

**Two subtypes:**
- **Homoscedastic**: same noise level across all inputs (common assumption)
- **Heteroscedastic**: noise varies with input (more realistic; different patients have different inherent ambiguity)

### Epistemic Uncertainty (Model Uncertainty)

Uncertainty due to lack of knowledge — from limited training data, wrong model class, or out-of-distribution inputs. **Can be reduced** by collecting more data or better modeling.

```
Examples:
  - A rare disease with few training examples
  - An input far from the training distribution
  - A new geographic region the model hasn't seen
  - Ambiguity between multiple plausible model configurations
```

Epistemic uncertainty is the target of active learning, data collection, and model improvement.

### Why the Distinction Matters for Evaluation

```
High aleatoric, low epistemic  →  Prediction is inherently noisy; model is well-informed
High epistemic, low aleatoric  →  Model needs more data; don't trust this prediction
High both                      →  Don't deploy; human review needed
Low both                       →  High-confidence, trustworthy prediction
```

Ideally, your UQ method should distinguish these. In practice, most methods estimate total uncertainty; separating aleatoric from epistemic requires more sophisticated approaches.

---

## 12.3 Conformal Prediction

The most principled, distribution-free approach to uncertainty quantification. Provides **guaranteed coverage** without distributional assumptions.

### The Core Idea

Instead of outputting a single prediction, output a **prediction set** — a set of labels guaranteed to contain the true label with at least (1-α) probability.

```
Standard prediction:
  Input → "cat"

Conformal prediction (α = 0.1, 90% coverage):
  Input → {"cat", "lynx"}          ← both plausible; model unsure
  Input → {"dog"}                  ← confident single prediction
  Input → {"cat", "dog", "wolf"}   ← very uncertain; many possible labels
```

**Key property:** The prediction set is guaranteed to contain the true label on ≥ 90% of new samples, with no distributional assumptions.

### How It Works: Split Conformal Prediction

**Step 1: Calibration set**
Hold out a calibration set (n samples) separate from training and test.

**Step 2: Nonconformity scores**
For each calibration sample, compute a nonconformity score — how "surprising" the true label is given the model:

```
For classification:
  Nonconformity score = 1 - p(true label | input)

For regression:
  Nonconformity score = |y - ŷ|
```

**Step 3: Quantile threshold**
Find the (1-α) quantile of the nonconformity scores on the calibration set:

```
q̂ = ⌈(n+1)(1-α)⌉/n quantile of {s₁, s₂, ..., sₙ}
```

**Step 4: Prediction sets**
At test time, include all labels k in the prediction set for which:

```
1 - p(k | x_test) ≤ q̂
⟺  p(k | x_test) ≥ 1 - q̂
```

### Worked Example

Calibration set: 1000 samples. α = 0.1 (90% coverage desired).

```
Nonconformity scores: [0.02, 0.05, 0.08, ..., 0.45, 0.72, 0.95]
90th percentile: q̂ = 0.38

At test time, include label k if:
  p(k | x) ≥ 1 - 0.38 = 0.62

Test sample 1: p(cat)=0.89, p(dog)=0.08, p(bird)=0.03
  → Include only "cat" (0.89 ≥ 0.62)
  → Prediction set: {"cat"}

Test sample 2: p(cat)=0.65, p(dog)=0.63, p(bird)=0.12
  → Include "cat" and "dog" (both ≥ 0.62)
  → Prediction set: {"cat", "dog"}   ← model is uncertain

Test sample 3: p(cat)=0.40, p(dog)=0.35, p(bird)=0.25
  → None exceed 0.62
  → Prediction set: {} (empty — unusual; increase coverage or flag OOD)
```

### Coverage Guarantee

**Marginal coverage theorem:** Under exchangeability,

```
P(y_test ∈ Ĉ(x_test)) ≥ 1 - α
```

This holds for **any** classifier, regardless of how well-calibrated it is, as long as the calibration set is exchangeable with the test set.

### Evaluating Conformal Predictors

| Metric | What It Measures |
|---|---|
| **Coverage** | Fraction of test samples where true label is in prediction set. Should be ≥ 1-α. |
| **Efficiency (Average Set Size)** | Mean size of prediction sets. Smaller = more informative. |
| **Singleton Rate** | Fraction of prediction sets with exactly one label. Higher = more confident. |
| **Empty Set Rate** | Fraction of inputs with no label in the set. Indicates OOD inputs. |

```python
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory

# Fit conformal predictor
icp = IcpClassifier(NcFactory.get_nc(model))
icp.fit(X_train, y_train)
icp.calibrate(X_calib, y_calib)

# Prediction sets at 90% coverage
prediction_sets = icp.predict(X_test, significance=0.1)

# Evaluate
coverage = np.mean([y_test[i] in prediction_sets[i] for i in range(len(y_test))])
avg_set_size = np.mean([len(s) for s in prediction_sets])
print(f"Coverage: {coverage:.3f}")       # Should be ≥ 0.90
print(f"Avg set size: {avg_set_size:.2f}")  # Smaller is better
```

### Conformal Prediction for Regression

For regression, output a **prediction interval** instead of a set:

```
Ĉ(x) = [ŷ(x) - q̂, ŷ(x) + q̂]

Where q̂ is the (1-α) quantile of |yᵢ - ŷᵢ| on the calibration set.
```

This gives a guaranteed-coverage interval. But it's constant-width — the same interval regardless of how confident the model is. More sophisticated variants (CQR) use quantile regression for adaptive-width intervals.

---

## 12.4 Prediction Intervals

Prediction intervals extend point predictions with uncertainty bounds, primarily for regression.

### Types of Intervals

**Confidence interval:** Uncertainty about the **mean** prediction.
```
CI: ŷ ± z × SE(ŷ)
Covers: the true mean response
Does NOT cover: individual new observations
```

**Prediction interval:** Uncertainty about a **new individual observation**.
```
PI: ŷ ± z × √(SE(ŷ)² + σ²)
Covers: individual new observations with stated probability
Wider than CI — accounts for irreducible noise
```

For ML applications, prediction intervals are almost always what you want.

### Methods for Prediction Intervals

**1. Conformal Prediction Intervals (above)**
- Distribution-free guarantee
- Constant width (basic) or adaptive (CQR)

**2. Quantile Regression**
Train separate models for lower and upper quantiles:

```python
from sklearn.ensemble import GradientBoostingRegressor

# 80% prediction interval: predict 10th and 90th percentiles
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.10)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.90)

model_lower.fit(X_train, y_train)
model_upper.fit(X_train, y_train)

lower = model_lower.predict(X_test)
upper = model_upper.predict(X_test)
```

**3. Bayesian Methods**
Model a full posterior distribution over predictions. Naturally produces intervals.

**4. Bootstrap**
Train K models on bootstrap samples. Use the distribution of their predictions as an interval.

### Evaluating Prediction Intervals

| Metric | Formula | Meaning |
|---|---|---|
| **Coverage** | P(y ∈ [lower, upper]) | Fraction of true values inside interval. Target: 1-α |
| **Width** | mean(upper - lower) | Average interval size. Smaller = more useful |
| **PICP** | Prediction Interval Coverage Probability | Same as coverage; explicit name |
| **MPIW** | Mean Prediction Interval Width | Average width |
| **CWC** | Coverage Width Criterion | Penalizes wide intervals: MPIW × exp(η(PICP-target)) |

**The coverage-width trade-off:** Any method can achieve 100% coverage by outputting infinitely wide intervals. A useful interval is both **well-covered** and **narrow**. Always report both.

```
Good:  Coverage = 90%, Width = 5.2   ← tight and accurate
Bad:   Coverage = 88%, Width = 5.2   ← undercoverage
Bad:   Coverage = 99%, Width = 52.0  ← overcoverage (uninformative)
```

---

## 12.5 Bayesian Approaches

Bayesian methods place a distribution over model parameters, naturally producing uncertainty estimates.

### Bayesian Neural Networks (BNNs)

Instead of point estimates for weights, learn distributions:

```
Standard NN:  w → single value
Bayesian NN:  w → p(w | data) — a distribution

At inference:
  p(y | x) = ∫ p(y | x, w) p(w | data) dw
```

The integral is intractable — approximated by sampling.

**Evaluation:** The posterior predictive distribution gives both a prediction and uncertainty. Evaluate with:
- Log-loss on the predictive distribution
- Coverage of credible intervals
- Separation of epistemic vs. aleatoric uncertainty

### Monte Carlo Dropout (MC Dropout)

A practical approximation to Bayesian inference using standard dropout.

```python
def mc_dropout_predict(model, X, n_samples=100):
    model.train()  # Keep dropout active at inference
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(X)
            predictions.append(torch.softmax(pred, dim=-1))

    predictions = torch.stack(predictions)  # (n_samples, batch, n_classes)

    mean_pred = predictions.mean(dim=0)     # Point estimate
    uncertainty = predictions.std(dim=0)    # Epistemic uncertainty proxy

    return mean_pred, uncertainty
```

**Evaluation:**
- Use mean prediction for standard metrics
- Use std as epistemic uncertainty signal: higher std → model is more uncertain
- Check: does std correlate with prediction error? (It should.)

### Deep Ensembles

Train K independent models with different random seeds. Their disagreement estimates epistemic uncertainty.

```python
# Ensemble of 5 models
predictions = [model_k(X_test) for model_k in ensemble]
# predictions: list of (n_samples, n_classes) tensors

mean_pred = torch.stack(predictions).mean(dim=0)
epistemic_uncertainty = torch.stack(predictions).std(dim=0)
```

**Empirically**, deep ensembles outperform MC Dropout and many BNN approximations on calibration benchmarks while being simple to implement.

---

## 12.6 Out-of-Distribution Detection

A model should recognize when an input is far from its training distribution — and express higher uncertainty or abstain.

### The OOD Problem

```
Training data: chest X-rays from Hospital A
Deployment:   chest X-ray from Hospital B with different scanner
              → Different image statistics
              → Model may be confidently wrong
              → High epistemic uncertainty should flag this
```

### OOD Detection Methods

**Maximum Softmax Probability (MSP)**
Baseline: flag samples where max class probability is below a threshold.

```python
confidence = softmax_probs.max(axis=1)
is_ood = confidence < threshold
```

Simple but not reliable — neural networks can be overconfident on OOD inputs.

**Temperature Scaling**
Higher temperature → softer distributions → lower max probability → better OOD separation (sometimes).

**Mahalanobis Distance**
Compute distance of test sample from class-conditional Gaussian distributions fitted to training features:

```
D(x) = min_k √[(f(x) - μₖ)ᵀ Σ⁻¹ (f(x) - μₖ)]
```

High Mahalanobis distance → far from any training class distribution → likely OOD.

**Energy Score**
```
E(x) = -log Σₖ exp(fₖ(x))
```
In-distribution samples have lower energy; OOD samples have higher energy.

### Evaluating OOD Detectors

Treat OOD detection as binary classification (in-distribution vs. OOD):

| Metric | What It Measures |
|---|---|
| AUROC (OOD) | Separability of in-dist vs OOD scores |
| AUPR-In | Average precision treating in-dist as positive |
| AUPR-Out | Average precision treating OOD as positive |
| FPR@95TPR | False positive rate when 95% of in-dist correctly accepted |

**Standard benchmark:** Train on CIFAR-10; test OOD on SVHN, LSUN, etc. Report FPR@95TPR.

---

## 12.7 Calibration of Uncertainty Estimates

Uncertainty estimates themselves need calibration. A model that says "I'm 80% uncertain" should actually be uncertain 80% of the time.

### Reliability Diagram for Uncertainty

For conformal predictors: plot empirical coverage vs. target coverage (α) across a range of α values.

```
Ideal: points on diagonal (empirical coverage = target coverage)
Over-conservative: points above diagonal (sets larger than needed)
Under-conservative: points below diagonal (coverage guarantee violated)
```

### Sharpness of Uncertainty

Calibrated uncertainty that is still tight (small prediction sets, narrow intervals) is more useful than calibrated uncertainty that is wide and uninformative.

**Sharpness metrics:**
- Average prediction set size (classification)
- Average interval width (regression)
- Fraction of singleton predictions

Always jointly report coverage (calibration) and sharpness (efficiency).

---

## 12.8 Evaluation Framework for UQ Systems

```
Step 1: Choose UQ method appropriate to task
  Classification  → Conformal prediction sets, MC Dropout, ensembles
  Regression      → Conformal intervals, quantile regression, Bayesian
  OOD detection   → Energy score, Mahalanobis, MSP

Step 2: Evaluate calibration of uncertainty
  Coverage: does 90% interval actually cover 90% of outcomes?
  Plot: empirical vs. target coverage across α values

Step 3: Evaluate sharpness
  Set size / interval width
  Are confident predictions correct more often?

Step 4: Evaluate OOD behavior
  Does uncertainty increase on OOD inputs?
  AUROC on OOD detection benchmark

Step 5: Evaluate downstream impact
  Does abstaining on high-uncertainty inputs improve precision?
  Does uncertainty routing to humans improve overall system accuracy?
```

---

## 12.9 Worked Example: Medical Triage System

**Task:** Classify skin lesion images as benign/malignant. Route uncertain cases to dermatologist.

**System design:**
- Conformal predictor at α = 0.05 (95% coverage)
- Three outcomes: {malignant}, {benign}, {malignant, benign} (uncertain → human review)

```
Test set: 1,000 lesions (150 malignant, 850 benign)

Results:
  Singleton "malignant":        120 cases  → 118 correct (98.3% precision)
  Singleton "benign":           700 cases  → 697 correct (99.6% precision)
  Set {"malignant","benign"}:   180 cases  → routed to dermatologist

Coverage check:
  True positives in prediction sets: 149/150 = 99.3% ≥ 95% ✓
  True negatives in prediction sets: 845/850 = 99.4% ≥ 95% ✓

Efficiency:
  Average set size: 1.18 (most predictions are singletons)
  Abstention rate: 18% (180/1000 routed to human)
```

**Business value:**
- Model handles 82% of cases automatically with >99% precision
- 18% routed to human — the genuinely hard cases
- Without UQ: model would make 32 confident wrong predictions
- With UQ: uncertain cases abstain; confident predictions are highly reliable

**Lesson:** UQ doesn't just measure uncertainty — it enables systems that know when to act and when to defer.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Aleatoric uncertainty | Irreducible noise in the data; can't be fixed with more data |
| Epistemic uncertainty | Model ignorance; reducible with more data or better models |
| Conformal prediction | Distribution-free guaranteed coverage; gold standard for prediction sets |
| Coverage | Fraction of true values inside prediction set/interval; must meet target |
| Efficiency | Sharpness of uncertainty; smaller sets/intervals are better |
| Prediction intervals | Always report coverage AND width together |
| MC Dropout | Practical Bayesian approximation; keep dropout on at inference |
| Deep ensembles | Simple, empirically strong UQ via model disagreement |
| OOD detection | Uncertainty should increase for out-of-distribution inputs |
| Calibration of UQ | Uncertainty estimates themselves need calibration |

---

## Further Reading

- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction* (2022) — the best practical intro
- Gal & Ghahramani — *Dropout as a Bayesian Approximation* (ICML 2016) — MC Dropout
- Lakshminarayanan et al. — *Simple and Scalable Predictive UQ using Deep Ensembles* (NeurIPS 2017)
- Hendrycks & Gimpel — *A Baseline for Detecting Misclassified and OOD Examples* (ICLR 2017)
- Romano et al. — *Conformalized Quantile Regression* (NeurIPS 2019) — adaptive-width intervals

---

*Next: Chapter 13 — NLP Evaluation*
