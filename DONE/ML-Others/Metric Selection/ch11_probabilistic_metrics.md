# Chapter 11: Probabilistic Metrics — Log-loss, Brier Score, and ECE

> *"A model that says 'I'm sure' when it's wrong is more dangerous than a model that says 'I'm not sure' when it's right. Probabilistic metrics measure whether your model knows what it doesn't know."*

---

## 11.1 Why Probabilistic Metrics Matter

Every metric we've discussed so far — accuracy, F1, AUC, NDCG — evaluates decisions, not uncertainty. They ask: *did the model get it right?*

Probabilistic metrics ask a different question: *does the model know how likely it is to be right?*

This distinction matters enormously when:

- Downstream systems use the raw probability score, not just the class label
- Decisions have asymmetric costs that depend on confidence level
- You are combining model outputs (ensembling, calibration stacking)
- The model's uncertainty is itself a useful signal (triage, human-in-the-loop)
- You want to know if you should trust a specific prediction

```
A model predicts p = 0.95 for a positive outcome.
The outcome is negative.

Bad calibration metric?   Yes — the model was overconfident.
Bad AUC?                  Not necessarily — AUC only cares about ranking.
Bad F1?                   Not necessarily — the prediction was still "positive" at threshold 0.5.

Only a probabilistic metric catches this.
```

### The Three Properties of Good Probability Estimates

**Calibration:** Predicted probabilities match empirical frequencies.
**Sharpness (Resolution):** Predictions are decisive — close to 0 or 1, not always near the base rate.
**Proper scoring:** The metric rewards honest probability reporting and cannot be gamed by distorting predictions.

A **proper scoring rule** is one where your expected score is maximized by reporting your true beliefs. You cannot improve your expected score by lying about your uncertainty. Log-loss and Brier score are both proper scoring rules.

---

## 11.2 Log-Loss (Binary Cross-Entropy)

The most widely used probabilistic metric. It is also the training loss for logistic regression and neural networks — making it directly connected to what the model optimizes.

### Formula

```
Log-Loss = -(1/n) × Σᵢ [yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ)]

yᵢ ∈ {0, 1}  = true label
pᵢ ∈ (0, 1)  = predicted probability of positive class
```

Broken down per sample:

```
If yᵢ = 1 (true positive):   loss = -log(pᵢ)
If yᵢ = 0 (true negative):   loss = -log(1 - pᵢ)
```

### Behavior: The Penalty Curve

Log-loss penalizes confident wrong predictions exponentially:

```
Prediction for a positive sample (y=1):

p = 0.99  →  loss = -log(0.99) = 0.010   ← tiny penalty (correctly confident)
p = 0.90  →  loss = -log(0.90) = 0.105
p = 0.70  →  loss = -log(0.70) = 0.357
p = 0.50  →  loss = -log(0.50) = 0.693   ← baseline (coin flip)
p = 0.30  →  loss = -log(0.30) = 1.204
p = 0.10  →  loss = -log(0.10) = 2.303
p = 0.01  →  loss = -log(0.01) = 4.605   ← catastrophic penalty
```

The penalty approaches infinity as confidence in the wrong answer approaches 1. This is why a single wildly overconfident wrong prediction can dominate log-loss across thousands of correct predictions.

### Baselines for Log-Loss

```
Perfect model:               Log-loss = 0.0
Random classifier (50/50):   Log-loss = 0.693 (= log(2))
Always predict base rate p̄:  Log-loss = -[p̄ log(p̄) + (1-p̄) log(1-p̄)]
                                        (binary entropy of the base rate)
```

Always compare against the "predict base rate" baseline. On an imbalanced dataset with 5% positive rate:

```
Base rate baseline log-loss = -[0.05 × log(0.05) + 0.95 × log(0.95)]
                             = -[0.05 × (-2.996) + 0.95 × (-0.051)]
                             = 0.199
```

A model with log-loss = 0.18 beats this baseline. A model with log-loss = 0.22 does not.

### Multi-class Log-Loss

Extends naturally to K classes:

```
Log-Loss = -(1/n) × Σᵢ Σₖ yᵢₖ log(pᵢₖ)

yᵢₖ = 1 if sample i belongs to class k, else 0
pᵢₖ = predicted probability of class k for sample i
```

This is the standard categorical cross-entropy used in neural network training.

### Properties and Limitations

**Properties:**
- Proper scoring rule — cannot be gamed
- Differentiable — natural training objective
- Penalizes overconfidence severely
- Sensitive to calibration

**Limitations:**
- Unbounded above — a single catastrophically wrong confident prediction can dominate
- Requires predicted probabilities (not raw scores)
- Hard to interpret in absolute terms without a baseline
- Sensitive to label noise — mislabeled samples cause large losses even for "correct" model behavior

### Clipping for Stability

In practice, clip predictions away from 0 and 1 to avoid numerical instability:

```python
import numpy as np
from sklearn.metrics import log_loss

# Clip to avoid log(0)
eps = 1e-15
p_clipped = np.clip(p, eps, 1 - eps)
loss = log_loss(y_true, p_clipped)
```

---

## 11.3 Brier Score

A quadratic scoring rule — less severe than log-loss on overconfident predictions, easier to decompose.

### Formula

```
Brier Score = (1/n) × Σᵢ (pᵢ - yᵢ)²

pᵢ ∈ [0, 1]  = predicted probability
yᵢ ∈ {0, 1}  = true label
```

Range: [0, 1]. Lower is better. 0 = perfect, 1 = maximally wrong.

### Behavior: Quadratic vs. Logarithmic Penalty

```
Prediction for a positive sample (y=1):

p = 0.99  →  BS = (0.99-1)² = 0.0001   ← tiny
p = 0.70  →  BS = (0.70-1)² = 0.090
p = 0.50  →  BS = (0.50-1)² = 0.250
p = 0.30  →  BS = (0.30-1)² = 0.490
p = 0.10  →  BS = (0.10-1)² = 0.810
p = 0.01  →  BS = (0.01-1)² = 0.980   ← severe but not infinite
```

Unlike log-loss, Brier score doesn't go to infinity. A single catastrophically wrong prediction is bounded. This makes Brier score more **robust to outliers in prediction** than log-loss.

### Baselines

```
Perfect model:              Brier Score = 0.0
Random (50/50):             Brier Score = 0.25
Always predict base rate p̄: Brier Score = p̄ × (1-p̄)
```

For a 5% positive rate, the base rate baseline:
```
BS_baseline = 0.05 × 0.95 = 0.0475
```

A model achieving BS = 0.03 is meaningfully better than this baseline.

### Brier Skill Score (BSS)

Normalized version for comparability:

```
BSS = 1 - (BS_model / BS_baseline)

BSS = 1.0   →  Perfect
BSS = 0.0   →  No better than predicting base rate
BSS < 0.0   →  Worse than predicting base rate
```

### The Brier Score Decomposition

The most powerful feature of the Brier score — it decomposes into three interpretable components:

```
Brier Score = Calibration + Resolution - Uncertainty

Uncertainty  = p̄ × (1 - p̄)
               Inherent difficulty; irreducible noise in the outcome.
               High when base rate ≈ 0.5; low when outcomes are predictable.

Calibration  = (1/K) × Σₖ nₖ × (p̄ₖ - ōₖ)²
               Mean squared difference between mean predicted probability
               and observed frequency per bin.
               Perfect calibration → Calibration term = 0.

Resolution   = (1/n) × Σₖ nₖ × (ōₖ - p̄)²
               Variance of observed frequencies across bins.
               High resolution = model makes decisive, varied predictions.
               Low resolution = model always predicts near the base rate.
```

**Reading the decomposition:**

| Scenario | Calibration | Resolution | Brier Score |
|---|---|---|---|
| Perfect model | 0 | High | Low (good) |
| Well-calibrated, indecisive | 0 | Low | Medium |
| Sharp but miscalibrated | High | High | Medium |
| Base rate predictor | 0 | 0 | = Uncertainty |

**The key insight:** A model can have low calibration error but also low resolution — it's well-calibrated but uninformative (always predicts near the base rate). Resolution measures whether the model actually commits to predictions. You want high resolution *and* low calibration error.

```python
# Brier score decomposition
from sklearn.calibration import calibration_curve
import numpy as np

def brier_decomposition(y_true, y_prob, n_bins=10):
    p_bar = np.mean(y_true)
    uncertainty = p_bar * (1 - p_bar)

    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts, _ = np.histogram(y_prob, bins=n_bins)
    n = len(y_true)

    calibration = np.sum(bin_counts * (mean_pred - fraction_pos)**2) / n
    resolution  = np.sum(bin_counts * (fraction_pos - p_bar)**2) / n

    return {
        'uncertainty': uncertainty,
        'calibration': calibration,
        'resolution': resolution,
        'brier_score': calibration - resolution + uncertainty
    }
```

---

## 11.4 Expected Calibration Error (ECE)

Introduced in Chapter 4 as a calibration diagnostic. Here we go deeper on its properties and limitations.

### Formula (Recap)

```
ECE = Σₘ (|Bₘ| / n) × |acc(Bₘ) - conf(Bₘ)|

acc(Bₘ)  = observed positive rate in bin m
conf(Bₘ) = mean predicted probability in bin m
```

### ECE vs. Brier Score Calibration Term

Both measure calibration, but differently:

| | ECE | Brier Calibration Term |
|---|---|---|
| Error function | Absolute | Squared |
| Sensitivity to large gaps | Linear | Quadratic |
| Range | [0, 1] | [0, 1] |
| Proper scoring rule? | No | Yes (as part of Brier) |
| Interpretability | "Off by X%" | Less direct |

ECE is not a proper scoring rule — it can be gamed. A model can achieve lower ECE by making predictions more uniform (less sharp) rather than by being more accurate. For a proper calibration assessment, use the Brier calibration term.

### Binning Issues and Adaptive ECE

Fixed-width binning has problems:
- Bins near 0 and 1 may have very few samples → noisy estimates
- Bins near 0.5 may have many samples → over-represented in ECE

**Adaptive (equal-mass) binning:** Use bins with equal numbers of samples instead of equal width.

```python
# Adaptive ECE
def adaptive_ece(y_true, y_prob, n_bins=15):
    sorted_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_idx]
    y_prob_sorted = y_prob[sorted_idx]

    bins = np.array_split(np.arange(len(y_true)), n_bins)
    ece = 0.0
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        bin_conf = np.mean(y_prob_sorted[bin_idx])
        bin_acc  = np.mean(y_true_sorted[bin_idx])
        ece += (len(bin_idx) / len(y_true)) * abs(bin_conf - bin_acc)
    return ece
```

### Maximum Calibration Error (MCE)

```
MCE = max_m |acc(Bₘ) - conf(Bₘ)|
```

Captures the worst-case calibration gap. Relevant when high-confidence predictions drive the most consequential decisions. A model with low ECE but high MCE is well-calibrated on average but unreliable when it's most certain.

---

## 11.5 Comparing Log-Loss, Brier Score, and ECE

| Property | Log-Loss | Brier Score | ECE |
|---|---|---|---|
| Proper scoring rule | Yes | Yes | No |
| Penalizes overconfidence | Extremely (unbounded) | Moderately (bounded) | No |
| Decomposable | No (standard) | Yes | Partial |
| Measures calibration only | No (also resolution) | No (also resolution) | Yes |
| Sensitive to outlier predictions | Very | Moderately | No |
| Interpretability | Low (needs baseline) | Medium | High ("X% off") |
| Used as training loss | Yes | Rarely | No |
| Best for | Model selection, training | Decomposition analysis | Calibration diagnosis |

### When to Use Each

```
During training:
    Log-loss (cross-entropy) — it's differentiable and standard

For model selection:
    Log-loss (standard benchmark) + PR-AUC (discrimination)

For calibration diagnosis:
    ECE + reliability diagram (visual)

For understanding WHY calibration fails:
    Brier decomposition (calibration vs. resolution vs. uncertainty)

For high-stakes tail behavior:
    MCE (worst-case calibration gap)

For reporting to stakeholders:
    Brier Skill Score (normalized, intuitive scale)
```

---

## 11.6 Proper Scoring Rules: The Theory

Why does it matter that a metric is a "proper scoring rule"?

### Definition

A scoring rule S(p, y) is **proper** if:

```
E[S(p*, y)] ≥ E[S(p, y)]   for all p ≠ p*
```

Where p* is the true probability distribution. In English: you maximize your expected score by reporting your honest beliefs. Misreporting your uncertainty cannot help you.

### Improper Metrics Can Be Gamed

Accuracy is not a proper scoring rule. You maximize expected accuracy by predicting 1 when p > 0.5 and 0 when p < 0.5, regardless of the actual probability value. This discards all the probabilistic information.

AUC is not a proper scoring rule. It only measures ranking, not magnitude. A model that outputs [0.9, 0.1] and one that outputs [0.6, 0.4] for the same pair have identical AUC but very different calibration.

ECE is not a proper scoring rule. You can improve ECE by flattening your predictions (moving them toward the base rate), even if that's not honest.

**Log-loss and Brier score are proper.** The only way to improve them is to make better predictions, not to strategically misrepresent confidence.

### Implications for ML Systems

When you optimize a proper scoring rule during training, you're incentivizing the model to output its true probability estimates. When you evaluate with a proper scoring rule, you can trust that a lower score means genuinely better probability estimates — not a calibration trick.

---

## 11.7 Probabilistic Metrics for Multi-class

### Multi-class Log-Loss

Already covered — categorical cross-entropy with K classes.

### Multi-class Brier Score

```
BS_multiclass = (1/n) × Σᵢ Σₖ (pᵢₖ - yᵢₖ)²
```

Sums squared errors across all K class probabilities per sample. Range: [0, 2] for K classes (normalized versions available).

### Confidence Calibration for Multi-class

For multi-class, calibration is often measured on the **confidence** (maximum predicted probability):

```
For each sample i:
  confidence_i  = max_k p_ik       (probability of predicted class)
  accuracy_i    = 1[argmax_k p_ik = y_i]  (was prediction correct?)

ECE = Σ_bins (|B_m|/n) × |mean_accuracy(B_m) - mean_confidence(B_m)|
```

This asks: when the model says "I'm 90% confident in my prediction," is it correct 90% of the time?

---

## 11.8 Worked Example: Weather Forecasting

**Context:** Binary forecast — will it rain tomorrow? Evaluate three forecasting systems.

Dataset: 365 days, 120 rainy days (32.9% base rate).

```
System A: Climatology (always predicts base rate 0.329)
  Log-Loss: 0.648
  Brier:    0.221  (= 0.329 × 0.671, the uncertainty term)
  ECE:      0.000  (perfectly calibrated — always says 32.9%)

System B: Numerical Weather Model
  Log-Loss: 0.441
  Brier:    0.148
  ECE:      0.032  (slight overconfidence)
  BSS:      0.330  (33% improvement over climatology)

System C: Neural Network (uncalibrated)
  Log-Loss: 0.521
  Brier:    0.163
  ECE:      0.089  (significant overconfidence)
  BSS:      0.263

System C (after temperature scaling):
  Log-Loss: 0.448
  Brier:    0.149
  ECE:      0.018  (much better calibrated)
  BSS:      0.326  (nearly matches System B)
```

**Brier decomposition for System B:**

```
Uncertainty  = 0.329 × 0.671 = 0.221
Calibration  = 0.004  (very well calibrated)
Resolution   = 0.077  (makes useful, varied predictions)
Brier Score  = 0.004 - 0.077 + 0.221 = 0.148  ✓
```

**Key insights:**
- System A has perfect ECE (0.0) but is useless — it has zero resolution
- System C has worse ECE than System B but nearly identical Brier score after calibration
- Calibration fixed most of System C's deficit; the underlying model was discriminative
- BSS reveals System C post-calibration is competitive with the expensive numerical model

**Lesson:** ECE alone is misleading. A well-calibrated but uninformative model scores well on ECE. Always pair calibration metrics with resolution/sharpness measures.

---

## 11.9 Practical Implementation

```python
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def evaluate_probabilistic(y_true, y_prob, n_bins=10, model_name="Model"):
    """Complete probabilistic evaluation suite."""

    # Core metrics
    ll   = log_loss(y_true, y_prob)
    bs   = brier_score_loss(y_true, y_prob)
    p_bar = np.mean(y_true)
    bs_baseline = p_bar * (1 - p_bar)
    bss  = 1 - bs / bs_baseline

    # ECE
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts = np.histogram(y_prob, bins=n_bins)[0]
    ece = np.sum(bin_counts * np.abs(mean_pred - fraction_pos)) / len(y_true)
    mce = np.max(np.abs(mean_pred - fraction_pos))

    print(f"\n{model_name} — Probabilistic Evaluation")
    print(f"  Log-Loss:    {ll:.4f}  (baseline: {-p_bar*np.log(p_bar)-(1-p_bar)*np.log(1-p_bar):.4f})")
    print(f"  Brier Score: {bs:.4f}  (baseline: {bs_baseline:.4f})")
    print(f"  Brier SS:    {bss:.4f}")
    print(f"  ECE:         {ece:.4f}")
    print(f"  MCE:         {mce:.4f}")

    # Reliability diagram
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(mean_pred, fraction_pos, 's-', label=model_name)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Reliability Diagram — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {'log_loss': ll, 'brier': bs, 'bss': bss, 'ece': ece, 'mce': mce}
```

---

## Summary

| Metric | Formula | Proper? | Measures | Best For |
|---|---|---|---|---|
| Log-Loss | -mean[y log p + (1-y) log(1-p)] | Yes | Calibration + Resolution | Training loss, model selection |
| Brier Score | mean[(p - y)²] | Yes | Calibration + Resolution | Decomposition, robust eval |
| Brier Skill Score | 1 - BS/BS_baseline | Yes | Normalized improvement | Reporting, comparison |
| ECE | weighted mean \|conf - acc\| | No | Calibration only | Diagnosis, reliability diagram |
| MCE | max \|conf - acc\| | No | Worst-case calibration | High-stakes tail behavior |

---

## Further Reading

- Gneiting & Raftery — *Strictly Proper Scoring Rules, Prediction, and Estimation* (JASA, 2007) — the definitive theory
- Murphy — *A New Vector Partition of the Probability Score* (1973) — original Brier decomposition
- Guo et al. — *On Calibration of Modern Neural Networks* (ICML 2017) — ECE for deep learning
- Bröcker — *Reliability, Sufficiency, and the Decomposition of Proper Scores* (Quarterly Journal of the Royal Meteorological Society, 2009)

---

*Next: Chapter 12 — Uncertainty Quantification*
