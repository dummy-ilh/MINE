# Partial Dependence Plots (PDP) and Accumulated Local Effects (ALE)

> PDP first — how it works and why it fails. ALE second — how it fixes exactly that failure. Both with full numerical examples and every plot type.

---

## Table of Contents

1. [What PDP and ALE Are For](#1-what-pdp-and-ale-are-for)
2. [PDP — Partial Dependence Plots](#2-pdp--partial-dependence-plots)
   - 2.1 The Core Idea
   - 2.2 The Algorithm
   - 2.3 Full Numerical Example
   - 2.4 What the Plot Shape Means
   - 2.5 Two-Feature PDP (Interaction Plot)
3. [The Extrapolation Problem — Where PDP Fails](#3-the-extrapolation-problem--where-pdp-fails)
   - 3.1 What Extrapolation Means
   - 3.2 Numerical Example of Bias
   - 3.3 When It Matters Most
4. [ALE — Accumulated Local Effects](#4-ale--accumulated-local-effects)
   - 4.1 The Core Idea — The Fix
   - 4.2 The Algorithm — Step by Step
   - 4.3 Full Numerical Example (Same Data as PDP)
   - 4.4 Why ALE Stays Inside the Distribution
   - 4.5 Two-Feature ALE (Interaction)
5. [PDP vs ALE — Side by Side](#5-pdp-vs-ale--side-by-side)
6. [PDP vs ALE vs SHAP Dependence Plot](#6-pdp-vs-ale-vs-shap-dependence-plot)
7. [ICE Plots — Individual Conditional Expectation](#7-ice-plots--individual-conditional-expectation)
   - 7.1 What ICE Adds Over PDP
   - 7.2 Centred ICE (c-ICE)
   - 7.3 Worked Example
8. [Bias-Variance of PDP and ALE](#8-bias-variance-of-pdp-and-ale)
9. [Interaction Plots — 2D PDP and 2D ALE](#9-interaction-plots--2d-pdp-and-2d-ale)
10. [Practical Decision Guide](#10-practical-decision-guide)
11. [Common Mistakes](#11-common-mistakes)
12. [Interview Q&A](#12-interview-qa)
13. [Summary Card](#13-summary-card)

---

## 1. What PDP and ALE Are For

So far in this folder, we've covered:
- **Feature importance** — which features matter most (a single number per feature)
- **SHAP / LIME** — why a specific prediction was made (per-sample decomposition)

PDP and ALE answer a different question entirely:

> **"How does the model's prediction change as a feature changes?" — for the dataset as a whole**

They reveal the **shape** of the relationship between a feature and the model's output — linear, nonlinear, monotone, U-shaped, plateau, threshold effect.

```
Importance:   income matters a lot  (a number)
PDP / ALE:    as income goes from $20k to $100k, predicted default risk
              drops steeply at first, then levels off above $60k  (a curve)
```

PDP and ALE are **global** methods — they describe the model's average behaviour across the dataset, not any single prediction.

---

## 2. PDP — Partial Dependence Plots

### 2.1 The Core Idea

Pick a feature j. Ask: "If I could set everyone's feature j to the same value v, what would the model predict on average?"

Do this for every plausible value v of feature j. Plot the average prediction vs v. That curve is the partial dependence function.

```
PD(v) = E_{X_{-j}} [f(v, X_{-j})]
       = average prediction when feature j is set to v
         and all other features keep their actual values
```

The subscript X_{-j} means "all features except j."

### 2.2 The Algorithm

```
INPUT:  trained model f, dataset X with N samples, feature j, grid of values v₁,...,vₖ

For each candidate value vₗ in the grid:
  1. Make a copy of X
  2. Replace column j with vₗ for EVERY row
  3. Run model on this modified dataset: f(X with col j = vₗ)
  4. PD(vₗ) = mean of the N predictions

Plot PD(vₗ) vs vₗ
```

**Key point:** All other features stay at their actual values for each sample. You're only changing feature j. The average marginalises out everything else.

### 2.3 Full Numerical Example

**Dataset:** 5 samples. Predict house price from [size, age]. Model is already trained.

```
 #   size   age   actual price
─────────────────────────────────
 1   1000    5      $180k
 2   1500   10      $220k
 3   2000   15      $260k
 4   2500   20      $300k
 5   3000   25      $340k
```

**Computing PDP for feature `size` at v = 1500:**

```
Replace size with 1500 for all rows, keep age as-is:

 #   size    age    f(modified)
─────────────────────────────────
 1   1500     5     $210k   (model sees size=1500, age=5)
 2   1500    10     $215k   (model sees size=1500, age=10)
 3   1500    15     $220k   (model sees size=1500, age=15)
 4   1500    20     $225k   (model sees size=1500, age=20)
 5   1500    25     $230k   (model sees size=1500, age=25)

PD(size=1500) = mean(210, 215, 220, 225, 230) = $220k
```

**Repeat for size = 1000, 2000, 2500, 3000:**

```
PD(size=1000) = $190k
PD(size=1500) = $220k
PD(size=2000) = $250k
PD(size=2500) = $280k
PD(size=3000) = $310k
```

**Plotting these 5 points gives the PDP curve for `size`:**

```
Price ($k)
 310 |                                          ●
 280 |                                ●
 250 |                      ●
 220 |            ●
 190 |  ●
     └──────────────────────────────────────────
       1000  1500  2000  2500  3000    size (sqft)
```

**Reading this:** As house size increases from 1000 to 3000 sqft, predicted price increases from $190k to $310k. The relationship is approximately linear in this range.

### 2.4 What the Plot Shape Means

```
Shape                      Interpretation
──────────────────────────────────────────────────────────────────
Linear slope up           Feature has constant positive effect
Linear slope down         Feature has constant negative effect
Flat (horizontal)         Feature has no marginal effect (in this range)
Step / threshold          Model has a learned cutoff for this feature
S-curve                   Saturating effect (diminishing returns at extremes)
U-shape                   Both low and high values increase prediction
                           (non-monotone, interesting — check if realistic)
Bump in the middle        Middle values increase prediction most
Wide confidence band      High variance in the dataset for these values
```

### 2.5 Two-Feature PDP (Interaction Plot)

Vary two features simultaneously, marginalise over all others. The result is a **heatmap** or **3D surface**.

```
PD(v₁, v₂) = E_{X_{-jk}} [f(v₁, v₂, X_{-jk})]

For each pair (size, age):
  Set size=v₁ for all rows
  Set age=v₂ for all rows
  Average the predictions
```

The heatmap shows predicted price for every combination of (size, age):

```
         age=5   age=10  age=15  age=20  age=25
size=1k   $210k   $200k   $190k   $185k   $180k
size=1.5k $230k   $220k   $215k   $205k   $195k
size=2k   $255k   $245k   $235k   $225k   $215k
size=2.5k $280k   $270k   $260k   $250k   $240k
size=3k   $305k   $295k   $285k   $275k   $265k
```

**Reading the interaction plot:**

If the rows (fixed age, varying size) are all parallel, there is no interaction — size and age affect price independently. If the rows converge/diverge or cross, there is an interaction — the effect of size depends on age.

In this example, the rows are roughly parallel → no strong interaction between size and age.

---

## 3. The Extrapolation Problem — Where PDP Fails

This is PDP's critical flaw. Understanding it is essential — and it's exactly the problem ALE was designed to fix.

### 3.1 What Extrapolation Means

When you set feature j to value v for every row, you create rows where feature j = v but all other features have their original values. If j is correlated with other features, some of these combinations are **unrealistic** — they don't exist in the training data.

```
Example: features = [house_size, num_rooms]
         These are strongly correlated: big houses have many rooms.

PDP at size=3000 sqft:
  Row 1: size=3000, age=5  ← realistic (large modern house)
  Row 2: size=3000, age=10 ← realistic
  Row 3: size=3000, age=15 ← realistic
  Row 4: size=3000, age=20 ← realistic
  Row 5: size=3000, age=25 ← realistic

But if num_rooms is also a feature:
  Row 1: size=3000, num_rooms=2, age=5  ← UNREALISTIC (3000 sqft with 2 rooms?)
  Row 2: size=3000, num_rooms=3, age=10 ← UNREALISTIC
  Row 3: size=3000, num_rooms=8, age=15 ← realistic
  Row 4: size=3000, num_rooms=5, age=20 ← realistic
  Row 5: size=3000, num_rooms=10, age=25 ← realistic
```

Rows 1 and 2 are "3000 sqft houses with only 2–3 rooms" — something almost never seen in the data. The model's prediction for these rows extrapolates into a region it was never trained on.

### 3.2 Numerical Example of Bias

**Setup:** Predict income from [age, education_years]. These are correlated (r=0.65) — more education usually means more years spent studying, so people with more education are often older at career start.

```
Training data summary:
  age=25: education_years mostly 12–16
  age=45: education_years mostly 16–22
  age=65: education_years mostly 12–18

The combination (age=25, education_years=22) almost never occurs.
```

**PDP at age=25:**

All rows have their original education_years values (from all ages), paired with age=25:

```
Row from age=65 person: [age=25, education_years=18] ← unrealistic (65-year-old's
                                                        education set to young age)
Row from age=45 person: [age=25, education_years=20] ← unrealistic
```

The model has to predict income for (age=25, education_years=20) — a combination it essentially never saw during training. The prediction may be extrapolating wildly.

**Result:** PDP at age=25 is biased because it averages over unrealistic (age=25, high education) rows that inflate the predicted income.

**The true marginal effect of age=25** should be computed using only education values that realistically co-occur with age=25 (12–16 years). PDP doesn't do this.

### 3.3 When It Matters Most

```
Correlation level         PDP bias
─────────────────────────────────────────────────────
Low (|r| < 0.3)         Small — PDP is usually fine
Moderate (0.3–0.6)      Noticeable — use caution, consider ALE
High (|r| > 0.6)        Severe — use ALE instead
Very high (|r| > 0.8)   PDP can produce completely misleading curves
```

**Rule of thumb:** If any feature pair has correlation > 0.5, prefer ALE over PDP.

---

## 4. ALE — Accumulated Local Effects

### 4.1 The Core Idea — The Fix

Instead of asking "what does the model predict when I set feature j to v for everyone?", ALE asks:

> **"For samples that actually have feature j ≈ v, how does the prediction change when I increase feature j slightly?"**

Two critical differences:
1. **Only uses samples with j ≈ v** — stays within the data distribution
2. **Measures local differences** — computes derivatives (prediction change), not absolute predictions

This means ALE never asks the model to predict for unrealistic input combinations.

### 4.2 The Algorithm — Step by Step

```
INPUT: trained model f, dataset X, feature j, number of bins B

Step 1 — Bin the feature
  Sort samples by x_j value
  Divide into B equal-frequency bins: [z₀, z₁], [z₁, z₂], ..., [z_{B-1}, zB]
  Each bin contains roughly N/B samples

Step 2 — For each bin [zₗ₋₁, zₗ]:
  Take only the samples whose x_j falls in this bin: S_l = {i : zₗ₋₁ ≤ xᵢⱼ < zₗ}

  For each sample i in S_l:
    Create two versions:
      x_i,high = sample i with feature j set to the RIGHT edge zₗ
      x_i,low  = sample i with feature j set to the LEFT edge zₗ₋₁

    local_effect_i = f(x_i,high) − f(x_i,low)
    (how much does the prediction change when we nudge j from left to right edge?)

  bin_effect_l = mean(local_effect_i) for all i in S_l
               = average local derivative in this bin

Step 3 — Accumulate (integrate) bin effects
  ALE(v) = Σ_{l: zₗ ≤ v} bin_effect_l   (running sum)

Step 4 — Centre
  ALE(v) = ALE(v) − mean(ALE(X_j))
  (shift so the average ALE value is zero — makes interpretation cleaner)
```

**Why this avoids extrapolation:**

In Step 2, when we compute local effects for a sample in bin [zₗ₋₁, zₗ], we only change feature j from zₗ₋₁ to zₗ — a small nudge within a realistic range. All other features (correlated or not) keep their original values for that sample. Since we're only using samples that naturally have j near v, the pairs (j=v, other_features) are all realistic combinations observed in the training data.

### 4.3 Full Numerical Example (Same Data as PDP)

**Same dataset:** 5 samples, features [size, age], predict house price.

```
 #   size   age   price
────────────────────────────────────
 1   1000    5     $180k
 2   1500   10     $220k
 3   2000   15     $260k
 4   2500   20     $300k
 5   3000   25     $340k
```

**Computing ALE for `size` with B=2 bins:**

Sorted by size, split into 2 bins:
```
Bin 1: [1000, 2000]  — samples 1, 2, 3
Bin 2: [2000, 3000]  — samples 4, 5
```

Wait — sample 3 has size=2000 which is the bin boundary. Let's use 4 bins for clarity:

**B=4 bins: [1000,1500], [1500,2000], [2000,2500], [2500,3000]**

```
Bin 1 [1000, 1500] — sample 1 (size=1000) and sample 2 (size=1500):

  Sample 1: size=1000 → size=1500 change:
    f(size=1500, age=5) − f(size=1000, age=5)
    = $210k − $180k = +$30k

  Sample 2: size=1500 → ??? (at right edge of its bin)
  [For simplicity, put sample 2 in next bin]

  Bin 1 contains sample 1 only.
  bin_effect_1 = +$30k (for 500 sqft increase in this range)
  Per-unit effect = +$30k / 500 sqft = $60 per sqft

Bin 2 [1500, 2000] — sample 2 (size=1500) and sample 3 (size=2000):

  Sample 2: f(size=2000, age=10) − f(size=1500, age=10)
            = $245k − $215k = +$30k

  Sample 3: f(size=2000, age=15) − f(size=1500, age=15)
            = $250k − $220k = +$30k

  bin_effect_2 = mean(30, 30) = +$30k per 500 sqft

Bin 3 [2000, 2500] — sample 4:
  f(size=2500, age=20) − f(size=2000, age=20)
  = $290k − $260k = +$30k
  bin_effect_3 = +$30k

Bin 4 [2500, 3000] — sample 5:
  f(size=3000, age=25) − f(size=2500, age=25)
  = $340k − $300k = +$40k  (slightly higher effect at large sizes)
  bin_effect_4 = +$40k
```

**Accumulate:**

```
ALE(size=1000) = 0                          (start at zero)
ALE(size=1500) = +$30k                      (after bin 1)
ALE(size=2000) = +$30k + $30k = +$60k      (after bin 2)
ALE(size=2500) = +$60k + $30k = +$90k      (after bin 3)
ALE(size=3000) = +$90k + $40k = +$130k     (after bin 4)
```

**Centre (subtract mean):**

```
mean ALE = (0 + 30 + 60 + 90 + 130) / 5 = 62k
ALE_centred(1000) = 0   − 62 = −$62k
ALE_centred(1500) = 30  − 62 = −$32k
ALE_centred(2000) = 60  − 62 = −$2k
ALE_centred(2500) = 90  − 62 = +$28k
ALE_centred(3000) = 130 − 62 = +$68k
```

**Reading the ALE curve:**

```
ALE ($k)
 +68 |                                          ●
 +28 |                                ●
  −2 |                      ●
 −32 |            ●
 −62 |  ●
      └──────────────────────────────────────────
        1000  1500  2000  2500  3000    size (sqft)
```

**Interpretation:** Compared to the average-sized house:
- A 1000 sqft house is predicted $62k below average
- A 3000 sqft house is predicted $68k above average
- The effect is roughly linear and monotone

**Why ALE doesn't extrapolate:**

In Bin 1, when we compute the effect for sample 1 (size=1000, age=5), we only nudge size from 1000 to 1500 — we never put a large house (size=3000) together with age=5. The computation only uses realistic local variations.

### 4.4 Why ALE Stays Inside the Distribution

```
PDP at size=3000:
  Takes ALL samples (including old, small houses) and sets size=3000
  → Creates (size=3000, age=25) but also (size=3000, age=5) — unrealistic!
  → Model interpolates/extrapolates into unseen regions

ALE at size=3000:
  Only uses samples that actually have size near 3000 (bin [2500,3000])
  → These samples naturally have large age values too (old houses tend to be larger)
  → No unrealistic combinations are created
  → Local effect = how does prediction change when size nudges from 2500 to 3000
    for samples that actually have size ≈ 2500–3000
```

### 4.5 Two-Feature ALE (Interaction)

The 2D ALE measures the **interaction effect**: how much does the joint effect of two features deviate from the sum of their individual main effects?

```
ALE₂D(v₁, v₂) = interaction between feature j (=v₁) and feature k (=v₂)

Positive values: features j and k have a synergistic interaction
                 (together they boost prediction more than individually)
Negative values: features j and k have a cancelling interaction
                 (together they matter less than individually)
Near zero: features j and k act independently
```

This is the 2D ALE plot — a heatmap where the colour represents the pure interaction effect (after removing the two main effects).

---

## 5. PDP vs ALE — Side by Side

| Property | PDP | ALE |
|---|---|---|
| **What it computes** | E[f(v, X_{-j})] — average prediction at fixed v | Accumulated local derivatives — centred |
| **How it handles absent features** | Uses all rows (ignores correlation) | Only uses rows where j ≈ v |
| **Extrapolation** | ✅ Yes — creates unrealistic combinations | ❌ None — stays in data distribution |
| **Correct with correlated features** | ❌ No — biased | ✅ Yes |
| **Interpretability of y-axis** | Absolute prediction scale | Deviation from average effect (relative) |
| **Centred** | Not by default | Yes (always centred at 0) |
| **Computation cost** | N model evaluations per grid point | N model evaluations per bin |
| **Parameter** | Grid of v values | Number of bins B |
| **When to use** | Features are uncorrelated | Features are correlated |
| **Variance** | Low | Higher (more sensitive to bin count) |

**The single key decision:**

```
Are features correlated (|r| > 0.5)?

YES → Use ALE
NO  → PDP is fine (and slightly simpler to interpret)
```

### Numerical Divergence — When They Give Different Answers

This is the key diagnostic: if PDP and ALE give very different curves for the same feature, that feature is correlated with others, and **ALE is more trustworthy**.

```
Example: predicting income from [age, education_years] (r=0.65)

PDP(age):    flat from age=25 to age=35, then rises steeply
             (because at age=25, PDP includes high-education rows that
              artificially inflate the young-age prediction)

ALE(age):    steady increase from 25 to 65, modest slope
             (uses only realistic age-education combinations per bin)

The truth: ALE is correct. PDP's flatness at young ages is an artifact
           of pairing young age with old people's education levels.
```

---

## 6. PDP vs ALE vs SHAP Dependence Plot

All three show how a feature's value relates to the model's prediction, but from different angles:

| | PDP | ALE | SHAP Dependence Plot |
|---|---|---|---|
| **Type of average** | Average prediction | Accumulated local derivatives | Per-sample SHAP values (no averaging) |
| **Shows heterogeneity** | ❌ (averages it away) | ❌ (averages within bins) | ✅ (one dot per sample) |
| **Shows individual variation** | ❌ | ❌ | ✅ |
| **Interaction detection** | Via 2D PDP | Via 2D ALE | Via colour dimension in dependence plot |
| **Extrapolation** | ❌ Has it | ✅ None | Depends on SHAP variant |
| **Y-axis meaning** | Absolute predicted value | Deviation from average | SHAP value for this feature |
| **Influenced by other features** | Yes (marginalises them) | Yes (marginalises them) | No (each SHAP value is per-sample) |

**When to use which:**

```
PDP:                Features are uncorrelated, you want average effect
ALE:                Features are correlated, you want average effect
SHAP Dependence:    You want the distribution of effects, not just average
                    You want to see heterogeneity
                    You want to detect interactions via colour
```

---

## 7. ICE Plots — Individual Conditional Expectation

### 7.1 What ICE Adds Over PDP

ICE (Friedman & Popescu, 2008) is an extension of PDP that shows each individual sample's prediction curve instead of just the average.

```
For PDP: at each v, average all N predictions → one average curve
For ICE: at each v, keep all N predictions separate → N individual curves
```

**Why this matters:** PDP's single curve can hide important heterogeneity. If some customers become riskier as income increases while others become less risky, the average might be flat — completely hiding the interaction.

### 7.2 Centred ICE (c-ICE)

To make ICE plots easier to read, each line is centred at a reference value (usually the minimum of feature j):

```
c-ICE_i(v) = ICE_i(v) − ICE_i(v_min)

Now all lines start at 0 at the leftmost x value.
The spread of lines shows heterogeneity.
Parallel lines → no interaction (feature j's effect is the same for everyone)
Diverging lines → interaction (feature j's effect depends on other features)
```

### 7.3 Worked Example

**Dataset:** Predict loan default from [income, age]. 4 samples.

```
At income = $30k:       predictions = [0.80, 0.72, 0.65, 0.88]
At income = $50k:       predictions = [0.60, 0.61, 0.50, 0.72]
At income = $70k:       predictions = [0.42, 0.55, 0.35, 0.58]
At income = $100k:      predictions = [0.25, 0.52, 0.20, 0.45]
```

**PDP (average):**

```
income   PDP
$30k     0.76
$50k     0.61
$70k     0.48
$100k    0.36
```

A smooth downward curve — more income = lower default risk.

**ICE (individual lines):**

```
income   Sample1  Sample2  Sample3  Sample4
$30k       0.80     0.72     0.65     0.88
$50k       0.60     0.61     0.50     0.72
$70k       0.42     0.55     0.35     0.58
$100k      0.25     0.52     0.20     0.45
```

Now we see: Samples 1 and 3 drop steeply with income. Samples 2 and 4 drop much more slowly. The average hides this split. There may be an interaction between income and age — older customers (samples 2,4) are less responsive to income changes.

**c-ICE (centred at $30k):**

```
income   Sample1  Sample2  Sample3  Sample4
$30k       0.00     0.00     0.00     0.00
$50k      −0.20    −0.11    −0.15    −0.16
$70k      −0.38    −0.17    −0.30    −0.30
$100k     −0.55    −0.20    −0.45    −0.43
```

c-ICE makes the different slopes obvious: Sample 2 has a much flatter slope. PDP's average line would show none of this.

---

## 8. Bias-Variance of PDP and ALE

### PDP Bias-Variance

```
Bias:
  Main source: correlated features → extrapolation → biased average
  Increases as: feature correlations increase
  Does NOT decrease with sample size (systematic, not random)

Variance:
  Source: small dataset → noisy average per grid point
  Decreases as: N increases (more samples → more stable average)
  Also: the model itself has variance if it was trained on limited data
```

### ALE Bias-Variance

```
Bias:
  Source: bin width too large → bin effects don't capture local curvature
  Coarse bins can miss non-linearities (e.g., a sharp threshold within a bin
  gets smoothed out by the bin average)
  Fewer bins = more bias

Variance:
  Source: few samples per bin → noisy local effect estimate
  More bins = fewer samples per bin = higher variance per bin
  The accumulated sum amplifies variance from earlier bins

Bias-Variance Tradeoff on B (number of bins):
  Fewer bins (small B):  more bias (misses curvature), less variance
  More bins (large B):   less bias, more variance
  Typical default: B=10 or B=20 is usually a good balance
```

**Practical guidance:**

```
Dataset size   Recommended B
< 500          5–10
500–5000       10–20
> 5000         20–50
```

---

## 9. Interaction Plots — 2D PDP and 2D ALE

Both methods can be extended to show interactions between two features.

### 2D PDP

```
For each pair (v₁, v₂) in a grid:
  Set feature j = v₁ for all rows
  Set feature k = v₂ for all rows
  PD(v₁, v₂) = mean prediction

Result: a 2D heatmap showing average prediction for each (j,k) combination
```

**Reading a 2D PDP:**

If the heatmap rows are parallel (same pattern across different k values), features j and k act independently. If the pattern changes across rows (e.g., j has a strong positive effect only when k is high), there is an interaction.

### 2D ALE

More complex — measures the **pure interaction** (removing both main effects):

```
ALE₂D(v₁, v₂) = interaction effect of (j=v₁, k=v₂)
                = how much the joint effect exceeds sum of individual effects

Positive → synergistic (both together is more powerful)
Negative → cancelling (together they partially cancel)
Near zero → independent main effects, no interaction
```

The 2D ALE is harder to compute (requires double binning) but more interpretable: a flat zero heatmap means no interaction; any deviation from zero reveals interaction patterns.

---

## 10. Practical Decision Guide

```
Question: How does feature X affect predictions on average?
│
├── Are features correlated? (check |r| > 0.5)
│   │
│   ├── NO (uncorrelated)
│   │   └── PDP is fine and simpler to interpret
│   │       Use ICE alongside PDP to check for heterogeneity
│   │
│   └── YES (correlated)
│       └── Use ALE — PDP will be biased
│
├── Do you want per-sample variation (not just average)?
│   └── Use ICE (with PDP) or SHAP dependence plot
│
├── Do you want to detect and quantify interactions?
│   └── Use 2D PDP (quick visual) or 2D ALE (pure interaction measure)
│
└── Is the relationship between feature and prediction the focus?
    Not the importance rank, but the shape (linear? threshold? nonlinear?)
    └── PDP/ALE — this is exactly what they are designed for
```

---

## 11. Common Mistakes

**Mistake 1: Using PDP when features are correlated**

Always check pairwise correlations before using PDP. If any feature pair has |r| > 0.5, PDP may give misleading curves. Use ALE.

**Mistake 2: Interpreting PDP/ALE as causal**

PDP(size=3000) = $310k doesn't mean "if you build a 3000 sqft house, it will sell for $310k." It means the model predicts $310k on average when size=3000. The model may have learned spurious correlations (e.g., large houses in this dataset happen to be in expensive neighbourhoods).

**Mistake 3: Ignoring ICE plots**

PDP is the average of ICE curves. A flat PDP can hide two steep opposing ICE trends that cancel out. Always look at ICE alongside PDP.

**Mistake 4: Choosing B too small for ALE**

With very few bins (B=3), ALE misses within-bin non-linearities. A sharp threshold effect spanning one bin gets averaged away. Use at least B=10 for most datasets.

**Mistake 5: Reading ALE y-axis as absolute predictions**

ALE values are centred deviations, not absolute predicted values. ALE(age=25) = −0.05 means: for people aged 25, the model predicts 5 percentage points below the average risk level. It doesn't mean the predicted probability is 5%.

**Mistake 6: Using 2D PDP to claim there's no interaction**

A 2D PDP that looks like parallel contours suggests no interaction, but this visual can be misleading with correlated features. Use 2D ALE for proper interaction detection.

---

## 12. Interview Q&A

**Q: What is a Partial Dependence Plot and what does it show?**

A PDP shows the marginal effect of one feature on the model's prediction, averaging out all other features. For each candidate value v of the feature, it sets that feature to v for every sample in the dataset, runs the model, and averages the predictions. The result is a curve showing how the average prediction changes as the feature varies.

---

**Q: What is the extrapolation problem in PDP?**

When features are correlated, setting feature j to an extreme value v while keeping all other features at their original values creates input combinations that don't exist in the training data. For example, if house size and number of rooms are correlated, PDP at size=3000 sqft might use rows with only 2 rooms — a combination the model was never trained on. The model extrapolates to regions outside its training distribution, producing biased average predictions.

---

**Q: How does ALE fix the extrapolation problem?**

Instead of setting feature j to a fixed value for all rows, ALE computes local derivatives within narrow bins. For bin [z₁, z₂], it only uses samples that actually have j in that range, computes the prediction change when j is nudged from z₁ to z₂ for each sample, then averages those local changes. Since it uses only real samples from within each bin, it never creates unrealistic feature combinations. The local effects are then accumulated (integrated) to give the total effect across the feature's range.

---

**Q: When would PDP and ALE give very different results, and what does that tell you?**

Large divergence between PDP and ALE indicates that the feature being visualised is correlated with other features. The more correlated, the bigger the divergence. The divergence tells you that PDP is biased (extrapolating into unrealistic regions) and ALE is more trustworthy. This divergence is itself a diagnostic signal for feature correlation.

---

**Q: What is an ICE plot and why is it useful?**

An ICE (Individual Conditional Expectation) plot shows the prediction curve for each individual sample, rather than averaging into a single PDP curve. It reveals heterogeneity that PDP hides — for example, if the feature has a positive effect for some customers and a negative effect for others, the PDP curve (average) might be flat while ICE curves show two distinct groups diverging. ICE plots are how you detect interactions with other features.

---

**Q: PDP and SHAP dependence plots both show how a feature relates to predictions. How are they different?**

PDP shows the average prediction at each feature value, marginalising over other features. SHAP dependence plots show the per-sample SHAP value for the feature vs its actual value. PDP averages away individual variation; SHAP dependence plots preserve it (each dot is one sample). Additionally, SHAP dependence plots can reveal interactions by colouring dots by a second feature — you can see if the SHAP value of feature j depends on feature k's value. PDP requires a separate 2D PDP to see the same interaction.

---

**Q: What is the ALE bias-variance trade-off when choosing the number of bins B?**

Fewer bins → larger bins → each bin averages over a wider range of feature values → the local effect estimate is smooth but may miss sharp non-linearities within the bin (bias). More bins → smaller bins → captures fine-grained curvature (less bias) → but fewer samples per bin → noisier local effect estimates (variance). The accumulated sum in ALE means variance from early bins compounds into later bins. The optimal B is typically 10–20 for moderate datasets.

---

**Q: Can you use PDP to determine causality? (E.g., "raising income will reduce default risk")**

No. PDP shows the model's learned relationship between income and predicted default risk. The model may have learned this pattern from the data, but the data itself may reflect spurious correlations (e.g., higher income areas have better financial education, lower stress, better access to credit — the income effect in the PDP is actually a joint effect of all correlated factors). PDP describes the model's behaviour, not the real-world causal mechanism. Causal claims require controlled experiments or causal modelling, not PDP.

---

## 13. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PDP and ALE — KEY FACTS                                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  PURPOSE                                                                 │
│    Show how a feature's value relates to predictions — the shape        │
│    of the relationship (linear? threshold? U-shaped?)                   │
│    Global methods: describe the model's average behaviour               │
│                                                                          │
│  PDP                                                                     │
│    Algorithm: set feature j=v for all rows → average predictions        │
│    Y-axis: average predicted value (absolute)                           │
│    Problem: extrapolates into unrealistic combinations when features    │
│             are correlated                                               │
│    Use when: features are uncorrelated (|r| < 0.5)                      │
│                                                                          │
│  ALE                                                                     │
│    Algorithm: within each bin, compute local derivatives → accumulate   │
│    Y-axis: deviation from average effect (centred at 0)                 │
│    Advantage: never creates unrealistic feature combinations             │
│    Use when: features are correlated (always preferred over PDP then)   │
│    Key parameter: B (bins) — default 10–20                              │
│                                                                          │
│  ICE PLOTS                                                               │
│    PDP per individual sample (N curves instead of 1)                    │
│    Use: reveal heterogeneity and interactions that PDP averages away     │
│    c-ICE: centred version — all lines start at 0                        │
│    Parallel lines → no interaction                                       │
│    Diverging lines → interaction with another feature                   │
│                                                                          │
│  2D PDP / 2D ALE                                                         │
│    Vary two features simultaneously → heatmap                           │
│    2D PDP: shows average prediction for each (j,k) combination          │
│    2D ALE: shows pure interaction effect (after removing main effects)  │
│                                                                          │
│  KEY RULE                                                                │
│    Correlated features? → ALE                                            │
│    Uncorrelated?        → PDP (simpler)                                  │
│    Want heterogeneity?  → ICE or SHAP dependence plot                  │
│                                                                          │
│  WHAT NEITHER CAN DO                                                     │
│    ✗ Explain a single prediction (use SHAP or LIME)                     │
│    ✗ Show per-sample variation (use ICE or SHAP dependence)             │
│    ✗ Make causal claims (they describe the model, not reality)          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Friedman (2001)** — *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics. — Original PDP paper.
- **Goldstein et al. (2015)** — *Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation.* — ICE plots.
- **Apley & Zhu (2020)** — *Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models.* Journal of the Royal Statistical Society B. — Original ALE paper.
- **Molnar (2022)** — *Interpretable Machine Learning*, Chapter 8 — https://christophm.github.io/interpretable-ml-book
- **sklearn PDP docs** — https://scikit-learn.org/stable/modules/partial_dependence.html
- **Companion** — `4_SHAP.md` for SHAP dependence plots.
- **Companion** — `8_Correlated_Features.md` for when correlations make PDP dangerous.
