# ICE Plots — Individual Conditional Expectation

> PDP shows the average. ICE shows everyone. This guide covers everything ICE adds, how to read it, and when the average is hiding the most important story.

---

## Table of Contents

1. [The Problem with Averages](#1-the-problem-with-averages)
2. [What ICE Is](#2-what-ice-is)
3. [The ICE Algorithm](#3-the-ice-algorithm)
4. [Full Numerical Example — From Scratch](#4-full-numerical-example--from-scratch)
5. [Reading an ICE Plot — Pattern Guide](#5-reading-an-ice-plot--pattern-guide)
6. [Centred ICE (c-ICE) — The Cleaner View](#6-centred-ice-c-ice--the-cleaner-view)
   - 6.1 Why Centre?
   - 6.2 Algorithm and Interpretation
   - 6.3 Numerical Example
7. [Derivative ICE (d-ICE) — Detecting Interactions](#7-derivative-ice-d-ice--detecting-interactions)
8. [Ceteris Paribus Plots — ICE for One Sample](#8-ceteris-paribus-plots--ice-for-one-sample)
9. [ICE vs PDP — When They Diverge (and What It Means)](#9-ice-vs-pdp--when-they-diverge-and-what-it-means)
10. [ICE vs SHAP Dependence Plots](#10-ice-vs-shap-dependence-plots)
11. [Bias-Variance of ICE](#11-bias-variance-of-ice)
12. [ICE for Correlated Features](#12-ice-for-correlated-features)
13. [Practical Tips — Making ICE Useful](#13-practical-tips--making-ice-useful)
14. [Interview Q&A](#14-interview-qa)
15. [Summary Card](#15-summary-card)

---

## 1. The Problem with Averages

Partial Dependence Plots show the **average** effect of a feature across the dataset. Averaging can destroy exactly the information you need.

Consider a model that predicts loan default risk from income. Suppose:

```
For young applicants (age < 35):
  High income strongly reduces risk (steep negative slope)
  
For old applicants (age > 55):
  High income barely matters (nearly flat)

Middle-aged applicants:
  Moderate slope in between
```

The PDP averages across all three groups:

```
PDP (average across all ages):  moderate negative slope, appearing linear

What PDP hides:
  - The strong income protection for young people
  - The irrelevance of income for older people
  - The fact that these are TWO different sub-populations
```

A flat PDP can result from two steep lines cancelling. A moderate PDP slope can hide one steep and one flat group. An upward PDP can hide one strongly positive and one moderately negative group that don't cancel.

**ICE keeps every individual's curve separate.** It reveals exactly what the average is hiding.

---

## 2. What ICE Is

ICE (Individual Conditional Expectation) was introduced by Goldstein et al. (2015) as a direct extension of PDP.

**PDP:** For each value v of feature j, set j=v for all N samples, predict, average → one point on the curve.

**ICE:** For each value v of feature j, set j=v for all N samples, predict → keep all N predictions separate → N curves on the same plot.

Each ICE curve shows how **one individual's** prediction would change if their value of feature j changed, holding all their other features constant.

```
ICE_i(v) = f(v, x_i,−j)

where x_i,−j = all features of sample i EXCEPT feature j
      v       = the candidate value of feature j

For each sample i, ICE_i is a curve over the grid of v values.
PDP = (1/N) × Σᵢ ICE_i   ← PDP is the average ICE curve
```

---

## 3. The ICE Algorithm

```
INPUT: trained model f, dataset X with N samples, feature j,
       grid of K values v₁, ..., vₖ

FOR each sample i = 1 to N:
  FOR each grid value vₗ:
    Create modified row: x_i' = x_i with column j replaced by vₗ
    Get prediction: ICE_i(vₗ) = f(x_i')
  Plot the curve (v₁, ICE_i(v₁)), ..., (vₖ, ICE_i(vₖ))

PDP(vₗ) = (1/N) × Σᵢ ICE_i(vₗ)   ← overlay the average
```

**Computational cost:**

```
N samples × K grid points × 1 model inference = N × K total predictions

For N=1000 samples and K=50 grid points: 50,000 predictions
For N=10000 and K=100: 1,000,000 predictions

Same cost as PDP — you get ICE for free when computing PDP.
Always compute and store ICE; you can always average to get PDP.
```

---

## 4. Full Numerical Example — From Scratch

**Dataset:** 6 customers. Predict loan default probability from [income, age, debt_ratio].

```
 #   income($k)  age   debt_ratio   default_prob (model output)
────────────────────────────────────────────────────────────────
 1      30        25       0.80         0.78
 2      50        35       0.55         0.52
 3      70        45       0.40         0.33
 4      90        55       0.30         0.21
 5      45        28       0.70         0.65
 6      65        42       0.35         0.28
```

**Computing ICE for `income` on a grid: [30, 45, 60, 75, 90]**

For each sample, replace income with each grid value, keep age and debt_ratio unchanged:

```
Grid value v = 30:
  Sample 1: [income=30, age=25, debt=0.80] → f = 0.78  (no change, this is original)
  Sample 2: [income=30, age=35, debt=0.55] → f = 0.68  (income forced down from 50)
  Sample 3: [income=30, age=45, debt=0.40] → f = 0.58  (income forced way down)
  Sample 4: [income=30, age=55, debt=0.30] → f = 0.45  (income forced way down)
  Sample 5: [income=30, age=28, debt=0.70] → f = 0.75  (similar to original)
  Sample 6: [income=30, age=42, debt=0.35] → f = 0.55  (income forced down)
  PDP(30) = mean = 0.63

Grid value v = 60:
  Sample 1: [income=60, age=25, debt=0.80] → f = 0.62  (income up from 30)
  Sample 2: [income=60, age=35, debt=0.55] → f = 0.45  (income up a bit)
  Sample 3: [income=60, age=45, debt=0.40] → f = 0.35  (small change, near original)
  Sample 4: [income=60, age=55, debt=0.30] → f = 0.24  (income up from 30)
  Sample 5: [income=60, age=28, debt=0.70] → f = 0.55  (income up from 45)
  Sample 6: [income=60, age=42, debt=0.35] → f = 0.30  (small change)
  PDP(60) = mean = 0.42

Grid value v = 90:
  Sample 1: [income=90, age=25, debt=0.80] → f = 0.52  (high income helps some)
  Sample 2: [income=90, age=35, debt=0.55] → f = 0.28  (income nearly doubles)
  Sample 3: [income=90, age=45, debt=0.40] → f = 0.18  (income triples)
  Sample 4: [income=90, age=55, debt=0.30] → f = 0.21  (no change — original)
  Sample 5: [income=90, age=28, debt=0.70] → f = 0.40  (big income boost)
  Sample 6: [income=90, age=42, debt=0.35] → f = 0.19  (big boost)
  PDP(90) = mean = 0.30
```

**Full ICE table (each row = one sample, each column = one grid value):**

```
Sample    v=30   v=45   v=60   v=75   v=90    Slope shape
──────────────────────────────────────────────────────────────────
  1       0.78   0.72   0.62   0.55   0.52   Moderate decrease, flattens
  2       0.68   0.56   0.45   0.35   0.28   Steep, near-linear decrease
  3       0.58   0.47   0.35   0.26   0.18   Steep, near-linear decrease
  4       0.45   0.34   0.24   0.21   0.21   Decreases then FLAT (plateau)
  5       0.75   0.65   0.55   0.47   0.40   Moderate, consistent decrease
  6       0.55   0.43   0.30   0.23   0.19   Steep decrease
──────────────────────────────────────────────────────────────────
PDP       0.63   0.53   0.42   0.35   0.30   Moderate decrease (average)
```

**Key observations from ICE that PDP misses:**

1. **Sample 1** (young, high debt): income reduces risk but much less than for others. Even at income=$90k, risk stays at 0.52. Income can't overcome this person's age=25 + debt=0.80 combination.

2. **Sample 4** (older, low debt): risk decreases with income up to $60k, then **plateaus**. The model has learned that for older low-debt applicants, income above $60k provides no additional safety. PDP doesn't show this plateau.

3. **Samples 2, 3, 6**: steep, nearly linear decreases — income is highly effective for these profiles.

**The heterogeneity is the story.** Income works very differently depending on the customer's profile. PDP's moderate slope hides all of this.

---

## 5. Reading an ICE Plot — Pattern Guide

```
Pattern               What It Means
──────────────────────────────────────────────────────────────────────────

All lines parallel,   No interaction. The feature's effect is the same
same slope            for everyone. PDP fully captures the story.
                      [safe to use PDP alone]

All lines parallel,   Same no-interaction story, just different risk levels.
different heights     Height difference = effect of other features.
                      [safe to use PDP alone; height = other features]

Lines with same       All agree on direction (e.g., higher income = lower risk)
shape but fan out     but magnitude differs. Interaction with another feature
at one end            exists at that end of the x-axis.
                      [PDP misleading at the fan region]

Lines cross           STRONG interaction. For some samples, increasing x
(crossing lines)      increases prediction. For others, it decreases it.
                      PDP may be near-flat despite all individuals having
                      steep effects — they cancel.
                      [PDP very misleading here]

Lines plateau at      Feature effect saturates. A threshold exists above
different values      which feature j stops mattering for different subgroups.
                      [PDP misses the threshold and subgroup difference]

One or few outlier    Those individuals have very unusual feature profiles.
lines far from rest   The feature interacts with other features uniquely
                      for those samples.
                      [PDP hides outlier behaviour]
```

---

## 6. Centred ICE (c-ICE) — The Cleaner View

### 6.1 Why Centre?

Raw ICE plots can be hard to read because the lines start at different heights — reflecting different baseline risks. Two lines that are actually parallel (same slope, same income effect) look different because one person has overall higher risk.

Centring removes the baseline difference and lets you see **the shape of each individual's response** to the feature.

### 6.2 Algorithm and Interpretation

```
c-ICE_i(v) = ICE_i(v) − ICE_i(v_ref)

where v_ref is a reference value — typically:
  v_ref = min(feature j values) — all lines start at 0 at the left edge
  or: v_ref = mean(feature j values) — all lines centred at the mean

After centring:
  If all lines are parallel → no interaction with this feature
  If lines diverge as v increases → interaction that grows with feature value
  If lines cross → strong two-directional interaction
```

**Key property:**

```
PDP (centred) = mean of c-ICE curves = flat line through 0 at v_ref
             → the centred PDP is always 0 at the reference point

Spread of c-ICE curves at any v = measure of interaction strength at v
```

### 6.3 Numerical Example

Using the previous data, centring at v=30 (minimum):

```
c-ICE_i(v) = ICE_i(v) − ICE_i(30)

Sample    v=30   v=45   v=60   v=75   v=90
──────────────────────────────────────────────────────────
  1        0    −0.06  −0.16  −0.23  −0.26  (flattens)
  2        0    −0.12  −0.23  −0.33  −0.40  (steeper)
  3        0    −0.11  −0.23  −0.32  −0.40  (similar to sample 2)
  4        0    −0.11  −0.21  −0.24  −0.24  (PLATEAU — stops at v=75)
  5        0    −0.10  −0.20  −0.28  −0.35  (moderate)
  6        0    −0.12  −0.25  −0.32  −0.36  (steeper)
```

Now all lines start at 0. It's immediately clear:
- **Sample 1 (blue, flatter line):** income has smaller effect — flattens after $60k
- **Sample 4 (green, plateau):** income stops mattering at $75k
- **Samples 2, 3, 6 (steeper):** income consistently reduces risk

The interaction structure is now visually unambiguous. Without centring, the different starting heights (0.78 vs 0.45) for samples 1 and 4 made their slopes harder to compare.

---

## 7. Derivative ICE (d-ICE) — Detecting Interactions

For more rigorous interaction detection, you can compute the derivative of each ICE curve:

```
d-ICE_i(v) = d ICE_i(v) / dv
           ≈ [ICE_i(v + δ) − ICE_i(v)] / δ    (numerical approximation)

High variance of d-ICE across samples at a specific v:
  → The slope of the income effect varies across individuals at income=v
  → Strong interaction at that value

Var(d-ICE at v) is a formal interaction measure:
  Low variance → all individuals respond similarly to feature j at value v
  High variance → heterogeneous response → interaction present
```

In practice, d-ICE plots are used to pinpoint **where** in the feature's range interactions are most prominent.

---

## 8. Ceteris Paribus Plots — ICE for One Sample

A **Ceteris Paribus (CP) plot** is simply one ICE curve — the prediction curve for a single specific individual as one feature varies.

```
CP_i(v) = ICE_i(v) = f(v, x_i,−j)
          for one specific sample i
```

"Ceteris paribus" is Latin for "all other things being equal." It asks: for this person, how would their risk change if only this feature changed?

**When to use CP plots:**

- Local explanation: "Show me how my prediction would change if I reduced my debt_ratio"
- Customer-facing tools: interactive slider adjusting one feature, showing prediction change in real time
- Counterfactual analysis: find the value of feature j where the prediction crosses the decision threshold

**CP plot vs Counterfactual:**

```
CP plot: shows the FULL curve f(v, x_i,−j) over all possible v values
         → "Here is how your risk changes as income varies from $20k to $120k"
         → Visual, shows the full range

Counterfactual: finds the SPECIFIC point where prediction crosses threshold
                → "You need income ≥ $68k to be approved"
                → Actionable, single answer

Use CP for exploration; use counterfactual for the specific actionable answer.
```

---

## 9. ICE vs PDP — When They Diverge (and What It Means)

The divergence between ICE and PDP is a diagnostic signal. Here's exactly what each scenario means:

**Case 1: ICE curves are parallel, PDP = average of parallels**

```
All ICE lines have the same slope → no interaction
PDP captures the story correctly
→ Safe to summarise with PDP alone
```

**Case 2: ICE spread around PDP, same direction**

```
All lines go the same direction (all negative slope)
but some are steeper than others
→ The feature has a consistent direction but variable magnitude
→ Interaction exists (the effect size depends on other features)
→ PDP shows the right direction but hides the variation
→ Report PDP + "effect ranges from X to Y across individuals"
```

**Case 3: ICE lines cross — the dangerous case**

```
Some lines have positive slope; others have negative slope
→ Feature j has OPPOSITE effects for different subgroups
→ PDP averages these out → may appear flat or moderate
→ PDP is ACTIVELY MISLEADING here

Example: for high-debt customers, more income reduces default risk
         for low-debt retirees, more income has no effect or slightly 
         increases risk (more spending power without enough debt management)
→ ICE shows the two groups; PDP shows a near-zero average that
   misses both effects
```

**Formal test:**

```
If Var(ICE_i(v) − ICE_i(v_ref)) > threshold for many values of v:
  → Significant heterogeneity → interaction present → PDP misleads

A quick practical test:
  Compute correlation between ICE slopes and a second feature X_k
  If |r| > 0.3: feature j interacts with feature k
```

---

## 10. ICE vs SHAP Dependence Plots

Both show how a feature's values relate to predictions per sample — but they show different things.

| Dimension | ICE | SHAP Dependence Plot |
|---|---|---|
| **Y-axis** | Raw model prediction f(v, x_{-j}) | SHAP value φⱼ(xᵢ) |
| **X-axis** | Candidate values of feature j (grid) | Actual value of feature j for each sample |
| **Each point** | One sample at one grid value | One sample at its actual feature value |
| **Counterfactual** | ✅ "What IF j=v for this person?" | ❌ "What IS the SHAP for j for this person?" |
| **Extrapolation** | ✅ Uses artificial j=v values | Uses actual observed values only |
| **Interaction detection** | Via line divergence / c-ICE | Via colour by second feature |
| **Shows interaction** | Direction of divergence | Sign of interaction (colour bands) |

**The key conceptual difference:**

ICE asks: "What would the model predict if I changed this person's feature j to value v?"
SHAP asks: "How much did this person's actual value of feature j contribute to their prediction?"

ICE is counterfactual (what if). SHAP is attributional (how much credit does the actual value get).

**Use ICE when:** you want to see the full prediction curve as a feature changes — useful for customer-facing tools, what-if analysis, finding thresholds.

**Use SHAP dependence when:** you want to understand how the model uses a feature globally — direction, magnitude, and interactions with other features at actual data points.

---

## 11. Bias-Variance of ICE

**Bias:**

ICE inherits PDP's main bias: extrapolation when features are correlated. When you set income=v for a sample where the original income was very different, all other features (which may correlate with income) stay unchanged — creating potentially unrealistic combinations.

```
Sample 3: income=70k, debt_ratio=0.40 (realistic: moderate income, moderate debt)
When we compute ICE_3(v=30):
  [income=30k, age=45, debt=0.40]
  
In the training data, people with income=30k rarely have debt_ratio=0.40
(low income people typically have higher debt ratios)
This is an unrealistic combination → the model may extrapolate
```

The same fix applies as for PDP: use ALE instead when features are correlated. ALE uses only samples within each bin, avoiding this problem.

**Variance:**

ICE has the same variance properties as PDP per grid point. Each ICE curve is an exact computation (deterministic given the model and the grid). The variance comes from the model itself — if the model was trained on a small or noisy dataset, the ICE curves will be noisy.

**For small datasets:**

```
With N=50 samples: ICE curves represent very few individuals — the spread you
see may reflect noise as much as true heterogeneity.

With N=1000+ samples: patterns in the spread are meaningful.

Rule: with fewer than 100 samples, interpret ICE patterns with caution.
```

---

## 12. ICE for Correlated Features

When features are correlated, ICE has the same extrapolation problem as PDP. However, ICE makes the problem **visible in a way PDP doesn't**.

**How ICE reveals the extrapolation problem:**

When income and debt_ratio are correlated (r=−0.55), varying income to extreme values (v=150k) for samples that have moderate debt_ratio creates an unrealistic scenario. The model's prediction for this combination reflects extrapolation, not the true income effect.

In the ICE plot, samples with unusual current values of the feature (far from v) produce the most unrealistic combinations when the feature is set to v. These samples often produce outlier ICE curves that look extreme or erratic.

**The diagnostic signal:**

```
If ICE curves for samples with x_j near v look smooth and consistent
but ICE curves for samples with x_j far from v look erratic or extreme:
→ The model is extrapolating for the second group
→ Those curves are not trustworthy
→ Consider restricting ICE computation to samples whose x_j is
  not too far from the grid value being evaluated
```

**Practical approach for correlated features:**

```
Option 1: Use ALE instead (avoids extrapolation entirely)
Option 2: Use ICE but only for samples where x_j is within one IQR
          of the grid value v — restrict the comparison to realistic samples
Option 3: Show ICE with confidence-band shading based on sample density
          (wider bands = fewer samples = less reliable)
```

---

## 13. Practical Tips — Making ICE Useful

**Tip 1: Always compute ICE when you compute PDP**

ICE is free — same computation, just don't average. If PDP and ICE look very different (crossing lines, high variance of slopes), you've discovered a heterogeneity that the PDP summary would have hidden.

**Tip 2: Plot a random subsample for large N**

With N=10,000 samples, 10,000 ICE lines is unreadable. Sample 50–200 random individuals and plot those. The pattern will be representative.

**Tip 3: Colour lines by a second feature to reveal interactions**

The most useful ICE visualisation: plot ICE curves and colour each line by a second feature's value (e.g., colour by age group). If you see red lines (old customers) behaving differently from blue lines (young customers), you've identified an interaction between your x-axis feature and age.

**Tip 4: Use c-ICE (centred) for comparing slopes**

When baseline risk levels differ substantially across samples, use c-ICE. It removes the height variation so you can compare slopes cleanly.

**Tip 5: Look for the crossing point**

If lines cross in a c-ICE or raw ICE plot, identify the approximate v value where they cross. That's the threshold where the feature's effect reverses for some subgroup. This is often a highly actionable finding.

**Tip 6: Check that your grid covers the data range**

If the grid doesn't cover the actual distribution of feature j values, you're extrapolating. Always set the grid to the observed range (min to max, or 5th to 95th percentile) of feature j in your dataset.

---

## 14. Interview Q&A

**Q: What is an ICE plot and how does it relate to a PDP?**

An ICE (Individual Conditional Expectation) plot shows the prediction curve for each individual sample as a feature varies, holding all other features constant. There is one curve per sample. The PDP is the average of all ICE curves — a single summary curve. ICE shows the full distribution of individual responses; PDP compresses that into a mean.

---

**Q: Give a concrete example where PDP is misleading and ICE is not.**

If half the customers show a strong positive effect of income on predictions (income reduces default risk) and the other half show a flat effect, the PDP averages these to show a moderate slope — appearing as if income has a consistent but moderate effect. ICE would show two distinct groups: one set of steep curves and one set of flat curves. PDP hides the two-subpopulation structure entirely.

---

**Q: What is a c-ICE plot and what does it reveal?**

Centred ICE (c-ICE) subtracts each sample's ICE value at a reference point (typically the minimum feature value) so all curves start at zero. This removes the baseline risk differences between samples, making it easy to compare the *slope* of each individual's response. Parallel lines in c-ICE → no interaction. Diverging lines → interaction that grows with the feature value. Crossing lines → the feature has opposite effects for different subgroups.

---

**Q: What is the relationship between ICE and feature interactions?**

If feature j has no interaction with any other feature, all ICE curves are parallel — the feature's effect is the same for everyone. If ICE curves diverge, converge, or cross, feature j interacts with some other feature — the effect of j depends on the values of that other feature. To identify which feature drives the interaction, colour ICE curves by each other feature in turn and look for the colouring that best separates the distinct behaviour patterns.

---

**Q: What is a Ceteris Paribus plot and when would you use one over a counterfactual?**

A Ceteris Paribus (CP) plot is a single ICE curve for one specific individual — it shows how that person's prediction changes across all possible values of one feature. Use it for exploration and customer-facing what-if tools ("how would your risk change as your income changes?"). A counterfactual is more specific — it finds the minimum change to reach a target prediction. Use CP for the full picture; use counterfactuals when you need a single actionable answer.

---

**Q: Why might ICE curves be unreliable for samples with feature values far from the grid values being evaluated?**

Because setting feature j to a very different value for a sample creates an input combination that may not exist in the training data — especially if j is correlated with other features. The model extrapolates to a region it was never trained on, producing unreliable predictions. This is the same extrapolation problem as PDP. Samples with j values close to the grid point v give realistic combinations; samples with j values far from v are extrapolating.

---

## 15. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ICE PLOTS — KEY FACTS                                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  WHAT IT IS                                                              │
│    One prediction curve per sample: how would prediction change         │
│    if only feature j changed?  (all other features constant)            │
│    PDP = average of all ICE curves                                       │
│                                                                          │
│  ALGORITHM                                                               │
│    For each sample i, for each grid value v:                            │
│      ICE_i(v) = f(v, x_i,−j)   ← replace j=v, keep all other features  │
│    Plot all N curves together                                            │
│                                                                          │
│  WHAT ICE ADDS OVER PDP                                                  │
│    Heterogeneity: different individuals respond differently              │
│    Interaction detection: diverging or crossing lines = interaction      │
│    Subgroup discovery: colour by a second feature to find who           │
│    Plateau/threshold: some individuals saturate at different values     │
│                                                                          │
│  VARIANTS                                                                │
│    c-ICE: subtract each curve's value at v_ref → all start at 0        │
│           Use: compare slopes cleanly (removes baseline height diff)    │
│    d-ICE: derivative of each curve → formal interaction measure         │
│    CP plot: one ICE curve for one individual (Ceteris Paribus)          │
│                                                                          │
│  READING PATTERNS                                                        │
│    Parallel lines  → no interaction, PDP is sufficient                  │
│    Diverging       → interaction grows with feature value               │
│    Crossing lines  → STRONG interaction, PDP is misleading              │
│    Plateau         → feature saturates for some individuals             │
│                                                                          │
│  ICE vs SHAP DEPENDENCE PLOT                                            │
│    ICE: "what IF j=v?" — counterfactual curves over a grid             │
│    SHAP: "how much does actual j contribute?" — attribution at actual   │
│           observed values                                                │
│                                                                          │
│  LIMITATIONS                                                             │
│    Same extrapolation problem as PDP (correlated features)              │
│    Use ALE instead when features are correlated                         │
│    With large N: subsample 50–200 curves for readability                │
│                                                                          │
│  PRACTICAL RULES                                                         │
│    Always compute ICE when computing PDP (same cost)                    │
│    If ICE and PDP diverge: the average was hiding something important   │
│    Colour by a second feature to identify interaction partners          │
│    c-ICE for slope comparison; raw ICE for absolute levels              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Goldstein et al. (2015)** — *Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation.* Journal of Computational and Graphical Statistics. — Original ICE paper.
- **Friedman (2001)** — *Greedy Function Approximation: A Gradient Boosting Machine.* — Original PDP paper (ICE is its extension).
- **Molnar (2022)** — *Interpretable Machine Learning*, Chapter 8.1 — https://christophm.github.io/interpretable-ml-book/ice.html
- **Companion** — `6_PDP_and_ALE.md` — PDP, ALE, and the extrapolation problem ICE shares.
- **Companion** — `4_SHAP.md` — SHAP dependence plots as an alternative to ICE.
