# Correlated Features in XAI — Complete Guide

> The single most common source of wrong feature importance conclusions. How every method breaks, how to detect it, and how to fix it — with numbers.

---

## Table of Contents

1. [Why Correlation Is the Central Problem in XAI](#1-why-correlation-is-the-central-problem-in-xai)
2. [Measuring Correlation — The Full Toolkit](#2-measuring-correlation--the-full-toolkit)
   - 2.1 Pairwise Correlation
   - 2.2 VIF — Variance Inflation Factor
   - 2.3 The Pre-Analysis Protocol
3. [How Each Method Breaks — Method by Method](#3-how-each-method-breaks--method-by-method)
   - 3.1 Permutation Importance
   - 3.2 MDI / Gini Importance
   - 3.3 SHAP (Marginal)
   - 3.4 LIME
   - 3.5 PDP
   - 3.6 Drop-Column Importance
4. [The Substitute Problem — Numerical Proof](#4-the-substitute-problem--numerical-proof)
5. [The Credit Absorption Problem — Numerical Proof](#5-the-credit-absorption-problem--numerical-proof)
6. [Grouped Permutation Importance — The Fix for Correlated Groups](#6-grouped-permutation-importance--the-fix-for-correlated-groups)
   - 6.1 The Algorithm
   - 6.2 Full Numerical Example
   - 6.3 Within-Group Attribution
7. [Conditional SHAP — The Fix for SHAP](#7-conditional-shap--the-fix-for-shap)
8. [ALE — The Fix for PDP](#8-ale--the-fix-for-pdp)
9. [The Full Diagnosis and Fix Protocol](#9-the-full-diagnosis-and-fix-protocol)
10. [Worked Case Study — Same Dataset, Every Method](#10-worked-case-study--same-dataset-every-method)
11. [Interview Q&A](#11-interview-qa)
12. [Summary Card](#12-summary-card)

---

## 1. Why Correlation Is the Central Problem in XAI

Nearly every XAI method was designed assuming features are uncorrelated or weakly correlated. Real datasets almost never satisfy this assumption.

When features are correlated:
- Permutation importance **underestimates** both correlated features
- Gini MDI gives **arbitrary credit** based on which feature got the root split
- Marginal SHAP **creates unrealistic inputs** and splits credit incorrectly
- LIME **creates unrealistic perturbations** leading to unstable explanations
- PDP **extrapolates** into impossible feature combinations

The result: you can publish a feature importance report from a well-trained model that is completely misleading because two important correlated features both appear unimportant.

---

## 2. Measuring Correlation — The Full Toolkit

### 2.1 Pairwise Correlation

The simplest check: compute the correlation matrix and flag high pairs.

```
Pearson r: for continuous features
  r = Cov(A,B) / (σ_A × σ_B)

Spearman ρ: for monotone relationships, or ordinal features
  ρ = Pearson r applied to the ranks of A and B

Cramér's V: for categorical features
  V = sqrt(χ²/N × 1/min(k₁-1, k₂-1))
  where k₁, k₂ = number of categories
```

**Flagging thresholds:**

```
|r| > 0.3   → weakly correlated — usually fine
|r| > 0.5   → moderately correlated — watch PDP/SHAP
|r| > 0.7   → highly correlated — use grouped methods
|r| > 0.9   → redundant features — one may be removable
```

**Example correlation matrix (credit dataset):**

```
                credit_score  late_payments  debt_ratio  income  age
credit_score        1.00          -0.88         -0.41    0.32   0.21
late_payments       -0.88          1.00          0.38   -0.28  -0.15
debt_ratio          -0.41          0.38          1.00   -0.55  -0.18
income              0.32          -0.28         -0.55    1.00   0.43
age                 0.21          -0.15         -0.18    0.43   1.00
```

Flagged pairs (|r| > 0.7):
- `credit_score` ↔ `late_payments`: r = −0.88 ← severe
- Everything else is moderate or weak

### 2.2 VIF — Variance Inflation Factor

Pairwise correlations miss **multicollinearity** — when one feature is a linear combination of several others. VIF catches this.

```
VIF(j) = 1 / (1 − R²_j)

where R²_j = R² from regressing feature j on ALL other features

Interpretation:
  VIF = 1        → no multicollinearity
  VIF = 2        → moderate: R²_j = 0.50
  VIF = 5        → concerning: R²_j = 0.80
  VIF = 10       → severe: R²_j = 0.90
  VIF = 20       → extreme: R²_j = 0.95
```

**Worked example:**

```
Feature: income
  Regress income on [credit_score, late_payments, debt_ratio, age]
  R²  = 0.72   (these 4 features explain 72% of income's variance)
  VIF(income) = 1/(1−0.72) = 1/0.28 = 3.57  → moderate concern

Feature: credit_score
  Regress on [late_payments, debt_ratio, income, age]
  R² = 0.84
  VIF(credit_score) = 1/(1−0.84) = 6.25  → concerning

Feature: late_payments
  R² = 0.82 (mostly explained by credit_score)
  VIF(late_payments) = 1/(0.18) = 5.56  → concerning
```

**Rule:** VIF > 5 → feature importance estimates for this feature may be unreliable. VIF > 10 → importance estimates are untrustworthy. Consider grouping with correlated features.

### 2.3 The Pre-Analysis Protocol

Before running any importance or explainability method:

```
Step 1: Build correlation matrix (Pearson for continuous, Cramér's V for categorical)
Step 2: Flag all pairs with |r| > 0.7
Step 3: Compute VIF for all features
Step 4: Flag features with VIF > 5
Step 5: Build correlation groups (clusters of correlated features)
        Use hierarchical clustering on |1 − correlation matrix|
Step 6: Decide per group:
          - If one feature dominates (r > 0.95): drop the redundant one
          - If both matter: use grouped methods (see Section 6)
          - If moderately correlated (0.5–0.7): use ALE over PDP, SHAP with caution
```

---

## 3. How Each Method Breaks — Method by Method

### 3.1 Permutation Importance

**The mechanism:** When feature A is permuted (shuffled), correlated feature B is still in the data with its original values. The model uses B as a substitute for A. Performance barely drops. A appears unimportant.

Same thing happens when B is permuted — A substitutes. B also appears unimportant.

**Result:** Both A and B look unimportant despite their group being the most important predictor.

**Severity:** Proportional to correlation strength.

```
r(A,B) = 0.4:  importance of A and B are each reduced by ~20%
r(A,B) = 0.7:  reduced by ~50% each
r(A,B) = 0.9:  reduced by ~80% each — both appear nearly useless
r(A,B) = 0.99: both appear completely useless
```

### 3.2 MDI / Gini Importance

**The mechanism:** When A and B are correlated, both carry similar information. Which one the tree splits on first depends on random factors (bootstrap, feature subsampling). The one that gets the root split absorbs most of the group's credit via the n_t/N weighting. The other gets pushed to lower nodes, earning far less credit.

**Result:** The winning feature appears very important. The other appears unimportant. But which one "wins" is essentially random — it varies across random seeds.

```
RF seed=42:  credit_score = 0.48,  late_payments = 0.12
RF seed=99:  credit_score = 0.21,  late_payments = 0.39
(same dataset, same model quality, opposite conclusions)
```

### 3.3 SHAP (Marginal)

**The mechanism:** When feature A is "absent" from a coalition, its value is drawn from its marginal distribution — independently of the other present features. If B is correlated with A, this draws values of A that don't co-occur naturally with B's actual value.

For example: when credit_score is absent and late_payments = 0, marginal SHAP might draw credit_score = 450 from an older person who had many late payments. The combination (late_payments=0, credit_score=450) is rare and unrealistic. The model's prediction for this combination is extrapolation.

**Result:** The attributions between A and B reflect model behaviour in unrealistic input regions, not the true contribution of each feature within realistic data.

```
True SHAP (with conditional expectation):
  credit_score: −0.03 (conditioned on late_payments=0, both imply good credit)
  late_payments: −0.20

Marginal SHAP (default TreeSHAP):
  credit_score: −0.15 (model uses both independently)
  late_payments: −0.08

Sum is the same (efficiency holds!), but the split is different.
```

The sum is always correct. The individual attributions for correlated features are unreliable.

### 3.4 LIME

**The mechanism:** When LIME perturbs features by turning them "on" or "off," it creates combinations like (credit_score=750, late_payments=8) — a very high credit score with many late payments. This combination almost never appears in training data, so the model extrapolates. The resulting prediction for this artificial input is unreliable, which makes the regression coefficient for credit_score noisy and unstable.

**Result:** LIME's instability is amplified for correlated features. The coefficients for each correlated feature vary wildly across runs because they're being estimated from highly unrealistic perturbed inputs.

### 3.5 PDP

**The mechanism:** Setting feature A to a fixed value v for all rows keeps B at its original value, creating combinations (A=v, B=b) that may not naturally co-occur. For rows where the natural co-occurrence would be (A=high, B=high), forcing A=low while B stays high is artificial.

**Result:** PDP's curve for feature A is biased toward the prediction for unrealistic (A, B) combinations, not the true marginal effect of A. Covered in full in `6_PDP_and_ALE.md`.

**Fix:** ALE — uses only rows with A ≈ v, so B is always its realistic co-occurring value.

### 3.6 Drop-Column Importance

**The mechanism:** Drop column A. Retrain the model. Now the model can learn to use B more heavily to compensate for the absence of A. The new model's performance is not much worse because B covers most of A's information.

**Wait — is this actually a problem?**

Actually, drop-column importance handles correlations **better** than permutation importance. The performance drop when dropping A measures "how much of A's unique information is irreplaceable by the other features." If B is a near-perfect substitute, the drop is small — correctly reflecting that A is redundant given B is present. This is the right question for feature selection: "Do I need A if I already have B?"

The "problem" with drop-column is computational cost (requires retraining), not correlation handling.

---

## 4. The Substitute Problem — Numerical Proof

**Setup:** Predict house price from [size_sqft, num_rooms]. Correlation = 0.85.

```
True relationship: price increases with both size and rooms, but they
                   carry mostly the same information.
                   True combined importance: 0.60 (major factor)

Test set: 100 houses. Baseline MAE = $15k.
```

**Permute size_sqft:**

```
For each house: replace size_sqft with a random size from training distribution.
BUT: num_rooms is still intact.

Large num_rooms (say, 5) paired with random small size (1200 sqft):
  → The model knows there are 5 rooms → predicts accordingly
  → size mismatch barely matters because rooms substitutes

Result: MAE after permuting size = $18k  (only $3k worse)
Permutation importance(size) = $3k  ← appears unimportant
```

**Permute num_rooms:**

```
Same logic: size_sqft stays intact, substitutes for rooms.
MAE after permuting rooms = $17k
Permutation importance(rooms) = $2k  ← also appears unimportant
```

**Ground truth (permute BOTH simultaneously):**

```
With both shuffled: model has no information about house size or rooms.
MAE = $42k  (catastrophic performance loss)
Grouped importance(size + rooms) = $27k  ← the true combined importance
```

**The problem visualised:**

```
Individual importances: size=$3k, rooms=$2k  → sum = $5k
True group importance:  $27k

The group is 5× more important than the sum of individuals suggests.
```

This is the substitute problem. Permutation of individuals underestimates the group by a factor proportional to the correlation.

---

## 5. The Credit Absorption Problem — Numerical Proof

**Setup:** Random Forest on same credit dataset. Two correlated features:
- credit_score (r = −0.88 with late_payments)
- late_payments

**Run 1 (seed=42):** Tree happens to split on credit_score first at root.

```
Root split: credit_score ≤ 650?
  → n_t/N = 1.0 → full weight → MDI gets maximum credit
  
Depth-1 splits: late_payments appears in some branches
  → n_t/N ≈ 0.4 → only 40% weight

MDI result:
  credit_score:  0.44   (got root, full weight)
  late_payments: 0.13   (only gets depth-1 splits)
```

**Run 2 (seed=99):** Tree splits on late_payments first.

```
Root split: late_payments ≥ 2?
  → MDI(late_payments) gets full n_t/N = 1.0 weight

MDI result:
  late_payments: 0.41
  credit_score:  0.16
```

**Both forests have identical test AUC = 0.88.**

**The problem:** The "winner" is determined by random bootstrap/seed, not by which feature is more informative. The attribution is arbitrary. A stakeholder reading Run 1 concludes credit_score dominates; reading Run 2 concludes late_payments dominates. Both reports come from equally valid models.

---

## 6. Grouped Permutation Importance — The Fix for Correlated Groups

### 6.1 The Algorithm

Instead of permuting one feature at a time, permute all features in a correlated group simultaneously. Now the model can't fall back on any substitute.

```
INPUT: model f, data (X, y), metric M, groups G = {g₁, g₂, ...},
       where each gₖ = list of correlated feature names

Baseline score: s₀ = M(y, f(X))

For each group gₖ:
  For each repeat r in 1..n_repeats:
    X_perm = copy of X
    shuffle_idx = random permutation of row indices
    X_perm[gₖ] = X[gₖ].iloc[shuffle_idx]   ← shuffle ALL columns in group
    s_perm = M(y, f(X_perm))
    drop[r] = s_perm - s₀

  group_importance[gₖ] = mean(drop)
  group_importance_std[gₖ] = std(drop)
```

Key: `X_perm[gₖ] = X[gₖ].iloc[shuffle_idx]` — all columns in the group are shuffled using the same index. This preserves the within-group correlations for each shuffled row (because we're using complete rows from elsewhere in the dataset) while breaking the group's relationship with the target.

### 6.2 Full Numerical Example

**Dataset:** 6 samples, predict house price. Feature groups:

```
Group 1 (size):  [size_sqft, num_rooms]  — r=0.85
Group 2 (location): [zip_code, neighbourhood_score]  — r=0.72
Group 3 (age): [house_age]  — standalone

 #   size  rooms  zip  nbhd  age   price($k)
──────────────────────────────────────────────
 1   1000    3    100   7.2   5     $185
 2   1500    4    200   8.1  10     $230
 3   2000    5    100   7.5  15     $265
 4   2500    6    200   8.3  20     $310
 5   3000    7    100   7.8  25     $355
 6   2000    4    200   6.9  12     $240

Baseline MAE = $8k
```

**Grouped permutation — Group 1 (size, rooms):**

Shuffle the [size, num_rooms] columns together (using the same row permutation):

```
Shuffle index: [3, 5, 1, 4, 2, 6]  (rows swapped together)

Permuted size values: [2500, 2000, 1000, 3000, 1500, 2000]
Permuted rooms:       [   6,    4,    3,    7,    4,    5]
(Note: the (size, rooms) pairs are still internally consistent — row 3's size
 and rooms are now in row 1's position, but they're the same (2500, 6) pair)

Other features: unchanged

MAE after permutation = $31k
Group 1 importance = $31k − $8k = +$23k  ← strong
```

**Grouped permutation — Group 2 (zip, nbhd):**

```
Shuffle [zip, neighbourhood_score] with same index:
MAE = $15k
Group 2 importance = $7k  ← moderate
```

**Individual feature (age):**

```
Shuffle age alone:
MAE = $10k
age importance = $2k  ← small
```

**Compare to individual permutation importances (showing the substitute problem):**

```
Method                 size  rooms  zip  nbhd  age
─────────────────────────────────────────────────────
Individual permutation  $4k   $3k   $4k   $3k  $2k
  (substitute problem underestimates correlated pairs)

Grouped permutation    $23k   (combined)  $7k (combined)  $2k
  (correctly captures group importance)
```

The individual importances say size+rooms matter "$7k combined." The grouped method says they matter "$23k." The true group importance is 3× larger than the sum of individuals suggests.

### 6.3 Within-Group Attribution

After establishing group importance, you may want to know which feature within the group contributes more. Options:

**Option 1: Conditional permutation**
Within the group, permute features one at a time but only among samples with similar values of the other group features. Expensive but principled.

**Option 2: SHAP within the group**
Use conditional or interventional SHAP to attribute credit within the group.

**Option 3: Accept ambiguity**
If r > 0.85, the within-group attribution is inherently ambiguous — both features carry nearly identical information. Report the group importance and note that attribution within the group is unreliable. For feature selection: try dropping each in turn and measuring performance.

---

## 7. Conditional SHAP — The Fix for SHAP

Default (marginal) SHAP draws absent feature values from P(X_j) independently. For correlated features, this creates unrealistic combinations.

**Conditional SHAP** draws absent feature values from P(X_j | X_present) — conditioned on the features that are present in the coalition.

```
Marginal SHAP when credit_score is absent, late_payments=0:
  Draw credit_score from P(credit_score)
  → includes low credit scores that don't co-occur with late_payments=0
  → unrealistic

Conditional SHAP when credit_score is absent, late_payments=0:
  Draw credit_score from P(credit_score | late_payments=0)
  → only high credit scores, which realistically co-occur with 0 late payments
  → realistic
```

**The trade-off:**

Conditional SHAP is more realistic but:
- Harder to compute (requires estimating conditional distributions)
- Can violate the Dummy axiom for features that correlate with influential features
- Not the default in TreeSHAP (use `feature_perturbation='interventional'` for a middle ground)

**Practical guidance:**
- For tree models: use `shap.TreeExplainer(model, feature_perturbation='interventional')` — this is closer to conditional SHAP than the default
- For any model: acknowledge that marginal SHAP values for highly correlated feature pairs are approximate and should be interpreted at the group level

---

## 8. ALE — The Fix for PDP

Covered in full in `6_PDP_and_ALE.md`. The summary:

PDP at feature value v uses all samples regardless of whether their other features realistically co-occur with A=v. ALE at value v only uses samples that naturally have A≈v, so the co-occurring values of correlated features B, C are always realistic.

**The rule:**

```
If |r(A, B)| > 0.5 for any pair, use ALE instead of PDP for feature A.
```

---

## 9. The Full Diagnosis and Fix Protocol

```
STEP 1 — DETECT
─────────────────────────────────────────────────────────
Compute: pairwise correlation matrix (Pearson/Spearman)
         VIF for all features

Flag:  pairs with |r| > 0.7  →  HIGH correlation
       features with VIF > 5  →  MULTICOLLINEARITY

─────────────────────────────────────────────────────────
STEP 2 — GROUP
─────────────────────────────────────────────────────────
Use hierarchical clustering on the correlation matrix to
form groups of correlated features.

Cut at |r| = 0.7 threshold.
Each cluster is a "correlation group."

─────────────────────────────────────────────────────────
STEP 3 — FIX BY METHOD
─────────────────────────────────────────────────────────

For feature importance:
  USE:   Grouped permutation importance (permute entire group)
  AVOID: Individual permutation importance for members of the same group
  AVOID: MDI for any correlated feature (arbitrary credit absorption)

For SHAP:
  USE:   TreeSHAP with feature_perturbation='interventional' (trees)
  USE:   Conditional SHAP if careful implementation available
  REPORT: Group-level sum of SHAP values for correlated features
  AVOID: Interpreting individual marginal SHAP values for r > 0.7 pairs

For PDP:
  USE:   ALE instead of PDP for any feature with |r| > 0.5 with another feature
  USE:   ICE plots alongside to detect interaction-driven heterogeneity

For LIME:
  USE:   Run 10+ times and check coefficient stability
  REPORT: High instability for correlated features as a known limitation
  CONSIDER: SHAP as the alternative

For feature selection:
  USE:   Drop-column importance (retrains model — handles correlation correctly)
  USE:   Grouped permutation to establish group importance
  THEN:  Ablation within the group (drop one at a time) to find the representative

─────────────────────────────────────────────────────────
STEP 4 — REPORT
─────────────────────────────────────────────────────────
Report group-level importance for correlated groups.
Never present individual importances for correlated features
without flagging the correlation and its effect on the estimate.
```

---

## 10. Worked Case Study — Same Dataset, Every Method

**Dataset:** 200 customers. Predict loan default. Features:

```
credit_score       (continuous, 400–850)
num_late_payments  (integer, 0–15)         r(credit, late) = −0.88
debt_ratio         (continuous, 0–1)       r(debt, income) = −0.55
income             (continuous, $20k–$150k)
age                (continuous, 20–70)     VIF(age) = 1.4
```

Correlation groups after hierarchical clustering at |r| > 0.7:
- **Group A:** credit_score, num_late_payments
- **Group B:** debt_ratio (moderate r with income, but below threshold)
- **Group C:** income
- **Group D:** age

---

**Method 1: Individual permutation importance (test set, n_repeats=20)**

```
Feature           Importance (AUC drop)   Note
─────────────────────────────────────────────────────────────
credit_score      0.031                   ← underestimated (late_payments substitutes)
num_late_payments 0.028                   ← underestimated (credit_score substitutes)
debt_ratio        0.065                   ← moderately affected by income correlation
income            0.041
age               0.022

Apparent conclusion: debt_ratio is the most important feature
```

**Method 2: Grouped permutation importance**

```
Group                     Importance    Note
──────────────────────────────────────────────────────────────
Group A (credit+late)     0.142         ← true importance, 2.5× sum of individuals
debt_ratio                0.075         ← slightly higher than individual (income less able to substitute when tested alone)
income                    0.048
age                       0.022

Correct conclusion: credit history (Group A) is by far the most important factor
```

**Method 3: MDI / Gini importance (RF with 100 trees)**

```
Seed=42:                           Seed=99:
  credit_score  0.38                 num_late_payments 0.40
  late_payments 0.14                 credit_score      0.12
  debt_ratio    0.22                 debt_ratio        0.21
  income        0.18                 income            0.19
  age           0.08                 age               0.08

Conclusion varies by seed — arbitrary credit absorption
```

**Method 4: Marginal SHAP (TreeSHAP, test set)**

```
Mean |SHAP|:
  credit_score      0.062
  num_late_payments 0.045
  debt_ratio        0.078
  income            0.052
  age               0.019

Individual SHAP for customer #47 (credit=750, late=0, debt=0.75, income=$38k):
  credit_score      −0.08  (good credit → lowers risk)
  num_late_payments  0.00  (0 late payments, near average)
  debt_ratio         0.28  (high debt → raises risk)
  income            −0.10  (low income → raises risk slightly)
  age               +0.02

Caveat: credit_score and late_payments attributions are marginal SHAP —
        under-attributes the group because they cover for each other.
        True group attribution: (−0.08 + 0.00) = −0.08 for Group A,
        but conditional SHAP would give Group A a larger attribution.
```

**Method 5: ALE curves**

```
ALE(credit_score): steep decrease from 400 to 600, flattens above 700
ALE(debt_ratio):   steady increase from 0.3 to 0.8, steep above 0.8
ALE(income):       steep decrease from $20k to $50k, gradual above $50k

Diagnostic: ALE and PDP curves differ significantly for credit_score
            (r=−0.88 with late_payments) → confirms PDP would be biased here
```

**Synthesis:**

```
Method                    What it correctly shows
─────────────────────────────────────────────────────────────────────
Individual permutation    debt_ratio looks most important — WRONG
Grouped permutation       Group A (credit history) is most important — CORRECT
MDI                       Varies by seed — UNRELIABLE for Group A features
Marginal SHAP             Group A sum = −0.08 for this customer, debt=+0.28
ALE                       Shape of relationships correctly shown (no extrapolation)
```

**Bottom line:** Without grouped permutation importance, you would conclude that debt_ratio is the most important feature. The correct conclusion is that credit history (credit_score + late_payments combined) is the most important — and debt_ratio is secondary. The correlation between credit_score and late_payments was hiding the true picture.

---

## 11. Interview Q&A

**Q: How does feature correlation affect permutation importance?**

When feature A is permuted, correlated feature B remains intact. The model uses B as a substitute for A, so performance barely drops. The same happens when B is permuted — A substitutes. Both A and B appear unimportant, even if their group is the most predictive factor. The degree of underestimation is proportional to the correlation: at r=0.9, each feature's individual importance is roughly 80% underestimated.

---

**Q: What is VIF and what does a value of 10 tell you?**

VIF (Variance Inflation Factor) for feature j = 1/(1 − R²_j), where R²_j is the R² from regressing feature j on all other features. VIF=10 means R²_j = 0.90 — 90% of feature j's variance is explained by the other features. The feature is highly multicollinear. Feature importance estimates for this feature are unreliable because many other features can substitute for it.

---

**Q: What is the grouped permutation importance and when should you use it?**

Grouped permutation importance permutes all features in a correlated group simultaneously using the same shuffled row index. This breaks the group's relationship with the target while preserving within-group correlations (because complete rows are swapped). Use it when any feature pair in your dataset has |r| > 0.7. It correctly estimates the group's combined importance, whereas individual permutation importance underestimates each member of the group.

---

**Q: SHAP satisfies the efficiency axiom for correlated features. So why is SHAP still problematic?**

The efficiency axiom guarantees that the sum of SHAP values equals f(x) − E[f(X)] — the total attribution is correctly distributed. But for correlated features, marginal SHAP creates unrealistic input combinations when computing coalition values, which affects how the total is *split* between the correlated features. The sum is correct, but the individual attributions for each correlated feature can be misleading. The efficiency axiom is about the total, not about within-group splits.

---

**Q: Why is drop-column importance better than permutation importance for correlated features?**

Drop-column importance removes feature A entirely and retrains the model from scratch. The retrained model learns to use remaining features (including B) as optimally as possible without A. The performance drop measures: "how much worse is the best possible model when A is completely unavailable?" This correctly measures A's unique contribution given that the other features can substitute.

Permutation importance measures: "how much worse does the already-trained model get when A's values are scrambled but B is still available?" The trained model was built expecting A to exist, so scrambling A but leaving B is more forgiving. Drop-column importance is more expensive (retraining) but asks the right question.

---

**Q: You're presenting feature importance to a business stakeholder and two highly correlated features both appear unimportant. How do you handle this?**

Explain that individual importance estimates underestimate correlated feature groups — this is a known limitation, not a data or model quality issue. Present the grouped permutation importance showing the combined importance of the correlated pair. Then explain the practical implication: both features carry the same information, so the model doesn't need both individually — but the signal they represent is highly important. For feature selection: test dropping each from the group and measuring performance to find the best representative. For business communication: present the group as a single signal ("credit history" rather than "credit_score and late_payments separately").

---

## 12. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CORRELATED FEATURES IN XAI — KEY FACTS                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  DETECTION                                                               │
│    |r| > 0.7           → flag as highly correlated                       │
│    VIF > 5             → flag as multicollinear                          │
│    ALE ≠ PDP curve     → confirms correlation affects that feature       │
│                                                                          │
│  HOW EACH METHOD BREAKS                                                  │
│    Permutation:  underestimates both (substitute problem)                │
│    MDI:          arbitrary credit (credit absorption)                    │
│    Marginal SHAP: unrealistic coalitions → wrong within-group split      │
│    LIME:          amplified instability from unrealistic perturbations   │
│    PDP:           extrapolates into unrealistic combinations             │
│                                                                          │
│  THE SUBSTITUTE PROBLEM (permutation)                                    │
│    r=0.85 → each feature's importance underestimated by ~60%            │
│    Grouped importance >> sum of individual importances                  │
│                                                                          │
│  THE CREDIT ABSORPTION PROBLEM (MDI)                                     │
│    Whichever correlated feature gets the root split absorbs most credit  │
│    Which one "wins" depends on random seed → arbitrary result            │
│                                                                          │
│  FIXES BY METHOD                                                         │
│    Permutation  → Grouped permutation importance                        │
│    MDI          → Don't use for correlated features                     │
│    SHAP         → Conditional/interventional; report group sums         │
│    PDP          → Use ALE instead                                        │
│    LIME         → Run many times; acknowledge instability               │
│    Feature sel. → Drop-column importance                                │
│                                                                          │
│  PROTOCOL                                                                │
│    1. Check correlations (pairwise + VIF)                               │
│    2. Build correlation groups (hierarchical clustering)                │
│    3. Use grouped methods for importance of each group                  │
│    4. Report group-level importance — not individual                    │
│    5. Within-group: use ablation or conditional SHAP                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Strobl et al. (2008)** — *Conditional Variable Importance for Random Forests.* — Correlation bias in MDI and MDA.
- **Molnar et al. (2020)** — *Pitfalls to Avoid when Interpreting Machine Learning Models.* — Comprehensive review of correlation pitfalls.
- **Janzing et al. (2020)** — *Feature relevance quantification: A causal problem.* — Marginal vs conditional SHAP.
- **Apley & Zhu (2020)** — *Visualizing effects of predictor variables.* — ALE as PDP fix.
- **Companion files:** `3_Permutation_Importance.md`, `4_SHAP.md`, `6_PDP_and_ALE.md`
