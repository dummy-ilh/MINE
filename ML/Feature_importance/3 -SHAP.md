# SHAP — Complete Guide

---

## Table of Contents

1. [The Problem SHAP Solves](#1-the-problem-shap-solves)
2. [The Core Idea in One Sentence](#2-the-core-idea-in-one-sentence)
3. [Where SHAP Comes From — A Story](#3-where-shap-comes-from--a-story)
4. [Translating the Story to ML](#4-translating-the-story-to-ml)
5. [The Four Guarantees (Axioms)](#5-the-four-guarantees-axioms)
6. [The Baseline — What You're Explaining Against](#6-the-baseline--what-youre-explaining-against)
7. [How SHAP Handles "Missing" Features](#7-how-shap-handles-missing-features)
8. [Computing SHAP Values by Hand](#8-computing-shap-values-by-hand)
   - 8.1 Two Features — Complete Walkthrough
   - 8.2 Three Features — All 6 Orderings
   - 8.3 Why This Becomes Expensive
9. [The Formula (After You Understand It)](#9-the-formula-after-you-understand-it)
10. [SHAP Variants — Pick the Right One](#10-shap-variants--pick-the-right-one)
11. [TreeSHAP — Why Trees Get an Exact Fast Version](#11-treeshap--why-trees-get-an-exact-fast-version)
    - 11.1 The Key Insight
    - 11.2 Worked Example on a Real Tree
    - 11.3 Extending to Random Forests and GBMs
12. [KernelSHAP — The Universal Approximation](#12-kernelshap--the-universal-approximation)
13. [LinearSHAP — Instant Exact for Linear Models](#13-linearshap--instant-exact-for-linear-models)
14. [Marginal vs Conditional vs Interventional SHAP](#14-marginal-vs-conditional-vs-interventional-shap)
15. [SHAP Plots — Every Type](#15-shap-plots--every-type)
    - 15.1 Force Plot (single prediction)
    - 15.2 Waterfall Plot (single prediction, cleaner)
    - 15.3 Beeswarm / Summary Plot (whole dataset)
    - 15.4 Bar Plot (global ranking)
    - 15.5 Dependence Plot (feature shape + interaction)
    - 15.6 Decision Plot (comparing groups)
16. [Global Importance from SHAP](#16-global-importance-from-shap)
17. [SHAP Interaction Values](#17-shap-interaction-values)
18. [Bias-Variance of SHAP Estimates](#18-bias-variance-of-shap-estimates)
19. [Correlated Features — Where SHAP Still Struggles](#19-correlated-features--where-shap-still-struggles)
20. [SHAP vs Other Methods](#20-shap-vs-other-methods)
21. [Common Mistakes](#21-common-mistakes)
22. [Summary Card](#22-summary-card)

---

## 1. The Problem SHAP Solves

You train a model. It predicts that customer #4521 will churn with 78% probability.

Your manager asks: **"Why?"**

You could answer: "Age and debt ratio are generally the most important features in this model." But that's a global statement about the model — not about this customer. Your manager wants to know what specifically about *this person's profile* drove the 78% prediction.

That is exactly what SHAP answers:

> For this specific prediction, each feature gets a number that says how much it pushed the result up or down, and all those numbers add up to explain the full prediction.

Before SHAP, the alternatives all had serious problems:

| Method | Problem |
|---|---|
| Global feature importance | Tells you about the model overall, not any specific prediction |
| LIME | Unstable — run it twice, get different explanations |
| Gradient attribution | Doesn't sum back to the prediction; can't verify |
| Coefficients | Only works for linear models |

SHAP solves all of these with a framework that has **mathematical fairness guarantees**.

---

## 2. The Core Idea in One Sentence

> A SHAP value for feature j answers: **"On average, across all the different orders in which features could have been introduced, how much did feature j change the prediction?"**

Everything else in this guide is explaining what "all the different orders" means and why averaging over them is the right thing to do.

---

## 3. Where SHAP Comes From — A Story

SHAP is built on **Shapley values**, invented by economist Lloyd Shapley in 1953 to solve a simple fairness problem.

Three workers — Alice, Bob, and Carol — collaborate on projects. Depending on who teams up, they generate different revenues:

```
Alone:
  Alice alone  →  $100k
  Bob alone    →  $120k
  Carol alone  →   $80k

In pairs:
  Alice + Bob        →  $280k
  Alice + Carol      →  $210k
  Bob + Carol        →  $250k

All three together  →  $400k
```

**Question: How do you fairly split the $400k among the three?**

Naive equal split ($133k each) ignores that Bob contributes more. "Give everyone their solo earnings" ignores the synergies when they work together.

**Shapley's answer:** For each worker, try every possible order in which they could have joined the group. Each time they join, measure how much revenue went up because of them. Average that across all joining orders. That average is their fair share.

For Alice:
- Alice joins first (before anyone): adds $100k (Alice's solo value)
- Alice joins second after Bob: adds $280k − $120k = $160k
- Alice joins third (after Bob and Carol): adds $400k − $250k = $150k
- Alice joins second after Carol: adds $210k − $80k = $130k
- ... (all 6 orderings total)
- Average = Alice's Shapley value = her fair share

This is the Shapley value. It is **the** fairness solution — not one of many options.

In ML: the workers are features, and the "revenue" is the model's prediction for one sample.

---

## 4. Translating the Story to ML

Here is the word-for-word translation:

```
Story concept         →   ML concept
────────────────────────────────────────────────────────────────
Workers               →   Features  (age, income, debt_ratio, ...)
A team / coalition S  →   A subset of features the model "knows"
Revenue of team S     →   Expected model prediction when the model
                           knows only the features in S
                           (all other features are averaged out)
Total revenue         →   Model's actual prediction for this sample
No workers (empty)    →   Average prediction across all samples
A worker's fair share →   SHAP value for that feature
```

So asking "what is the SHAP value for `income` for customer #4521?" means:

> "Across all orderings in which features could be revealed to the model, how much did knowing `income = $65k` for this customer change the expected prediction?"

---

## 5. The Four Guarantees (Axioms)

The Shapley value is the **only** attribution method satisfying all four of these simultaneously. This is what makes SHAP theoretically different from every other method.

---

### Guarantee 1 — Completeness (Everything adds up)

```
φ₁ + φ₂ + ... + φₚ  =  f(x) − E[f(X)]
                         ↑              ↑
                   this prediction   average prediction
```

**What it means:** All SHAP values sum to exactly the gap between this prediction and the average. Every dollar of "deviation from average" is accounted for. Nothing is lost, nothing is double-counted.

**Why it matters in practice:** You can always verify your SHAP values. Sum them up. If they don't equal `prediction − baseline`, something went wrong. LIME can't do this. Permutation importance can't do this. SHAP is the only method with a built-in correctness check.

---

### Guarantee 2 — Symmetry (No favouritism)

If two features contribute exactly the same amount to every possible coalition, they get the same SHAP value. The result can't be gamed by renaming features or changing the order they appear in your dataset.

---

### Guarantee 3 — Dummy (Zero credit for useless features)

If a feature never changes the prediction for any coalition — it literally contributes nothing — its SHAP value is 0.

**Practical implication:** A feature with SHAP values near zero for all samples is genuinely unused by the model. Safe to remove.

---

### Guarantee 4 — Additivity (Ensembles decompose cleanly)

If your model is a sum of two sub-models (like a Random Forest is a sum of trees), the SHAP values for the full model equal the sum of SHAP values from each sub-model.

**Practical implication:** For a Random Forest, you compute TreeSHAP on each tree separately and average. The result is exactly right.

---

## 6. The Baseline — What You're Explaining Against

SHAP explains the gap between this prediction and the **average prediction** (the baseline).

```
                 Average prediction   Prediction for this sample
                     (baseline)
                         │                       │
                         ▼                       ▼
Churn probability:      0.15     →→→→→→→→→      0.72
                                    gap = +0.57
                                    (SHAP values explain this)
```

The baseline is computed from a **background dataset** — usually a sample of the training data.

**What the background dataset does:**
- Sets the baseline `E[f(X)]`
- Defines what "unknown" feature values look like when computing coalitions

**Practical choice:** 100–300 randomly sampled rows from training data is enough. Using all training data is most correct but slow.

```
Different background → different baseline → different absolute SHAP values
BUT: the relative attribution between features stays stable
AND: the efficiency check always holds regardless of background choice
```

---

## 7. How SHAP Handles "Missing" Features

This is the mechanism behind everything. When a coalition S doesn't include a feature — when that feature is "absent" — SHAP needs to compute the model's prediction as if it doesn't know that feature's value.

It does this by **averaging the prediction over what that feature could plausibly be**.

**Step-by-step:**

```
Sample: age=45, income=$60k, debt=0.80
Coalition S = {age, income}  ← debt is absent

To compute v({age, income}):

  For each row b in the background dataset:
    Create a hybrid row:
      age   = 45        ← from our sample  (in S)
      income = $60k     ← from our sample  (in S)
      debt  = b.debt    ← from background  (not in S)

    Run model on this hybrid row → get a prediction

  v({age, income}) = average of all those predictions
```

You're essentially asking: "If we knew this person's age and income but had no idea about their debt, what would the model predict on average?"

**Why not just replace absent features with 0 or the mean?**

- Zero: creates inputs the model never saw during training → the prediction reflects "what does the model do with garbage input?" not "what does the model predict without this information?"
- Mean: same problem — a single fixed value doesn't represent uncertainty
- **Averaging over background:** correctly simulates not knowing the feature, while keeping all inputs realistic

This averaging approach is called **marginal SHAP** (the default). Two alternatives — conditional and interventional SHAP — differ in how they draw replacement values. Covered in Section 14.

---

## 8. Computing SHAP Values by Hand

### 8.1 Two Features — Complete Walkthrough

**Problem:** Predict house price. Features: `size` (large or small) and `location` (good or bad).

For a specific house with large size and good location, the model predicts **$280k**. Average prediction is **$200k**. Explain why this house is **$80k above average**.

**Step 1: Build the coalition value table**

We need to know what the model predicts when each subset of features is "known":

```
v(∅)                = $200k   no features known → just predict the average
v({size})           = $245k   only know size=large
v({location})       = $225k   only know location=good
v({size, location}) = $280k   know both → actual prediction
```

**Step 2: Enumerate all orderings (2! = 2)**

```
╔══════════════════════════════════════════════════════════════╗
║  ORDERING 1:  size first, then location                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. size joins the empty coalition                           ║
║       Before: v(∅)     = $200k                              ║
║       After:  v({size}) = $245k                             ║
║       size's contribution: +$45k                            ║
║                                                              ║
║  2. location joins (size is already in)                     ║
║       Before: v({size})          = $245k                    ║
║       After:  v({size,location}) = $280k                    ║
║       location's contribution: +$35k                        ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  ORDERING 2:  location first, then size                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. location joins the empty coalition                       ║
║       Before: v(∅)          = $200k                         ║
║       After:  v({location}) = $225k                         ║
║       location's contribution: +$25k                        ║
║                                                              ║
║  2. size joins (location is already in)                     ║
║       Before: v({location})        = $225k                  ║
║       After:  v({size,location})   = $280k                  ║
║       size's contribution: +$55k                            ║
╚══════════════════════════════════════════════════════════════╝
```

**Step 3: Average each feature's contributions**

```
φ(size)     = ($45k + $55k) / 2  =  $50k
φ(location) = ($35k + $25k) / 2  =  $30k
```

**Step 4: Verify (Efficiency check)**

```
φ(size) + φ(location) = $50k + $30k = $80k
f(x) − E[f(X)]        = $280k − $200k = $80k  ✅
```

**Reading the result:**

This house is $80k above the average. Size is responsible for $50k of that, location for $30k.

Notice that `size` contributed $45k when it went first (alone, not knowing about location) but $55k when it went second (adding the "finishing touch"). The SHAP value $50k is the fair average across both scenarios. It is not the prediction with size added first and not the prediction with size added last — it is genuinely the average.

---

### 8.2 Three Features — All 6 Orderings

**Problem:** Predict loan default probability. Features: `debt_ratio` (D), `income` (I), `age` (A).

**The coalition value table (2³ = 8 values):**

```
v(∅)      = 0.10   baseline, average default rate
v({D})    = 0.35   high debt ratio alone
v({I})    = 0.08   good income alone (below average — low risk)
v({A})    = 0.12   age alone (near average)
v({D,I})  = 0.30   high debt, partially offset by good income
v({D,A})  = 0.38   high debt + older age → higher risk
v({I,A})  = 0.07   good income + age → low risk
v({D,I,A})= 0.45   all known → actual prediction
```

**3! = 6 orderings, each feature's marginal at each step:**

```
Ordering  │  D contribution  │  I contribution  │  A contribution
──────────┼──────────────────┼──────────────────┼─────────────────
D→I→A     │  0.35−0.10=+0.25 │  0.30−0.35=−0.05 │  0.45−0.30=+0.15
D→A→I     │  0.35−0.10=+0.25 │  0.45−0.38=+0.07 │  0.38−0.35=+0.03
I→D→A     │  0.30−0.08=+0.22 │  0.08−0.10=−0.02 │  0.45−0.30=+0.15
I→A→D     │  0.45−0.07=+0.38 │  0.08−0.10=−0.02 │  0.07−0.08=−0.01
A→D→I     │  0.38−0.12=+0.26 │  0.45−0.38=+0.07 │  0.12−0.10=+0.02
A→I→D     │  0.45−0.07=+0.38 │  0.07−0.12=−0.05 │  0.12−0.10=+0.02
──────────┼──────────────────┼──────────────────┼─────────────────
Sum       │       1.74       │       0.00       │      0.36
Average   │      +0.290      │      0.000       │     +0.060
```

**Verification:**

```
φ(D) + φ(I) + φ(A) = 0.290 + 0.000 + 0.060 = 0.350
f(x) − E[f(X)]     = 0.45 − 0.10            = 0.350  ✅
```

**Reading the result:**

- **debt_ratio (+0.29):** Dominant driver. This applicant's high debt pushes their predicted default risk 29 points above average.
- **income (0.000):** Income helped in some orderings and hurt in others, exactly cancelling out. For this specific applicant, income has zero *net* SHAP value — not zero importance in general, just zero contribution for this person's profile across all orderings.
- **age (+0.06):** A small positive contribution — their age adds a little extra risk.

---

### 8.3 Why This Becomes Expensive

```
p features → p! orderings and 2^p coalitions

p = 3   →      6 orderings,      8 coalitions  (just did this by hand)
p = 10  →  3.6M orderings,   1024 coalitions
p = 20  →  2.4×10¹⁸ orderings   ← impossible
p = 50  →  3×10⁶⁴ orderings     ← wildly impossible
```

Each coalition evaluation requires running the model multiple times (once per background sample). For large p, exact computation is completely intractable.

**Solutions:**
- **TreeSHAP:** Exploits tree structure to avoid enumeration → exact, fast
- **KernelSHAP:** Samples a subset of coalitions → approximate, works on any model

---

## 9. The Formula (After You Understand It)

Now that you know what the Shapley value computes intuitively, here is the formal formula. Every symbol should now make sense:

```
φⱼ  =  Σ_{S ⊆ F\{j}}  ───────────────────  ×  [ v(S∪{j}) − v(S) ]
                         |S|! × (p−|S|−1)!
                         ───────────────────
                                p!
```

Breaking it down:

```
φⱼ                        the SHAP value for feature j

S ⊆ F\{j}                 every possible subset that doesn't include j
                           (every coalition j could be added to)

v(S∪{j}) − v(S)           j's marginal contribution when added to coalition S
                           "how much does the prediction change when j is added?"

|S|! × (p−|S|−1)! / p!    the weight for this coalition
                           = probability that this exact coalition S appears
                             if all p features join in a uniformly random order
                           = this is just the "average across all orderings"
                             rewritten as a weighted sum over coalitions
```

The weight can be expanded to show what it is:

```
For p=2, S={} (size 0):
  weight = 0! × (2−0−1)! / 2! = 1 × 1 / 2 = 0.5

For p=2, S={location} (size 1):
  weight = 1! × (2−1−1)! / 2! = 1 × 1 / 2 = 0.5
```

Both coalitions get weight 0.5 → average equally → matches our by-hand calculation above.

---

## 10. SHAP Variants — Pick the Right One

| Variant | For what model | Exact? | Speed | Use when |
|---|---|---|---|---|
| **TreeSHAP** | RF, XGBoost, LightGBM, CatBoost | ✅ Yes | Fast | Any tree model — always use this |
| **LinearSHAP** | Linear / Logistic regression | ✅ Yes | Instant | Any linear model |
| **KernelSHAP** | Any model (black box) | ❌ Approx | Slow | Non-tree, non-linear models |
| **DeepSHAP** | Neural networks | ❌ Approx | Medium | Deep learning |
| **GradientSHAP** | Neural networks | ❌ Approx | Medium | Deep learning (often better than DeepSHAP) |

**Simple decision tree:**

```
What type of model?
│
├── Tree (RandomForest, XGBoost, LightGBM, CatBoost)
│   └── → TreeSHAP  ✅
│
├── Linear / Logistic Regression
│   └── → LinearSHAP  ✅
│
├── Neural Network
│   └── → GradientSHAP or DeepSHAP
│
└── Anything else (SVM, kNN, custom ensemble)
    └── → KernelSHAP  (slow, but works)
```

---

## 11. TreeSHAP — Why Trees Get an Exact Fast Version

### 11.1 The Key Insight

Remember: to compute SHAP values, we need `v(S)` for every coalition S — the expected prediction when only features in S are known.

For trees, computing `v(S)` doesn't require looping over background samples separately. It can be done in a **single pass through the tree**, using the training sample counts stored at each node.

**The rule at each node:**

```
When you reach a node that splits on feature j:

  IF j is in S (it's "known"):
    → follow the branch that matches x's actual value of j
      (normal prediction, no uncertainty)

  IF j is NOT in S (it's "absent"):
    → split your probability mass across both branches,
      proportional to how many training samples went each way:
        left  branch gets  n_left / n_total  of the weight
        right branch gets  n_right / n_total  of the weight
```

At the leaves, `v(S)` = sum of (leaf prediction × accumulated weight).

This works because the tree's node counts already encode the distribution of training data — they tell you the probability a random training sample would go left or right, which is exactly the marginal average we need.

---

### 11.2 Worked Example on a Real Tree

**The tree:**

```
                    ┌──────────────────────┐
                    │      age ≤ 30?       │  ← split feature: age
                    │  n_total=100         │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │ YES (age≤30)                     │ NO (age>30)
              │ n=60                             │ n=40
              └─────────┬───────────────         └──────────┬──────────
                        │                                   │
              ┌─────────┴──────────┐             ┌──────────┴─────────┐
              │  income ≤ 50k?     │             │  income ≤ 70k?     │
              │  n=60              │             │  n=40              │
              └────────┬───────────┘             └──────────┬─────────┘
                       │                                    │
           ┌───────────┴──────────┐             ┌───────────┴──────────┐
       YES (≤50k)            NO (>50k)       YES (≤70k)            NO (>70k)
       Leaf A                Leaf B          Leaf C                Leaf D
       pred=0.1              pred=0.5        pred=0.4              pred=0.8
       n=40                  n=20            n=25                  n=15
```

**Baseline:**
```
E[f(X)] = (40×0.1 + 20×0.5 + 25×0.4 + 15×0.8) / 100
        = (4 + 10 + 10 + 12) / 100
        = 0.36
```

**Sample to explain:** age=25, income=60k.
**Actual prediction:** age≤30 → left → income>50k → Leaf B → **f(x*) = 0.5**
**Need to explain:** 0.5 − 0.36 = **+0.14**

---

**Computing v(S) for all 4 coalitions:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 v(∅) = E[f(X)] = 0.36  (no features known)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 v({age}) — only age=25 is known, income is absent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At root (splits on age):
  age IS known → age=25 ≤ 30 → go left, weight=1.0

At left node (splits on income):
  income is NOT known → split weight by training counts:
    → Leaf A gets weight: n_A/n_leftnode = 40/60 = 0.667
    → Leaf B gets weight: n_B/n_leftnode = 20/60 = 0.333

v({age}) = 0.667 × 0.1  +  0.333 × 0.5
         = 0.067 + 0.167  =  0.233

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 v({income}) — only income=60k is known, age is absent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At root (splits on age):
  age is NOT known → split weight by training counts:
    → left subtree gets  60/100 = 0.6
    → right subtree gets 40/100 = 0.4

In LEFT subtree (income ≤ 50k?), income=60k IS known:
  income=60k > 50k → Leaf B, weight = 0.6 × 1.0 = 0.6

In RIGHT subtree (income ≤ 70k?), income=60k IS known:
  income=60k ≤ 70k → Leaf C, weight = 0.4 × 1.0 = 0.4

v({income}) = 0.6 × 0.5  +  0.4 × 0.4
            = 0.30 + 0.16  =  0.46

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 v({age, income}) — both known
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Normal prediction: age=25≤30 → left; income=60k>50k → Leaf B
v({age, income}) = 0.5  (= f(x*))
```

**Now compute SHAP values (2 features → 2 orderings):**

```
Ordering 1: age first, income second
  age enters ∅:         v({age}) − v(∅)             = 0.233 − 0.36 = −0.127
  income enters {age}:  v({age,inc}) − v({age})      = 0.500 − 0.233 = +0.267

Ordering 2: income first, age second
  income enters ∅:      v({inc}) − v(∅)             = 0.46 − 0.36  = +0.100
  age enters {income}:  v({age,inc}) − v({inc})      = 0.50 − 0.46  = +0.040

φ(age)    = (−0.127 + 0.040) / 2  =  −0.044
φ(income) = (+0.267 + 0.100) / 2  =  +0.184
```

**Verification:**
```
φ(age) + φ(income) = −0.044 + 0.184 = +0.14
f(x*) − E[f(X)]    = 0.50 − 0.36   = +0.14  ✅
```

**Reading the result:**

This young (age=25) applicant with a decent income (60k) gets prediction 0.5.
- Their income (+0.184) is the main reason they're above average — income=60k places them in Leaf B (0.5) which is above the average 0.36.
- Their age (−0.044) actually slightly pulls the risk *down* from average — young age is associated with lower risk in this model.

---

### 11.3 Extending to Random Forests and GBMs

**Random Forest:** By the additivity axiom, SHAP values for the forest = average of SHAP values from each tree.

```
φⱼ(RF) = (1/T) × Σₜ φⱼ(tree t)
```

**Gradient Boosting (XGBoost, LightGBM):** Same logic — SHAP values from the full model = sum of SHAP values from each tree, weighted by the learning rate.

TreeSHAP runs on each tree and combines. The result is exact.

**Speed:** TreeSHAP runs in O(TLD²) time — where T=trees, L=leaves, D=depth. For a typical 100-tree forest with depth 6, this is roughly 100 × 64 × 36 ≈ 230,000 operations per sample. Fast enough for real-time use.

---

## 12. KernelSHAP — The Universal Approximation

KernelSHAP works on **any model** by treating it as a complete black box. It doesn't look inside the model at all — it only calls `model.predict()`.

**The approach:**

1. Sample M random coalitions (binary masks of which features are present/absent)
2. For each coalition, compute `v(S)` by replacing absent features with background values and averaging predictions
3. Fit a **weighted linear regression** on the (coalition, v(S)) pairs using a special SHAP-specific weight function
4. The regression coefficients are the SHAP values

**Why a weighted regression?** Mathematically, you can show that if you weight coalitions using the specific function:

```
π(S) = (p−1) / [ C(p, |S|) × |S| × (p−|S|) ]

where C(p,|S|) = "p choose |S|", the binomial coefficient
```

...then the regression coefficients equal the true Shapley values (as M → 2^p). The SHAP axioms are "baked into" this weighting.

**The approximation:** Since you can't evaluate all 2^p coalitions, you sample M of them. Larger M → smaller approximation error, but more computation.

```
M = 100  → rough estimate, fast
M = 500  → reasonable accuracy
M = 5000 → accurate, but 50× slower than M=100
```

**When to use:** Only when TreeSHAP/LinearSHAP don't apply (the model is not a tree or linear model). KernelSHAP is correct but slow.

---

## 13. LinearSHAP — Instant Exact for Linear Models

For a linear model `f(x) = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ`, the SHAP value for feature j has a closed-form solution:

```
φⱼ = βⱼ × (xⱼ − E[Xⱼ])
```

**Why this works:** In a linear model, features never interact. Feature j always contributes `βⱼ` per unit regardless of which coalition it joins. So the "average across all orderings" is always just `βⱼ` times the feature's deviation from its mean.

**Numerical example:**

```
Model: default_risk = 0.05 + 0.002×age − 0.00001×income + 0.4×debt_ratio

Background means: E[age]=40, E[income]=55000, E[debt_ratio]=0.35

Sample x*: age=28, income=45000, debt_ratio=0.75

φ(age)        = 0.002   × (28 − 40)    = 0.002 × (−12) = −0.024
φ(income)     = −0.00001 × (45000 − 55000) = −0.00001 × (−10000) = +0.100
φ(debt_ratio) = 0.4     × (0.75 − 0.35) = 0.4 × 0.40   = +0.160

Baseline: f(mean features) = 0.05 + 0.002×40 − 0.00001×55000 + 0.4×0.35
                            = 0.05 + 0.08 − 0.55 + 0.14 = −0.28

Sum check:
  −0.024 + 0.100 + 0.160 = +0.236
  f(x*) = −0.28 + 0.236 = −0.044  (low risk, as expected — young, decent income)
```

---

## 14. Marginal vs Conditional vs Interventional SHAP

When a feature is absent from a coalition, how do you replace it? There are three answers, and each gives a different SHAP value with a different interpretation.

---

**Option 1 — Marginal SHAP (default in TreeSHAP)**

Absent features are drawn from their **marginal distribution** — independently, ignoring correlations.

```
"Income is absent" → draw income values from the overall income distribution,
                     regardless of this person's age or other features
```

**What it measures:** How much does the model use this feature for predictions, averaged over the marginal feature distribution?

**Problem:** When features are correlated (e.g. age and retirement status), this creates unrealistic combinations. A 22-year-old might be paired with retirement_status=True from an older background person.

---

**Option 2 — Conditional SHAP**

Absent features are drawn from **P(X_j | features in S)** — conditioned on the known features.

```
"Income is absent, but we know age=22" → draw income values from the
                                          distribution of income for people aged 22
```

**What it measures:** Feature attribution within the realistic joint distribution.

**Problem:** Harder to compute. Can assign non-zero SHAP to features that don't directly appear in the model, if they're correlated with ones that do (violates the Dummy axiom in some cases).

---

**Option 3 — Interventional SHAP**

Absent features are drawn directly from background data, unconditionally — but this represents a **causal intervention** (breaking the feature's natural dependencies).

```
"Income is absent" → draw income from background regardless of age,
                     as if we physically set income independently
```

**What it measures:** Causal attribution — what would happen to the prediction if we intervened on this feature.

---

**Numerical example showing they differ:**

Model predicts credit risk from `credit_score` and `num_late_payments`. These are highly correlated (r = −0.90): more late payments → lower credit score.

For a customer with credit_score=750 (high) and num_late_payments=0:

```
Marginal SHAP:
  When credit_score is "absent," num_late_payments=0 is paired with credit scores
  drawn from the full background (including low scores from people with many payments).
  This creates unrealistic combos.
  Result:  φ(credit_score) = −0.18,  φ(late_payments) = +0.40

Conditional SHAP:
  When credit_score is "absent," num_late_payments=0, so we condition:
  backgrounds only from people with 0 late payments (all have high scores).
  That coalition already implies good credit → credit_score adds little.
  Result:  φ(credit_score) = −0.03,  φ(late_payments) = +0.55
```

Same prediction, different attribution. Both are "correct" — they answer different questions.

**Quick reference:**

```
Goal                                    → Use
────────────────────────────────────────────────────────────────
Model transparency (what the model uses)  → Marginal (default)
Realistic feature combinations            → Conditional
Causal attribution                        → Interventional
Fast tree-based models                    → TreeSHAP (marginal by default;
                                            interventional via option flag)
```

---

## 15. SHAP Plots — Every Type

### 15.1 Force Plot (Single Prediction)

**What it shows:** One prediction, fully explained. Features that pushed the prediction above baseline shown in red (pointing right). Features that pulled it down shown in blue (pointing left). Width of each bar = magnitude of the SHAP value.

```
                              Prediction: 0.72
Baseline: 0.15
│
├─────────────────────────────────────────────────────────────────►
│       ◄────────┐         ┌────────────────┐┌───────┐┌─────────┐
│        income  │         │  debt_ratio    ││  age  ││ student │
│        −0.12   │         │    +0.35       ││ +0.08 ││  +0.11  │
│                └─────────┘                └┴───────┘└─────────┘
└── 0.15                                                     0.72
```

**When to use:** Explaining a single prediction to a non-technical stakeholder. Debugging a specific unexpected prediction. Customer-facing explanations ("here's why you were declined").

---

### 15.2 Waterfall Plot (Single Prediction, Cleaner)

Same information as the force plot but laid out vertically, showing each feature's contribution as a step building from baseline to final prediction.

```
Starting from E[f(X)] = 0.15

       feature              SHAP     Running total
─────────────────────────────────────────────────
  debt_ratio = 0.82        +0.35  →  0.50  ████████
  is_student = True        +0.11  →  0.61  ████████████
  age = 28                 +0.08  →  0.69  ████████████████
  credit_score = 580       +0.05  →  0.74  (small + correction)
  income = $45k            −0.12  →  0.62
  (other features)          ...
  ──────────────────────────────
  Final: f(x*) = 0.72
```

**When to use:** Reports and presentations where you want a clean, readable single-prediction explanation. Cleaner than force plots for documents.

---

### 15.3 Beeswarm / Summary Plot (Whole Dataset)

![SHAP Beeswarm](https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.png)

**What it shows:** All features × all samples in one plot.

- **Rows** = features, sorted by mean |SHAP| (most important at top)
- **Each dot** = one sample from your dataset
- **X-axis** = SHAP value (right = pushed prediction up, left = pushed down)
- **Dot colour** = the feature's actual value for that sample (red = high, blue = low)

**How to read it:**

```
Feature: debt_ratio
  Red dots (high debt) cluster on the right  → high debt increases prediction
  Blue dots (low debt) cluster on the left   → low debt decreases prediction
  Consistent pattern                         → strong, reliable feature

Feature: age
  Red dots appear on BOTH sides of zero      → non-monotonic relationship!
  Very young AND very old increase risk,
  middle-age reduces it

Feature: zip_code (if shown as binary)
  Clean separation: one colour left, one right → binary feature, clear effect
```

**Why beeswarm beats a bar chart:**
- Direction of effect (does high value increase or decrease prediction?)
- Heterogeneity (do all samples behave the same or is there spread?)
- Non-linear effects (is the relationship monotone or U-shaped?)

A bar chart loses all three of these. Always look at the beeswarm.

---

### 15.4 Bar Plot (Global Ranking)

The simplest global SHAP view. Shows mean |SHAP value| per feature — the average magnitude of each feature's contribution across all samples.

```
debt_ratio     ████████████████████  0.29
income         ████████████          0.18
age            ████████              0.12
credit_score   ██████                0.09
is_student     ████                  0.06
zip_code       ██                    0.03
```

**Limitation:** Shows magnitude only, not direction. A feature that strongly increases risk for half your customers and strongly decreases it for the other half can have a high mean |SHAP| even though its average SHAP is near zero. Bar chart won't show this; beeswarm will.

**When to use:** Quick executive summary. Feature selection shortlist. When you need a one-number importance score per feature.

---

### 15.5 Dependence Plot (Feature Shape + Interaction)

**What it shows:** For one feature j:
- X-axis: the feature's actual values across the dataset
- Y-axis: the SHAP value for that feature, per sample
- Colour: optionally, a second feature (to reveal interactions)

The cloud of dots shows how the model uses this feature across different values.

```
Example: SHAP dependence plot for debt_ratio

SHAP(debt_ratio)
   +0.5 |                                          ·
        |                                       ···
   +0.3 |                                  ·····
        |                            ·····
   +0.1 |                   ·········
     0  |  ·················
  −0.1 |·
        └──────────────────────────────────────────
        0.1   0.3   0.5   0.7   0.9   debt_ratio value
```

In this case: the relationship is monotonically increasing — higher debt always means higher risk SHAP. The curve is slightly non-linear (steeper at high debt values).

**Adding interaction colour:** If you colour by `income` and see two bands at high debt_ratio — one red (high income, lower SHAP) and one blue (low income, higher SHAP) — that means there's an interaction between debt_ratio and income that the model has learned.

**How it relates to PDP:** A PDP shows the average prediction at each feature value. A SHAP dependence plot shows the per-sample SHAP value. The latter shows heterogeneity that averaging hides.

---

### 15.6 Decision Plot (Comparing Multiple Predictions)

**What it shows:** Multiple predictions explained simultaneously. Each sample is a line. The x-axis accumulates SHAP values feature by feature from baseline to final prediction. Each row adds the next most important feature's contribution.

```
baseline  0.15 ─── (all lines start here)
                │
    age         │  → lines diverge slightly here for different age groups
                │
    income      │  → lines diverge more; high-income lines go left
                │
    debt_ratio  │  → big divergence here; high-debt lines shoot right
                │
                └── final predictions  (lines land between 0.0 and 1.0)
```

**Reading it:** Lines that run close together → similar explanation structure. Lines that fan out at a particular feature → that feature creates the biggest difference between samples.

**When to use:** Comparing approved vs denied loan applications. Finding which feature causes the biggest split between two groups. Understanding why a model is inconsistent across similar customers.

---

## 16. Global Importance from SHAP

SHAP is fundamentally a local method (per-sample), but you can derive a global importance ranking:

```
Global Importance(j) = (1/N) × Σᵢ |φⱼ(xᵢ)|
                     = mean absolute SHAP value across all N samples
```

**Why this is better than other global importance measures:**

1. **It's consistent with local explanations.** The global ranking comes directly from summing the per-sample explanations. You can always drill into any specific prediction and see the local SHAP values that contribute to the global average.

2. **It gives you direction too.** Mean SHAP (not absolute) = average directional effect across the dataset. A feature with mean SHAP = +0.05 tends to push predictions up across the population.

3. **It handles interactions properly.** If feature A is only important in combination with feature B, this shows up in each sample's SHAP values and thus in the global average — unlike impurity importance which can miss interaction-driven contributions.

**The relationship to permutation importance:**

Both measure "how much does this feature contribute to predictions overall?" They often agree on top features but can differ for correlated features (permutation importance underestimates correlated features due to the substitute problem; SHAP splits credit between them).

---

## 17. SHAP Interaction Values

TreeSHAP can decompose each prediction into **main effects and pairwise interactions**:

```
f(x) = baseline + Σⱼ main_effect(j) + Σᵢ Σⱼ>ᵢ interaction(i,j)
```

The interaction value between features i and j measures: "How much extra prediction change comes from having both i and j together, beyond what you'd expect from their individual effects?"

```
Positive interaction (+):  i and j are synergistic
                           (having both is more powerful than each alone)
Negative interaction (−):  i and j are redundant
                           (having both gives less than the sum of each alone)
```

**Example — loan default:**

```
debt_ratio main effect:      +0.25  (high debt is bad)
income main effect:          −0.08  (good income is protective)
interaction(debt, income):   −0.03  (negative interaction = redundancy:
                                     good income already partly offsets high debt,
                                     so having both adds less than you'd expect)
```

**Practical use:** The interaction matrix (p × p heatmap of mean |interaction values|) reveals which feature pairs have the strongest interactions in your model — useful for feature engineering and for understanding complex model behaviour.

---

## 18. Bias-Variance of SHAP Estimates

| Variant | Bias | Variance | What controls error |
|---|---|---|---|
| TreeSHAP | None (exact) | None | — |
| LinearSHAP | None (exact) | None | — |
| KernelSHAP | Low (decreases with M) | 1/√M | Number of coalition samples M |
| DeepSHAP | Network approximation | Low | Network depth |
| GradientSHAP | Decreases with m | 1/√m | Number of gradient samples m |

**The background dataset adds another source of variation:**

If your background dataset is small or not representative, `E[f(X)]` (the baseline) is poorly estimated, and the distributions used to marginalise absent features are wrong. Use at least 100–300 representative background samples.

**The overfit trap:**

SHAP faithfully explains whatever model you give it. If your model overfits:

```
SHAP on training data: noise features have large |SHAP| values
                       (the model memorised them → genuinely uses them on training data)

SHAP on test data:     those same noise features have near-zero |SHAP| values
                       (the memorised patterns don't apply to new data)
```

**Always compute SHAP on held-out test data** if you want explanations that reflect generalisation, not memorisation.

---

## 19. Correlated Features — Where SHAP Still Struggles

SHAP is better than permutation importance for correlated features (no substitute problem) but still has an attribution ambiguity issue.

**The problem with marginal SHAP:**

When features A and B are correlated, computing `v(S)` with A in S but B absent means drawing B from its marginal distribution — which creates unrealistic (A=high, B=low) combinations.

**What happens in practice:**

```
True data: credit_score and num_late_payments are 90% correlated
           (they carry almost the same information)

With marginal SHAP:
  → Some credit goes to credit_score, some to late_payments
  → The split is somewhat arbitrary and depends on which the model
    happened to use more
  → Sum φ(credit_score) + φ(late_payments) is correct (efficiency holds!)
  → But the individual values can be misleading

With grouped permutation importance:
  → Permute both simultaneously → get their combined importance
  → More reliable for the group
```

**Practical guidance:**

```
When features are correlated (|r| > 0.7):
  ✅ Report group-level SHAP (sum φ values of the group)
  ✅ Use dependence plots to understand individual effects within the group
  ✅ Use conditional SHAP if you need within-group attribution
  ⚠️  Don't over-interpret individual SHAP values for correlated features
```

---

## 20. SHAP vs Other Methods

| | SHAP | Permutation Importance | LIME | Gini MDI |
|---|---|---|---|---|
| **Scope** | Local + Global | Global only | Local only | Global only |
| **Model-agnostic** | ✅ | ✅ | ✅ | ❌ Trees only |
| **Satisfies fairness axioms** | ✅ All 4 | ❌ | ❌ | ❌ |
| **Self-verifiable** | ✅ Efficiency check | ❌ | ❌ | ❌ |
| **Shows direction** | ✅ | ❌ | ✅ | ❌ |
| **Handles interactions** | ✅ | Partially | ❌ | Partially |
| **Stable / deterministic** | ✅ (TreeSHAP) | Approx | ❌ High variance | ✅ |
| **Fast for trees** | ✅ (TreeSHAP) | ✅ | ❌ | ✅ (free) |
| **Correlated features** | Better (not perfect) | Poor | Poor | Poor |

**When SHAP is the wrong choice:**

| Need | Better tool |
|---|---|
| "What change flips this prediction?" | Counterfactual explanations |
| "Give me a simple if-then rule" | Anchors |
| "Fast global ranking for 500 features" | Permutation importance |
| "Causal attribution with a causal graph" | Conditional / interventional SHAP + do-calculus |

---

## 21. Common Mistakes

**Mistake 1: Not running the efficiency check**

Always verify after computing SHAP values:
```
sum of SHAP values + baseline ≈ model prediction (for every sample)
```
If this fails, something went wrong. No other method lets you catch errors this way.

**Mistake 2: Interpreting SHAP as causal**

`φ(age) = +0.08` means "age pushed this prediction +0.08 above baseline in the model." It does NOT mean "if this person were older, their risk would increase by 0.08." The model uses age as a predictor — that's a statistical association, not causation.

**Mistake 3: Computing SHAP on training data and calling it "model explanation"**

If your model overfits, training-set SHAP shows what the model memorised — including spurious patterns. Compute SHAP on the test set.

**Mistake 4: Only looking at the bar chart**

Mean |SHAP| hides direction, heterogeneity, and non-linearity. Always look at the beeswarm. The bar chart is a summary; the beeswarm is the full picture.

**Mistake 5: Using KernelSHAP on a tree model**

TreeSHAP is exact and fast. KernelSHAP on a tree model is approximate and slow. There is no reason to use KernelSHAP on a tree model.

**Mistake 6: Comparing raw SHAP values across different models**

SHAP values are in the model's output space. A random forest outputting probabilities and an XGBoost outputting log-odds will have SHAP values on completely different scales, even for the same dataset. Normalise before comparing.

---

## 22. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SHAP — KEY FACTS                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WHAT IT IS                                                              │
│    For one prediction: how much did each feature push the result         │
│    above or below the average prediction?                                │
│    Answer: average marginal contribution across all feature orderings    │
│                                                                          │
│  THE BIG GUARANTEE (Efficiency axiom)                                    │
│    Σ SHAP values  =  f(x) − E[f(X)]                                     │
│    Always verify this. If it fails, something is wrong.                  │
│                                                                          │
│  FOUR AXIOMS (SHAP is the only method satisfying all four)               │
│    1. Efficiency:  values sum to prediction − baseline                   │
│    2. Symmetry:    equal contributors get equal values                   │
│    3. Dummy:       zero-contribution feature gets φ = 0                  │
│    4. Additivity:  ensembles decompose (RF = average of trees)           │
│                                                                          │
│  WHICH VARIANT TO USE                                                    │
│    Tree model (RF, XGB, LGB)  →  TreeSHAP (exact, fast)                 │
│    Linear model               →  LinearSHAP (φⱼ = βⱼ(xⱼ − E[Xⱼ]))     │
│    Neural network             →  GradientSHAP or DeepSHAP               │
│    Any black box              →  KernelSHAP (approximate, slow)         │
│                                                                          │
│  THREE FLAVOURS OF "ABSENT"                                              │
│    Marginal:      draw from P(Xⱼ) independently  [default]              │
│    Conditional:   draw from P(Xⱼ | features present) [realistic]        │
│    Interventional: draw from background directly [causal]               │
│                                                                          │
│  PLOTS (local → global)                                                  │
│    Force / Waterfall  →  single prediction, features as bars             │
│    Beeswarm           →  all samples, shows direction + heterogeneity    │
│    Bar                →  global ranking only (loses direction info)       │
│    Dependence         →  SHAP(j) vs x_j, reveals shape + interactions   │
│    Decision           →  multiple samples, cumulative lines              │
│                                                                          │
│  CRITICAL REMINDERS                                                      │
│    ▸ Always compute on test set, not training set                        │
│    ▸ SHAP ≠ causality. It's predictive attribution.                      │
│    ▸ For correlated features: report group-level, not individual         │
│    ▸ Bar chart alone is insufficient — always check beeswarm             │
│    ▸ KernelSHAP is an approximation; TreeSHAP is exact                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# SHAP — Interview Q&A

> 50 questions from fundamentals to senior-level. No code. Pure concepts, proofs, and the follow-ups that filter candidates.

---

## Table of Contents

1. [Foundations & Game Theory](#1-foundations--game-theory)
2. [The Four Axioms](#2-the-four-axioms)
3. [Computing SHAP Values](#3-computing-shap-values)
4. [TreeSHAP Internals](#4-treeshap-internals)
5. [SHAP Variants](#5-shap-variants)
6. [Marginal vs Conditional vs Interventional](#6-marginal-vs-conditional-vs-interventional)
7. [SHAP Plots](#7-shap-plots)
8. [Bias-Variance & Reliability](#8-bias-variance--reliability)
9. [Correlated Features](#9-correlated-features)
10. [SHAP vs Other Methods](#10-shap-vs-other-methods)
11. [Hard Senior Follow-ups](#11-hard-senior-follow-ups)

---

## 1. Foundations & Game Theory

---

**Q1. What is SHAP and where does it come from?**

SHAP (SHapley Additive exPlanations) is a framework for explaining individual model predictions by attributing the prediction to each input feature. It is grounded in cooperative game theory — specifically, the Shapley value originally defined by Lloyd Shapley in 1953. In the ML context, features are players, a single prediction is the game, and SHAP computes each feature's fair share of the difference between the model's prediction and its average prediction.

---

**Q2. What problem does SHAP solve that permutation importance or LIME don't?**

Permutation importance only provides a global score per feature — it can't explain why a specific prediction was made. LIME provides local explanations but is unstable (high variance between runs) and doesn't satisfy any formal fairness properties. SHAP provides both local and global explanations, is theoretically grounded in axioms that guarantee unique and consistent attribution, and is deterministic for tree models (TreeSHAP). It is the only common method that satisfies all four Shapley axioms simultaneously.

---

**Q3. Explain the Shapley value formula in plain English.**

Imagine adding features one at a time to the prediction in a random order. Every time you add a feature, you measure how much the prediction changes — its marginal contribution in that ordering. The Shapley value is the average of this marginal contribution across all possible orderings in which the features could have been added. This averaging ensures no feature benefits from being added first or last — it gets credited for exactly its expected marginal value.

---

**Q4. What is the baseline in SHAP and why does it matter?**

The baseline φ₀ is the model's expected prediction over the background dataset — typically E[f(X)]. SHAP values explain the deviation of a specific prediction from this baseline: Σφᵢ = f(x) − E[f(X)]. The baseline choice affects the absolute SHAP values but not the relative attribution between features. A different baseline (e.g., a specific reference point instead of the training mean) shifts all SHAP values by a constant. The efficiency axiom guarantees the sum is always correct regardless of baseline.

---

**Q5. Why does SHAP compute the expectation over missing features rather than setting them to zero?**

Setting absent features to zero changes the feature's distribution — the model sees inputs it was never trained on, and the prediction reflects out-of-distribution behaviour rather than the absence of information. Using the expected value (averaging over background samples) simulates marginalising out the feature — the model sees realistic inputs, and the attribution reflects the feature's contribution given realistic alternatives.

---

## 2. The Four Axioms

---

**Q6. State the four Shapley axioms and what each means in ML terms.**

**Efficiency:** All SHAP values sum to f(x) − E[f(X)]. The total "credit" for the prediction is fully distributed — nothing double-counted, nothing missing.

**Symmetry:** Two features with identical contributions to every possible coalition get equal SHAP values. No feature gets more credit just because of its label or position.

**Dummy:** A feature that changes no prediction for any coalition gets SHAP value zero. A truly useless feature gets zero attribution.

**Additivity:** SHAP values for a sum of two models equal the sum of SHAP values for each model separately. This is why SHAP values for a Random Forest equal the average SHAP values of its individual trees.

---

**Q7. Why is the efficiency axiom practically useful?**

It gives you a built-in verification test. After computing SHAP values for any sample, you can always check: does the sum of SHAP values equal the model's prediction minus the baseline? If it doesn't, something went wrong in the computation. No other feature importance method has this kind of self-verification property.

---

**Q8. How does the dummy axiom connect to feature selection?**

A feature with SHAP values near zero for all samples — not just on average but individually — is genuinely not being used by the model for any prediction. This is a stronger statement than having near-zero permutation importance, which can be confounded by correlated substitutes. If mean |SHAP| ≈ 0 for a feature, it's safe to remove it from the model without any performance impact.

---

**Q9. LIME doesn't satisfy the efficiency axiom. What does this mean practically?**

LIME fits a local linear model that approximates the black box around one sample. The sum of LIME coefficients (times the feature deviations) doesn't necessarily equal the model's actual prediction. This means you can't verify LIME explanations, and the attribution can silently be wrong — overstating or understating feature contributions without any way to detect it.

---

**Q10. What is the uniqueness property of Shapley values?**

The Shapley value is the **unique** function that satisfies all four axioms simultaneously. This means any attribution method that satisfies efficiency, symmetry, dummy, and additivity must compute Shapley values. No other method exists that satisfies all four. This uniqueness is what makes SHAP theoretically superior to ad-hoc attribution methods — there is only one "fair" attribution.

---

## 3. Computing SHAP Values

---

**Q11. Walk me through computing a Shapley value for a 2-feature model by hand.**

Say we have features A and B. For sample x, enumerate all 2! = 2 orderings: (A then B) and (B then A).

For ordering (A then B):
- A's marginal = v({A}) − v({}) = prediction with only A known minus baseline
- B's marginal = v({A,B}) − v({A}) = final prediction minus prediction with only A

For ordering (B then A):
- B's marginal = v({B}) − v({})
- A's marginal = v({A,B}) − v({B})

φ(A) = average of A's marginal across both orderings.
φ(B) = average of B's marginal across both orderings.
Verify: φ(A) + φ(B) = v({A,B}) − v({}) = f(x) − E[f(X)].

---

**Q12. For p features, how many orderings are there and why is exact Shapley computation intractable for large p?**

There are p! orderings. For p=20, that's 20! ≈ 2.4 × 10^18 orderings. Each ordering requires evaluating the characteristic function v(S) for each subset encountered, which itself involves model inference. Exact computation is completely intractable for any p beyond ~15. This motivates approximations (KernelSHAP) or structural exploitation (TreeSHAP).

---

**Q13. What does v(S) represent in the ML context, and how is it computed?**

v(S) is the expected model prediction when only the features in coalition S are known at their actual values for sample x, and all other features are integrated out over the background distribution. It is computed by:
1. Taking background samples
2. For each background sample, replacing features not in S with the background sample's values
3. Running the model on these hybrid inputs
4. Averaging the predictions

This is why each coalition evaluation requires multiple model inference calls (one per background sample), making naive Shapley computation expensive even with a fast model.

---

**Q14. Describe in one sentence why SHAP values and interaction effects are not the same thing.**

A SHAP main effect value φⱼ measures the average marginal contribution of feature j in isolation, averaging out all interactions with other features; the SHAP interaction value φᵢⱼ measures the additional prediction contribution that comes specifically from features i and j appearing together, beyond what their individual main effects would predict.

---

## 4. TreeSHAP Internals

---

**Q15. Why is TreeSHAP fast when naive Shapley computation on trees is not?**

For tree-structured models, `E[f(X) | X_S = x_S]` can be computed exactly by a weighted traversal of the tree, where at each split on a feature outside S, the sample is probabilistically split according to the training sample proportions (n_L/n_t to the left, n_R/n_t to the right). This allows computing v(S) for all 2^p subsets simultaneously in a single pass, reducing the complexity from O(T × 2^p) to O(TLD²) — where T is trees, L is leaves, D is depth.

---

**Q16. How does TreeSHAP handle a split on a feature that is "absent" from the current coalition?**

If the split feature j is absent from coalition S (meaning it's not fixed at its actual value), TreeSHAP propagates the sample through both branches simultaneously, weighted by the training sample proportions: n_L/n_t of the probability mass goes left, n_R/n_t goes right. This is the weighted traversal — the sample accumulates a weight at each leaf based on the product of these branching probabilities. The expected prediction is the sum of (leaf value × weight) over all leaves.

---

**Q17. What is the complexity of TreeSHAP and what do the terms represent?**

O(TLD²) where:
- T = number of trees in the ensemble
- L = maximum number of leaves per tree
- D = maximum depth of a tree (D² comes from tracking pairwise path statistics)

For a typical RF (T=100, D=10, L≤1024): roughly 100 × 1024 × 100 = 10^7 operations per sample — fast enough for real-time use in many applications.

---

**Q18. Can TreeSHAP be used on XGBoost and LightGBM, or only sklearn Random Forests?**

TreeSHAP works on any tree-based model — sklearn RandomForest, ExtraTrees, GradientBoostingClassifier, XGBoost, LightGBM, and CatBoost. The SHAP library implements TreeSHAP for all of these. For gradient boosted trees, SHAP values for the ensemble are the sum of tree-level SHAP values weighted by the learning rate, which follows directly from the additivity axiom.

---

**Q19. TreeSHAP computes "exact" SHAP values. Exact with respect to what assumption?**

Exact with respect to the **marginal** (independent) expectation assumption — it computes the exact Shapley values when absent features are drawn from their marginal distribution (approximated by the training set distribution stored in the tree's leaf statistics). It is not exact with respect to the conditional or interventional expectation. Using `feature_perturbation='interventional'` in the SHAP library changes the computation to the interventional expectation instead.

---

## 5. SHAP Variants

---

**Q20. When would you use KernelSHAP instead of TreeSHAP?**

KernelSHAP is used when the model is not tree-based — for example, SVM, k-NN, a custom ensemble, or any model without an internal tree structure. TreeSHAP is always preferred for tree models because it's exact, deterministic, and orders of magnitude faster. KernelSHAP should be considered a last resort for non-tree models, with awareness that it's slow and approximate.

---

**Q21. What is the "SHAP kernel" in KernelSHAP and why is it special?**

The SHAP kernel is a specific weight function applied to each coalition in the weighted linear regression:
`π(z') = (p−1) / [C(p,|z'|) × |z'| × (p−|z'|)]`

This particular weighting is mathematically derived from the Shapley axioms. It gives higher weight to coalitions with very few or very many features present. Using this specific kernel in the weighted regression makes the solution equal to the true Shapley values (as the number of sampled coalitions → all 2^p). This is what makes KernelSHAP, despite being a regression, equivalent to the game-theoretic Shapley formulation.

---

**Q22. What is the relationship between SHAP and Integrated Gradients for neural networks?**

Integrated Gradients attributes a neural network prediction by integrating the gradient of the output with respect to each input feature along a straight path from a baseline input x₀ to the actual input x. GradientSHAP is essentially Integrated Gradients averaged over multiple baseline samples (drawn from a background dataset), which aligns it with SHAP's expectation framework. Both methods satisfy the completeness property (equivalent to SHAP's efficiency axiom), but Integrated Gradients with a single baseline doesn't satisfy all four Shapley axioms; GradientSHAP is a better approximation of true SHAP values for neural networks.

---

**Q23. What is LinearSHAP and when is it exact?**

For a linear model f(x) = β₀ + Σⱼ βⱼxⱼ, the exact Shapley value for feature j is φⱼ = βⱼ(xⱼ − E[Xⱼ]). This follows because in a linear model, features don't interact — the marginal contribution of feature j is exactly βⱼ in every coalition ordering, so the average marginal contribution is simply βⱼ times the deviation from the mean. This is exact (not an approximation) and requires no background sampling.

---

## 6. Marginal vs Conditional vs Interventional

---

**Q24. What is the difference between marginal and conditional SHAP?**

Marginal SHAP replaces absent features with draws from their **unconditional (marginal) distribution** P(Xⱼ), ignoring correlations with present features. Conditional SHAP replaces absent features with draws from **P(Xⱼ | X_present)** — the distribution conditioned on the present features' actual values. When features are correlated, marginal SHAP creates unrealistic input combinations (e.g., young age with retirement status from older people). Conditional SHAP stays within the data distribution but is harder to compute and can cause different attribution patterns.

---

**Q25. When can conditional SHAP produce "surprising" results for a feature with no direct model effect?**

If feature A is correlated with feature B, and B has a direct effect on the model, then conditional SHAP can attribute non-zero importance to A — even if A never appears in the model and the dummy axiom would seem to require φ(A) = 0. This happens because conditioning on A changes the distribution of B (due to correlation), which changes v(S) for coalitions containing A. Conditional SHAP violates the dummy axiom for features that are correlated with informative features but have no direct model effect themselves.

---

**Q26. What does interventional SHAP measure, and when would you use it?**

Interventional SHAP uses background data directly as the replacement for absent features, without conditioning. This corresponds to the causal question: "What would the model predict if we physically intervened to set feature j to its background value, breaking any dependency between j and other features?" You would use this when you want attribution that reflects the causal structure of the problem rather than the joint distribution of features. It requires a causal graph or at least the assumption that setting one feature independent of others is meaningful.

---

**Q27. You run marginal and conditional SHAP on the same model and sample and get very different results for two correlated features. Which should you trust?**

Neither is universally "correct" — they answer different questions. Marginal SHAP answers: "How much does the model use this feature to make predictions, averaged over the marginal distribution?" Conditional SHAP answers: "How much does this feature contribute given the realistic joint distribution?" For model transparency (understanding what the model learned), marginal SHAP is more interpretable. For stakeholder communication where you want realistic scenarios, conditional SHAP is better. The key is to be explicit about which question you're answering.

---

## 7. SHAP Plots

---

**Q28. What information does a beeswarm plot convey that a bar chart doesn't?**

A beeswarm plot shows, for each feature, the full distribution of SHAP values across all samples — not just the mean. Each dot is one sample, coloured by the feature's actual value (red = high, blue = low). This reveals:
- The direction of the feature's effect (red dots on the right → high values increase prediction)
- Heterogeneity: whether the feature has a consistent or highly variable effect
- Non-monotonic effects: both red and blue dots on the positive side indicate a non-linear relationship
- Outliers: rare samples with unusually large |SHAP| values

A bar chart only shows the mean |SHAP| and loses all of this information.

---

**Q29. What does a SHAP dependence plot show, and how does it relate to a Partial Dependence Plot (PDP)?**

A SHAP dependence plot shows, for one feature j, the SHAP value φⱼ(xᵢ) (y-axis) plotted against the feature's actual value xᵢⱼ (x-axis). The shape of this scatter cloud reveals the functional form of the feature's effect on predictions. It is similar to a PDP in that both show how predictions change with feature value, but: PDP averages predictions (removing variation), while the SHAP dependence plot shows the same relationship at a per-sample level, revealing heterogeneity and interactions via the colour dimension.

---

**Q30. You have a SHAP dependence plot for "age" and you colour the dots by "income." The plot shows two distinct bands at high age values — one red and one blue — with very different SHAP(age) values. What does this indicate?**

This indicates a strong interaction between age and income. At high age, the SHAP contribution of age is very different depending on whether income is high (red) or low (blue). This means age's effect on the prediction is not independent — it changes substantially depending on the income level. This is precisely what SHAP interaction values φ(age, income) would capture. The two-band pattern is a visual signature of a significant interaction.

---

**Q31. When would you use a decision plot over a force plot?**

A force plot is designed for a single prediction — it shows one sample's features as arrows pushing the prediction left or right from the baseline. A decision plot shows multiple samples simultaneously — each sample is a line that accumulates SHAP values feature by feature from baseline to final prediction. You'd use a decision plot when you want to compare the explanation structure across a group of samples: for example, comparing approved vs denied loan applicants, or comparing a model's correct vs incorrect predictions. The spreading and converging of lines reveals where different groups diverge in their prediction drivers.

---

## 8. Bias-Variance & Reliability

---

**Q32. KernelSHAP is approximate. What controls the approximation error, and how do you reduce it?**

KernelSHAP samples M coalitions from the 2^p possible subsets and fits a weighted linear regression. Approximation error is proportional to 1/sqrt(M). Increasing M reduces error at this rate. Practically: M=500 gives rough estimates (acceptable for exploration), M=2000–5000 gives reasonable accuracy for most use cases. The background dataset size also matters — using 100–300 representative background samples is usually sufficient; too few gives noisy v(S) estimates.

---

**Q33. TreeSHAP is "exact" yet different background datasets give different SHAP values. Is this a contradiction?**

No. TreeSHAP is exact with respect to a specific expectation formula: it uses the training sample proportions stored in the tree's node statistics to compute marginalised predictions. Changing the background dataset changes E[f(X)] (the baseline) and changes the marginal distributions used to integrate out absent features. Different backgrounds → different SHAP values, but all of them correctly implement the Shapley formula given that background. The "exactness" refers to the Shapley calculation being exact, not to there being a single correct background.

---

**Q34. If your model overfits, what effect does this have on SHAP values computed on the training set?**

SHAP faithfully explains whatever model it's given. An overfit model has learned spurious patterns in the training data. On the training set, SHAP values will show noise features with large |SHAP| values because the model genuinely uses them for training-set predictions (it memorised them). On the test set, those same features will have near-zero SHAP values because the memorised patterns don't apply to new data. Always compute SHAP on the test set for interpretability that reflects generalisation, not memorisation.

---

**Q35. How do you verify that your SHAP computation is correct?**

Run the efficiency check: for every sample, the sum of SHAP values plus the baseline should equal the model's prediction:

`shap_values[i].sum() + expected_value ≈ model.predict(X[i])`

For classification with probability output, the model output should be the predicted probability. For log-odds output (XGBoost default), SHAP values are in log-odds space. Mismatch indicates: wrong model output type, wrong expected_value, or approximation error (for KernelSHAP). If the check fails for TreeSHAP, something is wrong with the implementation.

---

## 9. Correlated Features

---

**Q36. How does SHAP handle correlated features differently from permutation importance?**

Permutation importance suffers from the "substitute problem" — when feature A is permuted, the model can use correlated feature B as a substitute, making A appear unimportant. SHAP avoids this because it evaluates coalitions that include all combinations of features, not just "A present / A absent." In some orderings, both A and B are present, and their joint contribution is measured. However, marginal SHAP still creates unrealistic combinations by drawing absent features independently, which can misattribute credit between correlated features.

---

**Q37. Two features are perfectly correlated (r=1.0). What do their SHAP values look like under marginal SHAP?**

With marginal SHAP and perfectly correlated features A and B (r=1.0), the SHAP values depend on the model's parametrisation. If the model uses only A (B was excluded), then φ(B) = 0 and φ(A) carries all the credit. If the model uses both A and B with similar coefficients (or in a tree that splits on both), the marginal SHAP computation creates unrealistic inputs (A=high, B=low) which can lead to arbitrary credit splits between them. The sum φ(A) + φ(B) will always equal the total combined contribution (efficiency holds), but the individual splits can be unstable and depend on which feature the model happened to use more.

---

**Q38. A credit risk model uses both "credit_score" and "num_late_payments" which are highly correlated (r=−0.9). SHAP gives credit_score φ=−0.15 and late_payments φ=+0.45. How should you interpret this?**

Under marginal SHAP: the model uses both features. Credit_score (high value → low risk) is being pushed down from the baseline by −0.15, meaning having a good credit score reduces risk by that amount for this specific person. Late payments (low = good) is pushing risk up by +0.45 for this person's value. However, because of the correlation, part of credit_score's attribution may be reflecting the same underlying reality as late payments. This is not a bug — the model genuinely uses both features. But it means you can't add these up and say "the combined credit history effect is 0.30" — the split between the two is somewhat arbitrary due to the marginal expectation. Consider reporting them as a group for correlated features.

---

## 10. SHAP vs Other Methods

---

**Q39. Your colleague says "I'll just use SHAP for everything — permutation importance, feature selection, model debugging." What would you push back on?**

SHAP is excellent for local explanations and global summaries, but:

For **feature selection**, mean |SHAP| is a reasonable importance measure but still suffers from the correlated-feature attribution ambiguity under marginal SHAP. Drop-column importance or conditional SHAP may be more reliable for selecting which correlated features to keep.

For **debugging**, SHAP is excellent but if you just need a fast global ranking to compare features, permutation importance is faster (especially for non-tree models where KernelSHAP is slow).

For **regulatory explanations** requiring counterfactual reasoning ("what change would flip this decision?"), SHAP doesn't directly provide this — counterfactual explanations are better.

For **causal attribution**, marginal SHAP is not causal. You need interventional SHAP plus domain knowledge or a causal graph.

---

**Q40. SHAP says feature A is globally the most important. Permutation importance says feature B. Which is right?**

Both can be correct — they measure slightly different things. SHAP mean |φ| measures the average magnitude of each feature's marginal contribution in the Shapley framework (marginalising absent features). Permutation importance measures performance degradation when a feature is shuffled globally. They diverge when:

1. Features are correlated: permutation underestimates both A and B (substitute problem), while SHAP splits credit between them
2. The model has strong interaction effects: SHAP captures these in each feature's marginal contribution; permutation importance captures them in the performance drop when a feature is disrupted

Neither is definitively "correct" — they answer different questions. Investigate the discrepancy by looking at correlation structure and interaction plots.

---

**Q41. When would LIME and SHAP give the same local explanation, and when would they diverge most?**

They agree when the model is approximately linear in the neighbourhood of the sample, features are uncorrelated, and the LIME neighbourhood size is well-calibrated. They diverge most when:

- Features are correlated: LIME's random perturbation creates unrealistic samples; SHAP's coalition evaluation does too (under marginal), but with different weighting
- The model is highly non-linear near the sample: LIME's linear approximation fails; SHAP's Shapley-based attribution still correctly averages contributions
- LIME is run with few samples: LIME has high variance; TreeSHAP has zero variance

---

## 11. Hard Senior Follow-ups

---

**Q42. Explain why the Shapley value is the unique attribution satisfying all four axioms. What breaks if you drop one axiom?**

Shapley (1953) proved uniqueness constructively — any function satisfying the four axioms must produce the Shapley formula. Dropping any one axiom breaks uniqueness:

- Drop efficiency: now you can scale all values by any constant and still satisfy the remaining axioms. Attribution is no longer anchored to the total value.
- Drop symmetry: you can add arbitrary asymmetric biases (e.g., "first feature always gets double attribution") and still satisfy the other axioms.
- Drop dummy: you can assign non-zero values to features that contribute nothing, inflating or deflating attribution arbitrarily.
- Drop additivity: ensemble models can't be decomposed — the attribution for a forest isn't the sum of tree attributions, breaking the natural interpretation.

---

**Q43. SHAP interaction values decompose predictions into main effects and pairwise interactions. Is this decomposition exact or approximate?**

For TreeSHAP, it is exact — the SHAP interaction values are computed using the same exact algorithm with an additional term measuring the synergy between feature pairs. The decomposition satisfies: φ₀ + Σᵢ φᵢ + Σᵢ Σⱼ>ᵢ φᵢⱼ = f(x) (with appropriate definitions). The interaction matrix is exact for trees. For KernelSHAP, interaction values would require approximating higher-order terms and are not commonly computed.

---

**Q44. A Random Forest and a Gradient Boosted Tree achieve identical test AUC. Their SHAP rankings for the same dataset differ. Why, and which should you report?**

The Rashomon problem: multiple equally-valid models can attribute credit differently. The RF may rely more on feature A (perhaps split on it more often and earlier), while the GBM may rely more on feature B through residual-learning dynamics. Both are "correct" for their respective models. Reporting only one model's SHAP values implies a uniqueness of attribution that doesn't exist. Better practice: report SHAP values for both models, note where they agree (robust signal) and where they disagree (features where attribution is model-dependent), and focus stakeholder attention on the features that are important in both.

---

**Q45. Can SHAP values be negative, and what does that mean for both local and global contexts?**

Yes. Locally (for one sample), a negative SHAP value for feature j means that feature j's value for this specific sample **decreases** the prediction below the baseline. For example, income=$200k gives a negative SHAP value for default risk because high income reduces risk below the average.

Globally, negative SHAP values are averaged out in mean |SHAP|. But the average (non-absolute) SHAP across all samples can be negative for a feature — meaning that across the dataset, this feature tends to reduce predictions below baseline. This is sign information that the bar chart (mean |SHAP|) loses — the beeswarm plot preserves it.

---

**Q46. How does the choice of background dataset affect SHAP values, and how would you choose one in practice?**

The background dataset defines E[f(X)] (the baseline) and determines the distribution used to marginalise absent features. A background that is not representative of the population the model is deployed on can produce misleading SHAP values — features that are important in deployment but uncorrelated with the biased background may get underestimated.

In practice: use a diverse sample of 100–300 instances from the training distribution. If certain subgroups are important (e.g., high-risk customers), ensure the background represents them. K-means clustering on the training set and using cluster centroids is an efficient approach that preserves distributional coverage while reducing background size. For models in production, the background should reflect the inference distribution, not just training data.

---

**Q47. What is the "consistency" property of SHAP, and why does it matter more than people realise?**

Consistency means: if model A assigns more importance to feature j than model B in all coalitions (formally: model A's marginal contribution of j is always at least as high as model B's for any coalition S), then model A's Shapley value for j is at least as high as model B's. Permutation importance violates this: if you change a model to make feature j more important in every coalition, permutation importance can nonetheless decrease for j (due to correlated features absorbing the change). This consistency failure means permutation importance can give counterintuitive rankings even when the underlying model change is clear. SHAP's consistency guarantee means the ranking is monotone with the model's actual reliance on features.

---

**Q48. You're explaining a gradient boosted tree's predictions with SHAP to a regulator who asks: "Does high age cause higher credit risk?" How do you respond?**

You explain that the SHAP value for age shows that the model associates high age with higher predicted risk for this applicant — it pushes the prediction above baseline. However, this is a predictive association, not a causal claim. Age may correlate with income, employment type, and debt accumulation history in the training data. The model learned that age is a useful predictor, but we cannot say from SHAP alone that age causes higher risk. To make a causal claim, you would need to control for confounders via causal analysis. What SHAP tells the regulator is: "For this decision, the model relied on age as an input." Whether that reliance is appropriate — especially given fair lending regulations — is a separate policy question.

---

**Q49. SHAP values satisfy the efficiency axiom: they sum to f(x) − E[f(X)]. But what if your model outputs log-odds and you want SHAP values in probability space?**

This is a common practical trap. SHAP values are in the **model's output space**. If XGBoost outputs log-odds, SHAP values are log-odds differences. The efficiency axiom holds in log-odds space, not probability space. Transforming to probability space (applying the sigmoid) breaks additivity — you can't simply add probability-space SHAP values and expect them to sum to the probability difference.

The practical solution: accept that SHAP values are in log-odds space (which is fine for ranking and interpretation), or use a calibrated model that outputs probabilities natively (sklearn's `CalibratedClassifierCV`) and run SHAP on the calibrated output. There is no mathematically consistent way to get SHAP values in probability space for a log-odds model.

---

**Q50. What is the relationship between SHAP main effects, interaction values, and the full prediction, and why does this decomposition matter for model auditing?**

The full decomposition is:

`f(x) = φ₀ + Σᵢ φᵢᵢ + Σᵢ Σⱼ>ᵢ φᵢⱼ`

where φᵢᵢ is the main effect of feature i and φᵢⱼ is the interaction between i and j.

This matters for auditing because: if a protected characteristic (e.g., race) has a near-zero main effect φᵢᵢ but large interaction values φᵢⱼ with other features (e.g., zip_code, income), the model is still using race indirectly through interactions. A model can appear fair by main-effect SHAP values while being discriminatory through interaction terms. Full SHAP interaction analysis is therefore necessary for thorough fairness auditing — the main-effect bar chart alone is insufficient.

---

## Summary: What Interviewers Really Test at Each Level

```
Junior
  ├── Can you explain what a Shapley value is in plain English?
  ├── What does the efficiency axiom mean?
  ├── What is TreeSHAP and why is it faster than naive SHAP?
  └── What does a beeswarm plot show that a bar chart doesn't?

Mid
  ├── Compute a 2-feature Shapley value by hand
  ├── What is the difference between marginal and conditional SHAP?
  ├── When does KernelSHAP fail and how do you mitigate it?
  ├── Why is SHAP better than LIME for local explanations?
  └── What does a negative SHAP value mean globally?

Senior
  ├── What breaks if you drop one of the four axioms?
  ├── Why is SHAP consistent and permutation importance not?
  ├── Two models, same AUC, different SHAP rankings — what do you report?
  ├── SHAP in log-odds vs probability space — what's the right approach?
  └── How do SHAP interaction values reveal fairness violations that
      main effects miss?
```
