# Global vs Local Explainability — Complete Guide

> Surrogate models, Anchors, and Counterfactuals — what each answers, how each works, when each breaks, with full numerical examples.

---

## Table of Contents

1. [The Two Questions Explainability Answers](#1-the-two-questions-explainability-answers)
2. [The Full Landscape — Every Method in One Map](#2-the-full-landscape--every-method-in-one-map)
3. [Global Surrogate Models](#3-global-surrogate-models)
   - 3.1 The Idea
   - 3.2 The Algorithm
   - 3.3 Full Numerical Example
   - 3.4 Fidelity — How Good Is the Surrogate?
   - 3.5 Limitations
4. [Anchors — High-Precision Local Rules](#4-anchors--high-precision-local-rules)
   - 4.1 The Idea
   - 4.2 Precision and Coverage
   - 4.3 Full Numerical Example
   - 4.4 How Anchors Are Found
   - 4.5 Anchors vs LIME — Key Differences
   - 4.6 Limitations
5. [Counterfactual Explanations](#5-counterfactual-explanations)
   - 5.1 The Idea — The Contrastive Question
   - 5.2 What Makes a Good Counterfactual
   - 5.3 Full Numerical Example
   - 5.4 DiCE — Diverse Counterfactuals
   - 5.5 Actionability
   - 5.6 Limitations
6. [Comparing All Three — When to Use What](#6-comparing-all-three--when-to-use-what)
7. [Global vs Local — The Fundamental Tension](#7-global-vs-local--the-fundamental-tension)
8. [Bias-Variance Across Methods](#8-bias-variance-across-methods)
9. [Interview Q&A](#9-interview-qa)
10. [Summary Card](#10-summary-card)

---

## 1. The Two Questions Explainability Answers

Every explainability method answers one of two fundamentally different questions:

```
GLOBAL: "How does this model behave across all predictions?"
  → Describes the model as a whole
  → Audience: data scientists, model auditors, regulators
  → Output: rules, curves, feature rankings that hold generally

LOCAL:  "Why did this model make THIS specific prediction?"
  → Explains one individual decision
  → Audience: the person affected, fraud analysts, doctors
  → Output: feature attributions or rules for one sample
```

This distinction matters because a **globally important feature can be locally irrelevant**, and vice versa.

```
Example: credit scoring model

Global: "debt_ratio is the most important feature across all applicants"
Local for applicant #4521: "debt_ratio is near average — what drove the
                             denial was the combination of short employment
                             history AND recent address change"
```

You need both. Global methods tell you how the model behaves in general. Local methods tell you why it made a specific call.

---

## 2. The Full Landscape — Every Method in One Map

```
                      SCOPE
                  Global        Local
              ┌─────────────┬─────────────┐
              │ PDP / ALE   │ SHAP force  │
Model-   ANY  │ Surrogate   │ LIME        │
Agnostic      │ models      │ Anchors     │
              │ SHAP summary│ Counterfact │
              ├─────────────┼─────────────┤
              │ Gini MDI    │ TreeSHAP    │
Model-        │ Permutation │ (per sample)│
Specific      │ importance  │             │
              └─────────────┴─────────────┘
```

This file covers the three methods that complete the picture:

| Method | Scope | Output type |
|---|---|---|
| **Global Surrogate** | Global | Interpretable model (decision tree, linear) |
| **Anchors** | Local | If-then rule with precision guarantee |
| **Counterfactuals** | Local | "Minimal change to flip the prediction" |

---

## 3. Global Surrogate Models

### 3.1 The Idea

A complex model (RF, neural network, GBM) is powerful but opaque. What if you trained a **simple, interpretable model to imitate it**?

The surrogate doesn't try to learn the original labels — it tries to learn what the complex model *predicts*. If the surrogate is good enough (high fidelity), reading the surrogate gives you insight into the complex model.

```
Step 1:  Train your complex model f on (X, y)
Step 2:  Use f to predict on your training data: ŷ = f(X)
Step 3:  Train a simple model g on (X, ŷ)  ← note: ŷ, not y
Step 4:  Interpret g as a window into f
```

The surrogate g is not your production model — it's an approximation used only for understanding.

### 3.2 The Algorithm

```
INPUT: trained black-box model f, dataset X

1. Compute pseudo-labels:  ŷᵢ = f(xᵢ)  for all i

2. Choose surrogate type:
   - Decision Tree (max_depth=3–5):  produces human-readable rules
   - Linear model:                   produces feature coefficients
   - Rule list:                      produces ordered if-then rules

3. Train surrogate:  g = fit(X, ŷ)

4. Measure fidelity:
   R² = correlation between g(X) and f(X)
   (NOT correlation with y — fidelity is about mimicking f, not predicting y)

5. Interpret g as an approximation of f
   Extract: decision tree rules, feature coefficients, important splits
```

### 3.3 Full Numerical Example

**Dataset:** 10 customers. Predict churn. Features: tenure (months), monthly_charge ($), num_complaints.

Black-box RF predictions (ŷ):

```
 #  tenure  charge  complaints  ŷ (RF churn prob)
──────────────────────────────────────────────────
 1    2       90        3         0.92
 2    4       85        2         0.84
 3    6       70        1         0.61
 4   12       65        0         0.35
 5   18       60        0         0.28
 6   24       55        0         0.22
 7   30       50        0         0.15
 8   36       45        0         0.10
 9    3       95        4         0.97
10   48       40        0         0.08
```

**Train a decision tree on (X, ŷ) with max_depth=2:**

The tree finds:

```
                  [complaints ≥ 1?]
                  /                \
           YES (≥1)               NO (<1)
         [tenure ≤ 6?]         [charge ≤ 55?]
         /           \          /            \
    Leaf: 0.84    Leaf: 0.61 Leaf: 0.15   Leaf: 0.32
    "High risk"   "Med risk" "Low risk"   "Low-Med risk"
    (#1,2,9)      (#3)       (#7,8,10)    (#4,5,6)
```

**Reading the surrogate:**

> "The model appears to first check if there are complaints. If yes and the customer is new (tenure ≤ 6), they are very high churn risk. If no complaints, low monthly charge (≤$55) signals low churn risk."

**Fidelity check (R² between g(X) and f(X)):**

```
g predictions:  [0.84, 0.84, 0.61, 0.32, 0.32, 0.15, 0.15, 0.15, 0.84, 0.15]
f predictions:  [0.92, 0.84, 0.61, 0.35, 0.28, 0.22, 0.15, 0.10, 0.97, 0.08]

R² = 0.91 ← high fidelity: the surrogate captures 91% of the RF's variance
```

The surrogate is a good approximation. The rules extracted from it are a reasonable explanation of how the RF behaves.

### 3.4 Fidelity — How Good Is the Surrogate?

```
R² between g(X) and f(X)   Interpretation
──────────────────────────────────────────────────────
> 0.90                      Excellent — surrogate is a trustworthy proxy
0.75 – 0.90                Good — main patterns captured, details lost
0.50 – 0.75                Moderate — use with caution, major patterns only
< 0.50                      Poor — surrogate does not represent f well
                             Explanations based on g may be misleading
```

**Fidelity vs accuracy are different:**

A surrogate with R²=0.95 (high fidelity) but the black-box itself has accuracy=0.65 (mediocre model) means: the surrogate perfectly describes a mediocre model. High fidelity doesn't make the model good — it just means the surrogate is an accurate description of whatever the model does.

### 3.5 Limitations

**1. Fidelity caps interpretability**

To get high fidelity, the surrogate often needs to be complex (deeper tree, more leaves). But a deeper tree is less interpretable. There is a direct tension between fidelity and simplicity.

```
max_depth=2:   R²=0.72 (simple, interpretable, but misses detail)
max_depth=5:   R²=0.91 (faithful, but 32 leaves — hard to read)
max_depth=10:  R²=0.99 (nearly perfect, but uninterpretable)
```

**2. The surrogate explains the surrogate, not the model**

If fidelity is 0.75, the surrogate doesn't capture 25% of the model's behaviour. The rules you extract are correct for the surrogate but wrong for 25% of the black box's decisions. There is no way to know which decisions fall in that 25%.

**3. Global explanation hides local variation**

A decision tree surrogate gives global rules that apply on average. An individual prediction may be driven by a completely different part of the feature space that the simple surrogate misses.

---

## 4. Anchors — High-Precision Local Rules

### 4.1 The Idea

LIME gives you soft linear coefficients: "income contributed −0.18 to this prediction." That's precise but not intuitive.

Anchors give you a hard rule: "**IF** debt_ratio > 0.7 **AND** income < $40k **THEN** this prediction will be High Risk — 94% of the time."

An anchor is a set of conditions (predicates) that are **sufficient** to produce the same model prediction for any input that satisfies them — with high probability.

```
Anchor A is a rule on features of x* such that:
  P(f(x) = f(x*) | A(x) = True) ≥ τ

where:
  A(x) = True means x satisfies all conditions in A
  τ     is the precision threshold (typically 0.90 or 0.95)
```

In plain English: "Whenever this rule is satisfied, the model almost always makes the same prediction as it did for this specific sample."

### 4.2 Precision and Coverage

Two metrics define a good anchor:

```
PRECISION = P(f(x) = f(x*) | A(x) = True)
  How often the rule leads to the same prediction.
  High precision = the rule is reliable.
  Minimum target: τ = 0.90 or 0.95

COVERAGE = P(A(x) = True)
  Fraction of the dataset where this rule applies.
  High coverage = rule is broadly useful.
  Low coverage = very specific rule, only applies to rare cases.
```

**The tension:** Precision and coverage trade off against each other.

```
Narrow rule:  IF debt > 0.7 AND income < 40k AND age < 30 AND city=NYC
              → Precision: 0.99 (almost always correct)
              → Coverage:  0.02 (applies to 2% of dataset)

Broad rule:   IF debt > 0.7
              → Precision: 0.71 (correct 71% of the time — may not reach τ)
              → Coverage:  0.35 (applies to 35% of dataset)
```

Good anchors aim for **high precision (≥ τ) at reasonable coverage**. A rule that's only satisfied by 0.1% of the dataset isn't very useful even if it's 100% precise.

### 4.3 Full Numerical Example

**Sample:** applicant with debt_ratio=0.82, income=$38k, age=26, employment=1yr.
**Prediction:** 0.88 (high default risk).

After anchor search (explained in 4.4):

```
Anchor found: IF debt_ratio > 0.75 AND income < $45k
              THEN prediction = High Risk
              Precision: 0.93   (93% of applicants satisfying this rule
                                  also get High Risk prediction)
              Coverage:  0.14   (14% of the dataset satisfies this rule)
```

**Reading this:** "This applicant was denied because they have very high debt (>0.75) combined with low income (<$45k). Any applicant with this combination will be classified High Risk 93% of the time."

**Alternative anchors found:**

```
Anchor 2: IF debt_ratio > 0.80
          Precision: 0.89  Coverage: 0.08

Anchor 3: IF debt_ratio > 0.75 AND employment < 2yrs
          Precision: 0.96  Coverage: 0.06
```

Anchor 1 is preferred: highest coverage at precision ≥ 0.90.

### 4.4 How Anchors Are Found

Anchors are built using a **beam search** — a greedy breadth-first search over the space of possible rules, guided by statistical tests.

```
Algorithm sketch:

Start: empty rule A = {} (no conditions)

While precision(A) < τ:
  Candidate extensions: add one condition to A
    e.g., "debt > 0.5", "income < 50k", "age < 30", ...

  For each candidate A':
    Sample K random inputs x that satisfy A'
    Estimate precision: fraction where f(x) = f(x*)
    Use a statistical test (KL-LUCB bandit) to get confidence bounds

  Select B best candidates (beam width B)

Return the A with precision ≥ τ and highest coverage
```

The bandit approach is efficient: it stops sampling early for candidates that are clearly below threshold, focusing samples on promising candidates.

### 4.5 Anchors vs LIME — Key Differences

| Dimension | Anchors | LIME |
|---|---|---|
| **Output** | If-then rule | Linear coefficients |
| **Precision** | Guaranteed (≥ τ) | No guarantee |
| **Human-readable** | ✅ Yes (rule) | ⚠️ Requires interpretation |
| **Stable** | More stable | High variance |
| **Coverage** | Explicit metric | Not measured |
| **Interactions** | Naturally captured in rule | Misses (linear model) |
| **Negative features** | Not shown explicitly | Shows as negative coefficient |
| **Speed** | Slower | Faster |

**LIME** says: "income contributed −0.18, debt contributed +0.32 ..."
**Anchors** says: "IF debt > 0.7 AND income < 45k THEN High Risk (93% of the time)"

For a bank customer asking "why was I denied?", the Anchor explanation is more actionable and easier to understand.

### 4.6 Limitations

**1. Doesn't explain the features NOT in the anchor**

The anchor tells you what conditions are sufficient for the prediction. It doesn't tell you the full picture — which features helped, which didn't matter. A feature absent from the anchor might still have contributed.

**2. Can fail to find an anchor**

If no rule achieves precision ≥ τ, the algorithm returns the best it found but with precision below threshold. In models with noisy decision boundaries (e.g., SVMs near the boundary), no compact rule may be sufficient.

**3. Coverage can be very low**

An anchor specific to one person's profile may have coverage = 0.001. It explains this prediction but generalises to almost no one else.

**4. Anchors are not unique**

Multiple anchors can satisfy the precision requirement. The algorithm may return different anchors on different runs (though more stable than LIME).

---

## 5. Counterfactual Explanations

### 5.1 The Idea — The Contrastive Question

Anchors answer: "What is sufficient to produce this prediction?"
Counterfactuals answer: **"What is the minimum change that would produce a different prediction?"**

```
"You were denied a loan. What would need to change for you to be approved?"

Counterfactual: "If your debt_ratio had been 0.58 instead of 0.82
                 (keeping everything else the same),
                 the model would have approved you."
```

This is called a **contrastive explanation** — it explains the actual prediction by contrasting it with a nearby alternative. This is how humans naturally explain decisions: "I chose A over B because B lacked X."

Formally:

```
Find x' (the counterfactual) such that:
  1. f(x') = desired_outcome       ← prediction is flipped
  2. distance(x, x') is minimal   ← as few / small changes as possible
  3. x' is plausible               ← lies within the data distribution
```

### 5.2 What Makes a Good Counterfactual

Not all counterfactuals are useful. Four properties define a good one:

**1. Validity** — the prediction must actually flip

```
Bad: "If debt_ratio = 0.81 (was 0.82)" → still denied  (prediction didn't flip)
Good: "If debt_ratio = 0.58"            → approved      (prediction flipped)
```

**2. Proximity** — minimal change from original

```
Bad: "If you were 20 years younger with twice the income and in a different city"
     → technically valid but requires changing too much
Good: "If your debt_ratio were 0.24 lower"
     → one small specific change
```

**3. Plausibility / Actionability** — the counterfactual should represent a realistic state

```
Bad:  "If your age were 45 (currently 28)"
      → Not actionable. You can't change your age.
Good: "If you reduced your debt by $8k"
      → Actionable. The person can do this.

Bad:  "If you had 20 years of employment (currently 2)"
      → Technically plausible in the future, but requires 18 years of waiting
Good: "If you had 3 years of employment"
      → Achievable in 1 year
```

**4. Sparsity** — change as few features as possible

A counterfactual that requires changing 10 features simultaneously is less useful than one that requires changing 1 or 2.

### 5.3 Full Numerical Example

**Sample:** income=$38k, debt_ratio=0.82, age=26, employment=1yr, credit_score=580.
**Prediction:** 0.88 (denied — high default risk).
**Desired outcome:** approval (prediction < 0.40).

**Counterfactual search (minimise changes):**

```
Attempt 1 — change only debt_ratio:
  x' = [income=$38k, debt_ratio=0.55, age=26, employ=1yr, credit=580]
  f(x') = 0.38 ✅ prediction < 0.40
  Change required: reduce debt_ratio by 0.27
  Actionability: reduce total debt by ~$12k

Attempt 2 — change only credit_score:
  x' = [income=$38k, debt_ratio=0.82, age=26, employ=1yr, credit=660]
  f(x') = 0.42 ✗ still above 0.40
  
  x' = [income=$38k, debt_ratio=0.82, age=26, employ=1yr, credit=690]
  f(x') = 0.37 ✅
  Change required: credit_score +110 points
  Actionability: difficult — takes 12–18 months

Attempt 3 — change income only:
  x' = [income=$65k, debt_ratio=0.82, age=26, employ=1yr, credit=580]
  f(x') = 0.41 ✗ just above threshold

  x' = [income=$72k, ...]
  f(x') = 0.38 ✅
  Change required: income +$34k (84% increase)
  Actionability: unrealistic in the short term

Attempt 4 — change two features (employment + credit_score):
  x' = [income=$38k, debt_ratio=0.82, age=26, employ=3yr, credit=640]
  f(x') = 0.35 ✅
  Changes: employment +2yr, credit_score +60
  Actionability: achievable over 2 years
```

**Best counterfactual (most actionable):**

```
Current state:       debt_ratio=0.82, credit=580  →  DENIED (0.88)
Counterfactual 1:    debt_ratio=0.55              →  APPROVED (0.38)
                     Action: reduce debt by ~$12k

Counterfactual 4:    employment=3yr, credit=640   →  APPROVED (0.35)
                     Action: build credit and wait 2 years
```

Two valid, actionable counterfactuals are offered — the applicant can choose based on what's feasible.

### 5.4 DiCE — Diverse Counterfactuals

A single counterfactual can be misleading if it suggests the only path to approval when multiple paths exist. **DiCE** (Diverse Counterfactual Explanations, Mothilal et al., 2020) generates K diverse counterfactuals simultaneously.

```
DiCE objective: find K counterfactuals x'₁, ..., x'ₖ that:
  1. Are all valid (flip the prediction)
  2. Are close to x (minimal change)
  3. Are diverse from each other (change different subsets of features)
  4. Are plausible (within data distribution)
```

**DiCE output for our example:**

```
CF1: reduce debt_ratio to 0.55  (debt reduction)
CF2: increase credit_score to 690 + employment to 2yr  (credit building)
CF3: increase income to $62k + reduce debt to 0.70  (income + modest debt)
CF4: age=26, employment=4yr, credit=620  (wait and build history)
```

Presenting all four gives the applicant a realistic picture of their options.

### 5.5 Actionability

Not all feature changes are actionable. A well-designed counterfactual system should respect constraints:

```
Immutable features (cannot change):
  age, gender, race, nationality → must not appear in counterfactuals

Semi-mutable (change slowly / with effort):
  credit_score → can improve, but takes 6–18 months
  employment_years → can only increase over time

Freely changeable:
  debt_ratio → can reduce by paying down debt
  monthly_payment → can refinance
```

Counterfactual algorithms should accept **feature constraints** to ensure they only suggest realistic paths.

### 5.6 Limitations

**1. Multiple valid counterfactuals exist — which to present?**

An infinite number of counterfactuals can satisfy "prediction flips, minimal change." The choice of which to present affects the user's options. Always present multiple (DiCE) rather than one.

**2. Counterfactuals don't explain the model globally**

"If your debt were lower, you'd be approved" explains this denial but says nothing about how debt affects the model for other applicants.

**3. Plausibility is hard to enforce**

A counterfactual with income=$120k (currently $38k) is technically valid (flips prediction) but completely unrealistic. Enforcing plausibility requires estimating the data distribution — which is itself a hard problem.

**4. Causal validity vs statistical validity**

A counterfactual says "if X changed, f(x) would change." This is a statement about the model, not about the real world. "If your debt were lower, you'd be approved" doesn't mean reducing debt causes lower default risk — the model might have learned a spurious correlation between debt and other protective factors.

---

## 6. Comparing All Three — When to Use What

| | Surrogate Model | Anchors | Counterfactuals |
|---|---|---|---|
| **Scope** | Global | Local | Local |
| **Output** | Interpretable model (tree/linear) | If-then rule | Minimal change to flip prediction |
| **Answers** | "How does the model generally work?" | "What is sufficient for this prediction?" | "What would need to change?" |
| **Best for** | Model auditing, overall understanding | Customer-facing explanations of decisions | Actionable guidance for the person affected |
| **Audience** | Data scientists, regulators | Customer, analyst | Customer, case worker |
| **Captures interactions** | Through surrogate structure | Naturally in rule | Not directly |
| **Coverage** | Full dataset | Measured metric | Single point |
| **Stable** | ✅ (deterministic) | Moderate | Moderate |
| **Causal** | ❌ | ❌ | ❌ (model-level, not causal) |

**Practical decision guide:**

```
"I need to understand how my model works overall"
→ Global Surrogate (then verify with fidelity score)

"I need to explain to a customer why they were denied"
→ Counterfactuals (actionable) + optionally Anchor (rule-based)

"I need a rule that reliably predicts this outcome"
→ Anchors (with precision guarantee)

"I need per-feature attribution for a single prediction"
→ SHAP or LIME (not in this file — see 4_SHAP.md and 5_LIME.md)

"I'm building a compliance report for a regulatory audit"
→ Global Surrogate (model overview) + Counterfactuals (individual decisions)
```

---

## 7. Global vs Local — The Fundamental Tension

A globally unimportant feature can be the decisive factor locally. A locally small SHAP value can belong to a globally dominant feature that just doesn't vary much for this sample.

**Example:**

```
Model: predict hospital readmission

Global surrogate says: age is the most important feature

Local counterfactual for patient #247: "If you had attended the follow-up
  appointment within 7 days, you would not be predicted for readmission"
  → follow-up attendance is the decisive local factor for this patient,
    even though it's globally ranked 5th in importance

Global SHAP for patient #247: age SHAP = +0.02 (small — this patient's
  age is near average so it barely contributes to this prediction)
  appointment SHAP = +0.31 (dominant for this patient)
```

**Implication:** Never use global explanations to explain individual decisions. The global model tells you what matters across the population. The local explanation tells you what matters for one person. They answer different questions and should never be conflated.

---

## 8. Bias-Variance Across Methods

| Method | Bias Source | Variance Source |
|---|---|---|
| Surrogate | Fidelity gap (if fidelity < 1.0, the surrogate is biased) | Surrogate training variance (especially for small datasets) |
| Anchors | Precision threshold τ (lower τ = more bias in rule quality) | Beam search sampling; statistical test for precision |
| Counterfactuals | Distance metric choice (what "minimal" means) | Optimisation landscape (multiple optima) |

**Surrogate model fidelity is the key bias diagnostic:**

```
Fidelity R² = 0.95 → bias ≈ 5% of model behaviour not captured
Fidelity R² = 0.60 → bias ≈ 40% — explanations describe a different model
```

Always report fidelity alongside surrogate explanations. A surrogate without a reported fidelity score is meaningless.

---

## 9. Interview Q&A

**Q: What is a global surrogate model and what is its key limitation?**

A global surrogate is a simple interpretable model (decision tree, linear regression) trained to mimic a complex black-box model's predictions (not the true labels). Its key limitation is fidelity: the surrogate only captures a fraction of the black box's behaviour. Any part of the model not captured by the surrogate is a blind spot — the surrogate gives explanations that are wrong for those cases, with no way to know which cases they are.

---

**Q: What is the difference between surrogate model fidelity and accuracy?**

Fidelity measures how well the surrogate approximates the black-box model's predictions (R² between surrogate(X) and black_box(X)). Accuracy measures how well either model predicts the true labels (y). A surrogate can have high fidelity (closely mimics the RF) but low accuracy (because the RF itself is not very accurate). High fidelity means "this is a good explanation of the model." High accuracy means "this is a good predictive model." You want both, but they are independent.

---

**Q: What is an Anchor and what does its precision metric mean?**

An Anchor is a local if-then rule: a set of conditions on features such that whenever those conditions are met, the model gives the same prediction as it gave for the specific sample being explained. Precision = the probability that any new sample satisfying the rule receives the same prediction. A precision of 0.93 means 93% of inputs satisfying the rule get the same prediction — the rule is reliable 93% of the time.

---

**Q: What is coverage in the context of Anchors and why does it matter?**

Coverage is the fraction of the dataset where the anchor's conditions are satisfied. Low coverage means the rule only applies to a tiny subset of inputs — it explains this one prediction but generalises poorly. High coverage means the rule is broadly applicable across many similar cases. The ideal anchor has both high precision (reliable) and high coverage (general). In practice these trade off: broader rules have lower precision.

---

**Q: What is a counterfactual explanation and what makes one "good"?**

A counterfactual is the minimal change to an input that would flip the model's prediction to a desired outcome. A good counterfactual is: (1) valid — the prediction actually flips; (2) proximal — the change is small; (3) plausible — the counterfactual lies within the realistic data distribution; (4) sparse — as few features as possible are changed; (5) actionable — the suggested changes are ones the person can actually make.

---

**Q: Why should you present multiple counterfactuals instead of just one?**

A single counterfactual implies there is one path to a different outcome, when many paths may exist. Different paths require different efforts and may be feasible for different people. DiCE generates diverse counterfactuals that change different subsets of features — giving the person affected a realistic picture of their options rather than a single potentially impractical suggestion.

---

**Q: A data scientist says "our global surrogate tree explains the model." What questions would you ask?**

What is the fidelity R² between the surrogate's predictions and the black box's predictions? If fidelity is below 0.85, the surrogate is a significant distortion of the model. What is max_depth? Too shallow = low fidelity; too deep = uninterpretable. Does the surrogate capture the model's behaviour equally across all subgroups, or is fidelity lower for minority groups? And critically: are you using this surrogate to explain individual decisions? If yes, you need local methods (SHAP, counterfactuals) instead — the global surrogate may not represent this specific prediction.

---

**Q: What is the key difference between Anchors and LIME as local explanation methods?**

LIME fits a linear model locally and returns coefficients — soft, continuous attributions that say "each feature contributed this much." Anchors return a hard rule with a precision guarantee — a set of conditions sufficient to produce the same prediction. LIME's output requires interpretation ("a coefficient of +0.32 means debt matters"). Anchors' output is directly actionable ("IF debt > 0.7 AND income < 45k THEN High Risk — 93% reliable"). Anchors naturally capture feature interactions in their rules; LIME cannot because it uses a linear model.

---

## 10. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GLOBAL vs LOCAL — THREE METHODS                                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GLOBAL SURROGATE                                                        │
│    Train simple model g on (X, f(X)) — not on true labels y             │
│    Measure fidelity: R²(g(X), f(X)) — report this always               │
│    Output: decision tree rules, linear coefficients                     │
│    Limitation: fidelity gap = blind spot in explanations                │
│    R² < 0.75 → do not trust the explanations                            │
│                                                                          │
│  ANCHORS                                                                 │
│    Local if-then rule: IF A THEN prediction = f(x*) with prob ≥ τ      │
│    Precision: P(same prediction | rule satisfied)  — target ≥ 0.90     │
│    Coverage: fraction of dataset where rule applies                     │
│    Found via: beam search + KL-LUCB bandit sampling                     │
│    Output: "IF debt > 0.7 AND income < 45k THEN High Risk (93%)"       │
│    Advantage over LIME: stable, guaranteed precision, captures interact  │
│                                                                          │
│  COUNTERFACTUALS                                                         │
│    Minimal change to x that flips f(x) to desired outcome               │
│    Good CF properties: valid, proximal, plausible, sparse, actionable   │
│    Use DiCE for multiple diverse counterfactuals                        │
│    Immutable features (age, race) must be excluded from changes         │
│    Output: "If your debt_ratio were 0.55, you'd be approved"            │
│    Does NOT explain causal mechanism — only model behaviour             │
│                                                                          │
│  WHEN TO USE WHAT                                                        │
│    Model audit / overall understanding  →  Global Surrogate             │
│    Customer: "why was I denied?"        →  Counterfactuals              │
│    Need a reliable rule for a class     →  Anchors                      │
│    Feature attribution for one person   →  SHAP (see 4_SHAP.md)        │
│                                                                          │
│  THE FUNDAMENTAL RULE                                                    │
│    Never use global explanations to explain individual decisions         │
│    Global importance ≠ local importance for any specific sample         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Ribeiro et al. (2016)** — *"Why Should I Trust You?"* — LIME paper (surrogate models mentioned).
- **Ribeiro et al. (2018)** — *Anchors: High-Precision Model-Agnostic Explanations.* AAAI.
- **Wachter et al. (2017)** — *Counterfactual Explanations Without Opening the Black Box.* Harvard JOLT.
- **Mothilal et al. (2020)** — *Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations.* FAccT. — DiCE.
- **Molnar (2022)** — *Interpretable Machine Learning*, Chapters 5.4, 9.3, 9.4.
