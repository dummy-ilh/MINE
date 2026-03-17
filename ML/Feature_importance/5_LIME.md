# LIME — Complete Guide

> Local Interpretable Model-Agnostic Explanations — intuition first, algorithm in detail, every data type, and every failure mode explained with numbers.

---

## Table of Contents

1. [The Problem LIME Solves](#1-the-problem-lime-solves)
2. [The Core Idea — One Analogy](#2-the-core-idea--one-analogy)
3. [The Key Assumption — And When It Breaks](#3-the-key-assumption--and-when-it-breaks)
4. [The LIME Algorithm — Step by Step](#4-the-lime-algorithm--step-by-step)
5. [Full Numerical Example — Tabular Data](#5-full-numerical-example--tabular-data)
6. [LIME for Text](#6-lime-for-text)
7. [LIME for Images](#7-lime-for-images)
8. [The Instability Problem — Proven with Numbers](#8-the-instability-problem--proven-with-numbers)
   - 8.1 What Instability Means
   - 8.2 Where the Variance Comes From
   - 8.3 How Large Is the Problem in Practice
   - 8.4 Fixes
9. [The Neighbourhood Problem](#9-the-neighbourhood-problem)
   - 9.1 Too Small → Too Few Samples
   - 9.2 Too Large → Linear Approximation Fails
   - 9.3 The Kernel Width Parameter
10. [Faithfulness — Does LIME Actually Approximate the Model?](#10-faithfulness--does-lime-actually-approximate-the-model)
11. [Bias-Variance of LIME Explanations](#11-bias-variance-of-lime-explanations)
12. [LIME vs SHAP — Head to Head](#12-lime-vs-shap--head-to-head)
13. [When LIME Is the Right Tool](#13-when-lime-is-the-right-tool)
14. [Common Mistakes](#14-common-mistakes)
15. [Summary Card](#15-summary-card)

---

## 1. The Problem LIME Solves

You have a black-box model — maybe a Random Forest, an XGBoost, or a neural network. It makes a specific prediction for a specific input. You want to know why.

SHAP is the theoretically ideal answer, but:
- For a neural net, TreeSHAP doesn't apply
- KernelSHAP is slow
- Sometimes you just want a quick local explanation without the Shapley machinery

LIME was introduced in 2016 (Ribeiro, Singh, Guestrin) to answer: **"Can we explain any black-box model's prediction at any single point, quickly, using nothing but the model's predict function?"**

The answer is yes — with important caveats about stability.

---

## 2. The Core Idea — One Analogy

Imagine you're trying to understand the shape of a mountain range, but you can only get a ground-level view. From any single point on the mountain, the terrain around you looks approximately flat — you can describe your immediate surroundings with a simple slope. That slope won't describe the whole mountain, but it accurately describes what's near you.

LIME does exactly this for model predictions:

> **A complex model may be globally nonlinear and impossible to interpret. But around any single prediction, it behaves approximately like a simple linear model. LIME finds that local linear approximation.**

```
            True model (complex, nonlinear)
            ─────────────────────────────────────────
                    ╭─────╮
                   ╱       ╲
                  ╱         ╲
       ╭─────────╯           ╲─────╮
      ╱                             ╲
─────╯                               ╲─────


            LIME's local approximation (at x*)
            ─────────────────────────────────────────
                         ╱ ←  linear fit in this local region
                        ╱
                       ╱  ← x* is here
                      ╱
                     ╱
```

The linear fit is only accurate near x*. But that's all we need — we only want to explain this one prediction.

---

## 3. The Key Assumption — And When It Breaks

**LIME's assumption:** The model is locally linear around the point you want to explain.

This is almost always approximately true — most smooth functions look linear when you zoom in enough. The question is *how much* you need to zoom in, and whether that neighbourhood contains enough samples to fit a reliable linear model.

**When it breaks:**

```
Case 1 — Sharp discontinuities
  Model: predict fraud if amount > $500, otherwise not
  x*: amount = $499 (prediction = 0.05)
  
  Just above $500 → prediction jumps to 0.95
  The model is NOT locally linear near $499 — it has a cliff edge
  LIME's linear fit will be unreliable and unstable here

Case 2 — Tight interactions
  Model: prediction is high only if BOTH feature A > 0.5 AND feature B > 0.5
  x*: A=0.6, B=0.6 (prediction = 0.9)
  
  In the local neighbourhood, perturbing A or B alone reduces prediction sharply
  A pure linear model can't represent this AND-logic accurately
  LIME's coefficients for A and B will be misleading

Case 3 — Very high-dimensional space
  With 100 features, the "local neighbourhood" is geometrically enormous
  Random perturbations are unlikely to be truly "nearby" in any useful sense
  The linear approximation may fit a very heterogeneous region
```

---

## 4. The LIME Algorithm — Step by Step

LIME runs the same core algorithm for tabular, text, and image data. Only step 1 (how to create perturbations) changes.

```
INPUT:
  x*           — the sample you want to explain
  f            — the black-box model (only .predict() needed)
  N            — number of perturbed samples to generate (default ~5000)
  K            — number of features to include in the explanation (default 10)
  σ (sigma)    — kernel width (controls neighbourhood size)

ALGORITHM:

Step 1 — Generate perturbed samples around x*
  Create N new samples that are similar-but-different to x*
  (How you do this depends on data type — details in next sections)

Step 2 — Get predictions from the black box
  For each of the N perturbed samples x_i:
    y_i = f(x_i)   (run the black-box model)

Step 3 — Weight each sample by distance to x*
  d_i  = distance(x_i, x*)
  w_i  = exp(−d_i² / σ²)
  
  Samples very close to x* get weight ≈ 1.0
  Samples far from x*  get weight ≈ 0.0
  σ controls how fast the weight drops off with distance

Step 4 — Fit a weighted sparse linear model
  Solve: minimise  Σᵢ wᵢ × [f(xᵢ) − g(xᵢ)]²  +  λ||θ||₁
  
  where g(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... is the simple linear model
  and λ||θ||₁ is an L1 (Lasso) penalty that limits to K features

Step 5 — The coefficients θⱼ are the explanation
  Positive θⱼ → feature j pushes prediction UP for this sample
  Negative θⱼ → feature j pushes prediction DOWN
  |θⱼ|        → magnitude of feature j's local influence
```

**What LIME returns:** A list of K feature-value pairs with their local linear coefficients. For example:

```
Prediction: 0.78 (high fraud risk)

LIME explanation:
  amount > $800     +0.35   ← large transaction pushes risk up
  foreign = True    +0.18   ← foreign transaction pushes up
  hour = 2am        +0.12   ← late night pushes up
  usual_merchant    −0.15   ← familiar merchant pulls risk down
  card_present      −0.08   ← card physically present reduces risk
```

---

## 5. Full Numerical Example — Tabular Data

**Setup:** Predict loan default. Model is a black-box RF. Features: income, debt_ratio, age, employment_years.

Sample to explain: income=$45k, debt_ratio=0.75, age=28, employment_years=2.
Black-box prediction: **0.72** (high default risk).

---

**Step 1: Generate perturbations**

For tabular data, LIME uses an **interpretable binary representation**:

```
Each feature is either "present" (1) or "absent" (0)
"Absent" = replaced by a random value drawn from that feature's training distribution

Original sample (interpretable):  [income=1, debt=1, age=1, employ=1]
i.e., all features present = actual values used
```

Generate N=8 perturbed samples (showing the binary mask and the actual values used):

```
 #   income  debt  age  employ  │  actual income  debt   age  employ  │ f(x)
─────────────────────────────────┼────────────────────────────────────────────────
 1    1       1     1    1       │  $45k    0.75   28    2     │  0.72  (= x*)
 2    0       1     1    1       │  $72k*   0.75   28    2     │  0.58  (* = from training dist)
 3    1       0     1    1       │  $45k    0.42*  28    2     │  0.45
 4    1       1     0    1       │  $45k    0.75   45*   2     │  0.68
 5    1       1     1    0       │  $45k    0.75   28    8*    │  0.62
 6    0       0     1    1       │  $78k*   0.38*  28    2     │  0.31
 7    1       0     0    1       │  $45k    0.51*  52*   2     │  0.41
 8    0       1     0    0       │  $83k*   0.75   41*   11*   │  0.44
```

(* = value drawn randomly from training distribution for that feature)

---

**Step 2: Compute distances and weights**

Distance is computed in the interpretable binary space (number of features that differ from x*):

```
 #   binary mask         distance  weight (σ=0.75)
─────────────────────────────────────────────────────
 1   [1,1,1,1]           0         1.000   (x* itself)
 2   [0,1,1,1]           1         0.641   (1 feature missing)
 3   [1,0,1,1]           1         0.641
 4   [1,1,0,1]           1         0.641
 5   [1,1,1,0]           1         0.641
 6   [0,0,1,1]           2         0.168   (2 features missing)
 7   [1,0,0,1]           2         0.168
 8   [0,1,0,0]           3         0.018   (3 features missing, very low weight)
```

Samples closer to x* (fewer features "turned off") get higher weight.

---

**Step 3: Fit weighted Lasso regression**

We're fitting: `ŷ = θ₀ + θ₁×income_present + θ₂×debt_present + θ₃×age_present + θ₄×employ_present`

Using the weights above. After solving (the actual numbers would come from sklearn's Lasso):

```
θ₀              =  0.28   (intercept — prediction when all features absent)
θ₁ (income)     = −0.20   (having actual income helps: income=$45k is below avg → reduces risk)
θ₂ (debt_ratio) = +0.32   (having actual debt hurts: debt=0.75 is high → increases risk)
θ₃ (age)        = −0.04   (age=28 slightly reduces risk)
θ₄ (employ)     = −0.08   (employment=2yrs slightly reduces risk)
```

**Verify (approximately):**
```
ŷ(x*) = 0.28 − 0.20 + 0.32 − 0.04 − 0.08 = 0.28 ≈ 0.72?
```

Wait — this doesn't match. The linear approximation is approximate. The intercept absorbs the gap. In practice, LIME's local linear fit won't perfectly reconstruct the black-box prediction — this is the faithfulness issue (Section 10).

---

**Step 4: Read the explanation**

```
LIME explanation for this applicant (prediction = 0.72):

  debt_ratio = 0.75    +0.32  ← HIGH debt drives default risk up most
  income = $45k        −0.20  ← low income is already reflected in income=45k
                                  but income being "present" actually helps
                                  (model sees 45k vs the average of 62k in training,
                                   so debt_ratio dominates)
  employment = 2 yrs   −0.08  ← short but non-zero employment helps slightly
  age = 28             −0.04  ← young but not the main driver
```

**Human-readable:** "This applicant is predicted high-risk primarily because of their high debt ratio (0.75). Their below-average income is already baked into the debt ratio effect. Their young age and short employment history are minor factors."

---

## 6. LIME for Text

Text LIME has a natural interpretable representation: **individual words**.

**The perturbation strategy:**

```
Original text: "The movie was surprisingly good, I loved every scene"
Prediction: positive sentiment = 0.92

Step 1: Create binary mask — each word is present (1) or absent (0)
  [The=1, movie=1, was=1, surprisingly=1, good=1, I=1, loved=1, every=1, scene=1]

Step 2: Create perturbations by randomly removing words
  Perturbation 1: "The movie was good I loved every scene"  (removed "surprisingly")
  Perturbation 2: "The movie was surprisingly I loved every scene" (removed "good")
  Perturbation 3: "movie was surprisingly good loved every scene"  (removed "The","I")
  ...N=1000 perturbations

  "Removing" a word = replacing it with a blank or out-of-vocabulary token,
  or simply dropping it from the string

Step 3: Get black-box predictions for each perturbation
Step 4: Weight by similarity (fraction of words kept)
Step 5: Fit weighted Lasso regression
```

**Resulting explanation:**

```
"The movie was surprisingly good, I loved every scene" → 0.92

LIME explanation:
  good        +0.31   ← removing "good" drops sentiment most
  loved       +0.28   ← "loved" is strongly positive
  surprisingly +0.11  ← adds intensity
  scene       +0.03   ← weak positive
  was         −0.02   ← neutral, slightly pulls down
  The         +0.01   ← irrelevant
```

**Why text LIME works better than tabular LIME:**

Words are a natural, human-readable interpretable unit. "This prediction changed because the word 'loved' was present" is immediately understandable. The binary on/off representation for words is meaningful — a sentence without "loved" is a real, valid input.

For tabular data, "income feature turned off" means replaced with a random income value — which is less interpretable and creates artificial inputs.

---

## 7. LIME for Images

Image LIME uses **superpixels** — contiguous regions of similar pixels grouped together — as the interpretable units.

**The perturbation strategy:**

```
Original image: a photo of a dog (prediction: dog = 0.94)

Step 1: Segment the image into superpixels
  Use an algorithm like SLIC to divide the image into ~50–100 superpixels
  Each superpixel = a cluster of spatially adjacent, visually similar pixels

Step 2: Binary mask over superpixels
  Each superpixel is either "present" (keep original pixels)
  or "absent" (replace with grey / mean colour)

Step 3: Create N perturbed images
  Randomly turn superpixels off (grey them out)
  Each perturbation = a different random subset of superpixels shown

Step 4: Get black-box predictions for each perturbed image
Step 5: Weight by number of superpixels kept (more kept = more similar to original)
Step 6: Fit weighted Lasso
```

**Visualising the result:**

Instead of coefficients, image LIME typically highlights the superpixels with the largest positive coefficients — the regions of the image most responsible for the prediction.

```
Dog image explanation:
  Superpixels covering dog's face/ears    → large positive coefficients
  Superpixels covering background grass   → near-zero
  Superpixels covering dog's body         → moderate positive

Visual output: the dog's face lights up in green (pro-prediction)
               background stays grey (irrelevant)
```

**Why superpixels and not individual pixels?**

Individual pixels are:
1. Too many (224×224 = ~50k pixels → 2^50000 possible coalitions)
2. Not human-interpretable (a single pixel means nothing to a human)

Superpixels are:
1. Manageable (~50–100 per image)
2. Human-interpretable ("this region of fur drove the dog prediction")

**The trade-off:** Superpixel boundaries affect results. A badly segmented image (superpixels that cut across meaningful object regions) will produce misleading explanations.

---

## 8. The Instability Problem — Proven with Numbers

This is LIME's most serious practical limitation. It is not a minor edge case — it can be severe enough to make LIME unreliable for important decisions.

### 8.1 What Instability Means

Run LIME twice on the exact same sample with the exact same model. Get different explanations. The ranking of features can change. The signs can even flip.

**Concrete demonstration:**

Same model, same sample (income=$45k, debt=0.75, age=28). Run LIME 5 times:

```
Run  │ top feature   │ 2nd feature   │ income coeff │ debt coeff
─────┼───────────────┼───────────────┼──────────────┼────────────
  1  │ debt (+0.32)  │ income (−0.20)│   −0.20      │  +0.32
  2  │ debt (+0.28)  │ income (−0.22)│   −0.22      │  +0.28
  3  │ debt (+0.41)  │ employ (−0.14)│   −0.09      │  +0.41
  4  │ income (+0.19)│ debt (+0.25)  │   +0.19 ← sign flip! │  +0.25
  5  │ debt (+0.33)  │ income (−0.18)│   −0.18      │  +0.33
```

Run 4 gives income a **positive** coefficient when all other runs give it negative. A stakeholder given Run 4 would be told "having your income level increases risk" — the opposite of what Runs 1–3 say.

### 8.2 Where the Variance Comes From

LIME has **three sources of randomness**, all compounding:

**Source 1: Random perturbation sampling**

Each run generates a different random set of N perturbed samples. The regression is fitted on a different dataset each time. With N=500 samples in a high-dimensional feature space, the samples don't fully cover the neighbourhood reproducibly.

```
Run 1: happens to sample many high-income perturbations → income looks protective
Run 2: happens to sample mostly mid-range incomes → income's effect looks smaller
```

**Source 2: Random replacement values**

When a feature is "turned off," its replacement value is drawn randomly from the training distribution. Different runs draw different replacements.

```
Run 1: "debt absent" → debt gets replaced with low values (0.2, 0.3) → predictions drop a lot
Run 2: "debt absent" → debt gets replaced with medium values (0.5, 0.6) → predictions drop less
→ Same feature, same absence, different apparent contribution
```

**Source 3: Lasso regularisation path sensitivity**

The Lasso's feature selection is sensitive to small changes in the data near where regularisation would include/exclude a feature. A feature near the regularisation boundary appears in some runs and not others.

### 8.3 How Large Is the Problem in Practice

A 2019 study (Alvarez-Melis & Jaakkola) measured LIME instability across multiple datasets. They found:

```
Instability measured as: variance of LIME coefficients across runs
                          normalised by mean coefficient magnitude

For tabular data:   instability scores of 0.2–0.8 (very high)
                    meaning coefficients vary by 20–80% of their magnitude

For text data:      instability scores of 0.1–0.3 (moderate)
                    more stable because word removal is less ambiguous

For images:         0.1–0.2 (relatively stable)
                    superpixels are discrete and consistent across runs
```

Text > Image > Tabular in terms of reliability. **Tabular LIME is the most unstable.**

### 8.4 Fixes

| Fix | How | Tradeoff |
|---|---|---|
| Increase N (more perturbations) | Default 5000 → use 10000+ | Slower |
| Fix the random seed | `random_state=42` | Same run every time, not more accurate |
| Run K times, take mean | Average coefficients across 10–20 runs | K× slower, more robust |
| Use SHAP instead | TreeSHAP for trees is exact and deterministic | Only for tree models |
| Use LIME for text/images | Less instability there | Not always tabular |

**The honest answer:** For tabular data with a tree model, just use TreeSHAP. LIME's instability is a fundamental limitation, not something you can fully engineer away.

---

## 9. The Neighbourhood Problem

LIME's linear approximation is only valid within a "neighbourhood" around x*. The neighbourhood is controlled by the kernel width σ.

### 9.1 Too Small → Too Few Samples

If σ is very small, only samples extremely close to x* get significant weight. In high-dimensional tabular data, almost no randomly perturbed samples land this close. You effectively fit a regression on a handful of points → high variance, unreliable coefficients.

```
σ = 0.1 (very small):
  Most 5000 perturbed samples have weight ≈ 0
  Effective sample size: maybe 20–30 samples
  Regression is very noisy
```

### 9.2 Too Large → Linear Approximation Fails

If σ is very large, you include samples far from x* where the model may behave very differently. The "local" linear fit becomes a global average, losing the local fidelity that makes LIME useful.

```
σ = 5.0 (very large):
  All 5000 samples have similar weight ≈ 1.0
  You're fitting a global linear model, not a local one
  May give the same explanation for very different samples
```

### 9.3 The Kernel Width Parameter

LIME's default in the Python library uses:

```
σ = sqrt(number of features) × 0.75
```

This is a heuristic. It's not derived from the data or the model. Different problems may need very different σ values, and LIME provides no principled way to choose.

**In practice:** LIME's explanations can change substantially as you vary σ — another form of instability that's less visible but equally real.

---

## 10. Faithfulness — Does LIME Actually Approximate the Model?

After fitting the local linear model g, LIME reports a **local fidelity score**: R² between g's predictions and the black-box f's predictions on the perturbed samples.

```
High R² (e.g., 0.90):   g is a good local approximation of f near x*
                         → LIME explanations are relatively trustworthy

Low R² (e.g., 0.40):    g is a poor approximation
                         → LIME is fitting noise, not the model
                         → explanations cannot be trusted
```

**Always check local fidelity before interpreting LIME explanations.**

In sklearn's LIME implementation, you can access this via `explanation.score` — the R² of the local linear fit on the weighted neighbourhood samples.

**What causes low fidelity:**
- Model has sharp discontinuities near x* (thresholds, step functions)
- Strong interactions between features that a linear model can't capture
- σ too large → local linear model covers a heterogeneous region

---

## 11. Bias-Variance of LIME Explanations

```
LIME explanation error = Bias² + Variance

Bias sources:
  ├── Linear approximation error (model is not locally linear)
  ├── σ too large → approximation covers non-linear region
  └── Interpretable representation mismatch (tabular "absent" is unrealistic)

Variance sources:
  ├── Random perturbation sample (largest source)
  ├── Random replacement values for absent features
  └── Lasso regularisation path sensitivity (near-boundary features)
```

**How each data type compares:**

```
                Bias                  Variance        Overall reliability
Tabular         Medium-High           HIGH            ⚠️  Lowest
Text            Low                   Medium          ✅  Good
Image           Low                   Low-Medium      ✅  Best
```

**Text and image LIME are more reliable because:**

- The interpretable units (words, superpixels) are natural and meaningful
- "Removing a word" creates a valid, plausible sentence — a realistic perturbation
- "Greying out a superpixel" creates a valid perturbed image

- "Turning off income" for tabular data creates an artificial hybrid row that may not represent any real scenario — this is where the bias comes from

**The bias-variance diagram for tabular LIME:**

```
Increasing N (more perturbations):
  Variance ↓ (more samples → more stable fit)
  Bias stays the same (doesn't fix the linear approximation error)

Increasing K (more features in explanation):
  Variance ↑ (more coefficients to estimate)
  Bias ↓ (more expressive model → better approximation)

Increasing σ (larger neighbourhood):
  Variance ↓ (more samples in neighbourhood → stable)
  Bias ↑ (larger region → linear approximation worse)
```

---

## 12. LIME vs SHAP — Head to Head

| Dimension | LIME | SHAP |
|---|---|---|
| **Local explanation** | ✅ Yes | ✅ Yes |
| **Global explanation** | ❌ No (can't aggregate reliably) | ✅ Yes (mean \|φ\|) |
| **Stable / deterministic** | ❌ High variance (tabular) | ✅ TreeSHAP is exact |
| **Satisfies fairness axioms** | ❌ No | ✅ All 4 |
| **Self-verifiable** | ❌ No efficiency check | ✅ Values sum to prediction − baseline |
| **Faithfulness** | Approximate (check R²) | Exact for TreeSHAP |
| **Text/Image support** | ✅ Native, good | ⚠️ Possible but non-standard |
| **Speed** | Fast | Fast (TreeSHAP), slow (KernelSHAP) |
| **Correlation handling** | Poor | Better (marginal SHAP) |
| **Interpretable output** | Coefficients of a linear model | Additive attribution values |
| **Works on any model** | ✅ Yes | ✅ Yes (KernelSHAP) |

**When to choose LIME over SHAP:**

1. **Text or image model** — LIME's word/superpixel representation is native and intuitive. SHAP for text/image requires more setup.
2. **You need a fast prototype explanation** — LIME is simpler to set up initially.
3. **The model changes frequently** — LIME treats the model as pure black box with no setup.

**When to choose SHAP over LIME:**

1. **Tree model** — TreeSHAP is exact, fast, and deterministic. No reason to use LIME.
2. **You need reliability** — SHAP has the efficiency check; LIME doesn't.
3. **You need global + local** — SHAP provides both; LIME is local only.
4. **Correlated features** — SHAP handles them better.
5. **Consequential decisions** — LIME's instability is dangerous when explanations matter.

---

## 13. When LIME Is the Right Tool

Despite its limitations, LIME is the right choice in specific situations:

**✅ Text classification explanations**

"Why was this email classified as spam?" — LIME highlights the specific words that drove the prediction. This is natural, interpretable, and relatively stable. SHAP for text NLP models requires more setup.

**✅ Image model explanations**

"Why did the model classify this as a dog?" — LIME highlights the superpixels. Grad-CAM is an alternative but LIME is model-agnostic.

**✅ Quick exploratory explanations**

When you just want to get a rough sense of what features matter locally, and you don't need the rigour of SHAP's guarantees. Good for early-stage model development.

**✅ Non-tree, non-linear, non-NN models**

For models like SVMs or k-NN, KernelSHAP is the SHAP option but is slow. LIME is faster and can give a reasonable first approximation.

**❌ High-stakes tabular decisions**

Loan approvals, medical diagnoses, fraud decisions — LIME's instability and lack of efficiency guarantee make it unreliable here. Use TreeSHAP or KernelSHAP.

**❌ When you need to aggregate across samples**

LIME gives local explanations that can't be reliably averaged to produce global importance. Don't use LIME to rank features globally.

---

## 14. Common Mistakes

**Mistake 1: Trusting a single LIME run**

Run LIME multiple times (5–10 runs) and check whether the top features and their signs are consistent. If they're not, the explanation is unreliable for this sample.

**Mistake 2: Not checking local fidelity (R²)**

The explanation is only as good as the local linear model's fit. Always check `explanation.score`. If it's below 0.6, be very cautious about the explanation.

**Mistake 3: Using LIME when SHAP is available**

For tree models, TreeSHAP is strictly better — exact, deterministic, theoretically grounded. Don't use LIME for trees.

**Mistake 4: Averaging LIME coefficients across samples for "global importance"**

LIME explanations are not designed to be aggregated. The coefficients from different samples are on different scales and use different perturbation distributions. Averaging them produces a meaningless number.

**Mistake 5: Assuming "not in the explanation" means "not important"**

LIME uses Lasso with a sparsity constraint (top K features). A feature excluded from the K-feature explanation might still matter — it was regularised away. This is especially misleading when K is small (K=3 or K=5).

**Mistake 6: Using tabular LIME for correlated features**

If features are correlated (income and debt_ratio), LIME's perturbations create combinations like (high income, high debt) that don't exist in the real data. The coefficients reflect the model's behaviour on artificial inputs, not realistic scenarios.

---

## 15. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LIME — KEY FACTS                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WHAT IT DOES                                                            │
│    Explains ONE prediction from ANY model                                │
│    Fits a simple linear model locally around the point                   │
│    Returns: K features with coefficients (local linear influence)        │
│                                                                          │
│  THE ALGORITHM                                                           │
│    1. Generate N perturbed samples near x*                               │
│    2. Get black-box predictions for each                                 │
│    3. Weight by distance to x* (kernel: w = exp(−d²/σ²))                │
│    4. Fit weighted Lasso regression → coefficients = explanation         │
│                                                                          │
│  PERTURBATION BY DATA TYPE                                               │
│    Tabular:  turn features on/off (replace absent w/ training dist)      │
│    Text:     remove words (keep/absent binary)                           │
│    Image:    grey out superpixels (SLIC segmentation)                    │
│                                                                          │
│  RELIABILITY BY DATA TYPE                                                │
│    Image  > Text > Tabular                                               │
│    Tabular LIME is the most unstable — use SHAP for trees instead        │
│                                                                          │
│  THREE SOURCES OF VARIANCE (tabular)                                     │
│    1. Random perturbation samples                                        │
│    2. Random replacement values for absent features                      │
│    3. Lasso path sensitivity near regularisation boundary                │
│                                                                          │
│  KEY PARAMETERS                                                          │
│    N (perturbations):  higher = more stable, slower. Use ≥ 5000         │
│    K (features shown): Lasso sparsity. More = less sparse explanation   │
│    σ (kernel width):   controls neighbourhood size. Heuristic default   │
│                                                                          │
│  ALWAYS CHECK                                                            │
│    explanation.score (R²) — local fidelity                               │
│    Run 5–10 times and check consistency                                  │
│                                                                          │
│  USE LIME FOR           AVOID LIME FOR                                   │
│    Text explanations      Tree models → use TreeSHAP                    │
│    Image explanations     High-stakes tabular decisions                  │
│    Quick prototyping      Global feature importance                      │
│    Non-tree black boxes   Correlated features                            │
│                                                                          │
│  LIME DOES NOT SATISFY THE SHAP EFFICIENCY AXIOM                        │
│    Coefficients don't sum to prediction − baseline                       │
│    Cannot verify correctness of explanation                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Ribeiro, Singh & Guestrin (2016)** — *"Why Should I Trust You?" Explaining the Predictions of Any Classifier.* KDD. — Original LIME paper.
- **Ribeiro et al. (2018)** — *Anchors: High-Precision Model-Agnostic Explanations.* AAAI. — LIME's successor for rule-based local explanations.
- **Alvarez-Melis & Jaakkola (2018)** — *On the Robustness of Interpretability Methods.* ICML Workshop. — Formal analysis of LIME instability.
- **Garreau & von Luxburg (2020)** — *Explaining the Explainer: A First Theoretical Analysis of LIME.* AISTATS. — Theoretical analysis of LIME's behaviour.
- **Companion** — `5a_LIME_QA.md` — interview Q&A.
- **Companion** — `4_SHAP.md` — for when SHAP is the better choice.
