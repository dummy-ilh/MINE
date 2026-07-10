# Types of Feature Importance and Explainability — Complete Taxonomy

> The full map of every approach, how they relate, where each lives in the landscape, and the blurred lines between categories. Read this before diving into any individual method.

---

## Table of Contents

1. [Why a Taxonomy Matters](#1-why-a-taxonomy-matters)
2. [The Top-Level Split — Three Questions](#2-the-top-level-split--three-questions)
3. [Interpretability by Design (Intrinsic)](#3-interpretability-by-design-intrinsic)
   - 3.1 The Core Models
   - 3.2 Levels of Interpretability Within This Category
   - 3.3 The Rashomon Problem for Intrinsic Models
4. [Post-Hoc Interpretability](#4-post-hoc-interpretability)
   - 4.1 The SIPA Principle
   - 4.2 Model-Agnostic vs Model-Specific
5. [Feature Importance — Types and Subtypes](#5-feature-importance--types-and-subtypes)
   - 5.1 Model-Dependent Importance
   - 5.2 Model-Agnostic Importance
   - 5.3 Filter and Wrapper Methods
6. [Interpretation Scope — Global vs Local](#6-interpretation-scope--global-vs-local)
   - 6.1 Global Methods
   - 6.2 Local Methods
   - 6.3 Why You Can't Substitute One for the Other
7. [Types of Explainability](#7-types-of-explainability)
   - 7.1 Feature-Based
   - 7.2 Example-Based
   - 7.3 Visual
   - 7.4 Textual / Rule-Based
8. [Model-Agnostic vs Model-Specific — Deep Comparison](#8-model-agnostic-vs-model-specific--deep-comparison)
9. [The Full Method Map](#9-the-full-method-map)
10. [Blurred Lines — Where Categories Overlap](#10-blurred-lines--where-categories-overlap)
11. [How to Choose — Decision Guide](#11-how-to-choose--decision-guide)
12. [Interview Q&A](#12-interview-qa)
13. [Summary Card](#13-summary-card)

---

## 1. Why a Taxonomy Matters

When someone asks "how do I explain my model?", there are at least 15 distinct methods with real differences. Without a mental map:

- You might use a global method to explain an individual decision (wrong tool)
- You might apply a model-agnostic method when a faster model-specific one exists
- You might use a post-hoc method on an inherently interpretable model that doesn't need it
- You might confuse importance (a number) with explainability (a reasoning)

The taxonomy is the map. Every method in this folder has a specific location on it. Understanding the map means knowing when to reach for which tool.

---

## 2. The Top-Level Split — Three Questions

Every interpretability approach answers one or more of three questions:

```
Question 1 — WHEN do we interpret?
  Before/during training → Interpretability by design
  After training         → Post-hoc methods

Question 2 — WHAT do we interpret?
  The model overall           → Global methods
  A single prediction         → Local methods

Question 3 — WHO can we apply it to?
  Any model (black box)       → Model-agnostic
  One specific model type     → Model-specific
```

These three questions create the axes of the taxonomy.

---

## 3. Interpretability by Design (Intrinsic)

### 3.1 The Core Models

Interpretability by design means choosing a machine learning algorithm that produces a model humans can directly read and reason about — without any additional explanation layer.

```
Model                  Why It's Interpretable
────────────────────────────────────────────────────────────────────
Linear Regression      Each coefficient βⱼ = change in prediction
                        per unit increase in feature j, all else equal.
                        The model IS the explanation.

Logistic Regression    Coefficients map to log-odds. The sign tells
                        direction; magnitude tells strength.
                        Odds ratio = exp(βⱼ) is directly interpretable.

Decision Tree          Prediction = path from root to leaf.
(small, max_depth≤5)   Each split is a human-readable rule.
                        You can trace any prediction by eye.

Decision Rules         If-then rules extracted directly from data.
(rule lists)           Ordered or unordered sets of conditions.
                        Directly human-readable.

RuleFit                Sparse set of tree-based rules combined with
                        a Lasso linear model. Balances accuracy and
                        interpretability.

GAMs                   Prediction = f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)
(Generalised Additive  Each fⱼ is a 1D function, visualisable as a curve.
 Models)               No interactions by default → interpretable shape.
```

### 3.2 Levels of Interpretability Within This Category

Not all "interpretable" models are equally interpretable:

```
Level 1 — Entirely interpretable
  A decision tree with max_depth=3 (8 leaves)
  A linear model with 5 standardised features
  → Any person can trace a prediction and understand every decision

Level 2 — Parts interpretable
  A linear model with 200 features
  → The coefficients are interpretable in principle, but reading 200
    numbers is practically impossible
  → Global behaviour is interpretable; individual predictions less so

Level 3 — Predictions interpretable (with effort)
  Logistic regression with engineered features
  → Individual predictions can be traced, but require understanding
    what each feature means and how it was engineered
```

### 3.3 The Rashomon Problem for Intrinsic Models

Even for interpretable-by-design models, the Rashomon problem applies:

```
Scenario: fitting logistic regression to a credit dataset.
There are many sets of coefficients that achieve similar test AUC.

Model A: β(debt_ratio)=+1.8, β(income)=−0.9
Model B: β(debt_ratio)=+1.2, β(income)=−0.6, β(age)=+0.4

Both achieve AUC=0.88. Both are "interpretable."
But they give different explanations for why applicants are denied.

The interpretation depends on which of the equally-good models you choose
to fit — a choice often made by regularisation settings, not by truth.
```

This means: choosing an interpretable model doesn't escape the Rashomon problem. Interpretability by design gives you faithful explanations of the model. Whether the model is the "right" model among all equally-valid ones is a separate question.

---

## 4. Post-Hoc Interpretability

Post-hoc means after the model is trained. You apply an interpretation method to an already-fitted black box.

### 4.1 The SIPA Principle

All model-agnostic post-hoc methods follow the same underlying structure:

```
S — Sample:     Draw samples from the data (or generate perturbations)
I — Intervene:  Modify a feature or set of features
P — Predict:    Run the black-box model on modified inputs
A — Aggregate:  Summarise the resulting predictions into an explanation
```

**Examples of SIPA in action:**

```
Permutation Importance:
  S: use the test dataset
  I: shuffle column j
  P: get predictions → measure performance drop
  A: average drop across n_repeats = feature importance

PDP:
  S: use the full dataset
  I: set feature j = v for all rows
  P: predict on modified dataset
  A: average predictions at each v = partial dependence curve

LIME:
  S: generate N random perturbations near x*
  I: turn features on/off
  P: get black-box predictions for each
  A: fit weighted regression → local explanation

SHAP (KernelSHAP):
  S: sample coalitions
  I: replace absent features with background values
  P: predict for each coalition
  A: weighted regression → Shapley values
```

Understanding SIPA means you understand the family resemblance between all post-hoc methods — and why they all share similar failure modes around correlated features and distributional assumptions.

### 4.2 Model-Agnostic vs Model-Specific

```
Model-Agnostic:  applies SIPA externally — only uses f.predict()
                 Works on ANY model
                 Examples: Permutation, LIME, KernelSHAP, PDP, ALE

Model-Specific:  accesses model internals (weights, gradients, tree structure)
                 Only works for one model family, but is typically faster/exact
                 Examples: TreeSHAP, Gini MDI, gradient saliency maps
```

---

## 5. Feature Importance — Types and Subtypes

Feature importance answers: **"Which features matter most?"** — returning a scalar score per feature.

### 5.1 Model-Dependent Importance

These methods access model internals to compute importance.

```
Method                 Model Family       Mechanism
────────────────────────────────────────────────────────────────────────
Gini / MDI             Trees (RF, GBM)    Sum of impurity reductions per feature,
                                          weighted by n_t/N. Free from training.

Coefficient magnitude  Linear / Logistic  |βⱼ| after standardising features.
                                          Only valid if features are on same scale.

Attention weights      Transformers, NNs  Weight each token/feature by attention.
                                          Disputed as importance measure — attention
                                          ≠ attribution in general.

Gradient magnitude     Neural networks    |∂f/∂xⱼ| — how much output changes with
                                          a small nudge to feature j.

Split count            Trees              Number of times a feature is used for splits.
                                          Even more biased than MDI — avoid.
```

**Key weakness shared by all model-dependent importance:** they describe how the model uses features during training, not whether those features are actually informative for generalisation. An overfit model's MDI is a description of memorisation.

### 5.2 Model-Agnostic Importance

These work on any model by probing it externally.

```
Method                 Mechanism                              File
────────────────────────────────────────────────────────────────────────────
Permutation            Shuffle feature j → measure            3_Permutation_Importance.md
Importance             performance drop

Drop-Column            Remove feature j → retrain →           covered in 3_
                       measure performance drop

SHAP global            Mean |SHAP value| across all samples   4_SHAP.md
(mean |φ|)

LOFO                   Leave-One-Feature-Out:                 introduced here
(Leave One Feature Out) remove j, retrain, measure drop
                       Best for correlated features but costly

H-statistic            Measure of interaction strength         introduced here
                       between two features
```

**LOFO in detail:**

```
LOFO Importance(j) = Score(full model) − Score(model without j)

where "model without j" is RETRAINED from scratch (unlike drop-column
which just drops the column from a trained model)

Advantage: the retrained model optimally compensates for j's absence
           → correctly attributes importance even with correlations
Disadvantage: N+1 model trainings required (expensive)
```

**H-statistic (Friedman's):**

Measures how much of the variance in the model's predictions is explained by interactions between feature pairs, rather than their individual main effects:

```
H²(j,k) = Σ [PD(j,k) − PD(j) − PD(k)]² / Σ PD(j,k)²

≈ 0:   features j and k act independently
≈ 1:   all variance is due to j×k interaction
> 0.3: substantial interaction worth investigating
```

### 5.3 Filter and Wrapper Methods

These are feature selection methods (not post-hoc explanations), but they compute a form of importance:

```
Filter methods (pre-model, statistical):
  Pearson/Spearman correlation with target
  Mutual information
  Chi-squared test (categorical features)
  ANOVA F-statistic
  
  Properties: fast, computed before model training, ignore interactions,
              evaluate features independently of each other

Wrapper methods (model-based selection):
  Forward selection: greedily add features that improve performance
  Backward elimination: greedily remove features that don't hurt performance
  Recursive Feature Elimination (RFE): repeatedly retrain, remove weakest

  Properties: account for interactions, expensive (many model trainings),
              tend to overfit the selection to the training set
```

**When to use which:**

```
Filter:   Initial screening of 1000+ features to get to ~50
          Computationally cheap, no model needed
          Use: mutual information or correlation as first pass

Wrapper:  Final selection from a manageable feature set (~20–50)
          More accurate, captures interactions
          Use: RFE with cross-validation

Post-hoc: Explaining what the trained model uses
          Use: permutation importance, SHAP mean |φ|
          Note: these explain the model's behaviour, not the optimal feature set
```

---

## 6. Interpretation Scope — Global vs Local

### 6.1 Global Methods

Global methods summarise model behaviour **across the whole dataset**.

```
Method           What It Shows                              File
────────────────────────────────────────────────────────────────────────────────
PDP              Average effect of feature j on output      6_PDP_and_ALE.md
ALE              Marginal effect, corrected for correlation  6_PDP_and_ALE.md
ICE              Per-sample effect (N curves instead of 1)   12_ICE.md
H-statistic      Interaction strength between feature pairs  this file
Feature          Which features matter most                  3_, 4_SHAP.md
importance
SHAP summary     Distribution of SHAP values across dataset  4_SHAP.md
(beeswarm)
Surrogate model  Simple model mimicking the black box        7_Global_vs_Local.md
```

**What global methods can't do:**

A globally unimportant feature can be the decisive factor for a single prediction. A globally important feature may have zero contribution to a specific individual's outcome.

### 6.2 Local Methods

Local methods explain **a single prediction**.

```
Method           What It Shows                              File
────────────────────────────────────────────────────────────────────────────────
SHAP force/      Feature attributions for one prediction    4_SHAP.md
waterfall
LIME             Local linear approximation                  5_LIME.md
Anchors          If-then rule sufficient for this prediction 7_Global_vs_Local.md
Counterfactuals  Minimum change to flip the prediction       7_Global_vs_Local.md
ICE (one line)   This individual's prediction across         12_ICE.md
                 feature values
Ceteris Paribus  Effect of one feature for this individual   12_ICE.md
(CP plots)       (equivalent to one ICE curve)
```

### 6.3 Why You Can't Substitute One for the Other

```
Scenario: global permutation importance says income ranks 4th.
          Local SHAP for applicant #4521 shows income SHAP = +0.28
          (the single largest contributor to the denial)

Why: this specific applicant has an unusually low income
     relative to their high debt — the combination creates an
     outsized local effect even though income is globally moderate.

Rule: never use global importance to explain individual decisions.
      Never average local explanations to get global importance
      (except SHAP, which is designed so mean |φ| = global importance).
```

---

## 7. Types of Explainability

Explainability is broader than feature importance — it's about communicating model decisions to humans. Four distinct types:

### 7.1 Feature-Based Explainability

Explaining decisions by showing how much each feature contributed.

```
Output form:   a set of (feature, contribution) pairs
Examples:      SHAP waterfall, LIME coefficients, permutation importance bar chart
Best for:      technical audiences who understand features
Risk:          assumes features are the right unit of explanation
               (a person may not know what "debt_ratio" means)
```

### 7.2 Example-Based Explainability

Explaining decisions by reference to representative examples from the data.

```
Type            Description                              Example
────────────────────────────────────────────────────────────────────────
Counterfactuals "The closest case that got a             "If your debt were 0.55,
                different outcome"                        you'd be approved"

Prototypes      "The most representative examples        Show 3 typical 'approved'
                of each prediction class"                applicants for comparison

Criticisms      "Cases that the model handles           Flag unusual customers
                unusually or poorly"                    the model is uncertain about

k-NN in         "Your prediction is similar to          "Customers like you
embedding       these k training examples"              typically churn at 30%"
```

**Why example-based works for non-technical users:** Humans naturally reason by analogy. "Here's someone similar to you who was approved" is more intuitive than "your SHAP value for debt is +0.28."

### 7.3 Visual Explainability

Using plots and visual tools to communicate model behaviour.

```
Tool             What It Shows                  Best Audience
──────────────────────────────────────────────────────────────────
SHAP beeswarm    Feature importance + direction  Data scientists
PDP / ALE        Feature effect shape            Analysts, domain experts
ICE plots        Individual variation            Researchers
Saliency maps    Pixel importance for images     Computer vision teams
Force plots      Single prediction breakdown     Mixed audiences
Decision plot    Multiple predictions compared   Analysts
```

### 7.4 Textual / Rule-Based Explainability

Generating human-readable rules or natural language explanations.

```
Method          Output                              Properties
───────────────────────────────────────────────────────────────────────
Anchors         IF debt > 0.7 AND income < 45k     High precision guarantee
                THEN High Risk (93% reliable)       Naturally interpretable

Decision tree   IF age < 35 AND complaints ≥ 2     Globally applicable rules
(surrogate)     THEN Churn=True                    Fidelity may be limited

Rule lists      Ordered sequence of if-then rules  Deterministic, auditable
(e.g. RIPPER)   falling back to default

LLM-generated   "This customer was denied because  Requires careful grounding
summaries       their debt-to-income ratio..."      in actual model outputs
```

---

## 8. Model-Agnostic vs Model-Specific — Deep Comparison

```
Dimension           Model-Agnostic              Model-Specific
──────────────────────────────────────────────────────────────────────────────
What it requires    f.predict() only            Access to internals
                                                (weights, gradients, splits)

Model coverage      Any model                   One family only

Speed               Slower (many predict calls) Faster (uses internal computation)

Exactness           Approximate (usually)       Often exact (TreeSHAP, coefficients)

Correlated features Same limitations            Same limitations (TreeSHAP marginal)

Consistency         SHAP satisfies axioms       MDI violates efficiency axiom
                    Permutation does not        Attention ≠ attribution

Neural net support  LIME, KernelSHAP            Grad-CAM, DeepLIFT, IG, TCAV

Tree support        Permutation, LIME           TreeSHAP, MDI (all faster/exact)

Debugging value     External probing only       Can inspect exact learned patterns
                                                (useful for NN neuron analysis)

When to use         Black-box models,           When model family is known and
                    proprietary APIs,           internal methods are available
                    comparing models
```

**The hybrid approach:**

For tree models: use TreeSHAP (model-specific) for explanation + permutation importance (model-agnostic) for robustness check + grouped permutation (model-agnostic) for correlation handling. The two types complement each other.

---

## 9. The Full Method Map

```
                    WHEN
                    ────────────────────────────────────────────────
                    By Design           Post-Hoc
                    (Intrinsic)
                    ┌───────────────────┬──────────────────────────┐
        Any         │ Linear/Logistic    │   Model-Agnostic:        │
        model       │ Decision tree      │   Permutation, LOFO      │
WHO     (global)    │ GAMs, Rules        │   SHAP (global)          │
                    │ RuleFit            │   PDP, ALE, ICE          │
                    │                   │   H-statistic            │
                    │                   │   Surrogate              │
                    ├───────────────────┼──────────────────────────┤
        Any         │ (inherently        │   LIME, Anchors          │
        model       │  local: single     │   SHAP (local/force)     │
        (local)     │  tree path trace)  │   Counterfactuals        │
                    │                   │   Ceteris Paribus / ICE  │
                    ├───────────────────┼──────────────────────────┤
        Trees       │ (n/a — trees are   │   TreeSHAP               │
        only        │  also post-hoc     │   MDI / Gini             │
                    │  in RF/GBM)        │   MDA (OOB perm.)        │
                    ├───────────────────┼──────────────────────────┤
        NNs         │ ProtoViT           │   Grad-CAM, DeepLIFT     │
        only        │ (interpretable     │   Integrated Gradients   │
                    │  by architecture)  │   TCAV, Influential inst │
                    └───────────────────┴──────────────────────────┘
```

---

## 10. Blurred Lines — Where Categories Overlap

Real methods don't always fit neatly into one box. Understanding the blurred lines prevents categorical confusion.

**Blurred line 1: Logistic regression coefficients**

Are these intrinsic (model by design) or post-hoc (we're doing analysis after fitting)?

They are both. The model is intrinsically interpretable. Reading its coefficients is a post-hoc analysis step. The line blurs because for linear models, the model IS the explanation.

**Blurred line 2: Boosted stumps ≈ interpretable GAMs**

Gradient-boosted shallow trees (max_depth=1 or 2) with many estimators approximate an additive model — each tree captures a different feature's effect. The result is close to an interpretable GAM, even though GBMs are considered black boxes. The model-specific explanation (individual trees) gives you something interpretable by design.

**Blurred line 3: SHAP from a linear model**

For linear regression, SHAP gives φⱼ = βⱼ(xⱼ − E[Xⱼ]). This is a model-agnostic method (Shapley framework) applied to a model-specific quantity (the coefficient). The result happens to equal the exact model-specific interpretation. Model-agnostic and model-specific converge here.

**Blurred line 4: Global vs Local SHAP**

SHAP is a local method (per-sample values), but global importance (mean |φ|) is derived from local SHAP values by averaging. The beeswarm plot shows both simultaneously. SHAP is one of the few methods that natively spans the local-global divide.

**Blurred line 5: ICE vs PDP**

PDP is the average of ICE curves. ICE is a collection of local explanations (each curve = one sample). Together they span local (individual ICE line) and global (PDP average). The same computation gives both.

---

## 11. How to Choose — Decision Guide

```
Start here: What is your goal?
│
├── Understand model overall (audit, communication, selection)
│   └── Global methods
│       ├── Which features matter most?     → Permutation importance (test set)
│       │                                      or SHAP mean |φ|
│       ├── How does feature j affect pred?  → ALE (correlated) or PDP (uncorrelated)
│       ├── Are there feature interactions?  → H-statistic or 2D ALE
│       └── What rules describe the model?  → Global surrogate (check fidelity R²)
│
├── Explain one specific prediction
│   └── Local methods
│       ├── Feature attribution        → SHAP waterfall (trees) or LIME (text/image)
│       ├── Actionable guidance        → Counterfactuals
│       └── Reliable rule              → Anchors
│
├── Select features
│   ├── Initial screening (1000+ features) → Filter (mutual info, correlation)
│   ├── Final selection (~20–50)            → LOFO or RFE
│   └── Understanding what model uses       → SHAP mean |φ| or permutation importance
│
└── Choose a model type
    ├── High stakes, need trust       → Interpretable by design (linear, tree, GAM)
    ├── Performance priority          → Black-box + post-hoc explanation
    └── Regulation requires auditability → Interpretable by design OR
                                           black-box + counterfactuals + anchors
```

---

## 12. Interview Q&A

**Q: What is the difference between interpretability by design and post-hoc interpretability?**

Interpretability by design means choosing a model that is inherently understandable — like a decision tree or linear regression — so that the model itself is the explanation. Post-hoc interpretability applies explanation methods after training a model that may be complex and opaque. The key trade-off: interpretable-by-design models may sacrifice predictive performance; post-hoc methods explain whatever model you trained but the explanation is not the model itself and may be imperfect.

---

**Q: What is the SIPA principle and which methods follow it?**

SIPA stands for Sample-Intervene-Predict-Aggregate. Nearly all model-agnostic post-hoc methods follow this pattern: sample from the data, apply some intervention (shuffle a feature, turn features off, perturb an image), run the black-box model to get predictions, then aggregate those predictions into an explanation. Permutation importance, LIME, KernelSHAP, PDP, ALE, and H-statistic all follow SIPA. Understanding this makes it clear why they all share similar failure modes (correlated features, distributional assumptions).

---

**Q: What is the difference between feature importance and explainability?**

Feature importance produces a scalar score per feature answering "which features matter most?" It's a ranking or magnitude measure. Explainability is broader — it answers "why did the model make this specific decision?" returning an attribution, rule, or contrastive comparison. Feature importance is always global (a property of the model overall); explainability can be local (per-prediction) or global. SHAP spans both: local attributions (explainability) that can be aggregated to global importance.

---

**Q: Name three model-agnostic and three model-specific post-hoc methods. What is the key trade-off between these categories?**

Model-agnostic: permutation importance, LIME, KernelSHAP (or PDP/ALE). Model-specific: TreeSHAP, MDI/Gini importance, gradient saliency maps (Grad-CAM).

Key trade-off: model-agnostic methods work on any model (maximum flexibility) but only see inputs and outputs — they can be slower and less exact. Model-specific methods access internal structure, making them faster and often exact, but they are tied to one model family. For tree models, TreeSHAP strictly dominates KernelSHAP in speed and exactness — always use it.

---

**Q: What are the four levels of feature selection/importance methods from cheapest to most expensive?**

From cheapest to most expensive: (1) Filter methods — computed before training using statistical tests (e.g., mutual information); no model needed, ignores interactions. (2) Model-dependent importance — free by-product of training (e.g., MDI from a tree); biased but instant. (3) Permutation importance — requires N×K inference passes on trained model; moderate cost. (4) LOFO / Drop-column — requires retraining the model N+1 times; most expensive, most correct for correlated features.

---

## 13. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TYPES OF IMPORTANCE AND EXPLAINABILITY — TAXONOMY                       │
├──────────────────────────────────────────────────────────────────────────┤
│  TWO WHEN CATEGORIES                                                     │
│    By design:   model IS the explanation (linear, tree, GAM)             │
│    Post-hoc:    explanation applied to a trained model                   │
│                                                                          │
│  TWO SCOPE CATEGORIES                                                    │
│    Global:  average model behaviour across dataset                      │
│    Local:   why this specific prediction was made                       │
│    Rule:    never substitute global for local or vice versa             │
│                                                                          │
│  TWO ACCESS CATEGORIES                                                   │
│    Model-agnostic:   f.predict() only; any model; SIPA pattern          │
│    Model-specific:   accesses internals; one family; faster/exact       │
│                                                                          │
│  THE SIPA PATTERN (all model-agnostic post-hoc methods)                  │
│    Sample → Intervene → Predict → Aggregate                             │
│                                                                          │
│  FEATURE IMPORTANCE SUBTYPES                                             │
│    Model-dependent:  MDI, coefficients, attention, gradients            │
│    Model-agnostic:   Permutation, LOFO, SHAP mean|φ|, H-statistic       │
│    Filter/Wrapper:   pre-model selection (mutual info, RFE)             │
│                                                                          │
│  EXPLAINABILITY SUBTYPES                                                 │
│    Feature-based:   SHAP, LIME coefficients                             │
│    Example-based:   Counterfactuals, prototypes, k-NN                   │
│    Visual:          beeswarm, PDP, ALE, saliency maps                   │
│    Textual/rules:   Anchors, surrogate tree, rule lists                 │
│                                                                          │
│  BLURRED LINES                                                           │
│    SHAP spans local + global (mean|φ| is global importance)             │
│    ICE spans local (per-sample) + global (= PDP when averaged)          │
│    Linear model: intrinsic = post-hoc (model IS the explanation)        │
│    Boosted stumps ≈ GAM (black-box structure ≈ interpretable)           │
│                                                                          │
│  RASHOMON: multiple equally accurate models → different importances      │
│  This applies to intrinsic models too — interpretation is model-specific│
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## References

- **Molnar (2022)** — *Interpretable Machine Learning*, Chapter 4 — https://christophm.github.io/interpretable-ml-book — source of the SIPA principle and taxonomy structure.
- **Rudin (2019)** — *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions.* Nature Machine Intelligence. — Case for interpretable by design.
- **Caruana et al. (2015)** — *Intelligible Models for HealthCare.* KDD. — GAMs for interpretable ML.
- **Fisher, Rudin & Dominici (2019)** — *All Models are Wrong, but Many are Useful.* — Rashomon set.
- **Companion files:** All files in this folder — each method covered here in detail.
