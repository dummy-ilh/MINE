# Supervised vs Unsupervised Learning


---

## Table of Contents
1. [Formal Problem Setup](#1-formal-problem-setup)
2. [Objective Functions — The Real Divider](#2-objective-functions--the-real-divider)
3. [Identifiability](#3-identifiability)
4. [Evaluation](#4-evaluation)
5. [Bias–Variance Perspective](#5-biasvariance-perspective)
6. [Interpretation](#6-interpretation)
7. [Practical Usage](#7-practical-usage)
8. [Semi-Supervised Learning](#8-semi-supervised-learning)
9. [One-Line Mental Models](#9-one-line-mental-models)

---

## 1. Formal Problem Setup

### Supervised Learning

You observe **input–response pairs**:

$$\{(x_i, y_i)\}_{i=1}^n$$

The assumed data-generating process:

$$Y = f(X) + \varepsilon, \qquad \mathbb{E}[\varepsilon] = 0$$

**Goal:** Estimate $\hat{f} \approx f$.

This is a **well-posed** statistical estimation problem — there is a concrete target to aim at.

---

### Unsupervised Learning

You observe **inputs only** — no response variable:

$$\{x_i\}_{i=1}^n$$

No noise model for $Y$, because $Y$ does not exist.

**Goal:** Discover *structure* in $X$.

But "structure" must be defined by the algorithm designer, making this an **ill-posed problem without extra assumptions**.

---

## 2. Objective Functions — The Real Divider

This is the sharpest distinction between the two settings.

### Supervised — Clear Target, Clear Loss

$$\min_{\hat{f}} \; \mathbb{E}\bigl[L(Y,\, \hat{f}(X))\bigr]$$

Ground truth $Y$ exists, so every loss function has concrete meaning:

| Task | Loss |
|---|---|
| Regression | $L = (Y - \hat{f}(X))^2$ |
| Classification | $L = -\log P(Y \mid X)$ |
| Margin methods | Hinge loss |

---

### Unsupervised — No $Y$, No Canonical Loss

Without a response variable, we must choose **surrogate objectives** that encode our assumptions about what "structure" means:

| Algorithm | Surrogate Objective |
|---|---|
| k-means | Minimise within-cluster variance |
| PCA | Maximise variance explained |
| Density estimation | Maximise $\log p(X)$ |
| Topic modelling | Explain co-occurrence structure |

k-means explicitly:

$$\min_{\{C_k\}} \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

> **Key implication:** Different surrogate objectives yield different "truths." There is no single correct answer — only answers that are consistent with your chosen objective.

---

## 3. Identifiability

### Supervised

Most models are identifiable or nearly so. Even when multiple $\hat{f}$ fit the training data, prediction error on held-out data **anchors** the solution.

There is an external judge: *does it predict $Y$ well?*

### Unsupervised

Multiple explanations can fit equally well — different clusterings, latent factors, or manifolds — all optimising the same objective.

**There is no external notion of correctness.** The data cannot tell you which discovered structure is real.

---

## 4. Evaluation

### Supervised — Objective

$$\text{Test Error} = \mathbb{E}\bigl(Y - \hat{f}(X)\bigr)^2$$

Or accuracy, AUC, F1 — all computed against held-out ground truth. You can unambiguously say: *Model A beats Model B.*

### Unsupervised — Context-Dependent

There is no universal "best." Common proxies:

- **Internal metrics** — silhouette score, inertia
- **Stability** — consistency under resampling
- **Downstream performance** — does the structure help a later task?
- **Human interpretability** — do the outputs make sense?

> **Consequence:** Unsupervised results are *debated*, not verified. Two experts can justify opposite conclusions from the same clustering.

---

## 5. Bias–Variance Perspective

### Supervised

The classical decomposition applies directly:

$$\mathbb{E}(Y - \hat{Y})^2 = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

We explicitly manage overfitting, underfitting, and generalisation — all relative to a target.

### Unsupervised

Bias–variance is **implicit**:

| Term | What It Means Here |
|---|---|
| Bias | Structural assumptions (spherical clusters, linear manifolds, sparsity) |
| Variance | Sensitivity to which sample you drew |

But there is **no target error to decompose against** — so the framework is suggestive, not rigorous.

---

## 6. Interpretation

### Supervised Models

Interpretation answers: *How does $X$ affect $Y$?*

Coefficients, partial dependence plots, and feature importance all have direct meaning relative to the response.

### Unsupervised Models

Interpretation answers: *What regularities exist in $X$?*

But be careful:

| Output | What It Actually Is |
|---|---|
| Clusters | Regions of high density under your metric — **not** real classes |
| Principal components | Directions of variance — **not** causal factors |
| Topics | Co-occurrence patterns — **not** semantic truth |

These are **representations**, not explanations.

---

## 7. Practical Usage

### Supervised — Use When

- You know exactly what you care about predicting
- Labels are available and encode the objective cleanly
- The task is prediction or inference

**Examples:** credit default, disease diagnosis, demand forecasting

---

### Unsupervised — Use When

- You don't yet know the right question to ask
- You want to explore, compress, or organise
- Labels are expensive, scarce, or undefined

**Examples:** customer segmentation, feature learning, anomaly detection, pretraining representations

---

## 8. Semi-Supervised Learning

This section breaks the clean supervised / unsupervised dichotomy and addresses a setting that arises naturally in real systems.

### The Setup

You observe both:

$$\text{Labeled:} \quad \{(x_i, y_i)\}_{i=1}^{m} \qquad \text{Unlabeled:} \quad \{x_i\}_{i=m+1}^{n}$$

with $m \ll n$ — predictor measurements are **cheap**, response measurements are **expensive**.

### Why Not Just Drop the Unlabeled Data?

The unlabeled $x$'s carry information about:

- The geometry of the feature space
- Density structure of $P(X)$
- Natural clusters and manifolds

Ignoring them is often suboptimal.

### The Objective (Conceptually)

$$\min_{\hat{f}} \; \mathbb{E}[L(Y, \hat{f}(X))] \;+\; \lambda \cdot \mathcal{R}(\hat{f},\, P_X)$$

The regularisation term $\mathcal{R}$ encourages $\hat{f}$ to be consistent with the **distribution of $X$**, estimated using all $n$ points. Unlabeled data influences the model indirectly — it shapes the hypothesis space rather than directly supervising the loss.

### Core Assumptions

Semi-supervised learning is not magic. It relies on at least one of these:

**Cluster Assumption**
Points in the same high-density region share the same label. Decision boundaries should pass through low-density gaps.

**Manifold Assumption**
Data lie on a low-dimensional manifold embedded in high-dimensional space. Labels vary smoothly along the manifold.

**Low-Density Separation Assumption**
The optimal decision boundary should avoid regions of high density.

### When It Works

Medical imaging, speech recognition, NLP (self-training, language models), fraud detection — anywhere $X$ is abundant but $Y$ is expensive to obtain.

### When It Fails

> Semi-supervised learning can **hurt** performance (negative transfer) if:

- Unlabeled data distribution differs from labeled data
- The cluster or manifold assumption is violated
- Classes overlap heavily
- Label noise is high

---

### The Learning Spectrum

```
Unsupervised ────── Semi-Supervised ────── Supervised
  X only              few Y's               full Y
  structure         guides decision         optimise loss
  discovery          boundaries              directly
```

---

## 9. One-Line Mental Models

> **Supervised learning answers well-posed questions.**
> **Unsupervised learning proposes hypotheses about structure.**

That is why supervised learning dominates **deployment** and unsupervised learning dominates **exploration**.

---

> **Supervised learning is optimisation against reality.**
> **Unsupervised learning is optimisation against assumptions.**

---

> **Semi-supervised learning uses unlabeled data to shape the hypothesis space — not to directly define correctness.**
