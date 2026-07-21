# Logistic Regression — Part 1: Google · Apple · Meta
### Historic Interview Questions & Model Answers for L4–L6 / Senior–Staff

> **How to use this guide:** Each question is tagged with the company and the level it most commonly surfaces at. Answers are written the way a strong L5/L6 candidate would *speak* them — precise, grounded in math where it matters, and always connecting theory to production reality.

---

## Table of Contents
1. [Foundations & Mechanics](#1-foundations--mechanics)
2. [MLE, Loss, and Optimization](#2-mle-loss-and-optimization)
3. [Coefficients, Multicollinearity & Confidence Intervals](#3-coefficients-multicollinearity--confidence-intervals)
4. [Model Evaluation](#4-model-evaluation)
5. [Regularization & Overfitting](#5-regularization--overfitting)
6. [Class Imbalance](#6-class-imbalance)
7. [Assumptions & Diagnostics](#7-assumptions--diagnostics)
8. [Logistic Regression vs. Other Models](#8-logistic-regression-vs-other-models)
9. [Applied & System-Design Framing (Meta/Google)](#9-applied--system-design-framing-metagoogle)
10. [Apple-Specific: Propensity, Coefficient Interpretation, Causal Inference](#10-apple-specific-propensity-coefficient-interpretation-causal-inference)

---

## 1. Foundations & Mechanics

---

### Q1 — What is logistic regression, and why is it a regression model despite being used for classification?
**🏢 Google (L4/L5) | Apple (L4) | Meta (L4)**

**Answer:**

Logistic regression models the *log-odds* of a binary outcome as a linear function of the input features:

```
log( p / (1 - p) ) = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ
```

It is fundamentally a regression — it is regressing the log-odds (a continuous quantity) against the features. The "classification" step is a downstream decision: we apply a threshold (usually 0.5, but often tuned) to the *probability output* of the sigmoid function:

```
p = σ(z) = 1 / (1 + e^(-z))
```

The sigmoid maps any real-valued linear combination `z` into the range (0, 1), which we interpret as a probability. The name "logistic" comes from the logistic/sigmoid function.

**Why this matters at L5+:** A senior candidate recognizes that logistic regression outputs *calibrated probabilities*, not just labels. This distinction is critical in production — for example, in ad ranking at Meta or click-through rate (CTR) prediction at Google, the raw probability score is what the system consumes, not a binary label.

---

### Q2 — What is the sigmoid function and why is it specifically used in logistic regression?
**🏢 Google (L4) | Apple (L4) | Meta (L4)**

**Answer:**

The sigmoid function is:
```
σ(z) = 1 / (1 + e^(-z))
```

It's chosen for three concrete reasons:

1. **Probabilistic output:** It squashes any real number to (0, 1), making it interpretable as a probability.
2. **Derives naturally from the log-odds:** If you exponentiate both sides of `log(p/(1-p)) = z`, you recover the sigmoid. The sigmoid is not an arbitrary choice — it *is* the logistic distribution's CDF.
3. **Mathematically convenient for MLE:** The gradient of the log-likelihood with respect to weights has an elegant closed form: `(ŷ - y)x`, which is why gradient descent is efficient.

**L5/L6 nuance:** The sigmoid has a well-known flaw — it saturates at extreme values (z >> 0 or z << 0), causing vanishing gradients in deep networks. This is why ReLU replaced it in hidden layers of neural nets. But for the *output* layer of a binary classifier, sigmoid remains the right choice.

---

### Q3 — Walk me through what happens mathematically when logistic regression makes a prediction.
**🏢 Google (L5) | Meta (L5)**

**Answer:**

Given a feature vector **x** and learned weights **β**:

1. **Linear combination:** Compute `z = β₀ + β · x` (a scalar).
2. **Sigmoid activation:** Compute `p = 1 / (1 + e^(-z))` → a probability in (0, 1).
3. **Decision:** Apply threshold τ (default 0.5) → predict class 1 if `p ≥ τ`, class 0 otherwise.

The threshold τ is a tunable hyperparameter. Lowering τ increases recall (catches more positives) at the cost of precision. At senior levels, the interviewer expects you to mention that in production, you'd choose τ by optimizing the business metric on a held-out set using a precision-recall curve, not default to 0.5.

---

## 2. MLE, Loss, and Optimization

---

### Q4 — Derive the maximum likelihood estimator for logistic regression.
**🏢 Google (L5/L6) — asked verbatim in multiple reported Google DS loops**

**Answer:**

**Step 1 — Define the likelihood.**
For binary labels `yᵢ ∈ {0, 1}` and predicted probabilities `pᵢ = σ(β · xᵢ)`, the likelihood of the observed data under independent Bernoulli draws is:

```
L(β) = ∏ᵢ pᵢ^yᵢ · (1 - pᵢ)^(1 - yᵢ)
```

**Step 2 — Take the log-likelihood (more tractable).**

```
ℓ(β) = Σᵢ [ yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ) ]
```

This is the *binary cross-entropy loss* (negated). Minimizing cross-entropy = maximizing log-likelihood.

**Step 3 — Differentiate.**
The gradient w.r.t. βⱼ is:

```
∂ℓ/∂βⱼ = Σᵢ (yᵢ - pᵢ) · xᵢⱼ
```

This elegant result — the gradient is just the *residual* times the feature value — is why logistic regression is efficient to optimize.

**Step 4 — No closed form.**
Unlike linear regression, there is no closed-form solution for β. We maximize ℓ(β) iteratively, typically via gradient ascent or Newton-Raphson (IRLS — Iteratively Reweighted Least Squares).

**L6 follow-up:** Newton-Raphson converges quadratically (vs. gradient descent's linear convergence) but requires computing the Hessian, which is O(p²) in parameters. For high-dimensional problems (millions of features in ad systems), gradient descent with momentum or Adam is preferred.

---

### Q5 — Why does logistic regression use cross-entropy loss instead of mean squared error?
**🏢 Meta (L4/L5) | Google (L4)**

**Answer:**

There are two interconnected reasons:

**1. MSE produces non-convex loss surface for logistic regression.**
If you plug `p = σ(z)` into MSE `= (y - p)²`, the loss surface becomes non-convex with respect to `β`. This means gradient descent can get stuck in local minima. Cross-entropy, by contrast, produces a *convex* loss surface — there is a unique global minimum guaranteed.

**2. MSE gives vanishing gradients with sigmoid.**
When the model is confidently wrong (e.g., predicts p ≈ 0 when y = 1), the sigmoid is saturated. The MSE gradient `(y - p) · σ'(z)` is nearly zero because `σ'(z)` is tiny at saturation. Cross-entropy avoids this: its gradient is `(y - p)`, which is large when the model is confidently wrong — exactly when you *want* large gradient updates.

---

### Q6 — What optimization algorithm is typically used to train logistic regression, and what are the tradeoffs?
**🏢 Google (L5) | Apple (L5)**

**Answer:**

Common choices:

| Algorithm | Convergence | Memory | Best For |
|-----------|------------|--------|---------|
| Batch Gradient Descent | Stable, linear | O(n) per step | Small datasets |
| SGD | Noisy, can escape saddle points | O(1) per step | Large-scale |
| Mini-batch GD | Balanced | O(batch) | General use |
| L-BFGS / Newton | Quadratic (fast) | O(p²) Hessian | Low-dim, offline |
| IRLS | Exact for GLMs | O(p²) | Classical statistics |

**At FAANG scale (L5/L6 context):** For online learning systems (e.g., Meta's ad CTR model updated in near real-time), SGD variants with adaptive learning rates (Adam, Adagrad) are used because they handle sparse features efficiently — most users haven't clicked most ads, so gradients are sparse.

---

## 3. Coefficients, Multicollinearity & Confidence Intervals

---

### Q7 — If two predictors are highly correlated, what is the effect on the coefficients in logistic regression? What happens to their confidence intervals?
**🏢 Google (L4/L5/L6) — one of the most frequently reported Google DS questions**

**Answer:**

**Effect on coefficients:**
When two predictors `x₁` and `x₂` are highly correlated (multicollinearity), the model cannot distinguish their individual contributions to the outcome. The coefficients β₁ and β₂ become *unstable* — small perturbations in the training data cause large swings in their estimated values. The sum `β₁ + β₂` may be stable and meaningful, but the individual values are not.

Mathematically, this is because the design matrix X has near-linear dependence between columns, making the information matrix `Xᵀ X` (or the Hessian of the log-likelihood) near-singular, i.e., its determinant approaches 0.

**Effect on confidence intervals:**
Confidence intervals for β are derived from the *standard errors*, which are the square roots of the diagonal elements of the *inverse* of the Fisher information matrix (Hessian). When the information matrix is near-singular, its inverse has very large diagonal values → very large standard errors → very wide confidence intervals.

Concretely: the coefficients will have high variance (they could be very positive or very negative and both be consistent with the data), and the p-values will be inflated, making it look like neither predictor is significant even if together they are.

**Remedies:**
- Remove one of the correlated features
- Combine them (e.g., PCA, average)
- Apply L2 regularization (ridge), which adds a diagonal term to the Hessian, making it invertible and shrinking the coefficients toward zero
- Use domain knowledge to choose the more interpretable feature

**L6 extension:** In high-dimensional settings (e.g., thousands of ad features at Google), near-multicollinearity is pervasive. This is one of the core arguments for regularization being a default, not an option.

---

### Q8 — How do you interpret the coefficients of a logistic regression model?
**🏢 Apple (L4/L5) | Google (L4) | Meta (L4)**

**Answer:**

A coefficient βⱼ represents the change in the *log-odds* of the outcome for a one-unit increase in feature xⱼ, holding all other features constant.

To make this more interpretable, exponentiate:

```
Odds Ratio (OR) = e^βⱼ
```

- **OR > 1:** The feature is positively associated with the outcome (increases odds)
- **OR < 1:** The feature is negatively associated (decreases odds)
- **OR = 1:** No association

**Example:** If β = 0.693 for "email_opened", then OR = e^0.693 ≈ 2. Meaning: users who opened the email have *twice the odds* of converting compared to those who didn't, holding other features constant.

**L5/L6 nuance:** Log-odds interpretations are not always intuitive to stakeholders. For a senior candidate, it's worth noting:
- Coefficients assume *linearity in the log-odds*, which is an assumption worth testing for continuous features.
- Odds ratios ≠ relative risk (probability ratios). They coincide only when the baseline probability is low, which is why OR and RR diverge in high-prevalence outcomes.
- In production ML, coefficients also serve as a debugging tool: unexpectedly large or signed coefficients often signal data leakage, label bugs, or multicollinearity.

---

### Q9 — What happens to regression coefficients if you have omitted variable bias?
**🏢 Google (L5/L6)**

**Answer:**

Omitted variable bias occurs when a variable that is correlated with both a predictor *and* the outcome is excluded from the model. The included predictors then absorb part of the omitted variable's effect, making their coefficients biased and inconsistent.

**Direction of bias:**
If the omitted variable Z is positively correlated with both predictor X and outcome Y, then β_X is *upward biased* (overestimated). The sign and magnitude of the bias depend on the partial correlations.

**In logistic regression specifically:**
Omitted variable bias is more severe than in linear regression. Even if the omitted variable is *uncorrelated* with the included predictors, omitting it still attenuates coefficients in logistic regression (this is a property of the non-linear model — it's known as *incidental parameter bias*).

**Practical implication:** In causal inference work (propensity score models, uplift modeling), omitted variable bias is the central concern. At senior levels, you should be able to discuss how to partially address it: sensitivity analysis, instrumental variables, difference-in-differences, or, if the variable is unobserved, bounding the effect.

---

## 4. Model Evaluation

---

### Q10 — What does the ROC-AUC represent as an integral? How do you use it to evaluate logistic regression?
**🏢 Meta (L4/L5) | Google (L4)**

**Answer:**

**Geometric interpretation:** The ROC curve plots True Positive Rate (TPR/Recall) vs. False Positive Rate (FPR) as the classification threshold sweeps from 1 to 0. AUC-ROC is the area under this curve.

**Probabilistic interpretation (the cleaner L5 answer):**
AUC = P(score(positive example) > score(negative example))

That is: if you pick one random positive and one random negative from the dataset, AUC is the probability that the model assigns a higher score to the positive. An AUC of 0.5 is random; 1.0 is perfect.

**As an integral:**
```
AUC = ∫₀¹ TPR(FPR⁻¹(t)) dt
```
This equals the Mann-Whitney U statistic (a rank-based non-parametric test), which is why AUC is threshold-agnostic and robust to class imbalance.

**When AUC is not the right metric:**
- When class imbalance is extreme (e.g., 1:1000 fraud), AUC can be misleadingly high even for poor models. Prefer **PR-AUC** (precision-recall AUC) which focuses on model behavior in the positive class.
- When you care about calibration (predicted probabilities matching actual rates), AUC tells you nothing. Use **Brier score** or calibration plots instead.

---

### Q11 — How would you evaluate a logistic regression model beyond accuracy?
**🏢 Apple (L5) | Meta (L5) | Google (L5)**

**Answer:**

Accuracy is almost never the right metric. My evaluation framework:

**1. Discrimination:** How well does the model *rank* positives above negatives?
→ AUC-ROC, PR-AUC

**2. Calibration:** Are the predicted probabilities actually reliable?
→ Calibration plot (predicted vs. observed probability), Brier score, Hosmer-Lemeshow test
→ Crucial when downstream decisions depend on the raw probability (e.g., setting bid prices in ad auctions)

**3. Decision-threshold metrics:** At the chosen operating threshold:
→ Precision, Recall, F1, F-beta (weight recall more if false negatives are expensive)

**4. Business metrics on holdout / A/B test:**
→ Revenue impact, user retention, click rate — the model's ultimate job is to move these

**5. Failure modes:**
→ Confusion matrix slice analysis: where is it systematically wrong? (by cohort, time, feature values)
→ Are errors clustered in certain demographic groups? (fairness audit)

**L6 extension:** In production, I'd add *online* metrics — is the offline AUC predictive of online lift? A large gap signals train-serve skew.

---

## 5. Regularization & Overfitting

---

### Q12 — Explain L1 vs L2 regularization in the context of logistic regression. When would you use each?
**🏢 Google (L5) | Meta (L5) | Apple (L5)**

**Answer:**

Both add a penalty term to the loss to prevent overfitting:

**L2 (Ridge):** Penalty = λ · Σ βⱼ²
- Shrinks all coefficients toward zero but rarely to exactly zero
- Has a closed-form gradient, smooth optimization
- Preferred when most features are relevant (dense solution)
- Solves multicollinearity by adding λ to the diagonal of the Hessian

**L1 (Lasso):** Penalty = λ · Σ |βⱼ|
- Produces *sparse* solutions — drives some coefficients to exactly zero
- Performs implicit feature selection
- Non-differentiable at zero; requires subgradient methods or coordinate descent
- Preferred in high-dimensional settings with many irrelevant features

**Elastic Net:** Combines both — useful when you want sparsity but also handle correlated features (L1 tends to arbitrarily pick one from a correlated group).

**At L5+:** The choice also depends on interpretability needs and downstream usage. In an ad ranking model with millions of sparse features (Meta's GBDT + LR stacking), L1 is valuable for reducing inference cost — zero-weight features can be dropped at serving time.

**Bayesian view:** L2 regularization = Gaussian prior on coefficients; L1 regularization = Laplace prior. This matters when you want to reason about uncertainty in coefficients.

---

### Q13 — Why does L1 regularization produce sparse solutions while L2 does not?
**🏢 Google (L5/L6)**

**Answer:**

The geometric intuition: the constraint region for L1 is a *diamond* (hypercube rotated 45°) with corners on the axes. The loss function's level curves are ellipses. When you find the point of intersection (the constrained minimum), the corners of the diamond lie on coordinate axes — meaning one coefficient is exactly zero. This is geometrically likely.

For L2, the constraint region is a *sphere*. It has no corners. The intersection of the loss ellipse with the sphere occurs at a point where both coefficients are nonzero but small.

**Algebraically:** L1's subgradient at zero has a "kink" — the penalty function pulls coefficients exactly to zero when the loss gradient is insufficient to move them away from zero. L2's gradient at zero is zero, so it never creates the same pulling force toward sparsity.

---

## 6. Class Imbalance

---

### Q14 — How would you handle severe class imbalance when training a logistic regression model?
**🏢 Meta (L4/L5) | Google (L4/L5) | Apple (L5)**

**Answer:**

This is a common real-world problem — in fraud detection, churn, or ad click prediction, positive rates of 0.1%–5% are typical.

**Strategies, in order of preference:**

**1. Class weights (fastest, usually best starting point):**
Set `class_weight='balanced'` in scikit-learn. The loss for minority class examples is upweighted by `n_total / (n_classes · n_minority)`. This effectively re-weights the cross-entropy loss so the model pays more attention to rare positives.

**2. Threshold tuning:**
Default threshold of 0.5 is almost always wrong for imbalanced data. Use a precision-recall curve to select the threshold that hits your business target (e.g., recall ≥ 0.90 with maximum precision).

**3. Resampling:**
- *Oversampling (SMOTE):* Synthesize new minority-class examples by interpolating between existing ones. Useful when data is very scarce.
- *Undersampling:* Randomly drop majority-class examples. Simpler, loses information — acceptable at scale.
- Note: Resampling changes the calibration of probabilities. After resampling, re-calibrate with Platt scaling or isotonic regression.

**4. Change the metric:**
Stop optimizing accuracy. Use PR-AUC, F-beta, or a business-cost-weighted metric as your evaluation target.

**5. Collect more data** (often the most impactful, but hardest).

**Meta-specific context:** In ad CTR prediction with click rates ~0.5%, Meta historically used negative downsampling combined with a calibration correction (`q / (q + (1-q)/r)` where r is the sampling rate) to recover calibrated probabilities.

---

## 7. Assumptions & Diagnostics

---

### Q15 — What are the key assumptions of logistic regression?
**🏢 Apple (L5) | Google (L4)**

**Answer:**

1. **Binary (or ordinal/multinomial) outcome:** The dependent variable must be categorical. For continuous outcomes, use linear regression.

2. **Independence of observations:** Each data point should be independent. Repeated measures or clustered data (e.g., multiple purchases per user) violate this — use mixed-effects models or clustered standard errors.

3. **Linearity in the log-odds:** The relationship between each continuous feature and the log-odds of the outcome must be linear. This is testable (Box-Tidwell test, partial residual plots).

4. **No (or low) multicollinearity:** As discussed — highly correlated features destabilize coefficient estimation.

5. **Absence of complete separation:** If a feature perfectly separates the two classes, MLE does not converge (coefficients go to ±∞). Penalized regression (L1/L2) solves this.

6. **Large sample size:** Unlike OLS, logistic regression's MLE is asymptotically consistent. Small samples with many features lead to unreliable estimates (rule of thumb: ≥10 events per predictor).

**Important non-assumption:** Logistic regression does *not* require the features to be normally distributed (unlike LDA). It's a discriminative model.

---

### Q16 — How do you diagnose and handle complete separation in logistic regression?
**🏢 Google (L5/L6)**

**Answer:**

**Complete separation** occurs when a linear combination of features perfectly predicts the outcome in the training data. The log-likelihood has no maximum — as β → ±∞, the likelihood approaches 1 for all training examples, so gradient descent diverges.

**Symptoms:** Coefficients with extremely large magnitudes, huge standard errors, near-zero p-values or NaN outputs.

**Causes:** A feature that is a near-perfect proxy for the label (data leakage is a common culprit), or very small datasets.

**Solutions:**
- **Penalized regression (L1/L2):** The regularization term prevents coefficients from going to infinity. This is the standard production fix.
- **Firth's penalized likelihood:** Adds a Jeffreys-prior term to the log-likelihood, specifically designed for small sample/separation scenarios.
- **Drop or bucket the offending feature:** If a feature creates separation, examine whether it's a leak.
- **Bayesian logistic regression:** Priors on coefficients naturally prevent separation.

---

## 8. Logistic Regression vs. Other Models

---

### Q17 — Why might you choose GBM or neural networks over logistic regression? When would you stick with logistic regression?
**🏢 Google (L5) — reported as a direct question ("Why not logistic regression, why GBM?")**

**Answer:**

**When GBM/GBDT (e.g., XGBoost) is better:**
- The relationship between features and outcome is nonlinear or involves complex interactions — LR's linear decision boundary will underfit.
- Feature engineering effort is limited — tree models discover interactions automatically.
- Class separability is poor in the original feature space.
- You need robustness to outliers (trees are invariant to monotone feature transformations).

**When neural networks are better:**
- You have unstructured inputs (text, images, user embeddings) where representation learning matters.
- Scale: you have billions of training examples and can amortize the cost of deep architectures.
- You need end-to-end optimization of a pipeline (e.g., embedding lookup + interaction layers + output).

**When logistic regression wins:**
- **Interpretability requirement:** Regulators, legal, or trust-and-safety teams need to explain each prediction. LR coefficients are directly interpretable.
- **Latency constraint:** LR inference is a dot product — O(p). At < 1ms SLA, a deep model may not fit.
- **Baseline / sanity check:** Always train LR first. If your complex model doesn't beat LR significantly, you likely have a data problem, not a model problem.
- **Data is scarce:** LR is far less data-hungry than deep models.
- **Well-calibrated probabilities matter more than discrimination:** LR is often better calibrated out of the box.

**L6 synthesis:** The best answer acknowledges that in production FAANG systems, LR is rarely the final model, but is always the starting point. At Meta, the classic architecture is sparse feature LR → GBDT → deep & wide / DLRM, with each layer adding latency and complexity in exchange for AUC gains.

---

### Q18 — Differentiate between linear regression and logistic regression.
**🏢 Meta (L4) | Apple (L4)**

**Answer:**

| Dimension | Linear Regression | Logistic Regression |
|-----------|------------------|---------------------|
| Output | Continuous (ℝ) | Probability (0, 1) |
| Outcome variable | Continuous | Binary (or categorical) |
| Link function | Identity | Logit (log-odds) |
| Loss function | MSE (L2 loss) | Binary cross-entropy |
| Estimation | Closed form (OLS: β = (XᵀX)⁻¹Xᵀy) | Iterative (MLE via gradient descent) |
| Assumptions | Normality of residuals | None on feature distributions |
| Decision boundary | N/A | Linear in feature space |
| Interpretation | β = change in E[Y] per unit X | β = change in log-odds per unit X |

**Key insight for L5+:** Both are Generalized Linear Models (GLMs). Linear regression uses the identity link; logistic uses the logit link. Poisson regression (count outcomes), Cox regression (survival), and others fit the same family. Knowing this tells the interviewer you understand the unifying framework.

---

### Q19 — How does a random forest differ from logistic regression, and when would you use one over the other?
**🏢 Apple (L5) — reported question**

**Answer:**

**Decision boundary:**
- LR: Always linear in the original feature space.
- Random Forest: Non-linear; approximates arbitrary decision boundaries by ensembling many decision trees.

**Training mechanism:**
- LR: Gradient-based optimization of a global loss.
- RF: Bootstrap aggregation (bagging) of independently grown trees with random feature subsampling at each split.

**Interpretability:**
- LR: Globally interpretable via coefficients.
- RF: Black box globally, partially explained via feature importance (mean decrease in impurity) or SHAP values.

**Robustness:**
- RF: Robust to outliers, missing values (with imputation), and doesn't require feature scaling.
- LR: Sensitive to outliers, requires scaling for regularization to work correctly.

**When to use LR:**
- Interpretability is required, data is linearly separable, latency is critical, or you need calibrated probabilities.

**When to use RF:**
- Nonlinear interactions exist, feature engineering budget is low, interpretability is secondary, you need robustness without extensive preprocessing.

**Apple context:** Apple's on-device ML systems often prioritize model size and latency — LR or compact tree models are preferred over RF in inference-constrained environments.

---

## 9. Applied & System-Design Framing (Meta/Google)

---

### Q20 — How would you build a click-through rate (CTR) prediction model? Where does logistic regression fit?
**🏢 Meta (L5/L6) | Google (L5/L6)**

**Answer:**

CTR prediction is the foundational ML problem in digital advertising. The full pipeline:

**1. Label definition:**
Click = 1, No-click = 0. Label must be carefully defined: click-through, not just impression.

**2. Features:**
- User features: demographics, behavior history, device, time-of-day
- Ad features: creative type, advertiser, bid, historical CTR
- Context features: placement, page content, session context
- Interaction features: user × ad affinity signals

**3. Model evolution (the canonical Meta/Google trajectory):**
- **Stage 1 — Logistic Regression:** Fast to train, interpretable, handles billions of sparse features via online SGD. Google's original ad ranking system was a massive LR over hand-engineered features.
- **Stage 2 — GBDT + LR (Facebook 2014 paper):** Use GBDT to learn feature interactions and feed the leaf indices as sparse features into an LR. Captures non-linearities while keeping LR's efficient online update.
- **Stage 3 — Deep & Wide / DLRM:** Embeddings for sparse categorical features, deep network for interactions, wide LR for memorization of sparse feature crosses.

**4. Training at scale:**
- Negative downsampling (retain all positives, sample ~1% of negatives) for computational feasibility
- Recalibrate probabilities after downsampling: `p_corrected = p / (p + (1-p)/r)` where r = sampling rate
- Online learning: update model daily or hourly to capture freshness effects

**5. Evaluation:**
- AUC for offline ranking; NE (Normalized Entropy) is Meta's standard calibration metric
- A/B test for online lift in revenue per impression (RPI)

**Why LR still matters at L6:** Even in deep learning systems, the output layer is often a sigmoid applied to a linear combination of the final hidden layer — which is logistic regression. Understanding LR deeply is understanding the final decision mechanism of every binary classifier.

---

### Q21 — How would you build a model to detect fake accounts / bots? Where does logistic regression fit, and when would you move beyond it?
**🏢 Meta (L5/L6)**

**Answer:**

This is a security ML problem with heavy class imbalance and adversarial dynamics.

**Features (reported in Meta interview rounds):**
- Friend request patterns (high volume in short window)
- Friend network structure (low interconnection among friends — real people's friends tend to know each other)
- Profile completeness
- Content posting velocity and diversity
- Engagement asymmetry (high outbound, low inbound)
- IP/device fingerprinting: multiple accounts from same device
- Email domain reputation, phone number verification

**Model choice:**
- **Logistic Regression (baseline):** Interpretable, auditable by trust-and-safety policy teams, fast to iterate. Start here.
- **GBDT (XGBoost):** Captures non-linear patterns in behavioral features. Typically a large AUC jump over LR.
- **GNN (Graph Neural Network):** Models the social graph structure explicitly — can propagate fraud signals through the network (friends of fake accounts are suspicious).
- **Ensemble:** LR + GBDT + GNN outputs combined.

**Key challenges:**
- **Adversarial distribution shift:** Bad actors observe when their accounts are flagged and adapt. Model retraining cadence is critical.
- **Label quality:** Ground truth (confirmed fake accounts) lags real-time detection.
- **Precision vs. recall tradeoff:** False positives (banning real users) are costly reputationally. The threshold must be tuned to minimize FPR while catching most fakes.

---

## 10. Apple-Specific: Propensity, Coefficient Interpretation, Causal Inference

---

### Q22 — What is a propensity model and how are its beta estimates calculated by MLE?
**🏢 Apple (L5/L6) — reported verbatim**

**Answer:**

A **propensity model** estimates the probability that a unit (user, customer) receives a treatment (e.g., an email, a promotion, an intervention), given their observed covariates:

```
e(X) = P(T = 1 | X)
```

This is exactly logistic regression applied to the treatment assignment indicator as the outcome, with pre-treatment covariates as features.

**Why it matters:**
In observational studies (where treatment is not randomly assigned), propensity scores are used to control for confounding:
- **Propensity score matching:** Match treated and control units with similar propensity scores.
- **Inverse Probability Weighting (IPW):** Weight each unit by `1/e(X)` (for treated) or `1/(1-e(X))` (for control) to create a pseudo-population where treatment is independent of covariates.

**Beta estimation via MLE:**
Exactly as in standard logistic regression — maximize the log-likelihood:
```
ℓ(β) = Σᵢ [ Tᵢ log(eᵢ) + (1 - Tᵢ) log(1 - eᵢ) ]
```
where `eᵢ = σ(β · Xᵢ)` and `Tᵢ` is the treatment indicator.

**L6 depth:** Propensity models require careful covariate selection — include variables that predict both treatment *and* outcome. Including only treatment predictors creates no bias but increases variance; including only outcome predictors is harmful if they're also affected by treatment (post-treatment bias). The estimand (ATE, ATT, ATO) determines the weighting scheme.

---

### Q23 — How do you interpret logistic regression coefficients with respect to their influence on the response variable?
**🏢 Apple (L5) — reported verbatim**

**Answer:**

At three levels of depth:

**Level 1 — Log-odds (exact):**
β_j = the change in log-odds of the outcome per one-unit increase in x_j, holding others constant.

**Level 2 — Odds ratio (more communicable):**
e^{β_j} = the multiplicative factor by which the odds change per one-unit increase in x_j.
- e^{β_j} = 1.5 → odds increase by 50%
- e^{β_j} = 0.7 → odds decrease by 30%

**Level 3 — Marginal probability (most intuitive):**
The change in predicted probability depends on the current value of p:
```
∂p/∂xⱼ = βⱼ · p(1 - p)
```
This is maximized when p = 0.5 and approaches zero as p → 0 or p → 1. So the same coefficient has a bigger probability impact near the decision boundary than at extreme probabilities.

**Practical implication:** In Apple's health ML applications (e.g., predicting a health event), reporting the marginal probability effect at the average risk level is most meaningful for clinical interpretation.

---

### Q24 — How would you assess whether a logistic regression model is well-calibrated, and how would you fix poor calibration?
**🏢 Google (L5/L6) | Apple (L5) | Meta (L5)**

**Answer:**

**What calibration means:**
A model is calibrated if, among all predictions of p = 0.7, approximately 70% of those examples are actually positive. Formally: P(Y=1 | ŷ=p) = p for all p.

**Diagnosis:**
1. **Calibration plot (reliability diagram):** Bin predictions into deciles; plot mean predicted probability vs. actual positive rate per bin. A well-calibrated model follows the diagonal.
2. **Expected Calibration Error (ECE):** Weighted average of |predicted - actual| per bin.
3. **Brier Score:** MSE of predicted probabilities — penalizes both discrimination and calibration.
4. **Hosmer-Lemeshow test:** Chi-squared test comparing observed and expected frequencies in probability bins (less preferred — sensitive to sample size).

**Common calibration failures:**
- **Overconfident:** Predictions cluster near 0 and 1 (S-curve vs. diagonal)
- **Underconfident:** Predictions cluster near 0.5
- **Systematic bias:** Predictions off by a constant offset (indicates intercept issue, often from downsampling without correction)

**Fixes:**
- **Platt Scaling:** Fit a new logistic regression on the validation set with the original model's log-odds as the single feature. Simple and effective.
- **Isotonic Regression:** Non-parametric; more flexible. Requires more validation data.
- **Temperature scaling:** In neural networks, divide logits by a learned temperature T before applying softmax/sigmoid. Single-parameter fix.
- **Correct the intercept:** After downsampling, adjust `β₀ → β₀ - log(q/(1-q))` where q is the sampling rate.

---

### Q25 — How would you perform feature selection for logistic regression in a high-dimensional setting?
**🏢 Google (L5/L6) | Apple (L5)**

**Answer:**

Feature selection in high-dimensional logistic regression is a regularization and statistical problem. My approach:

**1. L1 regularization (Lasso):**
Fit logistic regression with L1 penalty; features with zero coefficients are effectively selected out. Tune λ via cross-validation. Limitation: in correlated feature groups, L1 arbitrarily picks one.

**2. Elastic Net:**
L1 + L2 combined — sparsity with better handling of correlated features.

**3. Univariate filtering (pre-screening):**
For each feature, compute AUC or log-likelihood ratio vs. the null model. Drop features below a threshold before fitting the full model. Fast but ignores interactions.

**4. Recursive Feature Elimination (RFE):**
Iteratively remove the least important feature, refit the model, repeat. Computationally expensive but thorough.

**5. Permutation importance:**
Permute each feature and measure loss in AUC. Features whose permutation causes large AUC drops are important.

**6. Variance Inflation Factor (VIF):**
Identify multicollinear features (VIF > 10 is a common threshold); remove or combine them before fitting.

**L6 system thinking:** At Google/Meta scale, feature selection also has an engineering dimension. Features that are expensive to compute at serving time (high-latency lookups, join-heavy queries) are candidates for removal even if statistically useful. The tradeoff is accuracy vs. serving cost, and the senior candidate quantifies both.

---

## Quick Reference: Question-to-Company Map

| # | Question Topic | Google | Apple | Meta |
|---|---------------|--------|-------|------|
| 1 | What is logistic regression | L4 | L4 | L4 |
| 2 | Sigmoid function | L4 | L4 | L4 |
| 3 | Prediction walkthrough | L5 | — | L5 |
| 4 | MLE derivation | **L5/L6** | — | — |
| 5 | Cross-entropy vs MSE | L4 | — | L4 |
| 6 | Optimization algorithms | L5 | L5 | — |
| 7 | Multicollinearity + CIs | **L4–L6** | — | — |
| 8 | Coefficient interpretation | L4 | **L5** | L4 |
| 9 | Omitted variable bias | L5/L6 | — | — |
| 10 | ROC-AUC as integral | L4 | — | **L4/L5** |
| 11 | Evaluation beyond accuracy | L5 | L5 | L5 |
| 12 | L1 vs L2 regularization | L5 | L5 | L5 |
| 13 | Why L1 gives sparsity | L5/L6 | — | — |
| 14 | Class imbalance | L4/L5 | L5 | **L4/L5** |
| 15 | Assumptions | L4 | L5 | — |
| 16 | Complete separation | L5/L6 | — | — |
| 17 | LR vs GBM/neural nets | **L5** | L5 | L5 |
| 18 | Linear vs logistic regression | L4 | L4 | L4 |
| 19 | LR vs Random Forest | — | **L5** | — |
| 20 | CTR prediction system design | L5/L6 | — | **L5/L6** |
| 21 | Fake account detection | — | — | **L5/L6** |
| 22 | Propensity model + MLE | — | **L5/L6** | — |
| 23 | Coefficient influence on response | — | **L5** | — |
| 24 | Calibration diagnosis & fixes | L5/L6 | L5 | L5 |
| 25 | Feature selection high-dim | L5/L6 | L5 | — |

---

## Key Formulas Cheat Sheet

```
Sigmoid:          σ(z) = 1 / (1 + e^(-z))

Log-odds:         log(p / (1-p)) = β · x

Log-likelihood:   ℓ(β) = Σ [y·log(p) + (1-y)·log(1-p)]

Gradient:         ∂ℓ/∂βⱼ = Σ (yᵢ - pᵢ) · xᵢⱼ

Odds ratio:       OR = e^β

Marginal effect:  ∂p/∂xⱼ = βⱼ · p(1-p)

Calibration fix:  β₀_corrected = β₀ - log(q/(1-q))   [after downsampling rate q]

AUC definition:   P(score(pos) > score(neg))
```

---

# Logistic Regression — Part 1: Historic Interview Questions & Model Answers
### Google · Apple · Meta — Calibrated for L4–L6 / Senior to Staff Level

---

> **How to use this guide:** Questions are grouped by company, then ordered from foundational → advanced → applied/system-level. Each answer is written at the bar interviewers actually hold for senior and staff candidates — not just definitions, but mathematical intuition, tradeoff awareness, and production context. Read actively: cover the answer and try it first.

---

## Table of Contents

1. [Google](#google)
2. [Apple](#apple)
3. [Meta (Facebook)](#meta-facebook)
4. [Cross-Company Synthesis](#cross-company-synthesis)

---

## Google

> Google L4–L6 interviewers probe whether you understand **why** an algorithm works, not just **what** it does. Expect to be pushed on derivations, edge cases, and how theory breaks down in practice.

---

### Q1 · Foundational
**Explain logistic regression. Why do we use the sigmoid function specifically?**

**Answer:**

Logistic regression is a discriminative probabilistic model for binary classification. The core idea: we model the **log-odds** (logit) of the positive class as a linear function of features:

```
log( P(y=1|x) / P(y=0|x) ) = β₀ + β₁x₁ + … + βₙxₙ
```

Solving for P(y=1|x) directly gives us the sigmoid:

```
P(y=1|x) = 1 / (1 + e^(−z))   where z = β·x
```

The sigmoid is the **natural inverse of the logit**. It isn't an arbitrary choice — it falls directly out of the exponential family. Specifically, if you assume the class-conditional distributions p(x|y) are Gaussian with equal covariance, logistic regression is exactly the resulting posterior (this connects it to LDA).

Properties that make sigmoid ideal:
- Maps any real number to (0, 1), giving valid probabilities
- Differentiable everywhere — enables gradient-based optimization
- Its derivative is self-referential: σ'(z) = σ(z)(1 − σ(z)), which simplifies backprop

*L6 depth:* The sigmoid can be derived from maximum entropy principles — it's the maximum-entropy distribution over {0,1} consistent with observed feature expectations. That's a deeper justification than "it squashes values."

---

### Q2 · Foundational
**Derive the maximum likelihood estimator for logistic regression.**

**Answer:**

Given n i.i.d. examples {(xᵢ, yᵢ)}, yᵢ ∈ {0,1}, the likelihood is:

```
L(β) = ∏ᵢ P(yᵢ|xᵢ; β)
      = ∏ᵢ σ(βᵀxᵢ)^yᵢ · (1 − σ(βᵀxᵢ))^(1−yᵢ)
```

Taking the log (log-likelihood):

```
ℓ(β) = Σᵢ [ yᵢ log σ(βᵀxᵢ) + (1−yᵢ) log(1 − σ(βᵀxᵢ)) ]
```

This is the **negative cross-entropy loss** (negated for minimization). Crucially:

- There is **no closed-form solution** (unlike linear regression with OLS) because the sigmoid makes the normal equations nonlinear
- The log-likelihood is **globally concave** in β, so gradient descent finds the global optimum — there are no local minima to escape

The gradient is elegantly:

```
∇ℓ(β) = Xᵀ(y − ŷ)
```

where ŷ is the vector of predicted probabilities. This means the gradient is just the **residuals projected back through the features** — the same form as linear regression, which is not a coincidence (GLM theory).

*L5+ insight:* The Hessian is H = −XᵀWX where W = diag(ŷᵢ(1−ŷᵢ)). This is always negative semi-definite → confirms global concavity. Newton's method (using H⁻¹) gives quadratic convergence but costs O(p³) per step; gradient descent is preferred at scale.

---

### Q3 · Google Historical (confirmed reported)
**If two predictors are highly correlated, what is the effect on the coefficients in logistic regression? What happens to the confidence intervals of those coefficients?**

**Answer:**

This is a **multicollinearity** problem. When predictors xⱼ and xₖ are highly correlated:

**Effect on coefficients:**
The coefficients βⱼ and βₖ individually become **unstable and unreliable**, even though their combined effect on predictions may be stable. Geometrically, the design matrix XᵀWX approaches singularity — its determinant approaches zero — making the inverse poorly conditioned. Small changes in data can cause large swings in βⱼ and βₖ in opposite directions (one shoots up, the other plunges) while the prediction P(y=1|x) barely changes.

**Effect on confidence intervals:**
The standard error of βⱼ is SE(βⱼ) = √[(XᵀWX)⁻¹ⱼⱼ]. As XᵀWX becomes near-singular, its inverse blows up → **confidence intervals widen dramatically**. A coefficient that "should" be 0.5 ± 0.1 might appear as 0.5 ± 8.3. This can make both correlated features appear statistically insignificant (p-value → 1) even if they jointly carry strong signal.

**How to detect:** Variance Inflation Factor (VIF). VIF > 5–10 warrants investigation.

**Mitigations:**
- Remove one of the correlated features
- PCA / dimensionality reduction before fitting
- L2 (Ridge) regularization — adding λI to XᵀWX makes the matrix well-conditioned by construction
- L1 regularization — Lasso will drive one correlated feature's coefficient to zero, implicitly selecting

*L6 note:* In production ads models at Google where features like "ad_click_rate_7d" and "ad_click_rate_14d" are often highly correlated, Ridge is the standard solution. You'd rarely rely on p-values anyway — model-level metrics (AUC, calibration) dominate.

---

### Q4 · Advanced
**Why do we use cross-entropy loss for logistic regression instead of mean squared error (MSE)?**

**Answer:**

Two reasons — one practical, one principled:

**Practical:** If you compose MSE with the sigmoid, the resulting loss function is **non-convex**. It has local minima and a very flat gradient when predictions are confidently wrong (e.g., ŷ ≈ 0 but y = 1 → σ'(z) ≈ 0 → gradient ≈ 0). The model stops learning in precisely the worst cases.

Cross-entropy avoids this. When ŷ is confidently wrong, the cross-entropy gradient is large: -log(ε) → ∞ as ε → 0. The model gets a strong signal to correct.

**Principled:** Cross-entropy **is** MLE for a Bernoulli-distributed output. Minimizing cross-entropy is identical to maximizing the log-likelihood, which is the statistically correct thing to do when your model is logistic regression. MSE corresponds to MLE under a Gaussian noise assumption — which is the right choice for linear regression, but wrong for binary outputs.

```
Cross-entropy = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

When y=1: loss = -log(ŷ) → ∞ as ŷ→0   ✓ (strong gradient)
When y=0: loss = -log(1-ŷ) → ∞ as ŷ→1  ✓ (strong gradient)
```

---

### Q5 · Advanced (Google-confirmed)
**A logistic regression model for spam detection performs well offline but degrades in production after a week. You suspect overfitting to short-lived tokens and distribution shift. What regularization and feature engineering changes do you make, and how do you validate them before redeploying?**

**Answer:**

Step 1 — **Diagnose before prescribing.** First verify the offline/online gap is real: check if the offline evaluation used a random train/test split (which leaks temporal patterns) rather than a time-based split. Redo evaluation with a time split that mimics production ordering.

Step 2 — **Address short-lived tokens (overfitting):**
- Switch to **L1 (Lasso) regularization** or increase existing L2 penalty. L1 will drive rare-token features to zero, effectively performing feature selection
- **Minimum frequency threshold:** only include token features seen ≥ k times in training (e.g., k=100). This alone cuts noise dramatically
- **Feature hashing** with a smaller vocabulary size forces generalization
- Consider **character n-grams** instead of word tokens — more robust to novel spam vocabulary

Step 3 — **Address distribution shift:**
- Implement **time-windowed retraining** (daily or weekly refresh)
- Use **sample weights** that upweight recent examples (e.g., exponential decay: wᵢ = e^(−λ·days_old))
- Add **time-based features** (day of week, hour) to help the model adapt its threshold
- Monitor **input feature distributions** with KS tests or PSI (Population Stability Index) to detect shift early

Step 4 — **Validate before redeployment:**
- Shadow mode: run new model in parallel, compare predicted probability distributions
- Check **calibration curves** (reliability diagrams) — not just AUC. A well-calibrated model should have P(y=1 | ŷ=0.7) ≈ 0.7
- Run a **champion/challenger A/B test** on 5% traffic with pre-specified success metrics (precision at high-recall threshold, false positive rate on known-good senders)

---

### Q6 · Advanced
**What is the difference between L1 and L2 regularization in logistic regression? Why does L1 produce sparse solutions?**

**Answer:**

Both add a penalty to the log-likelihood to prevent overfitting:

```
L1 (Lasso):  ℓ(β) − λ·Σ|βⱼ|
L2 (Ridge):  ℓ(β) − λ·Σβⱼ²
```

**L2 behavior:** Penalizes large coefficients quadratically. The gradient of the penalty at any βⱼ ≠ 0 is 2λβⱼ — it shrinks coefficients toward zero proportionally but **never exactly reaches zero**. All features stay in the model; they just get small weights. This is ideal when you believe all features contribute.

**L1 behavior:** Penalizes absolutely. The subgradient of |βⱼ| is ±λ (constant, regardless of βⱼ's magnitude). This creates a **fixed force pulling toward zero**. Once a coefficient's magnitude drops below a threshold where this force exceeds the data's gradient, it **gets pinned exactly at zero**. Sparse solutions emerge naturally.

**Geometric explanation:** In 2D, L2's constraint set is a smooth circle; L1's is a diamond with corners at the axes. Contours of the log-likelihood are ellipses. The optimum is where the ellipse first touches the constraint. The diamond has corners exactly on the axes — so solutions often land on a corner, giving βⱼ = 0.

**When to use which:**
- L1: high-dimensional feature spaces, when you want automatic feature selection (e.g., text classification with millions of tokens)
- L2: all features are meaningful, multicollinearity present
- Elastic Net: both (L1 + L2) — best of both worlds, useful in practice

*L6 production note:* At Google scale, L1 on text features is standard. Sparse weight vectors also reduce memory and serving latency since you only store/load non-zero weights.

---

### Q7 · Applied (L5–L6)
**Why would you choose GBM over logistic regression for a classification problem? Why not always just use logistic regression as a baseline?**

**Answer (mirrors a real Google interview case):**

You'd use GBM over logistic regression when:

1. **Non-linear feature interactions exist** — LR only models additive log-odds effects. Real-world signals like "clicks × time-of-day" or "user × item compatibility" require manual feature crosses in LR but are learned automatically in GBM

2. **High feature cardinality / mixed feature types** — GBM handles raw categoricals and numeric features natively. LR requires explicit encoding decisions

3. **Non-monotonic relationships** — e.g., conversion rate peaks at "moderate ad frequency" and drops at extremes. LR can't model this without binning

You still want LR as a baseline because:
- It's interpretable (coefficients = log-odds ratios)
- It trains in seconds even on millions of examples
- It serves in microseconds (dot product + sigmoid)
- When features are already well-engineered, LR often matches GBM within a few AUC points

*L6 framing:* At Google Ads, the production ranking stack often uses LR for the first retrieval/filtering stage (speed, interpretability, easy to debug with SHAP-like attribution) and GBM or deep nets for late-stage ranking. A 2014 Google paper described this two-stage architecture explicitly.

---

## Apple

> Apple interviews emphasize **clarity of explanation to non-technical audiences**, applied model building, and practical diagnostics. For senior roles (ICT4+), expect to design production ML pipelines and justify modeling choices.

---

### Q8 · Foundational (Apple-confirmed)
**Explain how you would fit data using logistic regression, so that someone outside data science could follow.**

**Answer:**

Imagine you're trying to predict whether a customer will return a product — yes or no. You have some information about them: how long they've owned the product, their rating, how many times they contacted support.

Here's the process:

**Step 1 — Prepare your data.** Clean missing values. Scale numeric features so no single one dominates (StandardScaler). Encode categoricals. Split into train/test.

**Step 2 — The model's "bet".** Logistic regression draws a line (or hyperplane in high dimensions) through your feature space. On one side: "probably will return." On the other: "probably won't." The line's position is defined by one coefficient per feature. A positive coefficient means "more of this feature → more likely to return."

**Step 3 — Convert to a probability.** The raw score from the line is fed through a sigmoid function, which squashes it to between 0 and 1. If the score is 0.8, we say "80% probability of returning."

**Step 4 — Training.** We adjust the line's coefficients so that our probability estimates match what actually happened as closely as possible. We use an algorithm called gradient descent — imagine rolling a ball downhill on an error landscape to find the lowest point.

**Step 5 — Evaluate.** On the held-out test set, we measure accuracy, precision (of our "will return" predictions, how many actually did?), recall (of everyone who returned, how many did we catch?), and AUC — a summary of our model's discrimination ability.

*For a technical follow-up:* "Fitting" = maximizing the log-likelihood via L-BFGS or gradient descent, which are equivalent to minimizing cross-entropy.

---

### Q9 · Apple-confirmed
**How can the coefficients in a logistic regression model be interpreted with respect to their influence on the response variable?**

**Answer:**

This is one of logistic regression's greatest strengths — interpretability.

**Coefficient β is the change in log-odds per unit increase in xⱼ, holding all else equal.**

More usefully: **e^β is the odds ratio.**

Example: Suppose you're predicting whether a user will subscribe to Apple One. You have a feature "number of Apple devices owned" with coefficient β = 0.4.

- e^0.4 ≈ 1.49
- This means each additional Apple device the user owns **multiplies the odds of subscribing by 1.49** — i.e., increases the odds by ~49%

Sign of β:
- β > 0: feature increases probability of the positive class
- β < 0: feature decreases it
- β ≈ 0: feature has little marginal effect (conditional on other features)

**What to watch out for:**
- Coefficients are **conditional** — they describe the effect of one feature given all others are held constant. In correlated features, individual coefficients can be misleading
- They describe **log-odds scale**, not probability scale. A unit increase in x never shifts probability by a constant amount — it depends on the baseline probability
- Large absolute β doesn't mean the feature is important in practice if the feature has very low variance (e.g., a feature that's always 0.001 barely moves predictions even with β = 10)

*For standardized features:* Scale features to mean 0, std 1 before fitting — then |βⱼ| is directly comparable across features as a measure of feature importance.

---

### Q10 · Advanced (Apple ICT4+)
**Your logistic regression model has high AUC but poor calibration. The business team is using the predicted probabilities to make budget allocation decisions. What's the problem and how do you fix it?**

**Answer:**

**The problem:** AUC measures ranking ability — whether the model separates positives from negatives. But it's invariant to monotonic transformations of the score. A model can have perfect AUC while predicting 0.9 for events that happen 10% of the time. If the business team treats ŷ = 0.7 as "70% probability," they're making decisions on wrong probabilities, which leads to systematic budget misallocation.

**Diagnosis:** Plot a **reliability diagram** (calibration curve). Bin predictions into deciles, plot mean predicted probability (x-axis) vs actual positive rate (y-axis). A perfectly calibrated model follows the diagonal. Common failures:
- **Overconfident**: predicted probs cluster near 0 and 1, actual rates are more moderate
- **Underconfident**: predicted probs cluster near 0.5, actual rates are more extreme
- **Systematic bias**: predicted probs are consistently too high or too low

Also report the **Brier score** = mean((ŷ - y)²) — sensitive to calibration, unlike AUC.

**Fixes:**

1. **Platt Scaling**: Fit a new logistic regression on top of the model's outputs (logistic(a·score + b)) using a held-out calibration set. Simple and usually effective.

2. **Isotonic Regression**: Non-parametric — learns a monotone step function mapping raw scores to calibrated probabilities. More flexible than Platt but needs more calibration data.

3. **Temperature Scaling** (neural net origin, but applicable): Divide the logit by a temperature T > 1 to soften predictions, T < 1 to sharpen. Find T by minimizing NLL on a calibration set.

*Production note:* At Apple Services (subscriptions, App Store recommendations), calibration matters for downstream decision-making in ways raw AUC doesn't capture. Always report calibration alongside discrimination metrics.

---

### Q11 · Applied (Apple ICT4+)
**You are building a model to predict whether a user will churn from Apple Music. You have 1 million users, 50 features, and a class imbalance of 5% churners. Walk through your end-to-end approach with logistic regression.**

**Answer:**

**1. Exploratory Data Analysis:**
Check feature distributions, missing rates, and class balance. 5% churn is moderate imbalance — not extreme, but needs handling.

**2. Feature Engineering:**
- Behavioral features: days since last play, sessions per week, skip rate, playlist additions
- Device context: iOS version, device age (proxy for loyalty)
- Engagement trend: 30-day vs 7-day session count (declining trend is a strong signal)
- Payment features: auto-renew status, number of previous cancellation attempts

**3. Handle Imbalance:**
Options (in order of preference for LR):
- **Class weights**: set `class_weight='balanced'` or manually `{0: 1, 1: 20}` — effectively up-weights the positive class in the loss
- **Threshold tuning**: don't assume 0.5 as the decision boundary; tune to maximize F1 or optimize at a business-specified precision/recall point
- SMOTE or oversampling (generally less effective for LR than tree methods)

**4. Regularization:**
With 50 features and 1M samples, regularization is less about overfitting and more about numerical stability and feature selection. Start with L2 (Ridge), tune λ via 5-fold cross-validation on log-loss.

**5. Evaluation Metrics:**
- **AUC-ROC**: overall discrimination
- **Precision-Recall curve**: more informative under imbalance
- **F1 at business threshold**: e.g., "flag top 10% of users by churn probability for retention campaign"
- **Calibration**: crucial if downstream decisions use raw probabilities

**6. Model Validation:**
Use a **temporal split**: train on months 1–9, validate on month 10, test on month 11. Avoids future leakage.

**7. Deployment:**
- Log-odds scores can be recomputed as a dot product — serving is O(p) per user, trivially scalable
- Set up weekly retraining as user behavior drifts (seasonality: users churn more after holiday gifted subscriptions expire in Jan/Feb)
- Monitor PSI (Population Stability Index) on key features; alert if PSI > 0.25

---

## Meta (Facebook)

> Meta (Facebook) interviews are highly applied — they expect you to connect statistical concepts to **real products** (News Feed, Ads, Marketplace). Questions often embed logistic regression into a larger system design context.

---

### Q12 · Foundational (Meta-confirmed)
**Differentiate between linear regression and logistic regression.**

**Answer:**

| Dimension | Linear Regression | Logistic Regression |
|---|---|---|
| **Output** | Any real number ∈ (−∞, +∞) | Probability ∈ (0, 1) |
| **Target variable** | Continuous | Binary (or multiclass via softmax) |
| **Link function** | Identity: μ = Xβ | Logit: log(p/1-p) = Xβ |
| **Loss function** | Mean Squared Error (MLE under Gaussian noise) | Cross-entropy (MLE under Bernoulli) |
| **Assumptions** | Linearity, normality, homoscedasticity of residuals | Linearity of log-odds, independence, no severe multicollinearity |
| **Closed-form solution** | Yes: β = (XᵀX)⁻¹Xᵀy | No — requires iterative optimization |
| **Model type** | Regression | Classification (despite the name) |

**Key insight for interviews:** Both are GLMs (Generalized Linear Models) with different link functions and error distributions. Understanding them as two instances of the same framework — not two different algorithms — signals mathematical maturity.

---

### Q13 · Meta-confirmed
**What does the ROC AUC represent as an integral? What is its probabilistic interpretation?**

**Answer:**

The **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)** has a beautiful probabilistic interpretation:

> **AUC = P(score(positive) > score(negative))**

where a "score" is the model's predicted probability, and the pair (positive, negative) is drawn uniformly at random from all such pairs in the test set.

In other words: if you randomly pick one user who clicked an ad and one who didn't, AUC is the probability that your model gives the clicker a higher score.

- AUC = 1.0: perfect discrimination
- AUC = 0.5: random — no better than a coin flip
- AUC = 0.0: perfectly wrong (but monotonically invertible → still useful!)

**As an integral:** The ROC curve plots TPR (y) vs FPR (x) across all decision thresholds. AUC is literally ∫₀¹ TPR(FPR) d(FPR). Under the trapezoidal approximation, this is equivalent to the Mann-Whitney U statistic.

**Meta context:** For ads click-through prediction, Meta cares deeply about **relative ordering** of ads for ranking — AUC directly measures this. But for bid pricing and budget allocation, **calibrated probabilities** matter more (see Q10 above). A well-calibrated CTR model with AUC 0.73 is more valuable than a miscalibrated one with AUC 0.75.

---

### Q14 · Advanced (Meta-confirmed)
**If a model performed poorly after launching, what potential causes do you suspect happened in the model training step?**

**Answer (logistic regression framing):**

Systematically diagnose across these failure modes:

**1. Data leakage:**
A feature that "knows the future" was included in training but isn't available at serving time (e.g., "number of post-click events" used to predict "will user click"). The model learns a spurious signal, looks great offline, fails live.

**2. Label definition mismatch:**
Training labels and production labels are computed differently. Example: training used "clicked within 1 hour," but production monitors "clicked within 5 minutes."

**3. Train-serve skew:**
Feature preprocessing (normalization constants, imputation logic) computed on training data but applied inconsistently at serving. A feature scaled to [0,1] during training arrives as raw [0,1000] in production.

**4. Distribution shift:**
The world changed. User behavior on a new product launch is different from historical data. For logistic regression specifically, the class prior shift is especially dangerous — if the training set had 10% positives but production has 2%, the model's intercept is wrong and all probabilities are systematically over-inflated.

**5. Regularization miscalibration:**
Too little λ → overfit to training distribution; degrades on slightly shifted production data.

**6. Imbalanced treatment:**
If you oversampled the minority class during training without correcting for it at inference time, your predicted probabilities will be shifted upward by log((1-ρ)/ρ · ρ_s/(1-ρ_s)) where ρ is true prevalence and ρ_s is sampling prevalence. This is a known correction for logistic regression under prior probability shift.

---

### Q15 · Meta Applied (E5–E6)
**You're designing the first-stage retrieval model for Meta's ad auction. You have billions of users and millions of ads. Why might you choose logistic regression for this stage, and what are the limits that require a more complex model downstream?**

**Answer:**

**Why logistic regression at the retrieval stage:**

*Speed:* A logistic regression prediction is a dot product + sigmoid. At Meta scale (millions of ad auction calls per second), latency is measured in microseconds. LR can serve in ~1μs; a deep neural network might take 10–100ms.

*Scalability:* LR with L1 regularization on sparse features (user interests, ad categories) produces sparse weight vectors. These can be stored in memory and computed trivially.

*Stability:* LR has a convex loss — guaranteed to converge, no hyperparameter sensitivity around learning rate schedules that could cause instability in production.

*Interpretability:* Engineers can directly inspect which features are driving predictions. Critical for debugging in a regulated (ads) environment.

**What LR can't do (why you need a downstream model):**

1. **Feature interactions:** CTR depends on (user × ad) compatibility — a user interested in running shoes interacting with a Nike ad. LR requires these crosses to be explicitly engineered; it can't discover them. Deep networks or GBDT learn them automatically.

2. **Non-monotonic effects:** Ad fatigue — CTR increases for a few exposures then drops. LR can't model this without manual feature engineering (binning impression count).

3. **Raw inputs:** LR can't process images (ad creative), text (ad copy), or user history sequences. DNNs with embeddings can.

**Meta's actual architecture (publicly documented):** A cascade — LR for coarse filtering (top-K candidates from billions), then a GBDT for re-ranking, then a DNN with embeddings for final scoring. Each stage trades accuracy for speed.

---

### Q16 · Meta Advanced
**How do you handle class imbalance in logistic regression for a fraud detection problem at scale?**

**Answer:**

Fraud is classically imbalanced (often 0.1–1% positive rate). Three axes of intervention:

**1. Loss function / training:**
- **Class weights:** Weight positive examples by `1/fraud_rate` in the cross-entropy loss. In scikit-learn: `class_weight={0: 1, 1: 100}`. Mathematically equivalent to oversampling positives.
- **Focal Loss:** A variant of cross-entropy that down-weights easy negatives: FL = −α(1−ŷ)ᵞ log(ŷ). γ > 0 focuses learning on hard, misclassified examples. Effective in extremely imbalanced settings.

**2. Sampling:**
- **Undersampling negatives:** Randomly drop majority class examples. Fast, loses information. For LR, if you undersample to 50/50, you must **correct the intercept** at inference: adjust β₀ by −log((1−ρ)/ρ) + log((1−q)/q) where ρ = true prevalence, q = sampled prevalence.
- **SMOTE:** Less natural for tabular data in LR contexts; generates synthetic examples in feature space that may not correspond to real fraud patterns.

**3. Evaluation:**
Never report accuracy on imbalanced data (99% accuracy by predicting all-negative). Use:
- **Precision-Recall AUC** (more sensitive to minority class performance)
- **F1 at operating threshold** — choose threshold by business constraint (e.g., "flag at most 2% of transactions")
- **False Positive Rate @ fixed recall** — the cost of annoying legitimate customers

**4. Calibration correction (critical for Meta at scale):**
If you train with oversampled positives, the model's raw probabilities are biased high. Correct with Platt scaling on a held-out set with the **true class distribution** (not the sampled one). This ensures downstream risk scoring is accurate.

---

### Q17 · Meta System Design (E6)
**Walk through building a real-time bidding CTR prediction system using logistic regression. What are the engineering challenges beyond model training?**

**Answer:**

**Model training pipeline:**
- Labels: click events joined with impression logs (with a conversion window, e.g., 1 hour post-impression)
- Features: user features (interests, demographics), ad features (category, format, historical CTR), context features (platform, time of day, placement)
- Training: mini-batch gradient descent on a rolling 30-day window; Hogwild!-style parallel updates for distributed LR
- Regularization: L1 on sparse features (thousands of ad categories → only hundreds matter), L2 on dense features (user embeddings)

**Serving challenges:**

1. **Feature freshness:** User behavioral features (sessions today, recent searches) must be updated in near-real-time. Stale features cause CTR miscalibration. Solution: a feature store with online and offline paths (Lambda architecture).

2. **Train-serve skew:** Features computed differently in batch training vs. online serving is the single biggest source of unexplained offline/online gaps. Solution: share feature computation code between training and serving (write features to a unified store used by both).

3. **Calibration drift:** As user behavior and ad inventory change seasonally, the model's absolute CTR estimates drift. Solution: lightweight recalibration (Platt scaling) that runs daily without full retraining.

4. **Latency budget:** The full auction pipeline at Meta has a latency SLA of ~10ms. LR scoring must take <1ms. Solution: pre-compute user features at impression time, cache ad features, dot product is trivially parallelizable.

5. **Model freshness:** Spam patterns and user behaviors shift. Use online learning (SGD with fresh data) or daily retraining pipelines; track AUC on a rolling holdout to detect degradation.

---

## Cross-Company Synthesis

> These are the meta-level questions interviewers at all three companies ask to separate L5 from L6 candidates.

---

### Q18 · Senior / Staff Level
**When would you NOT use logistic regression? What signals in the data would push you to a different model?**

**Answer:**

Use logistic regression as your baseline — always. Then move away when:

| Signal | Problem | Alternative |
|---|---|---|
| Non-linear decision boundary even after feature engineering | LR underfits | GBM, Random Forest, DNN |
| Strong feature interactions not captured by manual crosses | LR misses patterns | GBM (learns trees = feature crosses) |
| Raw text, images, or sequences as input | LR needs fixed-dimensional features | DNN with embeddings |
| Many features with zero correlation to target (p >> n setting) | LR can still overfit | Lasso LR or sparse DNN |
| Need to model uncertainty well (calibration is hard to get right) | — | Bayesian Logistic Regression or Platt-scaled GBM |
| Class separability is perfect (features fully separate classes) | Coefficients diverge to ±∞; optimization doesn't converge | SVM (handles this by definition), or add regularization |

*Staff-level addition:* Perfect separation (complete separation or quasi-complete separation) is a known failure mode of LR. The MLE doesn't exist — the likelihood can be increased indefinitely by scaling β → ∞. Detection: check if gradient descent never converges or coefficients explode. Fix: Firth's penalized regression or strong regularization.

---

### Q19 · Math (L5–L6)
**Derive the gradient of logistic regression's log-likelihood with respect to β. What is elegant about the result?**

**Answer:**

Log-likelihood for a single example:

```
ℓ(β) = y·log(σ(z)) + (1−y)·log(1−σ(z))    where z = βᵀx
```

Using the chain rule and σ'(z) = σ(z)(1−σ(z)):

```
dℓ/dβ = [y · (1−σ(z)) − (1−y) · σ(z)] · x
       = [y − σ(z)] · x
       = (y − ŷ) · x
```

For the full dataset:

```
∇ℓ(β) = Xᵀ(y − ŷ)
```

**Why it's elegant:**
- The gradient is **the residual (y − ŷ) projected by features** — the same structural form as OLS in linear regression
- It's **model-agnostic in form**: the cross-entropy + sigmoid combination collapses to a clean residual, despite the complexity under the hood
- It suggests a natural update rule: increase βⱼ when the model under-predicts (y − ŷ > 0), decrease when it over-predicts
- For mini-batch SGD: average the gradient over a batch → unbiased estimator of the full gradient

This result also generalizes: in the GLM framework, the gradient always takes the form Xᵀ(y − μ) where μ is the mean prediction under the model's distribution.

---

### Q20 · Calibration (L5–L6 Universal)
**Your logistic regression predicts CTR. What does it mean for the model to be "calibrated," and why is it especially important at companies like Google, Apple, and Meta?**

**Answer:**

A model is **calibrated** if its predicted probabilities match empirical frequencies. Formally:

```
P(y=1 | ŷ = p) = p   for all p ∈ [0,1]
```

If the model predicts ŷ = 0.05 (5% click probability), then across all impressions where the model predicted 5%, approximately 5% should actually click.

**Why this matters at FAANG scale:**

*Google Ads:* Advertisers bid in a second-price auction. The bid is bid_price × predicted_CTR. If CTR is systematically over-inflated by 20%, Google over-charges advertisers relative to actual value delivered — a trust and legal issue.

*Apple App Store:* Predicted conversion probability drives which apps are shown in search. Miscalibrated probabilities distort monetization calculations.

*Meta Ads:* Expected value = CTR × bid × revenue_per_click. All downstream optimizations assume probabilities are true probabilities. A 10% miscalibration at Meta's scale translates to billions in misallocated ad spend.

**How logistic regression can be miscalibrated even on training data:**
- Severe class imbalance (probabilities skew toward majority class)
- Excessive regularization (pushes coefficients — and therefore probabilities — toward 0.5)
- Underrepresented subgroups (calibration may hold on average but not for specific user segments)

**Correction:** Platt scaling (post-hoc logistic regression on held-out data) or isotonic regression. Always evaluate calibration on a **separate calibration set**, not the training or test set.

---

*End of Part 1 — Logistic Regression*

---

## Quick Reference: What Each Company Emphasizes

| Topic | Google | Apple | Meta |
|---|---|---|---|
| MLE / derivation | ✅ Heavy | Moderate | Moderate |
| Multicollinearity | ✅ Confirmed historical | Moderate | Moderate |
| Calibration | Advanced (L5+) | ✅ Product-critical | ✅ Ads-critical |
| L1/L2 regularization | ✅ Heavy | Moderate | ✅ Heavy |
| Class imbalance | Advanced | ✅ Applied problems | ✅ Heavy |
| System design w/ LR | L5+ | ICT4+ | ✅ E5–E6 |
| Interpretability of coefficients | Moderate | ✅ Confirmed | Moderate |
| LR vs GBM tradeoffs | ✅ Confirmed historical | Applied | ✅ Applied |
| AUC / ROC interpretation | ✅ Math depth | Standard | ✅ Confirmed |
| Production failures / shift | ✅ Confirmed | ICT4+ | ✅ Heavy |

---

# Regularisation, Assumptions & When Not to Use Logistic Regression

---

## 1. Regularisation — Full Treatment with Hand Calculations

### Why regularise?

Without regularisation, MLE finds weights that maximise likelihood on training data. Problems:

```
1. Overfitting:     model memorises noise, fails on new data
2. Perfect separation: MLE diverges (w → ∞), no finite solution
3. Multicollinearity: weights blow up in unpredictable directions
```

Regularisation adds a penalty term to the loss that discourages large weights.

---

### L2 Regularisation (Ridge)

**Loss function:**

```
L_ridge(w) = −ℓ(w) + λ · Σⱼ wⱼ²
           = cross-entropy + λ · ||w||²

Equivalently with C = 1/λ:
L_ridge(w) = −ℓ(w) + (1/C) · ||w||²
```

**Gradient:**

```
∇_w L_ridge = Xᵀ(p̂ − y) + 2λw

Update rule:
w ← w − α · [Xᵀ(p̂ − y) + 2λw]
  = w(1 − 2αλ) − α · Xᵀ(p̂ − y)

The (1 − 2αλ) factor shrinks w every step — "weight decay"
```

**Effect:** all weights shrink toward zero, none reach exactly zero. Correlated features share weight rather than one dominating arbitrarily.

---

### L1 Regularisation (Lasso)

**Loss function:**

```
L_lasso(w) = −ℓ(w) + λ · Σⱼ |wⱼ|
```

**Gradient (subgradient, since |w| not differentiable at 0):**

```
∂L_lasso/∂wⱼ = [∂(−ℓ)/∂wⱼ] + λ · sign(wⱼ)

where sign(wⱼ) = +1 if wⱼ > 0, −1 if wⱼ < 0, ∈ [−1,+1] if wⱼ = 0
```

**Soft-thresholding update (proximal gradient):**

```
w̃ⱼ = wⱼ − α · [∂(−ℓ)/∂wⱼ]   (gradient step ignoring penalty)

Then apply soft threshold:
wⱼ ← sign(w̃ⱼ) · max(|w̃ⱼ| − αλ, 0)
```

If |w̃ⱼ| < αλ, the weight is set to exactly zero → feature eliminated.

---

### Why L1 gives sparsity and L2 doesn't — the geometry

Both penalties constrain the weights to lie in a region:

```
L2 penalty: constraint region is a SPHERE  (smooth, no corners)
            ||w||² ≤ t

L1 penalty: constraint region is a DIAMOND (corners on axes)
            ||w||₁ ≤ t
```

The loss contours are ellipses. Where the ellipse first touches the constraint region is the solution.

```
L2 (sphere):   ellipse touches sphere on a smooth surface
               → solution has wⱼ ≈ small but rarely exactly 0

L1 (diamond):  ellipse very likely touches diamond at a CORNER
               corners sit ON the axes where some wⱼ = 0 exactly
               → solution is sparse
```

This is why L1 drives coefficients to exactly zero (feature selection) while L2 only shrinks them.

---

### Hand Calculation — Effect of Regularisation

**Setup:** one feature, intercept=0, one observation, y=1, x=2.

Log-likelihood:
```
ℓ(w) = log(σ(2w))
```

#### Unregularised MLE

Gradient: dℓ/dw = (y − p̂)·x = (1 − σ(2w))·2

Set gradient = 0: σ(2w) = 1 → w → +∞. No finite solution.

#### With L2 (λ = 0.5)

```
dL/dw = −(1 − σ(2w))·2 + 2·0.5·w
       = −2(1 − σ(2w)) + w

Set = 0:  w = 2(1 − σ(2w))

Solve by iteration starting at w=0:

Step 0: w = 0
  σ(0) = 0.5
  residual = 1 − 0.5 = 0.5
  w_new = 2 × 0.5 = 1.0

Step 1: w = 1.0
  σ(2) = 0.880
  residual = 1 − 0.880 = 0.120
  w_new = 2 × 0.120 = 0.240

  Wait — this doesn't look right. We also have the w on the left.
  Correct fixed point:  w = 2(1 − σ(2w))

  At w=0.8:   2(1−σ(1.6)) = 2(1−0.832) = 2×0.168 = 0.336  (too low)
  At w=0.3:   2(1−σ(0.6)) = 2(1−0.646) = 2×0.354 = 0.708  (too high)
  At w=0.5:   2(1−σ(1.0)) = 2(1−0.731) = 2×0.269 = 0.538  (close)
  At w=0.52:  2(1−σ(1.04)) = 2(1−0.739) = 2×0.261 = 0.522  ← converges near w ≈ 0.52
```

With L2, w converges to ~0.52 instead of ∞. Regularisation gives a finite, stable estimate.

#### With L1 (λ = 0.5)

Gradient of loss: −2(1−σ(2w)) + 0.5·sign(w)

At w=0.3:
```
  σ(0.6) = 0.646
  −2(1−0.646) + 0.5·(+1) = −0.708 + 0.5 = −0.208  (still negative → w should increase)

At w=0.1:
  σ(0.2) = 0.550
  −2(0.450) + 0.5 = −0.900 + 0.5 = −0.400 (negative → increase w)

At w=0.4:
  σ(0.8) = 0.690
  −2(0.310) + 0.5 = −0.620 + 0.5 = −0.120 (negative → increase w)

At w=0.6:
  σ(1.2) = 0.769
  −2(0.231) + 0.5 = −0.462 + 0.5 = +0.038 (positive → decrease w)

Root near w ≈ 0.57
```

L1 gives a slightly larger estimate here (0.57 vs 0.52 for L2) because L1 penalises small weights more aggressively than L2 near zero, but drives irrelevant weights to exactly zero.

---

### Full Gradient Descent Example — Two Features, Regularisation

**Data: 4 observations**

```
x₁:  [1,  2,  3,  4]
x₂:  [2,  1,  4,  2]
y:   [0,  0,  1,  1]

X = [[1, 2],   (no intercept for simplicity)
     [2, 1],
     [3, 4],
     [4, 2]]
```

**Init:** w = [0, 0], α = 0.1, λ = 0.5 (L2)

**Iteration 1:**

```
z = Xw = [0, 0, 0, 0]
p̂ = σ(z) = [0.5, 0.5, 0.5, 0.5]

residuals = y − p̂ = [−0.5, −0.5, 0.5, 0.5]

gradient of log-lik:
  ∂ℓ/∂w₁ = Σᵢ(yᵢ−p̂ᵢ)x₁ᵢ = (−0.5)(1)+(−0.5)(2)+(0.5)(3)+(0.5)(4) = −0.5−1+1.5+2 = 2.0
  ∂ℓ/∂w₂ = Σᵢ(yᵢ−p̂ᵢ)x₂ᵢ = (−0.5)(2)+(−0.5)(1)+(0.5)(4)+(0.5)(2) = −1−0.5+2+1 = 1.5

gradient of L2 penalty:  2λw = 2(0.5)[0,0] = [0, 0]

total gradient of loss = −[2.0, 1.5] + [0, 0] = [−2.0, −1.5]

w ← w − α·(−[2.0, 1.5]) = [0,0] + 0.1·[2.0, 1.5] = [0.20, 0.15]
```

**Iteration 2:**

```
z = Xw:
  z₁ = 1(0.20)+2(0.15) = 0.50
  z₂ = 2(0.20)+1(0.15) = 0.55
  z₃ = 3(0.20)+4(0.15) = 1.20
  z₄ = 4(0.20)+2(0.15) = 1.10

p̂ = σ(z):
  σ(0.50) = 0.622
  σ(0.55) = 0.634
  σ(1.20) = 0.769
  σ(1.10) = 0.750

residuals = y − p̂:
  [0−0.622, 0−0.634, 1−0.769, 1−0.750] = [−0.622, −0.634, 0.231, 0.250]

∂ℓ/∂w₁ = (−0.622)(1)+(−0.634)(2)+(0.231)(3)+(0.250)(4)
        = −0.622−1.268+0.693+1.000 = −0.197

∂ℓ/∂w₂ = (−0.622)(2)+(−0.634)(1)+(0.231)(4)+(0.250)(2)
        = −1.244−0.634+0.924+0.500 = −0.454

L2 penalty gradient: 2(0.5)[0.20, 0.15] = [0.20, 0.15]

total gradient of loss = −[−0.197, −0.454] + [0.20, 0.15]
                       = [0.197, 0.454] + [0.20, 0.15]
                       = [0.397, 0.604]

w ← [0.20, 0.15] − 0.1·[0.397, 0.604] = [0.160, 0.090]
```

Weights being pulled back by the L2 penalty — notice w₁ decreased from 0.20 to 0.16 partly due to the penalty term [0.20, 0.15] fighting the gradient.

---

### Elasticnet

Combines both penalties:

```
L_elastic(w) = −ℓ(w) + λ[α_mix·||w||₁ + (1−α_mix)·||w||²]

α_mix = l1_ratio in sklearn
α_mix = 0: pure L2
α_mix = 1: pure L1
α_mix = 0.5: equal mix
```

Gets sparsity (from L1) while handling correlated features better (from L2). L1 alone tends to arbitrarily pick one from a correlated group; elasticnet tends to keep both with shared weight.

---

### Choosing λ (or C) — cross-validation

```python
from sklearn.linear_model import LogisticRegressionCV

# Automatically finds best C via cross-validation
model = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1, 10, 100],  # C = 1/λ
    cv=5,
    penalty='l2',
    scoring='roc_auc'
).fit(X_train, y_train)

print(model.C_)  # best C found
```

---

## 2. Assumptions of Logistic Regression

Logistic regression is often presented as "assumption-free" compared to linear regression. It isn't. It has different assumptions.

---

### Assumption 1 — Binary (or categorical) outcome

The dependent variable must be binary for standard logistic regression. For ordered categories (low/medium/high), use ordinal logistic regression. For unordered multiple categories, use multinomial logistic regression.

**How to check:** inspect your target variable. Seems obvious but often violated when people discretise a continuous outcome arbitrarily.

---

### Assumption 2 — Linearity of log-odds in continuous predictors

The most important and most violated assumption. The log-odds must be a linear function of each continuous feature:

```
log(p/(1−p)) = w₀ + w₁x₁ + w₂x₂ + ...
```

If the true relationship is curved (e.g. U-shaped risk), the model silently fits the wrong shape and coefficients are meaningless.

**How to check:**

Method 1 — Box-Tidwell test: add interaction terms xⱼ·log(xⱼ) to the model. If they're significant, linearity fails for feature j.

```python
import numpy as np
import statsmodels.api as sm

# Add x*log(x) terms
X_bt = X.copy()
for col in continuous_cols:
    X_bt[col + '_logx'] = X[col] * np.log(X[col])

model_bt = sm.Logit(y, X_bt).fit()
# Test significance of _logx terms
```

Method 2 — Empirical log-odds plot: bin the feature into groups, compute observed log-odds per bin, plot against bin midpoint. Should be roughly linear.

```python
bins = pd.qcut(X['age'], q=10)
log_odds = y.groupby(bins).apply(
    lambda g: np.log(g.mean() / (1 - g.mean()))
)
# Plot log_odds vs bin midpoints — look for linearity
```

Method 3 — LOESS/GAM smoothing: fit a generalised additive model and compare to logistic regression. If GAM fits much better, linearity fails.

**How to fix:** add polynomial terms (x², x³), bin continuous variables, use interaction terms, or switch to a tree-based model.

---

### Assumption 3 — Independence of observations

Each observation must be independent. The model assumes:

```
P(y₁, y₂, ..., yₙ | X) = Π P(yᵢ | xᵢ)
```

Violated by: repeated measures on same subject, clustered data (students in classrooms), time series, spatial data.

**How to check:** think about your data collection process. Is there any grouping or longitudinal structure?

**How to fix:**
- Mixed-effects logistic regression (random effects for clusters)
- GEE (Generalised Estimating Equations) — estimates population-average effects with clustered data
- Robust standard errors (cluster-robust SEs)

---

### Assumption 4 — No perfect multicollinearity

If features are perfectly correlated, XᵀWX is singular — cannot be inverted to get (XᵀWX)⁻¹, so standard errors don't exist.

With near-perfect collinearity (VIF > 10): SEs are huge, coefficients flip signs with small data changes, model is numerically unstable.

**How to check:**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# VIF > 5-10: concerning. VIF > 10: serious problem.
```

**How to fix:** drop one of the correlated features, use PCA to create orthogonal components, L2 regularisation (stabilises estimates in collinear settings).

---

### Assumption 5 — No extreme outliers in continuous predictors

Logistic regression is sensitive to outliers in x (not just in y, since y is binary by definition). An extreme x value can have high leverage — dragging the fitted boundary.

**How to check:**

```python
# After fitting with statsmodels:
influence = model.get_influence()
leverage = influence.hat_matrix_diag
cook_d = influence.cooks_distance[0]

# High leverage: leverage > 2p/n
# Influential: Cook's D > 4/n or F(p, n-p) critical value
```

**How to fix:** investigate the outlier (error? genuine extreme case?). Winsorise, log-transform, or use robust logistic regression.

---

### Assumption 6 — Large enough sample size

MLE is asymptotic — guarantees only hold as n → ∞. With small samples: estimates are biased, standard errors underestimate uncertainty, p-values are inaccurate.

Rule of thumb:

```
EPV (Events Per Variable) ≥ 10

EPV = min(n₀, n₁) / p

where n₀, n₁ = count of each class
      p = number of predictors
```

Example: 100 events, 10 features → EPV = 10. Borderline acceptable.

With EPV < 10: use Firth's penalised likelihood (built-in bias correction), exact logistic regression, or reduce the number of predictors.

```python
# Firth's logistic regression
# pip install logistf (R) or use statsmodels with correction
```

---

### Assumption 7 — No perfect separation

If a feature or combination of features perfectly separates y=0 from y=1, MLE diverges. The model correctly "knows" the answer but cannot express this as a finite coefficient.

**Signs:** very large coefficients (|w| > 10), huge standard errors, warnings about convergence.

**How to check:**

```python
# Look for features that perfectly predict the outcome
for col in X.columns:
    crosstab = pd.crosstab(X[col], y)
    if 0 in crosstab.values or any(crosstab.sum(axis=1) == crosstab.max(axis=1)):
        print(f"Possible separation: {col}")
```

**How to fix:** Firth's penalised likelihood, L2 regularisation (most practical), collect more data.

---

### Assumption summary

| Assumption | How to check | How to fix |
|---|---|---|
| Binary outcome | Inspect target | Use appropriate GLM |
| Linearity in log-odds | Box-Tidwell, empirical log-odds plot | Polynomial terms, GAM |
| Independence | Data collection process | Mixed effects, GEE |
| No perfect multicollinearity | VIF | Drop features, PCA, L2 |
| No extreme outliers | Leverage, Cook's D | Investigate, winsorise |
| Adequate sample (EPV≥10) | Count events/variables | Firth's, reduce features |
| No perfect separation | Large coefficients/SEs | L2, Firth's |

---

## 3. When Not to Use Logistic Regression

### Case 1 — Outcome is not binary and can't be binarised meaningfully

If your outcome has a natural ordering (mild/moderate/severe), forcing it into binary (severe vs not-severe) throws away information. Use:
- **Ordinal logistic regression** (proportional odds model) for ordered categories
- **Multinomial logistic regression** for unordered categories

If your outcome is continuous, use linear regression or a GLM with the appropriate distribution.

### Case 2 — Decision boundary is highly nonlinear

Logistic regression can only draw hyperplanes in feature space. If the true boundary is curved, concentric, or discontinuous (like XOR), logistic regression will underfit regardless of regularisation.

**Signs:** train and validation accuracy are both low. Model doesn't improve with more data.

**Use instead:** random forests, gradient boosted trees (XGBoost, LightGBM), neural networks, SVMs with kernels. Or add explicit polynomial/interaction features if you know the shape.

### Case 3 — Very high-dimensional sparse features (text, genomics)

With p >> n (more features than observations), MLE is undefined — XᵀX is not invertible and you can always achieve zero training loss. Standard logistic regression fails.

**Use instead:**
- L1 logistic regression (sparse solution, selects relevant features)
- L2 logistic regression (regularises but doesn't select)
- Naive Bayes (works well for text, p >> n is natural)
- Linear SVM (equivalent to L2 logistic in large-margin sense but more efficient)

### Case 4 — Complex feature interactions you can't enumerate manually

Logistic regression with manual feature engineering can approximate nonlinear relationships, but only if you know which interactions matter. When interactions are high-order and unknown:

**Use instead:** gradient boosted trees automatically find interactions (splitting on x₁ then x₂ is a multiplicative effect). Neural networks learn arbitrary interactions.

### Case 5 — Missing data in features

Logistic regression has no native mechanism for missing values. Options:
- Impute (mean, median, KNN, MICE) before fitting — adds assumptions
- Include missingness indicator variables — sometimes informative but inflates feature count

Trees and XGBoost handle missing values natively via surrogate splits or learned missing-value directions. If your data has substantial missingness and you don't want to impute, use trees.

### Case 6 — Target classes are defined by a threshold on a continuous process

If y = 1 when some underlying continuous score exceeds a threshold, modelling the binary label directly loses information. You're throwing away how extreme the positive cases are.

**Use instead:**
- Tobit model (censored regression) — models the latent continuous variable
- Survival analysis (time-to-event) if the outcome is time-based
- Quantile regression if you care about different points of the distribution

### Case 7 — Sample size is very small with many predictors (EPV < 5)

With tiny data and many features, MLE is badly biased. Standard errors are too small, p-values are anti-conservative, and the model will overfit.

**Use instead:**
- Firth's penalised likelihood logistic regression
- Exact logistic regression (computationally expensive but exact inference)
- Bayesian logistic regression with informative priors
- Reduce the number of predictors before fitting (prescreen with univariate tests or domain knowledge)

### Case 8 — You need to generate synthetic data or model the joint distribution

Logistic regression is discriminative — it models P(y|x) directly. It cannot:
- Generate new realistic observations (x, y)
- Compute P(x) or P(x, y)
- Answer counterfactual questions about x itself

**Use instead:** Naive Bayes, Gaussian mixture models, VAEs, GANs — generative models that model the joint P(x, y).

### Case 9 — Outcome is a rate or proportion (not binary)

If yᵢ is a proportion (fraction of successes in nᵢ trials), standard logistic regression is wrong — it treats each observation as a single Bernoulli trial.

**Use instead:**
- Binomial logistic regression with weights (yi trials each, wᵢ = nᵢ)
- Beta regression if yᵢ is a continuous proportion in (0,1)

### Case 10 — Decision speed at inference matters more than interpretability

Logistic regression inference is fast (one dot product + sigmoid), but so are trees. If you need sub-millisecond inference on embedded hardware and interpretability isn't critical, the model choice is mostly about accuracy and a tree ensemble will likely win.

---

### Decision guide

```
Binary outcome?
  No  → GLM (Poisson, Gamma, ordinal logistic, multinomial)
  Yes ↓

Linear boundary sufficient?
  No  → XGBoost / Random Forest / Neural Net
  Yes ↓

p >> n?
  Yes → L1 logistic, Naive Bayes, Linear SVM
  No  ↓

EPV ≥ 10?
  No  → Firth's penalised logistic, reduce features
  Yes ↓

Need calibrated probabilities and interpretability?
  Yes → Logistic Regression  ✓
  No  → XGBoost / Random Forest (likely higher accuracy)
```

# Logistic Regression Interview Q&A — Part 1
### Google · Apple · Meta | L4–L6 Data Scientist

> **How to use this guide:** Questions are organized by company, then by theme. Answers are written at the L5 bar — technically precise, product-aware, and free of hand-waving. L4 candidates should master the core answer; L6 candidates should be able to extend every answer into a system-design or leadership discussion.

---

## 🟦 GOOGLE

---

### Q1. If two predictors are highly correlated, what happens to the coefficients in logistic regression?
*(Reported: Google DS onsite, L4–L5)*

**Answer:**

When two features are highly correlated — a condition called **multicollinearity** — the model can't reliably attribute the effect of the outcome to one feature vs. the other. The consequences are:

1. **Inflated variance on coefficients.** The estimates become unstable: small changes in training data cause large swings in coefficient values, even if predictions remain stable.
2. **Wide confidence intervals.** The standard errors of the affected coefficients blow up, making hypothesis tests on them unreliable (Wald tests are particularly distorted).
3. **Sign flip paradox.** You may see a coefficient flip sign relative to what domain knowledge predicts. This is a hallmark symptom of severe multicollinearity.

> The key distinction: **predictions may still be accurate**, but **interpretability collapses**. If you're using logistic regression primarily for a production scoring model (e.g., CTR prediction), multicollinearity is less urgent. If you need coefficients to be interpretable (e.g., informing a policy decision), it's critical to address.

**Detection & remediation:**
- Compute the **Variance Inflation Factor (VIF)**. VIF > 10 signals a problem; VIF > 5 warrants investigation.
- Drop one of the correlated features, or combine them (e.g., PCA).
- Apply **L2 (Ridge) regularization** — it distributes importance across correlated features rather than arbitrarily assigning it to one, stabilizing the coefficients.

**L6 extension:** At scale, multicollinearity manifests subtly. In a feature store with hundreds of user-behavior features, many are highly correlated (7-day vs. 30-day activity). Regularization is your default defense; periodic correlation audits and feature importance checks are part of a healthy ML hygiene process.

---

### Q2. Derive the Maximum Likelihood Estimator (MLE) for logistic regression.
*(Reported: Google DS onsite, L5–L6)*

**Answer:**

Logistic regression models the probability that label $y \in \{0, 1\}$ given features $\mathbf{x}$:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}$$

For $n$ i.i.d. observations, the **likelihood** is:

$$\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{n} \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{1 - y_i}$$

Taking the **log-likelihood** (standard practice — converts product to sum, avoids underflow):

$$\ell(\mathbf{w}) = \sum_{i=1}^{n} \left[ y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i) + (1 - y_i) \log(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right]$$

This is exactly **negative binary cross-entropy**. Maximizing $\ell$ is equivalent to minimizing cross-entropy loss.

Taking the gradient with respect to $\mathbf{w}$:

$$\nabla_{\mathbf{w}} \ell = \sum_{i=1}^{n} (y_i - \hat{y}_i) \mathbf{x}_i$$

where $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$.

There is **no closed-form solution** (unlike linear regression). We use iterative methods — gradient ascent, Newton-Raphson, or L-BFGS. The log-likelihood is **concave** in $\mathbf{w}$, so gradient methods converge to the global optimum.

> **Why this matters at Google:** Understanding MLE grounds every downstream conversation — why cross-entropy is the right loss, why calibration matters, and why regularization is a prior in the Bayesian interpretation.

---

### Q3. What happens to regression coefficients when you have omitted variable bias?
*(Reported: Google DS L5)*

**Answer:**

Omitted variable bias (OVB) occurs when a variable $Z$ that is (a) correlated with an included predictor $X$ and (b) causally affects the outcome $Y$ is left out of the model.

The coefficient on $X$ absorbs the effect of $Z$ proportional to their correlation. Formally:

$$\hat{\beta}_X^{\text{biased}} = \beta_X + \beta_Z \cdot \frac{\text{Cov}(X, Z)}{\text{Var}(X)}$$

**Implications:**
- The direction of the bias depends on the sign of $\beta_Z$ and the sign of the correlation between $X$ and $Z$.
- If $Z$ is positively correlated with $X$ and positively affects $Y$, $\hat{\beta}_X$ will be **upward-biased**.
- Adding $Z$ to the model reduces the coefficient on $X$.

**Practical relevance at Google:** Suppose you're modeling user retention. You include "sessions last week" but omit "user tenure." Tenure correlates with sessions and strongly predicts retention. The coefficient on sessions will be inflated, and any product decision based on that coefficient (e.g., "increasing sessions by 1 increases retention by X%") will be overstated. This is why Googlers are expected to think causally, not just predictively.

---

### Q4. How would you predict the likelihood of a customer churning using logistic regression?
*(Reported: Google DS phone screen, L4)*

**Answer:**

**Step 1 — Define the label precisely.**
Churn is time-bound. For example: "cancellation within 30 days after the current subscription period ends." Ambiguity here causes label leakage or noisy training signal. I avoid using "ever churned" as a label because it conflates short-term and long-term risk.

**Step 2 — Feature engineering.**
Key feature categories:
- *Recency/frequency/monetary:* days since last login, session count (7d, 30d), revenue generated.
- *Engagement depth:* features used, content consumed, support tickets filed.
- *Lifecycle signals:* tenure, plan type, historical churn in cohort.

Important: all features must be computed using data available **before** the prediction timestamp. This prevents target leakage.

**Step 3 — Handle class imbalance.**
Churn datasets are typically imbalanced (e.g., 5–10% churn rate). Options:
- Set `class_weight='balanced'` in scikit-learn, which adjusts the loss function.
- Downsample the majority class.
- Use **PR-AUC** (not ROC-AUC) as the primary evaluation metric, since PR-AUC is more sensitive to performance on the minority class.

**Step 4 — Model and evaluate.**
Train with L2 regularization. Evaluate on a time-based holdout (not random split — you want to simulate temporal generalization). Key metrics: PR-AUC, precision/recall at the operating threshold, calibration curve (Brier score).

**Step 5 — Calibration.**
A well-calibrated model means P(churn) = 0.3 actually implies 30% churn rate. Use Platt scaling or isotonic regression post-hoc if calibration is poor. At Google, predicted probabilities feed into downstream decision-making (intervention cost–benefit analysis), so calibration is non-negotiable.

---

### Q5. Why might a sigmoid function be beneficial for logistic regression? What are its limitations?
*(Reported: Google DS L4–L5)*

**Answer:**

**Why sigmoid works:**
The sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps any real-valued input to $(0, 1)$, which is exactly the range of a probability. This is mathematically required — we want $P(y=1 \mid \mathbf{x})$ to be a valid probability.

Additionally:
- Its derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ is elegant and computationally cheap.
- It naturally produces soft outputs (probabilities) rather than hard class boundaries.
- It's directly connected to the log-odds (logit) representation: $\log \frac{p}{1-p} = \mathbf{w}^T \mathbf{x}$, which has a clean linear interpretation.

**Limitations:**
1. **Vanishing gradients.** For large $|z|$, $\sigma'(z) \approx 0$. In deep networks, this kills gradient flow. (Less relevant for logistic regression itself, but worth knowing if you're comparing to neural networks.)
2. **Not symmetric around zero.** Unlike tanh, sigmoid outputs are always positive, which can slow convergence in gradient descent.
3. **Outputs never exactly 0 or 1.** This can be an issue if perfect separation exists in the data (complete separation problem — MLE diverges and coefficients blow up to infinity).

> **L6 nuance:** In production ad systems at Google, the raw sigmoid output from logistic regression is known to be **miscalibrated** when trained on downsampled negatives. A standard fix is post-hoc recalibration using the formula: $q = \frac{p}{p + (1-p)/w}$ where $w$ is the downsampling ratio. Knowing this signals production-level sophistication.

---

## 🍎 APPLE

---

### Q6. Explain how you would fit data using logistic regression. Walk me through the full process so that a non-technical stakeholder could follow.
*(Reported: Apple DS interview — verified by hiring managers)*

**Answer:**

I'll explain it in two layers — the intuition first, then the mechanics.

**The intuition:**
Logistic regression is a model that learns to answer "what is the probability that this observation belongs to a particular category?" Think of predicting whether a user will make a purchase: 1 = purchased, 0 = didn't. The model learns from historical examples and outputs a probability between 0 and 1 for each new user.

**The process:**

1. **Data preparation.** Logistic regression requires numeric inputs. Categorical features (e.g., device type: "iPhone", "iPad", "Mac") must be encoded numerically (one-hot encoding or ordinal encoding). Missing values must be imputed — mean/median for numeric features, mode for categorical.

2. **Feature scaling.** While not strictly required, standardizing features (zero mean, unit variance) speeds up convergence and ensures regularization treats features fairly.

3. **Train/test split.** Divide the data into training and holdout sets — typically 80/20. For time-series data (e.g., user behavior over time), split chronologically to avoid leakage.

4. **Model training.** The algorithm finds weights $\mathbf{w}$ that minimize binary cross-entropy loss. Under the hood, it's solving the MLE problem using gradient descent or L-BFGS.

5. **Evaluation.** On the holdout set, I assess:
   - **Accuracy** (for balanced classes)
   - **AUC-ROC** (discrimination ability across all thresholds)
   - **Precision/Recall** (for imbalanced classes like fraud or churn)
   - **Calibration** (do predicted probabilities match observed rates?)

6. **Threshold selection.** The default cutoff of 0.5 is rarely optimal. I choose a threshold based on the business cost tradeoff — e.g., at Apple, a false positive in fraud detection means blocking a real customer transaction, which has a high cost.

**To the non-technical stakeholder:** "The model reads thousands of past examples of users who bought and didn't buy, and learns which patterns — time of day, device, prior app engagement — best predict purchase. It then scores every new user with a probability, and we set a threshold (e.g., 0.6) above which we trigger a targeted message."

---

### Q7. What are the key assumptions of logistic regression? How would you test them in practice?
*(Reported: Apple DS, mid-level+)*

**Answer:**

Logistic regression has four core assumptions:

**1. Binary (or ordinal/multinomial) outcome.**
The dependent variable should be categorical. Continuous outcomes belong to linear regression. In practice, ensure your label is properly encoded and that the positive/negative class definitions are meaningful and stable.

**2. Independence of observations.**
Each row should be an independent data point. Violations occur with: repeated measures on the same user (use mixed-effects models), time-series data (use GEE or add lag features), or clustered data (cluster-robust standard errors).

**3. No severe multicollinearity.**
Covered in Q1. Test with VIF. Remediate with regularization or feature removal.

**4. Linear relationship between features and the log-odds.**
This is the most commonly violated assumption. The model assumes:
$$\log \frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \ldots + \beta_k x_k$$
So $x_1$ has a **linear** effect on the log-odds of the outcome.

**Testing this assumption:**
- Box-Tidwell test: add interaction terms $x_j \cdot \ln(x_j)$. If significant, the linearity assumption is violated.
- Visual: plot log-odds against each continuous feature using a loess smooth. Non-linearity in the plot = assumption violated.
- Remedy: apply log transforms, polynomial terms, or spline encoding.

**Goodness-of-fit:**
Use the **Hosmer-Lemeshow test** — it groups observations into deciles by predicted probability and checks if observed and predicted event rates match. A significant p-value indicates poor fit.

> **Apple-specific note:** In product analytics at Apple, you're often modeling with user telemetry features that have heavy right tails (e.g., number of app launches). Always check for log-linearity before trusting coefficient interpretations.

---

### Q8. How do you evaluate a logistic regression model? When would you choose AUC-ROC vs. PR-AUC?
*(Reported: Apple DS, L5)*

**Answer:**

**Core metrics:**

| Metric | What it measures | Use when |
|---|---|---|
| **Accuracy** | % correct predictions | Classes are roughly balanced |
| **AUC-ROC** | Discrimination across all thresholds | Balanced or slightly imbalanced data |
| **PR-AUC** | Precision-Recall tradeoff | Severely imbalanced (fraud, churn, rare events) |
| **Log-Loss** | Calibration quality of probabilities | Probabilities themselves matter (not just ranking) |
| **Brier Score** | Mean squared error of probabilities | Calibration; lower is better |
| **F1 Score** | Harmonic mean of precision/recall | Operating at a fixed threshold |

**AUC-ROC vs. PR-AUC — the key distinction:**

ROC-AUC measures performance across all classification thresholds using True Positive Rate (sensitivity) vs. False Positive Rate. It is **insensitive to class imbalance** because FPR is computed over all negatives — and when negatives dominate, even a poor model can achieve high FPR control.

PR-AUC, by contrast, uses Precision (what fraction of your predicted positives are real) and Recall (what fraction of real positives you caught). It is **dominated by performance on the positive class**, making it far more informative when positives are rare.

**Rule of thumb:** If the business cares about finding true positives (fraud detection, disease screening, churn intervention), use PR-AUC. If you need a general discrimination summary and classes are balanced, ROC-AUC is fine.

> **Apple context:** For push notification click prediction, if only 2% of users click, AUC-ROC can look great while the model misses most true clickers. PR-AUC will expose this. In a meeting with product stakeholders, I'd present recall at a fixed precision target (e.g., "at 80% precision, we capture 40% of clickers") — that's operationally meaningful.

---

### Q9. How would you handle class imbalance in logistic regression?
*(Reported: Apple DS, L4–L5)*

**Answer:**

Class imbalance occurs when one class dominates — e.g., 95% negative, 5% positive. A naive model that predicts "negative" for everything achieves 95% accuracy but is useless.

**Techniques, in order of preference:**

**1. Reframe the metric first.**
Don't optimize accuracy. Switch to PR-AUC, F1, or recall at a fixed precision. This often reveals that the raw model isn't as bad as it looks — it just needs better threshold selection.

**2. Threshold tuning.**
The default 0.5 cutoff isn't optimal under imbalance. Plot the Precision-Recall curve and choose the threshold that satisfies your business constraint (e.g., "we want precision ≥ 0.7"). This costs nothing and should always be your first lever.

**3. Class weighting.**
Set `class_weight='balanced'` in scikit-learn. This scales the loss function so that errors on the minority class receive higher penalty proportional to the imbalance ratio. Simple, effective, no data modification needed.

**4. Resampling.**
- **Oversample the minority class:** SMOTE generates synthetic minority examples via interpolation.
- **Undersample the majority class:** Randomly drop negatives to balance the dataset.
- Trade-off: oversampling risks overfitting; undersampling wastes data.

**5. Change the loss function.**
Focal loss (from Meta's RetinaNet paper) down-weights easy negative examples, forcing the model to focus on hard positives. Useful in extreme imbalance (e.g., 1:1000).

> **Apple-specific scenario:** In App Store review spam detection, the positive class (spam) might be 0.1% of all reviews. I'd: (a) weight classes, (b) tune the threshold toward higher recall to catch more spam, accepting some false positives, (c) use PR-AUC as the primary offline metric, and (d) run an A/B test to measure impact on user trust metrics before full deployment.

---

### Q10. When would you choose logistic regression over a tree-based model like XGBoost?
*(Reported: Apple DS, L5–L6)*

**Answer:**

This question tests whether you reach for the right tool or just the most fashionable one.

**Choose logistic regression when:**

1. **Interpretability is required.** If stakeholders (legal, policy, medical) need to understand exactly why a score is high, coefficient-based explanations are cleaner than SHAP values on a black-box ensemble.
2. **Data is sparse and high-dimensional.** Logistic regression with L1 regularization excels in NLP/ad feature spaces with millions of binary features. Tree models struggle with sparsity.
3. **You need well-calibrated probabilities.** Logistic regression is naturally calibrated. XGBoost outputs need Platt scaling or isotonic regression post-hoc.
4. **Latency constraints are tight.** Logistic regression inference is a dot product — nanoseconds. In production ad serving at Apple Search Ads, this matters.
5. **Training data is limited.** Fewer parameters mean lower variance. XGBoost can overfit when data is scarce.
6. **The relationship between features and log-odds is approximately linear.** If you've plotted this and it holds, logistic regression will generalize better than a more complex model that overfits the training signal.

**Choose XGBoost/tree-based when:**
- There are non-linear feature interactions you want to capture automatically.
- You have structured tabular data with enough samples.
- Calibration isn't critical and ranking (AUC) is the primary metric.

> **L6 framing:** The right answer is "start with logistic regression as the baseline." It's fast to train, easy to debug, sets a performance floor, and surfaces feature-quality issues. If logistic regression with good features gets you to AUC 0.78 and XGBoost gets to 0.81, ask whether the 3% lift justifies the operational complexity of deploying a tree ensemble. Often it doesn't.

---

## 🟪 META

---

### Q11. You are predicting click-through rate (CTR) for a new ad format at Meta. Connect the linearity and independence assumptions of logistic regression to concrete feature and data choices.
*(Reported: Meta DS interview — verified)*

**Answer:**

This question is testing whether you can connect statistical theory to engineering decisions.

**Linearity assumption (log-odds linearity):**

Logistic regression assumes $\log \frac{p}{1-p}$ is linear in your features. In CTR prediction, this means the model assumes, for example, that the log-odds of a click increases linearly with "user historical CTR." In reality:

- **User CTR** has a highly non-linear relationship with click probability (very low and very high historical CTR users behave differently).
- **Bid amount** has diminishing returns in its effect on click probability.

**Feature choices to preserve linearity:**
- Log-transform right-skewed features (e.g., `log(1 + user_click_history)`).
- Bucket continuous features into quantile bins and one-hot encode them — this approximates a step function and relaxes the linearity assumption at the cost of more parameters.
- Add explicit cross-features: `user_category_affinity × ad_category` captures interaction effects that logistic regression can't learn implicitly.

**Independence assumption:**

Logistic regression treats each training example as independent. In Meta's ad system, this is violated in multiple ways:
- The same user sees multiple ads in a session — their responses are correlated.
- Ads from the same advertiser are correlated in user response.
- Time-of-day creates clusters of similar behavior.

**Feature choices to mitigate dependence:**
- Don't use session-level features that bleed across observations (e.g., "clicks so far in this session" can cause leakage).
- Use user-level features computed over a look-back window, not the current session.
- If modeling sequences of impressions per user, consider GEE or add user-fixed effects (user embedding as a feature).

> **Production note:** At Meta scale, true independence is impossible. The practical approach is: acknowledge the violation, use robust standard errors if doing inference on coefficients, and focus on predictive performance (log-loss, AUC) rather than coefficient interpretability.

---

### Q12. Your logistic regression model for churn prediction gets 97% accuracy but misses most churners. What is the best next step?
*(Reported: Meta DS, L4–L5)*

**Answer:**

This is a classic class imbalance trap. The model has learned to predict the majority class ("not churn") exclusively, which gives high accuracy but zero recall on the class we care about.

**Step 1 — Switch the evaluation metric.**
Accuracy is the wrong metric here. Move immediately to:
- **PR-AUC** as the primary offline metric.
- **Recall at fixed precision** as the operational metric (e.g., "at 70% precision, what fraction of churners do we catch?").

**Step 2 — Audit the confusion matrix.**
Confirm the hypothesis: if precision for the positive class is ~0% and recall is ~0%, the model is predicting only negatives.

**Step 3 — Tune the decision threshold.**
Before retraining anything, plot the Precision-Recall curve for the existing model. The model may already have some discriminatory power — it's just that the 0.5 threshold is wrong. Lower the threshold (e.g., 0.1) and check if recall improves acceptably.

**Step 4 — Retrain with class weighting.**
Set `class_weight='balanced'`. This adjusts the effective loss so the minority class contributes proportionally more.

**Step 5 — Evaluate feature quality.**
High accuracy on a churn-imbalanced dataset sometimes means the model is using a near-perfect proxy for churn that you shouldn't have access to at prediction time (label leakage). Audit features carefully.

**Step 6 — Consider SMOTE or focal loss.**
If class weighting isn't sufficient, generate synthetic minority samples (SMOTE) or switch to focal loss to down-weight easy negative examples.

> **Meta framing:** "We should also ask what the business cost of a false negative vs. a false positive is. If missing a churner costs $50 in lost revenue and a false positive costs $2 in wasted retention spend, our optimal threshold is much lower than 0.5 — we should aggressively flag potential churners."

---

### Q13. Explain L1 vs. L2 regularization in logistic regression. When would you use each, and what do their geometric interpretations tell us?
*(Reported: Meta DS, L5)*

**Answer:**

Regularization adds a penalty term to the loss function to prevent overfitting:

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \Omega(\mathbf{w})$$

**L1 (Lasso):** $\Omega = \|\mathbf{w}\|_1 = \sum_j |w_j|$

**L2 (Ridge):** $\Omega = \|\mathbf{w}\|_2^2 = \sum_j w_j^2$

**Geometric intuition:**

- The L1 penalty region (in 2D) is a **diamond** with sharp corners at the axes. When the unconstrained optimum is off-center, the constrained solution tends to hit a corner — meaning one coefficient becomes exactly **zero**. This is why L1 performs automatic feature selection.
- The L2 penalty region is a **sphere**. The constrained solution lands on the sphere surface — coefficients are shrunk toward zero but almost never reach it exactly. L2 distributes weight smoothly across correlated features.

**Practical guidance:**

| Scenario | Regularization |
|---|---|
| High-dimensional sparse features (NLP, ads) | L1 — produces sparse models, interpretable |
| Many features, most are relevant | L2 — stabilizes all coefficients |
| Correlated features, want stable estimates | L2 — handles multicollinearity |
| Need to identify the most predictive features | L1 — built-in feature selection |
| Unsure | ElasticNet (blend of L1 + L2) |

**Hyperparameter $\lambda$:**
Higher $\lambda$ = more regularization = simpler model. Tune via cross-validation (e.g., `LogisticRegressionCV`). Note: scikit-learn uses `C = 1/λ`, so smaller `C` = more regularization.

> **Meta context:** In an ads CTR model with billions of sparse user-item features, L1 regularization is essential — it produces a sparse weight vector where most feature weights are zero, reducing memory and inference cost dramatically. L2 would keep all weights non-zero and balloon the model size.

---

### Q14. How does logistic regression handle multiclass problems?
*(Reported: Meta DS, L4)*

**Answer:**

Logistic regression is inherently a **binary** classifier. Two strategies extend it to multiple classes:

**1. One-vs-Rest (OvR / One-vs-All):**
Train $K$ separate binary classifiers, one for each class $k$: "class $k$ vs. all others." At inference, assign the class whose binary classifier returns the highest probability.

- Pros: Simple, parallelizable.
- Cons: Classifiers may not produce well-calibrated probabilities; the "rest" class is heterogeneous, making each binary problem potentially unbalanced.

**2. Multinomial Logistic Regression (Softmax Regression):**
Train a single model with $K$ weight vectors (one per class). The probability for class $k$ is:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

This is the **softmax function**, and the loss is categorical cross-entropy.

- Pros: Probabilities sum to 1 across classes (valid probability distribution); jointly trained, so classes interact.
- Cons: More parameters; assumes classes are mutually exclusive.

**One-vs-One (OvO):**
Train $\binom{K}{2}$ classifiers, one for each pair of classes. Final prediction by majority voting. Rarely used for logistic regression but common in SVM contexts.

**When to use which:**
- OvR is the sklearn default and works well in practice for most problems.
- Multinomial is preferable when class probabilities themselves matter (e.g., ranking use cases where you need P(class=k) to be a valid distribution).

---

### Q15. How would you use logistic regression to build a feed ranking model at Meta? What are the main challenges?
*(Reported: Meta DS, L5–L6)*

**Answer:**

**Framing the problem:**
Feed ranking is fundamentally a **learning-to-rank** problem, but logistic regression can solve a simplified version: predict the probability that a user engages with a given post (click, like, comment, share), then rank posts by this predicted probability.

**Model formulation:**
Label: binary — did the user engage with this post within a fixed window (e.g., 1 hour)?
Features: user features (demographics, historical engagement patterns), post features (content type, recency, author's historical CTR), contextual features (time of day, device, session context), interaction features (does the user historically engage with this content category?).

**Key challenges and solutions:**

**1. Massive class imbalance.**
In a feed with 100 candidate posts, a user engages with 3–5. That's a 95%+ negative rate. Use class weighting, downsample negatives, and evaluate on PR-AUC rather than ROC-AUC.

**2. Position bias.**
Posts shown higher in the feed get more clicks regardless of quality (exposure effect). Naive training conflates quality with position. Solutions: inverse propensity scoring (IPS) to reweight examples by their exposure probability, or position as a feature during training but not at serving.

**3. Feature interaction limitations.**
Logistic regression models additive effects; it can't automatically capture that "a video from a followed friend posted this morning" is much more engaging than each signal alone. Mitigation: engineer explicit cross-features (`content_type × author_relationship × recency_bucket`).

**4. Training-serving skew.**
Features available at training time must be available at serving time with the same latency. A feature like "engagement in the last 5 minutes" is powerful but may not be retrievable in the serving path without a real-time feature store.

**5. Concept drift.**
User behavior on Meta feeds evolves with product changes, news cycles, and seasonality. Retrain on a rolling window; monitor log-loss daily; alert on model degradation.

> **L6 framing:** A logistic regression feed ranking model is the right **baseline** — it's debuggable, fast to serve, and interpretable for stakeholder discussions. Once the baseline is established and features are validated, you layer in deep learning (e.g., DLRM-style models) where the lift is justified by the engineering cost. Never deploy a complex model without understanding where the gain comes from.

---

### Q16. What is the log-odds interpretation of a logistic regression coefficient? Give a concrete product example.
*(Reported: Meta DS, L4–L5)*

**Answer:**

In logistic regression, the model estimates:

$$\log \frac{P(Y=1)}{P(Y=0)} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots$$

The left-hand side is the **log-odds** (logit) of the outcome.

**Interpretation of $\beta_j$:**
A one-unit increase in $x_j$, holding all other features constant, increases the log-odds of $Y=1$ by $\beta_j$.

More intuitively: $e^{\beta_j}$ is the **odds ratio** — it tells you how the odds of the positive outcome multiply per unit increase in $x_j$.

**Concrete example:**
Suppose you're modeling whether a user clicks on a Marketplace listing. You have a feature `num_photos` (number of photos in the listing), and its coefficient is $\beta = 0.15$.

- $e^{0.15} \approx 1.16$
- Interpretation: Each additional photo **multiplies the odds of a click by 1.16**, i.e., increases the odds by 16%.

For a binary feature like `has_video` (0 or 1) with coefficient $\beta = 0.4$:
- $e^{0.4} \approx 1.49$
- Listings with a video have **1.49× higher odds of being clicked** than those without, all else equal.

**Caveats:**
- Odds ratios are multiplicative on the odds scale, not the probability scale. A 2× increase in odds from a 50% base probability takes you to 67%, but from a 5% base it takes you to ~9.5%.
- With multicollinearity, these interpretations break down.
- Coefficients reflect correlation in the training data, not necessarily causal effects. Be careful attributing causality without experimental evidence.

---

### Q17. How do you assess and ensure that your logistic regression model is well-calibrated?
*(Reported: Meta DS, L5–L6)*

**Answer:**

**What calibration means:**
A model is well-calibrated if, among all observations where it predicts P = 0.3, approximately 30% of them are actually positive. Calibration is distinct from discrimination (AUC): a model can rank examples perfectly while being systematically over- or under-confident.

**Why it matters at Meta:**
Predicted click probabilities from ad models feed directly into auction pricing: expected value = p(click) × bid. A miscalibrated model that inflates p(click) by 2× will inflate prices, distort the auction, and erode advertiser trust. Calibration is a production requirement.

**Assessment:**

1. **Calibration curve (reliability diagram):** Sort predictions into deciles. For each decile, plot mean predicted probability vs. observed positive rate. A perfectly calibrated model follows the diagonal.

2. **Hosmer-Lemeshow test:** Statistical test of whether observed and expected event rates match across deciles. Low p-value = poor calibration.

3. **Brier Score:** Mean squared error of predicted probabilities. Lower is better; baseline is the prevalence-squared.

4. **Expected Calibration Error (ECE):** Weighted average of calibration error across bins.

**Remediation:**

- **Platt Scaling:** Train a second logistic regression on top of the model's raw scores using a held-out calibration set. Simple and effective.
- **Isotonic Regression:** Non-parametric alternative; more flexible but needs more data and can overfit.
- **Temperature Scaling:** Divide logits by a learned scalar $T$ before the sigmoid. Fast, single parameter, works well for neural networks.

**Common miscalibration cause:** Training on downsampled negatives (common in CTR modeling). If negatives are downsampled by factor $w$, the model's raw probabilities need correction: $p_{\text{corrected}} = \frac{p}{p + (1-p)/w}$.

---

### Q18. What is the difference between a model trained to minimize log-loss vs. one trained to maximize AUC?
*(Reported: Meta DS, L5)*

**Answer:**

This is a subtle but important distinction that separates L5+ candidates.

**Log-loss training (cross-entropy minimization):**
- Directly minimizes $-\sum [y \log \hat{p} + (1-y) \log(1-\hat{p})]$
- Optimizes the **calibration** of predicted probabilities.
- If your downstream system uses the raw probability (e.g., ad auction pricing, expected-value decisions), you must use log-loss.
- Sensitive to class imbalance — the loss is dominated by the majority class.

**AUC maximization:**
- AUC is a **ranking metric** — it measures whether positives score higher than negatives, regardless of the absolute probability values.
- Models trained to maximize AUC may produce uncalibrated probabilities.
- Appropriate when you only care about the relative ordering of predictions (e.g., ranking a set of items).
- AUC is **not differentiable**, so direct AUC optimization requires surrogate losses (e.g., pairwise ranking loss, LambdaLoss).

**Practical guidance:**
- **Use log-loss when:** predicted probabilities drive downstream decisions (click probability × bid, churn probability × intervention cost).
- **Use AUC-oriented training when:** you need a ranked list and only the ordering matters (e.g., top-K recommendation, document ranking).

> At Meta, ad ranking systems care about **both**: they need well-calibrated probabilities for the auction (log-loss), but also need the ranking to be excellent to serve the most relevant ads first (AUC). In practice, the standard approach is to train with cross-entropy loss (which approximately maximizes AUC while preserving calibration) and then apply post-hoc calibration.

---

## 🔁 CROSS-COMPANY ADVANCED QUESTIONS

---

### Q19. Logistic regression is trained on historical data. What happens when the data distribution shifts at deployment (concept drift)? How do you detect and handle it?
*(Reported: Google/Meta senior DS, L5–L6)*

**Answer:**

**What concept drift is:**
The relationship between features and the outcome changes over time. In CTR prediction: user behavior on Meta Reels in 2023 ≠ 2025 due to product changes, new content types, and shifting user demographics.

**Types:**
- **Covariate shift:** $P(\mathbf{x})$ changes but $P(y|\mathbf{x})$ stays the same. Model may still generalize.
- **Label shift:** $P(y)$ changes (e.g., overall CTR drops). Calibration breaks.
- **Concept drift (true):** $P(y|\mathbf{x})$ changes. The model is fundamentally wrong.

**Detection:**
- Monitor **log-loss on live traffic** daily. A sustained increase is a strong signal.
- Monitor **feature distribution** with Population Stability Index (PSI). PSI > 0.2 for a key feature signals meaningful drift.
- Monitor **prediction distribution** (histogram of scores). A shift indicates the model is scoring differently on current traffic.
- Use a sliding holdout: compare model performance on last-week data vs. last-month data.

**Remediation:**
1. **Rolling retraining:** Retrain on data from a recent window (e.g., last 30 days). The right window depends on how fast your concept drifts.
2. **Weighting recent examples more:** Apply exponential decay to the training weights so recent examples have higher loss contribution.
3. **Online learning:** Update weights incrementally as new labeled data arrives (requires careful learning rate scheduling).
4. **Trigger-based retraining:** Retrain when PSI or log-loss exceeds a threshold.

> **L6 framing:** At Google/Meta, model monitoring is a first-class engineering concern. A well-functioning ML system has an automated pipeline that: (1) collects daily metrics, (2) triggers alerts, (3) automatically initiates retraining, and (4) runs shadow evaluation before promoting the new model. The data scientist's job is to define the alert thresholds and evaluate the retrained model — not to manually retrain.

---

### Q20. A logistic regression model works well in offline evaluation but underperforms in A/B test. What are the likely causes and how do you debug this?
*(Reported: Google/Meta DS, L5–L6)*

**Answer:**

The offline–online gap is one of the most important real-world ML problems. Common causes:

**1. Data leakage in offline evaluation.**
A feature used in training was computed with information available after the prediction timestamp. The model "cheats" offline but can't at serving time. **Debug:** Audit the feature pipeline for temporal leakage; ensure all features are computed using only data available at prediction time.

**2. Training-serving skew.**
The feature values computed in the training pipeline differ from those computed in the serving path (different codepaths, different data sources, different normalization). **Debug:** Log feature values at serving time; compare their distribution to the training set distribution. Use a feature monitoring dashboard.

**3. Selection bias / distribution shift.**
Offline evaluation uses historical data where the old model controlled which items were shown. The logistic regression model may score items differently, but the logged outcomes reflect the old model's selections (survivorship bias). **Debug:** Use counterfactual evaluation (inverse propensity scoring) or run the new model as a shadow ranker.

**4. Novelty effect.**
Users click on new recommendations simply because they're different, inflating the A/B test metric initially. Or conversely, users are confused by the new ranking, deflating early metrics. **Debug:** Run the test longer; look at metric trends over time.

**5. Metric mismatch.**
The offline metric (AUC, log-loss) doesn't align with the A/B test metric (session length, revenue, user satisfaction). **Debug:** Design offline evaluation to use a metric more correlated with the business outcome.

**6. Population mismatch.**
Offline evaluation used a random holdout; the A/B test exposes the model to a specific user segment (e.g., new users). **Debug:** Slice offline metrics by the same segments used in the A/B test.

> **Resolution framework:** When I see an offline-online gap, I go through: (1) feature audit for leakage, (2) distribution comparison between training and serving, (3) metric alignment review, (4) segment-level analysis. This structured approach narrows the cause systematically rather than guessing.

---

## 📚 Quick Reference: Logistic Regression Formulas

| Concept | Formula |
|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| Log-odds (logit) | $\text{logit}(p) = \log \frac{p}{1-p} = \mathbf{w}^T \mathbf{x}$ |
| Binary cross-entropy loss | $\mathcal{L} = -\frac{1}{n}\sum [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$ |
| MLE gradient | $\nabla_w \ell = \sum_i (y_i - \hat{y}_i)\mathbf{x}_i$ |
| L1 regularized loss | $\mathcal{L} + \lambda \|\mathbf{w}\|_1$ |
| L2 regularized loss | $\mathcal{L} + \lambda \|\mathbf{w}\|_2^2$ |
| Odds ratio for coeff $\beta_j$ | $e^{\beta_j}$ |
| Calibration correction (downsampled neg) | $p_{\text{corrected}} = \frac{p}{p + (1-p)/w}$ |

---

*Part 1 of the Logistic Regression Interview Series. Part 2 will cover: Bayesian logistic regression, multinomial extensions, time-series applications, fairness and bias considerations, and system design interviews where logistic regression is a component.*
