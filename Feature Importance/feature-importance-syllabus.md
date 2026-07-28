# Feature Importance — Prep Syllabus

**Goal:** Be able to (1) compute or reason through feature importance by hand for tree-based, linear, and black-box models, (2) explain the guarantees and failure modes of each method precisely enough to defend a choice in an interview, (3) diagnose why an importance ranking looks wrong (correlation, cardinality bias, non-linearity) and fix it, and (4) connect importance to real decisions — explaining one prediction, debugging a model, or informing (without overclaiming) an intervention.

This is a standalone deep-dive on **importance/explanation** specifically — separate from your Feature Selection syllabus, which covers *choosing* features (filter/wrapper/embedded). This one assumes a model is already trained and asks "what is it actually relying on?" Some foundational material (MDI, permutation importance, SHAP, linear coefficients) will overlap with Chapters 5–7 of that syllabus — this version goes deeper and treats importance as the main subject rather than a supporting topic.

---

## Chapter 1 — What "Importance" Actually Means (Motivation & Framing)
- Why "importance" is not one thing: global vs. local, predictive vs. causal, model-specific vs. model-agnostic — a taxonomy to hold onto through every later chapter
- Why two valid importance methods can disagree with each other on the same model, and why that's expected, not a bug
- Preview of the taxonomy: intrinsic (built into a model type) vs. post-hoc (computed after training, on any model)

## Chapter 2 — Intrinsic Importance: Tree-Based Models
- Mean Decrease in Impurity (MDI/Gini importance): full derivation, worked numeric example
- Mean Decrease in Accuracy as a variant, and how it differs from MDI
- The high-cardinality/continuous-feature bias in depth: why it happens, a synthetic-noise-feature demonstration, and when it's safe to ignore vs. dangerous
- Feature importance in boosted trees (XGBoost/LightGBM's `gain`, `weight`, `cover` importance types) — what each one actually measures and why they can rank features differently from each other

## Chapter 3 — Intrinsic Importance: Linear & Generalized Linear Models
- Raw vs. standardized coefficients, worked scale-distortion example
- Statistical significance (p-values, confidence intervals) vs. practical importance
- Multicollinearity and VIF, derived from first principles
- Odds ratios in logistic regression as an importance-adjacent, more interpretable quantity for classification

## Chapter 4 — Permutation Importance in Depth
- Full procedure, worked example
- Why it fixes MDI's cardinality bias — the precise mechanism
- Its own failure modes: correlated-feature masking, and the subtler problem of extrapolation (shuffled combinations of feature values may not resemble real data the model was ever trained on)
- Conditional/grouped permutation importance as a fix for correlated features

## Chapter 5 — Shapley Values and SHAP, Formally
- Full derivation of the Shapley value formula from cooperative game theory (the actual weighted-average-over-coalitions formula, not just the intuition)
- The four Shapley axioms (efficiency, symmetry, dummy, additivity) and why they make Shapley values the *unique* fair allocation — why this uniqueness matters for defending SHAP's guarantees in an interview
- TreeSHAP, KernelSHAP, DeepSHAP: what structural shortcut each one exploits to avoid the exponential cost
- Reading SHAP interaction values (going beyond single-feature attribution to pairwise interaction effects)

## Chapter 6 — LIME and Other Local Surrogate Methods
- Full procedure with a worked local-fit example
- The perturbation/sampling and kernel-weighting choices that determine LIME's explanation, and why they make it somewhat unstable across repeated runs
- Anchors (rule-based local explanations) as an alternative to LIME's linear surrogate approach
- When local surrogate methods mislead: non-smooth model behavior near the neighborhood being explained

## Chapter 7 — Partial Dependence & Individual Conditional Expectation (PDP/ICE)
- What a PDP shows: the model's average predicted response as one feature varies, holding others at their marginal distribution
- Why PDPs can be misleading under correlated features (extrapolating into unrealistic feature combinations) — connects directly to permutation importance's extrapolation problem from Chapter 4
- ICE plots: showing individual-example curves instead of the PDP's average, and what heterogeneity across ICE curves tells you that a PDP alone hides
- Accumulated Local Effects (ALE) plots as a fix for the correlated-feature extrapolation problem

## Chapter 8 — Global vs. Local, and Aggregating Local Explanations
- Formally defining global importance (dataset-wide) vs. local importance (single-prediction) and why some methods only give you one or the other natively
- How SHAP values get aggregated into a global ranking (mean absolute SHAP value) and what's lost in that aggregation
- Stability of importance rankings: how much do rankings change across bootstrap resamples or retraining runs, and how to measure/report that

## Chapter 9 — Pitfalls and Gotchas
- Correlated features splitting/masking credit — the recurring failure mode across every method, unified into one lesson
- High-cardinality bias (MDI) and extrapolation bias (permutation, PDP) as the two dominant "the number looks wrong" failure classes
- Data leakage risk when importance is computed on data the model has already seen, vs. proper held-out evaluation
- Predictive/associational importance vs. causal importance — the ice-cream/drowning pattern, and what to reach for instead when the goal is intervention, not explanation

## Chapter 10 — Practical Synthesis
- End-to-end worked case: given a trained model and a stakeholder asking "why did we deny this specific applicant," walk from global importance → local SHAP explanation → sanity-checking against a second method → producing a defensible, correct explanation
- A decision framework: which importance method to reach for, given model type, global-vs-local need, and compute budget
- Common interview questions on this topic and how to structure a strong answer (e.g. "Why do MDI and permutation importance disagree on this model?", "How would you explain a single denied loan application to a regulator?")

---

**Suggested build order:** 1 → 2 → 3 (intrinsic, model-specific methods) → 4 → 5 → 6 (post-hoc, model-agnostic methods, roughly in order of how principled/expensive they are) → 7 (a different lens — dependence rather than attribution) → 8 (tying global/local together) → 9 (pitfalls, best absorbed once you've seen every method) → 10 (synthesis).

Let me know if you want me to start drafting Chapter 1 in full.
