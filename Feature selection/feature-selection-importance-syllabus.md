# Feature Selection & Feature Importance — Prep Syllabus

**Goal:** Be able to (1) explain the main families of feature selection methods and when each applies, (2) compute or reason through feature importance for both tree-based and linear models by hand, (3) explain the pitfalls (correlated features, data leakage, high-cardinality bias) that interviewers love to probe, and (4) connect this topic back to your bias-variance and optimization prep, since feature selection is fundamentally a variance-reduction technique.

Same format as your other prep chapters — built from zero, plain-language math, diagrams where useful, delivered as `.md`.

---

## Chapter 1 — Why Select Features At All (Motivation & Framing)
- The curse of dimensionality, explained intuitively (why more features can hurt, not just cost more compute)
- Feature selection vs. feature extraction/dimensionality reduction (PCA, autoencoders) — what's different about keeping vs. transforming features
- How this connects to your Ch1 bias-variance tradeoff chapter: too many irrelevant/redundant features → higher variance, overfitting; too few → higher bias, underfitting
- The three families preview: filter, wrapper, embedded methods

## Chapter 2 — Filter Methods
- Univariate statistical tests: correlation coefficient, chi-squared test, ANOVA F-test — what each assumes and when to use which (continuous vs. categorical feature/target combinations)
- Mutual information: capturing non-linear relationships that correlation misses — worked numeric example
- Variance threshold (dropping near-constant features) as the simplest possible filter
- Pros/cons: fast and model-agnostic, but ignores feature interactions and redundancy between features

## Chapter 3 — Wrapper Methods
- Forward selection, backward elimination, and stepwise selection — worked walkthrough of the search procedure
- Recursive Feature Elimination (RFE): how it uses a model's own importance/coefficients to iteratively prune
- Why wrapper methods are more accurate but far more expensive (retraining a model at every step) — connecting to combinatorial search cost
- Cross-validation's role in wrapper methods (avoiding overfitting the feature selection itself to one split)

## Chapter 4 — Embedded Methods
- L1 regularization (Lasso) and how it drives coefficients exactly to zero — the geometric intuition (diamond-shaped constraint region vs. L2's circle), tying back to your Lagrange multipliers/constrained optimization chapter
- Elastic Net as a middle ground between L1 (sparsity) and L2 (stability with correlated features)
- Tree-based embedded selection: features that are never/rarely split on are implicitly de-selected
- Why embedded methods are the most commonly used in practice (selection happens "for free" during training)

## Chapter 5 — Feature Importance in Tree-Based Models
- Mean Decrease in Impurity (MDI / Gini importance): how it's computed, worked numeric example on a small tree
- Known bias of MDI: inflated importance for high-cardinality and continuous features — why, with a concrete illustration
- Permutation importance: shuffle-and-measure-performance-drop approach — worked example, and why it's more reliable than MDI
- Permutation importance's own pitfalls: correlated features splitting credit, and cost (requires re-scoring the model many times)

## Chapter 6 — SHAP and Other Model-Agnostic Importance Methods
- Shapley values from cooperative game theory — the intuition (fair credit allocation across "players"/features) before any formula
- How SHAP approximates Shapley values for ML models, and what a SHAP summary plot / force plot actually shows
- LIME as a simpler, local-approximation alternative — how it differs from SHAP's more principled but expensive approach
- When to reach for SHAP/LIME vs. simpler importance methods (model-agnostic need, local vs. global explanation need)

## Chapter 7 — Feature Importance for Linear Models
- Raw coefficient magnitude and why it's meaningless without standardized features — worked example showing how scale distorts naive coefficient comparison
- Standardized coefficients and how to compute/interpret them correctly
- p-values and confidence intervals on coefficients — statistical significance vs. practical importance (a coefficient can be "significant" but tiny)
- Multicollinearity's effect on coefficient stability (VIF — Variance Inflation Factor — as the standard diagnostic)

## Chapter 8 — Pitfalls and Gotchas
- Correlated/redundant features: why importance gets "split" between them, making individually low importance scores misleading
- Data leakage through feature selection done before train/test split (the single most common interview trap) — why you must select features only within each cross-validation fold
- High-cardinality categorical features inflating tree-based importance artificially
- Feature importance ≠ causal importance — correlation-driven importance can mislead if the goal is intervention/causal reasoning, not just prediction

## Chapter 9 — Practical Synthesis
- End-to-end worked case: given a dataset with 200 raw features, walk through filter → embedded → SHAP-based validation, catching a leakage bug along the way
- A decision framework: which method to reach for given dataset size, model type, interpretability requirement, and compute budget
- Common interview questions on this topic and how to structure a strong answer (e.g. "How would you reduce a 500-feature dataset down to the 20 that matter?", "Why might a feature have high importance in a random forest but a small coefficient in logistic regression?")

---

**Suggested build order:** 1 → 2 → 3 → 4 (the three selection families, in increasing sophistication) → 5 → 6 → 7 (importance methods, tree-based first since that's most commonly asked, then model-agnostic, then linear) → 8 (pitfalls, best absorbed after you've seen the methods) → 9 (synthesis). Same shape as your Fairness chapters — foundations, then techniques, then a practical synthesis chapter tying it together.

Let me know if you want me to start drafting Chapter 1 in full.
