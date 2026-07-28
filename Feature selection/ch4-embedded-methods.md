# Chapter 4 — Embedded Methods

Chapters 2 and 3 covered two extremes: filter methods (score features independently, no model) and wrapper methods (search over subsets by retraining a model many times). This chapter covers the middle ground that's actually used most often in practice: **methods where feature selection happens automatically as a side effect of training a single model, at no extra computational cost.**

## 4.1 The core idea: selection "for free"

With embedded methods, you don't run a separate selection step at all — you just train your model the normal way, and the training process itself naturally drives some features toward "not used" (a coefficient of exactly zero, or a feature the tree simply never splits on). This is why embedded methods are the most commonly used family day-to-day: you get the benefit of selection without the exponential-search cost of wrapper methods or the blind-spot-to-interactions problem of filter methods.

## 4.2 L1 regularization (Lasso) — the geometric intuition first

You've already seen L1 regularization in your optimization prep as a penalty term added to the loss. Here we're revisiting the exact same object, but asking a more specific question than before: *why does L1, specifically, drive some coefficients to exactly zero, while L2 only shrinks them toward zero without ever quite reaching it?*

**Setup:** Lasso adds a penalty proportional to the sum of the *absolute values* of the coefficients:

L_total = L_prediction(w) + λ · Σ|w_i|

compare this to L2/Ridge, which penalizes the sum of *squared* coefficients:

L_total = L_prediction(w) + λ · Σw_i²

**The geometric picture (this is the part worth really sitting with, since "why does L1 give sparsity" is a very common interview question).** Think of fitting a model with just two coefficients, w₁ and w₂, so you can draw the whole picture in 2D. The prediction loss L_prediction(w) traces out elliptical contours in (w₁, w₂) space — each ellipse is a set of coefficient pairs giving the same loss value, with the unconstrained best-fit point at the center of the smallest ellipse. The regularization term restricts you to a *constraint region* around the origin (this connects directly to the Lagrangian/constrained-optimization framing from your optimization prep — adding a penalty term is equivalent to constraining w to lie within some region and finding the best point inside it):

- **L2's constraint region is a circle** (or a sphere in higher dimensions): the set of points where w₁² + w₂² ≤ some budget.
- **L1's constraint region is a diamond** (a shape with sharp corners sitting exactly on the axes): the set of points where |w₁| + |w₂| ≤ some budget.

The fitted solution is wherever the loss ellipse first touches the constraint region as you shrink the ellipse toward the unconstrained optimum. **Here's the key visual fact:** a diamond has sharp corners that sit exactly on the axes (e.g., the point (budget, 0) — where w₂ = 0 exactly). Because ellipses are smooth, curved shapes, they are disproportionately likely to first touch a diamond's constraint region **exactly at one of its corners** — and a corner on the w₁-axis means w₂ = 0 exactly. A circle, by contrast, has no corners at all — an ellipse touching a circle can land anywhere on its smooth boundary, essentially never landing exactly on an axis. This geometric asymmetry — corners that sit on the axes vs. no corners at all — is the entire reason L1 produces exact zeros (true feature elimination) while L2 only shrinks coefficients toward small nonzero values (every feature technically still has *some* weight, just a small one).

**The practical consequence:** after fitting a Lasso model, simply look at which coefficients came out as exactly zero — those features have been automatically, implicitly deselected. No separate selection step needed; it happened inside the normal training process.

## 4.3 Elastic Net — combining L1 and L2

**The problem Elastic Net solves:** Lasso's sparsity is powerful, but it has a specific weakness when features are highly correlated with each other. If two features are nearly identical (say, `income` and a near-duplicate `annual_earnings` feature), Lasso tends to arbitrarily pick *one* of them to keep a nonzero coefficient on and zero out the other — which one it picks can be unstable (sensitive to small changes in the data), even though both features were equally informative.

**The fix:** Elastic Net combines both penalties:

L_total = L_prediction(w) + λ₁ · Σ|w_i| + λ₂ · Σw_i²

The L1 term still gives you sparsity (some coefficients driven to exactly zero), while the added L2 term encourages correlated features to be treated more similarly to each other (shrunk together, rather than one arbitrarily zeroed and the other kept) — a compromise between L1's aggressive, sometimes-unstable sparsity and L2's stability-but-never-quite-zero behavior.

## 4.4 Tree-based implicit selection

**How it happens automatically:** at every split in a decision tree, the algorithm searches over all available features (and split points within each) and picks whichever single split most reduces impurity (this connects to the impurity measures you'll see formalized in Chapter 5's MDI discussion). A feature that never provides a good enough split — because it's genuinely uninformative relative to the other features available — simply **never gets chosen for any split, in any tree, ever**, and is therefore implicitly excluded from the model's actual decision-making, with zero extra selection step required.

**Why this happens "for free" in a way filter and wrapper methods don't:** the tree-building algorithm was already searching over all features at every split anyway, purely to do its normal job of fitting the data well — the implicit feature selection is a byproduct of that search, not an additional procedure bolted on afterward.

**A caveat worth flagging now, expanded fully in Chapter 8:** a feature "never being split on" doesn't necessarily mean it's truly useless — it might be genuinely useful but **redundant** with another feature that happened to get picked first (two correlated features competing for the same splits, with only one "winning" at each node) — this is the same redundancy issue Elastic Net (4.3) addresses for linear models, just showing up in tree-based selection instead.

## 4.5 Why embedded methods dominate in practice

Pulling 4.2–4.4 together into the practical takeaway: embedded methods give you most of the benefit of a dedicated selection process (Lasso can capture some feature interactions indirectly through what the loss landscape looks like; trees directly capture interactions since splits can depend on prior splits) **without a separate, expensive search step** — you're already training the model anyway, and selection comes along for the ride. This is why, in practice, most real-world feature selection is done implicitly through the choice of model and regularization strength, with filter methods (Chapter 2) reserved for an initial cheap cut on very high-dimensional raw data, and wrapper methods (Chapter 3) reserved for cases where you specifically need to search over model *types*, not just within one model's own training process.

---

**Next: Chapter 5 — Feature Importance in Tree-Based Models**, shifting from *selecting* features to *explaining* which features a trained model actually relied on — starting with Mean Decrease in Impurity, its well-known bias toward high-cardinality features, and why permutation importance is generally the more trustworthy alternative.
