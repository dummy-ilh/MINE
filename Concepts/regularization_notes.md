# Regularization — Interview Notes

## 1. Core Definition

**Regularization:** any technique that deliberately constrains or penalizes model complexity/flexibility during training, in order to reduce **variance** (overfitting) at the cost of some added **bias** — a direct, deliberate lever on the bias-variance tradeoff (see the bias-variance notes). Rather than letting the model fit the training data as closely as possible, regularization adds a preference for "simpler" solutions, where "simpler" is defined by the specific regularization technique.

**Why it's needed:** an unconstrained, sufficiently flexible model will always achieve lower (even zero) training loss by fitting noise along with signal — training loss alone can never tell you when to stop adding flexibility (this is the same point from the hyperparameter-tuning notes about why capacity-controlling choices need a validation signal, not a training-loss signal).

## 2. L2 Regularization (Ridge)

**Penalty added to the loss:**
$$L_{ridge} = L_{original} + \lambda \sum_{j} w_j^2$$

**Effect:** shrinks all coefficients toward zero, proportionally more for larger weights (the penalty grows quadratically, so it "pushes harder" on big coefficients), but **essentially never drives them to exactly zero**. Every feature keeps some nonzero weight, just smaller than an unregularized fit would give it.

**Why it helps with variance:** large coefficients are typically a symptom of the model swinging wildly in response to small changes in the training data (a hallmark of high variance/overfitting) — constraining coefficient magnitude directly constrains how much predictions can swing.

**Closed-form intuition (for linear regression specifically):** ridge regression has a clean closed-form solution, $\hat\beta_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$ — the $\lambda I$ term added to $X^TX$ before inversion also has the practical side benefit of making the matrix invertible/numerically stable even when $X^TX$ is singular or near-singular (e.g., under multicollinearity) — a genuinely useful, separate-from-overfitting reason ridge is popular in ill-conditioned problems.

**When to prefer it:** many features are all likely at least somewhat relevant (you don't want to zero any out entirely), multicollinearity is present (correlated features — ridge tends to shrink correlated features' coefficients toward each other rather than picking one arbitrarily, which is more stable than L1's behavior in this case), or you specifically want smooth, stable coefficient estimates.

## 3. L1 Regularization (Lasso)

**Penalty added to the loss:**
$$L_{lasso} = L_{original} + \lambda \sum_{j} |w_j|$$

**Effect:** shrinks coefficients toward zero **and drives many of them to exactly zero** — L1 performs **automatic feature selection** as a byproduct of regularization, not just shrinkage. This is the single most important practical difference from L2, and the "why" is a geometry question (see §5 — the plain-language explanation you asked for).

**Why it helps with variance, additionally:** beyond generic shrinkage, exact sparsity also directly reduces effective model complexity/degrees of freedom (fewer active features = a genuinely simpler model, not just a model with smaller-but-still-nonzero coefficients everywhere) — connects to the bias-variance pitfall about irrelevant features increasing variance even when they "shouldn't hurt in theory."

**When to prefer it:** you suspect many features are truly irrelevant and want the model to actively select a sparse subset, you want a more interpretable model (fewer active features to explain), or feature count is very large relative to sample size (high-dimensional, sparse-truth settings — genomics, text with huge vocabularies) where sparsity is both statistically and computationally valuable.

**A genuine caveat/limitation:** under high multicollinearity, L1 tends to arbitrarily pick *one* of a group of correlated features and zero out the rest, somewhat unpredictably (small data perturbations can flip which one survives) — this instability is a real downside relative to ridge's more graceful shared-shrinkage behavior in that scenario.

## 4. Elastic Net — The Practical Compromise

$$L_{elastic} = L_{original} + \lambda_1\sum_j |w_j| + \lambda_2 \sum_j w_j^2$$

Combines both penalties, with a mixing parameter (often written $\alpha$, controlling the L1/L2 balance) plus overall strength $\lambda$. Gets sparsity (some coefficients exactly zero, like lasso) while handling correlated-feature groups more gracefully than pure L1 (grouped shrinkage of correlated features, closer to ridge's behavior) — specifically designed to fix lasso's instability-under-correlation weakness while keeping its sparsity benefit. A very standard, practical default when you want sparsity but are nervous about lasso's correlated-feature instability.

## 5. Why L1 Produces Sparsity and L2 Doesn't — The Geometric Explanation, In Simple Words

**Set-up, plain language:** think of fitting a model as trying to get as close as possible to the "best unconstrained fit" (the point that minimizes error with no penalty at all), but you're only *allowed* to pick a point inside some allowed "budget region" around the origin — the regularization penalty is really just a budget/boundary on how big your coefficients are allowed to get, and the shape of that allowed region is different for L1 vs. L2.

**The shape of the two budget regions (this is the whole trick):**
- **L2's allowed region is a circle (in 2D) or a sphere/ball (in higher dimensions).** Picture a perfectly round disc centered at zero — smooth, no corners anywhere on its boundary.
- **L1's allowed region is a diamond (in 2D) or its higher-dimensional generalization (a cross-polytope) — with sharp corners, and critically, those corners sit exactly on the axes** (i.e., at points like "weight 1 = something, weight 2 = exactly 0").

**Why corners matter — the actual mechanism:** you're trying to find the point *inside* the allowed region that's closest to the unconstrained best-fit point (think of concentric error contours — ellipses of "equally good fit" — expanding outward from the true best fit, and you're sliding outward until the first point where an error-contour just barely touches the boundary of your allowed region). 
- For the **circle** (L2), because it's perfectly smooth with no corners, the point where an expanding ellipse first touches the circle is, generically, somewhere on the smooth round part of the boundary — essentially never exactly on an axis, so essentially never exactly zero for any coordinate. Every touch-point is some "generic" combination of both weights being nonzero.
- For the **diamond** (L1), the corners stick out — they're the "pointiest," most protruding parts of the shape. An expanding ellipse is *much more likely* to first touch the diamond exactly at one of those protruding corners than to touch it along one of the smooth flat edges in between — and the corners are exactly the points where one coordinate is zero. So the optimal solution lands exactly on an axis far more often than not, meaning one of the weights is exactly zero.

**One-sentence version to say out loud in an interview:** "L1's constraint region has corners that sit exactly on the axes, and corners are the points an optimizer's expanding error contour is most likely to hit first — so L1 solutions tend to land exactly on an axis, meaning some weights are exactly zero, while L2's constraint region is a smooth circle with no special points at all, so its solutions are shrunk but essentially never exactly zero."

**Alternative, even simpler physical-intuition framing (good backup phrasing):** L2's penalty grows *quadratically* as a weight moves away from zero — right near zero, the penalty's slope (rate of increase) is nearly flat, so there's very little pressure pushing a small-but-nonzero weight the rest of the way to exactly zero. L1's penalty grows *linearly*, so it has a constant, non-vanishing "pull" toward zero all the way down — including right at zero itself, where the penalty has a sharp kink (a discontinuous derivative), which is precisely the mathematical reason the optimum can get "stuck" exactly at zero rather than sliding past it. (This is the calculus-flavored version of the same geometric fact — the kink at zero in $|w|$ is what the diamond's corner *is*, from a different angle.)

## 6. Other Major Regularization Techniques (Beyond L1/L2)

**Dropout (neural networks):** randomly zero out a fraction of neurons' activations during each training forward pass (different random subset each time), forcing the network to not rely too heavily on any single neuron/co-adapted group of neurons — effectively trains an implicit ensemble of many "thinned" sub-networks that share weights, and at inference time uses the full network (typically with activations scaled to compensate for the dropped-out training-time expectation). A variance-reduction technique conceptually related to bagging (bias-variance notes §6) — many "slightly different" sub-models being implicitly averaged.

**Early stopping:** stop training before the model has fully converged on the training loss, once validation loss starts increasing — treats the number of training iterations itself as an implicit capacity-control knob (fewer iterations ≈ less effective model complexity, especially for iterative optimizers like gradient descent, where more iterations generally let the model fit progressively finer, more idiosyncratic patterns including noise). Directly connects to the hyperparameter-tuning notes' point about early stopping blurring "training" and "hyperparameter search" into one loop.

**Data augmentation:** artificially expand the effective training set (image rotations/crops/flips, text back-translation, audio pitch-shifting) — doesn't change the model's parameter-penalty structure at all, but reduces variance the same way more genuine data would, by making the training distribution harder to simply memorize.

**Weight decay (a practical note on terminology):** in the context of SGD-style optimizers, "weight decay" is often used interchangeably with L2 regularization, but they're only exactly mathematically equivalent for plain SGD — for adaptive optimizers like Adam, naively adding an L2 penalty to the loss interacts with the adaptive per-parameter learning rates in a way that's *not* equivalent to true weight decay (directly shrinking weights each step, independent of the gradient-adaptive scaling) — this is precisely why **AdamW** (decoupled weight decay) exists as a distinct, now-standard fix, and it's a genuinely good precision-of-language point to raise if the conversation goes into deep learning optimizer territory.

**Max-norm constraint:** directly cap the norm of a weight vector (clip it back down if it exceeds a threshold after an update) rather than adding a soft penalty term to the loss — a "hard constraint" alternative to the "soft penalty" approach of L1/L2, occasionally used in conjunction with dropout in some architectures.

**Label smoothing:** instead of training a classifier against hard one-hot labels (100% confidence in the true class), soften the target distribution slightly (e.g., 90% true class, 10% spread across others) — regularizes by preventing the model from becoming arbitrarily overconfident, which otherwise pushes logits to extreme, high-variance-in-effect values chasing an unreachable perfect-confidence target.

**Batch normalization (a regularization *side effect*, not its primary purpose):** primarily an optimization-stabilization technique (normalizes layer inputs to control internal covariate shift-like effects during training), but it has a well-documented mild regularizing side effect too, because the batch statistics used for normalization introduce a bit of noise per-batch — worth naming as a "regularization bonus" rather than a designed-for-that-purpose technique, to show precision about primary vs. incidental effects.

**Pruning / model simplification (trees, networks):** directly restricting model capacity structurally (max tree depth, min samples per leaf, removing low-magnitude weights/connections post-training) — conceptually the most literal form of "regularization" in the plain-English sense (make the model structurally simpler), as opposed to the "soft penalty added to a loss function" flavor that L1/L2/dropout represent.

## 7. Comparison Table

| Technique | Sparsity? | Primary mechanism | Typical domain |
|---|---|---|---|
| L2 (Ridge) | No | Shrinks all weights smoothly, penalty grows quadratically | Linear/logistic regression, any differentiable model with weights |
| L1 (Lasso) | **Yes** | Drives some weights exactly to zero via the "corner" geometry / linear-penalty kink at zero | Linear/logistic regression, feature selection, high-dimensional sparse-truth settings |
| Elastic Net | Yes (less aggressive than pure L1) | Combines both, better handles correlated-feature groups than pure L1 | Same as above, when correlation is a known concern |
| Dropout | N/A (different axis) | Implicit ensembling via random neuron removal at train time | Neural networks |
| Early stopping | N/A | Limits effective optimization iterations/capacity | Any iteratively-trained model |
| Data augmentation | N/A | Effectively enlarges/diversifies training data | Images, text, audio |
| Pruning / depth limits | N/A (structural, not penalty-based) | Directly restricts structural capacity | Trees, neural networks |

## 8. Pitfalls / Trick Angles

1. **"L1 and L2 both just shrink weights, so they're basically the same idea with a different exponent."** True at a surface level (both are penalties in the loss), but the *qualitative behavior difference* (sparsity vs. no sparsity) is the entire point interviewers are testing — treating them as interchangeable "shrinkage flavors" misses the geometric/derivative reason they behave so differently right at zero (§5).

2. **Assuming more regularization is always better for reducing variance.** As covered in the bias-variance notes' pitfall list — over-regularizing can spike bias faster than it reduces variance, giving a net *worse* result; $\lambda$ needs to be tuned via validation, not maximized.

3. **Believing L1's feature selection means the zeroed-out features are truly irrelevant to the true underlying process.** Under correlated features, L1 may zero out a genuinely relevant feature simply because a correlated partner "absorbed" its signal instead — sparsity is a property of the *fitted solution given this specific data sample*, not a certified statement about true relevance, and the specific surviving/zeroed split can be unstable across resamples (§3's caveat) when features are correlated.

4. **Conflating L2 regularization with "weight decay" for adaptive optimizers like Adam.** As covered in §6 — they're only exactly equivalent for plain SGD; naively adding L2 to the loss under Adam doesn't behave like true decoupled weight decay, which is why AdamW exists as a distinct fix. A precise answer flags this rather than using the terms interchangeably by default.

5. **Treating dropout's train-time and inference-time behavior as identical.** Dropout is stochastic *only* during training; at inference, the full network is used (typically with a compensating scale adjustment) — describing dropout as "randomly disabling neurons" without this train/inference distinction is an incomplete answer.

6. **Forgetting that regularization strength interacts with feature scale.** L1/L2 penalties are applied directly to raw coefficient magnitudes — if features aren't on comparable scales (one in raw dollars, another in fractions between 0 and 1), the penalty implicitly punishes the large-scale feature's coefficient far more than intended, regardless of the feature's actual importance. Standardizing/scaling features before applying L1/L2 is a near-mandatory practical prerequisite, easy to forget to mention.

## 9. Interview Q&A

**Q1: In plain terms, why does L1 regularization produce exactly-zero coefficients while L2 doesn't?**
Picture the regularization penalty as a "budget shape" you're constrained to stay inside while getting as close as possible to the best unconstrained fit. L2's budget shape is a smooth circle/sphere with no corners, so the closest point on its boundary to any given target is essentially never exactly on an axis — every coordinate stays some small nonzero amount. L1's budget shape is a diamond with sharp corners that sit exactly on the axes, and those corners are the "pointiest" parts of the shape — geometrically the most likely place for the closest-point solution to land — so the solution frequently ends up exactly on an axis, meaning one or more coordinates are exactly zero.

**Q2: What's the calculus-level version of the same fact, without appealing to the diamond/circle picture?**
L1's penalty ($|w|$) has a constant-magnitude derivative everywhere except at $w=0$, where it has a sharp kink (an undefined/discontinuous derivative) — this constant "pull" toward zero, combined with the kink, means the optimum can genuinely land exactly at zero and stay there. L2's penalty ($w^2$) has a derivative that shrinks toward zero as $w$ approaches zero (it's smooth and flat right near the origin), so there's vanishingly little pressure pushing a small weight the rest of the way to exactly zero — it approaches but essentially never reaches it exactly.

**Q3: When would you pick Elastic Net over pure Lasso?**
When you want sparsity (automatic feature selection) but are dealing with meaningfully correlated features, where pure Lasso's tendency to arbitrarily pick one feature from a correlated group and zero out the rest (somewhat unstably across resamples) is a real concern — Elastic Net's added L2 component encourages correlated features to be shrunk together rather than one being arbitrarily favored, trading a bit of sparsity purity for more stable, reproducible feature selection.

**Q4: Is dropout a form of L1 or L2 regularization?**
No — it's a structurally different mechanism (stochastic neuron removal during training, implicitly training and averaging an ensemble of thinned sub-networks) rather than a weight-magnitude penalty added to the loss function. Both reduce variance/overfitting, but through entirely different mechanisms — a good answer avoids implying they're the same kind of thing just because both are labeled "regularization."

**Q5: Why does feature scaling matter specifically for L1/L2 regularization, in a way it might matter less for an unregularized model?**
Because the penalty term is applied directly to the raw coefficient values, and a feature's coefficient scale is inversely related to the feature's own numeric scale (a feature measured in small units needs a larger coefficient to have the same effect as one measured in large units) — without standardizing features first, the penalty implicitly and arbitrarily punishes coefficients on small-scale features more than large-scale ones, regardless of true relative importance, which can distort both which coefficients get shrunk most and, for L1 specifically, which ones get zeroed out.

**Q6 (clever): Could you construct a scenario where L1 regularization does NOT produce any exactly-zero coefficients?**
Yes, in principle — if the unconstrained best-fit point happens to already lie inside (not outside) the L1 diamond's allowed region for a given $\lambda$, no constraint is actively binding at all, and you'd just get the unconstrained solution back, sparse or not depending on what that unconstrained solution already looked like. More practically: for a sufficiently small $\lambda$ (very weak regularization), the diamond is large enough that the optimum may land on a flat edge close to a corner rather than exactly at a corner, especially in higher dimensions where "exactly on an axis" becomes a comparatively smaller target — sparsity is a strong empirical tendency of L1, driven by the geometry, but the exact degree of sparsity is $\lambda$-dependent, not an absolute guarantee at every regularization strength.
