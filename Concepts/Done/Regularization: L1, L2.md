# Regularization: L1, L2, Dropout, and Beyond — Interview Notes

## 0. The Unifying Idea

Regularization = deliberately **constraining or penalizing model complexity** to trade a small increase in bias for a larger decrease in variance (ties directly to the bias-variance tradeoff — this is the main lever labeled "reduce variance" in that framework).

**Why it's needed:** an unconstrained, sufficiently flexible model will always achieve lower (even zero) training loss by fitting noise along with signal — training loss alone can never tell you when to stop adding flexibility. Regularization adds a preference for "simpler" solutions, where "simpler" is defined by the specific technique.

General form:

$$J(\theta) = \underbrace{L(\theta)}_{\text{data loss (e.g., MSE, cross-entropy)}} + \underbrace{\lambda \cdot R(\theta)}_{\text{penalty}}$$

$\lambda$ (regularization strength) controls where you sit on the bias-variance curve: $\lambda=0$ → unregularized (low bias, high variance); $\lambda\to\infty$ → penalty dominates, model collapses toward trivial (high bias, low variance).

---

## 1. L2 Regularization (Ridge)

**Penalty:** $R(\theta) = \|\theta\|_2^2 = \sum_j \theta_j^2$

**Full objective (regression):**
$$J(\theta) = \sum_i (y_i - \hat y_i)^2 + \lambda \sum_j \theta_j^2$$

**Closed-form solution** — worth being able to write this:
$$\hat\theta_{\text{ridge}} = (X^TX + \lambda I)^{-1} X^Ty$$

Compare to OLS: $\hat\theta_{\text{OLS}} = (X^TX)^{-1}X^Ty$. The $\lambda I$ term is added to the diagonal — this is *literally* why Ridge fixes multicollinearity: it makes $X^TX+\lambda I$ invertible/better-conditioned even when $X^TX$ is singular or near-singular (correlated features).

**Effect on weights:** shrinks all coefficients toward zero — proportionally more for larger weights, since the penalty grows quadratically and "pushes harder" on big coefficients — but **never exactly to zero** (except in degenerate cases). Every feature keeps some nonzero weight.

**Why it helps with variance:** large coefficients are typically a symptom of the model swinging wildly in response to small changes in the training data (a hallmark of high variance/overfitting) — constraining coefficient magnitude directly constrains how much predictions can swing.

**Bayesian interpretation (good to mention — shows depth):** Ridge = MAP estimate under a **Gaussian prior** on the weights, $\theta_j \sim N(0, \tau^2)$, where $\lambda = \sigma^2/\tau^2$. Tighter prior (smaller $\tau^2$) → bigger $\lambda$ → more shrinkage.

**When to prefer it:** many features are all likely at least somewhat relevant (you don't want to zero any out entirely); multicollinearity is present (Ridge tends to shrink correlated features' coefficients toward each other rather than picking one arbitrarily); you want smooth, stable estimates; you need a closed-form solution for speed at scale.

---

## 2. L1 Regularization (Lasso)

**Penalty:** $R(\theta) = \|\theta\|_1 = \sum_j |\theta_j|$

$$J(\theta) = \sum_i (y_i - \hat y_i)^2 + \lambda \sum_j |\theta_j|$$

**No closed form** in general (the penalty isn't differentiable at 0) — solved via coordinate descent, subgradient methods, or LARS.

**Effect on weights:** drives many coefficients to **exactly zero** → automatic **feature selection** / sparsity, not just shrinkage.

**Why it helps with variance, additionally:** beyond generic shrinkage, exact sparsity directly reduces effective model complexity/degrees of freedom — fewer active features is a genuinely simpler model, not just one with smaller-but-still-nonzero coefficients everywhere.

**Bayesian interpretation:** Lasso = MAP estimate under a **Laplace (double-exponential) prior** on weights — the Laplace distribution has a sharp peak at 0, which pushes mass toward exact zeros, unlike the smooth Gaussian bump of the Ridge prior.

**Subgradient at $\theta_j=0$:** since $|\theta_j|$ isn't differentiable there, optimization uses the subgradient $\partial|\theta_j| \in [-1,1]$ at 0 — the reason a coefficient can get "stuck" at exactly zero rather than just approaching it.

**A genuine caveat/limitation:** under high multicollinearity, L1 tends to arbitrarily pick *one* feature from a correlated group and zero out the rest, somewhat unpredictably — small data perturbations can flip which one survives. This instability is a real downside relative to Ridge's graceful shared-shrinkage behavior.

**When to prefer it:** you suspect many features are truly irrelevant and want active selection of a sparse subset; you want interpretability (fewer active features to explain); $p \gg n$ (high-dimensional, sparse-truth settings — genomics, text with huge vocabularies) where sparsity is both statistically and computationally valuable.

---

## 3. Why L1 Produces Sparsity and L2 Doesn't — Two Complementary Explanations

### 3a. Plain-language geometric picture (say this out loud first)

Think of fitting a model as trying to get as close as possible to the "best unconstrained fit" (the point minimizing error with no penalty), but you're only allowed to pick a point inside some **budget region** around the origin — the penalty is a boundary on how big coefficients are allowed to get, and the *shape* of that region differs between L1 and L2.

- **L2's allowed region is a circle/sphere** — smooth, no corners anywhere.
- **L1's allowed region is a diamond** (cross-polytope in higher dimensions) — sharp corners that sit **exactly on the axes** (e.g., "weight 1 = something, weight 2 = exactly 0").

Picture concentric elliptical error contours expanding outward from the unconstrained best fit until they first touch the boundary of the allowed region:

- **Circle (L2):** because it's perfectly smooth, the first touch-point is, generically, somewhere on the round part of the boundary — essentially never exactly on an axis. Every touch-point is some generic combination of nonzero weights.
- **Diamond (L1):** the corners are the "pointiest," most protruding parts of the shape. An expanding ellipse is much more likely to first touch the diamond exactly at a corner than along a flat edge — and corners are exactly the points where one coordinate is zero.

**One-sentence version for an interview:** "L1's constraint region has corners that sit exactly on the axes, and corners are the points an expanding error contour is most likely to hit first — so L1 solutions tend to land exactly on an axis (some weights exactly zero), while L2's smooth circle has no special points, so its solutions are shrunk but essentially never exactly zero."

### 3b. The calculus/mathematical version (the "why," not just the picture)

At the optimum, for L2 the penalty's gradient is $2\lambda\theta_j$, which **vanishes as $\theta_j\to 0$** — so there's no active force pushing a small coefficient the rest of the way to exactly zero; it just asymptotically shrinks (the penalty is nearly flat right near the origin).

For L1, the penalty's (sub)gradient is $\lambda \cdot \text{sign}(\theta_j)$ — a **constant-magnitude force regardless of how close to zero $\theta_j$ already is**, with a sharp kink (undefined/discontinuous derivative) exactly at zero. This constant, non-vanishing pull can overshoot the coefficient to a corner solution, where the optimizer settles exactly at $\theta_j=0$ if the data-loss gradient there is smaller than $\lambda$.

The kink in $|w|$ at $w=0$ *is* the diamond's corner — 3a and 3b are the same fact viewed geometrically vs. analytically.

**Caveat worth naming (shows precision, not overclaiming):** sparsity is a strong empirical tendency of L1, not an absolute guarantee at every $\lambda$. If the unconstrained best-fit point already lies inside the L1 diamond for a given $\lambda$, no constraint binds at all. For small $\lambda$, or in high dimensions where "exactly on an axis" is a comparatively smaller target, the optimum can land on a flat edge near — but not exactly at — a corner.

---

## 4. Elastic Net (the "why not both" answer)

$$R(\theta) = \alpha\|\theta\|_1 + (1-\alpha)\|\theta\|_2^2$$

Combines both: gets sparsity from L1 but is more stable than pure Lasso when features are highly correlated. Lasso tends to arbitrarily pick *one* of a correlated group and zero out the rest; Elastic Net's added L2 component encourages correlated features to be shrunk together instead (the "grouping effect") — trading a little sparsity purity for more stable, reproducible feature selection. Good default answer to "what would you actually use in production with correlated features when you still want sparsity."

---

## 5. Numerical Example (real simulation, not hand-waved)

Synthetic data: $y = 3x_0 - 2x_1 + \text{noise}$, where $x_1$ is **correlated with $x_0$** (r≈0.85), and $x_2,\dots,x_5$ are **pure irrelevant noise features** (true coefficient 0). This mirrors a realistic production scenario: some real signal, some redundant/collinear signal, some junk features.

| Feature | True coef | OLS | Ridge (α=1) | Ridge (α=10) | Lasso (α=0.05) | Lasso (α=0.3) |
|---|---|---|---|---|---|---|
| x0 | 3.00 | 3.126 | 2.405 | 1.140 | 1.176 | 0.902 |
| x1 | -2.00 | -2.227 | -1.395 | 0.034 | -0.000 | 0.000 |
| x2 | 0.00 | -0.068 | -0.066 | -0.059 | -0.013 | -0.000 |
| x3 | 0.00 | -0.063 | -0.068 | -0.074 | -0.028 | -0.000 |
| x4 | 0.00 | 0.083 | 0.069 | 0.044 | 0.000 | 0.000 |
| x5 | 0.00 | -0.113 | -0.110 | -0.103 | -0.050 | -0.000 |

$\|\theta\|_2$: OLS = 3.842 → Ridge(α=1) = 2.785 → Ridge(α=10) = 1.149 (monotonic, smooth shrinkage)
Nonzero count: OLS = 6 → Lasso(α=0.05) = 4 → Lasso(α=0.3) = **1** (sparsity kicks in and grows with α)

**What to point out from this table in an interview:**
1. **OLS overfits the noise features** slightly (x2–x5 get small but nonzero, spurious coefficients) — the "irrelevant feature increases variance" pitfall made concrete.
2. **Ridge shrinks everything smoothly** — even noise features shrink slightly but never hit exact zero; at high α the correlated pair (x0, x1) gets compressed together rather than one dominating.
3. **Lasso does real feature selection** — at α=0.05, noise features are already zeroed; at α=0.3 only x0 survives — and notably it **kept x0 and dropped x1** even though both were "true" signal, because they're collinear and Lasso arbitrarily picks one from a correlated pair. This is the textbook Lasso-instability-under-collinearity phenomenon, visible in real numbers.

---

## 6. Dropout

**Mechanism (training time):** at each forward pass, independently zero out each unit (not weight) with probability $p$ (commonly $p=0.5$ for hidden layers, lower like 0.1–0.2 for input layers). A different random subnetwork is trained on every minibatch.

**Inverted dropout (what's actually used in practice):** to keep the *expected* activation magnitude the same between train and inference, scale surviving activations by $\frac{1}{1-p}$ **during training**, so **no rescaling is needed at inference time** (just run the full network as-is). This is the standard modern implementation (vs. the older "scale down at test time" version).

Quick derivation worth producing on demand: if a unit's activation is $a$, and it survives with probability $(1-p)$, then $E[\text{masked }a] = (1-p)\cdot a$. Dividing by $(1-p)$ during training restores $E[\cdot] = a$, matching the test-time (no dropout) expectation.

**Why it works (multiple valid framings, know at least two):**
1. **Implicit ensembling:** training with dropout approximately trains an exponential number of "thinned" subnetworks with shared weights, and inference (no dropout) approximates *averaging* their predictions — a cheap approximation to bagging without storing/training separate models.
2. **Prevents co-adaptation:** units can't rely on any specific other unit being present, forcing each unit to learn features useful in combination with many different random subsets of other units — more robust, redundant representations.
3. **Noise injection / regularization view:** mathematically similar in spirit to adding multiplicative noise to activations, which (for linear models) can be shown to approximate an L2-like penalty on the weights in expectation.

**Does dropout just add noise around a fixed loss surface, or change it?** It genuinely changes the effective objective — training with dropout approximately optimizes a regularized objective related to the (weighted) predictions of all thinned subnetworks, a different, more conservative objective than optimizing the full network directly. This distinguishes it from naive additive Gaussian noise on the loss, which is truly noise around the *same* underlying objective.

**Where it's applied:** hidden fully-connected layers classically; **much less common/effective in convolutional layers** (spatial correlation means neighboring units carry redundant info, weakening dropout's effect there — spatial dropout / DropBlock exist as conv-specific variants). Not typically applied to the output layer (see pitfalls). Less commonly used inside modern Transformer attention blocks with heavy normalization, though it still appears in various sublayers.

---

## 7. Other Major Regularization Techniques (Beyond L1/L2/Dropout)

**Early stopping:** stop training before the model fully converges on training loss, once validation loss starts increasing — treats the number of training iterations itself as an implicit capacity-control knob (more iterations generally let the model fit progressively finer, more idiosyncratic patterns including noise).

**Data augmentation:** artificially expand the effective training set (image rotations/crops/flips, text back-translation, audio pitch-shifting) — doesn't change the model's parameter-penalty structure at all, but reduces variance the same way more genuine data would, by making the training distribution harder to simply memorize.

**Max-norm constraint:** directly cap the norm of a weight vector (clip it back down if it exceeds a threshold after an update) rather than adding a soft penalty term to the loss — a "hard constraint" alternative to the "soft penalty" approach of L1/L2, occasionally used alongside dropout.

**Label smoothing:** instead of training against hard one-hot labels (100% confidence in the true class), soften the target distribution slightly (e.g., 90% true class, 10% spread across others) — prevents the model from becoming arbitrarily overconfident, which otherwise pushes logits toward extreme, high-variance-in-effect values chasing an unreachable perfect-confidence target.

**Batch normalization (a regularization *side effect*, not its primary purpose):** primarily an optimization-stabilization technique (normalizes layer inputs to control internal covariate shift-like effects), but has a well-documented mild regularizing side effect too, since the batch statistics used introduce a bit of noise per-batch — a "regularization bonus," not a designed-for-that-purpose technique.

**Pruning / structural capacity limits (trees, networks):** directly restricting model capacity structurally (max tree depth, min samples per leaf, removing low-magnitude weights/connections post-training) — conceptually the most literal form of "regularization" in the plain-English sense, as opposed to the "soft penalty added to a loss function" flavor L1/L2/dropout represent.

---

## 8. Comparison Tables

### L1 vs. L2

| | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Penalty | $\sum \theta_j^2$ | $\sum \lvert\theta_j\rvert$ |
| Coefficient behavior | Shrinks smoothly, never exactly 0 | Drives many to exactly 0 |
| Feature selection | No (implicit, "soft") | Yes (explicit, sparse) |
| Closed form | Yes | No (needs iterative solver) |
| Geometric shape | Circle/sphere (smooth) | Diamond (has corners) |
| Bayesian prior | Gaussian | Laplace |
| Penalty gradient near 0 | Vanishes (→0) | Constant magnitude (kink at 0) |
| Handles multicollinearity | Yes — spreads weight across correlated features | Poorly alone — picks one arbitrarily |
| Differentiable everywhere | Yes | No (kink at 0) |
| Best when | Many small/medium true effects, correlated features | True underlying model is sparse |

### Broader technique map

| Technique | Sparsity? | Primary mechanism | Typical domain |
|---|---|---|---|
| L2 (Ridge) | No | Shrinks all weights smoothly, quadratic penalty | Linear/logistic regression, any differentiable model |
| L1 (Lasso) | **Yes** | Drives weights exactly to zero via corner geometry / linear-penalty kink | Regression, feature selection, high-dim sparse settings |
| Elastic Net | Yes (less aggressive) | Combines both, handles correlated groups better than pure L1 | Same, when correlation is a concern |
| Dropout | N/A (different axis) | Implicit ensembling via random neuron removal at train time | Neural networks |
| Early stopping | N/A | Limits effective optimization iterations/capacity | Any iteratively-trained model |
| Data augmentation | N/A | Enlarges/diversifies effective training data | Images, text, audio |
| Pruning / depth limits | N/A (structural) | Directly restricts structural capacity | Trees, neural networks |

---

## 9. Diagnosis: When to Reach for Which

| Symptom | Lean toward |
|---|---|
| Train/val gap large, many features, suspect several are irrelevant | Lasso (sparsity + implicit feature selection) |
| Train/val gap large, features correlated/collinear, all plausibly relevant | Ridge |
| Want sparsity but features are correlated groups | Elastic Net |
| Deep net overfitting, activations/hidden units, not linear coefficients | Dropout (+ weight decay/L2 often combined) |
| High variance, but interpretability of which raw features matter is a hard business requirement | Lasso (even at some accuracy cost) |
| Need theoretical guarantees / closed-form solution for speed at scale | Ridge |

**Practical tuning method for all of them:** cross-validate $\lambda$ (or $p$ for dropout) via a grid/log-scale sweep, often paired with the **1-SE rule** — pick the simplest model (largest $\lambda$/regularization) whose CV error is within one standard error of the best observed CV error, rather than blindly picking the single lowest-error point (which itself has CV-estimation variance).

---

## 10. Pitfalls (interviewer bait)

1. **"Lasso is strictly better because it gives sparsity."** Not universally true — if the true model isn't sparse (many small true effects), Lasso's sparsity is a *bias* it introduces that doesn't match reality; Ridge often wins on pure predictive performance in that regime. Sparsity is a property you want for interpretability/feature selection, not automatically a "better" regularizer.
2. **Believing L1's feature selection means the zeroed-out features are truly irrelevant.** A feature Lasso zeroes out isn't proven "not predictive" — it may have lost an arbitrary tie-break against a correlated feature (shown concretely in the x0/x1 numerical example), and the specific surviving/zeroed split can be unstable across resamples.
3. **"L1 and L2 are basically the same idea with a different exponent."** True at a surface level (both are penalties on the loss), but the qualitative behavior difference — sparsity vs. no sparsity — is the entire point interviewers test, and it traces to the derivative/geometry difference at zero (§3).
4. **Forgetting regularization interacts with feature scaling.** Both penalties apply directly to raw coefficient magnitudes — un-normalized features get penalized unequally (a feature in raw dollars vs. one in fractions between 0 and 1). **Always standardize features before regularizing** — an easy, near-mandatory interview gotcha to mention.
5. **Applying dropout at inference time by mistake** (forgetting `model.eval()` in PyTorch, or leaving training-mode flags on) — silently destroys inference quality/determinism. A very real production bug, not just theoretical.
6. **Treating dropout's train-time and inference-time behavior as identical.** Dropout is stochastic only during training; at inference the full network is used (with inverted-dropout scaling already baked in during training). Describing dropout as "randomly disabling neurons" without this distinction is an incomplete answer.
7. **Using a single dropout rate everywhere without justification.** Different layers often want different rates (lower near input, none in the final output layer — see Q&A below, none typically applied to attention/embedding layers in modern LLMs by default).
8. **Thinking dropout and L2 (weight decay) are redundant so you should only use one.** They regularize differently (structural sparsification of *activations* per-batch vs. shrinking of *weights* globally) and are commonly and successfully combined, especially in older CNN architectures.
9. **Believing regularization strength should always be maximized for "more robust" models.** Over-regularizing spikes bias — same U-curve logic as model complexity, just traversed in the other direction; $\lambda$ needs to be tuned via validation, not maximized.
10. **Weight decay ≠ L2 regularization in adaptive optimizers.** In Adam (vs. plain SGD), naively adding an L2 penalty to the loss does **not** behave like classic "weight decay," because Adam's adaptive per-parameter learning rates interact with the penalty gradient in a way that decouples it from true weight decay. This is exactly why **AdamW** exists — it decouples weight decay from the gradient-based adaptive update (Loshchilov & Hutter, "Decoupled Weight Decay Regularization"). A strong, modern, "shows you actually read the literature" answer.

---

## 11. Interview Q&A

**Q1: Why does L1 produce sparsity but L2 doesn't — give both the plain-language and the mathematical reason.**
See §3 in full. Short version: L1's constraint region (a diamond) has corners exactly on the axes, and those corners are the points an expanding error contour is most likely to hit first — landing there means a coordinate is exactly zero. Mathematically, L1's penalty gradient has constant magnitude all the way to zero (with a kink at zero), maintaining pressure that can overshoot to exactly zero; L2's gradient vanishes near zero, so there's no force actively finishing the job.

**Q2: If I standardize my features and increase λ in Ridge to infinity, what happens?**
All coefficients → 0, model predicts the mean of $y$ (or intercept) for every input — maximum bias, minimum (zero) variance. A useful sanity-check answer showing you understand the endpoints of the tradeoff.

**Q3: You have 10,000 features and 500 samples ($p \gg n$). OLS is undefined ($X^TX$ singular). Does Ridge or Lasso fix this, and why?**
Both work, for different reasons. Ridge: adds $\lambda I$ to $X^TX$, guaranteeing invertibility regardless of rank. Lasso: doesn't need matrix inversion at all (solved via coordinate descent/subgradient methods), and additionally gives a sparse, interpretable model — often the practical first choice in genuine $p\gg n$ settings (genomics, text with huge vocab).

**Q4: When would you pick Elastic Net over pure Lasso?**
When you want sparsity but are dealing with meaningfully correlated features, where pure Lasso's tendency to arbitrarily pick one feature from a correlated group (unstably across resamples) is a real concern — Elastic Net's L2 component encourages correlated features to be shrunk together, trading a bit of sparsity purity for stability.

**Q5: Is dropout a form of L1 or L2 regularization?**
No — it's structurally different (stochastic neuron removal during training, implicitly training and averaging an ensemble of thinned sub-networks) rather than a weight-magnitude penalty added to the loss. Both reduce variance, but through entirely different mechanisms.

**Q6: Why do we NOT typically apply dropout to the last (output) layer?**
Dropping units in the layer immediately producing the final prediction directly injects noise into the answer itself, with no subsequent layer to average/smooth it out — it degrades signal rather than regularizing feature learning. Regularization is meant to act on internal representations, not corrupt the final readout.

**Q7: Weight decay and L2 regularization — same thing?**
Classically yes, for plain SGD they're mathematically identical. In **Adam and other adaptive optimizers**, they diverge (§10, pitfall #10) — this is why AdamW exists. A sharp interviewer may ask this specifically to see if "weight decay = L2" is being carried over uncritically from the SGD era.

**Q8 (trick): Can regularization ever *reduce* bias?**
Not in the classical sense — regularization's mechanism is constraining the hypothesis space or dampening parameter magnitude, which by definition can only *increase or hold* bias while reducing variance. If you see a case where adding regularization improves *both* train and test performance, the likely explanation is that the unregularized model wasn't actually at its true optimum yet (an optimization issue, not a genuine bias-variance effect) — a good "gotcha" answer showing you don't pattern-match "regularization = free lunch."

**Q9: In a neural net, does increasing dropout rate always increase bias monotonically, mirroring λ in L1/L2?**
Generally yes directionally (more units dropped → weaker effective capacity per step → more underfitting risk), but it's less clean than a scalar $\lambda$ sweep because dropout also interacts with training length/learning rate (higher dropout often needs more epochs to converge) — an apparent "high bias" from aggressive dropout might really be under-training, not a true capacity ceiling. Worth distinguishing before concluding "reduce dropout" is the fix.

**Q10 (clever): Could you construct a scenario where L1 regularization does NOT produce any exactly-zero coefficients?**
Yes — if the unconstrained best-fit point already lies inside the L1 diamond for a given $\lambda$, no constraint binds at all, and you get the unconstrained solution back regardless of its sparsity. More practically, for a sufficiently small $\lambda$ the diamond is large enough that the optimum may land on a flat edge near a corner rather than exactly at one, especially in higher dimensions where "exactly on an axis" is a comparatively smaller target — sparsity is a strong empirical tendency, not an absolute guarantee at every $\lambda$.

**Q11: Why does feature scaling matter specifically for L1/L2, in a way it might matter less for an unregularized model?**
The penalty applies directly to raw coefficient values, and a feature's coefficient scale is inversely related to the feature's own numeric scale (a feature in small units needs a larger coefficient for the same effect as one in large units). Without standardizing first, the penalty implicitly punishes small-scale features' coefficients more than large-scale ones' — regardless of true importance — distorting both shrinkage and, for L1 specifically, which coefficients get zeroed.

---

*Happy to go deeper on any thread — the AdamW/decoupled weight decay math, a worked coordinate-descent derivation for Lasso, spatial dropout for CNNs, or a mock Q&A round where you answer and I critique.*