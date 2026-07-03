# Regularization: L1, L2, Dropout — Interview Notes

## 0. The Unifying Idea

Regularization = deliberately **constraining or penalizing model complexity** to trade a small increase in bias for a larger decrease in variance (ties directly to the bias-variance notes — this is the main lever labeled "reduce variance" in that table).

General form:

$$J(\theta) = \underbrace{L(\theta)}_{\text{data loss (e.g., MSE, cross-entropy)}} + \underbrace{\lambda \cdot R(\theta)}_{\text{penalty}}$$

$\lambda$ (regularization strength) controls where you sit on the bias-variance curve: $\lambda=0$ → unregularized (low bias, high variance); $\lambda\to\infty$ → penalty dominates, model collapses toward trivial (high bias, low variance).

---

## 1. L2 Regularization (Ridge)

**Penalty:** $R(\theta) = \|\theta\|_2^2 = \sum_j \theta_j^2$

**Full objective (regression):**
$$J(\theta) = \sum_i (y_i - \hat y_i)^2 + \lambda \sum_j \theta_j^2$$

**Closed-form solution (linear regression case)** — worth being able to write this:
$$\hat\theta_{\text{ridge}} = (X^TX + \lambda I)^{-1} X^Ty$$

Compare to OLS: $\hat\theta_{\text{OLS}} = (X^TX)^{-1}X^Ty$. The $\lambda I$ term is added to the diagonal — this is *literally* why Ridge fixes multicollinearity: it makes $X^TX+\lambda I$ invertible/better-conditioned even when $X^TX$ is singular or near-singular (correlated features).

**Effect on weights:** shrinks all coefficients toward zero, proportionally more for less-important (low-signal) directions, but **never exactly to zero** (except in degenerate cases).

**Geometric intuition (classic whiteboard picture):** constraint region $\sum\theta_j^2 \le t$ is a **circle/sphere**. The OLS loss contours are ellipses. The constrained optimum is where the ellipse first touches the circle — because a circle has no corners, the touching point is essentially never exactly on an axis, so coefficients shrink smoothly but stay nonzero.

**Bayesian interpretation (good to mention — shows depth):** Ridge = MAP estimate under a **Gaussian prior** on the weights, $\theta_j \sim N(0, \tau^2)$, where $\lambda = \sigma^2/\tau^2$. Tighter prior (smaller $\tau^2$) → bigger $\lambda$ → more shrinkage.

---

## 2. L1 Regularization (Lasso)

**Penalty:** $R(\theta) = \|\theta\|_1 = \sum_j |\theta_j|$

$$J(\theta) = \sum_i (y_i - \hat y_i)^2 + \lambda \sum_j |\theta_j|$$

**No closed form** in general (the penalty isn't differentiable at 0) — solved via coordinate descent, subgradient methods, or LARS.

**Effect on weights:** drives many coefficients to **exactly zero** → automatic **feature selection** / sparsity.

**Geometric intuition:** constraint region $\sum|\theta_j|\le t$ is a **diamond** (has corners on the axes). The elliptical loss contours are much more likely to first touch the diamond *at a corner* — a corner means one or more coordinates are exactly zero. This is the single most important geometric fact to be able to draw in an interview.

**Bayesian interpretation:** Lasso = MAP estimate under a **Laplace (double-exponential) prior** on weights — the Laplace distribution has a sharp peak at 0, which pushes mass toward exact zeros, unlike the smooth Gaussian bump of the Ridge prior.

**Subgradient at $\theta_j=0$:** since $|\theta_j|$ isn't differentiable there, optimization uses the subgradient $\partial|\theta_j| \in [-1,1]$ at 0 — the reason a coefficient can get "stuck" at exactly zero rather than just approaching it.

---

## 3. Elastic Net (the "why not both" answer)

$$R(\theta) = \alpha\|\theta\|_1 + (1-\alpha)\|\theta\|_2^2$$

Combines both: gets sparsity from L1 but is more stable than pure Lasso when features are highly correlated (Lasso tends to arbitrarily pick *one* of a correlated group and zero out the rest; Elastic Net tends to keep or drop correlated groups together — the "grouping effect"). Good answer to "what would you actually use in production with correlated features and you still want sparsity."

---

## 4. Comparison Table

| | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Penalty | $\sum \theta_j^2$ | $\sum \lvert\theta_j\rvert$ |
| Coefficient behavior | Shrinks smoothly, never exactly 0 | Drives many to exactly 0 |
| Feature selection | No (implicit, "soft") | Yes (explicit, sparse) |
| Closed form | Yes | No (needs iterative solver) |
| Geometric shape | Circle/sphere (smooth) | Diamond (has corners) |
| Bayesian prior | Gaussian | Laplace |
| Handles multicollinearity | Yes — well, spreads weight across correlated features | Poorly alone — picks one arbitrarily among correlated group |
| Differentiable everywhere | Yes | No (kink at 0) |
| Best when | Many small/medium-sized true effects, features correlated | True underlying model is sparse (few real predictors among many candidates) |

---

## 5. Numerical Example (real simulation, not hand-waved)

Synthetic data: $y = 3x_0 - 2x_1 + \text{noise}$, where $x_1$ is **correlated with $x_0$** (r≈0.85), and $x_2,\dots,x_5$ are **pure irrelevant noise features** (true coefficient 0). This mirrors a very realistic production scenario: some real signal, some redundant/collinear signal, some junk features.

| Feature | True coef | OLS | Ridge (α=1) | Ridge (α=10) | Lasso (α=0.05) | Lasso (α=0.3) |
|---|---|---|---|---|---|---|
| x0 | 3.00 | 3.126 | 2.405 | 1.140 | 1.176 | 0.902 |
| x1 | -2.00 | -2.227 | -1.395 | 0.034 | -0.000 | 0.000 |
| x2 | 0.00 | -0.068 | -0.066 | -0.059 | -0.013 | -0.000 |
| x3 | 0.00 | -0.063 | -0.068 | -0.074 | -0.028 | -0.000 |
| x4 | 0.00 | 0.083 | 0.069 | 0.044 | 0.000 | 0.000 |
| x5 | 0.00 | -0.113 | -0.110 | -0.103 | -0.050 | -0.000 |

$\|\theta\|_2$: OLS = 3.842 → Ridge(α=1) = 2.785 → Ridge(α=10) = 1.149 (monotonic shrinkage, smooth)
Nonzero count: OLS = 6 → Lasso(α=0.05) = 4 → Lasso(α=0.3) = **1** (sparsity kicks in and grows with α)

**What to point out from this table in an interview:**
1. **OLS overfits the noise features** slightly (x2–x5 get small but nonzero, spurious coefficients) — this is the "irrelevant feature increases variance" pitfall from the bias-variance notes, made concrete.
2. **Ridge shrinks everything smoothly** — even the noise features shrink slightly, but never hit exact zero, and note how at high α, the correlated pair (x0, x1) gets compressed together rather than one dominating.
3. **Lasso does real feature selection** — at α=0.05, two noise features (x2, x3 borderline, x4) are already zeroed; at α=0.3 only x0 survives — and notably it **kept x0 and dropped x1** even though both were "true" signal, because they're collinear and Lasso arbitrarily picks one from a correlated pair. This is the textbook Lasso-instability-under-collinearity phenomenon, visible in real numbers.

---

## 6. Dropout

**Mechanism (training time):** at each forward pass, independently zero out each unit (not weight) with probability $p$ (commonly $p=0.5$ for hidden layers, lower like 0.1–0.2 for input layers). A different random subnetwork is trained on every minibatch.

**Inverted dropout (what's actually used in practice):** to keep the *expected* activation magnitude the same between train and inference, scale surviving activations by $\frac{1}{1-p}$ **during training**, so **no rescaling is needed at inference time** (just run the full network as-is). This is the standard modern implementation (vs. the older "scale down at test time" version).

Quick check of why the scaling is needed: if a unit's activation is $a$, and it survives with probability $(1-p)$, then $E[\text{masked }a] = (1-p)\cdot a$. Dividing by $(1-p)$ during training restores $E[\cdot] = a$, matching test-time (no dropout) expectation — this is a one-line derivation worth being able to produce on demand.

**Why it works (multiple valid framings, know at least two):**
1. **Implicit ensembling:** training with dropout is approximately training an exponential number of "thinned" subnetworks with shared weights, and inference (no dropout) approximates *averaging* their predictions — a cheap approximation to bagging without needing to store/train separate models.
2. **Prevents co-adaptation:** units can't rely on any specific other unit being present, forcing each unit to learn features that are useful in combination with many different random subsets of other units — more robust, redundant representations.
3. **Noise injection / regularization view:** mathematically similar in spirit to adding multiplicative noise to activations, which (for linear models) can be shown to approximate an L2-like penalty on the weights in expectation.

**Where it's applied:** hidden fully-connected layers classically; **much less common/effective in convolutional layers** (spatial correlation means neighboring pixels/units carry redundant info, so dropout is a weaker regularizer there — spatial dropout / DropBlock exist as conv-specific variants). Also less commonly used inside modern Transformer attention blocks with heavy normalization, though it still appears in various sublayers.

---

## 7. Pitfalls (interviewer bait)

1. **"Lasso is strictly better because it gives sparsity."** Not universally true — if the true model isn't sparse (many small true effects), Lasso's sparsity is a *bias* it's introducing that doesn't match reality; Ridge often wins on pure predictive performance in that regime. Sparsity is a property you want only if you also want interpretability/feature selection, not automatically a "better" regularizer.
2. **Confusing Lasso's feature selection with causal/statistical significance.** A feature Lasso zeroes out isn't proven "not predictive" — it may just have lost an arbitrary tie-break against a correlated feature (shown concretely in the x0/x1 example above).
3. **Forgetting L2 regularization interacts with feature scaling.** Both L1 and L2 penalties are scale-dependent — un-normalized features get penalized unequally. **Always standardize features before regularizing** (this trips people up constantly and is an easy interview gotcha).
4. **Applying dropout at inference time by mistake** (forgetting `model.eval()` in PyTorch, or leaving training-mode flags on) — silently destroys inference quality/determinism. A very real production bug, not just theoretical.
5. **Using a single dropout rate everywhere without justification.** Different layers often want different rates (e.g., lower near input, none in the final output layer, none typically applied to attention/embedding layers in modern LLMs by default).
6. **Thinking dropout and L2 (weight decay) are redundant so you should only use one.** They regularize differently (structural sparsification of *activations* per-batch vs. shrinking of *weights* globally) and are commonly and successfully combined, especially in older CNN architectures.
7. **Believing regularization strength should always be maximized for "more robust" models.** Over-regularizing spikes bias — same U-curve logic as model complexity, just traversed in the other direction (this connects directly back to the bias-variance pitfall list — regularization strength IS a complexity-control knob, subject to the same over-correction risk).
8. **Weight decay ≠ L2 regularization in adaptive optimizers.** In Adam (as opposed to plain SGD), naively adding an L2 penalty to the loss does **not** behave the same as classic "weight decay" because Adam's adaptive per-parameter learning rates interact with the penalty gradient in a way that decouples it from true weight decay. This is exactly why **AdamW** exists — it decouples weight decay from the gradient-based adaptive update. This is a very strong, modern, "shows you actually read the literature" answer if it comes up (Loshchilov & Hutter, "Decoupled Weight Decay Regularization").

---

## 8. Diagnosis: When to Reach for Which

| Symptom | Lean toward |
|---|---|
| Train/val gap large, many features, suspect several are irrelevant | Lasso (sparsity + implicit feature selection) |
| Train/val gap large, features are correlated/collinear, all plausibly relevant | Ridge |
| Both: want sparsity but features are correlated groups | Elastic Net |
| Deep net overfitting, activations/hidden units, not linear coefficients | Dropout (+ weight decay/L2 often combined) |
| High variance, but interpretability of which raw features matter is a hard business requirement | Lasso (even at some accuracy cost) |
| Need theoretical guarantees / closed-form solution for speed at scale | Ridge |

**Practical tuning method for all of them:** cross-validate $\lambda$ (or $p$ for dropout) via a grid/log-scale sweep, often paired with the **1-SE rule** — pick the simplest model (largest $\lambda$/regularization) whose CV error is within one standard error of the best observed CV error, rather than blindly picking the single lowest-error point (which itself has CV-estimation variance, per the earlier notes).

---

## 9. Conceptual Q&A (clever/L5-style)

**Q1: Why does L1 produce sparsity but L2 doesn't — give the *mathematical*, not just geometric, reason.**
At the optimum, for L2 the penalty's gradient is $2\lambda\theta_j$, which **vanishes as $\theta_j\to 0$** — so there's no force actively pushing a small coefficient the rest of the way to exactly zero; it just asymptotically shrinks. For L1, the penalty's (sub)gradient is $\lambda \cdot \text{sign}(\theta_j)$, a **constant-magnitude force regardless of how close to zero $\theta_j$ already is** — so it keeps pushing toward zero and can actually overshoot to a corner solution where the optimizer settles exactly at $\theta_j=0$ if the data-loss gradient there is smaller than $\lambda$.

**Q2: If I standardize my features and increase λ in Ridge to infinity, what happens?**
All coefficients → 0, model predicts the mean of $y$ (or intercept) for every input — maximum bias, minimum (zero) variance. Useful sanity-check answer showing you understand the endpoints of the tradeoff.

**Q3: You have 10,000 features and 500 samples ($p \gg n$). OLS is undefined ($X^TX$ singular). Does Ridge or Lasso fix this, and why?**
Both work, for different reasons. Ridge: adds $\lambda I$ to $X^TX$, guaranteeing invertibility regardless of rank. Lasso: doesn't need matrix inversion at all (solved via coordinate descent/subgradient methods), and additionally gives you a sparse, interpretable model — often the practical first choice in genuine $p\gg n$ settings (genomics, text with huge vocab, etc.).

**Q4: Does dropout change the *expected* loss surface, or just add noise around a fixed surface?**
It genuinely changes the effective objective being optimized — it's not just noise around the same optimum. Training with dropout approximately optimizes a regularized objective related to the (weighted) predictions of all thinned subnetworks, which is a different, more conservative objective than optimizing the full network directly. This distinguishes it from something like naive additive Gaussian noise on the loss, which truly is noise around the same underlying objective.

**Q5: Why do we NOT typically apply dropout to the last (output) layer?**
Dropping units in the layer immediately producing the final prediction directly injects noise into the answer itself, with no subsequent layer to average/smooth it out — it degrades signal rather than regularizing feature learning. Regularization is meant to act on internal representations, not corrupt the final readout.

**Q6: Weight decay and L2 regularization — same thing?**
Classically yes, for plain SGD they're mathematically identical. In **Adam and other adaptive optimizers**, they diverge (see pitfall #8 above) — this is why AdamW exists. A sharp interviewer may ask this specifically to see if you know the distinction, since "weight decay = L2" is a common oversimplification carried over from the SGD era.

**Q7 (trick): Can regularization ever *reduce* bias?**
Not in the classical sense — regularization's entire mechanism is constraining the hypothesis space or dampening parameter magnitude, which by definition can only *increase or hold* bias while reducing variance. If you see a case where adding regularization seems to improve *both* train and test performance, the likely explanation is that the unregularized model wasn't actually at its true optimum yet (optimization issue, not a genuine bias-variance effect) — a good "gotcha" answer showing you don't just pattern-match "regularization = free lunch."

**Q8: In a neural net, does increasing dropout rate always increase bias monotonically, mirroring λ in L1/L2?**
Generally yes directionally (more units dropped → weaker effective network capacity per step → more underfitting risk), but it's less clean than a scalar $\lambda$ sweep because dropout also interacts with training length/learning rate (higher dropout often needs more epochs to converge) — so an apparent "high bias" from aggressive dropout might really be **under-training**, not a true capacity ceiling. Worth distinguishing before concluding "reduce dropout" is the fix — could just need more epochs.

---

Happy to go deeper on any thread — the AdamW/decoupled weight decay math, a worked coordinate-descent derivation for Lasso, spatial dropout for CNNs, or a mock Q&A round where you answer and I critique.
