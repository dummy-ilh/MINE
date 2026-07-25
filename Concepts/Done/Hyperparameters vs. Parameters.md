# Hyperparameters vs. Parameters, and Hyperparameter Tuning — Interview Notes

## 1. Parameters vs. Hyperparameters — The Core Distinction

**Parameters** are values **learned from data** during training — they're what the optimization algorithm (gradient descent, closed-form solve, etc.) adjusts to minimize the loss function. You never set them by hand; the fitting process determines them.
- Examples: weights and biases in a neural network, coefficients $\beta$ in linear/logistic regression, split thresholds and leaf values in a decision tree, support vectors and their coefficients in an SVM, cluster centroids in k-means (after fitting).

**Hyperparameters** are values **set before training begins**, chosen by the practitioner (or by a search procedure) rather than learned from the training loss directly — they control *how* the learning process happens, not what it converges to for a fixed configuration.
- Examples: learning rate, number of trees / max depth in a tree ensemble, regularization strength $\lambda$, number of layers/units in a neural net, $k$ in k-NN or k-means, batch size, number of training epochs, dropout rate, kernel choice/kernel parameters in an SVM.

**Why the distinction matters operationally (the actual interview-relevant point, not just definitional trivia):** you cannot select hyperparameters by minimizing training loss the same way you fit parameters — a sufficiently flexible model with the "right" hyperparameters (e.g., unlimited tree depth, zero regularization) will always achieve lower (even zero) training loss than a more constrained one, so training loss is a monotonically misleading signal for hyperparameter choice. Hyperparameters must be evaluated on a validation set (or via cross-validation) that they were *not* fit to directly — this is precisely why the train/val/test split exists as a three-way split rather than two (see the foundations notes): train fits parameters, validation selects hyperparameters, test gives the final unbiased read.

**A genuinely tricky edge case worth naming (shows precision):** the line isn't always crisp. Some values are learned by an *outer* optimization loop but not by the primary gradient-based training loop — e.g., the number of boosting rounds selected via early stopping *is* technically chosen using validation performance during a form of automated search, blurring "hyperparameter tuning" and "training" into one loop. Similarly, in Bayesian modeling, "hyperparameters of a prior" are a different, older usage of the same word (the parameters of a *hyperprior*, one level up from the model's own parameters) — worth disambiguating if the interviewer's phrasing is ambiguous, since it's a genuinely overloaded term across ML sub-communities.

## 2. Why Hyperparameter Tuning Is Hard

- **The search space is often high-dimensional and mixed-type** — some hyperparameters are continuous (learning rate, regularization strength), some integer (depth, number of layers), some categorical (optimizer choice, activation function) — a single unified search strategy has to handle all three.
- **Evaluating one point in the space is expensive** — each "try" requires training (part of) a model, which for large models can take hours to weeks, ruling out naive exhaustive search at scale.
- **The objective (validation performance) is noisy**, not a smooth deterministic function of the hyperparameters — different random seeds, different data shuffles, and stochastic optimization all inject variance into a single evaluation (directly connects to "CV score has variance" from the bias-variance notes).
- **Interactions between hyperparameters** — the best learning rate depends on the batch size; the best regularization strength depends on model capacity; you can't always tune hyperparameters independently one at a time and expect to land on the joint optimum (a common naive mistake — see pitfalls).

## 3. Grid Search

**Mechanism:** define a discrete set of candidate values for each hyperparameter, then exhaustively evaluate **every combination** (the Cartesian product) via cross-validation, picking the combination with the best validation score.

**Pros:**
- Simple, deterministic, fully reproducible, embarrassingly parallel (every combination is independent).
- Guaranteed to find the best point *within the specified grid* — no randomness in what gets tried.
- Easy to reason about and explain to non-ML stakeholders.

**Cons:**
- **Curse of dimensionality:** cost grows exponentially with the number of hyperparameters — 5 values × 5 hyperparameters = 3,125 combinations, each requiring a full (cross-validated) training run. Rapidly becomes computationally infeasible beyond ~3-4 hyperparameters.
- **Wastes budget on unimportant dimensions.** If one hyperparameter barely affects performance and another is critical, grid search still spends equal resolution on both — it has no way to know in advance which dimensions matter more (this is the central argument for random search, §4).
- **Resolution is fixed and arbitrary** — the true optimum may fall between grid points, and there's no principled way to pick grid spacing without prior knowledge of the loss surface.

## 4. Random Search

**Mechanism:** instead of a fixed grid, define a *distribution* over each hyperparameter (e.g., log-uniform for learning rate, uniform for dropout), and draw a fixed budget of $N$ random combinations, evaluating each via cross-validation.

**Why it beats grid search at equal budget (the classic, citable result — Bergstra & Bengio, 2012):** if only a few hyperparameters actually matter for a given problem (very common in practice — most real loss surfaces are dominated by 1-2 "important" dimensions with the rest being nearly flat), grid search wastes most of its trials on redundant combinations along the unimportant axes, while random search's random draws naturally explore *more distinct values* along every axis, including the important ones, for the same total budget. Concretely: with a $k$-dimensional grid of resolution $m$ per axis, grid search only ever tries $m$ distinct values along any single axis; random search with the same total budget $N = m^k$ trials tries $N$ (up to) distinct values along every axis — a far denser exploration of each individually important dimension.

**Pros:**
- Better coverage of important dimensions at equal compute budget, especially as dimensionality grows.
- Easy to control the budget directly (just set $N$) rather than being locked into a combinatorial grid size.
- Still embarrassingly parallel, still simple to implement and reason about.

**Cons:**
- Still doesn't use information from *previous* trials to inform *future* trials — every draw is independent, so it can waste evaluations in clearly bad regions of the space that earlier trials already ruled out. This is exactly the gap Bayesian optimization fills (§5).
- No guarantee of finding the true optimum even with a large budget — it's a sampling procedure, not an optimization procedure.

## 5. Bayesian Optimization

**Mechanism:** treat the (expensive-to-evaluate, noisy) validation-performance-as-a-function-of-hyperparameters as an unknown black-box function, and build a cheap **surrogate probabilistic model** of it (classically a **Gaussian Process**, though tree-based surrogates like in TPE/SMAC are also common) from the trials evaluated so far. Use an **acquisition function** (e.g., Expected Improvement, Upper Confidence Bound) applied to the surrogate to decide which point in hyperparameter space to try *next* — explicitly balancing exploring uncertain regions against exploiting regions the surrogate believes are promising. After each real evaluation, update the surrogate and repeat.

**Why this is smarter than grid/random search:** every prior trial's result actively informs where to look next, rather than every draw being independent — this means Bayesian optimization typically finds a good configuration in **far fewer total evaluations** than grid or random search, which matters enormously when each evaluation is a multi-hour (or multi-day) training run.

**Key components worth being able to name precisely:**
- **Surrogate model:** a Gaussian Process is classic because it naturally provides both a mean prediction *and* a calibrated uncertainty estimate at every untried point — exactly what the acquisition function needs to balance exploration/exploitation. Tree-structured Parzen Estimators (TPE, used in Hyperopt/Optuna) are a common practical alternative that scale better to higher dimensions and mixed continuous/categorical/conditional spaces than GPs do.
- **Acquisition function:** the rule that converts "surrogate's current belief" into "which point to try next." **Expected Improvement (EI)** is the most common — it picks the point maximizing the expected amount by which the new evaluation would beat the current best observed value, naturally weighting both a good predicted mean *and* high uncertainty (since uncertain regions have more room for a pleasant surprise).
- **Exploration-exploitation tradeoff** — directly analogous to the same tradeoff in RL (§1 of the foundations notes) — an acquisition function that only exploits (always samples where the surrogate's mean is best) gets stuck in a local optimum; one that only explores (always samples where uncertainty is highest) never converges. Good acquisition functions balance both, often via an explicit tunable parameter.

**Pros:**
- Sample-efficient — finds good hyperparameters in far fewer trials, critical when each trial is very expensive (large models, limited compute budget).
- Naturally handles noisy objectives (GP surrogate models observation noise directly rather than treating each evaluation as ground truth).

**Cons:**
- **Inherently sequential** by construction (each new point depends on all prior results) — much harder to parallelize than grid/random search, though modern implementations support batched/asynchronous variants that partially mitigate this at some cost to sample efficiency.
- **The surrogate model itself has overhead** and can struggle in very high-dimensional spaces (classic GPs scale poorly, roughly cubically, with the number of observations) — usually most effective for a moderate number of hyperparameters (rough rule of thumb: tens, not hundreds).
- More complex to implement/reason about and debug than grid/random search — more moving parts (surrogate choice, acquisition function choice, kernel choice for GPs) that themselves have hyperparameters ("hyperparameters of your hyperparameter tuner" is a genuine, only-half-joking practical annoyance).
- Categorical/conditional hyperparameters (e.g., "if optimizer=Adam, tune beta1/beta2; if optimizer=SGD, tune momentum instead") are more naturally handled by tree-based surrogates (TPE) than by classic GP-based approaches.

## 6. Comparison Table

| | Grid Search | Random Search | Bayesian Optimization |
|---|---|---|---|
| Uses past trial results to pick next trial? | No | No | **Yes** |
| Parallelizable? | Fully (embarrassingly parallel) | Fully (embarrassingly parallel) | Inherently sequential (batched variants exist but reduce efficiency) |
| Sample efficiency (good result per evaluation) | Low | Moderate | **High** |
| Scales well to many hyperparameters? | Poor (exponential blowup) | Better than grid, degrades gradually | Poor beyond moderate dimensionality (surrogate cost) |
| Handles noisy objective well? | No explicit handling | No explicit handling | Yes (GP models noise natively) |
| Implementation/conceptual complexity | Lowest | Low | Highest |
| Best use case | Few (≤3) hyperparameters, cheap evaluations, want reproducibility/simplicity | Moderate dimensionality, cheap-ish evaluations, want a strong simple default | Expensive evaluations (large models, limited compute), moderate dimensionality, sample efficiency is the priority |

## 7. Other Approaches Worth Naming (Interviewers Sometimes Probe Breadth Here)

- **Successive Halving / Hyperband:** allocate a small budget (e.g., few epochs) to many configurations, discard the worst-performing half (or more), and reallocate the freed budget to the survivors, repeating until one/few configurations remain with a full budget. Exploits the fact that bad configurations are often identifiable early, without needing to fully train every candidate — a resource-allocation-first approach that's complementary to (and often combined with) Bayesian optimization (e.g., **BOHB** = Bayesian Optimization + Hyperband).
- **Population-Based Training (PBT):** trains a population of models in parallel, periodically "exploiting" (copying weights from better-performing members) and "exploring" (perturbing hyperparameters) — hyperparameters can change *during* training rather than being fixed upfront, useful for schedules (e.g., learning-rate schedules) that genuinely should vary over the course of training rather than being static.
- **Gradient-based hyperparameter optimization:** for some differentiable hyperparameters (rare, more research-flavored), you can compute gradients of validation loss with respect to the hyperparameter itself (e.g., via implicit differentiation) — not widely used in standard industry practice but worth a one-line mention for depth.
- **Manual/expert-guided search:** still extremely common in practice — an experienced practitioner's informed starting point plus a small local grid/random search around it often beats a naive from-scratch large-scale search, especially when compute is limited and strong priors exist from similar past problems.

## 8. Practical Considerations (What Actually Matters in Production Tuning)

**Log-scale search for scale-sensitive hyperparameters:** learning rate, regularization strength ($\lambda$), and similar hyperparameters that span orders of magnitude should be searched on a **log-uniform** scale, not linear — uniformly sampling learning rate between 0.0001 and 1 on a *linear* scale wastes almost all draws in the 0.1–1 range and almost never samples the 0.0001–0.001 range, even though the latter is often where the actually-useful values live.

**Nested cross-validation to avoid tuning-induced optimism:** if you both tune hyperparameters *and* want an unbiased final performance estimate, tuning on the same CV folds used for the final reported number leaks a small optimistic bias (the reported score benefits from having been selected, across many trials, specifically because it did well on those folds) — nested CV (outer loop for unbiased estimate, inner loop for tuning) avoids this, as covered in the foundations notes §4.

**Early stopping as an implicit, nearly-free hyperparameter search dimension:** rather than treating "number of epochs / boosting rounds" as a hyperparameter to grid-search explicitly, use validation-loss-triggered early stopping within a single training run — effectively searches that one dimension "for free" as a byproduct of a single training run instead of needing separate full runs at each candidate epoch count.

**Warm-starting / transfer of tuning knowledge across related problems:** if you've tuned a similar model on a similar dataset before, initializing the search (grid center, random search distribution, or Bayesian optimization's prior) around those known-good regions dramatically cuts the effective search cost versus starting from an uninformative default range every time.

**Budget-aware method selection is itself a judgment call worth stating explicitly:** cheap-to-train models with few hyperparameters → grid or random search is entirely adequate and not worth the added complexity of Bayesian optimization; expensive-to-train models (large neural nets, long training runs) with a moderate number of hyperparameters → Bayesian optimization's sample efficiency earns its complexity; very large hyperparameter spaces with cheap partial-training signal available → Hyperband/successive-halving-style methods.

## 9. Pitfalls / Trick Angles

1. **Tuning hyperparameters against the test set instead of a validation set (or CV).** Directly reintroduces the leakage/optimism problem from the foundations notes — the test set must be touched exactly once, at the very end, regardless of which tuning method is used.

2. **Assuming Bayesian optimization is "always better."** It has real overhead (sequential nature, surrogate maintenance, more moving parts) that isn't worth it for cheap-to-evaluate problems with few hyperparameters — random search is often the pragmatically correct choice, not a "worse" fallback, when evaluations are fast and the space is small.

3. **Linear-scale search on hyperparameters that should be log-scale.** As covered in §8 — a very common, easy-to-avoid mistake that silently wastes most of a search budget in an uninteresting region of the space.

4. **Tuning hyperparameters one-at-a-time (coordinate-wise) and assuming this finds the joint optimum.** Hyperparameters interact (§2) — the best regularization strength for one learning rate may not be the best for another. Coordinate-wise tuning can get stuck far from the true joint optimum; joint search (grid over combinations, random joint draws, or Bayesian optimization over the full joint space) is needed to actually respect interactions, though coordinate-wise tuning is sometimes used pragmatically as a cheap approximation when the full joint search is infeasible.

5. **Confusing "more hyperparameter search" with "better model."** Extensive tuning on a small validation set can itself overfit *to that validation set* (the tuning procedure effectively has many "degrees of freedom" to exploit validation noise, especially with many trials on a small validation set) — this is a real, if second-order, leakage-adjacent risk, and part of why nested CV / a genuinely held-out final test set matters even when you're confident your validation methodology is otherwise clean.

6. **Ignoring the noise/variance of the validation metric itself when comparing close configurations.** If two hyperparameter configurations differ by 0.1% on a validation metric that has 0.5% run-to-run variance (different seeds), treating the "winner" as genuinely better is chasing noise — repeat key comparisons across multiple seeds before concluding one configuration is truly superior, directly echoing the "CV variance" pitfall from the bias-variance notes.

7. **Not budgeting compute for re-validating the winning configuration.** Whatever method you use, the single "best" trial found during search is itself somewhat lucky (it won among many noisy comparisons) — a final confirmation run (or several, across seeds) on the chosen configuration, evaluated on genuinely fresh held-out data, is good practice before fully trusting the number that came out of the search procedure.

## 10. Interview Q&A

**Q1: What's the precise distinction between a parameter and a hyperparameter, and why can't you just add hyperparameters to the loss function and learn them the same way?**
Parameters are learned by minimizing training loss via the model's fitting procedure; hyperparameters control the fitting procedure or model capacity itself, and if you tried to learn them by minimizing training loss directly, you'd always drift toward the most flexible/least-regularized setting (e.g., zero regularization, unlimited depth) because that trivially minimizes training loss — training loss doesn't penalize overfitting, so it's not a valid objective for choosing capacity-controlling hyperparameters. They must be selected against a validation signal the model wasn't fit to directly.

**Q2: Why does random search outperform grid search at the same computational budget, in general?**
Because most real hyperparameter spaces have only a few dimensions that meaningfully affect performance — grid search spends equal resolution on every dimension regardless of importance, so at a fixed budget it only tries a handful of distinct values along any single (including the important) axis, while random search's independent random draws explore many more distinct values along every axis for the same total number of trials, giving denser coverage specifically along whichever dimensions turn out to matter.

**Q3: When would you choose grid search over Bayesian optimization, even though Bayesian optimization is more sample-efficient?**
When evaluations are cheap and the space is small (few hyperparameters, fast training) — Bayesian optimization's sequential nature and surrogate-model overhead aren't worth paying for when you could just run a full grid in parallel almost as fast, and grid search's simplicity/reproducibility/full parallelizability are genuine practical advantages that shouldn't be dismissed just because a fancier method exists.

**Q4: Explain the exploration-exploitation tradeoff as it appears in Bayesian optimization's acquisition function.**
The acquisition function decides where to sample next using the surrogate's predicted mean (favors points the surrogate currently believes are good — exploitation) and predicted uncertainty (favors points the surrogate is unsure about, which could turn out to be even better than currently known — exploration). Expected Improvement naturally balances both because a point can have high expected improvement either by having a good predicted mean *or* by having high uncertainty around a decent mean (more room to pleasantly surprise); pure exploitation risks getting stuck in a local optimum the surrogate is overconfident about, pure exploration never converges to refine a good region.

**Q5: Your validation accuracy for the best hyperparameter configuration found via extensive random search is 91.2%, versus 90.9% for the second-best. Should you confidently pick the 91.2% configuration?**
Not necessarily without more evidence — if the validation metric has meaningful run-to-run variance (different seeds, different folds), a 0.3% gap after searching over many trials could easily be noise rather than a genuine difference, especially since searching over *many* trials increases the chance that the observed "winner" simply got a lucky evaluation (a form of the same multiple-comparisons issue as picking the best of many noisy CV folds). Re-evaluate the top few candidates across multiple seeds/folds before trusting a small margin as real.

**Q6: What's Hyperband, and what problem does it solve that pure Bayesian optimization doesn't address as directly?**
Hyperband allocates a small training budget to many candidate configurations, aggressively discards the worst performers early, and reallocates freed budget to promising survivors — it exploits the fact that many configurations reveal themselves as bad well before full training completes, so you don't need to pay full training cost to rule most candidates out. Pure Bayesian optimization decides *which* configuration to try next, but by default still trains each one to some fixed budget/completion — Hyperband's resource-allocation strategy is a complementary axis (which is why they're often combined, e.g., BOHB), rather than a competing philosophy.

**Q7 (clever): Is choosing $k$ in k-fold cross-validation itself a hyperparameter? What about the random seed?**
In the loose everyday sense, yes — both are values set before the "learning" process rather than learned from data. But they're a genuinely different flavor from model hyperparameters like learning rate or tree depth: $k$ and the seed control the *evaluation procedure* itself rather than the model's capacity or fitting dynamics, so tuning them to maximize apparent validation performance would be a form of leakage/overfitting-to-the-evaluation-procedure, not legitimate hyperparameter search — you should pick $k$ and seeds for principled/practical reasons (data size, compute budget, reproducibility) rather than searching over them for the best-looking number.
