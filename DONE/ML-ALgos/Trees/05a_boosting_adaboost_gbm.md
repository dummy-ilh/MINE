# Chapter 5a — Boosting: Foundations, AdaBoost, Gradient Boosting

Chapter 2.3 previewed the bagging/boosting contrast: bagging reduces variance via parallel averaging of high-variance learners; boosting reduces **bias** via sequential correction using high-bias ("weak") learners. This chapter makes that precise.

---

## 5.1 Sequential Error-Correction — Why This Is Bias Reduction, Not Variance Reduction

**The core loop, at a level of abstraction that covers both AdaBoost and Gradient Boosting:**

1. Start with a simple initial prediction (e.g., the majority class, or the mean target).
2. Fit a weak learner (commonly a shallow tree — a "stump," `max_depth=1` or 2) to whatever the *current ensemble is getting wrong*.
3. Add this new weak learner into the ensemble, typically scaled down by a small weight/learning rate.
4. Repeat — each new learner targets the errors that remain **after** all previous learners' contributions.

**Why is this bias reduction and not variance reduction, formally?** Recall the bias-variance decomposition (Ch.2.1): $\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma_\epsilon^2$. Bagging's averaging mechanism (Ch.2.2) provably leaves bias unchanged and only shrinks variance. Boosting does the opposite: each round is explicitly constructed to reduce the *systematic, repeatable* part of the current ensemble's error (what's wrong on **every** run, not what varies randomly run-to-run) — that systematic error is exactly what "bias" means in the decomposition. A single weak learner (e.g., a depth-1 stump) has very low variance (it's simple and stable across resamples) but high bias (it can only represent one threshold's worth of decision boundary, per Ch.1.7). Summing many such learners, each correcting the last one's *remaining bias*, drives the ensemble's overall bias down round by round — while variance, if anything, tends to creep up slightly as the ensemble grows more complex (more terms, more capacity to eventually start fitting noise), which is precisely why boosting (unlike bagging) needs explicit regularization (learning rate, early stopping, tree depth limits) to keep that creeping variance in check.

**Why must the base learner be weak (high-bias) rather than strong?** If round 1's learner already fits the training data closely (low bias, e.g. a deep unpruned tree), there's very little systematic error left for round 2 onward to correct — whatever error remains is mostly irreducible noise ($\sigma_\epsilon^2$), and boosting has no mechanism to distinguish "real remaining signal" from "noise" in what it's correcting. It will happily fit the noise too, and since boosting has no averaging mechanism to smooth that out (unlike bagging), this shows up directly as overfitting. Weak learners leave deliberately large, genuine bias on the table for later rounds to keep meaningfully reducing.

---

## 5.2 AdaBoost — Full Derivation

AdaBoost (Freund & Schapire, 1997) is the original, and conceptually simplest, boosting algorithm. Binary classification, labels $y_i \in \{-1,+1\}$.

**The algorithm:**

1. Initialize sample weights uniformly: $w_i^{(1)} = \frac{1}{n}$ for all $i=1,\dots,n$.
2. For $m=1$ to $M$:
   a. Fit a weak classifier $h_m(x)$ to the training data, **using weights $w_i^{(m)}$** — i.e., the weak learner's own fitting criterion (e.g., Gini impurity, Ch.1.2) is computed as a weighted average, so heavily-weighted (currently-hard-to-classify) samples matter more to this round's split search.
   b. Compute the weak learner's **weighted error rate**:
   $$
   \varepsilon_m = \frac{\sum_{i=1}^n w_i^{(m)} \cdot \mathbb{1}(y_i \neq h_m(x_i))}{\sum_{i=1}^n w_i^{(m)}}
   $$
   c. Compute this round's model weight:
   $$
   \alpha_m = \frac{1}{2}\ln\left(\frac{1-\varepsilon_m}{\varepsilon_m}\right)
   $$
   d. Update sample weights — **increase** weight on misclassified samples, **decrease** on correctly-classified ones:
   $$
   w_i^{(m+1)} = w_i^{(m)} \cdot \exp\left(-\alpha_m \cdot y_i \cdot h_m(x_i)\right)
   $$
   then renormalize so $\sum_i w_i^{(m+1)} = 1$.
3. Final prediction: a weighted vote,
   $$
   H(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m h_m(x)\right)
   $$

**Where does $\alpha_m = \frac12\ln\frac{1-\varepsilon_m}{\varepsilon_m}$ come from — full derivation, no steps skipped:**

AdaBoost is provably equivalent to fitting an additive model $\sum_m \alpha_m h_m(x)$ to minimize the **exponential loss**:
$$
L(y, F(x)) = \exp(-y \cdot F(x)), \quad \text{where } F(x) = \sum_m \alpha_m h_m(x)
$$

At round $m$, we already have $F_{m-1}(x) = \sum_{k<m}\alpha_k h_k(x)$ fixed, and want to choose the new $(\alpha_m, h_m)$ to minimize total exponential loss of $F_{m-1}+\alpha_m h_m$:

$$
\sum_i \exp\left(-y_i\left(F_{m-1}(x_i)+\alpha_m h_m(x_i)\right)\right) = \sum_i \underbrace{\exp(-y_i F_{m-1}(x_i))}_{=\,w_i^{(m)} \text{ (up to a constant)}} \cdot \exp(-\alpha_m y_i h_m(x_i))
$$

This factorization is *exactly* why $w_i^{(m)} = \exp(-y_iF_{m-1}(x_i))$ is the natural definition of "current sample weight" — it's literally the accumulated exponential loss from all previous rounds, which is why misclassified samples (large $|F_{m-1}(x_i)|$ with wrong sign) end up with large weight automatically.

Split the sum by whether $h_m$ got sample $i$ right ($y_ih_m(x_i)=+1$) or wrong ($y_ih_m(x_i)=-1$):

$$
= e^{-\alpha_m}\sum_{i: \text{correct}} w_i^{(m)} + e^{\alpha_m}\sum_{i:\text{wrong}} w_i^{(m)}
$$

Let $W = \sum_i w_i^{(m)}$ (total weight) and $\varepsilon_m = \frac{\sum_{i:\text{wrong}} w_i^{(m)}}{W}$ (weighted error rate, matching step 2b). Then $\sum_{\text{correct}} w_i^{(m)} = W(1-\varepsilon_m)$ and $\sum_{\text{wrong}} w_i^{(m)} = W\varepsilon_m$. Substituting:

$$
= W\left[e^{-\alpha_m}(1-\varepsilon_m) + e^{\alpha_m}\varepsilon_m\right]
$$

**Now minimize over $\alpha_m$** — take the derivative with respect to $\alpha_m$ and set to zero:
$$
\frac{d}{d\alpha_m}\left[e^{-\alpha_m}(1-\varepsilon_m)+e^{\alpha_m}\varepsilon_m\right] = -e^{-\alpha_m}(1-\varepsilon_m) + e^{\alpha_m}\varepsilon_m = 0
$$
$$
e^{\alpha_m}\varepsilon_m = e^{-\alpha_m}(1-\varepsilon_m)
$$
$$
e^{2\alpha_m} = \frac{1-\varepsilon_m}{\varepsilon_m}
$$
$$
2\alpha_m = \ln\left(\frac{1-\varepsilon_m}{\varepsilon_m}\right)
$$
$$
\boxed{\alpha_m = \frac{1}{2}\ln\left(\frac{1-\varepsilon_m}{\varepsilon_m}\right)}
$$

This confirms step 2c is exactly the closed-form minimizer of exponential loss for this round, not an arbitrary heuristic choice.

**Why this formula's shape makes sense:**
- If $\varepsilon_m = 0.5$ (the weak learner is no better than a coin flip): $\alpha_m = \frac12\ln(1) = 0$. A useless learner gets **zero** weight in the final vote — exactly right.
- If $\varepsilon_m \to 0$ (a near-perfect weak learner): $\alpha_m \to +\infty$. A near-perfect round dominates the final vote.
- If $\varepsilon_m > 0.5$ (worse than random): $\alpha_m < 0$ — the learner's vote is *inverted* and still contributes usefully (this is a real edge case AdaBoost explicitly handles via the sign of $\alpha_m$).

**Worked numerical, full round:** $n=5$ samples, all start at $w_i^{(1)}=0.2$. Suppose weak learner $h_1$ misclassifies samples 2 and 4.

$$
\varepsilon_1 = \frac{0.2+0.2}{0.2\times5} = \frac{0.4}{1.0}=0.4
$$
$$
\alpha_1 = \frac12\ln\left(\frac{1-0.4}{0.4}\right)=\frac12\ln(1.5)=\frac12(0.4055)=0.2027
$$

Weight update (step 2d): correctly classified samples (1,3,5) get $w_i^{(2)} = 0.2\times e^{-0.2027} = 0.2\times0.8166=0.1633$. Misclassified samples (2,4) get $w_i^{(2)}=0.2\times e^{0.2027}=0.2\times1.2247=0.2449$.

Renormalize: sum $=3(0.1633)+2(0.2449)=0.4900+0.4899=0.9799$. Normalized weights: correct samples $\to 0.1633/0.9799=0.1667$; misclassified samples $\to 0.2449/0.9799=0.2500$.

Sanity check: misclassified samples' weight rose from 0.2 → 0.25 (a 25% relative increase); correctly-classified samples' weight fell from 0.2 → 0.1667 (a 16.7% relative decrease) — exactly the intended behavior: round 2's weak learner will now be trained with samples 2 and 4 weighted 1.5× as heavily as the others ($0.25/0.1667=1.5$), pushing it to prioritize getting those specific points right.

---

## 5.3 Gradient Boosting — Functional Gradient Descent

AdaBoost is elegant but tied specifically to exponential loss and classification. **Gradient Boosting (Friedman, 2001)** generalizes the same "sequentially correct the current ensemble's errors" idea to **arbitrary differentiable loss functions**, framed as gradient descent — not in parameter space, but in **function space**.

**The reframing — why "functional" gradient descent:** In ordinary gradient descent, you have parameters $\theta$ and update $\theta \leftarrow \theta - \eta\nabla_\theta L$. Gradient Boosting instead treats the *entire prediction function* $F$ as the thing being optimized, updating it as $F \leftarrow F + \eta \cdot (\text{something like } -\nabla_F L)$ — except you can't literally take a derivative "with respect to a function" and get another function back in closed form. The trick: evaluate the gradient of the loss **only at the $n$ training points** (a finite-dimensional vector, one gradient value per sample), then **fit a base learner (a regression tree) to approximate that gradient vector** as a function of $x$ — this converts the intractable "gradient step in function space" into a concrete, fittable regression problem.

**The algorithm, step by step:**

1. Initialize $F_0(x) = \arg\min_c \sum_i L(y_i, c)$ — the constant that minimizes total loss (for squared error, this is just $\bar y$, matching Ch.1.2.4's regression-tree leaf logic).
2. For $m=1$ to $M$:
   a. Compute the **pseudo-residuals** — the negative gradient of the loss with respect to the current prediction, evaluated at each training point:
   $$
   r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}
   $$
   b. Fit a regression tree $h_m(x)$ to predict these pseudo-residuals $r_i^{(m)}$ (as the target), using ordinary regression-tree fitting (Ch.1.2.4's MSE-splitting machinery — this is why Gradient Boosting's base learner is always a *regression* tree, even when the overall task is classification).
   c. Update: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$, where $\eta$ (the **learning rate**, `learning_rate` in sklearn) shrinks each round's contribution.
3. Final prediction: $F_M(x)$ (passed through a link function — e.g. sigmoid — for classification probabilities).

**Why does fitting a tree to the negative gradient make sense — the squared-error special case, worked with full derivation:**

For squared error loss, $L(y,F(x)) = \frac12(y-F(x))^2$:
$$
\frac{\partial L}{\partial F(x_i)} = \frac12 \times 2(F(x_i)-y_i) \times 1 = F(x_i)-y_i
$$
$$
r_i^{(m)} = -\left[F_{m-1}(x_i)-y_i\right] = y_i - F_{m-1}(x_i)
$$

**The negative gradient is exactly the ordinary residual** $y_i - F_{m-1}(x_i)$ — this is why Gradient Boosting is often introduced as "fitting trees to residuals," which is the correct intuition specifically for squared-error loss and becomes a generalization ("fit to the gradient") for any other loss.

**Worked numerical, one full round of squared-error Gradient Boosting:** $n=4$ samples, $y = [10, 15, 22, 28]$.

$F_0(x) = \bar y = (10+15+22+28)/4 = 75/4=18.75$ (Step 1, constant initialization).

Pseudo-residuals (Step 2a, squared-error case = ordinary residuals): $r^{(1)} = [10-18.75,\ 15-18.75,\ 22-18.75,\ 28-18.75] = [-8.75,\ -3.75,\ 3.25,\ 9.25]$

Suppose a shallow regression tree $h_1(x)$, fit to these residuals as targets (Step 2b), predicts $[-6.0,\ -6.0,\ 6.0,\ 6.0]$ (it found one split separating the first two samples from the last two).

With learning rate $\eta=0.1$ (Step 2c): $F_1(x) = F_0(x) + 0.1\times h_1(x)$
$$
F_1 = [18.75+0.1(-6.0),\ 18.75+0.1(-6.0),\ 18.75+0.1(6.0),\ 18.75+0.1(6.0)] = [18.15,\ 18.15,\ 19.35,\ 19.35]
$$

Compare to targets $[10,15,22,28]$: predictions moved *toward* the targets (18.75→18.15 toward 10; 18.75→19.35 toward 22 and 28) but only by a small step — exactly the intended "shrunk" incremental correction. Round 2 would compute fresh residuals against $F_1$ (e.g., new residual for sample 1: $10-18.15=-8.15$, still large — round 2's tree will keep correcting it) and repeat.

**Why the learning rate $\eta$, and why is it necessary here specifically (tying back to 5.1)?** Without shrinkage ($\eta=1$), each new tree would attempt to fully close the residual gap in one step, which — combined with boosting's lack of any averaging/variance-smoothing mechanism (Ch.2.2 doesn't apply here; these trees are trained sequentially and summed, not averaged over independent resamples) — makes the ensemble track the training data's noise very tightly, very fast. A small $\eta$ (commonly 0.01–0.3) forces many small, cautious steps, giving the (typically cross-validated or early-stopped) choice of $M$ finer-grained control over how far into "fitting noise" territory the ensemble is allowed to go — this is the direct mechanism behind the classic empirical finding (Friedman, 2001) that lower learning rate + more rounds, tuned together, reliably outperforms a high learning rate with few rounds, provided you can afford the extra compute.

---

## sklearn Parameters — `AdaBoostClassifier`/`Regressor` and `GradientBoostingClassifier`/`Regressor`

| Parameter | Class | What it controls | Notes |
|---|---|---|---|
| `estimator` | AdaBoost | Base weak learner | Default `DecisionTreeClassifier(max_depth=1)` — a "decision stump," matching 5.1's weak-learner requirement exactly |
| `n_estimators` | Both | Number of boosting rounds ($M$) | Unlike bagging/RF (Ch.3-4), **increasing this can eventually overfit** (5.1) — needs tuning via validation curve, not "set generously high" |
| `learning_rate` | Both | Shrinks each round's contribution | AdaBoost: multiplies $\alpha_m$ before use. Gradient Boosting: the $\eta$ in Section 5.3's Step 2c. Lower values need more `n_estimators` to compensate — classic trade-off pair |
| `loss` | GB | Loss function $L$ driving the pseudo-residual computation (5.3) | Regressor: `'squared_error'` (default, matches 5.3's worked example), `'absolute_error'`, `'huber'`, `'quantile'`. Classifier: `'log_loss'` (default — logistic/cross-entropy loss, the classification analogue of exponential loss) |
| `subsample` | GB | Fraction of training samples used to fit **each** round's tree (a stochastic variant — "Stochastic Gradient Boosting," also Friedman) | Default 1.0. Values <1.0 add a bagging-like row-randomization on top of boosting, which can reduce variance/overfitting somewhat, at the cost of some bias in the gradient estimate each round |
| `max_depth` | GB | Depth of each round's regression tree | Default 3 — deliberately shallow, per 5.1's weak-learner requirement (much shallower than typical standalone-tree or RF defaults) |
| `algorithm` (deprecated) / `SAMME` | AdaBoost | The specific multiclass extension formula | Newer sklearn versions use `SAMME` for both binary and multiclass, replacing the older `SAMME.R` |
| `validation_fraction`, `n_iter_no_change`, `tol` | GB | Early stopping — halt boosting rounds once validation loss stops improving | Directly operationalizes 5.1's "boosting needs explicit regularization" point |

**Why does `GradientBoostingClassifier` still fit *regression* trees internally even for a classification task?** Per 5.3's derivation, the pseudo-residuals $r_i^{(m)}$ are real-valued gradient values (not class labels), so the base learner at every round is solving a regression problem (predict this real number) regardless of what the overall task is — the classification behavior only re-enters at the very end, when $F_M(x)$ is passed through a link function (e.g. the sigmoid, for `log_loss`) to produce class probabilities.

---

## Quick Interview Q&A

**Q: "In AdaBoost, what happens to a weak learner with error rate exactly 0.5?"**
A: $\alpha_m = \frac12\ln(1) = 0$ (5.2's derivation) — it contributes nothing to the final weighted vote and its round effectively does not affect the ensemble's decision boundary, though its (unchanged, since $\alpha_m=0$ means the exponential weight-update term $e^{\mp\alpha_m}=1$ too) failure to shift sample weights means the next round sees the same weight distribution as this one.

**Q: "Why is Gradient Boosting described as 'gradient descent in function space' rather than just 'fitting to residuals'?"**
A: "Fitting to residuals" is only exactly correct for squared-error loss (5.3's derivation shows the negative gradient literally equals $y-F(x)$ in that specific case). For any other loss (log-loss, Huber, quantile), the pseudo-residual is the loss's negative gradient evaluated at each point — a different, loss-specific quantity that residuals-language doesn't generalize correctly to describe; "functional gradient descent" is the framing that correctly covers every loss function Section 5.3's algorithm supports.

**Q: "Would increasing `n_estimators` in `GradientBoostingClassifier` ever hurt performance, unlike in Random Forest?"**
A: Yes, and this is a genuine, frequently-tested contrast with Chapter 4's RF answer to the same question. Every additional boosting round is another opportunity to fit the current residual/pseudo-residual more closely (5.1) — with enough rounds, especially at a high learning rate, the ensemble eventually starts fitting noise in the residuals rather than remaining signal, directly increasing variance and test error even as training error keeps falling. This is exactly why `n_iter_no_change`/early stopping and learning-rate tuning are standard practice for boosting but not for bagging/Random Forest.

---

**Next up: Chapter 5b — XGBoost's regularized objective and second-order approximation, LightGBM's leaf-wise growth and histogram binning, and CatBoost's ordered boosting.**
