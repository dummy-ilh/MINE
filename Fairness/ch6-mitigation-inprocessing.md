# Chapter 6 — Mitigation: In-processing

Chapter 5 changed the *data*. This chapter changes the *training process itself* — the model architecture stays the same, but the objective it's optimized against now explicitly accounts for fairness, not just accuracy. In-processing techniques generally give you finer control over the fairness/accuracy tradeoff than pre-processing does, at the cost of more complex training.

## 6.1 The general idea: add a fairness term to the loss

The simplest version of in-processing is conceptually just what you already know from regularization: instead of only minimizing prediction error, minimize prediction error **plus** a penalty term that grows when a fairness metric (from Chapter 3) is violated.

**L_total = L_prediction + λ · L_fairness**

- L_prediction: your normal loss (cross-entropy, etc.)
- L_fairness: some differentiable proxy for a fairness gap — e.g., a penalty on the difference in average predicted score between groups (a soft proxy for demographic parity), or a penalty on the TPR/FPR gap between groups (a soft proxy for equalized odds)
- λ: a hyperparameter controlling how much you're willing to sacrifice prediction accuracy for fairness — this single number *is* the fairness/accuracy tradeoff dial that Chapter 8 discusses in depth

This is exactly the same pattern as L2 regularization (L_total = L_prediction + λ·‖w‖²) — you're already comfortable with this shape of objective from your optimization prep, just with a different penalty term.

**Why a fairness metric needs to be made "differentiable" at all:** TPR and FPR, as defined in Chapter 2, involve a hard threshold (Ŷ = 1 if S ≥ threshold, else 0) — and hard thresholds have zero gradient almost everywhere, so you can't directly backpropagate through them. In practice, the fairness penalty is usually computed on the raw score S (e.g., penalize the difference in mean score between groups, or use a smooth/sigmoid approximation of the threshold) so that gradients can flow.

## 6.2 Adversarial debiasing — the two-network setup

This is the technique most likely to come up by name in an interview, so it's worth understanding the mechanics in detail, not just the one-line summary.

**The setup:** two neural networks, trained against each other.

```
   Input features x
         │
         ▼
   ┌─────────────┐
   │  Predictor  │──── predicts Ŷ (the actual task, e.g., "will repay")
   └─────────────┘
         │
         │ Predictor's output (or an internal representation)
         ▼
   ┌─────────────┐
   │  Adversary  │──── tries to predict A (the protected attribute)
   └─────────────┘       *from the Predictor's output*
```

- The **Predictor** is trained normally to do the actual task well (minimize its own prediction loss on Y).
- The **Adversary** takes the Predictor's output (or an internal hidden layer) as its *input*, and tries to guess the protected attribute A from it.
- Here's the key trick: the Predictor is trained to do two things at once — get better at predicting Y, **and get worse at letting the Adversary predict A.**

**Why this works, intuitively:** if the Adversary *can* successfully guess someone's group from the Predictor's output, that means the Predictor's output still encodes group information — i.e., the model's decisions are still entangled with the protected attribute (directly or through proxies). By training the Predictor to actively *defeat* the Adversary, you're forcing the Predictor's output to become statistically independent of A — which is precisely the condition demographic parity requires (recall from Chapter 3: P(Ŷ=1|A=a) equal across groups is exactly saying "Ŷ carries no information about A").

**The training objective, a bit more formally:**

- Predictor's objective: minimize L_Y(Ŷ, Y) − λ · L_A(Â, A) — minimize its own task loss, while *maximizing* the adversary's loss (hence the minus sign) on predicting A.
- Adversary's objective: minimize L_A(Â, A) — get as good as possible at predicting A from whatever the Predictor gives it.

These two networks are trained in alternation (similar in spirit to how a GAN's generator and discriminator are trained in alternation, if you've encountered GANs) — take a step improving the Adversary, then a step improving the Predictor against the now-improved Adversary, and repeat. Over training, this pushes the Predictor toward a representation the Adversary genuinely cannot decode group membership from.

**What this technique targets specifically:** adversarial debiasing (as described above) most directly targets demographic-parity-style independence between Ŷ and A. Variants can target equalized odds instead by letting the adversary see Y as well as Ŷ — i.e., train the adversary to predict A from (Ŷ, Y) rather than Ŷ alone, which pushes the Predictor toward equal error rates rather than equal raw approval rates.

## 6.3 Fairness-constrained optimization (the "reductions" approach)

**The idea (Agarwal et al., 2018), at a conceptual level:** rather than adding a soft penalty term (6.1) or an adversarial game (6.2), treat fairness as a **hard constraint** and reduce the fairness-constrained learning problem to a sequence of ordinary (cost-sensitive) classification problems that you already know how to solve.

This connects directly to your Lagrange multipliers chapter: recall that a constrained optimization problem —

minimize L_prediction(θ) subject to (fairness gap) ≤ ε

— can be converted into an unconstrained problem using a Lagrangian:

minimize_θ maximize_{λ≥0} [ L_prediction(θ) + λ · (fairness gap(θ) − ε) ]

The "reductions" approach essentially operationalizes this: it alternates between (a) updating the model θ to minimize the Lagrangian for the *current* λ (which turns into a standard weighted classification problem — reweight examples based on the current λ, similar in spirit to Chapter 5's re-weighting, but now the weights are being *learned* through the optimization rather than computed once up front from the data), and (b) updating λ to increase pressure on whichever fairness constraint is currently most violated. This is exactly the same primal-dual / KKT structure as the SVM dual problem from your optimization prep — the fairness constraint plays the role the margin constraints played there.

**Why this approach is attractive:** it gives an explicit, tunable guarantee on the fairness gap (you set ε directly, rather than tuning an indirect penalty weight λ and hoping it lands where you want), and it can target any of the group fairness metrics from Chapter 3 by swapping in the corresponding constraint definition.

## 6.4 Tradeoffs of in-processing techniques generally

**Advantages over pre-processing (Chapter 5):**
- Finer, more direct control over the fairness/accuracy tradeoff (a tunable λ or an explicit ε, rather than hoping a reweighted dataset happens to produce the fairness property you want).
- Can target metrics that are about the *model's errors* (equalized odds, equal opportunity) more directly, since the objective sees Y, Ŷ, and A together during training — pre-processing only ever touches the data before the model has made any predictions at all.

**Disadvantages:**
- More complex training pipelines — adversarial debiasing in particular is known to be less stable to train than a standard single-objective model (the same instability concerns from your optimization/training-stability chapter apply: alternating min-max training can oscillate or fail to converge cleanly).
- The resulting model is tied to the specific λ or ε chosen at training time — if the desired fairness/accuracy tradeoff changes later (e.g., a new regulation, a new business requirement), you generally have to retrain, whereas post-processing (Chapter 7) lets you adjust after training with much less cost.
- Requires access to the protected attribute A during training, which is not always available or legally straightforward to collect and use, even for a good-faith fairness purpose.

---

**Next: Chapter 7 — Mitigation: Post-processing**, the third and final mitigation stage — adjusting a model's decisions *after* training, without retraining anything, including the legally sensitive question of using the protected attribute directly at inference time.
