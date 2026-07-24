# Chapter 7: Objective Function Design

## 1. Why This Chapter Exists

Chapters 5–6 covered how to *evaluate* a model given business cost structure. This chapter covers the earlier, more powerful lever: **baking the cost structure directly into the loss function the model optimizes during training**, rather than only checking for it after the fact at evaluation/threshold time. Getting this right means the model itself, not just your threshold choice, is aimed at what actually matters.

## 2. Why Loss Function Choice Is a Formulation Decision, Not Just a Modeling Detail

A default loss (cross-entropy, MSE) implicitly assumes a specific, often wrong, cost structure: cross-entropy assumes all misclassifications of a given confidence are equally bad; MSE assumes symmetric, quadratic cost for over- vs. under-prediction. When the real business cost is asymmetric, using a symmetric default loss produces a model that's *systematically miscalibrated toward the wrong operating point*, and no amount of post-hoc threshold tuning (Chapter 5) fully compensates, because threshold tuning only reslices an already-learned score distribution — it can't fix a model whose internal representation never learned to distinguish the costly error direction from the cheap one in the first place.

## 3. Asymmetric Losses: A Full Worked Derivation

Recall Chapter 5's fraud cost structure: FN costs \$500, FP costs \$5 — a 100:1 asymmetry. Standard cross-entropy loss for a single example is:

$$
\mathcal{L}_{CE} = -\big[y \log(\hat p) + (1-y)\log(1-\hat p)\big]
$$

This penalizes a confident wrong prediction on a positive example ($y=1, \hat p \to 0$) exactly as harshly as a confident wrong prediction on a negative example ($y=0, \hat p \to 1$) — symmetric, but our cost structure is not.

**Weighted cross-entropy** directly injects the cost ratio:

$$
\mathcal{L}_{WCE} = -\big[w_1 \cdot y \log(\hat p) + w_0 \cdot (1-y)\log(1-\hat p)\big]
$$

Setting $w_1 = 500$ (or any value preserving the 100:1 ratio, e.g., $w_1=100, w_0=1$) directly tells the optimizer that errors on positive (fraud) examples are 100x costlier, so gradient updates push far harder to correct false negatives than false positives during training itself — not just at a post-hoc threshold.

**Numeric check**: for a single false-negative example ($y=1$, $\hat p = 0.1$), unweighted loss $= -\log(0.1) = 2.303$. Weighted ($w_1=100$): $100 \times 2.303 = 230.3$. For a false-positive example ($y=0$, $\hat p=0.9$), unweighted loss is the same $2.303$ by symmetry, but weighted ($w_0=1$) stays at $2.303$. The optimizer now sees a ~100x larger gradient signal from the false-negative-type error, exactly matching the real dollar cost ratio, and will adjust decision boundaries to reduce false negatives even at the cost of somewhat more false positives — which is the economically correct tradeoff given \$500 vs. \$5 costs.

## 4. Quantile/Asymmetric Regression Losses

For regression with asymmetric costs (e.g., underestimating delivery time is worse than overestimating it, or vice versa depending on the product), use the **pinball/quantile loss**:

$$
\mathcal{L}_\tau(y, \hat y) = \begin{cases} \tau (y - \hat y) & \text{if } y \ge \hat y \\ (1-\tau)(\hat y - y) & \text{if } y < \hat y \end{cases}
$$

**Worked example**: an ETA-prediction model where underestimating delivery time (telling a customer 20 min when it takes 35) is worse for trust than overestimating (telling them 40 when it takes 35). Set $\tau = 0.7$ (weighting underestimation more heavily). For true $y=35$, prediction $\hat y = 20$ (underestimate by 15): loss $= 0.7 \times 15 = 10.5$. For prediction $\hat y = 50$ (overestimate by 15, same absolute error): loss $= (1-0.7)\times 15 = 4.5$. Same magnitude of error, more than double the loss for the direction we've decided is more costly — this directly trains the model to bias its predictions toward the safer overestimate side, exactly encoding the business preference into the optimization target rather than relying on a post-hoc adjustment added after training.

## 5. Ranking Losses Revisited: Encoding Position-Weighted Cost

Chapter 5's NDCG discounts relevance by position ($\log_2(i+1)$) at *evaluation* time. **LambdaRank/LambdaMART** losses go further and inject that same position-sensitivity directly into training gradients — the gradient for swapping a pair of items is scaled by $|\Delta NDCG|$, the actual change in NDCG that swapping them would cause. This means the optimizer is pushed hardest to fix inversions that would hurt NDCG the most (typically inversions near the top of the ranked list, since that's where the position discount is steepest), rather than treating every pairwise inversion in the list as equally important regardless of position — directly mirroring the real cost of ranking errors (an inversion at position 1–2 matters far more to user experience than one at position 40–41).

## 6. Multi-Term Objectives: Weighted Combination Basics

Often you need a loss that balances multiple concerns even before you get to Chapter 8's fuller multi-objective treatment — e.g., a recommendation model whose primary loss is relevance, but which also needs a regularization-like penalty discouraging near-duplicate items in a list:

$$
\mathcal{L} = \mathcal{L}_{\text{relevance}} + \lambda \cdot \mathcal{L}_{\text{diversity penalty}}
$$

**Worked numeric sanity check for $\lambda$ selection**: suppose $\mathcal{L}_{\text{relevance}}$ typically sits around 0.8–1.2 in magnitude during training and $\mathcal{L}_{\text{diversity penalty}}$ sits around 8–12 (different scale due to how it's computed, e.g., summed pairwise similarity across a 10-item list rather than averaged). Setting $\lambda = 1$ would let the diversity term dominate the gradient by roughly 10x purely due to scale mismatch, not because diversity is actually 10x more important. A sensible starting point is $\lambda \approx 0.1$, chosen specifically to roughly equalize the two terms' typical magnitude contribution (0.8–1.2 vs. ~0.8–1.2 after scaling), then tuned from there based on the actual tradeoff you want (e.g., via a held-out validation sweep watching both relevance and diversity metrics simultaneously) — never picked as a round number without checking the relative scales first.

## 7. Production Considerations

- **Class weights (Section 3) and quantile parameters (Section 4) are themselves business parameters, not just modeling hyperparameters** — they should be owned and periodically re-derived jointly with the team that owns the actual dollar-cost estimates (fraud losses, customer friction costs), since if those costs shift, the loss weighting should shift with them.
- **Weighted losses can destabilize training if weights are extreme** (e.g., $w_1=500$ verbatim rather than a more moderate ratio-preserving value) — in practice, teams often use a more moderate weight (e.g., capped at 10–20x) combined with post-hoc threshold tuning (Chapter 5) to split the correction between the loss function and the decision threshold, rather than pushing the entire 100x asymmetry into the loss alone.
- **Multi-term objectives (Section 6) need their component losses monitored individually during training**, not just the combined scalar — a combined loss can look like it's converging nicely while one component (e.g., diversity) has actually stalled or regressed, hidden inside the sum.
- **Loss function choice affects what the model's raw scores mean**, which has downstream implications for calibration (Chapter 5) — a model trained with heavily asymmetric weighting will produce probability-like outputs that are no longer well-calibrated probabilities, and needs a separate calibration step if downstream consumers need true probabilities rather than just a good decision boundary.

## 8. Common Interview Traps

- **Only discussing threshold tuning (a Chapter 5 evaluation-time lever) when asked about cost asymmetry, and never mentioning that the loss function itself can and should encode the same asymmetry.** This is the single biggest gap this chapter targets — many candidates know about threshold tuning but stop there.
- **Proposing an extreme class weight (e.g., literally 500x) without acknowledging the training-stability tradeoff** from Section 7.
- **Combining multiple loss terms with an arbitrary $\lambda$ without checking relative scale** (Section 6) — a classic, easily-caught mistake.
- **Not recognizing that asymmetric-loss training changes score calibration**, and then being surprised when raw model outputs no longer behave like well-calibrated probabilities.

## 9. L5-Differentiating Talking Points

- When discussing cost asymmetry, explicitly distinguish "encode it in the loss function (Section 3)" from "encode it in the decision threshold (Chapter 5)" and note these are complementary, not mutually exclusive, levers — proposing to use both, and explaining the stability tradeoff for why you might not push 100% of the correction into the loss.
- Bring up quantile/pinball loss unprompted for any regression problem with a plausible asymmetric cost (delivery ETAs, capacity planning, inventory forecasting) rather than defaulting to MSE.
- Note the calibration side-effect of asymmetric-loss training (Section 7) as a downstream consideration for any consumer that needs true probabilities.
- For multi-term objectives, explicitly mention checking relative loss-term scales before picking a combination weight, rather than treating $\lambda$ as a free hyperparameter to be swept blindly.

## 10. Comprehension Checks

1. Using the weighted cross-entropy formula in Section 3, if the true cost ratio between FN and FP is 20:1 (not 100:1), what weight values $w_1, w_0$ would you propose, and compute the loss for a false negative example ($y=1,\hat p=0.2$) under this weighting.
2. For a demand-forecasting model where overestimating demand wastes inventory (moderate cost) but underestimating causes stockouts (severe cost, lost sales + customer trust), propose a quantile loss $\tau$ value and justify the direction of your choice using Section 4's reasoning.
3. Explain why pushing an extreme class weight (e.g., 500x) entirely into the loss function, rather than splitting the correction between loss weighting and threshold tuning, risks training instability. What symptom would you look for in a training run to detect this?
4. A team combines a relevance loss (typical magnitude ~1) and a fairness-penalty loss (typical magnitude ~50) with $\lambda = 1$. Diagnose the likely problem and propose a corrected approach using Section 6's reasoning.

---

*End of Chapter 7. Chapter 8 will cover multi-objective and constrained formulations — combining objectives like relevance, diversity, and revenue using weighted sums, Lagrangian/constrained framings, and Pareto-frontier tradeoffs.*
