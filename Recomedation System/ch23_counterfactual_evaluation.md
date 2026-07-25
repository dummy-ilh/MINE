# Chapter 23: Counterfactual Evaluation & Off-Policy Learning (IPS, Doubly Robust)

## 1. Intuition

Chapter 22 introduced bandits as a way to *collect* data more wisely going forward. This chapter tackles a related but distinct problem: given data that was already collected under some **old policy** (the current production system, or a previous model), how do you honestly estimate how a **new policy** (a candidate replacement model) would have performed — without actually deploying it and running a live A/B test?

This matters enormously in practice: A/B tests are expensive (require real user traffic, take time, risk hurting real users if the new policy is bad) and you often want to screen many candidate policies offline before committing traffic to a live test. The naive approach — just evaluate the new policy's predictions against the logged data as if it were a normal supervised learning test set — is **subtly and seriously wrong**, and understanding exactly why is the core of this chapter.

## 2. Why Naive Offline Evaluation Fails

The logged data was generated under the **old policy's** action distribution — you only observe rewards for actions the old policy actually chose to take. If the new policy would recommend a *different* item than what was logged, you have **no observed reward** for what the new policy would have gotten — this is a missing-data problem, not a simple prediction-accuracy problem. Computing "accuracy" of the new policy's predictions against logged (context, old-policy-action, reward) triples silently and systematically favors any new policy that happens to agree with the old policy's choices, and gives you literally zero information about how the new policy performs when it disagrees — which is exactly the interesting, decision-relevant case.

This is the same fundamental issue as **exposure bias** (first raised in Ch. 6, echoed in Ch. 22) applied specifically to the evaluation problem rather than the training problem: absence of an observed reward for an action doesn't mean that action would have gotten zero reward, it means you simply don't know.

## 3. Inverse Propensity Scoring (IPS)

The standard fix: reweight the observed rewards by the **inverse probability that the old policy took the logged action**, correcting for the fact that some actions were more likely to be logged than others.

Let $\pi_0$ = old (logging) policy, $\pi_1$ = new policy being evaluated, $p_0(a|x)$ = probability the old policy would take action $a$ given context $x$ (the **propensity score**). For a logged example $(x_i, a_i, r_i)$:

$$\hat{V}_{IPS}(\pi_1) = \frac{1}{n}\sum_{i=1}^n \frac{\pi_1(a_i|x_i)}{p_0(a_i|x_i)} r_i$$

**Intuition for the weighting**: if the new policy $\pi_1$ would have taken action $a_i$ with high probability, but the old policy $p_0$ only rarely took that action (low propensity), that rare-but-informative observation gets **upweighted** — since it's a scarce data point that's especially informative about how $\pi_1$ would behave (because $\pi_1$ likes actions the old policy rarely tried). Conversely, if $\pi_1$ would rarely take action $a_i$, that observation contributes little to $\pi_1$'s value estimate regardless of its reward, since it's not representative of what $\pi_1$ would actually do.

This estimator is **unbiased** under a key assumption: every action with non-zero probability under $\pi_1$ must have had non-zero probability under $\pi_0$ too (called the **support/coverage assumption**) — if the old policy *never* tried some action that the new policy wants to take, IPS has literally no data to estimate that action's value from, and the estimator breaks down (typically manifesting as extremely high variance from tiny propensity scores in the denominator).

## 4. The Variance Problem, and Clipping

IPS is unbiased but can have **very high variance**, especially when propensity scores $p_0(a_i|x_i)$ are small (common for actions the old policy rarely took) — dividing by a small number amplifies both the reward's contribution and its noise. A common practical mitigation: **clip** the importance weight to a maximum value:

$$w_i = \min\left(\frac{\pi_1(a_i|x_i)}{p_0(a_i|x_i)}, M\right)$$

for some clipping threshold $M$ — this introduces a small, controlled amount of bias in exchange for a large reduction in variance, a standard bias-variance trade-off (directly analogous in spirit to Chapter 7's regularization/shrinkage discussion — trading a bit of correctness for a lot more reliability given limited data).

## 5. Doubly Robust Estimation

**Doubly Robust (DR)** estimators combine IPS with a direct reward-prediction model, gaining a genuinely valuable robustness property: the DR estimator is **unbiased if either** the propensity model $p_0$ **or** the reward model $\hat{r}(x,a)$ (a learned model predicting expected reward, i.e., a standard supervised model, foreshadowed by Chapters 5-15's whole toolkit) is correctly specified — you don't need both to be right, just one.

$$\hat{V}_{DR}(\pi_1) = \frac{1}{n}\sum_{i=1}^n\left[\hat{r}(x_i,\pi_1) + \frac{\pi_1(a_i|x_i)}{p_0(a_i|x_i)}\big(r_i - \hat{r}(x_i,a_i)\big)\right]$$

where $\hat{r}(x_i,\pi_1) = \sum_a \pi_1(a|x_i)\hat{r}(x_i,a)$ (the reward model's expected value under the new policy, computable directly since it doesn't depend on the logging policy at all). Read the formula as: start with the direct model-based estimate of $\pi_1$'s value (first term), then add an IPS-weighted **correction** for how wrong that model's predictions were on the actually-observed data (second term). If the reward model $\hat{r}$ is perfect, the correction term's expectation is zero and DR reduces to just the reliable direct-model estimate; if the reward model is bad but the propensity scores are accurate, the correction term does the heavy lifting and DR reduces to something behaving like plain IPS — hence "doubly robust," protected against either single point of failure.

## 6. Worked Numerical Example

Three logged examples, evaluating a new policy $\pi_1$ against old (logging) policy $\pi_0$:

| $i$ | $\pi_0(a_i|x_i)$ | $\pi_1(a_i|x_i)$ | $r_i$ |
|---|---|---|---|
| 1 | 0.5 | 0.8 | 1 |
| 2 | 0.2 | 0.1 | 0 |
| 3 | 0.1 | 0.6 | 1 |

**Plain IPS estimate:**
$$\hat{V}_{IPS} = \frac{1}{3}\left[\frac{0.8}{0.5}(1) + \frac{0.1}{0.2}(0) + \frac{0.6}{0.1}(1)\right] = \frac{1}{3}[1.6+0+6.0]=\frac{7.6}{3}=2.533$$

Notice example 3 contributes a weight of 6.0 — a huge amplification driven by the old policy's low propensity (0.1) for an action the new policy strongly favors (0.6) — this single data point is dominating the entire estimate, exactly the high-variance behavior flagged in Section 4. If example 3 had instead had reward 0 (easily could have, given it's a single noisy observation), the estimate would swing wildly: $\frac{1}{3}[1.6+0+0]=0.533$ — a nearly 5x difference in the final estimate driven by one coin-flip-like outcome, illustrating concretely why IPS alone is considered too high-variance to trust on small samples.

**With clipping** ($M=3$): example 3's weight of 6.0 gets clipped to 3.0:
$$\hat{V}_{IPS,clipped} = \frac{1}{3}[1.6+0+3.0(1)] = \frac{4.6}{3}=1.533$$

Lower estimate, but far less sensitive to that single example's exact reward value — a smaller swing if that reward had been 0 instead: $\frac{1}{3}[1.6+0+0]=0.533$ (this particular swing is unaffected by clipping since clipping only caps the reward=1 case, but in general clipping reduces the *maximum possible* influence of any single high-weight example, which is the point).

## 7. Production Considerations

- Off-policy evaluation (IPS/DR) is used specifically to **screen candidate policies offline** before committing to the expense and risk of a live A/B test (Ch. 24) — it doesn't replace A/B testing, but it lets teams reject clearly-bad candidates and prioritize the most promising ones for actual live testing, which is a genuine, valuable filtering step given how costly and slow live experiments are.
- Getting accurate propensity scores $p_0(a|x)$ requires the logging policy to actually be **stochastic** (have genuine randomness in its action selection, with known/computable probabilities) — a fully deterministic production system (always picks the single top-ranked item, no randomization at all) has degenerate propensities (each observed action has propensity 1, everything else 0), making principled off-policy evaluation of alternative policies essentially impossible. This is a concrete, often-overlooked reason production systems deliberately inject a small amount of randomization (sometimes framed as an extension of Chapter 22's exploration) specifically to enable trustworthy future counterfactual evaluation, not only to gather bandit-style feedback.
- Doubly robust estimation requires training a reward model $\hat{r}(x,a)$ specifically for this evaluation purpose — this is genuinely extra modeling infrastructure investment beyond the production ranking model itself, though it can often reuse similar features/architecture.

## 8. Interview Traps

- Evaluating a new candidate model by comparing its predictions to logged (context, old-policy-action, reward) data as if it were a standard supervised test set, without recognizing the missing-data/exposure-bias problem this creates — this is the single most important trap in this entire chapter, and the one interviewers most directly probe for.
- Describing IPS without acknowledging its high-variance failure mode, especially with small propensity scores — a genuinely important, commonly-tested practical caveat.
- Not knowing what "doubly robust" actually refers to (robustness to misspecification of *either* the propensity model *or* the reward model, not requiring both to be correct) — a common vague/wrong recollection of the term.
- Assuming production systems are always deterministic, and not recognizing that some deliberate randomization is often necessary specifically to make future off-policy evaluation possible at all (a subtle but real production requirement).

## 9. L5-Differentiating Talking Points

- Explain precisely *why* naive offline evaluation on logged data is wrong — the missing-data/exposure-bias framing, not just "you need special techniques" — since this precise articulation is what interviewers are actually listening for.
- Walk through the IPS variance problem concretely (as in Section 6) with a specific numerical illustration of how a single low-propensity, high-weight example can swing the entire estimate — showing genuine understanding of the failure mode, not just naming "high variance" abstractly.
- State the doubly-robust "protection from either failure" property precisely, and explain the formula's structure (direct estimate plus an IPS-weighted correction term) rather than treating DR as an unexplained black-box improvement over IPS.
- Note that production systems deliberately inject randomization partly *to enable future off-policy evaluation*, not only for bandit-style exploration (Ch. 22) — connecting this chapter's evaluation concern back to a concrete system design decision, showing the throughline across chapters.

## 10. Comprehension Check

1. Why is naively computing prediction accuracy against logged (context, action, reward) data the wrong way to evaluate a new candidate recommendation policy?
2. What does the propensity score represent, and what assumption is required for IPS to be unbiased?
3. Why can IPS have very high variance, and what's a standard practical mitigation?
4. What does "doubly robust" mean precisely, in terms of the two things the estimator can be robust to misspecification of?
5. Why might a production system deliberately inject randomization into its recommendation policy, beyond the bandit-exploration reasons covered in Chapter 22?
