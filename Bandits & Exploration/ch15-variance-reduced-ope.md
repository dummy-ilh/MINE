# Chapter 15 — Variance-Reduced Off-Policy Estimators

*(Same slower, simpler style — plain language first, light on notation.)*

---

## 15.1 Where we left off

Chapter 14 built IPS and showed, very concretely (Section 14.6), that it can produce wild, high-variance estimates — one lucky agreement on a rarely-picked arm swung the whole answer far outside the possible range of individual rewards. This chapter covers three practical fixes, each solving the same underlying problem in a different way. You'll want to know all three by name and be able to explain, in plain words, what each one does differently.

---

## 15.2 Fix #1: Clipped / Truncated IPS

**The idea, in the simplest possible words**: cap how small $p_0$ (the propensity score) is allowed to be before you divide by it. If $p_0$ is smaller than some chosen floor (say, $0.05$), just use the floor value ($0.05$) in the division instead of the tiny real value.

Revisit our Chapter 14 worked example: Round 5 had $p_0 = 0.2$, contributing $1/0.2 = 5.0$. If we'd had an even rarer round with $p_0 = 0.02$, plain IPS would divide by $0.02$, producing a contribution of $50$ — wildly larger. **Clipping simply refuses to divide by anything smaller than the floor**, so that round's contribution gets capped, no matter how rare the arm was.

**The honest tradeoff**: clipping directly trades away some accuracy (the estimate is no longer perfectly unbiased — the math is slightly distorted for rare cases) in exchange for much more stability (the estimate can't swing nearly as wildly). This is a completely standard "a little bit of bias in exchange for a lot less variance" trade — one of the most common tradeoffs in all of applied statistics, and worth stating in exactly those words if asked.

---

## 15.3 Fix #2: Self-Normalized IPS (SNIPS)

**The idea, in plain words**: instead of just dividing by the *number of rounds* ($n$) like plain IPS did, divide by the **sum of all the weights actually used** in that specific calculation.

Recall plain IPS's formula from Chapter 14:

$$\text{IPS} = \frac{1}{n}\sum \frac{\mathbb{1}[\text{agree}] \times \text{reward}}{p_0}$$

**SNIPS** changes only the denominator:

$$\text{SNIPS} = \frac{\sum \frac{\mathbb{1}[\text{agree}] \times \text{reward}}{p_0}}{\sum \frac{\mathbb{1}[\text{agree}]}{p_0}}$$

In plain words: instead of dividing the weighted sum by a fixed number ($n$, the round count), SNIPS divides by the **total weight actually collected** — which naturally shrinks or grows the estimate back toward a sensible range, because you're now computing a genuine **weighted average** (weights and values normalized consistently together) rather than "a weighted sum, divided by something that has nothing to do with the weights."

### Redoing the Chapter 14 example with SNIPS

Recall the agreement rounds were 1, 3, and 5, with weights ($1/p_0$) of $2, 2, 5$ respectively, and weighted-reward contributions of $2.0, 0.0, 5.0$.

$$\text{SNIPS} = \frac{2.0 + 0.0 + 5.0}{2 + 2 + 5} = \frac{7.0}{9} = 0.778$$

Compare: plain IPS gave $1.4$ (impossible, above the max possible reward of 1). **SNIPS gives $0.778$ — a genuinely plausible number, safely between 0 and 1.** This is exactly the practical benefit SNIPS provides: by normalizing consistently, the estimate is automatically kept within a sensible range, and in practice, SNIPS tends to have meaningfully lower variance than plain IPS, at the cost of a small amount of bias (a similar "trade a little accuracy for a lot of stability" tradeoff as clipping, just achieved a different way).

---

## 15.4 Fix #3: Doubly Robust (DR) Estimation

This is the most conceptually rich of the three fixes, so let's build it up slowly, piece by piece.

**Step back and notice a wasted opportunity in plain IPS**: on rounds where $\pi_1$ (the new policy) *disagrees* with what actually happened, plain IPS just throws that round away entirely (contributes exactly 0). But you still logged that round's *context* — you know who the user was, even though you don't know what reward they'd have gotten from $\pi_1$'s preferred arm. **Doubly Robust estimation's key idea: don't waste that context. Use a separately-trained reward-prediction model to fill in a reasonable guess for what would have happened, on the rounds where you don't have real logged data for $\pi_1$'s choice.**

**The two ingredients, named plainly**:
1. A **reward model** — an ordinary supervised-learning model (think: the straight-line prediction model from Chapter 11, or something fancier) trained on all your logged data, that predicts "how good would this arm probably be for this context" for *any* arm, not just the one that was actually shown.
2. The **IPS correction term** from Chapter 14 — used specifically to correct for whatever bias/error the reward model gets wrong, on the rounds where you *do* have real logged data to check it against.

**Putting them together, in plain words**: for every round, start with the reward model's prediction for whatever arm $\pi_1$ would have picked (this gives you *something* useful on every single round, unlike plain IPS which gives you nothing on disagreement rounds) — then, **only on the rounds where you have real logged data for that exact choice**, add a correction term (very similar in spirit to the IPS weighting) that nudges the estimate to account for wherever the reward model's prediction was off from the real observed outcome.

**Why "doubly robust"**: this method has a genuinely elegant property — **it gives a correct (unbiased) answer if *either* of the two ingredients is good, even if the other one is bad.** If your reward model is a poor predictor but your propensity scores ($p_0$) are accurate, you still get an unbiased estimate (the IPS correction term cleans up the reward model's mistakes). If your reward model happens to be excellent but your propensity scores are slightly off, you still tend to get a good estimate (you're relying mostly on the good reward-model predictions, with only a small correction layered on). **You get two independent chances to be right, instead of needing one single ingredient to be perfect** — this is exactly what "doubly robust" refers to, and it's worth being able to state this property in exactly this plain form, since it's the single most-quoted fact about DR estimation in interviews.

---

## 15.5 A simple worked comparison to build intuition (not a full formal DR calculation)

Let's stay conceptual here rather than working through the full DR formula symbol-by-symbol (it combines pieces from both a trained model and the IPS correction, which makes a fully faithful hand-worked numeric example fairly involved) — what matters for interview purposes is the clear plain-language mechanism above, plus this comparison table:

| Method | Uses every logged round, or only agreement rounds? | Needs an accurate reward-prediction model? | Needs accurate propensity scores? | Bias/variance tradeoff |
|---|---|---|---|---|
| Plain IPS | Only agreement rounds | No | Yes | Unbiased, but often high variance |
| Clipped IPS | Only agreement rounds | No | Yes (but tolerant of very small ones) | Slightly biased, lower variance |
| SNIPS | Only agreement rounds | No | Yes | Slightly biased, lower variance |
| Doubly Robust | **Every round** (via the reward model) | Helps if accurate, but not required | Helps if accurate, but not required | Low variance, gets an unbiased answer if either ingredient is good |

**A clean, interview-ready one-liner**: *"Plain IPS only learns from the rounds where the new and old policy happened to agree, and pays for that narrowness with high variance. Doubly Robust fixes this by also using a trained reward model to make use of every logged round, while still keeping IPS's correction term as a safety net in case that reward model is wrong."*

---

## 15.6 Production considerations (kept simple)

- **Doubly Robust estimation is the most commonly used method in serious industrial off-policy evaluation pipelines**, precisely because of the "two independent chances to be right" property (Section 15.4) — it's a genuinely robust default choice when you're not fully sure how accurate either your reward model or your propensity scores will be, which is most of the time in practice.
- **Clipping is often applied together with the other methods, not instead of them** — e.g., "Doubly Robust with clipped propensity scores" is a common, sensible combination in real pipelines, rather than treating these three fixes as mutually exclusive alternatives.
- **The reward model used inside Doubly Robust is exactly the kind of model you'd build for LinUCB or Linear Thompson Sampling (Chapters 11–12)** — this is a nice practical synergy: if you're already building a contextual bandit with a reward-prediction component, you often already have (most of) what you need to also build a Doubly Robust off-policy evaluator for testing future candidate policies.

---

## 15.7 Interview traps (kept simple)

- **Describing SNIPS as "just IPS with extra normalization" without being able to say *why* that normalization helps.** The precise reason: dividing by the total collected weight (rather than a fixed round count) automatically keeps the final estimate within a sensible, bounded range — as shown very directly in the Section 15.3 recomputation.
- **Describing Doubly Robust as "just IPS plus a model" without mentioning the "two independent chances to be right" property.** This property is the entire reason DR is considered a genuinely different, stronger idea rather than just a minor tweak — leaving it out misses the main point of the method.
- **Thinking clipping "fixes" IPS's variance problem for free.** It's a real bias/variance tradeoff, not a free lunch — a strong answer always names both sides of that tradeoff.

---

## 15.8 L5-vs-L6 differentiating talking points (kept simple)

- **L5 bar**: can name and briefly describe all three fixes (clipping, SNIPS, Doubly Robust), and know that all three exist specifically to address IPS's variance problem from Chapter 14.
- **L6 bar**:
  - Recomputes the Chapter 14 worked example under SNIPS unprompted (as in Section 15.3) to concretely demonstrate the fix, rather than just asserting SNIPS is "more stable."
  - States the "doubly robust" name's meaning precisely — unbiased if *either* the reward model or the propensity scores are accurate — rather than a vague "it combines two methods" gloss.
  - Notes the practical synergy between building a contextual-bandit reward model (Chapters 11–12) and building a Doubly Robust evaluator (Section 15.6), showing genuine systems-level, cross-chapter thinking about how a real bandit pipeline would actually be built end to end.

---

## 15.9 Comprehension checks — plain words, minimal formulas

1. In one sentence, what does clipping do to fix IPS's variance problem, and what's the honest cost of doing so?
2. Redo the SNIPS calculation from Section 15.3 in your own words — why does dividing by the total collected weight (instead of the round count) produce a more sensible-looking number?
3. What are the two "ingredients" Doubly Robust estimation combines, and what specific job does each one do?
4. In plain words, what does "doubly robust" mean — what property does this method have that plain IPS doesn't?
5. Why might a company that's already built a LinUCB-style contextual bandit have a head start on building a Doubly Robust off-policy evaluator?

---

*Next: Chapter 16 — The Replay Method & Counterfactual Learning, where we cover a much simpler, extremely widely-used practical alternative to the IPS-family methods — and briefly touch on training a policy directly from logged data, rather than just evaluating one.*
