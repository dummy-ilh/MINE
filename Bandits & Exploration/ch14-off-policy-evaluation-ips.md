# Chapter 14 — Off-Policy Evaluation: Importance Sampling

*(Same slower, simpler style — plain language first, light on notation.)*

---

## 14.1 The problem this whole chapter solves, in plain words

Here's a very common real situation: you have a bandit policy currently running in production (call it the **old policy**, or $\pi_0$ — "policy" just means "the rule that decides which arm to show"). You've come up with a **new policy** ($\pi_1$) that you think might be better — maybe a new version of LinUCB, maybe totally different logic.

**You want to know: would $\pi_1$ actually perform better than $\pi_0$, without actually turning it on and risking hurting real users if you're wrong.**

The obvious "safe" option is to run a live A/B test — show $\pi_1$ to some real traffic and measure. But that costs real traffic, real time, and real risk. **Off-policy evaluation (OPE)** asks a more ambitious question: **can we estimate how well $\pi_1$ would have performed, using only the data we already logged while $\pi_0$ was running — with no new live traffic at all?**

This is genuinely one of the most practically important ideas in this entire course — it's the difference between "test every new idea live, one at a time, slowly" and "evaluate dozens of candidate ideas overnight using data you already have."

---

## 14.2 Why this is hard: you only saw what the old policy showed

Here's the core obstacle, and it's just the bandit-feedback idea from Chapter 1, showing up again in a new form: **for every logged data point, you only know what happened for the arm that $\pi_0$ actually chose.** You have no idea what would have happened if $\pi_1$ had been running instead and had picked a *different* arm for that same user.

**A simple example to hold onto**: say $\pi_0$ showed User X the comedy show, and User X clicked. Now suppose your new policy $\pi_1$, looking at that same User X, would have shown the documentary instead. Did you get a click or not? **You have no idea — you never actually showed User X the documentary, so there's no logged outcome for that choice.** This missing-information problem is often called the **counterfactual problem** (we flagged this exact word back in Chapter 1, Section 1.8, as a preview) — you're trying to reason about "what would have happened," which you fundamentally didn't observe.

---

## 14.3 The key trick: only count the rounds where the two policies agree — but weight them cleverly

Here's the beautifully simple starting idea behind **Inverse Propensity Scoring (IPS)**, the first and most fundamental off-policy evaluation method:

**Only use the logged rounds where the new policy $\pi_1$ would have picked the *same* arm that $\pi_0$ actually picked.** On those specific rounds, you *do* actually know the outcome — because that's exactly the arm that got shown and logged. Throw away (or rather, count as zero contribution) the rounds where they'd have disagreed, since you genuinely have no data there.

But there's a subtlety: if you just average the reward over the "agreement" rounds, you get a **biased** answer — because some arms were shown more often than others by $\pi_0$, so the "agreement" rounds aren't a representative, fair sample of what $\pi_1$ would actually do overall. IPS fixes this with a weighting trick.

---

## 14.4 Propensity scores: "how likely was the old policy to have made this choice?"

For every logged round, define the **propensity score**: the probability that $\pi_0$ (the old, logging policy) would choose the specific arm it actually chose, given that round's context. Call this $p_0$ — just a number between 0 and 1, e.g., "$\pi_0$ had a 20% chance of showing the comedy show to this particular user, and it happened to do so."

**This should feel very familiar** — it's the exact same idea as EXP3's importance weighting from Chapter 8 (Section 8.4): there, we divided the observed reward by "how likely we were to have pulled this arm," to correct for the fact that we don't sample every arm equally. IPS is doing precisely the same correction, just now applied to a full logged dataset instead of one online round at a time.

---

## 14.5 The IPS formula, built up piece by piece

For one single logged round, define the **IPS estimate of reward** as:

$$\text{IPS estimate for this round} = \frac{\text{(1 if } \pi_1 \text{ would pick the same arm } \pi_0 \text{ did, else 0)} \times \text{(observed reward)}}{p_0}$$

In plain words: **"if the new policy would have agreed with what actually happened, count the observed reward — but divide by how likely that specific outcome was to have been logged in the first place, to correct for the fact that some arms got shown more than others."**

**Why dividing by $p_0$ fixes the bias**: an arm that $\pi_0$ rarely showed (small $p_0$) produces very little logged data — so on the rare rounds where you *do* get a data point for it, that data point needs to "count for more" to fairly represent how much that arm actually would have mattered, on average, across all the times $\pi_1$ would have wanted to pick it. Dividing by a small $p_0$ inflates that round's contribution — exactly mirroring EXP3's same trick from Chapter 8.

**The overall IPS estimate for $\pi_1$'s expected performance** is just the **average** of this per-round quantity across all your logged data:

$$\text{IPS estimate} = \frac{1}{n}\sum_{\text{all logged rounds}} \frac{\mathbb{1}[\pi_1 \text{ agrees}] \times \text{reward}}{p_0}$$

(The $\mathbb{1}[\cdot]$ symbol just means "1 if true, 0 if false" — a standard shorthand for "count it or don't.")

---

## 14.6 A very simple worked example

Say we logged 5 rounds under $\pi_0$. Here's what happened:

| Round | Arm shown by $\pi_0$ | $p_0$ (probability $\pi_0$ picked this arm) | Reward observed | Would $\pi_1$ pick the same arm? |
|---|---|---|---|---|
| 1 | Comedy | 0.5 | 1 (click) | Yes |
| 2 | Documentary | 0.2 | 0 | No |
| 3 | Comedy | 0.5 | 0 | Yes |
| 4 | Action | 0.3 | 1 (click) | No |
| 5 | Documentary | 0.2 | 1 (click) | Yes |

Now compute the IPS contribution for each round:

- Round 1: agrees, reward=1, $p_0=0.5$ → $1 \times 1 / 0.5 = 2.0$
- Round 2: disagrees → contributes $0$
- Round 3: agrees, reward=0, $p_0=0.5$ → $1 \times 0 / 0.5 = 0.0$
- Round 4: disagrees → contributes $0$
- Round 5: agrees, reward=1, $p_0=0.2$ → $1 \times 1 / 0.2 = 5.0$

$$\text{IPS estimate} = \frac{2.0 + 0 + 0.0 + 0 + 5.0}{5} = \frac{7.0}{5} = 1.4$$

**Notice something that should immediately jump out**: this "estimate" of $1.4$ is bigger than 1, even though every individual reward was just 0 or 1! This looks strange, but it's a completely normal and expected feature of IPS — Round 5's small $p_0 = 0.2$ caused a big division, inflating that single lucky agreement into a large contribution ($5.0$) that dominates the average. **This single example is actually the perfect illustration of IPS's core, well-known weakness — high variance — which is exactly what Section 14.7 explains next, and exactly what Chapter 15 is built to fix.**

---

## 14.7 The core weakness: high variance from rare-arm agreements

As the worked example just showed directly: whenever the "agreement" happens to occur on a round where $p_0$ was small (an arm the old policy rarely picked), dividing by that small $p_0$ creates a **huge** contribution from a **single data point** — and single data points swinging the whole average wildly is the textbook definition of **high variance**. Your overall IPS estimate can bounce around a lot from one logged dataset to another, even if the underlying true performance of $\pi_1$ is stable — this makes IPS estimates **unbiased in principle** (correct *on average*, over many repeated hypothetical datasets) but often **unreliable in any one specific practical dataset**, especially when your new policy $\pi_1$ often wants to pick arms that the old policy $\pi_0$ rarely picked.

**Simple intuition for why this matters practically**: if your new policy $\pi_1$ is *very* different from the old policy $\pi_0$ (agrees on very few rounds, and especially agrees mostly on rounds the old policy considered unlikely), your IPS estimate will be based on very little effective data and can be wildly noisy — even actively misleading. This single fact — "IPS gets unreliable when the new and old policies disagree a lot" — is one of the most important practical takeaways in all of Phase 3, and sets up exactly why Chapter 15's variance-reduction techniques (Doubly Robust estimation, self-normalization, clipping) are needed.

---

## 14.8 Production considerations (kept simple)

- **IPS-style off-policy evaluation is genuinely how many real companies vet new bandit/ranking policies before ever showing them to live users** — being able to reuse existing logged data to safely pre-screen many candidate policies, rather than needing a fresh live A/B test for each one, is a huge practical time and risk savings.
- **The "agrees vs. disagrees" idea generalizes beyond exact-match** — for continuous/contextual policies, you'd more precisely compare full probability distributions over arms (how likely was each policy to pick each arm, for this context) rather than a simple yes/no "same arm" check — the simplified yes/no version here is meant to build the core intuition clearly, and the full formula generalizes naturally.
- **Knowing the propensity scores $p_0$ requires the old policy to be genuinely randomized** (or at least have known, loggable probabilities) — if $\pi_0$ was a purely deterministic policy (like plain UCB1, which always makes the same choice given the same history — recall Chapter 4, Section 4.6), you simply don't have the probability information IPS needs, and off-policy evaluation becomes much harder. This is a real, concrete production reason companies sometimes prefer randomized policies (like Thompson Sampling) even when a deterministic policy might have slightly better online-serving regret — the randomization pays for itself later, at evaluation time.

---

## 14.9 Interview traps (kept simple)

- **Not immediately connecting IPS to EXP3's importance weighting from Chapter 8.** This is the exact same trick, and interviewers who've asked about both topics in one conversation are specifically listening for this connection.
- **Presenting IPS as unconditionally "the right tool" without mentioning its variance problem.** A strong answer always pairs "here's what IPS does" with "and here's its well-known weakness (high variance when policies disagree a lot), which is why in practice people usually reach for variance-reduced versions."
- **Forgetting that IPS fundamentally requires a *randomized* logging policy with known probabilities.** If asked "could you evaluate a new policy against logs from a purely deterministic old policy using IPS," the correct answer is essentially no — not without those known probabilities, which a deterministic policy doesn't naturally provide.

---

## 14.10 L5-vs-L6 differentiating talking points (kept simple)

- **L5 bar**: correctly explain the goal of off-policy evaluation, correctly describe IPS's "divide by propensity score" mechanism, and know that it's unbiased but can be high-variance.
- **L6 bar**:
  - Walks through a worked example like Section 14.6, and specifically points out — unprompted — the moment where a small $p_0$ produces a suspiciously large single-round contribution, using that as a live illustration of the variance problem rather than an abstract warning.
  - Explicitly connects IPS to EXP3's importance weighting (Chapter 8) as literally the same underlying technique, applied to logged offline data instead of live sequential rounds.
  - Raises the "you need a randomized logging policy with known probabilities" requirement unprompted, and connects it to a real production tradeoff (favoring Thompson-Sampling-style randomized policies partly *because* they make future off-policy evaluation possible) — a genuinely sophisticated, systems-level point.

---

## 14.11 Comprehension checks — plain words, minimal formulas

1. In one sentence, what question is off-policy evaluation trying to answer, and why can't you just directly observe the answer from your logged data?
2. What is a propensity score, in plain words?
3. Walk through, in your own words, why dividing by a small propensity score can create a single data point that dominates the whole IPS estimate.
4. Why does IPS require the old (logging) policy to have been randomized, with known probabilities, rather than fully deterministic?
5. In one sentence, what's the core practical weakness of IPS that motivates the methods covered in the next chapter?

---

*Next: Chapter 15 — Variance-Reduced Off-Policy Estimators, where we fix exactly the high-variance problem demonstrated in Section 14.6 — covering Doubly Robust estimation, Self-Normalized IPS, and clipping.*
