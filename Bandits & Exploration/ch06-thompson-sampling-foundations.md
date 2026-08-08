# Chapter 6 — Thompson Sampling: Foundations

---

## 6.1 The core idea, before any formula

Every algorithm so far has produced a single **point estimate** or **upper bound** per arm, and then deterministically (UCB1) picked the max. Thompson Sampling takes a genuinely different philosophical approach: **maintain a full probability distribution over what each arm's true mean might be, and choose an arm by randomly sampling one guess from each arm's distribution, then picking whichever sampled guess is highest.**

This is sometimes called **probability matching**, and here's the intuition for why it's a good idea: if you're genuinely 70% confident arm A is the best arm and 30% confident arm B is actually the best (given your current data), a sensible strategy is to pull arm A roughly 70% of the time and arm B roughly 30% of the time — not "100% of the time, whichever currently has the higher point estimate" (that's greedy) and not "some fixed exploration rate regardless of how confident you are" (that's ε-greedy). Thompson Sampling achieves *exactly* this proportional behavior, automatically, as a side effect of the sampling procedure — no one has to explicitly compute "70%" anywhere; it falls out naturally from sampling from the posterior distributions.

This is also the direct payoff of Chapter 5's Bayes-UCB discussion: Thompson Sampling uses the **exact same Bayesian posterior machinery** — priors, likelihoods, posteriors — but instead of taking a fixed high quantile of the posterior (Bayes-UCB's approach), it takes a **random draw** from the posterior. Same ingredients, different final step.

---

## 6.2 Bayesian setup: priors, likelihoods, posteriors — a fast, precise refresher

Since this machinery underlies the entire rest of the chapter, let's be precise about the three pieces, using plain language first.

- **Prior**: your belief about an arm's true mean $\mu_i$ *before* seeing any data from it. In the total-ignorance case, this is often a "flat"/uniform belief — "any value between 0 and 1 is equally plausible."
- **Likelihood**: given a *hypothetical* true mean $\mu_i$, how probable is the data you actually observed? (E.g., "if this coin's true bias were 0.5, how likely was it to produce 7 heads out of 10 flips?")
- **Posterior**: your **updated** belief about $\mu_i$ *after* combining the prior with the observed data via Bayes' rule:

$$\text{posterior}(\mu_i \mid \text{data}) \propto \text{likelihood}(\text{data} \mid \mu_i) \times \text{prior}(\mu_i)$$

The posterior is itself a full probability distribution — not a single number — capturing both your best guess *and* your remaining uncertainty. This is the object Thompson Sampling draws from.

---

## 6.3 The Beta distribution: the right tool for Bernoulli rewards

For Bernoulli rewards (click/no-click — our running example's setting), the standard choice of prior is the **Beta distribution**, written $\text{Beta}(\alpha, \beta)$, with two parameters $\alpha$ and $\beta$ (both positive numbers). A few facts about the Beta distribution you need, stated plainly:

- It's a distribution over values in $[0, 1]$ — exactly the right range for a probability/rate like $\mu_i$.
- **Mean of $\text{Beta}(\alpha, \beta)$**: $\dfrac{\alpha}{\alpha + \beta}$
- **Interpretation of $\alpha$ and $\beta$**: think of $\alpha$ as "prior successes + 1" and $\beta$ as "prior failures + 1." $\text{Beta}(1,1)$ is the flat/uniform prior on $[0,1]$ (equivalent to "0 prior successes, 0 prior failures" — total ignorance). As $\alpha$ and $\beta$ grow, the distribution concentrates more tightly around $\alpha/(\alpha+\beta)$ — more data, less uncertainty, exactly the behavior we want.
- **Shape intuition**: $\text{Beta}(1,1)$ is flat (uniform). $\text{Beta}(2,2)$ is a gentle hump centered at 0.5. $\text{Beta}(50,50)$ is a *very* narrow, tall spike centered at 0.5 — reflecting strong confidence the true mean is near 0.5 after lots of balanced evidence. $\text{Beta}(90, 10)$ is a narrow spike near 0.9 — strong confidence in a high mean.

---

## 6.4 The magic of conjugacy: the Beta-Bernoulli update rule

Here is the single most important computational fact in this chapter, and it's what makes Thompson Sampling so cheap and elegant to implement: **if your prior on $\mu_i$ is $\text{Beta}(\alpha, \beta)$, and you observe a new Bernoulli reward $X$, the posterior is *also* a Beta distribution** — specifically:

$$\text{Beta}(\alpha, \beta) \;\xrightarrow{\text{observe } X=1 \text{ (success)}}\; \text{Beta}(\alpha+1, \; \beta)$$
$$\text{Beta}(\alpha, \beta) \;\xrightarrow{\text{observe } X=0 \text{ (failure)}}\; \text{Beta}(\alpha, \; \beta+1)$$

This property — where the posterior stays in the *same family* as the prior — is called **conjugacy**, and the Beta distribution is said to be the **conjugate prior** for Bernoulli (and Binomial) likelihoods. The practical payoff: **updating your belief about an arm after a new observation is just "add 1 to $\alpha$ if you saw a success, add 1 to $\beta$ if you saw a failure."** No integration, no numerical optimization, no complicated math at update time — just a counter increment. This is a huge part of why Thompson Sampling is so widely deployed in practice: the update rule is essentially free, computationally.

**Connecting this back to Section 6.3's interpretation**: this is exactly why "$\alpha$ = prior successes + 1, $\beta$ = prior failures + 1" makes sense — every observed success literally increments the success-counting parameter, and every observed failure increments the failure-counting parameter. $\text{Beta}(1,1)$ (zero prior successes/failures, total ignorance) evolving after 7 successes and 3 failures becomes $\text{Beta}(1+7, 1+3) = \text{Beta}(8, 4)$, whose mean is $8/12 = 0.667$ — matching the raw empirical rate of $7/10 = 0.70$ closely but not exactly (the "+1, +1" from the prior gently pulls the estimate toward 0.5 when data is scarce — a mild, sensible regularization effect that fades as more data accumulates).

---

## 6.5 The full Thompson Sampling algorithm for Bernoulli bandits

**Setup**: for each arm $i$, maintain two counters, $\alpha_i$ and $\beta_i$, initialized to $\alpha_i = 1, \beta_i = 1$ (the flat/uniform prior — total ignorance about every arm at the start).

**At each round $t$**:
1. For each arm $i$, draw a random sample $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$ — a single random number between 0 and 1, drawn from that arm's *current* posterior distribution.
2. Pull the arm with the highest sampled value: $A_t = \arg\max_i \theta_i$.
3. Observe the reward $X_t \in \{0, 1\}$.
4. Update: if $X_t = 1$, set $\alpha_{A_t} \mathrel{+}= 1$; if $X_t = 0$, set $\beta_{A_t} \mathrel{+}= 1$. (All other arms' counters are untouched.)

That's the entire algorithm. Notice how radically simpler this is to *implement* than UCB1 or KL-UCB — no logarithms, no square roots, no root-finding — just two integer counters per arm and a random Beta draw.

---

## 6.6 Full worked numerical trace

Same running example: $\mu_1 = 0.30, \mu_2 = 0.50, \mu_3 = 0.45$ (unknown to the algorithm).

**Initialization**: $\alpha_1=\beta_1=1$, $\alpha_2=\beta_2=1$, $\alpha_3=\beta_3=1$ — every arm starts as $\text{Beta}(1,1)$, the flat prior.

**Round 1**: draw one sample from each arm's current posterior (all three are currently identical, $\text{Beta}(1,1)$, i.e. uniform on $[0,1]$ — so this first round is, appropriately, a pure random guess among the three arms):

- $\theta_1 \sim \text{Beta}(1,1) \to$ say we draw $\theta_1 = 0.42$
- $\theta_2 \sim \text{Beta}(1,1) \to$ say we draw $\theta_2 = 0.71$
- $\theta_3 \sim \text{Beta}(1,1) \to$ say we draw $\theta_3 = 0.18$

Highest sample: arm 2 ($\theta_2 = 0.71$) → pull arm 2. Suppose we observe $X_1 = 1$ (a click; true rate is 0.50, so this is plausible). Update: $\alpha_2 \mathrel{+}=1 \to \alpha_2 = 2$. Now arm 2's posterior is $\text{Beta}(2,1)$.

**Round 2**: draw fresh samples. Arm 2's posterior, $\text{Beta}(2,1)$, has mean $2/3 = 0.667$ and is now somewhat concentrated above 0.5 (though still fairly wide, with only 1 observation) — so a draw from it will *tend* to be higher than a draw from arm 1 or 3's still-flat $\text{Beta}(1,1)$, but won't *always* be higher, because there's still real uncertainty.

- $\theta_1 \sim \text{Beta}(1,1) \to$ say $\theta_1 = 0.85$ (a high draw from the flat prior is entirely possible — this is exactly how Thompson Sampling naturally explores under-sampled arms, exactly parallel to UCB1's under-sampled arms getting large bonus terms)
- $\theta_2 \sim \text{Beta}(2,1) \to$ say $\theta_2 = 0.61$
- $\theta_3 \sim \text{Beta}(1,1) \to$ say $\theta_3 = 0.30$

Highest sample: arm 1 ($\theta_1 = 0.85$) → pull arm 1, **even though arm 2 currently has a higher posterior mean** — this happened purely because of a lucky high random draw from arm 1's still-very-uncertain posterior. Suppose we observe $X_2 = 0$ (arm 1's true rate is 0.30, so this is likely). Update: $\beta_1 \mathrel{+}=1 \to \beta_1 = 2$. Arm 1's posterior is now $\text{Beta}(1,2)$, mean $1/3 = 0.333$ — pulled back down towards a more accurate estimate, and importantly, **narrower** than $\text{Beta}(1,1)$, so future draws from it will less often be wildly high like $0.85$ was.

This is the entire mechanic of the algorithm, playing out exactly as designed: **arms with little data get sampled with high variance, occasionally producing high draws that earn them exploration; arms with a lot of data get sampled with low variance, so their draws cluster tightly around their (increasingly accurate) true mean.** No one ever computed an explicit "exploration bonus" the way UCB1 does — the exploration emerges automatically from posterior width, which itself shrinks automatically as data accumulates.

---

## 6.7 Why "probability matching" is the right mental model

Let's make the probability-matching intuition from Section 6.1 fully concrete. Suppose, after many rounds, arm 2's posterior is $\text{Beta}(120, 61)$ (a lot of data, mean $\approx 0.663$) and arm 3's posterior is $\text{Beta}(80, 45)$ (also a lot of data, mean $\approx 0.640$) — genuinely close competitors given the accumulated evidence, even though we (the simulation designers) know the true means are 0.50 and 0.45 respectively (the algorithm doesn't know this, and in this hypothetical it has, through the accumulated noise of many rounds, ended up believing both arms are better than they truly are — entirely possible with finite data).

Because these two posteriors substantially overlap (both are estimating means in the low-to-mid 0.60s with real uncertainty), **Thompson Sampling will pull each of them a meaningful fraction of the time**, roughly proportional to the posterior probability that each one is actually the best arm — this is the literal meaning of "probability matching." Compare this to a hard cutoff rule that would deterministically always pick whichever currently has the (possibly noisily) higher mean — Thompson Sampling's randomized sampling naturally "hedges" across genuinely-uncertain near-ties, which is exactly the statistically sensible thing to do, and is a major part of why it performs so well empirically (a point we'll return to with real benchmark discussion in Chapter 7).

---

## 6.8 Production considerations

- **The Beta-Bernoulli update (Section 6.4) being literally just "increment a counter" is a massive practical advantage.** In a production system serving millions of requests per second, Thompson Sampling's update step is nearly free — no gradient computation, no matrix operations (in the simple, non-contextual case) — just atomic counter increments, which parallelize and shard trivially across distributed serving infrastructure. This is a frequently cited *real* reason companies choose Thompson Sampling over UCB variants in practice, beyond pure regret-bound comparisons.
- **Sampling from a Beta distribution is a well-optimized, constant-time-ish operation** in virtually every numerical library, so the "randomized draw" step (which sounds like it could be expensive) is not a practical bottleneck at serving scale.
- **The choice of prior matters more in low-traffic settings.** With $\text{Beta}(1,1)$ initialization and millions of observations, the prior's influence washes out almost entirely (the "+1, +1" becomes negligible next to large $\alpha, \beta$). But for a **new arm** just added to the system (e.g., a newly launched ad) with very few impressions so far, the prior choice directly shapes early behavior — an informative prior built from related historical data (rather than a flat, fully-agnostic prior) is a common and valuable production refinement, worth mentioning if a cold-start scenario comes up.

---

## 6.9 Interview traps

- **Describing Thompson Sampling as "just adding randomness to greedy."** This undersells the mechanism — the randomness isn't arbitrary noise bolted onto a point estimate; it's a **structured draw from a full posterior distribution** whose width is itself principled and data-driven (exactly the same uncertainty-quantification idea as UCB1's confidence intervals, just used differently). Conflating this with, say, ε-greedy's uniform random exploration is a common and serious mistake.
- **Forgetting the "+1, +1" (or more generally, prior-dependent) initialization and its regularizing effect** — a candidate should be able to explain why $\text{Beta}(1,1)$ starting parameters make sense (total ignorance = uniform prior) and why they gently pull early estimates toward 0.5.
- **Being unable to state the exact update rule** ($\alpha \mathrel{+}=1$ on success, $\beta \mathrel{+}=1$ on failure) — this is such a core, checkable fact that fumbling it signals surface-level familiarity only.
- **Not recognizing the direct link to Bayes-UCB from Chapter 5** when asked to compare the two — a strong answer immediately says "same posterior, Bayes-UCB takes a quantile, Thompson Sampling takes a random sample."

---

## 6.10 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly state the Beta-Bernoulli update rule, correctly describe the algorithm's four steps (sample, pick max, observe, update), and correctly explain probability matching in plain English.
- **L6 bar**:
  - Walks through a numerical trace like Section 6.6 and explicitly narrates the moment where a high random draw from an under-sampled arm's wide posterior causes exploration — mirroring the same kind of mechanical fluency expected for UCB1 in Chapter 4, showing the candidate can reason about *both* algorithm families at the same depth, not just recite one.
  - Explains *why* conjugacy (Section 6.4) is such a big practical deal — connecting the abstract mathematical property directly to concrete production infrastructure benefits (Section 6.8) unprompted.
  - When asked to compare Thompson Sampling to UCB1, immediately identifies that both are, at their core, "quantify uncertainty per arm, then use that uncertainty to bias exploration toward less-certain arms" — differing only in *how* they turn that uncertainty into an arm choice (a fixed upper quantile / bound vs. a random sample) — this is the single deepest unifying insight available in Phase 1 of the course, and stating it unprompted is a strong signal.

---

## 6.11 Comprehension checks

1. Write the Beta-Bernoulli posterior update rule from memory (what happens to $\alpha_i$ and $\beta_i$ on a success vs. a failure).
2. Why is $\text{Beta}(1,1)$ a sensible choice of "totally uninformed" prior, and what are its mean and shape?
3. In the worked trace (Section 6.6, round 2), why did Thompson Sampling pull arm 1 even though arm 2 had a higher posterior mean at that point? Is this a bug or a feature?
4. Explain "probability matching" in your own words, using the near-tied-arms example from Section 6.7.
5. What is the single most important practical (production/infrastructure) advantage of the Beta-Bernoulli conjugate update, compared to, say, KL-UCB's root-finding computation?

---

*Next: Chapter 7 — Thompson Sampling: Extended, where we cover the Gaussian-reward case (Normal-Normal conjugacy), formal regret bounds for Thompson Sampling, and a head-to-head empirical and theoretical comparison against UCB.*
