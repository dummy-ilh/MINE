# Chapter 2 — Regret: The Central Metric

---

## 2.1 Why we need a metric at all

Chapter 1 ended with a vague goal: "make $A_t$ equal to the best arm as often as possible, as early as possible." That's an intuition, not a number you can optimize or compare algorithms with. We need a single quantity that:

- Goes to zero (or grows slowly) when a policy is good
- Grows large when a policy is bad
- Lets us mathematically prove "algorithm X is better than algorithm Y"

That quantity is **regret**. Every bandit algorithm in this course is, underneath, a strategy for keeping regret small.

---

## 2.2 The core idea, in plain English

Regret answers one question: **"Compared to a psychic who already knew which arm was best and always pulled it, how much reward did I lose by not knowing that in advance?"**

Go back to our casino example: $\mu_1 = 0.30$, $\mu_2 = 0.50$, $\mu_3 = 0.45$. The psychic always pulls arm 2 (the best arm, $\mu^* = 0.50$) and earns $0.50$ per round, on average. Any round where you pull arm 1 or arm 3 instead, you're earning less than the psychic — that "less" is what gets counted as regret.

Regret isn't about whether you're making money (you always are, on average, as long as all $\mu_i > 0$) — it's about the **gap between what you got and what the best possible policy would have gotten.** This reframing is what makes bandit algorithms mathematically analyzable: we're not asking "is 0.42 average reward good?" in isolation — we're asking "how far below 0.50 (the ceiling) is 0.42, and how fast does that gap shrink?"

---

## 2.3 Formal definition: per-round regret

At round $t$, define the **instantaneous regret**:

$$r_t = \mu^* - \mu_{A_t}$$

where $\mu^* = \max_i \mu_i$ is the best arm's mean, and $A_t$ is the arm your policy actually picked at round $t$.

- If you pick the best arm, $r_t = 0$ (no regret that round).
- If you pick a suboptimal arm, $r_t > 0$ — the size of the gap between that arm's true mean and the best arm's true mean.

### Worked example

Using $\mu_1 = 0.30, \mu_2 = 0.50, \mu_3 = 0.45$ (so $\mu^* = 0.50$):

- If at round $t$ you pull arm 1: $r_t = 0.50 - 0.30 = 0.20$
- If you pull arm 2 (the best arm): $r_t = 0.50 - 0.50 = 0$
- If you pull arm 3: $r_t = 0.50 - 0.45 = 0.05$

Notice something important: **regret is defined in terms of true means $\mu_i$, not in terms of the noisy reward $X_t$ you actually observed.** This is a common point of confusion, so let's be explicit: even if you pull arm 2 and happen to observe $X_t = 0$ (bad luck on that one flip), your instantaneous regret is still $r_t = 0$, because you pulled the *correct* arm — you just got an unlucky sample from it. Regret measures the quality of your **decision**, not the luck of a single outcome.

---

## 2.4 Cumulative regret (total regret)

What we actually care about is regret summed over the whole horizon:

$$R_T = \sum_{t=1}^{T} r_t = \sum_{t=1}^{T} \big(\mu^* - \mu_{A_t}\big)$$

This is called the **cumulative regret** (or just "regret") over horizon $T$. It is *the* headline number reported in every bandit paper and expected in every interview answer.

### Worked example — tracing regret over 5 rounds

Suppose our policy picks arms in this order: $A_1 = 1, A_2 = 1, A_3 = 3, A_4 = 2, A_5 = 2$.

| Round $t$ | Arm picked $A_t$ | $\mu_{A_t}$ | $r_t = \mu^* - \mu_{A_t}$ | Cumulative regret $R_t$ |
|---|---|---|---|---|
| 1 | 1 | 0.30 | 0.20 | 0.20 |
| 2 | 1 | 0.30 | 0.20 | 0.40 |
| 3 | 3 | 0.45 | 0.05 | 0.45 |
| 4 | 2 | 0.50 | 0.00 | 0.45 |
| 5 | 2 | 0.50 | 0.00 | 0.45 |

After 5 rounds, $R_5 = 0.45$. Notice cumulative regret is **non-decreasing** — it can only stay flat (when you pick the best arm) or increase (when you don't). It never goes down. A good algorithm is one whose regret curve **flattens out** over time — meaning it eventually, almost always, picks the best arm, so each new round adds (close to) zero regret.

---

## 2.5 Why "flattening" regret is the goal — linear vs sublinear regret

This is the single most important shape-recognition skill in the whole bandit field. Plot cumulative regret $R_T$ against $T$:

- **Linear regret**: $R_T$ grows roughly proportional to $T$ (e.g., $R_T \approx 0.1 \cdot T$). This means the policy keeps making mistakes at a *constant rate*, forever — it never actually learns which arm is best, or it learns but keeps exploring anyway at a fixed rate. **This is the signature of a bad algorithm.** Pure ε-greedy with a *constant* ε (never decaying) produces linear regret — we'll prove this in Chapter 3.

- **Sublinear regret**: $R_T$ grows slower than $T$ — most commonly $O(\log T)$ or $O(\sqrt{T})$. This means the *rate* of mistake-making shrinks over time: the policy is learning, and eventually almost every pull goes to the best arm. **This is the signature of a good algorithm**, and it is the standard every algorithm from Chapter 4 onward (UCB, Thompson Sampling) is judged against.

Why does the shape matter so much more than the raw number? Because $R_T$ being "small" at $T=100$ doesn't tell you anything about whether the policy has actually learned — you need to know how $R_T$ *scales* as $T$ grows, and that's exactly what the $O(\cdot)$ notation captures.

**A good intuition check**: if regret is $O(\log T)$, then going from $T = 1{,}000$ to $T = 1{,}000{,}000$ (1000× more rounds) only grows regret by a factor of $\log(1{,}000{,}000)/\log(1{,}000) = 6/3 = 2\times$. That's an extraordinary result — a thousand-fold increase in traffic only doubles your cumulative mistakes. This is why $O(\log T)$ is treated as the gold standard, and why interviewers expect you to recognize it on sight.

---

## 2.6 Pseudo-regret vs expected regret vs realized regret — the three flavors

This is where most candidates get sloppy, and it's a favorite interview clarifying-question moment. There are three closely related but distinct quantities, and mixing them up is a real interview trap.

**1. Realized regret** — the actual number you'd compute after running one specific trial, using the actual random rewards observed:

$$\hat{R}_T = \sum_{t=1}^{T} \big(\mu^* - X_t^{\text{would-be-optimal}}\big)$$

This is noisy — it depends on the specific random draws that happened to occur, and would be a little different every time you re-ran the experiment.

**2. Expected regret** — take the expectation of realized regret over the randomness in both the rewards *and* the policy's own random choices (many policies, like Thompson Sampling, are themselves randomized):

$$\mathbb{E}[R_T] = \mathbb{E}\left[\sum_{t=1}^{T} (\mu^* - \mu_{A_t})\right]$$

This is what's usually meant by "regret" in a bandit paper, and matches the definition we built up in Sections 2.3–2.4 — note it's already written in terms of *means* $\mu_{A_t}$, not raw observed rewards, so taking $\mathbb{E}[\cdot]$ here is really just averaging over the randomness in which arm $A_t$ gets chosen.

**3. Pseudo-regret** — a subtly different and more common theoretical quantity:

$$\bar{R}_T = T\mu^* - \mathbb{E}\left[\sum_{t=1}^T \mu_{A_t}\right] = \mathbb{E}\left[\sum_{t=1}^T (\mu^* - \mu_{A_t})\right]$$

Wait — that looks identical to expected regret above. It is, **in the stochastic bandit setting** — this is a common source of confusion, and it's fine to treat them as the same thing in that setting. The distinction matters more in the **adversarial** setting (Chapter 8), where the "best arm" itself could be defined against the *realized* sequence of rewards versus the *best fixed arm in hindsight* — pseudo-regret specifically compares against the best single fixed arm in hindsight, which becomes a meaningfully different (and more tractable) benchmark than comparing against a fully clairvoyant round-by-round oracle once rewards can be adversarial.

**Practical interview guidance**: for stochastic bandits (essentially everything until Chapter 8), you can and should just say "expected regret" and use the definition from Section 2.4 — that's what 95% of interview conversations mean by "regret." Bring up the pseudo-regret vs expected-regret distinction only when the adversarial setting comes up, where it's the technically correct term.

---

## 2.7 The Lai-Robbins lower bound (stated and interpreted)

Here's a natural question: **is there a limit to how small regret can be made?** Yes — and this result, from Lai and Robbins (1985), is one of the most-cited facts in the whole field. You are not expected to derive it in an interview, but you are expected to state it and interpret it correctly.

For a stochastic bandit with $K$ arms, the theorem says that **any reasonable ("consistent") policy** must suffer regret that grows at least like:

$$R_T \geq \left(\sum_{i : \mu_i < \mu^*} \frac{\mu^* - \mu_i}{\text{KL}(\mu_i, \mu^*)}\right) \log T + o(\log T)$$

Don't worry about memorizing the constant in front of $\log T$ — what matters for an interview is the **shape** of this result:

- The lower bound is $\Omega(\log T)$ — meaning **no algorithm can do better than logarithmic regret**, asymptotically, in the stochastic setting. This is why $O(\log T)$ (achieved by UCB and others, Chapter 4) is called "asymptotically optimal" — it matches this unbeatable lower bound.
- The sum is over all *suboptimal* arms $i$ (arms where $\mu_i < \mu^*$) — each suboptimal arm contributes its own term to the unavoidable regret.
- Each term has $(\mu^* - \mu_i)$ in the numerator (the "gap" — how much worse that arm is) divided by a KL-divergence term in the denominator (how *statistically distinguishable* that arm's distribution is from the best arm's distribution).
- **Interpretation**: arms that are close in mean to the optimal arm ($\mu^* - \mu_i$ small) but hard to statistically distinguish (small KL-divergence) force you to pull them *more* before you can confidently rule them out — hence they contribute *more* to the unavoidable regret, not less. This is a genuinely counter-intuitive and interview-favorite point: **a "close but statistically ambiguous" arm can be a bigger regret driver than an arm that's obviously much worse.**

---

## 2.8 Problem-dependent vs problem-independent (minimax) bounds

One more distinction interviewers probe: bandit regret bounds come in two flavors, and you should recognize both on sight.

- **Problem-dependent bounds** (also called "instance-dependent") — the bound depends on the specific gaps $\Delta_i = \mu^* - \mu_i$ between arms. The Lai-Robbins bound above is problem-dependent — notice $\mu_i$ appears explicitly in the formula. These bounds say: "the harder your specific problem instance is (arms close together), the more regret is unavoidable for *that* instance."

- **Problem-independent bounds** (also called "minimax" or "worst-case" bounds) — a single bound that holds for *any* possible configuration of arm means, typically of the form $O(\sqrt{KT})$ or $O(\sqrt{KT\log T})$. These answer: "no matter how adversarially the arm means are set up, here's the worst this algorithm can do."

**Why both exist and matter**: a problem-dependent $O(\log T)$ bound can *look* better than a problem-independent $O(\sqrt{T})$ bound, but they're not directly comparable — the $\log T$ bound has a constant in front that depends on $1/\Delta_i^2$ (inverse-squared gaps), which **blows up** when arms are very close together (small $\Delta_i$). So an algorithm can have great problem-dependent asymptotic regret but still perform poorly in the worst case for hard, closely-spaced problem instances. Strong algorithms (like UCB, and MOSS in Chapter 5) are judged on **both** axes simultaneously — this is exactly the kind of nuance that separates an L6 answer from an L5 answer, covered in Section 2.10 below.

---

## 2.9 Production considerations

- **You never actually know $\mu^*$ or $\mu_i$ in production**, which means you can never *compute* regret directly on a live system the way we did in our worked examples — those required knowing the ground-truth means. In practice, regret is estimated in simulation (using historical data to construct a synthetic environment with assumed true means) or approximated via proxies like "estimated CTR uplift over control" in an A/B test. This is a genuine practical limitation worth naming.
- **Regret is a relative metric, not an absolute one.** A policy with "low regret" in an environment where all arms have similar, mediocre payouts might still be shipping a mediocre product. Regret tells you how close to the *best available option* you got — not whether the best available option was actually good. Always pair regret analysis with absolute performance metrics in a real system review.
- **The "best arm" itself can drift** (Section 1.7, and Chapter 13 in full) — in production, $\mu^*$ isn't a fixed constant the way it is in our textbook setup, so cumulative regret against a *moving* target requires the non-stationary variants covered later.

---

## 2.10 Interview traps

- **Confusing regret with negative reward.** Regret is a *gap*, not raw loss. An arm with $\mu_i = 0.45$ is still profitable in absolute terms — it just has $r_t = 0.05$ regret relative to the best arm at $0.50$. Candidates sometimes describe regret as if it means "you lost money" — correct this framing immediately if you catch yourself doing it.
- **Forgetting that cumulative regret is monotonically non-decreasing.** If asked to sketch a regret curve, it should never go down — a common candidate error is drawing a curve that dips, which isn't possible under this definition (each $r_t \geq 0$ always).
- **Reporting a raw regret number without stating $T$ or the scaling behavior.** "This algorithm has regret 50" means nothing without knowing the horizon and how that number would grow if $T$ were 10x larger. Always frame regret answers in terms of asymptotic scaling ($O(\log T)$, $O(\sqrt{T})$, or linear), not a bare number.
- **Treating pseudo-regret and expected regret as always identical** — true in the stochastic setting (fine to use interchangeably there), false in general once you're in the adversarial setting. Flagging this awareness, even briefly, reads as real depth.

---

## 2.11 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly define instantaneous and cumulative regret, correctly identify that $O(\log T)$ is sublinear and "good," and correctly explain why linear regret is bad (constant mistake rate forever).
- **L6 bar**:
  - Can state the Lai-Robbins lower bound's *shape* (sum over suboptimal arms of gap-over-KL-divergence terms) and, critically, explain *why* statistically-similar-but-suboptimal arms are the dominant regret contributors — this is the single most differentiating insight in this chapter.
  - Distinguishes problem-dependent from problem-independent bounds unprompted, and explains why an algorithm needs to be evaluated on both.
  - Proactively notes that regret can't be computed directly in production (Section 2.9) and names a practical workaround, showing awareness that textbook regret is a theoretical tool for algorithm design and comparison, not a metric you'd literally compute on a live dashboard.

---

## 2.12 Comprehension checks

1. Write the formula for instantaneous regret $r_t$ and cumulative regret $R_T$, in words and in notation.
2. Why can cumulative regret never decrease, round over round?
3. Explain, in your own words, why $O(\log T)$ regret is dramatically better than linear regret as $T$ grows large — use the "1000× more traffic" argument from Section 2.5.
4. What's the difference between a problem-dependent and a problem-independent regret bound, and why do strong algorithms need to be evaluated on both?
5. According to the Lai-Robbins lower bound, which contributes more to unavoidable regret: an arm that's much worse than optimal but easy to statistically distinguish, or an arm that's only slightly worse but hard to statistically distinguish? Why?

---

*Next: Chapter 3 — Naive Baselines (greedy, ε-greedy, explore-then-commit), where we'll use the regret machinery built here to formally prove why the "obvious" strategies fail.*
