# Chapter 8 — Adversarial Bandits and EXP3

---

## 8.1 Why we need a new framework at all

Every algorithm so far (ε-greedy, UCB1, KL-UCB, Thompson Sampling) rests on one shared assumption: **each arm's reward comes from a fixed, unknown probability distribution that doesn't change based on your behavior.** This is the **stochastic bandit** setting.

But this assumption can fail in the real world. Consider:
- **Fraud/spam/click-farms**: if you're choosing which ad-fraud-detection rule to apply, the adversary (a spammer) is actively adapting *in response* to your choices — there's no fixed "true click rate," because the environment is intelligently reacting to you.
- **Competitive/adversarial settings** more generally: pricing against a competitor who observes and reacts to your pricing, security systems facing an adaptive attacker, etc.
- **Worst-case robustness**: even without a literal adversary, you might want a guarantee that holds no matter *how* the reward sequence unfolds — not just "on average, assuming a fixed distribution," because you want protection against the possibility that your stochastic-distribution assumption is simply wrong.

**Adversarial bandits** drop the fixed-distribution assumption entirely. Instead, we assume an adversary (possibly worst-case, possibly just "the messy real world") chooses the reward for *every* arm at *every* round, arbitrarily — with one crucial restriction, discussed next, that keeps the problem from being hopeless.

---

## 8.2 The adversarial model, precisely

- At each round $t$, before seeing anything about your choice, an adversary picks a reward $g_t(i) \in [0,1]$ for **every** arm $i = 1, \dots, K$ (not just the one you'll end up pulling).
- You pick an arm $A_t$ (possibly randomly, using everything you've observed so far).
- You observe **only** $g_t(A_t)$ — the reward of the arm you actually pulled. This is the same **bandit feedback** restriction from Chapter 1 (you never see the rewards of arms you didn't pick) — it's what keeps this problem from being trivial even for an adversary.
- Crucially: the adversary's choice of $g_t(\cdot)$ for round $t$ **cannot depend on your randomized choice at round $t$** (it can depend on your *past* choices, but not on the coin flip you're about to make this round) — this is called an **oblivious adversary** in the simplest version of the theory, and it's what makes meaningful guarantees possible at all. (There's a more general "adaptive adversary" variant that can react even to your current-round randomness in more complex ways, which we'll flag but not dive into — a more advanced topic.)

**Why can we still hope for anything at all here?** Because even though there's no "true mean" to estimate, we can still define a benchmark: **the single best fixed arm, in hindsight, against the specific reward sequence that occurred.** That is:

$$G^* = \max_i \sum_{t=1}^T g_t(i)$$

— "the total reward you *would* have gotten had you known in advance which single arm to stick with the whole time." Regret is then defined relative to *this* benchmark:

$$R_T = G^* - \mathbb{E}\left[\sum_{t=1}^T g_t(A_t)\right]$$

This is precisely the **pseudo-regret** concept flagged back in Chapter 2, Section 2.6 — recall we noted the distinction between comparing against a fully clairvoyant *round-by-round* oracle versus the *best fixed arm in hindsight* becomes meaningful specifically in the adversarial setting. Here's that exact distinction, now fully justified: comparing against a round-by-round oracle would be hopeless against a true worst-case adversary (the adversary could always make whichever arm you didn't pick be the good one, every single round) — but "best fixed arm in hindsight" is an achievable, meaningful, and still-useful benchmark.

---

## 8.3 EXP3: the algorithm

EXP3 stands for **Exponential-weight algorithm for Exploration and Exploitation**. It maintains a set of **weights** $w_i(t)$ for each arm, converts those weights into a **probability distribution** over arms, samples an arm from that distribution, and updates weights multiplicatively based on an **importance-weighted** estimate of the reward received.

**Setup**: initialize $w_i(1) = 1$ for all arms $i = 1, \dots, K$. Fix a parameter $\gamma \in (0,1]$ (controls the exploration/exploitation balance).

**At each round $t$**:

1. Convert weights to a sampling distribution, **mixing in a uniform exploration term**:

$$p_i(t) = (1-\gamma)\frac{w_i(t)}{\sum_{j=1}^K w_j(t)} + \frac{\gamma}{K}$$

2. Sample $A_t \sim p(t)$ (i.e., pull arm $i$ with probability $p_i(t)$).
3. Observe reward $g_t(A_t)$.
4. Compute the **importance-weighted reward estimate** for the pulled arm only:

$$\hat{g}_t(A_t) = \frac{g_t(A_t)}{p_{A_t}(t)}, \qquad \hat g_t(i) = 0 \text{ for all } i \neq A_t$$

5. Update weights **multiplicatively**, only for the pulled arm:

$$w_{A_t}(t+1) = w_{A_t}(t) \cdot \exp\left(\frac{\gamma \cdot \hat g_t(A_t)}{K}\right)$$

(all other arms' weights are unchanged: $w_i(t+1) = w_i(t)$ for $i \neq A_t$).

---

## 8.4 Understanding the importance-weighting trick — this is the crux of the whole algorithm

Step 4 is the single most important and most commonly misunderstood piece of EXP3, so let's build real intuition for it.

**The problem it solves**: because of bandit feedback, you only observe $g_t(A_t)$ — the reward for the arm you actually pulled — and you get *zero* information about the other arms this round. If you just used the raw observed reward $g_t(A_t)$ directly to update arm $A_t$'s weight, arms that get pulled rarely would have their weight updated rarely, and — critically — there'd be a **systematic bias**: an arm you rarely explore looks "understudied" in a way that's hard to correct for using raw rewards alone, especially as you try to prove formal guarantees.

**The fix — dividing by the probability of having pulled it**: $\hat g_t(A_t) = g_t(A_t) / p_{A_t}(t)$ is an **unbiased estimator** of the *full* reward vector, in the following precise sense. Consider any arm $i$. If we had defined $\hat g_t(i) = g_t(i)/p_i(t)$ whenever $A_t = i$, and $\hat g_t(i) = 0$ otherwise, then:

$$\mathbb{E}_{A_t \sim p(t)}[\hat g_t(i)] = p_i(t) \cdot \frac{g_t(i)}{p_i(t)} + (1-p_i(t)) \cdot 0 = g_t(i)$$

In plain English: **even though you only ever directly observe one arm's reward per round, dividing by "how likely you were to have picked it" produces an estimate that, on average (over the randomness in your own arm selection), exactly equals the true reward — for every arm, not just the one you picked.** This is a foundational trick that reappears, in a much more developed form, in **Chapter 14 (off-policy evaluation)** — inverse propensity scoring there is *the exact same mathematical idea* applied to full logged-bandit-feedback datasets rather than a single online round. If you can already see this connection now, you're ahead of the curve for Phase 3.

**The intuitive cost of this trick**: if $p_{A_t}(t)$ is small (you rarely pull this arm), then $\hat g_t(A_t) = g_t(A_t)/p_{A_t}(t)$ can become a *very large number* even though the true reward $g_t(A_t) \in [0,1]$ is bounded — dividing by a small probability inflates the estimate. This is exactly why EXP3 needs the **uniform exploration mixing term** $\gamma/K$ in step 1 — it guarantees $p_i(t) \geq \gamma/K > 0$ for every arm, always, which caps how extreme the importance-weighted estimate can become, keeping the algorithm's variance under control. Without this floor, rarely-pulled arms could produce wildly unstable weight updates.

---

## 8.5 Worked numerical trace

Let's use $K=3$ arms and $\gamma = 0.3$ (a fairly large exploration parameter, chosen to make the numbers easy to trace by hand).

**Round 1**: $w_1=w_2=w_3=1$, so $\sum w_j = 3$.

$$p_i(1) = (1-0.3)\frac{1}{3} + \frac{0.3}{3} = 0.7 \times 0.333 + 0.1 = 0.233 + 0.1 = 0.333 \text{ for every arm } i$$

(Makes sense — with all weights tied, every arm is equally likely, exactly $1/3$ each, whether or not you mix in the uniform term.)

Suppose we sample and pull arm 2, observing $g_1(2) = 0.8$ (a good, high reward this round — remember, in the adversarial setting we don't think of this as "arm 2's true mean," just "whatever reward the adversary/environment produced this specific round").

Importance-weighted estimate: $\hat g_1(2) = 0.8 / 0.333 = 2.40$.

Weight update: $w_2(2) = 1 \times \exp\left(\frac{0.3 \times 2.40}{3}\right) = \exp(0.24) \approx 1.271$.

Arms 1 and 3 are untouched: $w_1(2) = w_3(2) = 1$.

**Round 2**: $\sum w_j = 1.271 + 1 + 1 = 3.271$.

$$p_2(2) = 0.7 \times \frac{1.271}{3.271} + 0.1 = 0.7 \times 0.3885 + 0.1 = 0.272 + 0.1 = 0.372$$
$$p_1(2) = p_3(2) = 0.7 \times \frac{1}{3.271} + 0.1 = 0.7 \times 0.3057 + 0.1 = 0.214 + 0.1 = 0.314$$

(Sanity check: $0.372 + 0.314 + 0.314 = 1.000$ ✓.)

Notice arm 2 is now *somewhat* more likely to be pulled (37.2% vs. the original equal 33.3%) because of its good round-1 reward — but it's nowhere close to dominating, and arms 1 and 3 still each retain a healthy 31.4% chance, well above what they'd get under a purely greedy scheme. This is precisely the **gradual, exponentially-weighted "lean toward what's worked" behavior**, tempered by the exploration floor — a very different flavor of exploration than either UCB's confidence bounds or Thompson Sampling's posterior sampling, but philosophically related: **all three families continuously hedge across arms in proportion to accumulated evidence, none of them ever fully commits (in finite time) to a single arm.**

---

## 8.6 The EXP3 regret bound

EXP3 achieves an expected regret bound (against the best-fixed-arm-in-hindsight benchmark from Section 8.2) of:

$$\mathbb{E}[R_T] = O(\sqrt{KT\ln K})$$

Notice this is $O(\sqrt{T})$, **not** $O(\log T)$ — a meaningfully worse asymptotic rate than the stochastic-setting algorithms from Chapters 4–7. This isn't a weakness of EXP3's design — **it's a fundamental fact about the adversarial setting itself**: it can be proven that *no* algorithm can achieve better than $\Omega(\sqrt{T})$ regret against a fully adversarial reward sequence — the problem is intrinsically harder than the stochastic case, and $O(\sqrt{T})$ is the best any algorithm can do (EXP3 is, up to log factors, minimax-optimal for this setting).

**This is a genuinely important interview-level takeaway, worth stating explicitly**: *the jump from $O(\log T)$ to $O(\sqrt{T})$ isn't a sign that EXP3 is a worse algorithm than UCB/TS — it's the unavoidable statistical price of dropping the fixed-distribution assumption.* If someone asks "why not just always use EXP3, since it doesn't need any distributional assumptions" — the correct answer is exactly this: you pay a real, provably-unavoidable regret penalty for that generality, so you should only reach for adversarial-bandit tools when you genuinely can't trust the stochastic assumption (e.g., real adversarial/adaptive-opponent settings), not as a "safer default."

---

## 8.7 Connection to online learning and no-regret dynamics (brief, interview-relevant framing only)

EXP3 is a bandit-feedback special case of a broader family of algorithms from **online learning** called **multiplicative weights** / **Hedge** algorithms, which assume *full-information* feedback (you get to see *every* arm's reward every round, not just the one you picked) — Hedge achieves an even better $O(\sqrt{T\ln K})$ bound in that easier full-information setting, and EXP3 is essentially "Hedge, adapted to bandit feedback via the importance-weighting trick from Section 8.4." This family of algorithms is foundational to **game theory and no-regret learning** — a "no-regret" algorithm played repeatedly against other no-regret algorithms provably converges to a Nash equilibrium in certain game classes, a fact occasionally referenced in interviews for context/breadth but essentially never requiring derivation. Knowing this connection exists — "EXP3 descends from the multiplicative weights / Hedge family of online-learning algorithms, and full-information versions of this idea connect to game-theoretic equilibrium concepts" — is enough depth for virtually any interview context.

---

## 8.8 Production considerations

- **Adversarial bandit techniques are most directly relevant in fraud detection, ad-auction/bidding systems facing strategic competitors, and security-adjacent applications** — settings where assuming a fixed, cooperative reward distribution is actively dangerous, because the "environment" includes rational actors adapting to you.
- **In practice, most production recommendation/ranking/ads systems use stochastic-bandit tools (UCB/TS), not EXP3** — because most user behavior, while noisy, is reasonably modeled as coming from a roughly-stable distribution over reasonable timescales, and the better ($O(\log T)$) regret rate is worth the (usually acceptable) risk of the stochastic assumption being imperfect. EXP3-style tools tend to be reserved for the specific sub-problems (fraud, adversarial competitors) where that assumption is genuinely untenable, rather than a company-wide default.
- **The importance-weighting trick (Section 8.4) is the conceptual seed of Phase 3's off-policy evaluation methods** — worth explicitly telling your interviewer you see this connection if the conversation naturally reaches both topics; it's a strong signal of synthesized understanding rather than siloed topic memorization.

---

## 8.9 Interview traps

- **Applying EXP3 by default "because it's more general/robust."** As emphasized in Section 8.6, generality has a real, provable regret cost ($\sqrt{T}$ vs $\log T$) — a candidate who reaches for EXP3 without being asked to justify the adversarial assumption is signaling they haven't internalized *why* the stochastic algorithms are preferred whenever their assumptions hold.
- **Forgetting the uniform-mixing term $\gamma/K$ in the sampling distribution**, or being unable to explain why it's needed (Section 8.4's variance-control argument) — this is a specific, checkable formula detail.
- **Misdescribing the importance-weighted estimator as biased.** It's unbiased in expectation over the algorithm's own randomization — a subtle but precise claim (Section 8.4) that's worth being able to state correctly, including the "expectation over what, exactly" qualifier.
- **Confusing the "best fixed arm in hindsight" benchmark with the round-by-round-optimal benchmark** used in the stochastic setting — this was flagged as a genuine subtlety back in Chapter 2 and is worth being precise about here, in the setting where the distinction actually bites.

---

## 8.10 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describe what makes the adversarial setting different from stochastic (no fixed distribution), correctly state EXP3's high-level mechanism (exponential weights + bandit feedback), and know the regret rate is $O(\sqrt{T})$.
- **L6 bar**:
  - Can derive/explain the unbiasedness of the importance-weighted estimator (Section 8.4) with the actual expectation calculation, not just assert it.
  - Explicitly connects EXP3's importance-weighting trick forward to Chapter 14's inverse propensity scoring, unprompted — a strong signal of holistic course synthesis.
  - Can give a precise, reasoned answer to "when would you actually choose EXP3 over UCB/TS in a real system," grounded in the fundamental $\log T$ vs. $\sqrt T$ tradeoff from Section 8.6, rather than a vague "EXP3 is more robust" hand-wave.

---

## 8.11 Comprehension checks

1. What is the key structural assumption that adversarial bandits drop, relative to stochastic bandits?
2. Write the importance-weighted reward estimator $\hat g_t(A_t)$, and explain in your own words why dividing by $p_{A_t}(t)$ makes it unbiased.
3. Why does EXP3 need the uniform-mixing term $\gamma/K$ in its sampling distribution — what specific problem would arise without it?
4. Why is EXP3's $O(\sqrt{T})$ regret rate not a sign of a poorly-designed algorithm, compared to UCB1/TS's $O(\log T)$?
5. In the worked trace (Section 8.5), why did arm 2's probability of being pulled only rise modestly (33.3% → 37.2%) after a single good outcome, rather than jumping dramatically?

---

*Next: Chapter 9 — Best-Arm Identification (Pure Exploration), where we switch objectives entirely — from minimizing cumulative regret while playing, to simply identifying the single best arm as efficiently as possible, with a PAC-style confidence guarantee.*
