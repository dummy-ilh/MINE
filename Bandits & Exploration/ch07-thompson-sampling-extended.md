# Chapter 7 — Thompson Sampling: Extended

---

## 7.1 What this chapter adds

Chapter 6 built Thompson Sampling for Bernoulli rewards (click/no-click), using the Beta-Bernoulli conjugate pair. Real production reward signals aren't always binary — watch-time, revenue-per-click, session length — these are **continuous**. This chapter extends Thompson Sampling to continuous (Gaussian) rewards, then covers the formal regret theory, and finally builds the full TS-vs-UCB comparison we've been promising since Chapter 5.

---

## 7.2 Gaussian bandits: the setup

Now assume each arm $i$'s reward is drawn from a Normal (Gaussian) distribution with unknown mean $\mu_i$ and — to keep things clean for this first pass — a **known, fixed variance** $\sigma^2$ (a common simplifying assumption; the unknown-variance case exists but is a more advanced extension we'll flag rather than derive). Think of $\mu_i$ as, e.g., the average revenue-per-impression of ad $i$, and individual impressions producing noisy revenue draws around that true average.

---

## 7.3 Normal-Normal conjugacy

Just as Beta was the conjugate prior for Bernoulli likelihoods, the **Normal distribution is its own conjugate prior** for a Normal likelihood with known variance. This is arguably an even more elegant conjugacy than Beta-Bernoulli.

**Prior**: $\mu_i \sim \mathcal{N}(m_i, s_i^2)$ — your current belief about arm $i$'s true mean, itself expressed as a Normal distribution with its own mean $m_i$ (best guess) and variance $s_i^2$ (uncertainty about that guess — note this is *not* the same $\sigma^2$ as the reward noise; it's uncertainty about the *mean*, which shrinks with data, versus $\sigma^2$, the inherent noise in individual reward draws, which does not shrink).

**Update rule after observing a new reward $X$ from arm $i$**: the posterior is again Normal, $\mathcal{N}(m_i', {s_i'}^2)$, with:

$$s_i'^2 = \left(\frac{1}{s_i^2} + \frac{1}{\sigma^2}\right)^{-1}, \qquad m_i' = s_i'^2 \left(\frac{m_i}{s_i^2} + \frac{X}{\sigma^2}\right)$$

This looks more intimidating than the Beta-Bernoulli "+1" rule, but the intuition is the same "precision-weighted averaging" idea used throughout Bayesian statistics: **your new belief's mean is a weighted average of your old belief and the new data point, where the weights are inverse-variances ("precisions") — more precise (lower-variance) information gets more weight.** As more data accumulates, $s_i'^2$ shrinks (more certainty), exactly mirroring how $\alpha_i, \beta_i$ growing shrinks the Beta posterior's spread in Chapter 6.

**A cleaner special case worth internalizing**: if you initialize with a very wide/uninformative prior (large $s_i^2$) and observe $n$ i.i.d. samples with sample mean $\bar{X}$, the posterior mean converges to essentially the raw sample mean $\bar X$, and the posterior variance converges to $\sigma^2/n$ — which should look familiar: **it's exactly the standard-error-shrinks-as-$1/\sqrt{n}$ behavior from classical statistics**, just recovered here as a special case of the Bayesian update. This is a satisfying and interview-friendly way to sanity-check the formula without memorizing it symbol-for-symbol.

### Worked numerical trace

Say arm 2's true mean is $\mu_2 = 0.50$ (as always), reward noise $\sigma^2 = 1$ (moderately noisy), and we start with a weakly informative prior $\mathcal{N}(m_2=0, s_2^2=100)$ — i.e., "I have almost no idea, but I'll center my guess at 0."

**After observing one reward, $X = 0.62$:**

$$s_2'^2 = \left(\frac{1}{100} + \frac{1}{1}\right)^{-1} = \left(0.01 + 1\right)^{-1} = (1.01)^{-1} \approx 0.990$$

$$m_2' = 0.990 \times \left(\frac{0}{100} + \frac{0.62}{1}\right) = 0.990 \times 0.62 \approx 0.614$$

New posterior: $\mathcal{N}(0.614, 0.990)$. Notice the posterior mean moved almost all the way to the observed data point (0.62) and barely stayed influenced by the prior's original guess of 0 — because the prior was so wide (variance 100, essentially "no information") relative to the single data point's precision, the data completely dominates after just one observation. This is the Normal-Normal analog of the Beta-Bernoulli "+1, +1" softly regularizing toward the prior — here, a wide/weak prior gets swamped almost immediately, exactly as it should.

**Thompson Sampling step**: draw $\theta_2 \sim \mathcal{N}(0.614, 0.990)$ — note the *variance* used for sampling is the *posterior* variance of the mean ($s_2'^2 = 0.990$), **not** the reward noise variance $\sigma^2 = 1$ — a detail worth being precise about if asked, since conflating "uncertainty about the mean" with "inherent noise in individual rewards" is an easy mistake.

---

## 7.4 Regret bounds for Thompson Sampling

Thompson Sampling achieves regret bounds of the same $O(\log T)$ **shape** as UCB1, but the theoretical analysis is structured a bit differently, and it's worth knowing the vocabulary precisely.

- **Frequentist regret** (the kind we defined in Chapter 2 — expectation taken only over the randomness in rewards and the algorithm's own random sampling, for one **fixed, specific** true configuration of arm means) has been proven, for Thompson Sampling, to match the same $O\left(\sum_i \frac{\ln T}{\Delta_i}\right)$-shaped problem-dependent bound as UCB-family algorithms, and in fact matches the Lai-Robbins lower bound asymptotically for Bernoulli bandits — i.e., **Thompson Sampling is also asymptotically optimal**, in the same precise sense KL-UCB was shown to be in Chapter 5.

- **Bayesian regret** is a different (and, historically, easier-to-prove) quantity: instead of fixing one true arm-mean configuration and analyzing worst-case behavior over the algorithm's own randomness, Bayesian regret *additionally* averages over a prior distribution on which arm-mean configuration is "true" in the first place. Formally:

$$\text{BayesRegret}(T) = \mathbb{E}_{\mu \sim \text{prior}}\left[\mathbb{E}[R_T \mid \mu]\right]$$

Thompson Sampling has a particularly clean and historically important $O(\sqrt{KT\log T})$ Bayesian-regret bound (matching the problem-independent/minimax rate up to a $\log T$ factor) — this was, in fact, easier to prove than the frequentist bound, and was proven first historically, which is worth knowing as context if asked "why does TS have two different flavors of regret guarantee?"

**Interview-level takeaway**: you don't need to reproduce these proofs. You need to be able to say: *"Thompson Sampling has strong theoretical guarantees under both a frequentist and a Bayesian analysis — frequentist analysis shows it matches the Lai-Robbins lower bound asymptotically, just like KL-UCB; Bayesian analysis (averaging over a prior on the true arm means) gives a clean minimax-shaped bound. Both confirm the same practical conclusion: TS achieves essentially the best possible regret shape."*

---

## 7.5 Why Thompson Sampling often *beats* UCB empirically, despite similar bounds

This is one of the most interview-relevant, genuinely subtle facts in the entire syllabus, so let's build real intuition for it rather than just asserting it.

Both UCB1 and Thompson Sampling achieve $O(\log T)$-shaped asymptotic regret — but **asymptotic** means "as $T \to \infty$," and it hides the *constant* in front of the $\log T$ term, as well as behavior at small/moderate $T$ (which is often what matters most in practice — a product doesn't get to run for "infinity" before someone checks the dashboard). Empirically, across a large number of benchmark studies (most famously Chapelle & Li's 2011 empirical evaluation), **Thompson Sampling consistently achieves lower regret than UCB1 in practice**, especially at moderate horizons. A few reasons commonly cited for this gap:

1. **UCB1's Hoeffding-derived bonus is a worst-case, "safe" bound** — it's built to hold with high probability across *any* bounded distribution, which, as we discussed in Chapter 5, makes it looser (more conservative, more exploration) than necessary for a specific, known distribution family like Bernoulli. Thompson Sampling's posterior, by contrast, is built using the *exact* correct likelihood (Bernoulli, in our example), so it isn't "wasting" exploration on distributional possibilities that can't actually occur — this is the exact same tightness argument that made KL-UCB better than UCB1, and Thompson Sampling gets a very similar benefit "for free" as a natural consequence of doing genuine Bayesian inference rather than using a generic concentration inequality.
2. **Randomized sampling naturally smooths decisions across near-ties** (the probability-matching behavior from Chapter 6, Section 6.7), rather than UCB1's hard deterministic arg-max, which can be more brittle when two arms' point-estimates/bounds are very close.

**A useful, precise way to summarize this for an interview**: *"UCB1 is intentionally conservative because Hoeffding's inequality has to hold for any bounded distribution. KL-UCB fixes this by using the exact Bernoulli likelihood, and gets close to Thompson Sampling's performance as a result — this suggests the gap isn't really 'UCB vs. TS' as competing philosophies, it's really 'generic worst-case bound vs. distribution-matched bound,' and Thompson Sampling gets distribution-matching for free as a natural consequence of Bayesian inference, while UCB needs a specifically-engineered variant (KL-UCB) to get the same benefit."* This single sentence, if you can produce it fluently, is close to a complete synthesis of Chapters 4 through 7.

---

## 7.6 Full comparison table: TS vs. UCB family

| Dimension | UCB1 | KL-UCB | Thompson Sampling |
|---|---|---|---|
| Decision rule | Deterministic arg-max of upper bound | Deterministic arg-max of (tighter) upper bound | Randomized: sample from posterior, arg-max of samples |
| Uses distribution-specific info? | No (generic, Hoeffding-based) | Yes (Bernoulli-specific KL bound) | Yes (full Bayesian likelihood) |
| Regret shape | $O(\log T)$, not asymptotically optimal constant | $O(\log T)$, asymptotically optimal (matches Lai-Robbins) | $O(\log T)$, asymptotically optimal (matches Lai-Robbins) |
| Typical empirical performance | Good, but usually beaten by KL-UCB/TS | Very good | Very good, often best in benchmarks |
| Reproducible given fixed history? | Yes (deterministic) | Yes (deterministic) | No (randomized) |
| Implementation complexity | Low (closed-form bonus) | Moderate (numerical root-finding for the index) | Low (Beta/Normal sampling + simple counter updates) |
| Needs a prior? | No | No | Yes (though its influence fades with data) |

This table is likely the single highest-value artifact in this whole chapter for rapid interview recall — worth being able to reproduce most of it from memory.

---

## 7.7 Production considerations

- **The empirical superiority of TS (Section 7.5) is a large part of why it's the more commonly deployed choice** in large-scale industrial systems (widely reported at companies including Microsoft, Google, and LinkedIn in published case studies) — despite UCB1 often being taught first academically because its analysis is more elementary.
- **Gaussian Thompson Sampling (Section 7.3) generalizes cleanly to continuous business metrics** (revenue, watch-time, session length) where Bernoulli/Beta doesn't directly apply — this is the practically important reason to know the Normal-Normal case, not just as a theoretical curiosity.
- **The "known variance" assumption in Section 7.2 is a simplification** — production systems typically don't know the true reward-noise variance in advance. The fully Bayesian fix (a Normal-Inverse-Gamma prior, jointly modeling unknown mean *and* unknown variance) exists and is used in practice, but is a level of detail beyond what's typically expected in an interview — knowing it exists and being able to name it ("Normal-Inverse-Gamma for unknown-variance Gaussian bandits") is enough to signal awareness without needing to derive it.

---

## 7.8 Interview traps

- **Claiming Thompson Sampling has "better" theoretical regret bounds than UCB1/KL-UCB.** This isn't quite right — TS's frequentist bound matches KL-UCB's (both are asymptotically optimal, matching Lai-Robbins), so the *shape and asymptotic optimality* are comparable. The genuine, well-documented empirical edge (Section 7.5) is a distinct claim from a theoretical superiority claim — conflating these is a common and avoidable mistake.
- **Confusing "Bayesian regret" with "the regret Thompson Sampling happens to be Bayesian about."** Bayesian regret is a specific *evaluation metric* (averaging over a prior on the true configuration), not a description of the algorithm's internal mechanics. An algorithm being "Bayesian" (like TS) and a regret bound being "Bayesian" (an evaluation framework) are related but distinct concepts, and interviewers will notice if you blur them.
- **In the Gaussian case, sampling using the wrong variance** — using $\sigma^2$ (reward noise) instead of $s_i'^2$ (posterior uncertainty about the mean) when drawing $\theta_i$ for the Thompson Sampling step. This is a precise, checkable error (flagged explicitly at the end of Section 7.3) that signals whether you actually understand what's being sampled from.

---

## 7.9 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describe the Gaussian/Normal-Normal Thompson Sampling setup, correctly state that TS achieves comparable regret to UCB, and know that TS tends to perform well empirically.
- **L6 bar**:
  - Can produce the full synthesis sentence from Section 7.5 unprompted — explaining *why* the "UCB vs TS" framing is somewhat misleading, and that the real axis is "generic worst-case bound vs. distribution-matched bound," with TS getting distribution-matching for free via genuine Bayesian inference.
  - Distinguishes frequentist regret from Bayesian regret precisely, and can explain historically why the Bayesian regret bound for TS was easier to establish first.
  - Names the Normal-Inverse-Gamma extension for the unknown-variance case unprompted when discussing the Gaussian setup's simplifying assumption, showing awareness of the boundary between "what we covered" and "what exists but is out of scope."

---

## 7.10 Comprehension checks

1. Write the Normal-Normal posterior update rule and explain, in plain English, why it's described as "precision-weighted averaging."
2. In the Gaussian Thompson Sampling algorithm, which variance do you sample with — the reward noise variance $\sigma^2$, or the posterior variance of the mean $s_i'^2$? Why does this distinction matter?
3. What's the difference between frequentist regret and Bayesian regret, as applied to analyzing Thompson Sampling?
4. Give the one-sentence synthesis (Section 7.5) explaining why Thompson Sampling tends to empirically outperform UCB1, and why that gap shrinks once you compare against KL-UCB instead.
5. Name one practical extension needed to apply Gaussian Thompson Sampling when the reward-noise variance is *not* known in advance.

---

*Next: Chapter 8 — Adversarial Bandits, where we drop the assumption that rewards come from any fixed distribution at all, and build EXP3 — an algorithm that still achieves strong regret guarantees even when an adversary is actively trying to make you look bad.*
