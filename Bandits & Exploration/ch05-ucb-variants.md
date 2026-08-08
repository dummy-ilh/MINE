# Chapter 5 — UCB Variants: UCB2, KL-UCB, Bayes-UCB, and MOSS

---

## 5.1 Why UCB1 isn't the end of the story

UCB1's bound $\sqrt{2\ln t / N_i(t)}$ came from Hoeffding's inequality (Chapter 4, Section 4.2), which is a **general-purpose** concentration bound — it works for *any* random variable bounded in $[0,1]$, regardless of its actual distribution shape. That generality is also its weakness: Hoeffding's inequality doesn't know or use anything about the *specific* distribution (e.g., Bernoulli vs. some other bounded distribution) — it just uses the boundedness. This means UCB1's confidence intervals are often **wider than they need to be**, which means UCB1 explores more than strictly necessary, which costs extra regret in the constant factor (even though the asymptotic $O(\log T)$ *shape* is already optimal).

This chapter covers four variants that improve on UCB1 along different axes:

- **UCB2**: reduces the *frequency* of recomputing/switching arms (an implementation-efficiency angle)
- **KL-UCB**: uses a *distribution-aware* confidence bound (specifically for Bernoulli-type rewards), giving a provably tighter bound than Hoeffding
- **Bayes-UCB**: reframes the confidence bound in Bayesian terms, blurring the line toward Thompson Sampling
- **MOSS**: optimizes specifically for the *worst-case* (minimax) regret bound rather than the problem-dependent bound

You are **not** expected to derive each of these from scratch in an interview — you're expected to know *what problem each one solves relative to UCB1*, and roughly *how*. That's the level of depth this chapter targets.

---

## 5.2 UCB2: reducing switching frequency via epochs

**The problem UCB2 solves**: UCB1 recomputes the UCB score for every arm at every single round, and can, in principle, switch which arm it's pulling every round. In some production settings, switching which "option" is being served has a real operational cost (e.g., re-ranking overhead, cache invalidation, a business rule against changing prices too frequently) — you might want an algorithm that commits to an arm for a short streak before reconsidering.

**How it works**: UCB2 introduces a parameter $\alpha \in (0,1)$ and organizes pulls of each arm into **epochs** of geometrically increasing length. When arm $i$ is chosen, it isn't pulled just once — it's pulled repeatedly for $\lceil (1+\alpha)^{r_i} \rceil - \lceil (1+\alpha)^{r_i - 1}\rceil$ rounds in a row (where $r_i$ is arm $i$'s current epoch counter), and only *then* does the algorithm recompute UCB scores and potentially switch arms.

### Worked intuition (not a full trace — the bookkeeping is more involved than UCB1's)

Think of $\alpha = 0.5$. The first time arm $i$ "wins" the UCB comparison, it might get pulled just once or twice. The *next* time it wins, its epoch length roughly grows by a factor of $(1+\alpha)$ — so it might get pulled 3-4 times in a row. Each subsequent win grows the streak length geometrically. The key effect: **as an arm accumulates evidence of being good, UCB2 commits to it for longer uninterrupted streaks**, reducing the total number of arm-switches over the full horizon, while asymptotically preserving the same $O(\log T)$ regret shape (with a slightly better constant than UCB1 in some analyses).

**Interview-level takeaway**: UCB2 is about *engineering practicality* (fewer switches) more than a fundamentally different statistical idea — good to know it exists and why, but it's a lower-frequency interview topic than the next two variants.

---

## 5.3 KL-UCB: a distribution-aware confidence bound

This is the **most theoretically important** variant in this chapter, and the one most likely to come up in a rigorous interview.

**The problem KL-UCB solves**: Hoeffding's inequality is "distribution-agnostic" — it only assumes rewards are bounded in $[0,1]$. But if you *know* your rewards are Bernoulli (e.g., click/no-click — extremely common in ad/ranking systems), you're throwing away information by using a generic bound. KL-UCB uses the **Kullback-Leibler (KL) divergence** between Bernoulli distributions instead, producing a **provably tighter** confidence bound — and, remarkably, one that exactly matches the Lai-Robbins lower bound's constant (Chapter 2, Section 2.7), making KL-UCB **asymptotically optimal**, not just asymptotically the right *shape* like UCB1.

### The KL-divergence between two Bernoullis

For two Bernoulli distributions with means $p$ and $q$:

$$\text{KL}(p, q) = p \ln\frac{p}{q} + (1-p)\ln\frac{1-p}{1-q}$$

Intuitively: this measures "how statistically distinguishable is a coin with bias $p$ from a coin with bias $q$, from observed flips." When $p$ and $q$ are close together, $\text{KL}(p,q)$ is small (hard to tell apart from data) — when they're far apart, $\text{KL}(p,q)$ is large (easy to tell apart). This directly mirrors the Lai-Robbins bound's denominator from Chapter 2 — which is exactly why KL-UCB achieves the matching optimal constant.

### The KL-UCB index

Instead of Hoeffding's *additive* bonus term ($\hat\mu_i + \text{bonus}$), KL-UCB defines the upper confidence bound **implicitly**, as the largest value $q$ such that the KL-divergence from the sample mean $\hat\mu_i(t)$ to $q$ is still small enough given the amount of data:

$$\text{KL-UCB}_i(t) = \max\left\{q \in [\hat\mu_i(t), 1] : N_i(t) \cdot \text{KL}\big(\hat\mu_i(t), q\big) \leq \ln t + c\ln\ln t\right\}$$

(where $c$ is a small constant from the specific theoretical analysis — details vary slightly by paper). This looks intimidating, but the *intuition* is exactly parallel to UCB1: **"what's the highest value $q$ that's still statistically plausible given my data?"** — it's just that "plausible" is now measured with the *exactly correct* Bernoulli-specific yardstick (KL-divergence) instead of the generic Hoeffding yardstick.

### Worked numerical comparison (why this matters in practice)

Suppose an arm has been pulled $N_i(t) = 100$ times with sample mean $\hat\mu_i(t) = 0.10$ (a low-CTR ad, say), at round $t = 10{,}000$.

- **UCB1's bonus**: $\sqrt{2\ln(10{,}000)/100} = \sqrt{2 \times 9.21/100} = \sqrt{0.1842} \approx 0.429$, giving $\text{UCB}_1 \approx 0.10 + 0.429 = 0.529$.
- **KL-UCB's bound** would be computed by solving for the largest $q$ satisfying $100 \cdot \text{KL}(0.10, q) \leq \ln(10{,}000) \approx 9.21$. Without walking through the full numerical root-finding here, the qualitative result (well-established in the literature) is that this $q$ comes out **meaningfully lower than 0.529** — often close to $0.20$–$0.25$ in cases like this — because KL-divergence "knows" that for a mean this close to 0, the plausible range of true values is naturally asymmetric and tighter than the symmetric $\pm$ bonus that Hoeffding produces.

**Why this matters**: a tighter, more accurate upper bound means KL-UCB is less "falsely optimistic" about arms that are actually unlikely to be good — translating into meaningfully lower regret in practice, especially for low-probability-of-success settings (very common in ads/CTR, where true click rates are often just a few percent).

---

## 5.4 Bayes-UCB: a Bayesian reframing

**The idea**: instead of deriving a confidence bound from a frequentist concentration inequality (Hoeffding or KL-based), put a **prior** distribution over each arm's unknown mean, update it into a **posterior** as data comes in (exactly the machinery we'll build in full in Chapter 6 for Thompson Sampling), and define the UCB index as a high quantile (e.g., the 95th percentile) of that posterior:

$$\text{Bayes-UCB}_i(t) = Q\Big(1 - \frac{1}{t}, \; \text{posterior of } \mu_i \text{ given data so far}\Big)$$

where $Q(\cdot)$ denotes the quantile function. For Bernoulli rewards with a Beta prior (the same conjugate setup we'll use for Thompson Sampling), this posterior has a closed form, and the quantile is computable directly from the Beta distribution.

**Why this is worth knowing, even briefly**: Bayes-UCB is the conceptual bridge between "UCB-style" thinking (pick the arm with highest *upper bound*) and "Thompson-Sampling-style" thinking (use a full posterior distribution, not just a point estimate). It shows these two algorithm families — which can seem like completely different philosophies — are actually two different ways of using the **same underlying Bayesian machinery**: UCB-style approaches take a *specific quantile* of the posterior; Thompson Sampling (next chapter) *samples randomly* from the posterior. Recognizing this connection unprompted is a strong interview signal, and we'll return to it explicitly at the start of Chapter 6.

---

## 5.5 MOSS: optimizing the worst case

Recall from Chapter 2 (Section 2.8) the distinction between **problem-dependent** bounds (depend on the actual gaps $\Delta_i$) and **problem-independent / minimax** bounds (hold for the worst possible configuration of arms). UCB1's problem-dependent bound was $O\left(\sum_i \frac{\ln T}{\Delta_i}\right)$ — but if some $\Delta_i$ is very small (two arms nearly tied), this bound can become very large, even though the *minimax* rate for that same problem is known to be no worse than $O(\sqrt{KT})$.

**MOSS (Minimax Optimal Strategy in the Stochastic case)** modifies the bonus term to be:

$$\text{MOSS}_i(t) = \hat\mu_i(t) + \sqrt{\frac{\max\left(0, \ln\frac{T}{K \cdot N_i(t)}\right)}{N_i(t)}}$$

Two differences from UCB1 worth flagging precisely:
1. It uses the **known horizon $T$** directly (UCB1 uses the *current* round $t$, adapting as it goes — MOSS assumes $T$ is fixed and known in advance, which is a real practical constraint/limitation worth noting).
2. The $\max(0, \cdot)$ term means the bonus can hit **exactly zero** once an arm has been pulled "enough" relative to $T$ and $K$ — a qualitatively different behavior from UCB1's bonus, which (as we emphasized in Chapter 4, Section 4.5) is never exactly zero for any finite $N_i(t)$.

**Why this achieves better worst-case performance**: by capping the exploration bonus more aggressively once an arm has clearly been sampled "enough" (relative to the total horizon and number of arms), MOSS avoids UCB1's tendency to keep exploring near-tied arms more than the worst-case analysis would recommend — trading away a small amount of the problem-dependent bound's sharpness for a much better guarantee in the hardest possible arm configurations.

---

## 5.6 Side-by-side summary table

| Variant | Key idea | Needs $T$ known in advance? | Best suited for |
|---|---|---|---|
| **UCB1** | Hoeffding-based additive bonus | No | General-purpose baseline; distribution-agnostic |
| **UCB2** | Geometric epochs, fewer switches | No | Settings where switching arms has real operational cost |
| **KL-UCB** | KL-divergence-based bound, tight for Bernoulli | No | CTR/conversion-style Bernoulli rewards; best problem-dependent constant |
| **Bayes-UCB** | Posterior quantile as the index | No | Conceptual bridge to Thompson Sampling; needs a prior |
| **MOSS** | Horizon-aware, hits exactly zero eventually | Yes | When worst-case (minimax) guarantees matter more than typical-case performance |

---

## 5.7 Production considerations

- **KL-UCB is the variant most worth actually implementing** in a real ads/ranking system with Bernoulli-like (click/no-click, convert/no-convert) rewards, precisely because most production reward signals in this space genuinely are Bernoulli or close to it — the "distribution-agnostic" generality of UCB1 is *not* buying you anything you need, so you might as well use the tighter, distribution-matched bound.
- **MOSS's requirement to know $T$ in advance** is a real practical limitation — many production systems run indefinitely (no fixed horizon), which makes MOSS awkward to apply directly without some workaround (e.g., periodically resetting with an assumed horizon, or using an "anytime" variant). This is a good example of a theoretically elegant algorithm having a real deployment friction point — worth naming if asked "would you actually use MOSS in production?"
- **UCB2's epoch-based switching reduction** maps onto real infrastructure concerns: if "which arm to serve" maps to something like a pricing decision or a UI variant that has real cost to change frequently, algorithms explicitly designed to reduce switching frequency (like UCB2) are more than just theoretical curiosities.

---

## 5.8 Interview traps

- **Treating all UCB variants as interchangeable "tighter versions of UCB1" without being able to say what specifically each one improves.** A weak answer says "there are some more advanced UCB variants." A strong answer names the specific axis each improves along (switching frequency for UCB2, distribution-matched tightness for KL-UCB, Bayesian framing for Bayes-UCB, worst-case guarantees for MOSS).
- **Presenting KL-UCB's index formula and being unable to explain why it's tighter than Hoeffding in plain English.** The correct plain-English answer: Hoeffding is generic/distribution-agnostic; KL-divergence uses the *actual* shape of the Bernoulli likelihood, so it doesn't waste "confidence budget" on distributional possibilities that a Bernoulli reward simply cannot produce.
- **Not knowing MOSS requires the horizon $T$ in advance** — this is a specific, checkable fact interviewers can probe directly ("does this algorithm need to know how long it'll run?").

---

## 5.9 L5-vs-L6 differentiating talking points

- **L5 bar**: aware that UCB1 has variants, and can name at least KL-UCB and roughly why it's tighter (distribution-specific vs. distribution-agnostic).
- **L6 bar**:
  - Explicitly connects Bayes-UCB to Thompson Sampling as "same posterior machinery, different way of using it" (quantile vs. sample) — before even reaching Chapter 6, showing genuine synthesis rather than chapter-by-chapter memorization.
  - Can state precisely *why* MOSS's bonus term is allowed to hit exactly zero while UCB1's cannot, and connect this back to the problem-dependent vs. problem-independent distinction from Chapter 2 unprompted.
  - Gives an implementation-informed opinion on which variant they'd actually deploy for a *specific* stated production scenario (e.g., "for CTR prediction with Bernoulli rewards and no fixed campaign end date, I'd reach for KL-UCB over MOSS, because I don't have a natural fixed horizon and Bernoulli-specific tightness matters more here than worst-case guarantees") — this kind of scenario-conditioned recommendation is the clearest L6 signal in this whole chapter.

---

## 5.10 Comprehension checks

1. What specific limitation of UCB1's Hoeffding-based bonus does KL-UCB address, and how?
2. Why does KL-UCB's bound become *tighter* than UCB1's specifically when the sample mean is far from 0.5 (very low or very high observed rates)?
3. What does UCB2 change relative to UCB1, and what practical/operational problem does that change solve?
4. What does Bayes-UCB have in common with Thompson Sampling (which we haven't formally covered yet, but can reason about from this description)? What's the key difference?
5. Why does MOSS require the horizon $T$ to be known in advance, and what production limitation does that create?

---

*Next: Chapter 6 — Thompson Sampling: Foundations, where we build the full Bayesian machinery (priors, posteriors, and posterior sampling) that Bayes-UCB only hinted at — starting with the Beta-Bernoulli conjugate model and a full hand-traced numerical example.*
