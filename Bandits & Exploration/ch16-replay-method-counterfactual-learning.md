# Chapter 16 — The Replay Method & Counterfactual Learning

*(Same slower, simpler style — plain language first, light on notation.)*

---

## 16.1 Why this chapter exists, given Chapters 14–15 already covered evaluation

Chapters 14 and 15 built up a genuinely rigorous statistical toolkit (IPS, clipping, SNIPS, Doubly Robust) — but there's one very simple, very widely-used practical method we haven't covered yet, and it's arguably **the single most commonly implemented off-policy evaluation technique in real industry systems**, precisely because of how simple it is. This chapter covers that method (the **Replay Method**), and then briefly looks at the natural next question: instead of just *evaluating* a candidate policy, can you *learn* a good policy directly from logged data in the first place?

---

## 16.2 The Replay Method — the simplest possible idea

**One extra requirement this method needs, stated up front**: it only works cleanly when the old logging policy $\pi_0$ picked arms **uniformly at random** (or you can restrict your analysis to only the portion of logged data where it effectively did) — this is a real constraint, and we'll come back to why it matters.

**The idea, in the simplest words possible**: go through your logged data, round by round, in order. For each round, check: **did the old policy's actual random choice happen to match what your new policy $\pi_1$ would have picked?** If yes — **keep that round**, and count its actual observed reward. If no — **just skip that round entirely, as if it never happened.** At the end, average the reward over only the kept rounds.

That's it. There's no dividing by propensity scores, no weighting, no reward model — just "keep matching rounds, throw away the rest, average what's left."

### Why this works cleanly (in plain words) when logging was uniform random

If the old policy picked arms **uniformly at random** (every arm equally likely, every round, regardless of context), then the rounds that happen to match $\pi_1$'s choice are automatically a **fair, representative, unbiased sample** of what would have happened under $\pi_1$ — no weighting correction is needed, because there was never any imbalance to correct for in the first place. This is the entire reason the Replay Method can skip all of IPS's propensity-score machinery: **it sidesteps the whole "some arms were shown more than others" problem by requiring, up front, that this imbalance never existed to begin with.**

---

## 16.3 A very simple worked example

Reuse the same 5 logged rounds as Chapter 14 — but now assume $\pi_0$ was choosing uniformly at random among 3 arms (so every $p_0$ genuinely was $1/3 \approx 0.33$, unlike Chapter 14's varying propensities — this is the "uniform random logging" requirement made concrete):

| Round | Arm shown | Reward | Would $\pi_1$ pick the same arm? |
|---|---|---|---|
| 1 | Comedy | 1 | Yes |
| 2 | Documentary | 0 | No |
| 3 | Comedy | 0 | Yes |
| 4 | Action | 1 | No |
| 5 | Documentary | 1 | Yes |

**Replay Method**: keep rounds 1, 3, 5 (the "Yes" rounds), throw away rounds 2 and 4 entirely.

$$\text{Replay estimate} = \frac{1 + 0 + 1}{3} = \frac{2}{3} = 0.667$$

Notice how much simpler this computation was — just a plain average of the kept rounds, no division by small numbers, no risk of a single rare round dominating the whole estimate the way it did with plain IPS in Chapter 14. **This is exactly why the Replay Method tends to be much lower-variance and easier to trust in practice than plain IPS** — at the cost of only working correctly when the uniform-random logging requirement genuinely holds.

---

## 16.4 The real practical limitation, stated plainly

Here's the catch, and it's a big one: **most production bandit policies are not uniformly random** — that would mean deliberately showing bad arms just as often as good ones, purely for evaluation purposes, which is expensive in terms of real business results (lots of "wasted" traffic on obviously-worse options). Real systems mostly want to run something smarter (UCB, Thompson Sampling, LinUCB) that concentrates traffic on good arms — but that smartness is exactly what breaks the Replay Method's clean uniform-random assumption.

**Common practical compromises**:
- Carve out a **small slice of traffic** (e.g., 5%) that genuinely is served uniformly at random, purely to support future Replay-Method-style evaluation — sacrificing a little performance on that slice in exchange for clean, trustworthy offline evaluation data going forward. This is a very common, very real production pattern, sometimes just called an "exploration bucket" or "random logging bucket."
- Fall back to the IPS-family methods from Chapters 14–15 (which don't require uniform logging, just *known* propensities, whatever they happen to be) when you don't have a uniform-random slice available.

**Simple interview-ready framing**: *"The Replay Method is simpler and lower-variance than IPS, but it requires uniformly random logging data, which most production systems don't naturally produce — so companies often deliberately carve out a small random-traffic bucket specifically to enable this kind of clean, simple offline evaluation later."*

---

## 16.5 Counterfactual (risk) learning — the natural next question, kept high-level

Chapters 14–16 so far have all been about **evaluating** a specific candidate policy using logged data. A closely related but distinct question: **can you use logged data to directly search for and train a *good* new policy, rather than one-at-a-time evaluating policies you've already designed?**

This is generally called **counterfactual learning** or **counterfactual risk minimization**. The core idea, kept simple: instead of picking one specific candidate $\pi_1$ and asking "how good is this one?", you set up an optimization problem — search over a whole *family* of possible policies, using an IPS-style (or Doubly-Robust-style) estimate of each candidate's performance as the thing you're trying to maximize, and let an optimization procedure search for the best-scoring policy within that family, directly from the logged data — without ever needing new live traffic during the search itself.

**What's expected for an interview**: recognizing this as the natural "next step up" from off-policy *evaluation* — evaluation asks "how good is this one specific policy," counterfactual learning asks "search over many possible policies and find a good one" — using fundamentally the same statistical machinery (IPS-style corrections) as the engine driving the search. Full derivations of counterfactual learning objectives are a more advanced, specialized topic than this course goes deep on — the plain-language framing above is the right depth for most interview conversations.

---

## 16.6 Production considerations (kept simple)

- **The Replay Method (Li et al.'s well-known formulation) is genuinely the most widely cited practical off-policy evaluation approach in published industry case studies** — its simplicity (no weighting, no model needed) makes it easy to implement, easy to explain to non-technical stakeholders, and easy to trust, whenever the uniform-random logging requirement can be arranged.
- **The "exploration bucket" pattern (Section 16.4) is a real, common piece of production infrastructure** — worth naming directly if asked how you'd set up a system to support this kind of ongoing, cheap policy evaluation, since it's a concrete, buildable answer rather than an abstract statistical concept.
- **Counterfactual learning connects directly to a broader theme you may see elsewhere in ML systems**: training directly on logged, biased data (rather than fresh, unbiased live data) is a recurring challenge across recommendation systems and ranking, not something unique to bandits — worth mentioning this broader connection if the conversation invites it.

---

## 16.7 Interview traps (kept simple)

- **Applying the Replay Method to logs from a non-uniform-random policy (like UCB1 or Thompson Sampling) without flagging the mismatch.** This is a serious, checkable error — the whole method's simplicity depends entirely on the uniform-random logging assumption; using it on non-uniform logs silently reintroduces the exact bias problem it was designed to avoid.
- **Confusing off-policy evaluation (Chapters 14–16 so far) with counterfactual learning (Section 16.5).** Evaluation scores one candidate; learning searches over many candidates for a good one. They share underlying machinery, but they answer different questions — mixing them up in an interview answer signals imprecision.
- **Presenting the Replay Method as strictly "worse" or "more primitive" than IPS/DR, rather than as a clean, low-variance option that's simply more restrictive about when it applies.** When its uniform-random assumption genuinely holds, the Replay Method is often the *preferred*, simplest choice — not a fallback.

---

## 16.8 L5-vs-L6 differentiating talking points (kept simple)

- **L5 bar**: correctly describe the Replay Method's "keep matching rounds, throw away the rest" mechanism, and know it requires uniform-random logging.
- **L6 bar**:
  - Proactively names the "exploration bucket" pattern (Section 16.4) as a concrete, real production solution to the Replay Method's core limitation, rather than treating the limitation as a dead end.
  - Clearly distinguishes off-policy evaluation from counterfactual learning (Section 16.5) when asked to compare them, using the "evaluate one vs. search over many" framing precisely.
  - Can reason about *when* the Replay Method's simplicity is actually preferable to IPS/DR's added complexity (i.e., whenever a genuine uniform-random slice of traffic is available) — showing situational judgment rather than treating "more sophisticated method" as automatically "better method."

---

## 16.9 Comprehension checks — plain words, minimal formulas

1. In one sentence, what does the Replay Method do with logged rounds where the new policy would have disagreed with the old policy's actual choice?
2. Why does the Replay Method not need to divide by propensity scores the way IPS does?
3. What's the practical limitation that makes the Replay Method hard to apply to most real production logging data, and what's a common workaround?
4. In one sentence, how does counterfactual learning differ from off-policy evaluation?
5. Why might a company deliberately serve a small slice of traffic uniformly at random, even though it's not the "best" policy for that slice in the moment?

---

*Next: Chapter 17 — Production Systems & Decision Frameworks, closing out Phase 3 with the bandits-vs-A/B-testing decision framework, cold-start and warm-starting, multi-objective bandits, and real-world case studies from ad ranking, feed ranking, and assistant suggestion systems.*
