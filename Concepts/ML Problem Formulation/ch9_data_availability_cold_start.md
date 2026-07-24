# Chapter 9: Data Availability & Cold-Start Framing

## 1. Why This Chapter Exists

Every chapter so far has implicitly assumed you have *enough* of the right kind of data. Real formulation work constantly runs into the opposite: a brand-new user with no history, a brand-new item with no interactions, a brand-new market with no local behavioral data, or a rare event with too few positive examples to train on directly. This chapter is about how to *reformulate* the problem itself — not just "collect more data and wait" — when data is structurally unavailable at decision time.

## 2. The Three Classic Cold-Start Cases

**User cold-start**: a new user has no interaction history to personalize from. **Item cold-start**: a new item has no interaction history to be recommended from (nobody has clicked/rated it yet, so collaborative-filtering-style signals are entirely absent). **System cold-start**: an entirely new product/market has neither user nor item history.

These require different reformulations, not the same fix:

- **User cold-start fix**: fall back to content-based or context-based features that don't require history — demographics (where available and appropriately used), device type, referral source, explicit onboarding preferences ("pick 3 topics you like") — and blend toward personalized collaborative signals as history accumulates.
- **Item cold-start fix**: fall back to content-based similarity (item metadata, embeddings from item description/image, category) to place a new item "near" similar existing items in embedding space, enabling recommendation before any interaction data exists at all.
- **System cold-start fix**: transfer learning from a related, more mature system (e.g., bootstrap a new market's ranking model using a mature market's model as a starting point, or as a source of transferable embeddings), combined with simple rule-based heuristics until enough local data accumulates.

## 3. Worked Numerical Example: Bandit-Style Exploration to Solve Item Cold-Start

Suppose a new item is added with zero interaction history. A pure content-based estimate gives it a predicted relevance score of $\hat r = 0.55$ for a given user segment, but this estimate has high uncertainty (few similar historical items to calibrate against). Standard "always show highest-predicted-relevance item" ranking would rank this uncertain item purely on its point estimate, potentially burying a genuinely great new item behind confidently-known mediocre old items forever (since it would never surface if it's never in the top slots — a rich-get-richer dynamic).

**Upper Confidence Bound (UCB)-style formulation** addresses this directly by ranking on an *optimistic* estimate rather than the raw point estimate:

$$
\text{UCB score} = \hat r + \alpha \sqrt{\frac{\ln(t)}{n}}
$$

where $n$ is the number of times this item has been shown/observed, $t$ is total elapsed rounds, and $\alpha$ controls exploration strength. **Numeric illustration**: new item ($n=5$ impressions so far) with $\hat r = 0.55$, at $t=10{,}000$, $\alpha=1$: exploration bonus $= \sqrt{\ln(10000)/5} = \sqrt{9.21/5} = \sqrt{1.84} = 1.36$ — a huge bonus relative to the 0–1 relevance scale, meaning this barely-tested item gets ranked far above its raw point estimate would suggest, deliberately trading a short-term relevance cost for the information gained by testing it more. Compare to an established item with $n=5{,}000$ impressions and the same $\hat r=0.55$: bonus $=\sqrt{9.21/5000}=\sqrt{0.00184}=0.043$ — negligible, since we're already confident in its estimate and don't need to keep "exploring" it. This numeric contrast is the whole point of UCB: **the exploration bonus shrinks as $n$ grows**, so the system naturally transitions from exploring uncertain new items to exploiting well-estimated ones, without needing a hand-tuned schedule.

## 4. Transfer Learning as a Formulation Choice, Not Just an Optimization Trick

When a new market/system has too little local data to train from scratch, the *formulation* changes from "train a model on local data" to "adapt a model pretrained on a related, data-rich domain." This has real quantitative implications for how you set up the problem:

- **Full fine-tuning** (updating all parameters on local data): needs a non-trivial minimum amount of local labeled data to avoid catastrophic forgetting of the useful pretrained structure — as a rule of thumb, teams often start noticing degraded transfer benefit below roughly a few thousand local labeled examples for a moderately complex model, though this varies significantly by model size and task.
- **Feature/embedding reuse** (freeze a pretrained encoder, train only a small task-specific head on top): works with far less local data (sometimes just hundreds of examples) because far fewer parameters are being fit locally, at some cost to how well the frozen representation matches local nuances.
- **Formulation implication**: the *choice* of how much of the pretrained model to freeze vs. fine-tune is itself a data-availability-driven formulation decision, not a separate hyperparameter search — you should state upfront, given an estimate of available local data volume, which regime you're in, rather than trying every option experimentally without an a priori expectation of which will work.

## 5. Reformulating Around Rare Events (Beyond Cold-Start)

A related but distinct problem: not "no data" but "too little positive-class data to train a direct supervised model well" (e.g., a rare catastrophic-failure event with only 12 historical positive examples total). Formulation options, each changing what's actually being predicted:

- **Reformulate as anomaly detection** (Chapter 2's task taxonomy): model the distribution of "normal" and flag deviations, sidestepping the need for many positive labels entirely, at the cost of not directly optimizing for the specific rare failure mode.
- **Reformulate as a proxy/surrogate prediction task with abundant labels** that's believed to correlate with the rare event (e.g., predict a precursor signal that occurs far more often than the rare event itself and is believed to be causally upstream of it), explicitly trading target fidelity for trainability — echoing Chapter 1's proxy-target tradeoff, but now driven by data scarcity rather than by label cost.
- **Synthetic data augmentation**, when physically/statistically justified (e.g., simulate additional rare-failure scenarios from a physics-based or statistical model of the failure mechanism) — with the important caveat that a model trained substantially on synthetic data inherits all the assumptions baked into the simulator, and validating against real (however few) positive examples remains essential before trusting the synthetic-augmented model.

## 6. Production Considerations

- **Cold-start fallback logic needs an explicit, monitored transition rule** — e.g., "use content-based scoring until $n \ge 50$ interactions, then blend in collaborative signal proportionally" — an unmonitored, ad-hoc transition creates inconsistent behavior that's hard to debug when something looks wrong for "new-ish" items specifically.
- **Exploration (Section 3) has a real, quantifiable cost** — showing users lower-confidence content to gather information trades off measurable short-term engagement for long-term catalog health; this tradeoff should be sized and agreed upon with stakeholders (echoing Chapter 8's Pareto-frontier framing), not treated as a free win.
- **Transfer learning's "which regime are we in" decision (Section 4) should be revisited as local data accumulates** — a system correctly started in feature-reuse mode with 300 local examples should have an explicit plan (and trigger) to migrate to full fine-tuning once, say, 5,000+ examples accumulate, rather than staying frozen in the initial regime indefinitely by default.
- **Synthetic-data-augmented models (Section 5) need ongoing validation against however few real positive examples exist**, on a recurring cadence, specifically because simulator drift or unmodeled real-world factors can silently invalidate the synthetic data's usefulness over time.

## 7. Common Interview Traps

- **Treating "not enough data" as a dead end rather than a formulation-change signal** — the strongest answers immediately pivot to "here's how I'd reformulate the problem," not "we need to wait and collect more data" as the only answer.
- **Proposing pure exploitation (always show highest point-estimate) for new items**, missing the systemic rich-get-richer bias this creates and not proposing an exploration mechanism (Section 3).
- **Not distinguishing cold-start (genuinely zero data) from merely rare-event/imbalanced (some data, just very little of the positive class)** — Section 5's reformulations are meaningfully different from Section 2's, and conflating them signals a shallow read of the problem.
- **Assuming full fine-tuning is always the right transfer-learning choice regardless of local data volume** (Section 4) — a common gap when local data is genuinely too sparse to support it.

## 8. L5-Differentiating Talking Points

- Explicitly separate user/item/system cold-start (Section 2) and name a distinct fix for each, rather than a single generic "use content-based features" answer for all three.
- Bring up an explore/exploit framing (UCB or similar) unprompted for any item/content cold-start scenario, and quantify the exploration-cost tradeoff rather than treating exploration as free.
- Explicitly state the local-data-volume threshold that would change your transfer-learning strategy (full fine-tune vs. frozen-feature-reuse), showing you treat this as a principled, data-driven formulation decision rather than a default choice.
- For rare-event problems, explicitly distinguish "reformulate as anomaly detection" from "find a correlated abundant proxy" from "synthetic augmentation" as three genuinely different strategies with different validity assumptions, rather than treating "get more data" as the only lever available.

## 9. Comprehension Checks

1. A grocery delivery app launches in a brand-new city with zero local order history but has mature models in 50 other cities. Using Section 4's framing, propose a transfer strategy and state what local data volume milestone would change your recommendation.
2. Using the UCB formula in Section 3, compute the exploration bonus for an item with $n=20$ impressions at $t=50{,}000$, $\alpha=1$, and explain qualitatively how this bonus would change if $\alpha$ were doubled.
3. A rare catastrophic machine-failure event has only 8 historical positive examples across 5 years of data. Propose two of the three reformulation strategies from Section 5 for this specific scenario, and state one validity risk for each.
4. Explain why "always rank by highest point-estimate relevance" creates a systemic bias against new items even when some new items are, in truth, excellent — and connect this back to why exploration bonuses (Section 3) shrink as $n$ grows rather than staying constant.

---

*End of Chapter 9. Chapter 10 will cover baseline design — non-ML baselines, heuristic baselines, and what "beating baseline" should actually mean before investing in a full ML system.*
