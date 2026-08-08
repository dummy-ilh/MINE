# Chapter 13 — Non-Stationary & Structured Bandits

*(Same slower, simpler style — plain language first, light on notation.)*

---

## 13.1 What this chapter covers, in one sentence each

Three separate real-world complications, each getting its own short section:

1. **Non-stationary bandits**: what if the "best answer" changes over time?
2. **Combinatorial bandits**: what if you need to pick a whole *list* of things, not just one?
3. **Dueling bandits**: what if you never get a direct reward at all — only "which of these two did the user prefer"?

Each of these closes out Phase 2 by pointing at a real gap between the clean textbook setup and the messier real world.

---

## 13.2 Non-stationary bandits: the problem

Every algorithm so far assumes each arm's true mean $\mu_i$ is **fixed forever**. In real life, this is often false: an ad's true click rate decays over time as people get tired of seeing it (this is called "creative fatigue" in ad-industry language). A comedy show's appeal might spike right when a new season is released, then fade. **The "best arm" can genuinely change over the course of the game** — this is what "non-stationary" means: the statistics of the environment are *not* staying still ("stationary") over time.

**Why this breaks our earlier algorithms**: UCB1 and Thompson Sampling both build their confidence using *all* the data they've ever seen for an arm — old data and new data treated equally. If an arm used to be great and has since gotten worse, an algorithm that still remembers "lots of old great data" will keep over-trusting that arm for a long time after it's actually declined — a real, common failure mode in production systems that don't account for drift.

---

## 13.3 Fix #1: Sliding-Window UCB

**The idea, in plain words**: instead of using *all* historical data to compute an arm's sample mean and confidence bound, only look at the **most recent $W$ pulls** of that arm (a "window" of size $W$ — you pick this number in advance, e.g., "only look at the last 500 pulls"). Older data just gets thrown away and stops influencing the decision.

**Why this works**: if an arm's true mean shifts, the algorithm "forgets" the stale old data within about $W$ rounds, and starts re-learning based on fresh, current data — the confidence bound formula is otherwise exactly the same UCB1 idea from Chapter 4, just computed only over the recent window instead of the full history.

**The tradeoff on choosing $W$** (this should feel familiar — very similar to the explore-then-commit tradeoff from Chapter 3): a **small** $W$ adapts to changes quickly, but has less data to work with at any moment, so its estimates are noisier. A **large** $W$ gives smoother, more reliable estimates when things *aren't* changing, but reacts sluggishly when they do change. There's no universally "correct" $W$ — it depends on how fast you expect the real world to actually drift.

---

## 13.4 Fix #2: Discounted UCB

**The idea, in plain words**: instead of a hard cutoff (Sliding-Window UCB's "throw away everything older than $W$ rounds"), **gradually down-weight older observations** — the most recent pull counts fully, the one before that counts slightly less, the one before that counts even less, and so on, fading out smoothly rather than dropping off a cliff.

This is done with a **discount factor** $\gamma$ (a number just under 1, like $0.99$) — each round, all previously-accumulated evidence gets multiplied by $\gamma$ before adding in the newest observation. A round that happened 100 pulls ago ends up contributing $\gamma^{100}$ times as much weight as a round that just happened — a small number if $\gamma < 1$, meaning old evidence fades out smoothly over time, without a hard cutoff.

**Sliding-window vs. discounted, in one line**: sliding-window is "hard forget everything past a cutoff," discounted is "smoothly fade out old evidence" — both solve the same non-stationarity problem, just with a hard-edge vs. soft-edge flavor of "forgetting."

---

## 13.5 Combinatorial bandits: picking more than one thing at once

**The problem, in plain words**: so far, every round, you pick exactly **one** arm. But a lot of real systems don't work that way — think of a search results page: you're not picking one single result to show, you're picking (and ordering!) a whole **list** of, say, 10 results. Or think of a set of ad slots on a page — you're filling several slots at once, not just one.

**Combinatorial bandits** generalize the bandit problem to this "pick a *set* (or ranked list) of arms per round" setting. The reward you observe is typically for the **whole combination** together (e.g., "did the user click on *anything* in this list of 10 results"), which creates a genuinely new wrinkle: **you often can't cleanly tell which specific item in your list of 10 deserves the credit** for the click — this is sometimes called the **credit assignment problem**, and it's a big part of what makes combinatorial bandits harder than plain single-arm bandits.

**How this is typically handled (high-level, not derived)**: many practical approaches make a simplifying assumption — e.g., assume each position/slot in the list independently contributes to the overall reward, so you can still reason about "how good is this specific item" even though you only observed one combined outcome for the whole list. This is closely related to ideas you may already associate with search/recommendation ranking (like position bias — users are more likely to click something near the top of a list regardless of its actual quality) — worth mentioning if this connection comes up.

**What's expected for an interview**: recognizing that "pick a list, not a single item" is a real, common, and meaningfully harder variant of the bandit problem, and being able to name the core new difficulty (credit assignment across the items in the combination) — full combinatorial-bandit algorithm derivations are a more specialized topic than this course goes deep on.

---

## 13.6 Dueling bandits: only relative feedback

**The problem, in plain words**: everything so far assumes you get a direct reward number (or at least a 0/1 click) for the arm you picked. Sometimes you genuinely can't get that — you can only ask a user to **compare two options and say which they prefer**, with no direct numeric reward at all. Think of: "here are two possible article headlines — which one would you rather click?" or, very relevantly to current ML practice, **this is exactly the structure behind RLHF (Reinforcement Learning from Human Feedback)** — human raters are typically asked "which of these two model responses is better," not "rate this response on a 1–10 scale," because relative comparisons tend to be much easier and more reliable for humans to give consistently.

**Dueling bandits** are the bandit framework built specifically for this setting: instead of pulling one arm and getting a reward, you pick a **pair** of arms, and observe only **which one won the comparison**. The goal is still to find the best arm overall (or minimize a regret-like quantity relative to always dueling with the true best arm), just using only this weaker, relative form of feedback.

**Why this connection to RLHF is worth knowing explicitly**: if you're interviewing for a role that touches LLM training/fine-tuning at all, being able to say *"preference-based reward modeling in RLHF is essentially a dueling-bandits-style feedback structure — comparisons rather than absolute rewards"* is a genuinely valuable, up-to-date connection that shows you can link classical bandit theory to how modern LLMs are actually trained.

---

## 13.7 Why this chapter matters for interviews, even at a lighter level of depth

None of these three topics (non-stationarity, combinatorial, dueling) need to be derived in full mathematical detail for most interviews — but **"the #1 follow-up question after any bandit whiteboard answer is almost always about non-stationarity"** (as flagged back in Chapter 1's syllabus overview), so being able to immediately and fluently say "I'd use sliding-window or discounted UCB to handle drift" the moment an interviewer asks "but what if the click rates change over time?" is one of the highest-value, lowest-effort pieces of prep in this entire course.

---

## 13.8 Production considerations (kept simple)

- **Real ad/content systems deal with non-stationarity constantly** — creative fatigue, seasonal trends, breaking news changing what's relevant — sliding-window and discounted variants of UCB/Thompson Sampling are standard, practical tools for this, and are much simpler to implement than they might sound (just a small tweak to how the sample mean/posterior gets computed).
- **Combinatorial bandits map directly onto ranking systems** (search results, feed ranking, ad slot allocation) — any time your product surfaces more than one recommendation at once, you're implicitly in combinatorial-bandit territory, even if the actual system in place uses simplifying heuristics rather than a full combinatorial-bandit algorithm.
- **Dueling bandits' connection to RLHF (Section 13.6) is a genuinely current, high-value talking point** given how central preference-based training has become to modern LLM development — worth having ready regardless of whether the interview is nominally "about bandits" or "about LLM training," since the connection can bridge both.

---

## 13.9 Interview traps (kept simple)

- **Not immediately recognizing "what if things change over time" as the standard non-stationarity follow-up**, and fumbling for an answer instead of confidently naming sliding-window or discounted UCB.
- **Describing combinatorial bandits as "just doing regular bandits multiple times."** The key new difficulty is specifically the **credit assignment problem** — not being able to cleanly attribute a combined reward back to individual items — glossing over this shows a shallow read of the topic.
- **Confusing dueling bandits with "just a bandit with two arms."** Dueling bandits are about the **type of feedback** (only relative/comparative, never an absolute reward), which applies no matter how many total arms/options exist — this is a completely different axis from "how many arms are there."

---

## 13.10 L5-vs-L6 differentiating talking points (kept simple)

- **L5 bar**: can name sliding-window and discounted UCB as fixes for non-stationarity, and can describe combinatorial and dueling bandits at a high, correct level.
- **L6 bar**:
  - Immediately and fluently answers a "but what about drift?" follow-up with sliding-window/discounted UCB, unprompted, exactly matching the "#1 follow-up question" framing from Section 13.7 — showing real interview-readiness, not just topic familiarity.
  - Explicitly names the credit-assignment problem for combinatorial bandits, and connects it to position bias in ranking systems.
  - Draws the dueling-bandits ↔ RLHF connection unprompted — a strong, current signal that shows the candidate connects classical theory to how state-of-the-art systems are actually built today.

---

## 13.11 Comprehension checks — plain words, minimal formulas

1. In one sentence, why do UCB1 and Thompson Sampling (as originally described) struggle when an arm's true mean drifts over time?
2. What's the difference between Sliding-Window UCB and Discounted UCB, in plain words?
3. What's the "credit assignment problem" in combinatorial bandits, and why does it arise?
4. In one sentence, what makes dueling bandits different from a regular two-armed bandit?
5. Why is the dueling-bandits framework relevant to how modern LLMs are trained via RLHF?

---

*This closes out Phase 2 (Contextual & Structured Bandits). Next: Chapter 14 — Off-Policy Evaluation: Importance Sampling, where Phase 3 begins — this is the material that tends to separate candidates who've only studied the algorithms from candidates who've actually reasoned about deploying them.*
