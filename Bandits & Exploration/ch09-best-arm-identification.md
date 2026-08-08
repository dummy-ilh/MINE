# Chapter 9 — Best-Arm Identification (Pure Exploration)

---

## 9.1 A completely different objective

Every algorithm so far has optimized **cumulative regret**: you're playing "for keeps," and every pull's reward counts toward your score, so you need to balance exploring (to learn) against exploiting (to cash in on what you've already learned).

**Best-arm identification (BAI)**, also called **pure exploration**, throws out that scoring entirely. The new goal: **you have a budget of pulls (or you want to use as few pulls as possible), and at the end, you must output your single best guess for which arm has the highest true mean.** You get **zero credit for the rewards observed during the pulls themselves** — only for whether your final answer is correct. This is a genuinely different problem, with genuinely different optimal algorithms, and interviewers specifically test whether candidates can tell the two objectives apart.

### A concrete framing to anchor this

Think of **offline model selection**: you've trained 5 candidate model architectures, and you want to find out which one has the best true validation accuracy, using as few (expensive) validation-set evaluations as possible. You don't care about "regret" during this process — nobody's serving live traffic to the models while you're testing them. You only care about correctly identifying the winner, using a minimal evaluation budget. This is a best-arm identification problem, not a regret-minimization problem — and recognizing which of the two framings applies to a given real scenario is itself a key interview skill.

---

## 9.2 The PAC bandit formulation

Best-arm identification is typically formalized using **PAC** guarantees — "Probably Approximately Correct," a framework borrowed from learning theory. Two parameters define the guarantee:

- $\varepsilon \geq 0$: how close to truly-optimal your answer needs to be (an "approximately" tolerance — $\varepsilon = 0$ means you must find the *exact* best arm; $\varepsilon > 0$ allows outputting any arm within $\varepsilon$ of the best arm's mean)
- $\delta \in (0,1)$: the allowed failure probability (the "probably" — you're allowed to be wrong with probability at most $\delta$)

**Goal**: design an algorithm that, using as few pulls as possible, outputs an arm $\hat{i}$ such that:

$$P\big(\mu_{\hat i} \geq \mu^* - \varepsilon\big) \geq 1 - \delta$$

In plain English: *"with probability at least $1-\delta$, the arm I output is within $\varepsilon$ of the true best arm."* Notice the two knobs trade off against the pull budget in the natural direction: smaller $\varepsilon$ (needing a more precise answer) or smaller $\delta$ (needing more confidence) both require **more** pulls — there's a real, provable tradeoff, and algorithms in this chapter are judged on how few pulls they need to hit a given $(\varepsilon, \delta)$ target.

---

## 9.3 Successive Elimination

This is the cleanest, most interview-friendly BAI algorithm — start here, since it makes the core mechanic (statistically-justified arm elimination) extremely transparent.

**Algorithm**: maintain a set $S$ of "still active/plausible" arms, initialized to all $K$ arms. Proceed in rounds; in each round, pull **every** currently-active arm once (round-robin), then check: for each active arm $i$, compute a confidence interval around $\hat\mu_i$ (a Hoeffding-style bound, very similar in spirit to UCB1's bonus term from Chapter 4). **Eliminate** any arm $i$ from $S$ whose confidence interval's *upper* bound is below some other active arm $j$'s confidence interval's *lower* bound — i.e., eliminate arm $i$ once you're statistically confident it's worse than some other specific arm. Stop when only one arm remains in $S$ (that's your answer), or when a pull budget is exhausted.

### Worked trace intuition

Using our familiar means $\mu_1=0.30, \mu_2=0.50, \mu_3=0.45$: early on, all three arms are active, and their confidence intervals are wide (little data), so no elimination happens yet — you keep round-robining. As pulls accumulate, arm 1's confidence interval (centered near 0.30) will, with enough data, have its **upper bound** drop below arm 2's (and eventually arm 3's) **lower bound** — at that point, arm 1 gets **eliminated**, and all future rounds only round-robin between the remaining active arms (arms 2 and 3), which lets their confidence intervals narrow *faster* per-round (since pulls are no longer being "spent" on arm 1 at all). Eventually arm 3's confidence interval similarly falls below arm 2's, arm 3 gets eliminated, and arm 2 (the true best) is the last one standing — the algorithm halts and outputs arm 2.

**Why this is elegant**: the algorithm automatically concentrates its remaining pull budget on the arms that are hardest to distinguish (the ones still active), while cheaply "writing off" clearly-worse arms early — directly parallel to the Lai-Robbins "hard-to-distinguish arms need more data" theme that has run through this entire course since Chapter 2.

---

## 9.4 LUCB (Lower-Upper Confidence Bound)

Successive elimination round-robins over *all* active arms every round — a reasonable but not maximally efficient use of pulls. **LUCB** refines this by being more targeted about which two arms to pull each round, focusing pulls specifically where they're most needed to resolve the current biggest ambiguity.

**Algorithm (high-level)**: at each round, look at your current empirical best arm and identify the single most "threatening" contender — specifically:
1. Let $h(t) = \arg\max_i \hat\mu_i(t)$ — your current best-looking arm ("h" for "highest").
2. Let $l(t) = \arg\max_{i \neq h(t)} \text{UCB}_i(t)$ — among *all other* arms, whichever one has the highest upper confidence bound (i.e., the strongest remaining challenger — "l" for the arm with the highest **l**ower-bound-competing UCB).
3. Pull **both** $h(t)$ and $l(t)$ this round (just these two, not a full round-robin over every arm).
4. Stop once $h(t)$'s confidence **lower** bound exceeds $l(t)$'s confidence **upper** bound (i.e., you're now statistically confident $h(t)$ truly beats its strongest remaining challenger, which — by construction — means it beats every other arm too).

**Why this is more efficient than successive elimination**: LUCB doesn't waste pulls on arms that are already-clearly-worse than *both* the current leader and the current top challenger — it narrowly focuses pulls exactly where the outcome is still genuinely uncertain (the leader-vs.-top-challenger race), rather than continuing to round-robin pulls across every arm still nominally "active." This targeted-pulling idea is a recurring pattern in efficient BAI algorithms generally.

---

## 9.5 Racing algorithms — the general family and vocabulary

"**Racing algorithm**" is the general umbrella term for this style of approach: successively eliminate clearly-inferior options while continuing to "race" the remaining plausible contenders against each other, allocating a shrinking pool of candidates an increasing share of the sampling budget as weaker options drop out. Successive Elimination (Section 9.3) and LUCB (Section 9.4) are both racing algorithms; other named variants (Sequential Halving, Exponential-gap elimination, and others) exist in the literature with different specific rules for how aggressively to eliminate and how to allocate pulls among survivors, but they all share this same "eliminate the clearly-worse, keep racing the plausible" DNA. **You don't need to know every named variant** — knowing the successive-elimination and LUCB mechanics in Sections 9.3–9.4 well, plus this umbrella vocabulary term, covers the interview-relevant depth for this topic.

---

## 9.6 Connection to hyperparameter search and Bayesian Optimization

This is a valuable, concrete bridge worth being explicit about, especially since it connects back to material you've already studied (your ANOVA/DOE curriculum's coverage of Response Surface Methodology and Bayesian Optimization).

**Hyperparameter search is best-arm identification** when framed as: "I have a finite/discrete set of candidate hyperparameter configurations, each with an unknown true validation performance (noisy due to random seeds, data splits, etc.), and I want to find the best one using a limited compute budget of training runs." This is *exactly* the BAI setup from Section 9.1's "offline model selection" framing.

**Successive Halving** (a specific, very widely-used racing algorithm in the ML systems literature, and the direct ancestor of the popular **Hyperband** algorithm) applies exactly this racing philosophy: start with many candidate configurations, each given a small compute budget; eliminate the worse-performing half; double the remaining budget for survivors; repeat. This is successive elimination's core idea (Section 9.3), specifically adapted to the hyperparameter-search setting where "pulling an arm" means "training a model configuration for a bit longer." **Being able to draw this connection explicitly — "best-arm identification is the theoretical foundation underlying Successive Halving and Hyperband, which I'd actually use for hyperparameter tuning in practice" — is a strong, concrete, practically-grounded interview answer**, and directly demonstrates the kind of cross-topic synthesis that separates strong candidates.

---

## 9.7 Production considerations

- **BAI is the right framework whenever you're doing offline evaluation/testing before a launch decision**, not live optimization — e.g., picking a winning creative from a set of ad variants using a controlled test budget, or picking the best of several candidate ranking models using held-out evaluation traffic, where you don't care about the "regret" incurred during testing (because it's explicitly a controlled test phase, not the live-serving phase) — only about correctly identifying the winner before full rollout.
- **The $(\varepsilon, \delta)$ framing maps directly onto practical experiment-design decisions**: choosing $\delta$ is essentially choosing your statistical significance threshold (directly analogous to concepts from your A/B testing curriculum), and choosing $\varepsilon$ is choosing how much of a "practically insignificant" performance gap you're willing to tolerate without needing to keep testing — this is a genuinely useful frame for justifying sample-size/budget decisions to stakeholders in real experiment design conversations.
- **Racing algorithms (Section 9.5) are directly used in industrial hyperparameter-tuning infrastructure** — Hyperband and its descendants (e.g., ASHA — Asynchronous Successive Halving) are standard components of production ML platforms (used in tools like Ray Tune) — so this chapter's content has very concrete, nameable production tooling behind it, not just theory.

---

## 9.8 Interview traps

- **Conflating best-arm identification with regret minimization**, and applying UCB1 or Thompson Sampling (regret-minimizing algorithms) to a stated pure-exploration problem without noting the mismatch. A candidate who's asked "how would you find the best of these 5 model configurations using a fixed compute budget" and immediately reaches for UCB1/TS without flagging the objective mismatch is signaling they haven't internalized this chapter's central distinction. (In fact, using a *regret-minimizing* algorithm for a BAI problem is provably suboptimal — regret-minimizing algorithms deliberately keep pulling the current-best-looking arm a lot, which is the *wrong* pull-allocation strategy when your only goal is confidently identifying the winner, not accumulating reward along the way.)
- **Forgetting that BAI algorithms get zero credit for the rewards observed during the identification process itself** — the entire objective structure is different, and reasoning about an algorithm's "regret" during a BAI process is a category error worth being able to name explicitly.
- **Being unable to state what makes LUCB more efficient than successive elimination** — the key, checkable idea is the targeted "current leader vs. strongest remaining challenger" pulling rule from Section 9.4, versus successive elimination's full round-robin over all active arms.

---

## 9.9 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly distinguish BAI's objective from regret minimization, correctly describe the PAC $(\varepsilon,\delta)$ formulation, and correctly describe successive elimination's high-level mechanic.
- **L6 bar**:
  - Draws the Successive Halving / Hyperband connection (Section 9.6) unprompted when hyperparameter tuning comes up anywhere in the interview — a strong signal of practically-grounded cross-topic knowledge, not siloed textbook recall.
  - Explicitly explains *why* using a regret-minimizing algorithm (UCB/TS) for a stated pure-exploration problem is provably suboptimal, rather than just noting they're "different" — showing understanding of *why* the pull-allocation strategies genuinely conflict between the two objectives.
  - Connects the $(\varepsilon, \delta)$ framing to statistical-significance/practical-significance concepts from experiment design (Section 9.7) unprompted, demonstrating the ability to translate abstract PAC theory into stakeholder-facing experiment-design language.

---

## 9.10 Comprehension checks

1. In your own words, how is the objective of best-arm identification fundamentally different from the objective of regret minimization?
2. What do $\varepsilon$ and $\delta$ each control in the PAC bandit formulation, and how does each one trade off against the required pull budget?
3. Walk through why successive elimination automatically concentrates its remaining pull budget on the hardest-to-distinguish arms over time.
4. What specifically does LUCB do differently from successive elimination that makes it more pull-efficient?
5. Explain, concretely, why applying a regret-minimizing algorithm like UCB1 to a stated best-arm-identification problem is a mismatch — what would go wrong?

---

*This closes out Phase 1 (Core Stochastic Bandit Algorithms). Next: Chapter 10 — Contextual Bandits: Formulation, where we move from "which arm is best overall" to "which arm is best for this specific user or query" — the setting that maps most directly onto real production ad/recommendation/ranking systems.*
