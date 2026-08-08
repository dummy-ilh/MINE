# Bandits & Exploration — Interview Mastery Syllabus
### Target: Google / Apple L5 MLE & Data Scientist interviews
### Starting point: zero prior knowledge of bandits or exploration theory
### Prerequisites assumed: basic probability (expectation, variance), basic linear algebra — nothing about RL or bandits assumed

---

## How this curriculum is structured

Every chapter, when we work through it, will follow the same format you've used in prior curricula:

1. **Intuition first** — a plain-English mental model before any notation
2. **Formal setup / formula** — rigorous, in the notation actually used in papers and interviews
3. **Worked numerical example** — hand-traceable, not abstract
4. **Production considerations** — how this actually gets deployed at companies like Google/Apple (ads, search ranking, App Store/Siri suggestions, RecSys)
5. **Interview traps** — the specific ways candidates lose points on this topic
6. **L5-vs-L6 differentiating talking points** — what separates a pass from a strong pass
7. **Comprehension checks** — questions you should be able to answer cold before moving on

This document is the map. We'll produce one self-contained markdown reference per chapter (or small cluster of chapters) as we go, same as the SVM, outlier-analysis, and recsys curricula.

---

## Phase 0 — Foundations (Ch. 1–3)
*Goal: understand what problem bandits solve and how we measure success.*

**Ch 1 — The Multi-Armed Bandit Problem**
- Origin story (clinical trials → the "one-armed bandit" slot machine framing)
- Formal setup: arms, reward distributions, horizon T, sequential decision-making
- The exploration-exploitation tradeoff, stated precisely
- Bandits as a degenerate/1-step Markov Decision Process — how this connects to full RL
- Stochastic vs adversarial vs contextual bandit families (the taxonomy we'll fill in over the whole syllabus)

**Ch 2 — Regret: The Central Metric**
- Reward-based vs regret-based framing of "good policy"
- Expected regret vs pseudo-regret vs realized regret
- Why O(log T) regret is the gold standard, and where that number comes from
- The Lai-Robbins lower bound (asymptotic, problem-dependent) — stated and interpreted, not proved
- Problem-dependent vs problem-independent (minimax) regret bounds — why interviewers care about the distinction

**Ch 3 — Naive Baselines**
- Pure greedy and why it fails (worked example showing it locking onto a suboptimal arm)
- ε-greedy: algorithm, regret behavior (linear regret — why)
- ε-decay / ε_t = 1/t schedules
- Explore-then-commit (ETC): algorithm, regret analysis, optimal explore phase length
- Why every one of these is the "wrong answer" interviewers expect you to identify the flaw in

---

## Phase 1 — Core Stochastic Bandit Algorithms (Ch. 4–9)
*Goal: derive and reason about the algorithms that actually appear in interviews.*

**Ch 4 — UCB1**
- Optimism in the face of uncertainty — the core principle
- Hoeffding's inequality → confidence radius derivation
- Full UCB1 algorithm and step-by-step numerical trace
- Regret bound O(log T) — where each term in the bound comes from, intuitively
- Why UCB is deterministic (no randomness) and what that implies operationally

**Ch 5 — UCB Variants**
- UCB2 (epoch-based, tighter constants)
- KL-UCB (using KL-divergence confidence sets instead of Hoeffding — much tighter for Bernoulli rewards)
- Bayes-UCB
- MOSS (minimax-optimal)
- When each variant is actually preferred in practice vs interview-only trivia

**Ch 6 — Thompson Sampling: Foundations**
- Bayesian reframing of the bandit problem — priors, posteriors, posterior sampling as decision rule
- Beta-Bernoulli conjugate model, worked update-by-update numerical example
- Full algorithm trace over several rounds by hand
- Why "probability matching" is the right mental model

**Ch 7 — Thompson Sampling: Extended**
- Gaussian bandits (Normal-Normal conjugacy)
- Regret bounds for TS (Bayesian regret vs frequentist regret — the distinction interviewers probe)
- Empirical performance: why TS usually beats UCB in practice despite similar theoretical bounds
- TS vs UCB: a structured comparison table you'll be able to reproduce from memory

**Ch 8 — Adversarial Bandits**
- Why stochastic assumptions can fail (adversarial reward sequences)
- EXP3 algorithm: importance-weighted estimators, exponential weighting
- EXP3 regret bound (O(√T)) and why adversarial bandits are fundamentally harder than stochastic
- Connection to online learning / no-regret dynamics and game theory (brief, interview-relevant only)

**Ch 9 — Best-Arm Identification (Pure Exploration)**
- Regret minimization vs pure exploration — a different objective entirely
- PAC bandit formulation (ε, δ)
- Successive elimination algorithm
- LUCB and racing algorithms
- Where this shows up in practice: offline model selection, hyperparameter search (connects to Bayesian Optimization from your ANOVA/DOE curriculum)

---

## Phase 2 — Contextual & Structured Bandits (Ch. 10–13)
*Goal: move from "which arm is best" to "which arm is best for this user/query."*

**Ch 10 — Contextual Bandits: Formulation**
- Why context changes everything (per-user/per-query personalization)
- Policy class, reward model, realizability assumption
- Regret definition in the contextual setting
- Connection to supervised learning ("bandit feedback" — only observing the reward for the chosen action)

**Ch 11 — LinUCB**
- Linear reward model assumption
- Ridge regression for reward estimation, confidence ellipsoids around parameter estimates
- Disjoint LinUCB vs Hybrid LinUCB — the practical distinction and when each is used
- Full worked example: features → confidence bound → arm selection
- This is likely the single most interview-tested algorithm in this entire syllabus — treated accordingly

**Ch 12 — Linear Thompson Sampling & Neural Bandits**
- Bayesian linear regression as the TS analog to LinUCB
- NeuralUCB / NeuralTS — the high-level idea of using a neural network as the reward model with a Bayesian/UCB wrapper (interview-depth, not paper-implementation depth)
- Why contextual bandits at Google/Apple scale usually end up here (ad ranking, News/Discover feed ranking, Siri suggestion ranking)

**Ch 13 — Non-Stationary & Structured Bandits**
- Sliding-window UCB and discounted UCB (handling reward distributions that drift over time)
- Combinatorial bandits (selecting a *set* of arms — e.g., a ranked list, not a single item)
- Dueling bandits (only relative/preference feedback available — connects to RLHF-style preference learning)
- Why "non-stationarity" is the #1 follow-up question after any bandit whiteboard answer

---

## Phase 3 — Offline Evaluation & Production Systems (Ch. 14–17)
*Goal: this is where L5/L6 answers are actually won or lost — everyone knows the algorithms, few can reason about deployment.*

**Ch 14 — Off-Policy Evaluation: Importance Sampling**
- The core problem: you have logged data from policy π₀, you want to evaluate a new policy π₁ *without* deploying it
- Inverse Propensity Scoring (IPS) — derivation, unbiasedness, and why variance explodes
- Propensity score estimation and its pitfalls

**Ch 15 — Variance-Reduced Off-Policy Estimators**
- Doubly Robust (DR) estimation — combining a reward model with IPS
- Self-Normalized IPS (SNIPS)
- Clipped/truncated IPS
- A comparison table of bias/variance tradeoffs across all four estimators — this is a very common interview whiteboard ask

**Ch 16 — The Replay Method & Counterfactual Learning**
- Li et al.'s replay method for evaluating bandit policies on logged data (the practical industry-standard approach)
- Counterfactual risk minimization — brief conceptual coverage
- Why offline evaluation is treated as *the* differentiator between candidates who've memorized algorithms and candidates who've shipped bandits

**Ch 17 — Production Systems & Decision Frameworks**
- Bandits vs A/B testing: a decision framework (ties directly to your existing A/B testing curriculum — when do you actually reach for a bandit instead of a fixed-horizon test)
- Cold-start problem and warm-starting bandits from offline data
- Multi-objective bandits (e.g., optimizing engagement *and* revenue simultaneously)
- Real-world case studies: ad ranking auctions, News/Discover feed ranking, App Store search ranking, Siri/Assistant suggestion ranking
- Guardrails: how bandits fail silently in production (feedback loops, delayed reward, non-stationarity from the bandit's own actions)

---

## Phase 4 — Interview Mastery & Synthesis (Ch. 18–20)
*Goal: convert the theory into fast, confident interview performance.*

**Ch 18 — Whiteboard Problem Bank**
- Derive UCB1's confidence bound from Hoeffding, live
- Code ε-greedy, UCB1, and Thompson Sampling from scratch (Python, no libraries)
- Trace regret growth by hand for a 3-arm example under each algorithm

**Ch 19 — System Design Case Studies (dialogue-format mock interviews)**
- "Design an explore-exploit system for [ad ranking / feed ranking / App Store search]"
- Full L5-style and L6-style answer breakdowns, same format as your prior system design mocks

**Ch 20 — Rapid-Fire Review & L5-vs-L6 Differentiators**
- Consolidated comparison tables (ε-greedy vs UCB vs TS vs EXP3; IPS vs DR vs SNIPS vs Replay)
- The 15 questions most likely to appear as follow-ups, with model answers
- Common traps checklist compiled from every chapter

---

## Suggested pacing

20 chapters, roughly 1 chapter per session (matching the pace you've used elsewhere) — this is a tighter, denser topic than something like recsys or outlier analysis, so 20 chapters over ~25–30 days of focused study is realistic, with Phase 3 (offline evaluation) deserving extra time since it's the most differentiating and least intuitive material.

Ready to start with **Chapter 1 — The Multi-Armed Bandit Problem** whenever you want to begin.
