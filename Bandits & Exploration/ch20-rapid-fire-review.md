# Chapter 20 — Rapid-Fire Review & L5-vs-L6 Differentiators

*(Final chapter. Same plain-language style — this one is designed for fast, repeated review in the days before an interview.)*

---

## 20.1 How to use this chapter

This is your last-mile review chapter — the one to reread the night before an interview. It has three parts: consolidated comparison tables covering everything in the syllabus, the 15 questions most likely to come up as follow-ups (with short model answers), and a full traps checklist pulled from every chapter. Nothing here is new material — it's all condensed from Chapters 1–19.

---

## 20.2 Master comparison table: the core algorithms (Chapters 3–9)

| Algorithm | Deterministic or randomized? | Regret shape | Needs to know gaps/horizon in advance? | One-line mechanism |
|---|---|---|---|---|
| Greedy | Deterministic | Linear (Theta(T)) | No | Always pick current highest sample mean; can lock on forever |
| Constant-epsilon epsilon-greedy | Randomized | Linear (Theta(T)), smaller constant | No | Explore uniformly at rate epsilon forever |
| epsilon-decay | Randomized | Can be O(log T) | Yes (needs gap-dependent constant) | Shrink epsilon over time |
| Explore-then-commit | Deterministic after commit | Can be O(log T) | Yes (optimal m needs gaps) | Fixed explore phase, then commit forever |
| UCB1 | Deterministic | O(log T) | No | Sample mean + Hoeffding-based bonus |
| KL-UCB | Deterministic | O(log T), asymptotically optimal constant | No | Sample mean + KL-divergence-based bonus |
| Thompson Sampling | Randomized | O(log T), asymptotically optimal | No | Sample from posterior, pick highest draw |
| EXP3 | Randomized | O(sqrt(T)) | No | Exponential weights + importance-weighted updates |
| MOSS | Deterministic | Best worst-case (minimax) bound | Yes (needs T) | Horizon-aware bonus that can hit exactly zero |

---

## 20.3 Master comparison table: contextual bandits (Chapters 10–13)

| Approach | Underlying model | Decision rule | Best used when |
|---|---|---|---|
| Disjoint LinUCB | One straight-line model per arm | Prediction + unfamiliarity bonus | Few arms, plenty of data per arm |
| Hybrid LinUCB | Shared weights + per-arm weights | Prediction + unfamiliarity bonus | Many arms, especially new ones needing to borrow strength |
| Linear Thompson Sampling | Straight-line model, Bayesian version | Sample a plausible line, use its prediction | Similar to LinUCB; often preferred empirically |
| NeuralUCB / NeuralTS | Neural network instead of a straight line | Same as above, fancier model underneath | Relationships too complex for a straight line |
| Sliding-Window UCB | Adds a hard recency cutoff | Same as UCB1, only recent data | Non-stationary environment, want fast forgetting |
| Discounted UCB | Adds smooth recency fade | Same as UCB1, exponentially down-weighted | Non-stationary environment, want gradual forgetting |

---

## 20.4 Master comparison table: off-policy evaluation (Chapters 14–16)

| Method | Uses every logged round? | Needs a reward model? | Bias/variance profile |
|---|---|---|---|
| Plain IPS | No, only agreement rounds | No | Unbiased, often high variance |
| Clipped IPS | No | No | Slightly biased, lower variance |
| SNIPS | No | No | Slightly biased, lower variance, self-normalizing |
| Doubly Robust | Yes, via the reward model | Helps, but not required | Low variance; unbiased if either ingredient is good |
| Replay Method | No, only agreement rounds | No | Unbiased and simple, but needs uniform-random logging |

---

## 20.5 The 15 most likely follow-up questions, with short model answers

**1. "What if the reward distributions change over time?"**
-> Sliding-Window UCB (hard cutoff) or Discounted UCB (smooth fade) — both adapt the confidence/posterior calculation to weight recent data more.

**2. "Why not just use epsilon-greedy everywhere, it's simple?"**
-> Constant-epsilon epsilon-greedy has linear regret — it keeps exploring at a fixed rate forever, even once confident. Fine for simplicity/predictability in some production settings, but not regret-optimal.

**3. "How is a bandit different from full reinforcement learning?"**
-> A bandit is a one-state, no-transition MDP — your action doesn't change "where you are" for next round. Contextual bandits add context (like state) but still no transitions caused by your own actions; full RL adds transitions back in.

**4. "Why does Thompson Sampling often beat UCB1 in practice?"**
-> UCB1's Hoeffding-based bound is generic/distribution-agnostic and therefore loose; Thompson Sampling (like KL-UCB) uses the actual likelihood, so it doesn't waste exploration budget on distributional possibilities that can't occur.

**5. "How would you evaluate a new bandit policy without deploying it?"**
-> Off-policy evaluation using logged data — IPS-family methods (ideally Doubly Robust) if logging wasn't uniform-random, or the Replay Method if you have (or can carve out) a uniform-random traffic slice.

**6. "What's the difference between regret minimization and best-arm identification?"**
-> Regret minimization cares about cumulative reward while playing; best-arm identification only cares about correctly identifying the winner at the end, with no credit for rewards observed along the way — using a regret-minimizing algorithm for a pure-identification goal is provably the wrong pull-allocation strategy.

**7. "How do you handle a brand-new arm with no data (cold start)?"**
-> Warm-start from shared/hybrid model weights, an informative prior built from similar historical arms, or borrow from a related existing arm's performance.

**8. "What if you care about more than one metric?"**
-> Either combine into one weighted reward (a business decision on weights), or optimize a primary metric subject to guardrail constraints on the others.

**9. "How is picking a whole list different from picking one arm?"**
-> Combinatorial bandits — the core new difficulty is credit assignment (can't cleanly attribute a combined outcome back to one item), often complicated further by position bias.

**10. "What if you only get comparisons, not direct rewards?"**
-> Dueling bandits — directly relevant to how RLHF trains on human preference comparisons rather than absolute scores.

**11. "When would you use a bandit instead of a plain A/B test?"**
-> Many options, ongoing traffic, and a real cost to exploring bad options during learning; A/B testing is better when you want one clean, precise, stakeholder-facing effect-size estimate and can afford a dedicated test period.

**12. "Why does UCB1's bonus term include ln(t), not just a function of N_i(t)?"**
-> The global round counter ensures every arm keeps getting a (shrinking) positive bonus forever, guaranteeing occasional revisits across the whole horizon — without it, an arm could get "closed off" too early.

**13. "What's the practical downside of plain IPS?"**
-> High variance — dividing by a small propensity score can let a single rare-arm agreement dominate the whole estimate, sometimes producing implausible results (as shown directly in the Chapter 14 worked example).

**14. "What does 'doubly robust' actually mean?"**
-> The estimate stays unbiased if *either* the reward model or the propensity scores are accurate — you get two independent chances to be right, not one all-or-nothing dependency.

**15. "How would you design an explore-exploit system for [some new scenario]?"**
-> Work through the three-question framework live: how many arms, is the environment stationary or drifting, is there one metric or several — then justify your algorithm choice against those specific answers, the way each Chapter 19 mock did.

---

## 20.6 Full traps checklist, consolidated from every chapter

- Regret is a *gap* relative to the best arm, not "you lost money" — and cumulative regret can never decrease, only flatten.
- Pure greedy and constant-epsilon epsilon-greedy both produce *linear* regret — epsilon-greedy just has a smaller constant, it isn't a fundamentally different (better) shape.
- UCB1 is fully *deterministic* — its "exploration" is optimistic bias in a score, not randomness.
- The regret bound's 1/gap-squared term means arms *close* to optimal get pulled *more*, not less — a common, counter-intuitive point worth stating explicitly.
- Thompson Sampling's randomness is a structured draw from a real posterior, not arbitrary noise like epsilon-greedy's uniform exploration.
- LinUCB is not "just linear regression" — the exploration bonus is the entire "bandit" part of "contextual bandit."
- EXP3's O(sqrt(T)) regret is not a design flaw — it's the unavoidable statistical cost of dropping the fixed-distribution assumption.
- Best-arm identification and regret minimization are genuinely different objectives — don't reach for UCB1/TS by default on a stated pure-exploration problem.
- IPS requires the logging policy to be randomized with *known* probabilities — it doesn't work cleanly with a fully deterministic logger.
- The Replay Method specifically requires *uniform-random* logging — using it on non-uniform logs silently reintroduces bias.
- Multi-objective reward weighting is a business decision, not a purely technical one — say so explicitly if asked.
- MOSS needs the horizon T known in advance — a real limitation for systems with no fixed end date.

---

## 20.7 Closing note

This closes the full 20-chapter Bandits & Exploration syllabus — from the basic exploration-exploitation tradeoff in Chapter 1, through UCB and Thompson Sampling, into contextual bandits, and finally the offline-evaluation and production material in Phase 3 that most separates textbook familiarity from real deployment judgment. The single habit worth carrying forward past this course is the three-question framework from Chapter 17/19: **how many arms, stationary or drifting, one metric or several** — it's a repeatable way to reason through any new scenario an interviewer invents on the spot, rather than needing to have memorized that exact case beforehand.

Good luck.
