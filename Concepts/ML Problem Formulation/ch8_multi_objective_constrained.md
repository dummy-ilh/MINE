# Chapter 8: Multi-Objective & Constrained Formulations

## 1. Why This Chapter Exists

Chapter 7 showed how to encode a single cost structure into a loss function. Real production systems almost never have exactly one objective — a feed ranker cares about relevance *and* diversity *and* revenue *and* fairness across content creators simultaneously, and these frequently trade off against each other. This chapter is about the formal machinery for combining objectives that conflict, and about when to use a soft (weighted-sum) versus hard (constrained) formulation.

## 2. Weighted-Sum Scalarization: Strengths and a Real Limitation

The simplest approach, extending Chapter 7 Section 6:

$$
\mathcal{L} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k
$$

This is easy to implement and differentiate through end-to-end, but has a specific, provable limitation: **weighted-sum scalarization can only reach points on the *convex hull* of the true Pareto frontier.** If the actual tradeoff frontier between two objectives (say, relevance and diversity) is non-convex (has a concave "dent"), no choice of weights $w_k$ can land you on a solution inside that dent, even though such solutions may be strictly better on both objectives than anything the weighted sum can reach.

**Illustrative numeric sketch**: suppose three candidate models achieve (relevance score, diversity score) pairs: Model X = (0.90, 0.40), Model Y = (0.60, 0.60), Model Z = (0.75, 0.72). Z Pareto-dominates a straight line drawn between X and Y (it sits above that line in objective space — better on diversity than the X–Y trade-off line would predict for its relevance level), meaning Z is achievable and desirable, but if the true tradeoff curve near Z bulges outward toward the origin (non-convex/concave region) rather than sitting on the outer convex boundary, no linear combination $w_1(0.90) + w_2(0.40)$ vs. $w_1(0.60)+w_2(0.60)$ vs. $w_1(0.75)+w_2(0.72)$ will ever prefer Z over some weighting that favors X or Y instead — because weighted-sum scalarization is, by construction, only capable of finding whichever point maximizes a *linear* function of the objectives, and can never select an interior point that only a nonlinear preference could favor. This is why teams solving genuinely non-convex multi-objective tradeoffs often move to constrained optimization (Section 3) or multi-model ensembles/portfolio approaches instead of a pure weighted sum.

## 3. Constrained Formulation: Optimize One, Constrain the Rest

An alternative to weighting everything together: pick the objective that matters most as the thing you directly optimize, and express the others as **hard constraints**:

$$
\max_\theta \; \mathcal{L}_{\text{primary}}(\theta) \quad \text{subject to} \quad \mathcal{L}_{\text{secondary}}(\theta) \ge c
$$

**Worked example**: maximize expected relevance subject to a diversity floor — e.g., "average pairwise item similarity within any returned list must not exceed 0.3" (a concrete, auditable constraint, rather than an abstract weighted penalty whose real-world effect on diversity is hard to predict in advance). This is often *more interpretable to stakeholders* than a weighted sum: "diversity will never fall below X" is a promise you can directly verify and communicate, whereas "diversity gets weight $\lambda=0.1$ in the loss" doesn't translate to any guaranteed floor on diversity in the delivered results.

**Lagrangian relaxation** is the standard technique to solve constrained problems via unconstrained optimization: introduce a multiplier $\mu \ge 0$ and optimize

$$
\mathcal{L}(\theta, \mu) = \mathcal{L}_{\text{primary}}(\theta) + \mu \cdot \big(c - \mathcal{L}_{\text{secondary}}(\theta)\big)
$$

with $\mu$ adjusted (increased when the constraint is violated, decreased when it's comfortably satisfied) so that at the optimum, either the constraint is exactly met or $\mu = 0$ (the constraint wasn't binding at all, meaning it wasn't actually limiting the solution). Notice the deep connection to Chapter 7: **the Lagrangian formulation actually turns your constrained problem back into a weighted-sum loss, but with the crucial difference that $\mu$ is learned/adapted to satisfy a real business constraint, rather than hand-picked from an ad-hoc scale-matching heuristic** (contrast with Chapter 7 Section 6, where $\lambda$ was picked just to balance loss magnitudes, not tied to a specific guaranteed outcome).

## 4. Pareto Frontier: The Formal Tool for Reasoning About Tradeoffs

A solution $\theta_1$ **Pareto-dominates** $\theta_2$ if it's at least as good on every objective and strictly better on at least one. The **Pareto frontier** is the set of non-dominated solutions — none can be improved on one objective without sacrificing another.

**Practical use in an interview**: when asked to "balance relevance and revenue," a strong answer doesn't just propose one weighting — it describes *sweeping* a range of weights (or constraint thresholds) to trace out the empirical Pareto frontier (e.g., relevance at 20 different revenue-constraint levels), then handing that frontier to product/business stakeholders to pick an operating point, rather than the ML engineer unilaterally deciding the "right" tradeoff. This reframes the model-building task correctly: **your job is often to construct the frontier and quantify its shape, not to secretly decide where on it the company should sit** — that's a product/business decision informed by your frontier, not a purely technical one.

## 5. Worked Numerical Example: Sweeping a Constraint to Build a Frontier

Suppose you train 5 models with the diversity-floor constraint from Section 3 set at increasingly strict values $c \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ (lower = stricter, since it's a max-similarity bound), and observe:

| Diversity floor $c$ | Achieved relevance |
|---|---|
| 0.10 (strictest) | 0.62 |
| 0.20 | 0.71 |
| 0.30 | 0.78 |
| 0.40 | 0.82 |
| 0.50 (loosest) | 0.84 |

The marginal relevance gain per 0.1 relaxation of the constraint shrinks as it loosens: $+0.09, +0.07, +0.04, +0.02$ — a classic diminishing-returns Pareto frontier shape. This numeric pattern is itself a decision-relevant finding: it suggests the constraint level around $c=0.3$ captures most of the achievable relevance (0.78 out of a max ~0.84, i.e., 93% of the way there) while still preserving meaningfully more diversity than looser settings — a concrete, defensible recommendation to hand to stakeholders, backed by the shape of the curve rather than a single arbitrarily chosen weight.

## 6. Production Considerations

- **Weighted-sum objectives (Section 2) drift silently as component loss scales change** (e.g., a data distribution shift changes typical relevance-loss magnitude) — a fixed $\lambda$ that was well-calibrated at launch may implicitly shift the real-world tradeoff months later without any code change, so component-loss monitoring (Chapter 7's point) is doubly important here.
- **Constrained formulations (Section 3) need the constraint itself to be monitored in production**, not just checked once offline — a diversity floor that held in offline evaluation can drift post-launch as the underlying content distribution shifts, silently violating the promised guarantee.
- **Pareto frontier sweeps (Section 5) are expensive** (multiple full training runs) — in practice, teams often approximate the frontier cheaply via post-hoc reranking/thresholding on a single trained multi-output model rather than fully retraining for every constraint level, reserving full retraining sweeps for periodic recalibration.
- **Stakeholder-facing communication should default to constraints, not weights**, wherever possible (Section 3's point) — "diversity never falls below X" is auditable and buildable into monitoring/alerting; "diversity has weight 0.1" is not directly auditable in production.

## 7. Common Interview Traps

- **Defaulting immediately to a weighted-sum loss for any multi-objective question without acknowledging the convex-hull limitation** (Section 2) — this is the single most differentiating gap in this chapter's material.
- **Not distinguishing "I decided the tradeoff" from "I built the frontier and stakeholders decided"** (Section 4) — many candidates implicitly assume the ML engineer should unilaterally pick the final weighting, missing that this is usually a product decision the ML system should *inform*, not make.
- **Proposing a constrained formulation without mentioning Lagrangian relaxation as the practical solving mechanism**, or not connecting it back to weighted-sum losses (Section 3).
- **Sweeping a range of hyperparameters without framing it as "tracing the Pareto frontier"** — doing the right experiment but missing the vocabulary/framing that signals you understand *why* you're doing it.

## 8. L5-Differentiating Talking Points

- Explicitly name the convex-hull limitation of weighted-sum scalarization when proposing any multi-objective loss, and use it to justify when you'd reach for constrained optimization instead.
- Frame multi-objective tradeoffs as "construct the Pareto frontier, hand it to stakeholders" rather than "I'll pick weights that feel right" — this reframes the ML engineer's role correctly and is a strong senior-level signal.
- Bring up production monitoring of the constraint itself (Section 6), not just of the primary metric, for any constrained formulation you propose.
- When discussing a weighted-sum tradeoff, note the diminishing-returns shape typical of real Pareto frontiers (Section 5) and use it to argue for a specific, defensible operating point rather than an arbitrary one.

## 9. Comprehension Checks

1. Explain, using Section 2's convex-hull argument, why a company that wants a specific point deep inside a *non-convex* region of the relevance/diversity tradeoff curve cannot reach it via any weighted-sum loss, regardless of how many weight combinations are tried.
2. Using the Lagrangian formulation in Section 3, explain what it means, in terms of $\mu$, if a diversity-floor constraint turns out not to bind at the optimum (i.e., is satisfied with room to spare). What does this suggest about the constraint's business value at that particular threshold?
3. Using Section 5's numeric table, if a stakeholder insists on the loosest diversity floor ($c=0.5$) purely to maximize relevance, what quantitative argument (using the marginal-gain figures) would you make for reconsidering, without an outright refusal?
4. Propose a business scenario where you would strongly prefer a hard-constrained formulation (Section 3) over a weighted-sum formulation (Section 2), and justify using the interpretability/auditability argument from Section 3.

---

*End of Chapter 8. Chapter 9 will cover data availability and cold-start framing — how to reformulate a problem when data is sparse, delayed, or biased by existing systems.*
