# Chapter 19: Conflicting Metrics & Ship Decisions

*(Module 4: Judgment & System-Level Thinking begins here)*

## 1. Intuition

Every chapter so far has built toward this moment. In a real Google L5 interview, the case study almost never resolves cleanly — you'll be given a scenario where the primary metric is flat or mixed, one guardrail regressed slightly, a secondary metric looks great, and you have to make an actual recommendation under ambiguity. **This is the single biggest differentiator between L4 and L5 candidates**: an L4 candidate can run the right statistical tests; an L5 candidate can synthesize ambiguous, partially-conflicting evidence into a clear, well-reasoned business recommendation, while explicitly stating their assumptions and the risks of being wrong.

The intuition to hold onto: there is rarely a purely "correct" answer to "should we ship this?" What interviewers are evaluating is whether your **reasoning process** is sound, structured, and appropriately calibrated to the stakes and reversibility of the decision — not whether you land on the specific verdict they had in mind.

## 2. A Structured Framework for the Decision

When metrics conflict, work through this sequence rather than free-associating:

**Step 1 — Return to the pre-specified OEC and guardrails (Chapters 9-10).** What was the actual, pre-committed decision rule? If the primary OEC alone gives a clear answer and all guardrails pass within tolerance, you may not have a real conflict at all — just noisy secondary metrics that were never supposed to drive the decision. State this explicitly; it often resolves apparent "conflicts" immediately.

**Step 2 — Check statistical validity before trusting anything (Chapters 14, 16, 17, 18).** Is there an SRM? Was there uncontrolled peeking? Are you looking at too many metrics without correction? Is a segment-level Simpson's Paradox distorting the pooled read? A "conflict" between metrics is sometimes actually a diagnostic problem in disguise, not a genuine business tradeoff.

**Step 3 — Distinguish statistical significance from practical significance (Chapter 2), for every metric in question.** A guardrail that "regressed significantly" but is well within its pre-agreed non-inferiority margin (Chapter 10) isn't actually a problem worth blocking on.

**Step 4 — Assess reversibility and cost of being wrong.** If shipping is easily reversible (a UI tweak that can be rolled back in an hour with a feature flag) versus difficult to reverse (a pricing change that erodes customer trust, or a data migration), your bar for confidence before shipping should differ substantially. This is a genuinely business, not statistical, consideration — and stating it explicitly is a strong signal.

**Step 5 — Consider a middle path before defaulting to binary ship/no-ship.** Common middle paths: extend the experiment for more power/duration (if underpowered or novelty-affected), ship to a limited rollout percentage while continuing to monitor, ship with a specific mitigation for the segment/guardrail that showed concern, or run a targeted follow-up experiment on the specific ambiguous dimension.

**Step 6 — Make an explicit recommendation with stated confidence and assumptions**, rather than hedging indefinitely. Interviewers specifically want to see you commit to a recommendation ("I'd ship this with continued monitoring on X" or "I'd hold and extend for two more weeks because Y"), not endlessly list considerations without landing anywhere.

## 3. Worked Example (Full Case Walkthrough)

**Scenario**: You tested a new ad format on a search results page.
- Primary OEC (revenue per search): +2.1%, 95% CI [+0.3%, +3.9%], p=0.02 — statistically significant positive.
- Guardrail 1 (search satisfaction survey score): -0.8%, within the pre-agreed 2% non-inferiority tolerance — passes, but trending toward the boundary.
- Guardrail 2 (page load latency): no significant change — clean pass.
- Secondary metric (not pre-specified as OEC or guardrail): click-through rate on organic (non-ad) results: -3.5%, p=0.01 — statistically significant decline, discovered during exploratory post-hoc analysis.

**Walking the framework**:
- Step 1: primary OEC is a clear, significant win. Guardrails pass their pre-agreed thresholds.
- Step 2: check for SRM (assume clean), check for peeking (assume the experiment ran its full pre-planned duration), check for multiple testing on the secondary metrics (this org CTR decline was one of ~12 secondary metrics examined, so a naive p=0.01 deserves some skepticism per Chapter 17 — though it's specific and directionally coherent enough with the "ads pushed organic results down" mechanism that it's not purely noise either).
- Step 3: the satisfaction score regression (-0.8%) is statistically non-significant relative to its 2% tolerance band but is directionally concerning combined with the organic CTR decline — both point toward a coherent, plausible mechanism (more ads → less room/attention for organic results → slightly worse perceived search quality), even though neither individually breaches a hard threshold.
- Step 4: this is a **moderately reversible** decision (ad formats can be rolled back or dialed down via a feature flag relatively quickly) but has real reputational stakes if search quality perception erodes at scale over time — a "slow-burn" risk that's harder to detect and reverse quickly if it compounds.
- Step 5: a strong middle path exists here: **ship at a limited rollout percentage** (e.g., 20-30% of traffic) while specifically instrumenting and monitoring the organic CTR and satisfaction metrics over a longer horizon, rather than a full 100% launch or a full kill.
- Step 6: **Recommendation**: "I'd recommend a partial rollout with continued monitoring on search satisfaction and organic CTR, rather than a full launch or a full hold. The primary revenue metric is a clear, statistically robust win, and both guardrails technically pass — but the coherent, mechanistically-plausible pattern in the secondary data (worse organic CTR, softening satisfaction) suggests real risk to search quality that a full launch would expose us to before we've validated whether it's transient (novelty/Simpson's-type effect) or a genuine tradeoff. A partial rollout gets us most of the revenue benefit now while giving us a real read on the quality risk with more data and time, at limited downside."

This is the shape of a strong L5 answer: it doesn't pretend the tension away, it doesn't refuse to commit, and it explicitly reasons through statistical rigor AND business judgment together.

## 4. Production Considerations

- **Document your reasoning, not just your decision.** In a real company, the "why" behind a ship/no-ship call — especially a nuanced one like a partial rollout — needs to be written down for future reference, both for accountability and so future teams facing a similar ambiguous tradeoff have a precedent to draw on.
- **Revisit "middle path" decisions with a pre-committed follow-up plan.** A partial rollout with "continued monitoring" is only a real plan if you specify, in advance, what result would make you go to 100% versus roll back to 0% — otherwise you risk an indefinite, undocumented partial state that never gets properly resolved (a governance failure mode, not a statistical one).
- **The cost of a Type I error (shipping something bad) vs Type II error (not shipping something good) is asymmetric and case-specific** — a company might reasonably tolerate more false positives on low-stakes UI experiments (fail fast, iterate) but demand much stronger evidence for pricing, trust & safety, or platform-wide ranking changes. Bringing up this asymmetry explicitly, and connecting it back to the $\alpha$/power tradeoff from Chapter 5, shows you see the whole curriculum as one connected decision-theoretic framework.

## 5. Interview Traps

- **Trap #1**: Giving a wishy-washy answer that lists every consideration but never actually commits to a recommendation — interviewers specifically want to see you land somewhere, with stated confidence and caveats, not hedge indefinitely.
- **Trap #2**: Jumping straight to a recommendation without first checking statistical validity (SRM, peeking, multiple testing) — an L5 answer should visibly walk through the diagnostic checks from Module 3 before trusting the conflict is even real.
- **Trap #3**: Treating "guardrail passed its threshold" and "no real risk here" as equivalent — a guardrail passing its numeric threshold doesn't mean you should ignore a coherent, mechanistically-plausible risk signal nearby; thresholds are decision aids, not the entirety of your judgment.
- **Trap #4**: Proposing "extend the experiment" or "partial rollout" as a vague deflection without specifying what you'd actually be looking for, and for how long, before making a final call — a middle path without a pre-committed resolution plan isn't really a decision, it's a postponement.

## 6. L5-Differentiating Talking Points

- Explicitly structuring your answer through a repeatable framework (validity checks → practical significance → reversibility → middle paths → committed recommendation) rather than free-associating shows you have a systematic, teachable process, not just intuition you can't articulate.
- Naming the specific mechanism connecting two seemingly separate concerning signals (ads crowding out organic attention, in the worked example) rather than treating them as two unrelated data points shows real product/mechanistic reasoning, not just pattern-matching on p-values.
- Proposing partial rollout with a pre-specified follow-up resolution criterion, rather than an open-ended "let's keep an eye on it," demonstrates the kind of operational rigor that separates senior ICs from junior ones.
- Explicitly naming the asymmetry between Type I and Type II error costs for this specific decision (not as a generic textbook statement, but applied to the actual stakes of this specific case) is a strong, concrete signal of business judgment layered on top of statistical fluency.

## 7. Comprehension Check

1. Walk through the six-step framework in this chapter using a hypothetical case: a primary OEC that's flat/non-significant, but a guardrail that shows a small, statistically significant improvement. What would you check first, and why?
2. Why might "the guardrail technically passed its non-inferiority threshold" not be sufficient grounds to ignore a related, concerning secondary signal?
3. What makes a "partial rollout" a genuine decision versus merely a postponement of the real decision?
4. Explain how reversibility of a decision should influence how much statistical evidence you require before shipping.
5. In the worked example, what specific follow-up metrics and resolution criteria would you propose before deciding whether to move from a partial rollout to a full launch?

---
*Next: Chapter 20 — Interleaving & Ranking Experiments*
