# Chapter 13: Google-Style Case Studies (Full Walkthroughs)

## 1. Explanation — how to structure a causal case-study answer

This final chapter doesn't introduce a new method — it's about **synthesizing everything from Chapters 1–12** the way an interviewer actually presents a problem: as a messy business situation, not a pre-labeled "use method X" prompt. The single biggest differentiator between a strong and a mediocre answer is *structure*. A strong causal case-study answer, every time, walks through the same sequence:

1. **Clarify the estimand** (Chapter 2) — what exactly are we trying to measure, and for whom? ATE over the full population? ATT for those who already got it? A specific rollout population?
2. **Draw the mental (or literal) DAG** (Chapter 4) — what's plausibly confounding this relationship, what's a mediator, what's a collider, what's the mechanism generating treatment assignment in the first place?
3. **Assess feasibility of randomization** (Chapter 3) — can we just run an RCT? If not, why not — cost, ethics, the thing already launched to everyone, or a suspected interference problem (Chapter 11) that would make even an RCT misleading?
4. **Pick the best available identification strategy** given the real-world constraints, and **name its core assumption explicitly** — parallel trends for DiD (Ch.8), continuity/no-manipulation for RDD (Ch.9), good pre-treatment fit for synthetic control (Ch.10), ignorability for PSM/regression (Ch.5-6), relevance/exclusion/independence for IV (Ch.7).
5. **State how you'd sanity-check that assumption** — pre-trends plots, placebo tests, balance checks, E-values (Chapter 12) — since none of these assumptions are provable, only more or less well-supported.
6. **Communicate the result with appropriate humility** — a point estimate, honest uncertainty, and the assumption caveat, rather than a bare number presented as settled fact.

Interviewers are explicitly listening for this sequence, in roughly this order — jumping straight to "run a regression" or "compute the difference in means" without steps 1-3 first is the most common way strong technical knowledge fails to land as a strong interview answer.

## 2. Examples — three full case studies

### Case Study A: "We already launched to 100% in one country, no experiment. Did it work?"

*Setup: A product team launched a new UI in Germany only, no holdback group. Six months later, leadership asks for a causal read on whether it worked.*

- **Estimand**: ATT for Germany specifically — since only Germany received treatment, and we have no reason yet to assume the effect generalizes elsewhere, the honest target is "the effect for Germany, as launched," not a universal ATE.
- **DAG**: candidate confounders to worry about — any concurrent marketing campaigns in Germany, seasonal effects specific to the German market, broader macro/economic shifts, or any *other* product changes that happened to be bundled with the same launch window.
- **Feasibility of RCT**: not possible retroactively — the launch already happened to 100% of German users; there's no held-back control group within Germany to compare against.
- **Identification strategy**: DiD (Germany vs. a single comparable, unaffected country) *if* a genuinely good single comparator exists and pre-trends look parallel; otherwise, **Synthetic Control** using several other European countries as a weighted donor pool — this is the more robust default here, since finding one single "comparable country" is a strong, risky assumption for something as idiosyncratic as a national market (recall Chapter 10's point: a weighted blend of several donors usually beats betting everything on one).
- **Sanity checks**: pre-treatment trajectory fit (for synthetic control) or a multi-period pre-trends plot (for DiD); a placebo test using a fake, earlier "launch date"; explicitly check whether any other Germany-specific event (marketing calendar, pricing change, a competitor's action) coincided with the real launch date.
- **Communication**: report the estimated effect alongside a placebo-based sense of uncertainty, explicitly flag that this result describes Germany specifically (not a universal effect), and recommend that any *future* launch hold back a small random percentage of users specifically to enable a cleaner causal read next time — turning this ad hoc, retrospective analysis into a lesson about better experimental infrastructure going forward.

### Case Study B: "User-level test shows a win, but we're worried about marketplace interference."

*Setup: A rideshare-adjacent product tests a new driver-incentive structure, randomized by individual driver. Treated drivers show higher earnings and more accepted trips. Leadership wants to roll it out to all drivers.*

- **Estimand**: leadership actually wants the *ATE of a full-market rollout* — but what was measured is closer to the ATT under **partial** treatment, which (per Chapter 11) can be a very different number when a shared resource is involved.
- **DAG/mechanism concern**: rider demand is a shared, roughly fixed pool in the short run — treated drivers may simply be capturing trip requests that would otherwise have gone to control drivers. This is a SUTVA violation: the observed "win" partly reflects a zero-sum reallocation of a fixed rider-demand pie among drivers, not genuine market growth.
- **Feasibility check**: a driver-randomized test cannot cleanly answer the full-rollout question here, precisely because rollout to 100% eliminates the very reallocation-from-control-drivers mechanism that inflated the observed effect during the partial test.
- **Better identification strategy**: a switchback design (the whole city gets the new incentive structure on some days/weeks, the old structure on others) or geo-based randomization across similar cities — both avoid the driver-vs-driver competition for the same shared rider pool that contaminates a driver-level test.
- **Sanity check**: compare the driver-randomized estimate to a smaller-scale switchback pilot; if the driver-randomized estimate is substantially larger, that gap is itself a rough, empirical estimate of how much interference-driven inflation was present in the original test.
- **Communication**: explicitly tell leadership something like: "the driver-level test measured redistribution among drivers more than genuine market growth; we need a city-level (switchback or geo) test to estimate the true full-rollout effect before committing to it." This is exactly the kind of pushback Google wants to see a data scientist give, rather than rubber-stamping a flawed-but-favorable-looking result because it's what the business wants to hear.

### Case Study C: "Observational log data only — need a defensible causal read quickly."

*Setup: The Search team wants to know if a new "related searches" module increases session length, using only existing logs — the module was rolled out based on an internal eligibility/targeting rule, not randomly.*

- **Estimand**: ATT for users who received the module under the existing targeting rule — a natural starting point given the constraint that only observational data on the actually-targeted population exists.
- **DAG**: the targeting rule itself is the key confounder-generating mechanism — whatever internal logic decided who gets the module (e.g., high-engagement users, certain query types) is a *fork* that independently affects both "getting the module" and "session length," even if the module itself did nothing.
- **Feasibility of RCT**: likely feasible *going forward* (and should be recommended), but leadership specifically wants a fast read *now* using data that already exists.
- **Identification strategy**: PSM or IPW (Chapter 6), conditioning specifically on the known targeting-rule inputs. This is a meaningfully stronger design than adjusting for generic user features, because if the exact targeting logic/features are known and logged, ignorability becomes much more defensible — you're controlling for literally the variables that generated the treatment assignment, not just a plausible-sounding proxy set. Check covariate balance after weighting, per Chapter 6's diagnostic.
- **Sanity checks**: run an E-value / sensitivity analysis (Chapter 12) on the resulting estimate, explicitly naming plausible unmeasured confounders — e.g., "user intent or mood within that specific session" isn't logged anywhere and could independently affect both whether the module was deemed useful enough to display and how long the session lasted.
- **Communication**: report the ATT alongside the E-value framing ("a hidden confounder would need to be at least this strong to overturn our conclusion"), and explicitly recommend a proper RCT — even a small-scale one — before any permanent rollout decision is made, framing the observational estimate as a fast, provisional signal rather than a final, decision-grade answer.

## 3. Interview Q&A (case-study style, rapid-fire)

**Q: What's the very first question you should ask yourself before reaching for any causal inference formula, in any of the above case studies?**
A: "What exactly is the estimand, and for whom?" Clarifying whether the business question wants ATE, ATT, or a specific subpopulation effect shapes everything downstream, including whether your chosen method can even answer the question actually being asked — many technically-correct analyses fail simply because they answer a different, unstated question than the one leadership cares about.

**Q: In Case Study B, what would likely happen if the team ignored your pushback and just rolled the incentive out to 100% of drivers based on the flawed driver-randomized estimate?**
A: They would likely be disappointed at the aggregate level — a large chunk of the "win" measured in the partial test came from reallocating a fixed pool of rider demand toward treated drivers at control drivers' expense. Once everyone is treated, there's no one left to "steal" trips from, so the true aggregate lift in total completed trips or total earnings across the whole market would likely be much smaller than the per-driver effect measured in the partial-rollout test suggested.

**Q: In Case Study C, why is it valuable to explicitly control for "the exact targeting-rule inputs" rather than just a generic set of user features?**
A: Because the targeting-rule inputs are, by construction, exactly what generated the confounding in the first place — they're the fork affecting both D and, likely, Y through their known correlation with engagement. Controlling for the precise mechanism that created the treatment assignment gives you the strongest, most directly-justified case for conditional ignorability, compared to a generic "throw in some plausible-sounding covariates" approach where you might still be missing the actual driver of selection.

**Q: How would you defend, to a skeptical engineering leader, spending extra time on synthetic control or switchback designs instead of just trusting whatever simpler analysis is already available?**
A: Frame it in terms of decision cost and risk, not methodological purity: "the simpler analysis is faster but has a specific, identifiable bias — interference, or a lack of a genuinely clean comparator — that could lead us to overstate the benefit and make a costly rollout decision based on a number that won't hold up once we scale. The more rigorous design costs more time upfront but protects us from the much larger cost of reversing course or being blindsided by an underwhelming full-scale rollout."

**Q: What's a good habit to build so you don't miss estimand confusion or SUTVA issues under interview time pressure?**
A: Explicitly say the checklist out loud, briefly, before computing anything: "Before I give you a number — what population does this describe? What's the identifying assumption here? Is there any reason to worry about interference or a shared resource in this setting?" Verbalizing this checklist, even in one or two sentences, signals rigor to the interviewer and often surfaces the exact issue (like the interference problem in Case Study B) that the question was specifically designed to test for.

**Q: Across all three case studies, what's the one thing every "better" identification strategy had in common, compared to the naive default?**
A: In every case, the improvement came from more explicitly modeling or exploiting the actual **mechanism that generated the treatment assignment** — using multiple weighted donors instead of one arbitrary comparator (Case A), switching the randomization unit to match where the real economic/marketplace boundary sits (Case B), or conditioning precisely on the known targeting logic rather than generic covariates (Case C). None of these are exotic new tricks — they're the same core methods from earlier chapters, applied with a clearer, more specific understanding of *why* the naive comparison would be biased in that particular setting.

**Q: If you had unlimited time and budget, which of the three case studies would you most want to convert into a "proper" experiment going forward, and why?**
A: Case C (the Search "related searches" module) is the easiest and cheapest to convert — a standard RCT is fully feasible there (just hold back a random subset from receiving the module), unlike Case A (already launched to 100%, can't retroactively randomize) or Case B (needs a more elaborate switchback/geo design due to genuine marketplace interference, not just a simple holdout). Prioritizing the "easy win" RCT conversion for future decisions, while using the best available quasi-experimental method for the harder, already-launched or interference-prone cases, is a practical, resource-aware way to think about where to invest experimentation infrastructure.

---

## Final Summary Table (quick review before an interview)

| Chapter | Core question it answers | Central assumption | What breaks it |
|---|---|---|---|
| 1. Potential Outcomes | What does "causal effect" even mean? | — (framework, not a method) | N/A |
| 2. Estimands | Average over whom? | — | Confusing ATT with ATE |
| 3. RCTs | How do we get a clean comparison? | Randomization → independence | Interference, attrition, non-compliance, SRM |
| 4. DAGs/Confounding | What should I control for? | Correct causal graph | Colliders/mediators mistaken for confounders |
| 5. Regression Adjustment | Can a model recreate randomization? | Ignorability + correct functional form | No common support, wrong functional form |
| 6. PSM/IPW | Can I match on one number instead of many? | Ignorability + overlap | Extreme propensity scores, poor overlap |
| 7. IV | What if confounders are unmeasured? | Relevance + exclusion + independence | Weak instruments, violated exclusion |
| 8. DiD | Can I use each group as its own baseline? | Parallel trends | Diverging pre-trends, staggered timing |
| 9. RDD | Can a sharp rule create local randomization? | Continuity, no manipulation | Bunching/manipulation at cutoff |
| 10. Synthetic Control | What if I have only 1 treated unit? | Good pre-treatment fit | Poor pre-fit, donor contamination |
| 11. Interference/SUTVA | What if units affect each other? | SUTVA (usually violated at scale) | Shared resources, network/marketplace effects |
| 12. Sensitivity Analysis | How fragile is my observational result? | — (a diagnostic layer, not its own method) | A plausible confounder as strong as the E-value threshold |
| 13. Case Studies | How do I combine all of this under pressure? | — (synthesis, not a new assumption) | Skipping the estimand/DAG/feasibility steps |

---
**Previous: Chapter 12 — Sensitivity Analysis, Placebo Tests, and Quasi-Experiments**

**This completes the 13-chapter deep dive.** You now have, chapter by chapter: heavy explanation, worked numericals, and interview Q&A for the full causal inference toolkit expected in a Google MLE/DS interview — potential outcomes and estimands, RCTs, confounding and DAGs, regression/matching/weighting, IV, DiD, RDD, synthetic control, interference/SUTVA, sensitivity analysis, and full case-study synthesis.
