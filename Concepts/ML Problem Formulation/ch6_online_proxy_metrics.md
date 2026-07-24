# Chapter 6: Metric Selection — Online & Proxy Metrics

## 1. Why This Chapter Exists

Chapter 5 covered how to judge a model offline. This chapter covers the uncomfortable truth every senior ML candidate needs to internalize: **offline metric improvement is necessary but not sufficient evidence of business impact.** A model can win decisively offline and lose, or do nothing, online — and the reasons are structural, not just "bad luck." This chapter is about naming those structural reasons and designing metrics that survive the transition from offline to online.

## 2. Why Offline and Online Metrics Diverge

Three distinct mechanisms, each worth naming separately in an interview:

**Mechanism A — Offline evaluation is on historical, policy-biased data.** Your offline test set was generated under the *old* model's serving policy (Chapter 11 will formalize this as a feedback-loop/selection-bias problem). A new model that would do brilliantly on candidates the old policy never showed anyone gets no credit offline, because those candidates barely appear in logged data at the volume needed to evaluate them reliably.

**Mechanism B — The offline metric is a proxy, and proxies can be gamed by the optimization process itself.** Chapter 1's click-vs-watch-time example generalizes: any model trained to directly maximize a proxy metric will find degenerate solutions that improve the proxy without improving the true objective, *especially* as model capacity and optimization budget increase — this is a form of Goodhart's Law ("when a measure becomes a target, it ceases to be a good measure") that gets *worse*, not better, with more powerful models.

**Mechanism C — User behavior adapts to system changes, and offline data can't capture that.** A ranking change that offline looks purely additive (surfaces new-but-relevant items) can change user *browsing habits themselves* online (e.g., users learn to scroll less because top results got better, changing the click-position distribution in ways no offline replay predicted).

### Worked Numerical Example: An Offline Win That Became an Online Loss

Suppose Model B beats Model A on offline NDCG@10: 0.71 vs. 0.68 (a real, statistically significant +4.4% relative gain). You ship an A/B test. Results after 2 weeks, $n = 200{,}000$ per arm:

| Metric | Control (A) | Treatment (B) | Relative change |
|---|---|---|---|
| Offline NDCG@10 (recomputed on logged online data) | 0.68 | 0.71 | +4.4% |
| Click-through rate | 0.084 | 0.086 | +2.4% |
| **7-day session return rate** | 0.61 | 0.58 | **−4.9%** |

CTR moved in the expected direction, but the 7-day return rate — a longer-horizon guardrail metric closer to the true retention objective — *dropped*. Investigation reveals Model B was trained to optimize NDCG using immediate-click relevance labels (Mechanism B: proxy), and it learned to surface more sensational, immediately-clickable content that satisfied users less on reflection, measurably reducing their propensity to return. Offline NDCG, computed against the *same click-derived relevance labels the model was trained to predict*, could not have detected this — it's structurally blind to the very failure mode it caused, because the offline metric and the training objective share the same biased label source.

**The lesson this quantifies**: an offline metric computed from labels that are themselves a proxy (click-derived relevance) cannot catch degradation in a *different*, longer-horizon true objective (retention) — you need an independent guardrail metric, measured on a different signal and a longer time horizon, specifically to catch this class of failure.

## 3. Guardrail Metrics

A **guardrail metric** is one you don't optimize for directly but monitor to ensure a launch decision doesn't quietly damage something you care about. Standard practice: define your **primary metric** (the one driving the ship/no-ship decision, ideally close to the true business objective) and 3–6 **guardrail metrics** (things that must not regress beyond a pre-specified threshold, e.g., latency, error rate, long-horizon retention, revenue-per-user, complaint rate).

**Numerical framing**: pre-register a guardrail threshold before launch, e.g., "ship only if 7-day return rate does not drop by more than 1% relative, at 95% confidence." In the example above, a −4.9% drop with tight confidence intervals would trip this guardrail and block the launch *despite* the primary offline metric improving — this is precisely the discipline that prevents Mechanism B failures from reaching all users.

## 4. Designing Proxy Metrics That Survive Contact With Reality

Since some horizon-appropriate metrics (like 90-day retention) are too slow to use for rapid iteration, teams need faster proxy metrics that still correlate well with the true objective. A rigorous way to validate a candidate proxy metric: **run a historical correlation study** — across past A/B tests, compute the correlation between the fast proxy's movement and the slow true metric's movement.

**Worked example**: across 20 past launched experiments, you have (proxy metric % change, true 90-day-retention % change) pairs. Suppose the Pearson correlation is $r = 0.81$ — a fairly strong positive relationship, suggesting the proxy is usable for fast iteration decisions, though not a perfect substitute (it explains $r^2 \approx 0.66$, or 66%, of the variance in the true metric's movement, leaving meaningful room for the two to diverge on any single experiment — exactly like the case in Section 2). Contrast this with a proxy metric that shows $r = 0.15$ against the true objective across the same 20 experiments — that proxy should be discarded or heavily caveated, regardless of how convenient it is to measure, since it explains only about 2% of the variance and is nearly as likely to mislead as to inform.

## 5. Statistical Rigor in Online Evaluation

Two recurring pitfalls, stated with numbers:

**Peeking / early stopping bias.** If you check significance repeatedly and stop as soon as $p<0.05$ is first hit, your true false-positive rate is inflated well above 5% — with continuous monitoring and no correction, the effective false-positive rate can exceed 20–30% depending on how often you peek. Fix: pre-register a fixed sample size/duration, or use a sequential testing correction (e.g., alpha-spending functions) designed for repeated looks.

**Underpowered tests on rare metrics.** If your guardrail metric (e.g., chargeback rate, a ~0.3% base rate event) needs to detect a 10% relative change, back-of-envelope power calculations often reveal you'd need millions of users per arm to have adequate statistical power — far more than your primary metric requires. This needs to be flagged *before* the test launches, not discovered as "the result was inconclusive" afterward.

## 6. Production Considerations

- **Every launch decision needs a pre-registered primary metric and guardrail set, with pre-registered thresholds, before the experiment starts** — deciding "well is this good enough" post-hoc invites motivated reasoning and inconsistent bar-setting across teams.
- **Guardrail metrics should be chosen to specifically catch known failure modes of the primary metric/proxy**, not just be a generic checklist — e.g., pairing an immediate-engagement primary metric with a longer-horizon retention guardrail directly targets Mechanism B from Section 2.
- **Proxy-metric validity (Section 4's correlation study) should be periodically re-validated**, not assumed permanent — as the underlying product or user base shifts, a previously well-correlated proxy can decouple from the true objective.
- **Sequential testing / peeking corrections need to be built into the experimentation platform itself**, not left to individual analysts' discipline, or Section 5's inflated false-positive problem will recur silently across the organization.

## 7. Common Interview Traps

- **Treating an offline metric win as sufficient justification to ship**, without discussing the need for online validation and guardrails.
- **Proposing only one online metric** (just CTR, say) without a guardrail set — missing the entire point of Sections 2–3.
- **Not recognizing that the *same biased label source* powering both training and offline evaluation makes offline evaluation structurally blind to certain failure modes** — this is the deepest point in the chapter and a strong differentiator when raised explicitly (Section 2's worked example).
- **Ignoring statistical validity issues** (peeking, underpowered rare-metric tests) and treating any observed p<0.05 result as automatically trustworthy.

## 8. L5-Differentiating Talking Points

- Explicitly name all three divergence mechanisms (policy bias, proxy-gaming/Goodhart, behavioral adaptation) when asked "would an offline win always translate online?" — most candidates can name one; naming all three with an example each is a strong signal.
- Propose a concrete guardrail metric tied to the *specific* failure mode a given proxy is most likely to enable (e.g., a longer-horizon retention guardrail specifically because the primary metric is proxy-click-derived).
- Bring up the historical correlation-study method (Section 4) as the rigorous way to validate any proposed fast proxy metric, rather than asserting a proxy is "probably fine" on intuition alone.
- Flag peeking/sequential-testing risk unprompted when discussing how a launch decision would actually be made, showing statistical maturity beyond metric selection itself.

## 9. Comprehension Checks

1. Using Section 2's three mechanisms, construct a plausible scenario (different from the article given) where a search-ranking model wins offline on NDCG but loses on an online guardrail metric. Identify which mechanism(s) are at play.
2. A team wants to use "average session length" as a fast proxy for "monthly active users" (which takes 30 days to measure). Using Section 4's method, describe exactly what analysis you'd run before trusting this proxy, and what correlation value would make you skeptical of it.
3. Explain, in your own words, why "the offline metric is computed on the same biased labels the model was trained to predict" makes that offline metric structurally unable to detect certain classes of failure — even with a perfectly held-out test set.
4. A guardrail metric (chargeback rate) has a 0.4% base rate and your primary metric test is well-powered at $n=50{,}000$ per arm. Explain qualitatively why the guardrail metric might still be underpowered at that same sample size, and what you'd do about it before launch.

---

*End of Chapter 6. Chapter 7 will cover objective function design — translating business cost structures directly into loss functions, including asymmetric losses and weighted objectives.*
