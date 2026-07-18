# Chapter 13 (continued): The Full Landscape of Proxy Metrics

> The chapter gives you one method (surrogate index) and one shortcut (single validated proxy). This extends it sideways: the actual *variety* of proxy types teams use, a repeatable design process for building one, and a do's/don'ts checklist for handling them once they're live.

---

## 1. Why proxies exist at all (why now)

The true outcome you care about — LTV, sustained retention, genuine satisfaction — is almost always **slow, noisy, or both**. A proxy metric exists to trade a small amount of accuracy for a large amount of speed: instead of waiting 12 months to learn if an experiment worked, you find something measurable in 1–4 weeks that's a reliable stand-in. The entire discipline of proxy metrics is about managing that trade honestly — knowing how much accuracy you gave up, and not pretending the proxy *is* the goal.

## 2. The variety of proxy metric types

Not all proxies play the same role. Conflating them is a common source of bad decisions.

| Type | What it measures | Example | Best for |
|---|---|---|---|
| **Behavioral engagement proxy** | Frequency/depth of product use shortly after treatment | Week-1 app opens, week-2 "core actions completed" | Early signal on features meant to build habitual use |
| **Funnel/conversion proxy** | Progress through a defined sequence toward a known-valuable action | Signup → activation → first purchase completion rate | Onboarding, acquisition, monetization-adjacent experiments |
| **Quality/satisfaction proxy** | Direct or indirect signal of perceived experience quality | CSAT/NPS-style survey score, support ticket rate, complaint rate | Changes with plausible risk to perceived quality (ads, UI density, latency) |
| **Composite / surrogate index** | A fitted combination of several of the above, weighted by their actual predictive power | The Chapter 13 worked example's regression-combined index (ρ=0.81) | High-stakes, frequently-run experiment categories where the modeling investment pays off repeatedly |
| **Negative / early-warning proxy** | Something that predicts *harm*, not benefit — used as a guardrail-style proxy rather than a primary OEC proxy | Early churn indicators, rage-clicks, error/crash rate spikes | Catching downside risk before it shows up in a slow lagging metric like retention |
| **Leading indicator proxy** | A metric known from prior research to causally precede the true outcome, not just correlate with it | Time-to-first-value (how fast a new user reaches their "aha moment") | Product areas with an established, well-studied causal mechanism connecting the leading indicator to the outcome |

**Why the distinction matters:** a behavioral engagement proxy and a negative/early-warning proxy answer different questions ("is this good?" vs. "is this actively harmful?") and should usually be tracked *together*, not chosen between — this mirrors the OEC-plus-guardrails structure from Chapter 9, just moved earlier in time.

## 3. How to design a proxy metric — a repeatable process

1. **Name the true long-term outcome first**, explicitly, before looking for a proxy. Skipping this and reaching straight for "what's easy to measure this week" is how proxies drift away from what actually matters.
2. **Generate several candidate short-term signals**, not just one. Pull from multiple proxy types in the table above — behavioral, funnel, quality — rather than defaulting to whichever metric is already on a dashboard.
3. **Validate each candidate against historical realized outcomes.** Compute the actual correlation (or fit a regression, if you have the infrastructure) between each candidate and the true long-term metric using historical cohorts where the long-term outcome is already known.
4. **Prefer combination over a single "best" candidate** when the infrastructure allows it — the worked example's jump from ρ=0.68 (best single proxy) to ρ=0.81 (combined index) is the general pattern, not a coincidence: different proxies usually capture different, only-partially-overlapping slices of what drives the real outcome.
5. **Stress-test the mediating assumption.** Ask directly: could treatment plausibly move the true long-term outcome *without* moving any of my chosen proxies? If yes (e.g., a change that improves loyalty through brand trust rather than any measured behavior), the proxy is incomplete regardless of how well it historically correlated.
6. **Set a re-validation cadence up front**, before shipping the proxy into production use — not as an afterthought once someone notices it's stopped working.

## 4. How to handle a proxy once it's live — do's and don'ts

| Do | Don't |
|---|---|
| Re-validate the proxy against real long-term outcomes on a fixed schedule (e.g., quarterly or whenever a new cohort's true outcome becomes available) | Treat a validation done once, years ago, as permanently trustworthy |
| Track a negative/early-warning proxy *alongside* your primary proxy, the same way you'd pair an OEC with guardrails | Optimize a single positive proxy in isolation and assume harm would show up there too |
| Reserve full surrogate-index modeling for high-stakes or frequently-repeated experiment categories | Build a full regression-based surrogate index for every minor, low-stakes test — the overhead usually isn't justified |
| Document the correlation strength and the mediating assumption behind your chosen proxy so future teams know its limits | Let a proxy metric quietly become "the OEC" in institutional memory with no record of how well-validated it actually is |
| Watch for the proxy becoming a Goodhart's Law target — a metric that's easy to move without moving the true outcome will eventually get moved that way | Assume a proxy that was hard to game when chosen will stay hard to game as teams learn to optimize for it |
| Recompute correlation strength after any meaningful product or user-base shift (new user segment, platform change, major redesign) | Assume the proxy's predictive power is stable just because the product "hasn't changed that much" |
| Use the lightweight single-proxy approach as the practical default for everyday experimentation | Assume every company runs full academic surrogate-index modeling for routine tests — most don't, and shouldn't |

## 5. Extending the worked example

The chapter's streaming-product case validated three candidates against 12-month retention: week-1 open rate (ρ=0.35), week-2 core-actions rate (ρ=0.68), and a combined index (ρ=0.81). Two things worth adding to that:

- **Week-1 open rate's weak correlation (ρ=0.35) is itself informative** — it's flagged as "too easily gamed by notification spam," which is a preview of the Goodhart's Law risk: an easy-to-move metric that's weakly tied to the true outcome is exactly the kind of proxy that looks fine in a validation study but degrades fastest once teams start optimizing it directly.
- **The negative/early-warning layer is missing from the original example and worth adding**: alongside the combined engagement index, a real onboarding program would also want a proxy like early support-ticket rate or early churn flags tracked as a guardrail — catching a redesign that boosts "core actions completed" while quietly frustrating a meaningful subset of new users.

## 6. Comprehension check

1. Pick two proxy types from the table in Section 2 and explain a scenario where you'd need both simultaneously, not just one.
2. A team wants to skip historical validation because "the metric is obviously related to retention." What's the risk in that reasoning, and what would you ask them to do instead?
3. Why does combining multiple weakly-correlated proxies often outperform picking the single best one, even before you know the exact regression weights?
4. Design a re-validation trigger policy (not just a fixed schedule) — what product changes should force an unscheduled re-check of a proxy's correlation with the true outcome?
5. A proxy metric was strong (ρ=0.7) two years ago and has never been re-checked. What's the most likely failure mode, and how would you detect it without waiting for another full historical validation cycle?
