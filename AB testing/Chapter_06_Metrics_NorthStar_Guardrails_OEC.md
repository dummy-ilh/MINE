# Chapter 6: Metrics — North Star, Guardrails, and the Overall Evaluation Criterion (OEC)

## 1. Definition

- **North Star Metric:** the single metric a team/product optimizes toward, meant to capture long-term, sustainable value delivered to users (and by extension, the business). It should be a leading indicator of business success, not just an easily-movable proxy.
- **Guardrail Metrics:** metrics you monitor but don't optimize — they exist to catch harm the North Star wouldn't detect on its own (e.g., latency, crash rate, revenue, spam rate). A test can win on the North Star and still get blocked if it breaches a guardrail.
- **Overall Evaluation Criterion (OEC):** a single composite score (often a weighted combination of multiple metrics) used specifically to make ship/no-ship decisions in an experiment — designed so a team can't win on one dimension while quietly damaging another. Coined and popularized by Kohavi's experimentation work at Microsoft/Google-adjacent teams.

## 2. Layman Explanation

Imagine you're a coach evaluating a basketball player using only "points scored." A player could rack up huge point totals by taking terrible shots on every possession, ignoring defense and never passing — the team as a whole would suffer even as this one stat looks great. The North Star metric is like "points scored" — useful, but dangerous if it's the *only* thing you look at.

Guardrails are like other stats you check to make sure the star player isn't quietly wrecking the team: turnovers, fouls, plus/minus. You're not trying to *maximize* low turnovers — you're just making sure they don't spiral out of control while chasing points.

The OEC is like a single overall "player rating" that blends points, turnovers, defense, and team impact into one number — so nobody can game just one stat and call it a win. In product terms: you don't want a feature that boosts clicks by 10% while quietly increasing complaints 300% — the OEC (or the guardrail check) is what stops that from shipping.

## 3. Formal Explanation

**North Star selection criteria:**
- Should correlate with long-term business outcomes (validated via historical data — does moving this metric actually predict revenue/retention months later?)
- Should be sensitive enough to detect real product changes in a reasonable experiment duration
- Should be hard to "game" via short-term tricks (e.g., "time spent" can be inflated by making the product more confusing, not more valuable — a known Facebook/YouTube-era lesson)

**Guardrail design:**
- Typically includes: system health (latency, error rate, crash rate), business health (revenue, ad load), and user trust (spam/abuse rate, unsubscribe rate, complaint rate).
- Guardrails are usually tested with a looser statistical bar in the "harm" direction — you want high sensitivity to detect any negative movement, even at the cost of some false positives, because the cost of shipping a harmful feature is asymmetric to the cost of a false alarm.

**OEC construction (formally):**
OEC = Σ wᵢ · metricᵢ (normalized)

Where weights wᵢ reflect the relative business importance of each component metric, and normalization (e.g., z-scoring each metric) ensures no single metric dominates purely due to differences in scale.

In practice, many companies (Google, Microsoft's own published experimentation literature, Booking.com) use OEC as a *decision framework* rather than a literal single formula — e.g., "ship only if North Star improves AND no guardrail breaches AND OEC composite doesn't regress" — rather than reducing everything to one number by force.

## 4. Levers — What Controls It, What Moves It

**Choice of North Star**
- Picking a North Star too close to a vanity metric (e.g., raw "clicks") invites gaming via dark patterns. Picking one too far from user value (e.g., "quarterly revenue" for a single UI experiment) makes it too slow/noisy to move in a 2-week test.
- Teams often use a **proxy metric** for short-term experiments that's been validated to correlate with the true North Star over longer horizons (e.g., a "quality" score validated against long-term retention).

**Guardrail thresholds**
- Setting a guardrail threshold too tight causes false alarms that block good launches; too loose lets real harm through. This threshold is typically set based on historical variance of the guardrail metric and the org's risk tolerance.
- Guardrails can also be *directional* (only care about degradation, not improvement) — e.g., you don't block a launch because latency got *better*.

**Metric gaming resistance**
- Metrics that can be trivially inflated by manipulative UX (infinite scroll, notification spam, dark patterns) need companion guardrails specifically to catch when "engagement" is being bought at the cost of user trust — this is the well-known lesson from social media platforms optimizing raw engagement without guardrails on wellbeing/complaint metrics.

**Weighting in OEC**
- Changing relative weights changes which tradeoffs are "acceptable" — e.g., weighting revenue heavily vs. user satisfaction heavily leads to different ship/no-ship decisions on the same experiment. This weighting is a business/product judgment call, not a purely statistical one — a key point for demonstrating product sense in an interview.

## 5. Famous Q&A (Google / Apple style)

**Q: Your team's North Star is "time spent in app." A new feature increases time spent by 20% but user satisfaction survey scores drop. What's going on, and what would you do?**
A: This is a classic case of a North Star metric being gamed unintentionally — time spent went up not because the product became more valuable, but likely because it became more confusing, frustrating, or addictive in a way users don't actually want (e.g., harder navigation, more required steps, or manipulative engagement loops). I'd treat satisfaction as a guardrail here and block the launch despite the North Star win, then dig into *why* time spent increased — segmenting by user type and looking at qualitative feedback — to understand if this is genuine added value or an artifact of friction. Longer term, I'd advocate reconsidering "time spent" as a North Star in favor of something more resistant to this failure mode, like a validated engagement-quality composite.

**Q: How would you decide the guardrail metrics for a new checkout flow experiment?**
A: I'd think in three buckets: system health (page load latency, error rate on checkout submission — since a broken checkout is catastrophic even if it "looks" like higher conversion due to retries), business health (revenue per user, refund/chargeback rate — since a flow that increases orders but also increases fraud or returns isn't actually a win), and user trust (complaint rate, support ticket volume tied to checkout). I'd also want directional guardrails — I only care if these get *worse*, not better — and I'd set thresholds based on the historical variance of each metric so we're not blocking launches on noise.

**Q: Explain why a company might use an OEC instead of just looking at multiple metrics separately.**
A: Looking at metrics separately creates ambiguity when they disagree — e.g., North Star up 2%, revenue flat, complaints up 5% — different stakeholders can walk away with different ship/no-ship conclusions based on which metric they personally weight most. An OEC forces the organization to pre-commit to relative weights and tradeoffs *before* seeing results, which removes post-hoc rationalization ("we'll call it a win because I like this metric") and makes the ship decision more consistent and defensible across many experiments run by different teams.

**Q: A stakeholder wants to add 15 different metrics to the "must not regress" guardrail list for every experiment. What's the risk, and how would you push back?**
A: The main risk is inflated false-positive rate from multiple testing — checking 15 guardrails at a 5% significance threshold each means a high chance that at least one flags "harm" purely by chance, even if the feature is actually fine, which could block good launches unnecessarily (see the Multiple Testing Correction chapter later in this curriculum). I'd push back by proposing a smaller set of high-priority guardrails tied to genuine catastrophic risks (system health, trust, core revenue), apply a multiple-comparisons correction if more are truly needed, and route lower-priority metrics to monitoring dashboards instead of hard ship-blocking gates.

---
*Next: Chapter 7 — Unit of Randomization: user vs. session vs. device, and interference risks.*
