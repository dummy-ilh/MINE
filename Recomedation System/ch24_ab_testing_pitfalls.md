# Chapter 24: A/B Testing Pitfalls Specific to Recsys (Network Effects, Position Bias, Feedback Loops)

## 1. Intuition

Standard A/B testing assumes each user's outcome is **independent** of every other user's treatment assignment, and that the treatment effect measured today predicts the effect that will persist going forward. Recommendation systems violate both assumptions in specific, well-documented ways — this chapter names those violations precisely, since "how would you A/B test this" is one of the most reliable L5 follow-up questions across this entire curriculum, and generic A/B testing knowledge (assumed already known — see Ch. 2's note on offline-vs-online evaluation) isn't sufficient without these recsys-specific caveats.

## 2. Position Bias — Revisited in the A/B Context

Chapter 2 and Chapter 6 both flagged that item position affects click probability independent of true relevance. In an A/B test specifically, this creates a subtle measurement problem: if the treatment (new ranking model) and control (old model) produce genuinely different orderings, and you naively compare raw click-through-rate between arms, some of the difference reflects **which positions items ended up in**, not purely whether the new model identified more relevant items. A treatment that's equally good at identifying relevance but happens to shuffle a few borderline-relevant items into top positions (versus the control's ordering) can show an inflated CTR lift that doesn't reflect genuine relevance-quality improvement — a specific instance of a general confound that needs explicit control (e.g., position-normalized metrics, or metrics like NDCG that already discount by position, rather than raw CTR) in a recsys A/B test.

## 3. Network Effects and Interference (SUTVA Violation)

Standard A/B testing relies on the **Stable Unit Treatment Value Assumption (SUTVA)**: one user's outcome shouldn't depend on which arm *other* users were assigned to. Recsys frequently violates this:

- **Social/network platforms**: if a friend's feed ranking changes what content they engage with/share, that can affect what appears in *your* feed even if you're in the control group — your outcome now depends on your friend's treatment assignment, violating SUTVA directly.
- **Marketplace/two-sided platforms**: in a marketplace recommender (e.g., ride-sharing, e-commerce), if a treatment arm changes how supply (drivers, sellers) gets allocated to demand (riders, buyers), this can affect the pool of available supply/inventory seen by the *control* arm too — the two arms aren't truly isolated because they're drawing from a shared, finite resource pool.
- **Content ecosystem effects**: if a new ranking algorithm shifts engagement toward a different set of creators/sellers, this can change those creators'/sellers' incentives and future content production — an effect that plays out over a timescale far longer than a typical A/B test window, and that isn't contained to just the treatment-group users at all (it affects the whole content ecosystem, including what content control-group users eventually see too).

**Mitigation approaches**: cluster-based randomization (randomize at the level of geographic region, social cluster, or marketplace segment rather than individual user, so that interference happens *within* a treatment condition rather than *across* treatment and control) and switchback experiments (randomize the *same* population across time periods rather than across users, common in marketplace settings where user-level randomization is especially prone to interference).

## 4. Feedback Loops and Novelty/Primacy Effects

Directly connecting to Chapter 21's filter-bubble discussion: a new ranking model, once deployed, starts influencing what data gets logged (which items get shown, therefore which get engagement), which in turn affects any model that gets retrained on that data — meaning the treatment arm's *own future performance* is partly determined by the feedback loop it creates during the test, not just its initial ranking quality. This is why short A/B tests can be systematically misleading for models that have meaningfully different exploration/diversity characteristics (Ch. 21, 22) — a test window too short to let the feedback loop play out won't reveal whether a treatment's compounding effects (e.g., narrowing content diversity over time) are actually harmful.

A related, opposite-direction confound: **novelty effects** — users may engage more with a treatment simply because it's *different/new* (curiosity), not because it's genuinely better, and this effect typically fades over the test duration; conversely, **primacy effects** — users accustomed to the old system's behavior may show short-term-depressed engagement with a genuinely better new system simply due to unfamiliarity, before adapting. Both effects mean a treatment effect measured in the first few days of a test can be a poor predictor of the steady-state, longer-run effect — a specific, well-documented reason recsys A/B tests are often run longer, or specifically monitored for a stabilizing trend, rather than concluded based on early results alone.

## 5. Worked Example — A Concrete A/B Test Gone Wrong

Suppose a team A/B tests a new ranking model that happens to promote more "clickbait-style" thumbnails/titles (higher predicted CTR per Chapter 1's clickbait caveat) against the existing production model.

**What a naive short (3-day) A/B test measuring raw CTR would show**: treatment arm shows a clear CTR lift — the new model appears to be a strong win.

**What's actually happening, unpacked using this chapter's concepts**: 
- Some of the lift is a **novelty effect** (Section 4) — users clicking out of curiosity at unfamiliar-looking content, likely to fade.
- The clickbait-style items likely have lower **post-click satisfaction** (lower watch-time-percentage, higher rage-clicks/immediate-back-button) — a guardrail metric (Ch. 19) that a CTR-only test would miss entirely.
- If the test runs long enough for the **feedback loop** (Section 4) to kick in, the model retrains on this new clickbait-favoring engagement pattern, potentially *further* amplifying the clickbait tendency over time — a compounding effect invisible in a short test window.
- If this is a social/creator platform, creators may observe the clickbait-rewarding pattern and start producing more clickbait content themselves (**ecosystem effect**, Section 3) — an effect that plays out over weeks/months, affecting even control-arm users' available content pool eventually, and is essentially invisible within the test's isolated treatment/control framing entirely.

The correct response: extend the test duration to let novelty effects fade and get a read on the feedback-loop trend, add explicit guardrail metrics beyond raw CTR (watch-time-percentage, post-click satisfaction surveys, longer-run retention), and consider whether cluster/ecosystem-level effects need a different experimental design (e.g., a holdback test at the market/region level rather than pure individual-user randomization) to properly capture ecosystem-level consequences.

## 6. Production Considerations

- Recsys teams commonly run **holdback groups** — a small population permanently excluded from a given change (or kept on an older model) for an extended period, specifically to measure long-run cumulative effects that a standard time-boxed A/B test would miss — this is a direct practical response to the feedback-loop and long-run-effect concerns in Sections 3-4.
- Guardrail metrics (Ch. 19) in recsys A/B tests routinely include: session length/frequency over time (not just immediate CTR), diversity of consumption (Ch. 21), creator/seller ecosystem health metrics (for marketplace/content platforms), and various measures of user-reported satisfaction (surveys) alongside pure behavioral engagement signals — precisely because pure short-term engagement metrics are exactly the ones most susceptible to the pitfalls named in this chapter.
- Statistical significance calculations themselves need adjustment when SUTVA is violated (Section 3) — standard variance estimators assume independence across users, and interference violates this, generally requiring cluster-robust variance estimation or a fundamentally different experimental unit (cluster/region rather than individual user) to get valid confidence intervals at all.

## 7. Interview Traps

- Proposing a standard, short-duration, simple-CTR-metric A/B test for a recsys change without naming any of this chapter's specific caveats — a strong signal of only having generic (non-recsys-specific) A/B testing knowledge.
- Not recognizing that a marketplace/social/content-ecosystem recommender may violate SUTVA, and proposing plain user-level randomization without considering cluster-based or switchback alternatives.
- Treating novelty effects and feedback-loop compounding as the same phenomenon — they point in different directions (novelty typically inflates short-term treatment effect and fades; feedback loops can compound and grow over a longer horizon) and require different experimental durations/monitoring approaches to detect.
- Forgetting to propose guardrail metrics beyond the primary engagement metric — a repeated theme throughout this curriculum (Ch. 2, 19, 21) that's especially critical to name explicitly in an A/B testing discussion specifically.

## 8. L5-Differentiating Talking Points

- Proactively name SUTVA and give a concrete recsys-specific example of its violation (marketplace supply-sharing, social-graph spillover, or ecosystem/creator-incentive effects) — this precise terminology and concrete grounding is a strong, checkable signal of genuine experimentation maturity beyond generic A/B testing knowledge.
- Distinguish novelty/primacy effects from feedback-loop compounding explicitly, and connect the latter directly back to Chapter 21's filter-bubble/rich-get-richer discussion — showing the throughline across the curriculum's system-level concerns.
- Propose concrete mitigations (cluster/switchback randomization for interference, holdback groups for long-run effects, guardrail metrics beyond raw engagement) rather than only naming the problems — showing you can operationalize the caveats into an actual experimental design.
- Walk through a worked scenario (as in Section 5) showing how a naive short CTR-based test could produce a misleadingly positive result for a genuinely harmful change — this kind of concrete, mechanism-grounded skepticism is exactly what separates L5 experimentation maturity from a textbook "run an A/B test" answer.

## 9. Comprehension Check

1. What is SUTVA, and give a concrete recsys example of it being violated.
2. Why can a short A/B test measuring raw CTR be systematically misleading for a change that shifts content toward more clickbait-style items?
3. What's the difference between a novelty effect and a feedback-loop compounding effect, and how does each affect the reliability of a short-duration test?
4. What is a holdback group, and what specific long-run risk does it help a team measure?
5. Why might standard statistical significance calculations be invalid in a recsys A/B test with network/marketplace interference, and what's a standard fix?
