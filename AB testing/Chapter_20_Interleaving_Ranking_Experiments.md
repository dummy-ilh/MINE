# Chapter 20: Interleaving & Ranking Experiments

## 1. Intuition

Everything so far has assumed a fairly generic "treatment vs. control" split where each user experiences one consistent version of the product. Ranking systems (search results, feed ranking, recommendation lists) break this assumption in an interesting way: **the thing you're testing is an ordering of items, and a standard A/B test on rankings is often much less statistically efficient than it needs to be**, because between-user variance in preferences and behavior is enormous relative to the actual difference two ranking algorithms produce for the same person.

**Interleaving** is a technique developed specifically for ranking experiments (with deep roots in search-engine evaluation research, including work at Google) that sidesteps this problem: instead of showing different users different rankings and comparing aggregate outcomes, you **show the SAME user a blended/interleaved combination of both rankings simultaneously**, and infer which ranking algorithm is "better" based on which algorithm's results the user preferentially interacts with within that blended list.

## 2. Why Interleaving Is So Much More Efficient Than Standard A/B Testing for Rankings

The core insight: in a standard between-subjects A/B test on a ranking algorithm, most of your outcome variance comes from between-user differences in behavior (some users click a lot, some barely click at all, regardless of ranking quality) — the *signal* (algorithm A vs. B being genuinely better) is often small relative to this between-user noise.

Interleaving converts this into a **within-subject comparison**: because the same user sees both rankings blended together in one session, you difference out nearly all of that between-user variance — you're now asking a much more precise question ("within this one user's single session, did they click more items that came from ranking A's list or ranking B's list") rather than the noisier between-user question. This is conceptually the same "control for what's constant across the comparison" principle behind the paired t-test (Chapter 6) and CUPED (Chapter 12), applied to a ranking-specific interleaving mechanism instead of a covariate adjustment.

**Practical consequence**: interleaving experiments often require **orders of magnitude fewer users** than an equivalent between-subjects ranking A/B test to reach the same statistical power — this dramatic efficiency gain is the single most important fact to know about interleaving, and is very likely to come up if a Google interviewer asks about search/ranking experimentation specifically.

## 3. How Interleaving Works Mechanically

**Team-Draft Interleaving** (a common, robust method): imagine two ranking algorithms, A and B, each producing an ordered list of results for the same query. Construct a single blended list by having A and B "take turns" picking their next-favorite unpicked result to add to the shared list (like a fantasy sports draft) — with the picking order randomized per-query to avoid position bias favoring whichever algorithm happens to go first. Each item in the final blended list is tagged (invisibly to the user) with which algorithm "contributed" it. Then observe which algorithm's tagged items the user actually clicks/engages with more.

**Scoring**: if the user clicks more items tagged "A" than tagged "B" (adjusted appropriately for how many each contributed), that's a vote for algorithm A being better *for this user, this query*. Aggregate these within-session votes across many users/queries to determine an overall winner — note that the unit of comparison here is fundamentally different from a standard A/B test's "per-user outcome," it's closer to a paired-comparison / preference-based framework.

**Why randomizing the draft order matters**: if algorithm A always picked first, its results would systematically occupy more top-of-list positions, and top-of-list position alone drives more clicks regardless of quality (a well-documented position bias in ranking/search behavior) — randomizing which algorithm picks first per query cancels this out over many queries.

## 4. Worked Example

Two ranking algorithms, A (current production) and B (candidate), are compared via team-draft interleaving over 10,000 search queries.

- Queries where user clicked more "A-tagged" results: 3,200
- Queries where user clicked more "B-tagged" results: 3,800
- Queries with ties (equal clicks from both, or no clicks): 3,000

**Win rate comparison** (excluding ties, a common convention): $\frac{3800}{3200+3800} = \frac{3800}{7000} \approx 54.3\%$ in favor of B.

**Statistical test**: this is naturally framed as a **binomial test** (or equivalently a sign test) — under $H_0$ (no true preference, 50/50 win rate), is 54.3% out of 7,000 non-tied comparisons significantly different from 50%?

$$SE = \sqrt{\frac{0.5\times0.5}{7000}} \approx 0.00598, \quad z = \frac{0.543-0.500}{0.00598} \approx 7.19$$

This is an overwhelmingly significant result ($z=7.19$ corresponds to $p \ll 0.0001$) — **and notice this used only 10,000 queries**, a sample size that would likely be far too small to detect a similarly-sized effect in a standard between-subjects ranking A/B test, precisely because of the variance reduction from the within-subject, paired comparison structure described in Section 2.

## 5. Limitations and When Standard A/B Testing Is Still Needed

- **Interleaving tells you which ranking users prefer to click on, not necessarily which ranking produces better long-term outcomes** (satisfaction, retention, revenue) — it's a fast, sensitive *relative preference* signal, not a replacement for measuring your actual OEC (Chapter 9). Many teams use interleaving as a fast pre-filter (quickly screen out clearly worse ranking candidates) before running a full, slower standard A/B test on the surviving best candidates to confirm business-metric impact.
- **Interleaving compares exactly two rankings at a time** — testing many ranking variants requires either pairwise interleaving tournaments or extensions like "team-draft interleaving with multiple algorithms," adding complexity.
- **Interleaving can be gamed or biased by algorithms that are tuned to be "clickbaity"** in ways that don't reflect genuine quality (similar concern to Chapter 9's Goodhart's Law point) — an algorithm that surfaces more clickable-but-lower-quality results could win an interleaving comparison while actually harming a true long-term OEC, which is exactly why the standard A/B test on real business metrics remains necessary as a final confirmation step, not something interleaving fully replaces.
- **Interleaving requires meaningfully different rankings to be informative** — if two algorithms produce nearly identical top results for most queries, there's little to interleave and not much signal to extract, regardless of sample size.

## 6. Production Considerations

- **Interleaving is best used as a fast, sensitive early-stage filter, with standard A/B testing reserved for final validation against the true OEC** — this two-stage pattern (fast/cheap relative comparison, then slower/expensive absolute business-metric validation) is a common and efficient real-world workflow at companies with mature ranking/search infrastructure.
- **Position bias correction is essential and non-trivial** — team-draft interleaving handles this reasonably well, but more sophisticated methods (e.g., probabilistic interleaving) exist with different bias-correction properties; knowing team-draft by name and mechanism is usually sufficient depth for an interview unless the interviewer specifically wants to go further.
- **This is a genuinely Google/search-relevant technique** — interleaving methodology has deep historical roots in information retrieval research broadly associated with search engine evaluation, so bringing it up specifically and unprompted when discussing search/ranking product experimentation is a strong, targeted signal for a Google Search & AI-adjacent role.

## 7. Interview Traps

- **Trap #1**: Not knowing interleaving exists as a distinct technique from standard A/B testing, and defaulting to "just A/B test the two rankings" for a ranking-specific question — this is a strong signal of not having ranking/search-specific experimentation depth.
- **Trap #2**: Treating interleaving as a complete replacement for standard A/B testing rather than understanding its proper role as a fast relative-preference filter that still needs OEC-based confirmation.
- **Trap #3**: Not knowing why interleaving is more statistically efficient (missing the within-subject/paired-comparison variance reduction mechanism) — being able to only name the technique without explaining why it works is a shallow answer.
- **Trap #4**: Forgetting to mention position-bias correction (randomized draft order) as an essential component of the method — presenting interleaving as simple naive result-blending without this safeguard is an incomplete and technically wrong description.

## 8. L5-Differentiating Talking Points

- Proactively bringing up interleaving, unprompted, the moment a ranking/search-relevance experiment scenario comes up (rather than defaulting to standard A/B framing) is a strong, targeted signal specifically valuable for a Google Search & AI-adjacent role.
- Explaining WHY interleaving is so much more efficient (within-subject variance reduction, same underlying principle as paired tests/CUPED) rather than just naming the technique, ties this chapter back into the curriculum's broader variance-reduction theme.
- Correctly positioning interleaving as a fast filter that still requires standard A/B validation against the true OEC — not an either/or choice — shows nuanced, practical understanding of how these techniques compose in a real experimentation pipeline.
- Being able to actually compute the win-rate significance test live (as in the worked example) shows you can operationalize the technique, not just describe it conceptually.

## 9. Comprehension Check

1. Explain why interleaving experiments typically require far fewer samples than standard between-subjects A/B tests to detect the same underlying difference in ranking quality.
2. Describe team-draft interleaving mechanically, and explain why randomizing the draft order (which algorithm picks first) is essential.
3. In the worked example, compute whether a 52% win rate (instead of 54.3%) out of the same 7,000 non-tied comparisons would still be statistically significant.
4. Why isn't interleaving a complete replacement for standard A/B testing on business metrics, even though it's more statistically efficient?
5. You're asked to design an experiment comparing two search ranking algorithms at Google. Would you propose interleaving, standard A/B testing, or both? Justify your answer.

---
*Next: Chapter 21 — Full Mock Case Interviews*
