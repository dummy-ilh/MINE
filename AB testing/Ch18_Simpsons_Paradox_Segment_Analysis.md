# Chapter 18: Simpson's Paradox & Segment Analysis

*(Closes Module 3: Failure Modes & Diagnostics)*

## 1. Intuition

This chapter closes Module 3 with maybe the single most counterintuitive failure mode in the whole curriculum: **it is mathematically possible for a treatment to look worse than control in EVERY individual segment, while looking better than control in the pooled, aggregate data — or vice versa.** This isn't a rare statistical curiosity; it happens in real production A/B tests whenever segment sizes differ between arms, and it's a favorite "gotcha" scenario in interviews because it directly tests whether you instinctively check segment-level consistency before trusting a pooled result.

The intuition for why it happens: an aggregate metric is a weighted average across segments, where the weights are each segment's *sample size within that arm*. If the treatment changes not just the average outcome, but also the *mix* of segment sizes between arms (e.g., treatment disproportionately attracts a segment that has a different baseline rate), the aggregate comparison can be dominated by this compositional shift rather than reflecting a genuine, consistent treatment effect.

## 2. The Classic Numerical Structure

Simpson's Paradox requires two ingredients: (1) different segments have meaningfully different baseline rates, and (2) the segment mix (relative sizes) differs between treatment and control arms. When both hold, the pooled comparison can reverse the sign of every within-segment comparison.

**Generic worked example** (constructed to show the exact mechanism):

| Segment | Control: conversions/users | Control rate | Treatment: conversions/users | Treatment rate |
|---|---|---|---|---|
| Mobile | 45/500 | 9.0% | 95/1000 | 9.5% |
| Desktop | 180/1000 | 18.0% | 38/200 | 19.0% |
| **Pooled** | **225/1500** | **15.0%** | **133/1200** | **11.1%** |

Notice: **treatment wins in BOTH segments individually** (9.5% > 9.0% on mobile; 19.0% > 18.0% on desktop), but **treatment loses in the pooled comparison** (11.1% < 15.0%). This happens because treatment's traffic mix is heavily skewed toward mobile (1000/1200 ≈ 83% mobile) which has a low baseline rate, while control's traffic mix is more heavily weighted toward desktop (1000/1500 ≈ 67% desktop) which has a high baseline rate. The pooled numbers are comparing "mostly-mobile treatment" against "mostly-desktop control" — not a clean apples-to-apples comparison at all, even though total sample sizes look reasonable.

## 3. Why This Happens in Real A/B Tests (Not Just Contrived Examples)

- **Differential exposure/rollout**: if a feature is rolled out gradually and disproportionately reaches certain platforms/regions first, and randomization is somehow correlated with rollout timing, segment mix can differ between arms even with a technically correct randomization mechanism.
- **SRM-adjacent causes**: some of the same root causes that produce SRM (Chapter 14) — e.g., a bug that causes certain device types to drop out of the treatment funnel more than control — can also produce segment-mix imbalance without necessarily tripping an overall SRM check, since the *total* counts might still look correctly balanced even if segment-level composition differs. This is an important nuance: passing your SRM check does NOT guarantee segment-level composition balance.
- **Genuine heterogeneous treatment effects combined with differential segment growth**: if treatment causes users to convert to a specific segment (e.g., upgrade to a premium tier) at a different rate than control, the resulting segment mix naturally differs between arms as a *consequence* of the treatment itself — a subtly different and more benign case, but one that still requires care in interpretation (are you measuring the treatment's effect on conversion, or its effect on which segment users end up in, or both simultaneously?).

## 4. The Fix: Weighted / Stratified Analysis

Instead of trusting the naive pooled comparison, use a **stratified estimator** that holds segment weights fixed:

$$ATE_{stratified} = \sum_{s} w_s \times (\bar{Y}_{1,s} - \bar{Y}_{0,s})$$

where $w_s$ is a chosen, **fixed** weight for segment $s$ (commonly, the overall population's segment proportions, or an average of the two arms' segment proportions) applied uniformly to both arms' within-segment effects — this directly prevents the segment-mix difference between arms from contaminating the aggregate comparison, since both arms are now being combined using the *same* weights.

**Applying this to the worked example**: using, say, equal weights (50% mobile, 50% desktop) or overall pooled population weights instead of each arm's own (differing) mix:

$$ATE_{stratified} = 0.5\times(9.5\%-9.0\%) + 0.5\times(19.0\%-18.0\%) = 0.5\times0.5\%+0.5\times1.0\% = 0.75\%$$

This correctly shows a **positive** treatment effect (+0.75pp), consistent with both individual segments, resolving the apparent paradox — the stratified estimate reflects the true, consistent within-segment improvement, rather than being distorted by the arms' differing segment compositions.

## 5. Production Considerations

- **Always check segment-level consistency as a standard part of experiment analysis**, not just when something looks suspicious — Simpson's Paradox by definition looks completely normal in the pooled numbers alone; there's no warning sign in the aggregate data itself that tells you to go check segments.
- **This connects directly to Chapter 17 (Multiple Testing)**: checking many segments for treatment effect consistency is itself a multiple-comparisons exercise, so don't over-interpret noise in any single segment's numbers as "the effect differs by segment" without appropriate correction — the goal here is checking for gross directional consistency/paradoxes, not fishing for significant segment-specific effects.
- **CUPED and stratified analysis are related but distinct tools**: CUPED (Chapter 12) reduces variance using a pre-experiment covariate; stratified analysis (this chapter) corrects for segment-mix imbalance between arms. You can and often should use both together — they solve different problems.
- **Segment mix imbalance can sometimes indicate an SRM-adjacent bug even when the top-line SRM check passes** — worth explicitly running segment-level chi-square balance checks (not just the overall count check from Chapter 14) as an additional diagnostic layer.

## 6. Interview Traps

- **Trap #1**: Not recognizing a Simpson's Paradox scenario when presented with "the treatment lost in the pooled numbers but should we check anything else" — this is one of the most classic direct interview prompts in this space, and failing to propose a segment breakdown as your first instinct is a significant miss.
- **Trap #2**: Assuming segment mix imbalance always indicates a bug/problem — as noted in Section 3, it can also be a genuine, benign consequence of the treatment itself (e.g., treatment causes users to upgrade tiers at different rates), which changes how you'd interpret and act on the finding.
- **Trap #3**: Proposing to just "look at the biggest/most important segment" instead of a properly weighted stratified estimate — cherry-picking one segment isn't a principled fix, and doesn't address the aggregate ship/no-ship decision the OEC (Chapter 9) is supposed to answer.
- **Trap #4**: Not connecting this to the SRM chapter — believing that a clean SRM check (Chapter 14) is sufficient to guarantee no compositional issues; SRM only checks total counts, not segment-level composition.

## 7. L5-Differentiating Talking Points

- Immediately proposing a segment-level breakdown as a standard diagnostic step whenever a pooled result looks surprising (or even when it doesn't), rather than needing to be walked into recognizing Simpson's Paradox, is a strong signal of real practitioner instinct.
- Being able to explain, mechanistically, WHY Simpson's Paradox happens (differential segment weights combined with different baseline rates per segment) rather than just recognizing the pattern by name, shows genuine understanding.
- Proposing the stratified/weighted estimator as the fix, and being able to compute it live as in the worked example, moves you from "I've heard of this" to "I know how to actually resolve it."
- Explicitly connecting this chapter to Chapter 14 (SRM doesn't catch this), Chapter 12 (CUPED solves a different problem), and Chapter 17 (segment fishing needs multiple-testing correction) demonstrates you see Module 3's failure modes as an interconnected diagnostic toolkit, not isolated trivia.

## 8. Comprehension Check

1. Explain, in your own words, why it's mathematically possible for treatment to win in every individual segment but lose in the pooled comparison.
2. What two conditions are jointly necessary for Simpson's Paradox to occur in an A/B test?
3. Using the stratified estimator formula, explain why fixing the segment weights (rather than using each arm's own differing segment mix) resolves the paradox.
4. Why doesn't passing a standard SRM check (Chapter 14) guarantee you're safe from Simpson's Paradox-style segment imbalance?
5. Your pooled experiment result shows a negative effect, but you suspect segment mix might differ between arms. Walk through the steps you'd take to check this and correct for it if needed.

---
*End of Module 3: Failure Modes & Diagnostics (Chapters 14-18).*
*Next: Chapter 19 — Conflicting Metrics & Ship Decisions (start of Module 4: Judgment & System-Level Thinking)*
