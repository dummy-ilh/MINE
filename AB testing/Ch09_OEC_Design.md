# Chapter 9: OEC Design

*(Module 2: Metrics & Measurement begins here)*

## 1. Intuition

Modules 1 taught you how to test whether a metric moved. This chapter asks a harder question that sits above the statistics entirely: **which metric should you even be testing, and what do you do when a change moves five metrics in five different directions?**

The **Overall Evaluation Criterion (OEC)** — a term popularized by Ron Kohavi's work at Microsoft/Google-adjacent experimentation research — is the single metric (or explicit combination of metrics) that you commit to using as the deciding factor for ship/no-ship decisions, defined *before* you run the experiment. The intuition: without a pre-committed OEC, teams tend to retroactively pick whichever metric looks good after the fact, which is a subtle form of p-hacking at the metric-selection level rather than the statistical-test level.

## 2. Why You Can't Just Optimize the "Obvious" Metric

Naively, you might think: "we're testing a checkout redesign, so the OEC is obviously conversion rate." But short-term, directly-observable metrics are often poor proxies for what actually matters long-term:

- **Click-through rate** can be trivially increased with clickbait, at the cost of long-term trust and retention.
- **Session count/time-on-site** can go up because your product got *worse* (harder to find what you need = more sessions/more time), not better.
- **Short-term conversion** can increase via aggressive upsell tactics that increase churn 3 months later.

This is why the OEC is ideally a **long-term, business-relevant outcome** (customer lifetime value, long-term retention) — but you usually can't wait months to observe that in every experiment. This tension — wanting a long-term outcome but needing a short-term, measurable proxy — is exactly what Chapter 13 (Proxy Metrics / Surrogate Index) will formalize; OEC design is the practical, immediate version of that same problem.

## 3. Constructing an OEC

**Approach 1 — Single well-chosen proxy metric**: pick the single short-term metric most causally/predictively linked to your actual long-term goal (e.g., "7-day retention" as a proxy for LTV, validated by prior research showing they're correlated).

**Approach 2 — Weighted composite metric**: combine several metrics into a single weighted score:

$$OEC = w_1 \cdot m_1 + w_2 \cdot m_2 + ... + w_k \cdot m_k$$

where weights $w_i$ reflect the relative business importance of each component metric $m_i$ (normalized to comparable scales first, since you can't directly add "conversion rate" and "session length" without standardizing them, e.g., via z-scores or percent-change from baseline).

**Approach 3 — Guardrail-gated primary metric** (most common in practice, and the one worth leading with in an interview): pick ONE primary metric as the actual decision criterion, and specify a set of **guardrail metrics** (Chapter 10) that must not regress beyond a pre-specified threshold, regardless of what the primary metric does. This avoids the difficulty of assigning defensible numeric weights (Approach 2) while still preventing "we won on the primary metric but silently broke something important" outcomes.

## 4. Worked Example

You're testing a new recommendation algorithm on a video platform. Candidate metrics observed post-experiment:

- Click-through rate (CTR): **+3.2%** (statistically significant)
- Average watch time per session: **-1.1%** (not statistically significant, CI includes 0)
- 7-day return rate: **+0.4%** (not statistically significant, underpowered — too few days elapsed)
- Video completion rate: **-2.8%** (statistically significant)

**Without a pre-committed OEC**, this is a mess — you could tell a story to justify shipping ("CTR is up!") or not shipping ("completion rate dropped, people are clicking then bailing") depending on which story you wanted to tell.

**With a pre-committed OEC** (say, defined in the experiment design doc as: "primary metric = 7-day return rate; guardrails = watch time (must not drop >2%) and completion rate (must not drop >3%)"):
- 7-day return rate is not significant → **inconclusive on primary metric**, don't ship based on current data
- Guardrails: watch time drop of 1.1% is within the 2% tolerance (pass), completion rate drop of 2.8% is within the 3% tolerance but close to the line (pass, but flag for monitoring)
- **Decision**: don't ship yet — either extend the experiment for more power on the primary metric, or treat this as a "no clear win" and iterate on the algorithm.

Notice how radically different this is from the "vibes-based" first analysis — the OEC forces a specific, defensible, pre-committed decision rule instead of a post-hoc narrative built around whichever number looks best.

## 5. Production Considerations

- **The OEC must be defined and written down BEFORE the experiment launches** — exactly the same principle as pre-registering test direction (Chapter 8). An OEC chosen after seeing results isn't an OEC, it's just a post-hoc justification.
- **Composite OECs need periodic revalidation** — the weights $w_i$ in Approach 2 encode assumptions about what drives long-term value; if those assumptions go stale (e.g., a metric that used to predict retention no longer does, because user behavior shifted), the OEC itself needs to be re-derived, not treated as permanent.
- **Google's internal experimentation culture (from public talks/papers by Kohavi and others)** emphasizes designing OECs around metrics that are trustworthy (sensitive enough to detect real effects, but not so noisy or gameable that teams learn to exploit them) — this "trustworthy metric" framing is worth bringing up if asked how you'd choose an OEC at Google specifically.
- **Beware Goodhart's Law**: "when a measure becomes a target, it ceases to be a good measure." If teams know the OEC is exactly CTR, they'll optimize CTR specifically, sometimes at the expense of the underlying goal CTR was originally a proxy for. This is a real, recurring failure mode at metric-driven companies and a great thing to flag proactively.

## 6. Interview Traps

- **Trap #1**: Proposing an OEC only after being shown experiment results, rather than emphasizing it must be locked in during experiment design — same pre-registration principle as Chapter 8, applied to metric choice instead of test direction.
- **Trap #2**: Choosing a purely short-term, easily-gamed metric (e.g., raw CTR) as the OEC without acknowledging its long-term risks (clickbait, aggressive dark patterns).
- **Trap #3**: Proposing a composite weighted metric without acknowledging the difficulty and inherent subjectivity of choosing defensible weights — interviewers will likely push on "how did you get those specific weights?"
- **Trap #4**: Not distinguishing between the OEC (the actual ship/no-ship decision driver) and guardrails (metrics that must not regress, but aren't the primary target) — conflating these is a very common and revealing mistake.

## 7. L5-Differentiating Talking Points

- Leading with the guardrail-gated single-primary-metric approach (Approach 3) as the pragmatic default, while being able to explain the weighted-composite alternative and its tradeoffs, shows both practical judgment and theoretical range.
- Bringing up Goodhart's Law unprompted when discussing OEC design signals you think about second-order effects of metric-driven organizations, not just the immediate experiment.
- Connecting OEC design forward to the proxy-metric/surrogate-index problem (Chapter 13) shows you see this as one instance of a general "short-term measurable vs. long-term true goal" tension that recurs throughout experimentation, not an isolated concept.
- Being explicit that OEC validation is an ongoing process (re-checking that your proxy still predicts your true goal) rather than a one-time setup task reflects real production maturity.

## 8. Comprehension Check

1. Why can't you just use the most obviously-relevant short-term metric (e.g., CTR) as your OEC without further thought?
2. Describe the three approaches to constructing an OEC discussed in this chapter, and state one tradeoff for each.
3. In the worked example, why does having a pre-committed OEC change the ship/no-ship decision compared to a post-hoc analysis of all four metrics?
4. What is Goodhart's Law, and how does it apply to choosing and publicizing an OEC within a product team?
5. Your team wants to add a 5th "candidate OEC metric" halfway through an already-running experiment because early results on the original OEC look weak. What do you tell them?

---
*Next: Chapter 10 — Guardrail Metrics*
