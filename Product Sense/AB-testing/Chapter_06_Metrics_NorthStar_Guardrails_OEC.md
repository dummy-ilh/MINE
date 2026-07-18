# Chapter 6: Metrics — North Star, Guardrails, and the Overall Evaluation Criterion (OEC)

---

## 1. The Big Picture (Intuition)

Everything covered elsewhere in this curriculum (hypothesis testing, confidence intervals, sample size/power) answers "did a metric move, and can we trust that?" This chapter answers a harder, prior question: **which metric should you even be testing, and what do you do when a change moves five metrics in five different directions?**

Without a pre-committed answer to that question, teams tend to retroactively pick whichever metric looks good after the fact — a subtle form of p-hacking at the *metric-selection* level rather than the statistical-test level. The concepts below — **North Star**, **OEC**, and **Guardrails** — are the toolkit for locking in "how we'll judge this" *before* you see results.

### Layman analogy
Imagine coaching a basketball player using only "points scored." A player could rack up huge point totals by taking terrible shots on every possession, ignoring defense and never passing — the team as a whole suffers even as this one stat looks great. A **North Star metric** is like "points scored" — useful, but dangerous if it's the *only* thing you look at.

**Guardrails** are like the other stats you check to make sure the star player isn't quietly wrecking the team: turnovers, fouls, plus/minus. You're not trying to *maximize* low turnovers — you're just making sure they don't spiral out of control while chasing points.

The **OEC** is like a single overall "player rating" that blends points, turnovers, defense, and team impact into one number (or one decision rule) — so nobody can game just one stat and call it a win. In product terms: you don't want a feature that boosts clicks by 10% while quietly increasing complaints 300% — the OEC (or the guardrail check) is what stops that from shipping.

---

## 2. Core Definitions

- **North Star Metric**: the single metric a team/product optimizes toward, meant to capture long-term, sustainable value delivered to users (and by extension, the business). It should be a leading indicator of business success, not just an easily-movable proxy.
- **Overall Evaluation Criterion (OEC)**: a single metric, or an explicit pre-committed combination/decision-rule over multiple metrics, used specifically to make ship/no-ship decisions in an experiment — designed so a team can't win on one dimension while quietly damaging another. Term coined and popularized by Ron Kohavi's experimentation work at Microsoft/Google-adjacent teams.
- **Guardrail Metrics**: metrics you monitor but don't optimize — they exist to catch harm the primary metric/North Star wouldn't detect on its own (e.g., latency, crash rate, revenue, spam rate). A test can win on the primary metric and still get blocked if it breaches a guardrail.

**How these three relate**: the North Star is your long-term compass; the OEC is the concrete, pre-registered decision rule for a *specific experiment* (often built around a proxy for the North Star); guardrails are the safety net that stops the OEC from being "won" at the expense of something important it doesn't capture.

---

## 3. North Star Metric — Selection Criteria

A good North Star metric should:
- **Correlate with long-term business outcomes** — validated via historical data (does moving this metric actually predict revenue/retention months later?).
- **Be sensitive enough** to detect real product changes within a reasonable experiment duration (ties directly to the sample-size/power chapter — an insensitive North Star means every experiment is underpowered by construction).
- **Be hard to "game" via short-term tricks** — e.g., "time spent" can be inflated by making the product more confusing, not more valuable (a known Facebook/YouTube-era lesson).

### Why you can't just pick the "obvious" metric
Naively, for a checkout redesign, you might think "the OEC is obviously conversion rate." But short-term, directly-observable metrics are often poor proxies for what actually matters long-term:
- **Click-through rate** can be trivially increased with clickbait, at the cost of long-term trust and retention.
- **Session count / time-on-site** can go up because the product got *worse* (harder to find what you need = more sessions/more time), not better.
- **Short-term conversion** can increase via aggressive upsell tactics that increase churn 3 months later.

This is why the ideal North Star is a **long-term, business-relevant outcome** (customer lifetime value, long-term retention) — but you usually can't wait months to observe that in every experiment. This tension — wanting a long-term outcome but needing a short-term, measurable proxy — is exactly why **proxy metrics** exist: teams often use a short-term proxy that's been validated to correlate with the true North Star over longer horizons (e.g., a "quality score" validated against long-term retention, or "7-day retention" validated as a proxy for LTV).

---

## 4. Constructing an OEC — Three Approaches

**Approach 1 — Single well-chosen proxy metric.** Pick the single short-term metric most causally/predictively linked to your actual long-term goal (e.g., "7-day retention" as a proxy for LTV, validated by prior research showing correlation).

**Approach 2 — Weighted composite metric.** Combine several metrics into one weighted score:

$$OEC = w_1 \cdot m_1 + w_2 \cdot m_2 + \dots + w_k \cdot m_k$$

Weights $w_i$ reflect the relative business importance of each component metric $m_i$, **normalized to comparable scales first** (you can't directly add "conversion rate" and "session length" without standardizing them — e.g., via z-scores or percent-change from baseline). Changing relative weights changes which tradeoffs are "acceptable" — e.g., weighting revenue heavily vs. user satisfaction heavily leads to different ship/no-ship decisions on the same experiment. This weighting is a **business/product judgment call, not a purely statistical one** — a key point for demonstrating product sense in an interview, and interviewers will likely push on "how did you get those specific weights?"

**Approach 3 — Guardrail-gated primary metric (most common in practice — lead with this in an interview).** Pick ONE primary metric as the actual decision criterion, and specify a set of **guardrail metrics** (Section 6) that must not regress beyond a pre-specified threshold, regardless of what the primary metric does. This avoids the difficulty of assigning defensible numeric weights (Approach 2) while still preventing "we won on the primary metric but silently broke something important" outcomes. In practice, many companies (Google, Microsoft's published experimentation literature, Booking.com) use this as a *decision framework* rather than a literal single formula — e.g., "ship only if the primary metric improves AND no guardrail breaches" — rather than forcing everything into one number.

---

## 5. Worked Example — OEC in Action (Video Recommendation Algorithm)

You're testing a new recommendation algorithm on a video platform. Candidate metrics observed post-experiment:

| Metric | Result | Significant? |
|---|---|---|
| Click-through rate (CTR) | +3.2% | Yes |
| Avg. watch time per session | −1.1% | No (CI includes 0) |
| 7-day return rate | +0.4% | No (underpowered — too few days elapsed) |
| Video completion rate | −2.8% | Yes |

**Without a pre-committed OEC**, this is a mess — you could tell a story to justify shipping ("CTR is up!") or not shipping ("completion rate dropped, people are clicking then bailing") depending on which narrative you wanted to tell.

**With a pre-committed OEC** (defined in the experiment design doc *before launch* as: "primary metric = 7-day return rate; guardrails = watch time (must not drop >2%), completion rate (must not drop >3%)"):
- 7-day return rate is not significant → **inconclusive on primary metric**; don't ship based on current data.
- Guardrails: watch-time drop of 1.1% is within the 2% tolerance (pass); completion-rate drop of 2.8% is within the 3% tolerance but close to the line (pass, but flag for monitoring).
- **Decision**: don't ship yet — either extend the experiment for more power on the primary metric, or treat this as "no clear win" and iterate on the algorithm.

This is radically different from the "vibes-based" first analysis — the OEC forces a specific, defensible, pre-committed decision rule instead of a post-hoc narrative built around whichever number looks best.

---

## 6. Guardrail Metrics — Deep Dive

### Common categories
- **Latency/performance**: page load time, time-to-interactive. A feature that improves conversion but doubles load time may be a net loss even if the primary metric looks great.
- **Quality/trust metrics**: crash rate, error rate, complaint/report rate, unsubscribe rate — protect against "we optimized engagement by making the product objectively worse in some dimension."
- **Revenue/monetization** (when revenue isn't the primary metric): you don't want a UX experiment to accidentally tank ad revenue, or vice versa.
- **Company-wide "trust" metrics**: at large companies, standing guardrails often apply to almost *every* experiment regardless of team (e.g., overall query volume, overall active users, crash-free sessions) — company-wide tripwires that catch unexpected systemic harm.

### Guardrails are tested differently than the primary metric (the L5-level insight)
- **Primary metric**: you're trying to detect an improvement — standard two-sided (or, if pre-registered, one-sided-for-improvement) test, powered to detect a specific MDE.
- **Guardrail metric**: you're trying to **rule out harm** — often tested as **one-sided in the "harm" direction** ("did this get significantly *worse*"), and importantly, you usually want a **much larger, pre-committed non-inferiority margin** rather than testing against exactly zero.

### The non-inferiority test
Instead of $H_0: \Delta = 0$, a guardrail non-inferiority test is:

$$H_0: \Delta \leq -\delta_{margin} \quad \text{(truly unacceptable regression)}$$
$$H_1: \Delta > -\delta_{margin} \quad \text{(acceptable — within tolerance)}$$

where $\delta_{margin}$ is the maximum tolerable regression pre-agreed as acceptable (e.g., "latency can regress by up to 50ms, that's within tolerance"). This is subtly different from testing against a null of exactly zero, and is the statistically correct way to formalize "didn't meaningfully break anything," rather than requiring "literally no measurable change" — an unreasonably strict, often unachievable bar (nearly any change adds *some* latency at sufficient sample size).

Guardrails can also be **directional** in a simpler sense — you don't block a launch because latency got *better*; you only care about the harmful direction.

### Worked example — non-inferiority in action
You're testing a new recommendation module that adds visual complexity to a page. Primary OEC: click-through rate on recommendations. Guardrail: page load latency (p50).

- Control p50 latency: 850ms
- Treatment p50 latency: 890ms (a 40ms / ~4.7% regression)
- Pre-agreed non-inferiority margin: latency regression must not exceed 60ms (agreed with the perf team before launch, based on known thresholds where users start reporting perceived slowness)

**Non-inferiority test setup:**
$$H_0: \Delta_{latency} \geq 60ms \quad \text{vs.} \quad H_1: \Delta_{latency} < 60ms$$

Given the observed 40ms regression and a computed SE of 8ms:

$$z = \frac{40-60}{8} = \frac{-20}{8} = -2.5$$

For a one-sided test at α=0.05, $z_{critical} = -1.645$. Since $-2.5 < -1.645$, we reject $H_0$ (the "unacceptably bad" null) — meaning we have statistical evidence the true regression is **less than** the 60ms tolerance threshold. **Guardrail passes** — not because latency didn't regress at all (it did, by 40ms), but because we've statistically confirmed the regression is within the pre-agreed tolerable range.

**Key teaching point**: this is a fundamentally different question than "is there a statistically significant regression in latency?" (there almost certainly is, given enough sample size). The non-inferiority framing is what lets you distinguish "technically measurable" from "practically consequential" — the statistical-vs-practical-significance distinction, now applied specifically to the guardrail context.

---

## 7. Levers — What Controls These Concepts

**Choice of North Star**
- Too close to a vanity metric (e.g., raw "clicks") invites gaming via dark patterns. Too far from user value (e.g., "quarterly revenue" for a single UI experiment) makes it too slow/noisy to move in a 2-week test.
- Teams often substitute a validated **proxy metric** for short-term experiments (see Section 3).

**Guardrail thresholds**
- Too tight → false alarms block good launches. Too loose → real harm gets through. Typically set from the guardrail's historical variance and the org's risk tolerance — and ideally with the non-inferiority margin ($\delta_{margin}$) set by domain experts (UX research, infra/perf teams), **not by statisticians alone**.
- Guardrails are usually tested with a looser statistical bar in the "harm" direction — high sensitivity to detect negative movement is prioritized even at the cost of some false positives, because the cost of shipping a harmful feature is asymmetric to the cost of a false alarm.

**Metric gaming resistance**
- Metrics trivially inflated by manipulative UX (infinite scroll, notification spam, dark patterns) need companion guardrails specifically to catch when "engagement" is being bought at the cost of user trust — the well-known lesson from social platforms that optimized raw engagement without wellbeing/complaint guardrails.

**Weighting in a composite OEC**
- A business/product judgment call, not a purely statistical one (see Approach 2, Section 4).

**Guardrail proliferation**
- Adding too many guardrails increases the chance that *some* guardrail trips by pure chance — a multiple-testing problem. More guardrails = more chances for a false alarm to block a real win. Guardrail sets should be curated, not exhaustive; apply a multiple-comparisons correction (e.g., Bonferroni, Benjamini-Hochberg/FDR) if a genuinely large set is needed, and route lower-priority metrics to monitoring dashboards instead of hard ship-blocking gates.

---

## 8. Goodhart's Law and Metric Gaming

**"When a measure becomes a target, it ceases to be a good measure."** If teams know the OEC is exactly CTR, they'll optimize CTR specifically, sometimes at the expense of the underlying goal CTR was originally a proxy for. This is a real, recurring failure mode at metric-driven companies (the "time spent" and "clickbait CTR" examples above are both instances of it) and is worth flagging proactively in an interview — it signals you think about second-order effects of metric-driven organizations, not just the immediate experiment.

---

## 9. Production Considerations

- **The OEC (and guardrails) must be defined and written down BEFORE the experiment launches** — the same pre-registration principle as choosing test direction ahead of time. An OEC chosen after seeing results isn't an OEC, it's a post-hoc justification.
- **Composite OECs need periodic revalidation** — the weights $w_i$ encode assumptions about what drives long-term value; if those assumptions go stale (a metric that used to predict retention no longer does, because user behavior shifted), the OEC needs to be re-derived, not treated as permanent.
- **"Trustworthy metric" framing (Kohavi/Google experimentation culture)**: design OECs and guardrails around metrics that are sensitive enough to detect real effects, but not so noisy or gameable that teams learn to exploit them. Worth bringing up if asked how you'd choose an OEC at a company like Google specifically.
- **Standing/company-wide guardrails should be automated into the experimentation platform**, not manually checked per-experiment — at scale, thousands of experiments run concurrently, and a small number of universal tripwire metrics (crash rate, major error rate) should auto-flag any experiment that trips them, regardless of owning team.
- **Guardrail violations should generally block shipping even if the primary metric is a huge win** — this is a policy/cultural decision needing org-level buy-in ahead of time; without it, teams under pressure to ship will rationalize past guardrail failures.
- **Beware guardrail metric proliferation** (see Section 7 — multiple-testing risk).

---

## 10. Interview Traps (Consolidated)

1. **Proposing an OEC only after seeing results**, rather than emphasizing it must be locked in during experiment design — the pre-registration principle, applied to metric choice instead of test direction.
2. **Choosing a purely short-term, easily-gamed metric** (e.g., raw CTR) as the OEC without acknowledging its long-term risks (clickbait, dark patterns).
3. **Proposing a composite weighted metric** (Approach 2) without acknowledging the difficulty and inherent subjectivity of choosing defensible weights.
4. **Conflating the OEC with guardrails** — not distinguishing "the actual ship/no-ship decision driver" from "metrics that must not regress but aren't the primary target." A very common and revealing mistake.
5. **Testing guardrails with the exact same statistical setup as the primary metric** (two-sided, against exactly zero) instead of recognizing the non-inferiority framing is usually more appropriate.
6. **Treating "guardrail showed a statistically significant regression" as automatically disqualifying**, without checking whether the regression is within a pre-agreed *practically* tolerable margin — conflating statistical and practical significance again.
7. **Not mentioning guardrail thresholds need non-statistical, domain-expert input** (UX research, infra team, etc.) to set meaningfully.
8. **Proposing an unbounded number of guardrail metrics** without acknowledging the multiple-testing risk this creates.

---

## 11. Famous Interview Q&A

**Q: Your team's North Star is "time spent in app." A new feature increases time spent by 20% but user satisfaction survey scores drop. What's going on, and what would you do?**
A: This is a classic case of a North Star metric being gamed unintentionally — time spent went up not because the product became more valuable, but likely because it became more confusing, frustrating, or addictive in a way users don't actually want (harder navigation, more required steps, manipulative engagement loops). I'd treat satisfaction as a guardrail here and block the launch despite the North Star win, then dig into *why* time spent increased — segmenting by user type, looking at qualitative feedback — to understand if this is genuine added value or an artifact of friction. Longer term, I'd advocate reconsidering "time spent" as a North Star in favor of something more resistant to this failure mode, like a validated engagement-quality composite.

**Q: How would you decide the guardrail metrics for a new checkout flow experiment?**
A: Three buckets: system health (page load latency, error rate on checkout submission — a broken checkout is catastrophic even if it "looks like" higher conversion due to retries), business health (revenue per user, refund/chargeback rate — a flow that increases orders but also increases fraud or returns isn't actually a win), and user trust (complaint rate, support ticket volume tied to checkout). I'd also want directional guardrails — only care if these get *worse*, not better — and set thresholds based on the historical variance of each metric so we're not blocking launches on noise.

**Q: Explain why a company might use an OEC instead of just looking at multiple metrics separately.**
A: Looking at metrics separately creates ambiguity when they disagree — e.g., North Star up 2%, revenue flat, complaints up 5% — different stakeholders can walk away with different ship/no-ship conclusions based on which metric they personally weight most. An OEC forces the organization to pre-commit to relative weights and tradeoffs *before* seeing results, removing post-hoc rationalization ("we'll call it a win because I like this metric") and making the ship decision more consistent and defensible across many experiments run by different teams.

**Q: A stakeholder wants to add 15 different metrics to the "must not regress" guardrail list for every experiment. What's the risk, and how would you push back?**
A: The main risk is an inflated false-positive rate from multiple testing — checking 15 guardrails at a 5% significance threshold each means a high chance that at least one flags "harm" purely by chance, even if the feature is actually fine, potentially blocking good launches unnecessarily. I'd push back by proposing a smaller set of high-priority guardrails tied to genuine catastrophic risks (system health, trust, core revenue), apply a multiple-comparisons correction if more are truly needed, and route lower-priority metrics to monitoring dashboards instead of hard ship-blocking gates.

**Q: Why is a guardrail metric typically tested with a non-inferiority framework rather than a standard two-sided test against zero?**
A: Because the question you actually care about isn't "did this metric change at all" (almost anything changes it slightly at scale) but "did it change by *more than* an amount we've pre-agreed is tolerable." A standard two-sided test against zero will flag "significant" for trivial regressions given enough sample size, generating false alarms. Non-inferiority testing against a pre-committed margin ($\delta_{margin}$) directly encodes "practically consequential" rather than "technically detectable."

**Q: In the latency worked example, the observed regression (40ms) was smaller than the tolerance margin (60ms), yet a naive two-sided test against zero would likely still show "statistically significant regression." Reconcile these two facts.**
A: These aren't contradictory — they're answering different questions. The two-sided test against zero asks "is there any detectable change at all," and with enough sample size, almost any nonzero regression will clear that bar. The non-inferiority test asks a different, more useful question: "is the regression small enough to be within our pre-agreed tolerance." A regression can be statistically significant (clearly nonzero) *and* pass the guardrail (safely below the harm threshold) at the same time — that's exactly the statistical-vs-practical-significance distinction playing out.

**Q: Your primary OEC shows a strong, statistically significant win, but one guardrail metric shows a small regression that's technically outside its pre-agreed tolerance. What do you recommend?**
A: I'd default to **not shipping as-is**, per the org-level policy that guardrail violations block shipping regardless of primary-metric wins — otherwise the guardrail is guardrail in name only. I'd then dig into whether the regression is a real, generalizable effect or a fluke (check robustness, segment by user type), and consider whether a modified version of the feature could capture the primary win without the guardrail cost, rather than either blindly shipping or blindly killing the feature.

**Q: Your team wants to add a 5th "candidate OEC metric" halfway through an already-running experiment because early results on the original OEC look weak. What do you tell them?**
A: This is exactly the metric-selection version of p-hacking / post-hoc rationalization that pre-registering the OEC is designed to prevent. Changing the decision criterion after partially seeing results — even if well-intentioned — undermines the credibility of the eventual conclusion, because you're now more likely to land on whichever metric happens to look good. I'd recommend sticking with the pre-committed OEC for this experiment's decision, but noting the new candidate metric as a hypothesis to test in a *future*, properly pre-registered experiment.

---

## 12. L5-Differentiating Talking Points (Consolidated)

- Leading with the guardrail-gated single-primary-metric approach (Approach 3) as the pragmatic default, while being able to explain the weighted-composite alternative and its tradeoffs, shows both practical judgment and theoretical range.
- Bringing up **Goodhart's Law** unprompted when discussing OEC design signals you think about second-order effects of metric-driven organizations, not just the immediate experiment.
- Connecting OEC design to the general **proxy-metric / surrogate-index problem** ("short-term measurable vs. long-term true goal") shows you see this as one instance of a recurring tension in experimentation, not an isolated concept.
- Being explicit that OEC validation is an **ongoing process** (re-checking that your proxy still predicts your true goal), not a one-time setup task, reflects real production maturity.
- Introducing the **non-inferiority testing framework** by name, and explaining why it differs from a standard two-sided test against zero, is a strong differentiator — most candidates only know the standard hypothesis-test framing.
- Explicitly stating that guardrail thresholds are set jointly with domain experts (not purely a stats decision) shows organizational/cross-functional maturity.
- Flagging the **multiple-testing risk of guardrail proliferation** demonstrates you see guardrail design as part of the broader statistical rigor of the whole experimentation system, not an isolated checklist item.
- Being explicit that "guardrail regressed, but within tolerance" vs. "guardrail regressed beyond tolerance" require org-level policy on which one blocks shipping — showing you understand this is partly a governance problem, not purely a technical one.

---

## 13. Comprehension Check (Self-Test)

1. Why can't you just use the most obviously-relevant short-term metric (e.g., CTR) as your OEC without further thought?
2. Describe the three approaches to constructing an OEC, and state one tradeoff for each.
3. In the video-recommendation worked example, why does having a pre-committed OEC change the ship/no-ship decision compared to a post-hoc analysis of all four metrics?
4. What is Goodhart's Law, and how does it apply to choosing and publicizing an OEC within a product team?
5. Your team wants to add a 5th "candidate OEC metric" halfway through an already-running experiment because early results look weak. What do you tell them?
6. Why is a guardrail metric typically tested with a non-inferiority framework rather than a standard two-sided test against zero?
7. In the latency worked example, the observed regression (40ms) was smaller than the tolerance margin (60ms), yet a naive two-sided test against zero would likely still show "statistically significant regression." Reconcile these two facts.
8. Who should be involved in setting a guardrail's non-inferiority margin, and why shouldn't this be a purely statistical decision?
9. What risk does adding an unbounded number of guardrail metrics introduce, and how would you mitigate it?
10. Your primary OEC shows a strong, statistically significant win, but one guardrail metric shows a small regression that's technically outside its pre-agreed tolerance. What do you recommend, and why?

---
*This tutorial merges three chapters into one self-contained reference: (1) an overview chapter introducing North Star metrics, guardrails, and OEC together; (2) a deep-dive on OEC design (the three construction approaches, Goodhart's Law, pre-registration); and (3) a deep-dive on guardrail metrics specifically (non-inferiority testing, worked latency example, threshold-setting governance). No external chapter references are needed — all cross-referenced concepts (pre-registration/one-sided testing, statistical vs. practical significance, multiple-testing correction, proxy metrics) are explained inline above.*
