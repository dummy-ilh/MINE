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
# North Star, Guardrail & OEC Metrics — Exhaustive Reference

---

## 0. How the three concepts relate

```
                    ┌─────────────────────┐
                    │   NORTH STAR METRIC  │   ← company/product-level, long-term
                    └──────────┬───────────┘
                               │ decomposes into
                    ┌──────────▼───────────┐
                    │   INPUT / DRIVER      │   ← the "metric tree"
                    │   METRICS             │
                    └──────────┬───────────┘
                               │ operationalized per experiment as
                    ┌──────────▼───────────┐
                    │        OEC            │   ← the metric(s) a specific
                    │ (Overall Evaluation   │     A/B test is judged on
                    │  Criterion)           │
                    └──────────┬───────────┘
                               │ composed of
       ┌───────────────┬───────┴────────┬────────────────┐
       ▼                ▼                ▼                ▼
   PRIMARY          SECONDARY        GUARDRAIL       DIAGNOSTIC /
   METRIC(S)        METRICS          METRICS         SUPPORTING METRICS
```

- **North Star** = the one metric the *whole org* rallies around (quarters/years).
- **OEC** = the metric(s) a *single experiment or feature* is judged by (days/weeks). Every OEC should trace back to the North Star, but not every OEC-moving change moves the North Star directly.
- **Guardrails** apply at *both* levels — company guardrails protect the North Star; experiment guardrails protect the OEC decision from being a false "win."

---

## 1. NORTH STAR METRIC (NSM)

### 1.1 Definition
A single metric that best captures the core value your product delivers to customers, chosen as a proxy for long-term, sustainable business success. Popularized by Sean Ellis / growth teams; central to the "North Star Framework" (Amplitude).

### 1.2 Criteria for a good North Star (the checklist)
| Criterion | What it means |
|---|---|
| **Expresses value delivered**, not value extracted | "Weekly active collaborative documents" not "revenue" |
| **Leading, not lagging** | Predicts future revenue/retention rather than reporting past revenue |
| **Reflects customer + business success jointly** | Should correlate with both retention and monetization |
| **Actionable** | Teams can influence it through concrete input metrics |
| **Understandable** | Non-technical stakeholders can grasp it instantly |
| **Measurable in near-real-time** | Not a quarterly-lagging survey number |
| **Resistant to gaming** | Hard to inflate without delivering real value |
| **Single metric, not a bundle** | Composite indices are for OEC, not NSM (though some orgs use a "constellation") |

### 1.3 Structure: the North Star + Metric Tree
- **North Star Metric** — the headline number.
- **Input Metrics / Driver Metrics** — 3–6 metrics that mechanically roll up into the NSM (e.g., breadth × depth × frequency × efficiency).
- **Sub-input metrics** — team-level metrics feeding each input metric; this is where individual squads find their KPIs.

Common decomposition patterns:
- **Breadth × Depth**: (# of users engaging) × (intensity of engagement per user)
- **Acquisition × Activation × Retention × Referral × Revenue (AARRR / Pirate Metrics)** used as an input framework feeding the NSM
- **Frequency × Adoption**: how often, by how many

### 1.4 Canonical examples by company/industry
| Company/type | North Star Metric |
|---|---|
| Airbnb | Nights booked |
| Spotify | Time spent listening |
| Facebook (historic) | Daily active users (DAU), later refined to "meaningful social interactions" |
| Slack | Messages sent between teammates (later: weekly active teams sending 2,000+ messages) |
| WhatsApp | Messages sent |
| LinkedIn | (Historic) daily active users engaging in professional actions |
| Netflix | Hours streamed |
| Amazon | Purchases (or, more precisely, "customer lifetime value" style proxies) |
| Uber/Lyft | Completed rides |
| Dropbox | Files saved / shared via Dropbox |
| Duolingo | Daily learners / streaks maintained |
| SaaS B2B (generic) | Weekly active accounts with ≥1 core workflow completed |
| Marketplace (generic) | Successful transactions (both sides matched) |
| Media/content | Engaged time / completion rate |

### 1.5 Nuances and failure modes
- **Vanity-metric trap**: signups, downloads, pageviews look good but don't reflect delivered value — avoid as NSM.
- **Single-sided marketplace bias**: picking a metric that only reflects supply or only demand (e.g., "listings created" ignores whether they get booked).
- **Metric myopia / Goodhart's Law**: once a metric becomes a target, people optimize the metric, not the value ("ships sent" vs. "messages read and replied to").
- **Lagging disguised as leading**: revenue and NPS are often lagging — poor NSM choices.
- **One-size-fits-all fallacy**: multi-product companies often need a **North Star per product line** plus a company-level "constellation" or portfolio view, since one number can't represent unrelated product value.
- **Novelty/seasonality distortion**: NSM can spike from external effects unrelated to product changes (e.g., COVID spikes in streaming) — needs cohort-normalized tracking.
- **Should evolve over life stage**: pre-PMF (activation-focused), growth stage (engagement-focused), maturity (monetization/retention-focused) — NSM sometimes gets redefined at each stage (Slack's NSM changed over time).
- **Doesn't replace P&L**: NSM is a *leading proxy*, not a substitute for financial reporting.

### 1.6 Choosing between competing NSM candidates — a scoring method
When a team has shortlisted 2–4 candidate metrics, score each 1–5 against every criterion in 1.2, then sanity-check with three questions:
1. If this metric doubled next quarter, would leadership genuinely believe the company is twice as healthy?
2. Can this metric be moved by a bad actor internally (growth hacking, dark patterns) without real value delivered?
3. Does at least one full team's roadmap map cleanly onto an input metric for this candidate?
If the answer to (3) is "no" for every team, the NSM is too abstract to be actionable.

### 1.7 North Star vs. mission/vision statements
NSM is often confused with mission statements. A mission statement ("connect the world") is not measurable; the NSM operationalizes a *sliver* of that mission into a trackable number. Expect the NSM to feel narrower and slightly reductive compared to the mission — that's a feature, not a bug, because it forces trade-off clarity.

### 1.8 Multi-sided and platform businesses
Platforms with distinct user types (buyers/sellers, riders/drivers, creators/viewers) often need a **paired NSM** or a single metric that only counts when *both* sides participate (e.g., "completed transactions" implicitly requires both a buyer and seller), which avoids the single-sided bias noted in 1.5. Some platforms maintain a secondary "health metric" per side (e.g., driver utilization, creator payout velocity) that acts as a semi-guardrail on the primary NSM.

### 1.9 Cadence and reporting nuances
- **Rolling windows** (7-day or 28-day rolling active metrics) are usually preferred over strict calendar week/month cuts to smooth weekday/weekend and month-boundary noise.
- **Cohort-normalized NSM**: reporting NSM per cohort-month-of-signup avoids conflating growth-driven increases with genuine per-user engagement increases.
- **NSM should have an owner** (often a VP of Product or Growth) accountable for the trend, distinct from feature-team leads who own input metrics.

---

## 2. GUARDRAIL METRICS

### 2.1 Definition
Metrics that must **not regress** (beyond a tolerance) as a side effect of a change, even if the primary/OEC metric improves. They exist to catch "you won the battle, lost the war" outcomes — e.g., engagement went up because you added dark patterns, but trust and long-term retention silently eroded.

### 2.2 Taxonomy of guardrail types
| Category | Purpose | Example metrics |
|---|---|---|
| **Trust / quality guardrails** | Protect user experience & brand trust | Complaint rate, unsubscribe rate, crash rate, spam-report rate, "regret" surveys, uninstall rate |
| **Business / revenue guardrails** | Protect monetization while optimizing engagement | Revenue per user, ARPU, refund rate, churn rate, contract renewal rate |
| **Organizational / company-wide guardrails** | Metrics every team must respect regardless of what they're testing | Overall DAU/MAU, page-load latency SLA, support-ticket volume, NPS |
| **Ethical / legal / compliance guardrails** | Non-negotiable constraints | Accessibility compliance, privacy/data-handling metrics, regulatory thresholds, fairness/bias metrics across user segments |
| **Technical / performance guardrails** | Protect system health | Latency (p50/p95/p99), error rate, crash-free sessions, server cost per request, battery/data usage on mobile |
| **Sample-ratio / data-quality guardrails** | Protect experiment validity itself (a "meta-guardrail") | Sample Ratio Mismatch (SRM) checks, instrumentation/logging completeness, bucketing balance |
| **Cannibalization guardrails** | Ensure a win in one surface isn't just moving activity from another | Cross-feature substitution metrics, organic vs. paid channel mix |
| **Ecosystem / two-sided guardrails** | In marketplaces, protect the "other side" | Seller/creator retention when optimizing for buyer/consumer metrics, ad-load tolerance |

### 2.3 Statistical nuance: how guardrails are tested differently
- **Primary/OEC metrics** are usually tested with a **superiority test** (is treatment significantly better?).
- **Guardrail metrics** are usually tested with a **non-inferiority test**: define an acceptable degradation margin (e.g., "latency may not worsen by more than 50ms") rather than requiring "no difference," because true zero-change is statistically unfalsifiable.
- **Directionality**: guardrails are typically one-sided tests (only care about the "bad" direction), whereas primary metrics are often two-sided or one-sided in the "good" direction.
- **Multiple-comparison correction**: because orgs track many guardrails simultaneously, false-positive "guardrail breach" alarms multiply — corrections (Bonferroni, Benjamini-Hochberg / FDR) or Bayesian shrinkage are commonly applied, or guardrails are monitored with wider confidence intervals / higher significance bar than the primary metric.
- **Power considerations**: guardrails often need larger samples to detect small regressions confidently — teams sometimes accept lower power on guardrails and instead monitor them continuously post-launch (sequential testing / always-valid p-values).
- **Auto-abort / kill-switch rules**: some orgs configure experiments to auto-halt if a guardrail breaches a hard threshold mid-flight (used heavily for latency, crash rate, revenue).

### 2.4 Nuances and failure modes
- **Guardrail sprawl**: tracking too many guardrails dilutes focus and increases false alarms — best practice is a short, curated "company guardrail list" (5–10 metrics) plus optional team-specific ones.
- **Static thresholds go stale**: guardrail tolerance bands need periodic recalibration as baseline traffic/behavior shifts.
- **Guardrails vs. counter-metrics**: related but distinct — a **counter-metric** is an *expected* trade-off you're willing to accept in exchange for the primary win (e.g., slight increase in cost-per-user is fine if LTV rises); a **guardrail** is something you are *not* willing to trade away.
- **Local vs. global guardrails**: a guardrail could be fine to violate locally (one team's feature) but not globally (company-wide threshold) — clarity on scope avoids false positives.
- **Guardrails must be pre-registered**: adding guardrails after seeing results is p-hacking; the list should be locked before the experiment starts.

### 2.5 Extended guardrail catalog by function
| Function | Guardrails commonly used |
|---|---|
| **Growth/Marketing** | Organic-to-paid traffic ratio, CAC (customer acquisition cost), channel mix concentration |
| **Search/Ranking** | Query abandonment rate, "reformulation" rate (user had to re-search), position-bias-adjusted CTR |
| **Ads/Monetization** | Ad load (ads per session), ad blindness/banner-blindness proxies, advertiser churn, fill rate |
| **Notifications/Messaging** | Opt-out rate, notification fatigue score, unsubscribe rate, spam-complaint rate |
| **Mobile** | App size growth, battery drain, cold-start time, crash-free session %, data usage per session |
| **Infrastructure/Cost** | Cost per 1,000 requests, compute cost per active user, storage growth rate |
| **Support/CS** | Ticket volume per 1,000 users, average handle time, first-contact resolution rate |
| **Legal/Privacy** | Consent opt-in rate, data-deletion request volume, PII exposure incidents |
| **Fairness/Responsible AI** | Outcome parity across demographic segments, false-positive/negative rate gaps across groups |
| **Content/Trust & Safety** | Policy-violating content view rate, report rate, moderator action volume |

### 2.6 Guardrail governance
- **Tiering guardrails**: "hard" guardrails (auto-block ship, e.g., legal/safety) vs. "soft" guardrails (require sign-off/justification to override, e.g., a small latency regression with a documented remediation plan).
- **Escalation path**: define who can approve an override when a soft guardrail is breached (typically a metrics/experimentation review board, not the shipping team itself, to avoid conflict of interest).
- **Guardrail dashboards vs. per-experiment checks**: company-wide guardrails are usually monitored continuously (weekly business review) *in addition to* being checked per-experiment — a metric can pass every individual experiment's guardrail check yet drift company-wide through accumulation of many small regressions ("death by a thousand cuts"), so an aggregate trend view is necessary alongside per-experiment gates.
- **Guardrail decay/review cadence**: revisit the guardrail list and thresholds quarterly or semi-annually; stale guardrails calibrated to an old baseline produce false alarms or fail to catch real regressions as usage patterns shift.

---

## 3. OEC — OVERALL EVALUATION CRITERION

### 3.1 Definition (Kohavi/Microsoft origin)
The metric, or small weighted set of metrics, that operationally defines "success" for a specific experiment or product decision — distinct from the North Star (which is directional/strategic) and distinct from a raw dashboard of everything you happen to track.

Key idea: the OEC should reflect **long-term** value even though experiments run short-term, because short-term metrics can be gamed easily (e.g., email spam increases short-term clicks but destroys long-term trust). Kohavi calls this "predicting the long-term from the short-term."

### 3.2 OEC construction approaches
| Approach | Description | Trade-off |
|---|---|---|
| **Single metric OEC** | One clear metric (e.g., "sessions per user per week") | Simple, interpretable, but may miss trade-offs |
| **Composite/weighted OEC** | Weighted combination of multiple signals into one score (e.g., Bing's use of weighted query share + revenue + satisfaction proxies) | Captures trade-offs in one number, but weighting is subjective and opaque |
| **Multi-metric decision matrix** | No single number; a *scorecard* of primary + secondary + guardrails reviewed together with a decision rule | Most common in practice; avoids false precision of composite scores |
| **Proxy/short-term surrogate for long-term OEC** | Because true long-term value (e.g., lifetime retention) can't be measured in a 2-week test, a short-term proxy validated to correlate with long-term outcomes is used | Requires periodic revalidation that the proxy still tracks the real long-term metric |

### 3.3 The full metric hierarchy used in an experiment "scorecard"

| Tier | Purpose | Decision weight | Example |
|---|---|---|---|
| **Primary metric(s)** | The metric(s) the ship/no-ship call is made on — usually 1, at most 2–3 | Must move in the right direction with statistical significance | Conversion rate, task-completion rate |
| **Secondary metrics** | Explain *why* the primary moved; diagnostic, not decisive alone | Informative, not decisive | Time-on-task, funnel step drop-off, click-through rate |
| **Guardrail metrics** | Must not regress beyond tolerance (see Section 2) | Veto power — can override a primary win | Latency, crash rate, revenue, complaint rate |
| **Quality / data-integrity metrics** | Confirm the experiment itself is trustworthy | Gate to even reading the results | Sample Ratio Mismatch, instrumentation coverage, bucket balance |
| **Exploratory / supporting metrics** | Hypothesis-generation for future experiments, not used for this decision | Informational only | Segment cuts, novel behavior patterns, heatmap/qual signals |

### 3.4 Primary metric selection nuances
- **Should be pre-registered** before the experiment launches (avoid HARKing — Hypothesizing After Results are Known).
- **Should be sensitive enough** to detect a meaningful effect within the available sample size/time (a metric that needs 10x current traffic to reach significance is a bad primary choice).
- **Should minimize variance** where possible — e.g., ratio metrics or capped/Winsorized metrics reduce noise from outliers (particularly relevant for revenue-based primaries).
- **One primary metric per decision is the safest default** — multiple co-primary metrics require a pre-defined resolution rule (e.g., "both must be positive," "at least one significant and none negative").
- **Directionality must be specified up front**: is this a one-tailed or two-tailed test?

### 3.5 Secondary metric nuances
- Used to build a causal story ("why did the primary move?"), catch unexpected mechanisms, and inform iteration.
- Should NOT be cherry-picked post-hoc to justify a ship decision when the primary metric failed ("secondary metric fishing").
- Often organized as a **funnel** (each step from exposure → primary outcome) so a regression can be localized to a specific step.

### 3.6 Composite OEC weighting nuances
- Weights should reflect genuine business value trade-offs, ideally validated against outcomes (e.g., "does a +1 point OEC composite score historically predict +X% revenue 6 months later?").
- Weighting schemes need periodic revalidation — business priorities shift (e.g., growth-stage weighting vs. monetization-stage weighting).
- Risk: composite scores can mask which underlying metric actually moved — always publish the decomposition alongside the composite.

### 3.7 Common OEC/primary-vs-secondary failure modes
- **Optimizing a short-term proxy that diverges from long-term value** (classic case: optimizing clicks leads to clickbait; optimizing session length can encourage addictive, low-value engagement).
- **HiPPO override**: ignoring guardrail vetoes because "the primary metric looks great" (Highest Paid Person's Opinion problem).
- **Simpson's paradox**: an OEC can improve in aggregate while regressing in every major segment (or vice versa) — always check segment-level cuts before trusting an aggregate primary metric.
- **Novelty effect**: primary metric spikes initially then decays — long-running or holdout experiments are used to check persistence.
- **Metric misalignment across teams**: Team A's primary metric is Team B's guardrail — requires a shared, org-wide metric dictionary/registry to avoid contradictory optimization.

---

## 4. Putting it together: a worked example (B2B SaaS collaboration tool)

| Layer | Example |
|---|---|
| **North Star** | Weekly Active Teams completing ≥1 shared workflow |
| **Input/driver metrics** | New team activation rate, invite-to-teammate conversion, workflow completion frequency, cross-team document shares |
| **Experiment**: "New onboarding checklist" | |
| — OEC / Primary metric | % of new teams completing 3+ workflows in week 1 |
| — Secondary metrics | Checklist completion rate, time-to-first-workflow, step-level drop-off |
| — Guardrails | Support ticket rate (must not rise >5%), app latency (must not rise >100ms), 30-day churn (non-inferiority margin: -1pt) |
| — Data-quality checks | SRM check on bucketing, event logging completeness |
| — Counter-metric (accepted trade-off) | Slightly longer time-to-first-value is acceptable if week-4 retention improves |

---

## 5. Statistical deep dive

### 5.1 Non-inferiority testing (the guardrail workhorse)
Standard superiority test: H0: Δ = 0 vs. H1: Δ ≠ 0.
Non-inferiority test: H0: Δ ≤ −m vs. H1: Δ > −m, where **m** is the pre-agreed non-inferiority margin (the maximum acceptable degradation). The guardrail "passes" if the confidence interval's lower bound stays above −m, even if the point estimate is slightly negative. Choosing **m** is itself a product/business judgment call, not a purely statistical one — too tight and every experiment gets blocked on noise; too loose and real regressions slip through.

### 5.2 Sample size and minimum detectable effect (MDE)
- Primary metrics: size the experiment for the smallest effect that's *decision-relevant* (not the smallest effect statistically detectable — those aren't the same thing).
- Guardrails: since you're trying to rule out harm, guardrail MDE is often set smaller (more sensitive) than the primary MDE, requiring either larger samples or longer runtimes, or accepting a wider non-inferiority margin as a trade-off.
- Underpowered guardrails are a common silent failure — teams report "no guardrail regression" when the test simply couldn't have detected one.

### 5.3 Sequential and always-valid testing
Because guardrails are often monitored continuously (peeking daily), naive fixed-horizon p-values inflate false-positive rates. Sequential testing methods (e.g., mSPRT, always-valid confidence sequences) let teams monitor safely without inflating Type I error — increasingly standard on experimentation platforms for guardrail dashboards.

### 5.4 Variance reduction techniques for primary/OEC metrics
- **CUPED (Controlled-experiment Using Pre-Experiment Data)**: uses pre-period behavior as a covariate to reduce variance and shrink confidence intervals without more traffic.
- **Stratification**: bucketing by known high-variance segments (e.g., country, device) before randomization.
- **Winsorization/capping**: bounding extreme values (common for revenue-per-user metrics, which are heavy-tailed) so a few whales don't dominate the estimate.
- **Ratio metrics vs. average of ratios**: define metrics as (sum of numerator)/(sum of denominator) across users rather than averaging per-user ratios, to avoid divide-by-zero and small-denominator instability.

### 5.5 Bayesian vs. frequentist framing
Some orgs run OEC decisions with Bayesian methods (posterior probability that treatment beats control by more than X%, expected loss framing) instead of p-values/confidence intervals — this makes "probability of guardrail harm" more directly interpretable to non-statisticians, at the cost of requiring agreed-upon priors.

---

## 6. Governance & process

### 6.1 Metric Definition Document (MDD) — recommended fields
Every primary, guardrail, and North Star metric should have a single-source-of-truth definition doc containing: metric name, precise SQL/computation logic, owner, numerator/denominator definition, inclusion/exclusion filters (bots, internal traffic, test accounts), time window/grain, known caveats, historical baseline + seasonality notes, and last-reviewed date. Without this, "conversion rate" silently means five different things across five teams.

### 6.2 Metric registry / catalog
A centralized, searchable registry (not just docs scattered in wikis) prevents duplicate or contradictory metric definitions and lets an experimenter check "is this already someone else's guardrail?" before finalizing an OEC.

### 6.3 Experiment review board / metrics council
Larger orgs use a lightweight review step before high-impact launches: confirms primary metric was pre-registered, guardrails weren't quietly dropped, and segment cuts were checked. This is a process guardrail against the *organizational* failure modes (HiPPO override, p-hacking) as much as a statistical one.

### 6.4 RACI sketch for the metric hierarchy
| Metric layer | Typically Responsible | Typically Accountable |
|---|---|---|
| North Star | Growth/Product leadership | CPO/CEO |
| Input/driver metrics | Feature-area product leads | VP Product |
| OEC / Primary metric per experiment | Experiment owner (PM/eng) | Team lead |
| Guardrails | Data science / experimentation platform team | Metrics council |
| Data-quality checks | Experimentation platform / data eng | Platform lead |

---

## 7. Anti-pattern catalog (illustrative, generalized cases)

| Anti-pattern | What happens | Why it's dangerous |
|---|---|---|
| **Metric whack-a-mole** | Team ships a change, primary metric flat, so they retroactively promote a secondary metric to "primary" post-hoc | Classic HARKing; inflates false-positive ship rate org-wide |
| **Guardrail shopping** | Team quietly narrows the guardrail list before a risky launch | Removes the very check meant to catch the risk |
| **Composite score gaming** | Team optimizes the weighted sub-component with the most "give" rather than genuine value | Composite OECs are especially vulnerable since improving any weighted input raises the score |
| **Engagement/dark-pattern trap** | Primary metric (session count) rises because of increased friction/notifications, guardrail (opt-out rate) creeps up slowly and is ignored because each individual experiment's guardrail move is "not significant" | "Death by a thousand cuts" — needs aggregate trend monitoring, not just per-experiment gates |
| **Local optimum trap** | Every team's OEC improves, North Star stagnates or declines | Poor input-metric-to-North-Star mapping, or metrics that trade off against each other across teams |
| **Segment-blind shipping** | Aggregate primary metric improves; one large segment (e.g., a country, an OS version) regresses sharply | Simpson's Paradox — always slice primary + guardrails by major segments before shipping |
| **Novelty-effect false win** | Metric spikes in week 1, launch is called a win, metric decays to baseline (or below) by week 4 | Needs longer holdouts or persistence checks before declaring long-term success |

---

## 8. Quick-reference glossary

| Term | One-line definition |
|---|---|
| North Star Metric | Single long-term value metric for the whole product/company |
| Input/Driver Metric | Metric that mechanically feeds the North Star |
| OEC | The metric(s) an experiment/decision is actually judged on |
| Primary Metric | The metric the ship/no-ship call hinges on |
| Secondary Metric | Diagnostic metric explaining the primary's movement |
| Guardrail Metric | Metric that must not regress; has veto power |
| Counter-Metric | An accepted, expected trade-off metric |
| Composite/Weighted OEC | Multiple metrics combined into one score |
| Non-inferiority test | Statistical test used for guardrails (bounded degradation, not zero-change) |
| Sample Ratio Mismatch (SRM) | A data-quality guardrail catching broken randomization |
| Novelty effect | Temporary metric spike/dip that fades over time |
| Simpson's Paradox | Aggregate trend reversing when segmented |
| HARKing | Choosing/adjusting metrics after seeing results (a bias to avoid) |
| Non-inferiority margin (m) | Max acceptable degradation allowed before a guardrail is considered breached |
| CUPED | Variance-reduction technique using pre-experiment covariate data |
| MDE (Minimum Detectable Effect) | Smallest effect size an experiment is powered to detect |
| Sequential/always-valid testing | Statistical methods allowing safe continuous monitoring without inflated false positives |
| Metric Definition Document (MDD) | Single-source-of-truth spec for how a metric is computed |
| Metric registry | Centralized catalog of all official metric definitions |
| Death by a thousand cuts | Many individually-insignificant regressions accumulating into a real aggregate harm |
| Winsorization | Capping extreme outlier values before computing a metric, to reduce variance |
| Holdout group | A long-running control group kept out of a launched feature to detect long-term/novelty-decay effects |
| Cohort-normalized metric | Metric reported by signup cohort to separate growth effects from per-user behavior change |

---

*Frameworks referenced: Sean Ellis / Amplitude's North Star Framework; Ron Kohavi et al., "Trustworthy Online Controlled Experiments" (OEC, guardrail, non-inferiority testing concepts); standard A/B testing practice at large-scale experimentation platforms (Microsoft, Google, Booking.com, LinkedIn, Netflix, Airbnb).*
