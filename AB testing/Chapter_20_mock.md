# Chapter 21: Full Mock Case Interviews

*(Closes Module 4 and the full 21-chapter curriculum)*

## 1. Intuition

This is the capstone chapter. Everything from Chapters 1-20 gets synthesized here into the actual format you'll face in the interview: an open-ended prompt, worked through out loud, in real time, with the interviewer probing your reasoning at each step. Five full mock cases follow, each in dialogue format, covering a different flavor of A/B testing question (design-from-scratch, diagnostic/debugging, conflicting-metrics judgment, ranking-specific, and a "explain a concept simply" case). Each ends with an explicit L5-vs-lower-level breakdown of what separates a strong answer from an adequate one.

---

## 2. Mock Case 1: Design an Experiment From Scratch

**Interviewer**: "We want to test adding a 'save for later' button to our e-commerce product pages. Design an experiment for this."

**Strong candidate response**:
"Let me start by clarifying the goal — is this meant to increase eventual purchases, or is it more about engagement/session depth? [Interviewer: 'assume the goal is ultimately more purchases, but we're open to it helping via a longer consideration window'] Okay, that's useful — it suggests the effect might not show up immediately in same-session conversion, so I'll want to think about time horizon.

**Randomization unit**: I'd randomize by user ID, not session, since a user's shopping behavior spans multiple sessions and I want consistent treatment exposure across their whole consideration journey. I'd check whether this product has any social/marketplace dynamics that could cause interference — probably not for a personal 'save for later' feature, so standard user-level randomization should be safe from SUTVA violations here.

**OEC**: primary metric would be purchase conversion rate over a reasonably extended window — say 14-21 days post-first-exposure, not just same-session, since the whole point of 'save for later' is enabling delayed purchases. I'd also track 'save' actions themselves as a diagnostic/adoption metric, but that's not the OEC — it's possible people save things and never buy, which wouldn't be a real win.

**Guardrails**: page load latency (adding a button shouldn't meaningfully slow the page), and same-session conversion rate — I want to make sure 'save for later' isn't just delaying purchases that would have happened immediately anyway, which would be a wash or worse if delay causes drop-off.

**Sample size**: I'd want a baseline conversion rate and a realistic MDE from the PM — let's say baseline is 3% and we care about detecting a 0.3 percentage point lift [walks through the sample size formula from Chapter 5]. Given e-commerce traffic volumes, I'd expect this needs a few weeks of data given the 21-day observation window requirement stacks on top of the enrollment period.

**Novelty risk**: I'd want to check for novelty effects specifically here — a new button might get curiosity clicks initially that don't reflect steady-state usage, so I'd plan to look at the day-by-day/week-by-week save-and-purchase pattern, not just a pooled average across the whole window."

**L5 vs. lower-level breakdown**: A lower-level answer stops at "randomize users, measure conversion, compute sample size." The L5 signal here is (1) clarifying ambiguous goal before designing, (2) recognizing the metric needs an extended observation window because of the feature's actual mechanism (delayed purchase), (3) proactively adding a same-session guardrail to catch potential cannibalization, and (4) flagging novelty effect risk unprompted, before being asked "how do you know this isn't just curiosity?"

---

## 3. Mock Case 2: Diagnose a Broken Experiment

**Interviewer**: "An experiment ran for two weeks. The treatment shows a massive +15% lift on the primary metric, way bigger than the team expected going in. What do you do?"

**Strong candidate response**:
"A result that's much bigger than expected is itself a signal worth investigating before celebrating — I wouldn't just accept it at face value. First thing I'd check is SRM: is the actual user split close to the intended ratio? [Interviewer: 'good instinct, let's say the split checks out fine, 50.1/49.9'] Okay, that rules out one common cause of spuriously large or misleading effects.

Next I'd check whether this could be a peeking/multiple-testing artifact — was this the pre-specified primary metric checked at a pre-planned endpoint, or did someone check the dashboard repeatedly and we're looking at a favorable moment? [Interviewer: 'it was checked exactly once, at the pre-planned 2-week mark'] Good, that also rules out a common inflation source.

Given SRM and peeking are ruled out, I'd next check for a segment-level Simpson's Paradox-type issue — does the +15% pooled effect hold up consistently across major segments (platform, geography, new vs. returning users), or is it being driven by an unusual shift in segment composition between arms? I'd also check whether this metric is a ratio metric (Chapter 11) and whether the delta-method variance was computed correctly, since a mis-specified SE could make a moderate effect look artificially more significant/extreme than it truly is.

I'd also just sanity check the effect size against domain knowledge — a 15% lift on a mature product's core conversion metric would be an unusually large effect for most incremental feature changes; that magnitude alone is a flag that something might be off in either the treatment mechanism or the measurement pipeline, even independent of statistical validity checks. I'd want to look at whether the treatment accidentally introduced some kind of bug that inflates the metric numerically (e.g., double-counting an event) rather than genuinely changing user behavior."

**L5 vs. lower-level breakdown**: A lower-level answer might just say "check for bugs" vaguely. The L5 signal is running through the Module 3 diagnostic checklist systematically and in a sensible order (SRM → peeking → segment consistency → metric computation correctness → domain-knowledge sanity check), rather than jumping straight to "there's probably a bug" without a structured process to actually find it.

---

## 4. Mock Case 3: Conflicting Metrics, Make the Call

**Interviewer**: "We tested a more aggressive email frequency (2x current frequency). Open rates per email are down 8% (statistically significant), but total emails opened per user per week is up 12% (statistically significant), and unsubscribe rate is up 0.4 percentage points (statistically significant, but small in absolute terms). What's your recommendation?"

**Strong candidate response**:
"There's a real tension here, so let me work through it rather than pick a side reflexively. First — what was the pre-specified OEC for this test? [Interviewer: 'let's say it was total weekly engagement, i.e., total opens per user'] Okay, so by the pre-committed primary metric, this is actually a clear win — the drop in per-email open rate is expected and somewhat mechanical (sending more emails to the same finite attention naturally dilutes per-email engagement), and the actual OEC moved in the right direction.

But I wouldn't stop there, because the unsubscribe increase is a guardrail-type signal, and 0.4 percentage points, while small in absolute terms, could be large in relative terms depending on the baseline unsubscribe rate — I'd want to know that baseline before dismissing it as 'small.' [Interviewer: 'baseline unsubscribe rate is about 1.2% per week, so this is a ~33% relative increase'] That changes my read substantially — a 33% relative increase in unsubscribes is a meaningful long-term cost, because unsubscribed users have zero future engagement value, not just reduced engagement — it's a much more severe and permanent loss than a temporary open-rate dip.

Given that, I'd want to model the long-term tradeoff: are we gaining more total engagement now at the cost of a meaningfully larger permanently-lost user base later? This feels like exactly the kind of proxy-metric problem from Chapter 13 — 'opens per week' is a short-term proxy, but the real long-term OEC should probably be something like a modeled long-term engagement value that accounts for the elevated unsubscribe rate reducing the future opportunity for engagement entirely.

My recommendation: I would NOT fully ship the 2x frequency change based on this data alone. I'd propose either (a) a middle frequency (e.g., 1.5x) to see if we can capture some of the engagement gain with a smaller unsubscribe cost, or (b) segment the frequency increase — apply it only to users showing high baseline engagement, who are probably more tolerant of higher frequency and less likely to unsubscribe, while keeping current frequency for lower-engagement users who are probably closer to their unsubscribe threshold already."

**L5 vs. lower-level breakdown**: A lower-level answer either says "ship it, the OEC won" or "don't ship it, unsubscribes went up" — a binary read of a single number. The L5 signal is (1) asking for the baseline to correctly interpret whether "0.4pp" is actually large or small, (2) recognizing this as a short-term-proxy-vs-long-term-value tension (explicitly named, tied back to Chapter 13), and (3) proposing a genuinely creative middle path (segmented rollout by engagement tier) rather than a binary ship/kill call.

---

## 5. Mock Case 4: Ranking-Specific Case

**Interviewer**: "We have a new search ranking algorithm and want to know if it's better than our current one before doing a full-scale business-metric rollout test. How would you evaluate it quickly?"

**Strong candidate response**:
"Given this is specifically about ranking quality, and the goal is a fast pre-filter before a slower full rollout test, I'd reach for interleaving rather than a standard A/B test as the first step. Interleaving lets us show the same user a blended combination of both algorithms' results and see which algorithm's results they preferentially engage with — because it's a within-subject comparison, it needs far fewer users and much less time than a between-subjects A/B test would to get a statistically confident read on relative preference.

Specifically I'd use team-draft interleaving: construct a blended results list where the two algorithms take turns contributing their next-best unpicked result, randomizing which algorithm picks first each query to avoid position bias, then compare which algorithm's tagged results get more clicks in aggregate across many queries.

I'd frame this explicitly as a fast filter, though, not the final word — interleaving tells us which ranking users click on more, which is a reasonable proxy for relevance, but it doesn't capture whether the new algorithm actually improves our true business OEC (search satisfaction, session value, long-term retention). So my plan would be: run interleaving first as a quick, cheap, highly-sensitive screen — if the new algorithm clearly loses on interleaving, we can kill it fast without spending a full standard A/B test's worth of traffic and time. If it wins on interleaving, THEN we run the full standard A/B test against real business metrics, with the standard guardrails (latency, session quality) and appropriate power analysis, before considering a full launch."

**L5 vs. lower-level breakdown**: A lower-level answer defaults straight to "run an A/B test" without recognizing rankings have a specialized, more efficient evaluation technique available. The L5 signal is naming interleaving specifically and unprompted, correctly explaining the within-subject efficiency mechanism, and — critically — correctly scoping interleaving's role as a fast filter rather than a full replacement for OEC-based validation.

---

## 6. Mock Case 5: Explain a Concept Simply (Communication Test)

**Interviewer**: "Explain a p-value to me as if I were a product manager with no stats background, in under 30 seconds."

**Strong candidate response**:
"Think of it like this: the p-value tells you how surprising your result would be if the new feature actually did absolutely nothing. A small p-value — say, below 5% — means 'if this feature really had zero effect, we'd almost never see numbers like this by pure chance,' so we conclude something real is probably going on. It's not the probability that the feature works — it's a measure of how hard it would be to explain our data away as random luck alone."

**L5 vs. lower-level breakdown**: A lower-level answer either recites the formal definition verbatim (fails the "PM with no stats background" framing) or gives a wrong simplification like "it's the probability the result is due to chance" (a common but subtly incorrect phrasing — it can easily be misheard as $P(H_0|data)$ rather than $P(data|H_0)$). The L5 signal is finding a genuinely accurate, plain-language analogy under a real time constraint, without sacrificing correctness for simplicity — this specific skill (rigorous but accessible communication) is exactly what a senior DS needs when explaining experiment results to non-technical stakeholders in actual day-to-day work.

---

## 7. How to Use This Chapter for Final Prep

- **Do these out loud, not silently in your head.** Reading the strong responses above is not the same as generating them yourself under time pressure — practice saying your version aloud before checking it against the model response.
- **Notice the recurring skeleton across all 5 cases**: clarify the goal/ambiguity first, ground your answer in the pre-specified OEC/guardrails where relevant, run through the Module 3 validity checklist before trusting any surprising result, distinguish statistical from practical significance, and end with an explicit, committed recommendation rather than an open-ended list of considerations.
- **The single biggest failure mode across all case types** is technically correct fragments without synthesis — knowing all 20 prior chapters' individual facts but not weaving them into a coherent, prioritized narrative under interview time pressure. That synthesis is specifically what this chapter was built to model and what deliberate practice should target in your remaining prep time.

---
*This concludes the 21-chapter A/B Testing curriculum for Google L5 MLE/DS preparation.*
