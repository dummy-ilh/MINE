# Chapter 10: Guardrail Metrics

## 1. Intuition

Chapter 9 established that most real experiments use a single primary OEC gated by guardrails, rather than one giant composite metric. This chapter goes deep on that second half: **what exactly is a guardrail metric, how do you choose thresholds for it, and how does testing it differ from testing your primary metric?**

The intuition: a guardrail metric answers "did we break anything important while chasing our primary goal?" It's not something you're trying to *improve* — you're trying to confirm it **didn't get meaningfully worse**. This asymmetry (you don't care about "guardrail got much better," only "guardrail didn't get much worse") is exactly the legitimate one-sided testing use case flagged in Chapter 8.

## 2. Common Categories of Guardrail Metrics

- **Latency/performance**: page load time, time-to-interactive. A feature that improves conversion but doubles load time may be a net loss even if the primary metric looks great.
- **Quality/trust metrics**: crash rate, error rate, complaint/report rate, unsubscribe rate. These protect against "we optimized engagement by making the product objectively worse in some dimension."
- **Revenue/monetization** (when revenue isn't the primary metric): you don't want a UX experiment to accidentally tank ad revenue or vice versa.
- **Company-wide "trust" metrics**: at large companies, there are often standing guardrails that apply to almost *every* experiment regardless of team (e.g., overall query volume, overall active users, crash-free sessions) — these act as company-wide tripwires that catch unexpected systemic harm.

## 3. Guardrails Are Tested Differently Than the Primary Metric

This is the part that separates a surface-level answer from an L5 answer: **guardrail metrics typically use a different statistical posture than the primary OEC.**

- **Primary metric**: you're trying to detect an improvement — standard two-sided (or, if pre-registered, one-sided-for-improvement) test, powered to detect a specific MDE (Chapter 5).
- **Guardrail metric**: you're trying to rule out harm — often tested as **one-sided in the "harm" direction** ("did this get significantly *worse*"), and importantly, you often want a **much larger, pre-committed non-inferiority margin** rather than testing against exactly zero.

This introduces the concept of a **non-inferiority test**: instead of $H_0: \Delta = 0$, you test $H_0: \Delta \leq -\delta_{margin}$ vs $H_1: \Delta > -\delta_{margin}$, where $\delta_{margin}$ is the maximum tolerable regression you've pre-agreed is acceptable (e.g., "latency can regress by up to 50ms, that's within tolerance"). This is a subtly different test than testing against a null of exactly zero, and is the statistically correct way to formalize "didn't meaningfully break anything," rather than requiring "literally no measurable change," which is an unreasonably strict and often unachievable bar.

## 4. Worked Example

You're testing a new recommendation module that adds visual complexity to a page. Primary OEC: click-through rate on recommendations. Guardrail: page load latency (p50).

- Control p50 latency: 850ms
- Treatment p50 latency: 890ms (a 40ms / ~4.7% regression)
- Pre-agreed non-inferiority margin: latency regression must not exceed 60ms (agreed with the perf team before launch, based on known thresholds where users start reporting perceived slowness)

**Non-inferiority test setup**:
$$H_0: \Delta_{latency} \geq 60ms \text{ (i.e., truly unacceptable regression)} \quad H_1: \Delta_{latency} < 60ms \text{ (i.e., acceptable)}$$

Given the observed 40ms regression and a computed SE of, say, 8ms:

$$z = \frac{40-60}{8} = \frac{-20}{8} = -2.5$$

For a one-sided test at $\alpha=0.05$, $z_{critical} = -1.645$. Since $-2.5 < -1.645$, we reject $H_0$ (the "unacceptably bad" null) — meaning we have statistical evidence the true regression is **less than** the 60ms tolerance threshold. **Guardrail passes** — not because latency didn't regress at all (it did, by 40ms), but because we've statistically confirmed the regression is within the pre-agreed tolerable range.

**Key teaching point**: notice this is a fundamentally different question than "is there a statistically significant regression in latency?" (there almost certainly is, given enough sample size — nearly any change adds *some* latency). The non-inferiority framing is what lets you distinguish "technically measurable" from "practically consequential" — exactly the statistical-vs-practical-significance distinction from Chapter 2, now applied specifically to the guardrail context.

## 5. Production Considerations

- **Guardrail thresholds should be set by domain experts, not statisticians alone** — the "60ms tolerance" in the example above should come from actual UX research on perceptible latency, not an arbitrary statistical convention. This is a great point to make in an interview: guardrails require cross-functional input to set correctly, they're not a purely stats problem.
- **Standing/company-wide guardrails should be automated into the experimentation platform**, not manually checked per-experiment — at Google scale, thousands of experiments run concurrently, and a small number of universal tripwire metrics (crash rate, major error rate) should auto-flag any experiment that trips them, regardless of what team owns it.
- **Guardrail violations should generally block shipping even if the primary metric is a huge win** — this is a policy/cultural decision that needs org-level buy-in ahead of time; without it, teams under pressure to ship will rationalize past guardrail failures.
- **Beware guardrail metric proliferation**: adding too many guardrails increases the chance that *some* guardrail trips by pure chance (a multiple-testing problem — foreshadowing Chapter 14), potentially blocking genuinely good launches on statistical noise. Guardrail sets should be curated, not exhaustive.

## 6. Interview Traps

- **Trap #1**: Testing guardrails with the exact same statistical setup as the primary metric (two-sided, testing against exactly zero) instead of recognizing the non-inferiority framing is usually more appropriate.
- **Trap #2**: Treating "guardrail showed a statistically significant regression" as automatically disqualifying, without considering whether the regression is within a pre-agreed *practically* tolerable margin — conflating statistical and practical significance again.
- **Trap #3**: Not mentioning that guardrail thresholds need non-statistical, domain-expert input (UX research, infra team, etc.) to set meaningfully.
- **Trap #4**: Proposing an unbounded number of guardrail metrics without acknowledging the multiple-testing risk this creates (more guardrails = more chances for a false alarm to block a real win).

## 7. L5-Differentiating Talking Points

- Introducing the non-inferiority testing framework by name, and explaining why it's different from a standard two-sided test against zero, is a strong differentiator — most candidates only know the standard hypothesis test framing.
- Explicitly stating that guardrail thresholds are set jointly with domain experts (not purely a stats decision) shows organizational/cross-functional maturity.
- Flagging the multiple-testing risk of guardrail proliferation, and connecting it forward to Chapter 14, demonstrates you see guardrail design as part of the broader statistical rigor of the whole experimentation system, not an isolated checklist item.
- Being explicit that "guardrail regressed, but within tolerance" and "guardrail regressed beyond tolerance" require org-level policy on which one blocks shipping — showing you understand this is partly a governance problem, not purely a technical one.

## 8. Comprehension Check

1. Why is a guardrail metric typically tested with a non-inferiority framework rather than a standard two-sided test against zero?
2. In the worked example, the observed latency regression (40ms) was smaller than the tolerance margin (60ms), yet a naive two-sided test against zero would likely still show "statistically significant regression." Reconcile these two facts.
3. Who should be involved in setting a guardrail's non-inferiority margin, and why shouldn't this be a purely statistical decision?
4. What risk does adding an unbounded number of guardrail metrics introduce, and how does it connect to a concept from a later chapter?
5. Your primary OEC shows a strong, statistically significant win, but one guardrail metric shows a small regression that's technically outside its pre-agreed tolerance. What do you recommend, and why?

---
*Next: Chapter 11 — Ratio Metrics & the Delta Method*
