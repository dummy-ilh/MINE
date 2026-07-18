# Chapter 8: One-sided vs Two-sided Testing

## 1. Intuition

This chapter closes out Module 1 by answering a deceptively simple question that trips up even experienced practitioners: **when you set up your hypothesis test, do you test for "any difference" (two-sided) or "an improvement specifically" (one-sided)?**

The intuition: a two-sided test asks "did anything change, in either direction?" A one-sided test asks "did it get *better* (or specifically *worse*)?" This sounds like a minor technicality, but the choice directly affects your statistical power and — more importantly for interviews — is a common vector for accidentally (or deliberately) gaming your result.

## 2. The Formal Distinction

**Two-sided test:**
$$H_0: \Delta = 0 \quad \text{vs} \quad H_1: \Delta \neq 0$$

Rejection region is split across both tails: reject if $z < -z_{\alpha/2}$ OR $z > z_{\alpha/2}$. For $\alpha=0.05$, that's $z_{critical}=\pm1.96$, splitting the 5% error rate as 2.5% in each tail.

**One-sided test:**
$$H_0: \Delta \leq 0 \quad \text{vs} \quad H_1: \Delta > 0$$

Rejection region is entirely in one tail: reject if $z > z_\alpha$. For $\alpha=0.05$, that's $z_{critical}=1.645$ — a *less extreme* threshold than 1.96, because you're putting the entire 5% error budget into one tail instead of splitting it.

**The direct consequence**: for the exact same data, a one-sided test is "easier" to achieve significance on (lower critical value) — which is precisely why the choice must be locked in *before* seeing the data, and why interviewers probe this so heavily.

## 3. When Is a One-Sided Test Actually Justified?

The honest criterion: a one-sided test is justified only when a result in the "wrong" direction is **actually inconsequential to your decision** — i.e., you would take the same action (don't ship) whether the effect is exactly zero or meaningfully negative, so there's no reason to "spend" error budget detecting how negative it is.

**Example where one-sided is defensible**: you're testing a pure engineering optimization (e.g., a caching improvement expected only to affect latency, with no plausible mechanism to make things worse) — though even here, many practitioners argue you should still use two-sided, because "no plausible mechanism for harm" is often wrong in practice (unexpected regressions happen constantly).

**Example where one-sided is NOT defensible**: testing a new ranking algorithm or UI redesign — these can very plausibly make things worse (confuse users, hurt conversion), so ruling out that possibility by using a one-sided test in the "improvement" direction would hide a genuine risk you care about.

**Industry norm at most large tech companies (including Google)**: default to **two-sided tests** for nearly everything, specifically because of the risk described above — treating a possible harm as "impossible" by construction is a dangerous simplification, and it's very rare that leadership would actually be indifferent to a strongly negative result.

## 4. Why the Timing of the Decision Matters So Much

The single biggest interview trap in this chapter: **choosing one-sided vs. two-sided AFTER looking at the direction of your result is a form of p-hacking.**

Here's the mechanism, concretely: suppose your two-sided test gives $p=0.08$ (not significant at $\alpha=0.05$), but the effect happens to be positive. If you then say "well, we only cared about positive effects anyway, let's redo this as one-sided," your one-sided p-value would be $p=0.04$ — suddenly "significant." You didn't learn anything new about the world; you exploited the fact that halving the two-sided p-value (roughly, for a symmetric distribution) is exactly what a one-sided test does when the effect happens to point the "right" way.

This is why the test direction must be specified in your **pre-registration / experiment design doc**, before the experiment launches, not chosen reactively based on which framing makes the result look better.

## 5. Worked Example

Continuing the checkout redesign example: $\hat{\Delta}=0.008$, $SE\approx0.00432$ (from Chapter 2/4).

**Two-sided test** ($H_1: \Delta \neq 0$):
$$z = 0.008/0.00432 \approx 1.85, \quad z_{critical}=1.96 \implies \text{fail to reject}, \quad p \approx 2\times(1-\Phi(1.85)) \approx 0.064$$

**One-sided test** ($H_1: \Delta > 0$), same data:
$$z_{critical}=1.645 \implies 1.85 > 1.645 \implies \textbf{reject } H_0!, \quad p \approx 1-\Phi(1.85) \approx 0.032$$

**Notice**: identical data, identical effect, but the decision flips (fail to reject → reject) purely based on which test framing you chose. This is exactly why the choice must be locked in during experiment design — if you had the flexibility to pick whichever framing gave you the answer you wanted *after* seeing this result, you'd effectively be operating at a higher true false-positive rate than your stated $\alpha=0.05$.

## 6. Production Considerations

- **Default to two-sided unless you have a strong, pre-registered, written justification for one-sided** — write the test direction into the experiment design doc before launch, and treat it as immutable once the experiment starts.
- **One-sided tests are sometimes appropriate for guardrail metrics** (Chapter 7 of Module 2) — e.g., "we only care if latency got *worse*, not better" — because the decision rule genuinely is asymmetric: you'd act the same way whether latency improved or stayed flat, but you'd act differently if it regressed. This is a legitimate, principled use case, distinct from choosing one-sided just to make a marginal result "significant."
- **Some experimentation platforms enforce two-sided by default** and require explicit sign-off to switch to one-sided, precisely to prevent the p-hacking pattern above from being exploitable by teams under pressure to show a win.

## 7. Interview Traps

- **Trap #1 (the big one)**: Not immediately recognizing that choosing test direction based on the observed data is p-hacking. This is one of the most direct "do you understand rigor or just formulas" tests in an A/B interview.
- **Trap #2**: Claiming one-sided tests are "always more powerful, so why not use them" without acknowledging the cost: you're implicitly declaring you don't care about detecting harm in the other direction, which is a real, substantive decision, not a free power boost.
- **Trap #3**: Not being able to produce the numeric flip (as in the worked example above) live — being asked "convert this two-sided result to one-sided, does the conclusion change?" is a common on-the-spot calculation.
- **Trap #4**: Confusing "one-sided test" with "one-tailed critical region only in the direction you're hoping for" as if the *direction itself* were adjustable after the fact — the direction must be part of the pre-specified $H_1$, not a post-hoc framing choice.

## 8. L5-Differentiating Talking Points

- Proactively stating "this needs to be decided and documented before the experiment launches, not after we see results" without being prompted shows you understand *why* the rule exists, not just that it exists.
- Being able to articulate the legitimate use case (guardrail metrics with genuinely asymmetric decision rules) vs. the illegitimate use case (gaming a marginal result) in the same breath shows nuanced judgment rather than a blanket rule.
- Volunteering that some experimentation platforms structurally prevent this p-hacking vector (locking test direction pre-launch) shows awareness of how good experimentation infrastructure encodes statistical rigor into the tooling itself — a very L5 systems-thinking angle.

## 9. Comprehension Check

1. Why does a one-sided test have a less extreme critical value ($z=1.645$) than a two-sided test ($z=1.96$) at the same $\alpha=0.05$?
2. Explain precisely why choosing test direction after observing the sign of your effect inflates your true false-positive rate above the stated $\alpha$.
3. Give one legitimate, pre-registered use case for a one-sided test, and explain why it's different from choosing one-sided just to flip a marginal result to "significant."
4. In the worked example, the exact same data was "not significant" two-sided ($p=0.064$) but "significant" one-sided ($p=0.032$). What should you conclude about the treatment from this, if the test direction wasn't pre-registered?
5. Would you recommend one-sided or two-sided testing for a core product ranking change, and why?

---
*End of Module 1: Statistical Foundations (Chapters 1-8).*
*Next: Chapter 9 — OEC Design (start of Module 2: Metrics & Measurement)*
