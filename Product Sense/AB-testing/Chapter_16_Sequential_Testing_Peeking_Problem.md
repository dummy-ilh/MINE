# Chapter 16 (continued): Sequential Testing & the Peeking Problem — Always-Valid P-Values

> This picks up the thread the multiple-testing chapter opened with: *"Chapter 16 covered inflation of false-positive rate from looking at the same metric multiple times over time."* That's a different multiplicity problem from testing many metrics at once — this note explains why it happens, why it's worse than people expect, and how to actually fix it.

---

## 1. Why this happens (the "why now")

A standard p-value is only valid **at one pre-committed sample size**. The moment you check it *before* that sample size is reached — "let's just peek at day 3," "let's check again tomorrow" — you've silently run a new test. Check every day for two weeks and you haven't run one test, you've run fourteen, all on the *same* metric, all correlated with each other.

This is why it's tempting and dangerous in practice: nothing about the workflow looks like "multiple testing." No one deliberately tested 14 hypotheses — they just watched a dashboard. The multiplicity is hidden in the monitoring habit, not in the experiment design, which is exactly why it's an easy trap to walk into even for careful analysts.

## 2. The formal problem

Under the null hypothesis (no real effect), the p-value computed from an accumulating sample doesn't sit still — it behaves like a **random walk** that wanders up and down as more data arrives. A fixed threshold like p < 0.05 is a boundary in that walk's path, and a random walk that's allowed to run long enough will cross *any* fixed boundary eventually, with probability approaching 1 (this is a consequence of the Law of the Iterated Logarithm). 

In plain terms: **if you peek often enough and are willing to stop the moment p < 0.05, you are *guaranteed* to eventually see a "significant" result — even with zero true effect.** It's not a matter of if, only when.

Even with a modest, discrete number of equally-spaced peeks, the inflation is severe. Approximate cumulative false-positive rates when peeking at a nominal α = 0.05 threshold each time (illustrative, from the repeated-significance-testing literature):

| Number of peeks (K) | True cumulative false-positive rate (≈) |
|---|---|
| 1 (standard, no peeking) | 0.050 |
| 2 | 0.083 |
| 3 | 0.107 |
| 5 | 0.142 |
| 10 | 0.193 |
| 20 | 0.248 |

Two peeks nearly doubles your real error rate. Twenty peeks — a metric checked daily for three weeks — puts your *actual* false-positive rate near 25%, five times what the dashboard's "p < 0.05, ship it" implies.

## 3. Three ways to actually fix it

### A. Fixed horizon (no peeking)
Pre-commit to a sample size / duration, don't look at the p-value until you get there.
- **Works when:** you can tolerate waiting; simplest possible guarantee.
- **Fails when:** it's the least practical — teams *will* look, and it wastes the chance to stop early on a clear winner or a clear loser (bad for user harm and opportunity cost).

### B. Group sequential testing (alpha-spending)
Pre-register a small number of planned looks (e.g., 25%, 50%, 75%, 100% of target sample), and spend your total α = 0.05 error budget across those looks using a spending function rather than using 0.05 at each one.
- **Pocock boundaries:** roughly constant, slightly-stricter-than-0.05 threshold at every look. Good when you want reasonably equal power to stop early at any look.
- **O'Brien-Fleming boundaries:** very strict early thresholds (nearly impossible to stop early), loosening toward something close to 0.05 near the final look. Good when you want to strongly discourage premature stopping but still preserve most of your power at the planned endpoint.
- **Works when:** you know roughly how many times you want to check in advance (common in A/B testing platforms with weekly reviews).
- **Fails when:** you actually want to check *continuously* (real-time dashboards) — you're locked into the specific look schedule you pre-registered; an unplanned extra peek re-breaks the guarantee.

### C. Always-valid p-values / confidence sequences (anytime-valid inference)
Built from a different mathematical object — a **test martingale** (or equivalently, a mixture sequential probability ratio test). The key property, from **Ville's inequality**, is:

**P(the statistic ever exceeds 1/α, at *any* stopping time, under the null) ≤ α**

This is a much stronger guarantee than a normal p-value gives you: it holds no matter *when* or *how often* you look, including looking after every single new data point, with no pre-registered schedule at all.
- **Works when:** you want a live dashboard where "p < 0.05, stop now" is always a safe decision, at any moment, without planning peeks in advance. This is the machinery behind modern "peeking-safe" experimentation platforms.
- **Fails when:** for a *fixed*, non-peeked sample size, an always-valid p-value is more conservative (less powerful) than a standard p-value — you're paying a permanent power tax in exchange for the freedom to look whenever you want. If you genuinely never peek, it's the wrong tool.

## 4. Comparison table

| Method | What it protects against | How | Best used when | Why it won't work / limitations |
|---|---|---|---|---|
| **Fixed horizon, no peeking** | Peeking inflation | Pre-commit sample size, check once | You can enforce discipline not to look early | Impractical — teams look anyway; can't stop early even for obviously harmful variants |
| **Group sequential (Pocock)** | Peeking inflation across a *few* planned looks | Constant, slightly-stricter boundary per look | You know the number/timing of looks in advance | Breaks if you add an unplanned extra look; not built for continuous monitoring |
| **Group sequential (O'Brien-Fleming)** | Same, but weighted to protect the final look | Very strict early thresholds, looser late | You want most of your power preserved at the planned endpoint, early stopping rare | Nearly impossible to stop early even with a real effect — slow to react |
| **Always-valid p-values (test martingales / mSPRT)** | Peeking inflation at *any* stopping time, unplanned | Ville's inequality guarantee on a martingale statistic | Live dashboards, continuous monitoring, no fixed look schedule | Lower power than a standard p-value if you truly never peek — pays for flexibility you didn't use |

## 5. Worked example

Say a metric has **no true effect**. A team checks the dashboard every day for 20 days, stopping the moment p < 0.05.

- **Naive interpretation:** "we hit p = 0.04 on day 12, ship it" — treated as a normal 5%-risk decision.
- **Reality:** with ~20 peeks, the true chance of hitting p < 0.05 at *some* point during the 20 days is roughly **25%**, not 5% — a fivefold understatement of risk.
- **Fix with always-valid p-values:** the same 20-day monitoring stream, but using an anytime-valid statistic, guarantees the true false-stop probability stays at 5% *regardless* of how many days they checked or when they stopped — the dashboard-checking habit stops being a hidden liability.

## 6. Production considerations

- **This is a different multiplicity axis than the metrics/segments chapter** — that one was "many hypotheses at one time," this one is "one hypothesis, many times." Real experimentation systems need to guard against both simultaneously; correcting for one doesn't fix the other.
- **Modern experimentation platforms increasingly default to always-valid p-values** for exactly this reason — it removes the operational burden of telling analysts "don't look until day 14," which is a rule people reliably break.
- **Group sequential design is still preferred in some regulated / high-stakes settings** (e.g., clinical trials) because the look schedule is auditable and pre-registered — always-valid methods are powerful but relatively newer and less universally standardized in some domains.
- **The two failure modes compound:** peeking on many metrics multiplies the inflation from both problems at once — this is the realistic worst case ("we checked 15 metrics every day for two weeks and one came back significant") and needs both a peeking-safe statistic *and* a multiple-comparison correction.

## 7. Interview traps

- **Trap 1:** Treating "we looked a few times and it was significant by the end" as a normal 5%-risk result — the actual risk compounds with every unplanned look.
- **Trap 2:** Assuming a stricter fixed threshold (like Bonferroni) fixes peeking — it doesn't; Bonferroni corrects for *many simultaneous hypotheses*, not repeated looks at the *same* one over time. These are genuinely different problems needing different machinery.
- **Trap 3:** Assuming always-valid p-values are "just a stricter p-value" — the mechanism (martingale + Ville's inequality) is structurally different from a corrected threshold, and it's what allows validity at *unplanned*, *arbitrary* stopping times, which no fixed correction can offer.

## 8. Comprehension check
1. **Why does an accumulating p-value behave like a random walk, and why does that make any fixed threshold eventually crossable?**

Under the null hypothesis, each new batch of data contributes a small, roughly independent nudge to your cumulative test statistic — sometimes up, sometimes down, centered on "no effect." As you accumulate more data and keep recomputing the p-value at each new total, that sequence of p-values (or more precisely, the underlying test statistic) traces out a path that looks like a random walk: it wanders, with no persistent drift in either direction, but it doesn't stay put either. A standard p-value threshold (e.g., $p<0.05$) is calibrated assuming you check it exactly *once*, at a pre-committed sample size. But a random walk, given enough steps, will cross *any* fixed boundary eventually — this is a basic property of random walks (they're recurrent; with probability approaching 1, they hit any fixed level if you let them run long enough). So if you keep checking your p-value after every new data point and stop the moment it dips below 0.05, you're not testing "is there a real effect" — you're testing "did this random walk hit a boundary yet," which it will do with near certainty if you check often enough and for long enough, even when the null is exactly true.

2. **A team peeks 10 times before deciding to ship. Roughly how inflated is the true false-positive rate?**

For continuous monitoring, the rule-of-thumb inflation is roughly the square root of the number of looks, though the exact number depends on the correlation structure between looks. A commonly cited approximation: with $k$ looks at $\alpha=0.05$ per look, the overall false-positive rate approaches something like $\alpha\sqrt{k}$ for smallish $k$ under Brownian-motion-style approximations, though this isn't exact — actual simulations for 10 independent-ish looks typically show the true FWER landing somewhere in the **25–35% range**, well over 5x the nominal rate. The precise number depends on how correlated consecutive looks are (looks close together in time are highly correlated, which dampens the inflation somewhat compared to 10 fully independent tests, which would give $1-(0.95)^{10}\approx 40\%$). The practical takeaway for an interview: don't worry about nailing the exact number — the important point is that peeking 10 times pushes your true false-positive rate from 5% into the **25–40% ballpark**, an order-of-magnitude-relevant inflation, not a rounding error.

3. **Pocock vs. O'Brien-Fleming spending functions — which protects early stopping power, which protects the final look?**

- **Pocock**: spends alpha roughly *evenly* across all looks — each interim look uses a similar, moderate significance threshold throughout. This makes it **easier to stop early** (you don't need overwhelming evidence at look 2 to declare significance), which is attractive if stopping early has real business value (e.g., saving cost, ending an experiment that's clearly a loser). The tradeoff: because you "spent" more alpha early, the threshold at the *final* look is stricter than a naive $\alpha=0.05$ would suggest, so Pocock has **less power at the final, full-sample-size look** relative to a design that saved its alpha budget for later.

- **O'Brien-Fleming**: spends alpha very conservatively early on and saves most of the budget for later looks — early thresholds are extremely strict (nearly impossible to stop early unless the effect is enormous), while the final-look threshold ends up very close to the nominal $\alpha$ you'd use in a single fixed-sample test. This makes O'Brien-Fleming **protect final-look power much better**, at the cost of making early stopping nearly impossible except for very large effects.

**One-line summary**: Pocock trades final-look power for early-stopping flexibility; O'Brien-Fleming trades early-stopping flexibility for final-look power. If you expect the effect (if real) to be modest and want your last look to have the most punch, O'Brien-Fleming is the standard choice; if early operational savings matter more and you're comfortable with a stricter final bar, Pocock fits better.

4. **What guarantee does Ville's inequality give an always-valid p-value that a standard p-value doesn't?**

A standard p-value only controls the false-positive rate at a *single, pre-specified* look — its validity guarantee is "if you check this exactly once, at the sample size you committed to in advance, the false-positive rate is $\alpha$." It says nothing about what happens if you check it again tomorrow.

Ville's inequality (the mathematical backbone behind always-valid p-values / sequential testing frameworks like mSPRT) gives a much stronger guarantee: it bounds the probability that a certain nonnegative supermartingale (which the running "evidence" process is constructed to be) **ever** exceeds a threshold, at **any** stopping time — not just at one pre-chosen point. Concretely, it guarantees that $P(\exists t: M_t \geq 1/\alpha) \leq \alpha$ for a martingale $M_t$ starting at 1. Translated into practice: an always-valid p-value stays a valid, uninflated p-value **no matter when you look, no matter how many times you look, and no matter what stopping rule you use** (even a stopping rule that peeks at the data to decide when to stop). That's the guarantee a standard p-value fundamentally cannot offer — it's only honest at one specific, pre-committed moment.

5. **Is Bonferroni-correcting across 5 planned check-ins the right fix for peeking? Why or why not, and what would you recommend instead?**

It's not the right tool, though it's a very understandable instinct since peeking *is*, structurally, a multiple-testing problem in disguise. The issue is that Bonferroni (and FWER correction generally) was built for testing several **distinct, separate hypotheses** at one point in time — it doesn't account for the fact that here, the 5 "tests" are the *same* hypothesis, tested on *increasingly overlapping data* (look 3 includes all the data from looks 1 and 2, plus more). That overlap creates strong correlation between consecutive looks that Bonferroni's independence-agnostic (and actually correlation-ignoring, conservative) correction doesn't properly exploit — it would work as a crude, overly conservative patch (dividing $\alpha$ by 5 does technically control the false-positive rate here too, in a worst-case sense), but it throws away a lot of power unnecessarily, and it doesn't give you the flexibility to stop at an *unplanned* time if something dramatic happens between check-ins.

The better-fit tool is a **purpose-built sequential testing framework**: either a **group sequential design with a spending function** (Pocock or O'Brien-Fleming, question 3) if the 5 check-ins are pre-planned and fixed in number, or an **always-valid p-value / mSPRT-based approach** (question 4) if you want the flexibility to check literally whenever you want, however often you want, without pre-committing to exactly 5 looks. Both are specifically designed to exploit the correlation structure of accumulating data rather than treating each look as an independent test — giving you a properly calibrated false-positive rate with meaningfully more power than a blunt Bonferroni correction would allow.
