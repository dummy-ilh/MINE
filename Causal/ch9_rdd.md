# Chapter 9: Regression Discontinuity Design (RDD)

## 1. Explanation

### The situation this method exploits

RDD works whenever treatment assignment changes abruptly at a known **threshold** of some continuously-measured "running variable" — a test score, an account age, a bid amount, a review count, a subscriber count. The core intuition: **a unit that scores 79.9 and a unit that scores 80.1 are, for all practical purposes, the same kind of unit** — there's nothing meaningfully different about them except which side of an arbitrary line they happened to fall on. So any sudden *jump* in the outcome exactly at that threshold must be attributable to whatever treatment kicks in there — not to some underlying difference between the two groups of units, because right at the boundary, there essentially isn't one.

This is sometimes described as "a locally randomized experiment hiding inside observational data" — you didn't randomize anything, but the arbitrary, sharp cutoff does the randomizing for you, at least in the immediate neighborhood of that cutoff.

### Sharp vs. Fuzzy RDD — the distinction is about what "crossing the cutoff" actually guarantees

**Sharp RDD**: crossing the cutoff *deterministically* assigns treatment — everyone with score ≥ 80 gets the scholarship, no exceptions, no one below gets it. Because treatment is a perfect step function of the running variable, the estimator is literally the jump in the outcome's conditional expectation function right at the cutoff:
```
τ = lim_{x→c⁺} E[Y|X=x] − lim_{x→c⁻} E[Y|X=x]
```
You're comparing the limit of the outcome function approaching the cutoff from above versus from below — conceptually, "what does the outcome function look like infinitesimally close to the cutoff on each side."

**Fuzzy RDD**: crossing the cutoff changes the *probability* of treatment, but isn't fully deterministic — for instance, some people just below the cutoff can still get the scholarship through an appeals process, and some people just above might decline it. Here, you can't simply read off the outcome jump as the effect, because you don't know how many people on each side actually got treated. Instead, you divide the outcome-jump by the *treatment-probability*-jump at the cutoff — this is structurally identical to the IV/Wald estimator from Chapter 7, using "crossed the cutoff" as an instrument for actual treatment receipt:
```
τ_fuzzy = [ lim_{x→c⁺}E[Y|X=x] − lim_{x→c⁻}E[Y|X=x] ] / [ lim_{x→c⁺}E[D|X=x] − lim_{x→c⁻}E[D|X=x] ]
```
This connection to IV isn't a coincidence — it's the exact same logic, just with "crossing a specific numeric threshold" playing the role of the instrument instead of a lottery or scheduling quirk.

### The identifying assumption: continuity, and the danger of manipulation

The core requirement: everything *else* that could plausibly affect Y must vary **smoothly** through the cutoff — no other factor should happen to jump discontinuously at that exact same X value (otherwise you can't tell whether the outcome-jump is due to your treatment or to this other coincidental jump). 

There's a second, equally critical requirement: units must not be able to **precisely manipulate** their own value of X to land on the favorable side of the cutoff. If people know exactly where the cutoff is and can nudge their score across it (think: a test-taker who knows 80 is the magic number and finds a way to get exactly 80 or 81 rather than 78), then the group "just above" the cutoff is no longer a random slice of similar units — it now contains a subset of people who successfully manipulated their way across, who may be systematically different (e.g., more resourceful, more informed, more motivated) from those who didn't or couldn't manipulate. This breaks the "just like each other except for treatment" logic that RDD depends on entirely.

**How you check for manipulation, concretely**: plot a histogram (density) of the running variable X. If units are manipulating their way past the cutoff, you'd expect to see an unusually high density of observations *just above* the cutoff and a corresponding unusually low density (a dip) *just below* it — a "bunching" pattern. This is formalized as the **McCrary density test**. A smooth, continuous density through the cutoff (no bunching) is reassuring — not proof of no manipulation, but supportive evidence against it.

### Bandwidth: the fundamental tradeoff, explained intuitively

You only trust comparisons "close" to the cutoff, since the "these units are basically identical" logic weakens the further away from the cutoff you look. This raises the question: how close is close enough? This is controlled by a **bandwidth** parameter $h$ — you only use observations within $h$ of the cutoff (or weight nearby observations more heavily than farther ones).

- **Too narrow a bandwidth**: you're left with very few data points near the cutoff, so your estimate becomes noisy (high variance) — you're very confident the *comparison* is valid, but you don't have enough data to pin down the number precisely.
- **Too wide a bandwidth**: you start including units that are further from the cutoff and therefore more likely to be genuinely different in other ways — you're comparing along more of the underlying relationship's slope, not just capturing the local jump, which reintroduces bias (essentially the same "curse" as regression adjustment's functional-form problem, since now you're also relying on correctly extrapolating the shape of the relationship within your bandwidth window).

Best practice: use a data-driven, MSE-optimal bandwidth selection procedure (e.g., Calonico-Cattaneo-Titiunik), and — critically for interviews — show that your result is **stable across several different reasonable bandwidth choices**, rather than reporting a single number from one arbitrary window and hoping it's robust.

## 2. Example

### Example A — Sharp RDD, straightforward case

A scholarship is awarded if test score ≥ 80. Suppose local averages, computed just on each side of the cutoff:
- Just below (scores 75–79): average future GPA = 2.8
- Just above (scores 80–84): average future GPA = 3.1

```
τ_RDD ≈ 3.1 − 2.8 = 0.3
```
**Interpretation**: for students right around the cutoff, the scholarship raises future GPA by about **0.3 points**. This is a strictly **local** estimate — it says absolutely nothing about the effect for a student who scored 40 (who never realistically competes for this scholarship) or 99 (who might have gotten strong outcomes regardless of the scholarship) — this local-vs-global limitation is the classic RDD external-validity tradeoff, and is one of the most reliably-tested interview concepts for this method.

### Example B — Fuzzy RDD, worked in full, with the IV connection made explicit

A creator monetization program requires 1,000 subscribers to be **eligible** to apply for a "Creator Fund," but eligibility ≠ automatic enrollment — some eligible creators don't bother applying, and some just-under-1000-subscriber creators get in anyway via a manual review exception. This is fuzzy RDD, because crossing the subscriber threshold changes the *probability* of enrollment, but doesn't determine it perfectly.

Running variable X = subscriber count, cutoff c = 1,000. Local averages for creators near the cutoff (binned):

| Bin | Avg X | E[D] (fraction enrolled) | E[Y] (avg monthly earnings, $) |
|---|---|---|---|
| 950-999 | 975 | 0.10 | 220 |
| 1000-1050 | 1025 | 0.65 | 410 |

**Jump in enrollment probability at the cutoff** (this is the "first stage," exactly analogous to Chapter 7's IV first stage):
```
E[D | just above] − E[D | just below] = 0.65 − 0.10 = 0.55
```
**Jump in earnings at the cutoff** (the "reduced form"):
```
E[Y | just above] − E[Y | just below] = 410 − 220 = 190
```
**Fuzzy RDD estimate**:
```
τ_fuzzy = 190 / 0.55 = 345.5
```
**Interpretation**: for creators near the 1,000-subscriber threshold whose enrollment status was actually *changed* by crossing the cutoff (the "compliers" in this local population — mirroring exactly the complier logic from Chapter 7's IV discussion), Creator Fund enrollment causes roughly **$345.50/month additional earnings**. This is a strictly local claim about creators right around 1,000 subscribers — a creator with 100 subscribers or 100,000 subscribers is not described by this number at all.

## 3. Interview Q&A

**Q: Why can't you just use RDD's estimate to justify lowering the eligibility threshold to, say, 500 subscribers, expecting the same dollar effect?**
A: RDD's estimate is local to the cutoff — it describes the effect for creators *right around 1,000 subscribers*. Creators around 500 subscribers are a fundamentally different population (likely much smaller, less established channels), and there's no data-driven reason to assume the same effect size applies there; the underlying conditional expectation function could look completely different away from the studied cutoff. This "external validity" limitation is one of RDD's most commonly tested interview points.

**Q: Describe the McCrary density test in plain language and what result would concern you.**
A: It's a check for whether units seem to have "sorted" themselves around the cutoff on purpose — you look at a histogram of the running variable and ask whether it's smooth through the cutoff, or whether there's a suspicious bump right above it (and a dip right below). A big bump just above suggests people knew the cutoff and successfully engineered their way just past it (e.g., inflating their subscriber count right before applying), which would mean the "just above vs just below" groups are no longer comparable, breaking the RDD logic entirely.

**Q: What's the difference in what sharp vs fuzzy RDD can identify, in the language of estimands from Chapter 2?**
A: Sharp RDD identifies the local ATE right at the cutoff (since treatment is deterministic there, it behaves like a mini-RCT at exactly that point). Fuzzy RDD, like IV, only identifies a local ATE for the "compliers" — units whose actual treatment status is moved by crossing the cutoff — technically a Local Average Treatment Effect (LATE), localized specifically to the cutoff.

**Q: If you shrink your bandwidth to just ±5 subscribers around the cutoff, what do you gain and what do you lose?**
A: You gain more credibility that units on either side are truly comparable (the "just like each other" assumption is far more plausible over ±5 than over ±500), reducing bias from picking up the underlying slope of the relationship — but you lose sample size dramatically, inflating variance and making your point estimate noisy with a wide confidence interval. This tradeoff is exactly why bandwidth selection is often done via a formal MSE-minimizing procedure, and why results should be shown across multiple bandwidths as a robustness check rather than relying on one arbitrary window.

**Q: How would a covariate balance check work in an RDD setting, analogous to the balance check in PSM (Chapter 6)?**
A: Check that *predetermined* covariates — things fixed before a unit could possibly know about or manipulate around the cutoff, like a creator's country, channel creation date, or content category — don't show a discontinuous jump at the cutoff themselves. If they do jump, that suggests either manipulation, or that some other factor coincidentally changes at exactly the same threshold, undermining the "as good as random locally" logic the whole design depends on.

**Q: Explain, in your own words, why the fuzzy RDD formula is structurally identical to the IV Wald estimator from Chapter 7.**
A: In both cases, you have some variable (an instrument Z, or "crossed the cutoff" in RDD) that shifts the probability of treatment without perfectly determining it, and you want to isolate the treatment's effect using only the "clean" variation driven by that variable. The formula in both cases divides the observed jump/difference in the outcome by the corresponding jump/difference in treatment probability — normalizing the "total observed effect of the instrument" by "how much of treatment the instrument is actually responsible for." Fuzzy RDD is really just IV where the instrument happens to be a specific numeric threshold-crossing indicator rather than a lottery or scheduling quirk.

**Q: A stakeholder asks you to also report the RDD effect for creators with 100 subscribers, using the same fitted local model. How do you respond?**
A: I'd explain this isn't something the RDD design can support — the whole logic of RDD's validity comes from comparing units in a vanishingly small neighborhood right around the cutoff, where "just like each other except for treatment" is plausible. Extrapolating the fitted local relationship out to 100 subscribers (far outside that neighborhood) would just be functional-form-driven extrapolation, the same kind of unsupported extrapolation problem discussed in Chapter 5 — a fundamentally different (and much weaker) exercise than what RDD is designed to deliver.

---
**Previous: Chapter 8 — Difference-in-Differences**
**Next: Chapter 10 — Synthetic Control**
