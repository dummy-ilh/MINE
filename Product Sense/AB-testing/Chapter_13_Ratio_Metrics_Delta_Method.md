# Chapter 13: Ratio Metrics & the Delta Method
# Chapter 13 (Rebuilt for Clarity): Ratio Metrics & the Delta Method

This is a from-scratch rebuild of Chapter 13, built slowly with diagrams at every step. If the formula-first version didn't click, this version is designed to get you to the same place — being able to derive and explain the Delta Method — by building up from a very concrete example first, and only introducing the formula once the *shape* of the problem is obvious.

---

## 1. Start Here: Two Completely Different Kinds of "Average"

Before ratios, you need to see the distinction this whole chapter hinges on.

**Type 1 — Simple mean.** "Average clicks per user." Each user contributes exactly ONE number (their click count). The "per user" part isn't random — there's always exactly 1 user per user.

```
User A: 5 clicks
User B: 3 clicks
User C: 8 clicks
                    Average = (5+3+8)/3 = 5.33
                    ↑
            Only ONE random quantity here (the click count).
            The denominator ("per user") is fixed = 1. Always.
```

**Type 2 — Ratio metric.** "Clicks per session." Now BOTH numbers are random and different per user — how many clicks, AND how many sessions.

```
User A: 10 clicks / 2 sessions  = 5.0
User B: 3 clicks  / 1 session   = 3.0
User C: 16 clicks / 4 sessions  = 4.0
                    ↑                ↑
            random number 1    random number 2
            (varies per user)  (ALSO varies per user)
```

This is the entire problem in one picture: **a ratio metric has two moving parts, not one.** Every formula and every mistake in this chapter comes back to this picture.

---

## 2. Why This Actually Breaks Things — The Seesaw Picture

Here's the intuition for *why* two moving parts causes trouble, before any math.

```
        Simple mean:                    Ratio metric:

           ●                              ●━━━━━━━━━━●
           │  (one thing wobbles,          A          B
           │   you measure the             (numerator)(denominator)
           │   wobble directly)            BOTH wobble — and if
           ▼                               they wobble TOGETHER
      just measure                         (correlated), the seesaw
      Var(clicks)                          barely tips even though
                                           each side is moving a lot
```

If a user has more sessions (denominator goes up) AND that same user tends to click more overall (numerator goes up too), the two movements partially cancel out in the ratio. A user with 2x the sessions and roughly 2x the clicks still has almost the SAME ratio. The ratio is more *stable* than either raw number alone — but only if you correctly account for the fact that they move together. If you ignore that they move together, you'll think the ratio is wobblier than it really is.

---

## 3. The Two Separate Mistakes People Make

```
MISTAKE #1: Computing each user's OWN ratio, then averaging those ratios

  User A: 1 click / 1 impression   = 1.00  ←  based on tiny, unreliable sample
  User B: 800 clicks / 1000 impr.  = 0.80  ←  based on huge, reliable sample

  Naively averaging: (1.00 + 0.80) / 2 = 0.90
                      ↑
        User A's flimsy, 1-data-point ratio counts EXACTLY as much
        as User B's rock-solid, 1000-data-point ratio. That's wrong.

  FIX: Don't average individual ratios. Pool everything first:
       total clicks / total impressions = (1+800) / (1+1000) = 0.80
       ↑ this correctly lets big, reliable users carry more weight


MISTAKE #2: Even using the correctly-pooled ratio, using the WRONG
            variance formula — i.e., forgetting that numerator and
            denominator move together (covariance)

  This is the one the Delta Method exists to fix. Section 4 onward.
```

Mistake #1 is a weighting problem (fixed by pooling correctly). Mistake #2 is a variance-formula problem (fixed by the Delta Method). This chapter is mostly about Mistake #2.

---

## 4. Building the Delta Method From Nothing

Forget the formula for a second. Here's the logic, step by step.

**Step 1**: You have a ratio $R = X/Y$. You want to know how much $R$ wobbles (its variance) when $X$ and $Y$ both wobble.

**Step 2**: If $X$ and $Y$ were completely unrelated to each other, you could just add up their separate wobbles (scaled appropriately). Easy case.

**Step 3**: But usually $X$ and $Y$ move together — that's the covariance. If they move in the *same* direction (both up or both down together), the ratio is MORE stable than the "unrelated" case suggests, because a rise in $Y$ (denominator) is accompanied by a proportional rise in $X$ (numerator), and the two effects on the ratio partially cancel.

**Step 4**: So the honest formula for the ratio's variance has to be: *(wobble from X) + (wobble from Y) − (a correction for how much they move together).*

That "minus a correction" is the covariance term. That's it. That's the whole idea:

```
   Var(ratio)  =  [wobble from numerator]
                + [wobble from denominator]
                − [how much they move together] × 2
                     ↑
          this term is EXACTLY what people forget,
          and forgetting it is the single most common mistake
          in this entire topic
```

**Step 5 — now the actual formula**, which is just Step 4 written precisely (this comes from a first-order Taylor expansion of $R=X/Y$, if you ever need to derive it from scratch):

$$Var(R) \approx \frac{1}{\bar{Y}^2}\Big[\,Var(X) + \bar{R}^2\,Var(Y) - 2\bar{R}\cdot Cov(X,Y)\,\Big]$$

where $\bar X, \bar Y$ are the average numerator/denominator, and $\bar R = \bar X / \bar Y$.

**Read it exactly like Step 4**: $Var(X)$ is "wobble from numerator," $\bar R^2 Var(Y)$ is "wobble from denominator" (scaled by $\bar R^2$ so the units match), and $-2\bar R \cdot Cov(X,Y)$ is "the correction for moving together."

---

## 5. 🧭 Diagram: Which Direction Does the Mistake Go?

This is the part most people get backwards, so it gets its own diagram.

```
              Are numerator and denominator
              POSITIVELY correlated?
              (common case — e.g., users who have
               more sessions also click more overall)
                          │
              ┌──────Yes──┴──No (negative)────────┐
              ▼                                    ▼
   The "moving together" term            The "moving together" term
   SUBTRACTS from the naive              now ADDS to what naive
   variance estimate.                    math would suggest.
              │                                    │
              ▼                                    ▼
   If you FORGET this term,              If you FORGET this term,
   your variance estimate               your variance estimate
   comes out TOO BIG.                    comes out TOO SMALL.
              │                                    │
              ▼                                    ▼
   Confidence interval too WIDE.         Confidence interval too
   You become too CONSERVATIVE           NARROW. You become too
   — you might miss a real              CONFIDENT — you might see
   effect (lost power).                  a "significant" result
                                          that's actually just noise.
```

**The one-sentence version**: forgetting the covariance term isn't automatically "safe" — whether it makes you overconfident or underconfident depends entirely on which way the correlation points. Positive correlation (the usual case) → you become too conservative. Negative correlation (rarer) → you become falsely confident.

---

## 6. Worked Example, Slowly — CTR (clicks / impressions)

Same numbers as before, but every line explained.

**Setup**: 1,000 users in the treatment group. For each user we know their clicks and impressions.
- Average clicks per user, $\bar X = 5$
- Average impressions per user, $\bar Y = 50$
- So the pooled CTR, $\bar R = 5/50 = 0.10$ (10%)
- How much clicks vary user-to-user: $Var(X) = 20$
- How much impressions vary user-to-user: $Var(Y) = 400$
- How much clicks and impressions move together: $Cov(X,Y) = 60$ (positive — busier users generate more of both)

**Plug into the formula, term by term:**

```
Var(R) ≈ (1/Ȳ²) × [ Var(X)  +  R̄² × Var(Y)  −  2R̄ × Cov(X,Y) ]
             │           │            │                │
             │      "wobble from   "wobble from   "correction for
             │       numerator"    denominator,     moving together"
             │       = 20          scaled"          = 2×0.10×60 = 12
             │                     = 0.10² × 400
             │                     = 4
             ▼
        1/50² = 1/2500

Var(R) ≈ (1/2500) × [ 20 + 4 − 12 ]  =  (1/2500) × 12  =  0.0048
```

Then, since this is a per-user variance and we have 1,000 users, the variance of the *average* ratio is $0.0048/1000$, so:

$$SE(R) = \sqrt{0.0048/1000} \approx 0.00219$$

**Now the mistake, side by side**: if you'd forgotten the covariance correction (the "−12" term):

```
Var(R)_naive ≈ (1/2500) × [20 + 4]  =  24/2500  =  0.0096

     0.0096  is EXACTLY DOUBLE  the correct  0.0048
                         ↑
        Forgetting the covariance term here made the estimated
        variance twice as large as it should be — an unnecessarily
        wide confidence interval, and a test with less power than
        it actually has. This matches the "positive correlation →
        too conservative" branch of the Section 5 diagram.
```

---

## 7. Second Worked Example — Same Idea, Different Numbers, So the Pattern Sticks

"Clicks per session": $\bar X = 8.0$ clicks, $Var(X)=16$; $\bar Y = 4.0$ sessions, $Var(Y)=4$; $Cov(X,Y)=5$ (positive again); $\bar R = 8.0/4.0 = 2.0$.

```
Var(R) ≈ (1/16) × [ 16 + (2.0)²×4 − 2(2.0)(5) ]
        = (1/16) × [ 16 + 16 − 20 ]
        = (1/16) × 12
        = 0.75

Naive (forgetting covariance):
Var(R)_naive ≈ (1/16) × [16 + 16] = 2.0

     2.0 vs. the correct 0.75  →  naive is ~2.7x too large
```

Same story, different metric: positive correlation → forgetting it inflates your variance estimate → overly conservative test.

---

## 8. When You DON'T Need Any of This

Quick sanity check, since this is the most common place people over-apply the method:

```
Is the "denominator" of your metric actually a FIXED number
(like "1 user"), not something that varies?
                    │
        ┌─────Yes───┴───No──────────┐
        ▼                            ▼
  This is just a SIMPLE MEAN.   This is a genuine ratio metric.
  ("average clicks per user")   Use the Delta Method (or bootstrap).
  Standard variance formula
  is already correct. Stop here.
```

Example: "average clicks per user" — the denominator is always exactly 1 (one user per user), so it's not random at all. No Delta Method needed. But "clicks per session" — sessions-per-user is a random, varying count — so the Delta Method applies.

---

## 9. The Practical Alternative: Bootstrap

If the Delta Method's approximation feels shaky — denominators near zero, very skewed data — there's a simulation-based escape hatch that needs no formula at all:

```
1. Take your 1,000 users' (clicks, impressions) pairs.
2. Resample 1,000 users WITH REPLACEMENT (some users picked
   twice, some not at all — that's the point).
3. Compute the pooled ratio on this resample.
4. Repeat steps 2-3 thousands of times.
5. Look at the spread of all those resampled ratios —
   that spread IS your variance estimate, no formula needed.
```

It's more computationally expensive, but it automatically captures the numerator/denominator covariance without you ever writing down $Cov(X,Y)$ explicitly — the resampling does it implicitly, because clicks and impressions are resampled together, per user, preserving whatever relationship they actually have.

---

## 10. Full Picture — One Diagram Tying It All Together

```
              You have a metric that looks like "A per B"
                                 │
                                 ▼
              Is B fixed (e.g., always "1 user")?
                    │                    │
                   Yes                   No
                    │                    │
                    ▼                    ▼
            Simple mean.          Genuine ratio metric.
            Standard formula      Both A and B vary.
            is correct. Done.              │
                                            ▼
                              Pool first: total A / total B
                              (don't average individual
                               per-user ratios — Mistake #1)
                                            │
                                            ▼
                              Estimate Var(A), Var(B), Cov(A,B)
                              from your data
                                            │
                                            ▼
                              Apply Delta Method formula
                              (Section 4, Step 5)
                                            │
                                            ▼
                    Is the approximation trustworthy?
                    (denominator not near zero, not
                     wildly skewed)
                          │                    │
                         Yes                   No
                          │                    │
                          ▼                    ▼
                   Use Delta Method       Use BOOTSTRAP instead
                   result directly        (Section 9)
```



## 12. Comprehension Check — Try These Before Moving On

1. Draw the "two moving parts" picture (Section 1) for a metric of your choosing that ISN'T clicks/impressions or clicks/sessions.
2. In your own words (no formula), explain why positive correlation between numerator and denominator makes a ratio metric *more stable* than the two raw numbers alone.
3. Using the Section 5 diagram, predict — before calculating anything — whether ignoring covariance in a metric with negative correlation will widen or narrow your confidence interval.
4. Redo the Section 6 worked example by hand, but imagine $Cov(X,Y) = -60$ instead of $+60$. What does the corrected variance come out to, and is it bigger or smaller than the naive (no-covariance) estimate?
5. Explain, using Section 8's flowchart, why "average session duration per user" does NOT need the Delta Method, but "average clicks per session" does.
6. When would you reach for bootstrap over the Delta Method, and why does bootstrap not require you to explicitly compute Cov(X,Y)?

---


## 6. Levers — What Controls Ratio-Metric Variance

**Correlation between numerator and denominator**
- Strong positive correlation shrinks the true variance of the ratio relative to naive estimates — ignoring this makes your CI too wide (overly conservative, underpowered, but not falsely overconfident).
- Strong negative correlation (rarer — e.g., users who visit more often might convert at a lower rate per visit due to habituation) inflates the true variance relative to naive estimates — ignoring this makes your CI too narrow, risking false "significant" results.

**Granularity / level at which the ratio is computed**
- A ratio metric defined at the user level (e.g., avg clicks-per-session, averaged first within-user then across users) behaves differently statistically than one computed as a pooled ratio across all sessions (total clicks / total sessions, ignoring user boundaries) — the latter can bias toward heavy users (a user with many sessions contributes many "observations," implicitly overweighting them) and needs its own correction via stratification or clustering-aware variance estimation.
- Computing session-level ratios and averaging those (rather than aggregating to the user level first) risks implicitly weighting all sessions equally regardless of how many sessions each user contributes, and can mask the true numerator/denominator covariance structure at the user level.

**Choice of denominator / metric redefinition**
- Some ratio metrics can be redefined to avoid the problem entirely — e.g., instead of "clicks per session" (both random), you might use "did the user click at all" (binary, single random variable per user) as a simpler, more robust proxy metric, trading some information for statistical simplicity.

---

## 7. Production Considerations

- **Most experimentation platforms at scale compute ratio-metric variances via the delta method or via bootstrap resampling.** Bootstrap is a more computationally expensive but assumption-free alternative that naturally captures the numerator-denominator covariance without needing the closed-form formula — worth mentioning as the practical alternative when the delta method's linear approximation may not hold well (e.g., very skewed or small-sample ratios, or denominators close to zero).
- **Randomization unit vs. analysis unit mismatch is a related, adjacent trap**: if you randomize by user but analyze at the impression or session level (treating each impression as an independent observation), you dramatically overstate your effective sample size and understate your true SE — a distinct but related error to the ratio-metric problem, since impressions/sessions from the same user are correlated, not independent (this connects to the clustering/interference concepts covered elsewhere in this curriculum).
- **Variance-reduction techniques (CUPED) apply to ratio metrics too**, but require extending the delta-method variance formula to also incorporate covariance with the pre-experiment covariate — a natural extension worth flagging if asked to go deep on ratio-metric variance reduction specifically.
- **When NOT to worry about the Delta Method**: when the denominator isn't actually random — e.g., "average clicks per user" where the denominator is just "1 user" (a fixed unit of observation, not a random count). This is a simple mean, and the standard variance formula applies directly with no ratio-metric correction needed.

---

## 8. Interview Traps (Consolidated)

1. **Treating a ratio metric like a simple mean** and plugging observed per-user ratios directly into the standard two-sample t-test SE formula — ignoring both the weighting problem (Section 3, Problem 1) and the covariance problem (Section 3, Problem 2).
2. **Using the delta method formula but forgetting the covariance term entirely**, silently introducing a (possibly large) bias in your variance estimate — direction of the bias depends on the sign of the correlation, so it's not automatically "safe" to omit.
3. **Assuming ignoring covariance is automatically conservative (or automatically liberal)** — it can go either way depending on the correlation's sign (Section 4).
4. **Not recognizing when the delta method's linear approximation breaks down** (e.g., ratios with denominators close to zero, or highly skewed numerator/denominator distributions) — in these cases, flag bootstrap as the safer alternative rather than forcing the delta method to apply.
5. **Confusing the "ratio metric" variance problem with the "clustered/non-independent observations" problem** (analysis unit ≠ randomization unit) — related but distinct sources of SE misestimation; interviewers may probe whether you can tell them apart.
6. **Computing ratio metrics by pooling at the wrong level** (e.g., session-level instead of user-level), implicitly overweighting heavy users.
7. **Applying the Delta Method to a metric where the denominator is actually fixed** (e.g., a true simple per-user average) — unnecessary complexity where the standard formula is already correct.

---

## 9. Common Mistakes / Red Flags — Quick Review

- ❌ Applying the standard variance-of-a-mean formula directly to a ratio metric without accounting for denominator variance and covariance
- ❌ Assuming ignoring the covariance term is automatically "safe" or conservative — the direction of the error depends on the sign of the correlation
- ❌ Computing ratio metrics by pooling at the wrong level (e.g., session-level instead of user-level), implicitly overweighting heavy users
- ❌ Applying the Delta Method to a metric where the denominator is actually fixed (e.g., a true simple per-user average) — unnecessary complexity where the standard formula is already correct
- ❌ Forcing the delta-method linear approximation onto a ratio with a denominator near zero or heavy skew, instead of switching to bootstrap
- ✅ Check whether both numerator and denominator vary at your unit of randomization before deciding a ratio-metric correction is needed
- ✅ Estimate Cov(X,Y) from your data rather than assuming independence by default
- ✅ Aggregate to the user level (or your randomization unit) before computing the ratio, rather than pooling at a finer, non-independent level

---

## 10. Famous Interview Q&A

**Q: You're testing a new search ranking algorithm and using "clicks per search session" as your primary metric. Why can't you just use the standard variance-of-the-mean formula on this ratio?**
A: Because both clicks and sessions vary randomly per user, and treating "clicks per session" as if it were a simple per-user average ignores the covariance between the two — if users who search more often also click proportionally more (a very plausible pattern), that covariance meaningfully changes the true variance of the ratio, and ignoring it produces a biased (often overly conservative, sometimes overconfident) estimate of the metric's variance. I'd apply the Delta Method to correctly account for Var(X), Var(Y), and Cov(X,Y) together rather than just using Var(X)/n.

**Q: If the numerator and denominator of your ratio metric are positively correlated, does ignoring the correlation make your test too conservative or too aggressive?**
A: Positive correlation between numerator and denominator actually *reduces* the true variance of the ratio relative to what you'd naively estimate by treating them as independent — so ignoring it means your naive variance estimate is too large, making your confidence interval too wide and your test too conservative (you might fail to detect a real effect that a properly-calculated, tighter interval would have caught). This is the opposite direction of failure from negative correlation, which would make naive estimates too small and the test falsely overconfident — so the direction of the mistake genuinely depends on the correlation's sign, and can't be assumed to always be "safe."

**Q: A junior analyst computes a confidence interval on "revenue per session" by just applying the standard formula for the variance of a mean to the ratio values directly (session-level revenue/session, averaged). What's the issue?**
A: The core issue is which level the ratio is computed and averaged at. If they're computing revenue-per-session for each individual session and then averaging those session-level ratios, they may be implicitly weighting all sessions equally regardless of how many sessions each user contributes — heavy users (many sessions) get proportionally more influence just by having more observations, and simple pooling can also mask the true numerator/denominator covariance structure at the user level. The Delta Method, applied correctly at the user level (using per-user aggregated numerator and denominator, X̄ and Ȳ), gives a more defensible variance estimate that properly accounts for the user-level covariance structure, rather than pretending session-level ratios are independent, identically distributed observations.

**Q: When would you NOT need to worry about the Delta Method for a metric that looks like a ratio?**
A: When the denominator isn't actually random — e.g., "average clicks per user" where the denominator is just "1 user" (a fixed unit of observation, not a random count). This is a simple mean, and the standard variance formula applies directly with no ratio-metric correction needed. The Delta Method specifically matters when BOTH the numerator and denominator are random quantities that vary across your randomization unit — like sessions-per-user or revenue-per-visit where the "per" part itself fluctuates.

**Q: What's a practical, assumption-free alternative to the delta method when the ratio's denominator is close to zero or highly skewed?**
A: Bootstrap resampling — repeatedly resample your randomization units (e.g., users) with replacement, recompute the pooled ratio on each resample, and use the empirical distribution of those resampled ratios to estimate variance/confidence intervals directly. It's more computationally expensive than the closed-form delta method, but it naturally captures the numerator-denominator covariance and doesn't rely on the delta method's linear (Taylor-expansion) approximation, which can break down for small or skewed denominators.


**Q: Why can't you just use the standard "variance of a mean" formula on a ratio metric like clicks-per-session?**
A: Because that formula assumes only one thing is random. A ratio has two random parts — numerator and denominator — and if you ignore that they move together (covariance), you get the wrong answer. Picture the seesaw in Section 2: both sides are moving, and how they move *together* determines how much the ratio itself actually wobbles.

**Q: If numerator and denominator are positively correlated, does ignoring that make your test too conservative or too aggressive?**
A: Too conservative. Positive correlation means the two sides partially cancel out (Section 5), so the true variance is smaller than naive math suggests. Ignoring the covariance term inflates your estimated variance, widens your confidence interval more than necessary, and can cause you to miss a real effect. This is exactly what happened in both worked examples (2x and ~2.7x too large).

**Q: When would ignoring covariance make you *falsely* confident instead?**
A: Only when numerator and denominator are negatively correlated — rarer, but it happens (e.g., users who visit more often convert at a lower rate each time, due to habituation). In that case the "moving together" correction goes the other way, and skipping it makes your variance estimate too small — narrower confidence interval, higher chance of a false "significant" result.

**Q: A junior analyst averages each user's own clicks/impressions ratio and runs a t-test on those averages. What's wrong?**
A: Two separate things could be wrong. First (Mistake #1, Section 3): if they're averaging individual per-user ratios rather than pooling total clicks / total impressions, they're giving equal weight to a user with 1 impression and a user with 1,000 impressions, even though the second is a far more reliable estimate. Second (Mistake #2, this whole chapter): even after pooling correctly, a standard t-test's variance formula doesn't know to account for the covariance between clicks and impressions — you need the Delta Method (or bootstrap) for that part.

**Q: What do you do when the Delta Method's approximation seems unreliable?**
A: Switch to bootstrap (Section 9) — resample your randomization units with replacement, recompute the pooled ratio each time, and read the variance directly off the spread of resampled ratios. It's more expensive computationally but makes no linear-approximation assumption, so it holds up better when denominators are near zero or the data is heavily skewed.

---

## 11. L5-Differentiating Talking Points

- Being able to write out the delta method's Taylor expansion derivation, even briefly, rather than just quoting the final formula, shows you understand where it comes from rather than having memorized a lookup-table result.
- Proactively raising that the covariance term's sign/magnitude determines whether ignoring it makes your test too conservative or too liberal — rather than assuming it's "always fine to ignore" or "always makes you conservative" — shows precise, non-hand-wavy understanding.
- Mentioning bootstrap resampling as the assumption-free alternative when the delta method's approximation may be shaky (small denominators, heavy skew) demonstrates breadth beyond the one canonical formula.
- Connecting this topic to both the randomization-unit-vs-analysis-unit problem and to CUPED/variance-reduction shows you see ratio-metric variance estimation as one node in a connected web of "getting your standard errors right" problems, not an isolated formula to memorize.
- Being able to reproduce a concrete numeric contrast (correct vs. naive variance, as in both worked examples) live, on request, is a strong differentiator over reciting the formula alone.

---

## 12. Comprehension Check (Self-Test)

These questions test whether you **understand the intuition behind the Delta Method and ratio metrics**, not just the formulas. Here's how I'd answer them in an interview.

---

# 1. In your own words (no formula), explain why positive correlation between numerator and denominator makes a ratio metric more stable than the two raw numbers alone.

Imagine you're measuring:

* **Clicks per session**
* Numerator = clicks
* Denominator = sessions

Suppose users who have more sessions also naturally generate more clicks.

That means the numerator and denominator **move together**.

For example:

| User | Sessions | Clicks |
| ---- | -------- | ------ |
| A    | 10       | 50     |
| B    | 20       | 100    |

Both doubled.

The ratio stayed around **5 clicks/session**.

Even though both raw numbers changed a lot, the **ratio hardly changed** because increases in one were accompanied by increases in the other.

That's why **positive correlation stabilizes the ratio**—the numerator and denominator "track" each other, so random fluctuations partially cancel out instead of amplifying each other.

---

# 2. Using the Section 5 diagram, predict—before calculating anything—whether ignoring covariance in a metric with negative correlation will widen or narrow your confidence interval.

Suppose numerator and denominator are **negatively correlated**.

Example:

When sessions go up, clicks tend to go down.

Now imagine the ratio.

The numerator and denominator move in **opposite directions**, making the ratio fluctuate more.

Negative correlation therefore **increases the variance** of the ratio.

If you ignore covariance, you're ignoring this extra source of variability.

So you'll **underestimate the true variance**, which means:

* standard error is too small
* confidence interval becomes **too narrow**
* you become overly confident
* Type I error increases

**Prediction:** Ignoring negative covariance **narrows the confidence interval** (incorrectly).

---

# 3. Redo the worked example assuming Cov(X,Y) = –60 instead of +60.

Assume the worked example used:

* Var(X) = 400
* Var(Y) = 100
* Cov(X,Y) = +60 originally

The **naive estimate** ignores covariance completely.

Suppose the Delta Method variance expression is:

[
\text{Variance} = \text{Naive part} - \text{Covariance contribution}
]

With **positive covariance (+60)**, that covariance term reduces the overall variance.

Now change it to:

[
Cov(X,Y)=-60
]

The sign flips.

Instead of reducing variance, covariance now **adds** to it.

If the covariance contribution was 120 in magnitude:

Naive variance:

[
500
]

Positive covariance:

[
500-120=380
]

Negative covariance:

[
500+120=620
]

So:

* **Corrected variance = 620**
* **Naive (ignoring covariance) = 500**

Therefore:

* corrected variance is **larger**
* ignoring covariance **underestimates uncertainty**

*(The exact number depends on the values in your Section 6 example, but the direction is always the same: negative covariance increases the variance of the ratio.)*

---

# 4. Why does "average session duration per user" NOT need the Delta Method, but "average clicks per session" does?

The key question is:

**What is the unit of analysis?**

### Average session duration per user

Each user contributes **one value**:

```
User A → 15 minutes
User B → 8 minutes
User C → 22 minutes
```

You simply compare two sample means.

No ratio of two random quantities is involved.

A standard t-test is sufficient.

---

### Average clicks per session

Now suppose:

User A

* Sessions = 10
* Clicks = 40

User B

* Sessions = 2
* Clicks = 6

The metric is

```
Total Clicks
-------------
Total Sessions
```

Both the numerator (clicks) and denominator (sessions) are random variables.

Because both vary from sample to sample—and they're often correlated—you need the **Delta Method** to correctly estimate the variance of the ratio.

So:

* **Session duration per user** → ordinary mean → no Delta Method.
* **Clicks per session** → ratio of two random variables → Delta Method.

---

# 5. When would you use bootstrap instead of the Delta Method, and why doesn't bootstrap require Cov(X,Y)?

Use the **bootstrap** when:

* the metric has a complicated distribution,
* the ratio is difficult to analyze mathematically,
* the sample size is modest,
* the data are highly skewed,
* or you don't trust the assumptions behind the Delta Method.

Examples include:

* Revenue per user with a few "whale" customers
* Average order value
* Complex funnel metrics
* Custom business metrics

### Why doesn't bootstrap need covariance?

The Delta Method is an **analytical approach**.

It derives the variance using formulas, so you must explicitly account for how the numerator and denominator move together (their covariance).

Bootstrap takes a different approach:

1. Resample users (with replacement) many times.
2. Recalculate the ratio metric for each resample.
3. Look at the spread of those bootstrap estimates.

Because each bootstrap sample preserves the original pairing of numerator and denominator for every observation, **their correlation is automatically reflected in the resampled data**. The variability across bootstrap estimates naturally includes the covariance effect, so you never have to calculate ( \text{Cov}(X,Y) ) yourself.


---
*
This is the perfect question to ask because it cuts straight to the *business value* of the Delta Method. 

The mathematical derivation is useless if you don't understand *when* to deploy it and *what the consequences* of ignoring it are. 

Here is the interview-ready breakdown of exactly **why**, **when**, and **what happens if you don't**.

---

### Part 1: Why do you NEED the Delta Method?

You need the Delta Method for one simple reason: **Variance doesn't "pool" the same way averages do.**

- When you compute a **simple mean** (Avg. Clicks/User), the variance is just \( Var(X) / N \). Easy. 
- But a **ratio metric** (Clicks/Session) is a *division problem*. 
  
You are dividing two random variables. **The variance of a division is NOT the division of the variances.** You cannot just take \( Var(Clicks) / Var(Sessions) \). That is mathematically illegal. 

If you want a Confidence Interval or a p-value for your ratio, you **must** figure out how the *combination* of Numerator-variance, Denominator-variance, and their relationship (Covariance) affects the final ratio. **The Delta Method is the mathematical crowbar that pries open a division problem so we can measure its variance.**

---

### Part 2: How do you USE it? (The Interview Script)

When the interviewer asks you to calculate the standard error for a ratio metric (e.g., CTR, Revenue per Session, Clicks per User), you do **NOT** dive straight into the formula. You use this 3-step script to show you understand the *process*:

**Step 1: Aggregate to the User Level First**
> *"I will not average each user's individual ratio. I will compute the pooled ratio for the treatment group as \( \bar{R} = \frac{\sum X_i}{\sum Y_i} \) (Total Clicks / Total Sessions). This correctly weights heavy users."*

**Step 2: Compute the Three Required Components**
> *"Using my user-level dataset, I will compute three things:*
> - *\( Var(X) \): How much Clicks vary across users.*
> - *\( Var(Y) \): How much Sessions vary across users.*
> - *\( Cov(X,Y) \): How much Clicks and Sessions move together (almost always positive)."*

**Step 3: Apply the Delta Method Formula**
> *"I plug these into the Delta Method formula to get the variance of the ratio, then divide by my sample size (N) to get the squared standard error:"*

\[
SE(\bar{R}) = \sqrt{ \frac{1}{N} \cdot \frac{1}{\bar{Y}^2} \left[ Var(X) + \bar{R}^2 Var(Y) - 2\bar{R} \cdot Cov(X,Y) \right] }
\]

> *"I now have a valid standard error to build my confidence interval and calculate my p-value."*

---

### Part 3: What happens if you DON'T use it? (The 3 Disaster Scenarios)

This is the "killer app" of your interview answer. Interviewers ask this to see if you understand the *practical* consequences. There are **three** distinct ways to do it wrong:

#### Disaster #1: Averaging the Individual Ratios (The Weighting Mistake)
- **What you do:** You compute each user's CTR (e.g., User A has 1 click / 1 session = 100% CTR), average all 1,000 users' CTRs, and use the standard deviation of those 1,000 ratios.
- **What happens:** A user with **1 session** gets the exact same statistical weight as a power user with **1,000 sessions**. Your variance explodes (huge standard errors), and you lose all statistical power. You will fail to detect a real 5% lift because the noise from tiny users drowns out the signal.

#### Disaster #2: Forgetting the Covariance (The "Naive Delta" Mistake)
- **What you do:** You correctly pool the ratio, but you use the *wrong* variance formula: \( Var(X) + \bar{R}^2 Var(Y) \) — completely ignoring the \( -2\bar{R}Cov(X,Y) \) term.
- **What happens:** As shown in the chapter, if Clicks and Sessions are positively correlated (which they almost always are), **your variance estimate is exactly double what it should be**. Your confidence intervals are twice as wide as necessary. You tell the PM: *"This 8% lift isn't significant."* But if you had used the Delta Method correctly, it *would* be significant. You just cost the company millions by falsely killing a winning feature.

#### Disaster #3: The "Washout" Effect (Negative Correlation)
- **What you do:** Same as #2 (forget covariance).
- **What happens:** *This is the hidden trap.* What if your metric is "Support Tickets per User"? Power users (high denominator) actually open *fewer* support tickets (low numerator). The correlation is **negative**. 
- If you forget the covariance here, the \( -2\bar{R}Cov \) term should have *added* to your variance (because negative covariance increases the wobble of a ratio). By forgetting it, you artificially **shrink** your standard error. You tell the PM: *"This feature drastically cuts support tickets, p=0.001!"* You ship it, it does nothing, and you just committed a costly Type 1 error (False Positive).

---

### Part 4: The "Lazy" Alternative (The Bootstrap)

If the interviewer pushes back and says: *"That formula is too complex for my engineers. How else can we do this?"*

**Your Answer:**
"We use the **Bootstrap** in practice. 

- I resample my users (with replacement) 10,000 times. 
- Each time, I compute the pooled ratio (Total Clicks / Total Sessions). 
- I look at the standard deviation of those 10,000 bootstrapped ratios. 
- That standard deviation *inherently* contains the covariance between Clicks and Sessions, because I resample the (Clicks, Sessions) pairs *together*. I get the exact same corrected standard error as the Delta Method, without scaring the engineers with a Taylor expansion."

---

### Part 5: The Ultimate Interview Cheat Sheet

| Scenario | What you do | The Delta Method Formula |
| :--- | :--- | :--- |
| **Metric is "Avg. Clicks per User"** | **Do NOT use Delta Method.** Denominator (1 user) is fixed. Use standard \( \sigma / \sqrt{N} \). | N/A |
| **Metric is "CTR" (Clicks/Impressions)** | **Use Delta Method.** Both numbers vary per user. | \( \frac{1}{\bar{Imp}^2}[Var(Clicks) + R^2 Var(Imp) - 2R \cdot Cov] \) |
| **Positive Correlation (Clicks ↑, Impressions ↑)** | **Mandatory.** If you forget covariance, your SE is **too big** (false negatives). | The `- 2R*Cov` term *shrinks* the variance. |
| **Negative Correlation (Revenue ↑, Returns ↑)** | **Mandatory.** If you forget covariance, your SE is **too small** (false positives). | The `- 2R*Cov` term *adds* to the variance. |
| **Manager hates math** | **Use Bootstrap.** Resample users 10k times, compute ratio each time, take SD of results. | No formula needed; the algorithm handles the covariance implicitly. |
