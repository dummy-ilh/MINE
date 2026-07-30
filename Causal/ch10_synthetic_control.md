# Chapter 10: Synthetic Control

## 1. Explanation

### The specific problem this method solves

Synthetic control addresses a situation you'll run into constantly at a company like Google: **you have exactly one (or very few) treated unit** — one country, one metro area, one product launched in only a single market — and no single available comparison unit looks like a genuinely fair match on its own. DiD (Chapter 8) handles "treatment group vs. control group over time," but if you only have one treated unit, picking *one* arbitrary comparison unit (or a simple average of several very different ones) is fragile: any single comparator might match you well on one dimension (say, population size) but badly on another (say, growth rate), and there's no principled way to decide which single comparator, if any, to trust.

### The core idea: build a synthetic comparison, don't pick a real one

Instead of choosing one real comparison unit, synthetic control constructs an **artificial comparison unit** as a *weighted combination* of several real, untreated "donor" units — chosen specifically so that this weighted blend's **pre-treatment trajectory** closely tracks the treated unit's actual pre-treatment history. The key insight: a weighted combination has far more "degrees of freedom" to match multiple dimensions of your pre-treatment pattern simultaneously than any single real comparator could. For example, "60% like Donor B's trajectory + 40% like Donor C's trajectory" might jointly reproduce your treated unit's specific historical pattern — level, trend, seasonality, whatever you're matching on — far better than either Donor B or Donor C could alone.

### The optimization, explained before the notation

You're solving: "what combination of donor units, each weighted between 0% and 100% (weights non-negative, summing to 100%), would have best reproduced my treated unit's pre-treatment history?" Once you've found that combination, you assume (this is the identifying assumption) it would have *continued* to track the treated unit equally well into the post-treatment period, had treatment not occurred — so the actual post-treatment gap between the real treated unit and this synthetic twin is your estimated treatment effect.

Formally, you choose weights $w_j \geq 0$ (for each donor $j$), summing to 1, minimizing:
```
minimize  Σ_k v_k · ( X_1k − Σ_j w_j X_jk )²
```
where $X_1$ represents the treated unit's values on various pre-treatment predictor variables $k$ (which might include several years of the pre-treatment outcome itself, plus other relevant covariates), $X_j$ are the donors' corresponding values, and $v_k$ are importance weights reflecting how much you care about matching each particular predictor well (e.g., you might weight "last year's outcome level" more heavily than matching some loosely-related demographic variable).

Once weights are found, the estimated treatment effect at each post-treatment time $t$ is simply:
```
τ_t = Y_1t − Σ_j w_j Y_jt
```
— actual observed outcome, minus the weighted-average "synthetic" counterfactual. Notice this gives you a **whole time series** of effects, not just a single number — letting you see whether the effect is immediate or delayed, growing, shrinking, or stable, which is often just as informative as the headline average effect.

### Why non-negativity and sum-to-one constraints matter

These constraints keep the synthetic control interpretable as a genuine "weighted average of real, observed units" — the synthetic twin's values stay within the range spanned by actual donors at each point in time, rather than extrapolating beyond what any real unit has shown (which an unconstrained regression, allowing negative weights or weights summing to something other than 1, could otherwise do). This is both more interpretable (you can literally say "our synthetic Metro A is 60% like Metro B and 40% like Metro C") and safer against overfitting, especially when you have many candidate donors but relatively few pre-treatment time periods to fit them against.

### Inference without a conventional "sample size": the placebo-in-space test

With typically just one treated unit, you can't compute a standard error the usual way (there's no natural notion of "repeated sampling" from a population of treated units). The standard solution: **placebo-in-space testing**. Pretend, one at a time, that each of the *donor* (untreated) units was actually the treated unit — apply the exact same synthetic control procedure to each of them, generating a "fake" placebo effect for each. Since nothing really happened to any of these donor units, their placebo effects should mostly cluster around zero. If your *real* treated unit's estimated effect is unusually large relative to this distribution of placebo effects, that's evidence the effect is genuine and not just noise inherent to the estimation procedure itself — this is conceptually a permutation-test, distribution-free approach to inference, well suited to a setting with just one true treated unit.

### The credibility check you must always report: pre-treatment fit

Since the entire method's validity rests on "the synthetic twin tracked the real unit well before treatment, so it likely would have continued to after treatment too, absent that treatment" — **poor pre-treatment fit is disqualifying**. If your synthetic control doesn't closely match the treated unit's actual history even during the period when nothing was happening, there's no reason at all to trust its extrapolated counterfactual once treatment starts. Reporting pre-treatment fit quality (visually and/or numerically) alongside the post-treatment effect is not optional — it's the load-bearing credibility check for the whole exercise.

## 2. Example

### A worked numerical with 3 donors

A ride-hailing company changes its surge-pricing algorithm in exactly one metro area (Metro A). Weekly rides (in thousands) for Metro A and 3 candidate donor metros, over the last 4 pre-treatment weeks and 2 post-treatment weeks:

| Week | Metro A (treated) | Metro B | Metro C | Metro D |
|---|---|---|---|---|
| -4 | 50 | 48 | 55 | 40 |
| -3 | 52 | 49 | 56 | 41 |
| -2 | 51 | 50 | 54 | 42 |
| -1 | 53 | 50 | 57 | 43 |
| **+1 (post)** | 60 | 51 | 58 | 44 |
| +2 | 63 | 52 | 59 | 45 |

**Step 1 — find weights** minimizing the pre-treatment gap between Metro A and the weighted donor blend. Suppose (via optimization — real implementations use quadratic programming, shown here as a plausible worked result) the best-fit weights are $w_B=0.5$, $w_C=0.3$, $w_D=0.2$ (summing to 1). Check how well this blend reproduces Metro A's actual pre-treatment history:

- Week -4: $0.5(48) + 0.3(55) + 0.2(40) = 24 + 16.5 + 8 = 48.5$ — actual Metro A = 50, gap = 1.5
- Week -3: $0.5(49) + 0.3(56) + 0.2(41) = 24.5 + 16.8 + 8.2 = 49.5$ — actual = 52, gap = 2.5
- Week -2: $0.5(50) + 0.3(54) + 0.2(42) = 25 + 16.2 + 8.4 = 49.6$ — actual = 51, gap = 1.4
- Week -1: $0.5(50) + 0.3(57) + 0.2(43) = 25 + 17.1 + 8.6 = 50.7$ — actual = 53, gap = 2.3

These gaps (1.4–2.5) are reasonably small relative to the ~50 level of the series — a decent, though not perfect, pre-treatment fit. (In a real analysis, you'd try alternative weight combinations or additional donors/predictors to see if this fit could be tightened further before proceeding.)

**Step 2 — compute the post-treatment effect**, using the *same* weights:
```
Week +1 synthetic = 0.5(51) + 0.3(58) + 0.2(44) = 25.5 + 17.4 + 8.8 = 51.7
Actual Metro A = 60.   τ₊₁ = 60 − 51.7 = 8.3

Week +2 synthetic = 0.5(52) + 0.3(59) + 0.2(45) = 26 + 17.7 + 9.0 = 52.7
Actual Metro A = 63.   τ₊₂ = 63 − 52.7 = 10.3
```

**Interpretation**: the new surge-pricing algorithm is estimated to have increased weekly rides in Metro A by about **8,300** in the first post-treatment week, growing to about **10,300** by the second week. The fact that the gap is *growing* rather than staying flat or shrinking suggests the effect may still be building — worth watching in subsequent weeks before concluding the effect has fully stabilized (compare this to Chapter 3's discussion of novelty effects — you'd want to distinguish a genuinely growing effect from a transient spike that later fades).

**Placebo check, briefly**: to gain confidence in this 8.3–10.3 effect, you'd re-run the identical procedure treating Metro B, then Metro C, then Metro D as if each were "treated" (even though nothing happened to them), generating a small distribution of placebo gaps. If those placebo gaps are all much smaller than Metro A's 8.3–10.3 range, that's supportive evidence the real effect isn't just estimation noise.

## 3. Interview Q&A

**Q: Why do you need positive, sum-to-one weight constraints on the donor pool rather than just running an unconstrained regression of the treated unit's outcome on all donors?**
A: The constraints (weights ≥0, summing to 1) keep the synthetic control as an interpretable "weighted average" or convex combination of real, observed units — it stays within the range of actually-observed data at each time point, with no extrapolation beyond what any real donor showed. This is both more interpretable and avoids the overfitting/extrapolation risk that an unconstrained regression (potentially with many donors and relatively few pre-periods) would be prone to.

**Q: If your synthetic control fits the pre-treatment trend poorly (large gaps well before treatment), should you still report a post-treatment effect?**
A: Be very cautious — poor pre-treatment fit undermines the core justification for trusting the post-treatment gap as causal. If the synthetic twin didn't track the real unit well even *before* treatment (when nothing should have been different), there's no reason to trust it as a valid counterfactual *after* treatment starts. I'd either seek a better donor pool or additional predictors to improve the fit first, or be explicit that the analysis isn't reliable enough to support a strong causal claim as currently constructed.

**Q: Explain the placebo-in-space inference procedure and what result would make you doubt your finding.**
A: Re-run synthetic control treating each *donor* unit as if it were the treated one (pretending a change happened there when it didn't), generating a distribution of placebo "effects." If several of these placebo effects are as large as (or larger than) the real treated unit's estimated effect, that undermines confidence that the real result reflects a genuine causal impact rather than noise inherent to the estimation method/data — you want your real effect to be a clear outlier relative to this placebo distribution, not just another point within it.

**Q: How does synthetic control's identifying assumption differ conceptually from DiD's parallel trends assumption?**
A: DiD assumes a single (or simply-averaged) control group would have trended in parallel absent treatment — an assumption you can partially eyeball with a pre-trends plot, but can't fully verify, and which relies on an essentially arbitrary choice of comparator(s). Synthetic control makes a conceptually similar assumption but supports it with *much stronger, more direct empirical evidence*: it's explicitly optimized to match the entire pre-treatment trajectory (potentially across multiple predictor variables, not just one outcome series), so you have direct, visible confirmation of how well the counterfactual tracked history before you ever extrapolate it forward.

**Q: Your donor pool includes a metro that itself experienced a major, unrelated pricing change during your post-treatment window. What do you do?**
A: Exclude it (or heavily reconsider its inclusion) from the donor pool — if a donor unit experiences its own confounding shock during the post-treatment period, its outcome trajectory no longer represents a valid "untreated" counterfactual component during that window, and including it would contaminate the synthetic control's post-treatment comparison, even if it happened to fit the pre-treatment period reasonably well.

**Q: In the worked example, why does a growing post-treatment gap (8.3 in week 1, 10.3 in week 2) deserve more scrutiny than a flat, constant gap would?**
A: A flat, immediately-appearing, stable gap is more consistent with a clean, one-time causal shift that has fully manifested. A *growing* gap is ambiguous — it could reflect a genuinely compounding effect (e.g., word-of-mouth adoption of a pricing change), or it could be an early sign of a novelty/momentum effect that hasn't yet peaked, or even a sign that the synthetic control's fit is starting to drift for reasons unrelated to the treatment (e.g., a donor-specific shock beginning to emerge). More post-treatment weeks of data would help distinguish between these explanations.

**Q: If you had a genuinely comparable single control region available, would you still prefer synthetic control over simple DiD?**
A: Not necessarily — if there really is one region that's a strong, well-justified match on both level and trend, DiD with that comparator can be simpler, more transparent, and easier to communicate. Synthetic control earns its extra complexity specifically when no single comparator is convincing on its own, or when you want the added, visible credibility of an explicitly-optimized pre-treatment fit rather than relying on a qualitative "these two regions seem similar" argument.

---
**Previous: Chapter 9 — Regression Discontinuity Design**
**Next: Chapter 11 — Interference and SUTVA Violations**
