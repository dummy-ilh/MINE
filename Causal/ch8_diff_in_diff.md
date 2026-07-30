# Chapter 8: Difference-in-Differences (DiD)

## 1. Explanation

### The situation this method is built for

DiD is designed for a very common real-world setup: you have a **treatment group** and a **comparison group**, and you observe both **before and after** some intervention — but you're worried the two groups might have been different in some *fixed* way even before treatment, so a simple post-only comparison would be confounded by that pre-existing difference.

### Building the intuition step by step

Consider three naive things you could compute, and see why each one alone is contaminated:

1. **Treated-after minus treated-before** (a simple before/after comparison within the treatment group): this tells you the *total* change experienced by the treatment group — but it mixes together the treatment effect with anything else that changed over that same time period for everyone (seasonality, macro trends, an unrelated concurrent product change, a holiday). You can't tell how much of this change is the treatment versus just "time passing."

2. **Treated-after minus control-after** (a post-only, cross-sectional comparison): this tells you the *level gap* between groups after treatment — but it mixes the treatment effect with any *pre-existing* difference between the groups that was there even before treatment started (e.g., the treatment group was already a stronger or weaker market).

3. **DiD subtracts these two contaminating pieces simultaneously**: it takes "how much did the treatment group change over time" and subtracts "how much did the control group change over time," on the logic that whatever common time-trend affected both groups equally (seasonality, macro shocks, unrelated company-wide changes) will show up in *both* of these changes and cancel out when you subtract, leaving only the treatment-specific, extra change.

### The formula, and what each piece means

```
DiD = [ Ȳ_treat,after − Ȳ_treat,before ] − [ Ȳ_control,after − Ȳ_control,before ]
```

Equivalently, and more useful for real analyses with many periods/units, as a regression:
```
Y_it = α + β·Treat_i + γ·Post_t + δ·(Treat_i × Post_t) + ε_it
```
- **α**: the baseline average (control group, pre-period) — the reference point everything else is measured against.
- **β**: the fixed, constant level-gap between treatment and control groups, present *even before* treatment started. DiD explicitly *allows* this — it does not require the two groups to start at the same level, only that any gap between them stays stable over time absent treatment.
- **γ**: the common time trend/shock — whatever changed for both groups between the pre- and post-periods, regardless of treatment.
- **δ**: the Treat×Post interaction — **this is your causal effect estimate**. It captures the *extra* change experienced by the treatment group, beyond whatever common trend γ already accounts for.

### The identifying assumption: Parallel Trends

The entire "cancellation" logic above is only valid if, **absent treatment**, the treatment group's outcome would have moved in *the same way over time* as the control group's did:
```
E[Y(0)_after − Y(0)_before | Treat=1] = E[Y(0)_after − Y(0)_before | Treat=0]
```
Notice this is fundamentally a claim about an unobserved counterfactual — what the treatment group's trend *would have been* had they not been treated — so no single number from your data can ever fully prove it. It is exactly as untestable, in principle, as ignorability was for regression/PSM methods; DiD simply substitutes a different, but equally unverifiable, structural assumption.

### How you build confidence in parallel trends anyway (since you can never prove it)

The standard practice is an **event-study / pre-trends check**: instead of collapsing to a single "before" period, plot the treatment-minus-control gap across *several* pre-treatment periods. If that gap is flat and stable across multiple pre-periods, and only jumps right when treatment starts, that's supportive (though never conclusive) evidence for parallel trends. If the gap was already drifting apart *before* treatment began, DiD is likely to give a biased answer — you'd be crediting the treatment for a divergence that was already happening on its own.

### Threats to DiD, beyond parallel trends itself

- **Anticipation effects**: units change their behavior *before* the treatment officially starts, because they know it's coming (e.g., firms adjusting hiring right before an announced minimum-wage hike takes effect) — this can contaminate what you're calling the "pre-period," making the true pre-trend look different from what actually would have happened.
- **Composition changes**: if your data is repeated cross-sections rather than the same tracked units over time, a shift in *who's* in each group (different users, different customers) between periods can masquerade as a treatment effect.
- **Staggered adoption pitfalls**: when treatment starts at different times for different units (e.g., a feature rolled out region-by-region over many months) and treatment effects are heterogeneous or change over time, a naive standard two-way fixed-effects regression can be **badly biased** — a modern, well-documented econometrics finding. The core issue: the standard estimator implicitly uses *already-treated* units from earlier-adopting regions as part of the comparison group for later-adopting regions. If those early adopters' own treatment effect is still evolving, this comparison is contaminated. Modern estimators (Callaway-Sant'Anna, Sun-Abraham decompositions) explicitly avoid using already-treated units as controls.

## 2. Example

### Example A — The classic 2x2 case

A city (treatment) raises minimum wage; a neighboring, similar city (control) does not. Employment (Y, in thousands):

|  | Before | After |
|---|---|---|
| Treatment city | 100 | 96 |
| Control city | 90 | 88 |

```
Δ_treat = 96 − 100 = −4
Δ_control = 88 − 90 = −2
DiD = Δ_treat − Δ_control = −4 − (−2) = −2
```
**Interpretation**: the minimum wage increase is associated with a **2,000-job decrease** relative to what would have happened absent the policy (using the control city's trend as the stand-in for the treatment city's counterfactual trend). Note that a naive before-after comparison in the treatment city alone (−4, i.e., −4,000 jobs) would have wrongly attributed the control city's general downward trend (−2, e.g., a broader regional economic softening affecting both cities) entirely to the minimum wage policy — DiD correctly nets that shared trend out.

### Example B — Multi-period, with an explicit pre-trends check (going beyond the simple 2x2)

Google rolls out a redesigned checkout flow (D) in Region A only, starting Week 5. Weekly conversion rate (%), Region A vs Region B (control):

| Week | Region A | Region B |
|---|---|---|
| 1 | 10.0 | 8.0 |
| 2 | 10.2 | 8.1 |
| 3 | 10.1 | 8.2 |
| 4 | 10.3 | 8.0 |
| **5 (launch)** | 12.5 | 8.3 |
| 6 | 12.8 | 8.1 |
| 7 | 13.0 | 8.4 |

**Pre-trend check (weeks 1-4).** Compute the treatment-minus-control gap in each pre-period week: Week 1: 10.0−8.0=2.0; Week 2: 10.2−8.1=2.1; Week 3: 10.1−8.2=1.9; Week 4: 10.3−8.0=2.3. The gap hovers around ~2.0-2.1 with minor noise and no clear drift — this is reasonably supportive of parallel trends (a small upward wobble to 2.3 in week 4 is worth noting but is much smaller than what happens right after launch).

**DiD estimate**, comparing pre-period averages (weeks 1-4) to post-period averages (weeks 5-7):
```
Region A: pre avg = (10.0+10.2+10.1+10.3)/4 = 10.15;  post avg = (12.5+12.8+13.0)/3 = 12.77
Region B: pre avg = (8.0+8.1+8.2+8.0)/4 = 8.075;       post avg = (8.3+8.1+8.4)/3 = 8.267

Δ_A = 12.77 − 10.15 = 2.62
Δ_B = 8.267 − 8.075 = 0.192

DiD = 2.62 − 0.192 = 2.43 percentage points
```
**Interpretation**: the checkout redesign is estimated to have increased conversion by about **2.43 percentage points**, net of whatever small common trend both regions experienced. Because the pre-trends check looked reasonably flat, this estimate carries more credibility than it would if the pre-period gap had been visibly drifting.

## 3. Interview Q&A

**Q: Explain, using the regression form, exactly which coefficient captures the causal effect and why the *other* coefficients are still important to include.**
A: δ (the Treat×Post interaction) is the causal effect. β (the Treat main effect) matters because it absorbs any fixed, time-invariant difference between the two groups (e.g., Region A always converts somewhat better for unrelated reasons) — without it, you'd wrongly attribute this constant gap to the treatment. γ (the Post main effect) absorbs any common time trend/shock hitting both groups equally — without it, seasonality or macro shifts would contaminate δ.

**Q: Your pre-trends plot shows the treatment group was already trending upward relative to control for the 3 periods before launch. What does this mean for your DiD estimate, and what would you do?**
A: It's evidence against parallel trends — some of the "post-launch" divergence may just be a continuation of a pre-existing trend, not a treatment effect, meaning your DiD estimate is likely biased upward (overstating the true effect). Options: extend the pre-period to better understand and potentially model/extrapolate the trend explicitly, use a trend-adjusted DiD specification, or find a different comparison group with a flatter, better-matched pre-trend — or consider synthetic control (Chapter 10), which explicitly optimizes for pre-trend matching rather than assuming it.

**Q: What's a "placebo test" in the DiD context, and how would you run one on the checkout-flow example?**
A: Pretend the treatment happened at an earlier, fake date (e.g., week 3 instead of week 5), using only pre-period data, and re-run the DiD estimator on that fake split. Since nothing actually happened at week 3, you should find a null (near-zero, non-significant) "placebo effect." Finding a significant effect at a fake treatment date would indicate your design has some other source of bias unrelated to the real treatment, undermining confidence in the true post-launch estimate.

**Q: If treatment rolled out to different regions at different times (staggered adoption) instead of all-at-once, what's the specific technical danger of naively running a standard two-way fixed-effects DiD regression?**
A: With staggered timing and treatment effects that vary over time (e.g., growing or shrinking post-launch), the standard two-way FE estimator implicitly uses *already-treated* units (from earlier-adopting regions) as part of the comparison group for later-adopting regions — these "already treated" comparisons are contaminated by the earlier group's own still-evolving treatment effect, and the overall estimate can be severely biased, sometimes even the wrong sign, relative to the true average effect. Modern estimators (Callaway-Sant'Anna, Sun-Abraham) explicitly restrict comparisons to not-yet-treated or never-treated units to avoid this contamination.

**Q: Does DiD require the treatment and control groups to have similar outcome *levels* before treatment?**
A: No — DiD explicitly allows a fixed, constant level difference (captured by the Treat main-effect coefficient β); what it requires is that this gap stays constant *over time* absent treatment (parallel *trends*, not parallel *levels*). This is a frequently tested distinction, since intuitively people assume DiD needs the groups to "look the same" to start, which isn't correct.

**Q: Suppose instead of one control region, you have five candidate control regions with quite different pre-trends from each other. How might you decide which to use, or whether to use a simple average of all five?**
A: I'd check each candidate's pre-trend against the treatment region individually, favoring whichever region (or subset) shows the flattest, most stable gap over multiple pre-periods — a simple average across very different regions can obscure the fact that some are poor matches while others are good ones. If no single region or simple average looks convincing, this is exactly the situation synthetic control (Chapter 10) is built to handle, since it can construct an optimally-weighted blend rather than relying on an ad hoc choice of comparison group(s).

**Q: A colleague argues "our pre-trends look flat, so parallel trends definitely holds." What's the flaw in this reasoning?**
A: Flat pre-trends are *supportive evidence*, not proof — parallel trends is fundamentally a claim about an unobserved counterfactual (what would have happened without treatment), and no amount of pre-treatment data can directly verify a claim about the post-treatment period. It's entirely possible for pre-trends to look flat purely by chance, or for something to change concurrently with treatment (a confounding shock coinciding with the treatment date) that a pre-trends check by its nature cannot detect, since it only looks at the period before that shock occurred.

---
**Previous: Chapter 7 — Instrumental Variables**
**Next: Chapter 9 — Regression Discontinuity Design**
