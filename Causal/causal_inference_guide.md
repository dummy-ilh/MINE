# Causal Inference — End to End
### For Google MLE / Data Scientist interviews

---

## 0. Roadmap

```
Foundations → RCT → PSM/IPW → Diff-in-Diff → IV → RDD → Interference/SUTVA → Synthetic Control
   (why)      (gold standard)         (no randomization, but you can approximate it)
```

Every method below is answering one question: **"How do I build a credible stand-in
for the counterfactual — what would have happened without treatment — when I can't
just randomize?"**

---

## 1. Foundations

### 1.1 The fundamental problem

For any unit $i$ (a user, a store, a country), define two **potential outcomes**:

| Symbol | Meaning |
|---|---|
| $Y_i(1)$ | outcome unit $i$ **would have** had *if treated* |
| $Y_i(0)$ | outcome unit $i$ **would have** had *if not treated* |
| $\tau_i = Y_i(1) - Y_i(0)$ | individual treatment effect |
| $D_i \in \{0,1\}$ | actual treatment received |
| $Y_i = D_i Y_i(1) + (1-D_i) Y_i(0)$ | the outcome you actually **observe** |

You only ever observe **one** of $Y_i(1), Y_i(0)$ per unit — never both. That's the
fundamental problem of causal inference. Everything else is a workaround.

Two estimands you'll be asked to distinguish in interviews:

$$ATE = E[Y_i(1) - Y_i(0)] \quad \text{(effect on everyone)}$$
$$ATT = E[Y_i(1) - Y_i(0) \mid D_i = 1] \quad \text{(effect on those who actually got treated)}$$

They differ whenever treatment effects vary across people (heterogeneous effects) —
e.g. a discount code helps price-sensitive users more, and price-sensitive users are
also more likely to have clicked it. ATE and ATT will diverge there.

### 1.2 Why naive comparison is biased

The naive estimator is just $E[Y \mid D=1] - E[Y \mid D=0]$. Decompose it:

$$
\underbrace{E[Y|D=1]-E[Y|D=0]}_{\text{naive comparison}}
= \underbrace{E[Y(1)-Y(0)|D=1]}_{ATT} \;+\;
\underbrace{\big(E[Y(0)|D=1]-E[Y(0)|D=0]\big)}_{\text{selection bias}}
$$

The selection bias term is nonzero whenever the treated and untreated groups would
have looked different **even without treatment**. This is the whole ballgame — every
method below is a different trick for making that term zero (or measuring/removing it).

### 1.3 Confounding — ASCII picture

```
   OBSERVATIONAL DATA                    RANDOMIZED EXPERIMENT
   (confounder biases everything)        (coin flip severs the bad arrow)

           Z                                      Z
      (confounder)                          (confounder)
        /      \                              /       \
       /        \                        - - -         \
      v          v                      (cut by         v
      X -------> Y                       randomization)  Y
   (treatment) (outcome)                       X -------> Y
                                          (treatment)   (outcome)

   Z -> X : who gets treated is not random   Z -> X severed: coin flip ignores Z
   Z -> Y : Z also drives the outcome        Z -> Y : still there, but doesn't matter
   Result: X-Y correlation is contaminated   Result: X-Y correlation is the true effect
```

Example: Z = "health consciousness." It pushes people toward taking vitamins (X) AND
toward being healthier (Y) regardless of vitamins. Naive comparison of vitamin-takers
vs non-takers overstates the vitamin effect — classic confounding.

### 1.4 SUTVA (Stable Unit Treatment Value Assumption)

Two parts, and Google interviewers love probing this because Google products are
**networked/marketplace** systems where SUTVA routinely breaks:

1. **No interference**: my treatment assignment doesn't affect your outcome.
   (Breaks in: social networks, two-sided marketplaces, ads auctions, ride-sharing
   supply/demand — see Section 7.)
2. **No hidden variation of treatment**: "treated" means the same thing for everyone.
   (Breaks if e.g. "saw the new UI" varies by device/bandwidth so it's really many
   different treatments lumped together.)

### 1.5 Worked numerical — selection bias in action

10 users tried a paid feature (D=1), 10 didn't (D=0). Suppose the TRUE per-user
effect is +5 revenue for everyone (homogeneous), but users who opted in were already
higher-spending (confounded by "engagement level" Z).

| Group | Avg $Y(0)$ (baseline, hypothetical) | Avg $Y(1)$ (if treated) | Observed Y |
|---|---|---|---|
| Treated (D=1) | 50 (would've spent this even untreated) | 55 | 55 |
| Untreated (D=0) | 30 | 35 (hypothetical, unobserved) | 30 |

Naive estimate: $55 - 30 = 25$.
True ATT: $E[Y(1)-Y(0)|D=1] = 55 - 50 = 5$.
Selection bias: $E[Y(0)|D=1] - E[Y(0)|D=0] = 50 - 30 = 20$.
Check: $25 = 5 + 20$. ✓ — the naive number is inflated 5x by pre-existing differences.

**This is the single most important number to internalize**: naive comparison =
true effect + selection bias, and selection bias is invisible unless you do something
clever.

---

## 2. Randomized Controlled Trial (the gold standard, briefly)

Randomization forces $E[Y(0)|D=1] = E[Y(0)|D=0]$ (treated and control are exchangeable
in expectation), so selection bias → 0 and:

$$\hat{ATE} = \bar{Y}_{D=1} - \bar{Y}_{D=0}$$

is unbiased for the ATE. Standard error via the usual two-sample formula:

$$SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}$$

**Numerical**: 1000 users randomized 50/50 into a new recommender. Treatment
conversion 8.4% ($s=0.277$), control 7.6% ($s=0.265$).

$$SE = \sqrt{\frac{0.277^2}{500}+\frac{0.265^2}{500}} \approx 0.0170$$

Effect $= 0.084-0.076 = 0.008$, z = 0.008/0.017 ≈ 0.47 → **not significant**. This is
the everyday A/B test — the baseline every other method is trying to approximate
when you *can't* randomize.

**When RCTs aren't enough at Google-scale**: interference (Section 7) — e.g.
randomizing "show ad" per-user in an auction still lets treated and control users
compete for the same ad inventory, contaminating control.

---

## 3. Propensity Score Matching (PSM) & Inverse Propensity Weighting (IPW)

### 3.1 Intuition

You can't randomize, but you *can* measure the confounders (age, past spend, device,
tenure...). Idea: find, for every treated user, an untreated "twin" who looked
identical on those confounders *before* treatment. Compare outcomes within twin pairs
— any Z-driven bias cancels because twins share the same Z.

Doing this on many covariates at once (curse of dimensionality) is hard. Rosenbaum &
Rubin's trick: you don't need to match on every covariate — matching on the single
**propensity score** $e(X) = P(D=1|X)$ is sufficient (if unconfoundedness holds).

```
        Raw covariates X (age, spend, tenure, device...)
                         |
                 collapse via
             e(X) = P(D=1 | X)     <- one number per user
                         |
        match treated <-> control with similar e(X)
                         |
        compare outcomes within matched pairs
```

### 3.2 Key assumption: unconfoundedness (a.k.a. ignorability)

$$Y(0), Y(1) \perp D \mid X$$

"Once you condition on the observed covariates X, treatment is as good as random."
This is **untestable** — it fails if there's an *unmeasured* confounder (e.g. "user
motivation" that you didn't log). This is the #1 thing interviewers probe: *"what if
there's an unobserved confounder?"* → PSM has no defense; you need IV or RDD instead.

Also requires **overlap/common support**: $0 < e(X) < 1$ for all X — every covariate
profile must have *some* chance of being treated and *some* chance of being control.
If some segment is always treated (e(X)=1), you can't find a twin for them.

### 3.3 Formulas

**Estimate propensity score** via logistic regression:
$$e(X_i) = P(D_i=1|X_i) = \frac{1}{1+e^{-(\beta_0+\beta_1 X_{i1}+\dots)}}$$

**IPW estimator for ATE** (weight each unit by inverse probability of the treatment
they actually got):

$$\hat{ATE}_{IPW} = \frac{1}{n}\sum_i \left[\frac{D_i Y_i}{e(X_i)} - \frac{(1-D_i) Y_i}{1-e(X_i)}\right]$$

Interpretation: a treated user who had only a 10% chance of being treated ($e=0.1$)
is "surprising" — up-weight them 10x, because they're standing in for many similar
untreated-in-expectation users. A treated user who had a 90% chance is unsurprising,
barely upweighted.

### 3.4 Worked numerical

6 users. X = "high engagement" (1) or "low" (0). D = got the feature. Y = revenue.

| User | X | D | Y | e(X) (estimated: P(D=1\|X=1)=0.75, P(D=1\|X=0)=0.25) |
|---|---|---|---|---|
| 1 | 1 | 1 | 60 | 0.75 |
| 2 | 1 | 1 | 55 | 0.75 |
| 3 | 1 | 0 | 50 | 0.75 |
| 4 | 0 | 1 | 40 | 0.25 |
| 5 | 0 | 0 | 20 | 0.25 |
| 6 | 0 | 0 | 25 | 0.25 |

Naive: mean(D=1) − mean(D=0) = (60+55+40)/3 − (50+20+25)/3 = 51.67 − 31.67 = **20.0**
(inflated — high-X users dominate the treated group).

IPW:
$$\hat{ATE} = \frac1 6\Big[\tfrac{60}{0.75}+\tfrac{55}{0.75}+\tfrac{40}{0.25} - \tfrac{50}{0.25}-\tfrac{20}{0.75}-\tfrac{25}{0.75}\Big]$$

Term by term: $80+73.3+160 - 200-26.7-33.3 = 313.3 - 260 = 53.3$; divide by 6 = **8.9**.

Compare to matching directly: pair user 1&2 (X=1,D=1) against user 3 (X=1,D=0) →
diff ≈ 9.5; pair user 4 (X=0,D=1) against users 5&6 (X=0,D=0) → diff = 40-22.5 = 17.5.
Weighted combo lands near IPW's ~9-13 range — much closer to a plausible true effect
than the naive 20.0. **The whole point: adjusting for X cut the estimate roughly in
half** because X was confounding the naive comparison.

### 3.5 Diagnostics (PSM-specific failure modes)

1. **Poor overlap** — histogram of $e(X)$ for treated vs control barely overlaps.
   Symptom: extreme weights (some $e(X)$ near 0 or 1) blow up variance. Fix: trim
   to common support, or use overlap weights instead of IPW.
2. **Covariate imbalance after matching** — check standardized mean difference (SMD)
   on each covariate post-match; should be < 0.1. If still imbalanced, the
   propensity model is misspecified (missing interactions/nonlinearity).
3. **Omitted confounder** — no diagnostic catches this from the data alone; use
   domain knowledge + sensitivity analysis (Rosenbaum bounds / E-value) to ask "how
   strong would an unmeasured confounder need to be to explain away the effect?"
4. **Positivity violation near the tails** — e.g. e(X)=0.98 means only 2% chance of
   being control; that one control unit is doing enormous work if matched there.

### 3.6 Q&A — PSM/IPW

**Q1 (easy).** Why match on the propensity score instead of raw covariates directly?
<details><summary>Answer</summary>Dimensionality — matching exactly on 10+ covariates leaves almost no exact matches. The propensity score is a *balancing score*: conditioning on this one number is (under unconfoundedness) as good as conditioning on all of X, per Rosenbaum-Rubin theorem.</details>

**Q2 (medium).** A colleague matches on post-treatment covariates (e.g. "number of
sessions after launch"). What's wrong?
<details><summary>Answer</summary>Post-treatment variables can themselves be caused by treatment — conditioning on them opens a collider/mediator path and biases the estimate (post-treatment bias). Only match on pre-treatment covariates.</details>

**Q3 (medium, calculation).** If e(X)=0.05 for a treated unit, what's its IPW weight, and why is that dangerous?
<details><summary>Answer</summary>Weight = 1/0.05 = 20 — that single unit counts as 20 units, dominating the estimate's variance. Dangerous because one noisy outcome can swing the whole ATE. This is why practitioners trim or use stabilized weights.</details>

**Q4 (hard).** How would you detect that unconfoundedness is violated without an RCT to compare against?
<details><summary>Answer</summary>You can't prove it, but you can (a) run a placebo test on a pre-treatment "pseudo-outcome" that should show zero effect if unconfoundedness holds, (b) do sensitivity analysis (Rosenbaum bounds, E-values) to quantify how strong a hidden confounder would need to be, (c) check if adding more covariates changes the estimate a lot — instability suggests remaining confounding.</details>

**Q5 (hard, Google-style).** You're estimating the effect of "opted into push notifications" on retention using PSM. Why is this a particularly bad candidate for PSM?
<details><summary>Answer</summary>Opt-in is a strong self-selected behavior almost certainly driven by unmeasured motivation/engagement that also drives retention — a textbook unmeasured confounder. PSM assumes you've measured everything that matters; self-selected behavioral treatments rarely satisfy that. Better: find a natural experiment (e.g. a randomized nudge to opt in) and use IV, or run an actual holdout experiment on the nudge itself.</details>

---

## 4. Difference-in-Differences (DiD)

### 4.1 Intuition

Used when a treatment (policy, feature launch) hits one group at one point in time,
and you have a comparable group that didn't get it. Instead of comparing levels
(which are confounded by pre-existing differences), compare **changes over time**.
The untreated group's change is your estimate of what "would have happened anyway."

```
  Outcome
    |                                    * Treated (actual)
    |                                  *
    |                                *
    |                    *  ------>*    <- gap = effect
    |                  *  (predicted, using
    |                *    control's trend)
    |          *   *
    |       *  o------o Control (actual)
    |    *  o
    | *  o
    +--------------------|------------------------> time
                       treatment
                        starts
    *=treated group actual, o=control group actual
    dashed = counterfactual (what treated WOULD have done)
```

### 4.2 The critical assumption: parallel trends

$$E[Y(0)_{treated,post} - Y(0)_{treated,pre}] = E[Y(0)_{control,post} - Y(0)_{control,pre}]$$

In words: absent treatment, the treated group's outcome *would have moved in
parallel* with the control group's. **Untestable in the post period** (that's exactly
the counterfactual we don't observe) — but you CAN and MUST check it visually /
statistically in the **pre-period** (do the two lines move together before
treatment?). This is the #1 thing to bring up unprompted in a DiD interview question.

### 4.3 Formula

Simple 2x2 (two groups, two periods) estimator:

$$\hat{\tau}_{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{ctrl,post}-\bar{Y}_{ctrl,pre})$$

Equivalently, run this regression (this generalizes to multiple periods/units and
gives you standard errors for free):

$$Y_{it} = \alpha + \beta \cdot \text{Treated}_i + \gamma \cdot \text{Post}_t + \delta \cdot (\text{Treated}_i \times \text{Post}_t) + \varepsilon_{it}$$

$\delta$ is the DiD estimate. Term by term: $\alpha$ = control-pre baseline, $\beta$ =
fixed level gap between groups (differences that exist even without treatment — DiD
allows this!), $\gamma$ = time trend common to everyone, $\delta$ = the *extra* jump
the treated group got, over and above its own baseline gap and the common trend.
That's the causal effect.

### 4.4 Worked numerical

A city launches a promo (treatment). Compare avg weekly orders vs a similar city
with no promo.

| | Pre-period | Post-period | Change |
|---|---|---|---|
| Treated city | 100 | 130 | +30 |
| Control city | 90 | 105 | +15 |

$$\hat{\tau}_{DiD} = 30 - 15 = 15$$

Interpretation: the treated city grew by 30, but 15 of that growth would have
happened anyway (seasonality, general trend — captured by the control city). The
**causal effect of the promo is +15 orders/week**, not +30.

Regression check: $\alpha=90$ (control pre), $\beta=10$ (treated city already runs
10 higher even before promo — 100-90), $\gamma=15$ (control's own change, the common
trend), $\delta=15$ (treated's *extra* change beyond trend) → predicted treated-post
$= 90+10+15+15=130$ ✓.

### 4.5 Diagnostics

1. **Pre-trend divergence** — plot both groups' outcome over several pre-periods;
   if they weren't parallel before treatment, DiD is invalid. Fix: find a better
   control group, or use synthetic control (Section 8).
2. **Anticipation effects** — units change behavior *before* the policy officially
   starts (e.g. users hear about a price hike coming and stock up). Symptom: a jump
   right before the "Post" cutoff. Fix: redefine treatment timing or drop the
   anticipation window.
3. **Composition changes** — if the treated/control "groups" aren't the same units
   over time (e.g. different users enter/leave each period), changes may reflect
   compositional shifts, not treatment. Fix: use a balanced panel of the same units.
4. **Staggered rollout bias** (the hot topic in modern econometrics /
   Goodman-Bacon 2021) — if different units get treated at different times and you
   naively run one two-way fixed effects regression, already-treated units can act
   as "control" for later-treated units, contaminating estimates, especially with
   heterogeneous effects. Fix: use Callaway-Sant'Anna or staggered-DiD estimators.

### 4.6 Q&A — DiD

**Q1 (easy).** Why not just compare treated-post to control-post directly?
<details><summary>Answer</summary>That's just the naive/cross-sectional comparison — it's biased by any pre-existing level differences between the groups (the β term above). DiD nets out fixed group-level differences by differencing.</details>

**Q2 (medium).** How do you check parallel trends in practice?
<details><summary>Answer</summary>Plot both groups' outcomes over several pre-periods and visually inspect; formally, run an event-study regression with leads (interaction terms for each pre-period) and test that they're jointly insignificant (no differential pre-trend).</details>

**Q3 (medium, calculation).** Treated store: pre=200, post=260. Control store: pre=180, post=216. What's the DiD estimate, and does it suggest the treatment worked?
<details><summary>Answer</summary>Treated change = +60. Control change = +36. DiD = 60−36 = 24. Yes, positive effect of 24 units beyond the common trend (control grew 20%, treated grew 30% — a 10pp excess that translates to 24 in level terms... actually compute directly: 24 is the answer via the level-difference formula above).</details>

**Q4 (hard).** Google launches a UI change only in the US, and you want to use "rest of world" as control. What's the biggest threat to validity here?
<details><summary>Answer</summary>Parallel trends is unlikely — the US market has different seasonality, competitor dynamics, and macro trends (e.g. a US-specific news event, holiday) than "rest of world." A more comparable control (similar English-speaking markets, or a synthetic control built as a weighted combination of countries that historically tracked the US) would be more credible. Always show the pre-trend plot.</details>

**Q5 (hard).** You have 50 stores that adopt a new POS system at different months over 2 years (staggered rollout). Why is a standard two-way-fixed-effects DiD regression risky here?
<details><summary>Answer</summary>With staggered adoption and heterogeneous treatment effects over time, already-treated units effectively serve as comparisons for later-adopting units in the standard TWFE estimator, and if effects change over time (e.g. grow), this can produce negative weights and a badly biased — even sign-flipped — overall estimate (Goodman-Bacon decomposition). Use modern staggered-DiD estimators (Callaway–Sant'Anna, Sun-Abraham) instead.</details>

---

## 5. Instrumental Variables (IV)

### 5.1 Intuition

Use this when treatment is confounded by something you **cannot measure** (so PSM is
out), but you can find a variable Z — an "instrument" — that:

1. **Relevance**: actually moves treatment D (Z → D is a real, ideally strong, link).
2. **Exclusion restriction**: affects the outcome Y **only through** D — no direct
   arrow Z → Y, and no back door through unmeasured confounders.

Think of Z as a *random nudge* toward treatment that has nothing to do with the
confounder. You then only look at the variation in D that was "caused" by this
clean, exogenous nudge — filtering out the messy self-selected variation.

```
     U (unmeasured confounder, e.g. "ambition")
         \                    \
          v                    v
  Z ----> D  ---------------> Y
(instrument) (treatment)   (outcome)

 Z->D: relevance (instrument must move treatment)
 Z->Y: MUST NOT EXIST DIRECTLY (exclusion restriction)
 Z-/->U: instrument must be unrelated to the confounder
```

Classic Google-relevant example: estimating the effect of "years of app usage" on
spending, where usage is confounded by underlying interest. Instrument: a random A/B
test that nudged some users toward the app (e.g. a promo email sent to a random
subset) — the email is unrelated to inherent interest, but shifts usage.

### 5.2 Formula — Wald estimator (binary instrument, binary treatment)

$$\hat{\tau}_{IV} = \frac{E[Y|Z=1]-E[Y|Z=0]}{E[D|Z=1]-E[D|Z=0]} = \frac{\text{reduced-form effect of Z on Y}}{\text{first-stage effect of Z on D}}$$

This estimates the **LATE** (Local Average Treatment Effect) — the effect only for
"compliers": people whose treatment status was actually changed by the instrument.
Not the ATE for everyone — an important distinction interviewers probe.

General case (2-Stage Least Squares, 2SLS):
- **Stage 1**: regress $D_i = \pi_0 + \pi_1 Z_i + \epsilon_i$, get fitted $\hat D_i$.
- **Stage 2**: regress $Y_i = \beta_0 + \beta_1 \hat D_i + u_i$. $\beta_1$ is the IV estimate.

### 5.3 Worked numerical

Randomized promo email (Z) nudges app usage (D = "used app 5+ days/week").
Outcome Y = monthly spend ($).

| | Z=1 (got email) | Z=0 (no email) |
|---|---|---|
| P(D=1) | 0.40 | 0.20 |
| Avg Y | 45 | 39 |

First stage: $E[D|Z=1]-E[D|Z=0] = 0.40-0.20 = 0.20$ (the email moved 20pp of users
into heavy usage — this is the "compliance rate").

Reduced form: $E[Y|Z=1]-E[Y|Z=0] = 45-39=6$.

$$\hat{\tau}_{IV} = \frac{6}{0.20} = 30$$

Interpretation: among the ~20% of users who were "compliers" (used the app heavily
*because* of the email, not otherwise), heavy usage caused **+$30/month** spend.
Note this can't just be read off the raw 6-point gap — you have to scale up by how
much the instrument actually moved treatment (the "first stage"). A weak first stage
(say 2% instead of 20%) would blow this estimate up to 300 — huge variance, a red flag.

### 5.4 Diagnostics

1. **Weak instrument** — check the first-stage F-statistic; rule of thumb F > 10.
   Weak instruments (Z barely moves D) produce huge-variance, badly biased IV
   estimates even with large samples. This is the #1 IV pitfall to mention.
2. **Exclusion restriction violation** — untestable directly (like parallel trends),
   defend with domain logic: "does the email have ANY other channel to affect
   spending besides pushing app usage?" (e.g. if the email also contained a discount
   code, exclusion is violated — the code affects spend directly).
3. **Non-compliance heterogeneity** — LATE only applies to compliers; if compliers
   differ systematically from the full population (e.g. only price-insensitive users
   respond to nudges), LATE ≠ ATE, and that's a genuine limitation to state, not a bug
   to "fix."

### 5.5 Q&A — IV

**Q1 (easy).** What are the two core assumptions an instrument must satisfy?
<details><summary>Answer</summary>Relevance (Z meaningfully predicts D) and exclusion restriction (Z affects Y only through D, no direct effect and no shared confounding with Y).</details>

**Q2 (medium, calculation).** First stage effect = 0.05, reduced form effect = 2.5. What's the IV estimate, and what's your first concern?
<details><summary>Answer</summary>IV = 2.5/0.05 = 50. Concern: 0.05 is a weak first stage — small denominators amplify noise, so this estimate likely has huge standard errors; check the F-stat before trusting it.</details>

**Q3 (medium).** Why does IV estimate LATE and not ATE?
<details><summary>Answer</summary>IV only identifies the effect for units whose treatment status the instrument actually changed (compliers) — it says nothing about "always-takers" (treated regardless of Z) or "never-takers" (never treated regardless of Z), since their behavior doesn't respond to the instrument.</details>

**Q4 (hard).** Product team proposes using "distance to nearest Google office" as an instrument for "works in tech" to study effect on product usage. Critique this.
<details><summary>Answer</summary>Likely violates exclusion restriction — distance to a tech office correlates with urbanicity, income, education, all of which independently affect product usage through channels other than "works in tech." A valid instrument needs a causal story where the ONLY path to Y is through D.</details>

**Q5 (hard, Google-style).** How would you use a randomized notification experiment as an instrument to estimate the causal effect of "session length" on ad revenue, when session length itself is confounded by user interest?
<details><summary>Answer</summary>Randomly assign a subset of users to receive a notification nudge (Z) known to increase session length but with no direct link to ad revenue except via engagement. Use Z as instrument: first stage = effect of notification on session length; reduced form = effect of notification on ad revenue; IV estimate = ratio. Caveat: only valid if notification doesn't also change ad load, ad relevance, or user mood in ways that hit revenue directly — a real risk to flag (e.g. if the notification IS an ad itself, exclusion breaks).</details>

---

## 6. Regression Discontinuity Design (RDD)

### 6.1 Intuition

Treatment is assigned by a sharp rule based on a running variable crossing a cutoff
(credit score ≥ 700 → approved; test score ≥ 70 → scholarship; account age ≥ 30 days
→ eligible for a feature). Right at the cutoff, units just above and just below are
essentially random with respect to everything else — nobody can perfectly manipulate
which side of 699.99 vs 700.01 they land on. Compare outcomes in a tiny window
around the cutoff.

```
   Outcome Y
      |                              o o
      |                          o o    o
      |                       o          <- treated side (jump!)
      |                    o
      |     - - - - - - - -|  <- discontinuity = causal effect
      |                 x  |
      |              x  x
      |           x  x
      |        x                 <- control side
      +-----------------------|------------------> Running variable X
                            cutoff c
   x = untreated (X<c), o = treated (X>=c)
```

### 6.2 Formula

Local comparison right at the cutoff $c$:

$$\tau_{RDD} = \lim_{x \to c^+} E[Y|X=x] - \lim_{x \to c^-} E[Y|X=x]$$

In practice, estimated via local linear regression on each side of the cutoff, within
a bandwidth $h$:

$$Y_i = \alpha + \beta (X_i - c) + \tau \cdot \mathbb{1}[X_i \geq c] + \gamma (X_i-c)\cdot\mathbb{1}[X_i\geq c] + \varepsilon_i, \quad |X_i - c| < h$$

$\tau$ is the jump at the cutoff = the causal effect, *local to units near the cutoff
only* (another LATE-like caveat — doesn't tell you the effect for units far from c).

**Sharp RDD**: treatment is a deterministic function of X (100% compliance with the
rule). **Fuzzy RDD**: crossing the cutoff only shifts the *probability* of treatment
(e.g. score≥700 makes approval likely but not certain) — then you effectively run
RDD as an instrument (IV) for treatment, dividing the Y-jump by the D-jump, same
Wald-ratio logic as Section 5.

### 6.3 Worked numerical

Users with account age ≥ 30 days become eligible for a loyalty badge (sharp cutoff).
Outcome = 90-day spend. Local linear fits just each side of cutoff=30:

| Side | Intercept at cutoff (fitted) |
|---|---|
| Just below 30 days (control) | $52 |
| Just above 30 days (treated) | $61 |

$$\hat{\tau}_{RDD} = 61 - 52 = 9$$

The loyalty badge causes **+$9** in 90-day spend, *for users near the 30-day
threshold* — this says nothing about the effect for a 5-day-old or 300-day-old
account; those are extrapolations, not what RDD identifies.

### 6.4 Diagnostics

1. **Manipulation of the running variable** — check the density of X around the
   cutoff (McCrary density test); a suspicious pile-up just above the cutoff (e.g.
   scores of exactly 700 far more common than 699 or 701) suggests people are
   gaming the rule, which breaks local randomization.
2. **Bandwidth sensitivity** — re-run with several bandwidths ($h$); if $\hat\tau$
   swings wildly, the result isn't robust. Use data-driven optimal bandwidth
   selection (Imbens-Kalyanaraman / Calonico-Cattaneo-Titiunik).
3. **Covariate discontinuity check** — pre-treatment covariates that shouldn't jump
   at the cutoff (e.g. gender, signup country) should show NO discontinuity; if they
   do, something else is changing at the cutoff too, confounding the design.
4. **Functional form misspecification** — fitting a straight line when the true
   relationship is curved can manufacture a fake "jump." Always plot binned averages
   as a visual check, and try polynomial/local-nonparametric fits as robustness.

### 6.5 Q&A — RDD

**Q1 (easy).** Why does RDD give a causal estimate right at the cutoff even though the sample isn't randomized overall?
<details><summary>Answer</summary>Because whether a unit lands at X=699.9 vs X=700.1 is essentially arbitrary/noise-driven (nobody can precisely control being just barely above/below), so locally the two sides are "as good as randomized," even though the full sample of high-X vs low-X people obviously differs.</details>

**Q2 (medium).** Difference between sharp and fuzzy RDD?
<details><summary>Answer</summary>Sharp: crossing the cutoff perfectly determines treatment (100% compliance). Fuzzy: crossing the cutoff only shifts the probability of treatment; estimated as an IV problem using cutoff-crossing as the instrument.</details>

**Q3 (medium, calculation).** Fuzzy RDD: jump in P(treated) at cutoff = 0.30. Jump in Y at cutoff = 12. What's the local treatment effect?
<details><summary>Answer</summary>12 / 0.30 = 40, the LATE for compliers near the cutoff (Wald-ratio logic, same as IV).</details>

**Q4 (hard).** Google runs an internal RDD around "score ≥ 700 gets loan approval" using a partner bank's data, and you notice a big spike in applicants with score exactly 700. What do you do?
<details><summary>Answer</summary>This is a manipulation red flag — likely loan officers or applicants are gaming/rounding scores to just clear the cutoff. Run the McCrary density test to confirm the spike is statistically anomalous, and if confirmed, RDD validity is compromised; consider a donut-hole approach (exclude observations very close to the cutoff) or find an unmanipulable running variable.</details>

**Q5 (hard).** Why can't you extrapolate an RDD estimate to "the effect of loan approval on everyone," not just people near the cutoff?
<details><summary>Answer</summary>RDD only uses local variation near the cutoff, so it identifies a Local ATE for units whose score is near 700. People with scores of 300 or 950 may have completely different responses to approval (heterogeneous treatment effects) that the design has no information about — extrapolating requires an additional (untested) assumption of constant effects.</details>

---

## 7. Interference / SUTVA Violations (highly relevant to Google)

### 7.1 Why this matters specifically for Google

Google products are **networks and marketplaces**: Search results affect other
results shown (ad auction competition), YouTube recommendations affect the whole
content ecosystem, ride-share/marketplace-style products (Search ads, Play Store)
have supply-demand interactions, and social products have direct peer effects. In
all these, treating one unit changes outcomes for "control" units too — violating
SUTVA's "no interference" clause and biasing standard A/B tests.

```
  STANDARD (WRONG) ASSUMPTION           REALITY WITH INTERFERENCE

  Treated user's outcome                Treated user's outcome
       depends only on                       depends on
      their OWN assignment                their own AND
                                          their neighbors'/market's
                                              assignment

     T1  T2  C1  C2                       T1--C1  (friends: T1's post
     (independent)                          reaches C1's feed too!)
                                          T2 <-> C2 (compete for same
                                                     ad slot/inventory)
```

### 7.2 Common interference patterns and fixes

| Pattern | Example | Fix |
|---|---|---|
| **Network/social spillover** | Treated user posts more → friends (control) see more content | **Cluster/graph randomization**: randomize whole friend-clusters together, not individuals |
| **Market/supply-demand** | Show more ads to treated riders → less driver supply for control riders | **Switchback experiments**: randomize by *time* (whole market gets treatment for an hour, then control) instead of by unit |
| **Budget/auction competition** | Treated ad campaign bids up prices, hurting control campaigns | **Geo-based randomization**: randomize by region so treated/control don't compete in the same auction |
| **General equilibrium / congestion** | New checkout flow speeds up treated users → server capacity frees up → control users also get faster response | Cluster or geo randomization; or model spillover explicitly |

### 7.3 Formula sketch — cluster randomization bias/variance tradeoff

If you ignore interference and just randomize individuals, your naive estimate is
biased by the **spillover effect** onto controls:

$$\hat{\tau}_{naive} = \underbrace{E[Y(1)|D=1]-E[Y(0)|D=0]}_{\text{includes both direct AND spillover onto controls}}$$

If control units are partially "contaminated" by treated neighbors (say control
outcomes rise by spillover amount $s$), the naive estimate becomes:

$$\hat{\tau}_{naive} \approx \tau_{direct} - s$$

i.e. it **understates** the true direct effect, because the "control" group isn't a
clean counterfactual anymore — it's already partially treated via spillover.

### 7.4 Worked numerical — switchback experiment

A ride-share pricing algorithm is tested via switchback design: alternate 2-hour
blocks between old pricing (A) and new pricing (B) in the same city (avoids
supply-demand contamination across arms since only one policy runs at a time
citywide).

| Block | Policy | Avg driver utilization |
|---|---|---|
| 8-10am | A | 62% |
| 10-12pm | B | 68% |
| 12-2pm | A | 60% |
| 2-4pm | B | 70% |

Naive within-city comparison isn't confounded by cross-arm market competition
(only one policy live at a time), but time-of-day is now a confounder — compare
matched time-blocks or use a regression controlling for hour-of-day fixed effects:

$$\hat\tau \approx \frac{(68-62)+(70-60)}{2} = \frac{6+10}{2}=8pp \text{ utilization lift}$$

(In practice you'd difference against a longer baseline history for each hour slot,
not just one A/B pair, to net out day-specific noise — but this shows the core idea:
switching the *whole market* avoids within-market contamination.)

### 7.5 Diagnostics

1. **Check for "dose-response" by neighborhood/cluster saturation** — deliberately
   run a saturation experiment (randomize the % of a cluster treated: 0%, 25%, 50%,
   100%) and see if control-cluster outcomes shift as treated-% rises. If they do,
   you've confirmed and can even quantify interference.
2. **Compare individual-randomized vs cluster-randomized estimates** — a big gap
   between the two signals interference contaminated the individual-level design.
3. **Falsification via geo-holdouts** — hold out an entire untouched region as a
   secondary control; large divergence from the "official" individual-level control
   group indicates spillover.

### 7.6 Q&A — Interference/SUTVA

**Q1 (easy).** Why does a standard A/B test on Search ad ranking risk violating SUTVA?
<details><summary>Answer</summary>Ads compete in a shared auction with limited inventory and shared advertiser budgets — treating some users changes which ads/how much budget is available for the "control" users too, so control isn't a clean untreated baseline.</details>

**Q2 (medium).** What's a switchback experiment, and why does it help with marketplace interference?
<details><summary>Answer</summary>It randomizes treatment across TIME rather than across units, so the entire market runs one policy at a time — eliminating cross-contamination between treated and control units competing for the same supply within the same time window. Trade-off: time itself becomes a confounder (time-of-day, day-of-week effects) that must be controlled for.</details>

**Q3 (medium).** How would you detect that your "control" group in a social feature test is contaminated?
<details><summary>Answer</summary>Run a saturation/dosage experiment: vary the fraction of each cluster (e.g. friend-group) that's treated, and check whether control users' outcomes trend with the treated-share in their cluster. A significant trend confirms spillover.</details>

**Q4 (hard).** You're evaluating a new YouTube recommendation algorithm with individual-level randomization. Explain a plausible bias direction and how you'd fix it.
<details><summary>Answer</summary>If the new algorithm surfaces more of certain creators' content, it can change the overall content supply/competition dynamics (creators shift strategy, or aggregate watch-time shifts affecting recommendations for everyone via shared collaborative-filtering signals) — control users' feeds indirectly change too, likely biasing the measured lift DOWNWARD (control partially "gets" the treatment effect via shared signal). Fix: cluster-randomize by content/creator community, or run at the geo level, or use a long-run holdout of untouched users/regions entirely isolated from the treated recommendation model's training data.</details>

**Q5 (hard).** Design an experiment to measure the causal effect of a new ranking algorithm on ad revenue for Google Search, accounting for auction interference.
<details><summary>Answer</summary>Use geo-based randomization (e.g. randomize by DMA/region) rather than individual users, since ad auctions and advertiser budgets are shared within a market but largely independent across geographies. Within each geo, apply the full algorithm to avoid partial-contamination effects. Analyze at the geo level (few dozen to ~100 units) using difference-in-differences or synthetic control against historical geo trends, since the "N" is now geos, not users, which also changes your power calculation.</details>

---

## 8. Synthetic Control & Other Quasi-Experiments

### 8.1 Intuition

Extension of DiD for the case of **one treated unit** (one country, one city, one
market) with no single good control — instead, build an artificial control as a
**weighted combination of several untreated units** that best reproduces the treated
unit's pre-treatment trajectory.

```
   Outcome
     |                                actual treated (post)
     |                              *
     |                            *      <- gap = effect
     |                          *  ` ` ` `  synthetic control (post, counterfactual)
     |                        *  o
     |                      *  o
     |         (pre-period: synthetic tracks
     |          actual treated almost exactly)
     |    *o *o *o *o *o
     +------------------------|-------------------> time
                            treatment starts
     * = actual treated unit,  o = synthetic control (weighted avg of donor pool)
```

### 8.2 Formula

Choose weights $w_j \geq 0$, $\sum w_j = 1$ over donor (untreated) units $j$ to
minimize pre-treatment fit:

$$\min_{w} \sum_{t \in \text{pre}} \left(Y_{1t} - \sum_j w_j Y_{jt}\right)^2$$

Then the effect post-treatment:

$$\hat\tau_t = Y_{1t} - \sum_j w_j Y_{jt}, \quad t \in \text{post}$$

Inference is done via **placebo tests**: run the same procedure pretending each
*untreated* donor unit was "treated" and see how large a gap you get by chance — if
your real treated unit's gap is much bigger than the placebo gaps, that's evidence
of a real effect (this is the standard way to get a p-value-like statistic since
there's only N=1 treated unit, so no closed-form SE).

### 8.3 Worked numerical (simplified)

One state raises minimum wage; 3 neighboring states didn't. Synthetic control weights
chosen to best match pre-period employment trend: $w_A=0.5, w_B=0.3, w_C=0.2$.

Post-treatment: actual state employment = 94 (index). Synthetic $=0.5(97)+0.3(96)+0.2(98) = 48.5+28.8+19.6=96.9$.

$$\hat\tau = 94 - 96.9 = -2.9$$

Interpretation: employment is 2.9 index points lower than the synthetic
counterfactual — suggestive of a negative employment effect, *if* placebo tests on
the 3 donor states show gaps much smaller than 2.9 in magnitude (otherwise -2.9 could
just be noise).

### 8.4 Diagnostics

1. **Poor pre-treatment fit** — if the synthetic control can't closely track the
   treated unit's pre-period trend, don't trust the post-period gap; the whole method
   depends on a good pre-fit.
2. **Donor pool contamination** — a "control" state that was ALSO affected by a
   related policy shouldn't be in the donor pool (violates the idea of a clean
   counterfactual).
3. **In-space placebo test** — as described above; if the real unit's post-treatment
   gap isn't much bigger than placebo gaps from untreated donors, effect isn't
   distinguishable from noise.

### 8.5 Q&A — Synthetic Control

**Q1 (easy).** When would you reach for synthetic control instead of DiD?
<details><summary>Answer</summary>When there's a single treated unit (one market/city/country) and no single comparable control — synthetic control builds a data-driven weighted composite control from a donor pool instead of relying on one arbitrary comparison unit.</details>

**Q2 (medium).** How do you get a "p-value" for a synthetic control estimate with only one treated unit?
<details><summary>Answer</summary>Placebo-in-space test: re-run the whole synthetic control procedure treating each donor unit as if it were the treated one, generate a distribution of "fake" post-pre gaps, and see where the real treated unit's gap ranks — a rank-based p-value (e.g. 1/20 if it's the most extreme of 20 placebo runs).</details>

**Q3 (hard).** Google wants to know the incremental effect of a marketing campaign that launched in one country only. What's your synthetic control approach and biggest risk?
<details><summary>Answer</summary>Build synthetic control from a donor pool of comparable countries' pre-campaign metrics (search interest, revenue trend, etc.), weight to best match pre-period trajectory, measure post-campaign gap. Biggest risk: donor pool countries getting hit by a correlated shock (e.g. a global product update, macro event) at the same time, which would masquerade as campaign effect — must verify no contemporaneous confounding events in the donor countries.</details>

---

## 9. Master Cheat Sheet — Decision Tree

```
                    Can you randomize treatment?
                        /                  \
                     YES                    NO
                      |                      |
                RCT (Section 2)      Can you measure ALL
              simple diff-in-means    confounders?
                                       /            \
                                    YES              NO
                                     |                |
                            PSM / IPW         Is there a policy that
                            (Section 3)       hit one group at one
                                               point in time, with a
                                               plausible parallel-trend
                                               control?
                                                /            \
                                             YES              NO
                                              |                |
                                        Diff-in-Diff    Is treatment assigned
                                        (Section 4)     by a sharp threshold
                                        (1 treated unit  rule on some running
                                         -> Synthetic     variable?
                                         Control, Sec 8)      /        \
                                                            YES          NO
                                                             |            |
                                                    Regression      Do you have a
                                                    Discontinuity   variable that shifts
                                                    (Section 6)     treatment but has no
                                                                    direct effect on Y?
                                                                        /        \
                                                                     YES          NO
                                                                      |            |
                                                              Instrumental    You likely can't
                                                              Variables       identify a causal
                                                              (Section 5)     effect credibly —
                                                                              say so explicitly

  AT EVERY NODE: also ask "do units interact with each other?"
  (network/marketplace/auction) -> if yes, SUTVA is violated regardless
  of which branch above you're on -> go to Section 7 (cluster/geo/switchback
  randomization) before trusting any of the above.
```

## 10. Cross-Cutting Interview Q&A (the "gotcha" round)

**Q1.** What's the difference between correlation, causation, and prediction, in one sentence each?
<details><summary>Answer</summary>Correlation: X and Y move together in observed data. Causation: intervening on X changes Y (holding everything else fixed). Prediction: X is useful for forecasting Y even if not causal (e.g. umbrella sales predict rain, don't cause it).</details>

**Q2.** Give an example where a variable should NOT be controlled for even though it's correlated with the outcome.
<details><summary>Answer</summary>A "collider" or a post-treatment variable/mediator. E.g. controlling for "number of app opens after seeing the ad" when estimating ad effect on purchases — app opens are caused by the ad (mediator), so controlling for it blocks part of the very effect you're trying to measure and can even reverse the sign.</details>

**Q3.** You find a huge, statistically significant DiD effect. What's your first instinct before believing it?
<details><summary>Answer</summary>Check pre-trends. A large effect with divergent pre-trends is much more likely a pre-existing trend difference than a treatment effect — this is the single most common way DiD is misused.</details>

**Q4.** Explain, without formulas, why an RCT can still be biased.
<details><summary>Answer</summary>Randomization guarantees no confounding on average, but doesn't fix interference/SUTVA violations (spillover between treated and control), differential attrition (control drops out of the study at a different rate than treatment, biasing who's left to compare), or non-compliance (people assigned to treatment don't take it).</details>

**Q5.** A PM says "we ran an A/B test and it was flat, so the feature has no effect — let's use observational data with propensity matching instead to find the 'real' effect." What's wrong with this reasoning?
<details><summary>Answer</summary>A null RCT result is strong causal evidence of no (or a small) effect; switching to a less rigorous observational method specifically because it might show a nonzero (and possibly spurious, confounded) effect is p-hacking / methodology shopping — the RCT should be trusted more, not less, and the PM should ask about power (was the test big/long enough to detect a plausible effect size) rather than abandon the gold-standard result.</details>

**Q6.** What is the "exclusion restriction" and why can it never be tested with data?
<details><summary>Answer</summary>It requires the instrument to affect the outcome ONLY through the treatment, with no other path — but any "other path" would necessarily run through something unobserved (by definition, if it were observed you'd control for it directly), so no dataset can rule it out; it's defended by domain-knowledge argument, not statistics.</details>

**Q7 (calculation).** RCT: treatment n=200, mean=5.2, sd=2.1. Control n=200, mean=4.8, sd=2.0. Compute the effect and its z-stat.
<details><summary>Answer</summary>Effect = 5.2−4.8 = 0.4. SE = sqrt(2.1²/200 + 2.0²/200) = sqrt(0.02205+0.02) = sqrt(0.04205) ≈ 0.205. z = 0.4/0.205 ≈ 1.95 — borderline significant at 5% (just under/around the ~1.96 threshold, worth flagging as marginal, not a clean win).</details>

**Q8 (hard, systems-design style).** Design an end-to-end causal measurement strategy for "does showing personalized recommendations increase watch time on YouTube," accounting for confounding, interference, and long-run effects.
<details><summary>Answer</summary>Start with an RCT (individual or geo-randomized depending on interference risk from shared recommendation-model training signals). Use geo/cluster randomization if the recommendation model is retrained on pooled treated+control behavior (a subtle interference channel — the model itself "leaks" treatment effect into control via shared training data). Run long enough to capture novelty-effect decay (short-run lift can overstate long-run effect — check trend over the experiment duration, not just endpoint). Use a holdback/holdout group even after full launch to keep measuring long-run causal effect. Complement with DiD/synthetic control at the launch-market level as a secondary robustness check outside the RCT (e.g. before/after national rollout vs. countries with delayed rollout).</details>

---

## Appendix: Formula Quick Reference

| Method | Core formula |
|---|---|
| ATE / ATT | $E[Y(1)-Y(0)]$ / $E[Y(1)-Y(0)\|D=1]$ |
| Naive bias decomp | naive = ATT + selection bias |
| RCT | $\bar Y_{D=1}-\bar Y_{D=0}$, $SE=\sqrt{s_1^2/n_1+s_0^2/n_0}$ |
| IPW | $\frac1n\sum[\frac{D_iY_i}{e(X_i)}-\frac{(1-D_i)Y_i}{1-e(X_i)}]$ |
| DiD | $(\bar Y_{t,post}-\bar Y_{t,pre})-(\bar Y_{c,post}-\bar Y_{c,pre})$ |
| IV (Wald) | $\frac{E[Y\|Z=1]-E[Y\|Z=0]}{E[D\|Z=1]-E[D\|Z=0]}$ |
| RDD (sharp) | $\lim_{x\to c^+}E[Y\|X=x]-\lim_{x\to c^-}E[Y\|X=x]$ |
| RDD (fuzzy) | Y-jump at cutoff / D-jump at cutoff |
| Synthetic control | $Y_{1t}-\sum_j w_jY_{jt}$, weights fit on pre-period |
