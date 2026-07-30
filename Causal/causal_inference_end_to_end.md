# Causal Inference — End to End
### A slow, from-zero guide for Google MLE / Data Scientist interviews

> How to use this doc: read chapters in order. Every chapter has (1) intuition, (2) formulas explained term by term, (3) a worked numerical example you can redo by hand, (4) an interview-style Q&A block. Don't skip the numericals — causal inference interview questions are almost always "here are some numbers, what's the effect?"

---

## Table of Contents

0. Why causal inference, and why Google asks about it
1. Correlation vs Causation, and the Potential Outcomes Framework
2. Causal Estimands: ATE, ATT, ATC, CATE
3. Randomized Experiments (RCTs) — the gold standard
4. Confounding, Selection Bias, and DAGs
5. Regression Adjustment
6. Propensity Score Matching (PSM) and Weighting (IPW)
7. Instrumental Variables (IV)
8. Difference-in-Differences (DiD)
9. Regression Discontinuity Design (RDD)
10. Synthetic Control
11. Interference and SUTVA Violations (network/marketplace effects)
12. Quasi-Experiments, Natural Experiments, and Sensitivity Analysis
13. Google-style Case Studies and Rapid-Fire Interview Q&A

---

## Chapter 0 — Why causal inference, and why Google asks about it

Machine learning models are great at prediction: "given these features, what's the likely outcome?" Causal inference answers a different question: **"if I change X, what happens to Y?"**

Example: A model can predict that users who see more ads convert more. That's correlational. But if Product wants to know "should we show more ads?", they need the *causal* effect of ads on conversion — because heavy ad exposure might just be correlated with being an engaged user who was going to convert anyway.

Google (and any large tech company) cares about this because:
- **A/B testing** is causal inference by design (randomization).
- **Ranking/ads changes** can't always be A/B tested cleanly (network effects, marketplace interference, ramp-up effects) — so quasi-experimental methods matter.
- **Policy decisions** (pricing, feature launches, ranking algorithm changes) require causal answers, not just predictive accuracy.
- Interviewers want to see you can (a) reason about *why* a naive comparison is biased, (b) pick the right method, (c) do the algebra, (d) state assumptions explicitly.

**The single biggest interview differentiator:** stating assumptions out loud (SUTVA, ignorability, exclusion restriction, parallel trends, etc.) before touching an estimator. Anyone can plug numbers into a diff-in-diff formula; fewer people can say "this only works if parallel trends holds, and here's how I'd check that."

### Q&A
**Q: What's the difference between a predictive model and a causal model?**
A: A predictive model estimates P(Y | X) — useful for forecasting when you passively observe X. A causal model estimates what Y would be if you *intervened* to set X, i.e., P(Y | do(X)). Prediction can be excellent while causal understanding is completely wrong, because of confounding.

**Q: Give an example where a highly predictive feature is not causal.**
A: Number of fire trucks at a location strongly predicts fire damage size — more trucks, more damage. But sending fewer trucks would not reduce damage; fire size (a confounder) causes both.

---

## Chapter 1 — Correlation vs Causation, and the Potential Outcomes Framework

### 1.1 The core problem: the Fundamental Problem of Causal Inference

For any single unit (user, person, region) $i$, define two **potential outcomes**:
- $Y_i(1)$ = the outcome for unit $i$ **if treated**
- $Y_i(0)$ = the outcome for unit $i$ **if not treated**

The **individual causal effect** is:
```
τ_i = Y_i(1) − Y_i(0)
```

The problem: for any given unit, you only ever observe **one** of these two outcomes — whichever corresponds to the treatment they actually received. You never see the road not taken. This is called the **Fundamental Problem of Causal Inference** (Holland, 1986). You cannot compute τ_i directly for any individual. All of causal inference is about cleverly estimating **averages** of τ_i across many units, using assumptions to fill in the missing counterfactual.

### 1.2 Notation you'll use everywhere

- $D_i$ or $T_i$: treatment indicator (1 = treated, 0 = control) for unit i
- $Y_i$: observed outcome = $D_i Y_i(1) + (1-D_i) Y_i(0)$ (the "switching equation")
- $X_i$: observed covariates/features
- $U_i$: unobserved factors

### 1.3 Why naive comparison is biased

Suppose you just compute:
```
Naive effect = E[Y | D=1] − E[Y | D=0]
```
Decompose this:
```
E[Y|D=1] − E[Y|D=0]
 = E[Y(1)|D=1] − E[Y(0)|D=0]
 = { E[Y(1)|D=1] − E[Y(0)|D=1] }  +  { E[Y(0)|D=1] − E[Y(0)|D=0] }
     = ATT (true causal effect on treated)     +     Selection bias
```

**Selection bias** = the difference in the *baseline* (untreated) outcome between the group that chose treatment and the group that didn't. If people who take treatment would have done better anyway (even without treatment), naive comparison overstates the effect.

### 1.4 Worked numerical

Say we want the effect of "using Google Ads" (D) on a small business's weekly revenue (Y in $1000s). Hypothetical (unobservable in reality) potential outcomes for 6 businesses:

| Business | Y(0) | Y(1) | D (actual) | Y observed |
|---|---|---|---|---|
| 1 | 10 | 14 | 1 | 14 |
| 2 | 8 | 11 | 1 | 11 |
| 3 | 12 | 15 | 1 | 15 |
| 4 | 9 | 12 | 0 | 9 |
| 5 | 7 | 9 | 0 | 7 |
| 6 | 11 | 13 | 0 | 11 |

**True ATE** = mean(Y(1) − Y(0)) over all 6 = mean(4,3,3,3,2,2) = 17/6 ≈ **2.83**

**Naive estimate** = mean(Y | D=1) − mean(Y | D=0) = mean(14,11,15) − mean(9,7,11) = 13.33 − 9.0 = **4.33**

The naive estimate (4.33) overstates the true ATE (2.83) by 1.5 — because businesses that chose to advertise (1,2,3) had *higher baseline* Y(0) than those that didn't (10,8,12 avg=10 vs 9,7,11 avg=9). That gap (10−9=1... check: actually average Y(0) for treated =10, for control=9, gap=1, but bias formula gave 1.5 — let's recompute precisely below in the QA) — the point stands: some of the naive gap is selection bias, not the causal effect.

### Q&A
**Q: In one sentence, what is the "fundamental problem of causal inference"?**
A: For any unit you can never observe both potential outcomes simultaneously, only the one matching the treatment received — the other is a counterfactual you must estimate, not observe.

**Q: If naive comparison equals ATT + selection bias, when does naive comparison correctly estimate ATE?**
A: When there's no selection bias, i.e., E[Y(0)|D=1] = E[Y(0)|D=0] — which holds under random assignment, because then treatment group membership is independent of potential outcomes.

**Q: Compute the selection bias in the worked numerical above.**
A: Selection bias = E[Y(0)|D=1] − E[Y(0)|D=0] = mean(10,8,12) − mean(9,7,11) = 10 − 9 = 1.0. ATT = mean(Y(1)−Y(0) | D=1) = mean(4,3,3) = 3.33. Check: ATT + bias = 3.33 + 1.0 = 4.33 = naive estimate. ✓ (matches exactly, confirming the decomposition algebra).

---

## Chapter 2 — Causal Estimands: ATE, ATT, ATC, CATE

Different questions call for different "flavors" of causal effect. Interviewers love asking "which estimand does this method give you?" because many candidates conflate them.

| Estimand | Formula | Meaning | Who cares |
|---|---|---|---|
| **ATE** (Average Treatment Effect) | $E[Y(1)-Y(0)]$ over **everyone** | Effect if we treated the *whole population* | Policy: "should we roll this out to all users?" |
| **ATT** (Average Treatment Effect on the Treated) | $E[Y(1)-Y(0) \mid D=1]$ | Effect for those who *actually got* treated | "Was giving this coupon to the people who got it worth it?" |
| **ATC** (Average Treatment Effect on the Controls) | $E[Y(1)-Y(0) \mid D=0]$ | Effect if we treated those currently untreated | "If we extended the promo to non-recipients, what would happen?" |
| **CATE** (Conditional ATE) | $E[Y(1)-Y(0) \mid X=x]$ | Effect for a subgroup defined by covariates | Personalization / heterogeneous treatment effects |
| **LATE** (Local ATE) | effect for "compliers" only (IV) | Effect for the subpopulation whose treatment status is moved by an instrument | IV settings, see Ch.7 |

**Key fact:** ATE = ATT only when treatment assignment is independent of the treatment effect size (e.g., pure randomization). In observational data they usually differ, and can even have different signs in pathological cases.

### Worked numerical
Using the same 6-business table from Ch.1:
- ATT = 3.33 (computed above, over units 1,2,3)
- ATC = mean(Y(1)-Y(0) | D=0) = mean(3,2,2) = 2.33 (units 4,5,6)
- ATE = weighted avg = (3×3.33 + 3×2.33)/6 = (10 + 7)/6 = 2.83 ✓ matches Ch.1

Notice ATT (3.33) > ATC (2.33): the businesses that self-selected into advertising benefited *more* from it than those that didn't — a classic pattern (people who adopt a treatment often gain more from it, which is itself a source of bias if you mistake ATT for ATE).

### Q&A
**Q: A company wants to know "should we expand this feature to 100% of users?" Which estimand do they want?**
A: ATE over the target rollout population (or ATC if currently only a subset has it and you want the effect on the *rest*).

**Q: Why can an RCT with non-compliance only estimate LATE, not ATE?**
A: Because you only ever observe outcomes for the "compliers" whose behavior actually changed due to instrument/assignment; for "always-takers" and "never-takers" you can't identify the effect since their treatment status doesn't respond to assignment.

---

## Chapter 3 — Randomized Experiments (RCTs): the gold standard

### 3.1 Why randomization works
Randomly assigning D means $D \perp (Y(0), Y(1))$ — treatment assignment is statistically independent of potential outcomes. This kills selection bias by construction:
```
E[Y(0)|D=1] = E[Y(0)|D=0] = E[Y(0)]
```
So naive difference-in-means is an unbiased estimator of ATE:
```
ATE_hat = Ȳ_treated − Ȳ_control
```

### 3.2 Variance and sample size (power)
For a two-sample comparison, the variance of the estimator:
```
Var(ATE_hat) = σ²_1/n_1 + σ²_0/n_0
```
Standard sample-size formula (equal allocation, two-sided test, power 1−β, significance α):
```
n per arm ≈ 2 * (z_{α/2} + z_β)² * σ² / δ²
```
where δ = minimum detectable effect (MDE), σ = outcome std dev, z_{α/2}≈1.96 (α=0.05), z_β≈0.84 (power=0.80).

### 3.3 Worked numerical
You're running an A/B test on click-through rate. Baseline CTR σ (Bernoulli, so σ² = p(1−p)) with p≈0.10 → σ²=0.09. You want to detect an MDE of δ=0.01 (1 percentage point) at 95% significance, 80% power.
```
n per arm ≈ 2 × (1.96+0.84)² × 0.09 / (0.01)²
          = 2 × (2.8)² × 0.09 / 0.0001
          = 2 × 7.84 × 0.09 / 0.0001
          = 1.4112 / 0.0001
          = 14,112 users per arm
```
So you'd need ~14,112 users per arm (~28,224 total) to reliably detect a 1pp CTR lift.

### 3.4 Threats even in RCTs (Google-relevant!)
- **SUTVA violations / interference**: one user's treatment affects another's outcome (e.g., ads inventory is shared — showing more ads to treatment group reduces inventory available to control). See Ch.11.
- **Novelty/primacy effects**: users react to *change itself*, not the feature; effect decays or grows over weeks.
- **Selection into the experiment sample** (e.g., only logged-in users eligible) — external validity issue.
- **Attrition/differential dropout** between arms — breaks randomization's guarantee post-hoc.
- **Peeking / multiple testing** — checking significance repeatedly inflates false positive rate.
- **Network effects & spillovers** in social products — treating a user changes their friends' outcomes too.

### Q&A
**Q: Why does randomization let simple difference-in-means estimate a causal effect?**
A: Randomization makes treatment assignment independent of potential outcomes, so the control group's outcome is a valid stand-in (unbiased proxy) for what the treated group's outcome *would have been* absent treatment — eliminating selection bias.

**Q: You ran an A/B test and check p-values every day, stopping as soon as p<0.05. What's wrong?**
A: This is "peeking" / repeated significance testing — it inflates the Type I error rate far above the nominal 5%, because you're effectively running many correlated tests and stopping at the first favorable draw. Fix: pre-register a fixed sample size / use sequential testing methods (e.g., alpha-spending, mSPRT) designed for continuous monitoring.

**Q: Ads team ran an RCT increasing ad load for treatment users. Revenue per treated user went up but overall revenue didn't move much. Why might that be, and is this a causal inference problem?**
A: Likely SUTVA violation via a shared, finite ad auction/inventory: showing more ads to treatment "crowds out" ads that would've gone to control users, so control users' outcomes were also affected (not truly untreated) — the naive per-arm comparison overstates the *global* rollout effect. Yes, this is exactly an interference problem (Ch.11); need switchback tests, geo-based randomization, or market-level experiments instead of user-level.

---

## Chapter 4 — Confounding, Selection Bias, and DAGs

### 4.1 What is a confounder?
A confounder Z is a variable that causally affects **both** the treatment D and the outcome Y. It creates a spurious association between D and Y even if D has no causal effect on Y at all.

Classic DAG:
```
      Z (confounder)
     /              \
    v                v
    D  ------------> Y
   (treatment)     (outcome)
```
Ice cream sales (D) and drowning deaths (Y) are correlated — confounded by hot weather (Z), which increases both.

### 4.2 Types of bias to distinguish
- **Confounding bias**: common cause of D and Y not controlled for.
- **Selection bias**: conditioning on (or the sample being restricted by) a variable that is a common **effect** of D and Y (a "collider"), or non-random sample inclusion. Conditioning on a collider *creates* spurious association where none existed causally.
- **Reverse causation**: Y actually causes D (e.g., sicker patients seek more treatment — treatment looks "harmful" because sicker people chose it).
- **Measurement / recall bias**: systematically different measurement of Y or D across groups.

### 4.3 Collider example (why "controlling for everything" is wrong)
```
D --> M <-- Y
```
M (e.g., "hospitalization") is caused by both D (a drug) and Y (disease severity). If you condition on M (e.g., only study hospitalized patients), you can induce a *spurious* correlation between D and Y even if none exists in the general population — this is "Berkson's paradox" / collider bias.

### 4.4 The Backdoor Criterion & Ignorability
A set of covariates X satisfies the **backdoor criterion** relative to (D,Y) if:
1. No node in X is a descendant of D, and
2. X blocks every "backdoor path" (non-causal, confounding path) between D and Y.

If X satisfies this, then conditioning on X gives **conditional ignorability** (aka "unconfoundedness," "no unmeasured confounders", CIA — Conditional Independence Assumption):
```
(Y(1), Y(0)) ⊥ D | X
```
This is the assumption almost every observational-data method (regression adjustment, matching, propensity scores) leans on. It is **untestable** — you can never prove there's no hidden confounder; you can only argue plausibility and run sensitivity analyses (Ch.12).

### 4.5 Worked numerical: confounding bias in a simple case
Suppose true model: Y = 2 + 3D + 5Z + ε, and Z also drives D: D = 0.5 + 0.8Z + ν, with Z ~ Uniform(0,1), all noise independent mean-zero.

If you regress Y on D alone (omitting Z), the omitted-variable bias formula is:
```
Bias = β_Z * Cov(D,Z)/Var(D)
```
With Z ~ U(0,1), Var(Z)=1/12≈0.083. Cov(D,Z) = 0.8×Var(Z) = 0.8×0.083 = 0.0667. Var(D) = 0.8²×Var(Z) + Var(ν). If Var(ν) is small (say 0.02), Var(D) ≈ 0.64×0.083 + 0.02 ≈ 0.053+0.02=0.073.
```
Bias ≈ 5 × (0.0667/0.073) ≈ 5 × 0.914 ≈ 4.57
```
So naive OLS coefficient on D would be roughly 3 (true) + 4.57 ≈ 7.6 — massively overstating the true effect of 3, purely because Z confounds both D and Y and was omitted.

### Q&A
**Q: What's the difference between confounding and selection bias in DAG terms?**
A: Confounding = an uncontrolled common **cause** of D and Y (a "fork" open path). Selection bias = conditioning on (or sample-restricting by) a common **effect** of D and Y or of D and an outcome-related variable (a "collider"), which opens a spurious path that wasn't there before conditioning.

**Q: Can you ever prove "no unmeasured confounders" from data alone?**
A: No — it's a structural, untestable assumption about the world (though domain knowledge, negative controls, and sensitivity analysis can make it more or less plausible).

**Q: Give a real Google-style example of collider bias.**
A: Studying "does feature X increase user retention?" using only users who completed onboarding (a collider potentially caused both by liking feature X and by unrelated engagement drivers) can create spurious correlations between X and retention that don't exist in the full user population.

---

## Chapter 5 — Regression Adjustment

### 5.1 Idea
If ignorability holds given X, then within each stratum of X, treatment is "as good as random." Regression adjustment models:
```
Y = α + τD + β'X + ε
```
and interprets τ as the causal effect, *if* the linear functional form is correctly specified and X satisfies the backdoor criterion.

### 5.2 The danger: functional form and "regression adjustment ≠ magic"
Regression adjustment assumes:
1. Ignorability given X (untestable, as above).
2. Correct functional form (linearity, or correctly specified interactions).
3. Common support (no extrapolation into regions where treated/control don't overlap in X).

If treated and control units don't overlap in X-space, the "effect" you estimate partly comes from extrapolating a linear model outside the data range — very dangerous, and a common interview gotcha.

### 5.3 Worked numerical
Data: effect of a coaching program (D) on test score (Y), confounded by prior GPA (X).

| Student | X (GPA) | D | Y (score) |
|---|---|---|---|
| 1 | 2.0 | 0 | 60 |
| 2 | 2.5 | 0 | 65 |
| 3 | 3.0 | 0 | 70 |
| 4 | 3.0 | 1 | 78 |
| 5 | 3.5 | 1 | 82 |
| 6 | 4.0 | 1 | 90 |

Naive diff: mean(Y|D=1) − mean(Y|D=0) = mean(78,82,90) − mean(60,65,70) = 83.33 − 65 = **18.33**

But GPA (X) also predicts Y directly and predicts D (treated students have higher average GPA: 3.5 vs 2.5). Fit Y = α + τD + βX (by hand, approximately, using two "GPA-matched" comparisons is easier for intuition):
- Comparing student 3 (X=3.0,D=0,Y=70) to student 4 (X=3.0,D=1,Y=78): **same X**, effect ≈ 78−70 = **8**.
This "exact match on X" comparison strips out the GPA confound and suggests the true effect is closer to **8**, not 18.33 — most of the naive 18.33 gap was because treated students had higher baseline GPA, which independently raises scores.

(A full OLS fit would use all 6 points; the matched-pair comparison above is exactly what motivates Chapter 6.)

### Q&A
**Q: Why is "just control for confounders in a regression" not a complete answer in an interview?**
A: Because it silently assumes (a) you've measured *all* confounders (ignorability), (b) the linear functional form is correct, and (c) there's common support between treated/control in X — none of which is guaranteed, and a good answer names these assumptions.

**Q: What happens if treated and control groups have no overlapping X values?**
A: The regression is extrapolating outside the observed data in at least one arm — your "effect" estimate is a model artifact, not something the data can actually support. Check propensity score overlap / trim non-overlapping regions.

---

## Chapter 6 — Propensity Score Matching (PSM) and Inverse Probability Weighting (IPW)

### 6.1 Propensity score definition
```
e(X) = P(D=1 | X)
```
**Key theorem (Rosenbaum & Rubin, 1983):** if ignorability holds given X, it also holds given the scalar e(X):
```
(Y(1),Y(0)) ⊥ D | e(X)
```
This is huge — instead of matching on a high-dimensional X, you can match on a single number, the propensity score.

### 6.2 Matching
For each treated unit, find one (or more) control unit(s) with the closest propensity score (nearest-neighbor matching, often with a "caliper" max distance), then compare outcomes within matched pairs. Estimated ATT:
```
ATT_hat = (1/n_1) Σ_{i:D=1} [ Y_i − Y_{j(i)} ]
```
where j(i) is the matched control for treated unit i.

### 6.3 Inverse Probability Weighting (IPW)
Weight each unit by the inverse of the probability of receiving the treatment it actually got:
```
w_i = D_i/e(X_i) + (1-D_i)/(1-e(X_i))
```
Then:
```
ATE_hat = (1/n) Σ w_i * [treated-indicator adjustment] 
```
more precisely, the Horvitz–Thompson-style estimator:
```
ATE_hat = (1/n)Σ [ D_i Y_i / e(X_i) ]  −  (1/n)Σ [ (1-D_i) Y_i / (1-e(X_i)) ]
```
Intuition: an unlikely-to-be-treated unit that *was* treated is "upweighted" because it stands in for many similar untreated units, balancing the pseudo-population as if treatment had been randomized.

**Danger:** if e(X) is close to 0 or 1 for some units, weights explode → huge variance. This is the **positivity/overlap** assumption: need 0 < e(X) < 1 for all X in the support.

### 6.4 Worked numerical
5 users, covariate X=1 (engaged) or X=0 (not engaged), D = got push notification.

| User | X | D | Y (sessions) |
|---|---|---|---|
| 1 | 1 | 1 | 10 |
| 2 | 1 | 1 | 12 |
| 3 | 1 | 0 | 8 |
| 4 | 0 | 1 | 4 |
| 5 | 0 | 0 | 3 |

Suppose from a larger population we estimated propensity scores: e(X=1)=0.7 (engaged users more likely treated), e(X=0)=0.3.

IPW weights:
- User 1 (X=1,D=1): w=1/0.7=1.4286
- User 2 (X=1,D=1): w=1/0.7=1.4286
- User 3 (X=1,D=0): w=1/(1-0.7)=1/0.3=3.333
- User 4 (X=0,D=1): w=1/0.3=3.333
- User 5 (X=0,D=0): w=1/(1-0.3)=1/0.7=1.4286

Weighted treated mean:
```
Σ(D_i * Y_i * w_i) / Σ(D_i * w_i)
= (10×1.4286 + 12×1.4286 + 4×3.333) / (1.4286+1.4286+3.333)
= (14.286+17.143+13.333) / 6.190
= 44.762 / 6.190 = 7.23
```
Weighted control mean:
```
= (8×3.333 + 3×1.4286) / (3.333+1.4286)
= (26.667+4.286)/4.762 = 30.952/4.762 = 6.50
```
IPW ATE_hat ≈ 7.23 − 6.50 = **0.73** sessions.
Compare naive: mean(Y|D=1)=(10+12+4)/3=8.67, mean(Y|D=0)=(8+3)/2=5.5, naive diff=3.17 — much larger, because naive comparison doesn't account for engaged users (X=1) being both more likely treated AND having higher baseline sessions.

### Q&A
**Q: Why match on propensity score instead of raw covariates?**
A: With many covariates, exact matching is often impossible ("curse of dimensionality"); the propensity score collapses all confounders into one scalar summary sufficient (under ignorability) to balance treatment groups.

**Q: What's the overlap/positivity assumption, and why does violating it break IPW?**
A: It requires every unit to have a nonzero probability of receiving *either* treatment level (0<e(X)<1). If e(X)→0 or 1 for some units, their inverse-probability weight explodes, causing huge variance and unstable, unreliable estimates — effectively there's no valid comparison group for that region of X.

**Q: PSM vs regression adjustment — when would you prefer PSM?**
A: When you're unsure of the correct functional form for Y|X, or when treated/control don't have much overlap and you want matching to explicitly show you the (lack of) common support, rather than have a regression model silently extrapolate.

---

## Chapter 7 — Instrumental Variables (IV)

### 7.1 When do you need IV?
When there's an **unmeasured confounder** you cannot adjust for — regression/matching/PSM cannot fix this since they all require ignorability given *observed* X. IV works around unmeasured confounding by finding a variable Z that:

1. **Relevance**: Z is correlated with D (affects treatment take-up). $Cov(Z,D) \ne 0$
2. **Exclusion restriction**: Z affects Y **only through** D, not directly. (untestable — must argue from domain knowledge)
3. **Independence/exogeneity**: Z is not correlated with the unmeasured confounder U. $Z \perp U$

DAG:
```
Z --> D --> Y
       ^
       |
       U (unmeasured confounder) --> Y  and --> D
```
(Z has no arrow directly into Y, and no arrow to/from U.)

### 7.2 The Wald / IV estimator (binary instrument, binary treatment)
```
IV estimate (LATE) = [ E(Y|Z=1) − E(Y|Z=0) ] / [ E(D|Z=1) − E(D|Z=0) ]
```
Numerator = "intent-to-treat" effect of the instrument on the outcome.
Denominator = effect of the instrument on treatment take-up (the "first stage").

This is **2-Stage Least Squares (2SLS)** in the linear case:
- Stage 1: regress D on Z (and controls) → get D̂ (predicted treatment)
- Stage 2: regress Y on D̂ (and controls) → coefficient on D̂ is the IV estimate

### 7.3 LATE, not ATE
IV only identifies the effect for **compliers** — units whose treatment status is actually moved by the instrument. It says nothing about "always-takers" (would take treatment regardless of Z) or "never-takers" (would never take treatment regardless of Z). This is the **LATE (Local ATE)** interpretation, and it's a favorite "gotcha" question.

### 7.4 Worked numerical
Classic example: effect of military service (D) on earnings (Y), instrumented by draft lottery number (Z, binary: drafted vs not).

Say:
- E[Y | Z=1 (drafted)] = 38,000 (avg earnings of drafted group, including non-compliers)
- E[Y | Z=0 (not drafted)] = 40,000
- E[D | Z=1] = 0.30 (30% of drafted actually serve — some exemptions)
- E[D | Z=0] = 0.05 (5% volunteer anyway despite not being drafted)

```
IV estimate = (38,000 − 40,000) / (0.30 − 0.05)
            = −2,000 / 0.25
            = −8,000
```
Interpretation: for the "compliers" (people who serve *because* they were drafted, and wouldn't have otherwise), military service causes an $8,000 decrease in earnings. This says nothing directly about always-takers/never-takers.

### 7.5 Google-relevant IV example
Effect of app crashes (D) on user churn (Y) is confounded by unmeasured device quality (U, e.g., old/cheap phones crash more AND churn more for unrelated reasons). An instrument could be a **randomly-assigned server-side experiment that varies crash rates for reasons unrelated to device** (e.g., a random feature flag rollout causing crashes) — Z affects D (crash rate) but (if well-designed) only affects churn Y *through* the crash, not directly and not through device quality.

### Q&A
**Q: State the three IV assumptions in your own words with a business example.**
A: (1) Relevance: the instrument must actually move treatment (a randomly assigned discount code must actually change purchase rates). (2) Exclusion: the instrument affects the outcome ONLY via the treatment, not any other channel (the discount code shouldn't itself remind people to buy for unrelated reasons like appearing in a marketing email that boosts general engagement). (3) Independence: instrument assignment must be unrelated to unmeasured confounders (discount codes randomly assigned, not targeted based on likely-to-purchase signals).

**Q: Why can't you test the exclusion restriction with data?**
A: Because it requires knowing that Z has *no* effect on Y except through D — this requires the unobserved counterfactual "Y if Z changed but D didn't," which you never observe; it's a structural/domain-knowledge assumption, not a statistical hypothesis you can directly test (some partial/indirect tests exist, e.g., checking Z uncorrelated with observed covariates, but the core assumption isn't fully testable).

**Q: If compliance is very low (denominator near 0), what happens to the IV estimator?**
A: It becomes very unstable/high-variance ("weak instrument" problem) — small sampling noise in the denominator gets amplified, and 2SLS estimates can be badly biased in finite samples; check first-stage F-statistic (rule of thumb: F>10) to assess instrument strength.

---

## Chapter 8 — Difference-in-Differences (DiD)

### 8.1 Setup
Used when you have a treatment group and control group observed **before and after** a treatment/policy change, and you're worried about *pre-existing* differences between groups (so simple post-only comparison would be confounded), but assume both groups would have moved **in parallel** absent treatment.

### 8.2 The estimator
```
DiD = [ Ȳ_treat,after − Ȳ_treat,before ] − [ Ȳ_control,after − Ȳ_control,before ]
```
Equivalently, from a regression:
```
Y_it = α + β·Treat_i + γ·Post_t + δ·(Treat_i × Post_t) + ε_it
```
δ is the DiD estimate — the coefficient on the interaction term.

### 8.3 The critical assumption: Parallel Trends
Absent treatment, treated and control groups **would have had the same trend** (not necessarily the same level) over time:
```
E[Y(0)_after − Y(0)_before | Treat=1] = E[Y(0)_after − Y(0)_before | Treat=0]
```
This is **untestable in the post period** (we never see treated group's untreated counterfactual), but you can build confidence by checking **pre-trends** — do the two groups move in parallel over *multiple* pre-periods? If pre-trends already diverge, parallel trends is suspect.

### 8.4 Worked numerical
A city (treatment) raises minimum wage; a neighboring city (control) doesn't. Employment (Y, in thousands):

|  | Before | After |
|---|---|---|
| Treatment city | 100 | 96 |
| Control city | 90 | 88 |

```
Δ_treat = 96 − 100 = −4
Δ_control = 88 − 90 = −2
DiD = Δ_treat − Δ_control = −4 − (−2) = −2
```
Interpretation: the minimum wage increase is associated with a **2,000-job decrease** relative to what would have happened absent the policy (using the control city's trend as the counterfactual trend). Note simple before-after in treatment alone (−4) would have wrongly attributed the control city's general downward trend (−2, e.g., broader regional recession) entirely to the policy.

### 8.5 Threats to DiD
- **Parallel trends violated** (biggest threat) — check with an event-study plot of pre-period coefficients.
- **Anticipation effects** — units change behavior *before* treatment starts (e.g., firms adjust hiring right before a minimum wage hike is announced).
- **Composition changes** — if it's repeated cross-sections (not the same units), changing population composition can masquerade as a treatment effect.
- **Staggered adoption pitfalls** — with multiple treatment times across units, naive two-way fixed effects regression can be badly biased (a well-known modern econometrics finding — Goodman-Bacon, Callaway-Sant'Anna decompositions address this).

### Q&A
**Q: What's the key identifying assumption of DiD, and how would you check it (even though you can't prove it)?**
A: Parallel trends — that treated and control groups would have evolved identically absent treatment. You check it indirectly via pre-trends: plot outcome trajectories for both groups over several pre-treatment periods and verify they move together before the treatment date; you can also do a placebo test using a fake treatment date in the pre-period.

**Q: Google rolls out a new ranking algorithm in the US only, and you want to measure its effect using an unaffected country as control. Name two things that could break parallel trends here.**
A: (1) Seasonality/macro shocks that differ by country (e.g., a US-specific holiday shopping season or economic event coinciding with launch); (2) different underlying growth trends in user base or usage patterns between US and control country unrelated to the launch (e.g., US market already saturating while control country still growing).

**Q: What's wrong with naively running two-way fixed effects DiD when treatment timing varies a lot across units (staggered rollout)?**
A: Standard two-way FE implicitly uses already-treated units as part of the "control" comparison for later-treated units, and if treatment effects are dynamic/heterogeneous over time, this can produce severely biased (even sign-reversed) estimates; modern estimators (Callaway & Sant'Anna, Sun & Abraham, etc.) explicitly avoid using already-treated units as controls.

---

## Chapter 9 — Regression Discontinuity Design (RDD)

### 9.1 Idea
Exploit a treatment rule that switches on/off at a known **cutoff** of a continuous variable (the "running variable" X, e.g., a test score threshold for a scholarship, a bid threshold for an ad auction, a review-score threshold for a badge). Units *just above* and *just below* the cutoff are assumed nearly identical in all other respects — so the discontinuity in outcomes at the cutoff isolates the causal effect.

### 9.2 Sharp vs Fuzzy RDD
- **Sharp RDD**: treatment is a deterministic function of X — everyone above cutoff c is treated, everyone below is not.
```
τ_RDD = lim_{x→c+} E[Y|X=x] − lim_{x→c-} E[Y|X=x]
```
- **Fuzzy RDD**: probability of treatment *jumps* at c but isn't 0/1 (e.g., compliance issues). Estimated like an IV, using the cutoff-crossing as instrument:
```
τ_fuzzy = [ lim_{x→c+}E[Y|X=x] − lim_{x→c-}E[Y|X=x] ] / [ lim_{x→c+}E[D|X=x] − lim_{x→c-}E[D|X=x] ]
```

### 9.3 Key assumption: continuity / no manipulation
All *other* factors affecting Y must be continuous through the cutoff — no other discontinuous jump should coincide with treatment at exactly c. Also, units must not be able to precisely manipulate X to land on the favorable side of the cutoff (e.g., if students knew the exact scholarship cutoff and could nudge their score, that breaks RDD — check for a suspicious "bunching" in the X density right at the cutoff, e.g., via a McCrary density test).

### 9.4 Worked numerical
Scholarship awarded if test score ≥ 80. Suppose local averages near the cutoff:
- Just below (scores 75–79): average future GPA = 2.8
- Just above (scores 80–84): average future GPA = 3.1

```
τ_RDD ≈ 3.1 − 2.8 = 0.3
```
Interpretation: for students right around the cutoff, the scholarship raises future GPA by about 0.3 points — but this is a strictly **local** estimate; it says nothing about the effect for a student who scored 40 or 99 (this is the classic RDD external-validity limitation).

### 9.5 Bandwidth choice
Only observations "close" to the cutoff are used (or weighted more heavily), controlled by a **bandwidth** h. Too wide → bias (comparing units that aren't really similar). Too narrow → high variance (few data points). Common practice: use local linear regression on each side with a data-driven optimal bandwidth (e.g., Imbens-Kalyanaraman or Calonico-Cattaneo-Titiunik methods) and check robustness across several bandwidths.

### Q&A
**Q: Why is RDD's estimate described as "local"? What's the tradeoff with this?**
A: It only identifies the causal effect for units right at the cutoff (an average over a vanishing neighborhood) — it can't tell you the effect far from the cutoff where the CEF (conditional expectation function) may look completely different. Tradeoff: strong internal validity near the cutoff, weak external validity/generalizability elsewhere.

**Q: How would you detect that people are manipulating their score to just barely qualify for a Google internal "top performer" bonus (X = performance score, cutoff = bonus threshold)?**
A: Plot the density (histogram) of the running variable X around the cutoff and test for abnormal bunching just above the cutoff (a "McCrary density discontinuity test"); a suspicious spike right above cutoff (and a dip below) signals manipulation, which would invalidate the RDD design (assignment is no longer "as good as random" near the cutoff).

**Q: What's the difference between sharp and fuzzy RDD in terms of estimator?**
A: Sharp RDD directly compares the jump in outcome at the cutoff (deterministic treatment); fuzzy RDD divides that outcome-jump by the jump in *treatment probability* at the cutoff, mirroring the IV/Wald estimator, because treatment isn't fully determined by crossing the cutoff.

---

## Chapter 10 — Synthetic Control

### 10.1 When to use it
When you have **one (or very few) treated unit** (e.g., one state, one city, one market) and **many potential control units**, and no single control looks like a great match — DiD with one arbitrary control city is fragile. Synthetic control builds a **weighted combination** of untreated units that best matches the treated unit's *pre-treatment* trajectory, then uses that synthetic combination as the counterfactual post-treatment.

### 10.2 The estimator
Choose weights $w_j \ge 0$, $\sum w_j = 1$ for donor units j, minimizing:
```
minimize  Σ_k v_k * ( X_1k − Σ_j w_j X_jk )²
```
over pre-treatment predictor variables k (e.g., pre-treatment outcome values, other covariates), where X_1 is the treated unit and X_j are donors. Then the effect estimate at each post-treatment time t:
```
τ_t = Y_1t − Σ_j w_j Y_jt
```
i.e., actual outcome minus the "synthetic" weighted-average counterfactual.

### 10.3 Worked numerical (simplified, 2 donors)
Treated market: Google launches a new feature in Market A only. Pre-treatment (avg of last 3 periods) engagement:
- Market A (treated) = 50
- Market B (donor) = 45
- Market C (donor) = 60

Solve for weights w_B, w_C (w_B+w_C=1) minimizing (50 − (w_B·45 + w_C·60))². Setting the bracket to 0: 50 = 45w_B + 60(1−w_B) = 60 − 15w_B → 15w_B = 10 → w_B=0.667, w_C=0.333.

Post-treatment: Market A = 58, Market B = 44, Market C = 61.
```
Synthetic counterfactual = 0.667×44 + 0.333×61 = 29.33 + 20.33 = 49.67
τ = 58 − 49.67 = 8.33
```
Estimated causal lift from the feature ≈ **8.33 engagement units**.

### 10.4 Inference: placebo tests
Since there's often just 1 treated unit, you can't use standard SEs. Instead, run the *same* synthetic control procedure treating each **donor** unit as if it were "treated" (placebo), and see how large its placebo effect is. If the real treated unit's effect is much larger than the distribution of placebo effects, that's evidence of a genuine effect (analogous to a permutation test / Fisher-style exact inference).

### Q&A
**Q: Why is synthetic control often better than a single "closest" comparison city?**
A: A weighted combination of multiple donors can match the treated unit's *entire pre-treatment trajectory* more closely than any single donor can, reducing bias from imperfect matching and being less reliant on one arbitrary, possibly idiosyncratic comparison unit.

**Q: How do you get a sense of statistical significance with only one treated unit?**
A: Placebo-in-space tests — re-run synthetic control pretending each untreated donor was the treated unit, generating a distribution of "placebo effects"; compare the actual treated unit's effect to this distribution (essentially a randomization/permutation-based inference).

**Q: What's required of the pre-treatment fit for synthetic control to be credible?**
A: The synthetic control should closely track the treated unit's outcome (and other predictors) for an extended pre-treatment period — poor pre-treatment fit undermines confidence that the same weights produce a valid post-treatment counterfactual.

---

## Chapter 11 — Interference and SUTVA Violations (crucial for Google-scale systems)

### 11.1 SUTVA
**Stable Unit Treatment Value Assumption**: unit i's potential outcome depends *only* on unit i's own treatment, not on the treatment assigned to other units, and there's only "one version" of each treatment level.
```
Y_i(D_1, D_2, ..., D_n) = Y_i(D_i)     ← SUTVA assumes this simplification is valid
```
When this fails, we call it **interference** or **spillover effects**.

### 11.2 Why this is a huge deal at Google/marketplace/social-network scale
- **Shared/limited resources**: ad auctions, search result slots, delivery driver pools — treating one user can literally take "supply" away from another (a zero-sum-ish channel). E.g., increasing ad load for treatment users reduces available inventory, indirectly affecting control users' ad experience too.
- **Social/network spillovers**: a feature shown to one user (e.g., a new sharing feature) changes behavior of their friends/followers regardless of the friends' own assignment (e.g., they receive more notifications, see more shared content).
- **Marketplace equilibrium effects**: e.g., a pricing experiment for riders can change driver behavior/supply, which affects *all* riders, not just those in the treatment arm.
- **General equilibrium / market-level effects**: aggregate changes (like overall ad prices in an auction) can shift due to treatment, contaminating the "control" baseline.

If interference exists but you ignore it and run a standard user-randomized A/B test, your estimated effect can be **badly biased in either direction** relative to the true effect of a **full rollout** (because in a full rollout, everyone is treated and there's no "unaffected control" to compare against — the whole market shifts).

### 11.3 Detecting and mitigating interference

| Method | Idea |
|---|---|
| **Cluster/graph randomization** | Randomize at the level of clusters (e.g., friend groups, geographic markets, cities) instead of individuals, so spillovers mostly stay within-cluster and don't cross treatment/control boundaries. |
| **Switchback experiments** | For marketplace-level effects, randomize treatment **over time** within the same unit (e.g., city gets pricing policy A on odd days, B on even days) rather than across units. |
| **Geo-based experiments** | Randomize at city/DMA/country level for products with local marketplace dynamics (rideshare, delivery, ads markets). |
| **Ego-cluster / graph cluster randomization** | For social networks, cluster densely-connected users together and randomize whole clusters. |
| **Two-sided / market-level "bidding" experiments** | e.g. budget-split A/B (holding total ad spend or supply fixed) to detect market-level saturation effects. |
| **Model-based correction** | Explicitly model exposure/spillover (e.g., "how many treated neighbors does this control unit have") and estimate dose-response as a function of exposure. |

### 11.4 Worked example — intuition, not just formula
An experiment gives treatment users an extra push notification. If notification sending capacity is unlimited and independent per user, SUTVA plausibly holds (no shared resource) → user-level randomization is fine.

But if the notification triggers a "shared feed ranking" recompute that also changes what *friends* (regardless of their own arm) see in their feed, then a friend in the control arm is contaminated by their treated friend's action → SUTVA violated → the naive user-level A/B test estimate is biased for the true "if we launched to everyone" effect.

**Quantifying bias direction (intuition):** if the spillover is "positive" (treated units help control units too, e.g., through shared social content), naive comparison **understates** the true effect of a full rollout (because "control" isn't a clean zero-effect baseline anymore — control implicitly received partial treatment). If spillover is "negative" (e.g., competing for the same finite ad slots — treating some users takes slots away from others), naive comparison **overstates** the effect of a full rollout, because in a full rollout there's no "un-treated" pool from which to steal resources.

### Q&A
**Q: Define SUTVA precisely and give a Google-relevant example of its violation.**
A: SUTVA requires a unit's potential outcome to depend only on its own treatment assignment, not on others'. Violation example: in an ads experiment, showing treatment users more ads reduces the ad inventory available for auction to control users (shared finite supply), so control users' outcomes are affected by the treatment assignment of *other* users — violating SUTVA.

**Q: Your user-randomized rideshare pricing experiment shows treatment (lower price) riders get more rides. Why might this NOT extrapolate to "lower prices for everyone will increase total rides by the same amount"?**
A: Driver supply is shared/limited; treatment riders got a bigger slice of a roughly fixed driver pool at control riders' expense (interference through the marketplace), so the user-level effect partly reflects a *reallocation* of scarce supply rather than a true supply-elastic increase in total rides — a full-population price cut wouldn't have this "steal from control" advantage, since there'd be no untreated pool to steal from.

**Q: Name one experimental design that mitigates marketplace-level interference for a rideshare pricing test.**
A: Switchback design (alternate pricing policy by time-block within the same city) or geo-based randomization (randomize whole cities) — both keep the "market" mostly consistent within an experimental unit, avoiding within-market cross-contamination between arms.

---

## Chapter 12 — Quasi-Experiments, Natural Experiments, and Sensitivity Analysis

### 12.1 Quasi/natural experiments
A situation where **some external, non-designed event** creates an "as-if random" assignment to treatment — you didn't design the randomization, but you argue it's plausibly exogenous. Examples: policy changes rolled out by legislature with an arbitrary cutoff date, weather shocks, lottery-based program allocation, natural disasters, algorithm bugs that accidentally created a mini-RCT.

These typically get analyzed with the tools you already learned — DiD (policy change over time), IV (lottery as instrument), RDD (arbitrary cutoff), synthetic control (single treated region) — quasi-experiments are less "a new method" and more "a new *source* of plausibly-exogenous variation to plug into the standard toolkit."

### 12.2 Sensitivity analysis for unmeasured confounding
Since ignorability is untestable, good practice is to ask: **"how strong would an unmeasured confounder need to be to overturn my conclusion?"**

**Rosenbaum bounds** (matching context): ask how much an unmeasured confounder could differentially affect the odds of treatment between matched pairs (parameter Γ) before your result would no longer be statistically significant.

**E-value** (modern epidemiology tool): the minimum strength of association (on the risk-ratio scale) that an unmeasured confounder would need to have with *both* treatment and outcome to fully explain away the observed effect, above and beyond measured covariates.
```
E-value ≈ RR + sqrt(RR × (RR − 1))     [for RR ≥ 1; use 1/RR if RR<1]
```
where RR is your observed (adjusted) risk ratio.

### 12.3 Worked numerical (E-value)
Observational study finds RR = 2.0 for "using feature X" → "churn."
```
E-value = 2.0 + sqrt(2.0 × 1.0) = 2.0 + 1.414 = 3.41
```
Interpretation: an unmeasured confounder would need to be associated with *both* feature-X-usage and churn by a risk ratio of at least ~3.4 (each) to fully explain away the observed RR=2.0 effect. If no plausible confounder is anywhere near that strong, your result is fairly robust; if you can think of one that plausibly is, be cautious.

### 12.4 Placebo / falsification tests (general-purpose sanity checks)
- **Placebo outcome test**: run your causal method on an outcome that treatment *shouldn't* affect. If you "find an effect" there too, your design likely has residual bias.
- **Placebo treatment/timing test**: pretend the treatment happened earlier (a period when nothing actually changed) — you should find no effect.
- **Balance checks**: verify treated/control groups look similar on *pre-treatment* observed covariates (common for RCTs and matching/PSM).

### Q&A
**Q: What makes an event a good "natural experiment," and what should you still worry about?**
A: A good natural experiment has assignment to "treatment" driven by something plausibly unrelated to potential outcomes (e.g., an arbitrary administrative cutoff date, a lottery). You should still worry about: whether the "as-if random" claim really holds (could affected parties anticipate/manipulate it?), whether other things changed at the same time (confounding the natural experiment itself), and generalizability of the specific context.

**Q: What does an E-value of 1.2 vs an E-value of 5 tell you differently about robustness?**
A: An E-value of 1.2 means only a very weak unmeasured confounder (barely correlated with both treatment and outcome) could explain away your result — the finding is fragile. An E-value of 5 means it would take an implausibly strong unmeasured confounder to overturn the result — much more robust to hidden bias.

**Q: You find that a new search-ranking algorithm "improves" a metric that has nothing plausibly to do with ranking (e.g., app crash rate on an unrelated tab). What should you conclude?**
A: This is a placebo-outcome red flag — it strongly suggests your causal estimation pipeline has residual bias (e.g., broken randomization, contaminated logging, or confounding from a concurrent unrelated change), not that ranking magically affects crashes; investigate the experiment infrastructure before trusting the main-metric result.

---

## Chapter 13 — Google-style Case Studies & Rapid-Fire Interview Q&A

### 13.1 Case Study 1: New feature rollout, can't randomize
*"Product wants to roll out a new onboarding flow. Leadership already launched it to 100% of new users in the EU last month (no experiment), and now asks you: 'did it work?' What do you do?"*

Approach:
1. Ask if there's a natural control (e.g., other regions not yet launched, or a staggered launch you can exploit) → candidate for **DiD** (EU vs comparable region, before/after) — check parallel pre-trends first.
2. If truly no comparison group exists, consider **synthetic control** using multiple other regions as weighted donors.
3. Flag confounding risks: any other EU-specific event around the same time (regulatory change, marketing campaign, holiday)? Run placebo/pre-trend checks.
4. Recommend for *next* launch: hold back a small randomized control group (even 5%) specifically to enable clean causal measurement going forward.

### 13.2 Case Study 2: Ads auction experiment shows conflicting user vs advertiser metrics
*"User-randomized experiment shows increasing ad relevance threshold increases user satisfaction but decreases total ad revenue. Advertiser-side team ran their own randomization and found revenue per advertiser went up. Reconcile."*

Approach: this smells like **interference through the shared auction** — user-level and advertiser-level randomizations interact with the same fixed inventory/auction, so estimates from each can be internally valid for their own randomization unit but *not* simply combinable or extrapolated to a global rollout; recommend a **market-level (e.g., geo-randomized) holdout experiment** that randomizes the *entire ecosystem* (all users AND advertisers within a geo) to estimate the true equilibrium effect of a full rollout.

### 13.3 Case Study 3: Observational log data only, need a quick causal read
*"We don't have time for an RCT. We have logs of which users organically adopted a power-user feature, and their subsequent retention. Give a defensible causal estimate."*

Approach: 
1. Draw the DAG — what plausibly confounds "adoption" and "retention"? (e.g., prior engagement level, tenure, device type.)
2. Use **PSM/IPW** on those observed confounders; check covariate balance after weighting/matching (standardized mean differences should shrink toward 0).
3. Report ATT with an **E-value** or Rosenbaum bounds sensitivity analysis, explicitly stating the ignorability assumption and how fragile/robust the conclusion is to plausible hidden confounders (e.g., "user motivation/tech-savviness" is a likely omitted confounder — flag it).
4. Recommend a follow-up RCT (e.g., randomized feature prompt / nudge) to validate before big decisions.

### 13.4 Rapid-fire Q&A

**Q: One-liner: what's the difference between DiD and RDD?**
A: DiD compares *before/after* trends between treated and control groups over time; RDD compares units just above/below a treatment-assignment *cutoff* at one point (or over time), leaning on local continuity rather than parallel trends.

**Q: One-liner: why is matching alone (without propensity scores or regression) potentially incomplete?**
A: Exact matching on many covariates is often infeasible (curse of dimensionality) — most units won't have exact matches, forcing you to either drop data or approximate, which reintroduces the modeling choices matching was meant to avoid.

**Q: When would ATE ≠ ATT even under full randomization?**
A: They're equal in expectation under randomization (that's the whole point) — any observed gap is due to sampling variance, not a structural reason; if someone claims ATE≠ATT under a truly randomized design, suspect broken randomization or differential attrition rather than a real structural difference.

**Q: A colleague says "we don't need a control group, we'll just look at before/after." What's your pushback?**
A: Before/after alone can't separate the treatment effect from any other time-varying factor (seasonality, macro trends, concurrent launches) — you need a comparison group (or a design like RDD/IV) to net out what would have happened *anyway*.

**Q: What's the "bad control" problem?**
A: Adding a control variable that is itself *caused by* the treatment (a mediator or a collider) can bias your estimate — e.g., controlling for "number of app opens" when studying "notification → revenue" if notifications cause app opens which cause revenue: you'd be blocking part of the true causal pathway, biasing the effect toward zero (or worse, opening a collider path if it's a collider rather than a mediator).

**Q: How do you communicate a causal estimate's uncertainty to a non-technical exec?**
A: State the point estimate with a confidence interval in business terms ("we estimate a 2–4% lift in revenue, 95% confident it's not zero"), name the key assumption in plain language ("this assumes the two markets would have moved similarly without the change"), and give a sensitivity read ("even if there's some hidden factor we didn't measure, it would need to be unusually strong to fully explain this away").

**Q: Rank these methods from strongest to weakest causal identification, roughly: RCT, RDD, DiD, IV, PSM, naive regression.**
A: Roughly: RCT > RDD ≈ IV (when instrument/cutoff is strong and assumptions plausible) > DiD (when parallel trends is plausible) > PSM/regression adjustment (rely on the strong, untestable full-ignorability assumption) > naive regression/correlation. (Reasonable people order RDD vs IV differently depending on context — the key point is RCT is the gold standard and naive correlation is the weakest, with quasi-experimental methods in between depending on assumption plausibility.)

---

## Quick-Reference Cheat Sheet

| Method | Key assumption | What you need | Estimand |
|---|---|---|---|
| RCT | Randomization | Ability to randomize | ATE |
| Regression adjustment | Ignorability given X + correct functional form | Rich observed X | ATE/ATT |
| PSM / IPW | Ignorability given X + overlap/positivity | Rich observed X | ATT/ATE |
| IV | Relevance + exclusion + independence | A valid instrument Z | LATE |
| DiD | Parallel trends | Pre/post data, treated & control groups | ATT |
| RDD | Continuity at cutoff, no manipulation | A sharp/fuzzy assignment rule | Local ATE at cutoff |
| Synthetic Control | Good pre-treatment fit, few treated units | Many untreated donor units | ATT for the one treated unit |
| (All observational methods) | No unmeasured confounders (untestable) | — | Run sensitivity analysis (E-value, Rosenbaum bounds) |

**Golden rule for interviews:** always state (1) the estimand you're targeting, (2) the identifying assumption, (3) how you'd sanity-check that assumption, *before* writing down an estimator formula.
