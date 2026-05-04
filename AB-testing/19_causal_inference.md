# 🔗 19 — Causal Inference: When You Can't Randomize

> *"Correlation is not causation — but with the right design, correlation can become causation."*

---

## Why Causal Inference?

A/B tests establish causality through **randomization** — by design, treatment assignment is independent of all confounders. But randomization is often impossible:

- Can't randomly assign people to smoke cigarettes
- Can't randomly assign countries to adopt a tax policy
- Can't randomly decide which users get a new feature when it's a platform-wide change
- Historical/observational data exists, but no experiment was run

**Causal inference** provides rigorous methods to estimate causal effects from observational data.

---

## The Fundamental Problem of Causal Inference

For individual $i$, define:
- $Y_i(1)$ = outcome if treated
- $Y_i(0)$ = outcome if not treated
- Individual Treatment Effect: $\tau_i = Y_i(1) - Y_i(0)$

**Problem:** You can only observe one potential outcome — the other is the **counterfactual**.

$$\text{Average Treatment Effect (ATE)} = E[Y_i(1) - Y_i(0)]$$

In an RCT: $ATE = E[Y \mid T=1] - E[Y \mid T=0]$ ✅ (valid because randomization ensures independence)

In observational data: $E[Y \mid T=1] - E[Y \mid T=0]$ is biased by selection bias:

$$\underbrace{E[Y \mid T=1] - E[Y \mid T=0]}_{\text{Observed difference}} = \underbrace{ATE}_{\text{True effect}} + \underbrace{E[Y(0) \mid T=1] - E[Y(0) \mid T=0]}_{\text{Selection bias}}$$

---

## Method 1: Difference-in-Differences (DiD)

### Intuition

Compare the **change over time** in the treatment group vs. the change over time in the control group.

$$\hat{\tau}_{DiD} = (\bar{Y}_{T, post} - \bar{Y}_{T, pre}) - (\bar{Y}_{C, post} - \bar{Y}_{C, pre})$$

This removes time-invariant confounds AND common time trends.

### Visual Intuition

```
Outcome
  │          /── Treatment group (actual)
  │        /  
  │      /──────── Counterfactual (what treatment would have done without treatment)
  │    /  \       ↑ DiD estimator = gap between actual and counterfactual
  │───────── Control group
  │
  ─────────────────── Time
       Pre    Post
```

### Key Assumption: Parallel Trends

The treatment and control groups would have followed the same trend in the absence of treatment. **This is untestable for the post-period** but can be validated by checking pre-treatment trends.

### Regression Form

$$Y_{it} = \alpha + \beta_1 \text{Post}_t + \beta_2 \text{Treated}_i + \tau (\text{Post}_t \times \text{Treated}_i) + \epsilon_{it}$$

The coefficient $\tau$ on the interaction term is the **DiD estimator**.

### Worked Example

> Amazon tests a new seller dashboard in one region, keeps old dashboard in another.
>
> | Group | Pre-Revenue | Post-Revenue | Difference |
> |---|---|---|---|
> | Treatment (new dashboard) | $1000 | $1300 | +$300 |
> | Control (old dashboard) | $900 | $1050 | +$150 |

$$\hat{\tau}_{DiD} = 300 - 150 = \$150$$

The new dashboard caused an estimated **$150 increase in revenue**, net of the overall time trend.

### Assumptions

1. **Parallel trends** (key, untestable post-treatment)
2. **No spillover** (SUTVA): control group not affected by treatment
3. **No compositional changes**: same types of units in each period

---

## Method 2: Regression Discontinuity Design (RDD)

### Intuition

People just above and just below a threshold are nearly identical — their treatment status is "as good as random" near the cutoff.

### Canonical Example

> A scholarship is awarded to students scoring ≥ 70. Students scoring 69 vs. 71 are very similar — the only difference is whether they got the scholarship. We estimate the causal effect of the scholarship by comparing outcomes for students just above vs. just below 70.

### Sharp RDD

Treatment is a deterministic function of the running variable $X$:

$$T_i = \mathbb{1}[X_i \geq c]$$

The causal effect at the cutoff:

$$\tau_{RDD} = \lim_{x \downarrow c} E[Y \mid X=x] - \lim_{x \uparrow c} E[Y \mid X=x]$$

### Regression Implementation

$$Y_i = \alpha + \tau T_i + \beta_1 (X_i - c) + \beta_2 T_i (X_i - c) + \epsilon_i$$

### Visual Intuition

```
Outcome
  │          ●●●●●●●●  ← Treatment group trend
  │        ●●
  │       ●
τ ↕───────────────────  ← Jump at cutoff = causal effect
  │     ○○○            ← Control group trend  
  │  ○○○
  │○○
  ─────────────────────── Running variable X
                  c (cutoff)
```

### Bandwidth Selection

Only use observations close to the cutoff — but tradeoff between:
- **Narrow bandwidth**: more valid (similar units), less data (noisier)
- **Wide bandwidth**: more data, potentially incomparable units

Use data-driven bandwidth selectors (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik).

### Key Assumption

No **manipulation** of the running variable (people shouldn't be able to precisely sort above the cutoff). Test with **McCrary density test**.

---

## Method 3: Instrumental Variables (IV)

### Intuition

An **instrument** $Z$ affects treatment $T$ but only affects outcome $Y$ through $T$ — it has no direct effect. This allows isolating exogenous variation in $T$.

### Conditions for a Valid Instrument

1. **Relevance**: $Z$ is correlated with $T$ (can be tested: run $T$ on $Z$, check F-stat ≥ 10)
2. **Exclusion restriction**: $Z$ only affects $Y$ through $T$ (untestable — requires theory)
3. **Independence**: $Z$ is independent of confounders

### Classic Example

> Effect of education on earnings. Problem: smart people both get more education AND earn more (confounder = ability).
>
> **Instrument**: Distance to college (proximity). Affects education attainment but doesn't directly cause higher earnings. People who happen to live near colleges get more education (exogenous variation) → use to estimate causal effect of education.

### 2SLS Estimation

**Stage 1:** Regress $T$ on $Z$ (and controls):

$$\hat{T}_i = \hat{\pi}_0 + \hat{\pi}_1 Z_i$$

**Stage 2:** Regress $Y$ on $\hat{T}$ (the fitted values from Stage 1):

$$Y_i = \hat{\alpha} + \hat{\tau} \hat{T}_i + \epsilon_i$$

The estimate $\hat{\tau}$ is the **IV/2SLS estimator** — the Local Average Treatment Effect (LATE).

### Weak Instrument Problem

If first-stage F-stat < 10, the instrument is weak — small correlations between $Z$ and $T$ can cause large bias. Always report the first-stage F-stat.

---

## Method 4: Propensity Score Matching (PSM)

### Intuition

Match treated units to similar control units based on their probability of being treated (propensity score). Creates a pseudo-experiment where treated and matched controls are comparable.

### Propensity Score

$$e(X_i) = P(T_i = 1 \mid X_i)$$

Estimated via logistic regression:

```python
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression()
ps_model.fit(X_features, T)
propensity_scores = ps_model.predict_proba(X_features)[:, 1]
```

### Matching Strategies

| Method | Description |
|---|---|
| 1-to-1 nearest neighbor | Match each treated unit to closest control by PS |
| Kernel matching | Weight all controls by distance |
| IPW (Inverse Probability Weighting) | Upweight underrepresented treated units |

### Estimated ATT (Average Treatment Effect on Treated)

$$\hat{\tau}_{ATT} = \frac{1}{n_T} \sum_{i: T_i=1} \left(Y_i - \sum_{j: T_j=0} w_{ij} Y_j\right)$$

### Critical Assumption: Ignorability

$$Y(0), Y(1) \perp T \mid X$$

Conditional on observed covariates $X$, treatment is as good as random. **This assumes no unobserved confounders** — a strong and untestable assumption.

---

## Method Comparison

| Method | Assumption | Best Used For | Can Handle Unobserved Confounders? |
|---|---|---|---|
| DiD | Parallel trends | Policy changes, rollouts | Partially (time-invariant only) |
| RDD | No manipulation at cutoff | Rule-based thresholds | Yes (near cutoff) |
| IV | Valid instrument exists | When instrument available | Yes |
| PSM | No unobserved confounders | Rich covariate data | ❌ No |
| A/B Test | Randomization | Any product change | ✅ Yes (by design) |

---

## Product Applications at FAANG

| Company | Situation | Method |
|---|---|---|
| Google | Measure impact of organic search rank change (can't randomize rank) | IV (position as instrument) |
| Meta | Measure impact of friend adding feature (network effects violate RCT) | Clustered RCT or DiD |
| Uber | Effect of surge pricing on driver supply (no pure control) | RDD (surge threshold as cutoff) |
| LinkedIn | Measure impact of premium features on job outcome | PSM with rich profile features |
| Airbnb | Measure impact of COVID policy on booking behavior | DiD (policy adoption across regions) |

---

## 💬 FAANG Interview Questions & Answers

**Q1: We can't run an A/B test for a feature because it's being launched globally. How would you measure its impact?**

> A: Several options depending on what data is available. (1) **DiD**: If the feature launched in some markets before others, compare treated vs. untreated markets pre/post launch. Assumes parallel trends. (2) **RDD**: If launch was based on a threshold (e.g., account age ≥ 30 days), compare users just above/below the cutoff. (3) **Synthetic control**: Build a weighted combination of untreated units that match the treated unit's pre-treatment trend. (4) **Holdout group**: Reserve a random 5% of users who don't get the feature even after global launch — maintain this holdout for ongoing measurement.

**Q2: What's the difference between ATE and ATT in causal inference?**

> A: ATE (Average Treatment Effect) = average causal effect over the entire population — both those who received treatment and those who didn't. ATT (Average Treatment Effect on the Treated) = average causal effect only among those who actually received treatment. PSM estimates ATT, which is often more policy-relevant: "How much did this feature help the users who used it?"

**Q3: Why is the exclusion restriction in IV untestable?**

> A: Because it states that the instrument $Z$ affects $Y$ only through $T$ — but there's no statistical test for this. It requires domain expertise and theoretical argument. For example, distance to college as an instrument: we assume living near a college doesn't directly affect future earnings (only through education). But what if wealthy families live near colleges AND those families give financial advantages? Then the exclusion restriction fails. This is why IV results should always be accompanied by theoretical justification for the instrument.

**Q4: You have observational data and want to compare two user groups. How do you decide between PSM and regression adjustment?**

> A: Both are valid under the same ignorability assumption. Regression adjustment is generally preferred when the outcome model is well-specified and sample size is reasonable — it's more efficient (lower variance). PSM is better when you want to explicitly ensure overlap between groups and avoid extrapolating the regression model outside the support of the data. In practice, I'd use doubly-robust estimation: combine both propensity score weighting and outcome regression. It's consistent if *either* the propensity model or outcome model is correctly specified.

---

## 🚨 Common Pitfalls

1. **DiD:** Checking parallel trends only visually instead of statistically; ignoring compositional changes.
2. **RDD:** Using a bandwidth that's too wide (units aren't comparable); not testing for manipulation.
3. **IV:** Using a weak instrument (F-stat < 10); asserting exclusion restriction without justification.
4. **PSM:** Claiming causal identification when there are unobserved confounders; forgetting balance checks post-matching.
5. **General:** Confusing correlation with causation; not stating identifying assumptions explicitly.

---

*← [18 — CUPED](18_cuped_variance_reduction.md) | [20 — Regression & Hypothesis Testing →](20_regression_hypothesis_testing.md)*
