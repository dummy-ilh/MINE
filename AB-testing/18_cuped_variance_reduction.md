# ⚡ 18 — CUPED & Variance Reduction Techniques

> *"CUPED is one of the highest-leverage ideas in experimentation: you get more statistical power for free, just by using data you already have."* — Netflix Technology Blog

---

## The Core Problem: Variance is the Enemy of Power

Recall the test statistic for a two-sample mean comparison:

$$Z = \frac{\bar{Y}_T - \bar{Y}_C}{\sqrt{2\sigma^2/n}}$$

Power increases when either:
- $n$ increases (more users — expensive or slow)
- $\sigma^2$ decreases (less variance — often free)

**CUPED reduces $\sigma^2$ by incorporating pre-experiment data.** Same users, same experiment, dramatically more power.

---

## What is CUPED?

**CUPED** = **C**ontrolled-experiment **U**sing **P**re-**E**xperiment **D**ata

Introduced by Microsoft Research (Deng et al., 2013). Widely adopted at **Netflix, LinkedIn, Airbnb, Booking.com, Lyft**.

### Core Idea

If a user's **pre-experiment behavior** is correlated with their **in-experiment behavior**, we can use it to reduce variance.

**Example:** A user's revenue last week predicts their revenue this week. By "subtracting out" what we could have predicted from pre-experiment data, we remove a large chunk of variance — leaving a cleaner signal.

---

## Mathematical Formulation

### Step 1: Define the CUPED Estimator

Let:
- $Y_i$ = metric for user $i$ during the experiment
- $X_i$ = pre-experiment covariate for user $i$ (e.g., revenue last week)

The **CUPED-adjusted metric** is:

$$\tilde{Y}_i = Y_i - \theta \cdot X_i$$

Where $\theta$ is chosen to minimize the variance of $\tilde{Y}_i$:

$$\theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

This is simply the **regression coefficient of $Y$ on $X$**.

### Step 2: Compute the Treatment Effect on Adjusted Metric

$$\hat{\delta}_{CUPED} = \bar{\tilde{Y}}_T - \bar{\tilde{Y}}_C$$

The expected value is unchanged (unbiased):

$$E[\hat{\delta}_{CUPED}] = E[\hat{\delta}]$$

### Step 3: Variance Reduction

$$\text{Var}(\tilde{Y}) = \text{Var}(Y)(1 - \rho^2_{Y,X})$$

Where $\rho_{Y,X}$ is the **correlation between Y and X**.

$$\text{Variance Reduction} = \rho^2_{Y,X} \times 100\%$$

**This is the key formula:** if pre-experiment and in-experiment metrics are 70% correlated, we reduce variance by **49%** — cutting required sample size nearly in half!

---

## Variance Reduction Impact on Sample Size

The required sample size scales with variance:

$$n_{CUPED} = n_{original} \times (1 - \rho^2)$$

| Correlation $\rho$ | Variance Reduction | Sample Size Reduction |
|---|---|---|
| 0.3 | 9% | 9% |
| 0.5 | 25% | 25% |
| 0.7 | 49% | 49% |
| 0.8 | 64% | 64% |
| 0.9 | 81% | 81% |

> LinkedIn reported **~50% variance reduction** on revenue metrics using CUPED (ρ ≈ 0.7 for weekly revenue).

---

## Worked Example

### Setting

> Testing a new recommendation feature on Airbnb.
> - Experiment metric: **bookings in the next 30 days** ($Y$)
> - Pre-experiment covariate: **bookings in the prior 30 days** ($X$)
> - $n$ = 10,000 per group

### Before CUPED

```
Control mean:   Y_C = 2.10 bookings
Variant mean:   Y_T = 2.25 bookings
Std deviation:  σ = 3.5
SE = 3.5 / sqrt(10000) = 0.035
Z = (2.25 - 2.10) / (0.035 * sqrt(2)) = 0.15 / 0.0495 = 3.03 → p = 0.0024
```

### After CUPED

```
Correlation ρ(Y, X) = 0.75
Variance reduction = 1 - 0.75² = 43.75%
σ_CUPED = 3.5 * sqrt(1 - 0.5625) = 3.5 * 0.75 = 2.625
SE_CUPED = 2.625 / sqrt(10000) = 0.02625
Z_CUPED = 0.15 / (0.02625 * sqrt(2)) = 0.15 / 0.0371 = 4.04 → p = 0.000053
```

**Same experiment, same users — but 3× better Z-statistic and significantly lower p-value.** Or equivalently: to achieve the same power, we'd need 43.75% fewer users.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

def cuped_adjustment(Y_control, Y_treat, X_control, X_treat):
    """
    CUPED variance reduction for A/B test analysis.
    Y: in-experiment metric
    X: pre-experiment covariate
    """
    # Pool pre-experiment data to estimate theta
    X_all = np.concatenate([X_control, X_treat])
    Y_all = np.concatenate([Y_control, Y_treat])
    
    # Compute theta = Cov(Y,X) / Var(X)
    theta = np.cov(Y_all, X_all)[0, 1] / np.var(X_all, ddof=1)
    
    # Adjust metrics (subtract off what X predicts, centered at grand mean)
    X_grand_mean = np.mean(X_all)
    Y_control_adj = Y_control - theta * (X_control - X_grand_mean)
    Y_treat_adj   = Y_treat   - theta * (X_treat   - X_grand_mean)
    
    # Run Welch's t-test on adjusted metrics
    t_stat, p_val = stats.ttest_ind(Y_treat_adj, Y_control_adj, equal_var=False)
    
    # Variance reduction
    var_reduction = 1 - np.var(Y_control_adj) / np.var(Y_control)
    
    return {
        'theta': theta,
        'mean_diff': np.mean(Y_treat_adj) - np.mean(Y_control_adj),
        'p_value': p_val,
        'variance_reduction_pct': var_reduction * 100
    }
```

---

## Choosing a Good Covariate $X$

### Requirements

1. **Correlated with the outcome** — higher $\rho$ = more variance reduction
2. **Pre-experiment** — must be measured before randomization to avoid endogeneity
3. **Not affected by treatment** — otherwise $\theta$ is biased

### Common Covariates by Domain

| Domain | Metric (Y) | Good Covariate (X) |
|---|---|---|
| E-commerce | Revenue this week | Revenue last week/month |
| Streaming | Watch time this week | Watch time last week |
| Ride-sharing | Trips this week | Trips last week |
| SaaS | Engagement score | Prior engagement score |
| Ads | CTR this week | CTR last 30 days |

---

## CUPED vs. Stratified Sampling

| | CUPED | Stratified Sampling |
|---|---|---|
| When applied | Post-randomization (analysis) | Pre-randomization (design) |
| Mechanism | Regression adjustment | Ensures balance across strata |
| Requires pre-experiment data? | ✅ Yes | ❌ No |
| Implementation | Analysis-time | Randomization-time |
| Variance reduction | Often higher | Moderate |
| Can combine? | ✅ Yes — both together | ✅ Yes |

---

## Other Variance Reduction Techniques

### 1. Winsorization
Cap extreme values at the $p$-th percentile. Reduces variance from outliers (e.g., revenue per user).

```python
from scipy.stats import mstats
Y_winsorized = mstats.winsorize(Y, limits=[0.01, 0.01])  # 1st and 99th percentile
```

### 2. Ratio Metrics (Delta Method)
For metrics like "revenue per session" = total_revenue / total_sessions, use the delta method to compute the correct SE:

$$\text{SE}\left(\frac{\bar{Y}}{\bar{X}}\right) \approx \frac{\bar{Y}}{\bar{X}} \sqrt{\frac{\text{Var}(Y)/\bar{Y}^2 + \text{Var}(X)/\bar{X}^2 - 2\text{Cov}(Y,X)/(\bar{Y}\bar{X})}{n}}$$

### 3. Trimming / Filtering
Remove non-activated users (users who never saw the treatment). Reduces noise from users unaffected by the change — but be careful of introducing selection bias.

---

## 💬 FAANG Interview Questions & Answers

**Q1: What is CUPED and why is it useful?**

> A: CUPED is a variance reduction technique for A/B tests that uses pre-experiment data to reduce the noise in outcome metrics. The key insight is that if a user's past behavior (e.g., last week's revenue) is correlated with their current behavior, we can subtract out the predictable part and focus on the residual variation — which is much smaller. This increases the effective signal-to-noise ratio, giving us more statistical power without collecting more data. With a pre-experiment correlation of 0.7, we can reduce variance by 49% — nearly halving the required sample size.

**Q2: CUPED adjusts for pre-experiment data. Could you just include X as a covariate in a regression?**

> A: Yes — CUPED is mathematically equivalent to including X as a covariate in a linear regression of Y on the treatment indicator and X. The treatment coefficient gives the same adjusted effect estimate. CUPED is essentially "OLS covariate adjustment" reframed for experimentation. The regression approach also handles multiple covariates naturally.

**Q3: What if your pre-experiment covariate has many zeros (e.g., new users who never purchased before)?**

> A: New users with no prior activity have X = 0, making the covariate uninformative for them. Options: (1) Use the covariate only for returning users and run a separate analysis for new users. (2) Use a different covariate for new users (e.g., time since account creation, device type). (3) Use a categorical variable: "active in prior period" (0/1) as a covariate — at least removes some between-group variance.

**Q4: Netflix says they reduced experiment runtime by 50% using CUPED. How?**

> A: If variance reduction is $\rho^2$, then sample size requirement drops by the same factor (since $n \propto \sigma^2$ in the sample size formula). With $\rho = 0.7$ → 49% variance reduction → 49% fewer users needed → roughly half the runtime for the same power. Netflix achieves this by using the prior week's viewing behavior as a covariate for current-week watch time — highly correlated.

---

## 🚨 Common Pitfalls

1. **Using a post-experiment covariate** — must be pre-treatment; otherwise adjustment induces bias.
2. **Forgetting to center $X$** — the CUPED adjustment should use $X_i - \bar{X}$ to keep the adjusted mean interpretable.
3. **Assuming high $\rho$ always** — if the product recently changed or users are mostly new, prior behavior may not correlate well.
4. **Overcomplicating with many covariates without regularization** — can overfit in small samples.
5. **Not validating the balance check** — still verify that $\bar{X}_T \approx \bar{X}_C$ (randomization check).

---

*← [17 — Bayesian Testing](17_bayesian_hypothesis_testing.md) | [19 — Causal Inference →](19_causal_inference.md)*
