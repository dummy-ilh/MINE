# Chapter 12: Variance Reduction (CUPED)

## 1. Intuition

You already know CUPED from your regression/DOE background — this chapter reframes it specifically for the A/B testing interview context, where it's usually asked as "how do you make experiments faster without more traffic," and the interviewer wants to see the mechanism, not just the acronym.

The core idea, restated in A/B testing language: **most of the variance in a user's outcome metric during the experiment is explained by how that user already behaved *before* the experiment even started.** A user who spent $50/month before the experiment is likely to spend around $50/month during it, regardless of treatment. That pre-experiment variation is "noise" from the experiment's perspective — it has nothing to do with the treatment — but it's inflating your variance and burying the actual treatment signal. CUPED's whole job is to strip that pre-existing, treatment-irrelevant variance out.

This is the direct generalization of the **paired t-test** idea from Chapter 6 — instead of requiring an exact "before/after on the same unit" pairing, CUPED uses *regression* to control for any pre-experiment covariate, even an imperfectly correlated one.

## 2. The Formal Method

**CUPED (Controlled-experiment Using Pre-Experiment Data)** adjusts your outcome metric $Y_i$ using a pre-experiment covariate $X_i$ (typically the same metric, measured before the experiment started):

$$Y_i^{CUPED} = Y_i - \theta(X_i - \bar{X})$$

where $\theta$ is chosen to minimize the variance of $Y_i^{CUPED}$. The optimal $\theta$ is:

$$\theta^* = \frac{Cov(Y,X)}{Var(X)}$$

which you'll recognize as exactly the OLS regression coefficient of $Y$ on $X$ — CUPED is, mechanically, just regression adjustment using a pre-experiment covariate, dressed up with a specific name in the experimentation literature.

**Why the adjustment doesn't bias your ATE estimate**: since $E[X_i - \bar{X}] = 0$ by construction, subtracting $\theta(X_i-\bar{X})$ doesn't shift the mean of $Y_i^{CUPED}$ relative to $Y_i$ — it only reduces variance around that same mean. This is the crucial property: **CUPED reduces variance without introducing bias**, provided $X_i$ is measured strictly *before* the experiment (so it can't itself be affected by treatment assignment — using a post-treatment covariate here would be a serious error, introducing exactly the kind of bias CUPED is supposed to avoid).

**The variance reduction achieved**:

$$Var(Y^{CUPED}) = Var(Y)(1-\rho^2)$$

where $\rho$ is the correlation between $Y$ (the outcome) and $X$ (the pre-experiment covariate). This is the single number to memorize: **CUPED reduces variance by a factor of $(1-\rho^2)$**, so a correlation of $\rho=0.5$ gives you a 25% variance reduction, $\rho=0.7$ gives ~51% reduction, $\rho=0.9$ gives ~81% reduction.

## 3. Connecting Variance Reduction to Sample Size / Power

Recall from Chapter 5: sample size is proportional to variance (in the numerator of the sample size formula). If CUPED reduces variance by $(1-\rho^2)$, it reduces your **required sample size by exactly that same factor** — this is the concrete payoff that makes CUPED so valuable in production: **you get the same statistical power with less data or less time**, without changing your MDE, $\alpha$, or desired power at all.

$$n_{CUPED} = n_{original} \times (1-\rho^2)$$

If $\rho=0.7$ between pre-experiment and during-experiment spend (a very realistic correlation for an engaged user base), you cut your required sample size roughly in half — turning, e.g., a 4-week experiment into a 2-week one, at zero cost to your Type I/II error rates.

## 4. Worked Example

You're testing a new feature intended to increase weekly spend. You have 2,000 users per arm.

- Without CUPED: $Var(Y) = 400$ (weekly spend variance, in $\text{dollars}^2$)
- Pre-experiment weekly spend (4 weeks before launch, same users) has correlation $\rho=0.6$ with during-experiment weekly spend.

**Variance after CUPED adjustment**:
$$Var(Y^{CUPED}) = 400 \times (1-0.6^2) = 400 \times 0.64 = 256$$

That's a 36% variance reduction. Translating to standard error terms ($SE \propto \sqrt{Var}$):

$$SE_{original} \propto \sqrt{400}=20, \quad SE_{CUPED} \propto \sqrt{256}=16$$

A 20% reduction in standard error — meaning your confidence intervals are 20% tighter and your test is meaningfully more powered, using the exact same 2,000 users per arm, purely by incorporating 4 weeks of pre-experiment data you likely already had sitting in your data warehouse.

**Equivalent sample size framing**: to hit the same power/precision *without* CUPED, you'd need $n_{original}=n_{CUPED}/(1-\rho^2) = n_{CUPED}/0.64 \approx 1.56\times n_{CUPED}$ — i.e., you'd need 56% more users to match what CUPED achieves "for free" with the existing 2,000/arm.

## 5. Production Considerations

- **Choosing the covariate $X$**: the best covariate is almost always the **same metric, measured pre-experiment**, since it typically has the highest correlation with the during-experiment outcome. For new metrics with no pre-experiment history (e.g., a brand-new user or a brand-new metric), you can use correlated proxy covariates instead (e.g., account age, historical engagement on a related metric) — the variance reduction will be smaller (lower $\rho$) but still positive.
- **CUPED requires the covariate to be available for every unit, unaffected by treatment.** New users with no pre-experiment history are a real practical wrinkle — common solutions include using the overall population mean as their covariate value (effectively giving them zero adjustment) or running CUPED only on the returning-user subpopulation and handling new users separately.
- **CUPED generalizes naturally to ratio metrics (Chapter 11)** by extending the delta-method variance formula to include the covariate — this is a genuinely more involved derivation, worth namedropping as "CUPED for ratio metrics needs the delta method variance extended with the covariate's covariance terms" if asked to go deeper, without necessarily deriving the full formula live.
- **CUPED vs. stratification vs. regression adjustment more broadly**: CUPED is a special, simple case of the more general technique of "covariate adjustment for variance reduction" — post-stratification (Chapter 9's broader methods) and full regression-adjusted estimators (which can include multiple covariates, not just one) are close cousins, and mentioning this family relationship shows you see CUPED as one instance of a general principle, not an isolated trick.

## 6. Interview Traps

- **Trap #1**: Using a covariate that could itself be affected by treatment (e.g., using *during-experiment* week-1 spend as the covariate to predict during-experiment week-4 spend) — this introduces bias, since the "covariate" is no longer strictly pre-treatment and can be causally downstream of the treatment itself.
- **Trap #2**: Not being able to state the $(1-\rho^2)$ variance reduction formula, or not connecting it back to the sample-size formula from Chapter 5 to make the practical payoff concrete.
- **Trap #3**: Presenting CUPED as if it somehow "increases your effect size" — it doesn't change the true treatment effect or the point estimate's expected value at all; it only tightens the precision (variance) around the same true estimate.
- **Trap #4**: Not having a plan for new users with no pre-experiment history — this is a very common practical follow-up question ("what do you do with users who don't have a covariate value?").

## 7. L5-Differentiating Talking Points

- Deriving $\theta^*=Cov(Y,X)/Var(X)$ and recognizing it as exactly the OLS regression coefficient connects this chapter directly to your existing regression background — worth stating explicitly, since it signals you see CUPED as an application of familiar tools rather than a totally new, opaque technique.
- Immediately translating the $(1-\rho^2)$ variance reduction into a concrete "this means we can run the experiment in half the time" framing, with actual numbers, is exactly the kind of business-impact fluency L5 interviewers reward over reciting the formula alone.
- Proactively raising the new-user/missing-covariate problem, and having a real answer for it, shows you've thought about CUPED as a production system, not just a clean textbook derivation.
- Framing CUPED as one member of a broader family (stratification, general regression adjustment, multiple covariates) shows conceptual range beyond the single named technique.

## 8. Comprehension Check

1. Derive (or state) the optimal $\theta^*$ for CUPED and explain why it's mathematically identical to an OLS regression coefficient.
2. Why doesn't CUPED introduce bias into your treatment effect estimate, despite adjusting every unit's outcome value?
3. If the correlation between your pre-experiment and during-experiment metric is $\rho=0.8$, what percentage variance reduction does CUPED achieve, and what does that imply for required sample size?
4. Why would using a *during-experiment* covariate (rather than pre-experiment) be a serious mistake in CUPED?
5. A new feature launch has no pre-experiment history for the metric being tested (it's a brand new metric). Can you still use CUPED, and if so, how would you choose a covariate?

---
*Next: Chapter 13 — Proxy Metrics for Long-Term Outcomes*
