# Chapter 3: Randomized Experiments (RCTs) — Why They Work, and Where They Break

## 1. Explanation

### The core mechanical reason randomization works

If you flip a coin to decide treatment $D$ for each unit, the coin has *nothing to do with* the unit's underlying characteristics — including their potential outcomes $Y(0), Y(1)$. Statistically, this means:
```
(Y(0), Y(1)) ⊥ D
```
"Independent of." This is the strongest possible version of the **ignorability** assumption you'll meet again in observational methods (Chapter 4) — except here it's *guaranteed by design*, not assumed and argued for after the fact.

Once independence holds, recall the decomposition from Chapter 1:
```
E[Y|D=1] − E[Y|D=0] = ATT + Selection bias
```
where Selection bias = E[Y(0)|D=1] − E[Y(0)|D=0]. Under randomization, D is independent of Y(0), so:
```
E[Y(0)|D=1] = E[Y(0)|D=0] = E[Y(0)]
```
The selection bias term becomes exactly zero. And from Chapter 2, under randomization ATT = ATC = ATE. So:
```
E[Y|D=1] − E[Y|D=0] = ATE  (exactly, in expectation)
```
This is why RCTs are called the "gold standard" — not because they're magic, but because they mechanically satisfy, by construction, the one assumption every other method has to argue for indirectly and can never fully verify.

### Statistical inference: how much can you trust a given estimate?

Randomization gives you an *unbiased* estimator, but any single experiment still has sampling noise. The variance of the difference-in-means estimator is:
```
Var(τ̂) = σ²_1/n_1 + σ²_0/n_0
```
where $\sigma^2_1, \sigma^2_0$ are the outcome variances in each arm and $n_1, n_0$ are the arm sizes. Two intuitive takeaways: (1) more samples per arm shrinks variance — this is why experiments need large N; (2) a noisier/more variable outcome metric needs more samples to pin down with the same precision — this is why choosing a lower-variance metric (or applying variance reduction like CUPED) can be as valuable as collecting more users.

### Sample size / power calculation, built up conceptually

Before running an experiment, you want to know: "how many users do I need to reliably detect a lift of a given size?" The standard formula (two-sided test, equal allocation across arms):
```
n per arm ≈ 2 · (z_{α/2} + z_β)² · σ² / δ²
```
- $z_{\alpha/2}$: how extreme a result needs to be to call it "significant" at your chosen false-positive rate α (≈1.96 for α=0.05)
- $z_\beta$: how much of a safety margin you want against missing a real effect — governs your desired power (≈0.84 for 80% power)
- $\sigma^2$: outcome variance (noisier metric → need more data)
- $\delta$: minimum detectable effect (MDE) — the smallest lift you actually care about catching

Notice $\delta$ is squared in the denominator — halving the MDE you want to detect *quadruples* the required sample size. This nonlinearity is one of the most common "gotcha" facts interviewers probe, because it explains why detecting small effects is disproportionately expensive.

### Where RCTs still break, even though the core logic (independence) holds

The independence assumption is guaranteed *at the moment of assignment* — but things can go wrong afterward that reintroduce exactly the bias randomization was supposed to prevent:

- **Interference/SUTVA violations** (full chapter later) — a shared environment (ad auction, feed ranking, marketplace supply) means "control" isn't actually a clean, unaffected baseline once treatment changes the shared environment.
- **Differential attrition** — if treated units are more likely to *drop out of the measured dataset* (e.g., a buggy feature causes churn/uninstall before the outcome is ever measured), the *remaining* treated sample is no longer the same random draw it started as — this is "post-randomization selection," and it silently reintroduces the very bias RCTs are supposed to eliminate, even though the initial coin flip was fair.
- **Non-compliance** — some units assigned to treatment don't actually take it (or vice versa). Comparing "as received" (per-protocol) instead of "as assigned" (intent-to-treat) reintroduces selection bias, because *who complies* is rarely random. ITT analysis avoids this bias but at the cost of diluting/understating the effect for those who do comply — IV/LATE machinery (Chapter 7) is the principled way to recover the complier-specific effect.
- **Hawthorne/novelty effects** — the mere fact of experiencing *something new or being observed* changes behavior temporarily, independent of the feature's true long-run value; a spike in week 1 that fades is a classic signature.
- **Sample Ratio Mismatch (SRM)** — if the realized split between arms deviates meaningfully from the intended split (e.g., 52/48 instead of 50/50, in a way too large to be chance), it's a red flag that the randomization mechanism itself is broken (a logging bug, bot filtering asymmetry, a caching issue) — and if the *mechanism* is broken, you can no longer trust that assignment was truly independent of potential outcomes, undermining everything downstream.

## 2. Example

### Example A — Power/sample size calculation

You're running an A/B test on click-through rate (CTR). Baseline CTR is roughly p=0.10, so this is a Bernoulli-like outcome with variance $\sigma^2 = p(1-p) = 0.10 \times 0.90 = 0.09$. You want to detect a minimum lift of δ=0.01 (1 percentage point) at the standard 95% significance / 80% power settings ($z_{\alpha/2}=1.96$, $z_\beta=0.84$).

```
n per arm ≈ 2 × (1.96 + 0.84)² × 0.09 / (0.01)²
          = 2 × (2.80)² × 0.09 / 0.0001
          = 2 × 7.84 × 0.09 / 0.0001
          = 1.4112 / 0.0001
          = 14,112 users per arm
```
So you'd need **~14,112 users per arm (~28,224 total)** to reliably detect a 1 percentage-point CTR lift.

**Now watch what happens if you want to detect half that effect (δ=0.005 instead of 0.01):**
```
n per arm ≈ 2 × 7.84 × 0.09 / (0.005)² = 1.4112 / 0.000025 = 56,448 users per arm
```
Halving the MDE **quadrupled** the required sample size (14,112 → 56,448) — exactly as the $\delta^2$ in the denominator predicts. This is worth internalizing numerically, not just abstractly: it's why teams often accept a larger MDE (settle for detecting bigger effects only) when traffic is limited, rather than run underpowered tests hoping to catch small effects.

### Example B — Attrition bias, worked through concretely

1,000 users randomized 50/50 to a new "aggressive session timeout" feature (D=1) vs old behavior (D=0). Suppose the feature is buggy and causes 15% of treated users to uninstall the app entirely within a day (vs. only 2% baseline uninstall in control) — and uninstalled users are *disproportionately* the ones who would have had low engagement anyway (frustrated, marginal users bail first).

Tracking what happens to each arm:
- **Control**: starts at 500 users, loses 2% (10 users) to uninstall → 490 users remain, and since dropout is small and roughly random, these 490 are still a fairly representative random subsample of the original 500.
- **Treatment**: starts at 500 users, loses 15% (75 users) to uninstall → 425 users remain — but because the 75 who left were skewed toward the *lowest-engagement* users, the **425 who remain are a higher-engagement-skewed subsample** than a genuinely random 425 would be.

If you now compute "average session length among users still active after 1 week," you're comparing:
- Control: ~490 users, close to the original random draw (unbiased comparison group)
- Treatment: ~425 users, but selectively filtered to exclude exactly the low-engagement types who would have dragged the average down

This means the treatment arm's average session length will look **artificially inflated** relative to the true effect — not because the feature genuinely improved sessions, but because the users who would have shown the worst outcomes were removed from the treatment arm's measured pool (while a comparable removal didn't happen in control). Even though the original 50/50 coin flip was completely fair, this **post-randomization attrition** silently reintroduces selection bias.

**The fix, concretely:** report intent-to-treat (ITT) using ALL 1,000 originally-assigned users, imputing a conservative value (e.g., session length = 0, or last-observed-value-carried-forward) for anyone who left, rather than restricting the analysis to "survivors only." This keeps the comparison anchored to the original, valid random assignment.

## 3. Interview Q&A

**Q: Why does "independence of D and potential outcomes" matter more than "independence of D and observed covariates"?**
A: Because it's the potential outcomes (the things you're trying to compare) that must be balanced across arms — covariate balance is just an observable *proxy check* for this (if age/tenure/etc. are balanced, it's suggestive evidence randomization worked), but the actual guarantee from a valid random assignment mechanism is at the level of potential outcomes, observed or not, measured or not.

**Q: Your RCT has balanced observed covariates between arms (age, tenure, geography all look similar) but you're still suspicious something's wrong. What else would you check?**
A: Check the randomization *unit and mechanism* itself (was it truly random, e.g., hash-based, or was there a bug/non-random assignment logic?), check sample ratio mismatch (SRM — are the arm sizes close to the intended split?), and check for differential attrition between arms (are dropout rates similar, and is dropout correlated with the outcome?).

**Q: What is "Sample Ratio Mismatch" (SRM) and why do experienced experimenters check it as step one, before looking at any treatment effect?**
A: SRM is when the actual traffic split between arms deviates significantly (via a chi-square test) from the intended split (e.g., expecting 50/50 but observing 52/48 with high statistical significance given the sample size). It's checked first because if SRM is present, it's strong evidence the randomization itself is broken (a bug in assignment, differential bot filtering, a caching issue) — and if randomization is broken, none of the downstream causal conclusions can be trusted, no matter how "significant" the treatment effect looks. It's a pre-condition check, not just another metric.

**Q: A feature causes a subset of treated users to crash and get automatically excluded from your analytics pipeline (their events never get logged). How does this bias your ITT estimate, and in which direction?**
A: This is attrition bias baked into the "observed" data itself — if crashing correlates with the outcome you'd have measured (e.g., crashers likely would have had poor engagement, or possibly the opposite — power users who push the app harder crash more), you lose exactly the units whose outcome would inform you the most, and the remaining treated users become a non-random, biased subsample. The direction of bias depends on who crashes: if crashers skew toward users who would have had low engagement, the remaining treated group looks artificially better than the true full effect (upward bias) — always investigate crash correlates before trusting the topline number.

**Q: How would you distinguish a novelty effect from a true, durable effect in a live experiment?**
A: Look at the treatment effect **trend over time** (e.g., a weekly or daily breakdown) rather than a single pooled number for the whole experiment window — a novelty effect typically shows a spike in the first days/week that decays toward zero over subsequent weeks, while a genuine effect is stable or grows. Running the experiment longer and plotting an event-study-style time series of the treatment effect is the standard diagnostic for this.

**Q: If halving your minimum detectable effect quadruples required sample size, what practical levers do you have besides "just collect more users" when traffic is limited?**
A: Increase the traffic *allocation* to make arms more balanced if currently skewed (e.g., 50/50 gives more power than 90/10 for the same total N); apply variance reduction techniques like CUPED (using a pre-experiment covariate to shrink outcome variance without introducing bias, since it's measured before treatment); choose a less noisy proxy metric if a reasonable one exists; or explicitly accept a larger MDE and communicate that smaller true effects won't be reliably detected at the available sample size, rather than silently underpowering the test.

**Q: Contrast intent-to-treat (ITT) and per-protocol analysis, and explain why ITT is usually the default headline number.**
A: ITT analyzes units according to their *assigned* group regardless of whether they actually complied with/received treatment; per-protocol only analyzes those who actually complied. ITT preserves the unbiasedness guarantee from randomization, since assignment (not actual behavior) is what was randomized. Per-protocol analysis implicitly compares self-selected compliers to everyone else, reintroducing selection bias, because compliance itself is rarely random (e.g., people who comply with a health intervention may be systematically healthier or more motivated). ITT is the safe, unbiased default; per-protocol/LATE-style analysis is a secondary, assumption-heavy lens.

---
**Previous: Chapter 2 — Causal Estimands**
**Next: Chapter 4 — Confounding, DAGs, and the Backdoor Criterion**
