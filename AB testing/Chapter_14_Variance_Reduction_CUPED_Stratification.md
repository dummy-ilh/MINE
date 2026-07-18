# Chapter 14: Variance Reduction — CUPED & Stratified Sampling

## 1. Definition

Variance reduction techniques lower the noise in your metric estimate *without* requiring more users — increasing statistical power (or shrinking required sample size) for the same underlying data.

- **CUPED (Controlled-experiment Using Pre-Experiment Data):** uses a user's pre-experiment behavior (a covariate correlated with the outcome metric) to adjust/"explain away" some of the variance in the outcome metric, isolating the treatment effect more precisely. Popularized by Microsoft's experimentation team (Deng et al.).
- **Stratified sampling:** splits the population into homogeneous strata (subgroups) based on a known characteristic before randomizing, ensuring balanced representation across strata in both arms and analyzing variance within-strata (which is smaller than overall variance if strata are well-chosen).

## 2. Layman Explanation

Imagine you're trying to measure whether a new workout program improves people's running speed. Some people were already fast runners before the program started; others were slow. If you just compare average speed between the "program" group and the "no program" group without accounting for pre-existing fitness, a lot of the variation you see is just pre-existing differences in fitness — noise that has nothing to do with the program's actual effect.

**CUPED** says: "let's measure everyone's running speed *before* the program starts, then use that baseline to adjust each person's post-program speed." If someone was already fast before and stayed fast, that's not attributed to the program — you're isolating the *change* attributable to treatment, not conflating it with pre-existing differences. This dramatically shrinks the noise in your estimate, because you've subtracted out variance that was never about the treatment to begin with.

**Stratified sampling** is like making sure your "program" and "no program" groups each have the same mix of fast and slow runners from the start (instead of hoping random assignment happens to balance them by luck) — then when you analyze results, you compare within each speed-tier separately (fast vs. fast, slow vs. slow) rather than lumping everyone together, which also reduces noise since you're not comparing apples across very different starting points.

## 3. Formal Explanation

**CUPED formula:**

Let Y = outcome metric during the experiment, X = the same (or a related) metric measured in the pre-experiment period. Define the CUPED-adjusted metric:

Y_cuped = Y - θ(X - X̄)

Where θ = Cov(Y,X) / Var(X), chosen to minimize the variance of Y_cuped.

**Why this works:**
Var(Y_cuped) = Var(Y) - θ²Var(X) = Var(Y) - Cov(Y,X)²/Var(X) = Var(Y)(1 - ρ²)

where ρ is the correlation between the pre-experiment covariate X and the outcome Y. The variance reduction is directly proportional to ρ² — a covariate correlated at ρ=0.5 with the outcome reduces variance by 25%; ρ=0.7 reduces variance by ~49%.

**Key property — CUPED doesn't change the expected treatment effect:**
Because θ is chosen based on the *pre-experiment* period (before treatment assignment), and X is unaffected by treatment (it was measured before treatment even started), subtracting θ(X - X̄) doesn't bias the estimated treatment effect — it only reduces variance. This is what makes it safe to apply without threatening the experiment's validity.

**Choosing the right covariate:**
The best covariate is the same metric measured pre-experiment (e.g., last month's conversion rate to predict this month's conversion rate) — the higher the correlation between pre- and post-period values, the more variance reduction you get. For new users with no pre-experiment history, CUPED can't be applied directly (no X exists), which is a known limitation.

**Stratified sampling formula (variance under stratification):**

Var(Ȳ_stratified) = Σ (Nₕ/N)² × (Var(Yₕ)/nₕ)

summed across strata h, where Nₕ is the stratum's population size and Var(Yₕ) is the within-stratum variance. If strata are chosen so that within-stratum variance is much smaller than the overall population variance (i.e., the stratifying variable explains a lot of the variance), this weighted sum is smaller than the unstratified variance estimate.

**CUPED vs. stratification — relationship:**
Both techniques exploit the same underlying idea: use a known, pre-treatment characteristic correlated with the outcome to strip out variance unrelated to treatment. CUPED does this via regression adjustment post-hoc (after data collection); stratification does it via the randomization design itself (before data collection). They can be combined — stratify at randomization time AND apply CUPED using a different or additional covariate at analysis time.

## 4. Levers — What Controls It, What Moves It

**Correlation strength (ρ) between covariate and outcome**
- Higher ρ → more variance reduction from CUPED. This is why picking a *good* covariate matters enormously — a weakly correlated covariate (ρ near 0) gives negligible benefit, while the same historical metric (e.g., last month's value of the exact outcome metric) often gives ρ in the 0.5-0.8 range for stable user behaviors.

**Availability of pre-experiment history**
- Users with long tenure have richer pre-experiment data, making CUPED very effective for them; new users (no history) get no benefit from CUPED directly — sometimes handled by using a proxy covariate (e.g., signup cohort characteristics) or accepting that CUPED's benefit is concentrated in the existing-user segment.

**Number and quality of strata**
- Good stratification variables are ones highly correlated with the outcome and stable pre-treatment (e.g., past purchase tier, geographic region, platform). Poorly chosen strata (weakly related to outcome) barely reduce variance and add unnecessary complexity to the design and analysis.

**Combining techniques**
- Using both stratified randomization AND CUPED adjustment compounds the benefit if the covariates used for each aren't perfectly redundant with each other — e.g., stratify by platform (categorical), then CUPED-adjust using pre-period continuous engagement score.

## 5. Worked Example

Suppose your outcome metric Y (this month's revenue per user) has Var(Y) = 400 (i.e., sd=20). You have a pre-experiment covariate X (last month's revenue per user) with Var(X) = 350, and Cov(Y,X) = 280.

Correlation: ρ = Cov(Y,X)/√(Var(Y)·Var(X)) = 280/√(400×350) = 280/374.2 ≈ 0.748

θ = Cov(Y,X)/Var(X) = 280/350 = 0.8

Variance reduction: Var(Y_cuped) = Var(Y)(1-ρ²) = 400 × (1 - 0.748²) = 400 × (1 - 0.560) = 400 × 0.44 = 176

That's a reduction from 400 to 176 — a **56% reduction in variance**. Recall from Chapter 9 that required sample size scales linearly with variance for a fixed MDE — so this CUPED adjustment effectively cuts your required sample size by more than half, without collecting a single additional user. A test that would have needed 8 weeks to reach sufficient power might now only need about 3.5 weeks with the same traffic, purely from this adjustment.

## 6. Famous Q&A (Google / Apple style)

**Q: Your team is under pressure to detect smaller effects without extending test duration. How would CUPED help, and what data do you need?**
A: CUPED reduces the variance of your outcome metric by adjusting for a pre-experiment covariate correlated with that outcome — since required sample size scales directly with variance, cutting variance in half (achievable with a covariate correlated around ρ=0.7) effectively cuts required sample size in half too, letting you either detect smaller effects in the same duration or hit your existing MDE faster. The key requirement is having reliable pre-experiment data for the same or a closely related metric, ideally for most of your user base — new users without history won't benefit from this adjustment directly.

**Q: Why doesn't applying CUPED bias your estimated treatment effect, even though you're adjusting the outcome metric using extra data?**
A: Because the covariate X is measured strictly in the pre-experiment period, before treatment assignment even happens — treatment couldn't possibly have affected X, so subtracting θ(X - X̄) doesn't remove any part of the true treatment effect, it only removes variance that was already present before the experiment started and is unrelated to treatment. The adjustment is mean-preserving by construction (E[Y_cuped] = E[Y] since E[X - X̄] = 0), which is exactly why it's considered a safe, unbiased variance-reduction technique rather than a risky manipulation of the data.

**Q: You want to apply CUPED, but 40% of users in your experiment are brand new and have no pre-experiment history. What do you do?**
A: I'd handle new and existing users somewhat separately — for existing users with history, apply CUPED using their own pre-period data as usual; for new users, since there's no X available, you'd either fall back to the raw (non-adjusted) outcome for that segment, or use a coarser proxy covariate that IS available for new users (e.g., signup channel, initial onboarding behavior) if it's meaningfully correlated with the outcome. It's also worth reporting the treatment effect separately for new vs. existing users regardless, since the underlying dynamics (and the degree of variance reduction achievable) genuinely differ between these populations.

**Q: How does stratified sampling differ from CUPED in terms of when each intervention happens, and can you use both together?**
A: Stratification happens at randomization time — you split the population into subgroups based on a known pre-treatment characteristic and ensure balanced allocation to treatment/control within each subgroup, which is a design-time intervention. CUPED happens at analysis time — after data collection, you use a pre-treatment covariate to adjust the outcome metric via regression, regardless of how randomization was originally done. Yes, you can combine both: stratify on a categorical variable (e.g., platform: iOS/Android/web) at randomization, then apply CUPED using a continuous covariate (e.g., pre-period engagement score) at analysis — as long as the two covariates aren't fully redundant with each other, the variance reductions can compound.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Using a post-treatment variable as the CUPED covariate (X must be strictly pre-experiment, or you risk biasing the treatment effect estimate)
- ❌ Assuming CUPED works equally well for all users — new users with no pre-period history get little to no benefit
- ❌ Picking a stratification variable weakly related to the outcome — wastes design complexity for minimal variance reduction
- ❌ Forgetting that θ should be estimated from the data (Cov(Y,X)/Var(X)), not just assumed or guessed
- ✅ Do: check the correlation (ρ) between your chosen covariate and the outcome before expecting meaningful variance reduction — the benefit scales with ρ²
- ✅ Do: consider combining stratified randomization design with CUPED analysis-time adjustment for compounding benefits

---
*Next: Chapter 15 — Multiple Testing Correction (Bonferroni, FDR) Across Many Metrics.*
