# Day 27 — A/B Testing: The Complete ML Practitioner's Guide
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Kohavi et al., *Trustworthy Online Controlled Experiments*; Deng et al. (Microsoft)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why A/B Testing is the #1 DS Interview Topic

Every major tech company — Google, Meta, Amazon, Netflix, Airbnb, Uber — runs thousands of A/B tests simultaneously. It is the primary method for making data-driven product decisions.

| Company | A/B Tests Per Year |
|---|---|
| Google | 10,000+ |
| Microsoft (Bing) | 1,000+ |
| Netflix | 250+ per quarter |
| Amazon | 1,000+ |
| Facebook/Meta | 10,000+ |

In DS/ML interviews at these companies, A/B testing questions are near-universal. You WILL be asked about it.

---

## 2. What is an A/B Test?

An **A/B test** (randomized controlled experiment) compares:
- **Control (A):** current system / no change
- **Treatment (B):** proposed change (new model, new feature, new UI)

Users are randomly assigned to A or B. After sufficient time, compare a **metric** of interest.

### Why Randomization?

Randomization ensures that any observed difference in the metric is caused by the treatment, not by confounding factors (e.g., heavy users vs light users, different time zones).

Without randomization: selection bias, confounding, spurious results.

---

## 3. The Complete A/B Testing Pipeline

```
Step 1: Define hypothesis and metric
Step 2: Power analysis → determine sample size
Step 3: Randomize assignment (A/A test to validate)
Step 4: Run experiment for required duration
Step 5: Check for SRM (Sample Ratio Mismatch)
Step 6: Statistical analysis (test statistic, p-value, CI)
Step 7: Multiple testing correction (if needed)
Step 8: Decision: ship, iterate, or abandon
```

---

## 4. Step 1: Hypothesis and Metric Selection

### Primary Metric (OEC — Overall Evaluation Criterion)

```
H₀: δ = μ_B − μ_A = 0      (no effect)
H₁: δ ≠ 0  or  δ > 0       (effect exists)
```

**Choosing the right metric:**

| Metric Type | Examples | Pitfall |
|---|---|---|
| **Guardrail metrics** | Latency, crash rate, revenue | Must not degrade |
| **Primary metric** | CTR, conversion rate, DAU | Main decision metric |
| **Secondary metrics** | Session length, retention | Supporting evidence |
| **Proxy metrics** | Clicks → revenue | Correlation may not hold |

**Metric hierarchy:**
```
Revenue ← hard to move directly
    ↑
Conversion rate ← your primary metric (days to weeks)
    ↑
CTR ← leading indicator (hours to days)
    ↑
Engagement ← fastest signal (hours)
```

**Avoid vanity metrics** (feel good but don't correlate with business outcomes).

---

## 5. Step 2: Power Analysis and Sample Size

Before running: know how large an experiment you need.

### Required Inputs

```
α    = significance level (typically 0.05)
1−β  = desired power (typically 0.80 or 0.90)
δ    = minimum detectable effect (MDE) — smallest effect worth detecting
σ    = standard deviation of metric (estimated from historical data)
```

### Sample Size Formulas

**For means (continuous metric):**
```
n = 2(z_{α/2} + z_β)² σ²/δ²    [per arm, two-sample]
```

**For proportions (binary metric like CTR):**
```
n = (z_{α/2} + z_β)² [p_A(1−p_A) + p_B(1−p_B)] / δ²

Simplified (p_A ≈ p_B ≈ p):
n = 2(z_{α/2} + z_β)² p(1−p) / δ²
```

| z_{α/2} | z_β | Power |
|---|---|---|
| 1.960 (α=0.05) | 0.842 | 80% |
| 1.960 (α=0.05) | 1.282 | 90% |
| 2.576 (α=0.01) | 0.842 | 80% |

**Runtime = n / (daily traffic per arm)**

### Intuition for Sample Size

```
n ∝ σ²/δ²    [noise-to-signal ratio]

Doubling the MDE → 4× fewer samples needed
Halving the noise σ → 4× fewer samples needed
Higher power → more samples needed
Lower α → more samples needed
```

---

## 6. Step 3: Randomization

### Unit of Randomization

Choose the **randomization unit** carefully:

| Unit | When to Use | Pros | Cons |
|---|---|---|---|
| **User** (most common) | Long-term effects | Consistent experience | Network effects |
| **Session** | Short-term UI tests | More data faster | Same user sees both |
| **Page/Request** | Algorithm tests | Maximum data | Within-user correlation |
| **Cookie** | Anonymous users | Simple | Cookie deletion |
| **Device** | Mobile features | Natural unit | Multi-device users |

**General rule:** Randomize at the unit that prevents contamination between A and B.

### A/A Test

Before running A/B, run an **A/A test**: assign users to two identical control groups. Check:
- p-values should be Uniform(0,1) — no false positives
- Sample sizes should be equal
- Metric distributions should be identical

A/A tests validate your randomization and logging infrastructure.

---

## 7. Step 4: Experiment Duration

**Minimum duration:** at least 1–2 full weekly cycles (to capture weekly seasonality).

**Don't stop early** because of promising results — p-values fluctuate and "peeking" inflates false positive rate (Day 26).

### Novelty Effect

Users may react differently to new features simply because they're new. New features often show:
1. Initial boost (novelty)
2. Decay back to baseline
3. True long-term effect

Run experiments long enough to see past the novelty effect.

### Primacy Effect

Opposite of novelty: users initially resist change, then adapt. Metrics may initially look worse before improving.

---

## 8. Step 5: Sample Ratio Mismatch (SRM)

> **SRM:** The observed ratio n_B/n_A differs significantly from the expected ratio.

**Check:** If expecting 50/50 split and you observe 48%/52%, is this by chance?

```
Under H₀ (true 50/50): n_B ~ Binomial(n_total, 0.5)
Chi-squared test: χ² = (n_A − n_expected)²/n_expected + (n_B − n_expected)²/n_expected
```

**If SRM detected:** Do NOT analyze the experiment. Something is wrong with:
- Assignment mechanism
- Logging/tracking
- Bots or outliers
- Filter differences between A and B

SRM is the most common silent killer of A/B test validity.

---

## 9. Step 6: Statistical Analysis

### The Two-Sample z-test for Proportions

```
δ̂ = p̂_B − p̂_A

SE = √[p̂_A(1−p̂_A)/n_A + p̂_B(1−p̂_B)/n_B]

or pooled: SE = √[p̂(1−p̂)(1/n_A + 1/n_B)]

Z = δ̂/SE ~ N(0,1) under H₀

p-value = P(|Z| > |z_obs|) = 2(1−Φ(|z_obs|))

95% CI: δ̂ ± 1.96·SE
```

### Delta Method for Ratio Metrics

Many metrics are ratios: CTR = clicks/impressions. These require the delta method (Day 20):

```
For θ = f/g:
    θ̂ = f̄/ḡ
    Var(θ̂) ≈ [Var(f)/ḡ² − 2Cov(f,g)·f̄/ḡ³ + Var(g)·f̄²/ḡ⁴] / n
```

In practice, use bootstrapping or the delta method approximation.

---

## 10. Step 7: Multiple Testing in A/B Tests

### Multiple Metrics

Testing 10 metrics at α=0.05: expect 0.5 false positives on average under H₀. Use BH correction if exploratory; pre-specify primary metric if confirmatory.

### Multiple Variants (A/B/C/D tests)

Testing k variants vs control: Dunnett's test or Bonferroni (α* = α/(k−1) per comparison).

### Continuous Monitoring (Peeking)

Checking p-value every day and stopping when p<0.05 inflates Type I error substantially:

| Checks | True Type I Rate |
|---|---|
| 1 (planned) | 5% |
| 5 | 14.2% |
| 10 | 18.9% |
| 100 | 37.0% |

**Fix:** Sequential testing (alpha spending), SPRT (Sequential Probability Ratio Test), or always-valid p-values.

---

## 11. Bayesian A/B Testing Alternative

Instead of frequentist hypothesis testing:

```
Prior: p_A ~ Beta(α_A, β_A), p_B ~ Beta(α_B, β_B)

After experiment:
    P(p_B > p_A | data) — computed from posterior samples

Decision: ship if P(p_B > p_A) > threshold (e.g., 95%)
```

**Advantages:**
- P(B is better) is directly interpretable
- Can stop early without inflating Type I error
- Naturally incorporates prior knowledge
- Expected loss framework for decisions

**Disadvantages:**
- Requires prior specification
- Less standard in industry (harder to audit)

---

## 12. Worked Numericals

---

### 🔢 Numerical 1 — Full A/B Test End-to-End

**Problem:** You're testing a new ranking algorithm for a search engine.

- Baseline CTR: p_A = 0.15
- MDE: δ = 0.02 (want to detect 2pp improvement)
- α = 0.05, Power = 80%
- Daily traffic per arm: 5,000 users

**(a)** Required sample size per arm.
**(b)** Required experiment duration.
**(c)** After 14 days: n_A=70,200, n_B=69,800, clicks_A=10,407, clicks_B=11,033. Analyze.
**(d)** 95% CI for the improvement.

**Solution:**

**(a)**
```
n = 2(1.960+0.842)² × p(1−p) / δ²
  = 2 × (2.802)² × 0.15×0.85 / (0.02)²
  = 2 × 7.851 × 0.1275 / 0.0004
  = 2 × 1.0011 / 0.0004
  = 5,005
```

Need **5,005 users per arm** (≈10,010 total).

**(b)** At 5,000/day per arm: 5,005/5,000 ≈ **1.001 days** — barely more than 1 day!

But: always run at least **7–14 days** to capture weekly cycles, novelty effects, etc. Use 14 days.

**(c) Analysis:**

```
p̂_A = 10,407/70,200 = 0.14824
p̂_B = 11,033/69,800 = 0.15806

δ̂ = 0.15806 − 0.14824 = 0.00982 ≈ 0.98pp improvement

Pooled p̂ = (10,407+11,033)/(70,200+69,800) = 21,440/140,000 = 0.15314

SE = √[0.15314×0.84686×(1/70,200+1/69,800)]
   = √[0.12963×0.00002859]
   = √0.000003706 = 0.001925

Z = 0.00982/0.001925 = 5.10
```

p-value = 2×P(Z > 5.10) ≈ **3.4×10⁻⁷** << 0.05

**Highly significant!** Reject H₀.

**(d) 95% CI:**
```
0.00982 ± 1.96×0.001925 = 0.00982 ± 0.00377 = (0.006, 0.014)
```

The new ranking algorithm improves CTR by approximately **0.98pp** (95% CI: 0.6pp to 1.4pp).

**Decision:** Ship the new algorithm. The improvement is statistically significant and practically meaningful.

---

### 🔢 Numerical 2 — SRM Detection

**Problem:** You expect 50/50 split. After 7 days:
- Group A: 48,230 users
- Group B: 51,770 users
- Total: 100,000

Is this a Sample Ratio Mismatch?

**Solution:**

Under H₀ (true 50/50): E[n_A] = E[n_B] = 50,000

Chi-squared test:
```
χ² = (48,230−50,000)²/50,000 + (51,770−50,000)²/50,000
   = (−1,770)²/50,000 + (1,770)²/50,000
   = 3,132,900/50,000 + 3,132,900/50,000
   = 62.658 + 62.658
   = 125.316
```

χ²₁ distribution: χ²_{0.001, 1} = 10.83

125.316 >> 10.83 → **p-value ≈ 0** → **SEVERE SRM DETECTED**

This experiment has a massive sample ratio mismatch. The ratio is 48.23%:51.77% vs expected 50:50. Do NOT analyze this experiment.

**Likely causes to investigate:**
- Assignment mechanism bug (e.g., different hash function behavior)
- Bot traffic filtered differently for A and B
- Users seeing both variants (contamination)
- Technical issue causing some users to not receive treatment

---

### 🔢 Numerical 3 — Dealing with Novelty Effect

**Problem:** You run a 4-week A/B test on a new UI feature. Weekly CTR data:

| Week | CTR (A) | CTR (B) | Lift |
|---|---|---|---|
| 1 | 0.120 | 0.145 | +2.5pp |
| 2 | 0.122 | 0.138 | +1.6pp |
| 3 | 0.121 | 0.130 | +0.9pp |
| 4 | 0.120 | 0.127 | +0.7pp |

**(a)** What pattern suggests novelty effect?
**(b)** Should you use Week 1 data for the decision?
**(c)** How to estimate the long-run effect?
**(d)** Would you ship this feature?

**Solution:**

**(a)** Novelty effect pattern: The lift is **decreasing over time** — 2.5pp → 1.6pp → 0.9pp → 0.7pp. This suggests initial excitement is fading. The first-week lift is artificially inflated.

**(b)** No. Week 1 data overestimates the true long-run effect.

**(c)** Estimate long-run effect: fit a trend to weeks 2–4 and project stabilization.

Linear trend in lift: 1.6, 0.9, 0.7... the decay is slowing. Estimate stabilized lift ≈ 0.5–0.6pp.

Alternatively: use a **holdback group** — keep 10% in treatment for 3+ months to measure true long-term lift.

**(d)** Decision depends on business context:

- If 0.5–0.6pp CTR improvement is meaningful for the business → Ship
- If expected lift was 2pp+ → Don't ship (novelty effect, true lift much lower)
- If IT cost (engineering time, maintenance) is high → Need clearer lift evidence

**ML insight:** Novelty effects are common with UI changes, new models, or any visible change to users. Always run experiments long enough to see past the novelty. Google's principle: "Experiments should run until the novelty effect is over."

---

### 🔢 Numerical 4 — Multiple Metrics: Which to Trust?

**Problem:** A/B test for a new recommendation model. Results (n=50,000 per arm):

| Metric | Lift | p-value | Notes |
|---|---|---|---|
| CTR | +1.5pp | 0.032 | Primary metric |
| Revenue per user | +$0.08 | 0.071 | Secondary |
| Session length | −0.3 min | 0.041 | Guardrail |
| Bounce rate | +0.8pp | 0.018 | Guardrail |
| Latency P95 | +12ms | 0.003 | Guardrail |

**(a)** Which metrics are significant (uncorrected α=0.05)?
**(b)** Apply BH correction at FDR=0.05.
**(c)** How do you interpret the guardrail metric violations?
**(d)** Overall decision?

**Solution:**

**(a) Uncorrected:** CTR (0.032), session length (0.041), bounce rate (0.018), latency (0.003) → 4 significant.

**(b) BH correction:** m=5, sort p-values:

| Rank k | Metric | p-value | Threshold k×0.05/5 |
|---|---|---|---|
| 1 | Latency | 0.003 | 0.010 | ✓ |
| 2 | Bounce rate | 0.018 | 0.020 | ✓ |
| 3 | CTR | 0.032 | 0.030 | ✗ (0.032>0.030) |
| 4 | Session length | 0.041 | 0.040 | ✗ |
| 5 | Revenue | 0.071 | 0.050 | ✗ |

BH critical rank: k*=2. Reject H₀ for ranks 1 and 2 only.

**After BH correction:** Only latency and bounce rate are significant.

**(c) Guardrail violations:**

- **Latency +12ms** (significant): The new model is slower. Users may experience degraded experience. This is a hard blocker if latency SLA is violated.
- **Bounce rate +0.8pp** (significant): More users are leaving immediately. Negative user experience signal.
- **Session length −0.3 min** (borderline): Users spending less time — could be efficiency (finding what they want faster) or dissatisfaction.

**(d) Decision: DO NOT SHIP**

Even though CTR shows a positive signal (pre-correction), the guardrail metrics show real harm:
- Slower latency is measurable and significant
- Higher bounce rate means more users are dissatisfied

Investigate why the new model is slower (optimize it) and why bounce rate increased. Fix these issues before shipping.

**ML insight:** A/B test decisions are never just about the primary metric. Guardrail metrics prevent optimizing one metric at the expense of overall user experience. This is the production reality of ML system deployment.

---

### 🔢 Numerical 5 — Sequential A/B Testing (Peeking Problem)

**Problem:** You check p-values daily on a 14-day experiment. True H₀ is true (no effect).

Daily p-values: 0.23, 0.41, 0.08, 0.12, 0.19, 0.04, 0.31, 0.27, 0.18, 0.09, 0.22, 0.35, 0.14, 0.19.

**(a)** If you stop on day 6 when p<0.05, what happened?
**(b)** What is the true Type I error rate with daily peeking over 14 days?
**(c)** How does the SPRT (Sequential Probability Ratio Test) fix this?

**Solution:**

**(a)** Day 6 p-value = 0.04 < 0.05. If you stop and declare victory, you've committed a **false positive**. The true effect size is 0 — this is the peeking problem.

**(b)** Simulating the peeking Type I error: over 14 daily checks at α=0.05, the true Type I error rate is approximately:

```
P(at least one p < 0.05 in 14 checks | H₀) ≈ 1 − (1−0.05)^14... 
```

Wait, this overestimates because checks are correlated (running cumulative tests). True rate from simulation:

| Number of checks | True Type I rate |
|---|---|
| 1 | 5% |
| 5 | 14.2% |
| 10 | 18.9% |
| 14 | ~22% |

**Peeking daily for 14 days inflates Type I error to ~22% instead of 5%.**

Day 6 "significant" result is likely a false positive — the p-value fluctuated below 0.05 by chance.

**(c)** SPRT (Sequential Probability Ratio Test):

Instead of a fixed p-value threshold, compute the likelihood ratio:
```
LR_n = L(H₁|data₁,...,datₙ) / L(H₀|data₁,...,datₙ)
```

Stop and reject H₀ when LR > B (upper bound)
Stop and accept H₀ when LR < A (lower bound)

A and B are chosen to satisfy exact Type I and II error guarantees at any stopping time.

SPRT gives **always-valid p-values** — you can stop at any time with guaranteed Type I error rate.

**ML insight:** This is a real problem in production A/B testing. Companies solve it with:
- Pre-specified fixed duration and single analysis
- Sequential testing (SPRT, mSPRT)
- Bayesian methods (natural sequential updating)
- False discovery rate control across many experiments

---

### 🔢 Numerical 6 — Bayesian A/B Testing: Full Analysis

**Problem:** You're testing two recommendation models. Prior: Beta(20, 80) for both (historical CTR ≈ 20%).

After experiment:
- Model A: 500 users, 97 clicks (CTR = 0.194)
- Model B: 500 users, 118 clicks (CTR = 0.236)

**(a)** Posterior distributions.
**(b)** P(B is better than A).
**(c)** Expected lift if you ship B.
**(d)** Expected loss if B is actually worse.

**Solution:**

**(a)**

Posterior A: Beta(20+97, 80+403) = **Beta(117, 483)**
```
E[p_A|data] = 117/600 = 0.1950
```

Posterior B: Beta(20+118, 80+382) = **Beta(138, 462)**
```
E[p_B|data] = 138/600 = 0.2300
```

**(b)** P(p_B > p_A):

Using Normal approximation:
```
p_A ~ N(0.195, 0.195×0.805/600) = N(0.195, 0.000261)  →  SD_A = 0.01616
p_B ~ N(0.230, 0.230×0.770/600) = N(0.230, 0.000295)  →  SD_B = 0.01717

diff = p_B − p_A ~ N(0.035, 0.000261+0.000295) = N(0.035, 0.000556)
SD(diff) = 0.02358

P(diff > 0) = P(Z > −0.035/0.02358) = P(Z > −1.484) = Φ(1.484) ≈ 0.931
```

**P(B is better) = 93.1%**

**(c)** Expected lift if you ship B:

```
E[p_B − p_A | data] = 0.230 − 0.195 = 0.035 = 3.5pp lift
```

**(d)** Expected loss if B is actually worse:

```
Expected loss = E[max(0, p_A − p_B) | data]
≈ ∫_{diff<0} (−diff) × N(diff; 0.035, 0.02358²) d(diff)
≈ 0.02358 × φ(−0.035/0.02358) + (−0.035)×Φ(−0.035/0.02358)
= 0.02358 × φ(1.484) + (−0.035)×0.069
= 0.02358 × 0.1306 + (−0.035)×0.069
= 0.003080 − 0.002415 = 0.000665 ≈ 0.067pp
```

Expected loss from shipping B even if it's wrong ≈ 0.067pp — very small compared to expected gain of 3.5pp.

**Decision:** Ship B. P(B better)=93.1%, expected gain=3.5pp, expected loss if wrong=0.067pp. This is an excellent risk/reward ratio.

---

### 🔢 Numerical 7 — Common A/B Testing Pitfalls: Diagnosis

**Problem:** An ML team reports: "We ran an A/B test for 3 days, saw p=0.03, and shipped the model. After shipping, metrics dropped. What went wrong?"

Diagnose each possible issue:

**(a)** Experiment ran only 3 days.
**(b)** p=0.03 interpreted as "97% probability the model is better."
**(c)** No SRM check was performed.
**(d)** The team peeked every 6 hours.
**(e)** Only CTR was measured; latency degraded.

**Solution:**

**(a) Only 3 days:**
- Missing weekly seasonality (weekday vs weekend behavior differs)
- Novelty effect inflated early CTR boost
- **Fix:** Run for at least 2 full weekly cycles (14 days minimum)

**(b) p=0.03 misinterpreted:**
- p=0.03 means "if H₀ were true, probability of observing this or more extreme = 3%"
- NOT "97% confidence model is better"
- **Fix:** Correct interpretation; use Bayesian P(B>A) for direct probability statement

**(c) No SRM check:**
- Assignment bug could cause biased samples to enter one arm
- Analysis invalid if SRM exists
- **Fix:** Always run SRM test as Step 0 of analysis; if SRM → investigate before proceeding

**(d) Peeked every 6 hours:**
- Over 3 days: 12 checks at 6-hour intervals
- True Type I error ≈ 14-18% instead of 5%
- p=0.03 has much higher chance of being false positive
- **Fix:** Pre-specify analysis time; use sequential testing methods

**(e) Only CTR measured:**
- CTR may go up while quality goes down (clickbait effect)
- Latency degradation hurts user experience even if CTR rises
- **Fix:** Always check guardrail metrics (latency, error rate, revenue, user satisfaction)

**ML insight:** This is a composite of the most common A/B testing mistakes. At top tech companies, a rigorous experiment review process checks all of these before decisions are made.

---

## 13. The A/B Testing Interview Cheat Sheet

### Questions You Must Answer Perfectly

**Q: "How would you design an A/B test for [X]?"**
```
1. Define metric (primary, secondary, guardrail)
2. Set H₀, H₁, α, power
3. Compute required sample size (power analysis)
4. Choose randomization unit
5. Set duration (min 2 weekly cycles)
6. Plan for: SRM check, novelty effect, multiple testing
7. Analysis: test statistic, p-value, CI, effect size
8. Decision framework: ship/iterate/abandon
```

**Q: "Our A/B test shows p=0.04. Should we ship?"**
```
Not necessarily. Also check:
- Did we pre-specify the test (or is this selective reporting)?
- Are guardrail metrics acceptable?
- Is the effect size practically significant?
- Did we check for SRM?
- Was there continuous monitoring / peeking?
- Are there multiple metrics being tested?
- Is the experiment representative of production traffic?
```

**Q: "What is a p-value? What does p=0.04 mean?"**
```
P(observing test statistic this extreme | H₀ true) = 4%
= If there were truly no effect, we'd see this or stronger result 4% of the time
NOT = 96% probability the new version is better
```

---

## 14. Key Formulas — Cheat Sheet for Day 27

```
Sample Size (two proportions, equal n):
    n = 2(z_{α/2}+z_β)²·p(1−p)/δ²

Two-proportion z-test:
    Z = (p̂_B−p̂_A)/√[p̂(1−p̂)(1/n_A+1/n_B)]
    p-value = 2(1−Φ(|Z|))

95% CI for difference:
    (p̂_B−p̂_A) ± 1.96·SE

SRM test:
    χ² = Σ(nᵢ−nᵢ,expected)²/nᵢ,expected ~ χ²(1)
    Significant χ² → SRM → do not analyze

Peeking inflation (approximate):
    True Type I ≈ 1−(1−α)^k  [k = number of checks, overestimate]
    True Type I is lower but increases with k

Bayesian update:
    Prior Beta(α,β) + data(k,n) → Posterior Beta(α+k, β+n−k)
    P(p_B>p_A) = integral over posterior — use Normal approximation

Power:
    Power = Φ(δ√n/σ − z_{α/2})

Effect sizes:
    Relative lift = δ/p_A
    Cohen's h = 2arcsin(√p_B) − 2arcsin(√p_A)

Multiple testing (m metrics):
    Bonferroni: α* = α/m
    BH: reject p_{(k)} ≤ k·α/m

Experiment duration:
    Minimum 2 full weekly cycles
    Additional time for novelty effect to wash out
```

---

## 15. Practice Problems (Solve Before Day 28)

1. A new search ranking algorithm has baseline CTR=8%. You want to detect a +1pp improvement (MDE=1pp) with 80% power at α=0.05. Daily traffic: 20,000 per arm. Compute n and experiment duration.

2. An experiment shows n_A=25,340, n_B=24,660, total=50,000 (expected 50/50). Test for SRM. Should you proceed with analysis?

3. You check p-values at days 7, 14, 21 (3 checks). If true α per-check is 0.05, what is the approximate Type I error? How would Bonferroni correction handle this?

4. A/B test results: Control CTR=12.3% (n=10,000), Treatment CTR=13.1% (n=10,000). Compute Z-statistic, p-value, 95% CI for the lift, and MDE that this experiment could detect at 80% power.

5. *(Interview-level)* You're a DS at a company. The product manager says "Just run the A/B test for 2 days — we need fast results." What concerns would you raise? How would you balance speed with rigor? What would a minimum viable experiment design look like?

---

## 16. Looking Ahead

**Day 28** — **Information Theory: Entropy, KL Divergence & Cross-Entropy.** The mathematical bridge between probability and machine learning. We formalize entropy as the fundamental measure of uncertainty, derive KL divergence as the "cost" of using the wrong distribution, and show how cross-entropy loss, VAE regularization, and information-theoretic feature selection all flow from these concepts.

---
*End of Day 27 | Next: Day 28 — Information Theory*
