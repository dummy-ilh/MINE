# Day 14 — Unit 2 Review & Hard Interview Problems on Distributions
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Coverage:** Days 7–13 — Random Variables, Distributions, Expectation, Variance, Covariance
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Unit 2 Master Cheat Sheet

Every formula from Days 7–13 in one place.

```
══════════════════════════════════════════════════════════
RANDOM VARIABLES (Day 7)
══════════════════════════════════════════════════════════
PMF:        p(x) = P(X=x),    Σ p(x) = 1
CDF:        F(x) = P(X≤x),    f(x) = F'(x)
Joint PMF:  p_{X,Y}(x,y),    marginal: pₓ(x) = Σᵧ p(x,y)
Independence: p(x,y) = pₓ(x)·pᵧ(y)
Softmax:    P(k) = exp(zₖ)/Σ exp(zⱼ)   [valid PMF]
Cross-entropy loss: L = −log P(true class)

══════════════════════════════════════════════════════════
EXPECTATION & VARIANCE (Day 8)
══════════════════════════════════════════════════════════
E[X] = Σ x·p(x)  or  ∫ x·f(x)dx
LOTUS:  E[g(X)] = Σ g(x)p(x)
Linearity: E[aX+bY+c] = aE[X]+bE[Y]+c   [no independence needed]
E[XY] = E[X]E[Y]   [independent only]
Var(X) = E[X²] − (E[X])²
Var(aX+b) = a²Var(X)
Var(X+Y) = Var(X)+Var(Y)   [independent]
Bias-Variance: MSE = Bias² + Variance

══════════════════════════════════════════════════════════
KEY DISCRETE DISTRIBUTIONS (Days 9–10)
══════════════════════════════════════════════════════════
Bernoulli(p):    P(X=1)=p,  E=p,  Var=p(1−p)
Binomial(n,p):   P(X=k)=C(n,k)pᵏ(1−p)^(n−k),  E=np,  Var=np(1−p)
Geometric(p):    P(X=k)=(1−p)^(k-1)p,  E=1/p,  Var=(1−p)/p²
                 Memoryless: P(X>m+n|X>m)=P(X>n)
Poisson(λ):      P(X=k)=e^(−λ)λᵏ/k!,  E=λ,  Var=λ
                 Additive: Pois(λ₁)+Pois(λ₂)=Pois(λ₁+λ₂)
                 Limit of Bin(n,λ/n) as n→∞

══════════════════════════════════════════════════════════
KEY CONTINUOUS DISTRIBUTIONS (Days 11–12)
══════════════════════════════════════════════════════════
Uniform(a,b):    f=1/(b−a),  E=(a+b)/2,  Var=(b−a)²/12
                 p-values ~ Uniform(0,1) under H₀
Exponential(λ):  f=λe^(−λx),  F=1−e^(−λx),  E=1/λ,  Var=1/λ²
                 Memoryless; inter-arrival of Poisson process
Normal(μ,σ²):    f=(1/σ√2π)exp(−(x−μ)²/2σ²)
                 E=μ, Var=σ², Z=(X−μ)/σ~N(0,1)
                 68/95/99.7 rule
                 Sum of indep Normals → Normal
                 MLE: μ̂=x̄, σ̂²=(1/n)Σ(xᵢ−x̄)²

══════════════════════════════════════════════════════════
COVARIANCE & CORRELATION (Day 13)
══════════════════════════════════════════════════════════
Cov(X,Y) = E[XY]−E[X]E[Y]
ρ(X,Y) = Cov(X,Y)/(σₓσᵧ),  −1≤ρ≤1
Var(X+Y) = Var(X)+Var(Y)+2Cov(X,Y)
Cov=0 ⟹ Independent (ONLY for jointly Normal)
PCA = eigendecomposition of Σ
MVN conditional: μ_{1|2}=μ₁+(Σ₁₂/Σ₂₂)(x₂−μ₂)
```

---

## 2. Distribution Identification — The Interview Skill

Before computing anything, you must identify which distribution applies.

### Decision Tree

```
Is the outcome binary (success/failure)?
├── Single trial → Bernoulli(p)
└── n trials, count successes → Binomial(n,p)
    └── n→∞, p→0, np=λ → Poisson(λ)

How many trials until first success?
└── Geometric(p)  [or Negative Binomial for r-th success]

Counting events in time/space?
└── Poisson(λ)  [check: Mean ≈ Variance]

Continuous, bounded interval?
└── Uniform(a,b)

Continuous, waiting/survival time, memoryless?
└── Exponential(λ)

Symmetric bell curve, sums/averages of many things?
└── Normal(μ,σ²)

Positive values, right-skewed, multiplicative effects?
└── Log-Normal

Count data, Var >> Mean?
└── Negative Binomial (overdispersed Poisson)
```

---

## 3. Hard Problems

---

### 🔢 Problem 1 — Distribution of a Transformation

**Problem:** X ~ Uniform(0,1). Find the PDF of Y = −log(X).

*(This appears in: generating Exponential samples, understanding cross-entropy loss, survival analysis.)*

**Solution:**

Use the CDF method (change of variables):

Step 1 — Find the CDF of Y:
```
F_Y(y) = P(Y ≤ y) = P(−log X ≤ y) = P(log X ≥ −y) = P(X ≥ e^(−y))
```

Since X ~ Uniform(0,1): P(X ≥ e^(−y)) = 1 − e^(−y) for y ≥ 0.

Step 2 — Differentiate to get PDF:
```
f_Y(y) = F_Y'(y) = e^(−y)    for y ≥ 0
```

**Y ~ Exponential(1)!**

**Insight:** This proves the inverse CDF method from Day 11:
- U ~ Uniform(0,1) → X = −log(U) ~ Exponential(1)
- Cross-entropy loss = −log P(correct class) ~ Exponential-like when predictions are uniform

**Generalization:** If X ~ Uniform(0,1), then −log(X)/λ ~ Exponential(λ).

---

### 🔢 Problem 2 — Minimum and Maximum of Random Variables

**Problem:** X₁, X₂, ..., Xₙ are i.i.d. Exponential(λ). Find the distribution of:
- M = min(X₁, ..., Xₙ) — first arrival in n parallel processes
- L = max(X₁, ..., Xₙ) — last arrival

*(Appears in: parallel vs serial systems, order statistics, early stopping.)*

**Solution:**

**Minimum M:**

P(M > t) = P(all Xᵢ > t) = P(X₁>t)ⁿ = (e^(−λt))ⁿ = e^(−nλt)

So: F_M(t) = 1 − e^(−nλt) → **M ~ Exponential(nλ)**

E[M] = 1/(nλ) — minimum of n exponentials is n times faster.

**Maximum L:**

P(L ≤ t) = P(all Xᵢ ≤ t) = (F(t))ⁿ = (1−e^(−λt))ⁿ

PDF: f_L(t) = n(1−e^(−λt))^(n−1) · λe^(−λt)

E[L] = (1/λ)(1 + 1/2 + 1/3 + ... + 1/n) = Hₙ/λ ≈ ln(n)/λ

**ML connections:**
- **Parallel inference**: n replicas each take Exponential(λ) time → response time = M ~ Exponential(nλ). Response time drops as 1/n.
- **Serial pipeline**: n stages each take Exponential(λ) → completion time ≠ max (stages aren't independent given process), but order statistics appear in hyperparameter search (best of n configs).
- **Coupon collector**: E[max] = Hₙ/λ ~ ln(n)/λ — same harmonic number appears.

---

### 🔢 Problem 3 — Moment Matching for Distribution Fitting

**Problem:** You observe click counts per hour over 10 days:
```
Data: 3, 8, 5, 7, 4, 6, 9, 5, 7, 6
```

Fit a Poisson distribution using method of moments. Then fit a Normal. Which is better?

**Solution:**

**Sample statistics:**
```
x̄ = (3+8+5+7+4+6+9+5+7+6)/10 = 60/10 = 6.0
s² = [(3−6)²+(8−6)²+(5−6)²+(7−6)²+(4−6)²+(6−6)²+
      (9−6)²+(5−6)²+(7−6)²+(6−6)²] / 9
   = [9+4+1+1+4+0+9+1+1+0] / 9 = 30/9 = 3.33
```

**Poisson fit (MOM: set E[X]=x̄):**
λ̂ = x̄ = 6.0

Check: Poisson requires Mean = Variance.
- Observed: Mean = 6.0, Variance = 3.33
- Dispersion ratio = 3.33/6.0 = 0.56 < 1 → **underdispersed**
- Poisson assumes Var=Mean=6.0, but observed Var=3.33

Poisson may not be the best fit here.

**Normal fit (MOM: set E[X]=x̄, Var(X)=s²):**
μ̂ = 6.0, σ̂² = 3.33, σ̂ = 1.83

**Comparison:**
```
Poisson(6):  P(X=6) = e^(−6)·6⁶/6! = 0.00248×46656/720 = 0.161
Normal(6,3.33): P(5.5≤X≤6.5) = P(−0.27≤Z≤0.27) = 2Φ(0.27)−1 = 0.213
```

For discrete data with underdispersion, Binomial(n,p) with E=np=6 and Var=np(1−p)=3.33 gives:
- 1−p = 3.33/6 = 0.555 → p = 0.445, n = 6/0.445 ≈ 13.5

**Best fit:** Binomial(14, 0.43) — matches both mean and variance.

**ML lesson:** Method of moments sets theoretical moments equal to sample moments. Matching 2 moments requires 2-parameter distribution. Always check if the data's variance matches your distribution's assumed variance before fitting.

---

### 🔢 Problem 4 — The St. Petersburg Paradox (Expectation Fails)

**Problem:** A casino game: flip a fair coin until Heads. If Heads appears on flip k, you win $2ᵏ. How much should you pay to play?

**Solution:**

X = winning, which takes value 2ᵏ with probability (1/2)ᵏ.

```
E[X] = Σₖ₌₁^∞ 2ᵏ · (1/2)ᵏ = Σₖ₌₁^∞ 1 = ∞
```

**Expected value is infinite** — yet no rational person would pay more than ~$20 to play.

**Why expectation fails here:**
- The distribution has extremely heavy tails
- Variance is also infinite
- Law of Large Numbers convergence is impractically slow

**Resolutions:**
1. **Utility theory**: log utility U(x)=log(x). E[log(2ᵏ)] = Σ k·log(2)·(1/2)ᵏ = 2log(2) ≈ $1.39 — finite and reasonable
2. **Truncation**: real casinos have finite bankrolls, capping the game
3. **Risk aversion**: humans weight losses more than gains

**ML lessons:**
- Heavy-tailed distributions break expectation-based reasoning
- Training loss can have infinite-variance gradients → **gradient clipping** is necessary
- Expected reward in RL can be misleading for heavy-tailed reward distributions
- Why log loss (cross-entropy) is better than 0/1 loss: it has finite, well-behaved gradients
- Risk-sensitive RL: optimizing E[reward] ignores variance — CVaR (Conditional Value at Risk) is the rigorous alternative

---

### 🔢 Problem 5 — Distribution of Sum: Binomial + Geometric

**Problem:** You run hyperparameter trials in two phases:
- Phase 1: Fixed n=5 trials. Each succeeds with p=0.3. Let X = number of successes.
- Phase 2: Keep running trials (each p=0.3) until first success. Let Y = trials until first success in Phase 2.

**(a)** Distribution and E[X], E[Y]
**(b)** P(X=0) — Phase 1 finds nothing
**(c)** Given X=0, E[total trials] = 5 + E[Y]
**(d)** P(find at least 1 success in Phase 1 OR Phase 2 within 10 total trials)

**Solution:**

**(a)**
```
X ~ Binomial(5, 0.3):  E[X] = 5×0.3 = 1.5
Y ~ Geometric(0.3):    E[Y] = 1/0.3 ≈ 3.33
```

**(b)**
```
P(X=0) = (0.7)⁵ = 0.16807 ≈ 16.8%
```

**(c)** Given Phase 1 fails (X=0), expect 5 + 3.33 = **8.33 total trials**

**(d)** P(at least 1 success in 10 total trials) — treat all 10 as Bernoulli(0.3):
```
P(at least 1) = 1 − P(all fail) = 1 − 0.7¹⁰ = 1 − 0.0282 = 0.9718
```

Note: This collapses Phase 1 and 2 into a single Binomial(10,0.3) since all trials are i.i.d. Bernoulli(0.3) — the two-phase structure doesn't change the math.

---

### 🔢 Problem 6 — Covariance of Functions: The Tricky Case

**Problem:** X ~ N(0,1). Define Y = X² and Z = X³.

**(a)** Find Cov(X, Y) — are they correlated?
**(b)** Find Cov(X, Z)
**(c)** Are X and Y independent?
**(d)** Are X and Z independent?

**Solution:**

**(a) Cov(X, Y) = Cov(X, X²):**
```
E[X] = 0,  E[X²] = Var(X) = 1  (since E[X]=0)
E[X·X²] = E[X³] = 0    [odd moment of symmetric N(0,1)]

Cov(X, X²) = E[X³] − E[X]E[X²] = 0 − 0×1 = 0
```
**Zero covariance!** (As shown in Day 13.)

**(b) Cov(X, Z) = Cov(X, X³):**
```
E[X⁴] = 3    [4th moment of N(0,1) = 3, proven via MGF on Day 16]
E[X·X³] = E[X⁴] = 3
E[X³] = 0    [odd moment]

Cov(X, X³) = E[X⁴] − E[X]E[X³] = 3 − 0 = 3
```
**Non-zero covariance** between X and X³!

**(c)** X and Y=X² are NOT independent — knowing X determines Y exactly. But they are uncorrelated.

**(d)** X and Z=X³ ARE dependent (same reason — X determines Z), and they ARE correlated (Cov=3).

**Insight table:**

| Pair | Correlated? | Independent? |
|---|---|---|
| X and X² | No (ρ=0) | No |
| X and X³ | Yes (ρ>0) | No |

**ML lesson:** Polynomial features (X, X², X³, ...) can be correlated or uncorrelated with each other and with the original feature. Polynomial regression is not immune to multicollinearity. Orthogonal polynomials (Legendre, Chebyshev) are used precisely to avoid this problem.

---

### 🔢 Problem 7 — Full ML System: Choosing Distributions End-to-End

**Problem:** Design the probabilistic model for a **fraud detection system** end to end.

Given:
- 1M transactions per day
- 0.1% fraud rate (1000 fraudulent transactions/day)
- Each transaction has 20 features (continuous)
- Model must output P(fraud | features)

Answer the following in distribution terms:

**(a)** Distribution of number of fraud cases per day
**(b)** Distribution of each feature within fraud/non-fraud class
**(c)** Decision rule using Bayes' theorem
**(d)** Distribution of model output scores under H₀ (no fraud)
**(e)** Threshold setting: if we alert on top 0.5% of scores, expected false positive rate?

**Solution:**

**(a)** Daily fraud count:
n = 1,000,000, p = 0.001, λ = np = 1000

Since n is huge and p is tiny: **X ~ Poisson(1000)**

E[X]=1000, SD(X)=√1000≈31.6

On 95% of days, between 938 and 1062 fraud cases (1000±2×31.6).

**(b)** Feature distributions:
Each feature Xᵢ | Class=c ~ N(μᵢc, σᵢc²) — **Gaussian Naive Bayes assumption**

In practice: fit separate Normal to each feature within each class from training data.

**(c)** Bayes decision rule:
```
P(fraud|x) ∝ P(x|fraud)·P(fraud)
           = [Πᵢ N(xᵢ; μᵢ,fraud, σᵢ,fraud²)] × 0.001

Predict fraud if P(fraud|x) > threshold τ
```

In log form (numerically stable):
```
log P(fraud|x) ∝ log(0.001) + Σᵢ log N(xᵢ; μᵢ,fraud, σᵢ,fraud²)
```

**(d)** Under H₀ (benign transaction scored by model):
Well-calibrated model outputs P̂(fraud|x) for benign transactions.

For a perfectly calibrated model, the output **score rank** ~ Uniform(0,1) over benign transactions.

**(e)** Alert top 0.5% of scores. Among 999,000 benign transactions:
Expected false alerts = 999,000 × 0.005 = **4,995 false alerts per day**

True positives: depends on model sensitivity. If recall=80%:
True alerts = 0.80 × 1000 = 800

Precision = 800/(800+4995) = 800/5795 = **13.8%**

Despite good accuracy, only 1 in 7 alerts is real fraud — the base rate problem (Day 4) strikes again!

**System design insight:** In high-volume, low-fraud-rate systems:
- Poisson for daily counts → monitors for rate changes
- Gaussian Naive Bayes → fast, interpretable classifier
- Score calibration ensures P̂ is a true probability
- Alert threshold must balance precision vs recall
- High volume makes false positives a severe operational burden — always compute expected FP rate

---

## 4. Rapid-Fire: 20 Distribution True/False

| # | Statement | Answer |
|---|---|---|
| 1 | Var(Bernoulli(p)) is maximized at p=0.5 | **True** — max = 0.25 |
| 2 | Binomial(n,p) has mean = variance | **False** — mean=np, var=np(1-p) |
| 3 | Poisson has mean = variance | **True** — both equal λ |
| 4 | Geometric is the only discrete memoryless dist. | **True** |
| 5 | Exponential is the only continuous memoryless dist. | **True** |
| 6 | Sum of independent Normals is Normal | **True** |
| 7 | Sum of independent Poissons is Poisson | **True** — Pois(λ₁+λ₂) |
| 8 | Sum of independent Exponentials is Exponential | **False** — it's Gamma |
| 9 | PDF values must be ≤ 1 | **False** — densities can exceed 1 |
| 10 | P(X=x)=0 for continuous X | **True** |
| 11 | Cov(X,Y)=0 implies X,Y independent | **False** — only for jointly Normal |
| 12 | If X,Y independent then Cov(X,Y)=0 | **True** — always |
| 13 | Var(X-Y) = Var(X) - Var(Y) | **False** — Var(X)+Var(Y)-2Cov |
| 14 | E[XY] = E[X]E[Y] always | **False** — only if independent |
| 15 | Linearity of expectation requires independence | **False** — always holds |
| 16 | MLE for Normal mean is the sample mean | **True** |
| 17 | MLE for Normal variance divides by n-1 | **False** — divides by n (biased) |
| 18 | L2 regularization = Gaussian prior on weights | **True** |
| 19 | p-values follow Uniform(0,1) under H₀ | **True** |
| 20 | Minimum of n i.i.d. Exp(λ) is Exp(nλ) | **True** |

---

## 5. Five Real ML Interview Q&As

---

**Interview Q1** *(Google)* — "Your model is trained on i.i.d. data but deployed on time-series data. Which distribution assumptions break?"

**Answer:** Several break simultaneously:
- **Independence** fails — consecutive observations are autocorrelated
- **Identical distribution** may fail — distribution shifts over time (concept drift)
- **Binomial** model for accuracy (n i.i.d. trials) invalid — test samples are dependent
- **Normal approximation** for sample mean (via CLT) requires independence — invalid here
- **Fix:** Use time-series cross-validation, autoregressive models (AR, LSTM), test for stationarity

---

**Interview Q2** *(Meta)* — "You observe that feature X has ρ=0.02 with target Y. Should you drop it?"

**Answer:** Not necessarily:
- ρ=0.02 only measures **linear** dependence — Y=X² would give ρ≈0 even with perfect nonlinear relationship
- Compute mutual information I(X;Y) to detect nonlinear dependence
- Check interaction effects — X might predict Y only when combined with another feature
- Tree-based feature importance (SHAP) captures what correlation misses
- **Only drop if** mutual information is also near zero AND the feature adds no predictive value in cross-validation

---

**Interview Q3** *(Amazon)* — "We're modeling daily page views. Mean=500, Variance=2500. Which distribution?"

**Answer:**
- Var/Mean = 2500/500 = 5 >> 1 → **overdispersed** — Poisson doesn't fit
- Use **Negative Binomial**: E[X]=μ, Var(X)=μ+μ²/r
  - 500 + 500²/r = 2500 → r = 500²/2000 = 62.5
  - NegBin(r=62.5, p=r/(r+μ)=0.111)
- Alternatively, if log(views) is approximately Normal → **Log-Normal** model
- Quick check: plot histogram, check for right skew and tail behavior

---

**Interview Q4** *(DeepMind)* — "Why does gradient clipping help training with heavy-tailed gradient distributions?"

**Answer:**
- In deep networks, gradient distributions can be heavy-tailed (Var=∞ for truly heavy tails)
- With infinite-variance gradients, SGD steps are dominated by rare extreme gradients
- Law of Large Numbers converges slowly (requires finite variance for CLT)
- Gradient clipping: if ||∇L|| > C, scale to ||∇L||=C
- Effect: truncates the distribution at C, giving finite variance = C²
- Allows stable SGD convergence with controlled step sizes
- **Mathematical framing:** Clipping converts a heavy-tailed distribution to a bounded one, ensuring E[||∇L||²] < ∞, which is required for SGD convergence guarantees

---

**Interview Q5** *(OpenAI/Research)* — "In a VAE, the ELBO is E_q[log p(x|z)] − KL(q(z|x) || p(z)). Explain each term in distribution language."

**Answer:**
```
ELBO = E_q[log p(x|z)] − KL(q(z|x) || p(z))
```

**Term 1: E_q[log p(x|z)] — Reconstruction term**
- q(z|x) = encoder posterior ~ N(μ(x), σ²(x)) [learned]
- p(x|z) = decoder likelihood ~ N(decoder(z), σ²) [for continuous x]
- E_q[log p(x|z)] = expected log-likelihood of reconstruction
- Maximizing this = minimizing expected reconstruction error
- For Gaussian decoder = −MSE between x and x̂

**Term 2: KL(q(z|x) || p(z)) — Regularization term**
- p(z) = N(0,I) — prior on latent space
- KL divergence measures how far encoder posterior is from prior
- Minimizing KL forces latent codes toward N(0,I) — continuous, structured latent space
- For two Gaussians: KL(N(μ,σ²)||N(0,1)) = ½(μ²+σ²−log σ²−1)

**Together:** ELBO balances reconstruction quality (likelihood) vs latent space regularity (prior). Maximizing ELBO = learning to compress (encoder) and decompress (decoder) while keeping latent space Gaussian.

---

## 6. Unit 2 → Unit 3 Preview

Starting **Day 15**, we enter the deeper theory:

**Unit 3: Expectation & Transforms (Days 15–18)**

| Day | Topic | Why It Matters |
|---|---|---|
| Day 15 | LOTUS, Linearity, Indicator RVs | Computing any expectation efficiently |
| Day 16 | Moment Generating Functions | Distribution identification, all moments at once |
| Day 17 | Conditional Expectation & Law of Total Expectation | Prediction, regression, causal reasoning |
| Day 18 | Inequalities — Markov, Chebyshev, Jensen's | Bounding probabilities, generalization theory |

These are the mathematical power tools — they're what distinguishes someone who knows distributions from someone who can **reason** about them.

---
*End of Day 14 — Unit 2 Complete | Next: Day 15 — LOTUS, Linearity & Indicator Random Variables*
