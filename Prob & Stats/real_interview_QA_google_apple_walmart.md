# Real Probability & Statistics Interview Q&A
## Google, Apple, Walmart & Other FAANG/Top-Tech Companies
### Sourced from public reports: NickSingh.com, Glassdoor, Blind, Medium, Towards Data Science

---

## HOW TO USE THIS FILE

Each question is tagged with:
- **[Company]** — where it was reportedly asked
- **[Difficulty]** — Easy / Medium / Hard
- **[Topic]** — which Day of the 30-Day course it maps to
- **Full worked answer** — not just hints

---

## SECTION 1: PROBABILITY QUESTIONS

---

### Q1. [Facebook — Easy] [Day 4: Bayes]
**Unfair Coin Detection**

There is a fair coin (one side heads, one side tails) and an unfair coin (both sides tails). You pick one at random, flip it 5 times, and observe tails all five times. What is the probability you are flipping the unfair coin?

**Answer:**

Let U = unfair coin, F = fair coin. P(U) = P(F) = 0.5.
Let 5T = event of 5 tails in a row.

P(5T | U) = 1 (always tails)
P(5T | F) = (1/2)^5 = 1/32

By Bayes:
P(U | 5T) = P(5T|U)*P(U) / [P(5T|U)*P(U) + P(5T|F)*P(F)]
           = (1 * 0.5) / (1*0.5 + (1/32)*0.5)
           = 0.5 / (0.5 + 0.015625)
           = 0.5 / 0.515625
           ≈ **0.970 = 97%**

---

### Q2. [Lyft — Easy] [Day 1: Sample Spaces]
**HH vs TH Coin Game**

You and a friend flip a coin repeatedly until HH or TH appears. If HH appears first, you win. If TH appears first, your friend wins. What is the probability you win?

**Answer:**

Key insight: After the first flip:
- If first flip is T: the sequence TH must appear before TH can be beaten by HH. But TH appears on very next H → friend wins immediately.
- If first flip is H: need another H for HH, or T ends your run.

More carefully: after any T, TH appears on the very next H → friend wins. After H, you need another H before a T.

P(you win) = P(HH before TH)

After first flip = H (prob 1/2): you need second flip = H (prob 1/2) → you win. Or second flip = T → TH just appeared → friend wins.
After first flip = T (prob 1/2): friend wins immediately on next H.

P(win) = P(first=H) * P(second=H) = 1/2 * 1/2 = **1/4**

---

### Q3. [Google — Easy] [Day 2: Counting]
**Seven-Game Series Goes to Game 7**

What is the probability that a 7-game series (first to 4 wins) goes to 7 games? Assume teams are equally matched.

**Answer:**

Series goes to 7 games iff after 6 games the series is tied 3-3.
Team A wins exactly 3 of first 6 games: C(6,3) ways.

P(3-3 after 6 games) = C(6,3) * (1/2)^6 = 20/64 = **5/16 ≈ 31.25%**

---

### Q4. [Facebook — Easy] [Day 4: Bayes]
**Spam Rater Detection**

90% of raters are diligent (label 20% spam, 80% good). 10% are lazy (label 0% spam, 100% good). A rater labels 4 pieces as good. What is the probability they are diligent?

**Answer:**

P(4 good | diligent) = 0.80^4 = 0.4096
P(4 good | lazy) = 1.0^4 = 1.0

P(diligent | 4 good) = P(4 good|diligent)*P(diligent) / P(4 good)

P(4 good) = 0.4096*0.9 + 1.0*0.1 = 0.36864 + 0.1 = 0.46864

P(diligent | 4 good) = (0.4096 * 0.9) / 0.46864 = 0.36864 / 0.46864 ≈ **0.787 = 78.7%**

---

### Q5. [Bloomberg — Easy] [Day 2: Counting]
**Probability Two Random Chords Intersect**

Draw a circle. Choose two chords at random. What is the probability they intersect?

**Answer:**

A chord is defined by 2 points on the circle. Two chords = 4 points on the circle.

Given 4 points on a circle, there are C(4,2)/2 = 3 ways to pair them into 2 chords:
- {(1,2),(3,4)}, {(1,3),(2,4)}, {(1,4),(2,3)}

Of these 3 pairings, exactly 1 produces intersecting chords (when the points alternate around the circle).

P(intersect) = **1/3**

---

### Q6. [Amazon — Easy] [Day 4: Bayes]
**Disease Testing (Classic Base Rate Problem)**

1/1000 people have a disease. A test is 98% accurate if you have it; 1% false positive rate. If someone tests positive, what is the probability they have the disease?

**Answer:**

P(D) = 0.001, P(D^c) = 0.999
P(T+ | D) = 0.98, P(T+ | D^c) = 0.01

P(T+) = 0.98*0.001 + 0.01*0.999 = 0.00098 + 0.00999 = 0.01097

P(D | T+) = 0.00098 / 0.01097 ≈ **0.0893 = 8.93%**

Despite 98% accuracy, only ~9% chance of disease — because disease is very rare (base rate dominates).

---

### Q7. [Facebook — Easy] [Day 2: Counting]
**Colored Numbered Cards**

50 cards: 5 colors × numbers 1-10. Pick 2 cards. What is P(different color AND different number)?

**Answer:**

Total pairs: C(50,2) = 1225

Same color pairs: 5 colors × C(10,2) = 5 × 45 = 225
Same number pairs: 10 numbers × C(5,2) = 10 × 10 = 100
Same color AND same number: impossible (unique cards)

By inclusion-exclusion:
P(same color OR same number) = (225 + 100) / 1225 = 325/1225

P(different color AND different number) = 1 - 325/1225 = 900/1225 = **36/49 ≈ 0.7347**

---

### Q8. [Tesla — Easy] [Day 5: Independence]
**Two Dice Rolls**

A fair six-sided die is rolled twice. P(1 on first roll AND not 6 on second roll)?

**Answer:**

By independence:
P = P(first=1) * P(second≠6) = (1/6) * (5/6) = **5/36 ≈ 0.139**

---

### Q9. [Facebook — Easy] [Day 10: Geometric/Coupon Collector]
**Expected Rolls to See All 6 Die Faces**

What is the expected number of rolls needed to see all 6 sides of a fair die?

**Answer:**

Coupon collector problem with n=6.

E[T] = 6*(1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1/1)
     = 6*(1 + 0.5 + 0.333 + 0.25 + 0.2 + 0.167)
     = 6 * 2.45
     ≈ **14.7 rolls**

---

### Q10. [Microsoft — Easy] [Day 4: Bayes]
**Three Friends All Say It's Raining**

Three friends in Seattle each say it's raining. Each has 1/3 probability of lying. P(rain) = 0.25 on any given day. What is P(it's actually raining)?

**Answer:**

P(all say rain | rain) = (2/3)^3 = 8/27  [truthful]
P(all say rain | no rain) = (1/3)^3 = 1/27  [all lying]

P(rain | all say rain) = P(all say rain|rain)*P(rain) / P(all say rain)

P(all say rain) = (8/27)*0.25 + (1/27)*0.75 = 2/27 + 0.75/27 = (2 + 0.75)/27 = 2.75/27

P(rain | all say rain) = (8/27 * 0.25) / (2.75/27) = 2/27 / (2.75/27) = 2/2.75 ≈ **0.727 = 72.7%**

---

### Q11. [Uber — Easy] [Day 2: Counting]
**Three Dice in Strictly Increasing Order**

Roll three dice one by one. P(three numbers in strictly increasing order)?

**Answer:**

Total outcomes: 6^3 = 216

Favorable: choose 3 distinct numbers from {1..6} (C(6,3) = 20 ways), only 1 arrangement is strictly increasing.

But each set of 3 distinct numbers has exactly 1 increasing ordering out of 3! = 6 total orderings.

P = C(6,3) / 6^3 = 20/216 = **5/54 ≈ 0.0926**

---

### Q12. [Bloomberg — Medium] [Day 1: Sample Spaces]
**Three Ants on Equilateral Triangle**

Three ants sit at corners of an equilateral triangle. Each randomly picks a direction (clockwise or counterclockwise). P(no collisions)?

**Answer:**

No collision only if ALL ants go clockwise or ALL go counterclockwise.

Total outcomes: 2^3 = 8
Favorable: 2 (all CW or all CCW)

P(no collision) = 2/8 = **1/4**

For k ants on k-gon: P = 2/2^k = 2^(1-k)

---

### Q13. [Two Sigma — Medium] [Day 9: Geometric]
**Expected Flips for Two Consecutive Heads**

Expected number of coin flips to get HH (two consecutive heads)?

**Answer:**

Let E = expected flips from start.
E[X|H] = 1 + (1/2)*0 + (1/2)*E = 1 + E/2  [if first H, then next H=done, next T=restart]

E = 1/2*(1 + E[X|H]) + 1/2*(1 + E)   [first flip T: restart; first flip H: go to E[X|H]]
E = 1/2*(1 + 1 + E/2) + 1/2*(1 + E)
E = 1/2*(2 + E/2) + 1/2*(1 + E)
E = 1 + E/4 + 1/2 + E/2
E = 3/2 + 3E/4
E/4 = 3/2
E = **6 flips**

---

### Q14. [Amazon — Medium] [Day 2: Counting / Expected Value]
**Expected Cards Before First Ace**

How many cards expected to draw from a standard deck before the first ace?

**Answer:**

52-card deck, 4 aces. By symmetry, the 4 aces divide the 48 non-aces into 5 equal groups on average.

E[non-aces before first ace] = 48/5 = 9.6

So expected draws before seeing first ace = **9.6** (draw 9.6 non-aces, then draw the ace = 10.6 total draws until first ace seen, but the question asks cards drawn before = 9.6)

---

### Q15. [Robinhood — Medium] [Day 5: Independence / Symmetry]
**A has n+1 Coins, B has n Coins**

A has n+1 coins, B has n coins. Each flips all. P(A has more heads than B)?

**Answer:**

Consider A's first n coins vs B's n coins. Three scenarios (by symmetry):
- P(A's first n > B's n) = P(A's first n < B's n) = x
- P(tie) = y
- 2x + y = 1

When tied: A flips coin n+1. With prob 1/2 it's H → A wins.

P(A wins total) = x + (1/2)*y = x + (1-2x)/2 = **1/2**

---

### Q16. [Airbnb — Medium] [Day 1: Probability]
**Fair Odds from Unfair Coin**

Given an unfair coin with unknown bias, how do you generate fair odds?

**Answer:**

Von Neumann's trick:
1. Flip the coin twice to get a pair.
2. If HT → output "Heads" (call this event A)
3. If TH → output "Tails" (call this event B)
4. If HH or TT → discard and repeat

P(HT) = p*(1-p)
P(TH) = (1-p)*p

Since P(HT) = P(TH) regardless of p, the conditional probability of each outcome is 1/2. This generates a fair coin from any biased coin with 0 < p < 1.

---

### Q17. [Quora — Medium] [Day 12: Normal]
**N Draws, Probability k Are Above Y**

N i.i.d. draws from N(μ, σ²). P(exactly k draws > Y)?

**Answer:**

Let p = P(single draw > Y) = 1 - Φ((Y-μ)/σ)

Each draw independently exceeds Y with probability p.

Number exceeding Y ~ Binomial(N, p)

P(exactly k exceed Y) = C(N,k) * p^k * (1-p)^(N-k)

---

### Q18. [Walmart — Medium] [Day 3: Conditional Probability]
**Two Kids, At Least One Boy**

A coworker has two kids. At least one is a boy. P(both are boys)?

**Answer:**

Sample space of two children (equally likely): {BB, BG, GB, GG}

Given at least one boy, eliminate GG:
Remaining: {BB, BG, GB} — each equally likely.

P(both boys | at least one boy) = 1/3 ≈ **0.333**

Note: This is a classic probability trap. The answer depends critically on HOW you learned "at least one is a boy." If you learned "the older one is a boy," answer would be 1/2.

---

### Q19. [Spotify — Hard] [Day 2: Counting]
**Largest Die Roll = r**

Roll a fair die n times. P(largest number rolled = r) for each r ∈ {1..6}?

**Answer:**

Let B_r = event all rolls ≤ r. P(B_r) = (r/6)^n

P(max = r) = P(B_r) - P(B_{r-1}) = (r/6)^n - ((r-1)/6)^n

For example, P(max = 6) = (6/6)^n - (5/6)^n = 1 - (5/6)^n

---

### Q20. [Two Sigma — Hard] [Day 15: LOTUS / Expected Value]
**Sample Until Sum Exceeds 1**

Sample i.i.d. Uniform(0,1) until sum exceeds 1. Expected number of samples?

**Answer:**

Let N = number of samples until sum exceeds 1. This is a classic result.

P(N > n) = P(X₁ + X₂ + ... + Xₙ ≤ 1) = 1/n!  [volume of n-simplex in [0,1]^n]

E[N] = Σ_{n=1}^∞ P(N ≥ n) = Σ_{n=0}^∞ P(N > n) = Σ_{n=0}^∞ 1/n! = e ≈ **2.718**

Expected number of samples = e.

---

## SECTION 2: STATISTICS QUESTIONS

---

### Q21. [Facebook — Easy] [Day 25: Confidence Intervals]
**Explain Confidence Interval to Non-Technical Audience**

**Answer:**

"Imagine polling 1,000 voters to predict an election. You get 54% saying they'll vote for candidate A.

A 95% confidence interval of (51%, 57%) means: if we ran this poll 100 times with different random samples, 95 of those polls would produce an interval that contains the true population percentage.

It does NOT mean there's a 95% chance the true value is in this specific interval — the true value is fixed; it's the interval that varies.

For the business: we're 'confident' the true support is somewhere between 51-57%, which tells us candidate A is likely ahead."

---

### Q22. [Two Sigma — Easy] [Day 13: Covariance / Regression]
**Correlated Predictors in Linear Regression (Multicollinearity)**

How do correlated predictors affect regression results?

**Answer:**

Two main problems:
1. **Unstable coefficients**: Estimates change dramatically depending on which correlated variables are included. Standard errors inflate, making it hard to determine individual effects.
2. **Misleading p-values**: Important variables may appear insignificant (high p-values) even though they genuinely predict Y, because their variance is shared with correlated predictors.

**How to diagnose**: Variance Inflation Factor (VIF). VIF > 5-10 signals problematic multicollinearity.

**Solutions**:
- Remove one of the correlated predictors
- Combine them (e.g., average, PCA)
- Add interaction terms
- Regularization (Ridge regression penalizes large coefficients, stabilizing estimates)
- Collect more data (narrows confidence intervals)

---

### Q23. [Uber — Easy] [Day 26: Hypothesis Testing]
**Explain p-value in Layman's Terms**

**Answer:**

"A p-value answers: 'If there were truly no effect, how surprising would our data be?'

Example: You test a new drug. p-value = 0.03 means: if the drug truly did nothing, there's only a 3% chance of seeing results as good as we observed (or better) just by random luck.

Small p-value = data is surprising under the 'no effect' assumption → evidence against the null.

What it is NOT: It's not the probability the drug works. It's not the probability the result was a fluke. A p-value of 0.05 doesn't mean 5% chance of error — it means your result is as surprising as getting 1 head in 4 fair coin flips."

---

### Q24. [Google — Medium] [Day 25: Confidence Intervals / Day 9: Binomial]
**CI from Coin Tosses**

How do you derive a confidence interval from a series of coin tosses?

**Answer:**

n = total flips, k = heads observed, p̂ = k/n

By CLT, p̂ ≈ Normal(p, p(1-p)/n) for large n.

95% CI (Wald method):
p̂ ± 1.96 * sqrt(p̂*(1-p̂)/n)

Example: 1000 flips, 550 heads:
p̂ = 0.55
SE = sqrt(0.55*0.45/1000) = sqrt(0.0002475) ≈ 0.01573

95% CI: 0.55 ± 1.96 * 0.01573 = 0.55 ± 0.0308 = **(0.519, 0.581)**

For small n or extreme p̂: use Wilson interval instead:
p̃ = (k + 1.96²/2) / (n + 1.96²)

---

### Q25. [Stripe — Medium] [Day 22: MLE / Day 15: Exponential]
**MLE for Exponential λ**

Model customer lifetime as Exponential(λ). Given n lifetime observations, what is your best estimate of λ?

**Answer:**

Log-likelihood:
ℓ(λ) = Σᵢ log(λe^{-λxᵢ}) = n*log(λ) - λ*Σxᵢ

Set derivative to zero:
dℓ/dλ = n/λ - Σxᵢ = 0
λ̂ = n / Σxᵢ = 1/x̄

**MLE = 1 / sample mean**

Intuition: if average lifetime is 10 months, estimated rate = 0.1 events/month.

---

### Q26. [Lyft — Medium] [Day 11: Uniform Distribution]
**Derive Mean and Variance of Uniform(a,b)**

**Answer:**

f(x) = 1/(b-a) for a ≤ x ≤ b

E[X] = ∫ₐᵇ x/(b-a) dx = [x²/(2(b-a))]ₐᵇ = (b²-a²)/(2(b-a)) = (a+b)/2

E[X²] = ∫ₐᵇ x²/(b-a) dx = (b³-a³)/(3(b-a)) = (a²+ab+b²)/3

Var(X) = E[X²] - (E[X])² = (a²+ab+b²)/3 - (a+b)²/4 = **(b-a)²/12**

---

### Q27. [Google — Medium] [Day 11: Uniform / Day 15: LOTUS]
**E[min(X,Y)] for X,Y ~ Uniform(0,1)**

**Answer:**

P(min(X,Y) > t) = P(X>t)*P(Y>t) = (1-t)²  for t ∈ [0,1]

E[min(X,Y)] = ∫₀¹ P(min > t) dt = ∫₀¹ (1-t)² dt = [(-(1-t)³/3)]₀¹ = 0 - (-1/3) = **1/3**

General result: E[min of n Uniform(0,1)] = 1/(n+1)

---

### Q28. [Spotify — Medium] [Day 22: MLE]
**MLE for Uniform(0,d) Upper Bound**

n samples from Uniform(0,d). Best estimate of d?

**Answer:**

MLE: the likelihood is (1/d)^n for all observations ≤ d.
This is maximized by the smallest valid d, which is max(X₁,...,Xₙ).

**d̂_MLE = max(X₁,...,Xₙ)**

Note: This is biased — E[max] = n*d/(n+1) < d.
Unbiased estimator: d̂ = (n+1)/n * max(Xᵢ)

---

### Q29. [Quora — Medium] [Day 12: Normal / Day 9: Geometric]
**Expected Days Until Drawing Z > 2**

X ~ N(0,1) drawn daily. Expected days until X > 2?

**Answer:**

P(X > 2) = 1 - Φ(2) = 1 - 0.9772 = 0.0228

Each day is an independent Bernoulli trial with p = 0.0228.
Days until first success ~ Geometric(p).

E[days] = 1/p = 1/0.0228 ≈ **43.9 days**

---

### Q30. [Facebook — Medium] [Day 9: Geometric]
**Derive E[X] for Geometric Distribution**

**Answer:**

X ~ Geometric(p): P(X=k) = (1-p)^{k-1} * p, k=1,2,...

E[X] = Σ_{k=1}^∞ k*(1-p)^{k-1}*p

Let q = 1-p. E[X] = p * Σ_{k=1}^∞ k*q^{k-1}

Using d/dq[Σq^k] = Σk*q^{k-1} and Σq^k = 1/(1-q) = 1/p:
Σk*q^{k-1} = d/dq[1/(1-q)] = 1/(1-q)² = 1/p²

E[X] = p * (1/p²) = **1/p**

---

### Q31. [Google — Medium] [Day 25: Confidence Intervals / Day 26: Hypothesis Testing]
**Is a Coin Biased? 1000 Flips, 550 Heads**

**Answer:**

H₀: p = 0.5 (fair coin) vs H₁: p ≠ 0.5

Under H₀: X ~ Binomial(1000, 0.5) ≈ N(500, 250) by CLT.

Z = (550 - 500) / sqrt(250) = 50 / 15.81 ≈ **3.16**

P(|Z| > 3.16) ≈ 0.0016 << 0.05

**Reject H₀.** Strong evidence the coin is biased.

95% CI for p: 0.55 ± 1.96*sqrt(0.55*0.45/1000) = (0.519, 0.581)
The CI does not include 0.5, consistent with rejecting H₀.

---

### Q32. [Uber — Hard] [Day 22: MLE / Day 24: MAP]
**MLE vs MAP: Mathematical Description**

**Answer:**

**MLE (Maximum Likelihood Estimation):**
θ̂_MLE = argmax_θ P(data | θ) = argmax_θ Σᵢ log f(xᵢ; θ)

Finds θ that makes observed data most likely. No prior.

**MAP (Maximum A Posteriori):**
θ̂_MAP = argmax_θ P(θ | data)
       = argmax_θ [log P(data|θ) + log P(θ)]
       = argmax_θ [ℓ(θ) + log P(θ)]

Adds log-prior as a regularizer.

**Key connections:**
- Gaussian prior N(0, τ²): log P(θ) = -||θ||²/(2τ²) → MAP = Ridge (L2) regression
- Laplace prior: → MAP = Lasso (L1) regression
- As n→∞: MAP → MLE (data dominates prior)
- MLE can overfit on small data; MAP regularizes

**Example (Bernoulli):**
MLE: p̂ = k/n
MAP with Beta(α,β) prior: p̂ = (k+α-1)/(n+α+β-2)

---

### Q33. [Google — Hard] [Day 8: Variance / Day 13: Covariance]
**Blended Mean and Variance of Two Subsets**

Two dataset subsets with known means and standard deviations. How to compute combined mean and SD?

**Answer:**

Given: subset 1 has n₁ samples, mean μ₁, std σ₁
       subset 2 has n₂ samples, mean μ₂, std σ₂

**Combined mean:**
μ = (n₁μ₁ + n₂μ₂) / (n₁ + n₂)

**Combined variance:**
Need to account for both within-group variance and between-group variance:

σ² = [n₁(σ₁² + μ₁²) + n₂(σ₂² + μ₂²)] / (n₁+n₂) - μ²
   = [n₁σ₁² + n₂σ₂² + n₁(μ₁-μ)² + n₂(μ₂-μ)²] / (n₁+n₂)

**Extension to K subsets:**
μ = (Σᵢ nᵢμᵢ) / N  where N = Σnᵢ
σ² = Σᵢ nᵢ(σᵢ² + (μᵢ-μ)²) / N

This is Eve's Law: Var(Y) = E[Var(Y|group)] + Var(E[Y|group])

---

### Q34. [Lyft — Hard] [Day 11: Uniform / Monte Carlo]
**Uniformly Sample a Point Inside a Unit Circle**

**Answer:**

Naive approach (rejection sampling):
1. Sample (x,y) uniformly from [-1,1]²
2. If x²+y² ≤ 1: accept; else reject

Acceptance rate = π/4 ≈ 78.5%. Efficient enough.

**Better (inverse CDF method):**
The area of circle of radius r is πr². So CDF of radius R is F(r) = r² for r ∈ [0,1].

Sample r: let u ~ Uniform(0,1), then r = sqrt(u)  [F⁻¹(u) = sqrt(u)]
Sample angle: θ ~ Uniform(0, 2π)

x = r*cos(θ),  y = r*sin(θ)

This avoids rejection and is exact.

**Why NOT sample r ~ Uniform(0,1)?** Because area grows as r², so you need more points near the edge. Sqrt-transforming corrects for this.

---

### Q35. [Two Sigma — Hard] [Day 8: Expected Value / Day 9: Geometric]
**E[flips for Two Consecutive Heads] — Alternate Approach**

(See Q13 for the answer E[X]=6. This question asks for the derivation using the law of total expectation.)

**Answer (full derivation):**

States: S₀ = start/last was T, S₁ = last was H
E₀ = E[additional flips from S₀], E₁ = E[additional flips from S₁]

From S₀ (start):
- Flip H (prob 1/2): move to S₁, cost 1 flip
- Flip T (prob 1/2): stay in S₀, cost 1 flip
E₀ = 1/2*(1+E₁) + 1/2*(1+E₀) = 1 + E₁/2 + E₀/2
E₀/2 = 1 + E₁/2   →   E₀ = 2 + E₁  ...(1)

From S₁ (last was H):
- Flip H (prob 1/2): done! Cost 1 flip
- Flip T (prob 1/2): go back to S₀, cost 1 flip
E₁ = 1/2*(1) + 1/2*(1+E₀) = 1 + E₀/2  ...(2)

Substitute (1) into (2): E₁ = 1 + (2+E₁)/2 = 1 + 1 + E₁/2
E₁/2 = 2 → E₁ = 4
E₀ = 2 + 4 = **6**

---

### Q36. [Walmart — Medium] [Day 4: Bayes / Day 3: Conditional Probability]
**Boy Born on Tuesday Problem**

You have two children. One is a boy born on a Tuesday. P(both children are boys)?

**Answer:**

This is a famous problem where the extra information (Tuesday) actually changes the answer.

Without the Tuesday constraint:
P(both boys | at least one boy) = 1/3

With Tuesday constraint:
Sample space: (day, sex) combinations for each child = 14 options each.

Total two-child combinations where at least one is a boy born on Tuesday:
- Child 1 is boy-Tuesday: 14 options for child 2 (all combos)
- Child 2 is boy-Tuesday: 14 options for child 1
- Both are boy-Tuesday: counted twice = subtract 1

Total = 14 + 14 - 1 = 27 combinations

Both boys: child 1 = boy-Tuesday, child 2 = any boy (7 options) + child 1 = any boy (7 options), child 2 = boy-Tuesday - 1 overlap = 7+7-1 = 13

P(both boys | one is boy born Tuesday) = 13/27 ≈ **0.481**

Remarkably, this is almost 1/2, not 1/3!

---

### Q37. [Walmart — Medium] [Day 9: Binomial / Day 10: Poisson]
**Sampling from a Discrete Distribution**

Given events [2,3,4] with probabilities [0.2, 0.3, 0.5], generate 100 samples programmatically.

**Answer (concept):**

Use the inverse CDF (quantile) method:
1. Compute cumulative probabilities: [0.2, 0.5, 1.0]
2. For each sample: draw u ~ Uniform(0,1)
3. Return the smallest event where cumulative prob ≥ u

```python
import numpy as np

events = [2, 3, 4]
probs = [0.2, 0.3, 0.5]
samples = np.random.choice(events, size=100, p=probs)
```

Expected counts: 20 twos, 30 threes, 50 fours.

**Connection to Day 11**: This is the inverse CDF method. F⁻¹(U) ~ the target distribution when U ~ Uniform(0,1).

---

## SECTION 3: ML STATISTICS QUESTIONS (COMPANY-SPECIFIC)

---

### Q38. [Google — Medium] [Day 27: A/B Testing]
**Design an A/B Test for Google Search Ranking**

**Answer (framework):**

1. **Hypothesis**: H₀: new ranking has same CTR as old; H₁: new ranking has higher CTR

2. **Metric**: Primary = CTR; Guardrail = latency (must not increase), zero-results rate

3. **Randomization unit**: User (not query/session) — ensures consistent experience and avoids novelty effects within-user

4. **Sample size**: 
   n = 2*(z_α/2 + z_β)² * p(1-p) / δ²
   If baseline CTR=0.30, MDE=0.5pp: n ≈ 100,000+ per arm

5. **Duration**: At least 2 full weeks (capture Mon-Sun cycle; search behavior differs by day)

6. **SRM check**: Verify n_control ≈ n_treatment before analyzing

7. **Analysis**: Two-proportion z-test; report CI for lift, not just p-value

8. **Decision**: Ship only if primary metric significantly improves AND all guardrails hold

---

### Q39. [Facebook/Meta — Medium] [Day 26: Hypothesis Testing]
**P-value is 0.04 — Should We Ship?**

**Answer:**

Not necessarily. p=0.04 alone is insufficient. Ask:

1. **Pre-specified?** Was this the planned primary metric, or one of many tested?
   If 20 metrics tested: expected 1 false positive at α=0.05. Apply BH correction.

2. **SRM check passed?** If not, p-value is invalid.

3. **Practical significance?** What is the actual effect size? A 0.01% lift with p=0.04 is statistically real but may not justify engineering cost.

4. **Guardrail metrics?** Did latency, revenue, or retention degrade?

5. **Novelty effect?** Did the test run long enough for novelty to wash out?

6. **Replication?** Has this been seen before in similar experiments?

Decision: p=0.04 is evidence in favor, but ship only after checking all the above.

---

### Q40. [Amazon — Medium] [Day 26: Hypothesis Testing / Type I Error]
**Multiple Testing When Evaluating Many Models**

You test 50 hyperparameter configurations and report the best one. What's the statistical problem?

**Answer:**

**Multiple comparisons problem:** Testing 50 configs at α=0.05 each:
P(at least one false positive) = 1-(1-0.05)^50 ≈ 92.3%

You'll almost certainly find one configuration that "beats" baseline by chance.

**Solutions:**
1. **Bonferroni correction**: α* = 0.05/50 = 0.001 per test
2. **Benjamini-Hochberg (BH)**: Control FDR at 5% — more powerful than Bonferroni
3. **Hold-out validation**: Select best config on validation set, THEN test once on held-out test set
4. **Pre-register**: Decide the top-K configurations before final evaluation

**Best practice for ML**: Hyperparameter search on val set; report final numbers on unseen test set. Never tune to the test set.

---

### Q41. [Apple — Medium] [Day 5: Independence / Day 13: Covariance]
**Implement Naive Bayes**

Explain and implement Naive Bayes classifier (reported Apple MLE interview question).

**Answer:**

Naive Bayes classifies using Bayes' theorem with conditional independence assumption:

P(y | x₁,...,xₙ) ∝ P(y) * Π P(xᵢ | y)

"Naive" = features assumed conditionally independent given class.

Gaussian Naive Bayes implementation concept:

```python
# Training: estimate parameters per class
for each class c:
    mu_c[feature] = mean(X[feature] where y==c)
    sigma_c[feature] = std(X[feature] where y==c)
    prior[c] = count(y==c) / n

# Prediction: log posterior
for each class c:
    log_prob[c] = log(prior[c])
    for each feature i:
        log_prob[c] += log N(x_i; mu_c[i], sigma_c[i])
predict = argmax(log_prob)
```

**Why log?** Products of probabilities underflow numerically; log converts to sums.

---

### Q42. [Meta — Hard] [Day 20: CLT / Day 22: MLE]
**Given Bernoulli Generator, Return Normal Sample**

Given a Bernoulli(p) random generator, return a value sampled from N(0,1).

**Answer:**

**Method 1: CLT approach**
Sum n Bernoulli(p) samples: S_n = Σ X_i where X_i ~ Bernoulli(p)
By CLT: (S_n - np) / sqrt(np(1-p)) →d N(0,1) as n→∞

With n=1000, p=0.5: Z ≈ (S_1000 - 500) / sqrt(250) is approximately N(0,1).

**Method 2: Box-Muller (needs Uniform first)**
Generate U₁, U₂ ~ Uniform(0,1) from Bernoulli via Von Neumann trick.
Then: Z = sqrt(-2*log(U₁)) * cos(2π*U₂) ~ N(0,1)

**Method 3: Inverse CDF**
Generate u ~ Uniform(0,1) from Bernoulli.
Then: Z = Φ⁻¹(u) ~ N(0,1) where Φ⁻¹ is the standard normal quantile function.

---

## SECTION 4: QUICK-REFERENCE COMPANY DIFFICULTY MAP

| Company | Typical Difficulty | Focus Areas |
|---------|-------------------|-------------|
| Google | Medium-Hard | Probability puzzles, statistics, A/B testing, system design |
| Facebook/Meta | Easy-Medium | Bayes, A/B testing, metrics, product sense |
| Amazon | Easy-Medium | Hypothesis testing, product metrics, regression |
| Apple | Medium (team-dependent) | ML fundamentals, Naive Bayes, probability |
| Walmart | Easy-Medium | Conditional probability, expected value, basic ML |
| Uber | Medium-Hard | MLE/MAP, probability puzzles, A/B testing |
| Two Sigma | Hard | Expected value, probability derivations, quant-style |
| Lyft | Medium | Distributions, simulation, statistics |
| Bloomberg | Easy-Medium | Probability puzzles, basic stats |
| Robinhood | Medium | Permutations, symmetry arguments, regression |
| Stripe | Medium | MLE, exponential distribution, survival analysis |
| Spotify | Hard | Combinatorics, die rolls, continuous distributions |
| Airbnb | Medium | Fair sampling, simulation |
| Snapchat | Hard | Combinatorics, graph probability |

---

## SECTION 5: TOP TOPICS BY FREQUENCY

Based on public interview reports (Glassdoor, Blind, NickSingh.com, Medium):

**Most Frequently Asked Topics:**
1. Bayes' theorem — disease testing, coin fairness (Facebook, Amazon, Microsoft)
2. A/B testing design and pitfalls (Google, Meta, Uber, Amazon)
3. Expected value puzzles — geometric, coupon collector (Facebook, Two Sigma)
4. Confidence intervals and hypothesis testing (nearly all companies)
5. MLE vs MAP (Uber, Google, Two Sigma)
6. Conditional probability — two children, three friends (Walmart, Lyft)
7. Counting / combinatorics — chords, cards, dice (Bloomberg, Google, Tesla)
8. CLT applications (Google, Meta, Amazon)
9. Regression and multicollinearity (Two Sigma, Uber)
10. Sampling methods — uniform circle, inverse CDF (Lyft, Two Sigma)

---

*Sources: NickSingh.com "40 Probability & Statistics Interview Questions Asked by FAANG", Glassdoor interview reports, Blind.com interview threads, Medium interview experience posts. Questions are paraphrased from public sources; answers are original derivations.*
