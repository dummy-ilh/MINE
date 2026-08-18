# Day 30 — Final Review: 20 Top ML Interview Q&As & Master Cheat Sheet
### 30-Day Probability & Statistics for AI/ML Interviews

---

## PART 1: THE 20 MOST IMPORTANT INTERVIEW QUESTIONS

### Q1. MLE vs MAP?
MLE: argmax logL(theta) — no prior
MAP: argmax [logL(theta) + log P(theta)] — log-prior = regularizer
Gaussian prior -> L2 reg (lambda=1/2tau^2); Laplace -> L1; Uniform -> MLE
As n->inf: MAP -> MLE

### Q2. Derive MSE = Bias^2 + Variance.
E[(Y-fhat)^2] = (E[fhat]-Y)^2 + Var(fhat) = Bias^2 + Variance
Cross term vanishes because E[fhat - E[fhat]] = 0.
Simple models: high bias, low variance. Complex: low bias, high variance.

### Q3. Why cross-entropy for classification?
CE = negative Bernoulli/Categorical log-likelihood.
Minimizing CE = MLE for P(Y|X).
Universal gradient: sum_i (y_i - p_hat_i) x_i  [true-predicted x feature]

### Q4. Bayes theorem in ML?
P(theta|data) = P(data|theta) P(theta) / P(data)
Posterior prop to Likelihood x Prior
L2 reg = Gaussian prior; L1 = Laplace prior
Beta(alpha,beta) + (k successes, n-k fails) -> Beta(alpha+k, beta+n-k)

### Q5. CLT and why it matters?
sqrt(n)(Xbar - mu)/sigma ->d N(0,1)
Justifies: CIs for any distribution, A/B test z-tests, SGD noise models,
He/Xavier initialization (pre-activations -> Normal by CLT).
Berry-Esseen: error <= C*E[|X-mu|^3]/(sigma^3 * sqrt(n)).

### Q6. What is a p-value? Common mistakes?
Correct: P(data this extreme | H0 true)
Wrong: P(H0 true), P(H1)=1-p, small p = large effect
Under H0: p-values ~ Uniform(0,1) -> 5% false positives at alpha=0.05
Fix: Bonferroni alpha/m or BH FDR correction.

### Q7. Type I vs Type II errors?
alpha = P(Type I) = P(reject | H0 true) = significance level
beta  = P(Type II) = P(accept | H0 false)
Power = 1-beta
Sample size: n = (z_{alpha/2} + z_beta)^2 * sigma^2 / delta^2

### Q8. How to design an A/B test?
1. Hypothesis + metrics (primary, guardrail)
2. Power analysis: n = 2(z+z)^2 p(1-p)/delta^2
3. Randomize at user level; duration >= 2 weekly cycles
4. SRM check: chi^2 on n_A vs n_B — if significant, abort
5. Z-test, CI for difference; check guardrail metrics
Pitfalls: peeking (Type I -> 22%), no SRM, too short.

### Q9. When does LLN fail?
LLN: P(|Xbar - mu| > eps) <= sigma^2/(n*eps^2) -> 0
Fails for: Cauchy (Xbar stays Cauchy for ALL n), heavy tails E[|X|]=inf,
non-i.i.d. data (time series, distribution shift).

### Q10. Frequentist CI vs Bayesian credible interval?
Frequentist: P(interval contains theta) = 95% over repeated sampling
Bayesian:    P(theta in interval | data) = 95% — direct probability statement
CI formula: Xbar +/- t_{alpha/2, n-1} * S/sqrt(n)  [use t, not z]
Halve width -> 4x sample size.

### Q11. KL divergence?
KL(P||Q) = sum P(x) log[P(x)/Q(x)] >= 0
Non-symmetric: KL(P||Q) != KL(Q||P)
Forward KL: mode-covering; Reverse KL: mode-seeking (VAE uses reverse)
KL(N(mu,sigma^2)||N(0,1)) = (mu^2 + sigma^2 - log sigma^2 - 1)/2
ELBO: log p(x) - ELBO = KL(q||p) >= 0  [via Jensen's]

### Q12. Mutual information for feature selection?
I(X;Y) = H(Y) - H(Y|X) >= 0, = 0 iff independent, symmetric
Advantage over Pearson: detects ANY dependence (nonlinear too)
Decision tree splitting: IG = I(Y;feature) = reduction in label entropy.

### Q13. Bernoulli / Binomial / Poisson?
Bernoulli(p): single trial, E=p, Var=p(1-p)
Binomial(n,p): n trials, E=np, Var=np(1-p)
Poisson(lambda): rare events, E=Var=lambda
Var ~ Mean -> Poisson; Var > Mean -> Negative Binomial (overdispersion).

### Q14. Why does Normal appear everywhere?
1. CLT: sum of any i.i.d. vars -> Normal
2. Max entropy: given fixed mean and variance, Normal has highest entropy
3. Closed under linear operations: sum of Normals is Normal
He init: N(0, 2/n_in); Xavier: N(0, 1/n_in); L2 reg = N(0,1/lambda) prior.

### Q15. Exponential distribution?
f(x)=lambda*exp(-lambda*x), E=1/lambda, Var=1/lambda^2
Memoryless: P(X>s+t|X>s) = P(X>t)  [unique among continuous]
Poisson process rate lambda -> inter-arrivals ~ Exponential(lambda).
M/M/1 queue: stable iff lambda < mu.

### Q16. Fisher Information and CRLB?
I(theta) = E[(d log f/d theta)^2] = -E[d^2 log f/d theta^2]
CRLB: Var(theta_hat) >= 1/(n*I(theta))  [unbiased estimators]
MLE achieves CRLB asymptotically -> most efficient.
Bernoulli: I(p)=1/p(1-p); Poisson: I(lambda)=1/lambda.

### Q17. Jensen's inequality in ML?
phi convex: phi(E[X]) <= E[phi(X)]
ELBO: log p(x) = log E_q[p/q] >= E_q[log p/q]  [log concave, Jensen]
KL >= 0: from Jensen on convex -log
H(P,Q) >= H(P): cross-entropy >= true entropy

### Q18. Markov chains in ML?
Markov: P(X_{n+1}|X_n,...,X_0) = P(X_{n+1}|X_n)
Stationary: pi*P = pi
Detailed balance: pi_i*P_ij = pi_j*P_ji -> pi stationary [MCMC]
M-H acceptance: A(i->j) = min(1, pi(j)/pi(i)) [symmetric proposal]
Uses: RL/MDPs (Bellman), MCMC (posterior sampling), PageRank, diffusion models.

### Q19. Entropy and ML connections?
H(X) = -sum P(x) log P(x) = E[-log P(X)]
Cross-entropy H(P,Q): classification loss
KL(P||Q) = H(P,Q) - H(P): regularization
I(X;Y) = H(Y)-H(Y|X): feature importance
Perplexity = exp(H): language model quality
Info gain = I(Y;feature): decision tree splits

### Q20. L2 regularization = Gaussian prior. Proof?
MAP with w ~ N(0, tau^2 I):
log P(w) = -||w||^2/(2*tau^2)
MAP: maximize l(w) - ||w||^2/(2*tau^2)
   = minimize -l(w) + lambda*||w||^2  where lambda = 1/(2*tau^2)
This IS L2 regularization. Large lambda -> small tau^2 -> strong regularization.
L1 reg = Laplace prior w ~ Laplace(0, 1/lambda).

---

## PART 2: MASTER CHEAT SHEET

### Unit 1: Probability Foundations (Days 1-6)
```
P(A^c) = 1-P(A)
P(A∪B) = P(A)+P(B)-P(A∩B)
C(n,r) = n!/[r!(n-r)!];  P(n,r) = n!/(n-r)!
P(A|B) = P(A∩B)/P(B)
LOTP: P(A) = sum_i P(A|B_i)P(B_i)
Bayes: P(A|B) = P(B|A)P(A)/P(B)
Independence: P(A∩B) = P(A)P(B)
Bootstrap OOB fraction -> 1/e ≈ 0.368
```

### Unit 2: Distributions (Days 7-14)
```
Bernoulli(p):  E=p, Var=p(1-p)
Binomial(n,p): E=np, Var=np(1-p)
Geometric(p):  E=1/p, Var=(1-p)/p^2, memoryless
Poisson(lam):  E=Var=lam, additive, PMF=e^{-lam}lam^k/k!
Uniform(a,b):  E=(a+b)/2, Var=(b-a)^2/12
Exp(lam):      E=1/lam, Var=1/lam^2, memoryless, F=1-e^{-lam*x}
Normal(mu,s2): Z=(X-mu)/sigma~N(0,1), 68/95/99.7 rule
Var(X+Y) = Var(X)+Var(Y)+2Cov(X,Y)
MSE = Bias^2 + Variance
PCA = eigendecomposition of covariance matrix Sigma
```

### Unit 3: Expectation Tools (Days 15-18)
```
LOTUS: E[g(X)] = sum g(x)p(x)
Linearity: E[aX+bY] = aE[X]+bE[Y]  [always, no independence needed]
Indicators: E[I_A] = P(A);  E[sum I_i] = sum P(I_i=1)
MGF: M(t)=E[e^{tX}]; E[X^n]=M^(n)(0)
Tower: E[Y] = E[E[Y|X]]
Eve's Law: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
Markov: P(X>=a) <= E[X]/a  [X>=0]
Chebyshev: P(|X-mu|>=k) <= sigma^2/k^2
Jensen: phi convex -> phi(E[X]) <= E[phi(X)]
Hoeffding: P(|Xbar-mu|>=eps) <= 2exp(-2n*eps^2)
```

### Unit 4: Limit Theorems (Days 19-21)
```
LLN: P(|Xbar-mu|>eps) <= sigma^2/(n*eps^2) -> 0
CLT: sqrt(n)(Xbar-mu)/sigma ->d N(0,1); Xbar ~ N(mu, sigma^2/n)
(n-1)S^2/sigma^2 ~ chi^2(n-1)
(Xbar-mu)/(S/sqrt(n)) ~ t(n-1)   [sigma unknown]
[t(nu)]^2 ~ F(1,nu)
S1^2/S2^2 ~ F(n1-1, n2-1)
```

### Unit 5: Inference (Days 22-27)
```
MLE: theta_hat = argmax sum log f(xi;theta)
  Bernoulli: k/n;  Normal: xbar, (1/n)sum(xi-xbar)^2
  Poisson: xbar;  Exponential: 1/xbar
MAP: Gaussian prior -> L2; Laplace -> L1
Beta-Bernoulli: Beta(alpha,beta) + data -> Beta(alpha+k, beta+n-k)
CI (sigma unknown): Xbar +/- t_{alpha/2,n-1} * S/sqrt(n)
CI (proportion): phat +/- z*sqrt(phat*(1-phat)/n)
n for CI: z^2*p(1-p)/E^2;  halve width -> 4x samples
p-value = P(data|H0) != P(H0)
Power=1-beta; n=(z_alpha+z_beta)^2*sigma^2/delta^2
BH correction: reject p_{(k)} <= k*alpha/m  [FDR control]
A/B SRM: chi^2 on n_A vs n_B
```

### Unit 6: Advanced (Days 28-30)
```
H(X) = -sum P(x)log P(x)
H(P,Q) = H(P) + KL(P||Q) >= H(P)
KL(P||Q) = sum P(x)log[P(x)/Q(x)] >= 0
KL(N(mu,s2)||N(0,1)) = (mu^2+s2-log s2-1)/2
I(X;Y) = H(Y)-H(Y|X) >= 0
Perplexity = exp(cross-entropy)
Markov: pi*P=pi; detailed balance: pi_i*P_ij=pi_j*P_ji
M-H: A(i->j) = min(1, pi(j)/pi(i))
```

---

## PART 3: KEY ML CONNECTIONS

```
LOSS           DISTRIBUTION    REGULARIZATION   PRIOR
MSE            Gaussian        L2 (lam||w||^2)  N(0,1/2lam)
MAE            Laplace         L1 (lam||w||_1)  Laplace(0,1/lam)
Binary CE      Bernoulli       —                —
Multiclass CE  Categorical     —                —
ELBO           Gaussian latent KL(q||p)         N(0,I)

INITIALIZATION   DISTRIBUTION      WHY
He init          N(0, 2/n_in)      Preserves variance through ReLU
Xavier init      N(0, 1/n_in)      Preserves variance through linear/tanh
```

---

## PART 4: LIGHTNING ROUND

1. MLE for Bernoulli p?       -> k/n
2. L2 reg = which prior?      -> Gaussian N(0, 1/2lambda)
3. MSE = Bias^2 + ?           -> Variance
4. P(A∪B) = P(A)+P(B) - ?    -> P(A∩B)
5. CLT: sqrt(n)(Xbar-mu)/s -> ?  -> N(0,1)
6. p-values under H0 follow?  -> Uniform(0,1)
7. E[Y|X] minimizes?          -> E[(Y-g(X))^2]
8. KL(P||Q) >= ?              -> 0
9. Stationary dist satisfies? -> pi*P = pi
10. H(P,Q) = H(P) + ?        -> KL(P||Q)

---

## CONGRATULATIONS — 30 DAYS COMPLETE

Days 1-6:   Probability foundations (axioms, counting, Bayes, independence)
Days 7-14:  Distributions (PMF/CDF, discrete, continuous, covariance)
Days 15-18: Expectation tools (LOTUS, indicators, MGFs, inequalities)
Days 19-21: Limit theorems (LLN, CLT, t/chi-squared/F)
Days 22-27: Inference (MLE, MAP, Bayes, CIs, hypothesis tests, A/B testing)
Days 28-30: Advanced (information theory, Markov chains, review)

You are ready for probability and statistics at any ML/DS interview.

*End of Day 30 — Course Complete*
