# Day 28 — Information Theory: Entropy, KL Divergence & Cross-Entropy
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Cover & Thomas, *Elements of Information Theory*; Bishop PRML Chapter 1
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Information Theory is Core to ML

Information theory was invented by Claude Shannon in 1948. It became the mathematical foundation of ML without most practitioners realizing it.

| ML Concept | Information Theory Origin |
|---|---|
| Cross-entropy loss | Cross-entropy H(P,Q) |
| KL regularization in VAEs | KL divergence KL(q\|\|p) |
| Decision tree splitting | Information gain (mutual information) |
| Feature selection | Mutual information I(X;Y) |
| Compression | Entropy H(X) = minimum bits needed |
| Language model perplexity | exp(cross-entropy) |
| Maximum entropy models | Principle of maximum entropy |
| Variational inference | ELBO = negative KL + reconstruction |
| Neural network training | MLE ≡ minimizing cross-entropy |

Information theory gives us the language to precisely measure uncertainty, information, and the "cost" of making wrong probabilistic predictions.

---

## 2. Self-Information (Surprisal)

> **Definition:** The **self-information** (surprisal) of an event with probability p is:
> ```
> I(p) = −log p    [in nats if log=ln, bits if log=log₂]
> ```

**Intuition:** How surprised are you when event with probability p occurs?

```
p = 1.0: I = 0        [certain event — no surprise]
p = 0.5: I = 1 bit    [fair coin flip]
p = 0.01: I = log₂(100) ≈ 6.64 bits  [rare event — very surprising]
p → 0:   I → ∞       [impossible event — infinitely surprising]
```

**Why −log p?**
- Additivity: for independent events A,B: I(A∩B) = I(A) + I(B)
  - log(p·q) = log p + log q → −log(pq) = −log p + (−log q) ✓
- Monotone decreasing: rarer events = more information
- The only function satisfying both properties (Shannon's uniqueness theorem)

---

## 3. Entropy

> **Definition:** The **entropy** of a discrete random variable X with PMF P is:
> ```
> H(X) = −Σₓ P(x) log P(x) = E[−log P(X)]
> ```

**Entropy = Expected surprisal = Average information content**

This is LOTUS (Day 15) with g(x) = −log P(x).

### Units

```
log base 2:  entropy in bits  [most common in information theory]
log base e:  entropy in nats  [used in ML with natural log]
log base 10: entropy in hartleys

Conversion: 1 nat = log₂(e) ≈ 1.443 bits
```

### Entropy of Bernoulli(p)

```
H(p) = −p log p − (1−p) log(1−p)
```

| p | H(p) in bits |
|---|---|
| 0.0 or 1.0 | 0 (no uncertainty) |
| 0.1 or 0.9 | 0.469 bits |
| 0.3 or 0.7 | 0.881 bits |
| **0.5** | **1 bit** (maximum) |

Maximum entropy at p=0.5: maximum uncertainty.

### Properties of Entropy

```
1. H(X) ≥ 0                    [non-negative]
2. H(X) = 0 iff X is deterministic
3. H(X) ≤ log|X|               [max entropy = log of support size]
4. H(X) = log|X| iff X is uniform [uniform = maximum entropy]
5. H(X,Y) ≤ H(X) + H(Y)       [joint ≤ sum of marginals]
6. H(X|Y) ≤ H(X)              [conditioning reduces entropy]
7. H(X,Y) = H(X) + H(Y|X)     [chain rule for entropy]
```

### Maximum Entropy Principle

Among all distributions with given constraints (mean, variance, support), the **maximum entropy distribution** is the "most uncertain" / "least informative":

| Constraint | Max Entropy Distribution |
|---|---|
| Finite support {1,...,n} | Uniform |
| Fixed mean μ on [0,∞) | Exponential |
| Fixed mean μ and variance σ² on ℝ | Normal |
| Fixed mean on ℝ⁺, log-scale variance | Log-Normal |

This is why the Normal is the default — it's the maximum entropy distribution given only mean and variance.

---

## 4. Joint, Conditional, and Mutual Information

### Joint Entropy

```
H(X,Y) = −Σₓ Σᵧ P(x,y) log P(x,y)
```

### Conditional Entropy

```
H(X|Y) = −Σₓ Σᵧ P(x,y) log P(x|y)
        = E_{Y}[H(X|Y=y)]
        = H(X,Y) − H(Y)
```

**Intuition:** Average entropy of X after observing Y. How much uncertainty remains about X given Y?

### Chain Rule for Entropy

```
H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
```

### Mutual Information

> **Definition:** The **mutual information** between X and Y is:
> ```
> I(X;Y) = H(X) − H(X|Y) = H(Y) − H(Y|X)
>         = H(X) + H(Y) − H(X,Y)
>         = Σₓ Σᵧ P(x,y) log[P(x,y)/(P(x)P(y))]
> ```

**Intuition:** How much does knowing Y reduce our uncertainty about X?

```
I(X;Y) = 0 iff X and Y are independent    [knowing Y gives no info about X]
I(X;Y) = H(X) iff X is a function of Y   [Y completely determines X]
I(X;Y) ≥ 0                                [always]
I(X;Y) = I(Y;X)                           [symmetric]
```

**ML use:** Mutual information is the feature selection criterion that detects ANY dependence (linear or nonlinear) between feature X and target Y.

---

## 5. Cross-Entropy

> **Definition:** The **cross-entropy** between distributions P (true) and Q (model) is:
> ```
> H(P,Q) = −Σₓ P(x) log Q(x) = E_P[−log Q(X)]
> ```

**Intuition:** Expected number of bits needed to encode samples from P when using a code optimized for Q.

### Key Relationship

```
H(P,Q) = H(P) + KL(P||Q)

where KL(P||Q) = Σₓ P(x) log[P(x)/Q(x)] ≥ 0
```

Cross-entropy = True entropy + KL divergence

Since H(P) is fixed (doesn't depend on Q):
```
Minimizing H(P,Q) over Q ⟺ Minimizing KL(P||Q)
⟺ Making Q as close to P as possible
```

### Cross-Entropy as ML Loss

For classification with true distribution P (one-hot labels) and model output Q:

```
H(P,Q) = −Σₖ P(k) log Q(k) = −log Q(true class)    [since P is one-hot]
```

**Cross-entropy loss = −log(probability assigned to correct class)**

This is why minimizing cross-entropy = maximizing log-likelihood = MLE (Day 23).

---

## 6. KL Divergence

> **Definition:** The **Kullback-Leibler divergence** from Q to P is:
> ```
> KL(P||Q) = Σₓ P(x) log[P(x)/Q(x)] = E_P[log P(X)/Q(X)]
> ```

**Intuition:** Extra bits needed to encode data from P using code optimized for Q, vs code optimized for P.

### Properties

```
KL(P||Q) ≥ 0                    [Gibbs' inequality — always non-negative]
KL(P||Q) = 0 iff P = Q
KL(P||Q) ≠ KL(Q||P)            [NOT symmetric — not a distance!]
KL(P||Q) = H(P,Q) − H(P)
```

### Proof of Non-negativity (via Jensen's)

```
KL(P||Q) = E_P[log P(X)/Q(X)] = −E_P[log Q(X)/P(X)]

Since −log is convex, by Jensen's:
−E_P[log Q/P] ≥ −log E_P[Q/P] = −log[Σₓ P(x)·Q(x)/P(x)] = −log[Σₓ Q(x)] = −log 1 = 0
```

KL(P||Q) ≥ 0 with equality iff P = Q. ∎

### Forward vs Reverse KL

```
KL(P||Q) — "forward KL":
    E_P[log P/Q] — expectation under P
    Minimizing → Q covers all modes of P (mode-covering)
    Q may spread mass where P has none

KL(Q||P) — "reverse KL":
    E_Q[log Q/P] — expectation under Q
    Minimizing → Q concentrates on modes of P (mode-seeking)
    Q avoids regions where P=0
```

**In variational inference:**
- ELBO maximization ≡ minimizing KL(q||p) (reverse KL)
- This causes the approximate posterior q to be mode-seeking

---

## 7. KL Divergence Between Gaussians

This formula appears constantly in VAEs and Bayesian ML:

```
KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²)/(2σ₂²) − 1/2
```

**Special case: KL from N(μ,σ²) to N(0,1):**
```
KL(N(μ,σ²) || N(0,1)) = (μ² + σ² − log σ² − 1) / 2
```

This is the VAE regularization term — derived from this exact formula (Day 18, Day 24).

---

## 8. Information Gain in Decision Trees

The **information gain** of splitting on feature X for target Y:

```
IG(Y; X) = H(Y) − H(Y|X) = I(Y; X)
```

Choose the feature X that maximizes information gain (= mutual information with Y).

**Gini impurity** is an approximation of entropy used in CART (Classification and Regression Trees):
```
Gini(p) = 1 − Σₖ pₖ² ≈ 2H(p) for binary case
```

Both are measures of node impurity. Entropy-based splitting = ID3 algorithm.

---

## 9. Perplexity

> **Definition:** The **perplexity** of a language model P on test data x₁,...,xₙ is:
> ```
> PP(P) = exp(H(P̃, P)) = exp(−(1/n) Σᵢ log P(xᵢ))
>       = 2^(cross-entropy in bits)
> ```

**Intuition:** Perplexity ≈ effective vocabulary size the model is "confused" about at each step.

```
Perplexity = 10: model is as confused as uniformly choosing among 10 words
Perplexity = 100: model is choosing among 100 equally likely words
Lower perplexity = better language model
```

**Relationship to cross-entropy:**
```
Cross-entropy (nats) H = log(Perplexity)
Perplexity = e^H
```

GPT-2 achieved perplexity ≈ 35 on Penn Treebank; GPT-4 achieves much lower.

---

## 10. Worked Numericals

---

### 🔢 Numerical 1 — Computing Entropy

**Problem:** A 4-class classifier output distribution P = (0.6, 0.2, 0.1, 0.1).

**(a)** Entropy H(P) in bits.
**(b)** Entropy of Uniform(4 classes) for comparison.
**(c)** Entropy of a deterministic prediction (one class = 1.0).
**(d)** What does this tell you about model confidence?

**Solution:**

**(a)**
```
H(P) = −0.6·log₂(0.6) − 0.2·log₂(0.2) − 0.1·log₂(0.1) − 0.1·log₂(0.1)
     = −0.6·(−0.737) − 0.2·(−2.322) − 0.1·(−3.322) − 0.1·(−3.322)
     = 0.442 + 0.464 + 0.332 + 0.332
     = 1.570 bits
```

**(b)** Uniform: H = log₂(4) = **2 bits** (maximum)

**(c)** Deterministic: H = −1·log₂(1) = **0 bits** (no uncertainty)

**(d)** Scale:
- 0 bits: certain (bad for calibration unless truly deterministic)
- 1.570 bits: some confidence (model picks class 1 with 60% confidence)
- 2 bits: completely uncertain (flat distribution)

This model has 78.5% of maximum uncertainty — it's moderately confident about one class but not certain.

**ML insight:** Entropy of model output is used in:
- Active learning: query samples with highest entropy (most uncertain)
- Confidence calibration: well-calibrated models have appropriate entropy
- Knowledge distillation: softer distributions (higher entropy) transfer more knowledge

---

### 🔢 Numerical 2 — Mutual Information: Feature Selection

**Problem:** Feature X and target Y have joint distribution:

|  | Y=0 | Y=1 |
|---|---|---|
| X=0 | 0.30 | 0.10 |
| X=1 | 0.20 | 0.40 |

**(a)** Marginal distributions P(X), P(Y).
**(b)** H(Y) — baseline uncertainty about Y.
**(c)** H(Y|X) — remaining uncertainty after observing X.
**(d)** I(X;Y) — how much does X tell us about Y?
**(e)** Compare to a useless feature X' where X' and Y are independent.

**Solution:**

**(a)**
```
P(X=0) = 0.30+0.10 = 0.40,  P(X=1) = 0.20+0.40 = 0.60
P(Y=0) = 0.30+0.20 = 0.50,  P(Y=1) = 0.10+0.40 = 0.50
```

**(b)** H(Y) = −0.5·log₂(0.5) − 0.5·log₂(0.5) = **1 bit** (maximum — 50/50)

**(c)**

P(Y=0|X=0) = 0.30/0.40 = 0.75, P(Y=1|X=0) = 0.25
P(Y=0|X=1) = 0.20/0.60 = 0.333, P(Y=1|X=1) = 0.667

```
H(Y|X=0) = −0.75·log₂(0.75) − 0.25·log₂(0.25) = 0.311+0.500 = 0.811 bits
H(Y|X=1) = −0.333·log₂(0.333) − 0.667·log₂(0.667) = 0.528+0.390 = 0.918 bits

H(Y|X) = P(X=0)·H(Y|X=0) + P(X=1)·H(Y|X=1)
        = 0.40×0.811 + 0.60×0.918
        = 0.324 + 0.551 = 0.875 bits
```

**(d)** I(X;Y) = H(Y) − H(Y|X) = 1.000 − 0.875 = **0.125 bits**

Knowing X reduces uncertainty about Y by 0.125 bits — a 12.5% reduction.

**(e)** If X' and Y are independent: I(X';Y) = 0. Knowing X' gives zero information about Y.

**ML insight:** This is the information gain criterion in ID3 decision trees. Feature X with I(X;Y)=0.125 bits would be selected if no other feature provides more information about Y.

---

### 🔢 Numerical 3 — Cross-Entropy Loss: Classification

**Problem:** A 3-class classifier outputs probabilities for 3 samples:

| Sample | True Label | Prob(class 0) | Prob(class 1) | Prob(class 2) |
|---|---|---|---|---|
| 1 | Class 0 | 0.7 | 0.2 | 0.1 |
| 2 | Class 1 | 0.1 | 0.8 | 0.1 |
| 3 | Class 2 | 0.3 | 0.3 | 0.4 |

**(a)** Cross-entropy loss for each sample.
**(b)** Average cross-entropy loss.
**(c)** Perplexity.
**(d)** What would perfect predictions give?

**Solution:**

**(a)**

```
Sample 1: H(P₁,Q₁) = −log(0.7) = 0.357 nats = 0.515 bits
Sample 2: H(P₂,Q₂) = −log(0.8) = 0.223 nats = 0.322 bits
Sample 3: H(P₃,Q₃) = −log(0.4) = 0.916 nats = 1.322 bits
```

**(b)**

Average loss = (0.357 + 0.223 + 0.916)/3 = 1.496/3 = **0.499 nats = 0.720 bits**

**(c)**

Perplexity = exp(0.499) = **e^0.499 ≈ 1.647**

The model is as uncertain as uniformly choosing among ~1.65 classes.

**(d)** Perfect predictions (Q = P = one-hot):

```
Each loss = −log(1.0) = 0
Average loss = 0
Perplexity = e⁰ = 1
```

Perplexity = 1 means the model is perfectly certain. For K-class uniform: perplexity = K.

---

### 🔢 Numerical 4 — KL Divergence: Comparing Distributions

**Problem:** True data distribution P = (0.5, 0.3, 0.2) over 3 categories.

Two model distributions:
- Model Q₁ = (0.4, 0.4, 0.2)
- Model Q₂ = (0.5, 0.3, 0.2) (perfect)
- Model Q₃ = (0.8, 0.1, 0.1)

**(a)** KL(P||Q₁), KL(P||Q₂), KL(P||Q₃).
**(b)** Cross-entropy H(P,Q) for each.
**(c)** Is KL symmetric? Compute KL(Q₁||P).
**(d)** Which model is best?

**Solution:**

**(a)**

```
KL(P||Q₁) = 0.5·log(0.5/0.4) + 0.3·log(0.3/0.4) + 0.2·log(0.2/0.2)
           = 0.5·log(1.25) + 0.3·log(0.75) + 0.2·log(1.0)
           = 0.5×0.223 + 0.3×(−0.288) + 0.2×0
           = 0.1115 − 0.0864 + 0
           = 0.0251 nats

KL(P||Q₂) = 0  [Q₂ = P exactly] ✓

KL(P||Q₃) = 0.5·log(0.5/0.8) + 0.3·log(0.3/0.1) + 0.2·log(0.2/0.1)
           = 0.5×(−0.470) + 0.3×1.099 + 0.2×0.693
           = −0.235 + 0.330 + 0.139
           = 0.234 nats
```

**(b)** H(P) = −0.5ln(0.5)−0.3ln(0.3)−0.2ln(0.2) = 0.347+0.361+0.322 = **1.030 nats**

```
H(P,Q₁) = H(P) + KL(P||Q₁) = 1.030 + 0.0251 = 1.055 nats
H(P,Q₂) = 1.030 + 0 = 1.030 nats  [minimum possible]
H(P,Q₃) = 1.030 + 0.234 = 1.264 nats
```

**(c)**

```
KL(Q₁||P) = 0.4·log(0.4/0.5) + 0.4·log(0.4/0.3) + 0.2·log(0.2/0.2)
           = 0.4×(−0.223) + 0.4×0.288 + 0
           = −0.0892 + 0.1152 = 0.026 nats
```

KL(P||Q₁) = 0.0251 ≠ KL(Q₁||P) = 0.026 — **NOT symmetric!** ✓

**(d)** Q₂ is best (KL=0, cross-entropy minimum). Q₁ is close. Q₃ is worst — it overestimates category 0 and underestimates categories 1 and 2.

---

### 🔢 Numerical 5 — KL for Gaussians: VAE Regularization

**Problem:** A VAE encoder for input x outputs:
- μ(x) = 1.5, σ²(x) = 0.5 (encoder posterior q(z|x))
- Prior: p(z) = N(0, 1)

**(a)** Compute KL(q(z|x) || p(z)).
**(b)** Interpret this as regularization.
**(c)** What values of μ and σ² minimize KL?
**(d)** The total VAE loss is: L = reconstruction_loss + KL. If reconstruction loss = 2.3, total loss?

**Solution:**

**(a)** Using the formula: KL(N(μ,σ²) || N(0,1)) = (μ² + σ² − log σ² − 1)/2

```
KL = (1.5² + 0.5 − log(0.5) − 1)/2
   = (2.25 + 0.5 − (−0.693) − 1)/2
   = (2.25 + 0.5 + 0.693 − 1)/2
   = 2.443/2
   = 1.222 nats
```

**(b)** This KL term penalizes the encoder for producing a posterior far from the standard Normal prior. It:
- Pushes μ toward 0 (encoder mean should be near zero)
- Pushes σ² toward 1 (encoder variance should be near 1)
- Prevents posterior collapse (σ² → 0) and prior mismatch (μ → ∞)

The KL acts as a regularizer on the latent space, keeping encodings close to the standard Normal.

**(c)** KL = 0 iff q = p, i.e., **μ=0 and σ²=1**

When encoder outputs N(0,1), there's no KL penalty — the latent codes are already distributed as the prior.

**(d)** Total VAE loss = 2.3 + 1.222 = **3.522 nats**

---

### 🔢 Numerical 6 — Information Gain in Decision Trees

**Problem:** Dataset: 10 samples, 5 positive (Y=1), 5 negative (Y=0). Two features:

Feature A splits: {left: 4 pos, 1 neg}, {right: 1 pos, 4 neg}
Feature B splits: {left: 3 pos, 3 neg}, {right: 2 pos, 2 neg}

**(a)** H(Y) before split.
**(b)** H(Y|A) and IG(A).
**(c)** H(Y|B) and IG(B).
**(d)** Which feature splits better?

**Solution:**

**(a)** H(Y) = −0.5·log₂(0.5) − 0.5·log₂(0.5) = **1 bit**

**(b) Feature A:**

Left node: P(pos|left) = 4/5 = 0.8, P(neg|left) = 0.2
```
H(Y|A=left) = −0.8·log₂(0.8) − 0.2·log₂(0.2) = 0.722 bits
```

Right node: P(pos|right) = 1/5 = 0.2, P(neg|right) = 0.8
```
H(Y|A=right) = −0.2·log₂(0.2) − 0.8·log₂(0.8) = 0.722 bits
```

```
H(Y|A) = (5/10)×0.722 + (5/10)×0.722 = 0.722 bits

IG(A) = H(Y) − H(Y|A) = 1.000 − 0.722 = 0.278 bits
```

**(c) Feature B:**

Left: P(pos|left) = 3/6 = 0.5
```
H(Y|B=left) = 1 bit
```

Right: P(pos|right) = 2/4 = 0.5
```
H(Y|B=right) = 1 bit
```

```
H(Y|B) = (6/10)×1 + (4/10)×1 = 1 bit

IG(B) = 1.000 − 1.000 = 0 bits
```

**(d)** Feature A: IG = **0.278 bits** >> Feature B: IG = **0 bits**

Choose **Feature A**. Feature B is completely useless — both child nodes have the same 50/50 distribution as the parent. Feature A creates much purer nodes.

**ML insight:** This is exactly how ID3 and C4.5 decision trees select features at each node. High information gain = high mutual information between feature and label = feature is predictive.

---

### 🔢 Numerical 7 — Cross-Entropy and KL Unification

**Problem:** Language model perplexity analysis.

Model Q is evaluated on text where the true distribution P has:
- H(P) = 3.2 bits (true entropy of language, about 3-5 bits for English)
- Cross-entropy H(P,Q) = 5.1 bits (what model achieves)

**(a)** KL divergence KL(P||Q).
**(b)** Perplexity of the model.
**(c)** A better model achieves H(P,Q₂)=3.8 bits. Its KL and perplexity?
**(d)** What is the theoretical minimum cross-entropy achievable?
**(e)** Why can't a model achieve H(P,Q) < H(P)?

**Solution:**

**(a)** KL(P||Q) = H(P,Q) − H(P) = 5.1 − 3.2 = **1.9 bits**

**(b)** Perplexity = 2^{H(P,Q)} = 2^{5.1} = **34.3**

Model is as uncertain as uniformly choosing among ~34 words per token.

**(c)** Better model:
KL(P||Q₂) = 3.8 − 3.2 = **0.6 bits**
Perplexity = 2^{3.8} = **13.9**

Significant improvement in perplexity from reducing the KL gap.

**(d)** Minimum cross-entropy = H(P) = 3.2 bits. Achieved when Q = P exactly (KL=0).

No model can do better than the true entropy of the language — the data has inherent irreducible randomness.

**(e)** By the non-negativity of KL:
```
KL(P||Q) = H(P,Q) − H(P) ≥ 0
→ H(P,Q) ≥ H(P)
```

Cross-entropy is always ≥ true entropy. The gap is the KL divergence — the "extra cost" of using the wrong distribution.

**ML insight:** This is why language model improvement has diminishing returns: models approach H(P) asymptotically. The gap to H(P) measures how much room for improvement remains. GPT-4's gap to human-level entropy is estimated to be small but non-zero.

---

## 11. Information Theory ↔ ML Unification Table

```
Information Theory Concept    ML Equivalent
─────────────────────────────────────────────────────────────
Entropy H(P)                  Model uncertainty / label noise floor
Cross-entropy H(P,Q)          Classification loss function
KL divergence KL(P||Q)        Regularization (VAE), distribution matching
Mutual information I(X;Y)     Feature importance, information gain
Conditional entropy H(Y|X)    Remaining uncertainty after using feature X
Perplexity exp(H)             Language model quality metric
Max entropy principle         Gaussian prior, uniform prior
Information gain IG           Decision tree splitting criterion
Relative entropy              KL divergence (alternative name)
ELBO = E_q[log p(x,z)] − E_q[log q(z)]  = −H(q) − KL(q||p) + const

Minimizing cross-entropy      = MLE = maximizing log-likelihood
KL(q||p) ≥ 0                 = ELBO ≤ log p(x)  [Day 18, Jensen's]
H(X) ≤ log|X|               = Uniform has max entropy
```

---

## 12. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is entropy?" | H(X) = E[−log P(X)] — expected surprisal, uncertainty measure |
| "What distribution maximizes entropy?" | Uniform (finite support); Normal (fixed mean/variance) |
| "What is cross-entropy loss and why use it?" | H(P,Q) = −E_P[log Q] = MLE objective; penalizes wrong confidence |
| "What is KL divergence?" | KL(P\|\|Q) = H(P,Q)−H(P) ≥ 0; extra cost of using Q instead of P |
| "Is KL divergence symmetric?" | No — KL(P\|\|Q) ≠ KL(Q\|\|P) in general |
| "How does mutual information relate to feature selection?" | I(X;Y) = H(Y)−H(Y\|X) = reduction in uncertainty about Y given X |
| "What is perplexity?" | exp(cross-entropy); effective vocabulary size model is uncertain about |
| "Why is the VAE KL term needed?" | Keeps posterior close to prior; regularizes latent space |
| "What is the minimum cross-entropy achievable?" | H(P) — the true entropy of the data generating process |
| "How does entropy relate to information gain in decision trees?" | IG = H(Y)−H(Y\|X) = mutual information |

---

## 13. Key Formulas — Cheat Sheet for Day 28

```
Self-information:
    I(x) = −log P(x)     [bits if log₂, nats if ln]

Entropy:
    H(X) = −Σₓ P(x)log P(x) = E[−log P(X)]
    H(Bernoulli(p)) = −p log p − (1−p)log(1−p)
    H(Uniform(n)) = log n     [maximum]
    H(deterministic) = 0      [minimum]

Mutual Information:
    I(X;Y) = H(X)−H(X|Y) = H(Y)−H(Y|X) = H(X)+H(Y)−H(X,Y)
    I(X;Y) = 0 iff independent
    Information gain = I(Y;X)

Cross-Entropy:
    H(P,Q) = −Σₓ P(x)log Q(x) = E_P[−log Q(X)]
    H(P,Q) = H(P) + KL(P||Q) ≥ H(P)
    Classification loss: H(P,Q) = −log Q(true class)

KL Divergence:
    KL(P||Q) = Σₓ P(x)log[P(x)/Q(x)] = E_P[log P/Q]
    KL(P||Q) ≥ 0,  = 0 iff P = Q
    KL(P||Q) ≠ KL(Q||P)  [not symmetric]

KL between Gaussians:
    KL(N(μ,σ²)||N(0,1)) = (μ²+σ²−log σ²−1)/2

Perplexity:
    PP = exp(H(P,Q))  [nats]  or  2^{H(P,Q)}  [bits]

Decision tree information gain:
    IG(Y;X) = H(Y) − H(Y|X) = I(Y;X)

Chain rule:
    H(X,Y) = H(X) + H(Y|X)

Properties:
    H(X|Y) ≤ H(X)          [conditioning reduces entropy]
    H(X,Y) ≤ H(X)+H(Y)     [subadditivity]
```

---

## 14. Practice Problems (Solve Before Day 29)

1. Compute H(X) for X with PMF: P(1)=0.5, P(2)=0.25, P(3)=0.125, P(4)=0.125. Compare to log₂(4)=2 bits.

2. True distribution P=(0.6, 0.3, 0.1). Model outputs Q=(0.5, 0.4, 0.1). Compute KL(P||Q), KL(Q||P), and H(P,Q). Verify H(P,Q)=H(P)+KL(P||Q).

3. A 5-class problem has 100 training samples with class distribution (40,25,15,12,8). What is the entropy of this class distribution? What is the max entropy for 5 classes? How much room is there for reduction in uncertainty?

4. **Prove** that H(X) ≤ log|X| — uniform distribution has maximum entropy. (Hint: use KL(P||Uniform) ≥ 0.)

5. *(Interview-level)* A language model is trained on English text where the true entropy is estimated at H(P)=3.2 bits/token. The model achieves cross-entropy 4.0 bits/token on a test set. 
   - What is the KL divergence between model and true distribution?
   - What is the perplexity?
   - If the model's architecture is doubled in size and achieves 3.5 bits/token, what fraction of the KL gap was closed?
   - Why can't perplexity reach 1?

---

## 15. Looking Ahead

**Day 29** — **Markov Chains & Stationarity.** The mathematical model for sequential dependence — the foundation of HMMs, text generation, reinforcement learning, PageRank, and MCMC sampling. We derive the stationary distribution and mixing time, and connect Markov chains to modern sequence models.

---
*End of Day 28 | Next: Day 29 — Markov Chains & Stationarity*
