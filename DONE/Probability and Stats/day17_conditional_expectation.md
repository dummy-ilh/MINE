# Day 17 — Conditional Expectation & Law of Total Expectation
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 9
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Conditional Expectation is the Core of Prediction

Every prediction problem in ML is a conditional expectation problem.

| ML Task | Conditional Expectation |
|---|---|
| Regression | Ŷ = E[Y\|X] — the optimal predictor |
| Classification | P(Y=k\|X) = E[Iᵧ₌ₖ\|X] |
| Reinforcement Learning | V(s) = E[G\|S=s] — value function |
| Causal inference | E[Y\|do(X=x)] — interventional expectation |
| Bayesian updating | E[θ\|data] — posterior mean |
| Time series | E[Xₜ₊₁\|X₁,...,Xₜ] — forecast |
| Attention mechanism | E[V\|Q,K] — weighted value aggregation |
| Variational inference | E_q[log p(x,z)] — ELBO computation |

E[Y|X] is not just a tool — it IS what a model learns to approximate.

---

## 2. Conditional Expectation — Definition

### For Discrete RVs

> **Definition:** The conditional expectation of Y given X=x is:
> ```
> E[Y|X=x] = Σᵧ y · P(Y=y | X=x)
> ```

This is just the expected value of Y, but using the conditional distribution P(Y|X=x) instead of P(Y).

### For Continuous RVs

```
E[Y|X=x] = ∫₋∞^∞ y · f_{Y|X}(y|x) dy
```

### E[Y|X] as a Random Variable

**Critical distinction:**
- E[Y|X=x] is a **number** — the expected value of Y when X is fixed at x
- E[Y|X] is a **random variable** — a function of X

```
E[Y|X] = g(X)    where g(x) = E[Y|X=x]
```

E[Y|X] is the random variable obtained by replacing x in g(x) with the random variable X.

**Example:** If E[Y|X=x] = 3x+2, then E[Y|X] = 3X+2 — a random variable that takes value 3x+2 whenever X=x.

---

## 3. The Law of Total Expectation (Tower Property)

> **Theorem (Law of Total Expectation / Tower Property):**
> ```
> E[Y] = E[E[Y|X]]
> ```

**Reading:** The expectation of Y equals the expected value (over X) of the conditional expectation of Y given X.

### For discrete X with values x₁, x₂, ...:
```
E[Y] = Σᵢ E[Y|X=xᵢ] · P(X=xᵢ)
```

### For two conditioning variables (Tower Property):
```
E[Y] = E[E[Y|X,Z]] = E[E[E[Y|X,Z]|X]]
```

The tower property says: you can always "peel off" one layer of conditioning.

### Proof (discrete case)

```
E[E[Y|X]] = Σₓ E[Y|X=x] · P(X=x)
           = Σₓ [Σᵧ y · P(Y=y|X=x)] · P(X=x)
           = Σₓ Σᵧ y · P(Y=y|X=x) · P(X=x)
           = Σₓ Σᵧ y · P(Y=y, X=x)
           = Σᵧ y · Σₓ P(Y=y, X=x)
           = Σᵧ y · P(Y=y)
           = E[Y]  ∎
```

---

## 4. Conditional Variance

> **Definition:**
> ```
> Var(Y|X=x) = E[(Y−E[Y|X=x])² | X=x]
>            = E[Y²|X=x] − (E[Y|X=x])²
> ```

Var(Y|X) is a random variable — a function of X.

### Law of Total Variance (Eve's Law)

```
Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
```

**Term by term:**
- E[Var(Y|X)] — average within-group variance (unexplained variance)
- Var(E[Y|X]) — variance of group means (explained variance)

**This is the variance decomposition underlying ANOVA, R², and the bias-variance tradeoff.**

**Proof:**
```
Var(Y) = E[Y²] − (E[Y])²

E[Y²] = E[E[Y²|X]]                     [tower property]
      = E[Var(Y|X) + (E[Y|X])²]         [since Var=E[Y²]−(E[Y])²]
      = E[Var(Y|X)] + E[(E[Y|X])²]

(E[Y])² = (E[E[Y|X]])² = (E[Y|X] mean)²

Var(Y) = E[Var(Y|X)] + E[(E[Y|X])²] − (E[E[Y|X]])²
       = E[Var(Y|X)] + Var(E[Y|X])  ∎
```

---

## 5. E[Y|X] is the Best Predictor of Y

> **Theorem:** Among all functions g(X), E[Y|X] minimizes the mean squared prediction error:
> ```
> E[(Y − g(X))²]
> ```

**Proof:**
```
E[(Y−g(X))²] = E[(Y−E[Y|X]+E[Y|X]−g(X))²]

Let a = Y−E[Y|X],  b = E[Y|X]−g(X)

= E[a²] + 2E[ab] + E[b²]

E[ab] = E[(Y−E[Y|X])(E[Y|X]−g(X))]
      = E[E[(Y−E[Y|X])(E[Y|X]−g(X)) | X]]
      = E[(E[Y|X]−g(X))·E[Y−E[Y|X]|X]]    [second factor]
      = E[(E[Y|X]−g(X))·0] = 0

So E[(Y−g(X))²] = E[(Y−E[Y|X])²] + E[(E[Y|X]−g(X))²]
                  ≥ E[(Y−E[Y|X])²]
```

Equality when g(X) = E[Y|X]. **E[Y|X] is the optimal predictor.** ∎

**This is the mathematical justification for regression:** linear regression finds the best linear approximation to E[Y|X]. Neural networks learn a more flexible approximation to E[Y|X].

---

## 6. Connection to R² and Explained Variance

From Eve's Law:
```
Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
 Total  =  Unexplained  +   Explained
```

**R² (coefficient of determination):**
```
R² = Var(E[Y|X]) / Var(Y) = Explained / Total = 1 − Unexplained/Total
```

R² measures the fraction of Y's variance explained by X through the conditional mean.

- R²=1: E[Y|X] perfectly predicts Y (no residual variance)
- R²=0: E[Y|X]=E[Y] (X explains nothing)

---

## 7. Worked Numericals

---

### 🔢 Numerical 1 — Computing E[Y|X] from Joint Distribution

**Problem:** X and Y have joint PMF:

|  | Y=0 | Y=1 | Y=2 |
|---|---|---|---|
| **X=0** | 0.20 | 0.10 | 0.10 |
| **X=1** | 0.05 | 0.30 | 0.25 |

**(a)** Find E[Y|X=0] and E[Y|X=1]
**(b)** Find E[Y] using the Law of Total Expectation
**(c)** Find Var(Y|X=0), Var(Y|X=1)
**(d)** Verify Eve's Law: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])

**Solution:**

**Marginals:**
P(X=0) = 0.20+0.10+0.10 = 0.40
P(X=1) = 0.05+0.30+0.25 = 0.60

**Conditional PMFs:**

P(Y=y|X=0): P(0|0)=0.20/0.40=0.5, P(1|0)=0.25, P(2|0)=0.25
P(Y=y|X=1): P(0|1)=0.05/0.60=1/12, P(1|1)=0.30/0.60=0.5, P(2|1)=0.25/0.60=5/12

**(a)**
```
E[Y|X=0] = 0×0.5 + 1×0.25 + 2×0.25 = 0 + 0.25 + 0.50 = 0.75

E[Y|X=1] = 0×(1/12) + 1×0.5 + 2×(5/12)
          = 0 + 0.5 + 10/12 = 0.5 + 0.8333 = 1.3333
```

**(b) Law of Total Expectation:**
```
E[Y] = E[Y|X=0]·P(X=0) + E[Y|X=1]·P(X=1)
     = 0.75×0.40 + 1.3333×0.60
     = 0.30 + 0.80 = 1.10
```

**Verify directly:** E[Y] = 0×0.25+1×0.40+2×0.35 = 0+0.40+0.70 = 1.10 ✓

**(c)**
```
E[Y²|X=0] = 0²×0.5 + 1²×0.25 + 2²×0.25 = 0+0.25+1.0 = 1.25
Var(Y|X=0) = 1.25 − 0.75² = 1.25 − 0.5625 = 0.6875

E[Y²|X=1] = 0²×(1/12) + 1²×0.5 + 2²×(5/12) = 0+0.5+20/12 = 0.5+1.6667 = 2.1667
Var(Y|X=1) = 2.1667 − 1.3333² = 2.1667 − 1.7778 = 0.3889
```

**(d) Eve's Law verification:**

E[Var(Y|X)] = 0.6875×0.40 + 0.3889×0.60 = 0.2750 + 0.2333 = 0.5083

Var(E[Y|X]) = E[(E[Y|X])²] − (E[E[Y|X]])²
E[(E[Y|X])²] = 0.75²×0.40 + 1.3333²×0.60 = 0.225 + 1.0667 = 1.2917
Var(E[Y|X]) = 1.2917 − 1.10² = 1.2917 − 1.21 = 0.0817

E[Var(Y|X)] + Var(E[Y|X]) = 0.5083 + 0.0817 = **0.5900**

Direct Var(Y): E[Y²] = 0²×0.25+1²×0.40+2²×0.35 = 0+0.40+1.40 = 1.80
Var(Y) = 1.80 − 1.10² = 1.80 − 1.21 = **0.59** ✓

Eve's Law verified. The explained variance (0.0817) is small relative to unexplained (0.5083) — X doesn't explain much of Y's variance (low R²).

---

### 🔢 Numerical 2 — Law of Total Expectation: Model Evaluation

**Problem:** A model is evaluated on two data subsets:
- Easy examples: 60% of data, E[loss | easy] = 0.3
- Hard examples: 40% of data, E[loss | hard] = 0.9

**(a)** Overall expected loss.
**(b)** A new model has E[loss|easy]=0.25, E[loss|hard]=0.75. Which model is better overall?
**(c)** The data distribution shifts — now 70% hard, 30% easy. Which model is better now?

**Solution:**

**(a)**
```
E[loss] = E[loss|easy]·P(easy) + E[loss|hard]·P(hard)
        = 0.3×0.60 + 0.9×0.40 = 0.18 + 0.36 = 0.54
```

**(b)** New model overall loss:
```
E[loss_new] = 0.25×0.60 + 0.75×0.40 = 0.15 + 0.30 = 0.45
```

New model is better: 0.45 < 0.54 ✓

**(c)** After distribution shift (70% hard):

Old model: 0.3×0.30 + 0.9×0.70 = 0.09 + 0.63 = **0.72**
New model: 0.25×0.30 + 0.75×0.70 = 0.075 + 0.525 = **0.60**

New model still better. The gap actually widens under distribution shift because the new model is proportionally better on hard examples.

**ML insight:** This is the Law of Total Expectation applied to subgroup analysis. Benchmark accuracy (overall expected loss) is a weighted average of subgroup performances. **Dataset shift** changes the weights — a model that's better on-average in one distribution may not be in another. Always evaluate subgroup performance, not just overall.

---

### 🔢 Numerical 3 — Conditional Expectation as Predictor: Regression

**Problem:** X ~ Uniform(0,2) and Y|X=x ~ Normal(3x−1, 1).

**(a)** What is E[Y|X=x]? Is this the best predictor?
**(b)** Find E[Y] using Law of Total Expectation.
**(c)** Find Var(Y) using Eve's Law.
**(d)** What is R²?

**Solution:**

**(a)** E[Y|X=x] = 3x−1. Yes — E[Y|X] is always the MSE-optimal predictor. The regression function is **linear in x**.

**(b)**
```
E[Y] = E[E[Y|X]] = E[3X−1]
     = 3E[X] − 1 = 3×(0+2)/2 − 1 = 3×1 − 1 = 2
```

**(c) Eve's Law:**

Var(Y|X=x) = 1 (given — the Normal conditional variance)
E[Var(Y|X)] = 1 (constant)

E[Y|X] = 3X−1
Var(E[Y|X]) = Var(3X−1) = 9·Var(X) = 9×(2−0)²/12 = 9×4/12 = 3

Var(Y) = E[Var(Y|X)] + Var(E[Y|X]) = 1 + 3 = **4**

**(d)**
```
R² = Var(E[Y|X])/Var(Y) = 3/4 = 0.75
```

X explains 75% of Y's variance. The remaining 25% is irreducible noise (the Normal error term with variance 1).

**ML insight:** This is the population R². In linear regression:
- Var(E[Y|X]) = explained variance (from the linear fit)
- E[Var(Y|X)] = σ² = residual variance (irreducible noise)
- R² = 1 − σ²/Var(Y) — the familiar formula

---

### 🔢 Numerical 4 — Tower Property: Value Function in RL

**Problem:** An agent follows a 2-step policy. State S₁ → action A₁ → State S₂ → action A₂ → Reward R.

- S₁ = 0 or 1, P(S₁=1) = 0.6
- If S₁=0: E[R|S₁=0] = 5
- If S₁=1: E[R|S₁=1, A₁=good] = 12, P(good|S₁=1) = 0.7
            E[R|S₁=1, A₁=bad]  = 3,  P(bad|S₁=1)  = 0.3

**(a)** E[R|S₁=1] using tower property over A₁.
**(b)** E[R] (overall expected reward) using tower property over S₁.
**(c)** Value function V(S₁) for each state.

**Solution:**

**(a)** Tower property: E[R|S₁=1] = E[E[R|S₁=1, A₁] | S₁=1]
```
E[R|S₁=1] = E[R|S₁=1,A₁=good]·P(good|S₁=1) + E[R|S₁=1,A₁=bad]·P(bad|S₁=1)
           = 12×0.7 + 3×0.3
           = 8.4 + 0.9 = 9.3
```

**(b)**
```
E[R] = E[R|S₁=0]·P(S₁=0) + E[R|S₁=1]·P(S₁=1)
     = 5×0.4 + 9.3×0.6
     = 2.0 + 5.58 = 7.58
```

**(c)** Value function V(s) = E[R|S₁=s]:
```
V(0) = 5      [low-value state]
V(1) = 9.3    [high-value state]
```

**ML insight:** The Bellman equation in RL is precisely the tower property:
```
V(s) = E[R + γV(S') | S=s] = E[R|S=s] + γ·E[V(S')|S=s]
```

The value function is a conditional expectation. Dynamic programming (Q-learning, policy gradient) is computing/approximating conditional expectations layer by layer — exactly the tower property applied recursively.

---

### 🔢 Numerical 5 — Eve's Law: Variance Decomposition in ML

**Problem:** A model's prediction error Y depends on:
- Data difficulty D ∈ {easy, hard}, P(easy)=0.7, P(hard)=0.3
- Given difficulty, error has:
  - Easy: E[Y|easy]=0.2, Var(Y|easy)=0.04
  - Hard: E[Y|hard]=0.8, Var(Y|hard)=0.16

**(a)** E[Y] — overall expected error
**(b)** Var(Y) using Eve's Law
**(c)** How much variance is due to difficulty (explained) vs. within-class noise (unexplained)?
**(d)** Compute R² for difficulty as a predictor of error.

**Solution:**

**(a)**
```
E[Y] = 0.2×0.7 + 0.8×0.3 = 0.14 + 0.24 = 0.38
```

**(b) Eve's Law:**

E[Var(Y|D)] = 0.04×0.7 + 0.16×0.3 = 0.028 + 0.048 = **0.076** [unexplained]

E[Y|D] takes values: 0.2 (prob 0.7), 0.8 (prob 0.3)
Var(E[Y|D]) = E[(E[Y|D])²] − (E[E[Y|D]])²
= (0.04×0.7 + 0.64×0.3) − 0.38²
= (0.028 + 0.192) − 0.1444
= 0.220 − 0.1444 = **0.0756** [explained]

Var(Y) = 0.076 + 0.0756 = **0.1516**

**(c)**
- Unexplained (within-difficulty noise): 0.076 / 0.1516 = **50.1%**
- Explained (between-difficulty variation): 0.0756 / 0.1516 = **49.9%**

Almost exactly half the variance is explained by difficulty.

**(d)**
```
R² = Var(E[Y|D]) / Var(Y) = 0.0756 / 0.1516 = 0.499 ≈ 50%
```

**ML insight:** This is exactly how ANOVA works — decomposing total variance into between-group and within-group components. In ML, this guides whether to stratify datasets, whether difficulty-aware training helps, and how much performance gain is theoretically possible from better difficulty estimation.

---

### 🔢 Numerical 6 — Conditional Expectation: Bayesian Posterior Mean

**Problem:** You observe n=10 coin flips, k=7 heads. Prior: p ~ Beta(2, 2) (your prior belief about the coin's bias).

The posterior is p|data ~ Beta(2+7, 2+3) = Beta(9, 5).

**(a)** Posterior mean E[p|data] — the Bayes estimate.
**(b)** Compare to MLE: p̂_MLE = k/n.
**(c)** As n→∞, what does E[p|data] approach?
**(d)** Interpret E[p|data] as E[E[p|data]] using the tower property.

**Solution:**

**(a)** Beta(α,β) has mean α/(α+β):
```
E[p|data] = 9/(9+5) = 9/14 ≈ 0.643
```

**(b)**
```
p̂_MLE = 7/10 = 0.700
```

Posterior mean (0.643) is shrunk toward the prior mean (0.5) compared to MLE (0.700). The prior pulls the estimate toward 0.5.

**(c)** General formula for Beta(α+k, β+n−k) posterior:
```
E[p|data] = (α+k)/(α+β+n) = k/n · n/(α+β+n) + (α/(α+β)) · (α+β)/(α+β+n)
```

As n→∞: E[p|data] → k/n = MLE. Data dominates prior.

**(d)** Tower property:
```
E[p] = E[E[p|data]]
```

Before seeing data, your prior mean E[p] = 2/(2+2) = 0.5.
After seeing data, E[p|data] = 0.643.
The tower says: averaging E[p|data] over all possible datasets gives back the prior mean 0.5.

**ML insight:** MAP estimation = posterior mode. Posterior mean = Bayes estimate (minimizes expected squared error). The tower property ensures Bayesian estimates are coherent across levels of information.

---

### 🔢 Numerical 7 — E[Y|X] as Neural Network Target

**Problem:** A neural network learns to predict house price Y from features X = (size, location).

From training data, you estimate:
- E[Y|size=1000, location=A] = $350,000
- E[Y|size=1500, location=A] = $420,000
- E[Y|size=1000, location=B] = $280,000
- E[Y|size=1500, location=B] = $340,000

**(a)** Marginal distribution: P(size=1000)=0.4, P(size=1500)=0.6, P(A)=0.5, P(B)=0.5.
Find E[Y] using the Law of Total Expectation.

**(b)** If a new house has size=1200 (not in training), how should the model extrapolate?

**(c)** Training loss MSE = E[(Y−f(X))²]. Show this is minimized when f(X) = E[Y|X].

**Solution:**

**(a)**
```
E[Y] = E[Y|1000,A]·P(1000)·P(A) + E[Y|1500,A]·P(1500)·P(A)
     + E[Y|1000,B]·P(1000)·P(B) + E[Y|1500,B]·P(1500)·P(B)

     = 350k×0.4×0.5 + 420k×0.6×0.5 + 280k×0.4×0.5 + 340k×0.6×0.5
     = 70k + 126k + 56k + 102k
     = $354,000
```

**(b)** Linear interpolation:
```
E[Y|size=1200,A] ≈ 350k + (1200−1000)/(1500−1000) × (420k−350k)
                 = 350k + 0.4×70k = 350k + 28k = $378,000

E[Y|size=1200,B] ≈ 280k + 0.4×60k = 280k + 24k = $304,000
```

This is exactly what a linear regression or neural network does — interpolates E[Y|X] between training points.

**(c)** From Section 5:
```
E[(Y−f(X))²] = E[(Y−E[Y|X])²] + E[(E[Y|X]−f(X))²]
               ≥ E[(Y−E[Y|X])²]
```

Minimized exactly when f(X) = E[Y|X]. **The network's training objective (MSE) drives it to learn E[Y|X].** ∎

This is not just "nice to know" — it's the fundamental theorem justifying all regression neural networks.

---

## 8. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is E[Y\|X] and why is it important?" | Optimal MSE predictor — what every regression model approximates |
| "State the Law of Total Expectation" | E[Y] = E[E[Y\|X]] — average conditional means gives marginal mean |
| "What is Eve's Law?" | Var(Y) = E[Var(Y\|X)] + Var(E[Y\|X]) — variance decomposition |
| "How does R² relate to conditional expectation?" | R² = Var(E[Y\|X])/Var(Y) — fraction of variance explained by model |
| "Why does MSE regression estimate E[Y\|X]?" | E[Y\|X] uniquely minimizes E[(Y−f(X))²] over all f |
| "What is the Bellman equation in terms of conditional expectation?" | V(s) = E[R+γV(S')\|S=s] — tower property applied recursively |
| "How does the posterior mean relate to conditional expectation?" | E[θ\|data] is the Bayes estimate — minimizes expected squared error |
| "What does the tower property say about nested conditioning?" | E[Y] = E[E[Y\|X]] — outer expectation averages over the conditioning variable |

---

## 9. Key Formulas — Cheat Sheet for Day 17

```
Conditional Expectation:
    E[Y|X=x] = Σᵧ y·P(Y=y|X=x)    [discrete]
    E[Y|X=x] = ∫ y·f_{Y|X}(y|x) dy  [continuous]
    E[Y|X] = g(X)  where g(x) = E[Y|X=x]  [random variable]

Law of Total Expectation:
    E[Y] = E[E[Y|X]] = Σₓ E[Y|X=x]·P(X=x)

Tower Property:
    E[Y] = E[E[Y|X,Z]] = E[E[E[Y|X,Z]|X]]

Law of Total Variance (Eve's Law):
    Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
           =  Unexplained  +   Explained

R²:
    R² = Var(E[Y|X]) / Var(Y)
       = 1 − E[Var(Y|X)] / Var(Y)

Optimal predictor:
    f*(X) = E[Y|X]  minimizes  E[(Y−f(X))²]

Conditional Variance:
    Var(Y|X=x) = E[Y²|X=x] − (E[Y|X=x])²

Bellman Equation (RL):
    V(s) = E[R + γV(S') | S=s]    [tower property]

Posterior Mean (Bayes):
    E[θ|data] = argmin_c E[(θ−c)²|data]   [Bayes estimate]
    E[θ] = E[E[θ|data]]                    [tower property]
```

---

## 10. Practice Problems (Solve Before Day 18)

1. X ~ Poisson(λ). Conditioned on X=n, Y|X=n ~ Binomial(n, p). Find E[Y] and Var(Y) using the tower property and Eve's Law. *(Answer: E[Y]=λp, Var(Y)=λp(1−p)+λp²=λp)*

2. A retrieval system returns a document. P(relevant)=0.3. If relevant, user rating Y|relevant ~ Uniform(7,10). If not relevant, Y|not ~ Uniform(1,4). Find E[Y] and Var(Y).

3. **Prove** Eve's Law: Var(Y) = E[Var(Y|X)] + Var(E[Y|X]) from first principles using the definition of variance and tower property.

4. In a neural network, the output layer computes f(X) ≈ E[Y|X]. The residuals R = Y − f(X) should satisfy:
   - E[R] = 0
   - E[R|X] = 0 (no systematic error)
   - Var(R) = E[Var(Y|X)] (irreducible noise)
   
   Prove each property assuming f(X) = E[Y|X] exactly.

5. *(Interview-level)* An RL agent learns Q(s,a) = E[G|S=s, A=a] (action-value function). Using the tower property over future states S', show that:
   Q(s,a) = E[R|S=s,A=a] + γ·E[max_{a'} Q(S',a') | S=s, A=a]
   
   This is the **Bellman optimality equation**. *(Hint: G = R + γG', apply tower property to G'.)*

---

## 11. Looking Ahead

**Day 18** — **Inequalities: Markov, Chebyshev & Jensen's.** These inequalities let you bound probabilities and expectations **without knowing the full distribution** — critical for generalization theory, understanding why deep learning works, and deriving PAC learning bounds.

---
*End of Day 17 | Next: Day 18 — Markov, Chebyshev & Jensen's Inequalities*
