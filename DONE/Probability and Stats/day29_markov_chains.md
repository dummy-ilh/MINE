# Day 29 — Markov Chains & Stationarity
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 11; Norris, *Markov Chains*
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Markov Chains Matter in ML

| ML Application | Markov Chain Role |
|---|---|
| **Language models (n-gram)** | Text as Markov chain over words |
| **Hidden Markov Models (HMM)** | Hidden state sequence is Markov |
| **Reinforcement Learning** | MDP = Markov Decision Process |
| **MCMC** | Sample from posterior distributions |
| **PageRank** | Web graph as Markov chain |
| **Diffusion models** | Forward noising process is Markov |
| **Queue modeling** | M/M/1 = Markov chain on queue length |

---

## 2. The Markov Property

> **Definition:** {Xₙ} is a **Markov chain** if:
> ```
> P(Xₙ₊₁=j | Xₙ=i, Xₙ₋₁,...,X₀) = P(Xₙ₊₁=j | Xₙ=i) = Pᵢⱼ
> ```

**"Given the present, the past is irrelevant."**

For **homogeneous** chains, Pᵢⱼ doesn't depend on time n.

---

## 3. Transition Matrix

Pᵢⱼ = P(Xₙ₊₁=j | Xₙ=i)

Properties:
```
Pᵢⱼ ≥ 0,    Σⱼ Pᵢⱼ = 1    [rows sum to 1 — stochastic matrix]
```

**n-Step probabilities:**
```
P(Xₙ=j | X₀=i) = [Pⁿ]ᵢⱼ    [entry of matrix power]
```

**Distribution after n steps:**
```
π₀ Pⁿ    [row vector × matrix]
```

---

## 4. State Classification

**Reachability:** j reachable from i (i→j) if ∃n: [Pⁿ]ᵢⱼ > 0

**Communication:** i↔j if i→j AND j→i (equivalence relation → communicating classes)

**Irreducible:** All states communicate (one class)

**Period:** d(i) = gcd{n≥1: [Pⁿ]ᵢᵢ>0}. Aperiodic if d(i)=1 for all i.

**Recurrent:** P(return to i | start at i) = 1. All states in finite irreducible chain are recurrent.

---

## 5. Stationary Distribution

> **Definition:** π is stationary if:
> ```
> πP = π,   Σⱼ πⱼ = 1,   πⱼ ≥ 0
> ```

π is the left eigenvector of P with eigenvalue 1.

**Convergence Theorem:** For irreducible, aperiodic chain:
```
lim_{n→∞} [Pⁿ]ᵢⱼ = πⱼ    for all i,j
```

No matter where you start, you converge to π. The long-run fraction of time in state j = πⱼ.

**Finding π:** Solve πP=π simultaneously with Σπⱼ=1.

---

## 6. Detailed Balance (Reversibility)

> **Definition:** Chain satisfies **detailed balance** with π if:
> ```
> πᵢ Pᵢⱼ = πⱼ Pⱼᵢ    for all i,j
> ```

**Theorem:** Detailed balance → π is stationary.

**Proof:** Σᵢ πᵢ Pᵢⱼ = Σᵢ πⱼ Pⱼᵢ = πⱼ Σᵢ Pⱼᵢ = πⱼ ∎

**MCMC use:** Design transitions satisfying detailed balance with target π → chain samples from π.

---

## 7. Mixing Time

Mixing time ≈ steps until chain is close to π.

```
t_mix ≈ 1/(1−λ₂)
```

where λ₂ = second largest eigenvalue of P.

- Large spectral gap (1−λ₂) → fast mixing
- λ₂ near 1 → slow mixing → long MCMC burn-in needed

---

## 8. Metropolis-Hastings MCMC

To sample from target π:

1. At state i, propose j from proposal q(j|i)
2. Accept with probability: A(i→j) = min(1, π(j)q(i|j) / [π(i)q(j|i)])
3. If accepted: move to j. Else: stay at i.

**Symmetric proposal:** A(i→j) = min(1, π(j)/π(i))

**Detailed balance holds** → stationary distribution = π ✓

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — Weather Markov Chain

**Problem:** Sunny (S) / Rainy (R) chain:
```
P = [0.8  0.2]  ← from S
    [0.3  0.7]  ← from R
```
Find: (a) P²; (b) stationary distribution; (c) rate of convergence.

**Solution:**

**(a)**
```
P² = [0.8×0.8+0.2×0.3  0.8×0.2+0.2×0.7] = [0.70  0.30]
     [0.3×0.8+0.7×0.3  0.3×0.2+0.7×0.7]   [0.45  0.55]
```

**(b)** πP = π:
```
πS = 0.8πS + 0.3πR  →  0.2πS = 0.3πR  →  πS = 1.5πR
πS + πR = 1  →  2.5πR = 1  →  πR = 0.40, πS = 0.60
```

Long run: **60% Sunny, 40% Rainy** — regardless of starting state.

Verify: (0.6, 0.4) × P = (0.6×0.8+0.4×0.3, 0.6×0.2+0.4×0.7) = (0.60, 0.40) ✓

**(c)** Eigenvalues of P: λ₁=1, λ₂=0.5 (from trace=1.5, det=0.56-0.06=0.50)

Convergence rate = 0.5ⁿ — halves each step. After 10 steps: 0.5¹⁰ ≈ 0.001 — essentially converged.

---

### 🔢 Numerical 2 — Absorbing Chain: Customer Churn

**Problem:** User states: Active (A), Passive (P), Churned (C — absorbing).
```
P = [0.7  0.2  0.1]  ← A
    [0.3  0.5  0.2]  ← P
    [0.0  0.0  1.0]  ← C (absorbing)
```
Find E[steps until churn | start Active].

**Solution:**

Let tA = E[steps until C | start A], tP = E[steps until C | start P].

```
tA = 1 + 0.7tA + 0.2tP    →  0.3tA − 0.2tP = 1   ...(1)
tP = 1 + 0.3tA + 0.5tP    →  −0.3tA + 0.5tP = 1  ...(2)
```

(1)+(2): 0.3tP = 2 → tP = 6.67
From (1): tA = (1 + 0.2×6.67)/0.3 = 2.333/0.3 = **7.78 steps**

Starting Active → expected **7.78 steps until churn**.

**ML insight:** Expected customer lifetime = expected absorption time. Reducing churn transition rates increases this — the Markov chain framing makes the lever clear.

---

### 🔢 Numerical 3 — Detailed Balance: Custom Chain

**Problem:** Build a chain on {1,2,3} with stationary distribution π=(0.2, 0.5, 0.3). Given P₁₂=0.4, P₁₃=0.3, find P₂₁ and P₃₁ via detailed balance.

**Solution:**

```
π₁P₁₂ = π₂P₂₁  →  0.2×0.4 = 0.5×P₂₁  →  P₂₁ = 0.16
π₁P₁₃ = π₃P₃₁  →  0.2×0.3 = 0.3×P₃₁  →  P₃₁ = 0.20
```

P₁₁ = 1−0.4−0.3 = 0.30

Choose P₂₃=0.20: then π₂P₂₃ = π₃P₃₂ → 0.5×0.20 = 0.3×P₃₂ → P₃₂ = 0.333

Full matrix:
```
P = [0.30  0.40  0.30]
    [0.16  0.64  0.20]
    [0.20  0.333 0.467]
```

Verify: πP = (0.2,0.5,0.3) × P = (0.2, 0.5, 0.3) ✓

**ML insight:** This is how Metropolis-Hastings is constructed. You design transitions via detailed balance to guarantee the target π is stationary.

---

### 🔢 Numerical 4 — PageRank

**Problem:** 4-page web: Page 1→{2,3}, Page 2→{1,4}, Page 3→{2}, Page 4→{1,2,3}.

Build transition matrix and find PageRank (stationary distribution).

**Solution:**

```
     1      2      3      4
P = [0      0.5    0.5    0   ]  ← 1
    [0.5    0      0      0.5 ]  ← 2
    [0      1.0    0      0   ]  ← 3
    [1/3    1/3    1/3    0   ]  ← 4
```

Solve πP = π:

From column 4: π₄ = 0.5π₂
From column 1: π₁ = 0.5π₂ + (1/3)π₄ = 0.5π₂ + π₂/6 = 2π₂/3
From column 3: π₃ = 0.5π₁ + (1/3)π₄ = π₂/3 + π₂/6 = π₂/2

Normalization: 2π₂/3 + π₂ + π₂/2 + π₂/2 = (8/6+6/6+3/6+3/6)π₂ = (20/6)π₂ = 1... 

Let me recount: 2/3 + 1 + 1/2 + 1/2 = 4/6+6/6+3/6+3/6 = 16/6 → π₂ = 6/16 = 0.375

| Page | Score |
|---|---|
| 1 | 0.250 |
| 2 | **0.375** — highest |
| 3 | 0.188 |
| 4 | 0.188 |

**Page 2 has highest PageRank** — it receives links from pages 1 and 3 (page 3 exclusively links to page 2).

---

### 🔢 Numerical 5 — Metropolis-Hastings Step

**Problem:** Target π=(0.1, 0.3, 0.4, 0.2). Symmetric random walk proposal. Current state: 2. Proposed: 3.

**(a)** Acceptance probability 2→3.
**(b)** Acceptance probability 2→1.
**(c)** Verify detailed balance.

**Solution:**

**(a)** A(2→3) = min(1, π(3)/π(2)) = min(1, 0.4/0.3) = min(1, 1.333) = **1.0** (always accept — moving to higher prob)

**(b)** A(2→1) = min(1, π(1)/π(2)) = min(1, 0.1/0.3) = **0.333** (accept 1/3 of the time)

**(c)** Detailed balance for 2→3 (proposal q(j|i)=0.5 for neighbors):
```
Left:  π(2)×q(2→3)×A(2→3) = 0.3×0.5×1.0 = 0.150
Right: π(3)×q(3→2)×A(3→2) = 0.4×0.5×min(1,0.3/0.4) = 0.4×0.5×0.75 = 0.150 ✓
```

Detailed balance holds → π is the stationary distribution ✓

---

### 🔢 Numerical 6 — Bigram Language Model

**Problem:** Corpus: "the cat sat on the mat the cat ate"

**(a)** Estimate transition matrix; **(b)** find stationary distribution; **(c)** limitations.

**Solution:**

**(a)** Bigram counts: the→cat(2), the→mat(1), cat→sat(1), cat→ate(1), sat→on(1), on→the(1), mat→the(1)

Transition probabilities:
```
P_the  = {cat: 2/3, mat: 1/3}
P_cat  = {sat: 1/2, ate: 1/2}
P_sat  = {on: 1}
P_on   = {the: 1}
P_mat  = {the: 1}
P_ate  = {} (absorbing — end of corpus)
```

**(b)** For the cyclic subchain {the, cat, sat, on, mat} (excluding "ate"):

Flow balance:
- Flow into "the": from on(×1) and mat(×1) → π_the proportional to in-flow
- The→cat with prob 2/3, the→mat with 1/3

Stationary (proportional to visit frequency in corpus):
π ∝ (3, 2, 1, 1, 1) for (the, cat, sat, on, mat) → normalize: (3/8, 2/8, 1/8, 1/8, 1/8)

"the" appears most — consistent with corpus.

**(c)** Limitations:
- Only 1-step memory: "the cat sat" and "the mat sat" look identical after "sat"
- Cannot capture long-range dependencies ("the cat [20 words later] ... it")
- Sparse: many valid bigrams unseen in training
- Modern solution: transformers with attention capture the FULL history, overcoming the Markov limitation

---

### 🔢 Numerical 7 — Convergence Rate from Eigenvalues

**Problem:** Chain on {1,2,3}:
```
P = [0.1  0.8  0.1]
    [0.3  0.4  0.3]
    [0.1  0.8  0.1]
```

**(a)** Stationary distribution; **(b)** second eigenvalue; **(c)** convergence to π.

**Solution:**

**(a)** Rows 1 and 3 are identical → π₁=π₃ by symmetry.

π₂ = 0.8π₁ + 0.4π₂ + 0.8π₃ = 1.6π₁ + 0.4π₂ → 0.6π₂ = 1.6π₁ → π₂ = 8π₁/3

Normalization: π₁ + 8π₁/3 + π₁ = (14/3)π₁ = 1 → π₁ = 3/14

**π = (3/14, 8/14, 3/14) ≈ (0.214, 0.571, 0.214)**

**(b)** Eigenvalues: λ₁=1 always. Trace = 0.1+0.4+0.1 = 0.6 = λ₁+λ₂+λ₃.

By the structure (rows 1&3 identical), λ₂ = 0, λ₃ = −0.4.

Second largest in magnitude: **|λ₂| = 0.4**

**(c)** Convergence: |P^n − 1π| ∝ 0.4ⁿ

After n=5: 0.4⁵ = 0.010 — within 1% of stationary in just 5 steps!

**Spectral gap = 1 − 0.4 = 0.6** — large gap → very fast mixing.

**ML insight:** For MCMC: large spectral gap → short burn-in → computationally cheap. Designing better proposal distributions in MCMC is equivalent to increasing the spectral gap of the Markov chain.

---

## 10. Markov Chains in Modern Deep Learning

```
ML Concept              Markov Chain Connection
──────────────────────────────────────────────────────────────
GPT/language models     Chain rule P(w₁,...,wₙ) = ΠP(wₜ|w₁,...,wₜ₋₁)
                        Transformer relaxes Markov by attending to ALL past

Diffusion models        Forward: x₀→x₁→...→xT (Gaussian noise, Markov)
                        Reverse: learned denoising (non-Markov in general)

RL/MDPs                 Sₜ₊₁ depends only on (Sₜ, Aₜ) — Markov
                        Bellman equation = stationarity condition

MCMC posterior sampling Detailed balance → target π = posterior P(θ|data)

RNNs/LSTMs             Hidden state hₜ ≈ sufficient statistic for history
                        Approximately Markov in hidden state space

Energy-based models     Contrastive divergence = short MCMC chain
```

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is the Markov property?" | Future depends only on present, not past |
| "What is a stationary distribution?" | π such that πP=π — equilibrium |
| "How do you find stationary distribution?" | Solve πP=π with Σπⱼ=1 |
| "What is detailed balance?" | πᵢPᵢⱼ=πⱼPⱼᵢ — implies stationarity, used in MCMC |
| "What is MCMC?" | Build chain with target π via detailed balance; run for samples |
| "How does PageRank use Markov chains?" | Web as chain; PR scores = stationary distribution |
| "What is mixing time?" | Steps until close to stationary; controlled by spectral gap |
| "Why is Markov property useful in RL?" | Makes Bellman equation valid; DP tractable |
| "Limitation of bigram language models?" | Only 1-step memory; no long-range dependencies |
| "What is an absorbing state?" | State with Pᵢᵢ=1 — once entered, never left |

---

## 12. Key Formulas — Cheat Sheet for Day 29

```
Markov Property:
    P(Xₙ₊₁=j|Xₙ=i,...,X₀) = Pᵢⱼ

Transition Matrix:
    Pᵢⱼ ≥ 0,  Σⱼ Pᵢⱼ = 1

n-Step:
    P(Xₙ=j|X₀=i) = [Pⁿ]ᵢⱼ

Stationary Distribution:
    πP = π,  Σπⱼ=1

Detailed Balance → Stationarity:
    πᵢPᵢⱼ = πⱼPⱼᵢ  for all i,j

Convergence:
    [Pⁿ]ᵢⱼ → πⱼ  (rate: |λ₂|ⁿ)
    Mixing time ≈ 1/(1−|λ₂|)

Metropolis-Hastings:
    A(i→j) = min(1, π(j)q(i|j)/[π(i)q(j|i)])
    Symmetric: A(i→j) = min(1, π(j)/π(i))

Absorption Time:
    tᵢ = 1 + Σⱼ∉absorbed Pᵢⱼ tⱼ  [linear system]

PageRank:
    P_PR = αP + (1−α)(1/n)11ᵀ
    π = stationary distribution of P_PR
```

---

## 13. Practice Problems (Solve Before Day 30)

1. Chain on {1,2,3}: P=[[0,0.5,0.5],[0.3,0,0.7],[0.4,0.6,0]]. Find stationary distribution. Is the chain irreducible? Aperiodic?

2. Random walk on {0,1,...,5}: from state i∈(0,5), move left (prob 0.4) or right (prob 0.6). States 0,5 absorbing. Starting at 2, find E[steps until absorption].

3. **Prove** detailed balance πᵢPᵢⱼ=πⱼPⱼᵢ implies πP=π.

4. 3-page web: 1→{2}, 2→{1,3}, 3→{1}. Build P and find PageRank. Which page ranks highest?

5. *(Interview-level)* In a diffusion model, forward process: xₜ = √(1−β)xₜ₋₁ + √β·εₜ, εₜ~N(0,I). (a) Show this is a Markov chain. (b) What is the stationary distribution as T→∞? (c) Why does the reverse process require a learned neural network?

---

## 14. Looking Ahead

**Day 30** — **Final Review: 20 Top ML Interview Q&As & Master Cheat Sheet.** The complete consolidation of all 30 days — the most important questions, model answers, and every formula in one place.

---
*End of Day 29 | Next: Day 30 — Final Review & Master Cheat Sheet*
