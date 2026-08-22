# Day 19 — Law of Large Numbers (LLN)
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 10 (Section 10.3)
> **Style:** Andrew Ng — intuition first, math second, numericals always

> 💡 **TL;DR before you read anything else:** if you average enough independent samples, the average lands closer and closer to the true mean. That single sentence *is* the Law of Large Numbers. Everything below is either (a) making that statement precise, (b) proving it, or (c) showing you where it quietly powers something you already use in ML.

---

## 1. Why the LLN is the Foundation of ML

The Law of Large Numbers is the mathematical reason that **learning from data works at all**. Every time you trust a training loss curve, a test accuracy number, or a Monte Carlo estimate, you are silently leaning on this theorem.

| ML Concept | LLN Guarantee |
|---|---|
| Training loss → true risk | Empirical loss converges to expected loss |
| Sample accuracy → true accuracy | Test accuracy converges to true accuracy |
| Sample mean → population mean | x̄ → μ as n → ∞ |
| Histogram → true distribution | Empirical distribution → true distribution |
| Monte Carlo estimation | Average of samples → expected value |
| SGD convergence | Average gradient → true gradient |
| Bootstrapping validity | Bootstrap distribution → sampling distribution |
| A/B test validity | Observed difference → true difference |

> 💬 **Comment:** notice the pattern in every row — *"[something computed from a finite sample] → [the true, infinite-population quantity]"*. That arrow is always doing the same job. If you remember nothing else, remember that the LLN is what licenses you to replace an unknowable population quantity with a computable sample quantity.

Without the LLN, there would be no guarantee that anything learned from finite data means anything about the underlying population. It is the bedrock everything else in this course sits on.

---

## 2. Setup and Notation

Let $X_1, X_2, X_3, \dots$ be an i.i.d. (independent, identically distributed) sequence with:

$$E[X_i] = \mu \quad \text{(finite mean)}, \qquad \text{Var}(X_i) = \sigma^2 \quad \text{(finite variance — needed for the Weak LLN via Chebyshev)}$$

> 💬 **Comment — what "i.i.d." is buying you:** *independent* means no $X_i$ tells you anything about any other $X_j$ (no correlation to exploit or worry about). *Identically distributed* means every sample is drawn from the same underlying distribution (no shifting target). Both assumptions get violated constantly in the real world — Section 8 covers what happens when they break.

Define the **sample mean**:

$$\bar{X}_n = \frac{X_1 + X_2 + \cdots + X_n}{n} = \frac{1}{n}\sum_{i=1}^n X_i$$

**Key properties of $\bar{X}_n$** — and *why* each one holds:

$$E[\bar{X}_n] = \mu \qquad \text{[unbiased — always, regardless of } n \text{]}$$

*Why:* expectation is linear, so the expectation of an average of things is just the average of their expectations. Averaging never shifts where you're "centered," even with just $n=2$ samples.

$$\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n} \qquad \text{[variance shrinks as } 1/n \text{]}$$

*Why:* variance of a sum of *independent* variables adds up ($\text{Var}(\sum X_i) = \sum \text{Var}(X_i) = n\sigma^2$), but dividing by $n$ to get the average divides the variance by $n^2$ (variance scales quadratically with any constant multiplier). Net effect: $n\sigma^2/n^2 = \sigma^2/n$.

$$\text{SD}(\bar{X}_n) = \frac{\sigma}{\sqrt{n}} \qquad \text{[the "standard error" — this exact quantity shows up in every confidence interval you'll ever compute]}$$

As $n \to \infty$: $\text{Var}(\bar{X}_n) \to 0$. The sample mean's spread shrinks toward a single point — it **concentrates** around $\mu$.

> 💡 **Simplification:** unbiased ($E[\bar X_n]=\mu$) is a *static* fact — true for any $n$, even $n=1$. Concentration (shrinking variance) is the *dynamic* fact that actually requires $n$ to grow. The LLN is fundamentally a statement about the second one.

---

## 3. Weak Law of Large Numbers (WLLN)

> **Theorem (WLLN):** For any $\varepsilon > 0$:
> $$P(|\bar{X}_n - \mu| > \varepsilon) \to 0 \quad \text{as } n \to \infty$$

In plain English: *no matter how small a "closeness" tolerance $\varepsilon$ you demand, the probability that the sample mean misses that tolerance shrinks to zero as you collect more data.*

This mode of convergence has a name — **convergence in probability**: $\bar{X}_n \xrightarrow{p} \mu$.

### Proof via Chebyshev

$$P(|\bar{X}_n - \mu| > \varepsilon) \;\le\; \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} \qquad \text{[Chebyshev's inequality]}$$

$$= \frac{\sigma^2/n}{\varepsilon^2} \qquad \text{[substitute } \text{Var}(\bar{X}_n) = \sigma^2/n \text{ from Section 2]}$$

$$\to 0 \quad \text{as } n \to \infty \qquad \blacksquare$$

> 💬 **Comment — walking through this proof like an interviewer would want you to:**
> - **Line 1** is just Chebyshev's inequality applied to the random variable $\bar X_n$: "the probability any random variable strays more than $\varepsilon$ from its mean is at most its variance divided by $\varepsilon^2$." This line requires *nothing* about $\bar X_n$ specifically — it's a generic bound that works for any random variable with finite variance.
> - **Line 2** is the only place the LLN's specific setup (i.i.d. samples) enters — we substitute in the formula for $\text{Var}(\bar X_n)$ that we derived in Section 2, which is where the $1/n$ shrinkage comes from.
> - **Line 3** is just algebra: as $n\to\infty$, a fixed number ($\sigma^2/\varepsilon^2$) divided by a growing number ($n$) goes to zero.
>
> This is genuinely one of the shortest, cleanest proofs in all of probability theory — three lines, one classical inequality. It's a favorite "prove it on the spot" interview question precisely because it's short but tests whether you actually understand *why* variance shrinking implies the probability bound shrinking.

### What WLLN Says (and Doesn't Say)

**Says:** for any *fixed* $\varepsilon > 0$, the probability that $\bar X_n$ deviates from $\mu$ by more than $\varepsilon$ goes to zero as $n$ grows.

**Doesn't say:** $\bar X_n$ converges to $\mu$ for *every single* specific sequence of outcomes you could ever draw. It's a probabilistic statement — rare, pathological sequences where $\bar X_n$ never settles down are technically allowed, as long as the *probability* of drawing such a sequence vanishes.

> 💡 **Simplification — the "says vs. doesn't say" distinction in one line:** WLLN promises the *probability of being wrong* shrinks to zero. It does not promise that *every possible run of bad luck* is ruled out. (Section 4's Strong LLN tightens this gap.)

---

## 4. Strong Law of Large Numbers (SLLN)

> **Theorem (SLLN):**
> $$P\left(\bar{X}_n \to \mu \text{ as } n \to \infty\right) = 1$$

In plain English: *if you imagine running the sampling process forever, the set of "unlucky" universes where the running average never settles down at $\mu$ has probability exactly zero.*

This is **almost sure convergence**: $\bar{X}_n \xrightarrow{a.s.} \mu$.

### Difference: Weak vs Strong

| | WLLN | SLLN |
|---|---|---|
| **Statement** | $P(\lvert\bar X_n-\mu\rvert>\varepsilon)\to 0$ for each fixed $\varepsilon$ | $P(\bar X_n\to\mu)=1$ |
| **Convergence type** | In probability | Almost surely |
| **Exceptions allowed** | $P(\bar X_n \text{ fails to converge}) \to 0$ (in the limit) | $P(\bar X_n \text{ fails to converge}) = 0$ (exactly, always) |
| **Strength** | Weaker | Stronger (SLLN $\Rightarrow$ WLLN) |
| **Requirements** | Finite mean + finite variance | Finite mean only ($E[\lvert X\rvert]<\infty$) |
| **Proof difficulty** | Easy (3 lines via Chebyshev, above) | Hard (needs measure-theoretic tools) |

> 💬 **Comment — why does SLLN need a *weaker* assumption (mean only) but give a *stronger* conclusion?** This trips people up at first glance. The resolution: SLLN's proof doesn't route through Chebyshev at all (that's the "hard" part) — it uses different machinery (e.g., martingale or truncation arguments) that manages to avoid ever needing a variance bound. So "stronger conclusion, weaker assumption" isn't a contradiction — it's evidence the SLLN's proof technique is genuinely more powerful, not that something's wrong.

**For ML purposes:** WLLN is sufficient for essentially everything you'll reason about day to day (test accuracy stability, gradient noise, Monte Carlo error). SLLN is the "complete" theoretical version that mathematicians care about; know it exists and know the table above, but don't lose sleep over its proof.

---

## 5. Convergence Types — A Full Picture

There are four standard notions of convergence for sequences of random variables, roughly ordered strongest → weakest:

```
Almost Sure (a.s.)
      ↓ implies
In Probability
      ↓ implies
In Distribution (→ CLT, Day 20)
```

with **In $L^p$** (e.g. $L^2$, "mean-square convergence") sitting alongside and implying convergence in probability too.

$$\text{a.s. convergence:} \quad P(X_n \to X) = 1$$
$$\text{In probability:} \quad P(|X_n - X| > \varepsilon) \to 0 \ \text{ for all } \varepsilon > 0$$
$$\text{In distribution:} \quad F_n(x) \to F(x) \ \text{ for all continuity points } x \text{ of } F$$
$$\text{In } L^2\text{:} \quad E[(X_n - X)^2] \to 0$$

> 💡 **Simplification — you don't need to memorize the arrow diagram, just this intuition:** "almost sure" is a statement about *entire sequences/outcomes*; "in probability" is a statement about a *single snapshot at large $n$*; "in distribution" is the weakest — it only says the *shape of the histogram* matches, not that individual values are close at all. A sequence can converge in distribution to something and still, at every single $n$, have values wildly different from what it's "converging to" — that's how weak this mode is.

**For interviews:** know a.s. and in-probability cold (they're the two that show up in LLN). "In distribution" is the one you need for the CLT (Day 20) — file it away for tomorrow.

---

## 6. The LLN and Empirical Risk Minimization

The entire framework of supervised ML rests on the LLN. Here's the chain of reasoning, spelled out:

**True risk** (what you actually want to minimize — but can't, because you don't have the true data distribution $P$):

$$R(f) = E_{(X,Y)\sim P}[L(f(X), Y)]$$

**Empirical risk** (what you actually *can* compute, from your finite training set):

$$\hat{R}_n(f) = \frac{1}{n}\sum_{i=1}^n L(f(X_i), Y_i)$$

> 💬 **Comment — what's happening in these two equations:** $R(f)$ is an *expectation* over the entire (infinite, unknown) population — you'd need to see every possible $(X,Y)$ pair to compute it exactly. $\hat R_n(f)$ replaces that expectation with a plain average over the $n$ examples you actually have. Structurally, $\hat R_n(f)$ is *exactly* a sample mean $\bar X_n$ in disguise, where the "random variable" being averaged is the per-example loss $L(f(X_i), Y_i)$.

Because $\hat R_n(f)$ is literally a sample mean of i.i.d. loss values, the LLN applies directly:

$$\hat{R}_n(f) \xrightarrow{p} R(f) \quad \text{as } n \to \infty$$

**Empirical Risk Minimization (ERM)** — the entire strategy of "just minimize training loss" — is justified precisely by this: minimizing the empirical risk is a reasonable proxy for minimizing the true risk *because* the LLN guarantees the two converge as data grows.

The gap $|\hat R_n(f) - R(f)|$ is called the **generalization gap**. The LLN says this gap $\to 0$ for any *one fixed* $f$. But real training involves searching over many candidate $f$'s (an entire function class) — guaranteeing the gap shrinks *uniformly across the whole class simultaneously* needs stronger tools (VC theory, Rademacher complexity) that build on top of, but go beyond, the plain LLN.

---

## 7. Monte Carlo Estimation — LLN in Action

**Problem:** you want $E[g(X)]$ for some function $g$ that's too hard to integrate in closed form.

**Monte Carlo method:**
1. Sample $X_1, \dots, X_n$ i.i.d. from $P(X)$.
2. Estimate: $\hat{E}[g(X)] = \frac{1}{n}\sum_i g(X_i)$.

By the LLN: $\hat E[g(X)] \xrightarrow{p} E[g(X)]$ as $n \to \infty$.

> 💬 **Comment:** same trick as Section 6 — $g(X_i)$ is just "some random variable," and averaging it converges to its expectation for exactly the same reason $\bar X_n \to \mu$ does. Monte Carlo isn't a separate theorem; it's the LLN wearing a different hat.

**Error rate:** by the CLT (Day 20, previewed here):

$$\text{SD}(\hat E[g(X)]) = \frac{\text{SD}(g(X))}{\sqrt n}$$

Monte Carlo error shrinks as $1/\sqrt n$ — **regardless of the dimensionality of $X$**. This is *the* reason Monte Carlo is the tool of choice for high-dimensional integration: grid-based numerical integration methods get exponentially more expensive as dimension grows, but Monte Carlo's $1/\sqrt n$ rate doesn't care how many dimensions $X$ lives in.

---

## 8. When LLN Fails

The LLN needs **finite mean**, $E[|X|] < \infty$. It breaks down for:

| Distribution | Problem |
|---|---|
| Cauchy distribution | No finite mean at all — $\bar X_n$ does **not** converge |
| Heavy-tailed (power law, tail index $\alpha \le 1$) | Infinite mean → LLN fails |
| Non-i.i.d. data | LLN's proof needs independence |
| Non-stationary data | Distribution shift → "$\mu$" itself changes mid-stream, so there's no fixed target to converge to |

**Cauchy example, explained:**

$X \sim \text{Cauchy}(0,1)$: $f(x) = \dfrac{1}{\pi(1+x^2)}$.

You might expect $E[X]=0$ by symmetry — the density is symmetric around 0. But expectation isn't just "does the symmetric cancellation happen" — it requires the integral to converge *absolutely* first:

$$\int_0^\infty \frac{x}{1+x^2}\,dx = \left[\frac{\ln(1+x^2)}{2}\right]_0^\infty = \infty$$

The positive tail alone diverges to $+\infty$. So $E[|X|] = \infty$ — an **infinite first absolute moment**. The "symmetric cancellation" argument doesn't rescue you, because you're not allowed to add $+\infty$ and $-\infty$ and call it zero — the expectation is simply undefined. This is exactly the condition the LLN requires to be finite, and it fails.

> 💬 **Comment — the punchline that surprises people:** it's not merely that the Cauchy mean converges *slowly*. It doesn't converge *at all*, ever, no matter how many samples you collect. The Cauchy's moment-generating function doesn't exist either, but its characteristic function does: $\varphi_X(t) = e^{-|t|}$. Working out the characteristic function of $\bar X_n$:

$$\varphi_{\bar X_n}(t) = \left[\varphi_X(t/n)\right]^n = \left[e^{-|t|/n}\right]^n = e^{-|t|}$$

This is *identical* to $\varphi_X(t)$ — meaning **$\bar X_n$ has exactly the same distribution as a single $X_1$, for every $n$.** Averaging a million Cauchy samples gives you a random variable that's every bit as spread out as looking at just one sample. There is no concentration whatsoever.

**ML consequence:** if your loss function has heavy tails (e.g., squared loss amplifying rare huge outliers), batch-averaged gradients can behave Cauchy-like — a single extreme outlier in the batch can dominate the entire gradient estimate, and *increasing batch size does not fix this* the way it would for a well-behaved (finite-variance) loss. This is the actual mathematical justification for **gradient clipping** (cap the size of any individual gradient contribution) and **robust losses** like Huber (quadratic near zero, linear — i.e., bounded influence — in the tails) in modern deep learning.

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — Convergence Rate of Sample Mean

**Problem:** $X \sim \text{Bernoulli}(0.7)$. Compute $\bar X_n$ on $n$ samples.

**(a)** $E[\bar X_n]$ and $\text{Var}(\bar X_n)$ for $n = 10, 100, 1000, 10000$.
**(b)** Chebyshev bound on $P(|\bar X_n - 0.7| > 0.05)$ for each $n$.
**(c)** How large must $n$ be so that $P(|\bar X_n - 0.7| > 0.01) < 0.01$?
**(d)** Describe the convergence conceptually.

**Solution:**

$\text{Var}(X) = p(1-p) = 0.7 \times 0.3 = 0.21$.

**(a)**

| $n$ | $E[\bar X_n]$ | $\text{Var}(\bar X_n) = 0.21/n$ | $\text{SD}(\bar X_n)$ |
|---|---|---|---|
| 10 | 0.7 | 0.021 | 0.145 |
| 100 | 0.7 | 0.0021 | 0.0458 |
| 1,000 | 0.7 | 0.00021 | 0.0145 |
| 10,000 | 0.7 | 0.000021 | 0.00458 |

The mean stays anchored at 0.7 (unbiased at every $n$); the variance shrinks as $1/n$ exactly as predicted. ✓

**(b)** $P(|\bar X_n - 0.7| > 0.05) \le \dfrac{\text{Var}(\bar X_n)}{0.05^2} = \dfrac{0.21/n}{0.0025} = \dfrac{84}{n}$:

| $n$ | Chebyshev bound |
|---|---|
| 10 | $84/10 = 8.4 \to$ **1.0** (bound exceeds 1, so it's vacuous — Chebyshev is only useful once $n$ is large enough) |
| 100 | $84/100 = $ **0.84** |
| 1,000 | $84/1000 = $ **0.084** |
| 10,000 | $84/10000 = $ **0.0084** |

The bound $\to 0$ as $n\to\infty$, which is the WLLN in action. ✓

**(c)** Set $\dfrac{0.21}{n \times 0.01^2} < 0.01$:

$$\frac{0.21}{0.0001\,n} < 0.01 \;\Rightarrow\; \frac{2100}{n} < 0.01 \;\Rightarrow\; n > 210{,}000$$

Need **$n > 210{,}000$** for the Chebyshev guarantee. (Hoeffding, from Day 18, gives a much tighter — smaller — requirement, because it exploits the boundedness of Bernoulli variables rather than just their variance.)

**(d)** Conceptual convergence:

```
n=10:    X̄ₙ wanders widely: values from 0.4 to 1.0 are common
n=100:   X̄ₙ mostly lands in [0.6, 0.8]
n=1000:  X̄ₙ almost always in [0.67, 0.73]
n=10000: X̄ₙ essentially pinned at 0.7 ± 0.005
         ↓
n→∞:     X̄ₙ = 0.7 exactly (almost surely)
```

**ML insight:** this is exactly why test-set accuracy is only trustworthy once $n$ is reasonably large. With only $n=100$ test samples, your accuracy estimate has $\text{SD} \approx 4.6\%$ — you could easily report "74% accuracy" when the true accuracy is 70%.

---

### 🔢 Numerical 2 — LLN for Loss Functions: Training Convergence

**Problem:** each training step computes loss on one sample. True expected loss $\mu = 0.3$, $\text{Var(loss)} = 0.25$.

**(a)** After $n$ steps, how close is the average loss to the true expected loss (Chebyshev bound)?
**(b)** For SGD's average loss to be within 0.01 of true loss with 95% confidence, how many steps are needed?
**(c)** Why does SGD work even without $\bar X_n$ fully converging?

**Solution:**

**(a)** After $n$ steps, $\bar X_n$ = average loss over $n$ samples.

Chebyshev: $P(|\bar X_n - 0.3| > \varepsilon) \le \dfrac{0.25}{n\varepsilon^2}$

For $\varepsilon = 0.05$: $P(|\bar X_n - 0.3| > 0.05) \le \dfrac{0.25}{n \times 0.0025} = \dfrac{100}{n}$

- After 100 steps: bound $= 1.0$ (vacuous)
- After 1,000 steps: bound $= 0.1$ (10% chance of being off by 0.05)
- After 10,000 steps: bound $= 0.01$ (1% chance)

**(b)** Set $\dfrac{0.25}{n\varepsilon^2} \le 0.05$ with $\varepsilon = 0.01$:

$$n \ge \frac{0.25}{0.05 \times 0.0001} = \frac{0.25}{0.000005} = 50{,}000$$

Need **50,000 steps** to guarantee, via Chebyshev, that average loss is within 0.01 of true loss with 95% probability.

**(c)** SGD doesn't actually need $\bar X_n \to \mu$ to have *already happened globally*. It only needs, at every single step:
- Each individual stochastic gradient to be an **unbiased estimate** of the true gradient — guaranteed by $E[\bar X_n]=\mu$ holding for *any* $n$, even $n=1$ (Section 2).
- The *noise* in that gradient to be something that tends to cancel out, on average, across many steps — this is the LLN's variance-shrinking guarantee playing out implicitly as training accumulates steps.

> 💬 **Comment:** this is a subtle but important point — SGD doesn't wait around for $\bar X_n$ to converge before doing anything useful. It exploits the *unbiasedness* property immediately (every single noisy step points in the right direction *on average*) and lets the *averaging-out effect accumulate naturally* over the course of training, via the optimization trajectory itself, rather than via one explicit averaging step.

---

### 🔢 Numerical 3 — Monte Carlo Integration (Estimating π)

**Problem:** estimate $\pi$ via Monte Carlo. Formally:

$$E\left[\mathbb{1}_{x^2+y^2 \le 1}\right] = P(\text{point falls in unit circle}) = \frac{\pi}{4}$$

where $(x,y) \sim \text{Uniform}(0,1)^2$.

**(a)** Show this is an LLN application.
**(b)** Expected error after $n=10{,}000$ samples.
**(c)** How many samples for error $< 0.001$ with probability $\ge 95\%$?

**Solution:**

Define $X_i = 1$ if point $i$ falls inside the unit circle, else 0. Then $X_i \sim \text{Bernoulli}(\pi/4)$.

$$\bar X_n = \frac{1}{n}\sum X_i \xrightarrow{p} \frac{\pi}{4} \quad \text{by LLN, so} \quad \hat\pi = 4\bar X_n \xrightarrow{p} \pi \quad \checkmark$$

**(a)** This is a direct LLN application: the sample mean of i.i.d. indicator variables converges to their shared expected value.

**(b)** $\text{Var}(X_i) = \dfrac{\pi}{4}\left(1-\dfrac{\pi}{4}\right) \approx 0.7854 \times 0.2146 \approx 0.1685$

$$\text{Var}(\bar X_n) = \frac{0.1685}{10000} = 0.00001685, \qquad \text{SD}(\bar X_n) = 0.004105$$

$$\text{SD}(\hat\pi) = 4 \times \text{SD}(\bar X_n) = \mathbf{0.01642}$$

Expected error (roughly 1 standard deviation) after 10,000 samples $\approx 0.016$.

**(c)** For error in $\hat\pi < 0.001$ with 95% confidence (using the Normal approximation, $z=1.96$):

$$4\sqrt{\frac{0.1685}{n}} < \frac{0.001}{1.96}$$

$$\sqrt{\frac{0.1685}{n}} < 0.0001276 \;\Rightarrow\; \frac{0.1685}{n} < 1.629\times 10^{-8} \;\Rightarrow\; n > \frac{0.1685}{1.629\times 10^{-8}} \approx 10{,}344{,}600$$

Need roughly **10 million samples** for 3-decimal accuracy in $\pi$. Monte Carlo is accurate but slow, precisely because of the $1/\sqrt n$ wall.

**ML insight:** Monte Carlo in ML (MC dropout for uncertainty, MCMC sampling, variational inference) inherits this same $1/\sqrt n$ convergence. This is exactly why:
- MC dropout needs *many* forward passes to get a stable uncertainty estimate.
- MCMC chains need to run for a long time to get accurate posterior estimates.
- Importance sampling and other variance-reduction tricks exist — they're specifically designed to cut down the *constant* in front of the $1/\sqrt n$ rate, since you usually can't escape the $1/\sqrt n$ itself.

---

### 🔢 Numerical 4 — LLN Failure: Cauchy Distribution

**Problem:** $X_1, \dots, X_n \sim \text{Cauchy}(0,1)$. What happens to $\bar X_n$?

**(a)** Why does the LLN fail for Cauchy?
**(b)** What is the distribution of $\bar X_n$?
**(c)** What is the ML consequence?

**Solution:** — worked through in full in Section 8 above. Summary:

**(a)** $E[|X|] = \infty$ (infinite first absolute moment) — the LLN's core requirement fails outright.

**(b)** $\bar X_n \sim \text{Cauchy}(0,1)$ — identical to the distribution of a single sample, for **every** $n$. No concentration ever happens.

**(c)** Heavy-tailed losses/gradients can behave this way; a single extreme outlier can dominate a batch gradient regardless of batch size. Fixes: gradient clipping, Huber loss, other bounded-influence robust losses.

---

### 🔢 Numerical 5 — LLN and Empirical Risk Minimization

**Problem:** binary classifier $f$ with true error rate $R(f) = P(f(X) \ne Y) = 0.12$. Evaluate on $n$ test samples, giving observed error rate $\hat R_n(f)$.

**(a)** $E[\hat R_n(f)]$ and $\text{Var}(\hat R_n(f))$.
**(b)** $P(|\hat R_n - 0.12| > 0.02)$ for $n = 100, 500, 1000$ (Hoeffding).
**(c)** What $n$ guarantees $|\hat R_n - R(f)| < 0.01$ with probability $\ge 99\%$?
**(d)** If you compare 5 models simultaneously, how does this change?

**Solution:**

Let $X_i = 1$ if sample $i$ is misclassified, so $X_i \sim \text{Bernoulli}(0.12)$ and $\hat R_n = \bar X_n$.

**(a)**
$$E[\hat R_n] = 0.12 \qquad \text{[unbiased, by LLN's mean property]}$$
$$\text{Var}(\hat R_n) = \frac{0.12 \times 0.88}{n} = \frac{0.1056}{n}$$

**(b)** Using Hoeffding's inequality (tighter than Chebyshev because it exploits boundedness, i.e. every $X_i \in \{0,1\}$):

$$P(|\hat R_n - 0.12| > 0.02) \le 2\exp(-2n \times 0.02^2) = 2\exp(-0.0008n)$$

| $n$ | Hoeffding bound |
|---|---|
| 100 | $2e^{-0.08} \approx 1.85 \to$ capped at 1.0 (still useless) |
| 500 | $2e^{-0.4} \approx 1.34 \to$ capped at 1.0 |
| 1,000 | $2e^{-0.8} \approx$ **0.899** |
| 5,000 | $2e^{-4.0} \approx$ **0.037** |

You need roughly **5,000 samples** before the bound becomes practically meaningful at $\varepsilon = 0.02$.

**(c)** Hoeffding with $\varepsilon = 0.01$, target failure probability $\delta = 0.01$:

$$2\exp(-2n \times 0.0001) \le 0.01 \;\Rightarrow\; \exp(-0.0002n) \le 0.005 \;\Rightarrow\; -0.0002n \le \ln(0.005) = -5.298$$

$$n \ge \frac{5.298}{0.0002} \approx 26{,}491$$

Need roughly **26,500 test samples** for 99% confidence within 0.01 error.

**(d)** For 5 models tested on the same set, use a **union bound** (the probability that *any one* of several events happens is at most the sum of their individual probabilities):

$$P(\text{any model's } \hat R_n \text{ off by} > \varepsilon) \le 5 \times 2\exp(-2n\varepsilon^2) = 10\exp(-2n\varepsilon^2)$$

Setting this equal to 0.01: $\exp(-2n\varepsilon^2) = 0.001 \Rightarrow n \ge \dfrac{\ln(1000)}{2 \times 0.0001} \approx 34{,}539$

Need roughly **34,500 samples** when comparing 5 models simultaneously (versus ~26,500 for evaluating just one).

> 💬 **Comment — the intuition behind (d):** every additional model you evaluate on the same test set is another "chance to get unlucky" — the best-looking model might just be the one that happened to draw a favorable sample, not the genuinely best model. This is the **multiple comparisons problem**, and the union bound quantifies exactly how much larger your test set needs to be to compensate as you evaluate more candidates.

---

### 🔢 Numerical 6 — LLN for Gradient Estimation in SGD

**Problem:** true gradient $\nabla L = -0.5$. Each stochastic gradient estimate $G \sim N(-0.5, 1.0)$ (noise variance = 1).

**(a)** $E[\bar G_n]$ and $\text{Var}(\bar G_n)$ for batch sizes $n = 1, 8, 32, 128$.
**(b)** $P(\bar G_n > 0)$ — probability of stepping in the *wrong* direction — for each batch size.
**(c)** Why does large-batch SGD converge faster per step but potentially worse overall?

**Solution:**

**(a)**

| $n$ | $E[\bar G_n]$ | $\text{Var}(\bar G_n) = 1/n$ | $\text{SD}(\bar G_n)$ |
|---|---|---|---|
| 1 | −0.5 | 1.00 | 1.000 |
| 8 | −0.5 | 0.125 | 0.354 |
| 32 | −0.5 | 0.031 | 0.177 |
| 128 | −0.5 | 0.0078 | 0.088 |

All batch sizes give an unbiased estimate of the true gradient (mean stays at −0.5). Variance shrinks as $1/n$. ✓

**(b)** "Wrong direction" means $\bar G_n$ ends up positive when the truth is negative. Standardizing:

$$P(\bar G_n > 0) = P\!\left(Z > \frac{0.5}{\text{SD}(\bar G_n)}\right)$$

| $n$ | threshold $=0.5/\text{SD}$ | $P(\text{wrong direction})$ |
|---|---|---|
| 1 | $0.5/1.0 = 0.5$ | $P(Z>0.5)\approx 30.9\%$ |
| 8 | $0.5/0.354 = 1.41$ | $P(Z>1.41)\approx 7.9\%$ |
| 32 | $0.5/0.177 = 2.83$ | $P(Z>2.83)\approx 0.23\%$ |
| 128 | $0.5/0.088 = 5.68$ | $P(Z>5.68)\approx 0\%$ |

**(c)** Larger batches:
- ✅ More accurate gradient estimates → fewer wrong-direction steps (as shown above)
- ✅ Faster convergence *per step*
- ❌ More compute *per step*
- ❌ Often converge to **sharper minima** that generalize worse — the well-documented "generalization gap" of large-batch training
- ❌ Less implicit regularization, since gradient noise itself acts as a mild regularizer that helps escape sharp, narrow minima

> 💡 **Simplification:** the noise the LLN shrinks as batch size grows is *not* purely a nuisance to be eliminated. It has a side benefit — it jostles the optimizer out of sharp minima and toward flatter, better-generalizing ones. That's the real reason practitioners often prefer moderate batch sizes (32–256) over very large ones (1024+), even when compute budgets would allow it.

---

### 🔢 Numerical 7 — Strong vs Weak LLN: Almost Sure Convergence

**Problem:** $X_1, X_2, \dots \sim \text{Bernoulli}(0.5)$. Define the "bad event" $A_n = \{|\bar X_n - 0.5| > 0.1\}$ (the sample mean is far from 0.5).

**(a)** $P(A_n)$ for various $n$ (Normal approximation).
**(b)** Does $\sum_n P(A_n)$ converge? (Borel–Cantelli connection.)
**(c)** What does the Strong LLN say about the sequence $\{A_n\}$?
**(d)** What does "almost surely" mean in practice?

**Solution:**

**(a)** $\bar X_n \approx N(0.5,\, 0.25/n)$ by the CLT approximation.

$$P(A_n) = P\!\left(|Z| > \frac{0.1}{\sqrt{0.25/n}}\right) = P(|Z| > 0.2\sqrt n)$$

| $n$ | $0.2\sqrt n$ | $P(A_n)$ |
|---|---|---|
| 1 | 0.2 | $2 \times 0.421 = 0.842$ |
| 25 | 1.0 | $2 \times 0.159 = 0.317$ |
| 100 | 2.0 | $2 \times 0.023 = 0.046$ |
| 400 | 4.0 | $2 \times 3.2\times 10^{-5} \approx 6.4\times10^{-5}$ |
| 10,000 | 20 | $\approx 0$ |

$P(A_n)\to 0$ — this is exactly the WLLN. ✓

**(b)** Because $P(A_n)$ decays roughly like a Gaussian tail (super fast — much faster than $1/n$), the infinite sum $\sum_n P(A_n)$ **converges** to a finite number. By the **Borel–Cantelli lemma**: if $\sum_n P(A_n) < \infty$, then $P(A_n \text{ occurs infinitely often}) = 0$.

**(c)** The SLLN says $P(\bar X_n \to 0.5) = 1$, which is the same statement as $P(A_n \text{ infinitely often}) = 0$: with probability 1, $\bar X_n$ eventually settles inside the band $(0.4, 0.6)$ and **stays there forever** — it doesn't just visit and leave repeatedly.

**(d)** "Almost surely" is a statement about the space of *all possible infinite sequences* you could ever draw. If you imagine running the sampling process infinitely many times, in parallel universes, the collection of universes where $\bar X_n$ never settles down has *total probability mass zero* — even though a specific "impossible-looking" sequence (like an unbroken run of a million heads in a row) is technically not forbidden, the set of such pathological sequences is so vanishingly rare it doesn't register in the probability measure at all.

> 💬 **Comment — the practical translation:** you will never, in any real-world run of a sampling process, actually observe $\bar X_n$ failing to converge. The exceptional sequences are mathematically real but practically irrelevant — a distinction that matters for measure theory, not for anything you'll ever debug in a training run.

---

## 10. LLN vs CLT — Preview

| | LLN (Day 19) | CLT (Day 20) |
|---|---|---|
| **Statement** | $\bar X_n \to \mu$ | $\sqrt n(\bar X_n - \mu)/\sigma \to N(0,1)$ |
| **What it gives you** | Convergence to a single point | The *rate* and *shape* of that convergence |
| **What's being scaled** | $\bar X_n$ itself | $\sqrt n \times (\bar X_n - \mu)$ — note the rescaling |
| **The limit is...** | A number ($\mu$) | A distribution (the Normal curve) |
| **Used for** | Justifying that estimators are consistent at all | Building confidence intervals, running hypothesis tests |

> 💡 **One-line summary to memorize:** *the LLN tells you WHERE the sample mean ends up; the CLT tells you HOW FAST it gets there, and in what SHAPE the leftover randomness is distributed.*

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "State the Law of Large Numbers." | $\bar X_n \xrightarrow{p} \mu$ (WLLN) or $\bar X_n \xrightarrow{a.s.} \mu$ (SLLN) |
| "Prove the WLLN." | Apply Chebyshev: $P(\lvert\bar X_n-\mu\rvert>\varepsilon) \le \sigma^2/(n\varepsilon^2) \to 0$ |
| "Difference between Weak and Strong LLN?" | In-probability vs. almost-sure convergence; SLLN is strictly stronger and needs only a finite mean |
| "Why does the LLN justify empirical risk minimization?" | $\hat R_n(f) \xrightarrow{p} R(f)$ for any fixed $f$, because $\hat R_n(f)$ is literally a sample mean of per-example losses |
| "When does the LLN fail?" | Infinite mean (Cauchy), non-i.i.d. data, or a distribution that shifts over time (non-stationarity) |
| "What is the distribution of $\bar X_n$ for Cauchy data?" | Cauchy(0,1) — identical to a single sample's distribution, for every $n$; it never concentrates |
| "How does batch size relate to the LLN?" | Larger batch $\Rightarrow$ variance of the gradient estimate shrinks as $1/n$, exactly the LLN's variance formula applied to gradients |
| "What is the Monte Carlo convergence rate, and why?" | Error $\propto 1/\sqrt n$ — direct consequence of $\text{SD}(\bar X_n) = \sigma/\sqrt n$ (LLN + CLT together) |

---

## 12. Key Formulas — Cheat Sheet for Day 19

```
Sample Mean:
    X̄ₙ = (1/n) Σᵢ Xᵢ
    E[X̄ₙ] = μ                [unbiased, holds for any n]
    Var(X̄ₙ) = σ²/n           [shrinks as 1/n]
    SD(X̄ₙ) = σ/√n            [the "standard error"]

WLLN:
    P(|X̄ₙ − μ| > ε) ≤ σ²/(nε²) → 0  as n→∞
    ↳ convergence in probability

SLLN:
    P(X̄ₙ → μ) = 1
    ↳ almost sure convergence

Proof of WLLN (3 lines, via Chebyshev):
    P(|X̄ₙ−μ|>ε) ≤ Var(X̄ₙ)/ε²   [Chebyshev, generic]
                 = σ²/(nε²)       [plug in Var(X̄ₙ) = σ²/n]
                 → 0               [as n→∞]  ∎

Empirical Risk (LLN in an ML costume):
    R̂ₙ(f) →ᵖ R(f)   [justifies ERM]

Monte Carlo (LLN in another costume):
    (1/n)Σg(Xᵢ) →ᵖ E[g(X)]        [LLN gives convergence]
    Error ~ σ_{g(X)}/√n            [CLT gives the rate, Day 20]

LLN Requires:
    i.i.d. samples
    E[|X|] < ∞    [enough for SLLN]
    E[X²] < ∞     [enough for WLLN via Chebyshev]

LLN Fails For:
    Cauchy (infinite mean)     → X̄ₙ ~ Cauchy for every n, no concentration
    Heavy tails (E[|X|]=∞)     → same failure mode
    Non-i.i.d. / non-stationary data → no fixed target to converge to

Convergence Types (strongest → weakest):
    Almost sure  ⟹  In Lᵖ  ⟹  In probability  ⟹  In distribution
```

---

## 13. Practice Problems (Solve Before Day 20)

1. $X \sim \text{Exponential}(\lambda=2)$. Using the WLLN, what does $\bar X_n$ converge to? What is $\text{Var}(\bar X_n)$? How large must $n$ be so that $P(|\bar X_n - 0.5| > 0.01) < 0.05$ by Chebyshev?

2. A model's per-sample accuracy is i.i.d. Bernoulli($p$). You observe $\bar X_{500} = 0.92$.
   - What does the LLN say about this estimate?
   - Give a Chebyshev bound for how far 0.92 could plausibly be from the true $p$.
   - Give a Hoeffding bound for the same question.

3. **Prove** that $E[\bar X_n] = \mu$ and $\text{Var}(\bar X_n) = \sigma^2/n$ for i.i.d. $X_1,\dots,X_n$, using linearity of expectation and independence explicitly.

4. You run Monte Carlo to estimate $\int_0^1 x^2\,dx = 1/3$, using $n$ samples from Uniform(0,1) and averaging $X_i^2$.
   - What is $\text{Var}(X^2)$ for $X \sim \text{Uniform}(0,1)$?
   - How many samples are needed for the Monte Carlo estimate to land within 0.001 of $1/3$ with 95% probability?

5. *(Interview-level)* A federated learning system has $K=100$ clients, each with $n_k = 50$ local samples. The global model gradient is averaged across clients. Each client's gradient estimate has variance $\sigma^2 = 1$.
   - What is $\text{Var}(\text{global gradient})$ if all clients share the same underlying data distribution?
   - What changes if clients have *different* distributions (non-i.i.d. data)? Does the LLN still apply in any form?
   - What is the main practical challenge federated learning faces, viewed through an LLN lens?

*(Full worked solutions to all five, plus a follow-up set of interview questions on this material, are available if you want them — just ask.)*

---

## 14. Looking Ahead

**Day 20 — The Central Limit Theorem (CLT).** Where the LLN tells us that $\bar X_n \to \mu$, the CLT tells us the *rate and shape* of that convergence: $\sqrt n (\bar X_n - \mu)/\sigma \to N(0,1)$. The CLT is why Normal distributions show up everywhere, why confidence intervals come out bell-shaped, and why the Normal distribution earns its reputation as the "attractor" of probability theory.

---
*End of Day 19 | Next: Day 20 — The Central Limit Theorem*
