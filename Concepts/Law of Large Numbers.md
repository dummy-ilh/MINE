## Law of Large Numbers (LLN)

**The core idea:**

As you increase the number of trials (or sample size) in a random experiment, the **average of the results gets closer and closer to the expected (true) value**.

Formally, if $X_1, X_2, ..., X_n$ are i.i.d. random variables with expected value $\mu$, then:

$$\bar{X}_n = \frac{X_1 + X_2 + ... + X_n}{n} \to \mu \quad \text{as } n \to \infty$$

**Intuition with the dice example from before:** roll one die — you might get a 6. Roll it 10 times — the average might be 4.2, still noisy. Roll it 100,000 times — the average will be extremely close to the true expected value of 3.5. LLN is *why* your simulation earlier converged toward 3.5 as n grew.

---

### Two flavors (this is the classic interview distinction)

**1. Weak Law of Large Numbers (WLLN)**
The sample mean converges *in probability* to μ. Meaning: for any tiny tolerance ε, the probability that $|\bar{X}_n - \mu| > \varepsilon$ shrinks to 0 as n grows. But it doesn't guarantee this holds for every single sequence — just that large deviations become increasingly unlikely.

**2. Strong Law of Large Numbers (SLLN)**
The sample mean converges *almost surely* to μ — meaning, with probability 1, the actual sequence of sample averages will settle down and stay near μ forever as n → ∞. This is a stronger, pathwise guarantee.

(In practice, for interviews you mostly need to know these two exist and that "strong" is the stronger guarantee — you rarely need the measure-theoretic details unless it's a quant/stats-heavy role.)

---

### LLN vs. CLT — the question that trips people up

| | LLN | CLT |
|---|---|---|
| What it tells you | **Where** the sample mean goes (converges to μ) | **How** it fluctuates around μ on the way there |
| Statement type | Convergence (a point) | Distributional shape (a curve) |
| Output | A single number (μ) | A distribution: $\mathcal{N}(\mu, \sigma^2/n)$ |

Think of it like: LLN tells you the dartboard your throws are converging on; CLT tells you the shape of the scatter pattern around the bullseye as you keep throwing.

---

## Interview Q&A

**Q1: What does LLN guarantee — and what does it NOT guarantee?**
A: It guarantees the sample mean converges to the true mean as n grows. It does **not** guarantee anything about a small sample "catching up" — this is the misconception behind the **gambler's fallacy** (e.g., "I've lost 5 coin flips in a row, so I'm due for a win" — LLN doesn't mean future outcomes correct for past imbalance; it just means past imbalance becomes negligible against a much larger n).

**Q2: Why do casinos rely on LLN, not CLT?**
A: Casino games have a fixed negative expected value for the player (house edge). LLN guarantees that over millions of bets, the *actual* average payout converges to that expected value — this is why casinos are reliably profitable at scale even though any single bet is random.

**Q3: In ML, where does LLN show up besides "more data is better"?**
A: Monte Carlo estimation — e.g., estimating expected reward in reinforcement learning, or estimating an integral/expectation via sampling — relies on LLN to justify that the sample average of many simulated rollouts converges to the true expected value as the number of simulations grows.

**Q4: Does a larger sample size guarantee your sample mean is closer to the true mean than a smaller sample, every single time?**
A: No — that's the WLLN vs. "guaranteed" trap. WLLN only guarantees convergence in probability (large deviations become rare, not impossible). Any individual large sample *could* still be unusually far from μ, just very unlikely.
