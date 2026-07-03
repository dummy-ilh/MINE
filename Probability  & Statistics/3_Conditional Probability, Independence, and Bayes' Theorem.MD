# Random Variables — Complete Interview Guide (L5 Google/Apple)

**Reference texts** (same set as the conditional probability guide, so notation stays consistent):
- Sheldon Ross, *A First Course in Probability* — Ch. 4 (discrete RVs), Ch. 5 (continuous RVs)
- Blitzstein & Hwang, *Introduction to Probability* (Stat 110) — Ch. 3 (discrete RVs, distributions), Ch. 4 (expectation), Ch. 5 (continuous RVs)
- DeGroot & Schervish, *Probability and Statistics* — Ch. 3–5
- Casella & Berger, *Statistical Inference* — Ch. 2–4 (more rigorous, if you want the measure-theoretic underpinnings)

This guide picks up exactly where the conditional-probability guide left off, and deliberately over-explains the *why* behind each definition — not just the formula — since that's what separates a memorized answer from one that survives follow-up questions in an interview.

---

## 1. What a Random Variable Actually Is

### 1.1 The formal definition (and why it looks weird at first)
A random variable (RV) is **a function that maps outcomes in the sample space to real numbers**: $X: S \to \mathbb{R}$.

This trips people up because "variable" makes it sound like a number, but it's actually a *function*. The reason this matters: probability is defined on events (subsets of the sample space), but we want to talk about "the probability that the die roll is 4" — a random variable is exactly the bridge that lets us translate outcomes of an experiment into numbers we can compute with, and it's what lets $P(X=4)$ make sense as shorthand for $P(\{s \in S : X(s) = 4\})$.

**Concrete example:** Roll two dice. The sample space $S$ has 36 outcomes like $(3,5)$. Define $X(s) = $ sum of the two dice. Then $X((3,5)) = 8$. $X$ is a function; $8$ is just one possible *value* $X$ can take. When we write "$P(X=8)$," we really mean "the probability of the *event* $\{s : X(s)=8\}$" — which is the set of all dice-pairs summing to 8.

### 1.2 Discrete vs. Continuous
- **Discrete RV**: takes a countable set of values (finite or countably infinite) — e.g., number of heads in 10 flips, number of customers arriving in an hour.
- **Continuous RV**: takes an uncountable range of values (an interval of reals) — e.g., height, waiting time, temperature.

**Why the distinction matters practically:** for a discrete RV, $P(X=x)$ can be a genuinely positive number. For a continuous RV, $P(X=x) = 0$ for every single point $x$ — probability only accumulates over an *interval*. This is not a technicality; it's why continuous RVs need a *density* rather than a probability mass, and it's a common interview trip-wire ("what's the probability a continuous RV equals exactly 2.5?" — the answer is always 0, and being able to explain *why* in one sentence signals real understanding, not just formula recall).

---

## 2. Discrete Random Variables

### 2.1 Probability Mass Function (PMF)
$$p_X(x) = P(X=x)$$

Must satisfy: $p_X(x) \ge 0$ for all $x$, and $\sum_x p_X(x) = 1$.

**Intuition:** the PMF is literally "how much probability mass is stacked on top of each possible value." Drawing it as a bar chart (spikes at each valid value) is usually the fastest way to build intuition before doing any algebra.

### 2.2 Cumulative Distribution Function (CDF)
$$F_X(x) = P(X \le x)$$

For a discrete RV, this is a step function: it jumps up by $p_X(x)$ at each value $x$ can take, and is flat in between. $F_X$ is non-decreasing, $\lim_{x\to-\infty}F_X(x)=0$, $\lim_{x\to\infty}F_X(x)=1$.

**Why bother with the CDF if you already have the PMF?** Two reasons that come up constantly in interviews: (1) it's the only well-defined object for continuous RVs (no PMF exists there), so building CDF fluency now pays off directly in Section 3; (2) many real questions are phrased as "at least" / "at most" ($P(X\le 5)$, $P(X>10)$), which the CDF answers directly without re-summing the PMF each time.

---

## 3. Continuous Random Variables

### 3.1 Probability Density Function (PDF)
For a continuous RV, there's a function $f_X(x) \ge 0$ (the **density**) such that:
$$P(a \le X \le b) = \int_a^b f_X(x)\,dx, \qquad \int_{-\infty}^{\infty} f_X(x)\,dx = 1$$

**Critical intuition — the density is NOT a probability.** $f_X(x)$ can exceed 1 (e.g., a Uniform(0, 0.5) density is 2 everywhere on its support). What must integrate to 1 is the *area under the curve*, not the height. Interviewers sometimes probe this directly ("can a PDF value exceed 1?" — yes, absolutely, as long as the total area is still 1).

### 3.2 CDF for continuous RVs
$$F_X(x) = P(X\le x) = \int_{-\infty}^x f_X(t)\,dt \qquad\Longrightarrow\qquad f_X(x) = F_X'(x)$$

The PDF is literally the derivative of the CDF. This relationship is the single most useful algebraic fact for continuous-RV problems: if you're ever given a CDF and asked for the density (or vice versa), differentiate or integrate — don't overthink it.

---

## 4. Expectation

### 4.1 Definition
**Discrete:** $E[X] = \sum_x x\, p_X(x)$
**Continuous:** $E[X] = \int_{-\infty}^{\infty} x\, f_X(x)\,dx$

**Intuition (long-run average interpretation):** if you repeated the experiment many times and averaged the observed values of $X$, that average converges to $E[X]$ — this is the Law of Large Numbers, and it's why "expectation" and "long-run average" are used almost interchangeably in casual interview conversation, even though formally $E[X]$ is defined *before* any experiment is run.

**Common trap — expectation is not "the most likely value."** For a fair die, $E[X] = 3.5$, a value the die can never actually show. Interviewers love asking "what does it mean for the expected value to not be achievable?" — the honest answer is that expectation is a weighted average, not a mode or a prediction of any single trial's outcome.

### 4.2 LOTUS (Law of the Unconscious Statistician)
To find $E[g(X)]$ for some function $g$, you do **not** need to first find the distribution of $g(X)$ — you can compute it directly from the distribution of $X$:
$$E[g(X)] = \sum_x g(x)\,p_X(x) \quad\text{or}\quad \int g(x)f_X(x)\,dx$$

**Why this saves enormous work in interviews:** if asked for $E[X^2]$, you do *not* need to derive the distribution of $X^2$ first (which can be genuinely painful, especially if $g$ isn't monotonic). Just plug $g(x)=x^2$ directly into the sum/integral over $X$'s own distribution. This single fact eliminates most of the pain in variance calculations (see Section 5).

### 4.3 Linearity of Expectation (the single most powerful tool in this whole guide)
$$E[aX+b] = aE[X]+b \qquad E[X+Y] = E[X]+E[Y] \quad\text{(always — no independence required!)}$$

**This is worth dwelling on, because it's the most under-appreciated fact in applied probability:** linearity of expectation holds *even when $X$ and $Y$ are dependent*. This is what makes the **indicator random variable trick** (Section 12) so powerful — you can decompose a horrendously complicated random variable into a sum of simple 0/1 indicators, and compute the expectation of the sum by just adding up each indicator's individual expectation, with zero regard for how they interact with each other.

---

## 5. Variance and Standard Deviation

### 5.1 Definition
$$\text{Var}(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2$$

**Why the second form is what you'll actually use 95% of the time:** the "$E[X^2]-(E[X])^2$" shortcut (sometimes called the computational formula) avoids ever needing to compute a distribution centered at the mean — just find $E[X]$ and $E[X^2]$ (via LOTUS with $g(x)=x^2$) and subtract the square. **Derive this once by hand** — expand $(X-\mu)^2 = X^2-2\mu X+\mu^2$, take expectation term by term using linearity, and you'll never forget where it comes from.

$$\text{SD}(X) = \sqrt{\text{Var}(X)}$$

Standard deviation is in the *same units* as $X$ itself (variance is in squared units) — that's the practical reason SD, not variance, is what you report when communicating spread to a non-technical audience.

### 5.2 Properties of Variance
$$\text{Var}(aX+b) = a^2\text{Var}(X) \quad\text{(shifting doesn't change spread; scaling does, and it scales quadratically)}$$

**Why variance is NOT linear like expectation (a favorite "explain the difference" interview question):** $E$ is linear because it's fundamentally a sum/integral (a linear operator). $\text{Var}$ involves *squaring* a deviation, and squaring is not a linear operation — so variance picks up cross-terms (covariance) whenever you add two RVs together:
$$\text{Var}(X+Y) = \text{Var}(X)+\text{Var}(Y)+2\,\text{Cov}(X,Y)$$
Only collapses to $\text{Var}(X)+\text{Var}(Y)$ when $X,Y$ are independent (or merely uncorrelated) — unlike linearity of $E$, which needs no such assumption. **Getting this distinction right, unprompted, is a strong positive signal in an interview.**

---

## 6. Named Discrete Distributions (with full derivations, not just formulas)

### 6.1 Bernoulli($p$)
A single trial, success with probability $p$, failure with probability $1-p$. $X \in \{0,1\}$.

$$E[X] = 0(1-p)+1(p) = p \qquad \text{Var}(X) = E[X^2]-(E[X])^2 = p - p^2 = p(1-p)$$

(Since $X^2=X$ when $X\in\{0,1\}$, so $E[X^2]=E[X]=p$ — a small but genuinely useful shortcut worth remembering: **for any indicator/Bernoulli RV, $E[X^2]=E[X]$.**)

### 6.2 Binomial($n,p$) — sum of $n$ i.i.d. Bernoullis
$X$ = number of successes in $n$ independent trials, each with success probability $p$.

$$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}, \quad k=0,1,\dots,n$$

**Derive the mean the easy way, via linearity (not brute-force summation):** write $X = X_1+\cdots+X_n$ where each $X_i \sim \text{Bernoulli}(p)$ indicates success on trial $i$. Then
$$E[X] = \sum_{i=1}^n E[X_i] = np$$
This is dramatically easier than summing $\sum_k k\binom{n}{k}p^k(1-p)^{n-k}$ directly — **this is exactly the indicator-variable trick from Section 12, previewed here.** Similarly, because the trials are independent, variances add:
$$\text{Var}(X) = \sum_{i=1}^n \text{Var}(X_i) = np(1-p)$$

**Worked example:** flip a fair coin 20 times. Find $E[\text{heads}]$ and $\text{Var}(\text{heads})$.
$E[X] = 20(0.5) = 10$. $\text{Var}(X) = 20(0.5)(0.5) = 5$, so $\text{SD}(X) = \sqrt5\approx 2.24$.

### 6.3 Geometric($p$) — trials until first success
$X$ = number of trials up to and including the first success (support: $1,2,3,\dots$).
$$P(X=k) = (1-p)^{k-1}p$$

**Mean, derived via a self-referential conditioning trick (structurally identical to the amoeba branching-process trick from the conditional probability guide):** condition on the first trial. If it's a success (prob $p$), $X=1$. If it's a failure (prob $1-p$), you've "wasted" one trial and are back to the exact same problem, so the *remaining* number of trials needed has the same distribution as $X$ itself:
$$E[X] = p(1) + (1-p)(1+E[X]) \;\Rightarrow\; E[X] = 1+(1-p)E[X] \;\Rightarrow\; E[X] = \frac1p$$

$$\text{Var}(X) = \frac{1-p}{p^2}$$

**Intuition check:** if $p=1/6$ (rolling a die until you get a 6), $E[X]=6$ — makes sense, you expect to need about 6 rolls. **The "condition on the first step, and if you fail you're back to the start" self-similarity argument is one of the highest-value general problem-solving patterns in this entire guide** — it also solves the amoeba extinction problem, and shows up constantly in random-walk and Markov chain interview questions.

**Memorylessness (a favorite theoretical-depth question):** $P(X>s+t \mid X>s) = P(X>t)$ — i.e., given you've already failed $s$ times, the additional number of trials needed has the *exact same distribution* as starting fresh. This is intuitive for coin flips (the coin has no memory) but is worth being able to prove: $P(X>n)=(1-p)^n$, so $P(X>s+t\mid X>s) = \frac{(1-p)^{s+t}}{(1-p)^s} = (1-p)^t = P(X>t)$. ✓ **The Geometric (and its continuous cousin, the Exponential — Section 7.2) are the *only* memoryless distributions**, which is itself a common interview fact to know and be able to justify.

### 6.4 Negative Binomial($r,p$) — trials until the $r$-th success
Generalizes Geometric: $X$ = number of trials until the $r$-th success.
$$E[X] = \frac{r}{p}, \qquad \text{Var}(X) = \frac{r(1-p)}{p^2}$$
(Immediate from linearity: $X$ is the sum of $r$ i.i.d. Geometric($p$) "gap" variables — one for each successive success — so mean and variance both just multiply by $r$.)

### 6.5 Poisson($\lambda$) — counts of rare events over a fixed interval
Models the number of events in a fixed interval (time, space, etc.) when events occur independently at a constant average rate $\lambda$.
$$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k=0,1,2,\dots \qquad E[X]=\text{Var}(X)=\lambda$$

**Why mean = variance is worth flagging explicitly:** it's an unusually clean and distinctive property — if you're ever handed real count data and asked "does a Poisson model seem reasonable here?", checking whether the sample mean and sample variance are roughly equal is literally the first diagnostic to run (in practice, real data is often *overdispersed*, i.e., variance > mean, which is a common flag that a Negative Binomial model fits better — a genuinely useful applied-statistics fact for a DS/analytics interview).

**Poisson as a limit of Binomial (derive this — it's a classic "show me you understand where distributions come from" question):** if $n\to\infty$, $p\to0$, with $np=\lambda$ held fixed, $\text{Binomial}(n,p) \to \text{Poisson}(\lambda)$. Intuition: split a fixed time interval into a huge number of tiny sub-intervals, each with a tiny probability $p=\lambda/n$ of containing an event — in the limit, that's exactly the Poisson set-up. **This is why Poisson is the standard model for rare, independent events (website hits per second, defects per unit area, calls per minute) — it's literally what Binomial becomes when trials become infinitely frequent and individually infinitesimally likely.**

### 6.6 Hypergeometric($N,K,n$) — sampling without replacement
Draw $n$ items without replacement from a population of $N$, of which $K$ are "successes." $X$ = number of successes drawn.
$$P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}, \qquad E[X] = n\cdot\frac{K}{N}$$

**Compare directly to Binomial — this comparison is a very common interview prompt:** Binomial assumes sampling *with* replacement (or an infinite population), so each draw is independent with fixed success probability $p=K/N$; Hypergeometric handles sampling *without* replacement, where each draw changes the remaining composition, so draws are dependent — yet the **mean formula looks identical** ($n\cdot K/N$ for Hypergeometric vs. $np$ for Binomial, and they agree exactly when $p=K/N$) because of linearity of expectation again: even though individual draws are dependent, each individual draw still has *marginal* probability $K/N$ of being a success, so summing gives the same mean regardless of the dependence structure. **The variance, however, differs** — Hypergeometric variance is smaller than the corresponding Binomial variance (by a "finite population correction factor" $\frac{N-n}{N-1}$), because sampling without replacement reduces randomness compared to independent draws.

---

## 7. Named Continuous Distributions

### 7.1 Uniform($a,b$)
$$f_X(x) = \frac{1}{b-a} \text{ for } x\in[a,b], \qquad E[X]=\frac{a+b}{2}, \quad \text{Var}(X) = \frac{(b-a)^2}{12}$$

**Intuition:** flat density — every value in $[a,b]$ is "equally likely" in the density sense. The mean being the midpoint is obvious by symmetry; don't bother re-deriving it from the integral in an interview, just say "by symmetry."

### 7.2 Exponential($\lambda$)
Models waiting time until an event, with constant hazard rate $\lambda$ (the continuous analog of Geometric).
$$f_X(x) = \lambda e^{-\lambda x},\; x\ge0 \qquad F_X(x) = 1-e^{-\lambda x} \qquad E[X]=\frac1\lambda, \quad \text{Var}(X)=\frac1{\lambda^2}$$

**Memorylessness (continuous version, same proof structure as Geometric):** $P(X>s+t\mid X>s)=P(X>t)$. Practical framing: if a lightbulb's lifetime is Exponential, the fact that it's already survived 5 years tells you *nothing* about how much longer it'll last — its "remaining lifetime" distribution is identical to a brand-new bulb's. This is often *unrealistic* for real components (which wear out, i.e., failure rate increases with age) — a good talking point if an interviewer asks "when would you NOT use an Exponential model?"

**Relationship to Poisson (a very common pairing in interview questions):** if events arrive according to a Poisson process with rate $\lambda$ (i.e., counts in fixed intervals are Poisson($\lambda t$)), then the *waiting time between consecutive events* is Exponential($\lambda$). This Poisson-process ↔ Exponential-interarrival-time duality is worth knowing cold — it's the standard model for server request arrivals, customer arrivals in queueing theory, and radioactive decay.

### 7.3 Normal / Gaussian($\mu,\sigma^2$)
$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)} \qquad E[X]=\mu,\quad \text{Var}(X)=\sigma^2$$

**Why it shows up everywhere (Central Limit Theorem, one sentence version):** the sum (or average) of a large number of i.i.d. random variables, *regardless of their individual distribution*, tends toward a Normal distribution as the number of terms grows — this is why measurement errors, aggregated user behavior metrics, and sums of many small independent effects are so often modeled as Normal in practice, even when nothing about the individual components looks Normal at all. **If an interviewer asks "why do we assume normality so often," the CLT — not "because most things happen to be naturally Normal" — is the correct answer.**

**Standardization:** $Z = \frac{X-\mu}{\sigma} \sim N(0,1)$. Every Normal probability question reduces to a Standard Normal table (or `norm.cdf` in practice) lookup after standardizing — this is the mechanical trick behind essentially all Normal-distribution interview arithmetic.

### 7.4 Gamma($\alpha,\lambda$) — brief mention
Generalizes Exponential: sum of $\alpha$ i.i.d. Exponential($\lambda$) RVs (for integer $\alpha$) — e.g., waiting time until the $\alpha$-th Poisson-process event, the continuous analog of Negative Binomial. $E[X]=\alpha/\lambda$, $\text{Var}(X)=\alpha/\lambda^2$. Rarely the star of an L5 interview question, but worth recognizing the Exponential→Gamma relationship the same way you recognize Geometric→Negative Binomial (same underlying pattern, discrete vs. continuous).

---

## 8. Joint Distributions

### 8.1 Joint PMF / PDF, Marginals
**Discrete:** $p_{X,Y}(x,y) = P(X=x, Y=y)$. Recover a **marginal** by summing out the other variable: $p_X(x) = \sum_y p_{X,Y}(x,y)$.
**Continuous:** $f_{X,Y}(x,y)$ with $\iint f_{X,Y}(x,y)\,dx\,dy=1$; marginal $f_X(x) = \int f_{X,Y}(x,y)\,dy$.

**Intuition for "summing/integrating out":** the marginal distribution of $X$ answers "what's the distribution of $X$ alone, ignoring what $Y$ did?" — you get it by adding up (or integrating over) every possible value $Y$ could have taken, for each fixed $x$. This is literally the continuous/multi-variable version of the Law of Total Probability from the conditional-probability guide (Section 1.3 there) — same idea, different notation.

### 8.2 Conditional Distributions
$$p_{Y\mid X}(y\mid x) = \frac{p_{X,Y}(x,y)}{p_X(x)} \qquad f_{Y\mid X}(y\mid x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

Exactly the same conditioning logic as Section 1.1 of the conditional-probability guide — just applied to densities/masses instead of raw event probabilities.

### 8.3 Independence of Random Variables
$X,Y$ independent iff $p_{X,Y}(x,y)=p_X(x)p_Y(y)$ for all $x,y$ (or the PDF analog) — equivalently, the joint factors into the product of marginals **everywhere**, not just on average.

**Common trap — independence is a much stronger condition than "uncorrelated" (Section 9.3).**

---

## 9. Covariance and Correlation

### 9.1 Covariance
$$\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY]-E[X]E[Y]$$

**Intuition:** positive covariance means $X$ and $Y$ tend to be above/below their means *together*; negative means when one's above its mean, the other tends to be below. The second form ($E[XY]-E[X]E[Y]$) is what you actually compute in practice — same "expand and use linearity" derivation as the variance shortcut in Section 5.1.

### 9.2 Correlation
$$\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1,1]$$

**Why we bother normalizing covariance into correlation:** covariance's magnitude depends on the units/scale of $X$ and $Y$ (e.g., covariance between height-in-cm and weight-in-kg is a different number than height-in-inches and weight-in-lbs, even though the underlying relationship is identical) — correlation is scale-invariant, always between $-1$ and $1$, which is what makes it interpretable and comparable across different variable pairs.

### 9.3 Independence ⟹ Uncorrelated, but NOT the Reverse (a favorite theoretical-depth interview question)
If $X\perp Y$, then $E[XY]=E[X]E[Y]$ (this follows directly from the factored joint density/mass under LOTUS), so $\text{Cov}(X,Y)=0$.

**But the converse is false — classic counterexample to have ready:** let $X\sim\text{Uniform}(-1,1)$ and $Y=X^2$. Clearly $Y$ is a deterministic (maximally dependent!) function of $X$ — knowing $X$ tells you $Y$ exactly. Yet:
$$\text{Cov}(X,Y) = E[XY]-E[X]E[Y] = E[X^3] - E[X]E[X^2]$$
By symmetry of the Uniform($-1,1$) distribution around 0, $E[X]=0$ and $E[X^3]=0$ (odd functions integrate to zero over a symmetric interval), so $\text{Cov}(X,Y) = 0 - 0 = 0$.

**$X$ and $Y$ are uncorrelated despite being perfectly (nonlinearly) dependent.** Correlation only measures *linear* association — this single example is one of the most efficient ways to demonstrate deep understanding in an interview, because it's short, concrete, and directly refutes the common misconception that "zero correlation means no relationship."

---

## 10. Law of Total Expectation & Law of Total Variance

### 10.1 Law of Total Expectation ("Adam's Law")
$$E[X] = E[E[X\mid Y]]$$

**Reading this correctly:** $E[X\mid Y]$ is itself a random variable — a function of $Y$ (different values of $Y$ give different conditional expectations). Taking the expectation of *that* random variable, over the randomness in $Y$, recovers the plain unconditional $E[X]$. This is the RV-flavored version of the Law of Total Probability (conditional-probability guide, Section 1.3): instead of averaging *probabilities* over a partition, you're averaging *conditional expectations* over the distribution of a conditioning RV.

**Worked example:** a store's daily revenue $X$ depends on which of 3 promotions ($Y\in\{1,2,3\}$, each equally likely) is running that day, with $E[X\mid Y=1]=500$, $E[X\mid Y=2]=800$, $E[X\mid Y=3]=650$.
$$E[X] = \frac13(500)+\frac13(800)+\frac13(650) = \frac{1950}{3}=650$$
This is literally LOTP with expectations substituted for probabilities — same mechanical pattern, same partition-and-weight logic.

### 10.2 Law of Total Variance ("Eve's Law")
$$\text{Var}(X) = E[\text{Var}(X\mid Y)] + \text{Var}(E[X\mid Y])$$

**Intuition — this decomposes total variance into two genuinely distinct sources:**
- $E[\text{Var}(X\mid Y)]$ = **"within-group" variance** — how much $X$ varies *inside* each fixed value of $Y$, averaged across groups.
- $\text{Var}(E[X\mid Y])$ = **"between-group" variance** — how much the *group means themselves* vary as $Y$ changes.

**Why this decomposition is genuinely important for applied/DS interviews:** it's the mathematical basis of ANOVA (analysis of variance) and of stratified-sampling variance-reduction techniques. If you're ever asked "why would segmenting/stratifying users reduce variance in an A/B test," Eve's Law is the precise answer: by conditioning on (i.e., stratifying by) a variable $Y$ that explains some of the between-group variance, you isolate and can control for that portion, reducing the *residual* variance you have to deal with statistically.

**Worked example (continuing the store-revenue scenario):** suppose $\text{Var}(X\mid Y=1)=100$, $\text{Var}(X\mid Y=2)=150$, $\text{Var}(X\mid Y=3)=120$, and (from above) $E[X\mid Y=1]=500,\;E[X\mid Y=2]=800,\;E[X\mid Y=3]=650$, with $Y$ uniform over $\{1,2,3\}$.

Within-group term: $E[\text{Var}(X\mid Y)] = \frac13(100+150+120) = \frac{370}{3}\approx123.3$

Between-group term: need $\text{Var}(E[X\mid Y])$, treating $500,800,650$ as a random variable with mean $650$ (computed above):
$$\text{Var}(E[X\mid Y]) = \frac13\left[(500-650)^2+(800-650)^2+(650-650)^2\right] = \frac13[22500+22500+0] = 15{,}000$$

$$\text{Var}(X) = 123.3 + 15{,}000 \approx 15{,}123.3$$

**Notice how completely dominated by the between-group term this is** — which correctly reflects the fact that in this example, which promotion runs matters vastly more to revenue variability than the natural day-to-day noise within a fixed promotion. This is exactly the kind of insight Eve's Law is built to surface.

---

## 11. Standard Error & the Sample Mean

For $n$ i.i.d. observations $X_1,\dots,X_n$ with mean $\mu$ and variance $\sigma^2$, define $\bar X = \frac1n\sum X_i$.

$$E[\bar X] = \mu \qquad \text{Var}(\bar X) = \frac{\sigma^2}{n} \qquad \text{SE}(\bar X) = \frac{\sigma}{\sqrt n}$$

**Derivation, using exactly the tools above:** $E[\bar X]=\frac1n\sum E[X_i]=\frac1n(n\mu)=\mu$ (linearity, no independence needed). $\text{Var}(\bar X) = \frac{1}{n^2}\sum\text{Var}(X_i) = \frac{1}{n^2}(n\sigma^2)=\frac{\sigma^2}{n}$ (variance of a sum needs independence to drop cross-covariance terms — this is exactly why the i.i.d. assumption matters here in a way it didn't for the mean).

**Why $\sqrt n$ specifically (a very common "explain the intuition" follow-up):** variance of the sample mean shrinks proportionally to $1/n$, so standard deviation (the interpretable, same-units quantity) shrinks proportionally to $1/\sqrt n$ — meaning **you need to quadruple your sample size to halve your margin of error.** This single fact is the entire mathematical justification behind "collect more data to reduce noise" in A/B testing and experimentation — genuinely one of the most interview-relevant one-liners in all of applied statistics.

---

## 12. The Indicator Random Variable Trick

**The single highest-leverage technique for solving "expected number of X" problems quickly, with zero need for combinatorics.**

**Recipe:** to find $E[\text{count of something}]$, write the count as a sum of indicator variables $I_1+I_2+\cdots+I_n$, where each $I_k \in \{0,1\}$ flags whether a specific sub-event happened. Then, by linearity of expectation (which needs **no independence assumption**):
$$E\left[\sum_k I_k\right] = \sum_k E[I_k] = \sum_k P(I_k=1)$$

### Worked Example — Matching Problem (hat-check / derangement classic)
$n$ people randomly get back one of $n$ hats (a uniformly random permutation). Find the expected number of people who get their own hat back.

Let $I_k = 1$ if person $k$ gets their own hat, else 0. By symmetry, $P(I_k=1) = 1/n$ for every person (each person's hat is equally likely to be any of the $n$ hats). So:
$$E[\text{matches}] = \sum_{k=1}^n E[I_k] = \sum_{k=1}^n \frac1n = n\cdot\frac1n = 1$$

**Regardless of $n$** — whether 5 people or 5 million — the expected number of correct matches is always exactly 1. This is a genuinely famous result specifically *because* it's so counterintuitive (you'd think more people means more matches) and *because* it's essentially impossible to derive quickly without the indicator trick (direct computation would require the full distribution of matches, involving derangement counts — dramatically messier). **This is one of the best examples to have fully memorized and ready to reproduce from scratch — it demonstrates the indicator trick, linearity of expectation without independence, and a symmetry argument, all in four lines.**

### Worked Example — Expected Number of Fixed Points Isn't the Only Payoff — Variance Too (extension, shows depth)
If additionally asked for $\text{Var}(\text{matches})$: this requires $\text{Cov}(I_j,I_k)$ for $j\ne k$, since the indicators are *not* independent (if person $j$ gets their hat, it changes the probability distribution over what person $k$ gets). $P(I_j=1,I_k=1) = \frac{1}{n(n-1)}$ (both match: fix two people's hats, the rest arbitrary), so $\text{Cov}(I_j,I_k) = \frac{1}{n(n-1)} - \frac1n\cdot\frac1n = \frac{1}{n^2(n-1)}$. Summing $\text{Var}(I_k)$ plus all pairwise covariances eventually gives $\text{Var}(\text{matches})=1$ as well (for $n\ge2$) — a nice bonus fact, but the mean-only version above is the one worth having cold.

---

## 13. Worked Numerical Examples (Mixed Practice)

### Example 1 — Basic expectation & variance from a PMF
$X$ has PMF: $P(X=1)=0.2,\;P(X=2)=0.5,\;P(X=3)=0.3$.
$$E[X] = 1(.2)+2(.5)+3(.3) = .2+1.0+.9 = 2.1$$
$$E[X^2] = 1(.2)+4(.5)+9(.3) = .2+2.0+2.7=4.9$$
$$\text{Var}(X) = 4.9-(2.1)^2 = 4.9-4.41=0.49 \;\Rightarrow\; \text{SD}(X)=0.7$$

### Example 2 — Poisson application
A website receives an average of 3 support tickets per hour (Poisson). What's $P(\text{exactly 5 tickets in an hour})$? What's $P(\text{no tickets in 2 hours})$?

$$P(X=5) = \frac{e^{-3}3^5}{5!} = \frac{e^{-3}\cdot 243}{120} \approx 0.1008$$

For 2 hours, $\lambda_2 = 6$ (rate scales linearly with the interval length — a core property of the Poisson process): $P(X=0) = e^{-6} \approx 0.00248$.

### Example 3 — Linearity of expectation without independence (dice)
Roll a fair die twice. Let $X$ = first roll, $Y$ = second roll. Find $E[X+Y]$ **and** $E[XY]$.

$E[X+Y] = E[X]+E[Y] = 3.5+3.5=7$ — trivial, holds regardless of independence.
$E[XY]$: here the rolls **are** independent, so $E[XY]=E[X]E[Y] = 3.5\times3.5=12.25$. (Flag this explicitly: had $X,Y$ been dependent, you could *not* factor $E[XY]$ this way — you'd need the joint distribution.)

### Example 4 — Continuous RV, CDF/PDF relationship
$X$ has CDF $F_X(x) = 1-e^{-2x}$ for $x\ge0$. Find the PDF, $E[X]$, and $P(1<X<2)$.

Differentiate: $f_X(x) = F_X'(x) = 2e^{-2x}$ — recognize this immediately as $\text{Exponential}(\lambda=2)$, so $E[X]=1/2$ by the known formula (no need to integrate from scratch).
$$P(1<X<2) = F_X(2)-F_X(1) = (1-e^{-4})-(1-e^{-2}) = e^{-2}-e^{-4} \approx 0.1353-0.0183=0.1170$$

### Example 5 — Joint distribution, covariance
$X,Y$ have joint PMF: $P(0,0)=0.1,\;P(0,1)=0.2,\;P(1,0)=0.3,\;P(1,1)=0.4$.

Marginals: $P(X=0)=0.3,\;P(X=1)=0.7$; $P(Y=0)=0.4,\;P(Y=1)=0.6$.
$E[X]=0.7,\;E[Y]=0.6$. $E[XY] = 0(0)(0.1)+0(1)(0.2)+1(0)(0.3)+1(1)(0.4) = 0.4$.
$$\text{Cov}(X,Y) = 0.4-(0.7)(0.6) = 0.4-0.42=-0.02$$

Small negative covariance — $X$ and $Y$ are very close to independent but not exactly (check: if independent, $P(1,1)$ should equal $P(X=1)P(Y=1)=0.7\times0.6=0.42$, but it's actually $0.4$ — the tiny discrepancy is exactly what produces the small negative covariance).

### Example 6 — Adam's Law application (two-stage randomness)
Number of eggs a bird lays is Poisson($\lambda=6$). Each egg independently hatches with probability $p=0.5$. Find the expected number of chicks.

Let $N$ = number of eggs, $X$ = number of chicks. Given $N=n$, $X\mid N=n \sim \text{Binomial}(n, 0.5)$, so $E[X\mid N]=0.5N$.
$$E[X] = E[E[X\mid N]] = E[0.5N] = 0.5\,E[N] = 0.5(6)=3$$

**This "random sum" pattern (a Binomial count with a random number of trials) is extremely common in applied interview questions** — e.g., expected number of purchases when the number of site visitors is itself random, expected number of conversions when the funnel size varies day to day. Adam's Law handles it in one line without ever computing the (messier) unconditional distribution of $X$ directly.

---

## 14. Cheat Sheet — Named Distributions at a Glance

| Distribution | PMF/PDF | $E[X]$ | $\text{Var}(X)$ | Typical use case |
|---|---|---|---|---|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | single yes/no trial |
| Binomial($n,p$) | $\binom nk p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | # successes in $n$ i.i.d. trials |
| Geometric($p$) | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | trials until 1st success |
| Neg. Binomial($r,p$) | — | $r/p$ | $r(1-p)/p^2$ | trials until $r$-th success |
| Poisson($\lambda$) | $e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | rare-event counts per interval |
| Hypergeometric($N,K,n$) | $\binom Kk\binom{N-K}{n-k}/\binom Nn$ | $nK/N$ | (Binomial var. × correction) | sampling without replacement |
| Uniform($a,b$) | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ | "equally likely" continuous range |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | waiting time, memoryless |
| Normal($\mu,\sigma^2$) | Gaussian bell curve | $\mu$ | $\sigma^2$ | CLT limit, measurement error |
| Gamma($\alpha,\lambda$) | — | $\alpha/\lambda$ | $\alpha/\lambda^2$ | waiting time to $\alpha$-th event |

**Core formula recap:**

| Concept | Formula |
|---|---|
| LOTUS | $E[g(X)] = \sum g(x)p(x)$ or $\int g(x)f(x)dx$ |
| Linearity | $E[aX+bY+c]=aE[X]+bE[Y]+c$ (always) |
| Variance shortcut | $\text{Var}(X)=E[X^2]-(E[X])^2$ |
| Var. of a sum | $\text{Var}(X+Y)=\text{Var}(X)+\text{Var}(Y)+2\text{Cov}(X,Y)$ |
| Covariance shortcut | $\text{Cov}(X,Y)=E[XY]-E[X]E[Y]$ |
| Correlation | $\rho=\text{Cov}(X,Y)/(\sigma_X\sigma_Y)$ |
| Adam's Law | $E[X]=E[E[X\mid Y]]$ |
| Eve's Law | $\text{Var}(X)=E[\text{Var}(X\mid Y)]+\text{Var}(E[X\mid Y])$ |
| Sample mean SE | $\text{SE}(\bar X)=\sigma/\sqrt n$ |

---

## 15. Practice Problems

1. $X\sim\text{Binomial}(10, 0.3)$. Find $E[X]$, $\text{Var}(X)$, and $P(X=3)$.
2. A random variable $X$ has $E[X]=4$, $\text{Var}(X)=9$. Find $E[3X-2]$ and $\text{Var}(3X-2)$.
3. Requests to a server arrive as a Poisson process at 10/minute. Find $P(\text{at least one request in the next 6 seconds})$. (Hint: convert rate to a 6-second window, then use the Exponential CDF or Poisson$(X=0)$ complement.)
4. $n$ people at a party each independently, with probability $p$, decide to give the person to their right a high-five (arranged in a circle). Using indicator variables, find the expected number of high-fives.

<details>
<summary>Answers</summary>

**1.** $E[X]=10(0.3)=3$. $\text{Var}(X)=10(0.3)(0.7)=2.1$. $P(X=3)=\binom{10}{3}(0.3)^3(0.7)^7 \approx 0.2668$.

**2.** $E[3X-2]=3(4)-2=10$. $\text{Var}(3X-2)=9\,\text{Var}(X)=9(9)=81$.

**3.** Rate over 6 seconds: $\lambda = 10\times(6/60)=1$. $P(X=0)=e^{-1}\approx0.3679$, so $P(\ge1)=1-e^{-1}\approx0.6321$.

**4.** Let $I_k=1$ if person $k$ high-fives their right neighbor. $P(I_k=1)=p$ for each of the $n$ people (each independently decides). $E[\text{total}] = \sum_{k=1}^n E[I_k] = np$ — same indicator-trick pattern as the hat-matching problem in Section 12, regardless of the circular arrangement or any dependence between neighbors' decisions.

</details>

---

*Next natural topics to build out in this same style, if useful: joint distributions of continuous RVs (bivariate Normal, transformations of RVs), moment generating functions, and the Central Limit Theorem with worked convergence examples.*
