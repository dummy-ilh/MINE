# Phase 2: Stochastic Processes, White Noise, Random Walk (Absolute Zero, continued)

Before this phase, you know: a time series has components (trend/seasonal/cyclic/noise) and you can pull them apart. Now we go one level deeper: **what mathematical "machine" actually produces a time series in the first place?** This is the foundation every future model (AR, MA, ARIMA...) sits on, so we build it very carefully.

---

## 1. What is a "stochastic process"? (New word #1: stochastic)

**Stochastic** just means "random" / "involving chance." That's it — it's a fancier word statisticians use instead of "random," but it means the same thing. So a **stochastic process** = "a random process," i.e., a system that produces random outcomes as time moves forward.

Here's the key idea that trips people up, so let's go slow:

Imagine flipping a coin. One flip gives you one outcome (heads or tails). That's a single random event.

Now imagine flipping a coin **once every day, forever**, and writing down each day's result. Day 1: heads. Day 2: tails. Day 3: heads. Day 4: heads... This *sequence of random outcomes, one for each time point*, is a stochastic process. **A time series that you actually observe (your real data) is just one single realized outcome of an underlying stochastic process** — like one specific run of that coin-flipping experiment.

**Why does this matter practically?** Because when we build a model like "AR(1)" or "ARIMA," we are NOT just describing the specific numbers you happen to have in your spreadsheet. We are proposing a *hypothesis about the random machine that generated those numbers* — and if we get that hypothesis right, we can use it to say something about *tomorrow's* number too (forecasting), and to compute how *uncertain* we should be about that forecast.

**Analogy to cement this:** Think of a stochastic process like a die-rolling machine bolted to a table. Every time you pull the lever (every time step), it spits out a number, following some fixed set of rules (maybe it's a fair 6-sided die, maybe it's weighted, maybe today's roll depends on yesterday's roll). Your actual dataset is the *specific sequence of numbers the machine happened to spit out* the one time you ran it. If you could rewind time and run the machine again, you'd get a *different* sequence of numbers, even though the *underlying machine/rules* are identical. This is the single most important mental model in this entire phase, so re-read this paragraph if anything below feels confusing.

---

## 2. Why do we care about "the machine" and not just "the numbers we have"?

Because with only ONE run of the machine (your one dataset), you cannot directly compute things like "the true average" or "the true variance" of the machine by just repeating an experiment many times (you don't have multiple parallel universes' worth of data). Instead, time series analysis has to develop clever tricks to estimate the machine's rules from a **single sequence**. This single fact — *we only get to see one realization* — is why time series statistics needs special tools instead of ordinary statistics.

**New word #2: Ensemble.** If you imagine running the same random machine many times (hypothetically), each run produces a different sequence. The *collection* of all these hypothetical sequences is called an **ensemble**. Ordinary statistics usually assumes you have access to many independent samples (like an ensemble). Time series statistics usually only has **one single sequence** (one path through time) and has to work with that.

---

## 3. Stationarity — a preview (fully covered in Phase 4, but you need the concept now)

For our "machine" idea to be useful for prediction, we generally need the *rules of the machine* to not change over time — e.g., the machine isn't secretly reprogramming itself to behave totally differently in year 5 vs year 1. When the statistical rules/behavior of the process stay the same across time, we call the process **stationary** (we'll define this rigorously with formulas in Phase 4). For now, just hold onto the intuition: **stationary = the "personality" of the random machine doesn't change over time.** Most of the classical models we'll build (AR, MA, ARIMA) require this assumption to work correctly, which is why so much energy in time series analysis goes into checking for it and fixing it when it's violated.

---

## 4. White Noise: the simplest possible "machine"

Let's build the absolute simplest stochastic process there is. This is the "atom" that everything else is constructed from.

**Definition, built up piece by piece:**

A sequence $\varepsilon_1, \varepsilon_2, \varepsilon_3, \dots$ is called **white noise** if it satisfies three properties. Let's take each one slowly.

**Property 1 — Zero mean:** $E[\varepsilon_t] = 0$ for every $t$.
Plain English: "$E[\cdot]$" is the **expectation operator** — a new piece of notation meaning "the average value you'd get if you could rerun the machine infinitely many times" (remember the ensemble idea from section 2). Zero mean just says: the machine isn't biased upward or downward — on average, across all hypothetical reruns, it produces 0. It might spit out +3 today and −2 tomorrow, but averaged over infinite reruns, it centers on 0.

**Property 2 — Constant variance:** $\text{Var}(\varepsilon_t) = \sigma^2$ for every $t$ (same $\sigma^2$ at every single time point).
Plain English: **Variance** measures how spread-out / how "wild" the random outcomes are around the mean. $\sigma^2$ (sigma-squared) is just the symbol for "the variance value." "Constant variance" means the machine is equally wild/unpredictable at every time step — it doesn't get calmer in some periods and wilder in others. (When this constant-variance property is violated, it's called **heteroskedasticity** — a word you'll meet again when we cover GARCH models much later; for now just recognize it as "non-constant variance.")

**Property 3 — No autocorrelation:** $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ for any two different time points $t \neq s$.
Plain English: **Covariance** measures whether two random quantities move together (e.g., "when one is high, is the other also usually high?"). Here we're asking about the covariance between the noise value *today* ($\varepsilon_t$) and the noise value at *any other different day* ($\varepsilon_s$). Zero covariance means: **knowing today's value tells you absolutely nothing about any other day's value.** Every single time step is a completely fresh, independent random draw, with zero memory of the past.

**Putting it together in plain English:** White noise is a sequence of random numbers that (1) average out to zero, (2) are equally unpredictable/wild at every time step, and (3) have absolutely no relationship to each other — today's value gives you zero information about tomorrow's value.

**Why call it "white" noise?** This name comes from physics/signal processing — "white" light contains all colors/frequencies equally, with no single frequency dominating. White noise in time series similarly contains "all frequencies equally" in a technical sense we'll cover in the spectral analysis phase much later — for now, just treat "white noise" as a name, not something to derive.

**The most common concrete example used everywhere:** Gaussian white noise, meaning each $\varepsilon_t$ is independently drawn from a Normal (bell-curve) distribution with mean 0 and variance $\sigma^2$, written $\varepsilon_t \sim N(0, \sigma^2)$. This "∼" symbol just means "is distributed as" / "is drawn from."

**Why white noise matters so much:** It is the "null model" of time series — the benchmark of *pure unpredictability*. Every model we build later (AR, MA, ARIMA) is really just a recipe for describing HOW today's observed value is built out of current and past white noise terms, plus some structure (memory/dependency) layered on top. If you ever fit a time series model and check the leftover residuals, and they look like white noise — that's the sign you did a good job: you've successfully extracted *all* the predictable structure, and only pure randomness remains. This is why the Ljung-Box test we mentioned in Phase 1 (and will formalize soon) exists: it's literally a formal test for "does this residual sequence look like white noise or not?"

### Numerical mini-example of white noise
Suppose $\sigma^2 = 4$ (so standard deviation $\sigma = 2$). A sample path (one realization) might look like:
```
t=1: +1.3
t=2: -2.7
t=3: +0.4
t=4: +3.1
t=5: -1.8
```
Notice: no visible pattern, values bounce unpredictably around 0, and knowing $t=1$'s value (+1.3) gives you zero clue about what $t=2$ will be. That's the signature of white noise. Contrast this with real data (like the ice cream example from Phase 1) where you COULD predict the next value reasonably well — that predictability is exactly what separates "interesting time series with structure" from "pure noise."

---

## 5. The Random Walk: the second-simplest "machine," and your first taste of non-stationarity

Now we build something one tiny step more complex than white noise — and this next process is *extremely* important because stock prices, exchange rates, and many real-world series behave remarkably like it.

**Definition, built step by step:**

Start with white noise $\varepsilon_1, \varepsilon_2, \varepsilon_3, \dots$ (as defined above — zero mean, constant variance, no memory). Now define a NEW sequence $x_t$ by this rule:

$$
x_t = x_{t-1} + \varepsilon_t
$$

**Plain English reading of this formula:** "Today's value equals yesterday's value, PLUS a fresh random white-noise shock." That's it. That's the entire random walk. Each new step, you take wherever you currently are, and you nudge it by a random amount (which could be positive or negative, per white noise's zero-mean property).

**Analogy:** Imagine you're standing at position 0 on a number line. Every minute, you flip a coin: heads you step +1, tails you step −1. Your position after many minutes is a random walk. You have no "pull back toward zero" — wherever the random steps happen to take you, you *stay* there, and the next step is relative to your *current* position, not relative to the starting point.

**Unrolling the recursion (this is important, let's do it slowly):**
$x_1 = x_0 + \varepsilon_1$
$x_2 = x_1 + \varepsilon_2 = (x_0 + \varepsilon_1) + \varepsilon_2 = x_0 + \varepsilon_1 + \varepsilon_2$
$x_3 = x_2 + \varepsilon_3 = x_0 + \varepsilon_1 + \varepsilon_2 + \varepsilon_3$

Do you see the pattern? In general:
$$
x_t = x_0 + \varepsilon_1 + \varepsilon_2 + \dots + \varepsilon_t = x_0 + \sum_{i=1}^{t} \varepsilon_i
$$
(The $\sum$ symbol just means "add up all of these terms" — a compact way of writing a long addition.)

**Plain English: a random walk's current value is simply the STARTING point plus the ACCUMULATED SUM of every single random shock that has ever occurred.** It never forgets a single shock — every random nudge from the past is permanently baked into the current position forever. This is fundamentally different from white noise, where each value is a fresh, forgetful, independent draw.

### 5.1 Why is this NOT stationary? (Your first concrete non-stationarity proof)

Let's compute the variance of $x_t$ using the unrolled formula above. Recall: variance of a SUM of *independent* random variables is the SUM of their individual variances (this is a basic probability rule — variances add for independent terms, assuming $x_0$ is a fixed constant, e.g. 0).

$$
\text{Var}(x_t) = \text{Var}(\varepsilon_1) + \text{Var}(\varepsilon_2) + \dots + \text{Var}(\varepsilon_t) = \underbrace{\sigma^2 + \sigma^2 + \dots + \sigma^2}_{t \text{ times}} = t \cdot \sigma^2
$$

**Read this result out loud: the variance of a random walk GROWS as time goes on — specifically, it grows linearly with $t$.** This is a huge deal. Recall from Phase 2's stationarity preview (section 3): stationary means "the machine's statistical personality doesn't change over time." But here, the variance at $t=100$ is literally 100 times bigger than the variance at $t=1$. The "spread" of possible outcomes keeps widening forever. **This proves, directly from the formula, that a random walk is NOT stationary.**

**Intuitive translation:** the farther into the future you go, the less certain you can be about where the random walk will be — the "cone of uncertainty" keeps widening. If you've ever seen a stock price forecast chart with a widening shaded uncertainty region around the future prediction — that widening cone is a direct visual consequence of this exact variance formula.

### 5.2 Random Walk with Drift (one more small addition)

We can add a constant "push" in one direction at every step:
$$
x_t = x_{t-1} + \delta + \varepsilon_t
$$
Here $\delta$ (delta) is a fixed constant called the **drift** — plain English: "on top of the random noise, there's also a steady, predictable nudge of size $\delta$ in the same direction every single step." Unrolling this the same way as before:
$$
x_t = x_0 + t\cdot\delta + \sum_{i=1}^t \varepsilon_i
$$
Notice the $t \cdot \delta$ term — this is a straight-line trend embedded directly into the random walk. **This is precisely why stock prices (which tend to drift upward over decades due to overall economic growth) are often modeled as a random walk with drift**: a steady upward push, plus accumulating unpredictable noise on top.

### 5.3 A critical interview-relevant nuance: Random Walk vs. AR(1) with a "pull back"

Compare the random walk $x_t = x_{t-1} + \varepsilon_t$ (coefficient on $x_{t-1}$ is exactly 1) against a slightly different rule:
$$
x_t = \phi \, x_{t-1} + \varepsilon_t \quad \text{where } |\phi| < 1
$$
(This is a sneak preview of the AR(1) model from Phase 6 — don't worry about mastering it now, just notice the difference.) When $\phi$ is less than 1 in absolute value, the process gets pulled back toward zero over time — a large deviation today gets shrunk down at the next step (multiplied by $\phi<1$), so shocks don't accumulate forever; they fade out. This "pulled back toward a stable center" behavior IS stationary. The random walk is the exact **boundary/special case** where $\phi = 1$ — the pull-back completely disappears, and every shock survives forever. This special case ($\phi=1$) has a specific name you'll hear constantly: a **unit root** (because in the underlying algebra, the root of the process's characteristic equation sits exactly at 1). Testing whether real data has $\phi$ exactly equal to 1 (a unit root/random walk, non-stationary) versus $\phi$ meaningfully less than 1 (stationary, mean-reverting) is EXACTLY what the Augmented Dickey-Fuller (ADF) test from Phase 4 is built to detect. You now understand, from first principles, *why* that test exists and what question it's actually asking.

---

## 6. Numerical worked example: simulate a tiny random walk by hand

Let's use white noise shocks: $\varepsilon_1=+2, \varepsilon_2=-1, \varepsilon_3=+3, \varepsilon_4=-2, \varepsilon_5=+1$, starting at $x_0 = 0$.

| $t$ | $\varepsilon_t$ | $x_t = x_{t-1}+\varepsilon_t$ | Running sum check |
|---|---|---|---|
| 0 | — | 0 | — |
| 1 | +2 | 0+2 = **2** | 2 |
| 2 | −1 | 2−1 = **1** | 2−1=1 ✓ |
| 3 | +3 | 1+3 = **4** | 2−1+3=4 ✓ |
| 4 | −2 | 4−2 = **2** | 2−1+3−2=2 ✓ |
| 5 | +1 | 2+1 = **3** | 2−1+3−2+1=3 ✓ |

Notice: even though every single $\varepsilon_t$ has zero mean (no bias up or down), the path $x_t$ wanders around and does NOT return to 0 — it drifts to wherever the accumulated sum happens to land. This is the key visual/numeric signature of a random walk: it looks like it's "trending" over any short window, even though there's no real trend rule built in — it's an illusion created purely by accumulated randomness. **This is a famous, very real trap**: humans (and even careless analysts) often look at a random walk and think they see a meaningful trend or pattern, when it's actually just noise that has accumulated. Recognizing this trap is a genuinely important, real interview topic (related to "spurious trend perception").

Now compute the variance check using our formula $\text{Var}(x_t) = t\sigma^2$. Our $\varepsilon$ values here were illustrative fixed numbers, not resampled from a distribution, but if these had come from white noise with $\sigma^2=4$ say, then:
$\text{Var}(x_1) = 1(4) = 4$
$\text{Var}(x_5) = 5(4) = 20$

Five times more spread by $t=5$ than at $t=1$ — exactly matching the "widening cone of uncertainty" intuition from section 5.1.

---

## 7. IID vs. Martingale Difference Sequence (a precise nuance, briefly — flag for later depth)

You'll sometimes see white noise defined slightly differently in different textbooks. **IID** stands for "independent and identically distributed" — meaning every $\varepsilon_t$ is not just uncorrelated with every other one (property 3 above), but *fully statistically independent* (a much stronger condition — uncorrelated only rules out *linear* relationships, while independent rules out ANY relationship, linear or not) and drawn from the exact same probability distribution every time. A **martingale difference sequence (MDS)** is a slightly weaker, more general condition than IID, defined as $E[\varepsilon_t \mid \text{all past information}] = 0$ — meaning "given everything you knew up to yesterday, your best guess/expectation for today's shock is still zero." This is subtle and mostly a theoretical refinement; the practical takeaway for now: **"white noise" in applied work almost always effectively means IID Gaussian noise, and that's the version to use in your head for now.** We flag the MDS distinction here only because it occasionally appears in rigorous derivations later (state space models, GARCH) and you shouldn't be surprised by the term.

---

## 8. Ergodicity — briefly, and why we mention it at all

**New word: Ergodic.** Recall from section 2: we normally only observe ONE realization (one run) of the random machine, not an ensemble of many reruns. A process is called **ergodic** if, roughly speaking, **time averages computed from that one single long realization converge to the same answer as ensemble averages would** (averaging across many hypothetical reruns). Plain English: ergodicity is the mathematical justification that lets us say "it's OK to estimate the process's true mean/variance just by averaging over time within our one dataset, instead of needing many parallel datasets." Without this property (or something like it), nothing in applied time series analysis would be statistically justified — we'd have no way to estimate anything from a single observed sequence. You don't need to derive or test this formally at your current stage — just understand it's the quiet assumption that makes everything else in this course legitimate.

---

## 9. Quick self-check questions

1. If $\varepsilon_t$ is white noise with $\sigma^2 = 9$, what is $\text{Var}(\varepsilon_5)$? What is $\text{Cov}(\varepsilon_5, \varepsilon_{12})$?
   *(Answer: Var = 9 for every single $t$ by definition. Cov = 0, since white noise has zero covariance between any two different time points.)*
2. A random walk starts at $x_0=0$ with white noise variance $\sigma^2=1$. What is $\text{Var}(x_{10})$?
   *(Answer: $10 \times 1 = 10$, using $\text{Var}(x_t)=t\sigma^2$.)*
3. True or false: in a random walk, if you know today's value is unusually high, that tells you nothing useful about tomorrow's value.
   *(Answer: FALSE — this is the opposite of white noise. In a random walk, today's value is exactly the starting point for tomorrow, i.e., $x_{t+1}=x_t+\varepsilon_{t+1}$ — today's value is fully "remembered" and carried forward. It's the white noise SHOCKS that have no memory of each other, not the random walk level itself.)*
4. Why can't we just directly measure the "true" mean and variance of a stochastic process the way we would for ordinary independent data (e.g., heights of 1000 people)?
   *(Answer: because we typically only observe ONE realization/path through time, not many independent reruns of the same random machine — this is the ensemble vs. single-realization issue from section 2, resolved in practice by relying on stationarity + ergodicity.)*

---

## What's next
Phase 3 builds the tool that lets us actually *detect and measure* the "memory" a process has — the **autocorrelation function (ACF)** and **partial autocorrelation function (PACF)**. This is the single most-used diagnostic tool in classical time series and directly determines how we'll identify AR vs MA models in Phase 6. We'll derive the formulas fully and compute ACF by hand on a small dataset.

Say "next" for Phase 3, or ask for more drilling on white noise / random walk first.
