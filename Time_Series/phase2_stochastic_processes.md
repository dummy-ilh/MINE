# Phase 2: Stochastic Processes, White Noise, Random Walk

Before this phase, you know: a time series has components (trend/seasonal/cyclic/noise) and you can pull them apart. Now we go one level deeper: what mathematical "machine" actually produces a time series in the first place? This is the foundation every future model (AR, MA, ARIMA...) sits on.

---

## 1. What is a stochastic process?

**Stochastic** just means "random." A **stochastic process** is a system that produces random outcomes as time moves forward.

Picture a vending machine that spits out a random number every day. You can't predict exactly what it'll give you, but you know its general personality — maybe it's a fair coin, maybe it's biased, maybe today's number depends a little on yesterday's.

**Your actual dataset — the numbers in your spreadsheet — is just one single day-by-day record of that machine running once.** If you could rewind time and let the exact same machine run again, you'd get a completely different sequence of numbers, even though the machine itself (its rules, its personality) never changed.

Stochastic process = the machine, not the numbers it happened to produce.

**Why this matters practically:** when you fit a model like AR(1) or ARIMA later, you're not just curve-fitting the specific numbers in front of you. You're making a hypothesis about what kind of machine produced them. Get that hypothesis right, and you can say something meaningful about what the machine will probably produce tomorrow, and how confident to be about that guess.

**Analogy:** daily weather is a stochastic process. There are underlying physical rules (a "machine"), but the exact temperature each day has randomness baked in. This July's actual daily temperatures are one realization of "what July weather could have looked like." A different July, same climate, would give different numbers.

---

## 2. Why we can't just use ordinary statistics

Ordinary statistics usually assumes you can collect many independent samples — measure 1,000 different people's heights, for example. You have 1,000 separate "runs" of the height-generating process.

With time series, you almost never get that. You get **one single run of the machine** — one actual history, once. You can't rewind 2020 and see what a slightly different year would have looked like.

**Ensemble vs. single realization:** the ensemble is the hypothetical collection of every possible run the machine could have produced. You only ever observe one path through that collection. Nearly every technique in time series analysis exists to extract reliable estimates from that one path — which is why time series needs its own toolkit instead of borrowing directly from intro stats.

---

## 3. Stationarity (preview)

A process is **stationary** if the machine's personality doesn't change over time — same average behavior, same "wildness," in year 1 as in year 20 (formalized rigorously later). Most classical models assume the machine isn't secretly changing its own rules mid-run, which is why so much effort in time series analysis goes into checking that assumption and fixing it when it's violated.

---

## 4. White noise — the "pure static" machine

The simplest possible machine: pure static, like TV noise or radio hiss between stations. A wall of sound with zero pattern.

Three properties define it, in plain English:

- **Centers on zero.** Not biased up or down — averaged over infinite reruns, it centers on 0.
- **Equally wild at every moment.** Not calmer on some days and wilder on others — the amount of unpredictability is constant throughout.
- **No memory whatsoever.** Today's value tells you nothing about tomorrow's. Every moment is a fresh, independent random draw.

**Everyday picture:** flip a fair coin every second, write +1 for heads and −1 for tails. There's no streak logic, no "due for tails" — each flip is oblivious to every flip before it. That's white noise.

### Formal definition (for reference)

A sequence $\varepsilon_1, \varepsilon_2, \varepsilon_3, \dots$ is white noise if:

1. **Zero mean:** $E[\varepsilon_t] = 0$ for every $t$. ($E[\cdot]$ = the average across infinite hypothetical reruns.)
2. **Constant variance:** $\text{Var}(\varepsilon_t) = \sigma^2$ for every $t$. (Variance = how spread out / wild the outcomes are.)
3. **No autocorrelation:** $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ for any $t \neq s$. (Covariance = whether two quantities move together.)

The most common concrete version is **Gaussian white noise**: each $\varepsilon_t$ independently drawn from a Normal distribution, $\varepsilon_t \sim N(0, \sigma^2)$.

**Why it matters:** every model built later (AR, MA, ARIMA) is really a recipe for saying "today's value is built from some structure/memory, plus a leftover splash of pure white noise." When you fit a good model and check the residuals, you're hoping they look exactly like static — no pattern, no memory left. If residuals still show structure, the model missed something real. This is exactly what the Ljung-Box test checks: do the leftovers look like static, or is there still structure hiding in there?

**Numeric feel:** a white-noise path might look like `+1.3, -2.7, +0.4, +3.1, -1.8` — bounces around zero, no visible rhythm, and knowing day 1 was +1.3 tells you nothing about day 2.

---

## 5. The random walk — "wherever you land, that's home now"

Add exactly one ingredient of complexity to white noise and you get something that behaves like real stock prices.

**Intuition — a drunk person leaving a bar:** each step, they stumble a random direction, purely by chance. Critically, each new stumble starts from wherever they currently are, not from the bar door. If a string of stumbles has carried them 20 feet down the street, the next stumble is relative to that spot, not pulled back toward the bar.

$$
x_t = x_{t-1} + \varepsilon_t
$$

Today's value = yesterday's value + a fresh random nudge.

Compare directly to white noise: white noise has no memory — every value forgets the past instantly. A random walk has total memory — it permanently carries forward every nudge it has ever received.

### Unrolling the recursion

$$
x_1 = x_0 + \varepsilon_1
$$
$$
x_2 = x_1 + \varepsilon_2 = x_0 + \varepsilon_1 + \varepsilon_2
$$
$$
x_3 = x_2 + \varepsilon_3 = x_0 + \varepsilon_1 + \varepsilon_2 + \varepsilon_3
$$

In general:

$$
x_t = x_0 + \sum_{i=1}^{t} \varepsilon_i
$$

A random walk's current value is simply the starting point plus the accumulated sum of every random shock that has ever occurred. It never forgets a single shock.

### 5.1 Why this is not stationary

Variance of a sum of independent random variables = sum of their variances:

$$
\text{Var}(x_t) = \underbrace{\sigma^2 + \sigma^2 + \dots + \sigma^2}_{t \text{ times}} = t \cdot \sigma^2
$$

The variance grows linearly with $t$. This is the mathematical version of the drunk-walk intuition: since every random nudge gets permanently absorbed into the current position, the range of places you could plausibly be keeps expanding the longer the walk runs. After 1 step you're probably close to the start. After 100 steps you could be almost anywhere within a much wider radius.

This is exactly why stock-price forecasts show a widening cone/funnel around the future prediction — a random walk's uncertainty compounding step after step. And it's why a random walk is **not stationary**: unlike white noise's constant wildness, the variance keeps growing the further out you look.

### 5.2 Random walk with drift

$$
x_t = x_{t-1} + \delta + \varepsilon_t
$$

Imagine the sidewalk itself is tilted, gently sloping downhill in one direction. Every step, on top of the random stumble, there's also a small, steady, predictable push — the **drift**, $\delta$. Unrolled:

$$
x_t = x_0 + t\cdot\delta + \sum_{i=1}^t \varepsilon_i
$$

The $t \cdot \delta$ term is a straight-line trend embedded directly into the random walk. This is why stock prices — random short-term, trending upward over decades because the economy grows — are often modeled as random walk with drift: accumulated randomness riding on top of a steady long-run push.

### 5.3 Random walk vs. AR(1): the unit root idea

Compare:

$$
x_t = x_{t-1} + \varepsilon_t \quad \text{(random walk)}
$$
$$
x_t = \phi\, x_{t-1} + \varepsilon_t, \quad |\phi| < 1 \quad \text{(a preview of AR(1))}
$$

In the second version, any big deviation today automatically shrinks a bit at the next step, like a rubber band gently pulling the process back toward a stable center. Shocks fade out instead of accumulating forever — this "pulled back toward center" behavior is stationary.

The random walk is the exact edge case where that pull-back fraction ($\phi$) hits exactly 1 — the rubber band goes slack, nothing ever pulls the process back. This edge case is called a **unit root**. Testing whether real data has $\phi$ exactly equal to 1 (non-stationary random walk) versus $\phi$ meaningfully less than 1 (stationary, mean-reverting) is exactly what the **Augmented Dickey-Fuller (ADF) test** is built to detect.

---

## 6. Hand-worked example

Shocks: $\varepsilon_1=+2, \varepsilon_2=-1, \varepsilon_3=+3, \varepsilon_4=-2, \varepsilon_5=+1$, starting at $x_0 = 0$.

| $t$ | $\varepsilon_t$ | $x_t = x_{t-1}+\varepsilon_t$ |
|---|---|---|
| 0 | — | 0 |
| 1 | +2 | 2 |
| 2 | −1 | 1 |
| 3 | +3 | 4 |
| 4 | −2 | 2 |
| 5 | +1 | 3 |

Every single nudge averaged zero on its own, yet the path never returns to 0 — it drifts to wherever the accumulated randomness happens to land.

**The spurious trend trap:** stare at a short chunk of a pure random walk and it will look like it's trending, even though there's no trend rule anywhere in the machine. This is an optical illusion created purely by accumulated noise. Recognizing this is a genuinely important real-world skill — it's the reason people often think they've spotted a pattern in a stock chart that's actually just noise piling up.

**Variance check** (using $\sigma^2=4$ hypothetically): $\text{Var}(x_1) = 1 \times 4 = 4$, $\text{Var}(x_5) = 5 \times 4 = 20$ — five times more spread by $t=5$ than at $t=1$, matching the widening-cone intuition.

---

## 7. Two footnotes for later

- **IID vs. no memory:** "no correlation between days" (white noise property 3) is technically weaker than full statistical independence (IID) — uncorrelated only rules out *linear* relationships, while independent rules out any relationship. A **martingale difference sequence** is an even weaker, more general condition: $E[\varepsilon_t \mid \text{past}] = 0$. In practice, "white noise" almost always effectively means IID Gaussian noise — that's the version to keep in mind for now.
- **Ergodicity:** the quiet assumption that lets us estimate a process's true mean/variance just by averaging over time within one dataset, instead of needing many parallel reruns. Without it, nothing in applied time series analysis would be statistically justified.

---

## 8. Self-check questions

1. Every day you flip a fair coin: +1 for heads, −1 for tails, no other rule. White noise or random walk?
   *White noise — no memory, each day is a fresh independent draw.*

2. You track your running total of wins/losses over time (add +1 or −1 to yesterday's total). White noise or random walk?
   *Random walk — same underlying coin flips, but now each day inherits and builds on everything before it.*

3. If $\varepsilon_t$ is white noise with $\sigma^2 = 9$, what is $\text{Var}(\varepsilon_5)$? What is $\text{Cov}(\varepsilon_5, \varepsilon_{12})$?
   *Var = 9 for every $t$ by definition. Cov = 0, since white noise has zero covariance between any two different time points.*

4. A random walk starts at $x_0=0$ with white noise variance $\sigma^2=1$. What is $\text{Var}(x_{10})$?
   *$10 \times 1 = 10$, using $\text{Var}(x_t)=t\sigma^2$.*

5. A random walk has been running for a while and is currently far above zero. Does that tell you anything about tomorrow's value?
   *Yes — tomorrow starts from wherever it is today, not from zero. Today's level is fully remembered. It's only the individual nudges that have no memory of each other, not the walk's position.*

6. Why can't you measure the "true" mean and variance of a stochastic process the way you'd measure average height across 1,000 people?
   *Because you typically observe only one realization — one path through time — not many independent reruns of the same machine. Stationarity + ergodicity are what make time-averaging within that one path a legitimate substitute.*

---

## What's next

Phase 3 builds the autocorrelation function (ACF) and partial autocorrelation function (PACF) — the tools that measure how much "memory" a process has. This directly determines how AR vs. MA models are identified in Phase 6.
