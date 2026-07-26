# Phase 1: What Is a Time Series? (Absolute Zero Start)

We go slow. Every new word gets defined the first time it shows up. No formula appears before you understand in plain English what it's trying to say.

---

## 1. What "data over time" actually means

Normal data (the kind you see in a spreadsheet of, say, house prices) usually looks like this: each **row** is a different house, and the rows don't care about each other. House #5 doesn't "come after" House #3 in any meaningful way. You could shuffle the rows and nothing would break.

A **time series** is different. It's a sequence of numbers where **order matters**, because each number is tied to a specific point in time — and today's number is connected to yesterday's number.

Example: Apple's daily app downloads.
```
Day 1: 120,000
Day 2: 121,500
Day 3: 119,800
Day 4: 135,000  <- big jump, why?
Day 5: 134,200
```
If you shuffled these into a random order, you'd destroy real information — because Day 4's jump might be explained by "Day 4 was a Friday" or "a new iPhone launched that day." The **position in time carries meaning**. That's the entire reason time series needs its own set of tools — regular statistics assumes rows are independent of each other, and here they're not.

**Formal definition** (now that you have the intuition): A time series is a sequence of observations $x_1, x_2, x_3, \dots, x_T$ recorded at ordered, usually equally-spaced, points in time (e.g., every day, every hour, every quarter). $T$ is just "the total number of time points we have."

---

## 2. Why can't we just use normal statistics on this?

Normal statistics (like a simple average) assumes every data point is drawn independently — like flipping a coin repeatedly. Each flip doesn't affect the next flip.

But in a time series, **today depends on yesterday**. If app downloads were high yesterday (say, because of a big marketing push), they're probably still elevated today — the effect hasn't fully worn off. This dependency between neighboring points is called **autocorrelation** ("auto" = self, "correlation" = relationship — so literally "a series correlated with itself, at a lag/delay"). We will spend an entire future phase just on this concept. For now, just remember: **the defining feature of time series is that neighboring points are related, and we must model that relationship rather than ignore it.**

---

## 3. The four things hiding inside every time series ("components")

When you look at a real time series — sales, downloads, temperature, stock prices — what you're actually looking at is usually a **mixture of several different patterns added or multiplied together**. Untangling these patterns is called **decomposition**. Let's meet each one.

### 3.1 Trend
**Plain English:** the long-term direction the series is heading — is it generally going up, going down, or staying flat, if you ignore the day-to-day wiggles?

Example: iPhone sales per year have generally trended upward for a decade (with some flattening recently). If you squint and ignore the year-to-year bumps, there's a slow, smooth movement — that's trend.

Trend does **not** have to be a straight line. It can curve (accelerate, decelerate). The key property is: it's the *slow-moving, long-run* part of the signal.

### 3.2 Seasonality
**Plain English:** a pattern that repeats at a **fixed, known frequency** tied to the calendar or clock.

Example: Retail sales spike every December (holiday shopping) — every single year, same month. Restaurant traffic is higher every Friday/Saturday — every single week. Electricity usage is higher every afternoon — every single day.

The key defining feature of seasonality: **you know exactly when it will happen again, because it's tied to a calendar unit** (hour of day, day of week, month of year). If it repeats every 12 months, we say the series has a **seasonal period of 12** (when data is monthly) or a **seasonal period of 7** if data is daily and repeats weekly.

### 3.3 Cyclicality (often confused with seasonality — this is a classic interview trap)
**Plain English:** a rise-and-fall pattern that does **NOT** have a fixed, known length — unlike seasonality.

Example: Economic recessions and booms. They happen repeatedly, but not every exactly-8-years — sometimes 6 years apart, sometimes 11. You can't set your calendar by it.

**The interview-critical distinction:**
| | Seasonality | Cycle |
|---|---|---|
| Fixed length? | Yes, always same (e.g., every 12 months) | No, varies |
| Cause | Calendar/clock (holidays, weekdays, daylight) | Economic/business conditions |
| Predictable timing? | Yes | Only roughly |

### 3.4 Irregular / Noise / Residual
**Plain English:** whatever is left over after you've removed trend, seasonality, and cycle. This is the random, unpredictable wiggle — measurement error, one-off events, pure randomness.

If you've correctly extracted trend + seasonality + cycle, what remains should look like **structureless static** — no visible pattern. If you can still see a pattern in what's "left over," it means your decomposition missed something.

---

## 4. How these four pieces combine: Additive vs Multiplicative

Now here's the "how to find how it affects it, how to alter it" part you asked about — this is exactly what decomposition is for.

### 4.1 The Additive model
**Plain English idea:** the components just **add up** to make the final observed number. The size of the seasonal swing stays the **same absolute amount** no matter how big the trend has grown.

$$
x_t = T_t + S_t + C_t + I_t
$$

Where (defining every symbol, nothing assumed):
- $x_t$ = the actual observed value at time $t$ (what you actually measured)
- $T_t$ = the trend value at time $t$
- $S_t$ = the seasonal effect at time $t$ (can be positive or negative — e.g., "+500 units in December", "−300 units in February")
- $C_t$ = the cyclical effect at time $t$
- $I_t$ = the irregular/noise term at time $t$

**When to use additive:** when the seasonal swings look like they stay a *constant size* even as the overall level of the series grows. Example: a shop's sales grow from 1,000/month to 2,000/month over 3 years, but December is *always about 200 units higher* than the trend, whether the trend is at 1,000 or 2,000.

### 4.2 The Multiplicative model
**Plain English idea:** the components **multiply** together. The seasonal swing grows *proportionally* as the trend grows.

$$
x_t = T_t \times S_t \times C_t \times I_t
$$

Here $S_t$ is no longer "+500 units," it's a **ratio**, like 1.20 (meaning "20% above the trend level this month") or 0.85 (meaning "15% below trend this month").

**When to use multiplicative:** when the seasonal swing grows *in proportion to* the trend. Example: December sales are always **20% above** the current trend level — so when the trend was 1,000/month, December added +200; but now the trend is 2,000/month, December adds +400. The *percentage* effect is constant, not the absolute number.

**How you actually tell which one applies to real data (the practical "how to find" part):** Plot the series. If the seasonal wiggles look like they're getting *bigger in absolute size* as the series level rises (like a widening megaphone shape), it's multiplicative. If the wiggles stay roughly the same absolute height throughout, it's additive.

### 4.3 A trick that connects the two
If you take the **logarithm** of a multiplicative series, it turns into an additive one:
$$
\log(x_t) = \log(T_t) + \log(S_t) + \log(C_t) + \log(I_t)
$$
This is *why* you'll often see people apply a log transform before modeling — it converts a "megaphone-shaped" multiplicative problem into a simpler additive one. This is one of the most common practical tricks in time series and interviewers love asking "why would you log-transform a series before modeling?" — now you know the real answer: **to convert proportional (multiplicative) seasonal/trend effects into additive ones**, because additive models are simpler to estimate and reason about.

---

## 5. A full numerical worked example (by hand, no code)

Let's use a tiny toy dataset: quarterly ice cream shop revenue (in $1000s) over 2 years (8 quarters). Ice cream sales are strongly seasonal (high in summer quarters).

| Quarter | Value ($1000s) |
|---|---|
| Q1 2023 | 10 |
| Q2 2023 | 18 |
| Q3 2023 | 22 |
| Q4 2023 | 12 |
| Q1 2024 | 14 |
| Q2 2024 | 22 |
| Q3 2024 | 26 |
| Q4 2024 | 16 |

**Step 1 — Estimate the trend using a moving average.**
A **moving average** just means: take a small window of consecutive points and average them, then slide the window forward one step at a time. This smooths out the seasonal bumps because each window contains one of every season.

Since our seasonal period is 4 (four quarters per year), we average over 4 quarters at a time. But averaging 4 points gives you a value that sits *between* two quarters (not centered on any one quarter), so the standard trick is a **centered moving average**: average two consecutive 4-quarter averages together. Let's do it step by step.

4-quarter moving averages (each is the average of that quarter and the 3 before it... actually let's use the standard centered approach):

MA at Q3 2023 (centered on Q2.5–Q3, average of Q1–Q4 2023): (10+18+22+12)/4 = **15.5**
MA at Q4 2023 (average of Q2 2023 – Q1 2024): (18+22+12+14)/4 = **16.5**
MA at Q1 2024 (average of Q3 2023 – Q2 2024): (22+12+14+22)/4 = **17.5**
MA at Q2 2024 (average of Q4 2023 – Q3 2024): (12+14+22+26)/4 = **18.5**
MA at Q3 2024 (average of Q1 2024 – Q4 2024): (14+22+26+16)/4 = **19.5**

Now center these (average adjacent pairs of the above) to line them up with actual quarters:
Centered trend at Q4 2023 = (15.5+16.5)/2 = **16.0**
Centered trend at Q1 2024 = (16.5+17.5)/2 = **17.0**
Centered trend at Q2 2024 = (17.5+18.5)/2 = **18.0**
Centered trend at Q3 2024 = (18.5+19.5)/2 = **19.0**

So our estimated trend $T_t$ is slowly rising: roughly 16 → 17 → 18 → 19 across those quarters (about +1/quarter). This matches the eyeball intuition: overall revenue is climbing year over year.

**Step 2 — Extract the seasonal effect (additive assumption: $S_t = x_t - T_t$).**

Q4 2023: $S = 12 - 16.0 = -4.0$
Q1 2024: $S = 14 - 17.0 = -3.0$
Q2 2024: $S = 22 - 18.0 = +4.0$
Q3 2024: $S = 26 - 19.0 = +7.0$

Interpretation, in plain English: Q3 (summer, peak ice-cream season) runs about **+7** above trend. Q4 (fall/winter) runs about **−4** below trend. This is exactly the seasonal fingerprint you'd expect for an ice cream shop.

**Step 3 — This is the "how to alter it" answer you asked for: seasonal adjustment.**
If you want to see the "true underlying growth" of the business with the seasonal noise removed (this is literally what government agencies do when they report "seasonally adjusted unemployment" or "seasonally adjusted retail sales"), you **subtract out** the average seasonal effect for that quarter from every observation:

$$
x_t^{\text{seasonally adjusted}} = x_t - S_t
$$

So if Q3's typical seasonal bump is +7, and this Q3 actual revenue was 26, the seasonally-adjusted value is $26 - 7 = 19$ — telling you "if there were no summer effect at all, the underlying business level this quarter is about 19," which you can now fairly compare to a Q4 or Q1 number without the seasonal distortion fooling you.

**This is the general recipe for "how do I find and remove an effect":**
1. Estimate the slow-moving part (trend) using smoothing (moving average).
2. Subtract it off to isolate what's left (seasonal + noise).
3. Average the leftover by season-position (all Q3s together, all Q4s together, etc.) to get one clean seasonal effect per season-position.
4. Subtract (additive) or divide (multiplicative) that clean seasonal effect out of the raw data to "adjust away" that effect.
5. Whatever's left after removing both trend and seasonality is your noise/irregular component — check that it looks like structureless static.

---

## 6. How to actually *detect* trend/seasonality in practice (not just by eyeballing)

You asked specifically "how to find how it affects" — here are the concrete, practical methods (we'll formalize the statistical tests for these in Phase 4, but here's the intuitive first pass):

1. **Plot the raw series.** Always step one. Look for: overall drift (trend), repeating bumps at fixed intervals (seasonality), irregular multi-year swings (cycle).
2. **Plot a moving average on top of the raw data** (as we did above) — this visually strips out the noise/seasonality and shows you the trend line directly.
3. **Seasonal subseries plot**: group all the January values together across years, all the Februarys together, etc., and look at each group's average. If the group averages differ a lot and are stable year to year, that's seasonality.
4. **Check if the seasonal swing size grows with the trend level** (megaphone shape) → tells you additive vs multiplicative, as discussed in section 4.

---

## 7. Quick self-check questions (answer in your head before moving on)

1. A coffee shop's sales are higher every single Monday morning (people need coffee after the weekend) — is that trend, seasonal, or cyclical?
   *(Answer: seasonal — fixed, known, calendar-tied repeat pattern, in this case weekly)*
2. A country's GDP grows for 7 years then shrinks for 2, then grows for 5, then shrinks for 3 — trend, seasonal, or cyclical?
   *(Answer: cyclical — repeats but without a fixed length)*
3. If a series' seasonal swings get visibly bigger in absolute dollar terms every year as the business grows, should you model it as additive or multiplicative?
   *(Answer: multiplicative — because the swing size scales with the level; alternatively, log-transform and treat it as additive)*

---

## What's next
Phase 2 will build directly on this: we'll formally define what a **stochastic process** is (the mathematical machine that "generates" a time series), and introduce **white noise** and the **random walk** — the two simplest building-block processes that every more complex model (AR, MA, ARIMA) is built from. We go just as slow.

Say "next" when ready, or ask me to drill deeper into anything in Phase 1 first (e.g., more worked examples, or the multiplicative version of the ice cream example worked by hand).
