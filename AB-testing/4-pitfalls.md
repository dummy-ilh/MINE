
---

# 🚫 1. Peeking Problem (Early Stopping Bias)

---

## 🧠 What it is

You repeatedly check results **before experiment completion** and stop when you see significance.

---

## ❗ Why it's dangerous

Every check = a new hypothesis test → increases false positives.

👉 You think:

> “p < 0.05, we won!”

Reality:

> You inflated Type I error (false discovery)

---

## 🔬 Intuition

If you flip a coin enough times and stop when it looks biased → you’ll *always* “find” a winner.

---

## 📉 Effect

* False positive rate can go from **5% → 20–30%+**

---

## ✅ Fixes

* Predefine sample size & duration
* Use **Sequential Testing** (we’ll cover below)

---

## 💬 Interview Gold Line

> “Peeking invalidates classical hypothesis testing assumptions.”

---

# 🚫 2. Multiple Testing Problem

---

## 🧠 What it is

You test:

* Many metrics OR
* Many variants OR
* Many segments

---

## ❗ Why it's dangerous

Probability of false positive increases:

[
P(\text{at least one false positive}) = 1 - (1 - \alpha)^k
]

Where:

* ( k ) = number of tests

---

## 🔥 Example

* 20 tests, α = 0.05
  👉 ~64% chance of false positive 😬

---

## ✅ Fixes

| Method                   | Idea                               |
| ------------------------ | ---------------------------------- |
| Bonferroni               | α / k                              |
| FDR (Benjamini-Hochberg) | control expected false discoveries |
| Pre-register metrics     | avoid fishing                      |

---

## 💬 Interview Gold Line

> “We should control for multiple hypothesis testing to avoid inflated false discovery rates.”

---

# 🚫 3. Selection Bias

---

## 🧠 What it is

Treatment & control groups are **not comparable**.

---

## 🔥 Example

* Power users → treatment
* Casual users → control

👉 Result is meaningless

---

## ❗ Why it happens

* Bad randomization
* Opt-in experiments
* Missing data

---

## 📉 Effect

You measure **bias**, not causal effect.

---

## ✅ Fixes

* Proper randomization
* Stratified sampling
* Check covariate balance

---

## 💬 Interview Gold Line

> “Without proper randomization, we lose causal validity.”

---

# 🚫 4. Simpson’s Paradox (FAANG FAVORITE)

---

## 🧠 What it is

Trend reverses when data is aggregated.

---

## 🔥 Example

| Group   | A   | B   |
| ------- | --- | --- |
| Mobile  | 60% | 70% |
| Desktop | 30% | 40% |

👉 B better in BOTH

BUT aggregated:

* A = 50%
* B = 48% 😱

---

## ❗ Why?

Different group sizes / distributions

---

## 📉 Effect

You make **wrong decision**

---

## ✅ Fix

* Always analyze by segments:

  * device
  * geography
  * user type

---

## 💬 Interview Gold Line

> “I’d check for Simpson’s paradox by segmenting the data.”

---

# 🚫 5. Novelty Effect

---

## 🧠 What it is

Users behave differently **just because something is new**

---

## 🔥 Example

* New UI → spike in engagement
* After 2 weeks → drops

---

## 📉 Effect

Short-term gains ≠ long-term value

---

## ✅ Fix

* Run experiment longer
* Monitor over time
* Compare returning users

---

## 💬 Interview Gold Line

> “We should account for novelty effects before making long-term decisions.”

---

# 🚫 6. Network Effects (Interference)

---

## 🧠 What it is

One user’s treatment affects another user.

---

## 🔥 Example

* Social media likes
* Messaging apps
* Marketplaces

---

## ❗ Why dangerous

Breaks assumption:

> “Units are independent”

---

## 📉 Effect

Biased estimates

---

## ✅ Fix

* Cluster randomization
* Graph-based experiments

---

## 💬 Interview Gold Line

> “This violates the SUTVA assumption (no interference between units).”

---

# 🚀 ADVANCED TOPICS (FAANG Differentiators)

---

# 7.1 🧪 CUPED (Variance Reduction)

---

## 🧠 Idea

Use **pre-experiment data** to reduce noise.

---

## 🔬 Intuition

Users behave similarly over time.

👉 Use past behavior to “predict” outcome and remove variance.

---

## 📉 Effect

* Same sample size → **more power**
* Or same power → **shorter experiment**

---

## 💬 Interview Gold Line

> “CUPED reduces variance using covariates from pre-experiment data.”

---

---

# 7.2 ⏱️ Sequential Testing

---

## 🧠 Idea

Allows **safe early stopping**

---

## ❗ Fixes peeking problem

Instead of fixed sample:

* Continuously monitor
* Adjust thresholds

---

## 📊 Methods

* SPRT
* Bayesian sequential
* Alpha spending

---

## 💬 Interview Gold Line

> “Sequential testing allows continuous monitoring while controlling error rates.”

---

---

# 7.3 🧠 Bayesian A/B Testing

---

## 🧠 Idea

Compute:

> “Probability that B is better than A”

---

## 🔬 Instead of:

* p-value → indirect

We get:

* Direct probability

---

## 🔥 Example

> “There’s a 95% probability variant B is better”

---

## ✅ Advantages

* More intuitive
* Works well with small data

---

## ❗ Trade-off

* Requires prior assumptions

---

## 💬 Interview Gold Line

> “Bayesian methods provide a probabilistic interpretation of treatment effects.”

---

---

# 7.4 📊 Heterogeneous Treatment Effects (HTE)

---

## 🧠 Idea

Effect is **not uniform across users**

---

## 🔥 Example

| Segment   | Effect |
| --------- | ------ |
| New users | +10%   |
| Old users | -5%    |

---

## ❗ Danger

Average = misleading

---

## ✅ Use Cases

* Personalization
* Targeted rollouts

---

## 🔬 Methods

* Segmentation
* Causal trees
* Uplift modeling

---

## 💬 Interview Gold Line

> “I’d check for heterogeneous treatment effects to ensure we’re not masking segment-level differences.”

---

# 🧠 Final Mental Model

---

```id="aj7d2k"
Pitfalls = Things that break validity

Advanced methods = Ways to fix or improve experiments
```


