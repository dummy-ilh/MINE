
### 📌 What is Data Drift?

Data drift is a fundamental concept in machine learning that refers to the change in the statistical properties of the input data over time, as compared to the data on which a machine learning model was originally trained. In simple terms: your model was trained on yesterday’s world, but today’s world looks different. If the input data distribution changes, the model may become less accurate and reliable because it is seeing data that no longer matches its training distribution.
Data drift refers to the change in the statistical properties of input data over time, compared to the data on which a machine learning model was originally trained.

In simple terms: your model was trained on yesterday’s world, but today’s world looks different.

If the input data distribution changes, the model may become less accurate because it is seeing data that no longer matches its training distribution.
Data drift is a key challenge in MLOps (Machine Learning Operations), as it can lead to a significant degradation in the performance of a deployed model. This phenomenon directly violates the **IID (Independently and Identically Distributed) assumption**, which underpins most machine learning models.

### ⚖️ Difference Between Related Concepts

It's crucial to distinguish between various types of drift to understand the root cause of model degradation. While often used interchangeably, these terms have specific meanings:

* **Data Drift:** A broad term for any change in the data distribution.
* **Covariate Drift (Feature Drift):** The distribution of the input features ($P(x)$) changes, while the relationship between the features and the target variable remains the same ($P(y|x)$).
* **Concept Drift:** The relationship between the input features and the target variable changes ($P(y|x)$), even if the distribution of the features themselves remains stable.
* **Prior Probability Shift (Label Drift):** The distribution of the target variable ($P(y)$) changes.

### 🔎 Types of Data Drift

Data drift can be categorized in several ways, providing a more granular view of the problem.

#### Based on What Drifts:

1.  **Covariate Drift (Feature Drift):**
    * **Description:** The statistical distribution of your input features changes over time.
    * **Example:** A fraud detection model trained on credit card data from 2020 may face new spending patterns in 2025 (e.g., more online shopping, new digital payment methods). The distribution of features like "transaction amount" or "merchant type" changes, but the underlying rules of what constitutes fraud might not.
2.  **Prior Probability Shift (Label Drift):**
    * **Description:** The distribution of the target variable itself changes.
    * **Example:** A medical diagnosis model is trained when a disease is rare (e.g., 1% prevalence). A new outbreak occurs, and the proportion of positive cases (Y=1) increases to 5%. The model's performance can drop significantly as its training data no longer reflects the true class balance.
3.  **Concept Drift:**
    * **Description:** The relationship between features and labels changes over time.
    * **Example:** A spam detection model's understanding of "spam" becomes outdated. Spammers invent new techniques, and words that were once "spammy" are now common, while new patterns emerge that the model doesn't recognize. Mathematically, the conditional probability $P(Y|X)$ changes.

#### Based on Severity:

* **Virtual Drift vs. Real Drift:**
    * **Virtual Drift:** Only the input distribution changes, but the model's optimal decision boundary does not. While the model's raw performance might not drop immediately, it's a sign of potential future issues.
    * **Real Drift:** The actual relationship between inputs and outputs changes, directly leading to a decline in model performance. This is a critical issue that requires immediate attention.

#### Based on Timing:

* **Sudden Drift:** The change in data distribution happens all at once.
    * **Example:** A major external event like a global pandemic or a new government regulation dramatically alters consumer behavior overnight.
* **Gradual Drift:** The change in data distribution happens slowly and steadily over time.
    * **Example:** The evolution of slang words in social media sentiment analysis, or a slow shift in demographic trends.
* **Incremental Drift:** A series of small, sudden changes where an old distribution is replaced by a new one step by step.
Perfect! You’re now asking about the **temporal patterns** of drift — i.e., *how drift unfolds over time*. This is often classified into **sudden, gradual, incremental, and recurring drift**. Let’s go through each in detail with examples and a bit of math intuition.

---

# ⏳ Types of Drift by Time Pattern
![Drift Detection Diagram — Drift Types](https://cymulate.com/uploaded-files/2025/01/Drift-Detection-Diagram-Drift-Types.png)

### 1. **Sudden Drift (Abrupt Drift)**

* **Definition:** The data distribution changes all at once at a specific point in time.
* **Intuition:** Imagine flipping a switch — the "world" is different overnight.
* **Example:**

  * COVID-19 → people instantly stopped traveling.
  * Stock market crashes → data patterns change in a day.
* **Mathematical view:**
  Before time $t$, data comes from distribution $P_1(X,Y)$.
  At time $t$, it instantly switches to $P_2(X,Y)$.

---

### 2. **Gradual Drift**

* **Definition:** The old distribution slowly fades out while the new distribution grows in importance. Both coexist for a while.
* **Intuition:** Think of **slow climate change** rather than a sudden storm.
* **Example:**

  * Customer preferences changing over months (e.g., fashion trends).
  * Slang words in language models slowly evolving.
* **Mathematical view:**
  Over time, a mixture distribution exists:

  $$
  P_t(X,Y) = \alpha_t P_1(X,Y) + (1 - \alpha_t) P_2(X,Y)
  $$

  where $\alpha_t$ gradually decreases from 1 to 0.

---

### 3. **Incremental Drift**

* **Definition:** The distribution changes continuously in small steps, and the old distribution disappears completely.
* **Intuition:** Imagine a river slowly carving a new path — you never notice an abrupt shift, but over years the course is different.
* **Example:**

  * Currency exchange rate behavior drifting slowly due to inflation.
  * Wear-and-tear sensor readings in machines.
* **Mathematical view:**
  Distribution $P_t(X,Y)$ evolves smoothly over time — no discrete jumps, only small incremental updates.

---

### 4. **Recurring Drift (Seasonal / Cyclical Drift)**

* **Definition:** Old data distributions reappear after some time.
* **Intuition:** Like **seasons** — patterns repeat.
* **Example:**

  * Retail sales → spike every December.
  * Electricity demand → cycles daily and seasonally.
  * Fashion → old trends come back.
* **Mathematical view:**
  $P_t(X,Y)$ is not monotonic — it revisits earlier states, e.g.

  $$
  P_t \approx P_{t+k}
  $$

  for some period $k$.

---

# 📊 Summary Table

| Drift Type      | How it happens                                | Example                                       |
| --------------- | --------------------------------------------- | --------------------------------------------- |
| **Sudden**      | Instant switch                                | COVID-19 changing consumer behavior overnight |
| **Gradual**     | Old and new distributions overlap for a while | Fashion trends                                |
| **Incremental** | Small continuous changes                      | Sensor wear, inflation effects                |
| **Recurring**   | Old distributions come back periodically      | Seasonal sales, electricity demand            |

---

👉 Would you like me to **make you a timeline-style diagram** (with curves showing sudden vs gradual vs incremental vs recurring drift) so you can *visualize* the differences clearly?

### 📊 Why Data Drift Matters

Data drift is not just a theoretical problem; it has real-world consequences:

* **Model Performance Degradation:** The most significant effect is a drop in model accuracy, precision, recall, and other key performance metrics.
* **Biased or Unreliable Predictions:** A drifted model's predictions may become systematically wrong, leading to poor decisions.
* **Relevance:** Data drift is a common and inevitable problem in real-world systems, especially in dynamic environments like finance, healthcare, e-commerce, and IoT.

### 🛠️ How to Detect Data Drift

Relying solely on model performance metrics is insufficient, as drift can occur without an immediate, visible drop in accuracy. It is essential to continuously monitor the data itself.

**Common Detection Methods:**

* **Statistical Tests:** Use statistical tests to compare the distribution of the training (baseline) data with the new, incoming data.
    * **Kolmogorov–Smirnov (KS) Test:** A non-parametric test to determine if two datasets are from the same distribution.
    * **Jensen–Shannon (JS) Divergence:** A method to measure the similarity between two probability distributions.
    * **Population Stability Index (PSI):** A common metric in credit scoring that measures the shift in population distribution between two periods.
* **Visualizations:** Plotting histograms, density plots, or time-series charts of key features over time provides a quick and intuitive way to spot changes.
* **Feature-Level Monitoring:** Tracking summary statistics (mean, median, variance) of each feature can help pinpoint exactly which features are drifting.
* **Unsupervised Drift Detection:** This is a powerful technique that doesn't require new labels to be available. It compares the statistical properties of the training data and the live production data.

### ✅ How to Handle Data Drift

Once drift is detected, several strategies can be employed to mitigate its effects:

* **Continuous Monitoring:** A robust MLOps pipeline is crucial. It should continuously monitor live data against training data and alert the team when drift exceeds a predefined threshold.
* **Model Retraining:** This is the most common solution. The model is retrained periodically or on-demand using a new, more recent dataset that reflects the current data distribution.
* **Adaptive Models (Online Learning):** For fast-drifting environments, models can be updated continuously with new data as it arrives, rather than through periodic retraining.
* **Domain Adaptation:** Techniques that adapt a model to a new data distribution without full retraining, such as transfer learning or re-weighting data points.
* **Data Augmentation:** Enriching the training dataset with diverse samples to make the model more robust to future changes.
Excellent — now you’re moving into **methods for detecting drift**. Let’s break down **PSI, KL-divergence, ADWIN, and SHAP** one by one, because they belong to different “families” of drift detection.

---

# 🛠️ Drift Detection Methods

## 1. **PSI (Population Stability Index)**

* **What it is:** A statistical measure used in credit risk and ML monitoring to detect changes in feature or score distributions.
* **How it works:**

  1. Split data into bins (like histograms).
  2. Compare the proportion of samples in each bin between **baseline (training)** and **current (production)**.
  3. Compute PSI = sum of weighted log differences.

     $$
     PSI = \sum_i (p_i - q_i) \cdot \ln \left(\frac{p_i}{q_i}\right)
     $$

     where $p_i$ = baseline proportion, $q_i$ = current proportion.
* **Interpretation:**

  * PSI < 0.1 → Stable (no drift).
  * 0.1 ≤ PSI < 0.25 → Moderate drift.
  * PSI ≥ 0.25 → Significant drift.
* **Strengths:** Simple, interpretable, widely used in finance.
* **Weakness:** Needs binning, may miss subtle changes.

---

## 2. **KL Divergence (Kullback–Leibler Divergence)**

* **What it is:** A measure of how one probability distribution diverges from another.
* **Formula:**

  $$
  D_{KL}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
  $$

  where $P$ = baseline distribution, $Q$ = current distribution.
* **Interpretation:**

  * KL = 0 → Identical distributions.
  * Higher KL → Larger drift.
* **Strengths:** Rigorous information-theoretic foundation.
* **Weakness:** Asymmetric (KL(P||Q) ≠ KL(Q||P)), sensitive to zero probabilities.

*(PSI is basically a variant of KL, adapted for business use.)*

---

## 3. **ADWIN (Adaptive Windowing)**

* **What it is:** An **online drift detection algorithm** (good for streaming data).
* **How it works:**

  * Maintains a sliding window of recent data.
  * Splits window into two subwindows (left & right).
  * Tests if their averages differ statistically.
  * If yes → detects drift, shrinks the window.
* **Advantages:**

  * No need to fix window size in advance.
  * Good for real-time monitoring (IoT, finance streams).
* **Weakness:**

  * Focuses mainly on mean changes (not higher-order distributional changes).

---

## 4. **SHAP (SHapley Additive exPlanations)**

* **What it is:** A model explainability tool, not a drift detector by itself — but **can be adapted for drift monitoring**.
* **How it works:**

  * SHAP values measure **feature importance** for each prediction.
  * If the **distribution of SHAP values** shifts over time, it suggests that features are being used differently → possible **concept drift**.
* **Example:**

  * A credit model used to rely heavily on **income** (SHAP importance high).
  * Over time, SHAP importance for **income** drops, and **transaction patterns** increase → drift in model’s decision logic.
* **Strengths:** Detects *explanatory drift* (not just input drift).
* **Weakness:** Requires model access + computationally heavy.

---

# 🔎 Quick Comparison

| Method            | Type                              | Good For                                                  | Limitation                              |
| ----------------- | --------------------------------- | --------------------------------------------------------- | --------------------------------------- |
| **PSI**           | Statistical (binned distribution) | Business monitoring (finance, credit scoring)             | Needs binning, heuristic thresholds     |
| **KL Divergence** | Info-theoretic                    | General distributional drift                              | Asymmetric, sensitive to 0s             |
| **ADWIN**         | Online drift detector             | Streaming data (real-time drift)                          | Focuses on mean, may miss subtle shifts |
| **SHAP**          | Model interpretability-based      | Explaining **concept drift** (feature importance changes) | Expensive, requires model internals     |

---

✅ In practice:

* **PSI/KL** → Detect input drift.
* **ADWIN** → Catch real-time drifts in streams.
* **SHAP** → Detect changes in **model reasoning** (concept drift).

---
Great — I’ll give clear, **worked numerical examples** for each method: **PSI**, **KL divergence**, **ADWIN (simple online example using Hoeffding bound style test)**, and a **SHAP-based example** (toy SHAP values and how to detect drift in them). I’ll show step-by-step arithmetic so you can follow exactly.

---

# 1) PSI — Population Stability Index (numeric example)

Suppose a feature's values are binned into 4 bins. Baseline (training) counts and current (production) counts:

* Baseline counts (train): \[200, 300, 400, 100]  (total 1000)
* Current counts (prod):  \[150, 350, 380, 120]  (total 1000)

Step 1 — compute proportions $p_i$ (baseline) and $q_i$ (current):

* $p = [200/1000, 300/1000, 400/1000, 100/1000] = [0.20, 0.30, 0.40, 0.10]$
* $q = [150/1000, 350/1000, 380/1000, 120/1000] = [0.15, 0.35, 0.38, 0.12]$

Step 2 — compute PSI per bin: $(p_i - q_i) \cdot \ln(p_i/q_i)$

We compute each bin carefully (digit-by-digit):

Bin 1:

* $p_1 - q_1 = 0.20 - 0.15 = 0.05$
* $p_1/q_1 = 0.20 / 0.15 = 1.333333...$
* $\ln(1.3333333) \approx 0.287682072$
* Contribution = $0.05 \times 0.287682072 = 0.0143841036$

Bin 2:

* $p_2 - q_2 = 0.30 - 0.35 = -0.05$
* $p_2/q_2 = 0.30 / 0.35 = 0.857142857$
* $\ln(0.857142857) \approx -0.1541506798$
* Contribution = $-0.05 \times -0.1541506798 = 0.00770753399$

Bin 3:

* $p_3 - q_3 = 0.40 - 0.38 = 0.02$
* $p_3/q_3 = 0.40 / 0.38 = 1.0526315789$
* $\ln(1.0526315789) \approx 0.0512932944$
* Contribution = $0.02 \times 0.0512932944 = 0.001025865888$

Bin 4:

* $p_4 - q_4 = 0.10 - 0.12 = -0.02$
* $p_4/q_4 = 0.10 / 0.12 = 0.8333333333$
* $\ln(0.8333333333) \approx -0.1823215568$
* Contribution = $-0.02 \times -0.1823215568 = 0.003646431136$

Step 3 — sum contributions:

PSI = $0.0143841036 + 0.00770753399 + 0.001025865888 + 0.003646431136 = 0.026763934614$

Round: **PSI ≈ 0.0268**

Interpretation (common cutoffs):

* PSI < 0.1 → stable. Here **0.0268** → **no significant drift**.

---

# 2) KL Divergence — simple numeric example

Use the same baseline $P$ and current $Q$ proportions from above:

* $P = [0.20, 0.30, 0.40, 0.10]$
* $Q = [0.15, 0.35, 0.38, 0.12]$

KL divergence $D_{KL}(P||Q)=\sum_i P_i \ln(P_i/Q_i)$.

Compute per bin:

Bin 1:

* $P_1 \ln(P_1/Q_1) = 0.20 \times \ln(0.20/0.15)$
* $0.20/0.15 = 1.3333333$, $\ln = 0.287682072$
* Contribution = $0.20 \times 0.287682072 = 0.0575364144$

Bin 2:

* $0.30 \times \ln(0.30/0.35)$
* $0.30/0.35 = 0.857142857,\ \ln \approx -0.1541506798$
* Contribution = $0.30 \times -0.1541506798 = -0.04624520394$

Bin 3:

* $0.40 \times \ln(0.40/0.38)$
* $0.40/0.38 = 1.0526315789,\ \ln \approx 0.0512932944$
* Contribution = $0.40 \times 0.0512932944 = 0.02051731776$

Bin 4:

* $0.10 \times \ln(0.10/0.12)$
* $0.10/0.12 = 0.8333333333,\ \ln \approx -0.1823215568$
* Contribution = $0.10 \times -0.1823215568 = -0.01823215568$

Sum contributions:

$D_{KL} = 0.0575364144 - 0.04624520394 + 0.02051731776 - 0.01823215568$

Compute stepwise:

* $0.0575364144 - 0.04624520394 = 0.01129121046$
* $0.01129121046 + 0.02051731776 = 0.03180852822$
* $0.03180852822 - 0.01823215568 = 0.01357637254$

So **KL(P||Q) ≈ 0.013576** (small). Interpretation: small divergence, distributions similar.

Note: KL is asymmetric; KL(Q||P) would be slightly different.

---

# 3) ADWIN-style (online) numeric example — detecting a sudden change in stream mean

ADWIN (Adaptive Windowing) is an online approach. I’ll do a **simple numerical demonstration** using the Hoeffding inequality style test ADWIN uses: split window into left (older) and right (recent) parts and check if means differ significantly.

Scenario: stream of numeric values. For simplicity, we maintain a window of last 20 observations and test one split (left 10, right 10). We’ll show a case where a sudden change occurs at t = 21 (mean jumps).

Data:

* First 20 samples: drawn around mean 0.0 — we’ll use exact values:
  window = \[0.1, -0.2, 0.05, 0.0, -0.1, 0.2, -0.05, 0.0, 0.1, -0.15, 0.05, -0.05, 0.0, 0.02, -0.02, 0.1, -0.1, 0.05, 0.0, 0.03]
  (these are roughly centered around 0)
* Next 5 new samples (recent) with mean ≈ 1.0: \[1.1, 0.95, 1.05, 1.0, 0.9]

We will form a combined current window of size 15 (last 10 of the original plus new 5) for demonstration: take last 10 of old window (positions 11–20) and 5 new ones:

Left chunk (older half) — last 10 old samples (positions 11–15 as left? For demonstration pick 7 older, 8 newer — but to be simple I'll use a split: Left = older 7 values, Right = recent 8 values). To keep numbers simple, do Left = the 7 values from indices 11–17:

Left (n0 = 7): \[0.05, -0.05, 0.0, 0.02, -0.02, 0.1, -0.1]
Right (n1 = 8): \[0.05, 0.0, 0.03, 1.1, 0.95, 1.05, 1.0, 0.9]

Compute means:

Left mean $\mu_0$:
Sum left = $0.05 -0.05 +0.0 +0.02 -0.02 +0.1 -0.1 = 0.0$
So $\mu_0 = 0.0/7 = 0.0$

Right mean $\mu_1$:
Sum right = $0.05 + 0.0 + 0.03 + 1.1 + 0.95 + 1.05 + 1.0 + 0.9$
Compute stepwise:

* 0.05 + 0.0 = 0.05
* 0.05 + 0.03 = 0.08
* 0.08 + 1.1 = 1.18
* 1.18 + 0.95 = 2.13
* 2.13 + 1.05 = 3.18
* 3.18 + 1.0 = 4.18
* 4.18 + 0.9 = 5.08

So sum = 5.08, n1 = 8 → $\mu_1 = 5.08 / 8 = 0.635$

Difference of means = $|\mu_1 - \mu_0| = 0.635$

ADWIN uses a statistical bound: the difference is significant if it exceeds $\epsilon$ where (Hoeffding-style) for confidence $\delta$:

$$
\epsilon = \sqrt{\frac{1}{2} \cdot \frac{\ln(2/\delta)}{n_0} } + \sqrt{\frac{1}{2} \cdot \frac{\ln(2/\delta)}{n_1}}
$$

(ADWIN derives a similar bound; this is for illustration.)

Choose $\delta = 0.05$ (95% confidence). Then $\ln(2/\delta) = \ln(40) \approx 3.688879454$.

Compute first term:

* $\sqrt{\frac{1}{2} \cdot \frac{3.688879454}{7}} = \sqrt{ \frac{3.688879454}{14} } = \sqrt{0.2634913896} \approx 0.513363$

Second term:

* $\sqrt{\frac{1}{2} \cdot \frac{3.688879454}{8}} = \sqrt{ \frac{3.688879454}{16} } = \sqrt{0.230555\ } \approx 0.480136$

Sum $\epsilon \approx 0.513363 + 0.480136 = 0.993499$

Compare difference of means 0.635 vs epsilon 0.9935: 0.635 < 0.9935 → not significant at 95% with this small sample split. But if we enlarge the right chunk (more recent samples), the epsilon terms shrink.

If instead we had n0 = 7 and n1 = 20 (say many more recent points around 1.0), the second term becomes much smaller and difference would become significant. This demonstrates ADWIN idea: with enough recent evidence, the mean shift will exceed the statistical bound → declare drift and shrink window.

**Takeaway:** ADWIN detects drift when the difference between subwindow statistics is larger than the calculated confidence-bound $\epsilon$. Small sample sizes may not show significance even for moderate mean shifts; ADWIN adapts window size to accumulate evidence.

---

# 4) SHAP — toy numerical example and detecting explanatory (concept) drift

SHAP assigns additive contributions (per feature) that sum to the prediction difference vs baseline. We'll use a tiny toy dataset and **made-up but consistent SHAP values** to demonstrate how to detect drift in feature importance.

Suppose a binary classifier predicts probability of positive class. For three features (A, B, C). For five predictions at time T0 (baseline), we have SHAP values per sample:

Time T0 — 5 samples (SHAP contributions for features A,B,C and base\_value = 0.2):

| Sample | base | SHAP\_A | SHAP\_B | SHAP\_C | model output = base+sum(SHAP) |
| ------ | ---: | ------: | ------: | ------: | ----------------------------: |
| 1      | 0.20 |   +0.10 |   +0.05 |   +0.00 |                          0.35 |
| 2      | 0.20 |   +0.08 |   +0.03 |   +0.01 |                          0.32 |
| 3      | 0.20 |   +0.12 |   +0.02 |   -0.01 |                          0.33 |
| 4      | 0.20 |   +0.09 |   +0.04 |   +0.00 |                          0.33 |
| 5      | 0.20 |   +0.11 |   +0.03 |   +0.00 |                          0.34 |

Compute mean absolute SHAP (mean magnitude) per feature at T0:

* mean |SHAP\_A| = (0.10+0.08+0.12+0.09+0.11)/5 = 0.50/5 = **0.10**
* mean |SHAP\_B| = (0.05+0.03+0.02+0.04+0.03)/5 = 0.17/5 = **0.034**
* mean |SHAP\_C| = (0.00+0.01+0.01+0.00+0.00)/5 = 0.02/5 = **0.004**

So at baseline, **A ≫ B ≫ C** in importance.

Now at a later time T1 (production), we observe 5 new samples and compute SHAP values (again made-up):

Time T1:

| Sample | base | SHAP\_A | SHAP\_B | SHAP\_C | output |
| ------ | ---: | ------: | ------: | ------: | -----: |
| 1      | 0.20 |   +0.02 |   +0.20 |   +0.10 |   0.52 |
| 2      | 0.20 |   +0.03 |   +0.22 |   +0.12 |   0.57 |
| 3      | 0.20 |   +0.01 |   +0.18 |   +0.09 |   0.48 |
| 4      | 0.20 |   +0.02 |   +0.19 |   +0.11 |   0.52 |
| 5      | 0.20 |   +0.00 |   +0.21 |   +0.10 |   0.51 |

Compute mean |SHAP| at T1:

* mean |SHAP\_A| = (0.02+0.03+0.01+0.02+0.00)/5 = 0.08/5 = **0.016**
* mean |SHAP\_B| = (0.20+0.22+0.18+0.19+0.21)/5 = 1.00/5 = **0.20**
* mean |SHAP\_C| = (0.10+0.12+0.09+0.11+0.10)/5 = 0.52/5 = **0.104**

Compare T0 → T1 mean abs SHAP:

* Feature A: 0.10 → 0.016  (big drop)
* Feature B: 0.034 → 0.20  (big increase)
* Feature C: 0.004 → 0.104 (big increase)

This indicates **explanatory / concept drift**: the model’s reasoning shifted from relying on A to relying on B and C.

**Detection procedure (numeric):**

* For each feature, compute a divergence measure between the distribution of SHAP values at baseline and production — e.g., compare mean absolute SHAP with a t-test or use KS test on distributions. For a simple numeric check use percent change in mean abs SHAP:

Percent change for A: $(0.016 - 0.10)/0.10 = -0.84 = -84\%$ (big decrease)
Percent change for B: $(0.20 - 0.034)/0.034 ≈ 4.882 ≈ +488\%$ (huge increase)
Percent change for C: $(0.104 - 0.004)/0.004 = 25 ≈ +2500\%$

Set practical thresholds (example): if mean |SHAP| changes by >50% for top features → flag explanatory drift. Here all three exceed that → **alert**.

**Why useful:** SHAP change shows the *model is using features differently* even if raw X distributions didn’t change much — strong sign of concept drift.

---

# Quick reference: when to use which numeric test

* **PSI**: good for binned univariate checks. Use if you have score/feature histograms. (Example above: PSI ≈ 0.027 → stable.)
* **KL**: info-theory; use when you want continuous distribution comparison (watch out for zeros). (KL ≈ 0.0136 → similar.)
* **ADWIN / online**: use for streaming numeric stats (means) — need enough samples for the bound $\epsilon$ to be small.
* **SHAP-based**: compute distributions of SHAP values per feature, compare baseline vs production (KS test, t-test, percent-change of mean abs SHAP). Good for concept/explanatory drift.




### ✨ In Summary

Data drift is a change in the input or target data distribution over time that causes a model's predictions to degrade. It can be categorized as **covariate, label, or concept drift** and can occur **suddenly, gradually, or incrementally.** Detecting and handling data drift is a critical part of maintaining the health of any machine learning system in the real world.
