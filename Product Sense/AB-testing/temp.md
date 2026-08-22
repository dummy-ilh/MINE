**Statistical Foundations & Experiment Sizing**

* **Core Hypothesis Testing:** Null ($H_0$) vs. Alternative ($H_1$) hypotheses, 1-tailed vs. 2-tailed tests.
* **Errors & Power:** $\alpha$ (Type I error / false positive), $\beta$ (Type II error / false negative), Statistical Power ($1 - \beta$, standardly 80%).
* **Sample Size & MDE:** Determining Minimum Detectable Effect (MDE), baseline conversion rate, variance, and required run time (accounting for weekly seasonality and novelty effects).
* **Statistical vs. Practical Significance:** Minimum business-meaningful threshold vs. large-sample $p$-value inflation.
* **Testing Distributions:** Z-test vs. T-test (Student's vs. Welch's for unequal variances), Chi-Square test, Mann-Whitney U / Bootstrap for non-normal metrics.

---

**Metrics Framework & Experiment Design**

* **Metric Hierarchy:**
* *North Star / Primary Metric:* Conversion, CTR, latency, retention.
* *Secondary / Diagnostic Metrics:* Engagement depth, feature usage.
* *Guardrail / Invariant Metrics:* App crash rate, load time, uninstalls, battery consumption (critical for Apple), page render time (Google).


* **Randomization Unit:** User ID vs. device/hardware token vs. session/query vs. cluster/geo-level.
* **Ratio Metrics & Delta Method:** When the randomization unit differs from the analysis unit (e.g., clicks per pageview randomized by user), using Taylor expansion to calculate correct standard error.
* **AA Testing:** Verifying no baseline bias, validating the false-positive rate ($\alpha$), and ensuring pipeline sanity.

---

**Advanced Experimentation & Variance Reduction**

* **CUPED (Controlled-experiment Using Pre-Experiment Data):** Utilizing historical pre-experiment covariates to drastically reduce metric variance and shrink required sample sizes.
* **Sequential Testing / Peeking Problem:** Alpha-spending functions, Always Valid $p$-values, or sequential probability ratio tests (SPRT) to safely monitor tests without false-positive inflation.
* **Multiple Testing Corrections:** Bonferroni correction, Benjamini-Hochberg (FDR) when evaluating multiple variants or metrics simultaneously.
* **Multi-Armed Bandits (MAB):** $\epsilon$-Greedy, Upper Confidence Bound (UCB), and Thompson Sampling for dynamic traffic routing (e.g., short-lived promotional campaigns, algorithmic explore/exploit).
* **Bayesian A/B Testing:** Posterior distribution updates, expected loss modeling, and credible intervals vs. Frequentist $p$-values.

---

**System Dynamics & Violation of Assumptions**

* **Sample Ratio Mismatch (SRM):** Detecting assignment skew via Chi-Square goodness-of-fit test; root causes include bot traffic, redirect latency, and logging drops.
* **SUTVA & Network Interference:** Spillover effects where one user's treatment affects control (social networks, sharing). Mitigations: Cluster-based randomization, graph partitioning, or ego-network randomization.
* **Marketplace / Resource Contention:** Switchback experiments (time-slot randomization) and Geo-experiments (synthetic control, diff-in-diff).
* **Behavioral Biases:** Novelty effect (temporary surge) vs. Primacy/Change aversion effect (initial drop before adoption).

---

**AI/ML & Platform-Specific Angles (Google & Apple)**

* **Offline vs. Online ML Evaluation:** Counteracting offline-online metric divergence (e.g., high NDCG/offline AUC not translating to online conversion).
* **Interleaving (Search/Ranking):** Merging ranked lists (Google Search/Play/App Store) to test ranking models with drastically higher sensitivity and lower sample requirements.
* **Apple-Specific (Privacy & On-Device Constraints):**
* Privacy-preserving telemetry, differential privacy in client-side metrics, and federated experimentation.
* Power/thermal efficiency and on-device compute trade-offs as guardrail metrics.


* **Google-Specific (Ecosystem & Long-Term Holdouts):**
* Long-term holdout groups to capture compound downstream effects and chronic user decay.
* Interdependent advertiser/creator/user two-sided market dynamics.
