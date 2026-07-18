# A/B Testing Interview Cheat Sheet
*Definitions, formulas, and answer templates for L5 MLE/DS interviews*

---

## 1. Core Definitions (Fast Recall)

| Term | One-line definition |
|---|---|
| **Null hypothesis (H₀)** | No difference between treatment and control (effect = 0) |
| **p-value** | P(seeing data this extreme or more \| H₀ true) — NOT P(H₀ is true) |
| **α (significance level)** | Your tolerance for Type I error (false positive); typically 0.05 |
| **Power (1−β)** | P(detecting a true effect of size δ \| effect is real); typically 80–90% |
| **Type I error** | False positive — declaring a winner when there's no real effect |
| **Type II error** | False negative — missing a real effect |
| **MDE** | Minimum Detectable Effect — smallest true effect your test is powered to reliably catch |
| **OEC** | Overall Evaluation Criterion — the single pre-registered primary metric that decides ship/no-ship |
| **Guardrail metric** | A metric you monitor to ensure the treatment doesn't cause unacceptable harm elsewhere (latency, crashes, revenue) — not the decision metric itself |
| **SRM** | Sample Ratio Mismatch — observed traffic split deviates from the intended split, signaling a broken randomization/logging pipeline |
| **CUPED** | Controlled-experiment Using Pre-Experiment Data — variance reduction technique using a pre-period covariate |
| **Network effect / interference** | SUTVA violation — one unit's treatment assignment affects another unit's outcome (common in marketplaces, social graphs) |
| **Novelty / primacy effect** | Effect that's inflated (novelty) or deflated (primacy/learning) early on and decays/grows as users adjust — a reason to distrust day-1 results |
| **FWER** | Family-Wise Error Rate — P(at least one false positive across a family of tests) |
| **FDR** | False Discovery Rate — expected proportion of false positives among declared discoveries |
| **Always-valid p-value / mSPRT** | A p-value construction that stays statistically valid under continuous monitoring / unplanned stopping |

---

## 2. Key Formulas

### Sample size (two-proportion z-test, equal allocation)
$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \, \bar{p}(1-\bar{p})}{\delta^2}$$
- $\bar p$ = pooled baseline conversion rate, $\delta$ = MDE (absolute), $z_{\alpha/2}$≈1.96 (α=0.05 two-sided), $z_\beta$≈0.84 (80% power)

### Sample size (two-sample t-test, continuous metric)
$$n = \frac{2(z_{\alpha/2}+z_\beta)^2\sigma^2}{\delta^2}$$

### Standard error of a difference in proportions
$$SE = \sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}$$

### Z-test statistic (two proportions)
$$z = \frac{\hat p_1 - \hat p_2}{\sqrt{\bar p(1-\bar p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}$$

### Confidence interval (general)
$$\hat\theta \pm z_{\alpha/2} \cdot SE(\hat\theta)$$

### CUPED-adjusted metric
$$Y_{cuped} = Y - \theta(X - \bar X), \quad \theta = \frac{Cov(Y,X)}{Var(X)}$$
Variance reduction achieved: $Var(Y_{cuped}) = Var(Y)(1-\rho^2)$ where ρ = correlation between metric and pre-period covariate.

### Delta method (variance of a ratio metric, e.g., CTR = clicks/sessions per user)
$$Var\left(\frac{\bar X}{\bar Y}\right) \approx \frac{1}{\bar Y^2}\left[Var(\bar X) - 2\frac{\bar X}{\bar Y}Cov(\bar X,\bar Y) + \frac{\bar X^2}{\bar Y^2}Var(\bar Y)\right]$$
Used when the unit of randomization (user) ≠ unit of analysis (session/click) — naive per-event variance understates true variance.

### SRM check (chi-square goodness of fit)
$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}, \quad df = k-1$$
Flag SRM if p < 0.001 (strict threshold, since SRM should basically never happen if randomization is correct).

### FWER — probability of ≥1 false positive across m independent tests
$$P(\text{≥1 false positive}) = 1-(1-\alpha)^m$$

### Bonferroni correction
$$\alpha_{\text{per test}} = \frac{\alpha}{m}$$

### Holm-Bonferroni (step-down)
Sort p-values ascending; compare $k$-th smallest to $\alpha/(m-k+1)$.

### Benjamini-Hochberg (FDR control)
Sort p-values ascending $p_{(1)} \le \dots \le p_{(m)}$; find largest $k$ s.t. $p_{(k)} \le \frac{k}{m}\alpha$; declare all $p \le p_{(k)}$ significant.

### Peeking inflation (naive, treating looks as independent)
Same form as FWER: $1-(1-\alpha)^{\text{\#looks}}$ — e.g., 20 daily peeks at α=0.05 → ≈64% naive false-positive rate (real correlated-peek inflation is lower but still commonly 20–30%+).

---

## 3. Answer Templates

### Template A — "Walk me through how you'd design an A/B test for [feature]"
1. **Clarify the goal** — what business question are we answering; who is the user population; what's the unit of randomization (user/session/device)?
2. **Define OEC** — single pre-registered primary metric tied to the hypothesis (not a vague "engagement," pick something measurable and directionally unambiguous).
3. **Define guardrails** — latency, crashes, revenue, unsubscribe rate, etc. — things that must not regress even if OEC wins.
4. **Power analysis** — baseline rate/variance, MDE (grounded in business relevance, not just "smallest detectable"), α, power → compute n → translate to duration given traffic.
5. **Randomization mechanism** — hash-based bucketing, consistent per-user assignment, check for interference/network effects if applicable (may require cluster or switchback design).
6. **Pre-registration** — analysis plan, stopping rule, primary vs. secondary/exploratory metrics labeled up front.
7. **Run + monitor** — SRM check on day 1, guardrail monitoring throughout (not stopping on it), no peeking-driven early stops unless using a proper sequential method.
8. **Analysis** — compute effect on OEC with CI, check guardrails, check for segment heterogeneity (with correction if exploratory), check novelty/primacy via day-over-day trend.
9. **Decision** — ship / no ship / iterate, tied back to the pre-registered decision rule, not vibes.

### Template B — "A PM wants to stop early because the p-value just crossed 0.05"
"That p-value doesn't account for the fact that you've been checking it repeatedly — stopping at the first significant-looking day inflates the true false-positive rate well past 5%, often into the 20–30% range. I'd want to know if our platform supports always-valid/sequential inference; if so, trust its peek-safe indicator. If not, either hold to the pre-planned sample size, or set up a group-sequential design (e.g., O'Brien-Fleming) with pre-specified interim looks next time."

### Template C — "We tested 15 secondary metrics and one came back significant"
"With 15 tests at α=0.05, you'd expect close to 1 false positive by chance alone even under a true null — this is a classic multiple-testing scenario. Since this wasn't the pre-specified OEC, I'd treat it as hypothesis-generating, not confirmatory. If we want to control for it formally: use Bonferroni/Holm if we need strict false-positive control on a single follow-up decision, or Benjamini-Hochberg if this is exploratory across many metrics and we can tolerate a controlled fraction of false leads."

### Template D — "One segment shows a huge effect, others don't — what do you do?"
"First check whether this segment was pre-specified or found by slicing after the fact — if it's the latter, this is Simpson's-paradox/multiple-testing territory: slicing into many segments and reporting the most extreme one is equivalent to running many hidden tests. I'd apply FDR/FWER correction across segments, check if sample size within the segment is adequate for the claimed effect size, and check whether the segment finding is consistent with a plausible causal story rather than reporting it as the headline result."

### Template E — "How do you know your test isn't broken before trusting the results?"
"Start with an SRM check — compare observed vs. expected traffic split with a chi-square test; if it fails at a strict threshold (e.g., p<0.001), something in randomization/logging is broken and the experiment's results aren't trustworthy regardless of what the OEC shows. I'd also check for interference/network effects if units interact (marketplace, social graph), and check day-over-day trend for novelty/primacy effects before trusting a short-duration read."

### Template F — "How would you increase power without just collecting more users?"
- Use a continuous proxy metric instead of a binary one if variance is lower for the same signal
- CUPED / covariate adjustment using pre-experiment data to reduce variance
- Reduce metric variance via winsorization/trimming of outliers (state tradeoffs — can bias the estimate if not done carefully)
- Paired/within-subject designs where feasible
- Longer duration / larger MDE if traffic is fundamentally limited
- Sequential/always-valid design to stop early when the true effect is large

### Template G — "Why not just look at as many metrics/segments as you want and report what's interesting?"
"That conflates exploratory and confirmatory analysis. Looking broadly is fine and useful for generating hypotheses, but reporting an uncorrected 'discovery' from scanning many cuts as if it were the pre-specified answer overstates confidence — you're implicitly running many tests and cherry-picking the one that crossed the threshold. The fix isn't 'don't explore,' it's 'label exploratory findings as exploratory, and follow up with a properly pre-registered, corrected test before acting on them.'"

### Template H — "What's the difference between statistical significance and practical significance?"
"Statistical significance just means we're confident the effect isn't exactly zero — it says nothing about whether the effect is big enough to matter for the business. With enough sample size, even a trivially small, practically irrelevant effect becomes statistically significant. That's why MDE should be chosen based on the smallest effect that would actually change a ship decision, not just 'whatever we're powered to detect with our current traffic.'"

---

## 4. Quick-Fire Concept Contrasts (common interview confusions)

| Confusion pair | Distinction |
|---|---|
| p-value vs. α | p-value is what you observed; α is the pre-set bar you compare it to |
| Power vs. confidence level | Power = 1−β (catching true effects); confidence = 1−α (avoiding false positives) |
| FWER vs. FDR | FWER = P(≥1 false positive); FDR = expected % of false positives among discoveries |
| Guardrail vs. OEC | OEC decides ship/no-ship; guardrails veto on harm but aren't the success metric |
| Monitoring vs. peeking-to-stop | Watching guardrails continuously = fine; using an uncorrected interim p-value as a stopping rule = not fine |
| Alpha-spending vs. always-valid | Alpha-spending = fixed, pre-planned number of looks; always-valid = continuous, unplanned monitoring, both control error but always-valid costs more sample size |
| SUTVA violation vs. selection bias | SUTVA violation = one unit's treatment leaks into another's outcome (interference); selection bias = the compared groups differ systematically before treatment |
| Novelty effect vs. true effect | Novelty decays over time as users adapt; a true effect should be stable/persist across the test duration |

---

## 5. Numbers Worth Memorizing

- z for α=0.05 two-sided: **1.96**
- z for 80% power: **0.84**
- z for 90% power: **1.28**
- Common SRM significance threshold: **p < 0.001** (stricter than normal 0.05, since SRM should essentially never trigger under correct randomization)
- Naive 20-look daily peeking inflation: **≈64%** false-positive rate (independent-look approximation); realistic correlated inflation: **20–30%+**
- Expected false positives from m=20 uncorrected tests at α=0.05: **≈1**

---
*Companion reference to Ch. 16 (Sequential Testing & Peeking) and Ch. 17 (Multiple Testing) in the applied experimentation curriculum.*
