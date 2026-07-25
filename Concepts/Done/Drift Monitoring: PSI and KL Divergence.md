# Drift Monitoring: PSI and KL Divergence — Interview Notes

## 1. Why These Two Specifically

PSI (Population Stability Index) and KL divergence are the two workhorses for **unlabeled distributional drift monitoring** — comparing a "baseline" distribution (usually the training set, or a stable recent reference window) against a "current" distribution (recent production traffic), *without needing any ground-truth labels*. This makes them the practical backbone of covariate-shift monitoring (see prior notes, §6) — they're cheap, run continuously, and don't wait on label latency.

Both work the same fundamental way: **bin a continuous feature (or use natural categories for a categorical one), compute the proportion of observations in each bin for both distributions, then compare the two histograms.** The difference is *how* they compare them and what the resulting number means.

## 2. Population Stability Index (PSI)

### Formula
$$\text{PSI} = \sum_{i=1}^{n} (Actual_i - Expected_i) \times \ln\left(\frac{Actual_i}{Expected_i}\right)$$

where, for bin $i$: $Expected_i$ = proportion of the baseline/reference population in that bin, $Actual_i$ = proportion of the current population in that bin, summed over all $n$ bins.

**Origin:** PSI comes from the credit-risk/banking world (scorecard monitoring) — it predates modern ML drift-monitoring tooling by decades, which is why it uses "Expected" (baseline/development sample) and "Actual" (current/validation sample) terminology rather than ML-native language.

### Interpretation — the industry-standard thresholds (state as heuristic, not law)
| PSI value | Interpretation |
|---|---|
| < 0.1 | No significant shift |
| 0.1 – 0.25 | Moderate shift — worth investigating |
| > 0.25 | Major shift — action likely needed (retrain, investigate pipeline, alert) |

**Caveat worth stating explicitly in an interview:** these thresholds are a widely-used industry convention from credit scoring, not a statistically derived universal law. The right threshold for *your* problem depends on the cost of a missed drift event vs. the cost of false-alarm retraining/investigation churn — a high-stakes fraud or medical model might act at PSI 0.1; a low-stakes internal dashboard might tolerate 0.3.

### Symmetry
PSI as commonly formulated is **symmetric** in the sense that swapping "Expected" and "Actual" doesn't just flip a sign trivially — it's actually mathematically symmetric overall (PSI(P,Q) = PSI(Q,P)), because it's really the sum of two KL-divergence-like terms in opposite directions (see §4). This symmetry is one reason it's popular for monitoring dashboards where "which one is the reference" can be a bit ambiguous to end users.

### Worked numeric example (whiteboard-able)
Feature: credit score, binned into 3 buckets.

| Bin | Baseline (Expected) % | Current (Actual) % |
|---|---|---|
| < 600 | 20% | 35% |
| 600–700 | 50% | 45% |
| > 700 | 30% | 20% |

$$\text{PSI} = (0.35-0.20)\ln\frac{0.35}{0.20} + (0.45-0.50)\ln\frac{0.45}{0.50} + (0.20-0.30)\ln\frac{0.20}{0.30}$$
$$= (0.15)(0.560) + (-0.05)(-0.105) + (-0.10)(-0.405)$$
$$= 0.0840 + 0.0053 + 0.0405 = 0.1298$$

PSI ≈ 0.13 → moderate shift, worth investigating (more low-score applicants showing up than the model was trained on — plausibly a real risk-profile shift in the applicant pool).

## 3. KL Divergence (Kullback-Leibler Divergence)

### Formula
$$D_{KL}(P \parallel Q) = \sum_{i} P(i) \ln \frac{P(i)}{Q(i)}$$

where $P$ is typically the "current/true" distribution and $Q$ is the "reference/model" distribution (or vice versa depending on convention — direction matters, see below).

### Key properties (interview-critical, frequently tested)

- **Not symmetric:** $D_{KL}(P\parallel Q) \neq D_{KL}(Q\parallel P)$ in general. This is the single most commonly tested KL fact — it is **not a true distance metric** (fails the symmetry axiom, and also fails the triangle inequality), even though it's often colloquially called a "distance."
- **Non-negative:** $D_{KL}(P\parallel Q) \geq 0$ always, with equality iff $P = Q$ everywhere (Gibbs' inequality).
- **Undefined when supports mismatch:** if $Q(i) = 0$ for any bin where $P(i) > 0$, $D_{KL}$ blows up to $+\infty$ (division by zero / $\ln(\infty)$). This is a real, common, practical problem: any bin present in current data but entirely absent from the training baseline (e.g., a brand-new categorical value) causes KL divergence to explode or become undefined — you need smoothing (see §5) to handle this.
- **Information-theoretic meaning:** $D_{KL}(P\parallel Q)$ is the expected number of extra "nats" (or bits, if log base 2) needed to encode samples from $P$ using a code optimized for $Q$ instead of the true distribution $P$ — i.e., the "surprise" or inefficiency of assuming $Q$ when reality is $P$. Good to be able to state this framing precisely in an interview; it's the reason KL shows up so pervasively across ML (cross-entropy loss, VAEs, information gain in decision trees are all built on this same object).

### Direction matters for drift monitoring
- $D_{KL}(\text{current} \parallel \text{baseline})$ answers: "how surprised would a model trained on the baseline distribution be, if it now had to explain today's traffic?" — this is usually the more natural direction for monitoring, since it directly measures how poorly the *deployed* model's implicit assumptions fit *today's* reality.
- $D_{KL}(\text{baseline} \parallel \text{current})$ answers a different, less intuitive question and is used less often for this purpose.
- Because of this asymmetry, teams sometimes report **Jensen-Shannon (JS) divergence** instead — a symmetrized, smoothed, and always-finite version:
$$D_{JS}(P \parallel Q) = \frac{1}{2}D_{KL}(P\parallel M) + \frac{1}{2}D_{KL}(Q\parallel M), \quad M = \frac{1}{2}(P+Q)$$
JS divergence is bounded (between 0 and $\ln 2$ in nats), always defined (no zero-probability blowup, because $M$ always has support wherever $P$ or $Q$ does), and symmetric — often the more practically convenient choice than raw KL for monitoring dashboards, at the cost of being slightly less standard/interpretable than PSI in a business context.

## 4. PSI vs. KL Divergence — How They Relate

PSI is actually a specific, symmetrized construction that can be decomposed in terms of KL-divergence-like quantities:

$$\text{PSI} = D_{KL}(Actual \parallel Expected) + D_{KL}(Expected \parallel Actual)$$

This is a genuinely good thing to be able to state precisely in an L5+ interview: **PSI is (approximately, under the same binning) the symmetrized sum of the two KL divergences in each direction** — which is exactly why PSI is symmetric while raw KL alone is not. It's essentially a "poor man's" symmetric divergence that predates the more formal Jensen-Shannon formulation but serves a similar practical purpose, wrapped in more business-friendly threshold conventions (§2).

## 5. Comparison Table

| | PSI | KL Divergence | Jensen-Shannon Divergence |
|---|---|---|---|
| Symmetric? | Yes | **No** | Yes |
| Bounded? | Not formally bounded above, but conventionally interpreted via 0.1/0.25 thresholds | Unbounded (can → ∞) | Bounded: [0, ln 2] |
| Defined when a bin has zero mass in one distribution? | Can be undefined/unstable at exact zero — needs smoothing in practice | **No** — blows up to infinity | Yes — always defined |
| Origin / typical domain | Credit risk / banking (industry-standard monitoring metric) | Information theory (foundational, used throughout ML: cross-entropy, VAEs, etc.) | Information theory (a "fixed" version of KL for practical use) | 
| Typical use in drift monitoring | Business dashboards, tabular feature monitoring, standard MLOps tooling defaults | Research contexts, deep-learning-internal comparisons (e.g., comparing predicted output distributions) | When you want a bounded, symmetric, always-finite drift score without PSI's specific threshold conventions |

## 6. Practical Implementation Details (the stuff that actually matters in production)

**Binning strategy matters a lot, and is a common source of unreliable PSI/KL scores:**
- **Equal-width bins** are simple but can be badly skewed by outliers (one huge outlier stretches bin width, emptying out most other bins).
- **Equal-frequency (quantile) bins**, computed on the *baseline* distribution and then reused (frozen) for the current distribution, are the standard practice — this ensures each baseline bin starts with equal weight (typically 10 bins of 10% each), making the metric more numerically stable and comparable across features with different scales.
- **Bin count tradeoff:** too few bins → the metric misses genuine distributional shape changes; too many bins → small-sample noise in each bin inflates the metric spuriously, especially for low-traffic features/segments. 10 bins is a common convention for PSI; the right number should scale with how much traffic you're monitoring per time window.
- **Categorical features:** use natural categories directly as "bins" rather than trying to force a continuous binning scheme; watch specifically for brand-new categories appearing only in current data (the zero-probability-in-baseline problem, below).

**The zero-probability problem (critical for both metrics, worse for KL):**
- If a bin has zero baseline mass but nonzero current mass (a genuinely new category/value appearing), raw KL divergence is undefined (÷0 inside the log). PSI is also numerically unstable at literal zero (the $\ln(Actual_i/Expected_i)$ term blows up even though the $(Actual_i - Expected_i)$ multiplier is small).
- **Standard fix: additive/Laplace smoothing** — add a small epsilon (e.g., 0.0001 or 1 pseudo-observation) to every bin's count before computing proportions, so no bin is ever exactly zero. This is a real, necessary implementation detail, not an edge case to hand-wave past — new categorical values showing up in production is one of the *most common and most genuinely important* drift signals (e.g., a new product SKU, a new device type, a new country of origin), so silently crashing or NaN-ing on it defeats the entire purpose of monitoring.

**Reference window choice:**
- Comparing against the original **training set** distribution answers "has production drifted from what the model was trained on" (most directly relevant to model validity).
- Comparing against a **rolling recent window** (e.g., last 7 days vs. previous 7 days) instead answers "is there sudden change happening right now," which is a different and complementary question — good for catching abrupt pipeline breaks or sudden real-world shocks, even if the model is still broadly aligned with its original training distribution.
- Best practice in mature monitoring setups: track **both** — drift-from-training (model validity signal) and drift-week-over-week (operational/pipeline-health signal) — because they catch different failure modes.

**Per-feature vs. aggregate monitoring:**
- Compute PSI/KL **per feature**, not just on model output/prediction distribution — output-distribution monitoring alone can mask feature-level shifts that happen to cancel out in aggregate (two features drifting in offsetting directions can leave the output distribution looking stable while individual features have genuinely moved out of the training support region).
- Also worth tracking PSI on the **model's predicted score/probability distribution** itself as a cheap, single-number summary alarm, complementary to (not a replacement for) per-feature monitoring — it's fast to compute and a reasonable first-pass alert trigger before drilling into which specific feature moved.

## 7. Pitfalls / Trick Angles

1. **"PSI = 0.3, so the model must be broken."** PSI/KL flag *distributional* change, not necessarily *model performance* degradation. A feature can shift substantially in a region the model was never sensitive to (low feature importance there), producing a high PSI with zero actual performance impact. Always pair distributional monitoring with (delayed, but eventual) performance monitoring — PSI is a leading indicator/triage tool, not a proof of model failure on its own.

2. **Using raw KL divergence as if it were a true "distance" for ranking or thresholding symmetrically.** Because $D_{KL}(P\parallel Q) \neq D_{KL}(Q\parallel P)$, picking the "wrong" direction can materially change your number and even your conclusion about which distribution has "more surprise" relative to the other. If you need a genuine symmetric distance-like drift score, use PSI or Jensen-Shannon, not raw KL, and be explicit about direction if you do use KL.

3. **Forgetting to freeze bin edges from the baseline.** If you recompute bin edges fresh on each new time window's data (rather than reusing baseline-derived edges), you're comparing two different binning schemes — this silently corrupts the whole comparison and can produce artificially low PSI even under real drift (because the bins "adapt" to wherever the new data actually is).

4. **Not smoothing zero-count bins, and either crashing or silently reporting infinity/NaN.** As covered in §6 — this is a very frequent, very avoidable production bug, and interviewers like probing whether you've actually implemented one of these metrics for real (versus only knowing the formula).

5. **Applying PSI/KL to high-cardinality categorical features (e.g., ZIP code, user ID) without bucketing.** Thousands of near-empty "bins" produce a noisy, inflated, largely meaningless PSI dominated by sampling noise in rare categories rather than genuine drift. Group rare categories into an "other" bucket first, or use a different technique (e.g., comparing top-K category frequency ranks) for very high-cardinality fields.

6. **Treating PSI thresholds (0.1/0.25) as portable across every feature and every business context without calibration.** As stressed in §2 — these are conventions from one industry (credit scoring), not universal statistical cutoffs; a defensible interview answer states them *and* immediately caveats that the right threshold should be tuned per feature/problem based on cost of false alarms vs. cost of missed drift.

7. **Monitoring only the label/output distribution and not inputs.** Output-distribution monitoring is cheap and useful but is a lagging, aggregate signal — by the time it moves, individual features may have been drifting for a while; and as noted in §6, offsetting feature-level shifts can hide entirely from output-only monitoring.

8. **Ignoring seasonality when interpreting a rolling-window PSI spike.** A PSI spike comparing "this week" to "last week" right around a holiday or a known seasonal pattern is expected, recurring, and often not actionable — conflating it with genuine unexpected drift (see the "recurring drift" concept from the broader distribution-shift notes) leads to noisy false alarms and alert fatigue, which erodes trust in the monitoring system over time.

## 8. Interview Q&A

**Q1: Why is PSI symmetric but raw KL divergence is not?**
Because PSI is explicitly constructed as the sum of both directional KL-like terms — $D_{KL}(Actual\parallel Expected) + D_{KL}(Expected\parallel Actual)$ — so swapping which distribution you call "Expected" vs. "Actual" doesn't change the total. Raw KL divergence only computes one direction, and because $\ln(P/Q) \neq \ln(Q/P)$ in general (they're negatives of each other but weighted by different, non-equal probability masses $P$ vs. $Q$ in the outer sum), the two directions give genuinely different numbers.

**Q2: A brand-new product category appears in this week's traffic that didn't exist in your training data. What happens to your KL divergence calculation, and how do you fix it?**
Raw KL divergence becomes undefined/infinite, because the baseline probability for that bin is exactly zero, and $\ln(x/0)$ diverges. The standard fix is additive (Laplace) smoothing — adding a small pseudo-count to every bin, including previously-unseen ones, before computing proportions, so no denominator is ever exactly zero. Also worth noting this scenario is itself an important, genuine drift signal (a new category existing at all is meaningful) — the smoothing fix should make the metric computable, not suppress the signal.

**Q2b: Would PSI have the same problem?**
Yes, in practice — the $\ln(Actual_i/Expected_i)$ term is equally undefined/unstable at $Expected_i = 0$, even though the linear $(Actual_i - Expected_i)$ prefactor might be small. The same smoothing fix applies to PSI's computation.

**Q3: You compute PSI on your model's output probability distribution and it's stable (< 0.05), but per-feature PSI shows two features individually flagged as major shift (> 0.25). How do you explain this to a stakeholder who's confused why the "important" aggregate number looks fine?**
The two features are likely shifting in ways that happen to offset each other in their combined effect on the model's output (e.g., one feature pushing predictions up, another pushing down, net effect near zero on the output distribution) — or the shifted features simply have low importance/influence on this particular model's predictions, so the input-level shift doesn't propagate to the output. This is exactly why per-feature monitoring is necessary in addition to output-level monitoring — output stability is a *lagging, aggregate* signal that can mask real underlying movement, and even if today's net effect is benign, it's worth understanding whether it's incidental (offsetting effects that might not offset next month) or structural (feature is genuinely low-importance).

**Q4: When would you choose Jensen-Shannon divergence over PSI for drift monitoring?**
When you want a metric that's always finite and mathematically well-behaved without needing ad-hoc smoothing conventions (JS divergence's use of the mixture distribution $M = \frac{1}{2}(P+Q)$ as an intermediate reference guarantees no zero-probability blowup by construction), and when you're working in a more research/ML-native context where PSI's credit-scoring-derived thresholds (0.1/0.25) feel like an awkward fit for communicating results — e.g., comparing embedding distributions or output softmax distributions in a deep learning pipeline, where JS divergence (or its square root, the Jensen-Shannon *distance*, which is a true metric) is more standard.

**Q5: True or false — a low PSI on all monitored features guarantees your model hasn't experienced concept drift.**
False, and an important gotcha. PSI (and KL) only detect changes in the *input* distribution $P(X)$ (or, if applied to labels, $P(Y)$) — they say nothing about whether the relationship $P(Y\mid X)$ itself has changed. A model can face severe concept drift (the same inputs now genuinely mapping to different outcomes — e.g., adversarial adaptation in fraud) while the input distribution looks completely stable by every PSI/KL measure, because concept drift is fundamentally undetectable from unlabeled inputs alone (see the broader distribution-shift notes, §6). PSI/KL monitor covariate shift; catching concept drift requires actual labels, however delayed.

**Q6: How many bins should you use for PSI, and does it matter?**
Convention is ~10 quantile-based bins computed from the baseline distribution and frozen going forward. It does matter: too few bins wash out genuine shape changes in the distribution (a shift concentrated in one region gets diluted across a wide bin and doesn't show up); too many bins, especially with a smaller current-window sample size, inflates noise — small random count fluctuations in near-empty bins produce spuriously high PSI even with no real drift. The right choice should scale with how much traffic/data volume you have per monitoring window — a feature evaluated on 100 samples/day shouldn't be binned as finely as one with 1M samples/day.

**Q7 (clever): Can PSI be exactly zero even if the two distributions are visibly different when plotted?**
Yes, in a specific pathological sense — if your binning is coarse enough that both distributions happen to have identical proportions *within each bin* even though the underlying shape differs *within* a bin (e.g., a distribution that shifted its internal skew but kept the same total mass in each of your 10 quantile buckets), PSI computed on that binning would report ~0 despite genuine (finer-grained) drift having occurred. This is a real limitation of any binned/histogram-based divergence measure — it's only as sensitive as your binning resolution, which is why binning-strategy choices (§6) aren't just an implementation detail but actually bound what kinds of drift the metric *can* detect at all.
