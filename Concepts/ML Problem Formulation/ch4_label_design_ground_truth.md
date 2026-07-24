# Chapter 4: Label Design & Ground Truth

## 1. Why This Chapter Exists

Chapter 3 nailed down *when* a label is measured (windows) and *what unit* it's attached to. This chapter is about *where the label actually comes from* — is it something a human explicitly said, or something inferred from behavior? Is it immediate or delayed? Is it trustworthy? Every production ML system inherits the noise and bias of its labeling process, and interviewers probe this because it's where "clean-looking" ML problems turn out to have messy real-world label sources.

## 2. Implicit vs. Explicit Labels

**Explicit labels**: a human deliberately provides the label as its own action — a star rating, a thumbs up/down, a human-reviewed fraud adjudication, a survey response. Pros: directly measures what you want. Cons: sparse (most users don't rate), subject to selection bias (only people with strong opinions rate), and expensive to collect at scale.

**Implicit labels**: inferred from behavior that wasn't intended as a label — click, dwell time, add-to-cart, scroll depth, replay. Pros: abundant, free, always-on. Cons: is a noisy, indirect proxy for the true latent preference (Chapter 1's click-vs-watch-time problem is exactly an implicit-label proxy issue), and is influenced by the very system that's serving content (position bias, prior-model bias — Chapter 11 territory).

### Worked Numerical Example: Comparing Label Sources for "Content Quality"

Suppose true content quality (unobservable) is $q \in [0,1]$ for 4 articles, and we have two label sources:

| Article | True quality $q$ | Explicit rating (1–5 stars, n=8 raters) | Implicit: click-through rate (n=50,000 impressions) |
|---|---|---|---|
| A | 0.90 | 4.8 avg (n=8) | 0.031 |
| B | 0.85 | 4.6 avg (n=6) | 0.089 |
| C | 0.40 | 2.1 avg (n=3) | 0.061 |
| D | 0.20 | — (n=0, no ratings) | 0.045 |

Two things jump out. First, explicit ratings track true quality reasonably well *where they exist* (A > B > C), but article D has **zero explicit labels** — nobody bothered to rate low-quality content, so explicit-only labeling silently drops your worst examples from the training set entirely (a missing-not-at-random problem, not just noise). Second, implicit CTR ranks the articles B > C > D > A — completely inverted relative to true quality, likely because article B has a sensational headline (high click, but the earlier ratings suggest lower delivered satisfaction than A). This numeric mismatch is the concrete version of "implicit labels are a noisy, biased proxy" — it's not just noisy, it can be *systematically* biased in a specific, headline-exploitable direction.

**Practical implication**: many production systems combine both — using sparse-but-reliable explicit labels to calibrate or de-bias a model trained primarily on abundant implicit signals (e.g., using explicit ratings as a held-out validation signal even when implicit signals are the primary training source), rather than picking one exclusively.

## 3. Delayed Feedback

Some labels are known instantly (did the user click — yes, within milliseconds). Others mature over days, weeks, or months (did the loan default — up to years; was this transaction fraud — the chargeback window is typically 60–120 days; did the user churn — Chapter 3's window).

**The formal issue**: if you train only on examples whose label has already matured, you are systematically excluding the most recent data — which is exactly the data most representative of current user behavior and current adversarial tactics (in fraud, attackers adapt fast; a model trained only on 90-day-stale labels is 90 days behind the newest attack pattern).

**A concrete mitigation pattern — surrogate/proxy early labels.** Define a fast proxy label (e.g., "flagged by a rules engine or reported by the merchant within 3 days") to enable near-real-time retraining, alongside the slow, high-fidelity true label (confirmed chargeback at 90 days) for periodic recalibration/evaluation. This is the direct production application of Chapter 3 Section 4's freshness-vs-fidelity tradeoff, applied specifically to label sourcing rather than just window length.

## 4. Noisy Labels: Quantifying and Mitigating

Suppose your true label is $y^* \in \{0,1\}$ but your observed label $y$ has a known or estimated noise rate: $P(y = 1 \mid y^* = 0) = \alpha$ (false positive rate in labeling) and $P(y=0 \mid y^*=1) = \beta$ (false negative rate in labeling). A classifier trained naively on noisy $y$ is effectively learning:

$$
P(y=1 \mid x) = (1-\beta) \cdot P(y^*=1\mid x) + \alpha \cdot P(y^*=0 \mid x)
$$

**Numerical illustration**: suppose true positive rate in the population is $P(y^*=1) = 0.05$ (5% of transactions are truly fraud), and your labeling process (say, a rules-based auto-flagging system used as ground truth) has $\alpha = 0.02$ (2% false-flag rate on legitimate transactions) and $\beta = 0.30$ (30% of true fraud slips through unflagged). Then:

$$
P(y=1) = (1-0.30)(0.05) + (0.02)(0.95) = 0.035 + 0.019 = 0.054
$$

Your *observed* positive rate (5.4%) looks deceptively close to the true rate (5%), which could mask the fact that 30% of actual fraud cases are being trained as negatives (label noise concentrated specifically among the hardest, most novel fraud patterns the rules engine doesn't catch — precisely the cases you most want the ML model to learn to catch). This is why "the observed base rate looks reasonable" is not sufficient evidence that a labeling process is trustworthy — you need to separately estimate $\alpha$ and $\beta$, e.g., via a small gold-standard human-audited sample.

**Mitigation approaches** (each with a real cost):
- **Human audit sampling**: periodically send a random (not just flagged) sample to expert review to estimate $\alpha, \beta$ directly and, if severe, apply noise-correction techniques (e.g., loss correction using the estimated noise transition matrix).
- **Consensus/multi-rater labeling**: for explicit labels, use multiple independent raters and require agreement (or model inter-rater reliability, e.g., Cohen's kappa) rather than trusting a single label source.
- **Weak supervision / programmatic labeling**: combine several noisy heuristic labeling functions (e.g., Snorkel-style) and model their individual noise rates jointly rather than trusting any single one as ground truth.

## 5. Selection Bias in Label Collection

Labels collected only from users who take some voluntary action (rating, reviewing) are conditioned on that action, which is rarely random. Formally, you observe $y$ only for the subpopulation where (using ratings as an example) "chose to rate" $= 1$, and if that indicator correlates with the true outcome (people with extreme experiences — very good or very bad — rate far more than people with mediocre ones), your label distribution is a biased sample of the true outcome distribution, not a representative one. This is distinct from noisy labels (Section 4) — here the *labels you do have* might be accurate, but *which examples get labeled at all* is non-random, which biases anything trained purely on the labeled subset.

## 6. Production Considerations

- **Label pipelines need monitoring like any other production system.** A silent change in an upstream labeling heuristic (e.g., the rules engine in the fraud example gets updated) changes $\alpha, \beta$ without any code change in your model pipeline — track label-source drift as its own monitored metric.
- **Delayed feedback requires an explicit "label maturity" join key in your data pipeline** — training jobs need to correctly exclude examples whose label window hasn't elapsed yet, or you silently reintroduce Chapter 3's leakage/immaturity problems.
- **Human labeling budgets are a real, finite resource** — prioritizing which examples get human review (e.g., active-learning-style, sending uncertain/borderline model predictions to human review) stretches a fixed labeling budget further than random sampling.
- **Weak-supervision label sources need versioning** — if you combine 5 heuristic labeling functions, changes to any one of them retroactively change what "the label" meant for historically trained models, which complicates any before/after comparison.

## 7. Common Interview Traps

- **Treating "we have data" as equivalent to "we have ground truth."** Implicit signals are data, not automatically a faithful label — Section 2's inverted CTR-vs-quality example is the canonical rebuttal to that assumption.
- **Not distinguishing label noise (Section 4) from selection bias (Section 5).** These require different mitigations (noise correction vs. re-weighting/counterfactual correction) and conflating them is a common giveaway of surface-level understanding.
- **Assuming a rules-engine or heuristic label source is ground truth just because it's already in production** — as the fraud example shows, an existing automated labeling system can itself have substantial, quantifiable error rates.
- **Ignoring delayed feedback entirely and assuming all labels are available instantly** — an easy gap to expose by asking "when exactly do you know this label is correct?"

## 8. L5-Differentiating Talking Points

- Explicitly name the noise-rate parameters ($\alpha$, $\beta$) when discussing a labeling process, and propose a concrete way to estimate them (audit sampling) rather than treating label quality as a binary "trustworthy/not."
- Distinguish selection bias in *which examples get labeled* from noise in *the labels themselves* — these are different failure modes with different fixes, and naming both shows depth.
- Propose a dual-cadence labeling strategy (fast proxy + slow high-fidelity label) for any target with meaningfully delayed ground truth, tying it back to the freshness/fidelity tradeoff introduced in Chapter 3.
- Flag that label-source changes (a rules engine update, a UI change affecting implicit signal collection) are a recurring operational risk requiring monitoring, not a one-time setup concern.

## 9. Comprehension Checks

1. For a video platform relying on "watch time" as an implicit label for content quality, describe one plausible selection-bias mechanism (Section 5) and one plausible noise mechanism (Section 4) that could each corrupt this label, and explain why they'd require different fixes.
2. Using the noise-rate formula in Section 4, if $\alpha = 0.05$ and $\beta = 0.10$ with true base rate $P(y^*=1) = 0.10$, compute the observed positive rate $P(y=1)$. Is it close to the true rate, and does that closeness tell you anything about whether the individual noise rates are acceptable?
3. A team proposes using "items added to cart" as ground truth for a "purchase intent" model, arguing it's more abundant than actual purchases. Identify one way this implicit label could diverge from true purchase intent, using the reasoning style from Section 2's table.
4. Design a dual-cadence labeling scheme (fast proxy + slow ground truth) for a content-moderation model where the true label (human policy review outcome) takes 5 days but you want daily model refresh. What fast proxy would you use, and what's the specific risk of relying on it alone long-term?

---

*End of Chapter 4. Chapter 5 will cover offline metric selection — matching accuracy/precision/recall/AUC/NDCG/RMSE to task type and to business cost asymmetries.*
