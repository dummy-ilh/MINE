--

## PART A: Rapid-Fire Conceptual Q&A (30+ Questions)

**1. Why can't you use linear regression for classification?**
Output isn't bounded to [0,1], violates residual normality assumptions for a 0/1 target, and can't capture the natural flattening of probability near the extremes.

**2. What is a link function?**
A translator between bounded probability space (0,1) and unbounded straight-line space (−∞,∞), enabling linear modeling of a bounded quantity.

**3. What is the logit?**
log(odds) = log(p/(1-p)) — the specific link function logistic regression uses.

**4. What is the sigmoid function and why is it used?**
1/(1+e^-z) — it's the inverse of the logit link, mapping any real number back into (0,1).

**5. Write the sigmoid's derivative.**
sigmoid(z) × (1 - sigmoid(z)), or p(1-p) — clean because it's expressed in terms of its own output.

**6. What does a logistic regression coefficient mean?**
A 1-unit increase in the feature adds w to the log-odds, holding other features constant. Exponentiating (e^w) gives the odds ratio — the multiplicative effect on odds.

**7. Coefficient = 0.5. What's the odds ratio?**
e^0.5 ≈ 1.65 — a 1-unit increase multiplies the odds by ~1.65x.

**8. Why not use MSE as the loss function?**
Non-convex when paired with sigmoid — risks getting stuck in local minima. Log-loss is convex for logistic regression.

**9. What is Maximum Likelihood Estimation, in plain terms?**
Choosing the parameters that make the observed data most probable under the model.

**10. What's the relationship between log-loss and MLE?**
Log-loss is negative log-likelihood — minimizing log-loss ≡ maximizing likelihood.

**11. Why does log-loss punish confident-wrong predictions so harshly?**
-log(p) approaches infinity as p→0 — the penalty explodes for confidently wrong predictions, not just increases linearly.

**12. State the gradient descent update rule.**
new_weight = old_weight − (learning_rate × gradient).

**13. What's the gradient formula for logistic regression?**
Average of (p−y)×x across data points — clean because sigmoid and log-loss derivatives combine nicely.

**14. Cost is oscillating during training — what's wrong?**
Learning rate likely too high — reduce it.

**15. Batch vs. stochastic vs. mini-batch gradient descent?**
Batch = full dataset per update (accurate, slow). SGD = one point per update (fast, noisy). Mini-batch = small random subset (industry standard).

**16. Why isn't 0.5 always the right threshold?**
It assumes false positives and false negatives cost the same — rarely true. Threshold should reflect actual business cost of each error type.

**17. Why is logistic regression's decision boundary linear?**
z is a linear combination of features; setting z=0 (at threshold 0.5) produces a straight-line equation.

**18. How would you get a non-linear decision boundary from logistic regression?**
Add engineered non-linear features (polynomial or interaction terms) — the boundary becomes curved in original feature space even though it's still "linear" in the new features.

**19. Why does unregularized logistic regression overfit?**
Gradient descent will assign large weights to any feature that reduces training loss, including pure noise/coincidence in the training data.

**20. Difference between L1 and L2 regularization?**
L2 (Ridge) squares weights — smooth shrinkage, rarely exactly zero. L1 (Lasso) uses absolute value — tends to zero out weak features (automatic feature selection).

**21. Why does L1 produce sparsity but L2 doesn't?**
Squaring a small weight gives a tiny penalty (diminishing pressure); absolute value keeps constant proportional pressure regardless of size, pushing weak weights all the way to zero.

**22. λ is set extremely high — what happens?**
All weights shrink toward zero, model approaches predicting a constant probability regardless of input — underfitting.

**23. Why is accuracy a bad metric for imbalanced data?**
A model that always predicts the majority class can score 99%+ accuracy while catching 0% of the rare class.

**24. Precision vs. recall — formulas and when each matters?**
Precision = TP/(TP+FP), matters when false positives are costly. Recall = TP/(TP+FN), matters when false negatives are costly.

**25. Why harmonic mean for F1, not a simple average?**
Harmonic mean is pulled down heavily by the smaller value — prevents one very high metric from masking a very low one.

**26. ROC-AUC vs PR-AUC — when to use which?**
ROC-AUC fine for balanced classes. PR-AUC preferred for heavily imbalanced data, since ROC-AUC's FPR denominator gets dominated by a huge TN count, making it look deceptively good.

**27. What does AUC = 0.5 mean? AUC = 0.3?**
0.5 = no better than random guessing. 0.3 = systematically backwards — flip the predictions and you'd get 0.7. Only 0.5 is truly uninformative.

**28. What is calibration, and how does it differ from AUC?**
Calibration measures whether predicted probabilities match observed frequencies (among 70% predictions, does the event happen ~70% of the time). AUC measures ranking only — a model can rank well but be poorly calibrated.

**29. What does logistic regression assume about the relationship between features and the outcome?**
Linearity in the log-odds (not probability, not raw outcome) — precise wording matters here.

**30. What's the difference between the independence assumption and the multicollinearity assumption?**
Independence is about rows (observations shouldn't be correlated with each other). Multicollinearity is about columns (features shouldn't be too correlated with each other).

**31. Does multicollinearity hurt predictions or coefficients?**
Mainly coefficients — individual weights become unstable/erratic (can even flip sign), but overall predictions often remain fine.

**32. How do you extend logistic regression to multiclass problems?**
One-vs-Rest (train N independent binary classifiers, pick highest) or Softmax regression (one unified model, probabilities guaranteed to sum to 1).

**33. Why does softmax use exponentiation?**
Raw z-scores can be negative — exponentiating guarantees positivity before normalizing into valid probability shares.

**34. How is softmax related to sigmoid?**
Sigmoid is softmax's special case with exactly 2 classes.

**35. Why would you choose logistic regression over gradient boosting?**
Interpretability (clean odds-ratio coefficients), latency (fast inference), and baseline value (cheap first pass to measure how much lift a complex model actually buys).

**36. Logistic regression is mathematically equivalent to what neural network structure?**
A single neuron, single layer, with sigmoid activation.

**37. Why do stacked layers need non-linear activation functions?**
Without non-linearity, stacked linear layers mathematically collapse into one equivalent linear layer — depth buys nothing.

---

## PART B: "Explain to a PM" Questions

**1. "Why does the model say a customer has an 80% chance of churning, but you said the coefficient for complaints was only 0.8 — shouldn't 80% and 0.8 be more related?"**

*Model answer:* "The 0.8 isn't measured on the probability scale — it's measured on something called log-odds, which is a technical, unbounded number the model works with internally. It's not a percentage or a probability itself. What the 0.8 tells us is: each additional complaint makes churning about 2.2 times more likely relative to not churning, in odds terms. The 80% probability is the final, human-readable output after we translate that internal number back into something we can act on. They're related, just not on a scale where you can directly compare 0.8 to 80%."

**2. "Why can't we just say 'above 50% means they'll churn' and be done with it?"**

*Model answer:* "50% assumes that missing a churner and wrongly flagging a loyal customer cost us exactly the same amount — and they usually don't. If a wrongly-flagged loyal customer just gets an unnecessary discount offer (cheap), but a missed churner means losing their full annual revenue (expensive), we should be willing to flag more people as at-risk, even if we're less than 50% sure, because the cost of missing a real churner is so much higher than the cost of a wasted offer."

**3. "The model is 97% accurate — why are you telling me it's not good enough?"**

*Model answer:* "If only 1% of our customers actually churn, a model that predicts 'nobody will churn' would already be 99% accurate — while being completely useless, since it identifies zero at-risk customers. Accuracy hides how well we're doing on the rare cases that actually matter. I'd rather show you: of everyone we flagged as at-risk, how many really were at-risk (precision), and of everyone who actually churned, how many did we catch (recall) — those numbers tell the real story."

**4. "Why do we need to keep checking on this model after it's launched? Didn't we already test it?"**

*Model answer:* "Customer behavior changes over time — a model trained on last year's patterns can quietly become less accurate as the world shifts, without any code breaking or any alarms going off on their own. We monitor it the same way you'd periodically recheck a forecast against actual results — if predictions start drifting from reality, we catch it early and retrain, instead of discovering it months later through a customer complaint or lost revenue."

---

## PART C: Take-Home-Style Case Questions

**Case 1: Design a fraud detection system using logistic regression as a baseline.**

*Strong answer structure:*
- **Features:** transaction amount, transaction velocity (count/sum over recent time windows), device/location consistency with account history, time-of-day patterns. Apply monotonic transforms (log(amount)) and binning where relationships are non-linear.
- **Class imbalance:** fraud is rare — start with class weights (cheap, no data loss), consider SMOTE oversampling if needed.
- **Metric:** PR-AUC as primary offline metric, given severe imbalance (ROC-AUC would look artificially good).
- **Threshold:** set based on the real cost ratio of missed fraud (high) vs. false alarms (lower, but not negligible — customer friction has a cost too). Likely biased toward a lower threshold given fraud's asymmetric cost.
- **Monitoring:** drift (fraud patterns evolve adversarially — fraudsters adapt to detection), calibration decay (fraud base rate shifts seasonally, e.g., holiday spikes).
- **Baseline justification:** fast, interpretable, cheap to serve at scale (needed for real-time transaction blocking) — explicit plan to benchmark against gradient boosting later via a proper A/B test, not just offline comparison.

**Case 2: Your company wants to replace a logistic regression credit approval model with a deep learning model that has a 3-point higher AUC. Do you recommend the switch?**

*Strong answer structure:* Not automatically yes. Key considerations: (1) **Regulatory interpretability** — credit decisions often legally require explainable reasoning ("why was this application denied"); logistic regression's odds ratios are directly defensible, a deep model is much harder to explain even with tools like SHAP. (2) **Magnitude of gain** — is 3 AUC points a meaningful real-world lift, or within noise/measurement variance? (3) **Segment-level check** — does the new model improve avg. performance while quietly hurting a protected subgroup (fairness/compliance risk)? (4) **Calibration** — better ranking doesn't guarantee better calibration; credit limits/pricing may depend on the actual probability value, not just ranking. (5) **Process** — recommend a proper A/B test with gradual rollout and rollback capability before any full replacement, rather than an offline-metric-only decision.

**Case 3: A logistic regression churn model performed well in offline validation (AUC 0.85) but the retention team says targeted customers "don't seem right" after 2 months in production. Diagnose the issue.**

*Strong answer structure:* Start by separating two possible failure modes: **ranking failure** (re-check AUC on recent live data — has it actually dropped?) vs. **calibration failure** (re-check the calibration curve — are predicted probabilities still matching real outcomes?). Also check for **data drift** (have input feature distributions shifted since training — e.g., new customer segments, behavior changes?) and **training-serving skew** (are features computed identically in production as they were during training — a very common, sneaky bug). Also consider whether the **threshold** was tuned for outdated conditions and needs recalibration given a shifted base rate. Walk through each systematically rather than jumping to "retrain from scratch" as the first move.

---

## Common Trick Questions and the Traps Within Them

**Trick 1:** *"Doesn't a coefficient of 0 mean the feature doesn't matter?"*
Trap: true in an unregularized model roughly, but in L1-regularized models, a coefficient of exactly 0 could also mean the feature was correlated with another feature that "won" the competition for that signal (multicollinearity), not necessarily that the feature has zero true relationship with the outcome.

**Trick 2:** *"If AUC is 0.5, does that mean the model is useless?"*
Trap: yes, 0.5 truly is uninformative — but candidates sometimes wrongly generalize this to "any AUC far from 1.0 is bad," missing that an AUC of 0.2 or 0.3 is actually informative (systematically backwards, exploitable by flipping predictions).

**Trick 3:** *"Since logistic regression is 'simple,' is it always outperformed by more complex models?"*
Trap: not necessarily — if the true relationship is genuinely close to linear in the log-odds, a complex model may add negligible lift while costing more in latency, interpretability, and maintenance. "Simple" doesn't mean "worse," it means "different tradeoffs."

**Trick 4:** *"Higher regularization always improves generalization, right?"*
Trap: no — too much regularization causes underfitting. It's a bias-variance tradeoff with a genuine sweet spot, not a monotonic "more is better" relationship.

**Trick 5:** *"A confidence interval on a coefficient tells you the probability the true value falls in that range, right?"*
Trap: this is the classic frequentist-vs-Bayesian confusion — a (frequentist) confidence interval means "if we repeated this sampling process many times, X% of such intervals would contain the true value," not "there's an X% probability the true value is in THIS specific interval." A subtle but real distinction some interviewers probe.

---

That's the full curriculum, Module 0 through 13. Your cheat sheet now covers every module — want me to append this final Module 13 drill section as well, so you have the complete rapid-fire reference in one document for last-minute review?
