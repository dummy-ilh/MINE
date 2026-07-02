
# Module 11 — Production & System Design Angle (L5-Critical)

## 1. WHY

Everything so far has been about building and understanding the model itself. But at the L5/ICT5 level, interviewers don't just want to know you can build a model — they want to know you can **operate one in a real production system**, over time, at scale, with real engineering tradeoffs. This module is where "I understand logistic regression" becomes "I could be trusted to own a production ML system." This is explicitly flagged as critical in your curriculum, so let's be thorough.

## 2. FEATURE ENGINEERING FOR LOGISTIC REGRESSION

**WHY this needs special attention (vs. tree-based models):** Logistic regression can ONLY express a straight-line relationship with log-odds (Module 10). Tree-based models (like gradient boosting) can automatically discover non-linear patterns and interactions on their own. Logistic regression can't — **you, the engineer, have to hand it those patterns explicitly**, through feature engineering.

**Monotonic transforms:** if a feature's true relationship with log-odds is curved but consistently increasing or decreasing (monotonic, even if not perfectly straight), applying a transform like `log(x)` or `sqrt(x)` before feeding it into the model can straighten out that relationship enough for logistic regression to capture it well. Example: "income" often has a highly skewed distribution; `log(income)` frequently has a more linear relationship with log-odds of many outcomes than raw income does.

**Binning:** convert a continuous feature into discrete buckets (e.g., tenure → "0-6 months," "6-12 months," etc., like our Module 10 example) and use one-hot/dummy encoding for each bucket. This lets the model assign a COMPLETELY independent coefficient to each bucket, capturing non-linear, even non-monotonic patterns (like a U-shape) — at the cost of losing the smooth, continuous nature of the original feature and requiring more data to estimate each bucket's coefficient reliably.

## 3. HANDLING CLASS IMBALANCE

We already established WHY this matters in Module 8 (accuracy trap with rare positive classes). Now, the practical toolkit for actually fixing it:

**Class weights:** tell the model to treat mistakes on the rare class as more "costly" during training — effectively, misclassifying one fraud case gets penalized as if it were, say, 100 mistakes on the non-fraud class (if fraud is roughly 1% of data). This directly modifies the log-loss calculation from Module 4, weighting each data point's contribution to the loss based on its class.
- **Pro:** simple, doesn't require creating/removing any actual data points, easy to implement (`class_weight='balanced'` in most libraries).
- **Con:** doesn't add any new information — it just tells the optimizer to care more about the rare class, which can sometimes lead to more false positives if pushed too aggressively.

**Resampling — oversampling the minority class:** duplicate (or synthetically generate, via a technique called SMOTE) more examples of the rare class so the training set is more balanced.
- **Pro:** can help the model "see" the minority class pattern more clearly.
- **Con:** naive duplication can cause overfitting to those specific duplicated examples; SMOTE (synthetic examples) helps but adds complexity and can create unrealistic synthetic data points if not done carefully.

**Resampling — undersampling the majority class:** randomly remove examples from the common class until the classes are more balanced.
- **Pro:** simple, reduces training time (smaller dataset).
- **Con:** you're literally throwing away real data — potentially losing useful information/patterns present only in the discarded majority-class examples.

**Practical L5 guidance:** class weights are usually the first thing to try (cheap, no data loss); resampling is considered when class weights alone aren't sufficient, understanding that oversampling/undersampling both carry real tradeoffs, not free wins.

## 4. MONITORING A LOGISTIC REGRESSION MODEL IN PRODUCTION

A model doesn't stop needing attention once it's deployed — the real world keeps changing, and a model trained on last year's data can silently degrade. Two key things to monitor:

**Drift:** the statistical properties of your input data change over time compared to what the model was trained on. Example: a churn model trained pre-pandemic might see a totally different distribution of "months since last purchase" post-pandemic, as customer behavior patterns shifted. **How to catch it:** monitor the distribution of each input feature over time (e.g., compare weekly feature distributions to the training distribution using statistical tests), and alert if they diverge significantly.

**Calibration decay:** even if the model's RANKING ability (AUC) stays stable, its probability calibration (Module 8) can degrade over time as the underlying base rate of the event changes. Example: if fraud rates naturally rise during a holiday shopping season, a model trained on non-holiday data will systematically UNDER-predict fraud probability during that period, even if it still ranks risky transactions above safe ones correctly. **How to catch it:** periodically recompute the calibration curve (Module 8) on fresh, recent data and compare it to the original training-time calibration curve.

## 5. WHEN TO CHOOSE LOGISTIC REGRESSION OVER GRADIENT BOOSTING / DEEP MODELS

This is a genuinely common, high-value L5 interview question: **"why would you use something as 'simple' as logistic regression when more powerful models exist?"**

**Interpretability:** logistic regression's coefficients have a clean, explainable meaning (odds ratios, Module 3) that regulators, stakeholders, and even the model-builders themselves can reason about directly. This matters enormously in regulated industries (credit lending, insurance, healthcare) where "why did the model deny this loan?" needs a clear, defensible answer — gradient boosting/deep models are much harder to explain this precisely (though tools like SHAP help narrow the gap).

**Latency:** logistic regression inference is just one matrix multiplication plus a sigmoid — extremely fast, trivially cheap to compute even at massive scale or on resource-constrained devices (e.g., real-time bidding systems needing sub-millisecond decisions). Gradient boosting and deep models can be meaningfully slower, especially at high query volumes.

**Baseline value:** logistic regression is fast to build, fast to train, and gives you a credible performance floor almost immediately. It's standard practice to build a logistic regression baseline FIRST, even if you expect to eventually deploy something more complex — this tells you how much lift a fancier model is actually buying you, and sometimes the honest answer is "not much," in which case the simpler, more interpretable, cheaper-to-maintain model wins on pure practicality.

**When more complex models win:** when the true relationship between features and the outcome is genuinely non-linear/has complex interactions that would require extensive manual feature engineering to capture with logistic regression, and when interpretability/latency constraints are less strict.

## 6. A/B TESTING CONSIDERATIONS WHEN REPLACING A LOGISTIC REGRESSION MODEL

**WHY this matters:** even if a new model (say, a gradient boosting replacement) shows better offline metrics (higher AUC, better F1), you should almost never fully replace a production model overnight without validating it live — offline metrics don't always translate perfectly to real-world business impact.

**Key considerations:**
- **Metric alignment:** make sure the offline metric you optimized (e.g., AUC) actually correlates with the real business metric you care about (e.g., revenue, retention) — these can diverge.
- **Threshold recalibration:** a new model likely needs its own newly-tuned threshold (Module 6) — you can't necessarily reuse the old model's threshold, since the new model's probability outputs may behave differently even if overall ranking quality improved.
- **Calibration comparison:** check whether the new model is equally well-calibrated (Module 8) — a model with better ranking (AUC) but worse calibration could still hurt decisions that depend on the raw probability value, not just the ranking.
- **Segment-level analysis:** a new model might improve performance on average while quietly hurting a specific important subgroup (e.g., new users, a specific region) — always check performance isn't just improving in aggregate while masking a regression somewhere important.
- **Gradual rollout:** ramp the new model to a small percentage of traffic first, monitor real business metrics (not just ML metrics) before full rollout, and always keep the ability to roll back quickly if something looks wrong.

## 7. INTERPRETATION

In real terms: at the L5 level, you're expected to think past "does the model work on my validation set" and into "how does this model behave, get monitored, and evolve responsibly once it's live and touching real customers/revenue." This is genuinely the difference between a strong IC4/mid-level answer and an L5/senior-level answer in interviews — depth on the modeling math alone isn't enough; you need to demonstrate you've thought about the full lifecycle.

## 8. FAANG L5 ANGLE

**Common interview question:** *"Design a fraud detection system using logistic regression as a baseline. Walk me through your approach."*
Strong answer structure: (1) feature engineering — transaction amount, velocity of transactions, device/location features, likely needing monotonic transforms and binning; (2) handle severe class imbalance — likely class weights first, given fraud's typical rarity; (3) choose PR-AUC over ROC-AUC as the primary offline metric given the imbalance (Module 8); (4) threshold selection driven by the real cost of false negatives (missed fraud) vs false positives (customer friction), likely erring toward a lower threshold (Module 6); (5) production monitoring for drift and calibration decay, since fraud patterns evolve constantly and adversarially; (6) explicitly justify logistic regression as a fast, interpretable, cheap-to-maintain baseline — with a plan to compare against gradient boosting once the baseline is solid, using a proper A/B test before any full replacement.

**Common follow-up:** *"Your logistic regression baseline has an AUC of 0.85. A gradient boosting model gets 0.89. Do you switch?"*
Nuanced answer: not automatically — consider interpretability requirements, latency constraints, whether that 0.04 AUC gain translates to meaningful real-world business impact, and whether it's been properly validated via A/B test rather than just offline comparison. "It depends on the constraints" is a legitimate, expected L5 answer here, not a dodge.

**Common trap:** giving a purely mathematical answer to a system design question without addressing monitoring, class imbalance, or the interpretability/latency/baseline tradeoffs — this is the single most common way strong "textbook" candidates underperform on this specific module's interview questions.

## 9. CHECK — before Module 12

1. You're told your production logistic regression fraud model's AUC has stayed stable at 0.85 for 6 months, but the fraud team says the model is "missing more fraud lately." What would you investigate first, given what you learned about drift vs. calibration?
2. Your manager asks why you're starting with logistic regression instead of jumping straight to a deep neural network for a new classification problem. Give a 2-sentence answer that would satisfy an L5 bar.

Model Answers — Module 11 Checks
1. AUC stable at 0.85 for 6 months, but fraud team says the model is "missing more fraud lately" — what to investigate first?
This is a classic AUC-stable-but-calibration-decayed scenario. AUC only measures ranking ability — whether risky transactions score higher than safe ones, relatively speaking. It says nothing about whether the actual probability NUMBERS are still accurate. The fraud team's complaint ("missing more fraud") points straight at calibration decay, not drift in the ranking sense: if fraud has become more common recently (base rate shift — say, a new fraud pattern emerged, or seasonal fraud spiked), the model may still correctly rank "this transaction is riskier than that one" while systematically under-predicting the actual probability across the board — meaning transactions that now deserve a 70% score are still getting scored 40%, falling below the threshold and slipping through undetected, even though the model's relative ranking is technically unchanged.
First investigation step: recompute a fresh calibration curve on recent data and compare it to the training-time calibration curve (Module 8's reliability diagram). If predicted probabilities are now systematically too low relative to actual outcomes, that confirms calibration decay — the fix would likely be recalibrating the threshold or retraining on more recent data, not necessarily rebuilding the whole model from scratch.
2. Why start with logistic regression instead of jumping straight to a deep neural network — 2-sentence L5 answer:
"Logistic regression gives us a fast, interpretable baseline that tells us how much signal actually exists in our features and how much lift a more complex model would realistically buy us, before we invest in the added training/serving cost and reduced interpretability of a deep network. If the logistic regression baseline already performs close to what we need, the simpler, cheaper, more explainable model may be the better production choice outright."
