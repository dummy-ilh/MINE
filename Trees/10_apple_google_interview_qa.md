# Recent Apple, Google & Meta Interview Questions — Trees, Random Forest, Bagging

Compiled from current candidate-report sites (Interview Query candidate reports, Exponent, DataInterview, Glassdoor, Blind, and similar interview-prep aggregators) as of August 2026. Where a source reported a near-verbatim question, it's flagged as **[reported]**; broader theme questions compiled from multiple sources are flagged as **[theme]**. Each includes a pointer to which of your existing notes files answers it.

---

## 1. Apple

Apple's ML Engineer loop (per Interview Query's 2026 guide, synthesized from candidate reports) typically runs: recruiter screen → hiring-manager call → a ~30-minute applied coding screen (often NLP-flavored, sometimes with no internet access allowed) → a full-day virtual/remote onsite with ~8 interviewers covering ML fundamentals, coding, LLMs, and a deep dive on your own ML project.

**Questions reported in Apple's active interview-question bank:**

- **"Random Forest Explanation"** [reported] — an open-ended prompt to explain Random Forest from the ground up.
  → *Bagging notes, Section 2 (algorithm) + Ensemble Foundations notes, Section 4 (why bagging pairs with high-variance learners)*

- **"Bias vs. Variance Tradeoff"** [reported] — a standalone conceptual question, separate from the RF one, suggesting Apple tests this as its own topic rather than only inside a tree/forest context.
  → *Ensemble Foundations notes, Section 2 (full bias-variance decomposition + worked numerical)*

- **"Reducing Error Margin"** [reported] — framed around shrinking prediction error/uncertainty, a natural fit for a bagging/variance-reduction discussion.
  → *Bagging notes, Section 5 (when bagging helps) + Ensemble Foundations notes, Section 3 (the variance formula and diminishing returns)*

**Theme questions consistent with Apple's reported style** (algorithm choice justified by data/deployment constraints, cross-checked against Apple's MLE guide on Interview Query and Exponent):

- **How does a random forest generate its trees, and when would you use it over logistic regression?** [reported, InterviewQuery Apple MLE guide] — expects you to describe bootstrap sampling, feature subsampling, and majority-vote/averaging aggregation, and contrast that with logistic regression's linear decision boundary. Apple is specifically testing whether you justify the choice using dataset complexity, interpretability needs, and computational constraints — not just describe the algorithm.
  → *Decision Tree Fundamentals notes, Section 1 + Bagging notes, Section 2; Practical Interview Ground notes, Section 1 (trees vs. linear models table) for the justification framing*

- **Explain the difference between XGBoost and Random Forest — when would you prefer one over the other?** [reported, InterviewQuery Apple MLE guide] — expects a contrast between RF's independent bagged trees and XGBoost's sequential, gradient-optimized boosting.
  → *Boosting vs. Bagging notes, full document — this is exactly the comparison it's built around*

- **On-device/applied framing:** given Apple's emphasis on "data privacy and on-device awareness" (per Exponent's 2026 Apple MLE guide), expect a tree/RF question to pivot into an on-device deployment follow-up (memory/latency cost of an ensemble, Core ML constraints) rather than stopping at pure theory.
  → *Bagging notes, Section 9 (Apple MLE Q&A) + Decision Tree Fundamentals notes, Section 12 (Apple MLE Q&A) — both already built around exactly this pivot*

---

## 2. Google

Google's ML interview loop (per Interview Query and PracHub's 2026 guides) mixes coding/algorithmic rounds with ML theory and system design, typically 5–6 rounds after an initial phone screen.

**Reported/aggregated question themes across current Google-focused prep sources:**

- **How do ensemble methods like Random Forest or Gradient Boosting work?** [theme, appears near-verbatim across multiple 2025–2026 Google-focused prep guides] — expects both algorithms explained and contrasted in the same answer, not just one.
  → *Boosting vs. Bagging notes, Sections 1–2*

- **What's the difference between bagging and boosting?** [theme, consistently reported] — the standard framing is: bagging trains in parallel on bootstrap samples and reduces variance; boosting trains sequentially, each model correcting the last, and tends to reduce bias but can overfit if uncontrolled.
  → *Boosting vs. Bagging notes, full document; Ensemble Foundations notes, Section 4 (the underlying bias/variance mechanism)*

- **When would you use a decision tree over logistic regression?** [theme] — expected answer centers on interpretability and non-linear decision boundaries for trees vs. linear separability and simplicity for logistic regression, with an explicit mention that trees can overfit without pruning.
  → *Practical Interview Ground notes, Section 1 (full comparison table) + Section 2 (overfitting diagnostics)*

- **Applied/scenario framing (churn prediction, reported by a Google MLE candidate account):** justify choosing Random Forest vs. Gradient Boosting for a churn model — expects you to weigh robustness, training speed, and interpretability (Random Forest) against squeezing out peak accuracy on subtle non-linear patterns at the cost of more careful tuning (GBM).
  → *Boosting vs. Bagging notes, Section 3 ("when would you reach for bagging/RF instead of boosting") + Section 4 (Google MLE Q&A)*

- **Design/scale framing:** Google's prep guides emphasize scaling reasoning (e.g., "how would this parallelize/stream over billions of records") even for core ML questions — expect a tree/RF question to get a "how would this work at scale" follow-up.
  → *Bagging notes, Section 8 (Google MLE Q&A, the distributed-training question) + Decision Tree Fundamentals notes, Section 11 (Google MLE Q&A, the $O(p \cdot n\log n)$ scaling question)*

---

## 3. Meta

Meta's ML Engineer loop (per Interview Query's 2026 guide) is coding-speed-heavy: a coding screen (often two LeetCode-style problems in 40–45 minutes), followed by a 4–5 round onsite mixing more coding, an ML system-design round (recommendation/ranking/feed-freshness heavy), and product-sense/behavioral rounds.

**Questions reported in Meta's active interview-question bank:**

- **"Random Forest Explanation"** [reported] — appears in Meta's question bank in the same form as Apple's, suggesting this is a standard, company-agnostic conceptual check both use.
  → *Bagging notes, Section 2*

- **"Bank Fraud Model"** [reported] — a scenario/design-style question; fraud detection is a canonical class-imbalance use case (rare positive class, high cost of false negatives).
  → *Pruning/Missing Values/Imbalance notes, Part 3 (class imbalance) — directly relevant, especially Section 3.7's Q&A on false-negative cost trade-offs; Boosting vs. Bagging notes, Section 4 (Google MLE Q&A has a closely related noisy-label fraud-detection question, same reasoning applies)*

- **"Reducing Error Margin"** [reported] — appears in both Apple's and Meta's banks; framed around shrinking prediction uncertainty.
  → *Ensemble Foundations notes, Section 3 (variance-reduction formula and worked numerical)*

- **"Booking Regression"** [reported] — a regression-flavored applied question (plausibly a travel/booking price or demand prediction scenario); tree-based regression fundamentals are a likely fit given Meta's product surfaces.
  → *Decision Tree Fundamentals notes, Section 3.4 (regression/variance-reduction splitting) + Applied/Advanced notes, Section 2 (monotonic constraints — relevant if the interviewer probes "should price only increase with X")*

- **"Fill None Values"** [reported] — a data-cleaning/coding-flavored question, directly about missing-value handling.
  → *Pruning/Missing Values/Imbalance notes, Part 2 (missing value handling) — the imputation vs. native-handling comparison is exactly this topic*

- **"Precision and Recall"** [reported, appears alongside the RF question in the same bank] — commonly paired with imbalanced-data or fraud-style scenarios at Meta specifically.
  → *Evaluation & Tuning notes, Section 3 (imbalance) + Section 6 (Google MLE Q&A has a directly transferable "95% accuracy but missing everything that matters" diagnostic walk-through)*

**Theme, from Meta's system-design emphasis:** expect a tree/RF/bagging question to be a smaller piece of a larger applied scenario (e.g., "build a model to detect X" or "why did engagement drop") rather than asked in isolation — Meta's process consistently pairs ML fundamentals with product/metrics reasoning.
→ *Practical Interview Ground notes, Section 6 (Google MLE Q&A's fraud-scoring interpretability design question) transfers directly to this style of question.*

---

## 4. Cross-Company Pattern Summary

| Pattern | Seen at | What it means for prep |
|---|---|---|
| "Random Forest Explanation" as a standalone open-ended prompt | Apple, Meta (identical phrasing in both question banks) | Have a tight, structured explanation ready: bootstrap sampling → per-tree training → aggregation → why it reduces variance, in under 2 minutes, before any follow-up |
| Bagging vs. boosting comparison | Google (multiple sources), Apple (XGBoost vs. RF framing) | The comparison table and "when would you reach for X" framing matters more than reciting either algorithm alone |
| Bias-variance tradeoff asked standalone, not just inside a tree question | Apple | Don't assume it'll only come up as a follow-up to a tree question — be ready for it cold |
| Imbalance / fraud / precision-recall scenarios | Meta (Bank Fraud Model, Precision and Recall) | Expect imbalance handling to be tested via a scenario, not asked as "define precision and recall" in isolation |
| Missing-value handling as a data-prep/coding question | Meta (Fill None Values) | Be ready to actually write the imputation/handling code, not just discuss it conceptually |
| Scenario/applied framing over pure theory | All three, increasingly | Across all three companies, tree/RF/bagging questions are trending toward "justify your choice for this specific situation" rather than "define the algorithm" — the on-device (Apple), scale (Google), and product-metrics (Meta) framings in your existing notes' company-specific Q&A sections are the right prep angle |

---

## 5. Sources

- Interview Query — Apple ML Engineer Interview Guide (2026 edition, candidate-report-sourced question bank)
- Interview Query — Meta ML Engineer Interview Guide (2026 edition, candidate-report-sourced question bank)
- Interview Query — Google Machine Learning Interview Questions Guide (2026 edition)
- Exponent — Apple Machine Learning Engineer Interview Guide (2026)
- DataInterview — Apple Machine Learning Engineer Interview guide (2026) and Top ML Interview Questions (2026)
- Interviews.chat — Google Machine Learning Engineer Interview Questions and Answers
- PracHub — Google Machine Learning Engineer Interview Questions (2026)
- Devinterview.io — Random Forest / Decision Tree interview question banks (2026 editions)

*Note: several of these aggregator sites compile and rephrase candidate reports rather than publishing verbatim transcripts, so exact wording can vary by source — the "reported" questions above are the specific phrasings that appeared directly in each company's active question bank on Interview Query, which sources individual candidate interview reports.*

---

**One-line summary to remember:** *Apple and Meta both have a standalone "Random Forest Explanation" prompt in their active question banks — have a tight, structured answer ready. Google leans on the bagging-vs-boosting comparison and scale follow-ups. Meta wraps tree/RF/imbalance questions inside applied scenarios (fraud, regression, data-cleaning) rather than asking pure theory. All three increasingly want justification for a specific situation, not just a textbook definition.*# Trees & Ensembles — Apple / Google Style Interview Practice Q&A

Mixed practice set, covering the full curriculum (Ch.1-9). Questions are phrased the way they tend to show up in Apple/Google-style ML interviews — often starting concrete/practical, then pushing into "why" follow-ups. Answers are kept in the same plain-language style as the rest of the curriculum.

---

### Warm-up / Foundations

**Q1. "Walk me through how a decision tree decides where to split."**
At every node, it checks every feature and every possible threshold for that feature, and picks whichever single (feature, threshold) pair reduces impurity the most — Gini or entropy for classification, variance/MSE for regression. It repeats this on each resulting child, growing the tree one split at a time. It's a greedy process — it never looks ahead to see if a "worse now" split might set up a better split later.

**Q2. "Why does a fully-grown, unpruned decision tree usually overfit?"**
Left alone, it keeps splitting until every leaf is pure (or has just one sample) — at that point it's basically memorized the training data, including its noise, not just the real underlying pattern. That's why pruning (limiting depth, minimum leaf size, or cost-complexity pruning) exists.

**Q3. "Gini impurity or entropy — which would you pick, and does it matter?"**
In practice it barely matters — the two usually agree on the best split. Gini is cheaper to compute (no logarithm), which is why it's the common default. This is a good question to answer honestly rather than inventing a strong preference that isn't really justified.

---

### Bagging & Random Forest

**Q4. "Explain Random Forest to someone who's never heard of it, in under 30 seconds."**
Train a bunch of decision trees, each on a random resample of the data, and each only allowed to consider a random subset of features at every split. Average all their predictions together. The randomness in both rows and features makes the trees different enough from each other that averaging them smooths out each individual tree's mistakes.

**Q5. "If I keep adding more trees to my Random Forest, will it eventually start overfitting?"**
No. Each tree is trained completely independently, so adding more of them only ever averages away more noise — it can't add new overfitting risk the way, say, adding more boosting rounds can. More trees costs more compute/memory, but it won't hurt accuracy.

**Q6. "Why does Random Forest randomly restrict which features each split can consider, instead of just letting every tree see every feature like in plain bagging?"**
If one or two features are clearly the strongest predictors, plain bagging's trees will nearly all pick those same features for their top splits anyway, no matter which random rows they trained on — so the trees end up fairly similar to each other, and averaging similar trees doesn't help much. Forcing each split to only consider a random handful of features means the strongest feature isn't always available, pushing trees to genuinely differ from each other more — and averaging genuinely different trees helps a lot more than averaging similar ones.

**Q7. "What's Out-of-Bag error, and why would you use it instead of a separate validation set?"**
Since each tree is trained on a random resample, roughly a third of the original data is left out of any given tree's training set. For each data point, you can get an honest prediction from just the trees that never saw it, and compare that to the real answer — giving you a validation-like error estimate for free, without setting aside any data purely for validation.

---

### Boosting

**Q8. "How is boosting fundamentally different from bagging?"**
Bagging trains many trees independently, all at once, and averages them to smooth out unstable, high-variance mistakes. Boosting trains trees one after another, where each new tree is specifically aimed at correcting whatever the current combined model is still getting wrong — it's targeting systematic, repeatable mistakes (bias), not random instability.

**Q9. "Why does Gradient Boosting typically use small, shallow trees, while Random Forest often uses big, deep ones?"**
Random Forest wants each individual tree to be as strong and detailed an opinion as possible, since the averaging step is what handles instability. Boosting is taking a series of small, cautious correction steps — a big, detailed tree at any one boosting round would try to fix everything in one shot, which is exactly the kind of aggressive move that leads boosting to start fitting noise instead of real signal.

**Q10. "You're training a Gradient Boosting model and notice training error keeps dropping but validation error has started rising. What's happening, and what would you do?"**
Classic boosting overfitting — extra rounds are now fixing mistakes that were actually just noise, not real signal. Fixes: stop training earlier (early stopping based on validation error), lower the learning rate, reduce tree depth, or use a smaller number of rounds overall.

**Q11. "What does XGBoost actually add on top of plain Gradient Boosting?"**
Two big things: it directly penalizes trees for being too complex (too many leaves, or leaves with extreme prediction values) as part of what it's optimizing, so it naturally builds simpler, more conservative trees. And it uses curvature information (not just the basic slope/gradient) to decide how big a correction each step should make, which tends to get to a good answer in fewer, more accurate steps.

**Q12. "Your dataset is huge — millions of rows — and training is too slow. Would you switch from XGBoost to LightGBM, and why?"**
Likely yes. LightGBM buckets feature values into a fixed number of bins before searching for splits, instead of checking every possible threshold, which is much faster on large data. It also grows trees by chasing whichever single leaf would benefit most from splitting, rather than expanding every branch evenly — spending its computation where it matters most. Trade-off: keep an eye on `num_leaves`, since this leaf-wise growth can overfit more easily on smaller datasets (less of a concern here, given the data size).

**Q13. "You have a dataset full of categorical features with many unique values — what's your approach?"**
Reach for CatBoost or LightGBM, both of which handle categorical features natively without needing manual one-hot encoding — which would otherwise blow up into hundreds of sparse columns and make it harder for the model to find good splits. CatBoost specifically also guards against a subtle leakage issue where a categorical encoding can end up quietly using a sample's own target value to encode itself.

---

### Stacking & Practical Judgment

**Q14. "Would you rather stack a Random Forest and an XGBoost model, or a Random Forest and three different XGBoost models with different seeds? Why?"**
The first option — Random Forest and XGBoost are genuinely different algorithms that tend to make different kinds of mistakes, giving a meta-learner real signal to combine. Three XGBoost models with different seeds are likely to fail on similar houses/cases in similar ways, so there's much less for stacking to gain over just averaging them.

**Q15. "What's the single most common bug when people build a stacked model, and why does it matter?"**
Leakage — training the meta-learner on base models' predictions on data those same base models were trained on. Those predictions look artificially accurate (the models have partially memorized those rows), so the meta-learner learns from dishonest inputs and performs worse than expected on genuinely new data. The fix is generating base model predictions using out-of-fold or held-out data only.

**Q16. "When would you choose a linear model over a tree-based model, even if the tree-based model tests slightly more accurate?"**
When you need simple, direct explanations for predictions (clean per-feature coefficients), when you expect the model to need to extrapolate beyond the range of training data, or when you have very little data and want the extra stability that comes from a simpler model with fewer moving parts.

**Q17. "Explain to a non-technical product manager why your model priced one specific house the way it did."**
This is where SHAP values fit in — rather than explaining the whole model, SHAP breaks down one specific prediction into how much each feature pushed it up or down from the average. E.g., "this house priced $120K above average: +$80K from its unusually large size, +$50K from its location, −$10K from its age" — a clear, additive, per-prediction explanation a non-technical audience can follow.

**Q18. "Your model predicts that a bigger house is worth less than a smaller one in the same neighborhood, all else equal. Business stakeholders are unhappy — how do you fix this?"**
Add a monotonic constraint forcing the prediction to never decrease as size increases. It's enforced directly during tree-building — the split search simply isn't allowed to consider splits that would violate the constraint — rather than trying to patch the finished model's predictions after the fact.

---

### Rapid-fire (short-answer style, common in phone screens)

- **"Does more trees in Random Forest ever hurt?"** → No.
- **"Does more rounds in Gradient Boosting ever hurt?"** → Yes, can overfit.
- **"Can Random Forest trees train in parallel?"** → Yes.
- **"Can boosting rounds train in parallel?"** → No, each depends on the last.
- **"What does a decision tree use to decide the best split?"** → Gini/entropy (classification) or variance/MSE (regression) reduction.
- **"What's the main risk of impurity-based feature importance?"** → Biased toward features with many possible split points.
- **"What fixes that bias?"** → Permutation importance.
- **"What's the fix for stacking leakage?"** → Out-of-fold predictions for the meta-learner.
- **"Can trees extrapolate beyond training data's range?"** → No.
- **"Which handles missing values natively, sklearn's trees or XGBoost?"** → XGBoost (and LightGBM); plain sklearn trees do not.

---

Want a second, harder-difficulty pass (system-design-style questions — e.g., "design an ML pipeline for X using tree-based models," latency/scale trade-offs), or is this the right level to practice with?
