

## Decision Trees — Foundation (Always Asked First)

**Core questions:**
- How does a DT split? Walk me through Gini vs. entropy vs. MSE
- How does a DT handle continuous vs. categorical features?
- What is pruning? Pre vs. post-pruning trade-offs
- Why do DTs overfit?

**Follow-ups to expect:**
- *"What's the time complexity of building a tree?"* → O(n · d · log n)
- *"How does a DT handle missing values at train vs. predict time?"*
- *"If a feature has high cardinality, how does splitting behavior change?"*
- *"What's the difference between CART, ID3, C4.5?"* — just need high-level: CART binary + Gini, ID3 multi-way + info gain, C4.5 + gain ratio

---

## Bagging — Conceptual Depth

**Core questions:**
- What is bagging and why does it reduce variance but not bias?
- What is bootstrapping? What fraction of data is left out on average? (~37%)
- How does aggregation work — voting vs. averaging?

**Follow-ups:**
- *"Why does averaging predictions reduce variance?"* → They want you to say: Var(mean of n iid vars) = σ²/n, but trees are correlated so it's less clean
- *"Can bagging reduce bias?"* → No, each tree is high-variance low-bias, averaging preserves bias
- *"What's the difference between bagging and pasting?"* → Pasting = without replacement

---

## Random Forest — This is the Bulk

**Core questions:**
- How does feature selection in RF differ from a single DT?
- Why is RF less prone to overfitting than a single DT?
- How do you estimate feature importance in RF?
- Key hyperparameters to tune?

**Specific follow-ups by topic:**

**On decorrelation:**
- *"Why does RF randomly subset features at each split?"*
- *"What happens if you set max_features = total features?"* → Degenerates to bagged DTs, trees are more correlated
- *"What's the typical rule of thumb for max_features?"* → √p for classification, p/3 for regression

**On OOB:**
- *"What is OOB error and how is it used?"* → Each tree is tested on ~1/3 of data it never saw; acts as built-in cross-validation without a separate hold-out set
- *"When would you prefer OOB over k-fold CV?"*

**On feature importance:**
- *"How is mean decrease in impurity (MDI) computed?"* → Weighted avg of impurity decrease across all splits on that feature
- *"What's the problem with MDI for high-cardinality features?"* → Biased upward
- *"How does permutation importance fix this? What's the downside?"* → Slower, can underestimate correlated features

**On hyperparameters:**
- n_estimators, max_depth, min_samples_split, max_features, min_samples_leaf
- *"How do you know when adding more trees stops helping?"*
- *"What's the bias-variance effect of increasing max_depth?"*

---

## Comparison / Trade-off Questions (Very Common at FAANG)

- *"Bagged DT vs. RF — when would you pick one over the other?"*
- *"RF vs. GBM — which do you prefer and why?"* — expect this even though you only prepped RF. Answer: RF = parallelizable, less tuning, robust to outliers; GBM = lower bias, better peak performance but prone to overfit
- *"Are regularization techniques like Lasso/Ridge applicable to RF? Why not?"* → No — RF is non-parametric, no loss function to penalize coefficients
- *"How does RF handle class imbalance?"* → class_weight, balanced subsampling

---

## Applied / Practical

- *"You trained an RF. Training accuracy is 99%, test is 72%. What do you do?"*
- *"A feature is important in RF but has near-zero coefficient in logistic regression. Why?"* → RF captures non-linearity and interactions
- *"How would you explain RF feature importance to a non-technical stakeholder?"*
- *"RF takes too long to predict at serving time. How do you fix it?"* → Fewer/shallower trees, distill into single DT, feature reduction

---



**One tip:** At all 5, be ready to derive *why* averaging reduces variance from first principles (the iid → correlated trees argument). That one answer separates strong candidates.



Here's everything laid out as plain text:

---

## Decision Trees

**1. Walk me through exactly how a DT chooses a split.**
- What's the time complexity of finding the best split?
- How does this change with continuous vs categorical features?
- What if two splits give identical Gini — how do you break the tie?
- Key point: Don't just say "minimize impurity" — they want: iterate features → thresholds → pick max IG. Complexity is O(n·d·log n).

**2. Gini impurity vs entropy — when does the choice actually matter?**
- Derive Gini from scratch. Now entropy.
- Which converges faster during training?
- Why do most libraries default to Gini?
- Key point: In practice almost never matters. Gini is faster (no log). They want you to say that, not hedge.

**3. What is information gain and what's its flaw?**
- How does gain ratio fix it?
- Why does ID3 prefer high-cardinality features?
- How does CART avoid this problem?
- Key point: IG flaw = biased toward features with many values (e.g. unique IDs). Gain ratio = IG / split_info.

**4. Explain pre-pruning vs post-pruning. Trade-offs.**
- What is cost-complexity pruning (ccp_alpha)?
- How would you pick the right ccp_alpha?
- Is pruning more important for DT or RF?
- Key point: Post-pruning (reduced error, cost-complexity) is usually better but more expensive. RF makes pruning less critical.

**5. How does a DT handle missing values — at train time and predict time?**
- What are surrogate splits?
- How does sklearn handle this vs XGBoost?
- What if most values for a feature are missing?
- Key point: sklearn doesn't handle missing natively — impute first. XGBoost/LightGBM have built-in missing routing. Surrogate splits = fallback splits using correlated features.

**6. Why do decision trees have high variance and low bias?**
- Draw the bias-variance curve as max_depth increases.
- If I set max_depth=1, what does the tree look like? What's the bias?
- How does this motivate bagging?
- Key point: A fully grown tree memorises training data perfectly → near-zero bias, massive variance. This is THE motivation for ensembles.

**7. What is the difference between CART, ID3, and C4.5?**
- Which handles regression? Which handles multi-class natively?
- Which one does sklearn use?
- Why did C4.5 replace ID3?
- Key point: CART = binary splits + Gini/MSE; ID3 = multi-way + entropy, no pruning, no regression; C4.5 = gain ratio + pruning + continuous. sklearn uses CART.

**8. How does a DT handle multicollinearity?**
- Is this a problem for DTs vs linear models?
- How does it affect feature importance?
- How does RF behave differently here?
- Key point: DTs are largely immune to multicollinearity in terms of predictions, but feature importance becomes unreliable — one of two correlated features gets all the credit.

---

## Bagging

**1. Explain bagging from first principles. Why does it reduce variance?**
- Derive: Var(mean of n iid variables) = σ²/n. Why doesn't this fully apply to trees?
- What's the correlation ρ between trees and how does it enter the variance formula?
- If trees were perfectly correlated, what would bagging achieve?
- Key point: Var(avg) = ρσ² + (1-ρ)σ²/n. Key insight: correlation ρ prevents full variance reduction. That's exactly why RF adds feature randomness.

**2. What is bootstrapping? What fraction of data is excluded per tree on average?**
- Derive the ~36.8% figure.
- What is the implication for OOB samples?
- What's the difference between bootstrap and subsampling?
- Key point: Derive: P(not picked in one draw) = (1-1/n)^n → e^(-1) ≈ 0.368 as n→∞. They will ask you to derive this.

**3. Can bagging reduce bias? Why or why not?**
- Can boosting reduce bias? How?
- What if your base learners are linear models?
- What is the bias of the ensemble vs bias of one tree?
- Key point: No. Each tree has the same bias. Averaging preserves expected value. Bagging ONLY attacks variance. This is a classic trap.

**4. Bagging vs pasting — what's the difference?**
- When would you prefer pasting?
- How does subsampling fraction affect the bias-variance trade-off?
- Does sklearn support pasting?
- Key point: Pasting = sampling WITHOUT replacement. Bagging = WITH replacement. Pasting trees are more similar to each other → less variance reduction but less bias inflation.

**5. Why does bagging work better on unstable (high-variance) learners?**
- Name three unstable learners besides DTs.
- Would bagging a linear regression help much?
- What's an "unstable" learner formally?
- Key point: Unstable = small data change → large model change. DTs, k-NN, NNs are unstable. Linear regression is stable → bagging barely helps.

**6. How does bagging affect interpretability?**
- Is there any way to recover interpretability from a bag of trees?
- How does SHAP help here?
- What's the interpretability trade-off vs a single DT?
- Key point: You lose single-tree interpretability. SHAP values aggregate explanations across trees and restore local interpretability.

---

## Random Forest Core

**1. What does RF add over bagged DTs? Why is it necessary?**
- If all trees see the same strong feature, what happens?
- What's the variance formula with and without feature randomness?
- What's the effect on individual tree accuracy?
- Key point: Feature randomness (attribute bagging) breaks correlations between trees. Individual trees get worse, but ensemble improves because Var(avg) decreases as ρ drops.

**2. How do you choose max_features? What's the rule of thumb and why?**
- What happens as max_features → n_features?
- What happens as max_features → 1?
- How do you tune this in practice?
- Key point: Classification: √p. Regression: p/3. These are heuristics, not laws. At max_features=all → bagged DT. At max_features=1 → maximally random (Extremely Randomized Trees).

**3. Explain OOB error. How is it computed? When do you prefer it over CV?**
- Is OOB error pessimistically or optimistically biased?
- How does n_estimators affect OOB reliability?
- When is OOB error NOT a good substitute for CV?
- Key point: OOB is slightly pessimistic (trained on ~63% data). Prefer OOB when data is large and CV is expensive. Not reliable with very few trees or heavy class imbalance.

**4. Two types of feature importance in RF — explain both and their flaws.**
- MDI vs permutation importance — which do you trust more?
- How do correlated features affect both?
- What does SHAP add that these miss?
- When is drop-column importance better?
- Key point: MDI = fast but biased toward high-cardinality features. Permutation = unbiased but slow and underestimates correlated features. Drop-column = most honest but requires O(n_features) retraining.

**5. Walk me through every key hyperparameter in RF and its effect.**
- How would you set up a grid search for RF?
- Which hyperparameter matters most?
- What's the effect of bootstrap=False?
- Key point: n_estimators (more is better until plateau), max_depth (controls bias-variance), max_features (controls decorrelation), min_samples_leaf (smoothing), bootstrap. n_estimators doesn't overfit — a common misconception.

**6. Does RF overfit as n_estimators increases?**
- Prove it or give the intuition.
- Does training error still go to zero?
- What's the practical implication for tuning?
- Key point: No — it's a classic trap question. Adding trees only reduces variance. The ensemble converges to a limiting function. Training error goes to ~0 but test error plateaus.

**7. How does RF handle class imbalance?**
- What is class_weight='balanced' doing internally?
- What is balanced subsampling in RF?
- Compare to SMOTE — when do you use each?
- Key point: class_weight inversely weights classes in split criterion. balanced_subsample reweights per bootstrap sample. SMOTE generates synthetic minority samples — better for severe imbalance.

**8. How does RF make predictions for regression vs classification?**
- What's the variance of the ensemble prediction in regression?
- Can you get probability estimates from RF? How?
- How does predict_proba work — is it well-calibrated?
- Key point: RF predict_proba = fraction of trees voting for each class. It's poorly calibrated (tends toward 0.5). Use Platt scaling or isotonic regression if you need calibrated probabilities.

---

## Comparisons & Trade-offs

**1. RF vs GBM — when do you pick each?**
- Which is more sensitive to outliers?
- Which trains faster on large data?
- Which is easier to tune?
- When does RF beat GBM on leaderboards?
- Key point: RF = parallel, robust to outliers, less tuning. GBM = lower bias, better peak performance, sequential so slower. RF rarely wins competitions but wins in production.

**2. Is regularization (Lasso/Ridge) applicable to RF? Why not?**
- How does RF regularize implicitly?
- What IS regularized in a DT or RF?
- Can you add explicit regularization to tree models?
- Key point: No coefficient vector → nothing to penalize. RF regularizes implicitly via max_depth, min_samples_leaf, max_features. XGBoost adds L1/L2 on leaf weights.

**3. Why might a feature have high RF importance but near-zero logistic regression coefficient?**
- What does this tell you about the feature's relationship with the target?
- How would you visualize this?
- What if the reverse is true?
- Key point: RF captures non-linearity and interactions. High RF importance + near-zero LR coefficient → likely non-linear relationship or interaction effect. Use partial dependence plots to diagnose.

**4. DT vs RF — when would you use a single DT?**
- Is there an ensemble accuracy threshold below which you'd prefer a single tree?
- What about in regulatory/compliance contexts?
- What's a surrogate model?
- Key point: Single DT when interpretability is non-negotiable (regulatory, medical). Surrogate model = train a DT to mimic RF predictions to explain them.

**5. Extra Trees (Extremely Randomized Trees) vs RF — what changes?**
- When does Extra Trees outperform RF?
- What's the bias-variance implication of random thresholds?
- Is training faster or slower?
- Key point: Extra Trees randomizes split thresholds (no best-split search), not just features. Faster to train. Higher bias, lower variance than RF. Useful when n is small or speed matters.

---

## Applied / Practical

**1. Train accuracy 99%, test accuracy 72%. What do you do?**
- Which hyperparameters do you tune first?
- How do you detect if it's a data quality issue vs model issue?
- What if test improves but train drops to 80% — is that acceptable?
- Key point: Sequence matters: first check data leakage, then regularize (max_depth, min_samples_leaf), then check feature importance for leaking features, then get more data.

**2. RF is too slow at prediction time for your production SLA. What do you do?**
- How do you pick which trees to prune from the ensemble?
- What is model distillation?
- What's the trade-off of fewer deeper trees vs more shallow trees?
- Key point: Options: fewer trees, shallower trees, distill to single DT, feature reduction, convert to ONNX, quantize leaf values. Distillation often best accuracy/speed trade-off.

**3. How would you explain RF feature importance to a non-technical PM?**
- What if two features are correlated — how do you explain the importance split?
- What if a feature has zero importance but domain experts insist it matters?
- Key point: "Shuffle this feature randomly — if accuracy drops a lot, it's important." Use permutation importance framing. Zero importance + domain knowledge → check for encoding bugs or collinearity masking.

**4. You have a 1M row dataset. How do you train and validate RF efficiently?**
- When is OOB enough vs when do you need CV?
- How do you parallelize RF training?
- What if you can't fit data in memory?
- Key point: OOB is fine for 1M rows. Use n_jobs=-1 for parallel tree building. For out-of-memory: subsample, use LightGBM (histogram-based), or incremental approaches.

**5. How do you detect and handle data leakage in an RF model?**
- What does suspiciously high OOB accuracy signal?
- How do RF feature importances help detect leakage?
- What's temporal leakage and how do you prevent it?
- Key point: High importance on a feature that shouldn't be predictive = leakage signal. Temporal leakage = future info in training. Always use time-based splits for time series.

**6. RF on a ranking/recommendation task — what changes vs classification?**
- How do you frame ranking as a tree model problem?
- What loss function is used in LambdaMART?
- How does feature importance interpretation change?
- Key point: Pure RF doesn't do ranking natively. Workaround: pointwise (predict relevance score). Better: LambdaMART (gradient boosted trees with ranking loss).

---

## Advanced / Gotchas

**1. Prove (or derive) why averaging reduces variance. Where does the derivation break for trees?**
- What is the exact formula for Var(ensemble) with correlated trees?
- What value of ρ does a typical RF achieve?
- How does max_features affect ρ?
- Key point: Var(avg of n with correlation ρ) = ρσ² + (1-ρ)σ²/n. As n→∞ → ρσ². The irreducible floor = ρσ². Only way to improve = reduce ρ (feature randomness) or reduce σ² (deeper trees + more data).

**2. What is the Rashomon effect and how does it relate to RF?**
- Why does it matter for feature importance?
- How does it affect model selection?
- What's the implication for explainability?
- Key point: Rashomon effect = many equally good models exist. In RF, feature importance is unstable across runs when features are correlated — multiple importance orderings are equally "correct".

**3. How does RF behave on high-dimensional sparse data (e.g. NLP bag-of-words)?**
- Why does RF underperform linear models on sparse high-D data?
- What's the max_features issue?
- How would you adapt RF for this setting?
- Key point: With p=100K features and max_features=√p≈316, most splits consider irrelevant features. Linear models exploit global structure better. RF needs much larger max_features or dimensionality reduction first.

**4. What is the proximity matrix in RF and what can you do with it?**
- How is the proximity matrix computed?
- What algorithms can you run on it?
- When is it useful vs just using the predictions?
- Key point: Proximity(i,j) = fraction of trees where i and j end in the same leaf. Use for: missing value imputation, outlier detection, clustering. Expensive: O(n²) matrix.

**5. How would you implement RF from scratch in Python? Walk me through.**
- Which part is the bottleneck?
- How do you parallelize it?
- What's the minimum viable implementation?
- Key point: Bootstrap sample → grow DT with feature subset at each node → collect trees → aggregate. They're testing whether you understand what sklearn is doing under the hood.
