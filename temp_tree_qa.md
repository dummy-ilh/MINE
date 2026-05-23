

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


