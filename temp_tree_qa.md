Here's a structured breakdown of what to expect — question types, specific questions, and likely follow-ups.

---

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

## Applied / Practical (DS-heavy rounds, especially Meta/Google)

- *"You trained an RF. Training accuracy is 99%, test is 72%. What do you do?"*
- *"A feature is important in RF but has near-zero coefficient in logistic regression. Why?"* → RF captures non-linearity and interactions
- *"How would you explain RF feature importance to a non-technical stakeholder?"*
- *"RF takes too long to predict at serving time. How do you fix it?"* → Fewer/shallower trees, distill into single DT, feature reduction

---



**One tip:** At all 5, be ready to derive *why* averaging reduces variance from first principles (the iid → correlated trees argument). That one answer separates strong candidates.



Decision trees 8 questions
Walk me through exactly how a DT chooses a split. easy
Gini impurity vs entropy — when does the choice actually matter? medium
What is information gain and what's its flaw? medium
Explain pre-pruning vs post-pruning. Trade-offs. medium
How does a DT handle missing values — at train time and predict time? medium
Why do decision trees have high variance and low bias? easy
What is the difference between CART, ID3, and C4.5? medium
How does a DT handle multicollinearity? hard
Bagging 6 questions
Explain bagging from first principles. Why does it reduce variance? easy
What is bootstrapping? What fraction of data is excluded per tree on average? easy
Can bagging reduce bias? Why or why not? medium
Bagging vs pasting — what's the difference? easy
Why does bagging work better on unstable (high-variance) learners? medium
How does bagging affect interpretability? medium
Random forest core 8 questions
What does RF add over bagged DTs? Why is it necessary? easy
How do you choose max_features? What's the rule of thumb and why? medium
Explain OOB error. How is it computed? When do you prefer it over CV? medium
Two types of feature importance in RF — explain both and their flaws. hard
Walk me through every key hyperparameter in RF and its effect. medium
Does RF overfit as n_estimators increases? medium
How does RF handle class imbalance? medium
How does RF make predictions for regression vs classification? easy
Comparisons & trade-offs 5 questions
RF vs GBM — when do you pick each? medium
Is regularization (Lasso/Ridge) applicable to RF? Why not? medium
Why might a feature have high RF importance but near-zero logistic regression coefficient? hard
DT vs RF — when would you use a single DT? easy
Extra Trees (Extremely Randomized Trees) vs RF — what changes? hard
Applied / practical 6 questions
Train accuracy 99%, test accuracy 72%. What do you do? medium
RF is too slow at prediction time for your production SLA. What do you do? hard
How would you explain RF feature importance to a non-technical PM? easy
You have a 1M row dataset. How do you train and validate RF efficiently? medium
How do you detect and handle data leakage in an RF model? hard
RF on a ranking/recommendation task — what changes vs classification? hard
Advanced / gotchas 5 questions
Prove (or derive) why averaging reduces variance. Where does the derivation break for trees? hard
What is the Rashomon effect and how does it relate to RF? hard
How does RF behave on high-dimensional sparse data (e.g. NLP bag-of-words)? hard
What is proximity matrix in RF and what can you do with it? hard
How would you implement RF from scratch in Python? Walk me through. hard 
