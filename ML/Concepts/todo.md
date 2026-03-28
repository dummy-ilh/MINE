
psi

    Kolmogorov–Smirnov (KS) Test: A non-parametric test to determine if two datasets are from the same distribution.
    Jensen–Shannon (JS) Divergence: A method to measure the similarity between two probability distributions.
    Population Stability Index (PSI): A common metric in credit scoring that measures the shift in population distribution between two periods.

https://docs.deepchecks.com/stable/general/guides/drift_guide.html
Based on your notes so far (Decision Trees, Splitting Criteria, Pruning, Feature Selection), here's what's missing:

**Decision Tree specific:**
- Handling missing values (at train vs test time)
- Multivariate / oblique splits (trees that split on linear combinations of features)
- Regression Trees vs Classification Trees — differences in leaf prediction & criteria
- Class imbalance — how it affects splits + fixes (class_weight)

**Ensemble methods (natural next step):**
- Random Forest — bagging + feature subsampling
- Gradient Boosting (XGBoost, LightGBM, CatBoost) — boosting mechanics
- Bagging vs Boosting vs Stacking
- Out-of-Bag (OOB) error

**Broader ML topics you haven't covered yet:**
- Cross-validation (k-fold, stratified, leave-one-out)
- Hyperparameter tuning (Grid Search, Random Search, Bayesian Optimisation)
- Handling imbalanced datasets (SMOTE, class weights, threshold moving)
- Evaluation metrics deep dive (ROC-AUC, Precision-Recall curve, F1)
- Feature engineering & selection methods
- Regularisation (L1, L2, ElasticNet)
- SVM
- k-Means & clustering
- Neural network fundamentals + backprop
- Transformers & Attention
- ML System Design

**Most critical gaps for FAANG right now:**
> Random Forest → XGBoost → ML System Design → Cross-Validation

Want me to continue with **Random Forest** next?
