## Top ML Algorithms for Product Data Scientist Interviews

**Regression**
- Linear & Logistic Regression — always fair game; know assumptions, regularization (L1/L2), and when to use each
- Gradient Boosting (XGBoost/LightGBM) — workhorse for tabular data; know how trees are built sequentially

**Classification & Ranking**
- Decision Trees / Random Forest — bagging vs boosting, feature importance, overfitting controls
- Naive Bayes — surprisingly common for text/spam; explain the independence assumption honestly

**Clustering & Segmentation**
- K-Means — user segmentation use case; know how to choose K (elbow, silhouette)
- DBSCAN — when K-Means fails (non-convex clusters, outliers)

**Recommendation & Personalization**
- Collaborative Filtering (matrix factorization) — critical for feed/content products
- Content-based filtering + hybrid approaches

**Experimentation Adjacent**
- Propensity score modeling — causal inference without A/B tests
- Uplift modeling — who to target; incremental effect estimation

---

**What makes product DS interviews different** from pure ML roles:

- You're expected to tie every algorithm back to a **business metric** (CTR, retention, LTV)
- Questions often start with *"how would you build a system to..."* not *"implement X"*
- Trade-offs matter more than perfection — explain why you'd choose gradient boosting over a neural net for a small dataset
- **Interpretability** is often required (can you explain the model to a PM?)

**High-signal things to know cold:** AUC-ROC vs precision/recall trade-offs, handling class imbalance (SMOTE, weighting), feature importance vs SHAP values, and when *not* to use ML.
