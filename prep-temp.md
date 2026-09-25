**Machine Learning**
- Bias-variance tradeoff
- L1 / L2 / ElasticNet regularisation
- Gradient descent (batch, SGD, mini-batch; Adam, RMSProp)
- Linear & logistic regression
- Decision trees — splits, Gini, entropy
- Random forests — bagging, feature subsampling
- Gradient boosting — XGBoost vs LightGBM (sequential vs leaf-wise)
- Hyperparameter tuning — grid search, random search, early stopping
- Time-series forecasting — stationarity, ARIMA, SARIMA, lag features, walk-forward CV, Prophet
- Class imbalance — SMOTE, class weights, threshold tuning, PR-AUC over ROC-AUC
- Evaluation metrics — precision, recall, F1, ROC-AUC, RMSE, MAE, MAPE
- Confusion matrix — derive all metrics from TP/FP/TN/FN
- Data leakage — sources, time-based splits
- Model monitoring — data drift (PSI, KS), concept drift, prediction drift, retraining triggers
- Feature importance — gain, permutation, SHAP

**Statistics**
- Probability — conditional probability, Bayes' theorem, independence
- Distributions — Normal, Binomial, Poisson, Exponential, log-normal
- Expected value and variance from scratch
- Central Limit Theorem vs Law of Large Numbers
- Hypothesis testing — p-value, Type I/II error, power, one vs two-tailed
- t-test vs z-test vs chi-squared vs Mann-Whitney
- Confidence intervals — correct interpretation, bootstrap CIs
- A/B testing — randomisation unit, sample size / power analysis, MDE, novelty effect, peeking problem, multiple testing correction (Bonferroni, BH)
- Causal inference basics — confounders, DiD, propensity score matching

**SQL**
- GROUP BY + HAVING vs WHERE
- Conditional aggregation — SUM(CASE WHEN ...)
- Window functions — ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD
- Running totals and moving averages with OVER()
- INNER / LEFT / FULL OUTER joins
- CTEs vs subqueries
- Classic patterns — top-N per group, duplicates, sequence gaps

**Python / Coding**
- Arrays, hash maps, sets — know time complexity
- Two-pointer and sliding window patterns
- Implement from scratch — precision, recall, F1, RMSE, MAE, cosine similarity, Gini impurity
- Implement linear regression with gradient descent from scratch
- Pandas — groupby, merge, apply, rolling, pivot, melt
- NumPy — broadcasting, np.where, vectorised ops
- Top-K with heap, running median with two heaps
- Binary search

**LLM & Agents**
- Transformer basics — attention, tokenisation, context window
- Temperature, top-p — effect on outputs
- Fine-tuning vs RAG vs prompt engineering — when to choose each
- RAG pipeline — chunk, embed, vector DB, retrieve, rerank, generate
- Embedding models — dense vs sparse (BM25) vs hybrid retrieval
- ReAct pattern — reason + act loop
- Multi-agent architecture — orchestrator, sub-agents, guardrail agent
- Tool / function calling
- Memory types — in-context, vector store, episodic
- Human-in-the-loop design
- Guardrails — output schema enforcement, confidence thresholds, fallback
- LLM evaluation — LLM-as-judge, golden datasets, hallucination detection
- Observability — tracing agent steps, latency, token usage, drift
