# 🧠 Google India Data Science — Complete Interview Prep Guide

> Covers: **Data Scientist (University Graduate 2026)** & **Data Analytics Solutions Specialist**
> Roles based in Bengaluru, Karnataka

---

## 📌 How Google Interviews Work

Google data science interviews typically span **4–6 rounds**:

1. **Recruiter Screen** — Background, motivation, basics
2. **Technical Phone Screen** — SQL / Python / Stats problem
3. **Onsite / Virtual Loop (4–5 rounds):**
   - Statistics & Probability
   - SQL & Data Manipulation
   - Machine Learning
   - Product Sense / Analytics Case
   - Coding (Python / R)
   - Behavioral (Googleyness + Leadership)

---

## 🗂️ SECTION 1 — Statistics & Probability

This is the **most heavily tested** area. Google expects you to go deep.

### Topics to Master

- Descriptive statistics: mean, median, mode, variance, standard deviation, skewness, kurtosis
- Probability fundamentals: sample spaces, events, independence, conditional probability
- Bayes' Theorem and Bayesian reasoning
- Probability distributions:
  - Discrete: Bernoulli, Binomial, Poisson, Geometric, Negative Binomial, Hypergeometric
  - Continuous: Normal (Gaussian), Uniform, Exponential, Beta, Gamma, Log-Normal, Chi-Squared, t-distribution, F-distribution
- Central Limit Theorem (CLT) — intuition and applications
- Law of Large Numbers
- Expectation, Variance, Covariance, Correlation
- Joint, Marginal, and Conditional distributions
- Moment generating functions
- Percentiles, Quartiles, IQR, Outlier detection
- Sampling methods: SRS, stratified, cluster, systematic
- Sampling bias and how to avoid it

### Hypothesis Testing

- Null and Alternative Hypotheses
- Type I and Type II errors (α and β)
- p-value — meaning, common misconceptions
- Statistical power and sample size calculations
- One-tailed vs. two-tailed tests
- z-test, t-test (one-sample, two-sample, paired)
- Chi-squared test (goodness of fit, independence)
- ANOVA (one-way, two-way)
- Mann-Whitney U, Wilcoxon signed-rank (non-parametric alternatives)
- Multiple testing problem — Bonferroni correction, FDR, Benjamini-Hochberg
- Confidence intervals — construction and interpretation
- Bootstrap methods for estimation and CI

### Top Interview Questions

1. What is the difference between Type I and Type II errors? How do you balance them?
2. Explain p-value to a non-technical stakeholder.
3. You have two groups. How do you determine if their means are significantly different?
4. What is the Central Limit Theorem? Why is it important?
5. You flip a coin 1000 times and get 600 heads. Is the coin biased? Show me the test.
6. What is the difference between a confidence interval and a credible interval?
7. A/B test ran for 2 weeks. How do you know if you can stop early?
8. Two data scientists ran A/B tests on the same feature. They got different p-values. Why might this happen?
9. What is statistical power? How do you compute required sample size?
10. Explain Bayes' Theorem with a real-world example (e.g., medical testing).
11. What is the difference between correlation and causation? Give an example.
12. How would you detect if a distribution is normal? What tests would you use?
13. What are the assumptions of a linear regression? How do you validate them?
14. When would you use a non-parametric test over a parametric one?
15. Explain the Monty Hall problem and its probabilistic solution.

---

## 🗂️ SECTION 2 — SQL & Data Manipulation

Google uses SQL heavily. Expect medium-to-hard LeetCode-style questions in BigQuery SQL dialect.

### Topics to Master

- SELECT, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT
- JOINs: INNER, LEFT, RIGHT, FULL OUTER, CROSS, SELF JOIN
- Subqueries and correlated subqueries
- CTEs (Common Table Expressions) with `WITH`
- Window functions: `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LAG()`, `LEAD()`, `NTILE()`, `SUM() OVER`, `AVG() OVER`, `PARTITION BY`
- Aggregation functions: COUNT, SUM, AVG, MIN, MAX, COUNT DISTINCT
- CASE WHEN statements
- String functions: CONCAT, SUBSTRING, TRIM, UPPER, LOWER, REPLACE, LIKE, REGEXP
- Date/time functions: DATE_DIFF, DATE_ADD, EXTRACT, TIMESTAMP, STRFTIME
- NULL handling: IS NULL, COALESCE, NULLIF, IFNULL
- UNION vs UNION ALL
- Deduplication strategies
- Query optimization and indexing concepts
- Pivoting and unpivoting data
- BigQuery specific: ARRAY, STRUCT, UNNEST, partitioning, clustering

### Top Interview Questions

1. Find the second (or Nth) highest salary in a table.
2. Write a query to find users who logged in on consecutive days.
3. Calculate a 7-day rolling average of daily active users.
4. Find all users who made a purchase in January but not in February.
5. Given a sessions table, compute the average session duration per user per day.
6. Identify duplicate records and keep only the most recent one.
7. Write a query to compute month-over-month revenue growth.
8. Compute the retention rate of users from their first week to their second week.
9. Given a tree-structured table (employee-manager), find all reports under a given manager (recursive CTE).
10. For each user, find the 3 most recent events (use window functions).
11. Write a query to compute the median of a column (no built-in MEDIAN function allowed).
12. Join three tables: users, orders, and products. Find the top 5 products by revenue per region.
13. Identify sessions that lasted less than 1 minute and flag them as bounces. What % of total sessions are bounces?
14. What is the difference between WHERE and HAVING? When does each get applied?
15. Explain the performance difference between a correlated subquery and a CTE.

---

## 🗂️ SECTION 3 — Python / R for Data Analysis

Google expects clean, idiomatic Python. You should be comfortable solving problems without an IDE.

### Topics to Master

**Core Python**
- Data structures: lists, dicts, sets, tuples
- List/dict/set comprehensions
- Lambda functions, `map()`, `filter()`, `zip()`
- Error handling: `try/except/finally`
- File I/O
- OOP basics: classes, inheritance, `__init__`, `__repr__`
- Generators and iterators
- Decorators (conceptual understanding)

**NumPy**
- Array creation, indexing, slicing, reshaping
- Broadcasting rules
- Vectorized operations vs. loops (performance)
- `np.where`, `np.argmax`, `np.argsort`
- Linear algebra: `np.dot`, `np.linalg.inv`, `np.linalg.eig`
- Random number generation and seeds

**Pandas**
- Series and DataFrame creation and manipulation
- `read_csv`, `read_json`, `read_excel`
- Indexing: `.loc`, `.iloc`, boolean indexing
- `groupby()`, `agg()`, `transform()`, `apply()`
- Merging: `merge()`, `join()`, `concat()`
- Handling missing data: `fillna()`, `dropna()`, `isnull()`
- `pivot_table()`, `melt()`
- `value_counts()`, `nunique()`, `describe()`
- Time series resampling and rolling windows
- String operations via `.str` accessor

**Visualization (Matplotlib / Seaborn / Plotly)**
- Line, bar, scatter, histogram, box, violin, heatmap plots
- Subplots and figure customization
- Communicating insights visually

### Top Interview Questions

1. How do you handle missing data in a Pandas DataFrame? What are the trade-offs of different strategies?
2. What is the difference between `apply()`, `map()`, and `applymap()` in Pandas?
3. You have a DataFrame with 1 million rows. Operations are slow. How do you speed it up?
4. Explain broadcasting in NumPy. Give an example.
5. How would you compute a rolling 30-day average grouped by user_id in Pandas?
6. You receive a CSV with inconsistent date formats across rows. How do you clean it?
7. Explain the difference between deep copy and shallow copy.
8. Write a Python function to detect and remove outliers from a column using IQR.
9. How would you merge two DataFrames where keys have slight misspellings (fuzzy join)?
10. Explain the GIL in Python. Does it affect data science workloads?

---

## 🗂️ SECTION 4 — Machine Learning

For the Data Scientist role, ML is central. For the Analytics Specialist role, conceptual depth is expected.

### Topics to Master

**Supervised Learning**
- Linear Regression: OLS, assumptions, regularization (Ridge, Lasso, ElasticNet)
- Logistic Regression: log-odds, sigmoid, decision boundary, multiclass (OvR, softmax)
- Decision Trees: splitting criteria (Gini, entropy, variance), pruning, overfitting
- Random Forests: bagging, feature importance, out-of-bag error
- Gradient Boosting: XGBoost, LightGBM, CatBoost — how boosting works
- Support Vector Machines: kernels, margin, soft margin, C parameter
- k-Nearest Neighbors: distance metrics, curse of dimensionality
- Naive Bayes: assumptions, types (Gaussian, Multinomial, Bernoulli)

**Unsupervised Learning**
- K-Means Clustering: algorithm, choosing K (elbow method, silhouette score)
- Hierarchical Clustering: linkage methods, dendrograms
- DBSCAN: core points, border points, noise, epsilon and min_samples
- PCA: variance explained, eigenvectors, dimensionality reduction
- t-SNE and UMAP: intuition, use cases, limitations
- Autoencoders (basic understanding)

**Model Evaluation**
- Train/Validation/Test splits
- Cross-validation: k-fold, stratified k-fold, time series split
- Bias-Variance tradeoff
- Metrics:
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, Matthews Correlation Coefficient
  - Regression: MAE, MSE, RMSE, MAPE, R², Adjusted R²
  - Ranking: NDCG, MRR, MAP
- Confusion matrix and when to prioritize Precision vs. Recall
- Calibration of probabilities (Platt scaling, isotonic regression)
- Handling class imbalance: oversampling (SMOTE), undersampling, class weights, threshold tuning

**Feature Engineering**
- Encoding: One-Hot, Label Encoding, Target Encoding, Frequency Encoding, Embedding
- Scaling: StandardScaler, MinMaxScaler, RobustScaler, Normalizer
- Feature selection: filter methods (correlation, chi-squared), wrapper (RFE), embedded (Lasso, tree importance)
- Handling skewed distributions: log transform, Box-Cox, Yeo-Johnson
- Interaction features, polynomial features
- Imputation strategies: mean/median/mode, KNN imputation, model-based

**Deep Learning Basics (for Data Scientist role)**
- Perceptron, activation functions (ReLU, sigmoid, tanh, softmax, GELU)
- Forward and backpropagation
- Optimizers: SGD, Adam, RMSprop, AdaGrad
- Batch normalization, dropout, early stopping
- CNNs: convolution, pooling, channels — for image tasks
- RNNs, LSTMs, GRUs — for sequential/time-series data
- Transformers and attention mechanism (conceptual)
- Transfer learning and fine-tuning

### Top Interview Questions

1. Explain the bias-variance tradeoff. How do you diagnose and fix each?
2. You trained a model with 99% accuracy. Is it a good model? What else would you check?
3. What is the difference between bagging and boosting? When would you use each?
4. Explain how Random Forest handles overfitting.
5. How does XGBoost differ from a traditional Gradient Boosting Machine?
6. Your model has high recall but low precision. What does that mean? When is that acceptable?
7. How do you handle a highly imbalanced dataset (e.g., 1% positive class)?
8. What is regularization? Explain Ridge vs. Lasso geometrically.
9. How do you choose the number of clusters in K-Means?
10. Explain PCA. How many principal components would you keep?
11. What is the difference between parametric and non-parametric models?
12. Walk me through how you'd build a churn prediction model end-to-end.
13. How does a decision tree decide which feature to split on?
14. Explain the vanishing gradient problem and how it's addressed.
15. What is the ROC curve? What does AUC represent? When would you use PR-AUC instead?

---

## 🗂️ SECTION 5 — Product Sense & Analytics Case Studies

This is a **Google signature** section. They want to see you think like a product analyst.

### Topics to Master

- Defining and tracking the right metrics for a product
- North Star metrics vs. guardrail metrics
- Funnel analysis and conversion optimization
- Cohort analysis
- User segmentation
- DAU/MAU/WAU definitions and relationships
- Retention and churn analysis
- Session analysis and engagement metrics
- Root cause analysis of metric changes
- A/B testing design: hypothesis, randomization unit, metrics, duration, analysis
- Network effects and experiment interference (SUTVA violations)
- Novelty effect in experiments
- Instrumentation and event logging

### Framework for Product Questions

**CMAAD Framework:**
1. **Clarify** — What product? What user segment? What time period?
2. **Metrics** — What are we optimizing? What are guardrail metrics?
3. **Analysis** — How would I break this down? (user funnel, geo, device, time)
4. **Action** — What do the data suggest? What would I do?
5. **Decision** — Launch or not? What further data do I need?

### Top Interview Questions

1. How would you measure the success of Google Search?
2. Google Maps usage dropped by 10% last week. How do you investigate?
3. How would you design an A/B test to measure the impact of a new feature in Google Pay?
4. You're the PM for YouTube Shorts. What metrics would you track?
5. How would you define "user engagement" for Gmail? What would you measure?
6. Google launches a new feature. DAU increases but average revenue per user drops. Is this a success?
7. How would you compute the lifetime value (LTV) of a Google Ads customer?
8. You notice that a metric is trending up globally but down in India. How do you explain this?
9. How do you prioritize which metric to focus on if you have conflicting signals?
10. Explain the difference between a leading indicator and a lagging indicator. Give examples.
11. A new onboarding flow increases D1 retention by 5% but decreases D7 retention by 2%. Do you ship?
12. How would you set up a holdout experiment for a major Google product change?
13. Two features are running A/B tests simultaneously. How do you ensure they don't interfere?
14. How would you detect and handle novelty effects in an experiment?
15. If you could only track one metric for Google Photos, what would it be and why?

---

## 🗂️ SECTION 6 — Experimentation & Causal Inference

Google runs thousands of experiments. Deep knowledge here is a strong differentiator.

### Topics to Master

- Randomized Controlled Trials (RCTs)
- Observational studies vs. experiments
- Randomization: user-level vs. page-level vs. session-level
- Stratified randomization and matched pairs
- Switchback experiments
- Multi-armed bandit vs. traditional A/B testing
- Interference and SUTVA (Stable Unit Treatment Value Assumption)
- Network effects in experiments
- Instrumentation accuracy and logging bugs
- Variance reduction: CUPED (Controlled-experiment Using Pre-Experiment Data), stratification
- Sequential testing and peeking problem
- Stopping rules: fixed horizon vs. sequential
- Regression Discontinuity Design
- Difference-in-Differences (DiD)
- Instrumental Variables
- Propensity Score Matching
- Synthetic Control Method
- Uplift modeling

### Top Interview Questions

1. How do you calculate sample size for an A/B test?
2. What is the peeking problem? How do you handle it?
3. What is CUPED? Explain intuitively how it reduces variance.
4. Your A/B test result is statistically significant but the effect size is tiny (0.001%). Do you ship?
5. How do you handle network effects when running an A/B test on a social platform?
6. Explain difference-in-differences. What assumptions does it make?
7. What is an instrumental variable? Give a real-world example.
8. How would you analyze an experiment where compliance was imperfect (not everyone in treatment group used the feature)?
9. When would you use a multi-armed bandit over a classic A/B test?
10. How do you detect if there's a selection bias in your experiment?

---

## 🗂️ SECTION 7 — Data Engineering & Systems Knowledge

You don't need to be an engineer, but Google expects strong data infrastructure literacy.

### Topics to Master

- Relational databases vs. NoSQL (when to use each)
- Data warehouse concepts: fact tables, dimension tables, star schema, snowflake schema
- ETL vs. ELT pipelines
- Google Cloud Platform basics: BigQuery, Cloud Storage, Dataflow, Pub/Sub, Dataproc
- Apache Spark and MapReduce (conceptual)
- Data modeling: normalization, denormalization
- Slowly Changing Dimensions (SCD types)
- Batch processing vs. stream processing
- Data quality monitoring: null checks, schema validation, freshness, volume anomalies
- Logging and event tracking design
- Data versioning and lineage

### Top Interview Questions

1. What is the difference between a data lake and a data warehouse?
2. How would you design a data pipeline to track user events in real time?
3. You discover that a critical metric is wrong due to a logging bug. How do you handle it?
4. What is a star schema? When would you use it vs. a normalized schema?
5. How does BigQuery differ from a traditional relational database?
6. Explain the CAP theorem and how it applies to distributed data systems.
7. How would you handle late-arriving data in a streaming pipeline?
8. What are the trade-offs between batch and real-time processing?

---

## 🗂️ SECTION 8 — Communication & Storytelling with Data

Google values the ability to communicate findings clearly to technical and non-technical audiences.

### Topics to Master

- Structuring data narratives: Situation → Complication → Question → Answer (SCR/SCQA)
- Choosing the right visualization for the message
- Avoiding misleading charts (truncated axes, inappropriate aggregation, cherry-picking)
- Executive summary vs. detailed methodology
- Handling stakeholder disagreement with data
- Writing clear, concise analytical memos
- Presenting uncertainty and confidence in findings
- Translating business problems into data questions

### Top Interview Questions

1. How would you explain p-values to a VP of Product who has no statistics background?
2. You found a counter-intuitive result. How do you present it to leadership?
3. You and a stakeholder disagree on what the data says. How do you resolve it?
4. How do you decide which chart type to use for a given dataset?
5. Walk me through a time you influenced a product decision with data.

---

## 🗂️ SECTION 9 — Behavioral Questions (Googleyness & Leadership)

Google evaluates 4 core attributes: **General Cognitive Ability, Googleyness, Leadership, Role-Related Knowledge**.

### Themes to Prepare

- Ownership and initiative
- Handling ambiguity
- Cross-functional collaboration
- Disagreeing constructively
- Learning from failure
- Impact at scale
- Intellectual humility

### STAR Method (Situation, Task, Action, Result)

Always structure behavioral answers with clear **quantified results**.

### Top Behavioral Questions

1. Tell me about a time you used data to change someone's mind.
2. Describe a situation where you had to work with incomplete data. What did you do?
3. Tell me about a time you made a mistake in an analysis. How did you handle it?
4. Describe a project where you had to balance speed with rigor. How did you decide?
5. Tell me about a time you disagreed with your manager. What happened?
6. Give an example of a time you went above and beyond the scope of your role.
7. Describe a time you had to explain a complex technical concept to a non-technical audience.
8. How do you prioritize when you have multiple high-stakes analyses to deliver?
9. Tell me about the most impactful analysis you've ever done.
10. What does "Googleyness" mean to you?

---

## 🗂️ SECTION 10 — Domain Knowledge Relevant to Google

Understanding Google's business model helps you answer product and analytics questions in context.

### Key Areas

- Google Ads ecosystem: how auctions work (second-price auction, quality score, ad rank)
- Search ranking signals (conceptual)
- YouTube monetization metrics
- Google Cloud competitive landscape
- Google Maps and local search metrics
- Android and Pixel product lines
- Google Pay and fintech in India
- Privacy regulations: GDPR, DPDP Act (India), and impact on data collection
- Privacy-preserving techniques: differential privacy, federated learning
- Responsible AI and fairness in ML

### Top Questions

1. How does Google's ad auction work? What is a second-price auction?
2. How would you measure ad quality on Google Search?
3. If you were a data scientist at YouTube, how would you measure recommendation quality?
4. How does Google ensure user privacy while still doing large-scale data analysis?
5. What is differential privacy? How does Apple/Google use it?

---

## 📚 SECTION 11 — Recommended Resources

### Books

| Book | Focus |
|------|-------|
| *Practical Statistics for Data Scientists* — Bruce & Bruce | Stats, probability, ML |
| *Designing Data-Intensive Applications* — Kleppmann | Data systems |
| *The Elements of Statistical Learning* — Hastie et al. | ML theory |
| *Ace the Data Science Interview* — Hargittai & Kim | Interview-focused |
| *Naked Statistics* — Wheelan | Intuitive stats for communication |
| *Thinking, Fast and Slow* — Kahneman | Decision-making, product thinking |

### Online Practice

| Platform | Use Case |
|----------|----------|
| [LeetCode](https://leetcode.com) — Database section | SQL problems (Hard recommended) |
| [StrataScratch](https://stratascratch.com) | Real Google/FB SQL + Python questions |
| [DataLemur](https://datalemur.com) | SQL + stats interview questions |
| [Kaggle](https://kaggle.com) | ML practice and competitions |
| [Mode Analytics SQL Tutorial](https://mode.com/sql-tutorial) | SQL practice |
| [Brilliant.org](https://brilliant.org) | Statistics and probability |

### Courses

- **Stanford CS229** (Machine Learning) — YouTube, free
- **fast.ai** — Practical Deep Learning
- **Coursera: Data Science Specialization** — Johns Hopkins
- **Google's ML Crash Course** — free at developers.google.com

---

## ✅ Study Plan (12-Week Suggested Timeline)

| Week | Focus |
|------|-------|
| 1–2 | Statistics & Probability (deep review + practice questions) |
| 3–4 | SQL (LeetCode Hard + StrataScratch Google-tagged) |
| 5 | Python / Pandas / NumPy coding |
| 6–7 | Machine Learning (supervised + unsupervised + evaluation) |
| 8 | Experimentation & Causal Inference |
| 9 | Product Sense + Metric Design (mock interviews) |
| 10 | Data Engineering & Systems |
| 11 | Behavioral prep + Storytelling |
| 12 | Full mock interviews + review weak areas |

---

> 💡 **Pro Tip for India Candidates:** Google Bengaluru often focuses on **ads, cloud, and search analytics** use cases. Tailor your product examples toward Google's India-specific products — Google Pay, Google for India (Next Billion Users), YouTube, and Maps are high-signal topics.

---

*Last updated: March 2026*
