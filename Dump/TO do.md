# ML / DS / MLE Interview Checklist
## ML Foundations
1. [ ] **ML Basics** — bias-variance tradeoff, overfitting, underfitting, cross-validation (k-fold, stratified, LOOCV), evaluation metrics (precision, recall, F1, AUC-ROC, AUC-PR), class imbalance handling
2. [ ] **Prob & Stats** — probability distributions (Gaussian, Bernoulli, Poisson, Beta), Bayes theorem, MLE vs MAP, hypothesis testing, CLT, confidence intervals, law of large numbers
3. [ ] **A/B Testing** — experiment design, p-values, statistical power, Type I/II errors, sample size calculation, multiple testing correction (Bonferroni, FDR), novelty effects, network effects, common pitfalls

## Classical ML
4. [ ] **Regression** — linear regression assumptions (LINE), ridge/lasso/elastic net regularization, multicollinearity, VIF, heteroscedasticity
5. [ ] **Logistic Regression** — log-odds, sigmoid, decision boundary, calibration, when to use vs other classifiers, multinomial extension
6. [ ] **Classifiers** — SVM (kernels, margin, support vectors), KNN (curse of dimensionality), Naive Bayes (conditional independence assumption); trade-offs and when each shines
7. [ ] **Trees** — decision trees (splitting criteria, pruning), random forests, bagging, OOB error, feature importance (impurity vs permutation)
8. [ ] **Boosting** — gradient boosting mechanics, XGBoost (regularization, tree construction), LightGBM (leaf-wise vs level-wise), CatBoost, DART, monotonic constraints, handling class imbalance; differences vs bagging

## Deep Learning
9. [ ] **DL Basics** — backprop (chain rule, vanishing/exploding gradients), activations (ReLU, GELU, Swish), optimizers (SGD, Adam, AdamW), batch norm vs layer norm, dropout, learning rate schedules (warmup, cosine decay, cyclical)
10. [ ] **DL Architectures** — CNNs (conv, pooling, receptive field), RNNs, LSTMs, GRUs (gating mechanisms), Transformers (self-attention, multi-head, positional encoding), attention variants (cross, sparse, flash)
11. [ ] **Multi-task & Transfer Learning** — fine-tuning strategies, catastrophic forgetting, prompt tuning vs full fine-tune at scale, LoRA, adapter layers, domain adaptation

## LLM & GenAI
12. [ ] **LLM Basics** — tokenization (BPE, WordPiece), pretraining objectives (CLM, MLM), RLHF, DPO, fine-tuning vs prompting, scaling laws (Chinchilla), emergent abilities
13. [ ] **RAG** — embeddings, vector DBs (FAISS, Pinecone), chunking strategies, hybrid retrieval (dense + sparse), reranking, RAG evaluation (faithfulness, relevance, groundedness)
14. [ ] **Agents** — tool use, ReAct framework, chain-of-thought, planning (MCTS, tree-of-thought), memory types (episodic, semantic, in-context)
15. [ ] **Multimodal Models** — vision-language models, contrastive learning (CLIP), image-text alignment, Gemini-style architecture basics, cross-modal attention

## Engineering
16. [ ] **SQL** — joins (all types), window functions (ROW_NUMBER, LAG, LEAD, RANK), aggregations, CTEs, query optimization (indexes, execution plans), handling NULLs
17. [ ] **Coding** — DS&A (arrays, hashmaps, graphs, trees, DP, sliding window), time/space complexity, LeetCode medium; ML-adjacent problems (matrix ops, reservoir sampling, top-K)
18. [ ] **System Design** — feature stores (online vs offline), training pipelines, online/batch/streaming serving, two-phase retrieval, embedding freshness, feature skew, monitoring, data drift detection, model cards, rollback strategies
19. [ ] **Data Engineering Basics** — Spark (RDDs, DataFrames, partitioning), Kafka (topics, partitions, consumer groups), batch vs streaming, pipeline orchestration (Airflow), data lake vs warehouse

## Specialized / High-Impact Topics
20. [ ] **Recommender Systems** — collaborative filtering, matrix factorization (SVD, ALS), content-based, two-tower models, cold start, diversity vs relevance, position bias
22. [ ] **Bandits & Exploration** — ε-greedy, UCB, Thompson sampling, contextual bandits, explore-exploit tradeoff, offline evaluation of bandit policies
23. [ ] **ML Metrics & Evaluation** — metric selection, offline vs online, ranking metrics (NDCG, MAP, MRR), calibration (Platt scaling, isotonic regression), business metric alignment, Goodhart's Law

## ML Problem Formulation & Feature Work
24. [ ] **Feature Engineering** — categorical encoding (OHE, target encoding, hashing), missing data strategies, scaling (standard, min-max, robust), feature selection, leakage detection, target encoding pitfalls, feature crosses
25. [ ] **ML Problem Formulation** — defining labels, data collection strategy, proxy metrics, metric decomposition, failure mode analysis from vague business problems; "design an ML system for X" style

## MLE / Applied Scientist Focus
26. [ ] **Model Deployment & MLOps** — versioning, CI/CD for ML, A/B testing in production, shadow deployments, canary releases, training-serving skew, latency vs accuracy tradeoffs, model governance
27. [ ] **Distributed Training** — data vs model vs pipeline parallelism, gradient accumulation, mixed precision (fp16/bf16), parameter servers vs all-reduce, TPU/GPU specifics, checkpointing strategies

## DS / Analytics Focus
28. [ ] **Causal Inference** — diff-in-diff, instrumental variables, propensity score matching, regression discontinuity, interference (SUTVA violations), quasi-experiments

## Time Series
29. [ ] **Time Series** — stationarity (ADF test, KPSS), differencing, ACF/PACF, ARIMA/SARIMA, exponential smoothing (Holt-Winters), temporal cross-validation, forecasting metrics (MAPE, SMAPE, WRMSSE), modern approaches (TFT, N-BEATS, TimesNet), when DL vs classical

## Often Forgotten
30. [ ] **Optimization** — gradient descent variants (SGD, momentum, Adam), convexity, saddle points, Lagrange multipliers, constrained optimization, learning rate sensitivity
31. [ ] **NLP Basics** — TF-IDF, Word2Vec (skip-gram, CBOW), GloVe, subword tokenization, NER, sequence labeling, pre-Transformer NLP pipeline
32. [ ] **Graph ML** — GNNs (GCN, GAT, GraphSAGE), message passing, PageRank, node/graph embeddings, use cases in search, social graphs, fraud detection
33. [ ] **Fairness & Responsible AI** — bias metrics (demographic parity, equalized odds, calibration across groups), mitigation strategies (re-weighting, adversarial debiasing), model documentation


-------------

-------------
# ML / DS / MLE Interview Checklist (Numbered)
# ML / DS / MLE Interview Checklist
## ML Foundations
1. [ ] **ML Basics** — bias-variance tradeoff, overfitting, underfitting, cross-validation (k-fold, stratified, LOOCV), evaluation metrics (precision, recall, F1, AUC-ROC, AUC-PR), class imbalance handling
2. [ ] **Prob & Stats** — probability distributions (Gaussian, Bernoulli, Poisson, Beta), Bayes theorem, MLE vs MAP, hypothesis testing, CLT, confidence intervals, law of large numbers
3. [ ] **A/B Testing** — experiment design, p-values, statistical power, Type I/II errors, sample size calculation, multiple testing correction (Bonferroni, FDR), novelty effects, network effects, common pitfalls

## Classical ML
4. [ ] **Regression** — linear regression assumptions (LINE), ridge/lasso/elastic net regularization, multicollinearity, VIF, heteroscedasticity
5. [ ] **Logistic Regression** — log-odds, sigmoid, decision boundary, calibration, when to use vs other classifiers, multinomial extension
6. [ ] **Classifiers** — SVM (kernels, margin, support vectors), KNN (curse of dimensionality), Naive Bayes (conditional independence assumption); trade-offs and when each shines
7. [ ] **Trees** — decision trees (splitting criteria, pruning), random forests, bagging, OOB error, feature importance (impurity vs permutation)
8. [ ] **Boosting** — gradient boosting mechanics, XGBoost (regularization, tree construction), LightGBM (leaf-wise vs level-wise), CatBoost, DART, monotonic constraints, handling class imbalance; differences vs bagging

## Deep Learning
9. [ ] **DL Basics** — backprop (chain rule, vanishing/exploding gradients), activations (ReLU, GELU, Swish), optimizers (SGD, Adam, AdamW), batch norm vs layer norm, dropout, learning rate schedules (warmup, cosine decay, cyclical)
10. [ ] **DL Architectures** — CNNs (conv, pooling, receptive field), RNNs, LSTMs, GRUs (gating mechanisms), Transformers (self-attention, multi-head, positional encoding), attention variants (cross, sparse, flash)
11. [ ] **Multi-task & Transfer Learning** — fine-tuning strategies, catastrophic forgetting, prompt tuning vs full fine-tune at scale, LoRA, adapter layers, domain adaptation

## LLM & GenAI
12. [ ] **LLM Basics** — tokenization (BPE, WordPiece), pretraining objectives (CLM, MLM), RLHF, DPO, fine-tuning vs prompting, scaling laws (Chinchilla), emergent abilities
13. [ ] **RAG** — embeddings, vector DBs (FAISS, Pinecone), chunking strategies, hybrid retrieval (dense + sparse), reranking, RAG evaluation (faithfulness, relevance, groundedness)
14. [ ] **Agents** — tool use, ReAct framework, chain-of-thought, planning (MCTS, tree-of-thought), memory types (episodic, semantic, in-context)
15. [ ] **Multimodal Models** — vision-language models, contrastive learning (CLIP), image-text alignment, Gemini-style architecture basics, cross-modal attention

## Engineering
16. [ ] **SQL** — joins (all types), window functions (ROW_NUMBER, LAG, LEAD, RANK), aggregations, CTEs, query optimization (indexes, execution plans), handling NULLs
17. [ ] **Coding** — DS&A (arrays, hashmaps, graphs, trees, DP, sliding window), time/space complexity, LeetCode medium; ML-adjacent problems (matrix ops, reservoir sampling, top-K)
18. [ ] **System Design** — feature stores (online vs offline), training pipelines, online/batch/streaming serving, two-phase retrieval, embedding freshness, feature skew, monitoring, data drift detection, model cards, rollback strategies
19. [ ] **Data Engineering Basics** — Spark (RDDs, DataFrames, partitioning), Kafka (topics, partitions, consumer groups), batch vs streaming, pipeline orchestration (Airflow), data lake vs warehouse

## Specialized / High-Impact Topics
20. [ ] **Recommender Systems** — collaborative filtering, matrix factorization (SVD, ALS), content-based, two-tower models, cold start, diversity vs relevance, position bias
21. [ ] **Information Retrieval** — BM25, dense vs sparse retrieval, approximate nearest neighbor (HNSW, IVF), reranking, BEIR benchmarks, query understanding, inverted indexes
22. [ ] **Bandits & Exploration** — ε-greedy, UCB, Thompson sampling, contextual bandits, explore-exploit tradeoff, offline evaluation of bandit policies
23. [ ] **ML Metrics & Evaluation** — metric selection, offline vs online, ranking metrics (NDCG, MAP, MRR), calibration (Platt scaling, isotonic regression), business metric alignment, Goodhart's Law

## ML Problem Formulation & Feature Work
24. [ ] **Feature Engineering** — categorical encoding (OHE, target encoding, hashing), missing data strategies, scaling (standard, min-max, robust), feature selection, leakage detection, target encoding pitfalls, feature crosses
25. [ ] **ML Problem Formulation** — defining labels, data collection strategy, proxy metrics, metric decomposition, failure mode analysis from vague business problems; "design an ML system for X" style

## MLE / Applied Scientist Focus
26. [ ] **Model Deployment & MLOps** — versioning, CI/CD for ML, A/B testing in production, shadow deployments, canary releases, training-serving skew, latency vs accuracy tradeoffs, model governance
27. [ ] **Distributed Training** — data vs model vs pipeline parallelism, gradient accumulation, mixed precision (fp16/bf16), parameter servers vs all-reduce, TPU/GPU specifics, checkpointing strategies

## DS / Analytics Focus
28. [ ] **Causal Inference** — diff-in-diff, instrumental variables, propensity score matching, regression discontinuity, interference (SUTVA violations), quasi-experiments

## Time Series
29. [ ] **Time Series** — stationarity (ADF test, KPSS), differencing, ACF/PACF, ARIMA/SARIMA, exponential smoothing (Holt-Winters), temporal cross-validation, forecasting metrics (MAPE, SMAPE, WRMSSE), modern approaches (TFT, N-BEATS, TimesNet), when DL vs classical

## Often Forgotten
30. [ ] **Optimization** — gradient descent variants (SGD, momentum, Adam), convexity, saddle points, Lagrange multipliers, constrained optimization, learning rate sensitivity
31. [ ] **NLP Basics** — TF-IDF, Word2Vec (skip-gram, CBOW), GloVe, subword tokenization, NER, sequence labeling, pre-Transformer NLP pipeline
32. [ ] **Graph ML** — GNNs (GCN, GAT, GraphSAGE), message passing, PageRank, node/graph embeddings, use cases in search, social graphs, fraud detection
33. [ ] **Fairness & Responsible AI** — bias metrics (demographic parity, equalized odds, calibration across groups), mitigation strategies (re-weighting, adversarial debiasing), model documentation
