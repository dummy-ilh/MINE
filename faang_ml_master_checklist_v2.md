# FAANG ML Interview — Master Concept Checklist
### 28 Domains · ~280 Concepts · No Algorithms — Pure Understanding

> Know the *why* behind every concept. That's what cracks FAANG.

---

## 1. Regression

- [ ] Linear regression assumptions (LINE)
- [ ] Ordinary least squares (OLS) intuition
- [ ] R² and adjusted R²
- [ ] Residual analysis & heteroscedasticity
- [ ] Multicollinearity & VIF
- [ ] Ridge vs Lasso vs Elastic Net
- [ ] Polynomial regression & feature interaction
- [ ] Weighted least squares
- [ ] Outlier sensitivity
- [ ] Prediction interval vs confidence interval

---

## 2. Logistic Regression

- [ ] Sigmoid function & probability output
- [ ] Log-odds & logit transformation
- [ ] Binary vs multinomial logistic regression
- [ ] Cross-entropy loss intuition
- [ ] Decision boundary concept
- [ ] Regularization in logistic regression
- [ ] Class imbalance handling (class weights, threshold tuning)
- [ ] Odds ratio interpretation
- [ ] Softmax for multi-class
- [ ] Calibration of probabilities (Platt scaling, isotonic)

---

## 3. Classifiers

- [ ] Naive Bayes & conditional independence assumption
- [ ] k-Nearest Neighbors — curse of dimensionality
- [ ] Support Vector Machines — margin, kernel trick
- [ ] Kernel types (RBF, polynomial, linear)
- [ ] Hard vs soft margin SVM
- [ ] Linear Discriminant Analysis (LDA)
- [ ] Quadratic Discriminant Analysis (QDA)
- [ ] Perceptron & linear separability
- [ ] One-vs-rest vs one-vs-one multi-class strategy
- [ ] Precision-recall tradeoff & threshold selection

---

## 4. Evaluation & Model Selection

- [ ] Bias-variance tradeoff
- [ ] Overfitting & underfitting
- [ ] Train / val / test split discipline
- [ ] Cross-validation (k-fold, stratified, LOOCV)
- [ ] Confusion matrix (TP, TN, FP, FN)
- [ ] Precision, Recall, F1, F-beta
- [ ] AUC-ROC vs AUC-PR
- [ ] Cohen's Kappa
- [ ] Hyperparameter tuning (grid, random, Bayesian)
- [ ] Early stopping
- [ ] Data leakage

---

## 5. Trees

- [ ] Decision tree splitting criteria (Gini, Entropy, MSE)
- [ ] Information gain & gain ratio
- [ ] Pruning (pre-pruning, post-pruning)
- [ ] Depth vs leaf size tradeoffs
- [ ] Handling missing values in trees
- [ ] Categorical vs numerical splits
- [ ] Random Forest — bagging & feature subsampling
- [ ] Feature importance (impurity-based vs permutation)
- [ ] Variance reduction via ensembling
- [ ] Extrapolation limitation of trees

---

## 6. Boosting & Ensembles

- [ ] Bagging vs boosting vs stacking
- [ ] AdaBoost — sequential weighting intuition
- [ ] Gradient Boosting — residual fitting concept
- [ ] XGBoost — regularized objective, leaf scores
- [ ] LightGBM — leaf-wise growth, GOSS, EFB
- [ ] CatBoost — ordered boosting, categorical handling
- [ ] Learning rate & number of trees tradeoff
- [ ] Shrinkage & subsampling in boosting
- [ ] Out-of-bag error
- [ ] Stacking & blending

---

## 7. ML Basics & Fundamentals

- [ ] Supervised vs unsupervised vs self-supervised
- [ ] Generalization & empirical risk minimization
- [ ] No free lunch theorem
- [ ] Curse of dimensionality
- [ ] Feature engineering vs feature selection
- [ ] Semi-supervised learning
- [ ] Online vs batch learning
- [ ] Inductive vs transductive learning
- [ ] Parametric vs non-parametric models
- [ ] Data-centric vs model-centric AI

---

## 8. Probability & Statistics

- [ ] Probability distributions (Gaussian, Bernoulli, Poisson, Beta, Dirichlet)
- [ ] Bayes' theorem
- [ ] MLE vs MAP
- [ ] Expectation, variance, covariance
- [ ] Law of large numbers & CLT
- [ ] Hypothesis testing (p-value, Type I & II errors)
- [ ] Confidence intervals
- [ ] KL divergence & information theory
- [ ] Entropy & mutual information
- [ ] Conditional independence

---

## 9. Deep Learning Basics

- [ ] Forward propagation
- [ ] Backpropagation & chain rule
- [ ] Activation functions (ReLU, sigmoid, tanh, GELU, Swish)
- [ ] Weight initialization (Xavier, He)
- [ ] Batch normalization
- [ ] Layer normalization
- [ ] Dropout & regularization in NNs
- [ ] Vanishing & exploding gradients
- [ ] Residual / skip connections
- [ ] Universal approximation theorem

---

## 10. Optimization & Training

- [ ] Gradient descent (batch, mini-batch, SGD)
- [ ] Momentum, RMSProp, Adam, AdaGrad, AdamW
- [ ] Learning rate schedules (step, cosine, warmup)
- [ ] Gradient clipping
- [ ] Loss landscapes, saddle points, local minima
- [ ] Convergence criteria
- [ ] Mixed precision training (FP16, BF16)
- [ ] Gradient checkpointing
- [ ] Overfitting in NNs — data aug, dropout, weight decay
- [ ] Numerical stability

---

## 11. DL Architectures

- [ ] CNNs — convolution, pooling, receptive field, stride, padding
- [ ] RNN, LSTM, GRU — sequential modeling, gating
- [ ] Attention mechanism (self, cross, multi-head)
- [ ] Transformer — encoder, decoder, full architecture
- [ ] Positional encoding (sinusoidal, learned, RoPE, ALiBi)
- [ ] Vision Transformer (ViT)
- [ ] Encoder-decoder & seq2seq
- [ ] Autoencoders & VAE
- [ ] GANs — generator, discriminator, mode collapse, training instability
- [ ] Graph Neural Networks (GNN) — message passing concept
- [ ] Diffusion models — forward/reverse process concept

---

## 12. LLM Basics

- [ ] Tokenization (BPE, WordPiece, SentencePiece)
- [ ] Embeddings (word2vec, GloVe, contextual)
- [ ] Masked language modeling (BERT)
- [ ] Causal language modeling (GPT)
- [ ] Pretraining vs fine-tuning
- [ ] Instruction fine-tuning (SFT)
- [ ] RLHF — reward model, PPO intuition
- [ ] Scaling laws (Chinchilla)
- [ ] Emergent capabilities
- [ ] Hallucination & calibration
- [ ] Prompt engineering & in-context learning (few-shot, zero-shot)
- [ ] Parameter-efficient fine-tuning (LoRA, QLoRA, adapters, prefix tuning)

---

## 13. RAG (Retrieval-Augmented Generation)

- [ ] Why RAG — knowledge grounding vs parametric memory
- [ ] Chunking strategies (fixed, semantic, hierarchical)
- [ ] Embedding models for retrieval
- [ ] Vector databases & similarity search (cosine, dot product, L2)
- [ ] ANN indexing concepts (HNSW, IVF, FAISS)
- [ ] Sparse vs dense retrieval (BM25 vs embedding)
- [ ] Hybrid search & re-ranking
- [ ] Context window management & stuffing
- [ ] RAG evaluation (faithfulness, relevance, groundedness)
- [ ] Advanced RAG — query rewriting, HyDE, multi-hop retrieval

---

## 14. Agents

- [ ] Agent loop — perceive, reason, act
- [ ] Tool use & function calling
- [ ] ReAct framework (reason + act)
- [ ] Chain-of-thought & scratchpad reasoning
- [ ] Planning — task decomposition, tree of thought
- [ ] Memory types (in-context, external, episodic)
- [ ] Multi-agent systems & orchestration
- [ ] Reflection & self-critique loops
- [ ] Agent evaluation & benchmarking
- [ ] Safety & guardrails for agents

---

## 15. A/B Testing & Experimentation

- [ ] Null vs alternative hypothesis
- [ ] Type I (false positive) & Type II (false negative) errors
- [ ] Statistical power & sample size calculation
- [ ] p-value interpretation
- [ ] t-test, z-test — when to use which
- [ ] Confidence intervals & practical significance
- [ ] Multiple testing problem & corrections (Bonferroni, FDR)
- [ ] Novelty effect & network effects
- [ ] Switchback & interleaving experiments
- [ ] Metric selection & guardrail metrics

---

## 16. SQL & Data

- [ ] SELECT, WHERE, GROUP BY, HAVING, ORDER BY
- [ ] JOINs (inner, left, right, full, self, cross)
- [ ] Window functions (ROW_NUMBER, RANK, LEAD, LAG, SUM OVER)
- [ ] CTEs & subqueries
- [ ] Aggregations & CASE WHEN
- [ ] Deduplication & handling NULLs
- [ ] Date/time manipulation
- [ ] Query optimization & indexes conceptually
- [ ] Funnel analysis patterns
- [ ] Cohort analysis patterns

---

## 17. ML System Design

- [ ] Requirement scoping — what to optimize, constraints
- [ ] Data collection, labeling & data flywheel
- [ ] Feature store concepts
- [ ] Training pipeline (batch vs streaming)
- [ ] Model versioning & experimentation
- [ ] Serving — online vs batch inference
- [ ] Latency vs throughput vs cost tradeoffs
- [ ] Model monitoring — data drift, concept drift, performance decay
- [ ] Feedback loops & retraining triggers
- [ ] Ranking & recommendation system design
- [ ] Embedding-based retrieval at scale
- [ ] Cold start problem

---

## 18. Clustering

- [ ] k-means — centroid concept, inertia, convergence
- [ ] Choosing k (elbow method, silhouette score)
- [ ] k-means++ initialization
- [ ] DBSCAN — density, epsilon, min-points, noise points
- [ ] Hierarchical clustering — linkage types (single, complete, average, Ward)
- [ ] Dendrogram interpretation
- [ ] Gaussian Mixture Models (GMM) — soft assignment
- [ ] Cluster evaluation (silhouette, Davies-Bouldin, Calinski-Harabasz)
- [ ] Limitations of k-means (non-convex shapes, scale sensitivity)
- [ ] Spectral clustering concept

---

## 19. Dimensionality Reduction

- [ ] Why reduce dimensions — curse, noise, visualization
- [ ] PCA — variance explained, principal components, scree plot
- [ ] PCA vs LDA (supervised vs unsupervised)
- [ ] t-SNE — perplexity, crowding problem, non-linear
- [ ] UMAP — topology preservation, speed vs t-SNE
- [ ] Autoencoders for compression
- [ ] Feature selection vs feature extraction
- [ ] Variance threshold & correlation filtering
- [ ] Manifold hypothesis
- [ ] Intrinsic dimensionality concept

---

## 20. Recommender Systems

- [ ] Collaborative filtering — user-based vs item-based
- [ ] Content-based filtering
- [ ] Hybrid approaches
- [ ] Matrix factorization (SVD, ALS) — latent factors concept
- [ ] Implicit vs explicit feedback
- [ ] Cold start problem — new user, new item
- [ ] Two-tower model architecture
- [ ] Candidate retrieval vs ranking stages
- [ ] Evaluation metrics (NDCG, MRR, Hit Rate)
- [ ] Popularity bias & diversity-relevance tradeoff
- [ ] Session-based & sequential recommendation

---

## 21. Feature Engineering & Selection

- [ ] Encoding categoricals (one-hot, ordinal, target, embedding)
- [ ] Feature scaling (standardization, min-max, robust)
- [ ] Handling missing values (imputation strategies)
- [ ] Handling skewed distributions (log, Box-Cox)
- [ ] Interaction features & polynomial features
- [ ] Binning & discretization
- [ ] Filter methods (correlation, mutual info, chi-squared)
- [ ] Wrapper methods (RFE concept)
- [ ] Embedded methods (Lasso, tree importance)
- [ ] Feature stores & training-serving skew

---

## 22. Data Preprocessing & EDA

- [ ] Univariate vs bivariate vs multivariate analysis
- [ ] Outlier detection (IQR, z-score, isolation forest concept)
- [ ] Missing data mechanisms (MCAR, MAR, MNAR)
- [ ] Distribution analysis & normality tests
- [ ] Correlation (Pearson, Spearman, Kendall)
- [ ] Data type handling (numerical, categorical, datetime, text)
- [ ] Data quality checks — duplicates, inconsistencies
- [ ] Train-test distribution mismatch detection
- [ ] Class balance analysis
- [ ] EDA for time-series (stationarity, seasonality, trend)

---

## 23. Imbalanced Learning

- [ ] Why standard accuracy fails on imbalanced data
- [ ] Oversampling — SMOTE, ADASYN
- [ ] Undersampling — random, Tomek links, ENN
- [ ] Class weights in loss functions
- [ ] Threshold tuning for imbalanced classification
- [ ] Precision-recall curve over ROC for imbalance
- [ ] Cost-sensitive learning
- [ ] Ensemble methods for imbalance (EasyEnsemble, BalancedBagging)
- [ ] Evaluation: F1, G-mean, MCC
- [ ] Stratified sampling in CV

---

## 24. Self-Supervised & Contrastive Learning

- [ ] Self-supervised learning — no manual labels
- [ ] Pretext tasks (rotation prediction, masked patches, next sentence)
- [ ] Contrastive learning — positive vs negative pairs
- [ ] SimCLR — projection head, NT-Xent loss
- [ ] MoCo — momentum encoder, memory bank
- [ ] BYOL — no negatives needed
- [ ] CLIP — image-text alignment
- [ ] Representation quality & linear probe evaluation
- [ ] Data augmentation as the core learning signal
- [ ] Foundation models & pretraining at scale

---

## 25. Reinforcement Learning Basics

- [ ] MDP — state, action, reward, transition, discount
- [ ] Policy vs value function
- [ ] Exploration vs exploitation tradeoff
- [ ] Epsilon-greedy strategy
- [ ] Q-learning concept & Bellman equation
- [ ] Policy gradient intuition
- [ ] Actor-critic concept
- [ ] Model-based vs model-free RL
- [ ] Reward shaping & sparse rewards
- [ ] RL from human feedback (RLHF) connection to LLMs

---

## 26. ML on Graphs

- [ ] Graph representation — nodes, edges, adjacency matrix
- [ ] Graph types (directed, undirected, bipartite, heterogeneous)
- [ ] Node, edge, graph-level tasks
- [ ] Message passing framework
- [ ] Graph Convolutional Networks (GCN) concept
- [ ] GraphSAGE — inductive learning
- [ ] Graph Attention Networks (GAT)
- [ ] Knowledge graphs & entity embeddings
- [ ] Link prediction & node classification
- [ ] Graph-level pooling

---

## 27. Metrics & Goal Setting

- [ ] North star metric concept
- [ ] Proxy metrics & leading indicators
- [ ] Metric decomposition (tree of metrics)
- [ ] Guardrail metrics
- [ ] Goodhart's law — when a measure becomes a target
- [ ] Online vs offline metric correlation
- [ ] Short-term vs long-term metric tradeoffs
- [ ] Counter-metrics & unintended consequences
- [ ] Metric sensitivity & minimum detectable effect
- [ ] Business metrics → ML objective alignment

---

## 28. ML Ethics & Fairness

- [ ] Sources of bias (data, label, feedback loop)
- [ ] Fairness definitions (demographic parity, equalized odds, calibration)
- [ ] Fairness-accuracy tradeoff
- [ ] Disparate impact
- [ ] Interpretability vs explainability
- [ ] SHAP & LIME concepts
- [ ] Differential privacy
- [ ] Adversarial robustness & attacks
- [ ] Model cards & datasheets
- [ ] Responsible AI frameworks

---

**Total: ~280 concepts across 28 domains**

> **Suggested Study Order:**
>
> **Week 1–2 — Classical Foundation**
> Domains 1–8 (Regression → Prob/Stats)
>
> **Week 3–4 — Deep Learning**
> Domains 9–11 (DL Basics → Architectures)
>
> **Week 5 — Modern ML & LLMs**
> Domains 12–14 (LLMs → RAG → Agents)
>
> **Week 6 — Applied & Systems**
> Domains 15–17 (A/B Testing → SQL → System Design)
>
> **Week 7 — Fill Gaps**
> Domains 18–28 (Clustering → Ethics)
>
> **The FAANG formula for every concept:**
> (1) Define it → (2) Intuition → (3) Tradeoffs → (4) Real use case
>
> Go crack it. 🚀
