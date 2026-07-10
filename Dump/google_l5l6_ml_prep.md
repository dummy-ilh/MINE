# Google L5/L6 Data Scientist — ML Interview Prep
**Timeline: 60–90 Days**

---

## 📅 Phase 1: Foundations (Days 1–20)

### 1.1 Math & Statistics
- Probability theory (Bayes' theorem, conditional probability, joint/marginal distributions)
- Common distributions: Normal, Binomial, Poisson, Beta, Dirichlet
- Expectation, variance, covariance, correlation
- Hypothesis testing: t-test, chi-square, ANOVA
- p-values, confidence intervals, statistical power, Type I/II errors
- Central Limit Theorem & Law of Large Numbers

### 1.2 Information Theory
- Entropy, cross-entropy, joint entropy
- KL divergence & JS divergence
- Mutual information
- Applications in ML (decision trees, VAEs)

### 1.3 Optimization
- Gradient descent variants: SGD, Mini-batch, Momentum, Nesterov
- Adaptive methods: AdaGrad, RMSProp, Adam, AdamW
- Learning rate schedules: step decay, cosine annealing, warmup
- Convexity, saddle points, local minima
- Lagrangian optimization & KKT conditions

### 1.4 Linear Algebra
- Matrix decompositions: SVD, eigen decomposition
- Matrix calculus (Jacobian, Hessian)
- Applications in PCA, recommendation systems

---

## 📅 Phase 2: Classical ML (Days 15–35)

### 2.1 Linear Models
- Linear regression: OLS, assumptions, residual analysis
- Regularization: L1 (Lasso), L2 (Ridge), Elastic Net
- Logistic regression: log-odds, sigmoid, multinomial
- Generalized Linear Models (GLMs)

### 2.2 Tree-Based Methods
- Decision trees: splitting criteria (Gini, entropy, variance reduction)
- Pruning: pre-pruning, post-pruning
- Bagging & Random Forests: feature importance, OOB error
- Boosting: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
- Key hyperparameters and tuning strategies

### 2.3 Support Vector Machines
- Hard vs. soft margin
- Kernel trick: RBF, polynomial, linear kernels
- Dual formulation & support vectors
- SVR for regression

### 2.4 Probabilistic Models
- Naive Bayes: Gaussian, Multinomial, Bernoulli
- Gaussian Mixture Models (GMM)
- Expectation-Maximization (EM) algorithm
- Hidden Markov Models (HMM) basics

### 2.5 Dimensionality Reduction
- PCA: variance explained, scree plot, reconstruction error
- LDA (Linear Discriminant Analysis)
- t-SNE: perplexity, crowding problem
- UMAP: topology-preserving properties
- Autoencoders for dimensionality reduction

### 2.6 Clustering
- K-Means: initialization (K-Means++), inertia, elbow method
- DBSCAN: epsilon, min_samples, noise points
- Hierarchical clustering: linkage methods, dendrograms
- Cluster evaluation: silhouette score, Davies-Bouldin

---

## 📅 Phase 3: Deep Learning (Days 25–50)

### 3.1 Neural Network Fundamentals
- Perceptron, multi-layer perceptron (MLP)
- Activation functions: ReLU, Leaky ReLU, GELU, Sigmoid, Tanh, Softmax
- Backpropagation: chain rule, computation graphs
- Vanishing & exploding gradients
- Weight initialization: Xavier, He

### 3.2 Regularization & Optimization Tricks
- Dropout: inverted dropout, test-time behavior
- Batch normalization: training vs. inference behavior
- Layer normalization, group normalization
- Weight decay, early stopping
- Data augmentation strategies

### 3.3 Convolutional Neural Networks (CNNs)
- Convolution operation: stride, padding, dilation
- Pooling: max pool, average pool, global average pool
- Classic architectures: VGG, ResNet (skip connections), Inception
- Transfer learning & fine-tuning strategies

### 3.4 Sequence Models
- RNNs: unrolling, BPTT
- LSTMs: cell state, forget/input/output gates
- GRUs: simplified gating
- Sequence-to-sequence models
- Teacher forcing

### 3.5 Transformers & Attention
- Self-attention mechanism: Q, K, V matrices
- Multi-head attention
- Positional encoding
- BERT: masked LM, next sentence prediction, fine-tuning
- GPT: autoregressive language modeling
- Vision Transformers (ViT)
- Efficient transformers: Longformer, Performer

### 3.6 Embeddings
- Word2Vec: CBOW vs. Skip-gram, negative sampling
- GloVe
- Sentence embeddings: BERT-based, Sentence-BERT
- Item embeddings in recommendation systems
- Embedding similarity: cosine, dot product

---

## 📅 Phase 4: ML System Design (Days 40–65)

### 4.1 Problem Framing
- Translating business goals to ML objectives
- Defining success metrics (online vs. offline)
- Proxy metrics and their pitfalls
- Single vs. multi-task learning decisions

### 4.2 Data
- Data collection strategies: labeling, crowdsourcing, weak supervision
- Train/validation/test splits: stratification, time-based splits
- Class imbalance: oversampling (SMOTE), undersampling, class weights
- Data versioning (DVC, Delta Lake)
- Feature stores (Feast, Tecton)

### 4.3 Feature Engineering
- Numerical: scaling, binning, log transforms, polynomial features
- Categorical: one-hot, ordinal, target encoding, embeddings
- Temporal: lag features, rolling statistics, seasonality
- Interaction features
- Missing value imputation strategies

### 4.4 Training Infrastructure
- Distributed training: data parallelism, model parallelism
- Mixed precision training (FP16/BF16)
- Hyperparameter tuning: grid search, random search, Bayesian optimization
- Experiment tracking (MLflow, W&B)

### 4.5 Model Serving & Deployment
- Batch vs. real-time inference
- Model compression: pruning, quantization, knowledge distillation
- Latency vs. throughput tradeoffs
- Shadow deployment, canary releases
- Feature serving: online vs. precomputed

### 4.6 Monitoring & Maintenance
- Data drift detection: PSI, KL divergence, KS test
- Concept drift vs. data drift
- Model performance degradation signals
- Retraining strategies: scheduled vs. triggered
- Training-serving skew

---

## 📅 Phase 5: Ranking, Recommendation & Ads (Days 50–70)

### 5.1 Recommendation Systems
- Collaborative filtering: user-based, item-based
- Matrix factorization: SVD, ALS
- Neural collaborative filtering
- Two-tower models (dual encoder)
- Session-based recommendations
- Cold start problem: new user, new item strategies

### 5.2 Learning to Rank
- Pointwise: regression/classification on relevance score
- Pairwise: RankNet, LambdaRank
- Listwise: LambdaMART, SoftRank
- NDCG, MRR, MAP evaluation metrics

### 5.3 Ads & CTR Prediction
- Logistic regression baseline
- Feature interactions: FM, DeepFM, DCN
- Position bias and debias techniques
- Calibration of predicted probabilities
- Exploration vs. exploitation in ads

### 5.4 Search & Retrieval
- Inverted index, BM25
- Dense retrieval: bi-encoders, cross-encoders
- ANN search: FAISS, HNSW, ScaNN
- Re-ranking pipeline
- Query understanding: intent classification, entity extraction

---

## 📅 Phase 6: Experimentation & Causal Inference (Days 55–75)

### 6.1 A/B Testing
- Hypothesis formulation (null vs. alternative)
- Sample size calculation: MDE, power, significance level
- Multiple testing corrections: Bonferroni, BH
- Sequential testing: always-valid p-values
- Network effects & interference (SUTVA violation)
- Switchback experiments for marketplace settings

### 6.2 Observational Causal Inference
- Potential outcomes framework (Rubin)
- Average Treatment Effect (ATE), ATT, CATE
- Propensity score matching & IPW
- Difference-in-Differences (DiD): parallel trends assumption
- Synthetic control
- Instrumental variables (IV)
- Regression discontinuity design (RDD)

### 6.3 Uplift Modeling
- Meta-learners: S-learner, T-learner, X-learner
- Causal forests
- Heterogeneous treatment effects

---

## 📅 Phase 7: Advanced & L6 Differentiators (Days 65–90)

### 7.1 Probabilistic ML
- Bayesian linear regression
- Gaussian Processes
- Variational inference & ELBO
- Variational Autoencoders (VAE)
- Monte Carlo methods: MCMC, importance sampling

### 7.2 Bandits & Reinforcement Learning
- Multi-armed bandits: epsilon-greedy, UCB, Thompson Sampling
- Contextual bandits
- MDP formulation: states, actions, rewards, transitions
- Q-learning, Deep Q-Networks (DQN)
- Policy gradient methods (REINFORCE, PPO)
- RL in recommendation & ads

### 7.3 Fairness, Bias & Ethics
- Sources of bias: historical, representation, measurement
- Fairness metrics: demographic parity, equalized odds, calibration
- Tradeoffs between fairness criteria
- Debiasing techniques: re-weighting, adversarial training
- Responsible ML at scale

### 7.4 Generative Models
- GANs: generator, discriminator, training instability, mode collapse
- Diffusion models: forward/reverse process, DDPM
- Applications: data augmentation, synthetic data generation

### 7.5 Large Language Models (Applied)
- Prompt engineering & in-context learning
- Fine-tuning vs. RAG (Retrieval-Augmented Generation)
- RLHF: reward model, PPO fine-tuning
- Evaluation of LLMs: BLEU, ROUGE, human eval, LLM-as-judge
- Hallucination mitigation

---

## 📅 Phase 8: Model Evaluation (Throughout)

### 8.1 Classification Metrics
- Accuracy, precision, recall, F1-score
- AUC-ROC: interpretation, when to use
- AUC-PR: better for class imbalance
- Cohen's Kappa, MCC

### 8.2 Regression Metrics
- MAE, MSE, RMSE, MAPE
- R², adjusted R²
- Residual diagnostics

### 8.3 Calibration
- Reliability diagrams / calibration curves
- Expected Calibration Error (ECE)
- Platt scaling, isotonic regression

### 8.4 Ranking & Retrieval Metrics
- NDCG, MRR, MAP
- Precision@K, Recall@K
- Hit rate

---

## 🗓️ Suggested 90-Day Schedule

| Week | Focus Area |
|------|-----------|
| 1–2 | Math, Statistics, Optimization |
| 3–4 | Classical ML (Linear, Trees, Clustering) |
| 5–6 | Deep Learning Fundamentals |
| 7–8 | Transformers, Embeddings, NLP |
| 9–10 | ML System Design |
| 11–12 | Ranking, Recommendations, Ads |
| 13 | Experimentation & Causal Inference |
| 14–15 | Advanced Topics (Bandits, RL, Fairness) |
| 16–17 | Mock interviews, system design practice |
| 18 | Review weak areas, final prep |

---

## 📚 Recommended Resources

- **Books**: ESLII (Hastie), Deep Learning (Goodfellow), Causal Inference (Hernán & Robins)
- **Courses**: Stanford CS229, CS224N, fast.ai
- **Papers**: Attention is All You Need, Deep & Cross Network, Wide & Deep
- **Practice**: Kaggle, LeetCode (ML-focused), ML System Design by Chip Huyen
- **Mock Interviews**: Interviewing.io, Pramp

---

*Focus on depth over breadth at L5/L6. Be ready to discuss tradeoffs, failure modes, and production considerations for every topic.*
