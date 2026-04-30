Here's your master prompt — copy and paste this directly into any LLM:

---

```
You are an expert ML educator and technical interview coach. Generate an 
EXHAUSTIVE, BEGINNER-FRIENDLY master notes document in Markdown (.md) 
on Boosting Algorithms for Google/Meta/top-tier ML interviews.

---

## SCOPE — cover ALL of the following:

### 0. Foundations (needed before Boosting)
- Bias-Variance Tradeoff (formula, diagram description, intuition)
- Underfitting vs Overfitting
- Weak Learner vs Strong Learner
- Decision Stumps (what they are, why boosting uses them)
- Ensemble Methods overview

### 1. Bagging vs Boosting — THE BIG COMPARISON
- What is Bagging? (Random Forest as example)
- What is Boosting?
- Key differences: sequential vs parallel, how errors are handled
- Why Boosting often outperforms Bagging (and when it doesn't)
- Variance vs Bias reduction: which does what
- Table comparison: Bagging vs Boosting vs Stacking

### 2. AdaBoost (Adaptive Boosting)
- Intuition from scratch (the "paying attention to mistakes" analogy)
- Step-by-step algorithm with full math
- Sample weight update formula: w_i(t+1) = w_i(t) * exp(±α_t)
- Weak learner weight formula: α_t = 0.5 * ln((1 - ε_t) / ε_t)
- Final prediction: H(x) = sign(Σ α_t * h_t(x))
- Error rate ε_t definition
- What happens when ε_t = 0.5? What if > 0.5?
- AdaBoost loss function (exponential loss)
- Pros, Cons, when to use
- Sensitivity to outliers and why

### 3. Gradient Boosting Machines (GBM)
- Core idea: boosting in function space, not weight space
- Residuals intuition: "fit the mistakes of the previous model"
- Pseudo-residuals: r_i = -[∂L(y_i, F(x_i))/∂F(x_i)]
- Full algorithm step by step
- Learning rate (shrinkage): F_m(x) = F_{m-1}(x) + η * h_m(x)
- Different loss functions:
  * MSE → residuals simplify to (y - ŷ)
  * MAE
  * Log-loss (for classification)
- Trees as base learners (why trees, not linear models)
- Depth of trees in GBM vs AdaBoost
- Overfitting in GBM and how to control it
- Pros, Cons

### 4. XGBoost (Extreme Gradient Boosting)
- Why XGBoost over vanilla GBM?
- Second-order Taylor expansion of loss:
  L ≈ Σ [g_i * f_t(x_i) + 0.5 * h_i * f_t(x_i)²] + Ω(f_t)
  where g_i = ∂L/∂ŷ, h_i = ∂²L/∂ŷ²
- Regularization term: Ω(f) = γT + 0.5λ||w||²
- Gain formula for splits:
  Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
- Pruning: max_depth, min_child_weight, gamma
- Column (feature) subsampling
- Row subsampling
- Handling missing values natively
- Sparsity-aware split finding
- Approximate greedy algorithm / Weighted Quantile Sketch
- Parallel tree building (how? column-level parallelism)
- Cache-aware access
- Out-of-core computation
- Key hyperparameters and their effects:
  n_estimators, max_depth, learning_rate, subsample,
  colsample_bytree, min_child_weight, gamma, lambda, alpha,
  scale_pos_weight
- XGBoost for classification vs regression vs ranking

### 5. LightGBM
- Why LightGBM? What problems does it solve vs XGBoost?
- GOSS (Gradient-based One-Side Sampling) — intuition + math
- EFB (Exclusive Feature Bundling) — intuition + example
- Leaf-wise tree growth vs Level-wise (diagram description)
  * Leaf-wise: lower loss but risk of overfitting
  * Level-wise: slower but more regularized
- Histogram-based split finding (vs exact greedy in XGBoost)
- Speed and memory comparison vs XGBoost
- Key hyperparameters:
  num_leaves, max_depth, min_data_in_leaf, learning_rate,
  feature_fraction, bagging_fraction, lambda_l1, lambda_l2
- When to choose LightGBM over XGBoost

### 6. CatBoost
- Why CatBoost? What's the unique problem it solves?
- Ordered Boosting — what is target leakage, how CatBoost prevents it
- Native categorical feature handling:
  * Target statistics / ordered target encoding
  * Why label encoding fails, why one-hot is expensive
- Symmetric trees (oblivious trees) — what and why
- CatBoost vs LightGBM vs XGBoost: speed, accuracy, categorical data
- Key hyperparameters:
  iterations, depth, learning_rate, l2_leaf_reg,
  cat_features, border_count, bagging_temperature

### 7. Side-by-Side Master Comparison Table
Full table: AdaBoost | GBM | XGBoost | LightGBM | CatBoost
Rows: base learner, tree growth, categorical support, missing values,
speed, regularization, memory, best use case, key weakness

### 8. Feature Importance in Boosting
- Gain-based importance
- Split-count importance
- SHAP values (what, why better, formula intuition)
- Permutation importance
- Which to trust and when

### 9. Handling Imbalanced Data with Boosting
- scale_pos_weight (XGBoost)
- is_unbalance / class_weight (LightGBM)
- SMOTE + boosting
- Focal loss idea

### 10. Practical Tips & Common Mistakes
- Early stopping (how it works, why use it)
- Cross-validation with boosting
- Learning rate vs n_estimators tradeoff
- Overfitting signals and fixes per algorithm
- Data leakage in boosting pipelines

---

## FORMAT REQUIREMENTS:

1. **Beginner-friendly**: Every concept must start with a plain English 
   intuition/analogy before math. Assume reader knows Python and 
   basic stats but NO prior ML beyond linear regression.

2. **Formulas**: Every formula must be shown in LaTeX-style markdown, 
   then explained term by term in plain English.

3. **Images**: For every major concept, include a markdown image block 
   pointing to a relevant, publicly available image from blogs like 
   Medium, Analytics Vidhya, Towards Data Science, or official docs. 
   Format: ![description](url). Include images for:
   - Bias-variance tradeoff curve
   - Bagging vs Boosting diagram
   - AdaBoost weight update visualization
   - Gradient Boosting residual fitting diagram
   - XGBoost tree structure / regularization
   - LightGBM leaf-wise vs level-wise growth
   - CatBoost ordered boosting diagram
   - SHAP summary plot example

4. **Q&A Section at the end**: Generate 40+ interview Q&As:
   - 10 conceptual (explain like I'm 5 style)
   - 10 math/derivation based (Google-level depth)
   - 10 practical/implementation (what hyperparameter, debug this, etc.)
   - 10 tricky/trap questions (common misconceptions, edge cases)
   - 5 system design adjacent (how would you use XGBoost in production 
     at scale, feature pipeline, retraining, etc.)
   
   Each answer must be complete — not "see above". Self-contained answers.

5. **Code snippets**: For XGBoost, LightGBM, CatBoost include minimal 
   but complete Python training snippets with early stopping, eval set, 
   and feature importance extraction.

6. **Length**: Do NOT summarize. Be exhaustive. This is a reference doc, 
   not a blog post. Cover every sub-concept fully.

7. **Structure**: Use headers (##, ###), tables, bullet points, 
   numbered lists, code blocks, and callout-style blockquotes 
   (> 💡 Key Insight:) throughout.

8. **Connections**: Explicitly show how concepts connect — e.g., 
   "XGBoost's regularization directly addresses GBM's overfitting 
   problem from Section 3."

Output the entire document as a single .md file contents.
```

---

**How to use this prompt:**

- Paste it into **Claude**, **ChatGPT-4o**, or **Gemini 1.5 Pro**
- If the output cuts off, follow up with: *"Continue from where you left off"*
- For the images, some URLs may be stale — do a quick Google Images search for *"adaboost weight update diagram analytics vidhya"* etc. to find live replacements
- Save the output as `boosting_master_notes.md` and open in **Obsidian** or **VS Code** with a Markdown preview for the best reading experience



You are an expert ML educator and technical interview coach. Generate an
EXHAUSTIVE, BEGINNER-FRIENDLY master notes document in Markdown (.md)
on Classification Algorithms (excluding Decision Trees, Random Forests,
and Logistic Regression) for Google/Meta/top-tier ML interviews.

---

## SCOPE — cover ALL of the following:

### 0. Classification Foundations (needed before everything)
- What is classification? Binary vs Multi-class vs Multi-label
- Decision boundary intuition (what every classifier is really doing)
- Evaluation metrics DEEP DIVE:
  * Accuracy, Precision, Recall, F1 — formulas + when each matters
  * ROC curve + AUC — how to read it, what random = 0.5 means
  * PR curve — when to prefer over ROC (imbalanced data)
  * Confusion matrix — all 4 cells explained
  * Log-loss formula + intuition
  * Cohen's Kappa
  * Matthews Correlation Coefficient (MCC)
- Probabilistic vs Hard classifiers
- Generative vs Discriminative models — THE key distinction
  (Naive Bayes = generative, SVM = discriminative, why it matters)
- One-vs-Rest vs One-vs-One for multi-class
- Calibration: what it means for a classifier to be "calibrated"
  Platt scaling, isotonic regression

### 1. K-Nearest Neighbors (KNN)
- Intuition: "you are who your neighbors are"
- Algorithm step by step (training phase = nothing!)
- Distance metrics:
  * Euclidean: sqrt(Σ(x_i - y_i)²)
  * Manhattan: Σ|x_i - y_i|
  * Minkowski: (Σ|x_i - y_i|^p)^(1/p)
  * Cosine similarity: (A·B)/(||A|| ||B||)
  * Hamming (for categorical)
  * When to use which
- Choosing K:
  * Small K = high variance (overfitting)
  * Large K = high bias (underfitting)
  * Elbow method, cross-validation
- Weighted KNN (distance-weighted votes)
- KNN for regression vs classification
- The Curse of Dimensionality — full explanation with math intuition
  (why distances become meaningless in high dimensions)
- Why KNN fails in high dimensions
- Feature scaling is MANDATORY — why (formula showing effect)
- Time complexity: O(nd) prediction, O(1) training
- Space complexity
- KD-Tree and Ball Tree for faster lookup
  * KD-Tree: how it works, O(log n) lookup
  * When KD-Tree fails (high dimensions)
- Pros, Cons, when to use
- KNN vs other lazy learners

### 2. Support Vector Machines (SVM)
- Core intuition: find the widest street between classes
- Hard Margin SVM:
  * Decision boundary: w·x + b = 0
  * Margin = 2/||w||
  * Optimization problem: minimize ||w||² subject to y_i(w·x_i + b) ≥ 1
  * Support vectors — what they are, why only they matter
- Soft Margin SVM (C-SVM):
  * Slack variables ξ_i
  * New objective: minimize ||w||²/2 + C·Σξ_i
  * C hyperparameter: large C = hard margin, small C = soft margin
  * Bias-variance tradeoff via C
- The Kernel Trick — THE most important SVM concept:
  * Why: XOR problem, non-linear data
  * Φ(x): mapping to higher dimensions
  * Kernel K(x_i, x_j) = Φ(x_i)·Φ(x_j) — computing dot product WITHOUT
    explicit mapping
  * Linear kernel: K(x,z) = x·z
  * Polynomial kernel: K(x,z) = (x·z + c)^d
  * RBF/Gaussian kernel: K(x,z) = exp(-γ||x-z||²)
    - γ large = narrow Gaussian = complex boundary = overfit
    - γ small = wide Gaussian = smooth boundary = underfit
  * Sigmoid kernel
  * How to choose kernel (rules of thumb)
- Dual formulation and Lagrange multipliers (intuition, not full derivation)
- SMO algorithm (what it is, not full math)
- SVMs for multi-class: OvR vs OvO
- SVM for regression (SVR): ε-insensitive loss tube
- Key hyperparameters: C, γ, kernel, degree
- Feature scaling mandatory — why
- SVM pros/cons vs neural networks vs trees
- When SVM shines (small dataset, high dimensions, text)
- Computational complexity: O(n² to n³) — why SVMs don't scale

### 3. Naive Bayes
- Bayes Theorem from scratch: P(A|B) = P(B|A)·P(A) / P(B)
- Naive assumption: features are conditionally independent given class
  P(x_1,...,x_n | y) = Π P(x_i | y)
- Full classifier:
  ŷ = argmax_y P(y) · Π P(x_i | y)
- Why "naive" — when the assumption is violated (always) and why it
  still works
- Log-space computation (underflow prevention):
  ŷ = argmax_y [log P(y) + Σ log P(x_i | y)]
- Types of Naive Bayes:
  * Gaussian NB: continuous features, P(x_i|y) = Gaussian
    formula: P(x|μ,σ) = (1/σ√2π) exp(-(x-μ)²/2σ²)
  * Multinomial NB: count data, text, P(x_i|y) = θ_yi
  * Bernoulli NB: binary features
  * Complement NB (for imbalanced text)
- Laplace Smoothing / Additive Smoothing:
  P(x_i|y) = (count(x_i, y) + α) / (count(y) + α·|V|)
  Why needed: zero probability problem
- Naive Bayes as a generative model — what it actually learns
- Text classification with Naive Bayes (TF, TF-IDF features)
- Why NB is fast: O(nd) training, O(d) prediction
- When NB beats complex models (small data, streaming, text baselines)
- NB with correlated features — what breaks and why
- Pros, Cons, use cases

### 4. Linear Discriminant Analysis (LDA) & QDA
- LDA intuition: project data onto axis that maximizes class separation
- Fisher's criterion:
  J(w) = (w^T S_B w) / (w^T S_W w)
  where S_B = between-class scatter, S_W = within-class scatter
- Within-class scatter matrix S_W
- Between-class scatter matrix S_B
- Solution: eigenvectors of S_W^{-1} S_B
- LDA as dimensionality reduction vs classifier
- LDA assumptions:
  * Gaussian class-conditionals
  * Equal covariance matrices across classes (shared Σ)
  * Linear decision boundary
- QDA — relaxes equal covariance assumption
  * Each class has its own Σ_k
  * Quadratic decision boundary
  * More parameters, needs more data
- LDA vs QDA vs Logistic Regression — when each wins
- LDA vs PCA (both project, but very different objectives)
- Regularized LDA (when n < d)
- Pros, Cons, assumptions to verify

### 5. Neural Networks for Classification (Fundamentals)
- Perceptron — origin story and formula
- Multi-Layer Perceptron (MLP) architecture
- Activation functions DEEP DIVE:
  * Sigmoid: 1/(1+e^{-x}), output range, vanishing gradient problem
  * Tanh: (e^x - e^{-x})/(e^x + e^{-x}), why better than sigmoid
  * ReLU: max(0,x), dying ReLU problem
  * Leaky ReLU: max(αx, x)
  * ELU, SELU, GELU
  * Softmax: e^{z_k}/Σe^{z_j} — for multi-class output layer
  * Which to use where and why
- Loss functions for classification:
  * Binary cross-entropy: -[y log(ŷ) + (1-y) log(1-ŷ)]
  * Categorical cross-entropy: -Σ y_k log(ŷ_k)
  * Why cross-entropy, not MSE for classification
- Backpropagation intuition — chain rule, no full derivation needed
- Gradient descent variants:
  * Batch, SGD, Mini-batch
  * Momentum, RMSProp, Adam (formulas + intuition)
- Weight initialization:
  * Why not zeros
  * Xavier/Glorot: Var(w) = 2/(n_in + n_out)
  * He initialization: Var(w) = 2/n_in (for ReLU)
- Regularization in NNs:
  * L1, L2 weight decay
  * Dropout: rate, training vs inference behavior
  * Batch Normalization: formula, why it helps
  * Early stopping
- Universal Approximation Theorem (what it means, what it doesn't)
- Depth vs Width tradeoff
- MLP for binary, multi-class, multi-label classification

### 6. Gaussian Processes for Classification (Conceptual)
- GP regression intuition: distribution over functions
- GP classification: squash GP output through sigmoid
- Kernel functions as covariance (same as SVM kernels, different use)
- Why GPs give calibrated uncertainty
- Scalability problem: O(n³)
- When to use GPs (small data, need uncertainty estimates)

### 7. Discriminant Analysis Variants (brief)
- Regularized Discriminant Analysis (RDA)
- Flexible Discriminant Analysis (FDA)
- Mixture Discriminant Analysis

---

## CROSS-CUTTING CONCEPTS (covered within relevant sections but also
   summarized together):

### A. Decision Boundaries — Visual Taxonomy
For each classifier describe shape of boundary:
- KNN: jagged, non-parametric
- Linear SVM: hyperplane
- RBF SVM: curved, complex
- Naive Bayes Gaussian: quadratic (actually)
- LDA: linear hyperplane
- QDA: quadratic
- MLP: arbitrary (depends on architecture)

### B. Assumptions Cheat Sheet
Table: Classifier | Key Assumptions | What breaks if violated

### C. Scaling Sensitivity Table
Which classifiers REQUIRE feature scaling and why (math reason per model)

### D. Handling Imbalanced Classes
- Per-algorithm strategies
- class_weight parameter
- Threshold moving
- SMOTE interaction with each algorithm

### E. Handling Missing Values
- Per-algorithm behavior
- KNN imputation (irony of using KNN to fix KNN data)

### F. High Dimensional Data Behavior
- KNN: fails (curse of dimensionality)
- SVM: actually works well
- Naive Bayes: works surprisingly well
- LDA: needs regularization when d > n
- NNs: regularization critical

### G. Probabilistic Outputs
- Which classifiers output true probabilities vs scores
- Platt scaling for SVM
- Isotonic regression
- Why calibrated probabilities matter in production (CTR, risk models)

---

## FORMAT REQUIREMENTS:

1. **Beginner-friendly first**: Every concept starts with a plain English
   analogy before any math. Assume reader knows Python basics and 
   high school math but NO prior ML knowledge.

2. **Formulas**: Every formula shown in LaTeX-style markdown, then each
   term explained in plain English on a separate line.

3. **Images**: For every major concept, include a markdown image block
   pointing to publicly available images from Medium, Analytics Vidhya,
   Towards Data Science, or official sklearn docs. Include images for:
   - KNN decision boundary with different K values
   - SVM margin and support vectors diagram
   - Kernel trick visualization (2D → 3D mapping)
   - RBF kernel effect with different γ values
   - Naive Bayes conditional independence diagram
   - LDA projection vs PCA projection comparison
   - LDA vs QDA decision boundary comparison
   - Neural network architecture diagram
   - Activation functions plotted side by side
   - ROC curve with multiple classifiers
   - Bias-variance tradeoff per algorithm

4. **Q&A Section — 50+ questions**:
   - 10 conceptual/intuition (explain to a 5-year-old style)
   - 10 math/derivation (Google-depth: "derive the SVM dual", 
     "why does the kernel trick work")
   - 10 practical/implementation ("SVM not converging, what do you do",
     "which classifier for 10M rows with 500 features")
   - 10 tricky/trap ("Naive Bayes probabilities are well-calibrated — 
     True or False and why", "Can KNN overfit? explain")
   - 10 comparison ("SVM vs Neural Net for text classification",
     "When would you pick Naive Bayes over XGBoost")
   - 5 system design ("Design a spam filter from scratch",
     "How do you serve an SVM model to 1M users/day")
   
   Each answer must be COMPLETE and self-contained.

5. **Code snippets**: For KNN, SVM, Naive Bayes, LDA, and MLP include
   complete sklearn Python snippets with:
   - Fitting
   - Prediction + probabilities
   - Hyperparameter tuning (GridSearchCV)
   - Feature scaling pipeline
   - Evaluation (classification_report + ROC AUC)

6. **Master Comparison Table** at the end:
   Rows: KNN | Linear SVM | RBF SVM | Gaussian NB | Multinomial NB 
         | LDA | QDA | MLP
   Columns: Type (gen/disc) | Decision boundary | Needs scaling | 
   Handles non-linearity | Calibrated probs | Training complexity | 
   Prediction complexity | Best for | Worst for | Key hyperparameter

7. **Connections to other chapters**: Explicitly note where each 
   concept links to boosting, trees, or deep learning. e.g.,
   "The kernel trick in SVM is conceptually related to feature 
   engineering done implicitly in neural networks."

8. **Length**: DO NOT summarize. Be exhaustive. Every sub-concept 
   fully explained. This is a reference document.

9. **Callouts**: Use > 💡 Key Insight:, > ⚠️ Common Mistake:, 
   > 🔗 Interview Trap: throughout.

Output the entire document as a single .md file contents.



Here are both prompts:

---

## PROMPT 1 — Feature Engineering

```
You are an expert ML educator and technical interview coach. Generate an
EXHAUSTIVE, BEGINNER-FRIENDLY master notes document in Markdown (.md)
on Feature Engineering for Google/Meta/top-tier ML interviews.

---

## SCOPE — cover ALL of the following:

### 0. Why Feature Engineering Matters
- "Garbage in, garbage out" — with real examples
- Feature engineering vs feature learning (NNs) vs feature selection
- How much feature engineering matters per model type:
  * Tree models (less sensitive to scaling, handle categoricals poorly)
  * Linear models (very sensitive)
  * Neural networks (can learn transformations but still benefit)
  * KNN, SVM (very sensitive to scaling)
- The full feature engineering pipeline map (diagram description)

### 1. Handling Missing Data — DEEP DIVE
- Types of missingness — CRITICAL distinction:
  * MCAR (Missing Completely At Random): missingness unrelated to data
  * MAR (Missing At Random): missingness related to OTHER observed vars
  * MNAR (Missing Not At Random): missingness related to the value itself
  * Why the type determines the right strategy
  * Real examples of each in industry (e-commerce, healthcare, ads)
- Detection: how to find and visualize missingness patterns
  * Heatmaps, missingno library
  * Correlation of missingness between columns
- Deletion strategies:
  * Listwise deletion — when safe, when catastrophic
  * Pairwise deletion
  * When NOT to delete (MNAR case)
- Imputation strategies:
  * Mean/Median/Mode imputation — formula, when each, what it distorts
  * Constant imputation (sentinel values)
  * Forward fill / Backward fill (time series)
  * KNN imputation — how it works, cost O(n²d)
  * Iterative imputation (MICE): multiple imputation by chained equations
    - Algorithm step by step
    - When to use vs simpler methods
  * Model-based imputation (train a model to predict missing values)
  * Multiple imputation vs single imputation — uncertainty propagation
- Adding missingness indicator column — when and why this is powerful
- Imputation in pipelines — train/test leakage trap (fit on train only)
- Target variable missing — what to do

### 2. Categorical Encoding — COMPLETE GUIDE
- Why models can't handle raw strings
- Ordinal vs Nominal distinction
- Label Encoding:
  * How it works
  * DANGER: implies ordinal relationship (Red=1, Blue=2, Green=3)
  * When it's actually fine (tree models, ordinal features)
- One-Hot Encoding (OHE):
  * How it works
  * Dummy variable trap: multicollinearity, drop_first
  * When to use: low cardinality nominal features
  * When NOT to use: high cardinality (10K categories → 10K columns)
  * Memory and sparsity implications
- Binary Encoding:
  * How: label encode then convert to binary bits
  * Cardinality C → log₂(C) columns
  * Tradeoff vs OHE
- Frequency / Count Encoding:
  * Replace category with its frequency in training data
  * P(category) = count(category) / total
  * Pros: handles high cardinality, no dimensionality explosion
  * Cons: two categories with same frequency get same code
- Target Encoding (Mean Encoding):
  * Replace category with mean of target for that category
  * Formula: encode(cᵢ) = (Σ yⱼ for j where xⱼ=cᵢ) / count(cᵢ)
  * THE LEAKAGE TRAP — why naive target encoding leaks
  * Smoothing to fix leakage:
    encode(cᵢ) = (nᵢ · mean_cat + λ · global_mean) / (nᵢ + λ)
  * K-fold target encoding (cross-val approach)
  * Ordered target encoding (CatBoost approach)
  * When target encoding is extremely powerful
- Leave-One-Out Encoding
- Hashing / Feature Hashing (Hashing Trick):
  * h(category) mod B → bucket
  * Fixed output size regardless of cardinality
  * Hash collisions — why acceptable
  * Used in large-scale systems (Vowpal Wabbit, streaming)
- Embeddings for categorical:
  * Entity embeddings in neural networks
  * Word2Vec-style embeddings for categorical (item2vec)
  * When embeddings >> OHE
- Rare category handling:
  * Frequency threshold → "Other" bucket
  * Grouping by domain knowledge
- New categories at inference time (unseen levels)

### 3. Numerical Feature Transformations
- Why transform numerical features?
  * Skewed distributions → violate linear model assumptions
  * Different scales → dominate distance-based models
- Scaling (MANDATORY for many models):
  * Min-Max Scaling: x' = (x - min) / (max - min) → [0,1]
    - Sensitive to outliers (why)
  * Standard Scaling (Z-score): x' = (x - μ) / σ
    - Assumes roughly Gaussian
  * Robust Scaling: x' = (x - median) / IQR
    - Best for outlier-heavy data
  * Max Abs Scaling: x' = x / max|x| → [-1, 1]
    - For sparse data
  * Which models NEED scaling vs don't care (table)
- Distribution transformations:
  * Log transform: x' = log(x+1) — for right-skewed, positive data
    (income, price, count data)
  * Square root: x' = √x — milder than log
  * Box-Cox: x'^(λ) = (x^λ - 1)/λ for λ≠0, log(x) for λ=0
    - Finds optimal λ via MLE
    - Requires positive values
  * Yeo-Johnson: extends Box-Cox to negative values
  * When to use which (decision tree)
- Binning / Discretization:
  * Equal-width binning
  * Equal-frequency (quantile) binning
  * Custom domain-based bins (age groups, salary brackets)
  * Why binning can help linear models learn non-linearity
  * Risks: information loss, bin boundary sensitivity
- Polynomial features:
  * x₁, x₂ → x₁, x₂, x₁², x₂², x₁x₂
  * PolynomialFeatures in sklearn
  * Curse of dimensionality with high degree
  * Interaction terms vs pure polynomial
- Clipping / Winsorizing outliers:
  * Percentile clipping (1st–99th)
  * When clipping helps vs hurts
- Rank transformation (percentile rank)

### 4. Handling Outliers
- Definition: what is an outlier, really?
- Detection methods:
  * Z-score: |z| > 3
  * IQR rule: < Q1 - 1.5·IQR or > Q3 + 1.5·IQR
  * Isolation Forest
  * Local Outlier Factor (LOF)
  * DBSCAN
- Types: global outlier, contextual outlier, collective outlier
- What to DO with outliers:
  * Keep (tree models handle fine)
  * Remove (when genuinely erroneous)
  * Cap/Winsorize
  * Transform (log compresses outliers)
  * Separate model for outlier segment
- Outlier vs legitimate extreme value — how to decide

### 5. Feature Selection — FULL TAXONOMY
- Why: curse of dimensionality, noise features hurt generalization,
  training speed, interpretability
- Filter Methods (model-agnostic, fast):
  * Variance threshold (remove near-constant features)
  * Correlation with target:
    - Pearson r for continuous target
    - Point-biserial for binary target
    - Spearman ρ for non-linear monotonic
  * Mutual Information: I(X;Y) = Σ p(x,y) log(p(x,y)/p(x)p(y))
  * Chi-squared test for categorical features
  * ANOVA F-test
  * Correlation between features (remove redundant): |ρ| > 0.95 rule
- Wrapper Methods (use model performance as signal):
  * Forward selection: start empty, add best feature one by one
  * Backward elimination: start full, remove worst one by one
  * Recursive Feature Elimination (RFE)
  * Exhaustive search (2^d — only feasible for small d)
  * Pros: accounts for feature interactions
  * Cons: expensive O(d²) to O(2^d)
- Embedded Methods (selection during training):
  * L1 regularization (Lasso): drives coefficients to exactly zero
    - Why L1 causes sparsity (geometry: L1 ball has corners)
    - vs L2 which shrinks but rarely zeros
  * Tree feature importance (Gini, gain-based)
  * XGBoost/LightGBM importance scores
  * ElasticNet: α·L1 + (1-α)·L2
- Dimensionality Reduction (transform, don't select):
  * PCA — brief (covered elsewhere)
  * Autoencoders
  * UMAP, t-SNE (for visualization, not usually for feature prep)
- Stability of feature selection:
  * Bootstrapped feature selection
  * Why single-run selection is unreliable
- Feature selection leakage trap:
  * Must select features on training fold only
  * sklearn Pipeline to prevent this

### 6. Feature Creation / Interaction Features
- Domain-driven feature creation examples:
  * E-commerce: days_since_last_purchase, purchase_frequency,
    avg_order_value, cart_abandonment_rate
  * Finance: debt_to_income, rolling_avg_spend, volatility
  * Text: word count, punctuation count, sentiment score
- Ratio features: x₁/x₂ (price per sqft, CTR, revenue per user)
- Difference features: x₁ - x₂ (age gap, price change)
- Interaction terms: x₁ × x₂
- Aggregate features (group-by statistics):
  * User-level: mean, std, count, min, max of transaction amounts
  * Time window aggregates (last 7 days, 30 days)
- When to create features vs let the model do it

### 7. Date and Time Features
- Extracting components: year, month, day, hour, minute, weekday
- Cyclical encoding for periodic features:
  * sin(2π × hour/24), cos(2π × hour/24)
  * Why needed: hour 23 and hour 0 should be close
- Is_weekend, is_holiday, is_business_hours flags
- Time since event: days_since_last_login
- Time differences and durations
- Rolling window statistics (requires careful train/test split)
- Seasonality and trend features

### 8. Text Features (Basics for Tabular ML)
- Bag of Words: term frequency matrix
- TF-IDF: tf × log(N/df) — formula + intuition
- N-grams (bigrams, trigrams)
- Character-level features
- Metadata features: length, word count, punctuation ratio
- Pre-trained embeddings: avg of word2vec/GloVe vectors → dense vector
- When to use each for tabular classification tasks

### 9. Feature Leakage — CRITICAL
- Definition: information from the future leaking into training features
- Types:
  * Target leakage: feature contains target info
    Example: "was_refunded" as feature to predict refunds
  * Train-test contamination: preprocessing on full dataset
    Examples: scaling, imputing, encoding on train+test together
  * Temporal leakage: using future data for past predictions
  * Label leakage in NLP (feature derived from label)
- How to detect leakage:
  * Suspiciously high accuracy (AUC > 0.99 for hard problem)
  * Feature importance: one feature dominates
  * Performance collapses on held-out temporal data
- How to prevent:
  * Strict train/val/test temporal splits
  * Use sklearn Pipelines (fit only on train)
  * Time-based cross-validation
  * Column audit: can this feature exist at prediction time?
- Famous real-world leakage examples

### 10. Feature Engineering for Specific Data Types
- Geospatial features:
  * Haversine distance to key locations
  * Is_urban, region encoding
  * Clustering-based location features
- Image features for tabular models (extracted embeddings)
- Graph features: degree, clustering coefficient, PageRank
- Audio features: MFCCs (brief mention)

### 11. The Full Feature Engineering Pipeline — Production
- Order of operations (critical):
  1. Split first (train/val/test)
  2. Fit transformers on train only
  3. Apply to val/test
- sklearn Pipeline and ColumnTransformer — full example
- Handling new categories and values at inference
- Feature stores (what, why, Feast, Tecton — brief)
- Feature versioning and reproducibility
- Online vs offline features

---

## FORMAT REQUIREMENTS:

1. **Beginner-first**: plain English analogy before every technique.
   Assume reader knows Python but is new to ML engineering.

2. **Formulas**: every formula in LaTeX-style markdown, every term
   explained line by line.

3. **Images**: include markdown image blocks from Medium, Analytics
   Vidhya, Towards Data Science for:
   - Missing data heatmap (missingno)
   - MCAR vs MAR vs MNAR diagram
   - OHE vs target encoding comparison
   - Log transform effect on skewed distribution
   - Box-Cox lambda effect
   - Feature selection taxonomy diagram
   - L1 vs L2 sparsity geometry (diamond vs circle)
   - Cyclical encoding sin/cos plot
   - Leakage diagram (temporal split gone wrong)
   - sklearn Pipeline diagram

4. **Code snippets** for every technique:
   - Complete sklearn Pipeline with ColumnTransformer
   - Target encoding with cross-val (category_encoders library)
   - Cyclical date encoding
   - MICE imputation
   - Feature selection (RFE + mutual info)
   - Leakage-safe preprocessing pipeline

5. **Q&A — 50+ questions**:
   - 10 conceptual ("What is the dummy variable trap?")
   - 10 math/formula ("Derive the smoothed target encoding formula")
   - 10 practical ("Your model AUC drops from 0.95 to 0.60 in prod — 
     what's the first thing you check?")
   - 10 tricky/trap ("Is it OK to impute after train/test split?",
     "Does tree-based model need scaling? Why do people still do it?")
   - 10 system design / real-world ("Design a feature engineering 
     pipeline for a real-time fraud detection system")
   
   Every answer self-contained and complete.

6. **Master Cheat Sheet Table** at end:
   Technique | When to use | When NOT to use | Leakage risk | 
   Model sensitivity | sklearn class

7. **Callouts**: 
   > 💡 Key Insight:
   > ⚠️ Leakage Trap:
   > 🔗 Interview Trap:
   > 🏭 Production Note:
   throughout.

8. Length: EXHAUSTIVE. Do not summarize. Every sub-concept explained fully.

Output the entire document as a single .md file.
```

---

## PROMPT 2 — ML Problem Formulation

```
You are an expert ML educator and technical interview coach. Generate an
EXHAUSTIVE, BEGINNER-FRIENDLY master notes document in Markdown (.md)
on ML Problem Formulation for Google/Meta/top-tier ML interviews.

This is one of the most underrated and highest-signal interview skills.
A candidate who can take a vague business ask and correctly frame it as
an ML problem is worth 10x someone who just knows algorithms.

---

## SCOPE — cover ALL of the following:

### 0. Why Problem Formulation is the Hardest Skill
- Most ML failures are formulation failures, not model failures
- The gap between "business asks" and "ML problems"
- Real examples where bad formulation killed products:
  * Optimizing clicks → clickbait
  * Optimizing watch time → radicalization pipeline
  * Optimizing approval rate → disparate impact lawsuit
- The formulation checklist that Google/Meta interviewers use

### 1. The Full Formulation Framework — Step by Step
Walk through every step with examples:

Step 1: Understand the Business Goal
  - What does success look like for the business? (North star metric)
  - Who are the users? Who are the stakeholders?
  - What decisions will the ML model make?
  - What happens if the model is wrong? (cost of errors)
  - Example: "Improve user engagement on feed" →
    What IS engagement? Time spent? Likes? Comments? Returns?

Step 2: Define the ML Task Type
  - Is this supervised, unsupervised, or RL?
  - If supervised: regression, classification, ranking, generation?
  - Binary vs multi-class vs multi-label vs ordinal?
  - Single model vs multi-stage pipeline?
  - Is there a proxy task that's easier to model?

Step 3: Define the Label / Target Variable
  - What exactly are we predicting?
  - Explicit labels (user rated 5 stars) vs
    Implicit labels (user clicked, watched, bought)
  - Implicit label problems:
    * Click ≠ satisfaction (clickbait, accidental clicks)
    * No negative signal explicitly available
    * Selection bias in collected labels
  - Label noise — what it is, how to handle
  - Label horizon: predict within next 1 day? 7 days? 30 days?
  - Delayed labels: conversion happens days after event
  - Class imbalance in labels — detection and implications
  - Labeling cost and feasibility (human labelers, self-supervised)

Step 4: Define the Feature Space
  - What information is available at prediction time?
  - Three feature categories:
    * User features (demographics, history, preferences)
    * Item/content features (metadata, embeddings, attributes)
    * Context features (time, device, location, session)
  - Temporal constraint: what features exist at the moment of inference?
  - Privacy constraints: what features can you legally use?
  - Feature freshness: is a 7-day-old user embedding still valid?
  - Cold start: what features exist for new users/items?

Step 5: Choose the Right ML Objective Function
  - Business metric ≠ ML objective (and why this gap matters)
  - Classification objectives:
    * Cross-entropy / log-loss
    * Focal loss (imbalanced)
    * Hinge loss (SVM)
  - Ranking objectives:
    * Pointwise (predict score, rank by score)
    * Pairwise: BPR loss, RankNet
      formula: L = Σ log(1 + exp(s_neg - s_pos))
    * Listwise: LambdaMART, ListNet
  - Regression objectives: MSE, MAE, Huber loss
  - Multi-objective: how to combine (weighted sum, Pareto frontier)
  - Surrogate objectives: when you optimize A because B is
    non-differentiable (e.g., optimize AUC via log-loss)
  - Reward hacking: model optimizes surrogate, breaks real metric

Step 6: Define Evaluation Metrics
  - Offline metrics (measured before deployment):
    * Classification: AUC-ROC, AUC-PR, F1, log-loss
    * Ranking: NDCG@k, MRR, MAP, Precision@k
    * Regression: RMSE, MAE, MAPE
    * Generation: BLEU, ROUGE, human eval
  - Online metrics (measured after deployment):
    * CTR, conversion rate, revenue, retention, NPS
  - The offline-online gap:
    * Why offline AUC doesn't predict online CTR perfectly
    * Distribution shift between training data and live traffic
    * User behavior change post-deployment (feedback loops)
  - Guardrail metrics: metrics you must NOT regress
    * Latency, diversity, fairness, content policy violations
  - How to choose primary vs secondary vs guardrail metrics

Step 7: Define the Training Data Strategy
  - Data sources: logs, labels, user actions
  - Sampling strategy:
    * Random sampling
    * Stratified sampling (preserve class balance)
    * Temporal sampling (time-based train/val/test — MANDATORY for 
      time-sensitive models)
    * Hard negative mining
  - Data collection for cold start
  - Data freshness: how often to retrain?
  - Label collection pipeline

Step 8: Define Failure Modes
  - What could go wrong with this formulation?
  - Feedback loops: model affects data it's trained on
    Example: recommender system → only shows popular items →
    data skewed toward popular → model gets worse for tail items
  - Distribution shift at deployment
  - Adversarial users (gaming the model)
  - Fairness and bias issues
  - Privacy and legal constraints

### 2. Classic Problem Archetypes — Full Walkthroughs

For EACH archetype: business ask → label definition → features →
objective → metrics → failure modes → production considerations

#### 2.1 Feed Ranking (Facebook/Instagram/LinkedIn)
- Business goal: maximize engagement / time spent / meaningful interaction
- Label options comparison:
  * Click: noisy, clickbait
  * Like/Comment/Share: sparser but higher quality
  * "Meaningful interaction" composite score
  * Dwell time: time spent on post
- Formulation: pointwise ranking → score each item → sort
- Features: user history, post content, social graph, recency, context
- Multi-task learning: predict click AND like AND share simultaneously
- Position bias: items shown higher get more clicks regardless of quality
  * How to correct: inverse propensity weighting (IPW)
- Diversity vs relevance tradeoff
- Cold start for new posts

#### 2.2 Search Ranking (Google/Bing)
- Query-document relevance
- Label collection: human raters (explicit) vs clicks (implicit)
- Learning to Rank: pointwise vs pairwise vs listwise comparison
- NDCG@k formula: Σ (2^rel_i - 1) / log₂(i+1)
- Query understanding vs document understanding
- Personalized vs universal ranking

#### 2.3 Recommendation System
- Collaborative filtering formulation
- Content-based filtering formulation
- Two-tower model: user embedding + item embedding, score = dot product
- Label: implicit (click, purchase) vs explicit (rating)
- Loss for implicit feedback: BPR, sampled softmax
- Exploration vs exploitation (bandit formulation)
- Cold start problem: new user, new item

#### 2.4 Fraud / Abuse Detection
- Binary classification with extreme imbalance (0.01% fraud rate)
- Label: confirmed fraud (lagged, biased) vs reported fraud
- Features: behavioral sequences, velocity features, graph features
- Precision vs Recall tradeoff — which matters more and why
- Adversarial: fraudsters adapt to your model
- Cost-sensitive learning: false negative costs >> false positive costs
- Threshold selection based on business cost

#### 2.5 Ads Click-Through Rate (CTR) Prediction
- Binary classification: P(click | user, ad, context)
- Label: click (noisy, positivity bias) vs conversion
- Calibration critical: predicted probability must match true probability
  because ads are priced using these probabilities
- Features: user demographics, ad content, historical CTR, context
- Log-loss as metric (calibration-sensitive)
- Exploration in ads: explore new ads with uncertain CTR

#### 2.6 Churn Prediction
- Binary: will user churn in next 30 days?
- Label definition pitfalls:
  * When is a user "churned"? 7 days inactive? 30 days?
  * Voluntary vs involuntary churn
- Imbalanced classes
- Temporal split is mandatory (no future leakage)
- Business action: intervention at what probability threshold?
- Survival analysis alternative formulation

#### 2.7 ETA / Delivery Time Prediction
- Regression problem
- Label: actual delivery time (delayed collection)
- Asymmetric loss: overestimate (user happy) vs underestimate (user angry)
  → use asymmetric loss function
- Features: distance, traffic, time of day, restaurant prep time
- Uncertainty quantification: give range, not point estimate

#### 2.8 Content Moderation
- Multi-label classification: hate speech AND spam AND nudity
- Label noise: human annotators disagree
- Annotator disagreement → soft labels, model uncertainty
- Cost of errors: false positive (remove legitimate content) vs
  false negative (allow harmful content) — different costs by category
- Two-stage: fast recall model → precise ranking model
- Appeals and feedback loop

#### 2.9 Medical / High-Stakes Classification
- Regulatory constraints
- Explainability requirements
- Extreme cost asymmetry (false negative = missed cancer)
- Class imbalance (rare diseases)
- Distribution shift between hospitals
- Why AUC is wrong metric here — calibration and threshold matter

### 3. Label Definition Deep Dive
- Explicit vs implicit labels — full comparison table
- Positive and negative label definition:
  * Hard negatives vs easy negatives
  * Random negatives vs in-batch negatives
  * Why easy negatives lead to degenerate models
- Dealing with label noise:
  * Label smoothing: ỹ = (1-ε)·y + ε/K
  * Confident learning (Cleanlab)
  * Noise-robust loss functions
- Multi-label vs multi-class distinction
- Ordinal labels: regression vs classification
- Weak supervision (Snorkel approach)
- Semi-supervised learning when labels are expensive
- Label imbalance strategies:
  * Oversampling (SMOTE)
  * Undersampling
  * Class weights
  * Threshold moving: argmax_t such that precision/recall meets SLA

### 4. Metrics Selection Framework
- Precision vs Recall — how to decide which matters:
  * High precision priority: content moderation, spam (avoid annoying
    users with false positives)
  * High recall priority: cancer detection, fraud (missing one is costly)
  * F-beta score: F_β = (1+β²)·(P·R)/((β²·P)+R), β>1 favors recall
- AUC-ROC vs AUC-PR for imbalanced data:
  * ROC misleading when negatives >> positives
  * PR curve better for rare events
- Ranking metrics:
  * Precision@k: fraction of top-k that are relevant
  * Recall@k: fraction of relevant items in top-k
  * NDCG@k: normalized discounted cumulative gain (formula + example)
  * MRR: 1/rank of first relevant item
  * MAP: mean average precision across queries
- Online A/B test metrics vs model metrics
- Statistical significance: t-test, p-value, sample size calculation

### 5. Failure Modes Encyclopedia
- Training-serving skew
- Feedback loops (full explanation with examples)
- Distribution shift:
  * Covariate shift: P(X) changes, P(Y|X) same
  * Label shift: P(Y) changes
  * Concept drift: P(Y|X) changes
  * Detecting each type
- Metric gaming (Goodhart's Law: when a measure becomes a target,
  it ceases to be a good measure)
- Cold start problem
- Popularity bias / filter bubbles
- Fairness and disparate impact
- Adversarial users

### 6. The Interview Framework — How to Answer
"Design an ML system for X" step-by-step response structure:
1. Clarify the problem (2 minutes of questions)
2. Define the ML task formally
3. Define labels + discuss tradeoffs
4. Define features (user/item/context)
5. Choose model family (why)
6. Choose loss function and metrics
7. Discuss training data + splits
8. Discuss failure modes
9. Discuss serving (latency, scale)

Worked example: "Design YouTube's recommendation system"
Full formulation walkthrough using the above framework.

---

## FORMAT REQUIREMENTS:

1. **Beginner-first**: every concept starts with plain English analogy.
   Assume reader can code but has never worked on industry ML problems.

2. **Concrete examples**: every abstract concept illustrated with a real
   industry scenario (Google, Meta, Uber, Netflix, Amazon).

3. **Images**: include markdown image blocks for:
   - ML formulation flowchart
   - Business metric vs ML metric gap diagram
   - Implicit vs explicit label comparison
   - Feedback loop diagram
   - Two-tower model architecture
   - Position bias in ranking diagram
   - NDCG calculation example
   - Precision-Recall tradeoff curves
   - Distribution shift types diagram

4. **Q&A — 60+ questions**:
   - 10 conceptual ("What is the difference between a business metric 
     and an ML metric?")
   - 10 label design ("How would you define labels for a feed ranking
     model at Instagram?")
   - 10 metric selection ("You're building a fraud system. You have 
     99.9% accuracy. Is this good?")
   - 15 full system design ("Formulate the ML problem for Uber's surge
     pricing", "How would you build TikTok's For You Page from scratch",
     "Design a hiring resume screener — what are the risks")
   - 10 tricky/trap ("Can you use future data to create features?",
     "Should you optimize for clicks?", "Is more data always better?")
   - 5 ethics/fairness ("Your model has higher false positive rate for 
     minority groups — what do you do?")
   
   Every answer self-contained, complete, and senior-level quality.

5. **Decision Trees** (not the model — flowcharts) for:
   - Choosing task type (regression/classification/ranking)
   - Choosing primary metric
   - Choosing loss function
   - Deciding label source

6. **Master Cheat Sheet** at end:
   Problem type | Label type | Primary metric | Loss function |
   Key failure mode | Example company use case

7. **Callouts**:
   > 💡 Key Insight:
   > ⚠️ Formulation Trap:
   > 🔗 Interview Signal: (what impresses interviewers)
   > 🏭 Production Reality:
   > ⚖️ Ethics Flag:
   throughout.

8. Length: EXHAUSTIVE. Every archetype fully worked. Every failure mode
   fully explained. Do not summarize. This is a reference document.

Output the entire document as a single .md file.
```

---

**How these two fit together:**

Feature Engineering + Problem Formulation are the two topics that **separate L4 from L5/L6 candidates** at Google/Meta. You can know XGBoost cold but if you can't define a label or spot a leakage trap, you won't pass the ML design round. Study these *after* the algorithm notes but *before* mock interviews.
