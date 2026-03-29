### One-Hot · Label · Target · Frequency · Embedding

---

## 1. Intuition First

### The Core Problem
Machine learning models are mathematical functions. They operate on **numbers**, not words. When your dataset has a column like `["cat", "dog", "cat", "bird"]`, a model can't process it — you need to convert these categories into numbers, but the *way* you do that matters enormously.

**The central tension**: You want to give the model *enough* information about the categories without accidentally injecting false information (like implied ordering or magnitude).

### Simple Analogies

| Encoding | Analogy |
|---|---|
| **Label Encoding** | Giving each student a roll number. Student "Alice" = 1, "Bob" = 2. Fast, but now the model might think Bob > Alice. |
| **One-Hot Encoding** | A multiple-choice exam with one checkbox per option. Only one box is ticked. No ordering implied. |
| **Target Encoding** | Rating each city by its average house price. "Mumbai" gets replaced by ₹80L — its historical outcome. |
| **Frequency Encoding** | Replacing each category with how often it appears. "Mumbai" appears 500 times → replace with 500. |
| **Embedding** | Giving every word a GPS coordinate in meaning-space. Words with similar meanings are close together. |

---

## 2. Formal Explanation

### 2.1 Label Encoding

**Definition**: Assigns a unique integer to each category in an ordinal or nominal feature.

$$\text{LabelEncode}(x_i) = k \quad \text{where } k \in \{0, 1, 2, \ldots, C-1\}$$

for $C$ total categories. The mapping is arbitrary (or alphabetical by default in sklearn).

**Formally**: A bijective mapping $f: \mathcal{C} \rightarrow \mathbb{Z}$ where $\mathcal{C}$ is the set of categories.

---

### 2.2 One-Hot Encoding (OHE)

**Definition**: Transforms a categorical feature with $C$ categories into a $C$-dimensional binary vector where exactly one element is 1.

$$\text{OHE}(x_i = c_k) = \mathbf{e}_k = [0, \ldots, 0, \underbrace{1}_{k\text{-th position}}, 0, \ldots, 0]^T$$

**Dummy Variable Trap**: Using all $C$ columns introduces perfect multicollinearity. In linear models, drop one column (use $C-1$ dummies). This is handled via `drop='first'` in sklearn.

**Formally**: A mapping $f: \mathcal{C} \rightarrow \{0,1\}^C$ subject to $\sum_{j=1}^{C} f(x)_j = 1$.

---

### 2.3 Target Encoding (Mean Encoding)

**Definition**: Each category is replaced by the **mean of the target variable** for all training samples belonging to that category.

$$\text{TargetEncode}(c_k) = \frac{\sum_{i: x_i = c_k} y_i}{\#\{i : x_i = c_k\}}$$

**Smoothed version** (to handle rare categories):

$$\text{TargetEncode}_{\text{smooth}}(c_k) = \frac{n_k \cdot \bar{y}_{c_k} + \lambda \cdot \bar{y}_{\text{global}}}{n_k + \lambda}$$

where $n_k$ is the count of category $c_k$, $\bar{y}_{\text{global}}$ is the global mean, and $\lambda$ is a smoothing hyperparameter.

---

### 2.4 Frequency Encoding (Count Encoding)

**Definition**: Replace each category with its frequency (count) or relative frequency in the training set.

$$\text{FreqEncode}(c_k) = \frac{\#\{i : x_i = c_k\}}{N} \quad \text{(relative)} \quad \text{or} \quad \#\{i : x_i = c_k\} \quad \text{(absolute)}$$

---

### 2.5 Embeddings

**Definition**: A trainable lookup table $E \in \mathbb{R}^{C \times d}$ where each category $c_k$ maps to a dense $d$-dimensional real-valued vector.

$$\text{Embed}(c_k) = E[k] \in \mathbb{R}^d$$

The embedding matrix $E$ is learned via backpropagation. The embedding dimension $d \ll C$ (usually $d \approx \min(50, \lceil C/2 \rceil)$ as a rule of thumb).

**Key property**: Embeddings can capture semantic similarity — if $c_i$ and $c_j$ are "similar" (as judged by the task), then $||E[i] - E[j]||$ will be small after training.

---

## 3. Worked Examples

### Example 1: Label Encoding

**Dataset**: `Color = ["Red", "Blue", "Green", "Blue", "Red"]`

**Step 1**: Identify unique categories → `{Red, Blue, Green}`  
**Step 2**: Assign integers (alphabetically by default) → `Blue=0, Green=1, Red=2`  
**Step 3**: Transform:

| Original | Encoded |
|---|---|
| Red | 2 |
| Blue | 0 |
| Green | 1 |
| Blue | 0 |
| Red | 2 |

**Problem exposed**: The model now sees `Red(2) > Green(1) > Blue(0)`. For a tree-based model, this is fine (it splits on thresholds). For a linear model, this is catastrophic — it implies Red is "twice" Green.

---

### Example 2: One-Hot Encoding

**Same dataset**: `Color = ["Red", "Blue", "Green", "Blue", "Red"]`

**Step 1**: $C = 3$ categories → 3 binary columns  
**Step 2**: Create columns `Color_Red`, `Color_Blue`, `Color_Green`

| Original | Color_Blue | Color_Green | Color_Red |
|---|---|---|---|
| Red | 0 | 0 | 1 |
| Blue | 1 | 0 | 0 |
| Green | 0 | 1 | 0 |
| Blue | 1 | 0 | 0 |
| Red | 0 | 0 | 1 |

**Drop first for linear models** → remove `Color_Blue`. When both `Color_Green=0` and `Color_Red=0`, the model infers Blue.

---

### Example 3: Target Encoding

**Dataset**:

| City | Sale Price (₹L) |
|---|---|
| Mumbai | 90 |
| Delhi | 60 |
| Mumbai | 80 |
| Delhi | 70 |
| Pune | 50 |

**Step 1**: Compute per-category target means:  
- Mumbai → (90 + 80) / 2 = **85**  
- Delhi → (60 + 70) / 2 = **65**  
- Pune → 50 / 1 = **50**

**Step 2**: Replace:

| City | Target Encoded |
|---|---|
| Mumbai | 85 |
| Delhi | 65 |
| Mumbai | 85 |
| Delhi | 65 |
| Pune | 50 |

**Critical issue**: This is computed on training data. If you compute it on the same rows used for training, the model sees the target leaking into features — **data leakage**. Always use cross-fold or out-of-fold encoding.

---

### Example 4: Frequency Encoding

**Dataset**: `Country = [India, USA, India, UK, USA, India]`  
- India appears 3 times → 3/6 = 0.5  
- USA appears 2 times → 2/6 = 0.333  
- UK appears 1 time → 1/6 = 0.167

| Country | Freq Encoded |
|---|---|
| India | 0.500 |
| USA | 0.333 |
| India | 0.500 |
| UK | 0.167 |
| USA | 0.333 |
| India | 0.500 |

---

### Example 5: Embedding (Conceptual)

Consider a vocabulary of 10,000 words. A one-hot vector would be 10,000-dimensional and sparse.

With embeddings ($d=300$):
- "King" → `[0.21, -0.45, 0.87, ..., 0.12]` (300 floats)
- "Queen" → `[0.19, -0.41, 0.85, ..., 0.14]` (close to King!)
- "Apple" → `[-0.88, 0.32, -0.11, ..., 0.55]` (far from King)

Famous property: `Embed("King") - Embed("Man") + Embed("Woman") ≈ Embed("Queen")`

This is learned purely from data — the model discovers structure on its own.

---

## 4. Variants / Types

### One-Hot Variants
| Variant | Description | When to Use |
|---|---|---|
| **Standard OHE** | Full $C$ columns | Most cases |
| **Dummy Coding** | $C-1$ columns (drop first) | Linear models to avoid multicollinearity |
| **Effect Coding** | $C-1$ cols, reference = -1 | ANOVA-style models |
| **Binary Encoding** | Log2(C) bits per category | High cardinality (compromise) |
| **Hashing Trick** | Fixed-size hash buckets | Online learning, very high cardinality |

### Target Encoding Variants
| Variant | Description |
|---|---|
| **Mean encoding** | Replace with mean of target |
| **Median encoding** | More robust to outliers in target |
| **Leave-One-Out (LOO)** | Exclude current row from mean computation |
| **Smoothed/Regularized** | Shrink rare categories toward global mean |
| **Bayesian Target Encoding** | Full posterior estimate; most robust |

### Embedding Variants
| Variant | Description |
|---|---|
| **Word2Vec** | Predict context from word (CBOW/SkipGram) |
| **GloVe** | Global co-occurrence statistics |
| **FastText** | Subword-level (handles OOV words) |
| **Entity Embeddings** | Tabular data (Guo & Berkhahn 2016) |
| **Fine-tuned embeddings** | Transfer from pretrained model + task-specific training |

---

## 5. Tradeoffs

### Label Encoding

| ✅ Pros | ❌ Cons |
|---|---|
| Memory efficient (1 column) | Implies false ordinal relationship |
| Works with tree models natively | Misleads linear/distance-based models |
| Fast to compute | Arbitrary integer assignment |
| No dimensionality explosion | |

**Fails when**: Used with linear regression, SVM, KNN on nominal data. The numeric ordering creates spurious gradients.

---

### One-Hot Encoding

| ✅ Pros | ❌ Cons |
|---|---|
| No false ordering | Dimensionality explosion (high cardinality) |
| Works with all model types | Sparse matrices, memory-heavy |
| Interpretable | Multicollinearity (dummy trap) |
| Standard for nominal features | Useless for unseen categories at inference |

**Fails when**: Cardinality > 100–1000. A `user_id` column with 1M unique values → 1M new binary columns. Computationally catastrophic.

---

### Target Encoding

| ✅ Pros | ❌ Cons |
|---|---|
| Handles high cardinality gracefully | **Severe data leakage risk** |
| Compact (1 column output) | Rare categories have noisy estimates |
| Captures target signal directly | Requires careful cross-fold implementation |
| Works well with tree models | Over-fits target distribution |

**Fails when**: Implemented naively on training data. Also fails with severe class imbalance or when the category-target relationship is unstable across time.

---

### Frequency Encoding

| ✅ Pros | ❌ Cons |
|---|---|
| No leakage risk | Loses category identity (two cats with same count = same value) |
| Handles high cardinality | Frequency may not correlate with target |
| No OOV problem (assign 0 or rare-bin) | Sensitive to dataset size changes |
| Simple and fast | |

**Fails when**: Two different categories have the same frequency — model can't distinguish them.

---

### Embeddings

| ✅ Pros | ❌ Cons |
|---|---|
| Captures semantic similarity | Requires large data to train well |
| Compact dense representation | Black-box — not interpretable |
| Handles high cardinality | Adds model complexity |
| Can transfer knowledge (pretrained) | Need to handle OOV at inference |
| Learns task-specific structure | |

**Fails when**: Data is scarce (embeddings underfit). Also fails when categories have no natural semantic clustering.

---

## 6. When to Use What

### Decision Framework

```
Is the feature ORDINAL (natural order exists)?
│
├── YES → Label Encoding (respect the order)
│         e.g., Low=0, Medium=1, High=2
│
└── NO → Is cardinality LOW (< 10–15 unique values)?
          │
          ├── YES → One-Hot Encoding (safe, interpretable)
          │
          └── NO → Is cardinality MEDIUM (15–1000)?
                    │
                    ├── Is it a regression/binary classification task?
                    │   └── YES → Target Encoding (with cross-fold)
                    │
                    └── Is data size / frequency distribution meaningful?
                        └── YES → Frequency Encoding (safe baseline)
                            NO  → Binary Encoding / Hashing Trick

For DEEP LEARNING or NLP tasks → Embeddings always
For HIGH CARDINALITY (>1000) in DL → Learned Embeddings
```

### Real-World Scenarios

| Scenario | Recommended Encoding | Reason |
|---|---|---|
| `Gender` (M/F) in linear model | OHE | Low cardinality, nominal |
| `Education level` (HS/BS/MS/PhD) | Label Encoding | Ordinal |
| `City` (500 unique cities) in XGBoost | Target Encoding | High cardinality, tree model |
| `UserID` (1M users) in a DL model | Embedding | Ultra-high cardinality |
| `Product category` (200 items) in GBM | Frequency or Target Encoding | Balanced approach |
| `Word tokens` in NLP | Word Embeddings | Semantic relationships matter |
| Online learning with streaming data | Hashing Trick | Fixed memory, no vocabulary needed |

---

## 7. Edge Cases & Pitfalls

### Pitfall 1: The Dummy Variable Trap
Using all $C$ OHE columns in linear regression causes **perfect multicollinearity** — the sum of all dummy columns equals 1 (a constant). This makes the design matrix singular and the model unidentifiable.  
**Fix**: Always drop one column with `drop='first'` or use regularization.

### Pitfall 2: Target Encoding Leakage
Computing target encoding using all training data including the current row allows the model to "see" the target value through the feature.  
**Fix**: Use **K-fold cross-validation** — compute the encoding for fold $k$ using all folds *except* $k$.

### Pitfall 3: Unseen Categories at Inference (OOV)
If a new category appears at test time that wasn't in training, most encoders fail.  
- OHE → all zeros (valid, but the sample looks like no category)  
- Label Encoding → throws an error or maps to -1  
- Target Encoding → no mean available  
**Fix**: Always handle OOV explicitly — assign global mean for target encoding, a separate "Unknown" bucket for OHE, or 0 for frequency encoding.

### Pitfall 4: Label Encoding for Nominal Features in Linear Models
Students often use `LabelEncoder` from sklearn for all categoricals. It's fine for trees but **fatal** for linear models.

### Pitfall 5: OHE on High-Cardinality Features
A `zip_code` column with 40,000 unique values → 40,000 binary columns. Most are near-zero, the matrix is impossibly sparse, and training time explodes.

### Pitfall 6: Treating Embedding Size as a Hyperparameter to Ignore
Using too small an embedding dimension → underfitting (can't represent category diversity). Too large → overfitting and slow training. Rule of thumb: $d \approx \min(50, \lceil C/2 \rceil)$.

### Pitfall 7: Target Encoding with Class Imbalance
In heavily imbalanced datasets, the global mean is close to 0 (or near 1). Smoothing toward a biased global mean introduces systematic bias for rare classes.

### Interview Trap: "Just use Label Encoding, it's simpler"
This is a red flag answer. Always ask: *Is the feature ordinal or nominal? What model will consume it?* Nominal + linear model = OHE or target encoding, not label encoding.

---

## 8. High Cardinality & Scale Considerations

### The Cardinality Problem
| Cardinality | # Unique Values | OHE Columns | Feasibility |
|---|---|---|---|
| Low | < 15 | < 15 | ✅ Easy |
| Medium | 15–100 | 15–100 | ⚠️ Manageable |
| High | 100–10,000 | 100–10,000 | ❌ OHE breaks down |
| Ultra-High | > 10,000 | > 10,000 | ❌ Must use embedding/target/freq |

### Memory Impact of OHE
For $N = 1,000,000$ rows and $C = 10,000$ categories:
- Dense matrix: $10^6 \times 10^4 = 10^{10}$ cells × 1 byte = **~10 GB**  
- Sparse matrix (scipy CSR): Only store non-zeros → $10^6$ non-zeros × ~12 bytes = **~12 MB** ✅

**Always use sparse matrices for OHE at scale.**

### Hashing Trick
Map categories to a fixed-size space using a hash function:

$$h(c_k) = \text{hash}(c_k) \bmod B$$

where $B$ is the number of buckets. Collisions occur but are manageable. Used in Vowpal Wabbit, sklearn's `HashingVectorizer`.  
- **Pro**: O(1) memory, no vocabulary needed  
- **Con**: Hash collisions reduce model expressiveness

### Target Encoding at Scale
For billion-row datasets, compute per-category statistics with a single MapReduce or groupby pass. Smoothing is essential — with $10^8$ rows, rare categories (e.g., appearing once) will have wildly unstable target means.

### Embeddings at Scale
- Embedding table for 1M users × 64 dims = 64M floats × 4 bytes = **256 MB** — totally feasible
- Embedding table for 100M users × 128 dims = **51 GB** — needs distributed training (model parallelism)
- Facebook/Google use embedding tables with billions of entries, sharded across GPUs

---

## 9. FAANG-Level Conceptual Questions

---

**Q1. Why can't you use label encoding for nominal features in linear models, but it works fine for decision trees?**

**Answer**: Linear models apply weights directly to feature values: $y = w \cdot x$. If `Red=2, Blue=0, Green=1`, then a unit increase in the encoded color means something — the model interprets Red as "twice" Green. This is nonsensical for nominal data.

Decision trees, however, only evaluate *threshold conditions* like `x < 1.5`. They don't assign magnitude to values — they just split. So label encoding doesn't inject false ordinal information into the model's inductive bias.

---

**Q2. How do you prevent data leakage in target encoding?**

**Answer**: Use **K-fold cross-validation encoding**:
1. Split training data into K folds
2. For each fold $k$, compute the target mean using rows from all *other* folds
3. Assign these means to the rows in fold $k$
4. For test data, use the global mean computed from the entire training set (with smoothing)

This ensures no row's target value contributed to its own encoding. LOO (Leave-One-Out) encoding is the extreme case of this (K = N).

---

**Q3. What is the dummy variable trap, and when does it actually matter?**

**Answer**: With $C$ OHE columns, the sum of all dummy variables equals 1 for every row (exactly one is active). This is a perfect linear combination — the columns are linearly dependent. In linear regression, this makes the design matrix $X^TX$ singular and non-invertible, so $(X^TX)^{-1}$ doesn't exist and least-squares has no unique solution.

**It only matters for linear models** (regression, logistic regression, linear SVM). Tree-based models are immune. Neural networks with regularization are practically immune (the redundancy gets regularized away).

---

**Q4. When would you prefer frequency encoding over target encoding?**

**Answer**: Choose frequency encoding when:
1. The task is **unsupervised** (no target available)
2. You're building a **preprocessing pipeline** that must be computed without the target (e.g., in a feature store that's reused across multiple tasks)
3. Target encoding has **leakage risk** that's hard to implement correctly in your infrastructure
4. You suspect the **frequency itself is a useful signal** (e.g., rare items behave differently from common ones)
5. **Speed matters** — frequency encoding is a single groupby operation; target encoding requires fold-aware computation

---

**Q5. How do word embeddings like Word2Vec actually learn meaningful vector representations?**

**Answer**: Word2Vec (Skip-Gram variant) trains a neural network to **predict surrounding context words** given a center word. The network's hidden layer weights become the embeddings.

Words that appear in similar contexts get similar gradient updates, causing their vectors to converge in embedding space. "King" and "Queen" both appear near words like "throne", "crown", "reign" → their vectors become close. "Apple" and "Orange" both appear near "fruit", "juice" → they cluster together, but far from royalty words.

The famous arithmetic `King - Man + Woman = Queen` arises because the difference between male-female word pairs consistently aligns in embedding space — a linear subspace encodes gender.

---

**Q6. How would you handle a new, unseen category at inference time for each encoding type?**

**Answer**:
- **Label Encoding**: Map to a reserved "Unknown" integer, or raise an error and impute with the mode
- **OHE**: All dummy columns = 0. Valid but the row looks like "no category" — consider adding an explicit "Unknown" column during training
- **Target Encoding**: Replace with the global target mean (the smoothed estimate for $n_k = 0$). With smoothing formula: as $n_k \to 0$, encode → $\bar{y}_{\text{global}}$
- **Frequency Encoding**: Assign 0 or a small epsilon (below the minimum training frequency)
- **Embeddings**: Add a special `<UNK>` token during training. Its embedding is learned to represent the "average unknown entity"

---

**Q7. Why do embeddings work better than OHE for high-cardinality features in deep learning?**

**Answer**: Four reasons:
1. **Dimensionality**: OHE creates $C$-dimensional sparse vectors. Embeddings create $d$-dimensional dense vectors with $d \ll C$. The model has far fewer parameters and trains faster.
2. **Generalization**: Two OHE vectors for different categories have zero dot product — they're completely orthogonal and the model can't generalize between them. Embedding vectors for similar categories will be close, enabling generalization.
3. **Gradient flow**: Sparse OHE inputs mean most neurons in the first layer receive zero gradient during most updates. Dense embeddings ensure every parameter gets updated every step.
4. **Learned structure**: OHE is a fixed, predefined encoding. Embeddings are **learned from data** — the model discovers which categories are similar based on the task objective.

---

**Q8. A feature has 50,000 unique categories. Walk through the pros and cons of your encoding choices.**

**Answer**:
- **OHE**: 50,000 columns → sparse matrix is manageable technically, but the model now has 50,000 weights just for this feature. High risk of overfitting, very slow training. **Not recommended.**
- **Label Encoding**: For trees (LightGBM, CatBoost) with native categorical support, this can work. Otherwise, numeric order is meaningless. **Only for trees with native support.**
- **Target Encoding + smoothing**: Reduces to 1 column, captures signal well, but requires careful OOF implementation. **Good choice for GBM models.**
- **Frequency Encoding**: 1 column, no leakage, fast. Doesn't capture target signal. **Safe baseline.**
- **Hashing**: Fixed $B$ buckets (e.g., $B=1000$). Lossy but memory-bounded. **Good for online/streaming.**
- **Embeddings (dim=64)**: Best for deep learning. Each category gets a 64-dim vector. 50K × 64 = 3.2M parameters — totally manageable. **Best choice for DL.**
- **CatBoost native**: CatBoost uses an online target encoding with permutations internally. **Best for GBM when leakage is a concern.**

---

**Q9. What's the difference between entity embeddings for tabular data vs. word embeddings for NLP?**

**Answer**:
Both are lookup tables trained via backpropagation, but they differ in:

| Aspect | Word Embeddings | Entity Embeddings (tabular) |
|---|---|---|
| **Pretrained?** | Often (Word2Vec, GloVe, BERT) | Usually trained from scratch on task |
| **Transfer?** | High — language is universal | Low — "City" embeddings don't transfer across datasets |
| **Signal** | Co-occurrence in text sequences | Target signal from supervised task |
| **Semantic structure** | Rich linguistic structure | Depends entirely on target variable |
| **OOV handling** | Subword models (FastText, BPE) | `<UNK>` token, or zero vector |

Entity embeddings (Guo & Berkhahn 2016 for tabular neural nets) work well when categories have rich interactions with the target — e.g., `store_id` in retail demand forecasting.

---

**Q10. You're building a recommendation system with 10M users and 1M items. How do you encode these IDs?**

**Answer**: This is the canonical case for **learned embeddings**.

- OHE: 10M + 1M = 11M binary columns. Completely infeasible.
- Target encoding: Leakage issues, and you'd need per-user mean ratings — essentially reinventing collaborative filtering
- Embeddings: User embedding matrix $U \in \mathbb{R}^{10M \times d}$, item embedding matrix $I \in \mathbb{R}^{1M \times d}$, with $d = 64$ or $128$
  - $U$: 10M × 64 × 4 bytes = **2.56 GB**
  - $I$: 1M × 128 × 4 bytes = **0.51 GB**
  - Predicted score: $\hat{r}_{ui} = U_u \cdot I_i$ (dot product)

This is **Matrix Factorization / Neural Collaborative Filtering** in its core form. The embeddings are the latent factors. At this scale, embeddings are sharded across multiple GPUs or parameter servers (as in Google's PinSage, Facebook's DLRM).

---

## 10. Summary Cheat Sheet

```
╔══════════════════════════════════════════════════════════════════════╗
║              ENCODING QUICK REFERENCE CARD                          ║
╠══════════════╦══════════════╦═══════════════╦══════════════════════╗
║ Method       ║ Output Dims  ║ Key Risk       ║ Best For             ║
╠══════════════╬══════════════╬═══════════════╬══════════════════════╣
║ Label        ║ 1            ║ False ordering ║ Ordinal features     ║
║ OHE          ║ C (# cats)   ║ Dim explosion  ║ Low cardinality      ║
║ Target       ║ 1            ║ Data leakage   ║ High card + GBM      ║
║ Frequency    ║ 1            ║ Collision      ║ High card + safe     ║
║ Embedding    ║ d (chosen)   ║ Needs data     ║ DL + ultra-high card ║
╚══════════════╩══════════════╩═══════════════╩══════════════════════╝

KEY RULES:
  ✦ Ordinal feature → Label Encoding
  ✦ Nominal + low cardinality (<15) → OHE
  ✦ Nominal + high cardinality + tree model → Target Encoding (OOF)
  ✦ Nominal + high cardinality + no target available → Frequency Encoding
  ✦ Any feature in deep learning → Embedding
  ✦ Ultra-high cardinality (1M+) → Embedding always

LEAKAGE RULE:
  ✦ Target Encoding MUST use out-of-fold computation on training data
  ✦ Test set uses global mean (smoothed toward global mean for OOV)

OOV RULE:
  ✦ OHE → all zeros (+ add explicit Unknown column at training time)
  ✦ Target Encoding → global mean
  ✦ Frequency Encoding → 0 or minimum observed frequency
  ✦ Embeddings → <UNK> token vector

DUMMY TRAP:
  ✦ Linear models: use C-1 OHE columns (drop='first')
  ✦ Tree/DL models: all C columns OK

CARDINALITY THRESHOLDS (rules of thumb):
  < 15     → OHE always safe
  15–50    → OHE or Target Encoding
  50–1000  → Target or Frequency Encoding
  > 1000   → Embeddings or Hashing

EMBEDDING DIMENSION:
  d ≈ min(50, ceil(C/2)) — classic heuristic
  d = 4, 8, 16, 32, 64, 128 — powers of 2 common in practice

FORMULAS:
  Target:    mean(y | x = c_k)
  Smoothed:  (n_k * ȳ_k + λ * ȳ_global) / (n_k + λ)
  Frequency: count(x = c_k) / N
  Embedding: E[k] where E ∈ ℝ^(C×d), learned via backprop
```
