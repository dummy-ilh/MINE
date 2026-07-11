# Feature Engineering — Complete Interview & Reference Notes

---

## Chapter 1: Foundations

### What is Feature Engineering?
The process of transforming raw data into inputs (features) that better represent the underlying problem to a model — through creation, transformation, selection, or extraction of variables.

### Why It Matters More Than Model Choice (often)
- Models learn patterns *within* the feature space you give them. If the signal isn't represented in the features, no model — however powerful — can recover it.
- A linear model with great features frequently beats a deep model with raw, unprocessed features.
- **What happens if you skip it:** models fit noise, underperform, or need far more data/capacity to approximate what a good feature would have handed them directly. Example: a model without a "day of week" feature has to *learn* weekly seasonality from raw timestamps — wasteful and often it never fully gets there.

### Where It Fits in the ML Pipeline
```
Raw Data → EDA → Feature Engineering → Train/Test Split (or split FIRST, see Ch.8)
        → Fit transformations on train → Apply to val/test → Model training → Evaluation
```
**Industry standard:** split first, then fit every transformer (scaler, encoder, imputer) on train only. Doing it in the wrong order is the single most common source of silent leakage in real pipelines (see Chapter 8).

### Missingness Mechanisms (sets up Chapter 3)
| Type | Meaning | Example |
|---|---|---|
| **MCAR** (Missing Completely at Random) | Missingness unrelated to any data, observed or not | Sensor randomly drops packets |
| **MAR** (Missing at Random) | Missingness depends on *observed* data | Income missing more often for younger respondents |
| **MNAR** (Missing Not at Random) | Missingness depends on the *unobserved* value itself | High earners refuse to disclose income |

**Why this distinction needs to exist:** the correct imputation strategy depends entirely on *why* data is missing. Treating MNAR as MCAR (e.g., blindly mean-imputing) systematically biases your feature — you're not filling in noise, you're erasing a pattern.

### Google-style Interview Questions — Chapter 1

**1. Why can feature engineering matter more than model selection? Give a concrete example.**
> A model can only find patterns that are *representable* in the feature space you give it. Example: predicting bike-share demand from a raw timestamp. A linear model on the raw Unix timestamp will completely miss weekday/weekend or rush-hour effects because there's no linear relationship between "seconds since 1970" and demand. Extract `hour_of_day`, `is_weekend`, `is_rush_hour` and the *same* linear model suddenly captures most of the signal. A far more powerful model (deep net) trained on the raw timestamp would need enormous amounts of data to approximately rediscover these patterns on its own — and might never do so cleanly. Good features effectively do the model's work for it.

**2. Explain MCAR vs MAR vs MNAR with real-world examples. How does each affect your imputation choice?**
> - MCAR: missingness is pure noise (e.g., a sensor randomly drops readings). Any imputation method (mean, median) is statistically safe because missingness carries no information.
> - MAR: missingness depends on *other observed* columns (e.g., survey non-response on "income" is higher among younger respondents — but "age" is observed). Here, imputing using a model that conditions on the observed columns (KNN, MICE, regression imputation) is far better than a blind global mean, because you can recover the conditional distribution.
> - MNAR: missingness depends on the *unobserved value itself* (high earners refuse to report income *because* it's high). No amount of conditioning on other observed columns fully fixes this — you need domain assumptions, sensitivity analysis, or an explicit missingness indicator, because the data is not "missing at random" in a way any imputation can cleanly reverse.

**3. Where exactly does feature engineering sit relative to train/test splitting, and why does the order matter?**
> Any transformation that has "memory" of the data distribution (mean, variance, category frequency, min/max, PCA components) must be split first, then `fit()` exclusively on train, then `transform()`-applied to validation/test. If you compute those statistics on the full dataset before splitting, information about the test set's distribution leaks into training — the model's evaluation becomes optimistic and unreliable, even though no label ever touched the test rows directly.

**4. Can a sufficiently powerful model (e.g., deep neural net) make feature engineering unnecessary? Defend your answer.**
> Partially, and only in specific regimes. Deep nets *can* learn feature representations automatically — this is exactly why CNNs replaced hand-crafted image features (SIFT/HOG) and why transformers replaced hand-crafted NLP features. But this requires (a) very large datasets, (b) a lot of compute, and (c) structured, high-signal-density raw input (pixels, tokens) where local patterns generalize. For tabular data (the typical "feature engineering" domain), datasets are usually small-to-medium, features are heterogeneous and not spatially/sequentially structured, and hand-engineered features (ratios, aggregates, domain knowledge) still consistently beat raw-feature deep learning in practice (this is empirically well-documented — gradient-boosted trees on engineered features still win most tabular Kaggle competitions). So: no, not in general, especially not for tabular data.

---

## Chapter 2: Numerical Feature Transformation

### 2.1 Scaling

| Method | Formula | Robust to Outliers? | Use When |
|---|---|---|---|
| **Standardization (Z-score)** | (x − μ) / σ | No | Data ~ Gaussian; algorithms assuming Gaussian input (linear/logistic regression, SVM, PCA, KNN, NN) |
| **Min-Max** | (x − min) / (max − min) | No | Need bounded range [0,1] (e.g., image pixels, NN inputs with sigmoid/tanh) |
| **Robust Scaling** | (x − median) / IQR | Yes | Data has outliers |
| **MaxAbsScaler** | x / max(\|x\|) | No | Sparse data (preserves sparsity, doesn't shift zero) |

**Why scaling needs to exist:**
Distance-based and gradient-based algorithms treat feature magnitude as importance. If "salary" ranges 0–500,000 and "age" ranges 0–100, Euclidean distance (KNN, K-Means) and gradient descent (linear models, NN) will be dominated by salary purely due to scale — not because it's more predictive.

**What happens if you don't scale:**
- KNN/K-Means: distance metric dominated by large-magnitude features → wrong neighbors/clusters.
- Gradient descent: ill-conditioned loss surface → slow/unstable convergence, needs tiny learning rate.
- Regularization (L1/L2): penalizes large-scale coefficients unfairly — coefficients on unscaled features get artificially shrunk/inflated relative to their real importance.

**When NOT to scale:**
- Tree-based models (Decision Trees, Random Forest, XGBoost, LightGBM) — splits are based on ordering/thresholds, not magnitude. Scaling is a no-op for them.
- Naive Bayes — feature distributions are handled probabilistically, not distance-based.

**Industry standard:** `StandardScaler` is the default in most tabular pipelines unless you have heavy outliers (then `RobustScaler`) or need [0,1] bounds (then `MinMaxScaler`). Always fit on train, `transform` on val/test — never `fit_transform` on the full dataset.

### 2.2 Distribution Fixes

- **Log transform** (`log(x+1)` to handle zeros): compresses right-skewed data (income, counts, prices). Makes distributions closer to Gaussian, stabilizes variance.
- **Box-Cox:** generalizes log transform via a learned parameter λ; requires strictly positive data.
- **Yeo-Johnson:** like Box-Cox but works with zero/negative values too.
- **Quantile Transform:** maps data to a uniform or Gaussian distribution using the empirical CDF — very aggressive, can distort relationships between values but is powerful against extreme skew/outliers.

**Why this needs to exist:** many models (linear regression, LDA) assume features are roughly normal or at least not extremely skewed; skew inflates the influence of a few large values (e.g., a handful of billionaires wrecking an "income" feature's mean and variance).

**What happens if you don't fix skew:** heavy-tailed features dominate loss functions built on squared error (MSE); a few outliers can swing model coefficients disproportionately.

### 2.3 Outlier Handling
- **Winsorizing/Clipping:** cap values at a percentile (e.g., 1st/99th) instead of removing them — preserves sample size while limiting extreme influence.
- **IQR method:** flag points outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR].
- **Z-score method:** flag points beyond ±3 standard deviations (assumes near-normal distribution).

**When/when not:** Don't blindly remove outliers — some are the signal (e.g., fraud detection literally targets outliers). Clip/cap when outliers are believed to be noise/errors; keep them, or engineer an "is_outlier" flag, when they might be meaningful.

### Google-style Interview Questions — Chapter 2

**1. Why do distance-based algorithms require scaling but tree-based models don't?**
> Distance metrics (Euclidean, Manhattan) sum differences across dimensions — a feature with a larger numeric range mechanically dominates the distance calculation regardless of its actual predictive relevance. Gradient-based models (linear/logistic regression, SVM, neural nets) also implicitly weight by scale because gradients are proportional to input magnitude, making optimization ill-conditioned. Tree-based models instead ask "is x > threshold?" at each split — the *rank order* of values within a feature determines the split, and rank order is invariant to any monotonic rescaling. So multiplying a feature by 1000 changes nothing about which splits a tree chooses.

**2. When would you choose RobustScaler over StandardScaler? Walk through the math.**
> StandardScaler uses `(x − mean) / std`; both the mean and standard deviation are heavily influenced by outliers (a single extreme value can inflate std and shift the mean). RobustScaler uses `(x − median) / IQR`, and both the median and the interquartile range (Q3−Q1) are robust statistics — resistant to a small number of extreme values since they depend only on the middle of the distribution. If a feature like "transaction amount" has legitimate fraud outliers at $1M against a typical range of $10–$500, StandardScaler will compress almost all normal transactions into a tiny sliver near zero, while RobustScaler keeps the bulk of the distribution well-spread.

**3. You have a heavily right-skewed revenue feature with zeros. Which transform do you apply and why?**
> `log1p` (i.e., `log(x + 1)`), because plain `log(x)` is undefined at zero. Log-transform compresses the long right tail (large revenue values) proportionally more than small ones, pulling the distribution toward something closer to Gaussian/symmetric, which helps linear-model assumptions and reduces the outsized influence of a few very large values on squared-error loss. If some rows can be negative (e.g., net revenue with refunds), Yeo-Johnson is used instead since Box-Cox/log require strictly positive input.

**4. What's the practical difference between clipping outliers and removing them? When would each hurt the model?**
> Clipping (winsorizing) caps extreme values at a percentile threshold but keeps the row — you lose the *magnitude* of the extreme but preserve sample size and the fact that "this row was extreme." Removing drops the row entirely — you lose sample size and, if the outliers aren't random (often they aren't), you introduce selection bias. Clipping hurts when the tail values genuinely matter for the task (e.g., fraud amounts — clipping erases exactly the signal fraud detection needs). Removing hurts when outliers are frequent enough that dropping them meaningfully shrinks or biases the training set, or when "being an outlier" itself correlates with the target.

**5. Why can applying MinMaxScaler before removing outliers be a mistake?**
> MinMaxScaler uses `(x − min) / (max − min)` — a single extreme outlier defines `max` (or `min`), compressing the entire rest of the distribution into a narrow sub-range near 0 (or 1). E.g., if 99% of values are between 10–100 but one outlier is 100,000, MinMaxScaler squashes the useful 10–100 range into roughly [0.0001, 0.001] — destroying almost all resolution among the values that actually matter. Outliers should be handled (clipped/removed/flagged) *before* Min-Max scaling, or RobustScaler should be used instead.

---

## Chapter 3: Missing Data Strategies

### Step 1: Diagnose the Mechanism (before picking a method)
- Plot missingness against other variables. If missing rate correlates with an observed feature → MAR. If it correlates with the value itself (unobservable directly, but domain logic implies it) → MNAR. If no pattern → MCAR.
- **Why this order matters:** picking an imputation method before understanding *why* data is missing is like prescribing medicine before diagnosis — mean imputation is "safe" only under MCAR; under MAR/MNAR it introduces bias.

### Strategies

| Method | How | Best For | Risk |
|---|---|---|---|
| Mean/Median/Mode | Fill with a central statistic | MCAR, quick baseline | Shrinks variance, distorts correlations |
| Forward/Backward Fill | Carry last/next known value | Time series | Assumes local stability; wrong for volatile series |
| KNN Imputer | Use k nearest neighbors' values | MAR, moderate-size datasets | Expensive at scale; sensitive to feature scaling |
| MICE (Multiple Imputation by Chained Equations) | Iteratively models each feature with missing values as a function of others, repeated across multiple imputed datasets | MAR, when relationships between features matter | Computationally heavy; needs careful convergence checks |
| Missingness Indicator Flag | Add binary column `is_missing` alongside imputed value | MNAR or when missingness itself is predictive | Adds dimensionality |

**Why "missingness as signal" needs to exist:** sometimes the *fact* that a value is missing is more predictive than any value you could impute. E.g., in credit scoring, "income not disclosed" is itself a risk signal. If you silently impute the mean, you erase that signal entirely.

**What happens if you don't add a flag:** you assume missingness is uninformative (MCAR), which — if false — quietly discards predictive information and can systematically bias the model toward the imputed value's neighborhood.

### Industry Standard
- Simple pipelines: median (numeric) / mode (categorical) imputation + missingness flag, wrapped in `sklearn.impute.SimpleImputer`.
- Higher-stakes modeling (finance, healthcare): MICE (`IterativeImputer` in sklearn) or model-based imputation, always paired with sensitivity analysis (does the model's outcome change much across imputations?).
- **Always fit imputers on train only** — same leakage risk as scalers/encoders.

### When / When Not
- **Use mean/median:** fast baseline, MCAR, low missingness (<5%).
- **Avoid mean/median:** high missingness (>20-30%) or MNAR — you're fabricating a large chunk of the dataset with a single number, killing variance.
- **Use MICE/KNN:** when features are correlated and you can afford the compute; when correlation structure matters for downstream inference.
- **Never:** impute after train/test split boundary is violated (fit imputer on full data) — classic leakage.

### Google-style Interview Questions — Chapter 3

**1. Why does the choice of imputation method depend on whether data is MCAR, MAR, or MNAR?**
> Imputation is only statistically unbiased when the method's assumption matches the true missingness mechanism. Mean/median imputation implicitly assumes MCAR (missingness carries zero information — any reasonable fill value is fine). Under MAR, the missing value is predictable from *other observed columns*, so a conditional method (regression/KNN/MICE) recovers more signal than a flat statistic. Under MNAR, missingness depends on the unobserved value itself, so no method conditioning only on observed data can fully de-bias it — you need domain knowledge or explicit missingness signaling instead of pretending the value is recoverable.

**2. Explain how mean imputation distorts feature correlations. Give a numeric intuition.**
> Say `income` correlates strongly with `age` (older → higher income), and 30% of `income` is missing. If you fill missing income with the *overall* mean, those 30% of rows now have zero relationship to their actual age — you've injected a block of constant values that dilutes the true age-income correlation, pulling the sample correlation coefficient toward zero even though the true underlying relationship is unchanged. Additionally, imputing with a single constant collapses the *variance* of the feature — the imputed rows have zero within-group variance, artificially shrinking the standard deviation and biasing anything downstream that relies on variance (e.g., feature importance, t-tests, regression standard errors).

**3. When would you prefer adding a missingness indicator over imputing a specific value?**
> When missingness itself correlates with the target (i.e., not MCAR) — the *fact* that a value wasn't recorded is informative on its own, independent of what the "true" value might have been. Example: in a loan application, "employer field left blank" might correlate with self-employment or fraud risk regardless of what income value you'd guess. Adding `is_missing_employer` as a separate binary feature lets the model use that signal directly, in addition to (not instead of) whatever value you impute for the missing field itself.

**4. How does MICE work at a high level, and why is it preferred over single imputation for MAR data?**
> MICE (Multiple Imputation by Chained Equations) treats each feature with missing values as the target of its own regression model, conditioned on all other features (imputed with a placeholder initially). It cycles through each feature, refitting the regression and updating the imputed values, over several iterations until estimates stabilize — and typically repeats this whole process multiple times with different random seeds to produce several completed datasets, whose results are pooled. This is preferred over single-value imputation because (a) it uses the full multivariate relationship between features rather than a single feature's marginal statistic, correctly handling MAR, and (b) generating *multiple* imputed datasets and pooling results propagates the uncertainty of imputation into your final estimates, instead of pretending a single guessed value is ground truth.

**5. You have a column that's 60% missing. Do you impute it, drop it, or engineer a flag? Justify with reasoning about information loss vs. noise.**
> At 60% missing, imputing the majority of a column with a single statistic effectively fabricates most of the feature's values — the risk of injecting noise (or worse, bias if MAR/MNAR) outweighs the modest signal the remaining 40% carries. The generally sound approach: (1) check whether missingness itself is predictive (compare target rates for missing vs. non-missing) — if yes, keep a binary `is_missing` flag as a feature even if you drop or crudely impute the value itself; (2) if the underlying value has no recoverable relationship to other features and missingness isn't informative, drop the column entirely rather than dilute the model with mostly-fabricated data; (3) if the column is domain-critical and correlated features exist, a model-based imputation (MICE/KNN) evaluated via cross-validation impact is worth the cost before committing either way.

---

## Chapter 4: Categorical Encoding

### 4.1 One-Hot Encoding (OHE)
Creates a binary column per category.
- **Why it exists:** most ML algorithms operate on numbers, and naively assigning integers (1, 2, 3...) to unordered categories (e.g., "Red", "Blue", "Green") implies a false ordinal relationship — the model would think Green > Blue > Red, or that Blue is "between" Red and Green.
- **What happens if you skip proper encoding and just label-encode nominal data:** linear models will assign meaningless weight gradients based on the arbitrary integer ordering; tree models are less harmed since they can split arbitrarily, but still often perform worse than with OHE for low-cardinality nominal features.
- **The curse of dimensionality problem:** a categorical feature with 50,000 unique values (e.g., zip code, user ID) turns into 50,000 sparse columns — model size explodes, most splits become uninformative, and you risk overfitting to rare categories.

### 4.2 Label / Ordinal Encoding
Maps categories to integers.
- **Use when** categories have a true order (e.g., "Low" < "Medium" < "High", education level).
- **Don't use** for nominal (unordered) categories — introduces false ordinal signal (see above).
- Tree-based models tolerate label encoding on nominal data reasonably well since splits don't assume linear ordering — but it's still not "correct," just "less harmful."

### 4.3 Frequency / Count Encoding
Replace category with its frequency (or count) in the training data.
- **Why:** compresses high-cardinality categories into a single informative numeric signal (popularity) without exploding dimensionality.
- **Risk:** two different categories with the same frequency become indistinguishable to the model.

### 4.4 Target (Mean) Encoding
Replace category with the mean of the target variable for that category.
- **Why it's powerful:** directly encodes the relationship between category and target — often the single most predictive encoding for high-cardinality categoricals (e.g., zip code → mean house price).
- **Why it's dangerous (leakage):** if you compute the mean target using rows that include the row you're about to predict, you've leaked the label into the feature. The model essentially "sees the answer."

**Target Encoding Pitfalls (dedicated focus):**
1. **Direct leakage:** computing the encoding on the full dataset (train+test) or without excluding the current row — inflates train performance, collapses on real test data.
2. **Overfitting on rare categories:** a category with 2 samples where both happen to be "positive" gets encoded as 1.0 — pure noise treated as strong signal.
3. **Fix — Smoothing/Regularization:** blend the category's mean with the global mean, weighted by category frequency:
   ```
   encoded = (n_category * category_mean + m * global_mean) / (n_category + m)
   ```
   where `m` is a smoothing strength hyperparameter. Small categories shrink toward the global mean.
4. **Fix — K-Fold Target Encoding:** split train into K folds; for each fold, compute the encoding using only the *other* K-1 folds. Prevents a row's own label from influencing its own encoded value.
5. **Fix — Leave-One-Out Encoding:** for each row, compute the category mean excluding that row specifically. More granular than K-fold but can still leak slightly via correlated noise; usually combined with added Gaussian noise for regularization.
6. **Industry standard:** always cross-validated / out-of-fold target encoding in production pipelines (e.g., `category_encoders.TargetEncoder` with CV, or manual K-fold loop). Never fit target encoding on the same rows you're encoding.

### 4.5 Hashing Encoding
Maps categories to a fixed number of buckets via a hash function.
- **Why it exists:** handles unbounded/unknown cardinality (new categories at inference time) without needing to store a category→index mapping; keeps dimensionality fixed regardless of how many unique categories exist.
- **Risk — collisions:** two different categories can hash to the same bucket, merging their signal. Larger hash space reduces collision risk but increases dimensionality — a tradeoff.
- **Use when:** streaming data, unbounded vocabulary (e.g., URLs, user agents), or when you can't maintain a static category list.

### 4.5b Entity Embeddings (the "embedding instead of OHE" approach)
Instead of a sparse one-hot vector, learn a **dense, low-dimensional vector** per category as part of model training (a `nn.Embedding` layer in PyTorch, or an `Embedding` layer in Keras).
- **How it works:** each category is mapped to an integer index, which indexes into a learnable weight matrix of shape `(num_categories, embedding_dim)`. Gradient descent updates the embedding vectors jointly with the rest of the network so that categories with similar effects on the target end up close together in vector space — the same idea as word embeddings (Ch.6), applied to arbitrary categoricals (user IDs, zip codes, product SKUs, store IDs).
- **Why it needs to exist / what OHE can't do:**
  - OHE dimensionality scales linearly with cardinality (50k categories → 50k columns); embeddings compress this to a fixed, small size (e.g., 16–50 dims) regardless of cardinality — no combinatorial blowup.
  - OHE treats every category as **equidistant** from every other category (all pairwise distances are equal) — it encodes *presence*, not *similarity*. Embeddings let the model discover that, say, "Toyota Camry" and "Honda Accord" behave similarly for a resale-price model, while "Ferrari 458" is far away — a relationship OHE structurally cannot represent.
  - Embeddings generalize better to **rare categories** that share structure with common ones (transfer of statistical strength), because they sit near similar categories in the learned space instead of being isolated one-hot columns with almost no training signal.
- **What happens if you use OHE instead when cardinality is high:** sparse, high-dimensional input → linear/logistic models get a huge, mostly-empty coefficient matrix, deep nets get an enormous and wasteful first-layer weight matrix, tree models fragment splits across thousands of near-useless binary columns, and rare categories essentially get no signal (a column that's 1 for 3 rows out of 5 million can't be learned reliably).
- **Cost / when NOT to use:** requires a model that can jointly learn the embedding (neural net, or a two-stage approach where you pretrain embeddings with a small NN then feed them to XGBoost/linear models) — you can't just "compute" an embedding the way you compute a mean or hash. Needs enough data per category to learn a meaningful vector; for small tabular datasets (a few thousand rows) target/frequency encoding is usually more practical than training embeddings from scratch.
- **Industry standard:** this is the default approach at scale for recommender systems and click-through-rate models (e.g., Google's Wide & Deep, YouTube's recommender, most ranking systems) where categorical cardinality (user ID, item ID) is in the millions. For classic tabular ML (XGBoost/LightGBM on a CSV), target/frequency/hashing encoding remains more common because training a full neural embedding table is overkill.

### 4.5c Weight of Evidence (WOE) Encoding
Common in credit scoring / risk modeling:
```
WOE = ln( %non-events in category / %events in category )
```
where "event" is typically the positive class (e.g., default).
- **Why it exists:** produces a monotonic, log-odds-scaled encoding that plugs directly into logistic regression (whose output *is* log-odds), making coefficients directly interpretable as risk multipliers — a specific industry need in regulated domains (banking) where model explainability is legally required.
- Often paired with **Information Value (IV)** for feature selection: IV quantifies how predictive a feature is overall based on its WOE bins, giving a standard, industry-accepted table (IV < 0.02 = useless, 0.02–0.1 = weak, 0.1–0.3 = medium, 0.3+ = strong) used heavily in credit-risk feature selection.

### 4.5d CatBoost's Ordered Target Encoding (an industry-grade leakage fix)
CatBoost's built-in categorical handling computes target encoding using **only rows that came before the current row** in a randomly permuted order (similar in spirit to how you'd only use past data in a time series), rather than K-fold. This avoids leakage without needing an explicit K-fold loop, and is one reason CatBoost handles high-cardinality categoricals well out-of-the-box without manual encoding.

### 4.6 Rare Category Grouping
Bucket categories below a frequency threshold into an "Other" category.
- **Why:** prevents overfitting to categories with too few samples to be statistically meaningful, and reduces OHE dimensionality.

### 4.7 Unseen Categories at Inference
- **Problem:** production data will eventually contain a category never seen in training.
- **Standard fixes:** map to "Other"/"Unknown" bucket (must be reserved during training), use hashing (inherently handles new categories), or use frequency/target encoding with a fallback to the global mean.
- **What happens if you don't handle this:** pipeline throws an error in production, or (worse) silently encodes as NaN/0 and corrupts predictions.

### Encoding Method Comparison Table

| Method | Handles High Cardinality | Leakage Risk | Preserves Unordered Nature | Typical Use |
|---|---|---|---|---|
| One-Hot | Poor | None | Yes | Low-cardinality nominal |
| Label/Ordinal | Good | None | No (implies order) | True ordinal data |
| Frequency | Good | Low | Partial | High-cardinality, tree models |
| Target/Mean | Excellent | High (if done wrong) | Yes (numeric signal) | High-cardinality, strong target relationship |
| Hashing | Excellent | None | Yes | Streaming / unbounded categories |

### Google-style Interview Questions — Chapter 4

**1. Why is label encoding dangerous for nominal categorical features but acceptable for ordinal ones?**
> Label encoding assigns arbitrary integers (Red=0, Blue=1, Green=2) — for a linear model, this implies Green is "twice" Blue and "three times" Red, and that the categories sit on a meaningful numeric line, none of which is true for unordered (nominal) categories. For a truly ordinal feature ("Low"=0, "Medium"=1, "High"=2), that numeric ordering *is* real information, so the encoding is a faithful, compact representation rather than a fabricated one.

**2. Walk through exactly how target encoding leaks information, and how K-fold target encoding fixes it.**
> Naive target encoding computes, for each category, `mean(target)` across *all* rows with that category — including the very row you're about to predict on. That row's own label directly influenced the feature value it now sees at training time, so the model learns to partially "look up" the answer rather than generalize — performance looks great in-sample and collapses on genuinely unseen data. K-fold target encoding fixes this by splitting the training set into K folds; for each fold, the encoding used for those rows is computed *only* from the other K−1 folds — so no row's encoded value is ever influenced by its own label.

**3. Why does smoothing help target encoding for rare categories? Write out the formula and explain each term.**
> `encoded = (n_category * category_mean + m * global_mean) / (n_category + m)`. `n_category` is how many training rows have this category; `category_mean` is the raw target mean for just this category; `global_mean` is the overall target mean across all data; `m` is a smoothing strength hyperparameter (effectively "how many prior pseudo-observations of the global mean to blend in"). When `n_category` is small (rare category), the formula is dominated by `m * global_mean`, pulling the estimate toward the safe overall average instead of trusting a noisy mean computed from just 1–2 samples. As `n_category` grows large, the term `n_category * category_mean` dominates and the encoding converges to the category's true empirical mean. This is a Bayesian-shrinkage idea — trust data more as you have more of it.

**4. Compare one-hot encoding and hashing encoding for a categorical feature with 1M unique values.**
> OHE would create 1M sparse binary columns — infeasible for memory, most models, and most rows would have almost no training signal per rare category. Hashing maps all 1M categories into a small fixed number of buckets (e.g., 2^16) via a hash function, keeping dimensionality constant regardless of cardinality, and gracefully handles brand-new categories at inference (they just hash into an existing bucket) without needing a stored vocabulary. The tradeoff is collisions — two unrelated categories landing in the same bucket blend their signal — mitigated but not eliminated by choosing a larger hash space.

**5. How would you handle a category at inference time that never appeared in training?**
> Options, in order of typical preference: (a) reserve an explicit "Unknown/Other" bucket during training (group all rare categories into it, so the model has already learned what to do with it) and route unseen categories there at inference; (b) use hashing encoding, which inherently assigns *any* string (seen or not) to a bucket; (c) fall back to the global target mean for target-encoded features. What you must avoid is letting the pipeline throw an error or silently produce NaN/0 — both corrupt downstream predictions without warning.

**6. Why can one-hot encoding hurt tree-based models even though trees don't care about magnitude?**
> Magnitude-invariance isn't the only property that matters — dimensionality does too. A single high-cardinality categorical OHE'd into thousands of sparse binary columns fragments the feature's information across many nearly-useless splits; a tree has to "rediscover" the aggregate pattern by combining many binary splits (each carrying very little information gain individually), which is inefficient and prone to overfitting on rare categories that happen to correlate with the target by chance in the training sample. Target/frequency encoding gives the tree a single, information-dense numeric column that a single split can exploit directly.

---

## Chapter 5: Date/Time Features

### Basic Extraction
Day, month, year, weekday, hour, quarter, week-of-year, is_weekend, is_month_end.
- **Why:** raw timestamps (Unix epoch or datetime objects) are essentially meaningless to most models as a single scalar — the model can't "see" that two timestamps fall on the same weekday or hour without you extracting that structure explicitly.

### Cyclical Encoding
Hour, day-of-week, and month are *cyclical* (23:00 is close to 00:00, December is close to January) — but a raw integer (0–23, 0–6, 1–12) treats them as linear, so 23 and 0 look maximally far apart to the model.
- **Fix:** encode as sine/cosine pairs:
  ```
  hour_sin = sin(2π * hour / 24)
  hour_cos = cos(2π * hour / 24)
  ```
- **Why this needs to exist:** without it, a linear model (or any distance-based method) sees 11:59 PM and 12:01 AM as nearly opposite points instead of adjacent — destroying real temporal locality.

### Recency Features
"Time since X" (e.g., days since last purchase, days since account creation).
- **Why:** recency is often one of the strongest predictors in behavioral models (RFM: Recency, Frequency, Monetary — a classic marketing feature framework).

### Lag Features & Rolling Window Aggregates
- **Lag features:** value of a variable N periods ago (e.g., `sales_lag_7`).
- **Rolling aggregates:** mean/std/min/max over a trailing window (e.g., 7-day rolling average).
- **Why:** time series models (or even standard regressors applied to time series) need explicit access to history since they don't inherently "remember" past rows the way RNNs/LSTMs do.
- **Critical rule:** rolling windows must only look *backward* — a rolling average that includes future timestamps is leakage (see Chapter 8).

### Holiday / Business-Day Flags
Binary flags for holidays, weekends, paydays, etc. — domain knowledge injected directly as a feature, since "is it a holiday" is very hard for a model to infer from a raw date.

### Google-style Interview Questions — Chapter 5

**1. Why is raw Unix timestamp usually a poor feature on its own?**
> A Unix timestamp is a single, essentially arbitrary large integer (seconds since 1970). Any periodic structure that actually drives the target — hour-of-day effects, weekday effects, seasonality — is not linearly related to that raw number, so a model would have to learn extremely non-linear, high-frequency mappings from a huge numeric range to recover patterns that become trivial once you extract `hour`, `weekday`, `month` directly.

**2. Explain why cyclical (sin/cos) encoding is needed for hour-of-day instead of a plain integer 0-23.**
> Hour is cyclical — hour 23 (11pm) is one hour away from hour 0 (midnight), but as plain integers they're 23 apart, the maximum possible distance. Any distance-based or linear method would treat 11pm and midnight as *maximally dissimilar* when they're actually adjacent. Representing hour as `(sin(2π·h/24), cos(2π·h/24))` places it on a circle in 2D space, where 23:00 and 00:00 are genuinely close together, correctly preserving the cyclical adjacency.

**3. What's the danger of a rolling average window that isn't strictly backward-looking?**
> If a "7-day rolling average" window includes days *after* the prediction timestamp, the feature contains information the model wouldn't actually have access to at prediction time in production — classic temporal leakage. The model will appear highly accurate offline (it's essentially peeking at a smoothed version of the near future) and then perform far worse in real deployment, where future data obviously isn't available yet.

**4. Give three time-based features you'd engineer for a demand-forecasting model and justify each.**
> (1) `day_of_week` / `is_weekend` — captures predictable weekly demand cycles (e.g., retail spikes on weekends). (2) `rolling_7day_avg_demand` (backward-looking only) — captures recent trend/momentum that a single day's raw value can't convey. (3) `is_holiday` / `days_to_next_holiday` — captures known demand spikes/dips around holidays that wouldn't be inferable from the date alone without external calendar knowledge injected as a feature.

---

## Chapter 6: Text & Unstructured Features (Advanced/Optional)

### Bag-of-Words / TF-IDF
- **Bag-of-Words:** counts of each word in a document, ignoring order.
- **TF-IDF (Term Frequency–Inverse Document Frequency):** down-weights common words (like "the") and up-weights rare, distinctive words.
  ```
  TF-IDF(t,d) = TF(t,d) * log(N / DF(t))
  ```
- **Why TF-IDF exists over raw counts:** raw word counts overweight frequent-but-uninformative words; TF-IDF corrects for this by penalizing words that appear across many documents (low information content).

### N-grams
Sequences of N consecutive words/tokens ("machine learning" as a 2-gram) — captures local word order/context that unigram BoW/TF-IDF misses.

### Word Embeddings
Dense vector representations (Word2Vec, GloVe) or contextual embeddings (BERT) capturing semantic meaning — "king" and "queen" end up close in vector space.
- **Why they exist over BoW/TF-IDF:** BoW treats every word as an independent, unrelated dimension — "good" and "great" share no similarity. Embeddings encode semantic similarity learned from large corpora.

### Simple Text Statistics
Character/word count, average word length, punctuation counts, sentiment score, keyword flags — cheap, interpretable signals that are often surprisingly predictive (e.g., spam detection, review-quality scoring).

### Google-style Interview Questions — Chapter 6

**1. Why does TF-IDF outperform raw word counts as a baseline text feature?**
> Raw counts give equal per-occurrence weight to every word, so extremely common but low-information words ("the", "is", "and") dominate the count vector purely because they appear often, drowning out rarer words that actually distinguish one document from another. TF-IDF multiplies term frequency by an inverse-document-frequency factor that down-weights words appearing in most documents and up-weights words that are frequent *within* a document but rare *across* the corpus — directly emphasizing the words that carry discriminative signal.

**2. What's the core limitation of Bag-of-Words that word embeddings solve?**
> BoW/TF-IDF represents each word as an independent dimension with no notion of meaning or similarity — "excellent" and "great" are as unrelated to each other as "excellent" and "terrible" in that representation, and word order is discarded entirely. Word embeddings (Word2Vec, GloVe) are trained so that words used in similar contexts end up close together in a dense vector space, capturing semantic similarity; contextual embeddings (BERT) go further and also capture word order and context-dependent meaning (e.g., "bank" of a river vs. a "bank" account get different vectors).

**3. When would simple text statistics (length, punctuation) actually outperform embeddings?**
> When the signal is genuinely *structural/stylistic* rather than semantic — e.g., spam detection where excessive exclamation marks, ALL-CAPS ratio, or unusually short/long message length are strong tells; or when the dataset is too small to fine-tune or meaningfully leverage a large embedding model (embeddings need either pretrained transfer or enough data to be useful, and can add noise/overfitting risk on tiny datasets where a handful of interpretable statistics generalize better and are far cheaper to compute and explain).

---

## Chapter 7: Feature Interactions & Crosses

### Polynomial Features
Adding x², x·y, etc. explicitly.
- **Why they exist:** linear models can only learn linear relationships between input and target. If the true relationship is `y ~ x1 * x2` (an interaction effect), a linear model literally cannot represent that unless you hand it the `x1*x2` term directly as a feature.
- **What happens if you don't add them:** the model systematically underfits any multiplicative/interaction relationship in the data — no amount of more data fixes this for a linear model; it's a representational limitation, not a data limitation.
- Tree-based models can *implicitly* learn some interactions via sequential splits, but explicit crosses still often help, especially for shallow trees or linear/FM-style models.

### Domain-Driven Ratios
E.g., debt-to-income, price-per-square-foot, click-through-rate (clicks/impressions).
- **Why:** ratios often encode the *actual* causal driver better than either raw component alone. Debt-to-income predicts default risk far better than debt or income individually — a person with $50k debt is fine if they earn $500k, risky if they earn $30k.

### Group-by Aggregations
Aggregating a feature by group (e.g., average purchase amount per customer, per store, per category).
- **Why:** captures entity-level behavior patterns beyond a single row's raw values — critical in customer/user-level modeling.
- **Leakage risk:** if you compute this aggregate using the full dataset including the row you're predicting (or including future data for that entity), it leaks — same principle as target encoding (Chapter 4/8).

### Automated Interaction Discovery
Use feature importance from a trained tree model (e.g., XGBoost's gain-based or SHAP interaction values) to identify which pairs of features have strong joint effects, then explicitly engineer those crosses for simpler models.

### When / When Not
- **Use crosses:** linear/logistic regression, factorization machines, shallow trees, when domain knowledge suggests a real interaction (e.g., ad relevance = ctr * bid).
- **Be cautious:** exhaustive polynomial expansion (degree 2+ across many features) explodes feature count combinatorially — leads to overfitting and computational blowup. Prune with feature selection (Chapter 9) afterward.

### Google-style Interview Questions — Chapter 7

**1. Why can't a linear model learn an interaction effect without an explicit cross term?**
> A linear model computes `y = w1*x1 + w2*x2 + ... + b` — a strictly additive combination of individual feature contributions. If the true relationship is multiplicative (`y` depends on `x1 * x2`, e.g., ad relevance = click-through-rate × bid), no combination of additive weights on `x1` and `x2` alone can represent that curved, joint dependency — it's a representational ceiling of the model class, not something more data can fix. Explicitly adding an `x1*x2` column turns the interaction into a new "input" the linear model *can* weight additively.

**2. Why is debt-to-income often a better feature than debt and income separately? Generalize this into a principle.**
> The target (default risk) doesn't actually depend on the absolute scale of debt or income individually — it depends on the *relationship* between them. $50k debt is low-risk for a $500k earner and high-risk for a $30k earner; neither raw feature alone captures this, but their ratio does. General principle: when domain knowledge tells you the causal driver is a *relationship between two quantities* (a rate, a ratio, a proportion) rather than either quantity in isolation, engineering that explicit relationship as a feature gives the model direct access to the true causal structure instead of forcing it to approximate the ratio through interaction terms or many data-hungry splits.

**3. What's the risk of computing a "average purchase per customer" feature without care about time ordering?**
> If the aggregate is computed using the *entire* dataset (including transactions that happen after the row/timestamp you're trying to predict), it leaks future information into a feature that's supposed to represent "what we knew about this customer at prediction time" — the model implicitly gets to see outcomes that haven't happened yet relative to the row being scored, inflating offline performance in a way that won't hold up in production, where future purchases obviously aren't known yet.

**4. How would you decide which pairwise feature crosses to create out of 50 features, without brute-forcing all C(50,2) combinations?**
> Use a tree-based model's structure to guide the search rather than exhaustively trying every pair: train a gradient-boosted tree on the base features and inspect SHAP interaction values (or simpler: how often two features co-occur in the same decision path / their gain-based interaction strength) to rank which feature pairs have the strongest joint effect on the target. Also lean on domain knowledge — if you know two features are plausibly related mechanistically (price and quantity, bid and CTR), prioritize those crosses directly rather than searching blindly. Only after narrowing to a shortlist do you explicitly engineer and validate those specific cross terms.

---

## Chapter 8: Data Leakage Detection

### Why Leakage Is the #1 Silent Killer of ML Projects
Leakage means information from outside the legitimate training scope (usually, directly or indirectly, the label, or future data) sneaks into your features. **The danger:** the model looks *excellent* offline (high validation/test accuracy) and then fails in production, because the leaked information doesn't exist at prediction time in the real world.

### 8.1 Temporal Leakage
Using future information to predict the past — most common in time series.
- Example: using "total monthly revenue" (computed from the full month, including future days) to predict daily churn mid-month.
- **Fix:** always construct features using only data available *strictly before* the prediction timestamp. Use time-based train/test splits, not random splits, for temporal data.

### 8.2 Target Leakage via Proxy Variables
A feature that is, functionally, a disguised version of the label.
- Classic example: predicting "will patient be diagnosed with disease X" using a feature "was prescribed medication for X" — the prescription *only happens after* diagnosis, so it's a proxy for the answer.
- **Why this is subtle:** the feature isn't literally the label, so it's easy to miss in a quick correlation check — it requires understanding the *data generating process*, not just statistics.

### 8.3 Group / Entity Leakage
Same entity (patient, user, device) appearing in both train and test sets.
- Example: multiple rows per patient across visits, split randomly — the model "memorizes" patient-specific quirks in train and then sees the same patient in test, inflating performance.
- **Fix:** split by entity (GroupKFold), never by row, when rows aren't independent.

### 8.4 Preprocessing Leakage
Fitting scalers, encoders, imputers, or feature selectors on the *full* dataset (train+test) before splitting.
- Example: computing `StandardScaler().fit()` on the whole dataset means the test set's mean/variance influenced the scaling of the training set — a subtle but real leak of test distribution info into training.
- **Fix:** always `fit` on train only, `transform` on val/test. Use `sklearn.Pipeline` to enforce this structurally — it refits everything correctly inside each cross-validation fold automatically.

### 8.5 Target Encoding Leakage (cross-ref Chapter 4)
Covered in depth in Ch.4 — computing category-target means without out-of-fold logic is one of the most common real-world leakage bugs, because it's easy to write correct-looking code that's actually wrong.

### How to Systematically Audit a Pipeline for Leakage
1. **Suspiciously high performance** (near-perfect AUC/accuracy) is the #1 red flag — investigate before celebrating.
2. Check feature importance — if one feature dominates overwhelmingly, ask "could this only exist *after* the label is known?"
3. Recreate the pipeline with a strict *temporal cutoff* simulation — does performance hold if you literally cannot see anything past the prediction time?
4. Verify all `fit()` calls (scalers, encoders, selectors, imputers) happen only on training folds, ideally inside a `Pipeline`/`ColumnTransformer` + cross-validation loop.
5. For grouped/repeated-entity data, confirm splits respect group boundaries (`GroupKFold`, `GroupShuffleSplit`).

### Google-style Interview Questions — Chapter 8

**1. Your model gets 99% AUC on validation but performs poorly in production. Walk through how you'd diagnose this.**
> First, treat the near-perfect AUC itself as the red flag rather than a win. Steps: (a) inspect feature importance / SHAP — if one or two features dominate overwhelmingly, ask whether they could only be known *after* the label is determined (proxy/target leakage); (b) check how the train/val split was made — if rows from the same entity (user, patient) appear in both, that's group leakage; (c) check whether any preprocessing (scaling, encoding, feature selection) was fit on the full dataset before splitting; (d) for time-indexed data, verify the split is chronological and that no feature uses information timestamped after the prediction point; (e) simulate the real production constraint directly — recompute every feature as if you could only see data available at actual prediction time, and see if performance collapses back down to something more realistic.

**2. Explain the difference between target leakage and temporal leakage with examples.**
> Target leakage: a feature is (directly or via a proxy) a disguised function of the label itself, regardless of time — e.g., using "was prescribed medication X" to predict "has disease X," where the prescription is causally downstream of the diagnosis. Temporal leakage: a feature legitimately isn't a proxy for the label, but it's computed using data that, chronologically, wouldn't have existed yet at prediction time — e.g., using the full month's total revenue (including future days) to predict mid-month churn. They can overlap (a temporally-leaked feature can also functionally be a target proxy), but the core distinction is "is this feature causally the answer" vs. "is this feature only available too late."

**3. Why does fitting a StandardScaler on the full dataset before splitting count as leakage, even though it doesn't touch the label at all?**
> Leakage isn't only about the label — it's about any information from data the model shouldn't have access to at train time influencing the training process. Computing `mean`/`std` over the full dataset means the test set's distribution characteristics silently shape how every training row gets scaled. This is subtle (it never touches y), but it means your validation metric is no longer a clean estimate of "how well does this model generalize to truly unseen data" — the test set's statistics already informed the pipeline in a small but real way, biasing the evaluation optimistic.

**4. How does GroupKFold prevent leakage that regular KFold wouldn't catch?**
> Regular KFold splits rows randomly, so if a single entity (patient, user) contributes multiple, correlated rows, those rows can be split across both train and test — the model can partially "memorize" patient-specific quirks in train and then get an inflated score recognizing the same patient in test, even though it hasn't truly generalized to a *new* patient. GroupKFold enforces that all rows sharing a group ID (e.g., `patient_id`) stay entirely within one fold, so the model is always evaluated on genuinely unseen entities — a much more honest test of generalization for grouped/repeated-measures data.

**5. Design a checklist you'd use to audit a teammate's feature engineering pipeline for leakage before it goes to production.**
> - Is the train/test (or train/val/test) split made *before* any transformer is fit?
> - Are all scalers/encoders/imputers/selectors wrapped in a Pipeline/ColumnTransformer so `fit()` only ever sees training folds?
> - For grouped or repeated-entity data, is the split done by group (GroupKFold), not by row?
> - For time-indexed data, is the split chronological, and does every engineered feature (rolling windows, aggregates, target encodings) use only data strictly before each row's timestamp?
> - Does target/mean encoding use out-of-fold or ordered computation, never in-fold means?
> - Does any feature's definition plausibly happen *after* the label is determined in the real-world process (proxy check)?
> - Is offline performance suspiciously high relative to reasonable domain expectations — if so, investigate before shipping?

---

## Chapter 9: Feature Selection

### Why Select Features At All
More features isn't always better: irrelevant/redundant features add noise, increase overfitting risk, slow training/inference, and hurt interpretability. **The bias-variance tradeoff:** too many features → low bias, high variance (overfitting); too few → high bias (underfitting).

### 9.1 Filter Methods (model-agnostic, fast, statistical)
- **Correlation:** drop features with near-zero correlation to target (numeric-numeric); also drop one of a highly-correlated *pair* of features (redundancy/multicollinearity).
- **Chi-square test:** measures independence between categorical feature and categorical target — used for categorical feature relevance.
- **Mutual Information:** captures both linear and non-linear dependency between feature and target (more general than correlation, which only captures linear relationships).
- **Variance Threshold:** drop features with near-zero variance (constant or near-constant columns carry no information).
- **Why filter methods first:** cheap, fast, model-independent — good for a first pass on high-dimensional data before more expensive methods.

### 9.2 Wrapper Methods (model-in-the-loop, expensive, more accurate)
- **RFE (Recursive Feature Elimination):** train model, drop least important feature, repeat.
- **Forward/backward selection:** iteratively add/remove features based on model performance change.
- **Why:** accounts for feature *interactions* and the actual downstream model's behavior — filter methods can't see interactions (a feature useless alone might be crucial combined with another).
- **Cost:** computationally expensive — requires retraining the model many times.

### 9.3 Embedded Methods (selection happens during model training)
- **Lasso (L1 regularization):** shrinks some coefficients exactly to zero, performing selection as a side-effect of training a linear model.
- **Tree-based importance:** Gini importance (impurity reduction) or permutation importance (drop in performance when a feature is shuffled) from Random Forest/XGBoost.
- **Why embedded methods are popular in industry:** cheaper than wrapper methods (no repeated retraining loop) while still being model-aware (unlike filter methods).

### 9.4 Multicollinearity Diagnostics
- **VIF (Variance Inflation Factor):** quantifies how much a feature's variance is inflated due to correlation with other features. VIF > 5–10 typically flags problematic multicollinearity.
- **Why it matters:** multicollinearity doesn't hurt prediction accuracy much for many models, but it destroys coefficient interpretability in linear models (unstable, sign-flipping coefficients) and can cause numerical instability in matrix inversion (`(XᵀX)⁻¹`).

### 9.5 SHAP Values (post-hoc, not pure selection)
Game-theoretic attribution of each feature's contribution to individual predictions.
- **Why it matters for feature engineering:** goes beyond global importance (like Gini importance) to show *direction* and *interaction* effects per prediction — useful for validating that engineered features behave as intended, and for spotting leakage (a feature with suspiciously large, one-directional SHAP impact is a leakage red flag).

### 9.6 Dimensionality Reduction (Selection vs. Compression — an important distinction)
- **PCA:** projects features into orthogonal components maximizing variance — this is *compression*, not selection; you lose the original interpretable features and instead get a linear combination.
- **t-SNE / UMAP:** primarily for visualization of high-dimensional data in 2D/3D — not typically used as input features for downstream models (they distort distances and aren't stable/deterministic across runs).
- **When to use compression instead of selection:** when the features are highly correlated/redundant and you care about downstream performance/speed more than interpretability of individual input features.

### Google-style Interview Questions — Chapter 9

**1. Compare filter, wrapper, and embedded feature selection methods — cost, accuracy, and when you'd use each.**
> Filter methods (correlation, chi-square, mutual information, variance threshold) are cheap, model-agnostic, and fast — good as a first pass on high-dimensional data, but they evaluate each feature in isolation and miss interaction effects. Wrapper methods (RFE, forward/backward selection) retrain the actual downstream model repeatedly, so they capture interactions and are more accurate, but are computationally expensive — impractical on very large feature sets. Embedded methods (Lasso, tree-based importance) build selection into a single model training pass, striking a middle ground — model-aware like wrapper methods but far cheaper, which is why they're the most common in industry pipelines.

**2. Why does mutual information capture relationships that correlation misses?**
> Pearson correlation measures only *linear* association — a feature with a strong U-shaped or otherwise non-linear relationship to the target can have correlation near zero despite being highly predictive. Mutual information measures the general statistical dependence between two variables (based on their joint vs. marginal distributions), capturing any kind of relationship — linear, non-linear, even non-monotonic — because it doesn't assume a functional form.

**3. Explain VIF and why high multicollinearity is a problem even if predictive accuracy doesn't drop.**
> VIF for a feature is computed by regressing that feature against all other features and measuring `1 / (1 − R²)` of that regression — a high VIF means the feature is well-predicted by the others, i.e., it's redundant. Multicollinearity often doesn't hurt a model's raw predictive accuracy much (the redundant information is still "in there" somewhere), but it destroys the *interpretability and stability* of linear model coefficients: when two features are highly correlated, the model can arbitrarily split credit between them, producing unstable, sometimes sign-flipped coefficients that change drastically with small data perturbations — making the model untrustworthy for any use case that relies on interpreting coefficient values (e.g., "which factor drives risk more").

**4. What's the fundamental difference between feature selection and dimensionality reduction (e.g., PCA)?**
> Feature selection *chooses a subset* of the original, interpretable features and discards the rest — the surviving features keep their original meaning. Dimensionality reduction (PCA) *transforms* the entire feature set into a new, smaller set of derived components, each a linear (or non-linear, for other methods) combination of all original features — you gain compactness and decorrelation but lose direct interpretability of any individual output dimension in terms of the original variables.

**5. How would SHAP values help you catch a leaked feature that a standard feature-importance plot wouldn't reveal clearly?**
> A standard importance plot (e.g., Gini importance) gives one aggregate number per feature and can be noisy or misleading about *why* a feature matters. SHAP values show, per individual prediction, how much each feature pushed the prediction up or down — a leaked feature typically shows an unnaturally clean, strong, one-directional effect (e.g., a single feature nearly determining the output by itself across almost all predictions) rather than the more distributed, context-dependent contribution pattern you'd expect from a genuinely causal feature. That stark, almost deterministic SHAP signature is a strong tell to go investigate the feature's construction for leakage.

**6. Why can Lasso zero out a coefficient entirely while Ridge (L2) never does?**
> Lasso's penalty is the L1 norm (sum of absolute coefficient values), which has sharp corners at zero in coefficient space — geometrically, the optimization solution is likely to land exactly on one of those corners, forcing some coefficients to exactly zero. Ridge's penalty is the L2 norm (sum of squared coefficients), which is smooth and circular/spherical in coefficient space with no corners — the optimum shrinks coefficients toward zero proportionally but essentially never lands exactly on zero, so Ridge regularizes magnitude without performing true feature selection.

---

## Chapter 10: Engineering Hygiene & Production

### Fit-on-Train-Only Discipline
Every stateful transformation (scaler, encoder, imputer, feature selector) must be `fit()` exclusively on the training partition of *each* CV fold, then `transform()`-applied to validation/test. This isn't optional — it's the boundary between a valid experiment and an inflated, misleading one.

### `sklearn.Pipeline` / `ColumnTransformer` Pattern
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])
full_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier())
])
```
**Why this pattern is the industry standard:** it structurally *forces* correct fit/transform boundaries inside cross-validation — you can't accidentally leak because the whole pipeline (preprocessing + model) is refit from scratch on each fold.

### Feature Versioning & Feature Stores
In production ML systems (recommendation, fraud, ranking), the same features must be computed *identically* at training time and serving time (this is the "training-serving skew" problem).
- **Feature stores** (e.g., Feast, Tecton, or internal equivalents at large tech companies) centralize feature definitions, computation logic, and serve both batch (training) and low-latency (real-time inference) access from a single source of truth.
- **Why they need to exist:** without one, teams often re-implement feature logic separately for training (batch/offline, e.g., Spark) and serving (online, e.g., a Java microservice) — subtle mismatches between the two implementations cause silent, hard-to-debug production degradation.

### Validating New Features with Cross-Validation, Not Intuition
A feature that "seems like it should help" must be validated empirically:
1. Add the feature, run k-fold CV, compare metric distributions (not a single split — use paired statistical tests if possible).
2. Check for leakage red flags (Chapter 8) if performance jumps suspiciously.
3. Check feature importance / SHAP to confirm the model is using it sensibly, not spuriously.

### Google-style Interview Questions — Chapter 10

**1. Why does wrapping preprocessing and model training in a single sklearn Pipeline matter for cross-validation correctness?**
> Without a Pipeline, it's easy to accidentally fit a scaler/encoder/selector once on the full training set and then reuse it across all CV folds — meaning each fold's "held-out" portion actually influenced the preprocessing statistics used to transform it, a subtle leak. Wrapping everything in a Pipeline means `cross_val_score`/`GridSearchCV` refits the *entire* pipeline (preprocessing + model) from scratch on each fold's training portion alone, structurally guaranteeing the held-out fold never influences its own preprocessing — correctness is enforced by the code structure rather than relying on the practitioner remembering the rule every time.

**2. What is training-serving skew, and how do feature stores address it?**
> Training-serving skew is when the feature values computed during offline model training don't exactly match the feature values computed during real-time inference — often because training features are computed in one system (e.g., a Spark batch job) and serving features are computed in a completely different system (e.g., a low-latency online service), implemented separately by different code paths that can subtly diverge (different rounding, different time windows, different null-handling). This causes a model that performed well offline to silently underperform in production because it's now seeing features that don't quite match what it was trained on. Feature stores solve this by defining each feature's computation logic *once*, in a single source of truth, and serving both the batch/offline view (for training) and the low-latency online view (for real-time inference) from that same definition — eliminating the two-implementation divergence risk.

**3. You added a new feature and offline AUC jumped from 0.75 to 0.95. What do you do before shipping it?**
> Treat the jump as a leakage red flag first, not a win. Check: is the feature computed using information that would genuinely be available at real prediction time in production, with correct temporal boundaries? Is it, directly or as a proxy, derived from the label or something that only exists after the label is determined? Inspect SHAP/feature importance — does this one feature dominate suspiciously? Recompute it with a strict "what did we know at prediction time" simulation and re-evaluate. Only after ruling out leakage would you trust and ship a jump that large.

**4. Design a system for computing "average order value in the last 30 days" consistently for both batch model training and real-time inference serving.**
> Define the feature's computation logic once (e.g., as a SQL/Spark transformation or a shared feature-definition config) inside a feature store, specifying the exact window logic (rolling 30 days strictly before the event timestamp), aggregation function, and null-handling. The batch/offline pipeline materializes this feature historically for training data (point-in-time correct — computed as of each historical row's own timestamp, not "as of today," to avoid leakage). The online serving path maintains a continuously updated, low-latency view (e.g., a precomputed rolling aggregate refreshed on each new order, stored in a fast key-value store keyed by customer ID) that implements the *same* windowing/aggregation definition. Integration tests periodically compare a sample of online-computed feature values against what the batch pipeline would compute for the same entity/timestamp to catch any drift between the two implementations.

---

## Master Conceptual Interview Set (Cross-Chapter, Google-style)

1. **"Walk me through your feature engineering process for a new tabular dataset from scratch."** — Expect: EDA → missingness diagnosis → type-specific transforms → leakage audit → selection → validation loop.
2. **"How do you decide whether a transformation should be fit before or after train/test split?"** — Always after split, fit on train only, structurally enforced via Pipeline.
3. **"Give an example where feature engineering created a feature that leaked the label, and how you'd catch it."**
4. **"Why might adding more features hurt a linear model but not hurt (or even help) a random forest?"** — Multicollinearity/noise hurts linear coefficient stability; trees are more robust to irrelevant features (though not immune — still adds variance/overfitting risk).
5. **"You're told a categorical feature has 2 million unique values. What are your options and their tradeoffs?"** — Hashing, frequency/target encoding (with CV), embeddings, rare-category grouping.
6. **"Explain feature engineering for a cold-start problem (new user/item with no history)."** — Fall back to content-based/demographic features, population-level priors, or default/global-mean encodings; explicit "is_new_user" flags.
7. **"How would you engineer features for a fraud detection model where the label is extremely imbalanced?"** — Rate-based/ratio features, recency/velocity features (transactions in last hour), entity-level aggregates, careful leakage checks (label-related proxy actions like "account frozen" must post-date the transaction).
8. **"What's the difference between feature engineering for a linear model vs. a gradient-boosted tree model?"** — Linear models need explicit encoding/scaling/interactions/normality; trees handle raw scale, non-linearity, and some interactions natively but still benefit from strong domain features and proper categorical encoding.
9. **"How do you keep training and serving features consistent in a real-time system?"** — Feature store / shared feature computation library, integration tests comparing batch vs. online feature values.
10. **"Your team lead says 'just one-hot encode everything.' What's wrong with that approach at scale, and what would you propose instead?"**

---

*End of notes.*
