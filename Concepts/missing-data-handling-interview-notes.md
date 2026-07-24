# Missing Data Handling — Interview Notes (End-to-End, With Examples)

## 1. Why the *Mechanism* of Missingness Matters More Than the Missingness Itself

The single most important idea in this entire topic, and the one interviewers probe hardest: **how you should handle missing data depends entirely on *why* it's missing, not just *how much* is missing.** There are three formal categories (Rubin, 1976) — know these cold.

**MCAR — Missing Completely At Random.** The probability of a value being missing is unrelated to any observed or unobserved data. Example: a lab sample was randomly lost due to a shipping error, unrelated to the patient's actual health.

**MAR — Missing At Random.** The probability of missingness depends on *observed* data, but not on the missing value itself. Example: older patients are less likely to report their income (missingness depends on the observed "age" column), but among patients of the same age, whether income is missing has nothing to do with the actual income value.

**MNAR — Missing Not At Random.** The probability of missingness depends on the *unobserved value itself*. Example: people with very high income are less likely to disclose it (missingness depends directly on the income value that's missing) — or a medical sensor that fails specifically when readings are extreme.

```
MCAR:  Missing ⊥ (Observed data, Missing value)     ← pure random loss, safest case
MAR:   Missing depends on Observed data only        ← recoverable if you use the observed data
MNAR:  Missing depends on the Missing value itself   ← the hardest case, potentially unfixable
                                                        without external assumptions/domain knowledge
```

**Why this hierarchy matters practically:** most standard imputation techniques (mean/median, regression imputation, MICE) implicitly assume **MAR or MCAR**. If your data is actually **MNAR**, these techniques can silently introduce bias no amount of clever imputation math can fix — you need to explicitly model the missingness mechanism itself (or accept the bias and document it).

---

## 2. Diagnosing Which Mechanism You're Dealing With

```
Step 1: Is missingness correlated with OTHER observed columns?
        │
        ├─ NO  →  looks like MCAR (test formally with Little's MCAR test,
        │          or informally by comparing distributions of other features
        │          between rows-with-missing vs rows-without-missing)
        │
        └─ YES →  at least MAR. Now ask:
                   │
                   Step 2: After conditioning on those observed columns,
                   is missingness STILL correlated with the (unobserved) 
                   value itself?
                   │
                   ├─ NO  →  MAR (the observed columns "explain" the missingness)
                   │
                   └─ YES / CAN'T RULE OUT →  possibly MNAR
                             (this is fundamentally hard to test directly,
                              since you don't have the missing values to check
                              against — usually requires domain reasoning,
                              not a statistical test)
```

**Practical diagnostic checklist (what you'd actually do on a real dataset):**
- Create a binary "is_missing" indicator column for the feature in question.
- Check if that indicator correlates with other observed features (a t-test or chi-square test between groups, or just visually compare distributions) — if yes, at minimum MAR, not MCAR.
- Use domain knowledge to reason about whether the value itself would plausibly influence its own missingness (income, health status, salary, embarrassing survey answers are classic MNAR suspects; sensor/shipping/random system failures are classic MCAR suspects).
- Little's MCAR test (a formal statistical test) can support/refute MCAR, but it cannot distinguish MAR from MNAR — that distinction fundamentally cannot be verified from the observed data alone, since MNAR is about the *unobserved* values.

---

## 3. Handling Method 1 — Deletion

**Listwise deletion (complete-case analysis):** drop any row with a missing value in any column used by the model.

```
Original data (X = missing):
┌─────┬─────┬─────┐         Listwise deletion result:
│ A   │ B   │ C   │         ┌─────┬─────┬─────┐
├─────┼─────┼─────┤         │ A   │ B   │ C   │
│ 1   │ 2   │ 3   │   ──►   ├─────┼─────┼─────┤
│ X   │ 5   │ 6   │         │ 1   │ 2   │ 3   │   ← only this row survives
│ 7   │ X   │ 9   │         └─────┴─────┴─────┘
│ 10  │ 11  │ X   │         (3 out of 4 rows lost!)
└─────┴─────┴─────┘
```

- **Safe under MCAR** (dropping random rows doesn't bias the remaining sample, since missingness is unrelated to anything) — but wastes data, and the waste compounds fast: with 10 columns each missing 5% independently, roughly $1-(0.95)^{10} \approx 40\%$ of rows get dropped.
- **Biased under MAR or MNAR** — since missingness correlates with observed or unobserved data, the surviving rows are a systematically non-representative sample of the population (e.g., if older patients are more likely to have missing income, listwise deletion skews your remaining data younger).

**Pairwise deletion:** for each specific calculation (e.g., each pairwise correlation in a correlation matrix), use all rows that have both relevant values, even if those rows would be dropped for a different pair. Preserves more data than listwise, but can produce statistically inconsistent results (e.g., a correlation matrix that isn't positive semi-definite) since different cells are computed from different subsets of rows.

**Column deletion:** drop an entire feature if it has too much missingness to be usably imputed (common rule of thumb: >40-50% missing, though this is a judgment call depending on the feature's importance).

---

## 4. Handling Method 2 — Simple Imputation

**Mean/Median imputation** (numeric features):

```
Original "Age" column:  [25, 30, X, 40, 35, X, 28]
Mean of observed values: (25+30+40+35+28)/5 = 31.6

Imputed:  [25, 30, 31.6, 40, 35, 31.6, 28]
```

- Median is preferred over mean when the feature is skewed or has outliers (median is robust to extreme values; mean is pulled by them).
- **Critical side effect: imputation artificially shrinks variance** — every imputed row gets exactly the same value, creating an artificial spike at that value and reducing the feature's true spread. This distorts downstream statistics (correlations, model coefficients, feature importance) in ways that are easy to overlook.
- Should be computed **only on the training fold** and applied to validation/test using the training statistic — computing the mean on the full dataset before splitting is a leakage bug, identical in spirit to fitting a `StandardScaler` on the full dataset (see Cross-Validation notes §14).

**Mode imputation** (categorical features): replace missing values with the most frequent category. Same variance-shrinkage concern applies (all imputed rows become identical, artificially inflating that category's apparent prevalence).

**Constant/placeholder imputation:** fill with a fixed sentinel value (e.g., "Unknown" for categorical, -999 or 0 for numeric).

- Works reasonably well for tree-based models, which can learn to treat the sentinel as its own meaningful "bucket" (this is often actually *useful* if missingness itself carries a signal — see §6).
- Dangerous for linear/distance-based models — a numeric sentinel like -999 can be treated as a real, extreme numeric value and badly distort feature scaling, distance calculations (k-NN), or coefficient estimates unless the model also gets an explicit missingness indicator (see §6) to separate "real -999" from "was missing."

**Forward-fill / backward-fill** (time series specific): carry the last observed value forward (or next observed value backward) to fill gaps.

```
Time:      t1   t2   t3   t4   t5   t6
Value:     10   X    X    15   X    20

Forward-fill:  10   10   10   15   15   20
                     ↑    ↑         ↑
              carried forward from the last known value
```

- Only appropriate when temporal continuity is a reasonable assumption (e.g., sensor readings that don't change quickly) — inappropriate if the underlying value could have changed meaningfully during the gap.

---

## 5. Handling Method 3 — Model-Based Imputation

**Regression imputation:** train a regression model to predict the missing feature from the other (observed) features, using rows where the target feature isn't missing as the training set.

```
Rows with "Income" observed:           Train a regression model:
┌──────┬────────┬─────────┐            Income ~ f(Age, Education, City)
│ Age  │ Educ   │ Income  │
│ 35   │ BS     │ 65000   │   ──►      Then predict Income for rows
│ 42   │ MS     │ 88000   │            where it's missing, using
│ ...  │ ...    │ ...     │            their Age/Education/City.
└──────┴────────┴─────────┘
```

- More sophisticated than mean imputation — captures relationships between features rather than assuming every missing value equals a single constant.
- **Still shrinks variance** (predictions are inherently less variable than real observed values, since regression predicts the conditional mean) — just less severely than simple mean imputation, since it at least varies row-by-row based on other features.
- Assumes MAR at minimum — if missingness depends on the value being imputed itself (MNAR), this method inherits the same bias problem as simple imputation, just dressed up with more sophisticated math.

**k-NN imputation:** for each row with a missing value, find the $k$ most similar rows (by the other, observed features) and impute using their average (numeric) or mode (categorical) value for the missing feature.

```
Row with missing Income: [Age=35, Education=BS, City=NYC, Income=?]

Find 5 nearest neighbors by (Age, Education, City) distance:
  Neighbor 1: Income=62000
  Neighbor 2: Income=68000
  Neighbor 3: Income=71000
  Neighbor 4: Income=59000
  Neighbor 5: Income=65000
                              → Imputed Income = mean = 65000
```

- Captures local structure well (similar rows likely have similar missing values) but computationally expensive at scale (distance computation for every missing row against every other row) and sensitive to the choice of $k$ and distance metric — needs the *other* features to be reasonably scaled/encoded first (a chicken-and-egg problem if those features also have missingness).

**MICE (Multiple Imputation by Chained Equations)** — the gold-standard statistical approach:

```
Step 1: Initialize all missing values with simple imputation (e.g., mean).
Step 2: For feature 1 (with real missingness): 
        - Regress feature 1 on all other features (using their CURRENT
          imputed/observed values) among rows where feature 1 is observed.
        - Predict and re-impute feature 1's missing values using this model.
Step 3: For feature 2: repeat the same process, now using feature 1's
        UPDATED values from Step 2.
Step 4: Continue cycling through every feature with missingness, each time
        using the most recently updated values of all other features.
Step 5: Repeat the entire cycle (all features) for several iterations
        until the imputed values stabilize/converge.
Step 6 (the "Multiple" in MICE): repeat the ENTIRE process multiple times
        with different random starting points/draws, producing several
        completed datasets — then train your actual model on each,
        and pool the results (e.g., average coefficients, combine
        variance estimates) using Rubin's combining rules.
```

- The "Multiple" part is what distinguishes MICE from single-pass regression imputation: by generating several plausible completed datasets and pooling results, MICE naturally captures and communicates the **uncertainty introduced by imputation itself** — something single-imputation methods (mean, regression, k-NN) fundamentally cannot represent, since they present a single "confident" imputed value with no indication of how uncertain that value really is.
- Computationally the most expensive method here, but generally the most statistically principled — standard in fields like epidemiology and social science where honest uncertainty quantification matters as much as the point estimate.

---

## 6. The Missingness Indicator Technique (often the most practically useful lever)

Add a new binary column flagging whether each value was originally missing, **alongside** whatever imputation method you use for the value itself.

```
Original:                    With missingness indicator + imputation:
┌───────┬─────┐              ┌───────┬─────┬─────────────────┐
│ Row   │ Age │              │ Row   │ Age │ Age_was_missing │
├───────┼─────┤              ├───────┼─────┼─────────────────┤
│ 1     │ 25  │      ──►     │ 1     │ 25  │        0        │
│ 2     │ X   │              │ 2     │ 31.6│        1         ← flagged!
│ 3     │ 40  │              │ 3     │ 40  │        0        │
└───────┴─────┘              └───────┴─────┴─────────────────┘
```

- **This is the single most important practical technique when missingness itself is informative** (which is extremely common in real-world data, especially under MAR/MNAR) — e.g., "income was left blank" might itself predict loan default risk, entirely separate from whatever number you end up imputing for the blank.
- Lets the model separate "the imputed value happens to be X" from "this was originally missing" — without this indicator, a tree-based model given a sentinel value can sort of infer this implicitly, but a linear model given mean-imputed data has literally no way to know which rows were imputed at all.
- Essentially free to add (one extra column per feature with missingness) and very commonly recommended as a default practice in production ML pipelines, not just an edge-case trick.

---

## 7. Special Case — Missingness in Tree-Based Models

Modern gradient boosting libraries (XGBoost, LightGBM, CatBoost) have **native missing value handling** — they don't require you to impute at all in many cases.

```
At each split, the tree learns the BEST DIRECTION to send missing values:

           [Feature X]
           /          \
      X < 5.0      X >= 5.0  OR  X is missing
      (go left)    (go right — the algorithm learned during training
                     that missing values behave more like the "right"
                     branch's population, based on where sending them
                     minimizes the loss function)
```

- The algorithm learns, per split, which direction (left or right) minimizes the loss when a missing value is encountered — effectively learning an optimal default routing for missingness, rather than you having to guess an imputation value upfront.
- This is a major practical reason gradient boosting is popular on real-world tabular data with messy missingness — often **no explicit imputation step is needed at all**, and manually imputing beforehand can sometimes even hurt performance by removing the signal the tree could have extracted from the missingness pattern itself.
- Still worth adding an explicit missingness indicator as an additional feature even here — it gives the tree an even more direct signal than relying solely on the implicit split-routing behavior.

---

## 8. Common Pitfalls (interviewers love probing these)

1. **Assuming MCAR by default without checking.** Most real-world missingness is MAR or MNAR, not MCAR — dropping rows or using simple imputation without at least attempting the diagnostic in §2 risks silently biasing the dataset.
2. **Fitting the imputer (mean, regression, k-NN, MICE) on the full dataset before splitting.** Leaks validation/test information into the imputation statistics — same leakage class as fitting a scaler or encoder on the full dataset (see Cross-Validation and Categorical Encoding notes).
3. **Forgetting that imputation shrinks variance.** Downstream statistics (correlations, feature importances, confidence intervals) computed on imputed data can look artificially more "confident" or "certain" than the data actually supports — a genuine statistical distortion, not just a cosmetic issue.
4. **Using single imputation and treating the result as if it were the true, certain data.** Single-imputation methods (mean, regression, k-NN) provide no way to express how uncertain each imputed value is — MICE's multiple-imputation-plus-pooling approach exists specifically to fix this, and skipping it silently overstates confidence in downstream analysis.
5. **Blindly imputing before checking if missingness itself is informative.** As in §6 — dropping the "was this missing" signal by imputing without an indicator can throw away real predictive information, especially under MAR/MNAR where missingness correlates with the outcome.
6. **Applying a numeric sentinel (like -999) without an indicator, for linear or distance-based models.** The model can interpret it as a real extreme value and badly distort scaling, coefficients, or distance calculations — sentinels are far safer for tree-based models that can isolate them via splits, and even then a missingness indicator is a good idea.
7. **Manually imputing before feeding data into a GBM that already has native missing-value handling.** Can remove exactly the missingness-pattern signal the tree would have otherwise exploited — check whether your model already handles missingness natively before reaching for manual imputation by default.
8. **Deleting rows/columns based on an arbitrary missingness percentage threshold without considering the feature's importance.** A feature with 60% missingness might still be highly predictive for the 40% where it IS present (worth keeping + indicator), while a feature with only 10% missingness that's otherwise useless might not be worth the imputation effort at all — the decision shouldn't be threshold-only.

---

## 9. FAANG-Level Interview Q&A

**Q1: You notice a "salary" feature is missing more often for higher-income job titles. What missingness mechanism is this, and how does it change your imputation strategy?**
This is likely MAR (or possibly MNAR, depending on the exact causal story) — missingness correlates with an *observed* variable (job title), which if fully accounted for, may explain the pattern (making it MAR), or it might correlate with the actual salary value itself even after conditioning on job title (which would make it MNAR — e.g., very high earners specifically declining to disclose regardless of title). Under MAR, a model-based imputation method (regression or MICE) that uses job title as a predictor can reasonably recover the missing values; if it's actually MNAR, no amount of clever imputation using observed features alone fully fixes the bias — you'd need to flag this explicitly, add a missingness indicator (since the missingness itself is informative), and communicate the residual bias risk to stakeholders rather than presenting the imputed values as unbiased.

**Q2: Why does mean imputation understate the true variance of a feature, and why does this matter for a downstream model?**
Every imputed row receives the exact same fixed value (the mean), creating an artificial spike in the feature's distribution at that single point and reducing its true observed spread — any correlation or regression coefficient computed using this feature will be distorted (typically attenuated/biased toward zero for that feature's true relationship with the target, since imputed rows contribute no real variance-driven signal, just noise around a fixed point). This matters because it can make a genuinely predictive feature look weaker than it is, or make downstream confidence intervals/statistical tests overstate certainty, since they don't know a chunk of the data was fabricated at a single value.

**Q3: A colleague suggests filling all missing numeric values with -999 and moving on. When is this reasonable, and when is it a serious mistake?**
Reasonable for tree-based models, which can isolate -999 via a split threshold and effectively treat it as a distinct "was missing" bucket without it corrupting other splits. Serious mistake for linear models, k-NN, SVMs with distance/scale-sensitive kernels, or neural nets without special handling — -999 gets treated as a real, extremely negative numeric value, badly distorting feature scaling, distance calculations, and learned coefficients, since the model has no way to know -999 means "missing" rather than "a real value of negative 999." Even for tree-based models, I'd still add an explicit missingness indicator column alongside the sentinel, since it's essentially free and removes any ambiguity.

**Q4: Explain why MICE is described as "multiple" imputation, and why that matters compared to just imputing once with a regression model.**
Single-pass regression (or mean, or k-NN) imputation produces one "confident" completed dataset with no representation of how uncertain each imputed value actually is. MICE instead generates several independently completed datasets (each with slightly different imputed values, reflecting the genuine uncertainty in the imputation process), runs the downstream analysis/model on each, and pools the results using established combining rules (e.g., averaging point estimates, combining within-imputation and between-imputation variance for confidence intervals) — this lets you report honestly wider (more realistic) uncertainty intervals that reflect imputation uncertainty, rather than falsely precise results that treat a single guessed value as ground truth.

**Q5: You're using XGBoost and have a feature with 30% missing values. Should you impute it before training?**
Not necessarily, and possibly not at all — XGBoost (and LightGBM, CatBoost) have native handling for missing values, learning at each split which direction (left/right) minimizes loss when a value is missing, which can capture genuinely useful missingness-pattern signal that manual imputation would erase. I'd first try leaving it as native-missing and compare validation performance against a manually-imputed version (with a missingness indicator added either way) — in many real cases the native handling performs as well or better, and skips an entire preprocessing step and its associated leakage risks.

**Q6: Two features have identical percentages of missing values, but you decide to keep one and drop the other. What's your reasoning?**
The missingness percentage alone doesn't determine whether a feature is worth keeping — what matters is (a) how predictive the feature is where it IS observed (a highly predictive feature with the same missingness rate is worth much more effort to retain/impute than a weakly predictive one), and (b) whether the missingness pattern itself carries signal (worth adding an indicator for) versus being pure noise/MCAR. I'd check feature importance/univariate association with the target on the observed subset for both features before deciding, rather than applying a blanket missingness-percentage cutoff to both equally.

**Q7: Your team imputed missing values using the entire dataset's mean (train + validation + test combined) before splitting for cross-validation. What's wrong with this, and what's the fix?**
This leaks information from the validation/test sets into the imputation statistic used to fill training data — the training set's imputed values are influenced by data it should never have seen, inflating cross-validated performance estimates in a way that won't hold up on truly unseen production data. This is the same leakage class as fitting a `StandardScaler` or a target encoder on the full dataset before splitting (see the Cross-Validation and Categorical Encoding notes) — the fix is to compute the imputation statistic (mean, regression model, k-NN reference set, MICE chains) using only each fold's training partition, then apply that fitted imputer to the corresponding validation/test partition, exactly mirroring how any other fold-dependent preprocessing step must be handled inside a proper CV pipeline.

**Q8 (clever): Can adding a missingness indicator ever leak information or otherwise cause a problem, even though it seems harmless?**
Generally low-risk, but one subtle issue: if the missingness mechanism in your training data is an artifact of a *data collection process bug* that gets fixed before production deployment (e.g., a temporary logging error that caused certain fields to be null during a specific time window, later patched), the model may learn to rely heavily on the "was_missing" indicator as a spurious signal correlated with that specific time period rather than anything causally meaningful — and then perform unpredictably once that missingness pattern disappears in production because the underlying bug was fixed. This is a good example of why understanding *why* data is missing (the mechanism, per §1) matters even for a technique as generally safe as adding an indicator column.

---

## 10. One-Line Interview Closers

- *"The question isn't 'how much is missing,' it's 'why is it missing' — MCAR, MAR, and MNAR call for genuinely different strategies, and treating them the same is where most real bias creeps in."*
- *"Missingness is often a feature, not just a problem to erase — an indicator column costs nothing and can capture signal that any single imputed value, however clever, throws away."*
- *"Single imputation gives you a confident-looking number with no idea how uncertain it really is — that's the entire reason MICE exists, and why I'd flag single-imputation results as a point estimate, not a fact."*
