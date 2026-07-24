# Data Leakage — Interview Notes

## 1. Core Definition

**Data leakage** occurs when information that would not be legitimately available at prediction time is used during model training or evaluation — causing the model to look far better offline than it will ever perform in production.

The defining test: **"Would this exact piece of information be available, in this exact form, at the moment I actually need to make the prediction in production?"** If no, it's leakage.

Leakage is insidious because it doesn't cause an error — it causes a *silently optimistic* metric. Train and validation error both look great; the model ships; production performance collapses. This makes it more dangerous than most bugs, which tend to announce themselves.

## 2. Taxonomy of Leakage

### A. Target Leakage (a.k.a. feature leakage)
A feature is, directly or indirectly, a proxy for or a consequence of the label — information that only exists *because* the outcome already happened.

- Classic example: predicting "will this patient be readmitted" using a feature `discharged_to_hospice` — that field is often only populated *after* the event you're trying to predict, or is causally downstream of it.
- Subtler example: predicting loan default using `account_closed_reason = "charged off"` — the reason code is only assigned once default has already occurred.
- Even subtler: a feature that's technically available before the label but was *generated using* the label (e.g., a "risk score" computed by another internal system that itself consumed outcomes from the same events you're now predicting).

### B. Train-Test Contamination
Information from the test/validation set leaks into training, so the held-out set is no longer a clean, independent measure of generalization.

- **Preprocessing before splitting:** fitting a scaler, imputer, PCA, or feature selector on the *full* dataset, then splitting — statistics like mean/std/min-max now encode test-set information.
- **Duplicate or near-duplicate rows** across train/test (common in scraped or augmented datasets) — the model effectively memorizes test answers.
- **Improper cross-validation with grouped data:** e.g., multiple rows per patient/user/session split across folds — the model learns patient-specific patterns in train and "predicts" the same patient in val. Needs `GroupKFold` (group by patient) instead of plain `KFold`.
- **Target encoding computed globally:** replacing a categorical value with its mean target value, computed using the *entire* dataset including validation rows.
- **Oversampling/SMOTE before splitting:** synthetic minority samples generated using neighbors that end up split across train and validation, so validation rows share nearly-identical synthetic neighbors with train rows.

### C. Temporal Leakage
Using future information to predict the past — the single most common leakage type in real production systems with time-series or event-based data.

- Random (non-time-aware) train/test split on time-ordered data — the model trains on data from *after* the point it's being validated against.
- A feature computed with a rolling window that accidentally includes the current or future row (off-by-one in window boundaries).
- **Feature computed at "pull time" instead of "as of event time":** e.g., using a user's *current* total lifetime purchases as a feature for a churn event that happened 6 months ago — at that historical point, the user hadn't made those later purchases yet.
- Backtesting a trading strategy where an indicator uses same-day closing price to predict same-day movement.

### D. Pipeline / Group Leakage
Related rows share information that shouldn't cross the train/val boundary, even without an explicit time or preprocessing bug.

- Multiple images of the same object/person appearing in both train and test (common failure in face recognition and medical imaging datasets — e.g., different scan slices of the *same patient* split across sets).
- Sensor data windows from the same continuous recording session split across train/test — adjacent windows are highly autocorrelated, so "different" rows aren't really independent.
- Deduplication failures in NLP datasets — same document or near-paraphrase in both train and test corpora (a known issue in some early LLM benchmark evaluations).

### E. "Leakage via the label-generation process itself"
Sometimes the *way the dataset was constructed* encodes leakage independent of any single feature.

- **Survivorship-conditioned sampling:** only including examples where the outcome is already known filters the population in a way correlated with the label (e.g., a "fraud detection" dataset that only contains transactions that were investigated — and investigation itself was triggered by suspicion of fraud).
- **Human-in-the-loop label leakage:** labels assigned by annotators who had access to information that won't exist at inference time (e.g., annotators labeling "spam" while seeing which account got banned).

## 3. Why It's Dangerous (Interviewer Angle)

- It **inflates offline metrics** — AUC of 0.99 that becomes 0.65 in production is a leakage red flag, not a modeling win.
- It's **asymmetric with normal bugs**: most bugs degrade performance and get caught early; leakage *improves* performance, so nobody questions it until it's shipped and the damage is visible in production monitoring — often weeks later.
- It **compounds with model complexity**: more flexible models (GBMs, deep nets) are better at *exploiting* a leaky feature than simple models — so leakage often masquerades as "wow, XGBoost crushed logistic regression," when actually XGBoost just found and exploited the leak harder.
- It can **survive a correct train/val split** if the leak is inside feature engineering that runs before the split (see §2B) — a naive "did I split correctly?" check isn't sufficient.

## 4. Detection

**Primary signal: suspiciously high performance.**
- Near-perfect AUC/accuracy on a genuinely hard, noisy real-world problem (not synthetic/toy data) is the single strongest tell. If a domain expert would expect 75% accuracy and you're getting 99%, be suspicious before celebrating.

**Feature importance / SHAP audit:**
- If one or two features dominate feature importance far beyond what's plausible domain-knowledge-wise, investigate what that feature actually *is* and *when* it's populated.
- Ask: "does this feature's value ever change *after* the label is known, and did I capture it before or after?"

**Timestamp audit (for temporal data):**
- For every feature, explicitly write down: "as-of timestamp" vs. "label event timestamp." Any feature whose as-of timestamp is ≥ the label timestamp is leakage, full stop.
- Check window boundaries for off-by-one errors (`<=` vs `<` at the cutoff).

**Single-feature ablation:**
- Drop the top-N suspicious features one at a time and watch for a cliff in performance — a feature whose removal causes an outsized, disproportionate drop (versus a smooth graceful degradation) is a leakage candidate.

**Train/test distribution comparison:**
- Compare feature distributions between train and test. Near-identical distributions where you'd expect some drift (e.g., over time) can indicate contamination.
- Check for literal duplicate rows (hash rows and count exact and near-duplicate matches across splits).

**Ask a domain expert "could you know this at prediction time?"**
- The most reliable check is often not statistical at all — it's asking someone who understands the business process to walk through each field and confirm availability timing. This catches subtle target leakage that no automated check will find, because the leak is *semantic*, not distributional.

**Held-out "deployment simulation":**
- Reproduce the exact production feature-fetching pipeline for a handful of historical rows and compare feature values against what training used. Any mismatch = leakage or a train/serve skew bug (closely related failure mode, see §7).

## 5. Prevention

**Split first, transform second — always.**
- Fit scalers, imputers, encoders, feature selectors, and PCA on the training fold only, then `.transform()` (never `.fit_transform()`) the validation/test fold. This is the single most common source of leakage and the easiest to prevent mechanically.
- Use `sklearn.Pipeline` (or equivalent) so fit/transform ordering is enforced automatically instead of manually — manual preprocessing before `train_test_split` is the classic beginner mistake.

**Time-aware splitting for time-series/event data.**
- Use a strict chronological split (train on past, validate on future) or `TimeSeriesSplit`-style expanding/rolling windows. Never shuffle time-ordered data into a random split.
- Enforce a **point-in-time correctness** discipline: every feature must be computable using only information available strictly before the label's event time. This is often implemented via a formal "feature store" with as-of joins.

**Group-aware splitting for grouped/clustered data.**
- Use `GroupKFold` / `GroupShuffleSplit` keyed on patient ID, user ID, session ID, document ID, etc., so no group straddles the train/val boundary.

**Target encoding done safely.**
- Compute target-mean encodings using only training-fold data (or with proper out-of-fold / K-fold target encoding schemes, and smoothing/regularization to avoid overfitting to rare categories).

**Remove/mask post-outcome fields explicitly.**
- Maintain an explicit "banned features" list for known post-outcome fields, enforced by feature-store schema or a lint check in the training pipeline — don't rely on manual review alone.

**Feature timestamping discipline.**
- Every feature in a feature store should carry an explicit "valid-as-of" timestamp so training pipelines can programmatically enforce point-in-time joins rather than relying on developers remembering by hand.

**Deduplicate before splitting.**
- Hash/fingerprint rows (or embeddings for near-duplicate detection in text/image data) and remove duplicates *before* any split, not after.

## 6. Common Pitfalls / Trick Angles (Interviewers Love These)

1. **"I split before preprocessing, so I'm safe."** Not necessarily — if the *feature engineering* (e.g., aggregating a user's historical stats) itself pulls from the full dataset regardless of the later split, the split happening "first" in code doesn't matter if the aggregation logic ignores split boundaries.

2. **"Cross-validation protects me from leakage."** CV protects against overfitting to a *specific split*, not against leakage baked into features before the fold assignment. If every fold has the same leaky feature, every fold looks great — CV doesn't detect this class of leakage at all.

3. **"High feature importance for a plausible-sounding feature is fine."** Plausibility is not proof. A feature with a plausible business name can still be populated *after* the event (e.g., `days_since_last_contact` might be computed relative to *today* rather than relative to the label's timestamp).

4. **Leakage via feature selection on the full dataset.** Selecting the "top K correlated features" using the entire dataset (including validation) before splitting leaks label information into feature selection itself — even if the final model is trained cleanly afterward.

5. **Leakage in ensemble/stacking pipelines.** If a stacked meta-model is trained on out-of-fold predictions but the base models were tuned (early stopping, hyperparameter search) using the *same* folds later used for meta-features, information leaks between layers.

6. **Assuming "no leakage" because train/test accuracy gap is small.** A *small* gap doesn't rule out leakage — if the leak is present in both train and test identically (e.g., a global preprocessing statistic), both numbers are inflated together and the gap looks healthy while the absolute numbers are both wrong.

7. **Synthetic/augmented data leakage.** Data augmentation (image rotations, SMOTE, back-translation) applied before splitting means augmented "copies" of a test-set row can end up in train, or vice versa — near-identical rows across the split.

8. **Believing leakage is only a tabular-data problem.** It shows up constantly in NLP (train/test corpus overlap in web-scraped pretraining data), computer vision (same subject across splits), and RL (evaluation episodes sharing environment seeds/state with training episodes).

9. **"The model's cross-validated score matches production for the first month, so we're fine."** Some leakage (e.g., slowly-drifting global statistics) only shows its effect once the *distribution itself* shifts post-launch — a clean initial launch doesn't retroactively prove leakage-free features.

## 7. Related but Distinct Failure Mode: Train/Serve Skew

Worth distinguishing explicitly in an interview, since they're often confused:

- **Data leakage:** training-time information includes something that *shouldn't exist yet* at prediction time (an information-availability problem).
- **Train/serve skew:** the *same* feature is computed differently (different code path, different library version, different data source) in the offline training pipeline vs. the online serving pipeline — a *consistency* problem, not an information-availability problem. This causes the opposite symptom pattern: training/offline metrics look fine, but production metrics are bad from day one (not a slow drift) because the online feature values simply don't match what the model was trained on.
- Both produce a "great offline, bad production" story, so the diagnostic first step is the same (compare offline-computed feature values against online-computed feature values for identical historical inputs) — but the *fix* differs: leakage is fixed by removing/re-timing a feature; skew is fixed by unifying the feature-computation code path (e.g., a shared feature store used by both training and serving).

## 8. Diagnosis Table (Interview-Style Quick Reference)

| Symptom | Likely Cause | Check |
|---|---|---|
| Near-perfect offline metric on a hard real-world problem | Target leakage | Audit top features for post-outcome timing |
| Great CV score, bad on true holdout / production | Train-test contamination (fit-before-split, group leakage) | Re-run pipeline with fit/transform strictly inside each fold |
| Great backtest, bad live trading/forecasting | Temporal leakage | Point-in-time audit of every feature's as-of timestamp |
| One feature dominates importance implausibly | Target leakage on that single feature | Ablate it; check its population timing |
| Offline metrics fine, production bad from day one (not drifting) | Train/serve skew (not leakage) | Compare online vs. offline computed feature values on identical rows |
| Performance degrades slowly over weeks post-launch | Likely genuine concept drift, not leakage | Monitor feature/label distribution drift over time |

## 9. Interview Q&A — Conceptual and Trick Questions

**Q1: Your model gets 0.98 AUC on a churn prediction problem where the business tells you even their best analysts only reliably beat a coin flip by a little. What's your first move?**
Don't celebrate — investigate. Rank feature importances, and for the top few features ask "could this literally not have existed at prediction time?" This is almost always target leakage before it's a genuine modeling breakthrough.

**Q2: You used `StandardScaler().fit_transform(X)` on the whole dataset before calling `train_test_split`. Is this leakage, and does it matter much in practice?**
Yes, it's leakage — the scaler's mean/std now include test-set statistics. In practice, for large datasets with i.i.d. rows and no distribution shift, the *magnitude* of the leak is often small (means/stds barely change with a few thousand fewer rows) — but it's still methodologically wrong, and the leak magnitude grows sharply for small datasets, skewed features, or if you're doing something more information-dense than a simple scaler (e.g., PCA, feature selection, or target encoding).

**Q3: Does k-fold cross-validation protect against target leakage?**
No — see pitfall #2 above. CV protects against split-specific overfitting/variance in your error *estimate*, not against a feature that's leaky in every fold identically. Leakage-via-features look the same across all folds, so CV will happily report a consistent, confidently wrong, optimistic number.

**Q4: You're doing time-series forecasting and used `sklearn.train_test_split(shuffle=True)`. Why is this a serious problem, more so than a "normal" random dataset?**
Because rows are not independent in time — nearby timestamps are autocorrelated, and a random shuffle puts future rows in train and past rows in test, meaning the model can "peek" at the future when predicting the past. This isn't just a minor variance-inflation issue like it might be for i.i.d. tabular data — it can make your metric almost meaningless for the actual deployed use case (forecasting forward in time from "now").

**Q5: A hospital dataset gives near-perfect readmission-prediction accuracy. Later you learn the dataset only includes patients who were flagged for post-discharge review. Is this leakage?**
Yes — this is leakage via the label-generation/sampling process (§2E), not a single bad feature. The population itself was filtered using information correlated with the outcome (only high-risk patients got flagged for review), so the training distribution doesn't represent the true deployment population, and any model trained on it will be miscalibrated once applied to the general patient population.

**Q6: True or false: if train accuracy and test accuracy are both around 70% and close to each other, you can rule out leakage.**
False — a small train/test gap only rules out *variance-flavored* overfitting; it says nothing about leakage that inflates both numbers *together* (e.g., a globally-fit preprocessing statistic, or a feature leaked identically into both splits). A small, healthy-looking gap can still be built entirely on top of a leaky foundation.

**Q7: What's the difference between data leakage and train/serve skew, and why does it matter which one you diagnose?**
See §7. They produce similar "great offline, bad production" symptoms but require different fixes — leakage needs the feature removed or re-timed; skew needs the online and offline feature computation unified (typically via a shared feature store). Misdiagnosing one as the other wastes an incident-response cycle chasing the wrong fix.

**Q8 (clever): Can a model have *no* leaky individual feature, yet still suffer from leakage overall?**
Yes — via group leakage (§2D) or contamination in feature *selection*/*engineering* steps rather than any single raw feature. E.g., PCA components fit on the full dataset are leaky even though no single original column looks suspicious; the leak lives in the transformation, not in a column you'd eyeball.

**Q9: Why do more powerful/flexible models sometimes make leakage *harder* to notice rather than easier?**
Because they're better at extracting maximal signal from whatever's available — including a leaky feature — so the metric improvement from adding that feature looks like a genuine, satisfying capability jump ("XGBoost found signal linear regression missed!") rather than raising suspicion. A simpler model might barely use the leaky feature, making the leak's contribution to the metric less visually obvious.

**Q10: You added a "top-K correlated features" selection step before your train/test split, then trained cleanly afterward on the selected features only. Is this leaky?**
Yes — the selection step itself used label information from what later becomes your test set to decide which features to keep, so even though the *model fitting* afterward is clean, the *feature set itself* was chosen with test-set leakage baked in. Feature selection must live inside the same fit-on-train-only discipline as scaling/encoding.
