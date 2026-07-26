# Phase 14: Feature Engineering for ML-Based Forecasting

Phases 1-13 built the classical statistical toolkit (ARIMA, ETS, GARCH, VAR...) where the MODEL ITSELF has a built-in notion of time and memory. Phase 15 (next) will introduce using general-purpose ML models (gradient boosting, etc.) for forecasting — but those models have NO built-in concept of time at all; they just see rows of numbers. This phase builds the bridge: **how do you turn a time series into a table of features an ordinary ML model can consume, while still preserving all the temporal information those classical models got "for free"?**

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $y_t$ | the target value we want to predict, at time $t$ |
| lag-$k$ feature | a column containing $y_{t-k}$ — the value from $k$ steps ago, placed as an ordinary predictor column |
| window | a stretch of consecutive past time points used to compute a rolling statistic |
| one-hot encoding | representing a category (like "Monday") as a column of 0s and 1s |
| cyclical encoding | representing a repeating category (like hour-of-day) using sine/cosine, explained fully in section 4 |

---

## 2. The core reframing: turning a time series into an ordinary supervised learning table

**Plain English, the single most important idea of this entire phase:** a generic ML model (like gradient boosting, which we'll fully cover in Phase 15) doesn't understand "time" as a concept at all — it just sees a table of rows and columns, and tries to predict one column (the target) from the others (the features), TREATING EACH ROW AS INDEPENDENT of the others — genuinely the exact same "rows don't care about each other" assumption from Phase 1, section 1, that we spent this entire course explaining WHY it's wrong for raw time series data!

**The resolution to this apparent contradiction: we don't feed the raw time series directly into the ML model. Instead, we MANUALLY construct features that ENCODE the relevant time-based information directly into each row, so that even though the ML model treats each row independently, each row ALREADY CONTAINS whatever temporal context it needs.** This is a genuinely important conceptual shift: **all the "memory" and "seasonality" that ARIMA/ETS built into their formulas (Phases 5-6) now has to be EXPLICITLY, MANUALLY engineered into columns of a table, since the downstream ML model has no other way to access it.**

**Concretely, what a converted table looks like:** instead of a single column of $y_t$ values in order, you build a table where EACH ROW corresponds to one time point $t$, with columns like "$y_{t-1}$" (yesterday's value), "$y_{t-7}$" (value from a week ago), "average of the last 7 days," "is this a Monday," "is this a holiday," and so on — and the TARGET column is $y_t$ itself. **Every single feature column in this table is really just a hand-built way of giving the model access to information a classical model like ARIMA or Holt-Winters would have used automatically, built into its recursive formula.**

---

## 3. Lag features and rolling statistics: directly encoding AR/MA-style memory

**Lag features — the direct table-based equivalent of AR(p), Phase 6, Part 1:** simply include $y_{t-1}, y_{t-2}, \ldots, y_{t-p}$ as ordinary predictor columns. **Plain English: this is EXACTLY what AR(p)'s formula does mathematically (today depends on the last $p$ values) — except instead of the model having this baked into its structure, we manually create these as separate columns, and let a flexible ML model (Phase 15) figure out on its own how strongly and in what way each lag matters (potentially discovering NON-linear relationships between lags, something plain AR(p)'s linear formula cannot do at all).**

**Rolling window statistics — a genuinely useful family of features that has no direct classical-model equivalent, built from Phase 1's moving average concept:** for each row/time point $t$, compute a summary statistic over a WINDOW of the most recent past points — e.g., "rolling 7-day mean" ($\frac{1}{7}\sum_{j=1}^{7}y_{t-j}$, directly reusing Phase 1, section 1's moving-average formula, just computed using ONLY past points rather than a centered window, since we can't use future data as a feature — a critical, genuinely important distinction from Phase 1's DESCRIPTIVE moving average, elaborated on in section 6 below), "rolling 7-day standard deviation" (a rolling measure of recent VOLATILITY — genuinely echoing the GARCH conditional-variance idea from Phase 10, just computed as a simple descriptive statistic rather than a fitted recursive model), "rolling max/min over the last 30 days," and so on.

**Expanding window statistics — a related but distinct idea:** instead of a FIXED-SIZE window, compute a statistic over ALL data from the start up through $t-1$ (a growing window, exactly analogous to the "expanding window" cross-validation variant from Phase 13, section 3) — e.g., "average of ALL historical data up to today," useful for capturing a genuinely stable, long-run baseline level that a short rolling window might not capture well.

---

## 4. Date/time features: cyclical encoding, derived carefully

**The naive, WRONG approach first (to build intuition for why the fix is needed):** you might think to just include "hour of day" as a plain number, 0 through 23. **The problem: this treats hour 23 (11pm) and hour 0 (midnight) as MAXIMALLY DIFFERENT (a raw numeric distance of 23), when in reality they are ADJACENT, only ONE HOUR APART on the actual clock.** A plain numeric encoding completely destroys the CYCLICAL, wrap-around nature of time-of-day (and similarly, day-of-week, month-of-year) — the ML model would incorrectly learn that hour 23 and hour 0 are as different as possible, when they're actually neighbors.

**The fix, directly reusing Phase 8's Fourier machinery:** encode a cyclical time feature (like hour-of-day, with period $m=24$) using BOTH a sine AND a cosine transform:
$$
\text{hour\_sin} = \sin\left(\frac{2\pi\cdot\text{hour}}{24}\right), \qquad \text{hour\_cos} = \cos\left(\frac{2\pi\cdot\text{hour}}{24}\right)
$$
**This is EXACTLY Phase 8, section 3's single Fourier wave-pair (the $k=1$ term), just now being used as an input FEATURE for a general ML model rather than as a regressor coefficient in a dedicated seasonal regression.** **Why does this fix the wraparound problem?** Because sine and cosine are themselves inherently CYCLICAL/periodic functions — plugging in hour=23 and hour=0 into these formulas produces two POINTS THAT ARE GENUINELY CLOSE TOGETHER on the resulting sin/cos "clock face" (mathematically, points on a circle), correctly reflecting their true adjacency on an actual 24-hour clock, unlike the naive raw-number encoding. **You need BOTH sine and cosine together (not just one alone) for exactly the same reason Phase 8, section 3 needed both** — a single sine value alone can't distinguish between two DIFFERENT times of day that happen to produce the same sine output (e.g., sine is the same value at both 3am and 9am in a simple 24-hour cycle, due to the symmetric shape of a sine wave) — having BOTH sine AND cosine together uniquely pins down exactly where you are in the cycle, with no ambiguity.

**The same exact technique applies to day-of-week ($m=7$), month-of-year ($m=12$), day-of-year ($m\approx365$), and any other cyclical calendar unit** — always the same sin/cos pair recipe, just changing the period $m$ each time.

---

## 5. Fourier features as regressors: directly importing Phase 8's whole toolkit

**A genuinely direct connection, worth stating explicitly:** everything from Phase 8, sections 3-4 (multiple Fourier wave-PAIRS, $K$ controlling flexibility, separate blocks for different seasonal periods added together) applies here EXACTLY as feature columns for an ML model, not just as regressors in a dedicated Prophet-style additive model. **You can include, say, $K=5$ pairs of yearly Fourier terms AND $K=3$ pairs of weekly Fourier terms as 16 total feature columns, handing the ML model rich, flexible seasonal information without it needing to "discover" the calendar structure from scratch out of a raw date field.**

---

## 6. Target encoding, holiday flags, and other business-context features

**Holiday/event flags:** exactly Phase 8, section 6.3's dummy/indicator regressors, directly reused as ML features — a column that's 1 on Christmas, 0 otherwise, and similar for other known important calendar events; you can also include "days until next holiday" or "days since last holiday" as genuinely useful continuous features capturing lead-up/afterglow effects.

**Target encoding (a new, ML-specific technique, briefly):** for a categorical feature with MANY possible values (e.g., "product category," with hundreds of distinct categories) — rather than creating hundreds of separate one-hot dummy columns (which can make the model unwieldy), **target encoding replaces each category with a single number: the historical AVERAGE of the target variable for that specific category** (e.g., replace "product category = electronics" directly with "the historical average sales level for the electronics category"). **A genuinely important nuance/trap here, directly connecting back to Phase 13's leakage warnings:** this average must be computed using ONLY data STRICTLY BEFORE the current row's time point (an expanding-window-style calculation, section 3) — computing it using the FULL dataset (including future data relative to that row) would be a direct instance of the exact same data leakage problem Phase 13 warned about, just showing up in FEATURE CONSTRUCTION rather than in cross-validation splitting.

---

## 7. Feature design for MULTIPLE related series (a brief preview, fully covered in Phase 15)

**A genuinely important practical scenario:** if you're forecasting THOUSANDS of related series at once (e.g., Apple forecasting App Store downloads for every single app, or Google forecasting ad revenue for every single advertiser) — rather than fitting thousands of SEPARATE classical models (one ARIMA per series, genuinely impractical at that scale), a common, powerful approach is a **"global" model**: one single ML model, trained on data from ALL series POOLED together, where "which series this row belongs to" becomes just another FEATURE (e.g., an identifier column, or better, features describing that series' general characteristics/category). **This lets the model learn SHARED patterns across all series simultaneously** (e.g., "app downloads generally spike after a marketing push, regardless of which specific app") **while still using series-specific features (recent lags, rolling stats for THAT particular series) to specialize the forecast for each individual case.** We'll fully develop this "global vs. local model" distinction, and why pooling often WINS at scale (a genuinely important, somewhat surprising finding from the M4/M5 forecasting competitions referenced in the original syllabus), in Phase 15.

---

## 8. THE critical trap: Lag Leakage (a direct, concrete instance of Phase 13's data-leakage principle)

**This is the single most important practical warning in this entire phase, and a genuinely common real interview "spot the bug" question.**

**The trap, concretely:** suppose you're building a rolling-7-day-average feature (section 3) to help predict $y_t$. **A subtle, easy-to-make mistake: accidentally including $y_t$ ITSELF (today's value, the very thing you're trying to predict) inside that rolling average window** — e.g., computing "average of days $t-6$ through $t$" (INCLUDING today, $t$) instead of the correct "average of days $t-7$ through $t-1$" (STRICTLY before today). **If you make this mistake, your feature literally contains information about the target you're trying to predict — during TRAINING, this can make your model look suspiciously, artificially good (since a feature that partially "contains" the answer will obviously help predict that answer very well) — but at actual DEPLOYMENT/production time, when you're trying to forecast a GENUINELY FUTURE, not-yet-observed value of $y_t$, you would NEVER actually have access to $y_t$ itself to compute that "rolling average including today" feature — the model would completely fail or need to be fed a nonsensical placeholder in production, despite having looked excellent during evaluation.**

**Why this is SPECIFICALLY dangerous, in a way that's easy to miss even for careful practitioners:** unlike the more obvious k-fold-CV leakage from Phase 13 (which is at least somewhat visible if you inspect your CV splitting code), **lag/feature leakage can hide silently INSIDE a seemingly innocent feature-engineering pipeline** — a rolling average function, a "days since last event" calculation, a target-encoding computation (section 6's exact warning) — any of these can ACCIDENTALLY include same-day or future information through a simple off-by-one indexing error, and the resulting model will often look GENUINELY EXCELLENT in your evaluation metrics (Phase 13), precisely because it's secretly cheating — making this a particularly INSIDIOUS trap, since good-looking metrics normally reassure you everything is fine, when here they're actually the symptom of the bug.

**The general defensive principle, worth memorizing as a checklist item for any real project: for EVERY single engineered feature, explicitly ask "would this exact same feature value have been genuinely, calculably AVAILABLE at the real point in time I'd need to make this forecast in actual production?"** If the honest answer is no (the feature secretly depends on data that wouldn't exist yet at prediction time), it's leakage, full stop, regardless of how it improves your evaluation metrics. **A good, concrete practical habit: build every lag/rolling feature using a STRICT "shift by at least 1 before computing" rule** — e.g., in pandas-flavored pseudocode, `df['y'].shift(1).rolling(7).mean()` (shift FIRST, ensuring you never include today, THEN compute the rolling window) rather than `df['y'].rolling(7).mean()` directly (which would include today's own value in the window — the exact bug just described).

---

## 9. A small numerical illustration of the lag-leakage trap, made concrete

Suppose daily sales: $y = [10, 12, 9, 15, 11, 14, 13, 20]$ (index $t=1$ through $8$; we want a 3-day rolling average feature to help predict $y_8=20$).

**The WRONG (leaky) way — includes today:** rolling mean of $t=6,7,8$ (days 6, 7, AND today, 8): $(14+13+20)/3 = 47/3\approx15.67$. **Notice this feature value DIRECTLY incorporates the actual value we're trying to predict ($y_8=20$) — an obvious, direct leak once you look closely, since 20 is literally one of the three numbers being averaged into its own predictive feature.**

**The CORRECT (properly shifted) way — strictly excludes today:** rolling mean of $t=5,6,7$ (the three days STRICTLY before today): $(11+14+13)/3=38/3\approx12.67$. **This feature value could genuinely have been computed at the real moment you needed to forecast day 8 — it uses ONLY information that would have actually existed at that point in time, with zero contamination from the very target being predicted.**

**The practical lesson, stated plainly: the leaky version (15.67) is suspiciously close to the true target of 20 PRECISELY BECAUSE it partially contains the target itself — a model trained on features built this way will show excellent-looking training/validation metrics that will NOT hold up even slightly once deployed on genuinely new, real future data where this shortcut is impossible.**

---

## 10. Quick self-check questions

1. Why does a generic ML model like gradient boosting need EXPLICIT lag/rolling features, when a classical ARIMA model doesn't need anything like that constructed manually?
   *(Answer: ARIMA's mathematical formula has memory/temporal dependence BUILT IN directly (today's value is defined in terms of past values via φ and θ coefficients); a generic ML model has no inherent concept of time or order at all, treating each row independently — so all the temporal/memory information has to be manually encoded into feature columns for the model to have any access to it whatsoever.)*
2. Why do you need BOTH sine and cosine (not just one) to properly encode a cyclical time feature like hour-of-day?
   *(Answer: a single sine (or cosine) value alone can correspond to two DIFFERENT points in the cycle (e.g., the same sine value can occur at two different hours due to the symmetric wave shape), so using only one function leaves genuine ambiguity about the actual position in the cycle; using both sine and cosine together uniquely and unambiguously pins down the exact point in the cycle.)*
3. Describe, in your own words, the exact mechanism by which a rolling-average feature can leak target information, and why this is especially dangerous compared to ordinary CV leakage.
   *(Answer: if a rolling window used to build a feature accidentally includes the current time point t (the value being predicted) rather than strictly stopping at t-1, the feature partially contains the answer itself; this is especially dangerous because it can hide inside seemingly ordinary feature-engineering code (an off-by-one indexing mistake) rather than being visible in an obvious CV-splitting step, and it produces suspiciously good-looking evaluation metrics that mask the underlying bug rather than revealing it.)*
4. What's the practical, concrete coding habit recommended to avoid lag/rolling-feature leakage?
   *(Answer: always explicitly shift the series by at least 1 step BEFORE computing any rolling/window statistic, ensuring the current time point t is never included in its own feature calculation — e.g., shift first, then apply the rolling window, rather than computing the rolling window directly on the unshifted series.)*

---

## What's next
Phase 15 covers **Machine Learning Models for Forecasting** — using the features you just learned to build gradient-boosted tree models (XGBoost/LightGBM) for forecasting, why tree-based models fundamentally struggle to EXTRAPOLATE a trend beyond the range they were trained on (a real, formula-level limitation worth understanding precisely, not just knowing as a rule of thumb), the "global vs. local model" distinction previewed in section 7 above, and hierarchical forecast reconciliation (MinT) for when you need forecasts at multiple levels — e.g., per-store AND total-company — to add up consistently.

Say "next" for Phase 15, or ask for more feature-engineering drilling first.
