# Phase 13: Time Series Cross-Validation & Evaluation Metrics

You now know how to BUILD a huge range of models (Phases 1-12). This phase answers the practical question every interview eventually asks: **how do you honestly judge whether a model is actually good, and how do you fairly compare two candidate models?** This is where a genuinely common, career-relevant mistake lives — using ordinary cross-validation on time series data — so we build the correct alternative carefully.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $y_t$ | the actual observed value at time $t$ |
| $\hat{y}_t$ | the forecasted/predicted value at time $t$ |
| $e_t$ | the forecast error at time $t$, $e_t = y_t - \hat{y}_t$ |
| $n$ | number of points being evaluated |
| $\lvert \cdot \rvert$ | absolute value — strips away the sign, keeps only the magnitude |
| $q$ | the target quantile in pinball loss (e.g., 0.9 for the 90th percentile) |
| $\mathbb{1}[\cdot]$ | the **indicator function** — equals 1 if the condition inside is true, 0 otherwise (a new, simple notational tool used in section 7) |

---

## 2. Why ordinary k-fold cross-validation is actually WRONG for time series

**Quick refresher on ordinary k-fold CV (general ML background):** you'd normally split your data into $k$ chunks, and repeatedly train on $k-1$ chunks while testing on the 1 held-out chunk, rotating which chunk is held out — this works great when rows are independent of each other (Phase 1, section 1's original distinction between time series and ordinary cross-sectional data).

**Why this breaks for time series, precisely:** ordinary k-fold CV would happily let you train on data from, say, March through December, and then TEST on January and February — **meaning you'd be using FUTURE data to predict the PAST.** This is called **data leakage**, and it's a genuinely serious, real practical error: your model gets to "see the future" during training in a way it never could in real deployment, making your evaluation metrics look artificially good — sometimes dramatically so — compared to how the model would actually perform making genuine forward-looking predictions in production. **The core principle that must never be violated: a forecasting model must only ever be evaluated using training data that occurred STRICTLY BEFORE the data it's being tested on** — respecting the fundamental "order matters" property of time series that has been the foundation of this entire course since Phase 1, section 1.

---

## 3. The correct alternative: Rolling-Origin (a.k.a. "walk-forward") Cross-Validation

**The core idea, in plain English:** instead of randomly shuffling chunks, you slide a "cutoff point" (the **origin**) FORWARD through time, always training on everything BEFORE the cutoff and testing on a small window immediately AFTER it — then moving the cutoff forward and repeating.

**Concretely, step by step:**
1. **Fold 1:** Train on data from the start through, say, month 12. Forecast month 13. Record the error.
2. **Fold 2:** Train on data from the start through month 13 (one more month of REAL data now included). Forecast month 14. Record the error.
3. **Fold 3:** Train on data from the start through month 14. Forecast month 15. Record the error.
4. ...continue sliding the origin forward, one step at a time, through as much of your dataset as you have.
5. **Final evaluation:** average the errors across all these folds (using one of the metrics from sections 4-7 below) to get an overall honest performance estimate.

**Two variants, worth distinguishing by name:**
- **Expanding window:** the TRAINING SET keeps GROWING with each fold (fold 2 uses everything fold 1 used, PLUS one more point) — as described above. This mimics a realistic production scenario where you retrain periodically on ALL available history.
- **Sliding/rolling window:** the training set stays a FIXED SIZE, but slides forward (e.g., always exactly the most recent 12 months) — dropping old data as new data comes in, rather than accumulating everything forever. This is useful when you specifically suspect older data may no longer be representative (e.g., after a genuine structural change in the business) and want your evaluation to reflect that.

**Why this is the honest, correct approach, tying back to Phase 6, Part 5:** this exactly mimics ACTUAL deployment — in real life, when you stand at some point in time and make a forecast, you genuinely only have access to data up through today, never data from the future. **Rolling-origin CV simply repeats this exact real, honest situation many times across your historical dataset, to get a robust, averaged sense of how well the model would have genuinely performed if deployed repeatedly over that period** — rather than the fundamentally dishonest, leakage-prone scenario ordinary k-fold CV would create.

---

## 4. Blocked and Purged Cross-Validation: a further refinement (especially relevant in finance)

**The problem rolling-origin alone doesn't fully solve:** even respecting strict time order, there can still be a SUBTLER leakage problem if your FEATURES (Phase 8's feature-engineering concepts, foreshadowing later phases) include things like rolling averages or lagged values that span ACROSS the train/test boundary — e.g., if your test point's feature is "average of the last 7 days," and some of those 7 days fall right at the edge of what was in the training set, there can be subtle overlap/contamination even in an otherwise properly time-ordered split.

**The fix — an "embargo" period:** **blocked/purged cross-validation** (a technique genuinely emphasized in quantitative finance, where this kind of subtle leakage is a serious, real, costly risk) adds a small GAP — an embargo period — between the end of the training window and the start of the test window, specifically to ensure no feature computed near the boundary can "peek" across it. **Plain English: leave a small buffer of unused time between train and test, just to be extra safe that no information genuinely leaks across the boundary through overlapping feature windows.** You don't need to derive this further — just recognize the name and the practical motivation (protecting against SUBTLE leakage through engineered features, beyond the more obvious kind rolling-origin CV already prevents).

---

## 5. Scale-dependent metrics: MAE, MSE, RMSE — built from scratch

**Recall $e_t = y_t - \hat{y}_t$** (the forecast error at time $t$, directly reusing the exact same idea as Phase 6, Part 5, section 2's residuals — just now specifically computed on held-out TEST data from the CV procedure above, rather than in-sample training residuals).

**Mean Absolute Error (MAE):**
$$
\text{MAE} = \frac{1}{n}\sum_{t=1}^n |e_t|
$$
**Plain English:** "on average, how far off (in absolute terms, ignoring direction) were our forecasts?" Simple, directly interpretable in the SAME UNITS as your original data (e.g., "on average we're off by 340 units").

**Mean Squared Error (MSE):**
$$
\text{MSE} = \frac{1}{n}\sum_{t=1}^n e_t^2
$$
**Plain English: same idea, but SQUARING instead of taking absolute value** (exactly the same "guarantee a positive number, but also PENALIZE large errors disproportionately more" logic you saw with Ljung-Box back in Phase 3, section 8, and with the likelihood function's sum-of-squared-errors connection in Phase 6, Part 4, section 4). **A single huge miss contributes MUCH more to MSE than to MAE** (since squaring a big number makes it disproportionately bigger) — meaning MSE is a metric that specifically cares more about avoiding LARGE, occasional errors, even at the cost of slightly worse typical/average performance, whereas MAE treats every unit of error equally regardless of size.

**Root Mean Squared Error (RMSE):** simply $\sqrt{\text{MSE}}$ — **the whole point of taking this square root is purely to bring the metric BACK into the same original units as your data** (since MSE, from squaring, is in SQUARED units — e.g., "dollars-squared," which isn't directly interpretable) — RMSE is genuinely just "MSE, made interpretable again," while still keeping MSE's "punish large errors extra hard" character.

**Which to choose, MAE or RMSE? A genuine, practical, defensible answer:** if occasional LARGE errors are especially costly/dangerous in your specific business context (e.g., a large under-forecast causing a stockout, or a large over-forecast causing expensive waste), RMSE's extra sensitivity to big misses is the more appropriate choice. If you want a metric that's more ROBUST to a few outlier bad forecasts, and treats typical performance fairly regardless of occasional large misses, MAE is more appropriate. **This is a genuinely common interview question with a real, substantive answer — not just "they're both fine."**

---

## 6. Percentage-based metrics: MAPE and its serious flaw, sMAPE

**Mean Absolute Percentage Error (MAPE):**
$$
\text{MAPE} = \frac{100\%}{n}\sum_{t=1}^n \left|\frac{e_t}{y_t}\right|
$$
**Plain English: instead of measuring the error in raw units, measure it as a PERCENTAGE of the actual value** — a genuinely appealing idea at first glance, because it's scale-free (you can compare MAPE across totally different series, e.g., comparing forecast accuracy for a $10 product against a $10,000 product on an equal footing, something raw MAE/RMSE can't do).

**The serious, well-known flaw — DERIVED directly from the formula, not just asserted:** look at the DENOMINATOR — $y_t$, the actual observed value. **If $y_t$ is close to zero (or exactly zero), this fraction explodes toward infinity (or is literally undefined, dividing by zero) — REGARDLESS of how small the actual error $e_t$ was.** Concretely: if actual sales were $y_t=1$ unit and you forecasted $\hat{y}_t=3$ units, your error is only 2 units in absolute terms — but your percentage error is a wild 200%! **This makes MAPE genuinely unusable/misleading for any series that has values near zero or that can be zero (a very real, common situation — e.g., daily sales of a slow-moving product, which might genuinely be 0 on many days) — a small handful of near-zero actual values can completely dominate and distort the average MAPE score, even if the model's ABSOLUTE performance was perfectly reasonable everywhere else.**

**A second, more subtle flaw worth knowing (asymmetry):** MAPE penalizes OVER-forecasting and UNDER-forecasting differently, even for equally-sized absolute errors — e.g., if $y_t=10$ and you either forecast $\hat y_t=15$ (over by 5, giving 50% error) or $\hat y_t=5$ (under by 5, giving also exactly 50% error in THIS specific symmetric case) — actually try a genuinely asymmetric example: if $y_t=10$, forecasting $\hat y_t=20$ (over by 10) gives 100% MAPE, but forecasting $\hat y_t=0$ (under by 10, the maximum possible under-forecast since you can't go below zero forecasts for typically-positive quantities) can only ever reach 100% MAPE too in THIS case — but more generally, because the denominator is always the ACTUAL value (never the forecast), MAPE structurally caps how much an UNDER-forecast can be penalized (bounded by 100% when forecasting all the way down to zero) while OVER-forecasts can produce arbitrarily large percentage errors with no such ceiling — a genuine, structural asymmetry.

**Symmetric MAPE (sMAPE) — the attempted fix:**
$$
\text{sMAPE} = \frac{100\%}{n}\sum_{t=1}^n \frac{|e_t|}{(|y_t|+|\hat y_t|)/2}
$$
**Plain English: instead of dividing by the actual value ALONE, divide by the AVERAGE of the actual and forecasted values** — this addresses the over/under asymmetry problem somewhat (both directions are now measured against a shared, symmetric baseline) but **does NOT fully fix the near-zero-denominator explosion problem** (if BOTH $y_t$ and $\hat y_t$ happen to be near zero simultaneously, the denominator is still near zero) — sMAPE is a genuine improvement over MAPE, but not a complete cure, a nuance worth knowing rather than assuming sMAPE fully "solves" the problem.

---

## 7. MASE: Hyndman's preferred metric — built from scratch, with full rationale

**The motivating question this metric answers, that NONE of the previous metrics can: "is my forecast actually BETTER than the simplest, laziest possible forecasting approach?"** This is a genuinely important, practical framing — a model's raw MAE or RMSE number, on its own, tells you nothing about whether that performance is actually GOOD relative to a trivial baseline, or whether a naive, effortless approach would have done just as well or better.

**Step 1 — define the simplest possible baseline: the "naive forecast."** For NON-seasonal data, the naive forecast for tomorrow is simply "whatever happened today" ($\hat{y}_t = y_{t-1}$ — genuinely the laziest possible forecasting method, requiring zero modeling effort at all). For SEASONAL data, the natural naive baseline is "whatever happened at this same point in the last cycle" ($\hat y_t = y_{t-m}$, directly reusing the seasonal-period notation $m$ from Phase 5/8).

**Step 2 — compute the average ABSOLUTE ERROR this trivial naive baseline would have achieved, on the TRAINING data:**
$$
\text{scale} = \frac{1}{n-1}\sum_{t=2}^{n}|y_t - y_{t-1}|
$$
(For non-seasonal data — this is literally just "the average size of period-to-period changes in the historical data," a genuinely simple, intuitive quantity.)

**Step 3 — the MASE formula: scale YOUR model's error by this naive-baseline error:**
$$
\text{MASE} = \frac{\text{MAE of your model's forecasts}}{\text{scale (MAE of the naive baseline)}} = \frac{\frac{1}{n}\sum|e_t|}{\frac{1}{n-1}\sum|y_t-y_{t-1}|}
$$

**Plain English, the complete, genuinely satisfying interpretation: MASE tells you, directly, how many TIMES better (or worse) your model is, compared to just guessing "tomorrow will be the same as today."**
- **MASE $< 1$**: your model is BETTER than the naive baseline (e.g., MASE $=0.7$ means your model's typical error is only 70% the size of the naive approach's typical error — genuinely valuable, your modeling effort is paying off).
- **MASE $= 1$**: your model is EXACTLY as good as just guessing "same as last time" — a genuinely important, humbling benchmark; if your sophisticated ARIMA/GARCH/whatever model can't beat MASE $=1$, all that modeling effort added zero real value.
- **MASE $> 1$**: your model is WORSE than the trivial naive baseline — a serious red flag, genuinely embarrassing for a deployed model, and a real, common check that catches over-engineered models that actually underperform doing nothing clever at all.

**Why this fixes MAPE's near-zero-denominator problem (directly, structurally):** the DENOMINATOR here is the naive baseline's AVERAGE absolute error across the WHOLE series — a single, stable, well-behaved number computed from MANY data points — NOT a single, potentially-near-zero individual $y_t$ value the way MAPE's denominator was. **This structural difference is exactly why MASE doesn't blow up or become undefined for series containing values near zero — a genuinely well-engineered fix, not a coincidence.** This is precisely why Hyndman (a co-author of the FPP3 textbook referenced throughout this syllabus) specifically recommends MASE as the generally preferred metric for comparing forecast accuracy ACROSS different series of different scales, addressing MAPE's core flaw directly while still being scale-independent (comparable across series) — a genuinely well-reasoned, complete interview answer if asked "what's wrong with MAPE, and what would you use instead."

---

## 8. Pinball Loss (Quantile Loss): evaluating PROBABILISTIC forecasts, not just point forecasts

**Motivation, connecting back to Phase 6, Part 5, section 5's prediction intervals:** everything above evaluates a single POINT forecast. But recall from Phase 6 that a responsible forecast often comes with a full RANGE/interval, not just one number. **How do you score whether a predicted QUANTILE (e.g., "I predict there's a 90% chance the actual value will be below X") was actually good?**

**Building the formula, piece by piece, for a target quantile $q$ (e.g., $q=0.9$ for the 90th percentile forecast):**
$$
L_q(y,\hat y_q) = \begin{cases} q\,(y - \hat y_q) & \text{if } y \geq \hat y_q \\ (1-q)\,(\hat y_q - y) & \text{if } y < \hat y_q \end{cases}
$$
**Plain English, working through the intuition carefully:** Suppose $q=0.9$ (we're trying to forecast a value such that we EXPECT the actual outcome to fall BELOW it 90% of the time, and ABOVE it only 10% of the time — an intentionally cautious, "high" quantile forecast). **If the actual value $y$ comes in ABOVE our forecast $\hat y_q$** (our high forecast wasn't high enough — a genuine miss, and specifically the LESS common, more surprising direction of miss since we expected only 10% of outcomes to land here) **— the loss formula applies the LARGER weight, $q=0.9$, to this error** — plain English: **being caught under-forecasting on a quantile you specifically intended to be conservative/high about is penalized HEAVILY.** Conversely, **if the actual value falls BELOW our forecast** (the expected, common 90%-of-the-time case) **— the SMALLER weight $(1-q)=0.1$ applies** — a genuinely small penalty, since this was the anticipated, "on target" outcome.

**Why "pinball" is a fitting name (a fun, memorizable visual):** if you plot this loss function against different possible actual outcomes $y$, it forms a V-shape with two differently-SLOPED sides (steeper on one side, shallower on the other, with the exact slopes determined by $q$ and $1-q$) — resembling the angled walls of a pinball machine, asymmetrically directing the "ball" (the loss penalty) more sharply on one side than the other, depending on which quantile you're targeting.

**Genuinely important special case, connecting back to something you already know:** **at exactly $q=0.5$ (the median), the pinball loss becomes PERFECTLY SYMMETRIC** ($q=1-q=0.5$ on both sides) — **and at this specific value, pinball loss reduces to exactly (half of) the ordinary Mean Absolute Error from section 5!** This is a genuinely satisfying unifying fact: **MAE is just the special, symmetric $q=0.5$ case of the more general pinball loss framework** — evaluating the MEDIAN forecast is the same as evaluating the 50th percentile, with equal penalty for over- and under-shooting, exactly matching plain MAE's symmetric treatment of errors.

---

## 9. CRPS: briefly, the natural extension across ALL quantiles at once

**New term, briefly: CRPS (Continuous Ranked Probability Score).** Plain English, without full derivation: instead of evaluating just ONE specific quantile (like pinball loss does for a single $q$), **CRPS essentially integrates/averages pinball loss ACROSS EVERY POSSIBLE quantile from 0 to 1 simultaneously** — giving a single, comprehensive score for how good an ENTIRE predicted probability distribution was (not just one particular percentile of it), genuinely useful when your model outputs a FULL distribution/density forecast (e.g., from a Bayesian approach like BSTS, previewed back in Phase 9) rather than just a couple of specific named quantiles. You don't need the full integral derivation for interview purposes — just recognize CRPS as "pinball loss, generalized/averaged across the WHOLE distribution rather than one single quantile," and know it's the standard metric for evaluating complete probabilistic/distributional forecasts.

---

## 10. Prediction Intervals vs. Confidence Intervals: a precise, often-blurred distinction

**A genuinely common point of sloppy language, worth being precise about (a real interview clarity check):**

**A confidence interval** quantifies uncertainty about a fixed, unknown PARAMETER (e.g., "we are 95% confident the TRUE value of $\phi$ in our AR(1) model lies between 0.55 and 0.65" — Phase 6, Part 4's estimation uncertainty). **The thing being estimated is a single, fixed (if unknown) number — the uncertainty is entirely about our ESTIMATION process, not about any future randomness.**

**A prediction interval** (exactly what you built in Phase 6, Part 5, section 5) quantifies uncertainty about a FUTURE, YET-TO-BE-OBSERVED, genuinely RANDOM outcome (e.g., "we're 95% confident tomorrow's ACTUAL sales will fall between 340 and 410 units"). **This uncertainty comes from TWO separate sources combined: (1) our own estimation uncertainty about the model's parameters (the confidence-interval-flavored piece), PLUS (2) the GENUINE, irreducible randomness of the future outcome itself (the process noise, Phase 2's white noise, which no amount of additional data could ever fully eliminate).** **This is precisely why prediction intervals are ALWAYS WIDER than the corresponding confidence interval for the same underlying parameter** — prediction intervals must account for BOTH sources of uncertainty simultaneously, while confidence intervals only account for the first. **A crisp, complete, interview-ready way to state this distinction: "a confidence interval tells you how uncertain you are about a parameter; a prediction interval tells you how uncertain you are about a future outcome — and since a future outcome carries genuine additional randomness on top of parameter uncertainty, prediction intervals are always at least as wide, usually strictly wider."**

---

## 11. Quick self-check questions

1. Precisely why does ordinary k-fold cross-validation cause data leakage when applied to time series?
   *(Answer: k-fold CV randomly assigns data points to folds without respecting time order, meaning a model could end up being trained on data from later in time and tested on data from earlier in time — effectively letting the model "see the future" during training, which could never happen in genuine real-world deployment, producing overly optimistic, dishonest performance estimates.)*
2. Derive, from the MAPE formula, exactly why it becomes unreliable for series containing values near zero.
   *(Answer: MAPE divides each error by the actual observed value y_t; as y_t approaches zero, this fraction's denominator shrinks toward zero, causing the percentage error to explode toward infinity (or become undefined at exactly zero) regardless of how small the raw absolute error actually was — a small number of near-zero actual values can dominate and distort the entire average.)*
3. If a fitted model achieves MASE = 1.3, what does this concretely tell you, and is it good or bad news?
   *(Answer: it means the model's typical forecast error is 1.3 times LARGER than simply guessing "the same as last period" (or "same as last seasonal cycle" for seasonal data) — this is bad news, since a MASE above 1 means the model is actually performing WORSE than the simplest possible naive baseline, despite presumably requiring far more modeling effort.)*
4. Why does pinball loss at q=0.5 reduce to (half of) ordinary MAE, and what does this reveal about the relationship between the two metrics?
   *(Answer: at q=0.5, both branches of the pinball loss formula get an equal weight of 0.5, applied symmetrically regardless of whether the actual value falls above or below the forecast — exactly matching MAE's symmetric treatment of over- and under-forecasts; this reveals that MAE is really just a special case of the more general pinball/quantile loss framework, specifically the median (50th percentile) case.)*

---

## What's next
Phase 14 moves into practical **feature engineering for machine-learning-based forecasting** — lag features, rolling statistics, cyclical date encodings, and a genuinely important, very-commonly-tested trap: lag leakage (accidentally letting a feature "see" information from the future during training, a specific, concrete instance of the exact data-leakage principle this phase just established in the cross-validation context, now showing up again in feature construction instead).

Say "next" for Phase 14, or ask for more drilling on evaluation metrics first (e.g., working a full numerical example computing MAE, RMSE, MAPE, and MASE side-by-side on the same small dataset, to see directly how differently they can rank the same set of forecasts).
