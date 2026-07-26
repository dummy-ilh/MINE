# Phase 15: Machine Learning Models for Forecasting

Phase 14 built the bridge (features). This phase covers what happens once you actually plug those features into general-purpose ML models — specifically, a genuine structural weakness of tree-based models that catches people off guard, the "pool everything into one model" strategy that won the major forecasting competitions referenced in the original syllabus, and how to make forecasts at DIFFERENT levels of a hierarchy (e.g., per-store vs. company-total) mathematically consistent with each other.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| tree / decision tree | a model that predicts by asking a sequence of yes/no questions about feature values, ending in a predicted number at each "leaf" |
| split | a single yes/no question a tree uses to divide data (e.g., "is lag-1 value > 50?") |
| leaf | the final endpoint of a tree, where a specific predicted value is stored |
| gradient boosting | a technique that builds MANY small trees in sequence, each one correcting the errors of the ones before it |
| global model | ONE model trained on data pooled from MANY different series at once |
| local model | a SEPARATE model fit individually to each single series |
| reconciliation | the process of adjusting a set of forecasts (made at different hierarchy levels) so they add up consistently |

---

## 2. Quick refresher: what a decision tree actually predicts, and why that matters here

**Plain English, built from scratch:** a decision tree makes predictions by repeatedly splitting the data based on feature values (e.g., "if lag-1 > 50, go right; otherwise go left"), continuing to split smaller and smaller groups, until it reaches a "leaf" — a final small group of TRAINING data points, for which the tree simply predicts the AVERAGE of those training points' target values. **Gradient boosting** (the technique behind XGBoost/LightGBM, genuinely the dominant practical tool in this space) builds many small trees IN SEQUENCE, where each new tree specifically tries to correct the ERRORS left over by all the previous trees combined (directly echoing the "residuals as a diagnostic/target" philosophy from Phase 6, Part 5 — except here the residuals become the actual TARGET for the next tree to predict, rather than just a diagnostic to inspect).

**The single crucial fact this entire phase's core warning depends on: a tree's prediction is ALWAYS an average of some subset of TRAINING data values it has already seen. A tree can never, structurally, output a number LARGER than the maximum value in its training data, or SMALLER than the minimum value in its training data — no matter how it's built, no matter how many trees you use.**

---

## 3. Deriving, precisely, why tree-based models can't extrapolate a trend

**The concrete problem, worked through step by step:** suppose your training data (Phase 14's engineered feature table) covers a series that has been steadily GROWING — say, monthly revenue ranging from 100 up to 500 over the training period. Now you want to forecast several months into the FUTURE, where — if the real underlying trend continues — revenue should genuinely reach something like 600 or 700.

**What actually happens with a tree-based model, precisely:** the tree was trained using FEATURES like lag values and rolling averages (Phase 14) that, during training, never exceeded roughly 500 (since that was the max value seen in the training target). When you feed the model NEW future feature values that are similarly built from recent (also below-500) history, the tree's leaves can ONLY ever output values that were present in its training data's leaves — **meaning the very BEST a tree-based model can do, no matter how sophisticated, is predict something CLOSE TO the historical maximum it already saw (around 500) — it structurally CANNOT extrapolate upward to 600 or 700, even if that's genuinely where the real trend is heading.**

**Why this is a formula-level fact, not just an empirical rule of thumb:** recall from section 2 — a leaf's prediction is LITERALLY an average of specific training target values. **There is no mathematical mechanism ANYWHERE in a decision tree's construction that could ever produce an output value outside the range of values it was trained on** — averaging numbers can never produce a result outside the range of the numbers being averaged (a basic, provable arithmetic fact: the average of any set of numbers always falls between the minimum and maximum of that set). **This is a hard, structural ceiling/floor, fundamentally different from a linear regression or ARIMA model, where the FORMULA itself (e.g., $\beta_1 \times t$, a straight line, Phase 4 section 4.1) can be mechanically evaluated at ANY future point, producing outputs that genuinely extend beyond the historical range, because the underlying mathematical operation (multiplication, addition) has no such built-in ceiling.**

**The practical, genuinely important consequence and fix:** if your data has a real, ongoing trend, **you must explicitly DETREND the data first (Phase 4, section 5 — exactly the differencing/detrending toolkit from early in this course!), fit the tree-based model to the DETRENDED (residual) series, and then ADD BACK a separately-modeled trend component (e.g., a simple linear extrapolation, section 4.1's deterministic trend function) to the tree's forecast.** **This is a genuinely important, frequently-tested practical insight: tree-based models are excellent at capturing complex, non-linear SEASONAL and INTERACTION patterns, but they must be paired with an explicit trend-handling mechanism from OUTSIDE the tree framework entirely** — a real, common production pattern, and a strong, complete interview answer to "what's a limitation of using XGBoost for time series forecasting?"

---

## 4. Global models vs. local models: a genuinely important, somewhat counter-intuitive finding

**The traditional/classical approach (implicitly assumed throughout Phases 5-12): fit ONE SEPARATE model PER series** — a "local" model, tailored specifically to that one series' own history. If you have 10,000 different products to forecast, this means fitting 10,000 separate ARIMA models, each one only ever seeing that ONE product's own past data.

**The "global model" alternative: pool ALL 10,000 series' data together into ONE SINGLE large training table (directly building on Phase 14, section 7's brief preview), and fit just ONE model across everything, with "which series this is" (and characteristics describing it) included as features.**

**Why would pooling ever work BETTER than fitting a dedicated model per series? (the genuinely important, somewhat surprising insight, backed by real findings from the M4/M5 forecasting competitions referenced in the original syllabus):**

1. **More effective training data.** A single product's own history might only be, say, 24 months long — genuinely not much data for a flexible ML model (Phase 14's feature-rich approach) to learn complex patterns from reliably. **Pooling 10,000 products together gives the model access to 240,000 effective data points' worth of PATTERNS to learn from** — even though each individual row still only directly describes one specific product at one specific time, the model can learn GENERAL relationships (e.g., "how promotions typically affect demand," "what a typical post-holiday demand drop looks like") that transfer usefully across products, even for products with limited individual history.

2. **Better handling of "cold start" series** (a genuinely important practical problem): a BRAND NEW product with only 2 weeks of history has nowhere near enough data to fit a reliable dedicated local ARIMA model — but a GLOBAL model, having already learned general patterns from thousands of OTHER, similar, more established products, can make a genuinely reasonable forecast for the new product from day one, by leveraging the shared, pooled structure.

3. **A single model to maintain, rather than 10,000.** A genuinely important, practical, non-statistical advantage: in production, retraining, monitoring, and debugging ONE global model is vastly more manageable than maintaining thousands of separate individually-fit models (directly foreshadowing the "system design for forecasting at scale" considerations from later in the original syllabus).

**The honest trade-off, worth stating clearly (not every finding favors global models unconditionally):** a global model can sometimes UNDER-fit any single series' own genuinely UNIQUE, idiosyncratic behavior (since it's trying to find patterns that generalize ACROSS many series, it may smooth over something truly specific to just one series) — **the practical, real-world answer, and the actual finding from the M4/M5 competitions, is that a HYBRID/ensemble approach (blending global and local model predictions together, section 6 below) often performs BEST of all, capturing both the shared, general patterns (from the global model) and the series-specific nuances (from local models) simultaneously.**

---

## 5. Hierarchical & grouped time series: the reconciliation problem

**A genuinely important, distinct practical problem, common at any real company with a natural organizational hierarchy:** suppose Apple wants forecasts for iPhone sales at THREE levels simultaneously: (a) total company-wide iPhone sales, (b) sales broken down by REGION (US, Europe, Asia...), and (c) sales broken down further by INDIVIDUAL STORE within each region. **If you fit a SEPARATE, independent forecasting model at EACH level (total, region, store) — say, using ARIMA for each — there's absolutely no mathematical guarantee that the individual store-level forecasts will actually ADD UP to match the region-level forecast, or that the region-level forecasts will add up to match the total company-wide forecast.** Each model was fit completely independently, with zero awareness of the others, so their individual outputs can (and typically will) be mutually inconsistent.

**Why does this inconsistency actually matter practically, beyond just looking untidy?** Because different parts of a real business genuinely rely on forecasts at DIFFERENT levels — a regional manager needs the region-level number, a specific store manager needs the store-level number, and company leadership needs the total — **and if these numbers don't actually add up to each other, it creates genuine confusion, conflicting plans, and erodes trust in the forecasting system entirely.** This genuinely common, practical problem is called **the reconciliation problem**, and it has several named solution approaches:

**Bottom-up:** forecast ONLY at the most granular level (individual stores), then simply SUM those forecasts upward to get region and total-level numbers automatically (guaranteeing consistency by construction, since the totals are LITERALLY computed by adding up the detailed forecasts). **The weakness:** the most granular, store-level data is often the NOISIEST, hardest-to-forecast-accurately level (small individual stores can have genuinely volatile, hard-to-predict day-to-day patterns) — so this approach inherits all that granular-level noise/inaccuracy, even in the AGGREGATED totals, which might otherwise have been much easier to forecast accurately directly (aggregate/total-level patterns are typically smoother and more stable, since individual random fluctuations partially cancel out when summed together — directly echoing the Central-Limit-Theorem-flavored intuition that sums of many individually-noisy things tend to be smoother/more stable than any single individual component).

**Top-down:** forecast ONLY the total company-wide number (typically the smoothest, easiest-to-forecast-accurately level, per the reasoning just given), then DISAGGREGATE that total DOWN to regions and stores using some fixed historical PROPORTION (e.g., "Store A has historically represented 2% of total sales, so give Store A 2% of the forecasted total"). **The weakness:** this approach completely fails to capture any genuine, real STORE-SPECIFIC dynamics (e.g., if Store A is genuinely growing much faster than the company average, the fixed historical proportion will systematically under-forecast it, since the top-down approach has no mechanism to let individual proportions shift over time).

**MinT (Trace Minimization) — the more sophisticated, modern, statistically-optimal solution:** **the genuinely elegant idea: fit SEPARATE forecasts at EVERY level FIRST (total, regions, stores — exactly as if you were going to use them independently, capturing whatever unique information exists at each level), then apply a mathematical RECONCILIATION ADJUSTMENT afterward that nudges ALL the forecasts (at every level simultaneously) toward the CLOSEST possible mutually-consistent set of numbers** — using a weighted-least-squares-style optimization (directly related in spirit to the OLS/MLE machinery from Phase 6, Part 4, and Phase 7) that specifically weighs adjustments according to how RELIABLE/precise each level's ORIGINAL independent forecast was estimated to be (levels with historically more accurate/stable forecasts get adjusted LESS; levels with historically noisier, less reliable forecasts get adjusted MORE) — genuinely using ALL the information from every level of the hierarchy simultaneously, rather than picking just one level (bottom or top) as authoritative and mechanically deriving the rest. **You don't need the full matrix-optimization derivation for interview purposes — the practical, complete answer is: "MinT reconciles forecasts by optimally combining information from every level of the hierarchy at once, minimizing the total adjustment needed while guaranteeing the final forecasts genuinely add up consistently — a real improvement over the more naive, information-discarding bottom-up or top-down approaches, since it lets EVERY level contribute useful signal rather than picking one level as the sole source of truth."**

---

## 6. Ensembling and stacking: briefly, tying section 4's hybrid idea together formally

**New term: ensemble.** Plain English: instead of picking ONE single "best" model, COMBINE the predictions of SEVERAL different models together (e.g., blend an ARIMA forecast, a global gradient-boosting forecast, and a Prophet forecast) — typically by a simple weighted average, or sometimes with weights themselves learned from a secondary model (called **stacking** — literally training a simple secondary model whose ONLY job is to learn the best way to COMBINE the predictions of the other, "base" models, using their individual outputs as ITS inputs/features). **Why does this often work better than any single model alone, intuitively?** Different models tend to make DIFFERENT KINDS of errors (e.g., ARIMA might be excellent at capturing genuine short-term momentum/autocorrelation but poor at complex non-linear seasonal interactions; a tree-based global model might excel at exactly the opposite) — **averaging/combining models whose errors aren't perfectly correlated with each other tends to CANCEL OUT some of each individual model's specific mistakes**, producing a combined forecast that's often more accurate and more ROBUST than any single contributing model alone — a genuinely common, practical, real production technique, and precisely the mechanism behind section 4's "hybrid global+local" recommendation from the M4/M5 competition findings.

---

## 7. A small numerical illustration: the tree extrapolation problem, made concrete

Suppose a tiny training set (trend-only, no other features, for simplicity) where "lag-1 value" is the only feature, and the tree has learned exactly ONE split rule from this data: "if lag-1 $\leq$ 300, predict 250; if lag-1 $>$ 300, predict 480" (these two numbers, 250 and 480, are literally just averages of whatever training target values fell into each of those two groups, per section 2).

**Training data ranged up to a maximum observed value of, say, 500.** Now suppose the REAL underlying trend continues, and next month's true lag-1 input value is actually 550 (genuinely beyond anything the tree has ever seen). **Since $550 > 300$, the tree follows the SAME "predict 480" branch it would use for ANY input above 300** — whether the input is 301, 500, or 5000, **the tree's prediction is STUCK at exactly 480 in every single case, completely unable to distinguish between a mild trend continuation and an enormous one, because it has literally no leaf ANYWHERE in its structure corresponding to values it never saw during training.** **This concretely demonstrates section 3's derived fact: the tree's predictions are hard-capped by the range of values (480, in this specific tiny example) it happened to see during training, regardless of how far the true future input actually extends beyond that range.**

---

## 8. Quick self-check questions

1. Derive, from the basic mechanics of how a tree makes predictions, precisely why it cannot extrapolate beyond its training range.
   *(Answer: a tree's prediction at any leaf is literally the average of the training target values that fell into that leaf; since the average of any set of numbers must fall between the minimum and maximum of that set, no leaf can ever produce a value outside the range of the training targets that built it — there is no mathematical mechanism in the tree's construction that could produce an out-of-range output.)*
2. What is the standard, practical fix for using a tree-based model on data with a genuine ongoing trend?
   *(Answer: explicitly detrend the data first (e.g., via differencing, Phase 4), fit the tree-based model to the detrended/residual series to capture seasonal and non-linear patterns, then add back a separately, explicitly modeled trend component (such as a simple linear extrapolation) to produce the final forecast.)*
3. Give two genuine, concrete reasons why a global model (pooling many series together) can outperform fitting a separate local model per series.
   *(Answer: (1) pooling gives the model access to far more effective training data/patterns to learn from, even though each row still describes just one series at one time, letting it learn generalizable relationships that transfer across series; (2) it handles "cold start" series with very little individual history far better, since it can leverage patterns already learned from other, more established, similar series.)*
4. Why does bottom-up hierarchical reconciliation tend to produce noisier aggregate forecasts than forecasting the aggregate directly, while top-down reconciliation tends to miss genuine store-specific dynamics?
   *(Answer: bottom-up sums up granular, individually noisy store-level forecasts, and that noise carries through into the totals even though the true aggregate pattern is often smoother/easier to forecast directly (since individual random fluctuations partially cancel out when summed); top-down instead disaggregates a smooth total forecast using FIXED historical proportions, which cannot capture genuine shifts in any individual store's relative growth or decline over time.)*

---

## What's next
Phase 16 is the largest single phase in the full syllabus: **Deep Learning for Time Series** — building up from why we need RNNs at all, through the full gate-by-gate derivation of LSTM and GRU (with the same "define every symbol, derive every formula" treatment you've had throughout this course), attention mechanisms, and modern architectures (TFT, N-BEATS, DeepAR) — the material most directly relevant to cutting-edge forecasting research and increasingly common in senior-level interviews at companies like Google and Apple.

Say "next" for Phase 16 (we'll break it into well-paced sub-parts, the same way we handled Phase 6), or ask for more drilling on tree extrapolation / global models / reconciliation first.
