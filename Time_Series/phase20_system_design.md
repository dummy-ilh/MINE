# Phase 20: System Design for Forecasting at Scale

This phase is different in character from everything before it: less "derive a formula," more "assemble everything you know into a coherent architecture." This is the material most directly aimed at a "design a forecasting system for [X]" interview question — genuinely common at Google/Apple-level system design rounds when the domain happens to be forecasting-flavored. Every piece below is something you've already built; this phase is about ASSEMBLY and TRADE-OFFS.

---

## 1. The full pipeline, end to end: a mental map

**Plain English overview before the details:** a real, production forecasting system is not just "a model." It's a PIPELINE: **data ingestion → feature store → training → serving → monitoring**, each stage with its own genuine engineering and statistical considerations, and each stage connecting to specific things you've learned throughout this course.

$$
\text{Raw data} \to \text{Feature engineering (Phase 14)} \to \text{Model training (Phases 5,6,10,15,16)} \to \text{Serving forecasts} \to \text{Monitoring \& retraining}
$$

We'll walk through each stage, and at each one, the genuinely important QUESTION an interviewer is testing you on.

---

## 2. Stage 1 — Data Ingestion: the unglamorous, genuinely critical foundation

**The core question: how does raw, real-world data (sales transactions, ad clicks, sensor readings) actually get INTO a form usable for forecasting?** **Genuinely important considerations, each connecting to something you already know:** (1) **handling missing/irregular timestamps** (Phase 1, section 1.7) — real production data streams have gaps, late-arriving data, and duplicate records, all needing resolution before ANY modeling can begin; (2) **aggregation level decisions** — do you ingest and store data at the finest possible granularity (individual transactions) or pre-aggregate (hourly/daily totals)? **This directly connects to Phase 15, section 5's hierarchical reconciliation discussion — storing at the finest granularity gives maximum flexibility (you can always aggregate UP later, but you can't recover detail that was thrown away), at the cost of significantly more storage and processing overhead.**

---

## 3. Stage 2 — Feature Store: making Phase 14's features REUSABLE and CONSISTENT

**The core, genuinely important production problem this solves: Phase 14 taught you HOW to build features (lags, rolling stats, Fourier terms) — but at PRODUCTION SCALE, with potentially THOUSANDS of models all needing similar features, you don't want every single model recomputing the SAME rolling averages from scratch, independently, with subtle inconsistencies creeping in between different implementations.** **A feature store is a centralized system that computes and STORES these engineered features ONCE, making them available consistently to ANY model that needs them** — a genuinely important practical concept, directly motivated by Phase 14's feature engineering material, just addressing the SCALE and CONSISTENCY problem rather than the "how do you compute a Fourier feature" problem.

**The single most important, genuinely critical correctness issue here, directly connecting back to Phase 14, section 8's lag leakage warning: a feature store MUST guarantee "point-in-time correctness"** — plain English, **when TRAINING a model using historical data, the feature store must serve EXACTLY the feature values that WOULD HAVE BEEN available at that historical moment in time, not values that incorporate information that only became available LATER** — **this is PRECISELY Phase 14's lag-leakage trap, now showing up as a genuine, serious INFRASTRUCTURE design requirement, not just a coding discipline for a single script.** A poorly-designed feature store can systematically leak future information into historical training data at MASSIVE scale, silently corrupting every single model trained against it — a genuinely severe, real production risk.

---

## 4. Stage 3 — Model Training: revisiting global vs. local at TRUE production scale

**Directly extending Phase 15, section 4's global-vs-local discussion, now at the scale an interviewer actually cares about: "forecast demand for every single SKU Apple sells, across every single store, worldwide" — potentially millions of individual series.** **Fitting millions of SEPARATE classical models (ARIMA, Phase 6) is often genuinely, practically INFEASIBLE at this scale** (even if each individual fit takes just 1 second, a million fits is over 11 days of pure sequential compute time — a real, concrete constraint, not a theoretical inconvenience) **— this is EXACTLY why the global-model strategy (Phase 15, section 4, and DeepAR, Phase 16, Part 5, section 5) becomes not just statistically appealing, but practically, operationally NECESSARY at true scale: ONE model (or a small, manageable handful of specialized global models, perhaps segmented by broad product category) trained ONCE on pooled data, versus millions of individually-maintained artifacts.**

**A genuinely important, nuanced middle ground worth knowing, beyond pure "global vs local": SEGMENTED global models** — plain English, **rather than ONE single global model for literally everything, or millions of fully separate local models, train a SMALL NUMBER of global models, each one specializing in a MEANINGFULLY DIFFERENT segment** (e.g., one global model for "high-volume, stable" products, a separate one for "new/sparse/cold-start" products, Phase 14 section 6/Phase 15 section 4's cold-start discussion) **— balancing the STATISTICAL benefits of pooling (more effective data, shared pattern-learning) against the reality that genuinely DIFFERENT kinds of products may have genuinely different underlying dynamics that a single, undifferentiated global model might blur together.**

---

## 5. Stage 4 — Serving: batch vs. real-time, a genuine, concrete trade-off

**The core question: once a model is trained, HOW do actual forecasts get delivered to whoever needs them?**

**Batch forecasting:** generate forecasts for ALL series at once, on a SCHEDULE (e.g., once per night, forecasting the next 30 days for every single product) — **plain English, genuinely simple, computationally efficient (can process everything together, taking advantage of parallelization), but forecasts can become STALE between batch runs** (if something significant happens at 9am, your forecast won't reflect it until the NEXT scheduled batch run, potentially many hours later).

**Real-time/on-demand forecasting:** generate a fresh forecast the MOMENT it's requested, using the LATEST available data — **plain English, always current, but genuinely more computationally expensive and architecturally complex to serve at low latency, especially for models with expensive inference (e.g., a large LSTM/Transformer from Phase 16, versus a comparatively cheap ARIMA coefficient lookup from Phase 6).**

**The genuinely practical, real answer most production systems actually use: a HYBRID.** **Batch-compute the bulk of the forecast (e.g., the underlying trend/seasonal structure, which genuinely doesn't change minute-to-minute) on a regular schedule, but layer a LIGHTWEIGHT, fast real-time ADJUSTMENT on top when serving** (e.g., directly analogous, in spirit, to the Kalman filter's predict-then-update loop, Phase 9, section 4 — the batch forecast is the "predict" step's output, and a lightweight real-time correction, using the very latest incoming data, plays the "update" step's role, correcting the stale batch prediction with fresh information, without needing to rerun the ENTIRE expensive model from scratch).

---

## 6. Stage 5 — Monitoring: drift detection, directly reusing Phase 17's entire toolkit

**The core, genuinely important production question: how do you know when a DEPLOYED model has stopped being good, and needs to be retrained or replaced?** **This is PRECISELY Phase 17's change point detection material (section 7), just applied to MODEL PERFORMANCE metrics (like rolling MAE/RMSE, Phase 13) over time, rather than to the raw business series itself** — **plain English: monitor the model's OWN forecast errors as a time series in their own right, and apply CUSUM (Phase 17, section 4) or a control chart (Phase 17, section 2) DIRECTLY to that error series, specifically to detect when the model's error behavior has genuinely, persistently SHIFTED (a sign the underlying data-generating process has changed — a genuine "concept drift" — and the model, trained on now-outdated patterns, needs retraining).**

**"Retraining cadence" — a genuinely practical question with a real trade-off, not just "retrain constantly":** **retraining too INFREQUENTLY risks the model going stale, missing genuine, real shifts in behavior (directly the drift-detection concern above). Retraining too FREQUENTLY is computationally expensive, and — a genuinely subtle, important point — can cause the model to overreact to ORDINARY statistical noise (Phase 2's white noise) as if it were a genuine, meaningful pattern shift, introducing needless INSTABILITY into your forecasts from one retraining cycle to the next.** **The practical, defensible answer: retrain on a REGULAR schedule appropriate to how fast the underlying business genuinely changes (e.g., weekly for a fast-moving ad-market business, quarterly for a slow-moving industrial-equipment business), SUPPLEMENTED by drift-triggered EARLY retraining specifically when monitoring (this section's CUSUM-on-errors approach) flags a genuine, statistically-significant shift outside of the regular schedule.**

---

## 7. Latency/scale trade-offs: connecting model choice back to the whole course

**A genuinely useful, complete interview framing: different models from across this ENTIRE course have VERY different computational costs at INFERENCE (forecast-generation) time, and system design requires matching the model choice to the actual latency/scale requirements of the specific use case:**

| Model family | Typical inference cost | Best-suited scenario |
|---|---|---|
| ETS/Holt-Winters (Phase 5) | Extremely cheap (a few arithmetic operations) | Millions of simple series, tight latency budgets |
| ARIMA/SARIMA (Phase 6) | Cheap-moderate (fixed formula, small number of coefficients) | Moderate-scale, need for interpretability/diagnostics |
| Gradient boosting (Phase 15) | Moderate (tree traversal, fast but not trivial) | Global model across many series, rich feature sets |
| LSTM/GRU (Phase 16, Parts 1-3) | Higher (sequential computation through the network) | Complex non-linear patterns, moderate scale |
| Transformer-based (Phase 16, Part 5) | Highest (self-attention scales with sequence length, though parallelizable) | Very long sequences, when accuracy gains justify the cost |

**The genuinely important, complete interview answer this table supports: "the choice of model isn't just about which one is theoretically most accurate — it's about matching computational cost to the actual production constraints (how many series, how often forecasts are needed, what latency is acceptable) — a simple ETS model serving millions of low-stakes forecasts in real time is often the RIGHT engineering choice, even if a Transformer might squeeze out marginally better accuracy on paper, precisely because the operational cost/benefit trade-off favors the cheaper model at that scale."**

---

## 8. A worked example: designing a system for "forecast App Store daily downloads per app"

**Walking through the full pipeline for this concrete, realistic Apple-flavored interview scenario, tying every stage together:**

**Ingestion:** raw download events, aggregated to daily counts per app (Stage 1) — genuinely need to handle apps with sparse/irregular download patterns (many apps have very few downloads on most days — directly recalling Phase 13, section 6's MAPE-near-zero warning: any evaluation metric used here must be chosen carefully given how common near-zero daily download counts will be, favoring MASE, Phase 13 section 7, over MAPE).

**Feature store:** lag features, rolling stats, calendar/holiday features (Phase 14), app-category identifiers (for the global-model strategy, Stage 4) — with strict point-in-time correctness (Stage 3's critical requirement).

**Model training:** given millions of apps, a GLOBAL model (Stage 4) is the practical choice — likely gradient boosting (Phase 15) or DeepAR (Phase 16, Part 5, section 5), specifically because DeepAR's NATIVE probabilistic output and its NATIVE global-pooling design (built specifically for exactly this "many related series, some with very little individual history" scenario) is a genuinely strong fit — new apps (cold start, Phase 15 section 4) benefit enormously from a global model's shared, pooled learning.

**Serving:** likely a HYBRID (Stage 5) — nightly batch forecasts for the bulk of apps, since download patterns don't typically shift minute-to-minute, with no need for expensive real-time serving for most apps (a low-latency-requirement use case, per section 7's table, favoring batch economics over real-time cost).

**Monitoring:** track rolling MASE (Phase 13, section 7) per app-category segment, with CUSUM-based drift detection (Phase 17, section 4, this phase's section 6) specifically flagging segments whose error behavior has genuinely, persistently shifted — e.g., a sudden change in Apple's App Store featuring/promotion algorithm could cause a genuine, detectable shift across many apps simultaneously, exactly the kind of SUSTAINED, SMALL, hard-to-spot-in-any-single-series shift CUSUM is specifically designed to catch.

**This worked example is worth being able to reproduce and adapt on the fly** — the specific technology choices matter less than demonstrating you can walk through EVERY stage of the pipeline, citing the RIGHT considerations at each stage, and making DEFENSIBLE, reasoned trade-off decisions rather than arbitrary ones.

---

## 9. Quick self-check questions

1. What is "point-in-time correctness" in a feature store, and what earlier-phase concept is it directly protecting against, now at infrastructure scale?
   *(Answer: point-in-time correctness means the feature store must serve exactly the feature values that would genuinely have been available at a given historical moment, never values incorporating information that only became available later; this directly protects against Phase 14's lag-leakage trap, now as a serious infrastructure design requirement rather than just a single script's coding discipline — a flaw here can silently corrupt training data at massive scale.)*
2. Why does the global-vs-local model choice become not just statistically appealing but practically NECESSARY at true production scale (e.g., millions of series)?
   *(Answer: fitting millions of separate individual models is often computationally infeasible in practice (even fast per-model fitting times add up to enormous total compute at that scale), making a global model (or a small number of segmented global models) the only practically operable choice, on top of its genuine statistical benefits like better cold-start handling.)*
3. Describe the hybrid batch/real-time serving strategy, and what earlier-phase concept its structure directly resembles.
   *(Answer: batch-compute the bulk of the forecast (e.g., trend/seasonal structure) on a regular schedule, then apply a lightweight, fast real-time adjustment on top using the latest incoming data when actually serving a forecast, rather than rerunning the full expensive model from scratch each time; this directly resembles the Kalman filter's predict-then-update loop (Phase 9) — the batch forecast plays the "predict" role, and the real-time adjustment plays the "update" role.)*
4. Why is retraining too FREQUENTLY, not just too infrequently, also a genuine problem?
   *(Answer: retraining too frequently is computationally expensive, and can cause the model to overreact to ordinary statistical noise (white noise, Phase 2) as though it were a genuine, meaningful shift in the underlying pattern, introducing unnecessary instability in forecasts from one retraining cycle to the next, rather than only updating in response to genuine, statistically-confirmed drift.)*

---

## What's next
Phase 21 moves into a dedicated **Interview Drill Bank** — rapid-fire conceptual Q&A, a set of "derive this formula on the whiteboard" prompts pulling from across every phase of this course, product-sense/case-study framings (Apple device-usage forecasting, Google ads-revenue forecasting), and a coding-round simulation for implementing and backtesting a model under time pressure — the consolidation phase that pulls everything together into interview-ready form.

Say "next" for Phase 21, or ask for more system-design drilling first — e.g., working through a different worked example end-to-end yourself (like "design a system to forecast Google Ads revenue by advertiser"), with me reviewing your reasoning against this phase's framework.
