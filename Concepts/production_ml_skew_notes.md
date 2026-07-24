# Production ML & Skew — Interview Notes

## 1. Training-Serving Skew — Summary (see dedicated RCA notes for full depth)

**Definition:** the same logical feature is computed differently offline (training) vs. online (serving), so the model receives different inputs at inference time than it was trained on — a *consistency* bug, not an information-availability problem (data leakage) and not a real-world-changed problem (distribution shift/concept drift).

**Signature:** production performance is bad **from the moment of deploy**, not gradually decaying — the sharpest differentiator from drift, which typically shows a decaying trend from an initially-good state.

**Fastest diagnostic:** pull real production requests, recompute their features using the actual offline training-pipeline code on the same raw inputs, and diff against what was actually served — turns a vague "something's wrong" into a concrete feature-by-feature checklist in one step.

**Common root causes:** duplicated/divergent code paths (offline Python/Spark vs. online Java/Go re-implementation of the same logic), library/dependency version mismatches, feature freshness/point-in-time mismatches, missing-value/default-handling differences, un-reused preprocessing artifacts (scaler/encoder refit instead of loaded), join/lookup timing differences, silent schema/type coercion.

**Durable fix:** unify the code path (shared feature-computation logic, ideally a feature store) rather than patching the specific observed numeric discrepancy — patching symptoms without unifying the path lets the next pipeline change reintroduce divergence. See the dedicated skew RCA notes for the full step-by-step workflow, root-cause taxonomy, and Q&A.

## 2. Feature Stores

### What problem they solve
Two independent pain points that compound each other in production ML:
1. **Train/serve skew** (§1) — different code computing "the same" feature in two places.
2. **Feature reuse and duplication of effort** — without a central system, every team/model reinvents feature engineering for commonly-needed signals (e.g., "user's 30-day purchase count"), leading to redundant computation, inconsistent definitions of the "same" business concept across teams, and wasted engineering effort.

### Core architecture — the offline/online split
A feature store is built around **one feature definition, computed once, served two ways**:
- **Offline store (batch layer):** stores historical feature values at scale, optimized for large, point-in-time-correct joins used to assemble training datasets — typically backed by a data warehouse/lake (e.g., a columnar store).
- **Online store (serving layer):** stores the *current* value of each feature, optimized for low-latency single-key lookups at inference time — typically backed by a key-value store (e.g., Redis, DynamoDB) since prediction-time requests need millisecond-scale reads for a single entity (one user, one transaction), not analytical scans.
- **The critical design property:** both layers are populated by the **same feature-computation pipeline/definition**, so the value materialized into the online store for "user 123's 30-day purchase count right now" is guaranteed (by construction, not by discipline) to be computed the same way as the historical values used to build training data — this is what structurally eliminates skew rather than relying on two teams staying manually in sync.

### Point-in-time correctness
A feature store's offline layer typically supports **as-of joins** — when assembling a training dataset, for each labeled historical event, the join pulls the feature value *as it existed at that event's timestamp*, not the current/latest value. This directly prevents the temporal-leakage failure mode (using future information to predict the past) and the "fresher than training ever saw" skew variant covered in the RCA notes — both are structurally hard to get right without this capability built into the store itself.

### Key components/concepts worth naming precisely
- **Feature definition / transformation logic:** the single source of truth for how a feature is computed from raw data — registered once, referenced by both batch and online materialization jobs.
- **Materialization:** the process of computing feature values and writing them into the offline and/or online stores on a schedule (batch) or continuously (streaming).
- **Entity:** the key a feature is associated with (user ID, product ID, transaction ID) — features are organized around entities so lookups and joins have a clear key.
- **Feature versioning:** as feature definitions evolve, versioning lets you reproduce exactly which definition a given deployed model was trained against — important for reproducibility and for safely rolling out a redefinition without breaking currently-served models.
- **Streaming/real-time features:** some feature stores support computing features from streaming data (e.g., "clicks in the last 5 minutes") with low latency between event occurrence and availability for serving — a harder engineering problem than batch-materialized features, and a common area where skew (or the "freshness" question) has to be handled explicitly.

### Tradeoffs / when a feature store is (and isn't) worth it
- **Worth it:** multiple models/teams sharing features, frequent retraining, strict low-latency serving requirements, and/or a track record of skew incidents — the infrastructure investment pays for itself by structurally preventing an entire class of recurring incidents.
- **Possibly overkill:** a single small model, infrequent retraining, generous latency budgets, and a small team where a disciplined shared-library approach (same function called by both pipelines, without full feature-store infrastructure) can achieve most of the skew-prevention benefit at a fraction of the operational overhead. A senior answer notes this tradeoff explicitly rather than reflexively recommending a feature store for every situation — infrastructure investment should match the actual scale/risk of the problem.

## 3. Shadow Deployment

**Definition:** deploy a new model (or pipeline version) to run **in parallel** with the currently-live production model on real live traffic, but the new model's predictions are **logged only** — never shown to users or used to make real decisions. The existing model continues to serve all real traffic and decisions throughout.

**Why it exists — the core motivation:** offline validation (backtesting, held-out test sets) can never perfectly replicate live production conditions — real-time feature computation, real request latency/timeout behavior, real (not historical-snapshot) data distributions, and real edge cases in the request format. Shadow deployment closes that gap by observing the new model's actual behavior on genuinely live traffic **before** any user or business outcome depends on it, at zero risk to the current production experience.

**What it catches that offline evaluation can't:**
- **Train/serve skew itself** — comparing the shadow model's *served* feature computations and predictions against expectations on live traffic is a direct, continuous instance of the parity-check idea from the RCA notes §5, and can catch a skew bug before it ever affects a real user, rather than after (versus the reactive RCA workflow, which is inherently after-the-fact).
- **Latency and infrastructure issues** — does the new model meet latency SLAs under real production load and real request patterns, including tail-latency edge cases that a synthetic load test might not surface.
- **Live data distribution mismatches** the offline test set didn't capture, because it's a static historical snapshot rather than the actual current traffic.
- **Genuinely unexpected edge cases in the real request stream** — malformed inputs, unusual entity combinations, request volume spikes — that a curated offline test set is unlikely to fully represent.

**What shadow deployment does NOT tell you (an important limitation to name):**
- It doesn't measure the new model's effect on actual **business outcomes**, since its predictions never influence real decisions — you can compare its raw predictions against the current model's predictions and against eventual ground-truth labels (once available), but you can't measure things that depend on the *model's own predictions changing behavior* (e.g., a recommendation model's shadow predictions don't tell you how users would have actually clicked differently, since users never saw them). For that, you eventually need a genuine **A/B test** (a live traffic split where the new model's predictions really do drive a fraction of real decisions) — shadow deployment is a de-risking step that typically *precedes* an A/B test, not a replacement for one.

**Cost consideration:** shadow deployment doubles (or more) the inference compute cost during the shadow period, since every request is scored by both models — worth naming as a real, non-trivial tradeoff when discussing rollout strategy, especially for expensive models (large deep nets, ensembles) at high request volume.

**Where it sits in a broader safe-rollout progression (good to sketch explicitly):**
offline evaluation → shadow deployment (log-only, zero user risk) → canary / small-percentage A/B test (small fraction of real traffic, real but limited business exposure) → gradual traffic ramp-up → full rollout — each stage de-risking the next, with shadow deployment specifically being the stage that validates *infrastructure and prediction-parity* concerns before any real business-outcome risk is taken on.

## 4. Simple vs. Complex Model Tradeoffs

This is fundamentally the same bias-variance tension (see the bias-variance notes) applied through a **production-systems lens** rather than a purely statistical one — the "right" answer is rarely "use the most accurate model," because accuracy is only one axis among several that matter in a deployed system.

### Axes beyond raw accuracy (the core of a strong interview answer)

**Interpretability / explainability**
- Simple models (linear/logistic regression, shallow decision trees, single-feature scorecards) let you explain *why* a specific prediction was made — critical in regulated domains (credit decisions requiring adverse-action reasons, healthcare, hiring) where a black-box explanation isn't legally or ethically sufficient, independent of whether the complex model is more accurate.
- Complex models (deep ensembles, large neural nets) can approximate explanations post-hoc (SHAP, LIME, attention visualization) but these are approximations of the model's behavior, not the model's actual decision logic — worth being precise about this distinction in an interview, since post-hoc explainability tools are sometimes oversold as equivalent to true interpretability.

**Latency and serving cost**
- Complex models (large ensembles, deep nets) cost more compute per prediction — at high request volume or tight latency SLAs (e.g., real-time bidding, fraud-at-checkout), this can rule out an otherwise-more-accurate model outright, independent of any accuracy gain, because it simply can't meet the latency budget or the cost-per-prediction the business can sustain.
- Model compression techniques (distillation, quantization, pruning) exist specifically to bridge this gap — train a complex "teacher" model, then distill its behavior into a smaller, faster "student" model that approximates it at a fraction of the serving cost — worth mentioning as the practical middle-ground answer to "we want the accuracy of the big model but can't afford to serve it."

**Maintainability and iteration speed**
- Simple models are faster to retrain, debug, and reason about when something goes wrong (easier root-cause analysis when a prediction looks off) — directly relevant to the skew/RCA workflow above, since a simpler, more inspectable pipeline is genuinely easier to diagnose.
- Complex models/pipelines have more moving parts (more hyperparameters, more preprocessing steps, more places for skew or bugs to hide) — increasing both the *likelihood* of a production incident and the *cost* of diagnosing one, a real operational cost that doesn't show up in an offline accuracy comparison at all.

**Data requirements**
- Complex, high-capacity models need substantially more data to avoid high variance/overfitting (bias-variance notes) — in a genuinely data-scarce setting, a simple model may not just be "acceptable," it may **outperform** a complex one in practice, not merely be a reasonable compromise.

**Robustness to distribution shift**
- Simpler models with strong, correct inductive bias can sometimes generalize more robustly under covariate shift than a highly flexible model that overfit to spurious correlations in the training distribution's specific quirks — this connects directly to the "right inductive bias moves the whole bias-variance frontier down" point from the bias-variance notes, applied to a production-robustness context rather than a pure-statistics one.

**Debuggability under skew specifically**
- A simpler feature set and model is mechanically easier to audit in a skew RCA (§1) — fewer features to diff, fewer preprocessing steps to trace, a shorter dependency chain to walk backward through. This is a genuine, if secondary, argument for simplicity that's easy to overlook when the conversation is framed purely around accuracy.

### The framing to lead with in an interview
"Model complexity is a resource-allocation decision across accuracy, latency, interpretability, maintainability, and data availability — not a single-axis 'best model' choice." A strong answer states the specific constraints of the given scenario (regulatory requirements? latency SLA? team size for ongoing maintenance? data volume?) before recommending a model family, mirroring the problem-framing discipline from the foundations notes (§2 there) rather than defaulting to "the most accurate model I know."

### A concrete worked framing (good to have ready)
"For a credit approval model, I'd lean simple (logistic regression / a scorecard) even if a gradient-boosted model scores 2 AUC points higher, because regulatory adverse-action requirements need genuine feature-level explanations, not post-hoc approximations, and the accuracy gap is unlikely to be worth the compliance and interpretability risk. For a real-time ad-ranking model at massive scale, I'd lean complex (a large learned ranking model, possibly distilled for serving), because the latency/cost constraint is solvable via engineering (distillation, caching, approximate serving) and the accuracy gain compounds directly into revenue at scale, in a domain with no comparable regulatory interpretability requirement."

## 5. Pitfalls / Trick Angles

1. **Reflexively recommending a feature store for every production ML system.** As covered in §2 — the infrastructure overhead is only justified once skew risk, feature-reuse-across-teams, or retraining frequency reach a certain scale; recommending it unconditionally, without checking the actual team/system size, is a shallow answer.

2. **Treating shadow deployment as sufficient validation before full rollout.** It validates infrastructure/prediction-parity concerns but not actual business-outcome impact (§3) — a model can pass shadow deployment cleanly and still underperform (or even harm) real business metrics once its predictions actually start influencing decisions, because shadow mode never lets user behavior respond to it.

3. **Assuming "complex model = more accurate = better" without naming the other axes.** The single most common shallow answer to a "which model would you use" system-design question — a strong answer proactively raises latency, interpretability, maintainability, and data-availability constraints rather than only comparing accuracy numbers.

4. **Conflating post-hoc explainability (SHAP/LIME on a black-box model) with genuine interpretability.** Post-hoc tools approximate what a complex model is doing locally around a specific prediction; they are not the same guarantee as a model whose decision logic is the explanation itself (e.g., a linear model's coefficients) — this distinction specifically matters in regulated domains where genuine interpretability, not an approximation of it, may be a hard requirement.

5. **Forgetting that shadow deployment costs real (often doubled) serving compute.** Presenting shadow deployment as a "free" validation step without acknowledging its infrastructure cost is an incomplete answer, especially for expensive models at high traffic volume.

6. **Not distinguishing feature-store online-store freshness limits from genuine skew.** Even with a unified feature store, an online feature can still be *stale* relative to the absolute latest event (materialization lag) — this is a legitimate freshness/latency tradeoff to design around (e.g., streaming materialization vs. batch), not the same failure mode as skew (§1), where the two pipelines compute different values for logically-the-same-timestamp feature. Conflating "my online feature is a few minutes stale" with "my online and offline pipelines disagree" muddies the diagnosis.

## 6. Interview Q&A

**Q1: Why does a feature store structurally prevent train/serve skew, rather than just making it less likely?**
Because both the offline (batch, training-time) and online (low-latency, serving-time) feature values are materialized from the **same registered feature definition/transformation code**, rather than from two independently maintained implementations — the duplicated-code-path root cause (the single most common skew cause per the RCA notes) is eliminated by construction, not merely reduced by process discipline like code review or documentation, which can always lapse.

**Q2: What can shadow deployment catch that a good offline test set can't, and what can it not tell you that an A/B test can?**
It catches live-infrastructure issues (real latency under real load, real request edge cases) and prediction-parity/skew problems, because it runs on genuinely live traffic and a live serving pipeline rather than a static historical snapshot. It cannot tell you the new model's actual effect on business outcomes, because its predictions are logged but never actually shown to users or used to make real decisions — measuring true behavioral/business impact requires a genuine A/B test where the new model's predictions really do influence a slice of real traffic.

**Q3: A stakeholder wants the most accurate model regardless of complexity for a real-time bidding system with a 10ms latency budget. How do you respond?**
Explain that "most accurate" and "usable within a 10ms budget" may be in direct tension — a large ensemble or deep model may simply not fit the latency SLA regardless of its offline accuracy, making it non-deployable as-is. Offer the standard middle-ground paths: model distillation (train the complex "teacher," serve a distilled, faster "student"), quantization/pruning, or a simpler model with strong feature engineering that captures most of the signal at a fraction of the compute — and frame the final choice as an explicit tradeoff against the latency constraint, not a pure accuracy comparison.

**Q4: Why is genuine interpretability sometimes a hard requirement rather than a nice-to-have, and how does that constrain model choice independent of accuracy?**
In regulated domains (credit, lending, hiring, some healthcare contexts), legal/regulatory requirements (e.g., adverse-action notices in credit decisions) require being able to state the actual reason a specific decision was made — a requirement that a black-box model's *approximate* post-hoc explanation may not satisfy, since it explains the model's local behavior around a prediction rather than being the literal decision logic. This can rule out an otherwise more accurate complex model entirely, independent of any accuracy gain, because the requirement isn't "explain reasonably well" but "explain the actual decision process."

**Q5: Your team is a single small ML team supporting one moderately-sized model, retrained quarterly, with generous latency requirements. Would you recommend building a full feature store?**
Probably not as a first move — the skew-prevention and feature-reuse benefits of a feature store are most valuable at a scale (many models/teams sharing features, frequent retraining, tight latency, and/or a history of skew incidents) this scenario doesn't clearly have. A lighter-weight fix (a shared feature-computation library called by both the training and serving code, with disciplined artifact versioning for preprocessing objects) likely captures most of the practical benefit at a fraction of the infrastructure investment — worth explicitly naming this as a proportionality judgment rather than defaulting to "always build the fanciest infrastructure available."

**Q6 (clever): Can a model pass shadow deployment perfectly (predictions match expectations, latency fine, no skew detected) and still be a bad choice to roll out?**
Yes — shadow deployment validates infrastructure correctness and prediction consistency, not business value. The new model could be technically well-served and skew-free while genuinely being a worse model for the actual business objective (e.g., optimizing a proxy metric that doesn't track the real outcome, or simply being less accurate than the currently deployed model in ways an A/B test would reveal but a shadow-mode comparison against the old model's predictions might not clearly surface if you're only checking parity/latency rather than a true business-outcome-oriented offline evaluation). This is precisely why shadow deployment is a *step* in a safe-rollout progression, not a final gate on its own — it needs to be paired with genuine offline accuracy evaluation before shadowing, and a real A/B test after.
