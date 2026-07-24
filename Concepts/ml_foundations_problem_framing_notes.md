# Foundations & Problem Framing — Interview Notes

## 1. Types of Machine Learning

### Supervised Learning
Learn a mapping $f: X \to Y$ from labeled examples $(x_i, y_i)$.
- **Regression:** $Y$ continuous (price, demand, temperature).
- **Classification:** $Y$ categorical — binary (fraud/not-fraud), multi-class (digit 0–9), multi-label (an article can be tagged "politics" *and* "economy" simultaneously — not mutually exclusive, easy to conflate with multi-class).
- Defining property for framing purposes: you have ground-truth answers for historical examples, and the deployment question is "what's $y$ for this new $x$."

### Unsupervised Learning
No labels — learn structure in $X$ alone.
- **Clustering** (k-means, DBSCAN, hierarchical): group similar points, no predefined "correct" grouping.
- **Dimensionality reduction** (PCA, t-SNE, UMAP, autoencoders): compress $X$ while preserving structure — used both for visualization and as a preprocessing/feature step for downstream models.
- **Density estimation / anomaly detection:** model $P(X)$ itself; flag low-density points as outliers.
- **Association rule mining** (market basket analysis): find "if A then B" co-occurrence patterns.
- Key framing distinction from supervised: there's no single "correct" objective to check against — evaluation is inherently softer (silhouette score, reconstruction error, human judgment) unless you can proxy it against a downstream supervised task.

### Self-Supervised Learning (worth naming explicitly — often conflated with unsupervised, but distinct and interview-relevant post-LLM era)
Labels are **automatically generated from the data itself**, not human-annotated — e.g., masked-language-modeling (predict a hidden word from context), next-token prediction, contrastive learning (predict whether two augmented views come from the same image). Sits between supervised and unsupervised: uses a supervised-style loss function, but the "labels" cost nothing to acquire since they're derived from the raw data's own structure. This is the paradigm underlying most modern pretraining (BERT, GPT-style models, SimCLR).

### Reinforcement Learning
An **agent** takes **actions** in an **environment**, receiving **reward** signals, trying to learn a **policy** $\pi(a\mid s)$ that maximizes expected cumulative reward over time.
- Fundamentally different framing from supervised learning: no fixed dataset of "correct answers" — the agent's own actions determine what data it sees next (the data distribution is *not* i.i.d. and depends on the policy being learned, which is why RL is notoriously harder to train stably).
- **Exploration vs. exploitation** is the central tension unique to RL: try new actions to learn more (exploration) vs. take the currently-best-known action (exploitation) — this tradeoff has no real analog in supervised learning.
- **Credit assignment problem:** a reward may arrive many steps after the action that actually caused it (e.g., winning a chess game 40 moves after a critical blunder) — figuring out *which* past action deserves credit/blame is a defining RL challenge.
- Common interview trap: describing a recommendation system that only uses past logged clicks as "RL" — that's supervised/bandit-adjacent learning on historical data unless the system is *actively* choosing actions and receiving new reward signal in a closed loop (true online RL) or being trained with logged-bandit-feedback correction (off-policy methods) — casually calling static historical-log training "RL" is a precision-of-language flag interviewers watch for.
- **Contextual bandits** are worth mentioning as the useful middle ground: single-step "episodes" (choose one action, get one reward, no long-horizon credit assignment) — this is what most real-world "RL-flavored" production systems (ad selection, content ranking) actually are, since true multi-step RL is rare and hard to deploy safely in most industry settings.

### Semi-Supervised Learning (worth a one-liner — commonly asked as a follow-up)
A small amount of labeled data plus a large amount of unlabeled data, exploiting structure in the unlabeled data (e.g., cluster assumption, manifold assumption) to improve on what the small labeled set alone would support. Practically relevant whenever labeling is expensive but raw data is abundant (medical imaging, specialized domains).

## 2. ML Problem Framing — The Step Interviewers Actually Care About Most

The most common **system-design-interview failure mode** isn't a modeling mistake — it's jumping straight to "I'll use XGBoost" without doing this step. A strong answer walks through:

**a) What decision does this prediction actually drive?**
Every ML problem exists to inform some downstream action. "Predict churn" isn't the real goal — "decide who gets a retention offer, and how much it's worth spending on them" is. This reframing often changes the entire problem: you may not need a calibrated probability at all, you may need a *ranking*, or you may need an *uplift/causal* estimate (will this specific offer change this specific user's behavior) rather than a plain churn-propensity score — a very common and important distinction (correlational risk score vs. causal treatment-effect estimate) that separates junior from senior answers.

**b) What's the unit of prediction, and what's the label definition — precisely?**
"Fraud" needs an exact operational definition: fraud as *confirmed by an investigator* (delayed, expensive, but clean) vs. fraud as *charged back by a customer* (fast, noisy, biased toward disputes people bother to contest) vs. fraud as *flagged by a rules engine* (circular — you'd be predicting your existing rules, not fraud itself). Choosing the label definition **is** a large part of the problem-framing work, and doing it sloppily is a direct path to the label-leakage and sampling-bias issues covered in the data-leakage notes (§2E there).

**c) What's the right evaluation metric, and does it match the business cost structure?**
Accuracy is almost never the right metric for imbalanced real-world problems (99% accuracy on 1%-prevalence fraud by predicting "no fraud" always). The metric should encode the actual asymmetric cost of errors: precision vs. recall tradeoff mapped to $/false-positive vs. $/false-negative, calibration requirements if the score feeds a downstream cost-sensitive decision (e.g., expected-value thresholding), ranking metrics (NDCG, MAP) if the output feeds a ranked list rather than a binary decision.

**d) What's the baseline, and is ML even justified?**
Always establish what a simple heuristic (a rule, a lookup table, the current manual process, a naive "predict the mean/mode") achieves first. This does two things: sets a floor any ML system must beat to be worth its complexity/maintenance cost, and often reveals that the "hard" part of the problem isn't modeling at all but data availability or a plumbing/labeling problem.

**e) What are the latency, scale, interpretability, and update-frequency constraints?**
A fraud model scoring transactions in real-time at checkout has millisecond latency budgets that rule out large ensembles or expensive feature lookups; a credit model may have regulatory interpretability requirements (adverse action notices) that rule out uninterpretable black boxes outright, independent of accuracy gains. These constraints often narrow the model-family choice *before* any accuracy consideration — a strong interview answer raises these before jumping to algorithm selection.

**f) What can go wrong once deployed, and how will you know?**
Anticipate feedback loops (a fraud model changes fraudster behavior — concept drift the model itself causes), monitoring needs (see drift-monitoring notes), and failure modes specific to the domain (a false negative in medical diagnosis is categorically worse than a false positive, in a way pure accuracy numbers don't capture).

## 3. The ML Workflow (End-to-End)

1. **Problem framing** (§2) — before touching data.
2. **Data collection / sourcing** — identify what data exists, what needs to be newly collected or labeled, and what's *legally and ethically* usable.
3. **Exploratory data analysis (EDA)** — distributions, missingness patterns, class balance, obvious data-quality issues, sanity-check against domain knowledge.
4. **Data cleaning / preprocessing** — handle missing values, outliers, encoding, normalization. **Must happen fit-on-train-only** (see data-leakage notes §5) to avoid contaminating validation.
5. **Feature engineering** — construct features from raw data, guided by domain knowledge and EDA findings; often the highest-leverage step in practice, more so than model choice.
6. **Train/val/test split** — see §4 below; must happen *before* any leakage-prone step.
7. **Baseline model** — simple heuristic or simple model, establishes the floor (§2d).
8. **Model selection & training** — try candidate model families, starting simple and increasing complexity only as justified by validation performance.
9. **Hyperparameter tuning** — via the validation set or cross-validation, never the test set.
10. **Evaluation on test set** — the *one-time*, final, unbiased read of how the chosen model generalizes. Using the test set more than once, or using it to make further modeling decisions, quietly turns it into a second validation set (a subtle form of leakage-by-repeated-peeking, worth naming explicitly).
11. **Error analysis** — look at *where* the model is wrong, not just aggregate metrics; slice performance by relevant segments (a model with good aggregate accuracy can be badly miscalibrated for a specific important subgroup).
12. **Deployment** — serving infrastructure, latency/scale requirements, ensuring feature computation matches training exactly (train/serve skew risk).
13. **Monitoring** — performance monitoring (once labels arrive), drift monitoring (PSI/KL on inputs), alerting.
14. **Retraining / iteration loop** — scheduled or triggered retraining, informed by monitoring; the workflow is a loop, not a one-shot pipeline.

**Interview framing point:** this is often drawn as a loop, not a line — production ML is characterized by the fact that step 13 feeds back into steps 1–2 (new labels, discovered edge cases, drift) far more than a textbook "collect data → train → done" picture suggests. A good answer acknowledges the loop explicitly, especially in a system-design context.

## 4. Train / Validation / Test Splits

**Purpose of each split — the core thing to be crisp about:**
- **Train:** what the model's parameters are fit on.
- **Validation:** used to make *modeling decisions* — hyperparameter tuning, model family selection, early stopping, feature selection. Because you use it repeatedly to make decisions, it stops being a clean measure of generalization the more you use it — this is why a separate held-out test set exists.
- **Test:** touched **exactly once**, at the very end, to report the final unbiased generalization estimate. If you tune anything based on test performance, it has silently become a second validation set, and your reported number is now optimistic — a subtle, high-value point to state explicitly in an interview.

**Typical splits:** 60/20/20 or 70/15/15 for moderate datasets; for very large datasets, the validation/test sets can be a much smaller *absolute* percentage (e.g., 98/1/1) since even 1% of a huge dataset is plenty of examples for a stable estimate.

**Cross-validation (k-fold) as an alternative/complement:**
- Rotates which fold plays "validation" across $k$ splits, averaging results — gives a lower-variance estimate of validation performance than a single split, at the cost of $k\times$ the compute, and is especially valuable for small datasets where a single validation split would itself have high sampling variance (tying back to the "CV score itself has variance" point from the bias-variance notes, §4 pitfall 7 there).
- A held-out **test set is still needed on top of CV** — CV replaces/stabilizes the *validation* step, not the final unbiased test evaluation.

**Special-case splitting strategies (a favorite "do you actually understand *why*" probe):**
- **Time-series data → chronological split**, never random shuffle (§ covered in depth in the distribution-shift notes) — training must only ever see the past relative to what it's validated/tested against.
- **Grouped data (multiple rows per patient/user/session) → `GroupKFold`**, keeping all of a given group's rows entirely within one split — otherwise the model "cheats" by seeing other rows from the same entity during training (this is the group-leakage failure mode from the data-leakage notes, §2D).
- **Stratified splitting** for classification with class imbalance — ensures each split preserves the overall class ratio, avoiding a validation fold that happens to have zero (or wildly skewed) positive examples by chance, especially important with rare classes and small data.
- **Nested cross-validation** when you're both selecting hyperparameters *and* wanting an unbiased performance estimate — an outer CV loop provides the final unbiased estimate, while an inner CV loop (run independently within each outer training fold) handles hyperparameter tuning, preventing hyperparameter selection from leaking into the final performance number (a specific, precise instance of the general "don't let tuning touch your evaluation set" principle).

**Common pitfalls (ties directly into data-leakage notes, worth cross-referencing explicitly in an interview to show connected understanding):**
- Preprocessing/feature-selection/target-encoding fit *before* the split (leakage into val/test).
- Repeatedly "peeking" at test performance during iteration, turning it into a de facto second validation set.
- Random splitting on time-ordered or grouped data.
- Forgetting to stratify on a rare class, producing a noisy/unrepresentative validation read purely from sampling luck.

## 5. Data Leakage — Summary (see dedicated notes for full depth)

Leakage is when information unavailable at real prediction time sneaks into training or evaluation, inflating offline metrics in a way that collapses in production. Core taxonomy: **target leakage** (a feature is a proxy for or consequence of the label), **train-test contamination** (preprocessing/encoding/splitting done wrong), **temporal leakage** (using future info to predict the past), **group leakage** (related rows split across train/test), and **leakage baked into the label-generation process itself** (biased sampling of who gets labeled). The single strongest detection signal is *implausibly good* offline performance for a genuinely hard real-world problem — the fix is almost always a strict fit-on-train-only, point-in-time-correct, group-aware pipeline discipline (see the dedicated data-leakage notes for the full taxonomy, detection methods, and Q&A).

## 6. Pitfalls Specific to Foundations/Framing (Interviewers Probe These Early)

1. **Jumping to "which algorithm" before defining the label and the metric.** The most common junior-vs-senior tell in a system-design interview — senior candidates spend real time on §2 before naming a single model family.

2. **Treating "high accuracy" as the goal instead of the actual business decision it informs.** A model can have excellent AUC and still be useless if it isn't calibrated for the cost-sensitive threshold the business actually needs, or if it predicts propensity when the business actually needed a causal/uplift estimate (§2a).

3. **Conflating correlational prediction with causal decision-making.** "Users who get a discount convert more" doesn't mean *giving* a random new user a discount will cause them to convert — many users in that correlation were already going to convert. This distinction (propensity vs. uplift) is one of the most senior-flavored framing points to raise proactively.

4. **Assuming unsupervised evaluation is "softer, so it doesn't matter much."** Poor clustering/dimensionality-reduction evaluation (e.g., picking $k$ in k-means purely by eyeballing) can silently produce a downstream feature or segmentation that's actively misleading — evaluation rigor matters just as much, it's simply a different toolkit (silhouette score, reconstruction error, stability across resamples, or ultimately a downstream supervised proxy task).

5. **Calling any system with historical logged data "RL."** As covered in §1 — without a genuine closed action-reward loop (or explicit off-policy correction for logged bandit feedback), it's supervised learning on historical outcomes, and conflating the two is a language-precision flag.

6. **Not distinguishing validation-set use from test-set use, and touching the test set more than once.** Directly causes an optimistic final number and is one of the easiest things an interviewer can catch by asking "how many times did you look at test performance during development?"

7. **Skipping the baseline.** Presenting a complex model's performance with no simple-heuristic comparison point makes it impossible to tell whether the complexity was actually earning its keep.

## 7. Interview Q&A

**Q1: Someone says "let's just use RL for our recommendation system since it maximizes long-term reward." What would you push back on?**
Ask whether there's a genuine closed-loop deployment where the system's own actions determine what data it sees next, with a real reward signal arriving in a reasonable timeframe — most recommendation systems in practice are trained on static logged historical data (supervised/contextual-bandit territory), and "true" multi-step RL introduces exploration risk (showing users bad recommendations to learn), non-stationarity, and safety/deployment complexity that's rarely justified unless the long-horizon credit-assignment structure is genuinely present and the business is prepared for the exploration cost.

**Q2: What's wrong with using accuracy as your only reported metric for a fraud model, even if you also plan to say "we got 99.5% accuracy"?**
With typical fraud prevalence (often well under 1%), a trivial "always predict not-fraud" classifier achieves >99% accuracy while being completely useless — accuracy is dominated by the majority class and hides the actual quantity of interest (how well you catch the rare positive class). Precision/recall, PR-AUC, or a cost-weighted metric reflecting the actual $-per-false-positive vs. $-per-false-negative asymmetry are the right lens instead.

**Q3: Why do you need both a validation set and a separate test set — isn't cross-validation on the full non-test data enough?**
CV (across the training data) is used to make modeling decisions (tune hyperparameters, pick a model family) — using it repeatedly across many iterations means it accumulates a small optimistic bias toward whatever choices happened to look good on those specific folds, purely by chance (multiple-comparisons-style overfitting to the validation procedure itself). The test set, touched exactly once at the very end, gives an estimate untouched by that iterative decision-making process — it's a fundamentally different role, not a redundant one.

**Q4: You're building a churn model. The business wants "the churn probability" for each customer. What's the first framing question you'd ask back?**
What decision will this probability actually drive — is the goal to rank customers by risk for a fixed-size outreach campaign (a ranking problem, calibration less critical), to threshold and trigger an automatic retention offer (calibration now critical, since the threshold assumes the probability is meaningful in absolute terms), or to decide *who specifically would be swayed by an intervention* (which is a causal uplift question, not a plain propensity question, and a churn-propensity model alone can't answer it — some high-churn-risk users would churn regardless of any offer, and some low-risk users are the ones actually persuadable).

**Q5: What's the difference between semi-supervised and self-supervised learning, and why does the distinction matter?**
Semi-supervised learning has a small set of *human-provided* labels plus a larger unlabeled set, and tries to propagate label information using structure in the unlabeled data (e.g., cluster/manifold assumptions). Self-supervised learning generates its "labels" automatically from the data's own structure (e.g., predicting a masked word from surrounding context) with no human annotation at all — it's a pretraining strategy, typically followed by fine-tuning on a smaller genuinely-labeled set for the actual downstream task. The distinction matters because they solve different bottlenecks: semi-supervised assumes labeling is the scarce resource and tries to stretch a small labeled set further; self-supervised assumes labeling is prohibitively expensive/unavailable at scale and sidesteps it almost entirely for the pretraining phase.

**Q6: Give an example of a problem where the "obvious" label definition would introduce data leakage, and how you'd reframe it.**
A "will this support ticket be escalated" model using a label taken from the ticket's *final* status field, which is only set once the ticket is closed (and may itself be influenced by how the agent handled it after your prediction would have needed to fire) — the label should instead be defined at a clean decision point (e.g., "was this ticket escalated within the first hour of being opened," using only information known at that hour), reframing the problem so the label's timing genuinely precedes and is independent of what your features can see.

**Q7: A candidate proposes k-means clustering for a segmentation task and picks k=5 "because it looked good on a 2D PCA plot." What would you probe?**
Whether that visual impression on a 2D projection (which can hide or distort real cluster structure in the full-dimensional space) is corroborated by an actual quantitative criterion — silhouette score across a range of $k$, the elbow method on within-cluster sum of squares, or better yet, stability of the resulting clusters across bootstrap resamples of the data. Also worth asking whether the segments are being validated against a downstream business use (do the resulting segments actually behave differently on some outcome that matters) — the ultimate test of an unsupervised result is often whether it's *useful*, not just whether it looks clean on paper.
