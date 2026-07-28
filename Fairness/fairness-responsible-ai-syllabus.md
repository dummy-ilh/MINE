# Fairness & Responsible AI — Prep Syllabus

**Goal:** Be able to (1) explain what "fair" means mathematically in 2-3 different ways, (2) show why those definitions conflict with each other, (3) describe how to fix an unfair model at each stage of the ML pipeline, and (4) talk fluently about model documentation/governance in an interview setting.

This follows the same format as your other prep chapters: built from zero, plain-language math, diagrams where useful, delivered as `.md`.

---

## Chapter 1 — Why Fairness Is Hard (Motivation & Framing)
- What "bias" means in an ML system vs. in everyday language
- Where bias enters the pipeline: historical data, sampling, labeling, feature choice, objective function, deployment feedback loops
- Real-world case studies (COMPAS recidivism, Amazon hiring tool, facial recognition disparities) — just enough to motivate the math, not a deep legal dive
- The central tension: you cannot satisfy every fairness definition at once (preview of the impossibility results in Ch3)

## Chapter 2 — Defining Groups and Setting Up Notation
- Protected attributes / sensitive groups (A), true label (Y), predicted label (Ŷ), score (S)
- Confusion matrix refresher through a fairness lens (TPR, FPR, PPV, NPV per group)
- Base rates and why they matter

## Chapter 3 — Group Fairness Metrics
- **Demographic Parity (Statistical Parity):** P(Ŷ=1 | A=a) equal across groups — worked numeric example
- **Equalized Odds:** TPR and FPR equal across groups — worked numeric example
- **Equal Opportunity:** TPR-only relaxation of equalized odds
- **Predictive Parity / Calibration across groups:** P(Y=1 | Ŷ=1, A=a) equal, and calibration curves per group
- The impossibility theorem: why you generally can't satisfy demographic parity, equalized odds, and calibration simultaneously when base rates differ across groups (Chouldechova / Kleinberg et al. result, explained intuitively with a small numeric example, not the full proof)
- Individual fairness (brief contrast): "similar individuals should get similar predictions" — Lipschitz-style definition, and why it's hard to operationalize

## Chapter 4 — Measuring Fairness in Practice
- Choosing a metric based on the use case (lending vs. hiring vs. criminal justice vs. ad targeting)
- Slicing metrics by intersectional subgroups, and the small-sample-size problem when you slice too finely
- Statistical significance / confidence intervals on fairness metrics (small subgroup = noisy metric)
- Fairness dashboards and what a "fairness report" typically contains

## Chapter 5 — Mitigation: Pre-processing
- Re-weighting training examples by group/label combination (worked example computing weights)
- Re-sampling (oversampling/undersampling by group)
- Disparate impact removal / feature transformation to reduce correlation with protected attribute
- Data augmentation for underrepresented groups
- Tradeoffs: pre-processing is cheap but can't always fix downstream model behavior

## Chapter 6 — Mitigation: In-processing
- Adding a fairness constraint or regularization term to the loss function (constrained optimization framing — ties back to your Lagrange multipliers chapter)
- Adversarial debiasing: main model vs. adversary predicting the protected attribute from the main model's output, trained in opposition — explained step by step with the two-network diagram
- Fairness-constrained optimization (reductions approach, e.g. Agarwal et al.) at a conceptual level
- Tradeoffs: better fairness/accuracy tradeoff control, but more complex training and less portable

## Chapter 7 — Mitigation: Post-processing
- Threshold adjustment per group to equalize a chosen metric (worked numeric example: different thresholds for different groups)
- Calibration post-hoc adjustment
- Reject option classification (abstain near the decision boundary)
- Tradeoffs: no retraining needed, but legally/ethically sensitive (explicitly using protected attribute at inference time)

## Chapter 8 — The Fairness/Accuracy Tradeoff
- Why mitigation almost always costs some accuracy, and how to reason about that tradeoff quantitatively
- Pareto frontier framing: plotting accuracy vs. fairness metric as you sweep a mitigation strength parameter
- How to talk about this tradeoff in an interview without sounding like you're dismissing fairness as "a cost"

## Chapter 9 — Model Documentation & Governance
- **Model Cards** (Mitchell et al.): what sections go in one, worked example for a toy model
- **Datasheets for Datasets** (Gebru et al.): what a datasheet documents and why
- Audit trails, versioning for compliance, and the role of a model risk/governance review board
- Regulatory context at a high level (EU AI Act risk tiers, NIST AI RMF, US sector-specific rules like ECOA in lending) — enough to name-drop correctly, not a legal deep dive

## Chapter 10 — Practical Synthesis
- End-to-end worked case: a loan approval model with a demographic parity gap — diagnose, choose a metric, choose a mitigation, show before/after numbers
- A decision framework: "which fairness metric and which mitigation stage should I pick, given this use case?"
- Common interview questions on this topic and how to structure a strong answer (e.g. "How would you detect and fix bias in a hiring model?")

---

**Suggested build order:** 1 → 2 → 3 → 4 (concepts and measurement) before 5 → 6 → 7 (mitigation, in pipeline order) → 8 → 9 → 10 (synthesis). This mirrors how you built the optimization and MLOps chapters — foundations first, then techniques, then a practical synthesis chapter at the end.

Let me know if you want me to start drafting Chapter 1 in full (same plain-language, build-from-zero style as your other chapters), with diagrams for the pipeline and confusion-matrix-by-group visuals.
