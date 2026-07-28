# Chapter 9 — Model Documentation & Governance

Chapters 1–8 gave you the technical core of this topic — how to measure fairness and how to mitigate it. This chapter covers the organizational layer around that work: how findings and decisions get written down, reviewed, and made accountable, so that the tradeoff choices from Chapter 8 are explicit and auditable rather than buried in someone's head or a Slack thread.

## 9.1 Model Cards

**Origin:** Mitchell et al., 2019 ("Model Cards for Model Reporting"). A Model Card is a short, standardized document that accompanies a trained model, describing what it does, how well, and for whom — specifically so a downstream user (another team, a regulator, the public) doesn't have to reverse-engineer that information.

**Typical sections in a Model Card:**
- **Model details:** what type of model, version, training date, who developed it, license, contact for questions.
- **Intended use:** the use cases the model was designed and evaluated for, and — just as important — the use cases it's explicitly **not** intended for (this section is often what prevents a model built for one purpose from later being misapplied to a higher-stakes purpose it was never evaluated against).
- **Training data:** where it came from, at a high level, and any known limitations or skews in it (this section should reference the datasheet, covered in 9.2, rather than duplicate it).
- **Evaluation data:** what was used to measure performance, and how it differs (if at all) from the training data.
- **Metrics:** the performance metrics reported, and critically, **broken down by relevant subgroup** — this is where Chapter 4's per-group and intersectional metrics actually get published and made reviewable, rather than existing only in an internal analysis notebook.
- **Ethical considerations:** known risks, potential misuse, and fairness findings — including disparities that were measured but not (or not fully) mitigated, stated plainly rather than omitted.
- **Caveats and recommendations:** known limitations, and guidance for anyone deciding whether to use or extend the model.

**A toy worked example** (abbreviated): a Model Card for the loan-screening model from Chapters 2–7 might state: *"Intended use: preliminary screening for personal loans under $10,000, in conjunction with human review — not intended for use as a sole/automated approval decision. Evaluation: TPR = 0.75 (Group A) vs. 0.75 (Group B) after equal-opportunity post-processing (Chapter 7); FPR gap of 12 points remains unmitigated and should be reviewed before deployment in jurisdictions with disparate-impact liability exposure."* That last clause is exactly the kind of explicit, honest disclosure a Model Card is for — noting a known, unresolved gap rather than only reporting the metric that was successfully fixed.

## 9.2 Datasheets for Datasets

**Origin:** Gebru et al., 2018 ("Datasheets for Datasets"), often cited alongside Model Cards but documenting the *input* rather than the *model*.

**What a datasheet documents**, organized (per the original paper) around the dataset's lifecycle:
- **Motivation:** why the dataset was created, and by whom.
- **Composition:** what the instances represent, how many, whether it's a sample of a larger population (and if so, how it was sampled), and — crucially for fairness — whether any protected-attribute or demographic information is included, and how it was determined (self-reported vs. inferred, which matters a great deal, since inferred demographic labels can themselves introduce a new source of measurement error into your fairness analysis).
- **Collection process:** how the data was collected, over what time period, and whether the collection mechanism itself could introduce the kind of measurement bias discussed in Chapter 2 (§2.4) — e.g., a dataset of loan defaults that only includes people who were *approved* has a collection-process-induced selection bias baked in by construction.
- **Preprocessing/cleaning/labeling:** what was done to the raw data, and whether the labeling process itself (Chapter 1, §1.2) could have introduced bias — e.g., who the human labelers were, what instructions they were given.
- **Uses:** other tasks the dataset has been or could be used for, and any uses the creators recommend against.
- **Distribution and maintenance:** licensing, how updates are handled, who to contact with questions.

**Why this matters specifically for fairness (not just general data hygiene):** most of the bias sources identified in Chapter 1 — selection bias, label bias, historical inequity — are properties of the *data*, not the model. A Model Card alone, describing only the trained model's behavior, can't surface *why* a disparity exists; the datasheet is where the root cause (a collection or labeling artifact, for instance) actually gets documented, which is what lets someone downstream decide whether a disparity is fixable via the mitigation techniques in Chapters 5–7, or whether it points to a deeper problem with the data itself that no amount of re-weighting will fully resolve.

## 9.3 Audit trails, versioning, and governance review

**Audit trails:** a record of what data, code, and hyperparameters produced a specific deployed model version, plus a record of what fairness metrics were measured and what mitigation decisions were made and by whom — so that if a disparity is discovered after deployment, there's a clear record to trace back through, rather than having to reconstruct the history from memory or scattered notebooks.

**Versioning for compliance:** every deployed model version should be traceable to its exact Model Card, datasheet, and fairness evaluation report — this connects directly to the model versioning/reproducibility practices from your MLOps prep, just with the fairness-specific artifacts (per-group metrics, chosen mitigation strategy, documented tradeoff point from Chapter 8) included as first-class parts of what gets versioned, not an afterthought.

**Model risk / governance review boards:** many organizations in regulated industries (especially finance) have a formal committee that reviews models above a certain risk threshold before deployment — checking the Model Card, datasheet, and fairness metrics against policy and legal requirements before sign-off. The board's role is essentially to be the "who decides where on the Chapter 8 tradeoff curve to land" answer, made into a formal, accountable process rather than an implicit choice made only by the model-building team.

## 9.4 Regulatory context (high level — enough to name-drop correctly)

You don't need legal expertise here, but you should be able to name these accurately and connect each to a concept from earlier chapters:

- **EU AI Act:** classifies AI systems into risk tiers (unacceptable, high, limited, minimal), with the strictest documentation, testing, and human-oversight requirements applying to "high-risk" systems — which explicitly includes categories like employment, credit-scoring, and law enforcement risk-assessment, i.e., exactly the use cases discussed throughout this topic.
- **NIST AI Risk Management Framework (AI RMF):** a US framework (voluntary, not a binding law) organized around four functions — Govern, Map, Measure, Manage — that gives organizations a structured process for identifying and mitigating AI risks, including fairness/bias risk; it's a process framework rather than a set of specific numeric fairness thresholds.
- **Sector-specific US rules:** e.g., the **Equal Credit Opportunity Act (ECOA)** in lending, which predates modern ML but applies to algorithmic credit decisions — prohibiting discrimination based on protected characteristics in credit decisions, and generally requiring lenders to be able to explain the specific reasons for an adverse credit decision, which has direct implications for model interpretability, not just fairness metrics.

**The interview-ready framing:** "documentation practices like Model Cards and Datasheets aren't just good hygiene — they're increasingly what regulatory frameworks like the EU AI Act actually require organizations to produce for high-risk systems, so being able to produce this kind of documentation is itself a compliance capability, not only a best practice."

---

**Next: Chapter 10 — Practical Synthesis**, the final chapter — a full end-to-end worked case (the loan model from Chapters 2, 3, and 7) taken from diagnosis through metric selection through mitigation through documentation, plus a decision framework and practice interview questions to pull everything in Chapters 1–9 together.
