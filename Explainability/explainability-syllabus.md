# Explainability & Interpretable ML — Prep Syllabus

**Goal:** Be able to (1) explain the difference between interpretable-by-design models and post-hoc explanation methods and argue when each is appropriate, (2) go beyond feature attribution into counterfactual explanations and explanation methods for deep learning specifically (saliency maps, attention, concept-based methods), (3) explain how to actually evaluate whether an explanation is good, not just generate one, and (4) speak to the human and regulatory side — what makes an explanation useful to a real person, and what "right to explanation" actually requires.

This is a broader companion to your Feature Importance syllabus, which already covers feature-attribution methods (MDI, permutation importance, SHAP, LIME, PDP/ICE/ALE) in depth. This syllabus intentionally does **not** re-derive those — it references them where relevant and spends its own chapters on everything attribution methods don't cover: interpretable-by-design modeling, counterfactuals, deep-learning-specific explanation, evaluation of explanation quality, and the human/regulatory dimension.

---

## Chapter 1 — Interpretability vs. Explainability (Motivation & Framing)
- The core distinction: **interpretable-by-design** models (the model's own structure is understandable — a shallow decision tree, a linear model, a rule list) vs. **post-hoc explainability** (an opaque model plus a separate method trying to explain it after the fact — everything in your Feature Importance syllabus)
- Why this distinction is contested territory: the case that post-hoc explanations of black boxes can be actively misleading vs. the case that interpretable-by-design models sometimes sacrifice too much accuracy
- The accuracy/interpretability tradeoff — is it real, and when
- A taxonomy preview: this syllabus's chapters, mapped onto the interpretable-by-design vs. post-hoc split

## Chapter 2 — Interpretable-by-Design Models
- Linear/logistic regression and shallow decision trees as the classic interpretable baselines — what interpretability actually buys you structurally
- Generalized Additive Models (GAMs): fitting a separate, visualizable function per feature instead of a single linear coefficient — capturing non-linearity while staying interpretable
- Rule-based models: decision lists, RuleFit, and scoring systems (e.g., point-based risk scores used in medicine) — how they're built and why clinicians/regulators often prefer them
- The case for "just use an interpretable model" (Rudin's argument) vs. counterarguments about when black-box + explanation genuinely outperforms

## Chapter 3 — Counterfactual Explanations
- What a counterfactual explanation is: "the model would have predicted approval if your income were $X higher" — actionable, contrastive explanation rather than attribution
- How counterfactuals are generated: optimization-based search for the nearest input that flips the prediction, with a worked conceptual example
- Desirable properties of a good counterfactual: proximity (close to the original), sparsity (changes few features), plausibility/actionability (the suggested change is realistic and achievable), and diversity (multiple different counterfactual paths offered)
- Counterfactuals vs. SHAP/LIME: attribution answers "why," counterfactuals answer "what would need to change" — different questions, different use cases

## Chapter 4 — Explaining Deep Learning Models: Saliency & Gradient-Based Methods
- Saliency maps: using the gradient of the output with respect to the input to highlight which pixels/tokens mattered most — the simplest gradient-based explanation
- Known problems with raw saliency: noise, and vulnerability to adversarial manipulation of the explanation itself
- Integrated Gradients as a more principled, axiom-satisfying alternative (a Shapley-adjacent idea — integrating gradients along a path from a baseline to the actual input)
- Grad-CAM for convolutional networks: using activation maps from a late convolutional layer to produce a coarse, class-specific localization map

## Chapter 5 — Attention as Explanation, and Its Limits
- The intuitive appeal: attention weights in a transformer seem to directly show "what the model is looking at"
- The "attention is not explanation" debate: evidence that attention weights don't reliably correspond to feature importance measured by other means, and that different attention patterns can produce the same output
- What attention weights can and can't legitimately be used to claim
- Concept Activation Vectors (TCAV) as an alternative: testing whether a human-defined concept (e.g., "stripes") is represented in a network's internals, rather than trusting a raw attention/gradient map

## Chapter 6 — Explainability for Specific Modalities
- NLP: token-level attribution methods, and why explaining text is harder than explaining tabular data (tokens interact non-locally; synonyms and paraphrase break naive attribution)
- Vision: saliency/Grad-CAM in practice, and the sanity-check literature showing some popular saliency methods don't actually depend on the model's learned weights (a critical, commonly-cited failure mode)
- Time series: the extra difficulty of attributing importance to *temporal* patterns rather than static feature values

## Chapter 7 — Evaluating Explanation Quality
- Why "it looks reasonable to me" is not evaluation — the need for quantitative criteria
- Faithfulness: does the explanation actually reflect what the model is doing, tested via deletion/insertion metrics (removing or adding the "important" features per the explanation and checking if the prediction changes as expected)
- Robustness/stability: does a tiny, meaningless perturbation to the input produce a wildly different explanation (a red flag if so)
- Human-grounded evaluation: user studies measuring whether an explanation actually helps a person predict the model's behavior or make a better decision, vs. just feeling satisfying

## Chapter 8 — Explainability in Practice: Stakeholders and Regulation
- Different audiences need different explanations: a data scientist debugging a model, a regulator auditing it, and an end user affected by a decision all want different things from "explainability"
- GDPR's "right to explanation" and similar provisions — what they actually require (and the ongoing legal debate about how much they require) at a high level
- Connecting back to Model Cards and Datasheets (from your Fairness & Responsible AI prep) as the documentation layer that operationalizes explainability for governance purposes
- Explainability as a debugging tool for the ML team itself (finding spurious correlations, data leakage, shortcut learning) — a use case distinct from explaining to an external audience

## Chapter 9 — Practical Synthesis
- End-to-end worked case: given a black-box model flagged for an audit, walk through choosing interpretable-by-design vs. post-hoc, generating both an attribution-style and a counterfactual explanation for one decision, and evaluating whether the explanation is trustworthy using the Chapter 7 criteria
- A decision framework: which explainability approach fits which audience, model type, and modality
- Common interview questions on this topic and how to structure a strong answer (e.g. "When would you choose an interpretable model over a black box plus SHAP?", "How do you know if an explanation is actually faithful to the model?")

---

**Suggested build order:** 1 → 2 (interpretable-by-design, the alternative to everything else in this syllabus) → 3 (counterfactuals, a genuinely different question from attribution) → 4 → 5 → 6 (deep-learning-specific methods, roughly simple-to-sophisticated) → 7 (evaluation — best absorbed once you've seen methods to evaluate) → 8 (stakeholders/regulation) → 9 (synthesis).

Let me know if you want me to start drafting Chapter 1 in full.
