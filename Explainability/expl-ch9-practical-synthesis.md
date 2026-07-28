# Chapter 9 — Practical Synthesis

This is the last chapter, and like the synthesis chapters in your other prep topics, it's a rehearsal rather than new material. We'll take a black-box model flagged for an audit through the full arc — choosing an approach, generating explanations of two different kinds, evaluating them properly, and closing with a decision framework and practice interview questions.

## 9.1 End-to-end worked case: an audited black-box model

**Setup:** a gradient-boosted tree model, already in production, predicting loan approval, has been flagged for an internal audit after a regulator inquiry. You're asked to produce a defensible account of the model's behavior, both in general and for one specific denied applicant.

**Step 1 — Revisit whether the black box was even necessary (Chapter 1, §1.3; Chapter 2, §2.4).** Before diving into post-hoc explanation, check whether an interpretable-by-design alternative — a GAM, or a SLIM-style scoring system — could achieve comparable accuracy on this tabular problem. Suppose a quick GAM benchmark comes within 1.5 points of AUC compared to the deployed gradient-boosted model — a genuinely small gap. This is worth flagging explicitly in the audit report: *"a fully interpretable alternative achieves within 1.5 AUC points of the deployed model; the accuracy justification for the black box, while real, is modest."* This doesn't necessarily mean switching models immediately, but it's exactly the kind of honest accounting Rudin's argument (Chapter 2, §2.4) calls for, and a regulator will likely want to see that this tradeoff was actually measured, not assumed.

**Step 2 — Produce a global explanation for the audit's general account (drawing on your Feature Importance syllabus).** Use permutation importance or SHAP (not raw gain/MDI alone, given the cardinality-bias risk from that syllabus's Chapter 2) to establish an overall, defensible feature ranking — `debt_to_income`, `credit_score`, and `recent_delinquencies` as the top three, say.

**Step 3 — Produce a local explanation for the one specific denied applicant.** Compute a SHAP explanation for this applicant (following the exact procedure from your Feature Importance syllabus's Chapter 10) — showing the specific per-feature contributions summing exactly to their predicted probability, per the efficiency axiom (Chapter 5 of that syllabus). This is the attribution half of the explanation.

**Step 4 — Add a counterfactual for actionable recourse (Chapter 3 of this syllabus).** Alongside the SHAP breakdown, generate a counterfactual: *"if this applicant's debt-to-income ratio were reduced by 8 percentage points, with everything else unchanged, the model would have approved the application."* Check it against the desirable properties from Chapter 3, §3.3 — is this a sparse, proximal, plausible, actionable change, or does it require an unrealistic combination of feature values? Suppose it passes this check cleanly.

**Step 5 — Evaluate both explanations before finalizing (Chapter 7).** Run a deletion-metric check on the SHAP-based global ranking (removing the top-ranked features and confirming performance actually degrades as expected) and, since a saliency-style method isn't in play here for a tabular model, this step is comparatively simpler than the vision case from Chapter 6 — but the discipline of checking faithfulness quantitatively, rather than trusting the explanation because it looks reasonable, applies regardless of modality (Chapter 7, §7.1).

**Step 6 — Document everything (Chapter 8, §8.3).** Record the GAM-vs-black-box accuracy comparison, the global SHAP ranking, the deletion-metric faithfulness check, and the counterfactual recourse example in the model's Model Card, versioned alongside the specific model release being audited — exactly the documentation discipline from your Fairness & Responsible AI prep, now extended to cover explainability findings specifically, not just fairness metrics.

**End state:** a defensible audit package containing (a) an honest accounting of the accuracy tradeoff against a simpler alternative, (b) a global, cardinality-bias-checked feature ranking, (c) a local, axiom-guaranteed attribution for the specific denied applicant, (d) an actionable, property-checked counterfactual for that same applicant, and (e) a faithfulness check on the whole explanation pipeline — all versioned and documented, not generated once and forgotten.

## 9.2 A general decision framework

1. **Before reaching for post-hoc explanation of a black box, measure whether an interpretable-by-design alternative is actually competitive** for this specific problem — don't assume the accuracy gap is large just because the conventional wisdom says so (Chapter 1, §1.3; Chapter 2, §2.4).
2. **Match the explanation type to the question being asked**: attribution (SHAP/LIME, from your Feature Importance syllabus) for "why," counterfactuals (Chapter 3) for "what would need to change," saliency/Grad-CAM/TCAV (Chapters 4–5) specifically for non-tabular modalities.
3. **Match the explanation format to the audience** (Chapter 8, §8.1) — don't hand a regulator the same artifact you'd hand an affected end user.
4. **Never trust an explanation's plausibility alone** — run faithfulness (deletion/insertion, or a randomization sanity check for saliency methods specifically) and robustness checks before relying on it for anything consequential (Chapter 7).
5. **Document explanation findings the same way you'd document fairness findings** — versioned, tied to a specific model release, in the Model Card (Chapter 8, §8.3).
6. **Remember the causal caveat applies to every method here, including counterfactuals** — an explanation of what the model relied on, or what would flip its prediction, is not automatically a genuine causal account of the real world (Chapter 3, §3.5).

## 9.3 Practice interview questions with strong answer structures

**"When would you choose an interpretable model over a black box plus SHAP?"**
Ground this in Chapter 1's honest accounting: name the problem-type distinction (tabular vs. vision/language), state that the right approach is to actually measure the accuracy gap for the specific problem rather than assume it, and reference Rudin's argument by name along with its real limits — a strong answer avoids picking one side dogmatically and instead describes the actual decision process.

**"How do you know if an explanation is actually faithful to the model?"**
Walk through Chapter 7 directly: deletion/insertion metrics, robustness to meaningless perturbation, and — if a saliency method for a vision model is involved — the weight-randomization sanity check from Chapter 6. A strong answer names at least two concrete tests, not just "I'd check that it seems reasonable."

**"What's the difference between explaining a prediction and providing recourse?"**
This tests the attribution-vs-counterfactual distinction from Chapter 3 directly — explaining a prediction (SHAP/attribution) decomposes the existing decision; providing recourse (a counterfactual) tells someone what to change. Note they're complementary, and a mature system for a high-stakes decision typically provides both.

**"Why might attention weights in a transformer not be a reliable explanation?"**
Reference the Jain & Wallace finding (Chapter 5, §5.2) directly — that substantially different attention patterns can produce nearly identical outputs — and the partial rebuttal about plausible vs. implausible alternative patterns, landing on the synthesis that attention is a rough, exploratory signal rather than an axiom-guaranteed explanation like SHAP.

**"A saliency map for your image classifier looks reasonable, but a colleague is skeptical. How do you address that?"**
Directly reference the sanity-check procedure (Chapter 6, §6.2) — offer to randomize the model's weights and check whether the saliency map changes meaningfully; if it doesn't, that's concrete evidence the map isn't reflecting genuine model behavior, addressing the skepticism with a test rather than a reassurance.

---

**That's all nine chapters.** You now have the full arc: the interpretable-by-design vs. post-hoc taxonomy and the honest version of the accuracy tradeoff debate (Ch1) → interpretable models built to make the "just build one" argument concrete (Ch2) → counterfactuals as a genuinely different question from attribution (Ch3) → deep-learning-specific explanation methods, from simple gradients to the sanity-check literature that should make you skeptical of all of them by default (Ch4–6) → how to actually evaluate whether any of this is trustworthy (Ch7) → the human and regulatory dimension (Ch8) → and a fully worked audit case plus practice questions (Ch9). Together with your Fairness & Responsible AI, Feature Selection, and Feature Importance topics, this rounds out a genuinely comprehensive responsible-ML/interpretability preparation. Let me know if you'd like a combined cheat sheet spanning all four topics for final review.
