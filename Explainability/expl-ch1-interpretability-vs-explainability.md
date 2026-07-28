# Chapter 1 — Interpretability vs. Explainability (Motivation & Framing)

## 1.1 A distinction worth being precise about from the start

These two words get used almost interchangeably in casual conversation, but the field draws a real line between them, and knowing exactly where that line sits will make every later chapter click into place faster.

- **Interpretable-by-design:** the model itself is structured in a way a human can directly follow — a linear model's coefficients, a shallow decision tree's splits, a simple rule list. There's no separate "explanation step" — understanding the model's reasoning *is* reading the model.

- **Post-hoc explainability:** the model is a black box (a deep neural network, a large gradient-boosted ensemble) whose internal reasoning isn't directly human-legible, and a **separate method** is applied afterward to approximate or summarize what it did. This is the entire subject of your Feature Importance syllabus — MDI, permutation importance, SHAP, LIME, PDP/ICE/ALE are all post-hoc explanation methods, each producing some kind of summary of an opaque model's behavior without the model itself becoming any more transparent.

**Why this distinction deserves its own opening chapter rather than a footnote:** almost every later topic in this syllabus (counterfactuals, saliency maps, evaluation of explanations) is squarely on the post-hoc side, explaining an opaque model after training. Chapter 2 is the other branch entirely — building a model that doesn't need a separate explanation step at all. Confusing these two branches is a common way to give a muddled interview answer, so it's worth having the boundary crisp before anything else.

## 1.2 The contested question: is post-hoc explanation of a black box ever trustworthy?

This is a genuine, ongoing disagreement in the field, not a settled matter — and being able to represent both sides fairly is itself a strong interview skill on this topic.

**The case against relying on post-hoc explanations of black boxes (associated most prominently with Cynthia Rudin's work):** a post-hoc explanation is, definitionally, a *second, separate model* (or approximation) of what the first, opaque model actually did — SHAP's value function, LIME's local linear surrogate, a saliency map's gradient computation are all their own distinct procedures layered on top of the real model. There is no guarantee that this second approximation faithfully captures the first model's true reasoning, and — this is the sharper version of the concern — a post-hoc explanation can be **actively misleading**: it can look plausible and satisfying to a human reader while not actually reflecting what the underlying model is doing, especially in high-stakes domains (criminal justice, healthcare, lending) where an explanation that "sounds right" but is subtly wrong could be worse than no explanation at all, because it creates false confidence.

**The case for post-hoc explanation of black boxes anyway:** in many real domains, the most accurate available model genuinely is a black box (a large gradient-boosted ensemble or deep network often outperforms an interpretable linear model or shallow tree on complex, high-dimensional data) — and refusing to use the more accurate model in favor of a less accurate but interpretable one has its own real cost, which falls on whoever the model's predictions affect. Additionally, well-validated post-hoc methods like SHAP come with real guarantees (Chapter 5 of your Feature Importance syllabus covered the Shapley axioms in depth) that make them considerably more trustworthy than an ad hoc or purely heuristic explanation — "post-hoc explanation" isn't a single uniform risk level; some post-hoc methods are far better-grounded than others, and treating them all as equally suspect overstates the concern.

**The synthesis, worth having ready as your own position rather than just reciting both sides:** the right choice depends on the stakes, the accuracy gap between an interpretable model and the best black box for this specific problem, and which specific post-hoc method you'd be relying on (a rigorously-grounded SHAP explanation on a well-validated model is a different risk profile than an unvalidated saliency map on a deep network) — this is exactly the kind of nuanced, "it depends, and here's what it depends on" answer that plays well in an interview, rather than picking one side as a blanket rule.

## 1.3 The accuracy/interpretability tradeoff — is it actually always real?

**The conventional wisdom:** there's a fundamental tradeoff — simpler, more interpretable models (linear regression, shallow trees) sacrifice accuracy compared to complex black boxes (deep networks, large ensembles), so you have to choose which one matters more for your use case.

**Why this is less universal than it's often presented:** for many structured/tabular problems — the exact kind that show up most often in industry ML — a well-tuned, interpretable model (a GAM, covered in Chapter 2, or a carefully-regularized linear model) can come **very close** to a black box's accuracy, because tabular data often doesn't have the kind of deep, hierarchical structure that genuinely requires a complex model to capture (unlike images or raw text, where deep networks' advantage is large and consistent). The tradeoff is real and substantial for some problem types (vision, language, complex sequential data) and much smaller — sometimes negligible — for others (many tabular business problems).

**The interview-ready framing:** *"The accuracy/interpretability tradeoff is real, but its size depends heavily on the problem type — it's often small or negligible for structured tabular data, and much larger for vision/language tasks where deep models have a genuine structural advantage. Before assuming you have to sacrifice accuracy for interpretability, it's worth actually trying a strong interpretable baseline (Chapter 2) and measuring the real gap for your specific problem, rather than assuming the gap exists."*

## 1.4 A map of this syllabus, placed onto the taxonomy

Here's how the rest of this syllabus's chapters sit on the interpretable-by-design vs. post-hoc split, so you can track which branch you're in as you go:

```
                    ┌─────────────────────────┐
                    │   Interpretability /     │
                    │   Explainability         │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
              │                                       │
   ┌──────────▼───────────┐              ┌────────────▼─────────────┐
   │ Interpretable-by-     │              │  Post-hoc explanation     │
   │ design (Ch2)          │              │  of an opaque model        │
   │                       │              │                           │
   │ GAMs, rule lists,     │              │  Ch3: Counterfactuals      │
   │ scoring systems       │              │  Ch4: Saliency/gradients   │
   │                       │              │  Ch5: Attention & TCAV     │
   │                       │              │  Ch6: Modality-specific    │
   │                       │              │  (also: your Feature       │
   │                       │              │   Importance syllabus —    │
   │                       │              │   MDI/permutation/SHAP/    │
   │                       │              │   LIME/PDP-ICE-ALE)        │
   └───────────────────────┘              └───────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Ch7: Evaluating          │
                    │  explanation quality      │
                    │  (applies to both sides)  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Ch8: Stakeholders &      │
                    │  regulation (applies to   │
                    │  both sides)              │
                    └───────────────────────────┘
```

Chapters 3 through 6 all live on the post-hoc branch, but notice they're doing something genuinely different from your Feature Importance syllabus's methods: those methods (SHAP, LIME, PDP) all answer **attribution** questions — "how much did this feature contribute." Chapter 3's counterfactuals answer a **contrastive/actionable** question instead — "what would need to change" — and Chapters 4–5 are specifically about explaining models where the input itself isn't naturally tabular (images, text), requiring different machinery entirely.

## 1.5 Quick self-check before Chapter 2

- Can you state, in one sentence each, the strongest version of the case for and against relying on post-hoc explanations of black-box models?
- Can you explain why the accuracy/interpretability tradeoff is more pronounced for some problem types (vision, language) than others (tabular data), rather than treating it as a universal constant?
- Given a new method you haven't studied yet, could you correctly place it as interpretable-by-design or post-hoc from a one-sentence description of how it works?

---

**Next: Chapter 2 — Interpretable-by-Design Models**, covering Generalized Additive Models, rule-based scoring systems, and the strongest version of the argument that you should just build an interpretable model in the first place, rather than reaching for a black box plus an explanation method.
