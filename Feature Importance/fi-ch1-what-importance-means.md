# Chapter 1 — What "Importance" Actually Means (Motivation & Framing)

## 1.1 Why "feature importance" isn't one thing

If you ask five different tools "how important is this feature?" you can get five different answers, and — this is the key point of this whole chapter — **that's not a bug, and it doesn't mean one of them is wrong.** They're often answering genuinely different questions that all happen to get called "importance." Before touching a single formula, you need a taxonomy that lets you say precisely which question a given method answers, so that "these two methods disagree" stops being confusing and starts being informative.

Three independent axes distinguish importance methods from each other. Any method you'll meet in this topic can be located on all three at once.

## 1.2 Axis 1: Global vs. Local

**Global importance** answers: "across the whole dataset, which features does this model rely on most, in general?" This is what you want when you're deciding which features to keep, writing a Model Card, or explaining the model's overall behavior to a stakeholder who isn't asking about any one specific person.

**Local importance** answers: "for this one specific prediction, on this one specific example, which features drove the outcome, and by how much?" This is what you want when a specific loan applicant asks "why was I denied," or when you're debugging one particular prediction that looks wrong.

**Why the distinction matters more than it first appears:** a feature can have **high global importance and low local importance for a specific example** — e.g., `credit_score` might be the single most important feature across your whole population on average, but for one particular applicant whose credit score sits exactly at the population average, that feature might contribute almost nothing to *their specific* prediction relative to the baseline, while some other unusual feature of theirs (say, an unusually short employment history) does all the work for that one case. Global importance tells you what matters "in general"; local importance tells you what mattered "for this person" — and conflating the two is a common, avoidable mistake.

## 1.3 Axis 2: Predictive/Associational vs. Causal

**Predictive (associational) importance** answers: "how much does this feature help the model predict the target, given the statistical patterns present in the data it was trained on?" Every single method in this topic — MDI, permutation importance, SHAP, LIME, linear coefficients — answers this question and only this question.

**Causal importance** answers a different question entirely: "if we intervened and actually changed this feature's value in the real world, how much would the target actually change?" None of the methods in this topic answer this question, no matter how sophisticated they get.

**Why this needs to be a first-class axis in your taxonomy, not a footnote:** it's tempting to think a more sophisticated method (SHAP, say) gets you "closer" to a causal answer than a cruder one (raw correlation). It doesn't — SHAP is a more principled, better-behaved way of measuring predictive/associational reliance, but it sits on exactly the same side of this axis as every other method here. The classic illustration (which you'll see again in Chapter 9): `ice_cream_sales` can be a genuinely, robustly high-importance predictor of `drowning_incidents` by every method in this topic, while having approximately zero real causal effect — both variables are driven by a shared cause (hot weather) that isn't in the dataset. No amount of methodological sophistication on the predictive side closes this gap; it requires a fundamentally different toolkit (randomized experiments, instrumental variables, and other causal-identification techniques), which is genuinely a separate topic from this one.

## 1.4 Axis 3: Model-specific (intrinsic) vs. Model-agnostic (post-hoc)

**Intrinsic importance methods** are only defined for one particular type of model, because they use that model's own internal structure directly — MDI only makes sense for tree-based models, because it's literally reading off the impurity reductions the tree-building algorithm already computed while it was growing the tree. Raw linear coefficients only make sense for linear/generalized-linear models, for the analogous reason.

**Post-hoc (model-agnostic) methods** treat the trained model as a black box — they only need the ability to feed it inputs and read off predictions, and don't look inside at all. Permutation importance, SHAP (in its general form), and LIME all work this way — you could apply the exact same permutation-importance procedure to a random forest, a neural network, or a hand-written rule-based system, and it would be defined identically in every case.

**Why this axis matters practically:** intrinsic methods are typically far cheaper to compute (MDI is essentially free, a byproduct of training that already happened) but are tied to one model type and can carry that model type's specific biases (MDI's cardinality bias, covered in depth in Chapter 2, is a direct consequence of exploiting tree-specific internal structure). Post-hoc methods are more flexible and often more trustworthy, but cost more compute, since they typically require re-running the model many times (permutation importance) or approximating an expensive combinatorial quantity (SHAP).

## 1.5 Putting it together: a small map of the rest of this topic

Here's how the methods you'll meet in Chapters 2–7 sit on these three axes — worth returning to after each chapter to check your understanding is tracking correctly:

| Method | Global or Local | Predictive or Causal | Intrinsic or Post-hoc |
|---|---|---|---|
| MDI (Ch2) | Global | Predictive | Intrinsic (trees only) |
| Linear coefficients (Ch3) | Global | Predictive | Intrinsic (linear models only) |
| Permutation importance (Ch4) | Global (can be localized, but natively global) | Predictive | Post-hoc |
| SHAP (Ch5) | Both — local by construction, aggregable to global | Predictive | Post-hoc |
| LIME (Ch6) | Local by construction | Predictive | Post-hoc |
| PDP/ICE/ALE (Ch7) | PDP/ALE: global; ICE: local | Predictive | Post-hoc |

Every single row says "Predictive," never "Causal" — worth letting that sink in now, since Chapter 9 will return to it as the single most important caveat to carry into any real-world use of this entire topic.

## 1.6 Why two valid methods can disagree — and why that's informative, not a bug

Given the taxonomy above, disagreement between two methods is often *not* a sign that one is broken — it can mean they're genuinely answering different questions (a feature can be globally important but locally irrelevant for one example, per §1.2), or it can mean one method has a known bias that the other doesn't share (MDI's cardinality bias vs. permutation importance's relative immunity to it, covered fully in Chapter 2). Learning to ask "which axis are these two methods actually disagreeing on" — rather than just "which one is right" — is the single most useful interview reflex this chapter can give you, and it's the lens every later chapter in this topic will keep coming back to.

## 1.7 Quick self-check before Chapter 2

- Given two importance scores for the same feature from two different methods that disagree, can you name at least two structurally different reasons they might disagree, without assuming one is simply wrong?
- Can you explain, without using the word "causal" in the explanation itself, why every method in this topic answers a fundamentally different question than "what would happen if we changed this feature"?
- Can you place a method you haven't even learned yet (hypothetically) onto this three-axis taxonomy, just from a one-sentence description of how it works?

---

**Next: Chapter 2 — Intrinsic Importance: Tree-Based Models**, going deep on Mean Decrease in Impurity, Mean Decrease in Accuracy, boosted-tree importance types (gain/weight/cover), and a full mechanistic account of the high-cardinality bias.
