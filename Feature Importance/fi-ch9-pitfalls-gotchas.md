# Chapter 9 — Pitfalls and Gotchas

Chapters 2–8 covered every method in this topic in depth. This chapter pulls the recurring failure modes into one unified, interview-focused reference — the same shape as your Feature Selection syllabus's Chapter 8, but deeper, since you now have more methods to draw the connections across.

## 9.1 Correlated features: masking, splitting, and extrapolation — three distinct mechanisms, one root cause

Every chapter in this topic has hit some version of "correlated features break things," but it's worth being precise that these are **three genuinely different mechanisms**, not one repeated problem, even though they all stem from the same root cause (redundant information across features).

- **Splitting (Chapters 3, 4's linear-model version):** a linear model (or Lasso) has to somehow divide credit between two correlated features, and the exact split is unstable/arbitrary — small data changes can shift which feature "gets" the credit. VIF (Chapter 3, §3.3) is the diagnostic; Elastic Net or combining features is the fix.
- **Masking (Chapter 4, §4.4; also affects tree-based split competition):** shuffling or removing just one of two correlated features barely hurts performance, because the model leans on the untouched, correlated partner instead — both features look individually unimportant even though the pair matters. Grouped permutation importance (Chapter 4, §4.6) is the fix.
- **Extrapolation (Chapter 4, §4.5; Chapter 7, §7.2 for PDP):** breaking the *joint* relationship between correlated features (by shuffling one independently, or forcing one to a fixed value across all examples) creates unrealistic combinations the model was never trained on, distorting the resulting curve or importance score in ways that reflect the model's arbitrary behavior on nonsense inputs rather than genuine importance. Conditional permutation importance (Chapter 4, §4.6) or ALE (Chapter 7, §7.4) are the fixes.

**The interview-ready unification:** *"Correlated features break importance methods in at least three distinct ways — unstable credit-splitting in linear models, mutual masking in perturbation-based methods, and unrealistic extrapolation when you break the joint relationship between features — and recognizing which one you're looking at determines which fix is appropriate, since they're not interchangeable."*

## 9.2 High-cardinality bias: precisely where it does and doesn't apply

**Where it applies:** MDI (Chapter 2, §2.4) and, even more directly, the "weight"/split-count importance type in boosted trees (Chapter 2, §2.3) — both are computed purely from training-data split statistics, with no held-out check, making them vulnerable to a high-cardinality feature looking good purely by having many chances to find a favorable-by-luck split.

**Where it does NOT apply, and it's worth being precise about this rather than over-generalizing the caution:** permutation importance (Chapter 4), SHAP (Chapter 5), and Mean Decrease in Accuracy (Chapter 2, §2.2) are all comparatively immune, since each explicitly validates against held-out data (or, for SHAP, uses the value function v(S) computed consistently regardless of a feature's cardinality) — a high-cardinality noise feature won't fool these methods, because their scoring doesn't depend on how many candidate splits were available during training.

**The interview-ready framing:** *"High-cardinality bias is specifically a training-data-only-computation problem — any importance method that validates against held-out data, in one form or another, is largely immune to it. The fix isn't 'avoid high-cardinality features' — it's 'don't trust a training-only importance measure when cardinality varies across your feature set.'"*

## 9.3 Data leakage in importance computation, not just selection

Your Feature Selection syllabus covered leakage through selecting features on the full dataset before splitting (Chapter 8, §8.2 there). The importance-specific version of this trap: **computing permutation importance, SHAP values, or any held-out-data-dependent importance measure using data the model was actually trained on**, rather than a genuinely held-out set.

**Why this specifically distorts importance measurement:** if you compute permutation importance on the *training* set, a feature the model has overfit to (memorized noise in) will show a large performance drop when shuffled — not because it's genuinely important for generalization, but because the model specifically learned to exploit that feature's training-set-specific noise, and shuffling destroys that memorized (but non-generalizing) pattern. This makes overfit, noise-exploiting features look important, exactly backwards from what you want an importance measure to tell you.

**The fix, stated as a firm rule:** always compute permutation importance (and, where feasible, evaluate SHAP's fidelity) on a genuinely held-out set the model never saw during training — the same discipline required for any legitimate performance evaluation applies equally to importance measurement, since importance measurement is, at its core, a performance-sensitivity measurement.

## 9.4 Predictive/associational importance vs. causal importance — restated, and deepened

Chapter 1 introduced this as an entire taxonomy axis, and Chapter 8's synthesis case (in your Feature Selection syllabus) touched on it briefly. Here's the deepened version, worth having fully rehearsed.

**The ice-cream/drowning pattern, generalized:** whenever two variables share a common unmeasured cause (a "confounder"), every method in this topic — MDI, permutation importance, SHAP, LIME, linear coefficients, PDP/ICE/ALE — will faithfully and correctly report a strong statistical relationship between them, because a strong statistical relationship genuinely exists in the data. **None of these methods have any way to detect, from the data alone, whether that relationship reflects a real causal effect or a shared-confounder pattern** — this isn't a limitation of any one method that a better method could fix; it's a fundamental property of what these methods are built to measure (associations present in observed data) versus what a causal question requires (what would happen under a hypothetical intervention that changes the joint distribution of the data).

**Why "just use SHAP, it's more rigorous" doesn't help here:** it's tempting to think SHAP's mathematical rigor (Chapter 5's axioms) somehow gets you closer to a causal answer than a cruder method would. It doesn't, and this is worth stating explicitly because it's a genuinely common misconception: SHAP's axioms guarantee a *unique, fair allocation of the observed statistical relationship* — they say nothing whatsoever about whether that relationship would survive an actual intervention. A confounded relationship gets a perfectly rigorous, axiom-guaranteed SHAP value, exactly as confidently as a genuinely causal one.

**What to actually reach for instead, when the goal is intervention:** randomized controlled experiments (the gold standard, when feasible), or observational causal-inference techniques (instrumental variables, difference-in-differences, propensity score matching, causal graphical models) that explicitly model the confounding structure rather than just measuring the observed association — these are a genuinely separate methodological toolkit from everything in this topic, worth knowing exists but out of scope for this syllabus specifically.

**The interview-ready framing, worth having ready verbatim:** *"Every method in this topic tells you what the model found useful for prediction, given the associations present in the data it was trained on. None of them tell you what would happen if you actually intervened and changed a feature's value in the real world — that's a fundamentally different question, requiring causal-inference methods, and no amount of sophistication in the importance method itself closes that gap."*

## 9.5 A unified pre-flight checklist before trusting any importance ranking

Pulling every pitfall in this chapter into one practical checklist you can run through before reporting or acting on an importance ranking:

1. **Was this computed on genuinely held-out data**, for any method that depends on held-out evaluation (permutation importance, and ideally SHAP's fidelity checks)? (§9.3)
2. **Does my feature set mix high- and low-cardinality features**, and if so, am I relying on MDI/split-count alone rather than a held-out-validated method? (§9.2)
3. **Have I checked for correlated features** (a correlation matrix or VIF pass), and if I find some, do I know which of the three mechanisms (§9.1) is likely at play for my specific method?
4. **Is my ranking stable** across a bootstrap resample or two (Chapter 8, §8.4), or does it shuffle around suspiciously?
5. **Am I about to use this importance ranking to justify an intervention** (changing a feature, pulling a business lever), rather than just to explain or debug a prediction? If so, have I explicitly flagged that a causal analysis, not just a better importance method, is what's actually needed? (§9.4)

---

**Next: Chapter 10 — Practical Synthesis**, the final chapter — an end-to-end worked case explaining one specific denied loan application from global importance through a local SHAP explanation, catching a correlated-feature and a cardinality-bias issue along the way, plus a decision framework and practice interview questions pulling all nine prior chapters together.
