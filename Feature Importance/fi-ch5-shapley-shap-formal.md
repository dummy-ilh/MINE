# Chapter 5 — Shapley Values and SHAP, Formally

Earlier material gave you the intuition for Shapley values and how SHAP approximates them. This chapter goes all the way: the actual formula, the axioms that make it the *unique* fair allocation (not just *a* reasonable one), and precisely what structural shortcut each SHAP variant exploits to make an exponential computation tractable.

## 5.1 The Shapley value formula, built up piece by piece

**Setup:** you have a set of n "players" (features), and a **value function** v(S) that tells you the payout (model output) achievable using any subset S of players (features) — v(∅) is the baseline (e.g., the model's average prediction with no feature information at all), and v({all features}) is the model's actual full prediction for one specific example.

**The marginal contribution of feature i, given a specific subset S not containing i:**

marginal contribution of i to S = v(S ∪ {i}) − v(S)

— "how much does the prediction change when feature i's information is added to whatever subset S already has?"

**The full Shapley value formula:**

φᵢ = Σ_{S ⊆ N\{i}} [ |S|! · (n − |S| − 1)! / n! ] · [ v(S ∪ {i}) − v(S) ]

This looks intimidating, but every piece has a plain-language reading:

- **The sum is over every possible subset S of the *other* features** (every subset not containing i) — every possible "context" feature i could be added into.
- **The term [v(S ∪ {i}) − v(S)]** is exactly the marginal contribution defined above — how much adding feature i changes the prediction, given that context S.
- **The weighting term |S|!·(n−|S|−1)!/n!** is the trickiest part, and it deserves its own explanation (§5.2).

## 5.2 Why that specific, strange-looking weighting term

**What the weighting term actually computes:** it's the probability that, if you picked a **uniformly random ordering** of all n features and added them one at a time, you would end up with **exactly the subset S already placed before feature i, and feature i placed immediately next**. In other words, the Shapley formula isn't just summing marginal contributions arbitrarily — it's computing the **exact probability-weighted average of feature i's marginal contribution, across every possible random ordering in which all n features could be revealed one at a time**, which is exactly the "average over every ordering" idea from the intuitive version of this material, now made precise.

**Why it's not simply "1 divided by the number of subsets":** different subset sizes |S| don't occur with equal probability across all n! random orderings — there are more distinct orderings that place a *medium-sized* set of features before feature i than orderings that place either a very small or very large set before it (for a everyday-sized n, most random shuffles don't happen to put, say, exactly 1 out of 10 features first) — the combinatorial weighting term correctly accounts for exactly how many of the n! total orderings correspond to each specific S, so that the final weighted sum genuinely equals "the average marginal contribution across all n! equally-likely random feature-revelation orders," not some other, differently-biased average.

## 5.3 The four Shapley axioms — why this formula is *the unique* fair solution

This is the part that lets you defend SHAP's guarantees confidently in an interview, rather than just describing the formula: the Shapley value isn't merely *a* reasonable way to split credit — it's **provably the only** allocation rule satisfying four natural fairness properties simultaneously.

- **Efficiency:** the sum of all features' Shapley values equals exactly v(N) − v(∅) — the total difference between the full prediction and the baseline. Nothing is left over, nothing is double-counted; every bit of the prediction gets attributed to some combination of features. (This is exactly the "additive" property from SHAP's name.)
- **Symmetry:** if two features i and j contribute identically to every possible subset (v(S∪{i}) = v(S∪{j}) for every S not containing either), they must receive exactly equal Shapley values. Two features that are functionally interchangeable in every context can't be arbitrarily assigned different importance.
- **Dummy (or "null player"):** if a feature never changes the prediction no matter what subset it's added to (v(S∪{i}) = v(S) for every S), its Shapley value must be exactly zero. A truly uninformative feature can't be assigned any nonzero credit, no matter how the calculation is sliced.
- **Additivity (linearity):** if you have two separate value functions v and v' (e.g., from two separate models, or a model's output split into two components), the Shapley value of the *combined* value function (v + v') for any feature equals the sum of that feature's Shapley values computed on v and v' separately. This lets Shapley-based explanations compose cleanly across model components.

**The theorem (Shapley, 1953):** these four properties, together, have **exactly one** function satisfying all of them simultaneously — the formula in §5.1. This is the precise sense in which Shapley values are "the" fair allocation, not merely "a" reasonable heuristic — any other allocation rule you might invent will necessarily violate at least one of these four properties.

**Why this matters practically, beyond just being a nice mathematical fact:** it's what lets you say, with real justification rather than just "it's a popular method," that SHAP values won't arbitrarily favor one of two functionally-identical features over the other (symmetry), won't assign blame to a feature that provably never mattered (dummy), and will always account for the entire prediction with nothing left unexplained (efficiency) — properties that LIME (Chapter 6), by contrast, does not guarantee.

## 5.4 The exponential cost problem, and what each SHAP variant does about it

**The problem:** computing the exact Shapley value requires summing over every possible subset S of the other n−1 features — 2^(n−1) subsets. For even a modest 20 features, that's over 500,000 subsets *per feature*, and you'd need this for every feature, for every single prediction you want to explain. Fully exact computation is infeasible beyond a handful of features.

**KernelSHAP:** a fully model-agnostic approximation. It samples a manageable number of subsets S (rather than enumerating all 2^(n−1)) and, cleverly, formulates the Shapley-value computation as a specially-weighted linear regression problem — fitting a linear model on the sampled subsets' value-function outputs, where the specific choice of sample weights is what makes the resulting linear regression's coefficients converge to genuine Shapley value estimates as you sample more subsets. Works for any model, but is the slowest variant, since it doesn't exploit any model-specific structure at all.

**TreeSHAP:** exploits the specific structure of a decision tree to compute *exact* Shapley values (not merely an approximation) in time that's low-order polynomial in the number of features and tree size, rather than exponential. The key structural shortcut: a tree partitions the feature space into a fixed set of regions (leaves), and by tracking, for every possible subset, which leaf an example would land in, the algorithm can efficiently aggregate the needed subset-conditional expectations without ever explicitly enumerating all 2^(n−1) subsets one by one — it reuses the tree's own branching structure to do this bookkeeping far more efficiently than a naive brute-force sum would.

**DeepSHAP:** exploits neural network structure specifically, adapting an idea from a related attribution method (DeepLIFT) that propagates contribution scores backward through the network's layers, combined with Shapley-consistent weighting — approximating Shapley values by decomposing the network's computation layer by layer rather than treating the whole network as an opaque black box the way KernelSHAP does.

**The practical takeaway:** always use the model-specific variant when one exists (TreeSHAP for any tree-based model, DeepSHAP for neural networks) — they're both faster and, in TreeSHAP's case, exactly correct rather than approximate. Reserve KernelSHAP for genuinely black-box models with no specialized variant available (e.g., an arbitrary scikit-learn pipeline or an external API you can only query for predictions).

## 5.5 SHAP interaction values — going beyond single-feature attribution

**The idea:** ordinary SHAP values attribute the prediction to individual features, one number per feature. **SHAP interaction values** extend this to attribute part of the prediction to *pairs* of features jointly — capturing "how much of the prediction is explained by feature A and feature B acting together, beyond what either explains alone." This is computed (for TreeSHAP specifically, where it's tractable) using a two-feature generalization of the same Shapley-value machinery, splitting each feature's ordinary SHAP value into a main effect (that feature acting alone) plus interaction terms (that feature's contribution that specifically depends on another feature's value).

**Why this matters practically:** a feature can show only modest ordinary SHAP importance while participating in a strong interaction effect — e.g., `age` might show a modest main effect on its own, but a strong interaction with `income` (young-and-high-income individuals get treated very differently than old-and-high-income individuals) that wouldn't be visible from ordinary single-feature SHAP values alone. Interaction values are how you'd actually surface and confirm a suspected interaction effect rather than just guessing at one from a domain hunch.

## 5.6 Quick self-check before Chapter 6

- Can you explain, in your own words, what the strange combinatorial weighting term in the Shapley formula is actually computing a probability of?
- Given the four axioms, can you explain why "just split credit equally among all features that were used at least once" would violate at least one of them?
- Can you name which SHAP variant you'd reach for given a specific model type (tree-based, neural network, arbitrary black-box), and why each one is faster than brute-force enumeration?

---

**Next: Chapter 6 — LIME and Other Local Surrogate Methods**, covering the full local-fit procedure, the sampling/kernel-weighting choices that make LIME somewhat unstable across repeated runs, and Anchors as a rule-based alternative.
