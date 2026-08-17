# Applied / Advanced Topics — Master Notes

## 1. SHAP Values — Explaining One Specific Prediction

Feature importance (earlier chapters) answers "across all houses, which features matter most on average?" SHAP answers a different, more specific question: **"for this one house, why did the model predict $420,000 specifically?"**

**Plain idea:** start from the average prediction across all houses (say, $300,000). SHAP breaks down the gap to this specific house's prediction ($420,000, a $120,000 gap) into a per-feature contribution — "+$80,000 because this house is unusually large," "+$50,000 for location," "−$10,000 for age" — and these add up exactly to the total gap.

**Where the idea comes from:** SHAP borrows Shapley values from game theory — originally invented to fairly split a group payout among people who contributed differently in combination with each other. SHAP treats each *feature* as a "player" and the *prediction* as the "payout." The "fair" part means it considers every possible order features could be "revealed" in and averages each feature's contribution across all those orderings — more principled than simpler tricks (like just reading one path down one tree), because it accounts for features that matter differently depending on which other features are already known.

**Why is this needed if we already have feature importance?** Overall importance can't tell you why one *particular* unusual house (a small house in an expensive neighborhood) got the price it did. SHAP is the one-house-at-a-time tool — critical in real deployments, e.g. explaining to a specific customer why their loan, insurance quote, or valuation came out the way it did.

**Practical note:** computing exact Shapley values naively means checking every possible feature ordering — expensive fast as feature count grows. `TreeSHAP` (built specifically for tree models) computes exact SHAP values efficiently by exploiting tree structure directly — this is why SHAP is used so heavily with tree models specifically, more than with model types where only slower approximations exist.

---

## 2. Monotonic Constraints

**The problem:** a house-price model might end up predicting a 2,100 sq ft house is worth *less* than a 2,000 sq ft house in the same neighborhood, just from a quirky pattern in the training data. Obviously wrong by common sense (bigger should never be worth less, all else equal) — even if it technically reduced training error slightly.

**The fix:** a monotonic constraint tells the model "the prediction must never decrease as this feature increases" (or never increase, depending on direction) — e.g., forcing price to be non-decreasing in size. XGBoost, LightGBM, and CatBoost all support this directly (e.g., XGBoost's `monotone_constraints`).

**How it works:** during split search, the algorithm is only allowed to consider splits that keep the constraint satisfied — it can't create a split where "bigger size" ends up predicting a lower price than "smaller size." Enforced structurally during tree-building, not patched afterward on finished predictions.

**When it's worth using:** when domain knowledge says a relationship should go one direction (more square footage → higher price; more experience → higher salary) and you want that guaranteed — both for common-sense correctness and often for regulatory/trust reasons in sensitive applications (lending, insurance, medical risk scoring), where a model behaving backwards on one feature is hard to justify to a regulator or customer, even if rare and only slightly hurting overall accuracy.

---

## 3. Native Missing-Value Handling: XGBoost/LightGBM vs. Plain sklearn

| | Plain sklearn trees | XGBoost / LightGBM |
|---|---|---|
| Handles missing values? | ❌ No — must impute yourself first | ✅ Yes, natively |
| How | — | At each split, learns which direction (left/right) missing values should default to, based on what performs best on training data |
| Consistency across splits | — | Different splits can send missing values in *different* directions — the best default can genuinely differ node to node |

**Why does letting the model learn this beat manually imputing first?** Manually filling missing "lot size" with the average assumes missingness is random — but maybe houses with missing lot size are systematically different (older, rural, never formally surveyed). Forcing them all to look like "the average house" throws that pattern away. Letting the tree decide the best direction for missing values based on what actually predicts price well can capture "missingness itself is informative" instead of erasing it.

**Example:** if lot size is missing specifically for older, rural, cheaper houses, XGBoost might learn to route "missing lot size" down the same branch as "small lot size" — picking up on the real pattern that missingness correlates with lower price, something a naive "fill in the average" approach would miss.

---

## 4. Quick Q&A (general)

**Q: What's the difference between feature importance and SHAP values?**
A: Feature importance tells you what matters on average across the whole dataset. SHAP tells you, for one specific prediction, exactly how much each feature pushed that one prediction up or down from the average — a per-case explanation, not an overall ranking.

**Q: Why would you ever restrict what a model is allowed to learn, like with a monotonic constraint?**
A: Because pure accuracy isn't the only goal — sometimes the model needs to match common sense or regulatory expectations (e.g., "more income should never predict a worse credit score, all else equal"), even if letting it learn a slightly weird, technically-more-accurate pattern from noisy data would squeeze out marginally lower training error.

**Q: Why does XGBoost handle missing data better than just filling in the average yourself?**
A: Missingness is often not random — it can carry real signal (older/rural houses more likely to have missing lot-size records) — and letting the model learn the best handling, potentially differently at different splits, captures that signal instead of erasing it the way one blanket imputed value would.

---

## 5. Google MLE Interview Q&A

**Q: You're asked to explain why TreeSHAP is efficient while naive Shapley value computation isn't, in enough detail to satisfy a rigor-focused interviewer. What's the actual mechanism?**
A: Naive Shapley computation needs to evaluate the model's prediction under every possible subset (or ordering) of "known" vs. "unknown" features — exponential in the number of features, since there are $2^p$ subsets for $p$ features. TreeSHAP avoids this by exploiting the fact that a tree's prediction only depends on which path a sample takes through the tree's actual splits — it can compute exact expected contributions by tracking, for each node, the proportion of training data that would flow through it under every feature subset simultaneously, using the tree's structure to share computation across subsets rather than evaluating each one independently. This turns an exponential problem into something polynomial in tree size (roughly $O(TLD^2)$ across $T$ trees, $L$ leaves, $D$ depth, per the original TreeSHAP paper), which is why SHAP is practical for tree ensembles specifically but needs slower model-agnostic approximations (like KernelSHAP) for arbitrary model types.

**Q: A model with a monotonic constraint on "years of experience → salary" shows slightly worse training accuracy than the unconstrained version. How would you defend shipping the constrained model to a skeptical stakeholder focused purely on metrics?**
A: Frame the accuracy loss as buying something the metric doesn't capture: a guarantee of directional correctness the unconstrained model doesn't have. The unconstrained model's "better" training accuracy may partly come from fitting noise where a specific segment of the training data happened to show experience negatively correlating with salary — a pattern that's very unlikely to generalize and that the model would be actively wrong to rely on. Beyond the accuracy framing, there's a trust/risk cost to an unconstrained model: if it's ever caught predicting that more experience lowers predicted salary for some real employee, that's a much harder conversation with legal, compliance, or the employee themselves than a marginally lower validation metric — the constraint is buying insurance against exactly that scenario, not just leaving accuracy on the table for no reason.

**Q: Design question — you're building a lending risk model where several features can be missing for different reasons (some missing-at-random, some missing because a program only recently started collecting them). Would you rely on XGBoost's native missing-value handling uniformly, or would you do something more careful?**
A: Native handling works well for the missing-at-random case and even for the "missingness is itself informative" case, since it lets the split search find the best default direction per node without you having to specify anything. But it can silently do the wrong thing when missingness has *different causes for different features* — e.g., if "recently started collecting" missingness is really a proxy for "this loan application predates the program" (a time/cohort signal), letting the tree treat it as an ordinary predictive missingness pattern conflates a data-collection artifact with a genuine risk signal, which could bake a spurious or even legally risky correlation into the model. The more careful approach: explicitly separate the two missingness mechanisms — e.g., add an explicit "predates program" indicator feature for the cohort-driven case, so the tree isn't forced to infer that distinction implicitly through native missing-value handling alone, and reserve native handling for features where missingness is genuinely just a property of the individual record.

---

## 6. Apple MLE Interview Q&A (on-device / practical flavor)

**Q: You want to explain individual on-device predictions to users (e.g., "why did this feature flag your battery usage as unusual?") without sending data to a server. What does computing SHAP on-device actually require, and is it feasible?**
A: TreeSHAP's efficiency comes from walking the tree structure itself rather than needing external calls, so computing SHAP values for a single prediction is a self-contained computation over the already-on-device model — feasible in principle without any server round-trip, which fits an on-device-privacy requirement well. The practical constraint is closer to compute/latency: TreeSHAP's cost scales with the ensemble's total size (roughly tree count × leaves × depth²), so a large boosted ensemble that would produce good SHAP explanations server-side may need to be pruned smaller, or explanations may need to be computed for a subset of predictions rather than every single one, to fit on-device latency and battery budgets — the same $M\times$-cost tension that shows up for plain inference applies again here, just for explanation instead of prediction.

**Q: A monotonic constraint makes sense for a feature like "years of experience → predicted salary" in a clean domain example. Give an on-device example where a monotonic constraint would help avoid a genuinely bad user-facing outcome, and explain the mechanism.**
A: Consider a battery-health or wear-estimate model: predicted battery degradation should never *decrease* as charge-cycle count increases, all else equal — but a model trained on noisy real-world data (measurement noise, inconsistent charging patterns across the training population) could easily learn a small dip somewhere in that curve just from sampling noise. Without a constraint, this could produce something obviously wrong and confusing user-facing ("your battery health went from 87% predicted degradation risk back down to 84%?"), undermining trust in the feature. A monotonic constraint on charge-cycle count forces the split search to never create that internal contradiction, guaranteeing the feature never contradicts basic physical intuition the way an unconstrained model technically could, even if the constraint costs a small amount of measured accuracy.

**Q: If sklearn's trees don't handle missing values, and you're training a model that will need to run inference on-device where a sensor might legitimately be temporarily unavailable, how does the choice between "impute upstream" and "use a native-missing-value library" change your on-device pipeline design?**
A: With "impute upstream" (sklearn-style), the imputation logic itself has to ship as part of the on-device pipeline and must be kept in exact sync with whatever imputation ran at training time — any drift between the two (a different default, a changed fallback rule) causes a silent training/serving mismatch, as covered in the Decision Tree Fundamentals notes' missing-data discussion. With native handling (XGBoost/LightGBM), the model itself encodes the "what to do when this sensor is missing" logic as part of its learned structure, so there's one less separate component to keep synchronized between training and on-device code — a real engineering simplification, though it does mean the on-device runtime needs an inference library that supports the same native-missing-value semantics the model was trained with, not just a generic "walk the tree" implementation.

---

**One-line summary to remember:** *SHAP explains one prediction by fairly splitting credit among features (Shapley values, made tractable for trees via TreeSHAP) → monotonic constraints trade a little accuracy for guaranteed common-sense/regulatory-safe direction, enforced during split search itself → XGBoost/LightGBM learn the best default direction for missing values per split, capturing "missingness is informative" instead of erasing it the way manual imputation does.*
