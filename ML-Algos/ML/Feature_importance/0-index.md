# Feature Importance & Explainability — Master Index


---

## Folder Map

| File | What It Covers | Status |
|---|---|---|
| `1_Importance_vs_Explainability.md` | Core distinction, full taxonomy, every method compared, SHAP numerics, LIME, PDP, ALE, counterfactuals, anchors, Rashomon, bias-variance table | ✅ |
| `2_Impurity_Importance.md` | Gini/entropy math, high-cardinality bias, training-set bias, MDI vs MDA, numerical worked example | ✅ |
| `3_Permutation_Importance.md` | Full algorithm, numerical example, bias-variance, correlated features, train vs test, n_repeats, negative importances | ✅ |
| `3a_Permutation_Importance_QA.md` | 45 interview Q&As, fundamentals → senior follow-ups | ✅ |
| `4_SHAP.md` | TreeSHAP internals, all variants, every plot, marginal vs conditional vs interventional | 🔲 |
| `4a_SHAP_QA.md` | SHAP interview Q&A | 🔲 |
| `5_LIME.md` | Algorithm, instability, tabular vs text vs image LIME | 🔲 |
| `5a_LIME_QA.md` | LIME interview Q&A | 🔲 |
| `6_PDP_and_ALE.md` | PDP extrapolation problem, ALE fix, 2D interaction plots | 🔲 |
| `7_Global_vs_Local.md` | Surrogate models, anchors, counterfactuals deep dive | 🔲 |
| `8_Correlated_Features.md` | Per-method failure modes, VIF, grouped permutation, fix strategies | 🔲 |
| `9_Bias_Variance_in_XAI.md` | Rashomon set, overfitting effects per method, train vs test, sample size | 🔲 |
| `10_Master_QA.md` | Cross-topic questions that connect concepts — the senior filter | 🔲 |

---

## The One-Page Concept Map

```
                         ML Interpretability
                               │
              ┌────────────────┴─────────────────┐
              │                                   │
      IMPORTANCE                          EXPLAINABILITY
  "Which features matter?"           "Why this prediction?"
              │                                   │
    ┌─────────┴──────────┐             ┌──────────┴──────────┐
    │                    │             │                      │
 Model-Specific    Model-Agnostic    GLOBAL               LOCAL
    │                    │         (whole model)        (one sample)
    ├─ Impurity/Gini     ├─ Permutation  │                   │
    ├─ Coefficients      ├─ Drop-Column  ├─ PDP          ├─ SHAP
    └─ Attention         └─ SHAP global  ├─ ALE          ├─ LIME
                                         ├─ Surrogate    ├─ Counterfactual
                                         └─ SHAP summary └─ Anchors
```

---

## Quick-Select Decision Guide

### "What do I need right now?"

```
Goal                                             → Best Method
─────────────────────────────────────────────────────────────────
Rank features for selection / dropping           → Permutation Importance (test set)
Understand average effect of feature X           → ALE (correlated) or PDP (uncorrelated)
Explain one prediction to a business user        → SHAP waterfall or Counterfactual
Explain one prediction for debugging             → SHAP force plot
Generate a rule: "prediction = X because..."    → Anchors
"What change flips this prediction?"             → Counterfactual
Compare feature use across the whole model       → SHAP beeswarm / summary plot
Fast approximate importance (tree model)         → Gini importance (know its biases)
Detect data leakage                              → Compare train vs test permutation importance
Check if features are correlated (pre-step)      → VIF matrix → use grouped permutation
Regulatory / GDPR explanation                    → Counterfactual or SHAP waterfall
Fairness audit per subgroup                      → SHAP per subgroup
Model is a neural network                        → Integrated Gradients or GradSHAP
Model is a tree (RF/XGB/LGBM)                    → TreeSHAP (exact, fast)
Model is linear                                  → Standardised coefficients
Model is any black box                           → KernelSHAP or Permutation Importance
```

---

## The Bias Cheatsheet — Every Method, Every Bias

| Method | Bias Source | Variance Source | Fix |
|---|---|---|---|
| Impurity/Gini | High-cardinality features; training-set only | Low (deterministic) | Use permutation importance instead |
| Permutation | Correlated features; train set if overfit | 1/n_repeats | Test set; grouped permutation |
| Drop-Column | Retrained model compensates | Model training variance | Accept it; still best for correlations |
| PDP | Extrapolation over unrealistic inputs | Low | Use ALE instead |
| ALE | Bin width (coarse = loses detail) | Bin sample count | Tune bin count |
| LIME | Neighbourhood definition; linear approximation | High (random sampling) | Run 5–10x; prefer SHAP |
| KernelSHAP | Coalition sampling | 1/n_coalitions | Increase nsamples |
| TreeSHAP | None (exact) | None (deterministic) | — |
| Counterfactuals | Distance metric choice | Optimisation instability | Use DiCE for diverse set |
| Anchors | Precision threshold choice | Rule search randomness | Tune beam width |

---

## The Correlation Red Flag Protocol

Before running any importance or explainability method, always do this first:

```
Step 1: Compute correlation matrix
Step 2: Flag all pairs with |r| > 0.7
Step 3: Compute VIF for all features
        VIF > 5   → moderate concern
        VIF > 10  → severe; importance estimates unreliable
Step 4: For correlated groups → use grouped permutation importance
Step 5: For explainability    → use ALE (not PDP), TreeSHAP with caution
Step 6: For feature selection → use drop-column importance, not permutation
```

---

## What Interviewers Actually Test — Topic by Topic

### Junior Level
- What is permutation importance and how does it work?
- What is the difference between Gini importance and permutation importance?
- What is SHAP? What property makes it unique?
- What is the difference between global and local explainability?

### Mid Level
- Why should you use the test set (not train) for permutation importance?
- What happens to SHAP / permutation importance when features are correlated?
- When would you use ALE instead of PDP?
- Why is LIME unstable? What would you do about it?
- What does a negative permutation importance mean?

### Senior Level
- Explain the Rashomon problem and its implications for feature importance.
- What is the difference between marginal, conditional, and interventional SHAP?
- Walk me through computing a Shapley value by hand for a 3-feature model.
- Why does drop-column importance handle correlations better than permutation importance?
- How do overfitting and underfitting affect permutation importance estimates on train vs test?
- LIME says feature A is most important for this prediction; SHAP says feature B. Why might they disagree?
- Can feature importance be used to make causal claims? Under what conditions?

---

## The Non-Negotiable Things to Know Cold

```
1.  Permutation importance → TEST SET always (unless debugging leakage)
2.  Gini importance is biased toward high-cardinality features
3.  Correlated features → both appear unimportant under permutation
4.  SHAP efficiency axiom: all φⱼ sum to f(x) − E[f(x)]
5.  TreeSHAP is exact; KernelSHAP is an approximation
6.  LIME is high-variance — never trust a single run on tabular data
7.  PDP extrapolates; ALE does not
8.  Rashomon: many equally good models → different importance rankings
9.  Importance ≠ causality (ever)
10. Explainability explains the MODEL, not the data-generating process
```

---

## How the Files Connect

```
Start here:  1_Importance_vs_Explainability.md   ← big picture taxonomy
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
  2_Impurity.md   3_Permutation.md     4_SHAP.md
  (model-specific  (model-agnostic     (best method
   baseline)        workhorse)          for trees)
          │               │               │
          └───────────────┴───────────────┘
                          │
                          ▼
               8_Correlated_Features.md    ← how ALL methods break
                          │
                          ▼
               9_Bias_Variance_in_XAI.md  ← why estimates are unreliable
                          │
                          ▼
                  10_Master_QA.md          ← interview simulation
```
