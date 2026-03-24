# LIME — Interview Q&A

> 35 questions from fundamentals to senior level. Pure concepts — no code.

---

## 1. Fundamentals

**Q1. What is LIME and what problem does it solve?**

LIME (Local Interpretable Model-Agnostic Explanations) explains a single prediction from any black-box model by fitting a simple interpretable linear model in the local neighbourhood of that prediction. It solves the problem of explaining *why* a model made a specific decision, for any model type, using only the model's predict function.

---

**Q2. What is LIME's core assumption, and when does it break?**

LIME assumes the model is locally linear around the point being explained — that is, in a small neighbourhood around x*, the model's behaviour can be approximated by a linear function. This breaks when the model has sharp discontinuities near x* (e.g., a decision boundary runs right through the neighbourhood), or when the model has strong feature interactions that a linear model structurally cannot represent (e.g., prediction is high only when BOTH features exceed a threshold).

---

**Q3. Walk me through the LIME algorithm step by step.**

1. Generate N perturbed samples near the sample x* being explained (how depends on data type — random feature removal for tabular, word removal for text, superpixel greying for images).
2. Run the black-box model on each perturbed sample to get predictions.
3. Weight each perturbed sample by its similarity to x* using an exponential kernel: `w = exp(−d² / σ²)`, so nearby samples matter more.
4. Fit a weighted sparse linear model (Lasso) on the (perturbed samples, predictions) pairs using the weights from step 3.
5. The Lasso coefficients are the explanation — positive means the feature pushed the prediction up locally, negative means it pulled down.

---

**Q4. What does the kernel width σ control in LIME?**

σ controls the size of the neighbourhood LIME considers "local." Small σ: only samples very close to x* get significant weight — very local but with fewer effective samples, leading to high variance. Large σ: samples far from x* also get weight — more samples but the "local" approximation covers a larger region where the model may not be linear, leading to bias. There is no principled data-driven way to choose σ; the default is a heuristic `sqrt(n_features) × 0.75`.

---

**Q5. Why does LIME use a sparse linear model (Lasso) instead of a regular linear regression?**

High-dimensional data can have many features, and a full linear model would give non-zero coefficients to all of them — hard to interpret. Lasso (L1 regularisation) forces most coefficients to exactly zero, producing a sparse explanation with only K important features. This matches human cognitive limits — a person can understand "these 5 features drove the prediction" better than "here are 50 coefficients."

---

## 2. Data Types

**Q6. How does LIME differ for tabular, text, and image data?**

The core algorithm is identical. Only the perturbation strategy changes:

- **Tabular:** Features are represented as binary (present=actual value / absent=random draw from training distribution). Perturbations randomly flip features between present and absent.
- **Text:** Words are the units. Perturbations randomly remove words from the input text. "Absent" = word omitted from the string.
- **Image:** The image is first segmented into superpixels using SLIC or similar. Perturbations randomly grey out subsets of superpixels. "Absent" = superpixel replaced with grey/mean colour.

---

**Q7. Why is text LIME more reliable than tabular LIME?**

Text LIME's perturbations — removing words from a sentence — create valid, realistic inputs. A sentence with a word removed is a real sentence. The interpretable unit (a word) is meaningful to humans. In tabular LIME, "removing a feature" means replacing it with a random value from the training distribution, creating artificial hybrid rows that may not correspond to any realistic data point. This introduces a bias not present in text LIME.

---

**Q8. Why do image LIME explanations use superpixels instead of individual pixels?**

Two reasons: (1) With ~50,000 pixels in a typical image, the perturbation space is astronomically large and completely intractable. (2) Individual pixels are not interpretable — a single pixel means nothing to a human. Superpixels group spatially adjacent, visually similar pixels into ~50–100 regions that correspond to meaningful image parts (a dog's ear, the background, a foreground object), making the explanation interpretable.

---

**Q9. What is the downside of using superpixels for image LIME?**

The segmentation algorithm (SLIC or similar) may create superpixels that cut across semantically meaningful boundaries. For example, a superpixel might include half a dog's ear and part of the background. If the relevant region is split across multiple superpixels, the explanation incorrectly distributes credit. The quality of image LIME explanations depends substantially on segmentation quality.

---

## 3. Instability

**Q10. What is LIME instability, and why is it a serious problem?**

LIME instability means that running LIME multiple times on the exact same sample with the exact same model produces different — sometimes contradictory — explanations. A feature might appear positive in one run, negative in another, or not appear at all. This makes LIME unreliable for consequential decisions: two stakeholders could receive opposite explanations for the same prediction depending on which random run they see.

---

**Q11. What are the three sources of variance in LIME?**

1. **Random perturbation sampling** — each run generates a different set of N random samples. The regression is fitted on a different dataset each time.
2. **Random replacement values** — when a feature is "turned off," its replacement is drawn randomly from the training distribution. Different draws change the effective input distribution.
3. **Lasso regularisation path sensitivity** — features near the regularisation threshold appear in some runs and are excluded in others, depending on the random sample.

---

**Q12. How would you measure LIME's instability in practice?**

Run LIME K times (K=10–20) on the same sample. For each feature, compute the coefficient across K runs. Instability metrics:
- Standard deviation of the coefficient / mean coefficient = coefficient of variation (CV). CV > 0.3 indicates unreliable coefficient magnitude.
- Rank correlation of feature importance between runs — low correlation means even the ordering is unstable.
- Sign flip rate — fraction of runs where a feature's sign differs from the majority sign.

---

**Q13. Increasing N (more perturbations) reduces LIME's variance. Does it also reduce bias?**

No. Increasing N reduces variance (more samples → more stable regression fit) but does not reduce bias. The bias comes from LIME's assumption that the model is locally linear, and from using an arbitrary neighbourhood size σ. Neither of these is fixed by adding more perturbations. More N makes LIME more consistently wrong (if the linear approximation is bad), not more correct.

---

**Q14. A colleague runs LIME and gets a negative coefficient for income. A second run gives a positive coefficient. They're alarmed. What do you tell them?**

This is LIME's instability problem. The two runs sampled different random perturbations of income, leading to different apparent local relationships. This is not a fluke — it's a known fundamental limitation of tabular LIME. Options: (1) run LIME 10+ more times and check the average sign and magnitude, (2) compute the coefficient CV to quantify instability, (3) switch to SHAP if the model is tree-based (deterministic, exact), or (4) at minimum, report that the explanation for this sample is unreliable.

---

## 4. Faithfulness and Fidelity

**Q15. What is local fidelity in LIME, and how is it measured?**

Local fidelity measures how well LIME's local linear model g approximates the black-box model f in the neighbourhood of x*. It is measured as R² between g's predictions and f's predictions on the N weighted perturbed samples. High R² → the linear model is a good local approximation → explanation is trustworthy. Low R² → the linear model poorly approximates the black box → explanation may be misleading.

---

**Q16. You compute a LIME explanation and the local R² is 0.35. Should you trust the explanation?**

No. An R² of 0.35 means LIME's linear model only explains 35% of the variance in the black-box predictions in the neighbourhood of x*. The explanation is largely fitting noise, not the model. Possible causes: the model has a sharp discontinuity or strong interactions near this point; σ is too large. You should either increase N, decrease σ, or accept that LIME cannot faithfully explain this prediction and use a different method.

---

**Q17. LIME doesn't satisfy the efficiency axiom (SHAP's Guarantee 1). What practical consequence does this have?**

SHAP's efficiency axiom guarantees that SHAP values sum to f(x) − E[f(X)], providing a built-in verification test. LIME has no such guarantee — the coefficients don't sum to any meaningful total. This means you cannot verify whether a LIME explanation is correct. A LIME explanation that looks clean and sensible might be entirely wrong, with no way to detect it from the explanation itself. This is a fundamental reliability disadvantage versus SHAP.

---

## 5. Bias-Variance

**Q18. LIME is described as "model-agnostic." Does being model-agnostic create any problems?**

Yes. Being model-agnostic means LIME only uses the model's predict function — it knows nothing about the model's structure. This means LIME cannot exploit structure that would improve accuracy. For a tree model, TreeSHAP exploits the tree's internal path statistics to compute exact explanations efficiently. LIME, by treating it as a black box, throws away all that structure and produces an approximation that is both less accurate and less stable. Model-agnosticism is a feature (works universally) but also a liability (wastes information when it's available).

---

**Q19. What is the bias-variance trade-off in choosing the kernel width σ?**

Smaller σ → smaller neighbourhood → more locally linear (less bias) but fewer effective samples (higher variance). Larger σ → larger neighbourhood → more samples (less variance) but the region may not be locally linear (higher bias). The optimal σ minimises total error (bias² + variance), which depends on the local geometry of the model and cannot be determined analytically. LIME uses a heuristic default that may not be optimal for any specific problem.

---

## 6. LIME vs SHAP and Other Methods

**Q20. For a Random Forest model, should you use LIME or SHAP? Why?**

SHAP (TreeSHAP) always. TreeSHAP is exact (not an approximation), deterministic (same result every run), theoretically grounded (satisfies all 4 Shapley axioms), and faster than LIME for tree models. LIME for a tree model is approximate, unstable, slower, and has no efficiency check. There is no scenario where LIME is preferable to TreeSHAP for a tree model.

---

**Q21. For a text sentiment classification model (BERT-based), should you use LIME or SHAP?**

LIME is often the more practical choice here. Text LIME's word-removal perturbations produce realistic inputs and relatively stable explanations. SHAP for transformer models is possible (using KernelSHAP or the `shap.Explainer` with a masker), but is significantly slower and more complex to set up. For a quick, interpretable "which words drove the sentiment?" answer, LIME is a reasonable pragmatic choice — just run it multiple times to check stability.

---

**Q22. A business user asks: "Which features are most important across all customers?" Can you use LIME to answer this?**

No. LIME produces local explanations for individual predictions. You cannot reliably aggregate LIME coefficients across samples to get a global feature ranking. The coefficients from different samples are not on comparable scales, use different perturbation distributions, and may be unstable individually. For global feature importance, use permutation importance or SHAP's mean |φ| instead.

---

**Q23. LIME and SHAP both claim to explain why a model made a prediction. If they give different answers, which is right?**

They answer slightly different questions. LIME finds the best-fitting local linear model in a neighbourhood and reports its coefficients — it is an approximation of the model's local behaviour. SHAP computes the exact average marginal contribution of each feature across all coalitions — it satisfies formal fairness axioms. When they differ:
- If the model is highly non-linear near x*, LIME's linear approximation is poor (check R²) and SHAP is more reliable.
- If features are correlated, both struggle but SHAP handles it more gracefully.
- If SHAP is from TreeSHAP (exact), it is strictly more reliable than LIME for that prediction.

---

## 7. Senior Level

**Q24. LIME uses Lasso for sparsity. What are the risks of this regularisation choice?**

Lasso is sensitive to correlated features — when two features are correlated, it arbitrarily picks one and zeroes out the other. This means an important feature can be excluded from the explanation not because it's unimportant, but because a correlated feature happened to absorb its coefficient. A user told "feature A is important, feature B is not" might not realise B was excluded due to correlation with A, not due to irrelevance. This is compounded by LIME's instability: different runs may include A in one run and B in another.

---

**Q25. You're explaining a loan denial to a customer under GDPR's "right to explanation." You use LIME. What are the risks?**

Several serious risks: (1) Instability — the explanation the customer receives could be different from an explanation generated moments later, undermining consistency. (2) No self-verification — you cannot confirm the explanation is correct without running SHAP or another method. (3) Lasso exclusions — important features may be absent from the top-K explanation, making the explanation incomplete. (4) Tabular LIME's perturbations create unrealistic hybrid rows — the explanation may reflect model behaviour on inputs that don't correspond to any real scenario. For GDPR compliance, TreeSHAP (for tree models) is significantly more defensible.

---

**Q26. Explain why LIME's instability is particularly dangerous when features are correlated.**

When features A and B are correlated, perturbations that "turn off" A create rows where A takes a random value but B retains its actual value — an unrealistic combination. Sometimes the random A value happens to be high (preserving the A-B correlation by coincidence), sometimes low (violating it). This inconsistency creates large variance in A's apparent contribution across runs. Combined with Lasso's tendency to pick one of two correlated features arbitrarily, the resulting explanation is both unstable and potentially misleading about which feature is truly driving the prediction.

---

**Q27. A researcher proposes using LIME to compare feature importance across two different model versions to evaluate which model "makes sense." What would you caution?**

LIME cannot reliably be used for this comparison. First, LIME's instability means the comparison would reflect random sampling noise as much as true model differences. Second, LIME produces local explanations — comparing at one sample does not generalise to the full model. Third, the local fidelity (R²) may differ between models, meaning you might be comparing a good approximation of one model to a poor approximation of the other. A more reliable approach: compute permutation importance or mean |SHAP| for both models and compare those global metrics.

---

**Q28. What is the relationship between LIME and Anchors?**

Anchors (Ribeiro et al., 2018) is the follow-up to LIME from the same authors, addressing LIME's instability and imprecision. While LIME gives linear coefficients (soft, every feature contributes a little), Anchors gives if-then rules with precision guarantees: "IF debt_ratio > 0.7 AND income < $50k THEN prediction = high_risk with 95% confidence." Anchors produces sparser, more interpretable explanations, are more stable (rule search is more principled), and include precision/coverage metrics (unlike LIME which has no equivalent reliability guarantee). The trade-off: Anchors are harder to compute and may not always find a high-precision anchor.

---

## Summary: What Interviewers Test at Each Level

```
Junior
  ├── Explain the LIME algorithm step by step
  ├── How does LIME differ for tabular, text, and image data?
  └── What is the kernel width and what does it control?

Mid
  ├── What is LIME instability and what causes it?
  ├── What is local fidelity (R²) and when should you be worried?
  ├── When would you use LIME vs SHAP?
  └── Why can't you aggregate LIME coefficients for global importance?

Senior
  ├── LIME for tabular vs text — which is more reliable and why?
  ├── Why is LIME dangerous for correlated features?
  ├── How would you measure and report LIME's instability?
  └── GDPR explanation using LIME — what are the risks?
```
