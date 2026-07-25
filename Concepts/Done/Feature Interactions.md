# Identifying Feature Interactions — Interview Notes

## 1. Intuition First

An interaction exists when **the effect of one feature on the target depends on the level of another feature** — not just that two features are correlated with each other.

Classic example: drug dosage effect depends on whether patient is male/female. Marketing spend's effect on sales depends on whether it's holiday season. Price sensitivity depends on income bracket. In each case, you can't describe "the effect of X" with a single number — it changes depending on another variable.

**Critical distinction to state upfront in an interview:** correlation between two features is not the same as interaction. Correlation = X1 and X2 move together. Interaction = the *effect* of X1 on Y changes depending on X2's value. You can have zero correlation between two features and still have a strong interaction between them (e.g., a checkerboard pattern).

---

## 2. Formula (Linear Model Framing)

Baseline additive model: `y = b0 + b1·x1 + b2·x2 + ε`
Here the effect of x1 on y is always `b1`, regardless of x2 — that's the "no interaction" assumption.

Interaction model: `y = b0 + b1·x1 + b2·x2 + b3·(x1·x2) + ε`

Now the effect of x1 on y is: `∂y/∂x1 = b1 + b3·x2` — **the slope of x1 depends on the value of x2**. That's the mathematical definition of an interaction: a cross-derivative term is non-zero.

---

## 3. Worked Numerical Example

Model: `sales = 10 + 2·ad_spend + 1·is_holiday + 3·(ad_spend·is_holiday)`

- Non-holiday (is_holiday=0): `sales = 10 + 2·ad_spend` → each $1 of ad spend adds 2 units of sales
- Holiday (is_holiday=1): `sales = 10 + 2·ad_spend + 1 + 3·ad_spend = 11 + 5·ad_spend` → each $1 of ad spend now adds **5** units of sales

The interaction coefficient (b3=3) tells you ad spend is **2.5x more effective during holidays**. An additive-only model would report a single "average" ad_spend effect and completely miss this — this is exactly why interaction detection matters for both accuracy and business insight.

---

## 4. Methods to Detect Interactions

### A. Domain knowledge / hypothesis-driven (start here)
Always the first move: does business logic suggest an interaction? (e.g., "does discount effectiveness plausibly differ by customer segment?"). Cheapest and most interpretable method — test the specific hypothesis rather than searching blindly.

### B. Visual: interaction plots
Plot y vs x1, with separate lines for different levels/bins of x2. **Parallel lines → no interaction. Non-parallel/crossing lines → interaction.** This is the single fastest, most intuitive diagnostic and should be your go-to first check for any suspected pair.

### C. Statistical: ANOVA / t-test on the interaction coefficient
In a linear model, fit both the additive model and the model with the interaction term, then test whether `b3` is significantly different from zero (t-test on the coefficient, or F-test comparing nested models). Standard, well-understood, but only captures the *linear* interaction form (x1·x2) — misses nonlinear interaction shapes.

### D. Partial Dependence Plots (2D) and ICE plots
For any model (not just linear), plot the model's predicted output as a 2D surface over (x1, x2), holding other features at their observed distribution. **If the surface isn't simply "additive" (i.e., you can't decompose it into f(x1)+g(x2)), there's an interaction.** ICE (Individual Conditional Expectation) plots show this per-instance rather than averaged, and reveal interactions that PDP's averaging can hide.

### E. Friedman's H-statistic (the model-agnostic gold standard)
Quantifies **what fraction of the joint effect of x1 and x2 is NOT explained by their individual (additive) effects.**

`H²_jk = Σᵢ [ PD_jk(x_j,x_k) − PD_j(x_j) − PD_k(x_k) ]² / Σᵢ PD_jk(x_j,x_k)²`

Where PD_jk is the joint partial dependence and PD_j, PD_k are the individual partial dependences. H² close to 0 → no interaction (fully additive); H² close to 1 → the joint effect is almost entirely interaction, not explainable by main effects alone.

**Why this is the strongest answer to give:** it's model-agnostic (works on any black-box model, not just linear), gives a normalized 0-1 interaction strength score, and lets you rank *all pairs* of features by interaction strength to prioritize which ones to investigate.

### F. SHAP interaction values
Decomposes each prediction's SHAP value into a main effect for each feature plus a pairwise interaction term for every feature pair (based on the Shapley interaction index from game theory). Gives you **per-prediction, per-pair interaction magnitude** — more granular than H-statistic's global summary, useful for explaining individual predictions ("this customer's churn risk is driven by an interaction between tenure and support tickets, not either alone").

### G. Tree-based models — implicit interaction capture + explicit detection
Trees/GBMs (XGBoost, LightGBM) capture interactions *natively* through sequential splits (a split on x2 nested under a split on x1 is literally an interaction). But **standard feature importance (gain/split count) does NOT tell you which interactions exist** — it only ranks individual features. To explicitly surface interactions from a trained tree ensemble:
- Look at which feature pairs co-occur frequently on the same root-to-leaf path
- Use SHAP interaction values on the fitted GBM (works very well here)
- Restrict `max_depth=1` (stumps) as a baseline vs `max_depth>1` — if deeper trees dramatically outperform depth-1 trees, that performance gap is evidence of interaction effects being captured

### H. Automatic interaction generation + regularized selection
Generate all pairwise products (`PolynomialFeatures(interaction_only=True)`), then fit **Lasso (L1)** regression — the regularization will zero out uninformative interaction terms and keep the ones that matter. Practical for tabular data with a manageable number of base features; combinatorially explodes with high feature counts (careful — d features → d²/2 pairwise terms).

### I. Mutual information / interaction information
From information theory: **Interaction Information** = `I(X1;X2;Y) = I(X1;Y|X2) − I(X1;Y)` — measures whether knowing X2 changes how much X1 tells you about Y. Captures nonlinear, non-parametric interactions without assuming a functional form, at the cost of needing more data to estimate reliably (curse of dimensionality on the joint distribution).

---

## 5. Interview Traps (say these unprompted)

**Trap #1 — Correlation ≠ interaction.** Many candidates conflate "these two features are correlated" with "these two features interact." They are orthogonal concepts. You can have highly correlated features with zero interaction, and completely uncorrelated features with a strong interaction (checkerboard/XOR pattern is the classic example — this is also why a single-layer linear/logistic model provably cannot learn XOR without an explicit interaction term).

**Trap #2 — Adding interaction terms often causes severe multicollinearity.** `x1·x2` is typically highly correlated with `x1` and `x2` individually, especially if they aren't centered. **Fix: mean-center both variables before creating the product term** — this substantially reduces the induced collinearity between the interaction term and the main effects (standard textbook remedy, always mention it if asked about interaction terms in regression).

**Trap #3 — Tree ensembles "automatically" capture interactions, but you still can't see them without explicit tooling.** Don't say "gradient boosting handles interactions so I don't need to check" — you need SHAP interaction values or H-statistic to actually surface and explain *which* interactions the model found, both for debugging and for stakeholder communication.

**Trap #4 — PDP averaging can hide heterogeneous interactions.** If an interaction effect flips sign across different subpopulations, the averaged PDP surface can look deceptively flat/additive. Always sanity check with ICE plots (individual curves) alongside PDP.

---

## 6. L5 Differentiators

- Lead with the correlation-vs-interaction distinction unprompted — signals conceptual clarity
- Know the H-statistic formula and what it actually measures (fraction of joint effect not explained by additive main effects)
- Mention SHAP interaction values as the modern, model-agnostic, per-prediction tool — shows current best-practice awareness
- Bring up centering as the fix for interaction-term multicollinearity — ties back cleanly to a multicollinearity discussion if asked
- Frame it as a workflow (domain hypothesis → visual check → formal statistic → model-based confirmation), not a single technique

---

## Comprehension Check

Let's work through these — this is a nice cluster testing whether you understand interactions beyond just "multiply two features together."

## Q1: Why can two completely uncorrelated features still have a strong interaction effect?

Correlation and interaction are answering entirely different questions, and conflating them is the single most common mistake here.

- **Correlation** asks: do X₁ and X₂ move together across the dataset? (a property of the *input distribution*)
- **Interaction** asks: does the *effect of X₁ on Y* change depending on the value of X₂? (a property of the *underlying function/relationship*)

These are orthogonal. You can have two features that are statistically independent (zero correlation, generated completely separately) but the true data-generating function is something like:

$$Y = X_1 \times X_2 + \epsilon$$

Here X₁ and X₂ could be sampled independently (say, both uniform random and uncorrelated by construction), yet the marginal effect of X₁ on Y is literally 0 when X₂ = 0, and grows as X₂ grows. That's a textbook interaction with zero correlation.

**Concrete example:** `ad_spend` and `is_holiday` might be completely uncorrelated (you spend the same average amount regardless of whether it's a holiday), but the *effectiveness* of ad spend on sales could be much higher during holidays. Correlation measures co-occurrence; interaction measures whether one feature moderates the *slope* of another's effect. A model (or a human) that only checks correlation before deciding "these features are unrelated, no need to model them together" will completely miss this.

## Q2: In the ad_spend/holiday example, what does the sign and magnitude of the interaction coefficient tell you?

Say your model is:
$$Y = \beta_0 + \beta_1(\text{ad\_spend}) + \beta_2(\text{holiday}) + \beta_3(\text{ad\_spend} \times \text{holiday}) + \epsilon$$

- **β₃ (the interaction coefficient) sign:**
  - **Positive β₃** → ad spend is *more effective* during holidays than non-holidays. The slope of sales-on-ad-spend steepens when holiday = 1.
  - **Negative β₃** → ad spend is *less effective* during holidays (diminishing returns, or holiday sales are driven by other factors and additional spend adds less marginal lift, maybe due to market saturation or attention competing with other advertisers).

- **β₃ magnitude:** tells you *how much* the slope changes. Specifically, the effect of ad_spend on Y is:
  $$\frac{\partial Y}{\partial(\text{ad\_spend})} = \beta_1 + \beta_3 \times \text{holiday}$$
  So on a non-holiday, the marginal effect of ad spend is just β₁. On a holiday, it becomes β₁ + β₃. If β₃ is large relative to β₁, the holiday effect dominates — meaning your "main effect" β₁ alone is a misleading summary of how ad spend actually behaves in the real world.

**Practical takeaway for an interview:** you should never interpret β₁ (the "main effect" of ad_spend) in isolation once an interaction term is in the model — its meaning is now conditional on holiday=0. This trips people up constantly.

## Q3: Why does mean-centering variables before creating an interaction term reduce multicollinearity?

Without centering, `ad_spend` and `ad_spend × holiday` are often highly correlated with each other and with `ad_spend` itself — mechanically, because the product term inherits scale and structure directly from its parent variables. If `ad_spend` ranges from, say, 1000 to 10000, then `ad_spend × holiday` (when holiday=1) is just a scaled subset of the same large numbers, so the two columns move together almost by construction, inflating VIF (variance inflation factor) and making coefficient estimates unstable (large standard errors, coefficients that flip sign with tiny data changes).

**Mean-centering** (replacing X with X − X̄) shifts the variable so its mean is 0. When you then form the interaction term (X₁ − X̄₁)(X₂ − X̄₂), the product term becomes mathematically closer to *orthogonal* to the original centered main effects — because the correlation between a variable and its product with another mean-zero variable drops substantially (in the case of one continuous and one binary 0/1 variable, centering removes the shared "level" component that was driving the collinearity).

**Important nuance to state in an interview:** centering reduces *non-essential* multicollinearity (the artificial kind introduced purely by the scale/construction of the interaction term) — it does **not** change the actual interaction effect, its statistical significance, or the model's predictions. It only stabilizes the individual coefficient estimates and their standard errors, making them more interpretable. If asked "does centering fix real multicollinearity between two genuinely correlated predictors" — no, it doesn't; that's a different problem entirely.

## Q4: What does Friedman's H-statistic actually measure, and why is it preferred over eyeballing feature importance from a GBM?

**What it measures:** Friedman's H-statistic quantifies how much of the joint effect of two (or more) features on the prediction is **not** explainable by summing their individual (additive) partial dependence effects. It's built directly on **partial dependence functions**:

$$H^2_{jk} = \frac{\sum_i \left[ PD_{jk}(x_j^{(i)}, x_k^{(i)}) - PD_j(x_j^{(i)}) - PD_k(x_k^{(i)}) \right]^2}{\sum_i PD_{jk}(x_j^{(i)}, x_k^{(i)})^2}$$

In words: take the joint partial dependence of features j and k together, subtract what you'd expect if their effects were purely additive (PD_j + PD_k), square the leftover ("interaction residual"), and normalize by the total joint variance. If H² = 0, the features act completely additively — no interaction. If H² is large, a substantial chunk of the joint effect comes specifically from their combination, not from either alone.

**Why it's preferred over eyeballing GBM feature importance:**
1. **Feature importance (e.g., gain/split-count based) tells you *how much* a feature matters overall, but says nothing about whether two features work together or independently.** A feature can be individually important via many different interaction partners, and raw importance scores can't distinguish "this feature matters on its own" from "this feature matters because of how it interacts with another specific feature."
2. **It's model-agnostic-ish and quantitative**, not a subjective visual read of tree splits. Manually inspecting hundreds of trees in a GBM to spot which features co-occur in splits is impractical and unreliable — a tree might split on X1 then X2 either because they interact, or purely coincidentally due to greedy splitting order.
3. **It gives you an actual number to rank interaction strength across all feature pairs**, so you can prioritize which interactions to investigate further (e.g., via 2D PDPs or SHAP interaction values) rather than guessing.

**Caveat worth mentioning:** H-statistic is computationally expensive (requires many PDP evaluations) and can be unstable with correlated features, since PDPs themselves assume features are independent when marginalizing — a violated assumption when features are correlated, which can distort the numerator/denominator.

## Q5: Why might a 2D partial dependence plot show a flat, additive-looking surface even when a real interaction exists in the data?

A few distinct mechanisms, all worth naming:

1. **Averaging washes out heterogeneous interactions.** PDPs compute the *average* prediction while marginalizing over the rest of the features. If the interaction effect is strong but flips sign or varies a lot across subgroups (e.g., positive interaction for one customer segment, negative for another), the average across all those subgroups can cancel out and look flat — even though real, strong interactions exist *within* subgroups. This is the classic PDP blind spot; **ICE (Individual Conditional Expectation) plots** reveal this by showing individual curves instead of the average, exposing heterogeneity that the averaged PDP hides.

2. **Correlated features distort the marginalization.** PDPs assume you can freely vary X_j and X_k independently while holding everything else fixed — but if X_j and X_k are correlated in the real data, this creates unrealistic, "extrapolated" combinations (e.g., evaluating ad_spend=10000 at holiday=0 when that combination never occurs in reality). The model's behavior in these unrealistic regions can be poorly defined/near-constant if it never saw such combinations during training, artificially flattening the surface. (**ALE — Accumulated Local Effects — plots** are often preferred here since they avoid this extrapolation problem.)

3. **The model itself hasn't learned the interaction well**, even if it exists in the true underlying data-generating process. If the GBM is under-regularized in a way that limits interaction depth (e.g., very shallow trees, `max_depth=1`, meaning literally no interactions can be captured — each tree is a single split), or if there's insufficient data density in the region where the interaction matters most, the fitted surface can look additive purely because the model failed to capture the true interaction, not because it doesn't exist.

4. **Scale/resolution issues** — a real interaction can be small in absolute magnitude relative to the main effects' scale, so on a naive 2D PDP plot (especially with a linear color scale), a genuine but modest curvature/twist in the surface can visually look flat unless you specifically compute and highlight the interaction residual (which is exactly what the H-statistic does numerically instead of relying on visual inspection).

