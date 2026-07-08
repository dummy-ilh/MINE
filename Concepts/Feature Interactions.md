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

**Q1.** Why can two completely uncorrelated features still have a strong interaction effect?
**Q2.** In the ad_spend/holiday example, what does the sign and magnitude of the interaction coefficient tell you?
**Q3.** Why does mean-centering variables before creating an interaction term reduce multicollinearity?
**Q4.** What does Friedman's H-statistic actually measure, and why is it preferred over eyeballing feature importance from a GBM?
**Q5.** Why might a 2D partial dependence plot show a flat, additive-looking surface even when a real interaction exists in the data?

Want a worked H-statistic calculation next, or should we go into how interaction detection changes for high-cardinality categorical features?
