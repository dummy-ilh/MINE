# Chapter 25 — Interview Capstone

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. This closing chapter does not introduce new material — it's a single whiteboard-style walkthrough showing how the entire 24-chapter curriculum gets deployed live in an interview, followed by a compiled Q&A bank organized by section.*

---

## 25.1 The Whiteboard Prompt

*"I'm going to give you a dataset: students' exam scores, hours studied, and number of practice tests taken. Walk me through how you'd build and validate a regression model, and tell me what you can and can't conclude from it."*

This is a deliberately open-ended prompt — exactly the kind that rewards a structured walkthrough over a single formula. Below is the full arc, chapter by chapter, using this curriculum's own running numbers as the concrete answer at each step.

---

## 25.2 The Full Walkthrough

**Step 1 — Start simple, understand the mechanics (Chapters 1–3).** Before jumping to multiple predictors, fit exam score against hours studied alone: $\hat{y}=41.5+7.5x$ (Chapter 1). State plainly what each term means — $\hat{\beta}_1=7.5$ is the average score increase per additional hour studied, and the line necessarily passes through $(\bar{x},\bar{y})$. Mention that under the hood, this is solved via the matrix normal equations $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ (Chapter 3) — the same formula that will handle any number of predictors without modification.

**Step 2 — Quantify uncertainty, don't just report a point estimate (Chapter 2).** $SE(\hat{\beta}_1)=0.5$, giving $t=15$ against $H_0:\beta_1=0$ — decisively significant here, though note explicitly that a 5-observation toy example produces unusually dramatic statistics; real data would rarely look this clean. Report the 95% CI, $(5.91,9.09)$, and be precise about its correct interpretation (a statement about the long-run behavior of the interval-construction procedure, not a probability statement about the fixed true $\beta_1$).

**Step 3 — Add the second predictor, and immediately flag the interpretation shift (Chapters 4–5).** With both hours studied and practice tests in the model, $\hat{\beta}_1$ drops from 7.5 to 4.6, and $\hat{\beta}_2=7$ — state explicitly that this is expected, not a red flag, since the two predictors are correlated and multiple regression correctly reallocates credit between them (Chapter 4, §4.5). Then surface the subtlety that makes this dataset genuinely interesting: the overall F-test is overwhelming ($F\approx279.5$), yet $\hat{\beta}_2$'s individual t-test fails to reject at 5% ($t=3.5$ vs. critical $4.303$) — a mismatch worth naming out loud rather than glossing over.

**Step 4 — Diagnose why that mismatch happened (Chapters 6–9).** Note that OLS remains BLUE under Gauss-Markov (Chapter 6) regardless, but instability specifically among *individual* coefficients under correlated predictors is the signature of multicollinearity. Compute VIF: $\approx9.33$ for both predictors (Chapter 9) — right at the conventional red-flag threshold, directly explaining Step 3's puzzle with a precise number rather than a vague impression.

**Step 5 — Check the diagnostic panel, not just the coefficients (Chapters 7–8).** Studentized residuals and Cook's distance flag one specific observation ($D=1.83$, dwarfing every other point) as highly influential — and note explicitly that this isn't just "the largest residual" (that distinction belongs to a *different*, lower-leverage point) — a clean illustration that leverage and residual size must combine to produce real influence.

**Step 6 — Check the other classical assumptions systematically (Chapters 10–11).** Breusch-Pagan for heteroscedasticity ($BP\approx1.85$, not significant here) and Durbin-Watson for autocorrelation ($DW\approx2.02$, no signal) — both come back clean in this dataset, but state the general remedies you'd reach for if they hadn't (WLS/robust SEs; GLS/Newey-West), since an interviewer often follows up with "what if it had failed?"

**Step 7 — Address functional form and structure (Chapters 12–13).** If residuals showed curvature, a log transform or polynomial term would be the fix (with the centering trick to avoid manufacturing fresh collinearity between $x$ and $x^2$); if there's a categorical predictor like study method, dummy coding plus an interaction term lets slopes differ by group — and flag the classic trap that the main-effect coefficient alone isn't "the" effect once an interaction is present.

**Step 8 — Decide which predictors actually belong (Chapters 14–15).** Adjusted $R^2$, AIC, BIC, and Mallows' $C_p$ all agree here: keep both predictors, despite $x_2$'s shaky individual t-test — a good moment to explain *why* these criteria and a single coefficient's significance can disagree. Back this up with out-of-sample validation: leave-one-out cross-validation gives a test MSE of $\approx3.26$, meaningfully higher than the optimistic in-sample MSE of $0.48$ — and note that the point driving the worst CV error is the *same* point flagged as most influential in Step 5, tying diagnostics and predictive validation together as two views of the same fact.

**Step 9 — If multicollinearity remains a genuine problem, regularize (Chapters 16–18).** Ridge shrinks the unstable $\hat{\beta}_2$ smoothly toward zero without eliminating it; lasso, at a high enough penalty, drops it entirely (exact sparsity); elastic net splits the difference, keeping both predictors active via the grouping effect. Choice among the three depends on whether the goal is stability (ridge), sparsity/interpretability (lasso), or both under correlated predictor groups (elastic net) — each tuned via the same cross-validation machinery from Step 8.

**Step 10 — Handle whatever's left (Chapters 19–21).** If the error structure is more complex than plain heteroscedasticity or AR(1), GLS is the general unifying formula both special cases fall out of. If a small number of points look like genuine data errors, RANSAC or Huber M-estimation guard against them — noting the important caveat that M-estimators alone don't fully protect against *bad leverage points* specifically. If the outcome isn't continuous, the same WLS/IRLS machinery generalizes directly into logistic regression via the GLM framework.

**Step 11 — Close with the honest caveat, every time (Chapters 23–24).** State plainly: none of this establishes causation. Walk through the omitted variable bias logic if a predictor was dropped, and — critically — note that whether a given variable *should* be controlled for (confounder) or *should not* be (mediator, collider) is a question the data alone cannot answer; it requires an explicit causal assumption about the relationship between the variables, not just a statistical test.

**This eleven-step arc — not any single formula — is what a strong interview answer to an open-ended regression question actually sounds like:** start simple, quantify uncertainty, add complexity deliberately, diagnose before regularizing, validate out-of-sample, and close with honest limits on interpretation.

---

## 25.3 Compiled Interview Q&A Bank (Highest-Yield, By Section)

**Foundations (Chapters 1–6):**
- Why minimize squared error, not absolute error? *(Differentiable, closed-form, MLE under Gaussian errors.)*
- Does the regression line always pass through $(\bar{x},\bar{y})$? *(Yes, always — a direct consequence of the first normal equation.)*
- What does BLUE mean, and does it require normality? *(Best Linear Unbiased Estimator; no, normality is only needed for exact inference, not for BLUE.)*
- Write $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ and explain every term.

**Inference & Diagnostics (Chapters 7–11):**
- Why can the overall F-test be significant while an individual t-test isn't? *(Multicollinearity.)*
- Difference between leverage and influence? *(Leverage depends only on $x$; influence requires a large residual too.)*
- Difference between internally and externally studentized residuals?
- What does a VIF of 10 mean, precisely?
- Does heteroscedasticity bias OLS coefficients? *(No — only standard errors and BLUE-status are affected.)*

**Model Building & Regularization (Chapters 12–18):**
- Why must predictors be centered before adding polynomial/interaction terms?
- Why does lasso produce exact zeros while ridge doesn't? *(Geometry: diamond corners vs. a smooth circle.)*
- What's the grouping effect in elastic net?
- Why can't plain $R^2$ guide variable selection?
- Difference between AIC and BIC's penalty, and the practical consequence?

**Beyond OLS (Chapters 19–22):**
- How is WLS a special case of GLS?
- What's a "bad leverage point," and why do M-estimators only partially handle it?
- How does IRLS connect logistic regression back to ordinary WLS?
- What causes perfect separation, and why is it dangerous?

**Judgment & Limits (Chapters 23–24):**
- A formal test fails to reject, but the diagnostic plot still looks suspicious — what do you conclude? *(Absence of evidence isn't evidence of absence, especially in small samples.)*
- Derive omitted variable bias and explain when controlling for a variable helps vs. hurts a causal estimate. *(Confounder: control for it. Mediator: don't. Collider: definitely don't.)*

---

## 25.4 Closing Note on Structuring a Live Answer

When an interviewer gives an open-ended prompt like §25.1's, the strongest signal isn't reciting every formula in this curriculum — it's demonstrating the **order of operations**: simple model first, uncertainty quantified before complexity added, diagnostics before remedies, out-of-sample validation before trusting in-sample fit, and an honest causal caveat at the end regardless of how clean the statistics look. That structure, applied consistently, is what every chapter of this curriculum has been building toward — the formulas are the vocabulary; this workflow is the sentence.

---

*This concludes the 25-chapter synthesized Linear Regression curriculum.*
