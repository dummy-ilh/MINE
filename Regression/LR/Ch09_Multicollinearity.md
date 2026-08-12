# Chapter 9 — Multicollinearity

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations, a simplified formula cheat-sheet, and a complete numerical walkthrough. Formalizes the instability first observed in Chapter 5, §5.4 and Chapter 3, §3.8, using the same $x_1$ (hours studied), $x_2$ (practice tests) predictors.*

**Full dataset (5 students, used throughout this chapter):**

| Student | $x_1$ (hours) | $x_2$ (practice tests) | $y$ (score) |
|---|---|---|---|
| 1 | 1 | 1 | 50 |
| 2 | 2 | 1 | 55 |
| 3 | 3 | 2 | 65 |
| 4 | 4 | 2 | 70 |
| 5 | 5 | 3 | 83 |

---

## 9.1 The Motivating Question

Chapter 5 found something strange: the overall F-test was overwhelmingly significant ($F\approx279.5$), yet the individual t-test for $\hat\beta_2$ failed to reject at $\alpha=0.05$. Chapter 3 (§3.8) predicted mathematically *why* this can happen: when predictors are correlated with each other, $\mathbf X^T\mathbf X$ becomes close to non-invertible, inflating the variance of individual coefficient estimates.

This chapter builds the formal diagnostic tools to **detect and quantify** this — turning "the coefficients seemed unstable" into a precise, standard, interview-ready number, and then (§9.6) walks the *entire* pipeline end to end with real numbers so you can see exactly where the instability comes from and exactly how it propagates.

**Shopping-cart analogy:** two predictors that move together are like two people pushing the same cart in sync. You can measure the *total* push (the model's overall fit) confidently, but splitting credit between them individually becomes shaky. That shakiness is multicollinearity — this chapter measures exactly how bad it is, and traces the numbers that make it bad.

---

## 9.2 What Multicollinearity Actually Is (and Isn't)

**Multicollinearity** is a strong linear relationship among two or more predictors in $\mathbf X$. **Perfect multicollinearity** (one predictor is an exact linear function of others) makes $\mathbf X^T\mathbf X$ singular — no unique OLS solution exists. **Near-multicollinearity** (strong but imperfect correlation) is the far more common real-world problem, and the one this chapter's diagnostics target.

| | Affected? | Why |
|---|---|---|
| Predictions / $\hat y$ / $R^2$ | **No** — stay reliable | The *combined* information in correlated predictors is still there |
| Precision / interpretability of *individual* coefficients | **Yes** — degrades badly | The model can't cleanly split credit between near-redundant predictors |

**The single most important sentence of this chapter:** multicollinearity doesn't make your model bad at *predicting* — it makes your model bad at *explaining, in isolation, which specific input deserves the credit.*

**Two flavors, in plain words:**
- **Perfect** multicollinearity = trying to solve for two unknowns with only one real piece of information — mathematically impossible, the equation-solving equivalent of dividing by zero.
- **Near** multicollinearity = not impossible, just *unstable*. Small changes in the data can cause individual coefficients to swing wildly, even while overall predictions barely move. §9.6 shows exactly how much they swing, with real numbers.

---

## 9.3 Variance Inflation Factor (VIF) — Definition and Intuition

For each predictor $x_j$, regress it on **all the other predictors**, and take the resulting $R_j^2$:

$$ VIF_j = \frac{1}{1-R_j^2} $$

**Stripped-down reading:**

$$ VIF_j = \frac{1}{1-\underbrace{R_j^2}_{\text{"how redundant is }x_j\text{ with the other predictors?"}}} $$

- $R_j^2$ near 0: $x_j$ is nearly independent of the others → $VIF_j\approx 1$ → no inflation.
- $R_j^2$ near 1: $x_j$ is almost entirely explainable by the others → denominator shrinks toward 0 → $VIF_j$ blows up toward infinity.

**One-sentence version:** $VIF_j$ tells you how many times larger $\hat\beta_j$'s variance is, compared to a hypothetical world where $x_j$ were completely uncorrelated with everything else. $VIF_j=1$ is "no penalty"; $VIF_j=10$ is "10× the noise you'd otherwise have."

**Building the intuition, one layer at a time:**
1. **The hidden regression.** VIF asks: "if I predict $x_j$ using the *other* predictors, how well can I do it?" This has nothing to do with $y$ — it's purely about redundancy among the inputs.
2. **Turning redundancy into a number.** Near-total redundancy ($R_j^2\to1$) shrinks the denominator toward zero, so $VIF_j\to\infty$.
3. **Why redundancy causes instability.** If $x_1$ and $x_2$ carry almost the same information, the model has little independent evidence for "was it $x_1$ or $x_2$ that moved $y$?" Small data changes can flip that decision back and forth. VIF is a number for exactly how confused the model is likely to get.

---

## 9.4 Worked Example — Computing VIF by Hand

With two predictors, $R_j^2$ (regressing one on the other) is the same in both directions and equals the squared correlation between them.

**Step 1 — regress $x_2$ on $x_1$** (same mechanics as simple regression, applied to the predictors instead of $y$):

$$ \bar x_1=3,\quad \bar x_2=1.8,\quad S_{x_1x_1}=\sum(x_1-\bar x_1)^2=10 $$

$$ S_{x_1x_2}=\sum(x_1-\bar x_1)(x_2-\bar x_2) = 1.6+0.8+0+0.2+2.4 = 5.0 $$

Slope $= S_{x_1x_2}/S_{x_1x_1} = 5/10 = 0.5$; intercept $= 1.8-0.5(3)=0.3$. Fitted: $\hat x_2 = 0.3+0.5x_1$.

**Step 2 — compute $R_2^2$:**

$$ SSE = \sum(x_2-\hat x_2)^2 = 0.2^2+(-0.3)^2+0.2^2+(-0.3)^2+0.2^2 = 0.30 $$

$$ SST = \sum(x_2-\bar x_2)^2 = 2.8 \qquad\Rightarrow\qquad R_2^2 = 1-\frac{0.30}{2.8}=1-0.1071=0.8929 $$

**Step 3 — compute VIF:**

$$ VIF_2 = \frac{1}{1-0.8929}=\frac{1}{0.1071}\approx \mathbf{9.33} $$

By the two-predictor symmetry noted above, $VIF_1=VIF_2\approx9.33$ too.

**Thresholds:** $VIF>5$ deserves attention; $VIF>10$ is a clear red flag. At **9.33**, this dataset sits right at the edge of serious concern — confirming Chapter 5's symptom with a precise number.

**In plain words:** $x_1$ alone explains about 89% of the variation in $x_2$. That's a lot of overlap — knowing hours studied already tells you most of what you'd need to guess practice tests taken. That 89% overlap translates into "$\hat\beta_2$'s variance is about 9.3× bigger than it would be if $x_1,x_2$ were unrelated." That's the mathematical fingerprint of Chapter 5's symptom: a coefficient that's unbiased but too noisy to confidently distinguish from zero.

---

## 9.5 Condition Number — A Complementary Diagnostic

VIF diagnoses one predictor at a time. **Condition number** looks at the *overall* stability of $\mathbf X^T\mathbf X$ via its eigenvalues:

$$ \kappa = \sqrt{\lambda_{max}/\lambda_{min}} $$

**Plain framing:** eigenvalues here are loosely "how much independent information is stretched along each direction of the data." A tiny eigenvalue next to a large one means one direction carries almost no independent signal — a near-collapse.

**Worked example**, using the correlation between $x_1,x_2$: $r = S_{x_1x_2}/\sqrt{S_{x_1x_1}S_{x_2x_2}} = 5/\sqrt{10\times2.8}=5/5.29\approx0.945$.

$$ \mathbf R = \begin{bmatrix}1&0.945\\0.945&1\end{bmatrix},\qquad \lambda = 1\pm0.945=\{1.945,\ 0.055\} $$

$$ \kappa = \sqrt{1.945/0.055}=\sqrt{35.4}\approx \mathbf{5.95} $$

**Plain words:** $x_1,x_2$ move together at $r\approx0.945$ — nearly lockstep. That produces a big eigenvalue (1.945) and a small one (0.055), a roughly 35-to-1 gap. Taking the square root lands at ~5.95: one direction in the data is nearly 6× "shakier" to estimate than the sturdiest direction.

**Caveat:** conventions differ (this simplified two-predictor correlation-matrix version vs. Belsley's scaled-design-matrix approach including the intercept, standard in software, which typically flags severe multicollinearity above roughly 30). VIF is the more standardized, more commonly interview-tested tool; treat condition-number cutoffs with caution unless you know the convention.

---

## 9.6 End-to-End Numerical Walkthrough — What Multicollinearity *Actually Does*, Step by Step

This section traces one continuous numeric thread from raw data → the estimation machinery → the final t-statistic, so you can see exactly where instability enters and exactly how large its effect is — not just "VIF is high," but *what that produces downstream.*

### Step 1 — Build $\mathbf X^T\mathbf X$

$$ \mathbf X = \begin{bmatrix}1&1&1\\1&2&1\\1&3&2\\1&4&2\\1&5&3\end{bmatrix} \qquad\Rightarrow\qquad \mathbf X^T\mathbf X = \begin{bmatrix}5&15&9\\15&55&32\\9&32&19\end{bmatrix} $$

The off-diagonal entry linking $x_1$ and $x_2$ (32) is large relative to what independent predictors would produce — the first visible fingerprint of collinearity.

### Step 2 — The determinant is small relative to the entries

$$ \det(\mathbf X^T\mathbf X) = 5(55\cdot19-32^2) - 15(15\cdot19-32\cdot9) + 9(15\cdot32-55\cdot9) = 5(21)-15(-3)+9(-15) = 105+45-135 = \mathbf{15} $$

A determinant of 15 against entries in the hundreds/thousands is the numerical signal of near-singularity — the matrix is a long way from healthy, even though it's technically still invertible.

### Step 3 — Invert it, and watch the diagonal entries inflate

$$ (\mathbf X^T\mathbf X)^{-1} = \frac{1}{15}\begin{bmatrix}21&3&-15\\3&14&-25\\-15&-25&50\end{bmatrix} = \begin{bmatrix}1.400&0.200&-1.000\\0.200&0.933&-1.667\\-1.000&-1.667&3.333\end{bmatrix} $$

The $x_2$ diagonal entry (**3.333**) is more than **3.5×** the $x_1$ diagonal entry (**0.933**) — this asymmetry is exactly why $\hat\beta_2$ ends up far noisier than $\hat\beta_1$, and it's a direct, mechanical consequence of the small determinant in Step 2.

**Cross-check against VIF:** the identity $[(\mathbf X^T\mathbf X)^{-1}]_{jj} = VIF_j/S_{jj}$ should tie Steps 3 and 9.4 together exactly:

$$ \frac{VIF_1}{S_{x_1x_1}} = \frac{9.33}{10}=0.933\ \checkmark \qquad \frac{VIF_2}{S_{x_2x_2}}=\frac{9.33}{2.8}=3.333\ \checkmark $$

Both match the matrix inversion exactly — VIF isn't a separate idea from the inverted matrix, it *is* the inverted matrix, repackaged into an interpretable unit.

### Step 4 — Fit the model and get the residual variance

Solving the normal equations gives:

$$ \hat y = 38.2 + 4.6\,x_1 + 7.0\,x_2 $$

Residuals (matching Chapter 7/8's table): $e=(0.2,\ 0.6,\ -1.0,\ -0.6,\ 0.8)$, so:

$$ SSE=\sum e_i^2 = 0.04+0.36+1.00+0.36+0.64=2.40,\qquad s^2 = \frac{SSE}{n-p}=\frac{2.40}{2}=1.20 $$

### Step 5 — Turn the inflated diagonal entries into real standard errors

$$ \text{Var}(\hat\beta_j) = s^2\cdot[(\mathbf X^T\mathbf X)^{-1}]_{jj} $$

| Coefficient | $[(\mathbf X^T\mathbf X)^{-1}]_{jj}$ | $\text{Var}(\hat\beta_j)=1.20\times(\cdot)$ | $SE(\hat\beta_j)=\sqrt{\cdot}$ |
|---|---|---|---|
| $\hat\beta_1$ | 0.933 | 1.120 | **1.058** |
| $\hat\beta_2$ | 3.333 | 4.000 | **2.000** |

$\hat\beta_2$'s standard error (2.0) is almost double $\hat\beta_1$'s (1.058) — even though $\hat\beta_2$'s *point estimate* (7.0) is larger than $\hat\beta_1$'s (4.6). Bigger effect, noisier estimate: the classic multicollinearity signature.

### Step 6 — The t-statistics, and where Chapter 5's puzzle finally resolves

$$ t_1 = \frac{4.6}{1.058}=4.35 \qquad\qquad t_2=\frac{7.0}{2.000}=3.50 $$

With $df=n-p=2$, the two-tailed critical value at $\alpha=0.05$ is $t_{0.025,2}=4.303$.

| Coefficient | $t$ | vs. $4.303$ | Conclusion |
|---|---|---|---|
| $\hat\beta_1$ | 4.35 | $4.35>4.303$ | Barely significant |
| $\hat\beta_2$ | 3.50 | $3.50<4.303$ | **Not significant** |

**This is Chapter 5's exact puzzle, now fully explained numerically, start to finish:** the overall F-test is dominated by the strong *combined* signal in $(x_1,x_2)$ together, but the small determinant in Step 2 inflated $[(\mathbf X^T\mathbf X)^{-1}]_{22}$ in Step 3, which inflated $SE(\hat\beta_2)$ in Step 5, which pulled $t_2$ below the critical threshold in Step 6 — despite $\hat\beta_2=7.0$ being, if anything, the *larger* of the two coefficients. Nothing here is a coincidence or a separate phenomenon; it's one determinant's smallness propagating mechanically through five downstream steps.

### Step 7 — Sensitivity demo: how much do the coefficients swing?

Recall from Chapter 8: removing a *single* observation (Student 5, one-fifth of the data) sent the coefficients here:

| | With all 5 students | Without Student 5 | Swing |
|---|---|---|---|
| $\hat\beta_1$ | 4.6 | 5.0 | $-0.4$ (about $-8.7\%$) |
| $\hat\beta_2$ | 7.0 | 5.0 | $+2.0$ (about $+28.6\%$) |
| Fitted values | close to $y$ (max residual 1.0) | **exact**, $SSE=0$ | barely moved |

**One data point out of five changed $\hat\beta_2$ by almost 29%**, while the model's *predictions* barely changed at all (residuals were already small, and vanished entirely). That gap — huge coefficient swing, tiny prediction swing — is multicollinearity's signature in a nutshell, and it's the same asymmetry from the table at the top of §9.2, now shown with real before/after numbers instead of just asserted.

### Step 8 — What happens if the correlation gets even worse? (Holding everything else fixed)

Using $VIF=1/(1-r^2)$ and $SE(\hat\beta_2)=\sqrt{s^2\cdot VIF_2/S_{x_2x_2}}$ with $s^2=1.20$, $S_{x_2x_2}=2.8$, and $\hat\beta_2=7.0$ held fixed, escalating the correlation $r$ between $x_1$ and $x_2$ shows the blow-up is **not linear** — it accelerates sharply as $r\to1$:

| $r$ (correlation) | $VIF_2=\frac{1}{1-r^2}$ | $SE(\hat\beta_2)=\sqrt{1.2\cdot VIF_2/2.8}$ | $t_2=7.0/SE$ | Verdict at $df=2$ ($t_{crit}=4.303$) |
|---|---|---|---|---|
| 0.80 | 2.78 | 1.09 | 6.42 | Significant |
| 0.90 | 5.26 | 1.50 | 4.67 | Significant |
| **0.945 (our data)** | **9.33** | **2.00** | **3.50** | **Not significant** |
| 0.95 | 10.26 | 2.10 | 3.33 | Not significant |
| 0.99 | 50.25 | 4.64 | 1.51 | Not significant |
| 0.999 | 500.25 | 14.64 | 0.48 | Not significant |

**Plain reading of this table:** the "true" effect size ($\hat\beta_2=7.0$) never changes in this comparison — only the correlation between predictors does. Yet by $r=0.99$, the t-statistic has collapsed from a healthy 6+ down to 1.51, and by $r=0.999$ it's essentially zero. **A perfectly real, perfectly important predictor can be statistically strangled to death purely by being correlated with another predictor — with no change whatsoever in how much it actually matters.** This is the precise numerical mechanism behind the warning in §9.6's "what not to do": never drop a predictor on a failed t-test alone without checking whether VIF, not irrelevance, is the culprit.

---

## 9.7 Remedies for Multicollinearity

In rough order of preference:

1. **Drop one of the correlated predictors**, if theoretically justified.
2. **Combine correlated predictors** into a single composite (e.g., a combined "study effort" index), if defensible.
3. **Center the predictors** before creating interaction/polynomial terms (Chapter 13) — removes *artificial* multicollinearity introduced by the modeling choice itself.
4. **Collect more, more varied data** — multicollinearity is a property of the *sample*; more spread in $x_1,x_2$ independently reduces $R_j^2$.
5. **Ridge regression** (Chapter 16) — trades a small amount of bias for a large reduction in variance, previewed in Chapter 6's Gauss-Markov discussion.

**One-liners:**
1. **Drop** — if two inputs say the same thing, keep one; lose little, gain stability.
2. **Combine** — merge near-twins into one composite instead of forcing an awkward credit-split.
3. **Center first** — prevents the model from manufacturing artificial correlation via $x_1\times x_2$ or $x_1^2$ terms.
4. **More data** — a bigger, more diverse sample can break a coincidental correlation pattern.
5. **Ridge** — accept a tiny bit of "wrongness" on purpose in exchange for coefficients that stop swinging wildly.

**What NOT to do:** don't drop a predictor purely because its t-test wasn't significant (as with $\hat\beta_2$) without checking VIF first — Step 8 above shows exactly how a real, important predictor gets buried by correlation alone, with zero change in its actual importance.

**Plain words:** a non-significant t-test under high multicollinearity is like two people pushing a cart in sync — you can't tell how much either contributes individually, but that doesn't mean either is doing nothing. Removing one without checking VIF risks discarding a real predictor just because the model couldn't cleanly separate its effect from its correlated partner.

---

## 9.8 Where the Textbooks Differ

| Source | Distinctive contribution |
|---|---|
| **Kutner** | Most rigorous — ties VIF directly back to the $(\mathbf X^T\mathbf X)^{-1}$ diagonal entries from Chapter 3. |
| **Montgomery** | Strongest on condition number / eigenvalue diagnostics — an industrial-statistics text where design matrices are deliberately structured to avoid collinearity. |
| **Sheather** | Emphasizes `vif()` in R, and demonstrates the effect via simulation — coefficients swinging across simulated resamples of correlated predictors (conceptually the same as Step 7's sensitivity demo above). |
| **ESL/ISL** | Brief treatment — mainly *motivation* for ridge regression, whose $\lambda(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}$ correction directly addresses the near-singularity at its algebraic root (i.e., it fixes the small determinant from Step 2 directly). |

---

## 9.9 Formula Cheat-Sheet

| Quantity | Formula | Plain-English question |
|---|---|---|
| $R_j^2$ | $R^2$ from regressing $x_j$ on the other predictors | "How redundant is $x_j$ with everything else?" |
| VIF | $VIF_j = \dfrac{1}{1-R_j^2}$ | "How many times noisier is $\hat\beta_j$ because of that redundancy?" |
| Condition number | $\kappa=\sqrt{\lambda_{max}/\lambda_{min}}$ | "How lopsided/fragile is the whole predictor system?" |
| Coefficient variance | $\text{Var}(\hat\beta_j)=s^2\cdot[(\mathbf X^T\mathbf X)^{-1}]_{jj} = \dfrac{s^2\cdot VIF_j}{S_{jj}}$ | "How much does that noisiness actually translate into estimate uncertainty?" |

**Thresholds:**

| Diagnostic | Rule of thumb | Value here |
|---|---|---|
| VIF | $>5$ attention, $>10$ red flag | 9.33 |
| Condition number (simplified 2-predictor form) | context-dependent; Belsley's full-design version flags $\gtrsim30$ | 5.95 |

---

## 9.10 Interview Q&A

**Q: What does a VIF of 10 mean, precisely?**
A: The variance of that predictor's coefficient estimate is 10× larger than it would be if the predictor were completely uncorrelated with the others — a direct measure of how much multicollinearity inflates your uncertainty about that one coefficient.
*(Simple version: your estimate is about 10× "wobblier" than it would be if this input weren't tangled up with your other inputs.)*

**Q: Does multicollinearity bias your coefficient estimates?**
A: No — OLS remains unbiased under multicollinearity (Gauss-Markov still holds). The problem is inflated *variance*, not bias.
*(Simple version: on average across many samples you'd still land on the right answer — but any single sample's estimate could be way off in either direction. Step 7's before/after swing is exactly that "way off" in a single real sample.)*

**Q: If VIF is high for a predictor, should you always drop it?**
A: Not automatically — first consider whether it's theoretically essential, whether combining it with the correlated predictor makes sense, or whether ridge regression better serves stable prediction without discarding information.
*(Simple version: high VIF is a "go investigate" signal, not an automatic "delete this" order.)*

**Q: Can a model have severe multicollinearity and still predict well?**
A: Yes — $R^2$ and predictive accuracy are largely unaffected; only individual coefficient precision suffers. Step 7 shows this directly: fitted values barely moved even as coefficients swung by ~29%.
*(Simple version: the model can still give a good final answer, even if it can't cleanly say which ingredient deserves the credit.)*

**Q: How does VIF relate to the variance-covariance matrix from Chapter 3?**
A: $VIF_j$ is a rescaling of the diagonal entry of $(\mathbf X^T\mathbf X)^{-1}$: specifically $[(\mathbf X^T\mathbf X)^{-1}]_{jj}=VIF_j/S_{jj}$, verified numerically in Step 3 above. It's a friendlier repackaging of a number already buried inside $\text{Var}(\hat{\boldsymbol\beta})=\sigma^2(\mathbf X^T\mathbf X)^{-1}$.

**Q: Concretely, what breaks first when multicollinearity gets worse — and what doesn't?**
A: The determinant of $\mathbf X^T\mathbf X$ shrinks first (Step 2); that inflates the matrix inverse (Step 3), then coefficient standard errors (Step 5), then shrinks t-statistics (Step 6) — potentially past the significance threshold even for a real, large effect (Step 8's escalating-$r$ table). What does *not* break: predictions, $R^2$, and overall model fit, since those depend on the *combined* information in the correlated predictors, which stays intact throughout.

---

*End of Chapter 9. Next: Chapter 10 — Heteroscedasticity (Breusch-Pagan and White tests, Weighted Least Squares, and robust/sandwich standard errors as three different ways of handling unequal error variance).*
