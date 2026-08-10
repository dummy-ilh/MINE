# Chapter 10 — Heteroscedasticity

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — rewritten with plain-language explanations, ASCII visuals, and extra worked examples. Uses Chapter 5's dataset (residuals $e = 0.2, 0.6, -1, -0.6, 0.8$; $SSE = 2.4$) for the formal test, plus a simple-regression example for the Weighted Least Squares walkthrough.*

---

## 10.1 The Motivating Question

**In one sentence:** heteroscedasticity means your model's mistakes aren't equally noisy everywhere.

Picture two scatter plots of residuals against a predictor $x$:

```
HOMOSCEDASTIC (good)              HETEROSCEDASTIC (funnel shape)

  e |  .   .    .   .              e |            .
    |    .   .    .                  |          .   .
  0 |--.----.---.----.----  x      0 |--.--.-.------.------- x
    |    .    .   .                  |  . . .        .
    |  .    .    .   .               |  ..              .
    |__________________              |______________________

  spread of e is constant          spread of e GROWS as x grows
  across all x  → OK                → variance is not constant
```

The left plot is what you want: residuals scattered in an even, constant-width band. The right plot is the classic warning sign — a "funnel" that widens as $x$ increases. Chapter 7 flagged this by eye. This chapter turns "that looks a little funnel-shaped" into an actual number.

**What breaks and what doesn't — the single most important fact in this chapter:**

| Still fine under heteroscedasticity | Broken under heteroscedasticity |
|---|---|
| $\hat\beta_0, \hat\beta_1$ remain **unbiased** (Gauss-Markov's unbiasedness doesn't need constant variance) | OLS is no longer **BLUE** — some other weighted estimator now has lower variance |
| Your point predictions | Standard errors, t-tests, F-tests, confidence intervals — all built assuming $\text{Var}(\varepsilon_i)=\sigma^2$ for every $i$ |

**Analogy to hold onto:** a bathroom scale that's accurate to the ounce for light objects but could be off by five pounds for heavy ones. The scale isn't *biased* — on average it's still right — but the *precision* of its readings depends on what you're weighing. That's heteroscedasticity: your line is still the right line on average, but your confidence about how precise any single prediction is now depends on where you are on the x-axis.

---

## 10.2 The Breusch-Pagan Test — Formalizing the Funnel Shape

**Plain-language idea, before any formula:** if the noisiness of your mistakes truly depends on $x_1, x_2$, then squaring each residual and trying to *predict that squared residual* from $x_1, x_2$ should actually work to some degree. If the noise is equally spread out everywhere, there's nothing for $x_1, x_2$ to explain about the size of the mistakes — the attempt should fail, and that failure (a low $R^2$) is itself the evidence that nothing suspicious is happening.

```
Step 1: fit original model  →  get residuals e_i
Step 2: square them         →  e_i^2  (our stand-in for the unobservable variance)
Step 3: regress e_i^2 on x1, x2   (the "auxiliary regression")
Step 4: does that auxiliary regression explain a meaningful amount? → test it
```

**The test statistic, simplified:**

$$ BP = n \times R^2_{aux} $$

Read this as: *(sample size)* × *(how well the predictors explain the squared residuals)*. Bigger sample, or a stronger pattern in the squared residuals, both push $BP$ up. Under the null hypothesis of homoscedasticity, $BP$ follows a $\chi^2_p$ distribution ($p$ = number of predictors in the auxiliary regression).

**Worked example.** Residuals: $e = 0.2, 0.6, -1, -0.6, 0.8$, so $e^2 = 0.04, 0.36, 1, 0.36, 0.64$.

Regressing $e^2$ on $x_1, x_2$ gives fitted line $\hat z = 0 - 0.08x_1 + 0.4x_2$ (where $z = e^2$), with:

$$ SSE_{aux} = 0.3264, \quad SST_{aux} = 0.5184 \quad\Rightarrow\quad R^2_{aux} = 1 - \frac{0.3264}{0.5184} \approx 0.370 $$

$$ BP = 5 \times 0.370 \approx 1.85 $$

Compare to $\chi^2_2$ at $\alpha = 0.05$ (critical value $\approx 5.99$):

```
        reject region  →|███████████████
0    1.85           5.99
     ^BP here            ^critical value

1.85 < 5.99  →  fail to reject homoscedasticity
```

**Reading the result honestly:** $x_1, x_2$ explained about 37% of the variation in the squared residuals — that sounds like something, but with only 5 data points there isn't enough evidence to call it a real pattern rather than coincidence. **This is "no signal detected," not "homoscedasticity confirmed."** With $n=5$, almost nothing passes this test convincingly either way.

---

## 10.3 White's Test — A More General Alternative

Breusch-Pagan only checks: *does noise rise or fall in a straight line as $x_1$ or $x_2$ increases?* White's test checks richer patterns by adding squared and cross-product terms to the auxiliary regression:

$$ e_i^2 \sim x_1,\ x_2,\ x_1^2,\ x_2^2,\ x_1 x_2 $$

$$ White = n \times R^2_{aux} \sim \chi^2_{df}, \quad df = \text{number of auxiliary terms (here, 5)} $$

**What the extra terms buy you, in plain words:**
- $x_1^2, x_2^2$ → catches noise that follows a *curve* rather than a straight line as $x_1$ or $x_2$ increases
- $x_1 x_2$ → catches noise that spikes specifically when $x_1$ *and* $x_2$ are both high together (an interaction effect)

**Why it's skipped here, concretely:** 5 observations, 5 auxiliary predictors + an intercept = 0 degrees of freedom left over. There's literally no data left to test with. This is the cleanest possible illustration of why White's test is a *large-sample* tool — not a stylistic choice, a hard mathematical wall.

```
Breusch-Pagan:  needs df ≥ p        (p = 2 here → fits, barely)
White's test:   needs df ≥ p + p(p+1)/2   (5 here → n=5 leaves nothing)
```

---

## 10.4 Remedy 1 — Weighted Least Squares (WLS)

**Plain-language idea:** if you know some data points are inherently noisier than others, let the *cleaner* points have more say in where the line goes, and let the *noisier* points have less say. That's the entire mechanism — a "weighted vote."

$$ \hat{\boldsymbol\beta}_{WLS} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y}, \qquad w_i = \frac{1}{\hat\sigma_i^2} $$

Noisier point → larger assumed variance → smaller weight → less influence on the fitted line. WLS is, in fact, the **new BLUE** once homoscedasticity is violated but you know the weight structure — a direct instance of Chapter 6's point that a different linear-unbiased estimator becomes optimal when Gauss-Markov's conditions change.

```
OLS: every point gets an equal vote
   [====][====][====][====][====]

WLS (variance grows with x1): later points get downweighted
   [======][====][===][==][=]
    trust ↓ as x1 rises →
```

**Worked illustration** (simple regression, $y$ on $x_1$, assuming $\text{Var}(\varepsilon_i)\propto x_{1i}$, so $w_i = 1/x_{1i}$):

| Student | $x_1$ | $w_i = 1/x_1$ |
|---|---|---|
| 1 | 1 | 1.000 |
| 2 | 2 | 0.500 |
| 3 | 3 | 0.333 |
| 4 | 4 | 0.250 |
| 5 | 5 | 0.200 |

Weighted means: $\bar x_w = \dfrac{\sum w_i x_{1i}}{\sum w_i} = \dfrac{5}{2.283} \approx 2.190$, $\bar y_w = \dfrac{\sum w_i y_i}{\sum w_i} = \dfrac{133.27}{2.283} \approx 58.37$

Weighted slope (same shape as Chapter 1's $S_{xy}/S_{xx}$, just with a $w_i$ tucked into every term):

$$ \hat\beta_{1,WLS} = \frac{\sum w_i(x_{1i}-\bar x_w)(y_i - \bar y_w)}{\sum w_i(x_{1i}-\bar x_w)^2} = \frac{31.17}{4.05} \approx 7.70 $$

$$ \hat\beta_{0,WLS} = \bar y_w - \hat\beta_{1,WLS}\bar x_w = 58.37 - 7.70(2.190) \approx 41.5 $$

**Compare to plain OLS** (Chapter 5, §5.5): $\hat\beta_1 = 8.1$. WLS pulls the slope down to **7.70**, because student 5 — the highest-$x_1$ point, assumed to be the noisiest under $\text{Var}\propto x_1$ — gets the *smallest* weight (0.2) and loses influence over the line. Student 1, assumed cleanest, keeps full weight (1.0) and gains relative influence. That single reweighting is the entire difference between the two slopes.

---

## 10.5 Remedy 2 — Robust (Sandwich) Standard Errors

**The problem WLS doesn't solve:** WLS needs you to *know or guess* the shape of the variance (e.g., "grows proportionally with $x_1$"). What if you don't know that, and don't want to guess wrong?

**Robust standard errors sidestep the whole problem.** They keep the *exact same* $\hat{\boldsymbol\beta}_{OLS}$ — no reweighting of any data point — and only fix the error bars around it:

$$ \widehat{\text{Var}}_{robust}(\hat{\boldsymbol\beta}) = \underbrace{(\mathbf{X}^T\mathbf{X})^{-1}}_{\text{bread}} \ \underbrace{\big(\mathbf{X}^T \,\text{diag}(e_i^2)\, \mathbf{X}\big)}_{\text{filling}} \ \underbrace{(\mathbf{X}^T\mathbf{X})^{-1}}_{\text{bread}} $$

```
        ┌──────────────┐
        │  bread        │  (X'X)^-1  — same matrix used everywhere in OLS
        ├──────────────┤
        │  filling      │  built from OBSERVED e_i^2 — adapts to whatever
        │  (X' diag(e²) X) noise pattern is actually in the data
        ├──────────────┤
        │  bread        │  (X'X)^-1  — identical outer slice again
        └──────────────┘
```

**Why this works, in plain words:** instead of assuming a theoretical shape for the noise (like WLS does), the "filling" is built directly out of the squared residuals you actually observed — whatever lumpy, uneven pattern is really there gets absorbed automatically. The trade-off: it's a large-sample (asymptotic) guarantee, so it's less trustworthy in a tiny 5-observation dataset like this chapter's running example.

**Choosing between WLS and robust SEs — the simplest possible rule:**

- Know *why* the noise varies (e.g., proportional to a known predictor) → use **WLS**. It improves both efficiency and inference.
- Don't know the shape, don't want to guess → use **robust SEs**. Same point estimate, honest error bars, no theory required. This is the modern default in most applied work, precisely because the true variance structure is rarely known with confidence.

---

## 10.6 Where the Textbooks Differ

- **Kutner** — fullest derivation, ties the auxiliary regression back to the same normal-equations machinery used throughout the book.
- **Montgomery** — leans hard on WLS, reflecting industrial/quality-control roots where variance structures (e.g., measurement error scaling with quantity produced) are often known from the physical process.
- **Sheather** — leans hardest into robust standard errors as the modern default, consistent with its applied, software-output-driven style.
- **ESL/ISL** — barely touch this topic; it's a classical-inference concern, and prediction-focused, cross-validation-driven frameworks are comparatively insensitive to it (out-of-sample accuracy isn't corrupted by heteroscedasticity the way p-values and CIs are).

---

## 10.7 Interview Q&A

**Q: Does heteroscedasticity bias OLS coefficient estimates?**
A: No. The line is still right on average. What breaks are the standard errors — and everything built on them (t-tests, F-tests, CIs) — plus OLS stops being BLUE, since a differently-weighted estimator becomes minimum-variance instead.

**Q: Breusch-Pagan vs. White's test — what's the actual difference?**
A: Breusch-Pagan checks for straight-line noise patterns using only the original predictors. White's test also checks for curved or interaction-driven noise patterns by adding squared and cross-product terms — at the cost of needing a lot more data (more terms = more degrees of freedom consumed).

**Q: When would you pick WLS over robust standard errors?**
A: WLS if you know (or can justify) the shape of the variance — it fixes both efficiency and inference. Robust SEs if you don't want to commit to a specific weighting theory — same point estimate, corrected error bars only.

**Q: What does "sandwich" refer to in sandwich standard errors?**
A: Two identical $(\mathbf{X}^T\mathbf{X})^{-1}$ "bread" layers on the outside, with a "filling" built from the observed squared residuals in the middle — a filling that automatically adapts to whatever noise pattern is actually present, rather than assuming one in advance.

**Q: If Breusch-Pagan fails to reject homoscedasticity, does that prove the errors have constant variance?**
A: No — it only means no significant evidence was found, which is especially uninformative with a small, low-power sample. "No evidence of a problem" is not the same as "proof there's no problem."

---

*End of Chapter 10. Next: Chapter 11 — Autocorrelation (Durbin-Watson test, time-series residual patterns, and Generalized Least Squares as the remedy when errors are correlated across observations rather than just unequal in variance).*
