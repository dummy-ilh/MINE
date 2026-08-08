# Chapter 23 — Assumption Violations in Practice

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Unlike every prior chapter, this one introduces no new formulas — it's a worked integration of Chapters 7–14 and 20's toolkit, applied in the order a real analyst would actually use them, on one deliberately messier dataset.*

**Case study dataset** — 8 observations, constructed with genuine noise (unlike most prior chapters' clean or exact-fit examples), including one point that's both high-leverage and high-residual:

| $x$ (hours) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $y$ (score) | 16 | 19 | 27 | 27 | 39 | 35 | 52 | 70 |

---

## 23.1 The Motivating Question

Every diagnostic chapter so far (7 through 14, plus 20) isolated **one** specific problem on data engineered to make that one problem clean and legible. Real data almost never cooperates this way — a single messy dataset typically shows hints of several issues at once, and knowing the **order** in which to investigate them matters as much as knowing each individual tool. This chapter works through that order on one dataset, start to finish.

---

## 23.2 Step 1 — Fit the Model, Look at the Numbers

Ordinary OLS (Chapter 1 mechanics): $\bar{x}=4.5$, $\bar{y}=35.625$, $S_{xx}=42$, $S_{xy}=289.5$:

$$ \hat{\beta}_1 = 289.5/42 \approx 6.89, \qquad \hat{\beta}_0 = 35.625-6.89(4.5) \approx 4.61 $$

**Residuals** (fitted values from $4.61+6.89x$): $4.50,\ 0.61,\ 1.71,\ -5.18,\ -0.07,\ -10.96,\ -0.86,\ 10.25$. $SSE\approx276.3$, $MSE\approx46.05$.

**First-pass read, before any formal diagnostics:** residuals are notably larger in magnitude at the extremes ($x=6$: $-10.96$; $x=8$: $10.25$) than in the middle ($x=5$: $-0.07$) — a pattern worth investigating, but not yet a conclusion. This is exactly the point in a real analysis where you'd generate Chapter 7's four-panel diagnostic plot before doing anything else.

---

## 23.3 Step 2 — Leverage and Influence (Chapter 8), Before Formal Testing

**Why leverage/influence first, and not a formal heteroscedasticity test first:** a single highly influential point can distort *every* downstream diagnostic — including the Breusch-Pagan auxiliary regression itself — so it's standard practice to check for influential points before trusting any formal test that assumes the fitted model is reasonably trustworthy in the first place.

Using $h_{ii}=1/n+(x_i-\bar{x})^2/S_{xx}$ (Chapter 3/7 formula, simple-regression case): $h_{88}=1/8+3.5^2/42\approx0.417$ (the highest leverage point, tied with $x=1$, both being furthest from $\bar{x}=4.5$).

**Cook's distance for point 8** (Chapter 8 formula, using $r_8=e_8/(s\sqrt{1-h_{88}})\approx10.25/(6.786\times0.764)\approx1.98$):

$$ D_8 = \frac{r_8^2}{p}\cdot\frac{h_{88}}{1-h_{88}} = \frac{3.91}{2}\times\frac{0.417}{0.583} \approx 1.40 $$

Comparing to the $4/n=0.5$ threshold: **$D_8=1.40$ is flagged as clearly influential.**

**Contrast with point 6** (large residual, $-10.96$, but lower leverage, $h_{66}\approx0.179$): working through the same formula gives $D_6\approx0.35$ — **below** the 0.5 threshold, **not flagged**, despite having the second-largest raw residual in the dataset. **This is the same leverage-vs-influence lesson from Chapter 8, reappearing naturally in a messier dataset rather than a constructed one:** a large residual alone doesn't guarantee high influence; point 8's combination of extreme $x$ *and* a large residual is what makes it the one genuinely worth investigating first, not point 6.

---

## 23.4 Step 3 — Formal Heteroscedasticity Testing (Chapter 10)

With point 8 flagged but not yet removed (following Chapter 8, §8.5's guidance — investigate, don't reflexively delete), run the Breusch-Pagan auxiliary regression of $e_i^2$ on $x$: this gives $R^2_{aux}\approx0.294$, so:

$$ BP = n\times R^2_{aux} = 8\times0.294 \approx 2.35 $$

Comparing to $\chi^2_1$ at $\alpha=0.05$ (critical value $\approx3.84$): **$2.35<3.84$ — formally, we fail to reject homoscedasticity.**

**This is the most important practical lesson of the whole chapter, worth internalizing:** the *visual* pattern from Step 1 (residuals growing at the extremes) looked like a real heteroscedasticity signal, but the *formal* test, at this small sample size, doesn't reach significance. **A careful analyst doesn't treat this as "problem solved, ignore it"** — with $n=8$, this test has very low power (as flagged repeatedly throughout Chapters 10–11), so a non-significant result here is weak evidence of *absence*, not strong evidence *for* homoscedasticity. The honest conclusion is: **there's a suggestive pattern, an identified influential point likely driving much of it, and not enough data to formally confirm or rule out heteroscedasticity independent of that point** — precisely the kind of ambiguous, real-world verdict that a clean textbook example rarely produces.

---

## 23.5 Step 4 — Deciding What To Do

Following the decision framework built across Chapters 8–10, in order of preference:

1. **Investigate point 8 directly** (Chapter 8, §8.5) — is $y=70$ a data-entry error, or a genuine (if unusual) observation? Nothing in the diagnostics alone can answer this; it requires domain knowledge or checking the original data source.
2. **If point 8 is legitimate**, consider whether its large residual reflects genuine curvature the linear model is missing (worth a Chapter 7-style residuals-vs-fitted look with point 8 specifically in mind) rather than pure noise or heteroscedasticity.
3. **Given the ambiguous, underpowered formal test result**, a reasonable default is to report results with **robust (sandwich) standard errors** (Chapter 10, §10.5) regardless of the inconclusive Breusch-Pagan result — a low-cost safeguard against heteroscedasticity that doesn't require committing to a specific verdict from an underpowered test.
4. **Report sensitivity** — refit excluding point 8 and note how much the slope changes, being transparent about the dependency on a single observation rather than presenting only the full-data result as if it were unambiguous.

**What this chapter deliberately does NOT do:** manufacture a clean, decisive resolution. Real analyses frequently end at exactly this kind of qualified, honest conclusion — a flagged point, a suggestive-but-inconclusive formal test, and a recommended safeguard (robust SEs) rather than a dramatic fix. **Presenting appropriately hedged conclusions, rather than forcing false certainty, is itself a skill interviewers are testing for** when they ask open-ended "walk me through how you'd handle this dataset" questions.

---

## 23.6 The General Workflow, Abstracted

1. **Fit the model, examine residuals visually first** (Chapter 7) — cheap, fast, and often reveals where to look next.
2. **Check leverage and influence** (Chapter 8) **before** trusting any diagnostic that itself depends on the fitted residuals, since one bad point can corrupt downstream tests.
3. **Run formal tests for the specific violations visual inspection suggested** (Chapters 9–11: multicollinearity, heteroscedasticity, autocorrelation) — don't run every test on every model reflexively; let Step 1's visual read guide which formal tests are worth the effort.
4. **Weigh formal test results against their statistical power**, especially in small samples — a non-significant formal test is not the same as confirmed compliance with assumptions.
5. **Choose a remedy proportional to the evidence** — robust standard errors as a low-commitment safeguard; WLS/GLS/transformations/robust regression only when there's good reason to believe a specific structure; outright point removal only as an investigated, justified last resort (Chapter 8, §8.5; Chapter 20).
6. **Report honestly**, including remaining uncertainty — sensitivity analyses (with/without a flagged point) are often more valuable than a single confident-looking final number.

---

## 23.7 Where the Textbooks Differ

- **Montgomery** is the strongest source for this kind of integrated, workflow-oriented case study, with extensive real (not toy) engineering datasets walked through exactly this way across multiple chapters.
- **Kutner** presents each diagnostic tool thoroughly but more often in isolation, chapter by chapter, leaving the *integration* largely to the reader/instructor rather than devoting a dedicated chapter to it.
- **Sheather** is close behind Montgomery in this respect, particularly strong on the "what does the software output actually tell you to do next" angle.
- **ESL/ISL** don't really have an equivalent chapter — their prediction-focused, cross-validation-driven philosophy handles this kind of ambiguity differently (comparing out-of-sample performance with and without a flagged point, rather than a sequence of classical hypothesis tests).

---

## 23.8 Interview Q&A

**Q: You're given a messy real dataset. What's the first thing you do before running any formal diagnostic tests?**
A: Fit the model and look at the residual plots (Chapter 7) — visual inspection is fast and tells you which formal tests are actually worth running, rather than reflexively testing for every possible violation.

**Q: Why check leverage and influence before running a formal heteroscedasticity test?**
A: A highly influential point can distort the fitted residuals that any subsequent formal test (like Breusch-Pagan) relies on — investigating influential points first ensures you're not drawing conclusions about the whole dataset's error structure based substantially on one unusual observation.

**Q: A formal test for heteroscedasticity comes back non-significant, but the residual plot still looks visually suspicious. What do you conclude?**
A: Not that homoscedasticity is confirmed — especially in small samples, a non-significant result mainly reflects low statistical power, not positive evidence of no violation. A reasonable response is to still apply a low-cost safeguard like robust standard errors, and to note the ambiguity honestly rather than treating the formal test as the final word.

**Q: You've identified one highly influential point in your data. What's your recommended sequence of actions?**
A: Investigate it first (data error vs. genuine observation) using domain knowledge, not statistics alone; if legitimate, check whether it reflects missing model structure (nonlinearity); report results both with and without the point for transparency; only exclude it outright as a last resort, with clear justification.

**Q: Why is reporting "sensitivity to a single point" often more valuable than reporting one confident final coefficient estimate?**
A: It's honest about how fragile the conclusion is — a coefficient that shifts dramatically when one point is removed reflects a fundamentally less trustworthy finding than one that's stable regardless, and stakeholders relying on the result deserve to know which situation they're in.

---

*End of Chapter 23. Next: Chapter 24 — Causal Inference Considerations (confounding, omitted variable bias, and precisely why a well-fitting, well-diagnosed regression still doesn't establish causation without additional assumptions).*
