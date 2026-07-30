# Chapter 5: Regression Adjustment

## 1. Explanation

### The basic idea

Regression adjustment is the most familiar tool from standard ML/stats, repurposed for a causal question. You fit:
```
Y = α + τ·D + β'X + ε
```
and interpret $\tau$ — the coefficient on treatment — as the causal effect of D, *controlling for* X.

### Why this "works" (when it does) — borrowing, not creating, identification

It's important to be precise about what regression is and isn't doing here. If X satisfies the backdoor criterion from Chapter 4 — i.e., conditioning on X achieves ignorability, $(Y(1),Y(0)) \perp D \mid X$ — then *within* each stratum/value of X, treatment is "as good as randomly assigned." The regression is essentially computing a weighted average of many within-stratum comparisons (compare treated to control units that share the same X value, then average those comparisons across all X values). **The causal validity comes entirely from the ignorability assumption** — a fact about the world you're arguing for via your DAG — not from the regression machinery itself. Regression is just an efficient computational device for doing the stratified comparison, assuming the relationship is smooth/linear enough that you don't need to literally bin the data into exact-X strata (which would often leave too few observations per stratum).

### Two silent assumptions that can each independently wreck the whole thing

**1. Functional form.** OLS assumes Y is (approximately) linear in X, and — unless you explicitly add interaction terms $D \times X$ — that the treatment effect $\tau$ is *constant* across all values of X. If the true relationship between X and Y is curved, or if the treatment effect genuinely varies by X (heterogeneous effects), a plain linear specification can give you a badly misleading single number, and worse, it may look perfectly "statistically significant" while being systematically wrong.

**2. Common support / overlap.** For the "as good as random within each stratum" logic to actually apply, every relevant stratum of X needs *both* treated and control units present. If treated units only exist at high X values and controls only exist at low X values, there's no stratum where you can make a real, data-supported comparison. In this situation, OLS doesn't refuse to give you an answer — it happily fits a straight line through each group and reports the gap between the lines *at any X value you ask about*, including regions where neither line has any actual data. This is silent extrapolation, and the resulting "effect" is a pure artifact of the assumed linear shape, not something the data can support. This is one of the most consequential and frequently tested traps in applied causal inference interviews.

### A useful mental model

Think of regression adjustment as a "smoothed" version of matching (Chapter 6). Matching literally finds, for each treated unit, an actual similar control unit and compares them directly. Regression instead fits a smooth line/plane through all the data and reads off the vertical gap between the "D=1" line and the "D=0" line at each X. If the true relationship is curved but you force a straight line, or if there's a region where only one arm has data, this "read off the gap" trick can produce numbers with no real empirical grounding — matching would have simply refused to produce a match in that region (visibly signaling the lack of support), whereas naive regression extrapolates silently.

## 2. Example

### Example A — Overlap failure made explicit

Effect of a "premium support" tier (D) on customer satisfaction score (Y, 0-100), confounded by account size (X, in $1000s Monthly Recurring Revenue).

| Customer | X (MRR $k) | D | Y |
|---|---|---|---|
| 1 | 1 | 0 | 60 |
| 2 | 2 | 0 | 63 |
| 3 | 3 | 0 | 68 |
| 4 | 4 | 0 | 70 |
| 5 | 15 | 1 | 92 |
| 6 | 18 | 1 | 95 |
| 7 | 20 | 1 | 97 |

Notice: there is **zero overlap** — every D=0 customer has X∈[1,4], every D=1 customer has X∈[15,20]. A linear regression of Y on D and X will "identify" τ almost entirely by extrapolating the control-group line (fit only on the X=1–4 range) all the way out to X=15–20, and comparing it there against the treated-group line — but there is **no actual data** anywhere in that region for the control counterfactual. Whatever number this regression spits out is a mathematical artifact of assumed linearity, not something grounded in observed comparisons.

**The correct interview answer here is explicitly not to report a causal number.** The right response is: "There's no common support between treated and control on the key confounder — this design cannot answer the causal question with the data as given. I'd either need additional data (some mid-size accounts represented in both arms) or a completely different identification strategy (e.g., an RDD if there's a sharp qualifying MRR threshold for premium support eligibility, covered in Chapter 9)."

### Example B — A case where regression adjustment does work reasonably, contrasted with the naive comparison

Effect of a coaching program (D) on test score (Y), confounded by prior GPA (X) — but this time with actual overlap in X between arms.

| Student | X (GPA) | D | Y (score) |
|---|---|---|---|
| 1 | 2.0 | 0 | 60 |
| 2 | 2.5 | 0 | 65 |
| 3 | 3.0 | 0 | 70 |
| 4 | 3.0 | 1 | 78 |
| 5 | 3.5 | 1 | 82 |
| 6 | 4.0 | 1 | 90 |

Naive difference-in-means: mean(Y|D=1) − mean(Y|D=0) = mean(78,82,90) − mean(60,65,70) = 83.33 − 65 = **18.33**

But GPA (X) predicts Y directly *and* predicts D (treated students average GPA 3.5, control students average GPA 2.5) — so part of this 18.33 gap is confounding, not the coaching effect. Notice, crucially, that student 3 (X=3.0, D=0) and student 4 (X=3.0, D=1) share the **same X value** — this is a region of actual overlap. Comparing exactly these two:
```
78 (student 4, treated) − 70 (student 3, control) = 8
```
This "exact match on X" comparison strips out the GPA confound entirely for this pair and suggests the true effect is closer to **8**, not 18.33 — most of the naive 18.33 gap was because treated students had higher baseline GPA, which independently raises test scores regardless of coaching. A full regression fit on all 6 points (which internally does a version of this same "compare within similar X" logic, smoothed across the whole range rather than just one exact-match pair) would land in a similar ballpark, precisely because — unlike Example A — there's genuine, if limited, overlap in X to anchor the comparison.

## 3. Interview Q&A

**Q: What does it mean for a regression-adjustment causal estimate to be "model-dependent," and why is that a weakness compared to design-based methods like RCTs or RDD?**
A: It means the numeric answer changes if you change the functional form (linear vs. quadratic vs. with interaction terms) even using the exact same data and the exact same conceptual control set — the estimate is partly a product of modeling choices, not purely of the data and design. Design-based methods derive their validity primarily from *how the data was generated* (randomization, a sharp cutoff), making them far less sensitive to these kinds of modeling choices.

**Q: How do you check for common support in practice before trusting a regression-adjusted causal estimate?**
A: Plot (or tabulate) the distribution of key covariates — or, more practically with many covariates, the estimated propensity score (Chapter 6) — separately for treated and control groups, and visually or statistically check for overlapping ranges. Formally, trim or drop observations outside the region of common support (e.g., extreme propensity scores) before estimating effects, and report the estimate as applying only to the overlap population.

**Q: If you add an interaction term D×X to your regression, what new estimand are you now able to compute that you couldn't before?**
A: CATE as a function of X — i.e., how the treatment effect *varies* with the covariate (e.g., "the coaching boost is larger for students with lower starting GPA than higher"), rather than assuming a single constant τ applies to everyone regardless of their covariate profile.

**Q: Your regression-adjusted estimate changes substantially when you switch from a linear to a quadratic specification for a key covariate. What does this tell you, and what would you do?**
A: It tells you the true relationship is likely nonlinear and your causal conclusion is sensitive to functional form assumptions — a red flag for fragility, since the "true" causal effect shouldn't depend on an essentially arbitrary modeling choice. I'd move toward more flexible/nonparametric adjustment (e.g., matching, or machine-learning-based outcome models combined with doubly-robust estimation, Chapter 6) that doesn't force a rigid parametric shape, and I'd report the sensitivity of the estimate across specifications rather than silently picking whichever one and hiding the instability.

**Q: In Example A (premium support, zero overlap), what would you tell a stakeholder who insists "just give me a number, we need to make a decision"?**
A: I'd explain that any number I produce from this data would be extrapolation dressed up as an estimate — literally computed from a region with zero real comparisons — and that reporting it without flagging this would be misleading, not rigorous. I'd offer concrete alternatives instead: collect a small amount of data on mid-size accounts in both arms (even a small randomized pilot targeting that range), or look for a different identification strategy (e.g., if there's a sharp assignment rule at some MRR threshold, an RDD could work); giving a fabricated-looking point estimate to satisfy an artificial urgency would do more harm than acknowledging the limitation clearly.

**Q: Why can regression adjustment sometimes look *more* precise (tighter standard errors) than matching, and is that necessarily a good thing?**
A: Regression uses a smooth parametric form and (implicitly) all the data, including regions with sparse or no true overlap, which can make standard errors look small — but that apparent precision can be an illusion if it comes from confidently extrapolating a possibly-wrong functional form into unsupported regions, rather than reflecting genuine information in the data. Precision is only meaningful if the underlying model/assumption is correct; a tight confidence interval around a badly biased point estimate is worse than a wider interval around an honest one.

---
**Previous: Chapter 4 — Confounding, DAGs, and the Backdoor Criterion**
**Next: Chapter 6 — Propensity Score Matching & Inverse Probability Weighting**
