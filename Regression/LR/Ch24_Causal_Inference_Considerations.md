# Chapter 24 — Causal Inference Considerations

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Revisits Chapter 4's full model ($\hat{\beta}_1=4.6,\hat{\beta}_2=7$) and Chapter 5's reduced model ($\hat{\beta}_1=8.1$) — not as two separate curve fits anymore, but as the key exhibit for a rigorous derivation of omitted variable bias.*

---

## 24.1 The Motivating Question

Every chapter so far has treated $\hat{\beta}_1$ as "the effect of $x_1$ on $y$" without asking a harder question: **effect in what sense?** A regression coefficient is, by construction, a statement about **association** in the observed data — nothing in the OLS machinery from Chapters 1–23 distinguishes "x1 causes y" from "x1 and y are both driven by some third factor" or "y actually causes x1" or "x1 and y are correlated by pure coincidence." **A well-fitting, well-diagnosed regression is necessary but nowhere near sufficient for a causal claim** — this chapter is about exactly what's missing, and it's one of the most consistently tested conceptual areas in applied interviews, precisely because it doesn't require any new algebra, only a shift in how you think about what a coefficient means.

---

## 24.2 Omitted Variable Bias — Derived Exactly

Suppose the **true** model includes both predictors: $y=\beta_0+\beta_1x_1+\beta_2x_2+\varepsilon$. Suppose instead you fit the **reduced** model, omitting $x_2$ entirely: $y=\alpha_0+\alpha_1x_1+\varepsilon^*$. What is $E[\hat{\alpha}_1]$?

Substituting the true model's $y$ into the reduced-model's OLS formula for $\hat{\alpha}_1$ (algebra omitted, but it follows directly from the definition of $\hat{\alpha}_1=S_{x_1y}/S_{x_1x_1}$ and $y=\beta_0+\beta_1x_1+\beta_2x_2+\varepsilon$):

$$ E[\hat{\alpha}_1] = \beta_1 + \beta_2\delta_1, \qquad \text{where } \delta_1 = \frac{S_{x_1x_2}}{S_{x_1x_1}} $$

**Plain-English reading:** $\delta_1$ is exactly the slope you'd get from regressing the **omitted** variable $x_2$ on the **included** variable $x_1$ (Chapter 9's auxiliary-regression setup). The reduced model's coefficient on $x_1$ isn't just "$x_1$'s own effect" — it's $x_1$'s own effect **plus** a contamination term: $x_2$'s true effect ($\beta_2$), scaled by how strongly $x_2$ moves together with $x_1$ ($\delta_1$). **If $x_2$ is omitted and correlated with $x_1$, the coefficient on $x_1$ silently absorbs part of $x_2$'s effect** — this is precisely the phenomenon first observed informally back in Chapter 4, §4.5, now made exact.

---

## 24.3 Verifying the Formula Against Numbers You Already Have

$\delta_1$ is exactly Chapter 9's auxiliary regression slope: $\delta_1=S_{x_1x_2}/S_{x_1x_1}=5/10=0.5$.

**Plugging into the formula**, using the full model's true coefficients from Chapter 5 ($\beta_1=4.6,\beta_2=7$):

$$ E[\hat{\alpha}_1] = 4.6+7(0.5) = 4.6+3.5 = 8.1 $$

**This exactly matches Chapter 5, §5.5's actually-computed reduced-model slope of 8.1** — not an approximation, an **exact numerical confirmation** of the omitted variable bias formula, using data already fully worked through in this curriculum. The "bias" here is precisely $\beta_2\delta_1=3.5$ — the gap between the reduced model's 8.1 and the full model's 4.6.

---

## 24.4 The Question the Math Can't Answer — Confounder, Mediator, or Collider?

Here is the chapter's central, genuinely subtle point: **the arithmetic in §24.2–24.3 is completely agnostic to the causal story connecting $x_1$, $x_2$, and $y$** — but the *correct advice* about whether to include $x_2$ in the model depends entirely on which causal story is true, and **that can't be determined from the data alone.**

**Scenario A — $x_2$ is a confounder** (a common cause of both $x_1$ and $y$, e.g., some underlying "student motivation" drives both hours studied *and* practice tests taken independently, and motivation itself also affects score through channels not captured by either predictor): here, **omitting $x_2$ is a genuine mistake** — the reduced model's $8.1$ is contaminated by confounding, and the full model's $4.6$ is the more trustworthy estimate of $x_1$'s isolated causal effect. **This is the standard textbook case, and it's the one every earlier chapter implicitly assumed.**

**Scenario B — $x_2$ is a mediator** (hours studied, $x_1$, *causes* practice tests taken, $x_2$, which in turn affects score — i.e., part of how studying more hours helps is precisely *by* leading to more practice tests): here, **including $x_2$ is the mistake.** Controlling for a mediator blocks part of the very causal pathway you're trying to measure — the full model's $4.6$ would then understate $x_1$'s **total** effect on $y$ (which properly includes its indirect effect *through* $x_2$), while the reduced model's $8.1$ would actually be **closer** to the total causal effect you likely care about. This is sometimes called "**bad control**" in the causal inference literature — a variable that seems like it obviously belongs in the model, based on statistical significance or improved fit alone, but whose inclusion actively distorts the causal quantity of interest.

**Scenario C — $x_2$ is a collider** (both $x_1$ and $y$ independently cause $x_2$ — harder to construct plausibly in this specific example, but common in selection-effect settings, e.g., $x_2$="admitted to a selective program," influenced by both hours studied and underlying ability/score): here, **including $x_2$ actively introduces bias that wasn't present before** — conditioning on a collider creates a spurious association between $x_1$ and $y$ even if no true causal relationship between them exists at all. This is the most counterintuitive of the three, precisely because it's the one case where adding a plausible-looking predictor makes things categorically worse, not just imperfect.

**The practical upshot, worth stating explicitly and often in an interview:** "should I control for $x_2$" is not a statistical question that can be resolved by looking at $R^2$, p-values, or any diagnostic from Chapters 1–23 — it requires domain knowledge about the causal structure (formally represented as a **directed acyclic graph**, or DAG) connecting the variables. The same numerical bias formula from §24.2 applies identically in all three scenarios; **only the correct interpretation and recommended action differ, and that difference comes entirely from outside the dataset.**

---

## 24.5 Correlation, Causation, and What Regression Alone Can (and Can't) Establish

To be direct about the limits: even a regression with excellent diagnostics (Chapters 7–11), a well-justified functional form (Chapter 12, 21), appropriately handled categorical/interaction structure (Chapter 13), and a defensible model selection process (Chapter 14) **still only ever estimates an association**, conditional on whatever's included in the model. Moving from that association to a causal claim requires either:

- **A randomized experiment** — randomization breaks any link between the treatment variable and unmeasured confounders by design, which is precisely why RCTs remain the gold standard for causal claims.
- **A credible, explicit causal assumption** (a DAG, or equivalent domain reasoning) about which variables are confounders, mediators, or colliders — as in §24.4 — combined with including exactly the confounders (and none of the mediators or colliders).
- **Quasi-experimental designs** that approximate randomization in observational data — briefly, for recognition: **instrumental variables** (using a variable that affects $x_1$ but has no direct effect on $y$ except through $x_1$, to isolate exogenous variation), **difference-in-differences** (comparing changes over time between a treated and untreated group), and **regression discontinuity** (comparing observations just above/below an arbitrary treatment-assignment threshold, where assignment is plausibly as-good-as-random near the cutoff).

These quasi-experimental methods are each substantial topics in their own right, well beyond linear regression's scope — they're flagged here mainly so their names and core logic are recognizable if raised in an interview, not to be derived in depth in this curriculum.

---

## 24.6 Where the Textbooks Differ

- **Kutner and Montgomery** derive omitted variable bias algebraically (essentially as done in §24.2) but generally stop there, treating it as a technical/statistical result rather than extending into the confounder/mediator/collider causal framework — that framework is more recent and more associated with the dedicated causal inference literature (e.g., Pearl's work on DAGs) than with either of these classical regression texts.
- **ESL/ISL** barely touch causal inference at all — the entire book is explicitly and consistently framed around predictive accuracy rather than causal interpretation, treating "what does this coefficient mean causally" as outside its scope by design.
- **Sheather** occasionally flags the correlation-versus-causation distinction in examples but doesn't develop the formal confounder/mediator/collider framework in depth.
- **This chapter's DAG-based framing** (§24.4) draws primarily from the dedicated causal inference literature rather than any of the four core regression textbooks — flagged explicitly here because it's a meaningfully different intellectual tradition from classical regression theory, even though it's expressed using the exact same OLS machinery throughout.

---

## 24.7 Interview Q&A

**Q: Derive the omitted variable bias formula and explain each term.**
A: $E[\hat{\alpha}_1]=\beta_1+\beta_2\delta_1$, where $\delta_1$ is the slope from regressing the omitted variable $x_2$ on the included variable $x_1$. The reduced model's coefficient absorbs part of the omitted variable's true effect, scaled by how strongly the omitted and included variables move together.

**Q: If including a variable improves your model's $R^2$ and has a significant t-test, should you always include it if your goal is a causal estimate?**
A: Not necessarily — if that variable is a mediator (part of the causal pathway from your predictor of interest to the outcome) or a collider (a common effect of your predictor and the outcome), including it can bias or distort the causal estimate you actually care about, even though it improves the statistical fit. Statistical significance and improved fit say nothing about whether the variable belongs in a causally-interpretable model.

**Q: What's the difference between a confounder and a mediator, and why does the difference matter for whether to control for it?**
A: A confounder is a common cause of both the predictor and outcome — omitting it biases the causal estimate, so you should control for it. A mediator lies on the causal pathway between the predictor and outcome — controlling for it blocks part of the true causal effect you're trying to measure, so you generally should NOT control for it if you want the total causal effect.

**Q: What is a collider, and why is conditioning on one dangerous?**
A: A collider is a variable that is a common effect of both your predictor and your outcome (or of the predictor and some other cause of the outcome). Conditioning on (controlling for, or even just restricting your sample by) a collider can introduce a spurious association between the predictor and outcome even when no true causal relationship exists between them.

**Q: Regression alone can't establish causation. What are some ways to get closer to a causal claim using observational data?**
A: Explicitly reasoning through the causal structure (a DAG) to identify and control for confounders while avoiding mediators and colliders; or using quasi-experimental designs like instrumental variables, difference-in-differences, or regression discontinuity, which each approximate the logic of randomization in different ways using observational data.

---

*End of Chapter 24. Next: Chapter 25 — Interview Capstone (an end-to-end whiteboard-style case study integrating every chapter of this curriculum, plus a compiled interview Q&A bank spanning Chapters 1–24).*
