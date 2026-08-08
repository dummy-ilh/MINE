# Chapter 13 — Categorical Predictors & Interactions

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Introduces a new small dataset with a categorical predictor, since none of the running datasets so far have included one.*

**New example dataset** — students studying via two methods (self-study vs. tutor), with hours studied ($x$) and exam score ($y$):

| Method | $x$ (hours) | $y$ (score) |
|---|---|---|
| Self-study | 1 | 45 |
| Self-study | 2 | 50 |
| Self-study | 3 | 55 |
| Tutor | 1 | 58 |
| Tutor | 2 | 66 |
| Tutor | 3 | 74 |

---

## 13.1 The Motivating Question

Every predictor so far has been numeric (hours, practice tests). But plenty of real predictors are **categories** — study method, device type, treatment vs. control, region. Regression can't multiply a coefficient by the word "tutor" — categories have to be converted into numbers first, in a way that preserves a sensible interpretation. That conversion is **dummy coding**, and it's the entire subject of the first half of this chapter.

---

## 13.2 Dummy (Indicator) Coding

For a two-level category, create a single **indicator variable**:

$$ D_i = \begin{cases}0 & \text{if self-study}\\1 & \text{if tutor}\end{cases} $$

The chosen $D=0$ group (self-study, here) is called the **reference level** — every other category's effect is interpreted *relative to it*. The model:

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_i+\varepsilon_i $$

**Reading each coefficient:** for self-study ($D=0$), the model reduces to $y=\beta_0+\beta_1x$. For tutor ($D=1$), it becomes $y=(\beta_0+\beta_2)+\beta_1x$ — **the same slope, but a shifted intercept.** $\beta_2$ is literally "the vertical shift in the line for the tutor group, relative to self-study, holding hours studied fixed." This model (no interaction yet) forces both groups to have the *same slope* — an assumption we test and relax next.

**General rule for $k$ categories:** you need $k-1$ dummy variables, not $k$. Including all $k$ would create perfect multicollinearity with the intercept (the **dummy variable trap**) — the $k$ dummy columns plus the intercept column would sum to a constant vector, making $\mathbf{X}^T\mathbf{X}$ singular (a direct callback to Chapter 3, §3.4, and Chapter 9's multicollinearity discussion). For a 3-level category (e.g., self-study/tutor/online), you'd use 2 dummies, with one level as the omitted reference.

---

## 13.3 Adding an Interaction Term — Letting the Slope Differ Too

The data above doesn't actually have equal slopes across groups — the tutor group's scores climb faster per hour than self-study's. To let the slope itself differ by group, add an **interaction term**: the product of $x$ and $D$.

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_i+\beta_3(x_i\times D_i)+\varepsilon_i $$

**Reading it group by group:**

- **Self-study** ($D=0$): $y=\beta_0+\beta_1x$ — intercept $\beta_0$, slope $\beta_1$.
- **Tutor** ($D=1$): $y=(\beta_0+\beta_2)+(\beta_1+\beta_3)x$ — intercept $(\beta_0+\beta_2)$, slope $(\beta_1+\beta_3)$.

**Fitting this model** to the dataset above (design matrix columns: intercept, $x$, $D$, $x\times D$) gives an **exact fit** (constructed that way for clarity):

$$ \hat{\beta}_0=40,\quad \hat{\beta}_1=5,\quad \hat{\beta}_2=10,\quad \hat{\beta}_3=3 $$

**Verification** (tutor group, $x=2$): $(40+10)+(5+3)(2) = 50+16=66$ — matches the table exactly. Every other row checks out the same way.

---

## 13.4 What Each Coefficient Actually Means Here

| Coefficient | Value | Meaning |
|---|---|---|
| $\hat{\beta}_0$ | 40 | Self-study group's predicted score at $x=0$ (baseline intercept) |
| $\hat{\beta}_1$ | 5 | Self-study group's slope — each additional hour adds 5 points, **for self-study specifically** |
| $\hat{\beta}_2$ | 10 | The tutor group's intercept is 10 points **higher** than self-study's, at $x=0$ |
| $\hat{\beta}_3$ | 3 | The tutor group's slope is 3 points **steeper** than self-study's — each additional hour is *more valuable* under tutoring |

**The single most commonly misread coefficient in this entire model is $\hat{\beta}_1$.** With the interaction term present, $\hat{\beta}_1$ is **not** "the overall effect of hours studied" — it's specifically **the effect of hours studied for the reference group only** ($D=0$). The *actual* effect of one more hour of study depends on which group you're in:

$$ \frac{\partial y}{\partial x} = \beta_1+\beta_3 D $$

For self-study: effect $=5$. For tutor: effect $=5+3=8$. **You cannot meaningfully talk about "the effect of $x$" in an interaction model without specifying which level of $D$ you mean** — this exact trap is a favorite interview question, because people routinely quote $\hat{\beta}_1$ alone as "the effect of hours studied," which is only true for the reference group.

---

## 13.5 Testing Whether the Interaction Is Necessary

Before concluding the slopes genuinely differ, test $H_0: \beta_3=0$ (no interaction — a single common slope suffices) using the same individual t-test machinery from Chapter 2/5, or equivalently a partial F-test (Chapter 5, §5.5) comparing the interaction model to the no-interaction model from §13.2. **Practical guidance from Montgomery and Kutner alike:** if the interaction term is significant, keep both main-effect terms in the model regardless of their own individual significance — removing a main effect while keeping its interaction badly distorts the interpretation of the remaining terms (this is called maintaining **hierarchical/marginality** in the model).

---

## 13.6 Categorical Predictors With More Than Two Levels

For a 3-level factor (self-study / tutor / online), with self-study as reference:

$$ D_{tutor,i} = \mathbb{1}[\text{tutor}], \qquad D_{online,i} = \mathbb{1}[\text{online}] $$

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_{tutor,i}+\beta_3D_{online,i}+\varepsilon_i $$

$\beta_2$ is "tutor vs. self-study" and $\beta_3$ is "online vs. self-study," both **relative to the same reference group** — never directly comparable to each other without an additional calculation (tutor vs. online is $\beta_2-\beta_3$, not something read directly off either coefficient alone).

---

## 13.7 Where the Textbooks Differ

- **Kutner** uses the term "indicator variables" throughout and gives the most complete general treatment of the dummy-variable-trap/multicollinearity connection.
- **Montgomery** emphasizes **effect coding** (using $-1/+1$ instead of $0/1$) as an alternative scheme common in designed experiments, where coefficients are interpreted as deviations from a grand mean rather than from a specific reference category — worth recognizing by name even if $0/1$ dummy coding remains the default in most applied regression work.
- **Sheather** leans on visualizing interaction effects directly — plotting separate fitted lines for each group side by side — as the primary tool for building intuition, over the algebraic decomposition in §13.3–13.4.
- **ESL/ISL**, reflecting a machine-learning perspective, calls this **one-hot encoding** and treats it as a standard, almost automatic preprocessing step rather than a topic requiring careful interpretive discussion — the interpretability concerns in §13.4 are far more central to classical statistics than to ML practice, where predictive accuracy is often the only priority.

---

## 13.8 Interview Q&A

**Q: Why do you use $k-1$ dummy variables for a $k$-level categorical predictor, not $k$?**
A: Including all $k$ creates perfect multicollinearity with the intercept term (the dummy-variable trap) — the $k$ dummy columns would sum to the all-ones intercept column, making $\mathbf{X}^T\mathbf{X}$ non-invertible.

**Q: In a model with an interaction term $x\times D$, what does the coefficient on $x$ alone mean?**
A: It's the effect of $x$ specifically for the reference group ($D=0$) only — not an overall/average effect across all groups. The effect for the other group requires adding the interaction coefficient.

**Q: If your interaction term is statistically significant but one of the main effects isn't, should you drop the non-significant main effect?**
A: Generally no — removing a main effect while retaining its interaction violates the hierarchy/marginality principle and distorts the interpretation of the remaining coefficients; standard practice is to keep both main effects whenever their interaction is retained.

**Q: How would you test whether two groups have significantly different slopes?**
A: Test $H_0:\beta_3=0$ on the interaction coefficient — either via its individual t-test or an equivalent partial F-test comparing the interaction model to a reduced, common-slope model.

**Q: What's the difference between dummy coding and effect coding?**
A: Dummy (0/1) coding interprets coefficients relative to a specific reference category. Effect coding ($-1/+1$) interprets coefficients as deviations from the overall grand mean across all categories — common in designed experiments, less common in general applied regression.

---

*End of Chapter 13. Next: Chapter 14 — Model Selection (stepwise methods, AIC/BIC, adjusted $R^2$, and Mallows' $C_p$ as tools for deciding which predictors actually belong in the model).*
