# Chapter 13 — Categorical Predictors & Interactions

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations. Introduces a new small dataset with a categorical predictor, since none of the running datasets so far have included one.*

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

**Plain-language framing before anything else:** every formula we've used so far has needed actual numbers to multiply against — you can't do "$\beta_2 \times \text{"tutor"}$" the way you can do "$\beta_1 \times 3\text{ hours}$." So the whole first half of this chapter is about a simple but important trick: relabeling a category ("tutor" vs. "self-study") as a plain 0-or-1 number, so the same regression machinery from every earlier chapter can chew on it without any changes. The second half is about a genuinely new idea: letting the *slope itself* — not just the starting point — differ between groups, which is what "interaction" means.

---

## 13.2 Dummy (Indicator) Coding

For a two-level category, create a single **indicator variable**:

$$ D_i = \begin{cases}0 & \text{if self-study}\\1 & \text{if tutor}\end{cases} $$

The chosen $D=0$ group (self-study, here) is called the **reference level** — every other category's effect is interpreted *relative to it*. The model:

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_i+\varepsilon_i $$

**Reading each coefficient:** for self-study ($D=0$), the model reduces to $y=\beta_0+\beta_1x$. For tutor ($D=1$), it becomes $y=(\beta_0+\beta_2)+\beta_1x$ — **the same slope, but a shifted intercept.** $\beta_2$ is literally "the vertical shift in the line for the tutor group, relative to self-study, holding hours studied fixed." This model (no interaction yet) forces both groups to have the *same slope* — an assumption we test and relax next.

**Plain-language walkthrough of what's happening:** think of $D$ as an on/off switch. When it's "off" (self-study, $D=0$), the $\beta_2 D$ term disappears entirely — you're left with the plain-old regression line you already know from Chapter 1. When it's "on" (tutor, $D=1$), that same $\beta_2$ term switches on and adds a fixed bump to the intercept. So this simple version of the model says: "both groups improve at exactly the same rate per hour studied, but one group just starts from a higher baseline." Picture two parallel lines on a graph, one sitting above the other, always the same vertical distance apart — that's what this model forces onto the data, whether or not it's actually true.

**General rule for $k$ categories:** you need $k-1$ dummy variables, not $k$. Including all $k$ would create perfect multicollinearity with the intercept (the **dummy variable trap**) — the $k$ dummy columns plus the intercept column would sum to a constant vector, making $\mathbf{X}^T\mathbf{X}$ singular (a direct callback to Chapter 3, §3.4, and Chapter 9's multicollinearity discussion). For a 3-level category (e.g., self-study/tutor/online), you'd use 2 dummies, with one level as the omitted reference.

**Why the "dummy variable trap" happens, in plain words:** imagine you tried to use a separate 0/1 column for *every* category, including self-study itself. Then for any given student, exactly one of those columns is a 1 and all the rest are 0's — meaning if you add up all the category columns for any row, you always get exactly 1. But the intercept column is *also* always 1 for every row. So you'd have two different sets of columns that are secretly saying the exact same thing ("this row = 1, always") — that's perfect redundancy, the regression equivalent of trying to solve for two unknowns using only one real equation. Dropping one category (making it the reference) breaks that redundancy and lets the math work again.

---

## 13.3 Adding an Interaction Term — Letting the Slope Differ Too

The data above doesn't actually have equal slopes across groups — the tutor group's scores climb faster per hour than self-study's. To let the slope itself differ by group, add an **interaction term**: the product of $x$ and $D$.

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_i+\beta_3(x_i\times D_i)+\varepsilon_i $$

**Plain-English framing before the group-by-group breakdown:** the previous model (§13.2) only let the two groups have different *starting points* — it forced them to improve at the exact same rate per hour. But look at the actual data: self-study goes up by 5 points per hour (45→50→55), while tutor goes up by 8 points per hour (58→66→74) — a genuinely different *rate* of improvement, not just a different starting point. The interaction term is the tool that lets the model capture "different starting point AND different rate," instead of forcing both groups onto parallel lines.

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

**Plain-language version of the whole table, in one paragraph:** if you never studied at all ($x=0$), tutoring alone gives you a 10-point head start over self-study — that's $\hat\beta_2$. But it's not just a head start: every additional hour of tutoring is *also* worth 3 points more than an additional hour of self-study — that's $\hat\beta_3$. So tutoring helps you in two separate ways at once: a one-time bonus right at the start, plus a bigger payoff for every hour you put in afterward. That's the real-world story hiding behind these four numbers.

**The single most commonly misread coefficient in this entire model is $\hat{\beta}_1$.** With the interaction term present, $\hat{\beta}_1$ is **not** "the overall effect of hours studied" — it's specifically **the effect of hours studied for the reference group only** ($D=0$). The *actual* effect of one more hour of study depends on which group you're in:

$$ \frac{\partial y}{\partial x} = \beta_1+\beta_3 D $$

For self-study: effect $=5$. For tutor: effect $=5+3=8$. **You cannot meaningfully talk about "the effect of $x$" in an interaction model without specifying which level of $D$ you mean** — this exact trap is a favorite interview question, because people routinely quote $\hat{\beta}_1$ alone as "the effect of hours studied," which is only true for the reference group.

**Why this trap is so easy to fall into, in plain words:** in every model *before* this chapter, "the coefficient on $x$" was a single, universal number — one effect size that applied to everyone in the dataset. Once you add an interaction term, that's no longer true, but the label "$\beta_1$, the coefficient on $x$" *looks* exactly the same as before, so it's natural to keep reading it the old way out of habit. The fix is simple once you remember it: whenever an interaction term is in the model, always ask "for which group?" before quoting any effect size involving that interacting variable.

---

## 13.5 Testing Whether the Interaction Is Necessary

Before concluding the slopes genuinely differ, test $H_0: \beta_3=0$ (no interaction — a single common slope suffices) using the same individual t-test machinery from Chapter 2/5, or equivalently a partial F-test (Chapter 5, §5.5) comparing the interaction model to the no-interaction model from §13.2. **Practical guidance from Montgomery and Kutner alike:** if the interaction term is significant, keep both main-effect terms in the model regardless of their own individual significance — removing a main effect while keeping its interaction badly distorts the interpretation of the remaining terms (this is called maintaining **hierarchical/marginality** in the model).

**In plain words:** before believing "tutoring really does help more per hour," you should formally check whether that pattern could just be noise — the same kind of check (a t-test or F-test) you've been running since Chapter 2, just applied to the new interaction coefficient. And the "keep both main effects" rule exists because dropping, say, $\hat\beta_1$ while keeping $\hat\beta_3$ (the interaction) would leave your remaining coefficients meaning something entirely different and much harder to interpret cleanly — it's a bit like removing the foundation of a building while insisting on keeping the upper floors exactly as they are.

---

## 13.6 Categorical Predictors With More Than Two Levels

For a 3-level factor (self-study / tutor / online), with self-study as reference:

$$ D_{tutor,i} = \mathbb{1}[\text{tutor}], \qquad D_{online,i} = \mathbb{1}[\text{online}] $$

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_{tutor,i}+\beta_3D_{online,i}+\varepsilon_i $$

$\beta_2$ is "tutor vs. self-study" and $\beta_3$ is "online vs. self-study," both **relative to the same reference group** — never directly comparable to each other without an additional calculation (tutor vs. online is $\beta_2-\beta_3$, not something read directly off either coefficient alone).

**Plain-language version:** with three categories, you now have two "on/off switches" instead of one, and self-study is still the silent baseline both switches are measured against. $\beta_2$ tells you "how much better is tutor than self-study," and $\beta_3$ tells you "how much better is online than self-study" — but neither one directly tells you "how does tutor compare to online." For that specific comparison, you have to subtract the two coefficients from each other ($\beta_2-\beta_3$), since neither coefficient was ever measuring that pairing directly in the first place.

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
*(Simple version: using all $k$ categories secretly duplicates the intercept column, which breaks the math the same way trying to solve one equation for two unknowns does.)*

**Q: In a model with an interaction term $x\times D$, what does the coefficient on $x$ alone mean?**
A: It's the effect of $x$ specifically for the reference group ($D=0$) only — not an overall/average effect across all groups. The effect for the other group requires adding the interaction coefficient.
*(Simple version: it's the effect for the baseline group only — you have to ask "for which group?" before trusting it.)*

**Q: If your interaction term is statistically significant but one of the main effects isn't, should you drop the non-significant main effect?**
A: Generally no — removing a main effect while retaining its interaction violates the hierarchy/marginality principle and distorts the interpretation of the remaining coefficients; standard practice is to keep both main effects whenever their interaction is retained.
*(Simple version: don't remove the foundation while keeping the upper floors — main effects and their interactions travel together.)*

**Q: How would you test whether two groups have significantly different slopes?**
A: Test $H_0:\beta_3=0$ on the interaction coefficient — either via its individual t-test or an equivalent partial F-test comparing the interaction model to a reduced, common-slope model.
*(Simple version: check whether the interaction coefficient is significantly different from zero — the same t-test or F-test machinery you already know.)*

**Q: What's the difference between dummy coding and effect coding?**
A: Dummy (0/1) coding interprets coefficients relative to a specific reference category. Effect coding ($-1/+1$) interprets coefficients as deviations from the overall grand mean across all categories — common in designed experiments, less common in general applied regression.
*(Simple version: dummy coding compares each group to one chosen baseline group; effect coding compares each group to the overall average of everyone.)*

---

*End of Chapter 13. Next: Chapter 14 — Model Selection (stepwise methods, AIC/BIC, adjusted $R^2$, and Mallows' $C_p$ as tools for deciding which predictors actually belong in the model).*
