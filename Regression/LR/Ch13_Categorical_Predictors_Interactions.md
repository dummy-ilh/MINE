# Chapter 13 — Categorical Predictors & Interactions (Mastery Edition v2)

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations, ASCII diagrams, additional real-world examples, two fully-worked numerical examples, and a dedicated section on the practical choices you have to make when modeling categorical predictors.*

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

**More real-world examples of categorical predictors, to build a mental catalog beyond just "tutor vs. self-study":**

| Field | Categorical predictor | Typical numeric predictor it interacts with |
|---|---|---|
| Marketing | Ad channel (social / search / email) | Ad spend |
| Medicine | Treatment vs. control | Dosage or time since treatment |
| Real estate | Urban vs. rural | Square footage |
| Manufacturing | Machine A vs. Machine B | Production speed setting |
| Economics | Recession vs. non-recession year | Interest rate |
| Tech/product | Free-tier vs. paid-tier user | Number of logins per month |

**Why this matters for building intuition:** in every one of these, the question this chapter is built to answer is always the same shape: *"does the category just shift the baseline (parallel lines), or does it change how strongly the numeric predictor matters (fanning/converging lines)?"* — e.g., "does ad spend pay off equally well on social vs. search, or does one channel have sharper diminishing/increasing returns than the other?"

**ASCII picture — the two competing "shapes" this whole chapter is about, side by side:**

```
   NO INTERACTION                        WITH INTERACTION
   (parallel lines)                      (lines that fan out/converge)

 y                                     y
 |                    Tutor            |                         Tutor
 |               ,--*                  |                    ,--*
 |          ,--*                       |               ,--*
 |     ,--*         Self-study         |          ,--*
 |,--*          ,--*                   |     ,--*         ,--* Self-study
 |          ,--*                       |,--*         ,--*
 |     ,--*                            |        ,--*
 +---------------------------- x       +---------------------------- x

 Same slope for both groups —          Different slopes — the lines
 tutor line is just shifted UP         spread apart as x increases
 by a constant amount everywhere       (tutoring isn't just a head
                                        start, it's a steeper climb)
```

This picture is the entire chapter in one glance: **is the extra benefit of one group over another the same at every value of $x$ (left), or does that gap grow or shrink as $x$ changes (right)?**

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

## 13.3 CHOICES — Picking a Reference Level (and Why It's Not Just a Formality)

Before fitting anything, you have to choose *which* category becomes the silent baseline ($D=0$). This choice doesn't change the model's actual predictions or its overall fit — but it changes what every coefficient *directly tells you*, which matters enormously for interpretation.

**Three common ways people choose a reference level, each with a real reason:**

1. **The "natural control" choice.** If one group is obviously a baseline/control (placebo, no-treatment, standard-plan), make that the reference. Every other coefficient then directly answers "how much better/worse than doing nothing?" — usually the most useful framing in medicine, A/B testing, and policy analysis.
2. **The most common / largest group.** If there's no natural control, statistical software (like R) defaults to whichever level is alphabetically first or has the most observations. This isn't wrong, but it's worth deliberately overriding if it produces an awkward comparison (e.g., comparing "Tuesday" against "Friday" as the baseline, purely because T comes before F alphabetically, when "weekday average" would be more meaningful).
3. **The theoretically meaningful comparison.** Sometimes you specifically care about one comparison — e.g., "how does our new product tier compare to the old flagship tier?" — and you should set the reference level to whichever group makes *that* comparison read directly off a single coefficient, rather than requiring extra subtraction.

**What changes and what doesn't, if you flip the reference level:** flip self-study and tutor (make tutor the $D=0$ reference instead), and the intercept and the sign of $\hat\beta_2$ flip accordingly — but the *fitted line for each individual group* stays exactly the same. You're not changing the model, only which coefficient directly hands you which comparison. This is worth stating explicitly in an interview, since it's a common point of confusion: **reference-level choice is a labeling decision, not a modeling decision.**

---

## 13.4 NUMERICAL 1 — Fitting the No-Interaction Model by Hand

Let's actually fit the parallel-lines model from §13.2 to the student dataset, step by step, **before** jumping to the already-fit interaction model in §13.5. This shows exactly what happens when you *force* equal slopes onto data that doesn't really have them.

**Design matrix** (columns: intercept, $x$, $D$):

| Student | Intercept | $x$ | $D$ | $y$ |
|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 45 |
| 2 | 1 | 2 | 0 | 50 |
| 3 | 1 | 3 | 0 | 55 |
| 4 | 1 | 1 | 1 | 58 |
| 5 | 1 | 2 | 1 | 66 |
| 6 | 1 | 3 | 1 | 74 |

**Step 1 — means:** $\bar{x}=2$, $\bar{D}=0.5$, $\bar{y}=58$.

**Step 2 — solving the normal equations** (same machinery as Chapter 4) gives:

$$ \hat\beta_0 = 41.33, \qquad \hat\beta_1 = 6.5, \qquad \hat\beta_2 = 15.33 $$

**Step 3 — fitted values and residuals:**

| Student | Group | $x$ | Fitted $\hat y$ | Actual $y$ | Residual |
|---|---|---|---|---|---|
| 1 | Self-study | 1 | $41.33+6.5(1)=47.83$ | 45 | $-2.83$ |
| 2 | Self-study | 2 | $41.33+6.5(2)=54.33$ | 50 | $-4.33$ |
| 3 | Self-study | 3 | $41.33+6.5(3)=60.83$ | 55 | $-5.83$ |
| 4 | Tutor | 1 | $41.33+15.33+6.5(1)=63.16$ | 58 | $-5.16$ |
| 5 | Tutor | 2 | $41.33+15.33+6.5(2)=69.66$ | 66 | $-3.66$ |
| 6 | Tutor | 3 | $41.33+15.33+6.5(3)=76.16$ | 74 | $-2.16$ |

**The tell-tale sign this model is wrong:** notice the residuals for self-study start small-ish and get **more negative** as $x$ increases ($-2.83\to-4.33\to-5.83$), while tutor's residuals start very negative and **improve** toward zero ($-5.16\to-3.66\to-2.16$). **This is a systematic pattern, not noise** — exactly the fingerprint of a missing interaction term. The parallel-lines model can't bend to match tutor's steeper real slope, so it under-predicts tutor early and over-predicts self-study late, in a completely predictable, structured way.

**ASCII picture of what just happened:**

```
 y                                    Forced parallel-line fit
 76 |                              *  <- actual tutor point,
 74 |                          ,-'      model UNDER-predicts here
    |                      ,-'
 66 |                  ,-*    <- model line (tutor)
    |              ,-'
 58 |          ,-*      <- actual tutor point,
    |      ,-'              model way UNDER-predicts here
    |  ,-*
 55 |,'        *  <- actual self-study point,
    |              model OVER-predicts here (fitted line is above it)
    +------------------------------------- x
     1        2        3

 The straight parallel-line model simply CANNOT capture that
 tutor's real slope is steeper — it compromises with one
 "average" slope for both, missing systematically at both ends.
```

**The formal fix, and the payoff of doing this numerical:** this residual pattern is exactly what should push you toward testing $H_0:\beta_3=0$ (§13.8) and very likely rejecting it — which leads straight into the interaction model worked next.

---

## 13.5 Adding an Interaction Term — Letting the Slope Differ Too

The data above doesn't actually have equal slopes across groups — the tutor group's scores climb faster per hour than self-study's. To let the slope itself differ by group, add an **interaction term**: the product of $x$ and $D$.

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_i+\beta_3(x_i\times D_i)+\varepsilon_i $$

**Plain-English framing before the group-by-group breakdown:** the previous model (§13.2–13.4) only let the two groups have different *starting points* — it forced them to improve at the exact same rate per hour. But look at the actual data: self-study goes up by 5 points per hour (45→50→55), while tutor goes up by 8 points per hour (58→66→74) — a genuinely different *rate* of improvement, not just a different starting point. The interaction term is the tool that lets the model capture "different starting point AND different rate," instead of forcing both groups onto parallel lines.

**Reading it group by group:**

- **Self-study** ($D=0$): $y=\beta_0+\beta_1x$ — intercept $\beta_0$, slope $\beta_1$.
- **Tutor** ($D=1$): $y=(\beta_0+\beta_2)+(\beta_1+\beta_3)x$ — intercept $(\beta_0+\beta_2)$, slope $(\beta_1+\beta_3)$.

**Fitting this model** to the dataset above (design matrix columns: intercept, $x$, $D$, $x\times D$) gives an **exact fit** (constructed that way for clarity):

$$ \hat{\beta}_0=40,\quad \hat{\beta}_1=5,\quad \hat{\beta}_2=10,\quad \hat{\beta}_3=3 $$

**Verification** (tutor group, $x=2$): $(40+10)+(5+3)(2) = 50+16=66$ — matches the table exactly. Every other row checks out the same way.

**ASCII picture — this time the model actually captures the fanning-out pattern:**

```
 y
 74 |                                *  <- exact match now
 66 |                          *
    |                    ,-'  (tutor slope = 8/hour)
 58 |              *
    |
 55 |                    *  <- exact match
 50 |              *          (self-study slope = 5/hour)
 45 |        *
    +------------------------------------- x
     1        2        3

 Two DIFFERENT slopes, correctly diverging — zero residuals,
 because this model has the right shape to match the data.
```

---

## 13.6 What Each Coefficient Actually Means Here

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

## 13.7 NUMERICAL 2 — A Second Worked Example, With Interaction That Goes the Other Way (Converging Lines)

To make sure the concept isn't only understood in the "gap grows wider" direction, here's a second, independent numerical example where the interaction makes two groups **converge** instead of diverge — a common real-world pattern (e.g., a new drug's benefit over placebo shrinking at higher doses due to a ceiling effect).

**New tiny dataset** — a drug trial, dose level ($x$, in mg) vs. symptom improvement score ($y$), for Drug and Placebo groups:

| Group | $x$ (dose, mg) | $y$ (improvement score) |
|---|---|---|
| Placebo | 1 | 10 |
| Placebo | 2 | 14 |
| Placebo | 3 | 18 |
| Drug | 1 | 25 |
| Drug | 2 | 27 |
| Drug | 3 | 29 |

**Setting up the model** ($D=1$ for Drug, $D=0$ for Placebo):

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_i+\beta_3(x_i\times D_i)+\varepsilon_i $$

**Solving (design matrix machinery identical to §13.5), this fits exactly:**

$$ \hat\beta_0=6,\quad \hat\beta_1=4,\quad \hat\beta_2=17,\quad \hat\beta_3=-2 $$

**Verification** (Drug group, $x=2$): $(6+17)+(4-2)(2) = 23+4=27$ — matches the table.

**Reading the coefficients — notice $\hat\beta_3$ is now NEGATIVE:**

- **Placebo** ($D=0$): slope $=\hat\beta_1=4$ points per mg.
- **Drug** ($D=1$): slope $=\hat\beta_1+\hat\beta_3=4-2=2$ points per mg.

**In plain words:** placebo's effect keeps climbing steadily (4 points per mg, no ceiling in sight). The drug starts far ahead ($\hat\beta_2=17$-point head start at dose zero) but its *additional* benefit per extra mg is actually smaller (only 2 points per mg) — a classic **ceiling effect**, where a treatment that's already working well has less room left to improve further. The gap between drug and placebo is **shrinking** as dose increases, even though drug is still ahead at every dose shown here.

**ASCII picture — the converging-lines case:**

```
 y
 29 |                                    *  Drug (gap = 11)
 27 |                              *
 25 |                        *
    |                                          (gap keeps
 18 |                                    *      SHRINKING
 14 |                              *             as x grows)
 10 |                        *
    +------------------------------------- x
     1        2        3

 Drug starts WAY ahead (gap=15 at x=1) but placebo is catching
 up (gap=11 at x=3) — a NEGATIVE interaction coefficient means
 the lines converge instead of fan out.
```

**The general lesson from comparing Numerical 1 (implicitly diverging tutor data) and Numerical 2 (explicitly converging drug data):** the **sign** of $\hat\beta_3$ tells you the direction of the interaction — positive means the reference group's advantage (or disadvantage) grows with $x$; negative means it shrinks. The **size** of $\hat\beta_3$ relative to $\hat\beta_1$ tells you how dramatic that change is. Always check both.

---

## 13.8 NUMERICAL 3 — A Third Example: When There's NO Real Interaction (a "Negative Control")

It's just as important to see a case where fitting the interaction model *doesn't* pay off — so you know what "no interaction" genuinely looks like numerically, not just conceptually.

**Dataset** — website A/B test, ad spend ($x$, in hundreds of dollars) vs. signups ($y$), for two channels:

| Channel | $x$ (spend, $100s) | $y$ (signups) |
|---|---|---|
| Social | 1 | 12 |
| Social | 2 | 17 |
| Social | 3 | 22 |
| Search | 1 | 20 |
| Search | 2 | 25 |
| Search | 3 | 30 |

**Both channels climb by exactly 5 signups per $100 spent** — a genuinely constant, equal slope. Fitting the full interaction model here would give $\hat\beta_3\approx0$ (no meaningful interaction) — search simply has a constant 8-signup head start ($\hat\beta_2=8$) at every spend level, with no change in slope.

**Why this matters as a lesson:** don't reflexively add interaction terms to every categorical-numeric pair just because Chapters 13.3–13.7 showed how useful they *can* be. If the true slopes really are parallel, the extra interaction term adds unnecessary complexity (Chapter 14's model-selection criteria would penalize it) without buying any real explanatory power. **Always let the data (via the significance test in §13.9) decide, rather than assuming interaction is always present.**

---

## 13.9 Testing Whether the Interaction Is Necessary

Before concluding the slopes genuinely differ, test $H_0: \beta_3=0$ (no interaction — a single common slope suffices) using the same individual t-test machinery from Chapter 2/5, or equivalently a partial F-test (Chapter 5, §5.5) comparing the interaction model to the no-interaction model from §13.2. **Practical guidance from Montgomery and Kutner alike:** if the interaction term is significant, keep both main-effect terms in the model regardless of their own individual significance — removing a main effect while keeping its interaction badly distorts the interpretation of the remaining terms (this is called maintaining **hierarchical/marginality** in the model).

**In plain words:** before believing "tutoring really does help more per hour," you should formally check whether that pattern could just be noise — the same kind of check (a t-test or F-test) you've been running since Chapter 2, just applied to the new interaction coefficient. And the "keep both main effects" rule exists because dropping, say, $\hat\beta_1$ while keeping $\hat\beta_3$ (the interaction) would leave your remaining coefficients meaning something entirely different and much harder to interpret cleanly — it's a bit like removing the foundation of a building while insisting on keeping the upper floors exactly as they are.

---

## 13.10 CHOICES — A Practical Decision Guide for This Whole Chapter

Putting §13.3 (reference level) and §13.9 (interaction testing) together with a few more common real-world decision points, here's a consolidated guide for the choices you'll actually face:

```
  START: I have a categorical predictor and a numeric predictor.

  CHOICE 1 — How many categories?
    2 levels  --> 1 dummy variable (D = 0/1)
    k levels  --> k-1 dummy variables (never k — dummy variable trap)

  CHOICE 2 — Which level is the reference?
    Natural control/baseline exists?  --> use it (most interpretable)
    No natural baseline?              --> pick whichever makes your
                                           main comparison of interest
                                           read directly off ONE coefficient

  CHOICE 3 — Dummy (0/1) or effect (-1/+1) coding?
    Standard applied regression, comparing to one baseline --> dummy coding
    Designed experiment, comparing to the grand mean        --> effect coding

  CHOICE 4 — Include an interaction term?
    Look at the data/residuals: do the group's rates of
    change visibly differ (Numerical 1, 2) or look the
    same (Numerical 3)?
       Visibly differ / unsure --> fit WITH interaction, then TEST it
       Look the same           --> can still fit WITH interaction and
                                    test it — let the t-test/F-test decide,
                                    don't skip the test just because it
                                    "looks" parallel

  CHOICE 5 — Interaction term tests significant?
    YES --> keep both main effects regardless of their own
            individual significance (hierarchy principle)
    NO  --> drop the interaction term, revert to the simpler
            parallel-lines model (more interpretable, fewer
            parameters, likely better by Chapter 14's criteria)
```

**The single biggest practical mistake this guide is meant to prevent:** deciding whether to include an interaction term by eyeballing a plot alone, without ever running the formal test in §13.9. Numerical 1's residual pattern was dramatic and obvious; real data is very often much noisier, and what looks like "clearly different slopes" by eye can turn out to be statistically indistinguishable from a common slope once you actually test it — and vice versa, subtle-looking differences can be genuinely significant with enough data.

---

## 13.11 Categorical Predictors With More Than Two Levels

For a 3-level factor (self-study / tutor / online), with self-study as reference:

$$ D_{tutor,i} = \mathbb{1}[\text{tutor}], \qquad D_{online,i} = \mathbb{1}[\text{online}] $$

$$ y_i = \beta_0+\beta_1x_i+\beta_2D_{tutor,i}+\beta_3D_{online,i}+\varepsilon_i $$

$\beta_2$ is "tutor vs. self-study" and $\beta_3$ is "online vs. self-study," both **relative to the same reference group** — never directly comparable to each other without an additional calculation (tutor vs. online is $\beta_2-\beta_3$, not something read directly off either coefficient alone).

**Plain-language version:** with three categories, you now have two "on/off switches" instead of one, and self-study is still the silent baseline both switches are measured against. $\beta_2$ tells you "how much better is tutor than self-study," and $\beta_3$ tells you "how much better is online than self-study" — but neither one directly tells you "how does tutor compare to online." For that specific comparison, you have to subtract the two coefficients from each other ($\beta_2-\beta_3$), since neither coefficient was ever measuring that pairing directly in the first place.

**A worked micro-example of the subtraction trick:** suppose $\hat\beta_2=10$ (tutor is 10 points better than self-study) and $\hat\beta_3=4$ (online is 4 points better than self-study). Then tutor vs. online $=\hat\beta_2-\hat\beta_3=10-4=6$ — tutor beats online by 6 points. Note this comparison also has its *own* standard error, computed from the full variance-covariance matrix (Chapter 3) — you can't just eyeball "10 vs. 4 looks like a gap of 6 with unknown uncertainty," you'd formally need $\text{Var}(\hat\beta_2-\hat\beta_3)=\text{Var}(\hat\beta_2)+\text{Var}(\hat\beta_3)-2\text{Cov}(\hat\beta_2,\hat\beta_3)$ to test whether that 6-point gap is itself significant.

---

## 13.12 Where the Textbooks Differ

- **Kutner** uses the term "indicator variables" throughout and gives the most complete general treatment of the dummy-variable-trap/multicollinearity connection.
- **Montgomery** emphasizes **effect coding** (using $-1/+1$ instead of $0/1$) as an alternative scheme common in designed experiments, where coefficients are interpreted as deviations from a grand mean rather than from a specific reference category — worth recognizing by name even if $0/1$ dummy coding remains the default in most applied regression work.
- **Sheather** leans on visualizing interaction effects directly — plotting separate fitted lines for each group side by side — as the primary tool for building intuition, over the algebraic decomposition in §13.5–13.6.
- **ESL/ISL**, reflecting a machine-learning perspective, calls this **one-hot encoding** and treats it as a standard, almost automatic preprocessing step rather than a topic requiring careful interpretive discussion — the interpretability concerns in §13.6 are far more central to classical statistics than to ML practice, where predictive accuracy is often the only priority.

---

## 13.13 Interview Q&A

**Q: Why do you use $k-1$ dummy variables for a $k$-level categorical predictor, not $k$?**
A: Including all $k$ creates perfect multicollinearity with the intercept term (the dummy-variable trap) — the $k$ dummy columns would sum to the all-ones intercept column, making $\mathbf{X}^T\mathbf{X}$ non-invertible.
*(Simple version: using all $k$ categories secretly duplicates the intercept column, which breaks the math the same way trying to solve one equation for two unknowns does.)*

**Q: In a model with an interaction term $x\times D$, what does the coefficient on $x$ alone mean?**
A: It's the effect of $x$ specifically for the reference group ($D=0$) only — not an overall/average effect across all groups. The effect for the other group requires adding the interaction coefficient.
*(Simple version: it's the effect for the baseline group only — you have to ask "for which group?" before trusting it.)*

**Q: Does the choice of reference level change your model's predictions?**
A: No — it only changes which comparisons each coefficient directly represents. The fitted line for every individual group is identical no matter which level you designate as the reference; it's purely a labeling/interpretation choice, not a modeling choice.

**Q: What does a negative interaction coefficient mean, physically, versus a positive one?**
A: A positive interaction coefficient means the non-reference group's slope is steeper than the reference group's — the gap between groups widens as $x$ increases. A negative interaction coefficient means the opposite — the non-reference group's slope is flatter, so the gap narrows as $x$ increases (a "ceiling effect" or converging pattern).
*(Simple version: positive interaction = lines fan apart; negative interaction = lines converge.)*

**Q: If your interaction term is statistically significant but one of the main effects isn't, should you drop the non-significant main effect?**
A: Generally no — removing a main effect while retaining its interaction violates the hierarchy/marginality principle and distorts the interpretation of the remaining coefficients; standard practice is to keep both main effects whenever their interaction is retained.
*(Simple version: don't remove the foundation while keeping the upper floors — main effects and their interactions travel together.)*

**Q: How would you test whether two groups have significantly different slopes?**
A: Test $H_0:\beta_3=0$ on the interaction coefficient — either via its individual t-test or an equivalent partial F-test comparing the interaction model to a reduced, common-slope model.
*(Simple version: check whether the interaction coefficient is significantly different from zero — the same t-test or F-test machinery you already know.)*

**Q: What's the difference between dummy coding and effect coding?**
A: Dummy (0/1) coding interprets coefficients relative to a specific reference category. Effect coding ($-1/+1$) interprets coefficients as deviations from the overall grand mean across all categories — common in designed experiments, less common in general applied regression.
*(Simple version: dummy coding compares each group to one chosen baseline group; effect coding compares each group to the overall average of everyone.)*

**Q: If you fit a no-interaction (parallel-lines) model to data that actually has different slopes per group, what pattern would you expect to see in the residuals?**
A: A systematic pattern within each group as $x$ changes — e.g., the group with the true steeper slope will be under-predicted at low $x$ and over-predicted at high $x$ (or vice versa for the flatter-sloped group), since the forced common slope is a compromise between the two true slopes. This is a direct sign, per Chapter 7, that an interaction term is missing.

**Q: With a 3-level categorical predictor, how would you test whether two non-reference levels differ significantly from each other (e.g., tutor vs. online, neither of which is the reference)?**
A: Compute the difference of their coefficients (e.g., $\hat\beta_2-\hat\beta_3$) and its standard error using the full variance-covariance matrix — $\text{Var}(\hat\beta_2-\hat\beta_3)=\text{Var}(\hat\beta_2)+\text{Var}(\hat\beta_3)-2\text{Cov}(\hat\beta_2,\hat\beta_3)$ — rather than assuming the two individual t-tests (each vs. the reference group) answer this directly; neither coefficient alone tests that specific pairwise comparison.

**Q: Should you always include an interaction term when you have a categorical and a numeric predictor together?**
A: No — only if theory or the data suggests the relationship's slope genuinely differs by group. Always test $H_0:\beta_3=0$ rather than assuming interaction is present; adding an unnecessary interaction term increases model complexity without added explanatory power, which model-selection criteria (Chapter 14) would penalize.

---

*End of Chapter 13. Next: Chapter 14 — Model Selection (stepwise methods, AIC/BIC, adjusted $R^2$, and Mallows' $C_p$ as tools for deciding which predictors actually belong in the model).*
