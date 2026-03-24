Below are **clean, notebook-ready Markdown notes** on **“Causal meaning with respect to regression and experiments”**, written at a **graduate / FAANG-interview level**.

---

# 📌 Causal Meaning in Regression & Experiments

## 🔹 What Does *Causal* Mean?

A relationship is **causal** if changing **X** *directly produces* a change in **Y**, **holding everything else constant**.

> **Causation ≠ Correlation**

* **Correlation**: X and Y move together
* **Causation**: Intervening on X changes Y

Mathematically:
[
\text{Causal Effect} = \mathbb{E}[Y \mid do(X=x_1)] - \mathbb{E}[Y \mid do(X=x_0)]
]

The `do(·)` operator means **active intervention**, not passive observation.

---

## 🔹 Regression: What It Does (and Does NOT) Guarantee

### Standard Regression Model

[
Y = \beta_0 + \beta_1 X + \varepsilon
]

* (\beta_1) measures **association**
* It is **causal only under strong assumptions**

---

## 🔹 When Does Regression Have a Causal Interpretation?

Regression coefficient (\beta_1) is **causal** *iff*:

### ✅ 1. No Confounding (Exogeneity)

[
\mathbb{E}[\varepsilon \mid X] = 0
]

Meaning:

* All variables that affect both **X** and **Y** are:

  * Controlled for, **or**
  * Absent

📌 Example:

* X = Education
* Y = Income
* Confounder = Ability
  ➡ Omitted ability biases (\beta_1)

---

### ✅ 2. Correct Model Specification

* Functional form is correct
* No important nonlinearities omitted
* No omitted interactions

---

### ✅ 3. No Reverse Causality

* X → Y (not Y → X)

📌 Example:

* Ads → Sales (OK)
* Sales → Ads (violates causality)

---

### ✅ 4. Stable Units (SUTVA)

* No interference between units
* One person’s treatment doesn’t affect another’s outcome

---

## 🔹 Why Regression Alone Is Usually *Not* Causal

### ❌ Omitted Variable Bias

[
\hat{\beta}_1 = \beta_1 + \text{Bias}
]

Bias arises when:

* Z affects both X and Y
* Z is not included

---

### ❌ Selection Bias

* Individuals **self-select** into X
* Common in observational data

📌 Example:

* People who exercise more are healthier
* But health-conscious people choose exercise

---

### ❌ Post-Treatment Bias

* Controlling for variables affected by X
* Destroys causal meaning

---

## 🔹 Experiments: Gold Standard for Causality

### Randomized Controlled Trial (RCT)

Randomization ensures:
[
X \perp \varepsilon
]

➡ Treatment is **independent of confounders**

---

### Key Idea

Randomization simulates:

> “All else equal”

So:
[
\mathbb{E}[Y \mid X=1] - \mathbb{E}[Y \mid X=0]
]
**is causal**

---

## 🔹 Regression in Experiments

Regression **inside an experiment** is causal because:

* Randomization eliminates confounding
* Regression improves **precision**, not identification

Example:
[
Y = \beta_0 + \beta_1 Treatment + \beta_2 Controls + \varepsilon
]

Here:

* (\beta_1) = **Average Treatment Effect (ATE)**

---

## 🔹 Observational Data vs Experimental Data

| Aspect                 | Observational    | Experimental |
| ---------------------- | ---------------- | ------------ |
| Treatment Assignment   | Self-selected    | Random       |
| Confounding            | Likely           | Eliminated   |
| Regression Coefficient | Associational    | Causal       |
| Identification         | Assumption-heavy | Design-based |

---

## 🔹 Causal Language: What You Can & Cannot Say

### ❌ Regression Alone

> “X is associated with Y”

### ✅ With Strong Assumptions / Experiment

> “Increasing X causes Y to increase by β units”

---

## 🔹 Common Interview Trap

**Q:** “Does regression imply causation?”
**A:**

> No. Regression estimates causal effects **only if** treatment assignment is exogenous, typically ensured by randomization or strong identification assumptions.

---

## 🔹 How Causality Is Achieved Without Experiments

Methods that **restore causal meaning**:

* Randomized experiments
* Instrumental Variables (IV)
* Difference-in-Differences (DiD)
* Regression Discontinuity (RD)
* Matching / Propensity Scores

(All try to approximate `do(X)`)

---

## 🔹 Mental Model (Very Important)

* **Regression** → *Description*
* **Causal inference** → *Intervention*
* **Design > Model**

> Causality comes from **how data is generated**, not how fancy the regression is.

---

## 🔹 One-Line Summary

> Regression estimates **associations**.
> Experiments (or strong identification strategies) turn those associations into **causal effects**.

---

Below are **high-signal, interview-grade *tricky causal inference questions*** with **precise answers**.
These are exactly the kind of questions used to test whether you **actually understand causality**, not just regression.

---

# 🎯 Tricky Causality Questions (Regression & Experiments)

## 1️⃣ *“If I control for everything I can measure, is the regression causal?”*

### ❌ Tempting Answer

> Yes, because we controlled for all variables.

### ✅ Correct Answer

> No. Causality requires controlling for **all confounders**, including **unobserved** ones. If any unobserved variable affects both X and Y, the estimate remains biased.

📌 Key Insight

> **Controlling for many variables ≠ controlling for the right variables**

---

## 2️⃣ *“If regression coefficient is statistically significant, does it imply causality?”*

### ❌ Wrong

> Yes, significance means effect exists.

### ✅ Correct

> No. Statistical significance only indicates a **non-zero association**, not causality.

📌 Example
Ice cream sales significantly predict drowning deaths — but temperature is the confounder.

---

## 3️⃣ *“Why does randomization make regression causal?”*

### ❌ Shallow

> Because it removes bias.

### ✅ Deep

> Randomization makes treatment **independent of both observed and unobserved confounders**, ensuring:
> [
> \mathbb{E}[\varepsilon \mid X] = 0
> ]
> which is the key condition for causal interpretation.

---

## 4️⃣ *“Can adding more control variables ever increase bias?”*

### ✅ Yes — and this is very tricky

**Reasons:**

1. **Post-treatment bias**

   * Controlling for variables caused by treatment blocks causal paths.
2. **Collider bias**

   * Conditioning on a common effect creates spurious correlation.

📌 Interview gold line:

> “Bad controls can introduce bias even when good controls remove it.”

---

## 5️⃣ *“Is a randomized experiment always unbiased?”*

### ❌ Common belief

> Yes.

### ✅ Correct

> No. Bias can still arise due to:

* Non-compliance
* Attrition
* Spillover effects
* Measurement error

Randomization identifies **intent-to-treat (ITT)**, not always the true treatment effect.

---

## 6️⃣ *“Why shouldn’t we control for a mediator?”*

### ❌ Incorrect

> Because it’s unnecessary.

### ✅ Correct

> Because mediators lie **on the causal path** from X to Y. Controlling for them removes part of the causal effect.

📌 Example

* X = Education
* M = Occupation
* Y = Income

Controlling for occupation removes part of education’s effect.

---

## 7️⃣ *“What does the regression coefficient measure in an experiment?”*

### ❌ Weak

> Effect of treatment.

### ✅ Strong

> The **Average Treatment Effect (ATE)**, assuming perfect randomization and no interference.

Regression is used for **precision**, not identification.

---

## 8️⃣ *“Can correlation be causal?”*

### ✅ Correct Answer

> Yes — if the correlation is generated by a causal mechanism, typically through randomization or a valid identification strategy.

📌 Important:

> Correlation is necessary for causation, but not sufficient.

---

## 9️⃣ *“What exactly breaks causality in observational regression?”*

### Best Answer

> Endogeneity — arising from confounding, reverse causality, or measurement error.

---

## 🔟 *“Why is ‘no omitted variables’ an untestable assumption?”*

### Key Insight

> Because you cannot test for variables you do not observe.

This is why **design-based** methods dominate causal inference.

---

## 1️⃣1️⃣ *“Is R² relevant for causal inference?”*

### ❌ Common trap

> Higher R² means better causal model.

### ✅ Correct

> R² measures predictive power, not causal validity. A model can have high R² and zero causal meaning.

---

## 1️⃣2️⃣ *“Can a bad experiment be worse than observational data?”*

### ✅ Yes

If:

* Randomization is compromised
* Units interfere
* Sample is non-representative
* Ethical or practical constraints distort behavior

---

## 1️⃣3️⃣ *“What does the `do(X)` operator change compared to conditioning?”*

### Perfect Answer

> Conditioning observes X; `do(X)` forces X. They are equal **only when there is no confounding**.

[
P(Y \mid X) \neq P(Y \mid do(X))
]

---

## 1️⃣4️⃣ *“Why is causal inference more about design than statistics?”*

### Model Answer

> Because causal identification comes from how data is generated, not from how complex the model is.

📌 Killer line:

> “You can’t regression your way out of bad data.”

---

## 1️⃣5️⃣ *“Give a one-sentence definition of causality.”*

### Interview-Perfect

> X causes Y if intervening on X changes the distribution of Y, holding everything else fixed.

---

## 🧠 Meta Interview Tip

When unsure, anchor your answer to:

* **Exogeneity**
* **Randomization**
* **Intervention vs observation**

Those three words signal real causal understanding.

---

In **simple words**:

> **Causal means “X makes Y happen.”**

Not just that they move together — but that **if you change X, Y will change because of it**.

---

### A very plain explanation

* **Correlation**:

  > When X changes, Y also changes
  > (they are related)

* **Causation**:

  > Changing X **forces** Y to change
  > (X is the reason)

---

### Simple example

* 🟢 **Causal**
  Turning the switch **causes** the light to turn on.

* 🔴 **Not causal (just correlation)**
  Ice-cream sales and drowning both increase in summer.
  Ice-cream does **not** cause drowning — heat causes both.

---

### In regression / experiments

* Regression alone says:

  > “X is associated with Y”

* Causal claim says:

  > “If I intervene and change X, Y will change”

---

### One-line memory rule

> **Causal = what would happen if I deliberately changed it**




