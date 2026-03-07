# Chapter 7: Multiple Regression II


---

## Table of Contents
1. [Extra Sums of Squares — Basic Ideas](#extra-sums-of-squares)
2. [Definitions and Decompositions](#definitions-and-decompositions)
3. [Uses of Extra SS in Tests for Regression Coefficients](#uses-in-tests)
4. [Summary of All Tests Concerning Regression Coefficients](#summary-of-tests)
5. [Coefficients of Partial Determination](#partial-determination)
6. [Standardized Multiple Regression Model](#standardized-model)
7. [Multicollinearity and Its Effects](#multicollinearity)
8. [Key Formulas Reference Sheet](#formulas-reference)
9. [Common Mistakes & Misconceptions](#common-mistakes)
10. [Practice Problems](#practice-problems)

---

## 1. Extra Sums of Squares — Basic Ideas {#extra-sums-of-squares}

### The Core Concept

An **extra sum of squares** measures the *marginal reduction* in the error sum of squares when one or more predictor variables are **added** to a regression model that already contains other predictors.

Equivalently, an extra SS measures the *marginal increase* in the regression sum of squares from adding those same predictors.

> 💡 **Intuition:** You already have a model with $X_1$. How much does adding $X_2$ *additionally* help explain $Y$, beyond what $X_1$ already explains? That's the extra sum of squares $SSR(X_2|X_1)$.

This is fundamentally a **conditional** question: given what's already in the model, what more does the new variable contribute?

---

### Motivating Example: Body Fat Study

20 healthy females aged 25–34. We want to predict **body fat** ($Y$) from easy-to-measure body dimensions:
- $X_1$ = triceps skinfold thickness
- $X_2$ = thigh circumference
- $X_3$ = midarm circumference

The data (Table 7.1) contains 20 cases. We fit four different models and record their sums of squares (Table 7.2):

| Model | $SSR$ | $SSE$ | $df_E$ |
|-------|-------|-------|--------|
| $Y$ on $X_1$ alone | 352.27 | 143.12 | 18 |
| $Y$ on $X_2$ alone | 381.97 | 113.42 | 18 |
| $Y$ on $X_1, X_2$ | 385.44 | 109.95 | 17 |
| $Y$ on $X_1, X_2, X_3$ | 396.98 | 98.41 | 16 |
| **Total ($SSTO$)** | — | **495.39** | 19 |

**Computing an extra SS:** When $X_2$ is added to a model already containing $X_1$:

$$SSR(X_2|X_1) = SSE(X_1) - SSE(X_1, X_2) = 143.12 - 109.95 = 33.17$$

Equivalently (same number!):

$$SSR(X_2|X_1) = SSR(X_1, X_2) - SSR(X_1) = 385.44 - 352.27 = 33.17$$

**Adding $X_3$ given $X_1$ and $X_2$ are already in:**

$$SSR(X_3|X_1, X_2) = SSE(X_1, X_2) - SSE(X_1, X_2, X_3) = 109.95 - 98.41 = 11.54$$

**Adding both $X_2$ and $X_3$ given $X_1$ is already in:**

$$SSR(X_2, X_3|X_1) = SSE(X_1) - SSE(X_1, X_2, X_3) = 143.12 - 98.41 = 44.71$$

Note that this multi-variable extra SS can also be computed as:
$$SSR(X_2, X_3|X_1) = SSR(X_1, X_2, X_3) - SSR(X_1) = 396.98 - 352.27 = 44.71$$

> 🔑 **Why are both formulas the same?** Because $SSTO = SSR + SSE$ always, so a reduction in $SSE$ equals an identical increase in $SSR$.

---

## 2. Definitions and Decompositions {#definitions-and-decompositions}

### Formal Definitions

For a model with variables $X_1$ already in and $X_2$ added:

$$SSR(X_1|X_2) = SSE(X_2) - SSE(X_1, X_2) \tag{7.1a}$$
$$SSR(X_1|X_2) = SSR(X_1, X_2) - SSR(X_2) \tag{7.1b}$$

$$SSR(X_2|X_1) = SSE(X_1) - SSE(X_1, X_2) \tag{7.2a}$$
$$SSR(X_2|X_1) = SSR(X_1, X_2) - SSR(X_1) \tag{7.2b}$$

For three variables:

$$SSR(X_3|X_1, X_2) = SSE(X_1, X_2) - SSE(X_1, X_2, X_3) \tag{7.3a}$$
$$SSR(X_3|X_1, X_2) = SSR(X_1, X_2, X_3) - SSR(X_1, X_2) \tag{7.3b}$$

$$SSR(X_2, X_3|X_1) = SSE(X_1) - SSE(X_1, X_2, X_3) \tag{7.4a}$$
$$SSR(X_2, X_3|X_1) = SSR(X_1, X_2, X_3) - SSR(X_1) \tag{7.4b}$$

> 📌 **Reading the notation:** $SSR(X_2|X_1)$ is read "the extra regression sum of squares due to $X_2$, given $X_1$ is already in the model." The vertical bar $|$ means "given that."

---

### Decomposition of SSR into Extra Sums of Squares

This is the powerful part. For two predictor variables, we can write $SSTO$ as:

Starting from:
$$SSTO = SSR(X_1) + SSE(X_1) \tag{7.5}$$

Substituting $SSE(X_1) = SSR(X_2|X_1) + SSE(X_1, X_2)$:

$$SSTO = SSR(X_1) + SSR(X_2|X_1) + SSE(X_1, X_2) \tag{7.6}$$

But also:
$$SSTO = SSR(X_1, X_2) + SSE(X_1, X_2) \tag{7.7}$$

Therefore:

$$\boxed{SSR(X_1, X_2) = SSR(X_1) + SSR(X_2|X_1)} \tag{7.8}$$

Or equivalently (order swapped):

$$SSR(X_1, X_2) = SSR(X_2) + SSR(X_1|X_2) \tag{7.9}$$

> 💡 **What this means:** The total regression SS when both $X_1$ and $X_2$ are in the model can be decomposed into: (1) what $X_1$ contributes on its own, plus (2) what $X_2$ adds *given $X_1$ is already there*. The order matters — decompositions (7.8) and (7.9) are **both valid** but give different component pieces.

![Schematic of SSR decompositions](placeholder-figure-7-1.png)

*Figure 7.1: The total bar $SSTO = 495.39$. The bar decomposes into $SSR$ and $SSE$ components, and $SSR$ decomposes further into extra sums of squares.*

**For three predictor variables**, we have many possible decompositions. Three important ones:

$$SSR(X_1, X_2, X_3) = SSR(X_1) + SSR(X_2|X_1) + SSR(X_3|X_1, X_2) \tag{7.10a}$$
$$SSR(X_1, X_2, X_3) = SSR(X_2) + SSR(X_3|X_2) + SSR(X_1|X_2, X_3) \tag{7.10b}$$
$$SSR(X_1, X_2, X_3) = SSR(X_1) + SSR(X_2, X_3|X_1) \tag{7.10c}$$

And since:
$$SSR(X_2, X_3|X_1) = SSR(X_2|X_1) + SSR(X_3|X_1, X_2) \tag{7.11}$$

we can see that a multi-df extra SS can always be broken into single-df pieces.

---

### ANOVA Table with SSR Decomposition

Most regression software provides the decomposition into **single-degree-of-freedom** extra sums of squares, in the order variables are entered. Table 7.3 shows the general form, and Table 7.4 shows the body fat example with all three predictors:

**Table 7.4 — Body Fat Example ANOVA with Decomposition:**

| Source of Variation | SS | df | MS |
|---------------------|----|----|-----|
| Regression | 396.98 | 3 | 132.33 |
| $X_1$ | 352.27 | 1 | 352.27 |
| $X_2 \mid X_1$ | 33.17 | 1 | 33.17 |
| $X_3 \mid X_1, X_2$ | 11.54 | 1 | 11.54 |
| Error | 98.41 | 16 | 6.15 |
| Total | 495.39 | 19 | |

> 📌 **Degrees of freedom:** Each single extra SS has **1 df**. An extra SS involving $q$ additional variables has **$q$ df**. The multi-df SS in (7.11) has 2 df because it encompasses two single-df pieces.

**How to get a multi-variable extra SS from software output:**

If software gives you $SSR(X_1)$, $SSR(X_2|X_1)$, $SSR(X_3|X_1,X_2)$ (entering in order $X_1, X_2, X_3$), you can get $SSR(X_2, X_3|X_1)$ by **summing**:

$$SSR(X_2, X_3|X_1) = SSR(X_2|X_1) + SSR(X_3|X_1, X_2) = 33.17 + 11.54 = 44.71$$

**What if you need $SSR(X_1, X_3|X_2)$?** Enter variables in order $X_2, X_1, X_3$ to get:
$$SSR(X_2), \quad SSR(X_1|X_2), \quad SSR(X_3|X_1, X_2)$$
Then sum the last two.

---

## 3. Uses of Extra SS in Tests for Regression Coefficients {#uses-in-tests}

### The General Linear Test Framework

Recall the general linear test approach from Chapter 2: compare a **full model** to a **reduced model** (what you get when $H_0$ holds).

$$F^* = \frac{[SSE(R) - SSE(F)] / (df_R - df_F)}{SSE(F)/df_F}$$

We now show that these differences in $SSE$ are always **extra sums of squares**.

---

### Test whether a Single $\beta_k = 0$

**Question:** Can we drop $X_k$ from the model?

$$H_0: \beta_k = 0, \quad H_a: \beta_k \neq 0$$

**Full model** (three predictors, testing $\beta_3$):

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \beta_3 X_{i3} + \varepsilon_i \quad \text{(Full)} \tag{7.12}$$

**Reduced model** when $H_0$ holds ($\beta_3 = 0$):

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \varepsilon_i \quad \text{(Reduced)} \tag{7.14}$$

The general linear test statistic:

$$F^* = \frac{SSE(X_1, X_2) - SSE(X_1, X_2, X_3)}{(n-3)-(n-4)} \div \frac{SSE(X_1,X_2,X_3)}{n-4}$$

The numerator difference is exactly $SSR(X_3|X_1, X_2)$! So:

$$\boxed{F^* = \frac{SSR(X_3|X_1, X_2)}{1} \div \frac{SSE(X_1, X_2, X_3)}{n-4} = \frac{MSR(X_3|X_1, X_2)}{MSE(X_1, X_2, X_3)}} \tag{7.15}$$

Under $H_0$: $F^* \sim F(1, n-p)$. Large values lead to $H_a$.

**Equivalent test using $t$:** The $t^*$ statistic from Chapter 6:

$$t^* = \frac{b_k}{s\{b_k\}} \tag{7.25}$$

satisfies $(t^*)^2 = F^*$. Both are valid for testing a single $\beta_k = 0$.

---

### 📊 Example: Testing Whether $X_3$ Can Be Dropped (Body Fat)

Full model: $Y$ on $X_1, X_2, X_3$. Testing $H_0: \beta_3 = 0$ at $\alpha = .01$.

From Table 7.4: $SSR(X_3|X_1, X_2) = 11.54$, $SSE(X_1, X_2, X_3) = 98.41$, $n-4 = 16$.

$$F^* = \frac{11.54/1}{98.41/16} = \frac{11.54}{6.15} = 1.88$$

Critical value: $F(.99; 1, 16) = 8.53$.

Since $F^* = 1.88 \leq 8.53$, **conclude $H_0$**: $X_3$ (midarm circumference) can be dropped from the model containing $X_1$ and $X_2$.

**Verify with $t^*$:** From Table 7.2d, $t^* = b_3/s\{b_3\} = -2.186/1.596 = -1.37$.

$(t^*)^2 = (-1.37)^2 = 1.88 = F^*$ ✓

> 💡 **Why this is a *partial* F test:** We're testing $\beta_3 = 0$ *given* $X_1$ and $X_2$ are already in the model. The "partial" means conditional on the other predictors. This is fundamentally different from the "overall" F test which tests all $\beta_k$ simultaneously.

---

### Test whether Several $\beta_k = 0$

**Question:** Can we drop multiple variables at once?

$$H_0: \beta_q = \beta_{q+1} = \cdots = \beta_{p-1} = 0$$
$$H_a: \text{not all of the } \beta_k \text{ in } H_0 \text{ equal zero} \tag{7.26}$$

The reduced model drops the last $p - q$ predictors, leaving $X_1, \ldots, X_{q-1}$.

**Test statistic (partial F test with multiple df):**

$$F^* = \frac{SSR(X_q, \ldots, X_{p-1}|X_1, \ldots, X_{q-1})}{p-q} \div \frac{SSE(X_1, \ldots, X_{p-1})}{n-p}$$

$$= \frac{MSR(X_q, \ldots, X_{p-1}|X_1, \ldots, X_{q-1})}{MSE} \tag{7.27}$$

Under $H_0$: $F^* \sim F(p-q, n-p)$.

> 📌 **Key:** The numerator has $p - q$ degrees of freedom (the number of parameters being tested), and the denominator has $n - p$ df (from the full model).

The numerator extra SS can be built from single-df pieces (7.28):

$$SSR(X_q, \ldots, X_{p-1}|X_1, \ldots, X_{q-1}) = SSR(X_q|X_1, \ldots, X_{q-1}) + \cdots + SSR(X_{p-1}|X_1, \ldots, X_{p-2})$$

**In terms of $R^2$** (when the model contains intercept $\beta_0$):

$$F^* = \frac{R_F^2 - R_R^2}{p-q} \div \frac{1-R_F^2}{n-p} \tag{7.19}$$

where $R_F^2$ and $R_R^2$ are the $R^2$ values for the full and reduced models.

---

### 📊 Example: Testing Whether $X_2$ and $X_3$ Can Both Be Dropped (Body Fat)

Full model: $Y$ on $X_1, X_2, X_3$. Testing $H_0: \beta_2 = \beta_3 = 0$ at $\alpha = .05$.

From (7.11): $SSR(X_2, X_3|X_1) = 33.17 + 11.54 = 44.71$ (2 df)

$$F^* = \frac{44.71/2}{6.15} = \frac{22.355}{6.15} = 3.63$$

Critical value: $F(.95; 2, 16) = 3.63$.

Since $F^* = 3.63$ is exactly at the boundary, $p$-value $= .05$. This is borderline — we'd want more analysis before deciding to drop both $X_2$ and $X_3$.

**Same result via $R^2$** formula (7.20):

$$F^* = \frac{R_{Y|123}^2 - R_{Y|1}^2}{(n-2)-(n-4)} \div \frac{1-R_{Y|123}^2}{n-4}$$

$$= \frac{.80135 - .71110}{2} \div \frac{1-.80135}{16} = \frac{.04513}{.01241} = 3.63 \checkmark$$

---

### Other Tests (Non-Zero Hypotheses)

When the hypothesis doesn't involve testing $\beta_k = 0$, extra sums of squares cannot be used. The general linear test approach requires **separate fittings** of the full and reduced models.

**📊 Example — Test $H_0: \beta_1 = \beta_2$** (equal coefficients):

Full model (7.30): $Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \beta_3 X_{i3} + \varepsilon_i$

Under $H_0$ ($\beta_1 = \beta_2 = \beta_c$):

Reduced model (7.32): $Y_i = \beta_0 + \beta_c(X_{i1} + X_{i2}) + \beta_3 X_{i3} + \varepsilon_i$

Create a new variable $W_i = X_{i1} + X_{i2}$ and regress $Y$ on $W$ and $X_3$. Then use the general F test with 1 and $n-4$ df.

**📊 Example — Test $H_0: \beta_1 = 3, \beta_3 = 5$** (specific values):

Reduced model (7.34): $Y_i - 3X_{i1} - 5X_{i3} = \beta_0 + \beta_2 X_{i2} + \varepsilon_i$

The new response variable is $Y_i^* = Y_i - 3X_{i1} - 5X_{i3}$. Regress $Y^*$ on $X_2$ and use the general F test with 2 and $n-4$ df.

---

## 4. Summary of All Tests Concerning Regression Coefficients {#summary-of-tests}

Here is the complete taxonomy of hypothesis tests, organized from most general to most specific:

### Test 1: Whether ALL $\beta_k = 0$ (Overall F Test)

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_{p-1} = 0 \tag{7.21}$$

$$F^* = \frac{SSR(X_1, \ldots, X_{p-1})}{p-1} \div \frac{SSE(X_1, \ldots, X_{p-1})}{n-p} = \frac{MSR}{MSE} \tag{7.22}$$

Under $H_0$: $F^* \sim F(p-1, n-p)$.

---

### Test 2: Whether a SINGLE $\beta_k = 0$ (Partial F Test, 1 df)

$$H_0: \beta_k = 0, \quad H_a: \beta_k \neq 0 \tag{7.23}$$

$$F^* = \frac{SSR(X_k|X_1,\ldots,X_{k-1},X_{k+1},\ldots,X_{p-1})}{1} \div \frac{SSE}{n-p} = \frac{MSR(X_k|\text{all others})}{MSE} \tag{7.24}$$

Under $H_0$: $F^* \sim F(1, n-p)$. Equivalent $t$ test: $t^* = b_k/s\{b_k\} \sim t(n-p)$.

---

### Test 3: Whether SOME $\beta_k = 0$ (Partial F Test, Multiple df)

$$H_0: \beta_q = \beta_{q+1} = \cdots = \beta_{p-1} = 0 \tag{7.26}$$

$$F^* = \frac{MSR(X_q, \ldots, X_{p-1}|X_1, \ldots, X_{q-1})}{MSE} \tag{7.27}$$

Under $H_0$: $F^* \sim F(p-q, n-p)$.

> ⚠️ **Critical Warning: Don't test several $\beta_k = 0$ simultaneously using separate $t$ tests!**
>
> This is one of the most common and damaging mistakes in multiple regression. In the body fat example with $p-1 = 2$ predictors and $\alpha = .05$, both $t^*$ statistics (for $\beta_1$ and $\beta_2$) are below the Bonferroni-adjusted critical value of $t(.9875; 17) = 2.46$, suggesting both can be dropped. Yet the proper F test for $H_0: \beta_1 = \beta_2 = 0$ gives $F^* = 192.72/6.47 = 29.8 \gg F(.95; 2, 17) = 3.59$, strongly rejecting $H_0$!
>
> The reason: each $t$ test is marginal (testing the contribution *given the other predictor is in*). Together they don't test if both can be dropped simultaneously. **Always use the partial F test for joint hypotheses.**

---

### Test 4: Other Hypotheses (General Linear Test)

For tests not of the form $\beta_k = 0$ (e.g., $\beta_1 = \beta_2$, or $\beta_1 = 3$), use the general approach:
1. Fit full model → get $SSE(F)$
2. Fit reduced model under $H_0$ → get $SSE(R)$
3. Compute $F^* = [SSE(R) - SSE(F)]/(df_R - df_F) \div SSE(F)/df_F$

---

## 5. Coefficients of Partial Determination {#partial-determination}

### Motivation

The coefficient of multiple determination $R^2$ measures the total proportionate reduction in variation explained by the *entire set* of $X$ variables. But we often want to measure the **marginal contribution of a single $X$ variable** when the others are already in the model. That's what the **coefficient of partial determination** does.

---

### Definition for Two Predictor Variables

For a model with $X_1$ and $X_2$:

$$R_{Y1|2}^2 = \frac{SSE(X_2) - SSE(X_1, X_2)}{SSE(X_2)} = \frac{SSR(X_1|X_2)}{SSE(X_2)} \tag{7.35}$$

**Interpretation:** The proportionate reduction in the variation of $Y$ remaining after $X_2$ is in the model, that is gained by also including $X_1$.

Similarly:

$$R_{Y2|1}^2 = \frac{SSR(X_2|X_1)}{SSE(X_1)} \tag{7.36}$$

**General pattern for three or more variables:**

$$R_{Y1|23}^2 = \frac{SSR(X_1|X_2, X_3)}{SSE(X_2, X_3)} \tag{7.37}$$

$$R_{Y2|13}^2 = \frac{SSR(X_2|X_1, X_3)}{SSE(X_1, X_3)} \tag{7.38}$$

$$R_{Y3|12}^2 = \frac{SSR(X_3|X_1, X_2)}{SSE(X_1, X_2)} \tag{7.39}$$

$$R_{Y4|123}^2 = \frac{SSR(X_4|X_1, X_2, X_3)}{SSE(X_1, X_2, X_3)} \tag{7.40}$$

> 📌 **Reading the subscripts:** In $R_{Y1|23}^2$, the variable to the left of $|$ is being *added* (here: $X_1$), and the variables to the right are *already in the model* (here: $X_2$ and $X_3$). The $Y$ prefix reminds us $Y$ is the response.

**Range:** $0 \leq R_{Yk|\ldots}^2 \leq 1$, just like a regular $R^2$.

---

### 📊 Example: Body Fat Data

From Tables 7.2 and 7.4:

$$R_{Y2|1}^2 = \frac{SSR(X_2|X_1)}{SSE(X_1)} = \frac{33.17}{143.12} = .232$$

$$R_{Y3|12}^2 = \frac{SSR(X_3|X_1, X_2)}{SSE(X_1, X_2)} = \frac{11.54}{109.95} = .105$$

$$R_{Y1|2}^2 = \frac{SSR(X_1|X_2)}{SSE(X_2)} = \frac{3.47}{113.42} = .031$$

**What these tell us:**
- When $X_2$ (thigh) is added to a model already containing $X_1$ (triceps), the error variation is reduced by **23.2%** more.
- When $X_3$ (midarm) is added to the model with $X_1$ and $X_2$, the remaining error variation is reduced by another **10.5%**.
- If we already have $X_2$, adding $X_1$ only reduces remaining variation by **3.1%** — $X_1$ provides little unique information once $X_2$ is there (they're highly correlated).

---

### Deep Insight: Partial Determination as Simple Determination of Residuals

> 💡 **Beautiful geometric interpretation:** $R_{Y1|2}^2$ equals the *ordinary* $R^2$ from a simple regression of $e_i(Y|X_2)$ on $e_i(X_1|X_2)$, where:
> - $e_i(Y|X_2) = Y_i - \hat{Y}_i(X_2)$ are the residuals from regressing $Y$ on $X_2$
> - $e_i(X_1|X_2) = X_{i1} - \hat{X}_{i1}(X_2)$ are the residuals from regressing $X_1$ on $X_2$
>
> Both sets of residuals have been "purged" of the effect of $X_2$. So $R_{Y1|2}^2$ measures the relationship between $Y$ and $X_1$ **after both have been adjusted for $X_2$**. This is exactly the "added variable plot" or "partial regression plot" described in Chapter 10.

---

### Coefficients of Partial Correlation

The **coefficient of partial correlation** is the square root of the coefficient of partial determination, with the sign matching the corresponding regression coefficient:

$$r_{Y1|2} = \sqrt{R_{Y1|2}^2} \cdot \text{sign}(b_1)$$

For the body fat example:

$$r_{Y2|1} = \sqrt{.232} = .482 \quad \text{(positive, since } b_2 = .6594 > 0\text{)}$$
$$r_{Y3|12} = -\sqrt{.105} = -.324 \quad \text{(negative, since } b_3 = -2.186 < 0\text{)}$$
$$r_{Y1|2} = \sqrt{.031} = .176 \quad \text{(positive, since } b_1 = .2224 > 0\text{)}$$

**Algebraic formula** relating partial correlation to simple correlations:

$$R_{Y2|1}^2 = [r_{Y2|1}]^2 = \frac{(r_{Y2} - r_{12}r_{Y1})^2}{(1-r_{12}^2)(1-r_{Y1}^2)} \tag{7.41}$$

Partial correlation coefficients are used in **variable selection** (Chapter 9) to identify the best next predictor to add to the model.

> ⚠️ **Caution:** Coefficients of partial determination are more interpretable than partial correlations; the latter are frequently used but don't have as clear a meaning. Use $R^2_{Yk|\ldots}$ when the goal is interpretation, and $r_{Yk|\ldots}$ when using forward selection algorithms.

---

## 6. Standardized Multiple Regression Model {#standardized-model}

### Two Problems with Ordinary Regression Coefficients

**Problem 1: Roundoff errors in normal equation calculations.** When predictors are on very different scales, the $\mathbf{X'X}$ matrix can have elements spanning many orders of magnitude (e.g., 15 to 49,000,000), making $(\mathbf{X'X})^{-1}$ hard to compute accurately. The problem is especially severe when predictors are highly correlated.

**Problem 2: Non-comparability of coefficients.** In $\hat{Y} = 200 + 20{,}000X_1 + 0.2X_2$, it's tempting to say $X_1$ matters much more than $X_2$. But if $Y$ is in dollars, $X_1$ in thousand dollars, and $X_2$ in cents, then a $\$1,000$ increase in $X_1$ (1 unit) gives $+\$20,000$, and a $\$1,000$ increase in $X_2$ (100,000 units) *also* gives $+\$20,000$ in sales. The coefficients can't be compared without knowing the units.

---

### The Correlation Transformation

To fix both problems, we standardize all variables using the **correlation transformation**:

$$Y_i^* = \frac{1}{\sqrt{n-1}} \left(\frac{Y_i - \bar{Y}}{s_Y}\right) \tag{7.44a}$$

$$X_{ik}^* = \frac{1}{\sqrt{n-1}} \left(\frac{X_{ik} - \bar{X}_k}{s_k}\right) \quad (k = 1, \ldots, p-1) \tag{7.44b}$$

where:

$$s_Y = \sqrt{\frac{\sum(Y_i - \bar{Y})^2}{n-1}}, \qquad s_k = \sqrt{\frac{\sum(X_{ik} - \bar{X}_k)^2}{n-1}} \tag{7.43c,d}$$

The $\frac{1}{\sqrt{n-1}}$ factor (vs. the usual standardization) is what makes this the *correlation* transformation specifically — it ensures $\mathbf{X'X} = \mathbf{r}_{XX}$ (correlation matrix).

---

### The Standardized Regression Model

$$Y_i^* = \beta_1^* X_{i1}^* + \cdots + \beta_{p-1}^* X_{i,p-1}^* + \varepsilon_i^* \tag{7.45}$$

> 📌 **No intercept term!** When all variables are mean-centered, the intercept is always zero. This is why the standardized model has no $\beta_0^*$.

### X'X Matrix for Transformed Variables

$$\underset{(p-1)\times(p-1)}{\mathbf{X'X}} = \underset{(p-1)\times(p-1)}{\mathbf{r}_{XX}} \tag{7.50}$$

The **correlation matrix of X variables** is:

$$\mathbf{r}_{XX} = \begin{bmatrix} 1 & r_{12} & \cdots & r_{1,p-1} \\ r_{21} & 1 & \cdots & r_{2,p-1} \\ \vdots & \vdots & & \vdots \\ r_{p-1,1} & r_{p-1,2} & \cdots & 1 \end{bmatrix} \tag{7.47}$$

All entries are between $-1$ and $1$, so the matrix is well-conditioned and easy to invert.

The **X'Y vector** for transformed variables:

$$\underset{(p-1)\times 1}{\mathbf{X'Y}} = \mathbf{r}_{YX} = \begin{bmatrix} r_{Y1} \\ r_{Y2} \\ \vdots \\ r_{Y,p-1} \end{bmatrix} \tag{7.48, 7.51}$$

### Standardized Normal Equations and Estimators

$$\mathbf{r}_{XX} \mathbf{b} = \mathbf{r}_{YX} \tag{7.52a}$$

$$\mathbf{b} = \mathbf{r}_{XX}^{-1} \mathbf{r}_{YX} \tag{7.52b}$$

This is elegant: the standardized regression coefficients $b_1^*, \ldots, b_{p-1}^*$ are obtained by inverting the correlation matrix of the $X$'s and multiplying by the correlations of $Y$ with each $X$.

---

### Algebraic Form for Two Predictors

When $p - 1 = 2$:

$$\mathbf{r}_{XX}^{-1} = \frac{1}{1-r_{12}^2} \begin{bmatrix} 1 & -r_{12} \\ -r_{12} & 1 \end{bmatrix} \tag{7.54c}$$

$$b_1^* = \frac{r_{Y1} - r_{12} r_{Y2}}{1 - r_{12}^2} \tag{7.55a}$$

$$b_2^* = \frac{r_{Y2} - r_{12} r_{Y1}}{1 - r_{12}^2} \tag{7.55b}$$

> 💡 **Intuition:** $b_1^*$ is the correlation of $Y$ with $X_1$ ($r_{Y1}$), adjusted for the overlap between $X_1$ and $X_2$ (captured by $r_{12}$). When $r_{12} = 0$ (uncorrelated predictors), $b_1^* = r_{Y1}$ — the standardized coefficient equals the simple correlation!

---

### Converting Back to Original Coefficients

$$b_k = \left(\frac{s_Y}{s_k}\right) b_k^* \quad (k = 1, \ldots, p-1) \tag{7.53a / 7.46a}$$

$$b_0 = \bar{Y} - b_1 \bar{X}_1 - b_2 \bar{X}_2 - \cdots - b_{p-1} \bar{X}_{p-1} \tag{7.53b / 7.46b}$$

---

### 📊 Example: Dwaine Studios (Standardized)

Original data: $\bar{Y} = 181.90$, $s_Y = 36.191$; $\bar{X}_1 = 62.019$, $s_1 = 18.620$; $\bar{X}_2 = 17.143$, $s_2 = .97035$.

**Transformed values (first case):**

$$Y_1^* = \frac{1}{\sqrt{20}}\left(\frac{174.4 - 181.90}{36.191}\right) = -.04634$$

$$X_{11}^* = \frac{1}{\sqrt{20}}\left(\frac{68.5 - 62.019}{18.620}\right) = .07783$$

$$X_{12}^* = \frac{1}{\sqrt{20}}\left(\frac{16.7 - 17.143}{.97035}\right) = -.10208$$

**Fitted standardized model (Table 7.5c):**

$$\hat{Y}^* = .7484 X_1^* + .2511 X_2^*$$

**Interpretation:** A one-standard-deviation increase in target population ($X_1$), holding income constant, increases expected sales by **.7484 standard deviations** of sales. For income ($X_2$), the increase is only .2511 SDs. Target population has a **much larger standardized effect** than income.

> ⚠️ **Caution!** Even standardized regression coefficients can be misleading when predictors are correlated (as here, $r_{12} = .781$). The magnitude of $b_k^*$ depends on both the partial effect of $X_k$ AND the correlations with other predictors. See Section 7.6.

**Converting back to original scale:**

$$b_1 = \frac{36.191}{18.620}(.7484) = 1.4546$$

$$b_2 = \frac{36.191}{.97035}(.2511) = 9.3652$$

$$b_0 = 181.90 - 1.4546(62.019) - 9.3652(17.143) = -68.860$$

Same as Chapter 6 (minor rounding): $\hat{Y} = -68.860 + 1.455X_1 + 9.365X_2$ ✓

---

## 7. Multicollinearity and Its Effects {#multicollinearity}

### What is Multicollinearity?

When the predictor variables included in a regression model are **correlated with each other**, *intercorrelation* or **multicollinearity** is said to exist. The term "multicollinearity" is often reserved for *severe* cases.

Questions we'd love to answer in multiple regression:
1. What is the relative importance of each predictor?
2. What is the magnitude of each predictor's effect?
3. Can any predictors be dropped?
4. Should any currently-excluded predictors be added?

These questions have **simple, unambiguous answers only when predictors are uncorrelated**. Otherwise, multicollinearity complicates everything.

---

### The Clean Case: Uncorrelated Predictor Variables

**📊 Example: Work Crew Productivity** (Table 7.6)

Experiment: control crew size ($X_1$) and bonus pay ($X_2$). By design, $r_{12}^2 = 0$ (perfectly balanced).

| Variables in Model | $b_1$ | $b_2$ |
|-------------------|-------|-------|
| $X_1$ alone | 5.375 | — |
| $X_2$ alone | — | 9.250 |
| $X_1$ and $X_2$ | 5.375 | 9.250 |

**The regression coefficients don't change when the other predictor is added or removed!** When predictors are uncorrelated:
- $SSR(X_1|X_2) = SSR(X_1)$ — $X_1$ contributes the same extra SS regardless of whether $X_2$ is in the model
- $SSR(X_2|X_1) = SSR(X_2)$ — same for $X_2$

> 🔑 **This is the ideal case.** Controlled experiments are valuable precisely because they allow the experimenter to make predictor variables uncorrelated by design.

---

### The Crisis: Perfectly Correlated Predictor Variables

**📊 Example: Perfect Multicollinearity** (Table 7.8)

Four observations, two predictor variables related by: $X_2 = 5 + 0.5 X_1$ (exactly).

Mr. A fits the data and proudly gets: $\hat{Y} = -87 + X_1 + 18X_2$ — perfect fit!

Ms. B fits the same data and gets: $\hat{Y} = -7 + 9X_1 + 2X_2$ — also perfect fit!

**Both are correct.** In fact, infinitely many response functions fit the data perfectly when $X_1$ and $X_2$ are perfectly correlated. This happens because $(\mathbf{X'X})$ is **singular** (determinant = 0) and cannot be inverted — the normal equations have infinitely many solutions.

![Two response planes intersecting where X2 = 5 + 0.5X1](placeholder-figure-7-2.png)

*Figure 7.2: The two response planes (7.58) and (7.59) intersect only along the line $X_2 = 5 + .5X_1$ (where the observations lie). They give the same fitted values for the observations but very different predictions elsewhere.*

**Two key lessons from this example:**
1. Perfect multicollinearity did NOT inhibit getting a good fit — $R^2 = 1.0$ for both!
2. But we CANNOT interpret the individual coefficients meaningfully — infinitely many sets give the same fit

---

### Real-World Multicollinearity: Effects on Regression Coefficients

In practice, perfect multicollinearity is rare, but *near* perfect multicollinearity is common (especially in economics, social sciences, biology). The effects are a continuous function of how strong the correlation is.

**📊 Body Fat Example — Coefficients Change Dramatically with Model:**

| Variables in Model | $b_1$ | $b_2$ |
|-------------------|-------|-------|
| $X_1$ only | .8572 | — |
| $X_2$ only | — | .8565 |
| $X_1, X_2$ | .2224 | .6594 |
| $X_1, X_2, X_3$ | 4.334 | −2.857 |

Note how $b_2$ **changes sign** (from $+.6594$ to $-2.857$) when $X_3$ is added! This is a hallmark of severe multicollinearity. The reason: $X_1$ and $X_2$ are highly correlated ($r_{12} = .924$), so the regression coefficient of any one variable depends heavily on which other correlated variables are included.

> 🔑 **The Fundamental Lesson:** When predictor variables are correlated, a regression coefficient does NOT reflect any inherent effect of that predictor on the response — it reflects only the *partial* effect given the particular set of other predictors in the model. A coefficient can even change sign as predictors are added or removed.

---

### Four Effects of Multicollinearity

#### Effect 1: On Regression Coefficients
As shown above, coefficient estimates become erratic. The "same" physical relationship leads to wildly different coefficient values depending on which correlated variables are in the model.

**Why?** Algebraically: (7.56) shows $b_1$ in a two-predictor model:

$$b_1 = \frac{\sum(X_{i1}-\bar{X}_1)(Y_i-\bar{Y}) - \left[\frac{\sum(Y_i-\bar{Y})^2}{\sum(X_{i1}-\bar{X}_1)^2}\right]^{1/2} r_{Y2} \cdot r_{12}}{1 - r_{12}^2}$$

When $r_{12} = 0$: $b_1$ reduces to the simple regression slope. When $r_{12} \to 1$: the denominator $1 - r_{12}^2 \to 0$, making $b_1$ explode.

#### Effect 2: On Standard Errors of Regression Coefficients

As more correlated predictors are added, the standard errors of the $b_k$ **blow up**:

| Variables in Model | $s\{b_1\}$ | $s\{b_2\}$ |
|-------------------|-----------|-----------|
| $X_1$ only | .1288 | — |
| $X_2$ only | — | .1100 |
| $X_1, X_2$ | .3034 | .2912 |
| $X_1, X_2, X_3$ | 3.016 | 2.582 |

With all three predictors, $s\{b_1\}$ is nearly **24 times larger** than with $X_1$ alone! This massive inflation means the individual coefficients are very imprecisely estimated — confidence intervals are wide and t-tests are weak.

**Formula confirming this:** For the standardized two-predictor model (7.65):

$$\sigma^2\{b_1^*\} = \sigma^2\{b_2^*\} = \frac{(\sigma^*)^2}{1 - r_{12}^2}$$

As $r_{12} \to 1$: variance $\to \infty$. The higher the correlation, the worse the precision of individual coefficient estimates.

#### Effect 3: On Extra Sums of Squares

When predictors are correlated, extra sums of squares are NOT unique:

$$SSR(X_1) = 352.27 \qquad \text{but} \qquad SSR(X_1|X_2) = 3.47$$

$X_1$ on its own explains 352.27 units. But once $X_2$ is in the model, $X_1$ adds only 3.47 more units — because highly correlated $X_2$ has already captured most of $X_1$'s information.

> **There is no unique "amount of variation explained" attributable to any one correlated predictor variable.** The attribution depends on which other correlated variables are in the model.

#### Effect 4: On Fitted Values and Predictions (GOOD NEWS!)

Despite all the chaos above, multicollinearity does **NOT** harm:
- The fitted values $\hat{Y}_i$ within the observation region
- The $MSE$ (which decreases steadily as more variables are added)
- Prediction precision within the region of observations

**📊 Body Fat — Estimated mean body fat at $X_{h1} = 25, X_{h2} = 50$:**

| Variables in Model | $\hat{Y}_h$ | $s\{\hat{Y}_h\}$ |
|-------------------|------------|-----------------|
| $X_1$ only | 19.93 | .632 |
| $X_1, X_2$ | 19.36 | .624 |
| $X_1, X_2, X_3$ | 19.19 | .621 |

The estimated mean body fat is nearly identical across all three models, and the precision is essentially unchanged. This stability occurs even though adding $X_2$ doubled $s\{b_1\}$ — the covariance between $b_1$ and $b_2$ is negative and counteracts the increased variance.

> 🔑 **Bottom Line on Multicollinearity:**
> - **Individual coefficients:** unreliable, erratic, hard to interpret ❌
> - **Fitted values within the data region:** reliable and stable ✅
> - **Prediction within the data region:** reliable ✅
> - **Extrapolation outside the data region:** unreliable ❌

---

### The Variance Inflation Factor (VIF) — Preview

The variance formula (7.65) $\sigma^2\{b_k^*\} = (\sigma^*)^2 / (1-r_{12}^2)$ shows that $1/(1-r_{12}^2)$ inflates the variance due to multicollinearity. This is the **Variance Inflation Factor (VIF)**. For a two-predictor model, $VIF = 1/(1-r_{12}^2)$. When $r_{12} = .9$, $VIF = 1/(1-.81) = 5.26$ — the variance is over 5 times what it would be with uncorrelated predictors.

> 📌 Chapter 10 discusses VIF and other formal diagnostics for multicollinearity in depth.

---

### The Simultaneous Test Paradox

> ⚠️ **A paradox that trips up many analysts:**
>
> It is possible that:
> - The overall F test rejects $H_0$: all $\beta_k = 0$ → the model has predictive power
> - Yet EVERY individual $t$ test fails to reject $H_0$: $\beta_k = 0$ → each individual predictor seems dispensable
>
> This apparent contradiction arises because each $t$ test is testing the marginal contribution *given all others are present*. With highly correlated predictors, no single one provides much extra beyond the others, yet together they clearly predict $Y$.
>
> **Lesson:** Never judge the importance of a predictor solely from its individual $t$ statistic when multicollinearity is present.

---

## 8. Key Formulas Reference Sheet {#formulas-reference}

### Extra Sums of Squares

| Expression | Formula |
|------------|---------|
| $SSR(X_j\|X_k)$ | $SSE(X_k) - SSE(X_j, X_k)$ |
| $SSR(X_j\|X_k)$ | $SSR(X_j, X_k) - SSR(X_k)$ |
| $SSR(X_1, X_2)$ | $SSR(X_1) + SSR(X_2\|X_1)$ |
| $SSR(X_1, X_2)$ | $SSR(X_2) + SSR(X_1\|X_2)$ |
| $SSR(X_2, X_3\|X_1)$ | $SSR(X_2\|X_1) + SSR(X_3\|X_1, X_2)$ |

### Hypothesis Tests

| Test | Statistic | Distribution under $H_0$ |
|------|-----------|--------------------------|
| All $\beta_k = 0$ | $F^* = MSR/MSE$ | $F(p-1, n-p)$ |
| Single $\beta_k = 0$ (F) | $F^* = MSR(X_k\|\text{rest})/MSE$ | $F(1, n-p)$ |
| Single $\beta_k = 0$ (t) | $t^* = b_k/s\{b_k\}$ | $t(n-p)$ |
| Several $\beta_k = 0$ | $F^* = MSR(X_q,\ldots\|X_1,\ldots)/MSE$ | $F(p-q, n-p)$ |

### Partial Determination

$$R_{Yk|\text{rest}}^2 = \frac{SSR(X_k|\text{rest})}{SSE(\text{rest})}$$

### Standardized Coefficients

| Formula | Use |
|---------|-----|
| $X_{ik}^* = \frac{1}{\sqrt{n-1}} \cdot \frac{X_{ik}-\bar{X}_k}{s_k}$ | Transform predictors |
| $Y_i^* = \frac{1}{\sqrt{n-1}} \cdot \frac{Y_i - \bar{Y}}{s_Y}$ | Transform response |
| $\mathbf{b} = \mathbf{r}_{XX}^{-1}\mathbf{r}_{YX}$ | Standardized normal equations |
| $b_k = (s_Y/s_k) b_k^*$ | Convert back to original scale |
| $b_0 = \bar{Y} - b_1\bar{X}_1 - \cdots - b_{p-1}\bar{X}_{p-1}$ | Recover intercept |

### Multicollinearity (Two Predictors)

$$\sigma^2\{b_1^*\} = \sigma^2\{b_2^*\} = \frac{(\sigma^*)^2}{1-r_{12}^2}$$

$$VIF = \frac{1}{1-r_{12}^2}$$

---

## 9. Common Mistakes & Misconceptions {#common-mistakes}

1. **"Dropping a non-significant variable simplifies the model and costs nothing."**
   → Wrong! Dropping a predictor that has even weak marginal contributions may bias the remaining coefficients if the dropped variable is correlated with predictors that remain.

2. **"Running separate t-tests for each $\beta_k$ is equivalent to testing them jointly."**
   → **This is the most dangerous mistake in multiple regression.** Separate t-tests at level $\alpha$ for $g$ parameters do NOT test the joint hypothesis at level $\alpha$. Use the partial F test for joint hypotheses.

3. **"Large standardized coefficient = important predictor."**
   → Only valid when predictors are uncorrelated. With correlated predictors, standardized coefficients are affected by the intercorrelations and do not cleanly reflect "importance."

4. **"Multicollinearity ruins the model and all predictions are unreliable."**
   → Wrong! Multicollinearity only ruins individual coefficient interpretation and extrapolation. Fitted values within the data region remain reliable.

5. **"$SSR(X_1|X_2) = SSR(X_1)$ always."**
   → Only when $X_1$ and $X_2$ are uncorrelated. In general, $SSR(X_1|X_2) \neq SSR(X_1)$.

6. **"The partial F test and overall F test test the same thing."**
   → No! The overall F tests all $\beta_k = 0$ simultaneously. The partial F tests a *subset* of them, conditional on others.

7. **"If $X_2$ was correlated with $Y$ in a simple regression, it must be useful when added to a larger model."**
   → Not true. A variable highly correlated with already-included predictors may add negligible extra information ($SSR(X_2|X_1)$ could be tiny even if $SSR(X_2)$ is large).

8. **"A negative regression coefficient proves a predictor inhibits the response."**
   → When multicollinearity is present, a coefficient can even change sign depending on which other predictors are in the model (as seen with $b_2$ in the body fat example). Always consider the full model context.

---
## 7.6 Multicollinearity and Its Effects (Page 278)

**Multicollinearity** refers to the situation in multiple regression where two or more predictor variables are highly correlated with each other. It poses a significant challenge to the interpretation and stability of the regression model.

### Uncorrelated Predictor Variables (Page 279)
This is an ideal (but rare in observational studies) scenario. If predictor variables are perfectly uncorrelated (orthogonal), then:
* The regression coefficients $b_k$ are independent of which other predictors are in the model.
* The extra sum of squares $SSR(X_k | \text{other } X\text{s})$ would simply be $SSR(X_k)$.
* Coefficient estimates are highly stable and precise.
* Interpretation of individual coefficients is straightforward.
* This is often achieved in designed experiments (e.g., using orthogonal designs).

### Nature of Problem when Predictor Variables Are Perfectly Correlated (Page 281)
**Perfect Multicollinearity** occurs when one predictor variable is an exact linear combination of one or more other predictor variables.
* **Cause:** This happens if, for example, you include dummy variables for *all* categories of a categorical variable *and* an intercept in the model (the sum of dummies equals the intercept's column of ones). Or, if you include a variable and its exact duplicate.
* **Consequence:** The $\mathbf{X}^T \mathbf{X}$ matrix becomes **singular** (its determinant is zero), meaning its inverse $(\mathbf{X}^T \mathbf{X})^{-1}$ does **not exist**. Consequently, the least squares estimates $\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$ cannot be uniquely determined. Statistical software will usually issue an error and drop one of the perfectly correlated variables.

---

## Chapter Summary

Chapter 7 introduced three powerful tools for deeper multiple regression analysis:

**Extra Sums of Squares** let you measure the marginal contribution of any variable or set of variables, conditional on what's already in the model. They are the building blocks of all partial F tests, which are the correct tool for testing whether subsets of regression coefficients equal zero.

**Partial Determination Coefficients** ($R^2_{Yk|\text{rest}}$) express those marginal contributions as proportions — how much of the *remaining* unexplained variation does adding $X_k$ explain? They have an elegant interpretation as the $R^2$ from regressing residuals on residuals.

**Standardized Regression** solves two practical problems: numerical stability when predictors are on very different scales, and the desire to compare coefficient magnitudes across predictors with different units.

**Multicollinearity** is the central challenge of multiple regression in observational data. Its effects on individual coefficients and their standard errors can be severe, but — crucially — fitted values and predictions within the data region remain reliable. Understanding this distinction is essential for appropriate use and interpretation of regression models.
