# Chapter 4: Simultaneous Inferences and Other Topics in Regression Analysis

## Overview

This chapter explores advanced topics in simple linear regression, focusing on how to make **multiple statistical inferences simultaneously** from the same dataset. When we need to draw several conclusions at once (e.g., confidence intervals for both β₀ and β₁), special techniques are required to maintain our overall confidence level.

**Key Question:** If we construct two 95% confidence intervals separately, is there a 95% chance that both are correct simultaneously? **No!** The probability drops to (.95)² = .9025 or 90.25%.

---

## 4.1 Joint Estimation of β₀ and β₁

### The Problem with Multiple Inferences

When making multiple inferences from the same dataset, we need to distinguish between:

1. **Statement Confidence Coefficient**: The familiar confidence coefficient (e.g., 95%) that applies to individual estimates
2. **Family Confidence Coefficient**: The probability that an *entire set* of estimates (a "family") are all correct simultaneously

### Key Definitions

> **Family of Estimates**: A set of estimates or tests of interest. In our context, this typically means estimates for both β₀ and β₁.

> **Statement Confidence Coefficient**: The proportion of correct individual estimates when repeated samples are selected and confidence intervals are calculated for each sample.

> **Family Confidence Coefficient**: The proportion of families of estimates that are entirely correct when repeated samples are selected and confidence intervals are calculated for the entire family in each sample.

**Intuitive Example**: If we want 95% confidence intervals for both β₀ and β₁:
- Using separate 95% intervals means each has a 95% chance of being correct
- But the probability that **both** are correct is less than 95%
- For 5% of samples, either one or both intervals would be incorrect

---

### Bonferroni Joint Confidence Intervals

The **Bonferroni procedure** provides a simple, general method for constructing simultaneous confidence intervals with a specified family confidence coefficient.

#### The Bonferroni Principle

To achieve a family confidence coefficient of at least 1 - α:
- Construct each individual confidence interval with statement confidence coefficient **1 - α/g**
- Where g = number of estimates in the family

This ensures the family confidence coefficient is **at least** 1 - α.

#### Derivation of Bonferroni Inequality

Starting with ordinary confidence limits for β₀ and β₁ with statement confidence coefficients 1 - α each:

$$b_0 \pm t(1 - \alpha/2; n - 2)s\{b_0\}$$
$$b_1 \pm t(1 - \alpha/2; n - 2)s\{b_1\}$$

Let:
- $A_1$ = event that the first confidence interval does not cover β₀
- $A_2$ = event that the second confidence interval does not cover β₁

We know: $P(A_1) = \alpha$ and $P(A_2) = \alpha$

**Step 1**: Using probability theory, the probability that one or both intervals are incorrect:

$$P(A_1 \cup A_2) = P(A_1) + P(A_2) - P(A_1 \cap A_2)$$

**Step 2**: Using complementation, the probability that both intervals are correct:

$$P(\bar{A_1} \cap \bar{A_2}) = 1 - P(A_1 \cup A_2) = 1 - P(A_1) - P(A_2) + P(A_1 \cap A_2)$$ 

**(4.1)**

**Step 3**: Since $P(A_1 \cap A_2) \geq 0$, we can establish a lower bound (the **Bonferroni inequality**):

$$P(\bar{A_1} \cap \bar{A_2}) \geq 1 - P(A_1) - P(A_2)$$ 

**(4.2)**

For our situation:

$$P(\bar{A_1} \cap \bar{A_2}) \geq 1 - \alpha - \alpha = 1 - 2\alpha$$ 

**(4.2a)**

**Interpretation**: If β₀ and β₁ are separately estimated with 95% confidence intervals (α = .05), the Bonferroni inequality guarantees a family confidence coefficient of **at least 90%**.

#### Bonferroni Confidence Limits for Regression

To obtain a family confidence coefficient of at least 1 - α for both β₀ and β₁:

$$b_0 \pm Bs\{b_0\} \quad\quad b_1 \pm Bs\{b_1\}$$ 

**(4.3)**

where:

$$B = t(1 - \alpha/4; n - 2)$$ 

**(4.3a)**

**Why α/4?** Each statement needs confidence coefficient 1 - α/2, which requires the $(1 - \alpha/2/2) = (1 - \alpha/4)$ percentile of the t distribution for a two-sided interval.

---

### 📊 Example 4.1: Toluca Company (Bonferroni Intervals)

For the Toluca Company advertising example, we want 90% family confidence intervals for β₀ and β₁.

**Given** (from Chapter 2):
- $b_0 = 62.37$, $s\{b_0\} = 26.18$
- $b_1 = 3.5702$, $s\{b_1\} = .3470$
- n = 25 observations

**Calculate B:**
$$B = t(1 - .10/4; 23) = t(.975; 23) = 2.069$$

**Confidence Limits:**

For β₀:
$$62.37 \pm 2.069(26.18) = 62.37 \pm 54.16 = [8.21, 116.53]$$

For β₁:
$$3.5702 \pm 2.069(.3470) = 3.5702 \pm 0.718 = [2.85, 4.29]$$

**Conclusion**: With family confidence coefficient of at least .90:
- β₀ is between 8.20 and 116.5
- β₁ is between 2.85 and 4.29

---

### Important Comments on Bonferroni Procedure

#### 1. Conservative Nature

The Bonferroni 1 - α family confidence coefficient is actually a **lower bound** on the true family confidence coefficient. 

**Why?** Because estimates of β₀ and β₁ tend to pair up in families:
- If β₀ is incorrect and too high, β₁ tends to be incorrect and too low (and vice versa)
- This correlation means incorrect intervals occur together more than (1 - α)% of the time
- So correct pairs occur more often than the Bonferroni bound predicts

**Practical Implication**: Family confidence coefficients are frequently specified at **lower levels** (e.g., 90%) than when a single estimate is made, because the Bonferroni procedure is conservative.

#### 2. Extension to g Simultaneous Intervals

The Bonferroni inequality easily extends to g confidence intervals with family confidence coefficient 1 - α:

$$P\left(\bigcap_{i=1}^{g} \bar{A_i}\right) \geq 1 - g\alpha$$ 

**(4.4)**

**Implementation**: Construct each of the g intervals with statement confidence coefficient 1 - α/g.

#### 3. Optimal Family Size

For a given family confidence coefficient, the larger the number of confidence intervals g, the greater the multiple B becomes, which widens all intervals.

**Guideline**: The Bonferroni technique is most useful when the number of simultaneous estimates is **not too large** (typically g ≤ 5-10).

#### 4. Different Statement Confidence Coefficients

The confidence intervals don't need to have the same statement confidence coefficient. For instance:
- β₀ could be estimated with a 92% confidence interval
- β₁ could be estimated with a 98% confidence interval
- Family confidence coefficient by (4.2) would still be at least 90%

This flexibility allows focusing precision where needed most.

#### 5. Direct Use for Hypothesis Testing

Joint confidence intervals can be used directly for testing. 

**Example**: An industrial engineer theorized:
- Intercept should be 30.0
- Slope should be 2.50

Since 30.0 falls in the confidence interval for β₀ but 2.50 does **not** fall in the confidence interval for β₁, the engineer's theoretical expectations are **not correct** at the α = .10 family significance level.

#### 6. Correlation Between Estimators

The covariance between $b_0$ and $b_1$ is:

$$\sigma\{b_0, b_1\} = -\bar{X}\sigma^2\{b_1\}$$ 

**(4.5)**

**Key Insight**: When $\bar{X}$ is positive (which is typical), $b_0$ and $b_1$ are **negatively correlated**:
- If estimate $b_1$ is too high, estimate $b_0$ is likely too low
- If estimate $b_1$ is too low, estimate $b_0$ is likely too high

**Geometric Intuition**: Since observed points $(X_i, Y_i)$ fall in the first quadrant and the slope of the fitted line is too steep (β₁ overestimates β₁), the intercept is most likely to be too low (β₀ underestimates β₀).

**Important Note**: When the independent variable is $X_i - \bar{X}$ (the alternative model from 1.6), $b_0^*$ and $b_1$ are **uncorrelated** because the mean of the $X_i - \bar{X}$ observations is zero.

---

## 4.2 Simultaneous Estimation of Mean Responses

### The Problem

Often we need to estimate the **mean response** E[Y] at multiple levels of X from the same dataset.

**Example - Toluca Company**: Estimate the mean number of work hours required for lots of sizes:
- $X_h = 30$ units
- $X_h = 65$ units  
- $X_h = 100$ units

We already know how to estimate the mean response for **any one level** of X with a given confidence coefficient (from Chapter 2). Now we need procedures for **simultaneous estimation** with a family confidence coefficient.

**Why It Matters**: The separate interval estimates of E[Y_h] at different X_h levels need not all be correct or all be incorrect. Sampling errors in β₀ and β₁ may combine such that interval estimates are correct over some range of X levels and incorrect elsewhere.

---

### Working-Hotelling Procedure

The **Working-Hotelling procedure** uses the confidence band for the regression line (discussed in Section 2.6).

#### Key Concept: Confidence Band

Recall that the confidence band with coefficient 1 - α:

$$\hat{Y}_h \pm W s\{\hat{Y}_h\}$$ 

**(4.6)**

where:

$$W^2 = 2F(1 - \alpha; 2, n - 2)$$ 

**(4.6a)**

and $\hat{Y}_h$ and $s\{\hat{Y}_h\}$ are defined in (2.28) and (2.30).

**Critical Property**: The confidence band contains the **entire regression line**. Therefore, it contains the mean responses at **all X levels**.

**Family Confidence Coefficient**: Since the entire confidence band for the regression line is correct with probability 1 - α, the family confidence coefficient for these simultaneous estimates will be **at least** 1 - α.

---

### 📊 Example 4.2: Toluca Company (Working-Hotelling)

We require a family of estimates of the mean number of work hours at lot sizes $X_h = 30, 65, 100$ with family confidence coefficient .90.

**Given** (from Chapter 2):
- For $X_h = 65$ and 100: $\hat{Y}_h$ and $s\{\hat{Y}_h\}$ were calculated
- For $X_h = 30$: Need to calculate (process similar to Chapter 2)

**Summary Table**:

| $X_h$ | $\hat{Y}_h$ | $s\{\hat{Y}_h\}$ |
|-------|------------|------------------|
| 30    | 169.5      | 16.97            |
| 65    | 294.4      | 9.918            |
| 100   | 419.4      | 14.27            |

**Calculate W²:**

For family confidence coefficient .90, we need $F(.90; 2, 23) = 2.549$:

$$W^2 = 2F(.90; 2, 23) = 2(2.549) = 5.098$$
$$W = 2.258$$

**Confidence Intervals:**

For $X_h = 30$:
$$131.2 = 169.5 - 2.258(16.97) \leq E[Y_h] \leq 169.5 + 2.258(16.97) = 207.8$$

For $X_h = 65$:
$$272.0 = 294.4 - 2.258(9.918) \leq E[Y_h] \leq 294.4 + 2.258(9.918) = 316.8$$

For $X_h = 100$:
$$387.2 = 419.4 - 2.258(14.27) \leq E[Y_h] \leq 419.4 + 2.258(14.27) = 451.6$$

**Conclusion**: With family confidence coefficient .90, the mean number of work hours required is:
- Between 131.2 and 207.8 for lots of 30 parts
- Between 272.0 and 316.8 for lots of 65 parts
- Between 387.2 and 451.6 for lots of 100 parts

---

### Bonferroni Procedure for Mean Responses

The Bonferroni procedure can also be used for simultaneous estimation of mean responses at different X levels.

#### Bonferroni Confidence Limits

When E[Y_h] is to be estimated for g levels $X_h$ with family confidence coefficient 1 - α:

$$\hat{Y}_h \pm Bs\{\hat{Y}_h\}$$ 

**(4.7)**

where:

$$B = t(1 - \alpha/2g; n - 2)$$ 

**(4.7a)**

and g is the number of confidence intervals in the family.

---

### 📊 Example 4.3: Toluca Company (Bonferroni for Mean Responses)

For the same data as the Working-Hotelling example, we want Bonferroni simultaneous estimates for $X_h = 30, 65, 100$ with family confidence coefficient .90.

**Calculate B:**

We need g = 3 intervals, so:
$$B = t(1 - .10/2(3); 23) = t(.9833; 23) = 2.263$$

**Confidence Intervals:**

For $X_h = 30$:
$$131.1 = 169.5 - 2.263(16.97) \leq E[Y_h] \leq 169.5 + 2.263(16.97) = 207.9$$

For $X_h = 65$:
$$272.0 = 294.4 - 2.263(9.918) \leq E[Y_h] \leq 294.4 + 2.263(9.918) = 316.8$$

For $X_h = 100$:
$$387.1 = 419.4 - 2.263(14.27) \leq E[Y_h] \leq 419.4 + 2.263(14.27) = 451.7$$

---

### Comparison: Working-Hotelling vs. Bonferroni

**When to Use Which?**

1. **Small Number of Statements**: When the number of statements is small (g ≤ 3-4), the Bonferroni limits may be **tighter** than Working-Hotelling limits

2. **Large Number of Statements**: For larger families, Working-Hotelling confidence limits will always be **tighter** since W stays the same regardless of g, while B increases with g

3. **Number of Intervals Unknown**: When the number of confidence intervals is determined during analysis (levels of interest discovered after examining data), it's better to use the **Working-Hotelling procedure** because it encompasses all possible X levels

**In This Example**: The Working-Hotelling and Bonferroni intervals are nearly identical because:
- W = 2.258 (Working-Hotelling)
- B = 2.263 (Bonferroni for g = 3)

For g = 3, the procedures give essentially the same results.

---

### Important Comments on Mean Response Estimation

#### 1. Which Procedure is Better?

In this instance the Working-Hotelling limits are **slightly tighter** than (or the same as) the Bonferroni limits. In other cases where the number of statements is small, the Bonferroni limits may be tighter.

For larger families:
- Working-Hotelling stays constant (W doesn't change)
- Bonferroni gets wider (B increases as g increases)

**General Practice**: Once the family confidence coefficient has been decided, calculate both W and B multiples to determine which procedure provides tighter confidence limits.

#### 2. Lower Bounds on True Family Confidence Coefficient

Both procedures provide **lower bounds** to the actual family confidence coefficient, similar to the case with simultaneous estimation of β₀ and β₁.

#### 3. Unknown Levels of Interest

The levels of predictor variable X for which the mean response is to be estimated are sometimes not known in advance. Instead, levels of interest are determined as the analysis proceeds.

**Example - Toluca Company**: Lot size levels of interest were determined after analyses of other factors affecting optimum lot size were completed.

**Best Practice**: In such cases, use the **Working-Hotelling procedure** because the family confidence coefficient encompasses **all possible levels of X**, not just a fixed number g.

---

## 4.3 Simultaneous Prediction Intervals for New Observations

### The Problem

We may wish to predict g **new observations** on Y at g independent trials at g different levels of X. Simultaneous prediction intervals are frequently of interest.

**Example**: A company may wish to predict sales in each of its sales regions from a regression relation between region sales and population size in the region.

---

### Two Procedures Available

Both the **Scheffé procedure** and **Bonferroni procedure** can be used for simultaneous predictions. They differ in the multiple used:

1. **Scheffé Procedure**: Uses the F distribution
2. **Bonferroni Procedure**: Uses the t distribution

The choice depends on which provides tighter prediction limits (can be determined by calculating both multiples).

---

### Scheffé Simultaneous Prediction Limits

For g predictions with family confidence coefficient 1 - α:

$$\hat{Y}_h \pm Ss[\text{pred}]$$ 

**(4.8)**

where:

$$S^2 = gF(1 - \alpha; g, n - 2)$$ 

**(4.8a)**

and $s[\text{pred}]$ is defined in (2.38).

---

### Bonferroni Simultaneous Prediction Limits

For g predictions with family confidence coefficient 1 - α:

$$\hat{Y}_h \pm Bs[\text{pred}]$$ 

**(4.9)**

where:

$$B = t(1 - \alpha/2g; n - 2)$$ 

**(4.9a)**

**Note**: The S and B multiples can be evaluated in advance to see which procedure provides tighter prediction limits.

---

### 📊 Example 4.4: Toluca Company (Simultaneous Predictions)

The Toluca Company wishes to predict work hours required for each of the next two lots, which will consist of 80 and 100 units. The family confidence coefficient is to be 95%.

**Determine Which Procedure Gives Tighter Limits:**

Calculate the S and B multiples:

$$S^2 = 2F(.95; 2, 23) = 2(3.422) = 6.844 \quad\Rightarrow\quad S = 2.616$$
$$B = t(1 - .05/2(2); 23) = t(.9875; 23) = 2.398$$

**Conclusion**: The Bonferroni procedure will yield **somewhat tighter** prediction limits (B = 2.398 < S = 2.616).

**Calculate Needed Values** (calculations not shown in excerpt):

| $X_h$ | $\hat{Y}_h$ | $s[\text{pred}]$ | $Bs[\text{pred}]$ |
|-------|------------|------------------|-------------------|
| 80    | 348.0      | 49.91            | 119.7             |
| 100   | 419.4      | 50.87            | 122.0             |

**Simultaneous Prediction Limits** (95% family confidence):

For $X_h = 80$:
$$228.3 = 348.0 - 119.7 \leq Y_h(\text{new}) \leq 348.0 + 119.7 = 467.7$$

For $X_h = 100$:
$$297.4 = 419.4 - 122.0 \leq Y_h(\text{new}) \leq 419.4 + 122.0 = 541.4$$

---

### Important Comments on Simultaneous Predictions

#### 1. Width Comparison with Single Predictions

Simultaneous prediction intervals for g new observations at g different levels of X with family confidence coefficient 1 - α are **wider** than the corresponding single prediction intervals of (2.36).

**Why?** When the number of simultaneous predictions is not large, the difference in width is only moderate.

**Example**: A single 95% prediction interval for the Toluca Company utilizes a t multiple of:
$$t(.975; 23) = 2.069$$

which is only moderately smaller than the Bonferroni multiple:
$$B = 2.398$$

for two simultaneous predictions.

#### 2. Multiples for Simultaneous Predictions Become Larger

Note that both the B and S multiples for simultaneous predictions **become larger** as g increases.

**Contrast with Mean Response Estimation**:
- For mean responses, B becomes larger but W stays the same
- For predictions, both B and S increase with g

**Implication**: When g is large, both multiples may become so large that prediction intervals will be **too wide to be useful**.

**Solution**: Other simultaneous estimation techniques might be considered (discussed in Reference 4.1).

---

## 4.4 Regression through Origin

### When It Occurs

Sometimes the regression function is known to be **linear** and to **go through the origin** at (0, 0).

**Examples**:

1. **Output and Variable Cost**: When X is units of output and Y is variable cost, Y is zero by definition when X is zero

2. **Beer Brands and Volume**: X is the number of brands of beer stocked in a supermarket (including some supermarkets with no brands stocked) and Y is the volume of beer sales in the supermarket

---

### Model

The normal error model for regression through the origin is the same as regression model (2.1) except that **β₀ = 0**:

$$Y_i = \beta_1 X_i + \varepsilon_i$$ 

**(4.10)**

where:
- $\beta_1$ is a parameter
- $X_i$ are known constants
- $\varepsilon_i$ are independent $N(0, \sigma^2)$

The regression function for model (4.10) is:

$$E[Y] = \beta_1 X$$ 

**(4.11)**

which is a **straight line through the origin**, with slope $\beta_1$.

---

### Inferences

#### Least Squares Estimator of β₁

The least squares estimator of $\beta_1$ in regression model (4.10) is obtained by minimizing:

$$Q = \sum(Y_i - \beta_1 X_i)^2$$ 

**(4.12)**

with respect to $\beta_1$. The resulting normal equation is:

$$\sum X_i(Y_i - b_1 X_i) = 0$$ 

**(4.13)**

leading to the point estimator:

$$b_1 = \frac{\sum X_i Y_i}{\sum X_i^2}$$ 

**(4.14)**

**Key Properties**:
- The estimator $b_1$ in (4.14) is the **maximum likelihood estimator** for the normal error regression model (4.10)
- It is also an **unbiased estimator**

---

#### Fitted Values and Residuals

The fitted value $\hat{Y}_i$ for the ith case is:

$$\hat{Y}_i = b_1 X_i$$ 

**(4.15)**

The ith residual is defined as usual as the difference between the observed and fitted values:

$$e_i = Y_i - \hat{Y}_i = Y_i - b_1 X_i$$ 

**(4.16)**

---

#### Error Variance Estimator

An unbiased estimator of the error variance $\sigma^2$ for regression model (4.10) is:

$$s^2 = MSE = \frac{\sum(Y_i - \hat{Y}_i)^2}{n - 1} = \frac{\sum e_i^2}{n - 1}$$ 

**(4.17)**

**Important Note**: The denominator is **n - 1** (not n - 2) because only **one degree of freedom** is lost in estimating the single parameter β₁ in the regression function (4.11).

---

#### Confidence Limits and Tests

Confidence limits for $\beta_1$, $E[Y_h]$, and a new observation $Y_h(\text{new})$ for regression model (4.10) are shown in **Table 4.1**.

**Key Differences from Model with Intercept**:
- The t multiple has **n - 1 degrees of freedom** (not n - 2)
- Terms involve $X_i^2$ and $X_h^2$ (not $(X_i - \bar{X})^2$ and $(X_h - \bar{X})^2$)
- This is because the regression goes through the origin

**Table 4.1: Confidence Limits for Regression through Origin**

| Estimate of | Estimated Variance | Confidence Limits |
|-------------|-------------------|-------------------|
| $\beta_1$ | $s^2\{b_1\} = \frac{MSE}{\sum X_i^2}$ | $b_1 \pm ts\{b_1\}$ (4.18) |
| $E[Y_h]$ | $s^2\{\hat{Y}_h\} = \frac{X_h^2 MSE}{\sum X_i^2}$ | $\hat{Y}_h \pm ts\{\hat{Y}_h\}$ (4.19) |
| $Y_h(\text{new})$ | $s^2[\text{pred}] = MSE\left(1 + \frac{X_h^2}{\sum X_i^2}\right)$ | $\hat{Y}_h \pm ts[\text{pred}]$ (4.20) |

where: $t = t(1 - \alpha/2; n - 1)$

---

### 📊 Example 4.5: Charles Plumbing Supplies Company

The Charles Plumbing Supplies Company operates 12 warehouses. In an attempt to tighten procedures for planning and control, a consultant studied the relation between number of work units performed (X) and total variable labor cost (Y) in the warehouses during a test period.

**Data** (Table 4.2):

| Warehouse i | Work Units Performed $X_i$ | Variable Labor Cost $Y_i$ (dollars) | $X_i Y_i$ | $X_i^2$ | $\hat{Y}_i$ | $e_i$ |
|-------------|---------------------------|-------------------------------------|-----------|---------|-------------|-------|
| 1           | 20                        | 114                                 | 2,280     | 400     | 93.71       | 20.29 |
| 2           | 196                       | 921                                 | 180,516   | 38,416  | 918.31      | 2.69  |
| 3           | 115                       | 560                                 | 64,400    | 13,225  | 538.81      | 21.19 |
| ...         | ...                       | ...                                 | ...       | ...     | ...         | ...   |
| 10          | 147                       | 670                                 | 98,490    | 21,609  | 688.74      | -18.74|
| 11          | 182                       | 828                                 | 150,696   | 33,124  | 852.72      | -24.72|
| 12          | 160                       | 762                                 | 121,920   | 25,600  | 749.64      | 12.36 |
| **Total**   | **1,359**                 | **6,390**                           | **894,714** | **190,963** | **6,367.28** | **22.72** |

A scatter plot is shown in **Figure 4.1**.

![Scatter Plot and Fitted Regression through Origin - Warehouse Example](placeholder-figure-4-1.png)

**Figure 4.1**: The scatter plot shows work units performed vs. variable labor cost with the fitted regression line $\hat{Y} = 4.685X$ passing through the origin.

---

### Analysis

Model (4.10) for regression through the origin was employed since Y involves variable costs only and the other conditions of the model appeared to be satisfied as well.

From Table 4.2, columns 3 and 4:
$$\sum X_i Y_i = 894,714 \quad\text{and}\quad \sum X_i^2 = 190,963$$

**Estimate β₁:**

$$b_1 = \frac{\sum X_i Y_i}{\sum X_i^2} = \frac{894,714}{190,963} = 4.68527$$

**Estimated Regression Function:**

$$\hat{Y} = 4.68527X$$

**Interpretation**: Each additional work unit performed increases variable labor cost by an estimated $4.69.

---

### Calculate MSE

In Table 4.2:
- Column 5 shows the fitted values
- Column 6 shows the residuals

The fitted regression line is plotted in Figure 4.1 and appears to be a good fit.

**Interval Estimate of β₁** (desired with 95% confidence coefficient):

By squaring the residuals in Table 4.2, column 6, and summing:

$$s^2 = MSE = \frac{\sum e_i^2}{n - 1} = \frac{2457.6}{11} = 223.42$$

From Table 4.2, column 4:
$$\sum X_i^2 = 190,963$$

Therefore:

$$s^2\{b_1\} = \frac{MSE}{\sum X_i^2} = \frac{223.42}{190,963} = .0011700$$
$$s\{b_1\} = .034205$$

For a 95% confidence coefficient, we require $t(.975; 11) = 2.201$. 

The confidence limits, by (4.18) in Table 4.1:
$$4.68527 \pm 2.201(.034205)$$

The 95% confidence interval for $\beta_1$ is:

$$4.61 \leq \beta_1 \leq 4.76$$

**Conclusion**: With 95% confidence, it is estimated that the mean variable labor cost increases by somewhere between $4.61 and $4.76 for each additional work unit performed.

---

### Important Cautions for Using Regression through Origin

#### ⚠️ Caution 1: Residuals Don't Sum to Zero

In using regression-through-the-origin model (4.10), **the residuals must be interpreted with care** because they do **not** sum to zero usually.

**Evidence**: In Table 4.2, column 6, the residuals sum to 22.72, not zero.

**Why?** From the normal equation (4.13), the only constraint on the residuals is:
$$\sum X_i e_i = 0$$

Thus, in a residual plot, the residuals will usually **not** be balanced around the zero line.

---

#### ⚠️ Caution 2: SSE May Exceed SSTO

Another important caution: **The sum of the squared residuals** may exceed the total sum of squares for regression through the origin.

Recall for the intercept model:
$$SSE = \sum e_i^2 \quad\text{and}\quad SSTO = \sum(Y_i - \bar{Y})^2$$

This can occur when the data form a **curvilinear pattern** or a **linear pattern with an intercept away from the origin**.

**Consequence**: The coefficient of determination:
$$R^2 = 1 - \frac{SSE}{SSTO}$$
may turn out to be **negative**!

**Therefore**: $R^2$ has **no clear meaning** for regression through the origin.

---

#### ⚠️ Caution 3: Linearity and Origin Assumptions

Like any statistical model, regression-through-the-origin model (4.10) needs to be evaluated for aptness.

**Critical Assumptions to Check**:
1. The regression function must go through the origin
2. The function may not be linear
3. The variance of the error terms may not be constant

**Best Practice**: It is generally a **safe practice NOT to use regression-through-the-origin** model (4.10) and instead use the intercept regression model (2.1).

**Why?**

Even when the regression line does go through the origin:
- $b_0$ with the intercept model will differ from 0 only by a small sampling error
- Unless the sample size is very small, use of the intercept regression model (2.1) has **no disadvantages of any consequence**

**Advantages of Using Intercept Model**:
- If the regression line does **not** go through the origin, use of intercept model (2.1) will avoid potentially serious difficulties
- The intercept model is more robust and safer

---

### Important Comments on Regression through Origin

#### 1. Interval Estimation and Prediction

In interval estimation of E[Y_h] or prediction of $Y_h(\text{new})$ with regression through the origin, note that:

**The intervals (4.19) and (4.20) in Table 4.1 widen the further $X_h$ is from the origin.**

**Reason**: The value of the true regression function is known precisely at the origin, so the effect of the sampling error in the slope $b_1$ becomes increasingly important the farther $X_h$ is from the origin.

---

#### 2. Simultaneous Estimation Not Required

Since with regression through the origin only one parameter, $\beta_1$, must be estimated for regression function (4.11), **simultaneous estimation methods are not required** to make a family of statements about several mean responses.

**Why?** For a given confidence coefficient 1 - α, formula (4.19) in Table 4.1 can be used repeatedly with the given sample results for different levels of X to generate a family of statements for which the family confidence coefficient is still 1 - α.

---

#### 3. Calculating $R^2$ for Regression through Origin

Some statistical packages calculate $R^2$ for regression through the origin according to (2.72):

$$R^2 = 1 - \frac{SSE}{SSTO}$$

and hence will sometimes show a **negative value** for $R^2$.

Other statistical packages calculate $R^2$ using the **total uncorrected sum of squares** $SSTOU$ in (2.54):

$$SSTOU = \sum Y_i^2$$

This procedure avoids obtaining a negative coefficient but **lacks any meaningful interpretation**.

---

#### 4. ANOVA Tables for Regression through Origin

The ANOVA tables for regression through the origin shown in the output for many statistical packages are based on:

$$SSTOU = \sum Y_i^2$$
$$SSRU = \sum \hat{Y}_i^2 = b_1^2 \sum X_i^2$$
$$SSE = \sum(Y_i - b_1 X_i)^2$$

where **SSRU** stands for the uncorrected regression sum of squares.

**Important Note**: It can be shown that these sums of squares are **additive**:
$$SSTOU = SSRU + SSE$$

This is analogous to the breakdown in the intercept model, but uses uncorrected sums of squares.

---

## 4.5 Effects of Measurement Errors

Up to this point, we have not explicitly considered the presence of measurement errors in the observations on either the **response variable Y** or the **predictor variable X**.

We now examine briefly the effects of measurement errors in the observations on the response and predictor variables.

---

### Measurement Errors in Y

When random measurement errors are present in the observations on the response variable Y, **no new problems are created** when these errors are:
- **Uncorrelated** with each other
- **Not biased** (positive and negative measurement errors tend to cancel out)

**Example**: Consider a study of the relation between:
- Time required to complete a task (Y)
- Complexity of the task (X)

The time to complete the task may not be measured accurately because the person operating the stopwatch may not do so at the precise instants called for.

**As long as such measurement errors are**:
- Of a random nature
- Uncorrelated
- Not biased

These measurement errors are simply **absorbed in the model error term ε**.

**Key Insight**: The model error term always reflects the composite effects of a large number of factors not considered in the model, one of which now would be the random variation due to inaccuracy in the process of measuring Y.

---

### Measurement Errors in X

Unfortunately, a **different situation** holds when the observations on the predictor variable X are subject to measurement errors.

**Common Scenarios**:

Frequently, to be sure, the observations on X are:
- **Accurate**, with no measurement errors
- When X is the price of a product in different stores
- When X is the number of variables in different optimization problems
- When X is the wage rate for different classes of employees

However, at other times, **measurement errors may enter** the value observed for the predictor variable, for instance:
- When X is pressure in a tank
- When X is temperature in an oven
- When X is speed of a production line
- When X is reported age of a person

---

### Development of the Problem

We shall use the last illustration in our development of the nature of the problem.

**Setup**: Suppose we are interested in the relation between:
- Employees' piecework earnings (Y)
- Their ages (X)

**Define**:
- $X_i$ = true age of the ith employee
- $X_i^*$ = age reported by the employee on the employment record

Needless to say, **the two are not always the same**.

We define the **measurement error** $\delta_i$ as follows:

$$\delta_i = X_i^* - X_i$$ 

**(4.21)**

---

### The Regression Model We Would Like to Study

The regression model we would like to study is:

$$Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i$$ 

**(4.22)**

However, we observe only $X_i^*$, so we must replace the true age $X_i$ in (4.22) by the reported age $X_i^*$, using (4.21):

$$Y_i = \beta_0 + \beta_1(X_i^* - \delta_i) + \varepsilon_i$$ 

**(4.23)**

We can now rewrite (4.23) as follows:

$$Y_i = \beta_0 + \beta_1 X_i^* + (\varepsilon_i - \beta_1\delta_i)$$ 

**(4.24)**

---

### The Problem with Model (4.24)

Model (4.24) may appear like an ordinary regression model, with:
- Predictor variable $X^*$
- Error term $\varepsilon - \beta_1\delta$

**But it is not!**

The predictor variable observation $X_i^*$ is a **random variable**, which, as we shall see, is **correlated with the error term** $\varepsilon_i - \beta_1\delta_i$.

---

### Formal Conditions

Intuitively, we know that $\varepsilon_i - \beta_1\delta_i$ is **not independent** of $X_i^*$ since (4.21) constrains $X_i^* - \delta_i$ to equal $X_i$.

To determine the dependence formally, let us assume the following simple conditions:

$$E\{\delta_i\} = 0$$ 

**(4.25a)**

$$E\{\varepsilon_i\} = 0$$ 

**(4.25b)**

$$E\{\delta_i\varepsilon_i\} = 0$$ 

**(4.25c)**

**Interpretation**:

1. **(4.25a)**: $E\{X_i^*\} = E\{X_i + \delta_i\} = X_i$, so in our example the reported ages would be **unbiased estimates** of the true ages

2. **(4.25b)**: The usual requirement that the model error terms $\varepsilon_i$ have expectation 0, balancing around the regression line

3. **(4.25c)**: Requires that the measurement error $\delta_i$ not be correlated with the model error $\varepsilon_i$

This follows because by (4.21a):
$$\sigma\{\delta_i, \varepsilon_i\} = E\{\delta_i\varepsilon_i\}$$

since:
$$E\{\delta_i\} = E\{\varepsilon_i\} = 0$$

by (4.25a) and (4.25b).

---

### Covariance Between $X_i^*$ and Error Term

We now wish to find the covariance between the observations $X_i^*$ and the random terms $\varepsilon_i - \beta_1\delta_i$ in model (4.24) under the conditions in (4.25), which imply that:

$$E\{X_i^*\} = X_i$$

and:

$$E\{\varepsilon_i - \beta_1\delta_i\} = 0$$

The covariance is:

$$\sigma\{X_i^*, \varepsilon_i - \beta_1\delta_i\} = E[(X_i^* - E\{X_i^*\})[(\varepsilon_i - \beta_1\delta_i) - E\{\varepsilon_i - \beta_1\delta_i\}]]$$

$$= E[(X_i^* - X_i)(\varepsilon_i - \beta_1\delta_i)]$$

$$= E[\delta_i(\varepsilon_i - \beta_1\delta_i)]$$

$$= E\{\delta_i\varepsilon_i - \beta_1\delta_i^2\}$$

Now:
- $E\{\delta_i\varepsilon_i\} = 0$ by (4.25c)
- $E\{\delta_i^2\} = \sigma^2\{\delta_i\}$ by (A.15a) because $E\{\delta_i\} = 0$ by (4.25a)

We therefore obtain:

$$\sigma\{X_i^*, \varepsilon_i - \beta_1\delta_i\} = -\beta_1\sigma^2\{\delta_i\}$$ 

**(4.26)**

**Conclusion**: This covariance is **not zero** whenever there is a linear regression relation between X and Y.

---

### Implications of Correlated Predictor and Error

If we assume that the response Y and the random predictor variable $X^*$ follow a **bivariate normal distribution**, then the **conditional distribution** of the $Y_i$, $i = 1, \ldots, n$, given $X_i^*$, are normal and independent, with:

- **Conditional mean**: $E\{Y_i | X_i^*\} = \beta_0^* + \beta_1^* X_i^*$
- **Conditional variance**: $\sigma_{Y|X^*}^2$

Furthermore, it can be shown that:

$$\beta_1^* = \beta_1 \frac{\sigma_X^2}{(\sigma_X^2 + \sigma_\delta^2)}$$

where:
- $\sigma_X^2$ is the variance of X
- $\sigma_\delta^2$ is the variance of Y

Hence, the **least squares slope estimate** from fitting Y on $X^*$ is **not an estimate of** $\beta_1$, but is an estimate of:

$$\beta_1^* \leq \beta_1$$

---

### The Attenuation Effect

The resulting estimated regression coefficient of $\beta_1^*$ will be **too small on average**, with the magnitude of the bias dependent upon the relative sizes of $\sigma_X^2$ and $\sigma_\delta^2$.

**Cases**:

1. If $\sigma_\delta^2$ is **small relative** to $\sigma_X^2$, then the bias would be small
2. Otherwise the bias may be substantial

**This is called the attenuation effect** - measurement errors in X cause the estimated slope to be attenuated (reduced) toward zero.

---

### Discussion of Possible Approaches

Discussion of possible approaches to estimating $\beta_1^*$ that are obtained by estimating these unknown variances $\sigma_X^2$ and $\sigma_\delta^2$ will be found in specialized texts such as Reference 4.2.

---

### Instrumental Variables

Another approach is to use **additional variables** that are known to be related to the true value of X but not to the errors of measurement δ.

Such variables are called **instrumental variables** because they are used as an instrument in studying the relation between X and Y.

Instrumental variables make it possible to obtain **consistent estimators** of the regression parameters.

Again, the reader is referred to Reference 4.2.

---

### Comment: Why Are Measurement Errors in X Special?

What, it may be asked, is the distinction between the case when X is a random variable (considered in Chapter 2), and the case when X is subject to random measurement errors, and why are there special problems with the latter?

**Key Distinction**:

When X is a random variable, the observations on X are **not under the control of the analyst** and will vary at random from trial to trial, as when X is the number of persons entering a store in a day.

**But**: If this random variable X is **not subject to measurement errors**, it can be **accurately ascertained** for a given trial.

Thus, if there are no measurement errors in counting the number of persons entering a store in a day, the analyst has accurate information to study the relation between number of persons entering the store and sales, even though the levels of number of persons entering the store that actually occur cannot be controlled.

**On the other hand**: If measurement errors are present in the observed number of persons entering the store, a distorted picture of the relation between number of persons and sales will occur because the sales observations will frequently be matched against an incorrect number of persons.

---

## 4.6 Berkson Model

### The Special Case with No Problem

There is one situation where **measurement errors in X are no problem**.

This case was first noted by **Berkson** (Ref. 4.3).

**Description**: Frequently, in an experiment the predictor variable is set at a **target value**.

**Examples**:

1. In an experiment on the effect of room temperature on word processor productivity, the temperature may be set at target levels of:
   - 68° F
   - 70° F
   - 72° F
   
   according to the temperature control on the thermostat

2. The observed temperature $X_i^*$ is fixed here, whereas the actual temperature $X_i$ is a random variable since the thermostat may not be completely accurate

3. Similar situations exist when:
   - Water pressure is set according to a gauge
   - Employees of specified ages according to their employment records are selected for a study

---

### The Berkson Measurement Error Model

In all of these cases, the observation $X_i^*$ is a **fixed quantity**, whereas the unobserved true value $X_i$ is a **random variable**.

The measurement error is, as before:

$$\delta_i = X_i^* - X_i$$ 

**(4.27)**

However, here there is **no constraint** on the relation between $X_i^*$ and $\delta_i$, since $X_i^*$ is a fixed quantity.

Again, we assume that $E\{\delta_i\} = 0$.

---

### Why the Berkson Case Has No Problem

Model (4.24), which we obtained when replacing $X_i$ by $X_i^* - \delta_i$, is **still applicable** for the Berkson case:

$$Y_i = \beta_0 + \beta_1 X_i^* + (\varepsilon_i - \beta_1\delta_i)$$ 

**(4.28)**

The expected value of the error term, $E\{\varepsilon_i - \beta_1\delta_i\}$, is zero as before under conditions (4.25a) and (4.25b), since:

$$E\{\varepsilon_i\} = 0$$

and:

$$E\{\delta_i\} = 0$$

However, $\varepsilon_i - \beta_1\delta_i$ is now **uncorrelated with** $X_i^*$, since $X_i^*$ is a constant for the Berkson case.

Hence, the following conditions of an ordinary regression model are met:

1. The error terms have expectation zero
2. The predictor variable is a constant, and hence the error terms are not correlated with it

---

### Conclusion for Berkson Model

Thus, **least squares procedures can be applied for the Berkson case without modification**, and the estimators $b_0$ and $b_1$ will be **unbiased**.

If we can make the standard **normality and constant variance assumptions** for the errors $\varepsilon_i - \beta_1\delta_i$, the usual tests and interval estimates can be utilized.

---

## 4.7 Inverse Predictions

At times, a regression model of Y on X is used to make a **prediction of the value of X** which gave rise to a new observation Y.

This is known as an **inverse prediction**.

We illustrate inverse predictions by two examples:

---

### Example 1: Trade Association

A trade association analyst has regressed the selling price of a product (Y) on its cost (X) for the 15 member firms of the association.

The selling price $Y_h(\text{new})$ for another firm not belonging to the trade association is known, and it is desired to estimate the cost $X_h(\text{new})$ for this firm.

---

### Example 2: Drug Dosage

A regression analysis of the amount of decrease in cholesterol level (Y) achieved with a given dosage of a new drug (X) has been conducted, based on observations for 50 patients.

A physician is treating a new patient for whom the cholesterol level should decrease by the amount $Y_h(\text{new})$.

It is desired to estimate the appropriate dosage level $X_h(\text{new})$ to be administered to bring about the needed cholesterol decrease $Y_h(\text{new})$.

---

### The Inverse Prediction Method

In inverse predictions, regression model (2.1) is assumed as before:

$$Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i$$ 

**(4.29)**

The estimated regression function based on n observations is obtained as usual:

$$\hat{Y} = b_0 + b_1 X$$ 

**(4.30)**

A new observation $Y_h(\text{new})$ becomes available, and it is desired to estimate the level $X_h(\text{new})$ that gave rise to this new observation.

**Natural Point Estimator**: A natural point estimator is obtained by solving (4.30) for X, given $Y_h(\text{new})$:

$$\hat{X}_{h(\text{new})} = \frac{Y_{h(\text{new})} - b_0}{b_1} \quad\quad b_1 \neq 0$$ 

**(4.31)**

where $\hat{X}_{h(\text{new})}$ denotes the point estimator of the new level $X_h(\text{new})$.

**Figure 4.2** contains a representation of this point estimator for an example to be discussed shortly.

![Scatter Plot and Fitted Regression Line - Calibration Example](placeholder-figure-4-2.png)

---

### Confidence Interval for $X_h(\text{new})$

It can be shown that the estimator $\hat{X}_{h(\text{new})}$ is the **maximum likelihood estimator** of $X_h(\text{new})$ for normal error regression model (2.1).

Approximate **1 - α confidence limits** for $X_h(\text{new})$ are:

$$\hat{X}_{h(\text{new})} \pm t(1 - \alpha/2; n - 2)s[\text{predX}]$$ 

**(4.32)**

where:

$$s^2[\text{predX}] = \frac{MSE}{b_1^2}\left[1 + \frac{1}{n} + \frac{(\hat{X}_{h(\text{new})} - \bar{X})^2}{\sum(X_i - \bar{X})^2}\right]$$ 

**(4.32a)**

---

### 📊 Example 4.6: Galactose Concentration (Calibration)

A medical researcher studied a new, quick method for measuring low concentration of galactose (sugar) in the blood.

Twelve samples were used in the study containing known concentrations (X), with three samples at each of four different levels.

The measured concentration (Y) was then observed for each sample.

Linear regression model (2.1) was fitted with the following results:

$$n = 12 \quad\quad b_0 = -.100 \quad\quad b_1 = 1.017 \quad\quad MSE = .0272$$

$$s\{b_1\} = .0142 \quad\quad \bar{X} = 5.500 \quad\quad \bar{Y} = 5.492 \quad\quad \sum(X_i - \bar{X})^2 = 135$$

$$\hat{Y} = -.100 + 1.017X$$

The data and the estimated regression line are plotted in **Figure 4.2**.

**First Step - Test for Linear Association**:

The researcher first wished to make sure that there is a linear association between the two variables.

A test of $H_0: \beta_1 = 0$ versus $H_a: \beta_1 \neq 0$, utilizing test statistic:

$$t^* = \frac{b_1}{s\{b_1\}} = \frac{1.017}{.0142} = 71.6$$

was conducted for $\alpha = .05$.

Since $t(.975; 10) = 2.228$ and $|t^*| = 71.6 > 2.228$, it was concluded that $\beta_1 \neq 0$, or that a linear association exists between the measured concentration and the actual concentration.

---

### Inverse Prediction for New Patient

The researcher now wishes to use the regression relation to ascertain the actual concentration $X_h(\text{new})$ for a new patient for whom the quick procedure yielded a measured concentration of $Y_h(\text{new}) = 6.52$.

It is desired to estimate $X_h(\text{new})$ by means of a 95% confidence interval.

**Calculate Point Estimate:**

Using (4.31) and (4.32a), we obtain:

$$\hat{X}_{h(\text{new})} = \frac{6.52 - (-.100)}{1.017} = 6.509$$

$$s^2[\text{predX}] = \frac{.0272}{(1.017)^2}\left[1 + \frac{1}{12} + \frac{(6.509 - 5.500)^2}{135}\right] = .0287$$

so that:
$$s[\text{predX}] = .1694$$

We require $t(.975; 10) = 2.228$, and using (4.32) we obtain the confidence limits:

$$6.509 \pm 2.228(.1694)$$

Hence, the **95% confidence interval** is:

$$6.13 \leq X_{h(\text{new})} \leq 6.89$$

**Conclusion**: Thus, it can be concluded with 95% confidence that the actual galactose concentration for the patient is between 6.13 and 6.89.

This is approximately a **±6 percent error**, which is considered reasonable by the researcher.

---

### Important Comments on Inverse Predictions

#### 1. Calibration Problem

The inverse prediction problem is also known as a **calibration problem** since it is applicable when:

- Inexpensive, quick, and approximate measurements (Y) are related to precise, often expensive, and time-consuming measurements (X) based on n observations

The resulting regression model is then used to estimate the precise measurement $X_h(\text{new})$ for a new approximate measurement $Y_h(\text{new})$.

We illustrated this use in the calibration example.

---

#### 2. Appropriateness of Approximate Confidence Interval

The approximate confidence interval (4.32) is appropriate if the quantity:

$$\frac{[t(1 - \alpha/2; n - 2)]^2 MSE}{b_1^2 \sum(X_i - \bar{X})^2}$$ 

**(4.33)**

is small, say less than .1.

For the calibration example, this quantity is:

$$\frac{(2.228)^2(.0272)}{(1.017)^2(135)} = .00097$$

so that the approximate confidence interval is appropriate here.

---

#### 3. Simultaneous Prediction Intervals

Simultaneous prediction intervals based on g different new observed measurements $Y_h(\text{new})$, with a 1 - α family confidence coefficient, are easily obtained by using either the Bonferroni or the Scheffé procedures discussed in Section 4.3.

The value of $t(1 - \alpha/2; n - 2)$ in (4.32) is replaced by either:

$$B = t(1 - \alpha/2g; n - 2)$$

or:

$$S = [gF(1 - \alpha; g, n - 2)]^{1/2}$$

---

#### 4. Controversy Over Inverse Regression

The inverse prediction problem has aroused controversy among statisticians.

Some statisticians have suggested that **inverse predictions should be made in direct fashion by regressing X on Y**.

This regression is called **inverse regression**.

However, this approach has its own issues and is less commonly used than the calibration method presented here.

---

## 4.8 Choice of X Levels

When regression data are obtained by **experiment**, the levels of X at which observations on Y are to be taken are under the control of the experimenter.

Among other things, the experimenter will have to consider:

1. **How many levels of X should be investigated?**
2. **What shall the two extreme levels be?**
3. **How shall the other levels of X, if any, be spaced?**
4. **How many observations should be taken at each level of X?**

---

### No Single Answer

There is **no single answer** to these questions, since different purposes of the regression analysis lead to different answers.

**Possible Objectives**:

The possible objectives in regression analysis are varied, as we have noted earlier:

1. The main objective may be to **estimate the slope** of the regression line
2. Or, in some cases, to **estimate the intercept**
3. In many cases, the main objective is to **predict one or more new observations** or to **estimate one or more mean responses**
4. When the regression function is curvilinear, the main objective may be to **locate the maximum or minimum mean response**
5. At still other times, the main purpose is to **determine the nature** of the regression function

---

### How Purpose Affects Design

To illustrate how the purpose affects the design, consider the variances of $b_0$, $b_1$, $\hat{Y}_h$, and $s[\text{pred}]$ for predicting $Y_h(\text{new})$, which were developed earlier for regression model (2.1):

$$\sigma^2\{b_0\} = \sigma^2\left[\frac{1}{n} + \frac{\bar{X}^2}{\sum(X_i - \bar{X})^2}\right]$$ 

**(4.34)**

$$\sigma^2\{b_1\} = \frac{\sigma^2}{\sum(X_i - \bar{X})^2}$$ 

**(4.35)**

$$\sigma^2\{\hat{Y}_h\} = \sigma^2\left[\frac{1}{n} + \frac{(X_h - \bar{X})^2}{\sum(X_i - \bar{X})^2}\right]$$ 

**(4.36)**

$$\sigma^2[\text{pred}] = \sigma^2\left[1 + \frac{1}{n} + \frac{(X_h - \bar{X})^2}{\sum(X_i - \bar{X})^2}\right]$$ 

**(4.37)**

---

### Minimizing Variance of Slope $b_1$

If the main purpose of the regression analysis is to **estimate the slope** $\beta_1$, the variance of $b_1$ is minimized if $\sum(X_i - \bar{X})^2$ is **maximized**.

**Implementation**: This is accomplished by using **two levels of X**, at the two extremes for the scope of the model, and placing half of the observations at each of the two levels.

Of course, if one were not sure of the linearity of the regression function, one would be hesitant to use only two levels since they would provide no information about possible departures from linearity.

---

### Estimating the Intercept $\beta_0$

If the main purpose is to **estimate the intercept** $\beta_0$, the number and placement of levels does not affect the variance of $b_0$ as long as $\bar{X} = 0$.

On the other hand, to estimate the mean response or to predict a new observation at the level $X_h$, the relevant variance is minimized by using X levels so that $\bar{X} = X_h$.

---

### General Advice from D. R. Cox

Although the number and spacing of X levels depends very much on the major purpose of the regression analysis, the **general advice given by D. R. Cox** is still relevant:

> Use two levels when the object is primarily to examine whether or not ... (the predictor variable) ... has an effect and in which direction that effect is. Use three levels whenever a description of the response curve by its slope and curvature is likely to be adequate; this should cover most cases. Use four levels if further examination of the shape of the response curve is important. Use more than four levels when it is required to estimate the detailed shape of the response curve, or when the curve is expected to rise to an asymptotic value, or in general to show features not adequately described by slope and curvature. Except in these last cases it is generally satisfactory to use equally spaced levels with equal numbers of observations per level (Ref. 4.4).

---

### Visual Summary

The variances in (4.34)-(4.37) all have the term $\sum(X_i - \bar{X})^2$ in the denominator, highlighting its importance in determining precision of estimates and predictions.

**Key Principle**: Spreading out the X values (maximizing $\sum(X_i - \bar{X})^2$) generally improves precision, but the optimal design depends on whether you're primarily interested in:
- The slope
- The intercept
- Predictions at specific X values
- Understanding the shape of the relationship

---

## Summary

This chapter covered essential techniques for making **simultaneous statistical inferences** from regression data:

### Key Concepts

1. **Family vs. Statement Confidence**: When making multiple inferences, we must distinguish between individual statement confidence and the overall family confidence coefficient

2. **Bonferroni Procedure**: A simple, general method for constructing simultaneous confidence intervals by using more stringent individual confidence levels (1 - α/g instead of 1 - α)

3. **Working-Hotelling Procedure**: Uses the regression line confidence band to make simultaneous inferences about mean responses at multiple X levels

4. **Simultaneous Predictions**: Both Scheffé and Bonferroni procedures can be used for predicting multiple new observations

5. **Regression through Origin**: When the regression line is known to pass through (0,0), a simpler model can be used, but care must be taken as residuals don't sum to zero and R² may be negative

6. **Measurement Errors**: 
   - Errors in Y are generally absorbed into the model error term
   - Errors in X can cause serious bias (attenuation effect) unless they follow the Berkson model

7. **Inverse Predictions (Calibration)**: Regression models can be used to estimate X values that gave rise to observed Y values, useful in calibration problems

8. **Experimental Design**: The choice of X levels and number of observations depends critically on the primary objective of the analysis

### Practical Takeaways

- Always consider whether you need simultaneous inference procedures when making multiple estimates
- The Bonferroni procedure is conservative but simple and widely applicable
- Regression through origin should be used sparingly - the intercept model is generally safer
- Be extremely cautious about measurement errors in predictor variables
- Inverse predictions require special confidence interval formulas
- Design experiments with your primary inferential goal in mind

---

## References

**Reference 4.1**: [More advanced simultaneous estimation techniques for large families of estimates]

**Reference 4.2**: [Detailed treatment of measurement errors and instrumental variables in regression]

**Reference 4.3**: Berkson, J. "Are There Two Regressions?" *Journal of the American Statistical Association* 45 (1950), pp. 164-80.

**Reference 4.4**: Cox, D. R. *Planning of Experiments*. New York: John Wiley & Sons, 1958.

---

## Practice Problems

### Conceptual Questions

1. Explain in your own words why the probability that two 95% confidence intervals are both correct is less than 95%.

2. When would you choose the Bonferroni procedure over the Working-Hotelling procedure for simultaneous estimation of mean responses?

3. Why is regression through the origin generally not recommended even when you believe the true intercept is zero?

4. What is the "attenuation effect" and why is it problematic?

### Computational Problems

1. Given a regression with n = 30 observations, calculate the Bonferroni multiplier B for:
   - g = 2 simultaneous intervals with 90% family confidence
   - g = 5 simultaneous intervals with 95% family confidence

2. For regression through origin with $\sum X_i Y_i = 1250$, $\sum X_i^2 = 300$, and $\sum e_i^2 = 48$ (n = 15):
   - Estimate β₁
   - Construct a 95% confidence interval for β₁
   - Calculate MSE

3. In an inverse prediction problem with $\hat{Y} = 10 + 2X$, $Y_h(\text{new}) = 25$:
   - Calculate the point estimate $\hat{X}_{h(\text{new})}$
   - If MSE = 4, n = 20, $\bar{X} = 6$, $\sum(X_i - \bar{X})^2 = 100$, construct a 95% confidence interval

---

*End of Chapter 4 Notes*
