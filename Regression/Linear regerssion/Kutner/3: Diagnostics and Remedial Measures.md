
# Chapter 3: Diagnostics and Remedial Measures

## 🎯 Chapter Overview

**The Big Picture:** Before trusting any regression model, we must verify it's actually appropriate for our data. This chapter teaches you how to be a "regression detective" - spotting problems and fixing them.

> **Critical Insight:** Just because you *can* fit a regression line doesn't mean you *should*. Model diagnostics are not optional - they're essential!

**What We'll Learn:**
1. How to check if our predictor variable ($X$) is behaving properly
2. What residuals tell us about model adequacy
3. Visual and formal tests for detecting model violations
4. How to fix problems when we find them (transformations!)
5. A complete real-world case study

---

## 3.1 Diagnostics for Predictor Variable

### Why Check the Predictor Variable First?

Before diving into residual analysis, we need to understand our predictor variable $X$. Think of it like checking your ingredients before cooking - if the raw materials are problematic, the final dish will be too.

**Key Questions We're Asking:**
- Are there outliers in $X$ that could distort our regression line?
- What's the range and spread of $X$ values?
- For time series: Is there a pattern in how $X$ was collected?

### Diagnostic Plots for $X$

#### 1. **Dot Plot**

**What it shows:** Distribution of $X$ values along a number line

**📊 Example 3.1a - Toluca Company (Lot Sizes)**

![Dot plot showing lot sizes from 20 to 120](placeholder-dot-plot.png)

- Minimum lot size: 20
- Maximum lot size: 120
- Spread throughout interval
- **No extreme outliers detected** ✓
- Several repeated lot sizes (multiple runs at same size)

**What to look for:**
- Gaps in coverage
- Extreme values far from the rest
- Clustering patterns

---

#### 2. **Sequence Plot** (for time-ordered data)

**What it shows:** Values of $X$ plotted against their order of occurrence

**When to use:** Whenever data collected over time or in sequence (geographic regions, production runs, etc.)

**📊 Example 3.1b - Toluca Company (Production Run Sequence)**

```
Lot Size (Y-axis) vs. Run Number (X-axis)
Shows ups and downs but no systematic pattern
```

**What to look for:**
- Systematic trends (small lots early, large lots later)
- Seasonal patterns
- Any structure that could affect the regression

**Why this matters:** If smaller lot sizes were systematically used early in the study and larger sizes later, this time-related factor might confound our analysis.

> **Good news for Toluca:** No special pattern detected - lot sizes appear randomly scattered over time ✓

---

#### 3. **Stem-and-Leaf Plot**

**What it shows:** Frequency distribution (like a histogram but preserves actual values)

**📊 Example 3.1c - Toluca Company**

```
2  | 0
3  | 000
4  | 00
5H | 000
6  | 0
7M | 000
8  | 000
9H | 0000
10 | 00
11 | 00
12 | 0
```

**Key:** M = Median location, H = Hinges (quartiles)

**Reading this:** All lot sizes are multiples of 10

---

#### 4. **Box Plot**

**What it shows:** Five-number summary (min, Q1, median, Q3, max) in one visualization

**📊 Example 3.1d - Toluca Company**

![Box plot of lot sizes](placeholder-box-plot.png)

**Key features revealed:**
- Middle half of data: 50 to 90 (the "box")
- Median: Centered in the box (fairly symmetric distribution)
- Range: 20 to 120

**Advantages:** 
- Excellent for spotting outliers
- Shows symmetry/skewness at a glance
- Particularly useful with many observations

---

## 3.2 Residuals: The Heart of Diagnostics

### What Are Residuals, Really?

**Residual = Observed Reality - Model's Prediction**

$$e_i = Y_i - \hat{Y}_i \quad \text{(3.1)}$$

Where:
- $e_i$ = **residual** for observation $i$ (what we can observe)
- $Y_i$ = actual observed value
- $\hat{Y}_i$ = fitted value from regression

**vs. Error Terms:**

$$\varepsilon_i = Y_i - E\{Y_i\} \quad \text{(3.2)}$$

Where:
- $\varepsilon_i$ = **true error** (theoretical, unknown)
- $E\{Y_i\}$ = true expected value

> **Key Distinction:** 
> - **Errors** ($\varepsilon_i$): Unknown theoretical quantities
> - **Residuals** ($e_i$): Observable, calculable quantities we use to estimate errors

**The Philosophy:** If our model is correct, residuals should behave like the theoretical errors. If they don't, our model has problems!

---

### Properties of Residuals

Understanding these properties helps us interpret diagnostic plots correctly.

#### **Property 1: Mean of Residuals**

$$\bar{e} = \frac{\sum e_i}{n} = 0 \quad \text{(3.3)}$$

**Why:** By least squares construction, residuals always sum to zero.

**Implication:** Mean of residuals tells us nothing about whether errors have mean zero!

---

#### **Property 2: Variance of Residuals**

$$s^2 = \frac{\sum(e_i - \bar{e})^2}{n-2} = \frac{\sum e_i^2}{n-2} = \frac{SSE}{n-2} = MSE \quad \text{(3.4)}$$

**Key insight:** Since $\bar{e} = 0$, the variance formula simplifies beautifully.

**If model is appropriate:** $MSE$ is an unbiased estimator of $\sigma^2$

---

#### **Property 3: Residuals Are NOT Independent**

**Common misconception:** "Residuals are independent because errors are independent."

**Reality:** Residuals are related because they all come from the same fitted regression line!

**Why this matters:** 
- All residuals involve the same $b_0$ and $b_1$ estimates
- Two constraints bind them:
  - Constraint 1: $\sum e_i = 0$ (from equation 1.17)
  - Constraint 2: $\sum X_i e_i = 0$ (from equation 1.19)

**Practical impact:** With large samples ($n$ large relative to parameters), dependency is negligible and can be ignored for most purposes.

---

### Semistudentized Residuals

**The Problem:** Raw residuals $e_i$ are measured in the original units of $Y$. Hard to compare across different datasets or identify outliers objectively.

**The Solution:** Standardize them!

$$e_i^* = \frac{e_i - \bar{e}}{\sqrt{MSE}} = \frac{e_i}{\sqrt{MSE}} \quad \text{(3.5)}$$

Since $\bar{e} = 0$, we just divide by $\sqrt{MSE}$.

**Why "semistudentized"?**
- **True studentized** would use the actual standard deviation of each $e_i$
- That SD is complex and varies for different residuals
- $\sqrt{MSE}$ is only an *approximation* of that standard deviation
- Hence "semi"

> **Important Note:** We'll learn about fully **studentized residuals** in Chapter 10, which account for the varying precision of different residuals. Those are even better for identifying outliers!

---

## 3.3 Using Residuals to Diagnose Model Problems

### The Six Deadly Sins of Regression Models

When we fit a regression model $Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i$, we're assuming certain conditions hold. Residual analysis helps us check if they actually do.

**Six Major Departures to Check:**

1. **Regression function is not linear** → Need different functional form
2. **Error variance is not constant** → Violates homoscedasticity assumption
3. **Error terms are not independent** → Usually in time series or spatial data
4. **Model fits all but a few outlier observations** → Data quality issues
5. **Error terms are not normally distributed** → Affects inference validity
6. **Important predictor variables omitted** → Model misspecification

**The Diagnostic Toolkit** (7 key plots):

1. Residuals vs. predictor variable ($e$ vs. $X$)
2. Absolute/squared residuals vs. predictor ($|e|$ or $e^2$ vs. $X$)
3. Residuals vs. fitted values ($e$ vs. $\hat{Y}$)
4. Residuals vs. time or sequence ($e$ vs. order)
5. Residuals vs. omitted variables ($e$ vs. $Z$)
6. Box plot of residuals
7. Normal probability plot of residuals

---

### 📊 Example 3.2 - Toluca Company Diagnostic Plots

Let's see these plots in action using our running example.

![Figure 3.2: Four diagnostic plots for Toluca](placeholder-figure-3-2.png)

#### **(a) Residual Plot against $X$ (Lot Size)**

**What we see:**
- Residuals scattered around horizontal line at zero
- No systematic pattern (no funnel shape, no curve)
- Random scatter above and below zero

**Interpretation:** ✓ Linear relationship appears appropriate ✓ Constant variance appears reasonable

---

#### **(b) Sequence Plot (Residuals vs. Run Number)**

**What we see:**
- Residuals bounce up and down
- No trend over time
- No cyclical pattern

**Interpretation:** ✓ No evidence of time-related correlation ✓ Independence assumption appears satisfied

---

#### **(c) Box Plot of Residuals**

**What we see:**
- Box fairly symmetric around zero
- No extreme outliers extending far from whiskers

**Interpretation:** ✓ Distribution appears reasonably symmetric ✓ No major outliers detected

---

#### **(d) Normal Probability Plot**

**What we see:**
- Points fall reasonably close to straight line
- No severe S-curve or other departures

**Interpretation:** ✓ Normality assumption appears reasonable

**Conclusion for Toluca Company:** All diagnostic plots support the appropriateness of regression model (2.1) for this data! 🎉

---

## 3.3.1 Nonlinearity of Regression Function

### How to Detect It

**Best diagnostic:** Residual plot against $X$ (or against $\hat{Y}$)

**What to look for:**
- Systematic curved pattern in residuals
- Residuals consistently positive in some regions, negative in others

### 📊 Example 3.3 - Transit Ridership (Clear Nonlinearity)

**Context:** Studying relationship between:
- $X$ = Number of maps distributed (thousands)
- $Y$ = Increase in ridership (thousands)

**The Data:**

| City | Maps Distributed ($X$) | Ridership Increase ($Y$) | Fitted ($\hat{Y}$) | Residual ($e$) |
|------|------------------------|--------------------------|---------------------|-----------------|
| 1 | 80 | 1.66 | 1.66 | -1.06 |
| 2 | 220 | 7.75 | 7.75 | -1.05 |
| 3 | 140 | 4.27 | 4.27 | 1.03 |
| 4 | 120 | 3.40 | 3.40 | 0.60 |
| 5 | 180 | 6.01 | 6.01 | 0.54 |
| 6 | 100 | 2.53 | 2.53 | -0.38 |
| 7 | 200 | 6.88 | 6.88 | -0.28 |
| 8 | 160 | 5.14 | 5.14 | 0.61 |

**Fitted regression:** $\hat{Y} = -1.82 + 0.0435X$

---

#### **Figure 3.3a: Scatter Plot with Fitted Line**

![Scatter plot showing curved relationship](placeholder-scatter-curved.png)

**What we see:**
- Data points suggest **curved** relationship
- Straight line is clearly a poor fit

---

#### **Figure 3.3b: Residual Plot**

![Residual plot showing systematic pattern](placeholder-residual-curved.png)

**The smoking gun:**
- Residuals **negative** for small $X$ values
- Residuals **positive** for medium $X$ values  
- Residuals **negative again** for large $X$ values
- Clear **systematic fashion** (not random scatter!)

**Interpretation:** The residual plot actually reveals the problem MORE clearly than the scatter plot! The systematic pattern proves our linear model is inappropriate.

> **Key Lesson:** When scatter plot has steep slope or tight clustering, the systematic pattern in residuals can be easier to see than in the original scatter plot.

---

### Residuals vs. $\hat{Y}$ - When It's Useful

**Figure 3.4 - Comment Section:**

For **simple linear regression**, plotting residuals against $\hat{Y}$ gives equivalent information to plotting against $X$, because:

$$\hat{Y}_i = b_0 + b_1 X_i$$

The fitted values are just a linear transformation of $X$ values!

**When residuals vs. $\hat{Y}$ becomes crucial:**
- **Curvilinear regression** (Chapter 7)
- **Multiple regression** (Chapter 6+)

In these cases, the relationship between $X$ and $\hat{Y}$ is more complex, so separate plots against fitted values and predictor variables are both helpful.

---

### 📊 Prototype Patterns for Residual Plots

**Figure 3.4: What "Good" and "Bad" Look Like**

#### **(a) Appropriate Linear Model** ✓

```
    e |     •        •
      |  •     •  •    
    0 |___•_____•___•____
      |    •  •    •
      |___________________ X
```

**Characteristics:**
- Horizontal band centered at zero
- No systematic patterns
- Random scatter of positive and negative residuals

---

#### **(b) Indicates Need for Curvilinear Function** ✗

```
    e |  •           •
      |    •       •
    0 |______•_•________
      |        •
      |___________________ X
```

**Pattern:** Residuals change from negative → positive → negative (or vice versa)

**Solution:** Try polynomial or other nonlinear function

---

#### **(c) Another Curvilinear Pattern** ✗

```
    e |              
      |        •  •  •
    0 |____•______•____
      | •    •  
      |___________________ X
```

**Solution:** Different transformation needed than (b)

---

#### **(d) Time-Related Effect** ✗

```
    e |           •  •  
      |        •        •
    0 |____•_____________
      | •    
      |___________________ Time
```

**Pattern:** Systematic trend over time

**Implication:** Important predictor (time) has been omitted!

---

## 3.3.2 Nonconstancy of Error Variance

### The Problem: Heteroscedasticity

**What we want:** $\sigma^2$ (error variance) constant for all levels of $X$

**What sometimes happens:** Error variance increases (or decreases) with $X$

### How to Detect It

**Plots to examine:**
- Residuals vs. $X$ (or vs. $\hat{Y}$)
- **Absolute residuals** vs. $X$
- **Squared residuals** vs. $X$

**Why absolute/squared?** The signs of residuals aren't meaningful for checking variance - we care about magnitude!

---

### 📊 Example 3.5 - Blood Pressure vs. Age

**Context:** Studying diastolic blood pressure ($Y$) as function of age ($X$) in healthy adult women

**Figure 3.5a: Regular Residual Plot**

![Residual plot showing increasing spread](placeholder-residual-variance.png)

**What we see:**
- Spread of residuals **increases** as age increases
- Funnel or megaphone shape
- Older women show more variability in blood pressure

**Interpretation:** Error variance is **larger for older women** than for younger ones

> **Medical insight:** This makes biological sense - older individuals often show more variability in health measures.

---

**Figure 3.4a: Prototype for Constant Variance** ✓

```
    e |  • • •   • • •
      |  • • •   • • •
    0 |___• •_____• •___
      |  • • •   • • •
      |___________________ X
```

**Characteristics:**
- Vertical spread stays the same across all $X$ values
- Band has uniform width

---

**Figure 3.4c: Prototype for Increasing Variance** ✗ (Megaphone)

```
    e |            •  •  •
      |         •   •   •
    0 |____•___•___•_____
      |         •   •   •
      |            •  •  •
      |___________________ X
```

**Pattern:** Spread widens as $X$ increases

---

### Using Absolute Residuals to See Variance Pattern More Clearly

**Why this helps:** Placing all information about changing magnitudes above the zero line makes the pattern more obvious.

---

**Figure 3.5b: Absolute Residual Plot**

```
|e| |              • • •
    |           •  • • •
    |        •  •  •
    |     •  •  •
    |  •  •  •
    |_____________________ Age
```

**The pattern is unmistakable:** Clear upward trend shows increasing variability with age.

> **Pro tip:** When you have many observations, absolute or squared residual plots are often clearer than regular residual plots for diagnosing variance issues.

---

## 3.3.3 Presence of Outliers

### What Are Outliers in Regression?

**Definition:** Observations with **extreme residuals** - they lie far from the fitted regression line.

**Identification methods:**
1. **Visual:** Residual plots, box plots, stem-and-leaf plots, dot plots
2. **Numerical:** Semistudentized residuals

### The "Rough Rule of Thumb"

With large samples, consider semistudentized residuals with **absolute value ≥ 4** as potential outliers.

$$|e_i^*| \geq 4 \quad \text{suggests outlier}$$

> **More refined approaches in Chapter 10!** We'll learn more sophisticated methods for identifying and handling outliers.

---

### 📊 Example 3.6 - Toluca Company Outlier Check

**Figure 3.6: Semistudentized Residuals**

![Plot showing one observation circled](placeholder-outlier-plot.png)

$$\frac{e}{\sqrt{MSE}}$$

**What we see:**
- One observation circled
- This residual represents observation **almost 6 standard deviations** from fitted value!

**Interpretation:** This is definitely an outlier deserving investigation.

---

### Should You Just Delete Outliers?

**NO! Not automatically.** Outliers can result from:

1. **Mistakes** → Recording error, calculation error, equipment malfunction
   - **Action:** Discard if confirmed mistake
   
2. **Extraneous causes** → Unusual circumstances not representative of normal conditions
   - **Action:** May discard if cause identified and atypical

3. **Model inadequacy** → Maybe the model itself is wrong!
   - **Action:** Don't discard - fix the model instead

4. **Valuable information** → Outlier from interaction with omitted predictor
   - **Action:** Keep and investigate - it's telling you something important!

> **Important:** A **major reason** for discarding outlier observations is that under least squares, the fitted line can be disproportionately pulled toward outliers, distorting the fit for all other observations.

---

### 📊 Example 3.7 - The Distorting Effect of Outliers

**Figure 3.7: How One Outlier Can Ruin Everything**

#### **(a) Scatter Plot**

![Scatter plot with clear outlier](placeholder-outlier-scatter.png)

**What we see:**
- All observations except outlier follow straight-line relationship
- One observation way off the pattern

---

#### **(b) Residual Plot**

![Residual plot showing systematic pattern](placeholder-outlier-residual.png)

**The problem:**
- Fitted regression line pulled toward outlier
- Creates **systematic pattern** in other residuals
- Suggests lack of fit for the linear model
- **But the model IS linear for the other observations!**

**The lesson:** An outlier can make a good model look bad. Always investigate outliers - they might be the problem, not your model!

> **Safe approach:** Only discard an outlier if there's **direct evidence** it represents an error in recording, miscalculation, equipment malfunction, or similar circumstance.

---

## 3.3.4 Nonindependence of Error Terms

### When Does This Occur?

**Common scenarios:**
- **Time series data** - observations collected over time
- **Spatial data** - adjacent geographic areas
- **Repeated measures** - multiple observations from same subject

**The problem:** When errors are correlated, nearby residuals tend to be similar.

---

### How to Detect It

**Best diagnostic:** **Sequence plot of residuals**

Plot residuals against:
- Time order
- Geographic sequence  
- Any other relevant ordering

---

### 📊 Example 3.8 - Welding Strength Over Time

**Context:** Studying relationship between weld diameter ($X$) and shear strength ($Y$)

**Figure 3.8a: Residual Time Sequence Plot - Trend Effect**

![Time sequence showing upward trend](placeholder-time-trend.png)

**What we see:**
- Clear **upward trend** in residuals over time
- Negative residuals in early trials
- Positive residuals in later trials

**Interpretation:**
- **Correlation between error terms is evident**
- Some time-related effect is present (learning by welder? gradual equipment change?)
- Shear strength tended to be greater in later welds

> **The real problem:** An important variable (time) has been **omitted from the model**!

---

**Figure 3.8b: Cyclical Nonindependence**

![Time sequence showing up-down pattern](placeholder-cyclical.png)

**What we see:**
- Residuals follow wave-like pattern
- Adjacent error terms are clearly related
- No overall trend, but strong serial correlation

**Interpretation:** Different type of correlation - cyclical pattern.

---

**Figure 3.2b: Random Pattern (Good!)** ✓

For Toluca Company example:

```
    e |  •    •        •
      |    •    •  •    
    0 |__•___•__•___•___
      | •       •    •
      |_________________ Run
```

**What we see:**
- Residuals fluctuate randomly
- More or less equal scattering around zero line
- No systematic pattern

**Interpretation:** ✓ No evidence of correlation between error terms

---

### Two Types of Non-Random Patterns

1. **Too much alternation** 
   - Residuals bounce back and forth too regularly
   - Rarely occurs in practice

2. **Too little alternation** (more common)
   - Residuals stay positive for stretches, then negative for stretches
   - Indicates positive correlation between errors

> **Reality check:** In practice, we rarely worry about "too much alternation." The concern is almost always "too little alternation."

---

### Comment: Residual Plot vs. Scatter Plot for Sequence Data

**Figure 3.3b (Transit example):** When residuals are plotted against $X$ but the basic problem is probably **not lack of independence** of error terms but a **poorly fitting regression function**.

The scatter may not appear random in this plot, but that doesn't mean the errors are correlated - it means the linear function is wrong!

---

## 3.3.5 Nonnormality of Error Terms

### Why Check Normality?

**Good news:** Small departures from normality don't create serious problems because:
1. **Confidence intervals and tests are robust** (still work reasonably well)
2. **Central Limit Theorem** helps - sampling distributions of $b_0$ and $b_1$ approximately normal even when errors aren't perfectly normal
3. With large samples, can use $z$ values instead of $t$ values

**When to worry:** **Major departures** from normality should be of concern.

---

### How to Check Normality

Three complementary approaches:

1. **Distribution plots** (informal)
2. **Comparison of frequencies** (semi-formal)
3. **Normal probability plot** (formal-ish)

---

### Method 1: Distribution Plots

**Available tools:**
- **Box plot** - shows symmetry and outliers
- **Histogram** - shows shape of distribution
- **Dot plot** - preserves individual values
- **Stem-and-leaf plot** - combines features of histogram and dot plot

**Figure 3.2c: Box Plot of Residuals** (Toluca Company)

![Box plot showing symmetric distribution](placeholder-boxplot-residuals.png)

**What we see:**
- No serious departures from symmetry
- No extreme outliers

**Conclusion:** ✓ Distribution appears reasonably normal

> **Caveat:** For distribution plots to convey reliable information about error distribution shape, **sample size must be reasonably large**. With small samples, these plots can be misleading.

---

### Method 2: Comparison of Frequencies

**The idea:** Compare actual frequencies of residuals in different ranges against expected frequencies under normality.

**Procedure:**
1. Determine expected frequencies under normality
2. Compare to actual observed frequencies
3. Large samples: Can use $t$ distribution comparison

---

### 📊 Example - Toluca Company Frequency Comparison

**From Table 3.2:** Let's check if about 90% of residuals fall within $\pm 1.645\sqrt{MSE}$

**Calculations:**
- $\sqrt{MSE} = 48.82$
- Range: $\pm 1.714(48.82) = \pm 83.68$ to approximately $\pm 83.68$

**Results:**
- Expected under normality: ~90% (22 out of 25 residuals)
- Actual: 88% (22 residuals)

**Similarly:**
- Expected between $-41.89$ and $41.89$: ~60%
- Actual: 52% (13 residuals)

**Interpretation:** Actual frequencies reasonably consistent with those expected under normality ✓

---

### Method 3: Normal Probability Plot (Most Important!)

**What it does:** Plots each residual against its **expected value under normality**

**Key principle:**
- If error distribution is normal → points fall on or near straight line
- Departures from linearity → suggest error distribution is not normal

---

#### How to Construct Normal Probability Plot

**Step 1:** Order the residuals from smallest to largest

Let $e_{(k)}$ denote the $k$-th smallest residual (rank $k$)

**Step 2:** Calculate expected value of the $k$-th smallest observation in a sample of size $n$ from $N(0, \sqrt{MSE})$

The expected value is approximately:

$$\sqrt{MSE} \cdot z\left(\frac{k - 0.375}{n + 0.25}\right) \quad \text{(3.6)}$$

Where $z(A)$ is the $A \times 100$ percentile of standard normal distribution.

---

### 📊 Example 3.2 - Toluca Company Normal Probability Plot

**Table 3.2: Residuals and Expected Values**

| Run $i$ | Residual $e_i$ | Rank $k$ | Expected Value |
|---------|---------------|----------|----------------|
| 1 | 51.02 | 22 | 51.95 |
| 2 | -48.47 | 5 | -44.10 |
| 3 | -19.88 | 10 | -14.76 |
| ... | ... | ... | ... |
| 23 | 38.83 | 19 | 31.05 |
| 24 | -5.98 | 13 | 0 |
| 25 | 10.72 | 17 | 19.93 |

**Example calculation for Run 1** ($e_1 = 51.02$, rank $k = 22$):

$$\frac{k - 0.375}{n + 0.25} = \frac{22 - 0.375}{25 + 0.25} = \frac{21.625}{25.25} = 0.8564$$

Expected value: 
$$\sqrt{2,384} \cdot z(0.8564) = \sqrt{2,384}(1.064) = 51.95$$

---

**Figure 3.2d: Normal Probability Plot**

![Normal probability plot showing points near straight line](placeholder-normal-prob.png)

**What we see:**
- Points fall **reasonably close to a straight line**
- No major S-curves or severe departures

**Interpretation:** ✓ Distribution of error terms does **not depart substantially** from normal distribution

---

### 🚨 What Do Departures Look Like?

**Figure 3.9: Normal Probability Plots When Error Distribution Is NOT Normal**

---

#### **(a) Skewed Right** (Heavy right tail)

```
      |              •••
      |           •••
 Res. |        •••
      |      ••
      |    ••
      |  ••
      |••
      |______________ Expected
```

**Characteristics:**
- Concave-upward shape
- Points curve above the line in upper tail

**Common causes:**
- Exponential-type distributions
- Count data without transformation

---

#### **(b) Skewed Left** (Heavy left tail)

```
      |••
      |  ••
 Res. |    ••
      |      •••
      |        •••
      |           •••
      |              •••
      |______________ Expected
```

**Characteristics:**
- Concave-downward shape  
- Pattern opposite of (a)

---

#### **(c) Symmetrical with Heavy Tails**

```
      |               •
      |             ••
 Res. |           •
      |         ••••••
      |       ••
      |     •
      |   •
      |  •
      | •
      |______________ Expected
```

**Characteristics:**
- S-shaped curve
- Concave downward at left end
- Linear in middle
- Concave upward at right end
- More extreme values than normal distribution would predict

**Common cause:** Contaminated normal distribution (outliers present)

---

### Comments on Normal Probability Plots

1. **Software variations:** 
   - Some packages use semistudentized residuals
   - Some omit the $\sqrt{MSE}$ factor in (3.6)
   - These variations don't affect the nature of the plot - it's the **pattern** that matters, not the exact values!

2. **Ties in continuous data:**
   - Should occur rarely
   - If two residuals tied, use **average rank** for both
   - Example: If ranks 1 and 2 are tied, use rank $(1+2)/2 = 1.5$ for both

---

### Difficulties in Assessing Normality

**The reality:** Analyzing model departures with respect to normality is, in many respects, **more difficult than for other types of departures**.

**Why?**

1. **Random variation can be mischievous**
   - Particularly when sample size is small
   - Probability distributions can be peculiar
   - Even with large samples, other types of departures can affect residual distribution

2. **Other departures affect residual distribution**
   - Inappropriate regression function → residuals may appear not normally distributed
   - Nonconstant error variance → distribution of residuals affected

3. **Residuals may appear non-normal when regression function used is inappropriate** (or error variance not constant)

> **Strategic approach:** It's usually a good strategy to investigate other types of departures first, before concerning yourself with normality of error terms. Fix function and variance issues first, then check normality!

---

## 3.3.6 Omission of Important Predictor Variables

### The Detective Work

**Question:** Should additional variables be included in the model?

**How to investigate:** Plot residuals against **potential predictor variables** not currently in the model.

**What to look for:**
- If residuals show **systematic pattern** against omitted variable → that variable should be included
- If residuals show **no pattern** → variable probably not needed

---

### 📊 Example 3.10 - Productivity Study

**Context:** Regression of output ($Y$) on age of worker ($X$) in piece-rate assembly operation

**Figure 3.10a: Residual Plot - Both Machine Types Combined**

![Residual plot showing no clear pattern](placeholder-both-machines.png)

**What we see:** 
- No clear indication of problems
- Seems randomly scattered

**But wait...** 🤔

---

#### Separating by Machine Type Reveals Hidden Pattern!

**Figure 3.10b: Company A Machines Only**

![Residuals for Company A machines](placeholder-company-A.png)

**What we see:** Residuals tend to be **positive**

---

**Figure 3.10c: Company B Machines Only**

![Residuals for Company B machines](placeholder-company-B.png)

**What we see:** Residuals tend to be **negative**

---

**Interpretation:**
- Type of machine appears to have **definite effect on output**
- When properly included in model, predictions will be far superior!
- This is a **qualitative variable** - we'll learn how to handle these in later chapters

> **Key lesson:** The purpose of residual analysis to identify other important predictor variables is to test the **adequacy of the model** and see whether it could be improved materially by adding one or more predictor variables.

---

### Important Note on Qualitative Variables

While this second example dealt with a **qualitative variable** (type of machine), the residual analysis for an additional **quantitative variable** is analogous:
- Plot residuals against the additional variable
- Look for systematic patterns
- If pattern exists → include the variable

---

## 3.3.7 Some Final Comments on Residual Analysis

### Reality Check #1: Multiple Departures Happen

We discussed model departures **one at a time**, but in reality:
- **Several types of departures may occur together**
- Example: Linear function may be poor fit AND error variance may not be constant
- The **prototype patterns** in Figure 3.4 can still be useful
- They would need to be combined into **composite patterns**

---

### Reality Check #2: Graphical Analysis Is Subjective

**The truth:** Although graphic analysis of residuals is **inherently subjective**, it's incredibly valuable!

**Why it's still worth it:**
- Subjective analysis of interrelated residual plots **frequently reveals difficulties** more clearly than formal tests
- Visual patterns can suggest solutions

> **Bottom line:** In many cases, informal graphic analysis suffices for examining the aptness of a model.

---

### Reality Check #3: Principles Apply Beyond Simple Linear Regression

The **basic approach to residual analysis** explained here applies not only to simple linear regression but also to:
- More complex regression models (Chapters 6+)
- Other types of statistical models throughout the book

---

### Reality Check #4: Methods Keep Improving

Several types of departures from simple linear regression model (2.1) have been identified by diagnostic tests of residuals:

1. **Nonlinearity or omission of predictor variables** → Serious, leading to biased estimates of regression parameters and error variance
   - Discussed further: Section 3.9, Chapter 10

2. **Nonconstancy of error variance** → Less serious, leading to less efficient estimates and invalid error variance estimates
   - Problem discussed in depth: Section 11.1

3. **Presence of outliers** → Can be serious for smaller data sets when their influence is large
   - Influential outliers discussed: Section 10.4

4. **Nonindependence of error terms** → Results in estimators that are unbiased but whose variances are seriously biased
   - Alternative estimation methods: Chapter 12

5. **Nonnormality of error terms** → Can lead to serious inefficiencies
   - Discussed further: Section 3.9, Chapter 11
   - Problem is discussed in depth: Chapter 11

---

## 3.4 Overview of Formal Tests Involving Residuals

**The reality:** Graphic analysis of residuals is inherently subjective.

**However:** Subjective analysis of a variety of interrelated residual plots will frequently reveal difficulties with the model more clearly than particular formal tests!

**When formal tests are useful:** When you want to put specific questions to a test.

**Most statistical tests require independent observations** - but we know residuals are dependent! Fortunately, dependencies become quite small for large samples, so we can usually ignore them.

---

### Four Categories of Formal Tests

#### 1. **Tests for Randomness**

**Purpose:** Test for lack of randomness in residuals arranged in time order

**Example test:** Runs test (discussed in Chapter 12)

**Another test:** Durbin-Watson test (specifically designed for least squares residuals) - Chapter 12

---

#### 2. **Tests for Constancy of Variance**

**Purpose:** Test whether error variance varies systematically with level of $X$ (or $E\{Y\}$)

**Method:** Simple test based on **rank correlation** between:
- Absolute values of residuals
- Corresponding values of predictor variable

**More sophisticated tests:**
- **Brown-Forsythe test** (Section 3.6)
- **Breusch-Pagan test** (Section 3.6)

---

#### 3. **Tests for Outliers**

**Purpose:** Identify outlier observations

**Simple approach:** 
- Fit new regression line to $n-1$ observations (excluding suspect)
- Calculate probability that in $n$ observations, a deviation from fitted line as great as outlier's would be obtained by chance
- If probability sufficiently small → reject outlier
- Otherwise → retain outlier

**Detailed discussion:** Chapter 10

**Advanced methods:** Many other tests to aid in evaluating outliers (Reference 3.1)

---

#### 4. **Tests for Normality**

**Purpose:** Examine normality of error terms

**Example tests:**
- **Chi-square test** (or **Kolmogorov-Smirnov test**)
- **Lilliefors test** (modification for better properties)

**Simple test:** Based on normal probability plot of residuals (Section 3.5)

---

### Important Comment

The runs test, rank correlation, and goodness of fit tests are **commonly used statistical procedures** discussed in many basic statistics texts.

---

## 3.5 Correlation Test for Normality

**The Idea:** In addition to visually assessing the approximate linearity of points in a normal probability plot, we can calculate a **formal test** based on the **coefficient of correlation** between:
- Residuals $e_i$  
- Their expected values under normality

---

### The Test Procedure

**Correlation coefficient:** Measures the linear association between the residuals and their expected values

Formula: (2.74) - we learned this in Chapter 2!

---

### Interpretation

**High correlation value** → indicative of normality

**Critical values:** Table B.6 (prepared by Looney and Gulledge, Ref. 3.2) contains percentiles for various sample sizes when error terms are normally distributed.

---

### Decision Rule

**If observed coefficient ≥ critical value for given $\alpha$ level:**
→ Conclude error terms are reasonably normally distributed

**If observed coefficient < critical value:**
→ Conclude error terms are not normally distributed

---

### 📊 Example 3.5 - Toluca Company Normality Test

**From Table 3.2:** Coefficient of correlation between ordered residuals and expected values under normality = **0.991**

**Test setup:**
- $\alpha = 0.05$ risk level
- $n = 25$ observations

**Critical value from Table B.6:** 0.959

**Decision:**
Since $0.991 > 0.959$, we conclude that distribution of error terms does **not depart substantially** from a normal distribution ✓

---

### Comment

**Comparison to Shapiro-Wilk test:**

The correlation test for normality presented here is **simpler** than the Shapiro-Wilk test (Ref. 3.3).

The Shapiro-Wilk test can be viewed as based approximately also on the coefficient of correlation between ordered residuals and their expected values under normality.

---

## 3.6 Tests for Constancy of Error Variance

We present two formal tests for ascertaining whether the error terms have constant variance:
1. **Brown-Forsythe test**
2. **Breusch-Pagan test**

---

## 3.6.1 Brown-Forsythe Test

### Key Features

**Robustness:** The Brown-Forsythe test (modification of Levene test, Ref. 3.4) **does NOT depend on normality** of error terms!

**When it's relatively efficient:** Even when error terms have equal variances but distribution is far from normal

**What it tests:** Whether error variance either increases or decreases with $X$, as illustrated in the prototype megaphone plot (Figure 3.4c)

**Sample size requirement:** Must be **large enough** so dependencies among residuals can be ignored

---

### The Test Logic

**Basic idea:** Error variance is related to **variability of residuals**

If error variance is **larger**, residuals will tend to be **more variable** - larger variability means residuals further from their group mean.

---

### Test Procedure

**Step 1: Divide into Two Groups**

Divide data into two groups according to level of $X$:
- **Group 1:** Cases where $X$ level comparatively low
- **Group 2:** Cases where $X$ level comparatively high

Let:
- $n_1$ = sample size of group 1
- $n_2$ = sample size of group 2

$$n = n_1 + n_2 \quad \text{(3.7)}$$

---

**Step 2: Calculate Medians**

Let $\tilde{e}_1$ and $\tilde{e}_2$ denote the **medians of residuals** in the two groups

---

**Step 3: Calculate Absolute Deviations**

For each group, calculate **absolute deviations of residuals around their group median**:

$$d_{i1} = |e_{i1} - \tilde{e}_1|$$
$$d_{i2} = |e_{i2} - \tilde{e}_2|$$ 

(3.8)

---

**Step 4: Two-Sample $t$ Test**

The test statistic is the **two-sample $t$ test statistic** (A.67):

$$t_{BF}^* = \frac{\bar{d}_1 - \bar{d}_2}{s\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \quad \text{(3.9)}$$

Where:
- $\bar{d}_1$ and $\bar{d}_2$ = sample means of the $d_{i1}$ and $d_{i2}$ respectively
- $s^2$ = pooled variance (A.63):

$$s^2 = \frac{\sum(d_{i1} - \bar{d}_1)^2 + \sum(d_{i2} - \bar{d}_2)^2}{n-2} \quad \text{(3.9a)}$$

---

### Decision Rule

**If error terms have constant variance** and $n_1$ and $n_2$ are not extremely small:

$t_{BF}^*$ follows approximately the $t$ distribution with $n-2$ degrees of freedom

**Large absolute values of $t_{BF}^*$** indicate error terms do NOT have constant variance

---

### 📊 Example 3.6 - Toluca Company Variance Constancy Test

**Setup:** Test whether error term variance varies with level of $X$ (lot size)

**Step 1: Divide into groups**

Since $X$ levels spread fairly uniformly (Figure 3.1a), divide 25 cases into two groups with approximately equal $X$ ranges:

- **Group 1:** 13 runs with lot sizes 20-70
- **Group 2:** 12 runs with lot sizes 80-120

---

**Table 3.3: Brown-Forsythe Calculations**

#### **Group 1** (Lot sizes 20-70)

| $i$ | Run | Lot Size | Residual $e_{i1}$ | $d_{i1}$ | $(d_{i1} - \bar{d}_1)^2$ |
|-----|-----|----------|-------------------|----------|--------------------------|
| 1 | 14 | 20 | -20.77 | 0.89 | 1,929.41 |
| 2 | 2 | 30 | -48.47 | 28.59 | 263.25 |
| ... | ... | ... | ... | ... | ... |
| 12 | 12 | 70 | -60.28 | 40.40 | 19.49 |
| 13 | 25 | 70 | 10.72 | 30.60 | 202.07 |
| **Total** | | | | 582.60 | 12,566.6 |

$$\tilde{e}_1 = -19.88 \quad \bar{d}_1 = 44.815$$

---

#### **Group 2** (Lot sizes 80-120)

| $i$ | Run | Lot Size | Residual $e_{i2}$ | $d_{i2}$ | $(d_{i2} - \bar{d}_2)^2$ |
|-----|-----|----------|-------------------|----------|--------------------------|
| 1 | 1 | 80 | 51.02 | 53.70 | 637.56 |
| 2 | 8 | 80 | 4.02 | 6.70 | 473.06 |
| ... | ... | ... | ... | ... | ... |
| 11 | 20 | 110 | -34.09 | 31.41 | 8.76 |
| 12 | 7 | 120 | 55.21 | 57.89 | 866.71 |
| **Total** | | | | 341.40 | 9,610.2 |

$$\tilde{e}_2 = -2.68 \quad \bar{d}_2 = 28.450$$

---

**Step 2: Calculate sample means of absolute deviations**

$$\bar{d}_1 = \frac{582.60}{13} = 44.815$$

$$\bar{d}_2 = \frac{341.40}{12} = 28.450$$

---

**Step 3: Calculate pooled variance**

$$s^2 = \frac{12,566.6 + 9,610.2}{25-2} = \frac{22,176.8}{23} = 964.21$$

$$s = 31.05$$

---

**Step 4: Calculate test statistic**

$$t_{BF}^* = \frac{44.815 - 28.450}{31.05\sqrt{\frac{1}{13} + \frac{1}{12}}} = \frac{16.365}{31.05 \times 0.3869} = 1.32$$

---

**Step 5: Decision**

Control $\alpha$ risk at 0.05, require $t(0.975; 23) = 2.069$

**Decision rule:**
- If $|t_{BF}^*| \leq 2.069$ → conclude error variance is constant
- If $|t_{BF}^*| > 2.069$ → conclude error variance is not constant

**Since $|t_{BF}^*| = 1.32 \leq 2.069$:**

We conclude error variance is **constant** and does not vary with level of $X$ ✓

**Two-sided $P$-value:** 0.20

---

### Comments on Brown-Forsythe Test

**1. Multiple groups possible**

If data set contains many cases, two-sample $t$ test for constancy of error variance can be conducted after dividing cases into **three or four groups** according to level of $X$, using the **two extreme groups**.

---

**2. Robustness to nonnormality is valuable**

A robust test for constancy of error variance is desirable because:
- **Nonnormality** and **lack of constant variance** often go hand in hand
- Distribution of error terms may become increasingly skewed and more variable with increasing levels of $X$

---

## 3.6.2 Breusch-Pagan Test

### Key Features

**Large-sample test:** Assumes:
- Error terms are **independent**
- **Normally distributed**
- Variance of error term $\varepsilon_i$, denoted by $\sigma_i^2$, is related to level of $X$ in the following way:

$$\log_e \sigma_i^2 = \gamma_0 + \gamma_1 X_i \quad \text{(3.10)}$$

**Note:** (3.10) implies $\sigma_i^2$ either **increases or decreases** with level of $X$, depending on sign of $\gamma_1$

**Test hypotheses:**
- **Constancy of error variance** corresponds to $\gamma_1 = 0$
- Test of $H_0: \gamma_1 = 0$ vs. $H_a: \gamma_1 \neq 0$

---

### Test Procedure

**Test is carried out by:**
1. Regressing **squared residuals** $e_i^2$ against $X_i$ in usual manner
2. Obtaining regression sum of squares (denoted by $SSR^*$)

---

**Test statistic $X_{BP}^2$:**

$$X_{BP}^2 = \frac{SSR^*}{2} \div \left(\frac{SSE}{n}\right)^2 \quad \text{(3.11)}$$

Where:
- $SSR^*$ = regression sum of squares when regressing $e^2$ on $X$
- $SSE$ = error sum of squares when regressing $Y$ on $X$

---

**Distribution:** If $H_0: \gamma_1 = 0$ holds and $n$ is reasonably large, $X_{BP}^2$ follows approximately the **chi-square distribution with one degree of freedom**

**Large values of $X_{BP}^2$** lead to conclusion $H_a$, that error variance is not constant

---

### 📊 Example 3.6 (continued) - Toluca Company Breusch-Pagan Test

**Setup:** Conduct Breusch-Pagan test for Toluca Company example

**Step 1: Regress squared residuals**

Regress squared residuals in Table 1.2, column 5, against $X$ 

Result: $SSR^* = 7,896,128$

From Figure 2.2: $SSE = 54,825$

---

**Step 2: Calculate test statistic**

$$X_{BP}^2 = \frac{7,896,128}{2} \div \left(\frac{54,825}{25}\right)^2 = \frac{3,948,064}{4,804,756.25} = 0.821$$

---

**Step 3: Decision**

Control $\alpha$ risk at 0.05, require $\chi^2(0.95; 1) = 3.84$

**Since $X_{BP}^2 = 0.821 \leq 3.84$:**

We conclude $H_0$, that error variance is **constant** ✓

**$P$-value:** 0.64

Data are quite consistent with constancy of error variance.

---

### Comments on Breusch-Pagan Test

**1. Modification possible**

Breusch-Pagan test can be modified to allow for different relationships between error variance and level of $X$ than the one in (3.10).

---

**2. Alternative reference**

Test statistic (3.11) was developed independently by Cook and Weisberg (Ref. 3.7), and test is sometimes referred to as the **Cook-Weisberg test**.

---

## 3.7 $F$ Test for Lack of Fit

**Purpose:** Formal test for determining whether a specific type of regression function adequately fits the data

**This section:** We illustrate this test for ascertaining whether a **linear regression function** is a good fit for the data

---

## 3.7.1 Assumptions

The lack of fit test assumes:
1. Observations $Y$ for given $X$ are **independent**
2. Observations $Y$ are **normally distributed**  
3. Distributions of $Y$ have the **same variance $\sigma^2$**

---

### Repeat Observations Required

The lack of fit test requires **repeat observations at one or more $X$ levels**.

**In nonexperimental data:** May occur fortuitously (by chance), as when:
- In productivity study relating workers' output and age
- Several workers of same age happen to be included in study

**In experiment:** Can ensure repeat observations by design
- Assure there are repeat observations by including them in experimental design

---

### 📊 Example 3.7 - Bank Experiment on New Accounts

**Context:** Experiment involving 12 similar suburban branch offices of commercial bank

**Study variables:**
- $Y$ = Number of new money market accounts opened during test period
- $X$ = Size of minimum deposit (dollars)

**Design features:**
- Holders of checking accounts at offices offered gifts for setting up new money market accounts
- Minimum initial deposits in new money market account specified
- Value of gift directly proportional to specified minimum deposit
- Various levels of minimum deposit and related gift values used

**Experimental structure:**
- Six levels of minimum deposit
- Two branch offices assigned at random to each level (**replications**)
- One branch office had fire during period and was dropped

**Result:** 11 observations total (Table 3.4a)

---

**Table 3.4a: Data**

| Branch $i$ | Size of Minimum Deposit (dollars) $X_i$ | Number of New Accounts $Y_i$ |
|------------|----------------------------------------|------------------------------|
| 1 | 125 | 160 |
| 2 | 100 | 112 |
| 3 | 200 | 124 |
| 4 | 75 | 28 |
| 5 | 150 | 152 |
| 6 | 175 | 156 |
| 7 | 75 | 42 |
| 8 | 175 | 124 |
| 9 | 125 | 150 |
| 10 | 200 | 104 |
| 11 | 100 | 136 |

---

### Notation for Replications

**Modified notation to recognize replications:**

Let:
- $c$ = number of distinct $X$ levels in study (for bank example, $c = 6$)
- $X_1, \ldots, X_c$ denote the different $X$ levels

For bank example:
- $X_1 = 75$ (smallest minimum deposit level)
- $X_2 = 100$
- ...
- $X_6 = 200$

---

**Number of replications at each $X$ level:**

Denote number of observations for $j$th level of $X$ as $n_j$

For our example:
- $n_1 = n_2 = n_3 = n_5 = n_6 = 2$ (two observations at each level)
- $n_4 = 1$ (single observation)

**Total number of observations:**

$$n = \sum_{j=1}^{c} n_j \quad \text{(3.12)}$$

For bank example: $n = 2 + 2 + 2 + 2 + 2 + 1 = 11$

---

**Notation for individual observations:**

Denote observed value of response variable for:
- $i$th replicate
- $j$th level of $X$

by $Y_{ij}$, where $i = 1, \ldots, n_j$ and $j = 1, \ldots, c$

For bank example (Table 3.5):
- $Y_{11} = 28$ (first observation at $X_1 = 75$)
- $Y_{21} = 42$ (second observation at $X_1 = 75$)
- $Y_{12} = 112$ (first observation at $X_2 = 100$)
- $Y_{42} = 152$ (observation at $X_4 = 150$, single observation so no subscript 1)

**Mean $\bar{Y}_j$:** Mean of $Y$ observations at level $X = X_j$ by $\bar{Y}_j$

Thus: $\bar{Y}_1 = (28 + 42)/2 = 35$ and $\bar{Y}_4 = 152/1 = 152$

---

**Table 3.5: Data Arranged by Replicate Number and Minimum Deposit**

| | **Size of Minimum Deposit (dollars)** | | | | | |
|--|-------|-------|-------|-------|-------|-------|
| **Replicate** | $j=1$ $X_1=75$ | $j=2$ $X_2=100$ | $j=3$ $X_3=125$ | $j=4$ $X_4=150$ | $j=5$ $X_5=175$ | $j=6$ $X_6=200$ |
| $i=1$ | 28 | 112 | 160 | 152 | 156 | 124 |
| $i=2$ | 42 | 136 | 150 | | 124 | 104 |
| **Mean $\bar{Y}_j$** | 35 | 124 | 155 | 152 | 140 | 114 |

---

## 3.7.2 Full Model

The general linear test approach begins with specification of the **full model**.

**Full model** used for lack of fit test makes **same assumptions as simple linear regression model (2.1)** except for assuming a linear regression relation (the subject of the test!)

**Full model:**

$$Y_{ij} = \mu_j + \varepsilon_{ij} \quad \text{Full model} \quad \text{(3.13)}$$

Where:
- $\mu_j$ = parameters ($j = 1, \ldots, c$)
- $\varepsilon_{ij}$ = independent $N(0, \sigma^2)$

---

**Since error terms have expectation zero:**

$$E\{Y_{ij}\} = \mu_j \quad \text{(3.14)}$$

**Interpretation:** Parameter $\mu_j$ ($j = 1, \ldots, c$) is the **mean response when $X = X_j$**

---

**Key difference from regression model (2.1):**

Full model (3.13) is like regression model (2.1) in stating each response $Y$ is made up of two components:
1. Mean response when $X = X_j$
2. Random error term

**The difference:** In full model (3.13) there are **no restrictions** on the means $\mu_j$, whereas in regression model (2.1) the mean responses are **linearly related to $X$** (i.e., $E\{Y\} = \beta_0 + \beta_1 X$).

---

### Fitting the Full Model

To fit full model to data, we require **least squares or maximum likelihood estimators** for parameters $\mu_j$

**It can be shown these estimators are simply the sample means $\bar{Y}_j$:**

$$\hat{\mu}_j = \bar{Y}_j \quad \text{(3.15)}$$

**Estimated expected value:** For observation $Y_{ij}$ is $\bar{Y}_j$

---

**Error sum of squares for full model:**

$$SSE(F) = \sum_j \sum_i (Y_{ij} - \bar{Y}_j)^2 = SSPE \quad \text{(3.16)}$$

**Context:** In test for lack of fit, full model error sum of squares (3.16) is called the **pure error sum of squares** and denoted by $SSPE$

---

**Note about $SSPE$:**

$SSPE$ is made up of sums of squared deviations at each $X$ level

At level $X = X_j$, this sum of squared deviations is:

$$\sum_i (Y_{ij} - \bar{Y}_j)^2 \quad \text{(3.17)}$$

These sums are then added over all $X$ levels ($j = 1, \ldots, c$)

---

**For bank example:**

$$SSPE = (28-35)^2 + (42-35)^2 + (112-124)^2 + (136-124)^2 + (160-155)^2$$
$$+ (150-155)^2 + (152-152)^2 + (156-140)^2 + (124-140)^2$$
$$+ (124-114)^2 + (104-114)^2$$
$$= 1,148$$

---

**Note:** Any $X$ level with **no replications** makes **no contribution** to $SSPE$ because $n_j - 1 = 1 - 1 = 0$ then

For bank example: $(152 - 152)^2 = 0$ for $j = 4$

---

**Degrees of freedom associated with $SSPE$:**

Recognizing sum of squared deviations (3.17) at given level of $X$ is like an ordinary total sum of squares based on $n_j$ observations, which has $n_j - 1$ degrees of freedom associated with it:

Here, there are $n_j$ observations when $X = X_j$; hence degrees of freedom are $n_j - 1$

Just as $SSPE$ is sum of sums of squares (3.17), so number of degrees of freedom associated with $SSPE$ is sum of component degrees of freedom:

$$df_F = \sum_j (n_j - 1) = \sum_j n_j - c = n - c \quad \text{(3.18)}$$

For bank example: $df_F = 11 - 6 = 5$

**Note:** Any $X$ level with no replications makes **no contribution** to $df_F$ because $n_j - 1 = 1 - 1 = 0$ then (just as such an $X$ level makes no contribution to $SSPE$)

---

## 3.7.3 Reduced Model

General linear test approach next requires consideration of **reduced model under $H_0$**

For testing appropriateness of linear regression relation, alternatives are:

$$H_0: E\{Y\} = \beta_0 + \beta_1 X$$
$$H_a: E\{Y\} \neq \beta_0 + \beta_1 X \quad \text{(3.19)}$$

**Thus $H_0$ postulates:** $\mu_j$ in full model (3.13) is **linearly related to $X_j$**:

$$\mu_j = \beta_0 + \beta_1 X_j$$

---

**Reduced model under $H_0$:**

$$Y_{ij} = \beta_0 + \beta_1 X_j + \varepsilon_{ij} \quad \text{Reduced model} \quad \text{(3.20)}$$

**Note:** Reduced model is **ordinary simple linear regression model (2.1)**, with subscripts modified to recognize existence of replications

---

**We know estimated expected value for observation $Y_{ij}$ with regression model (2.1) is fitted value $\hat{Y}_{ij}$:**

$$\hat{Y}_{ij} = b_0 + b_1 X_j \quad \text{(3.21)}$$

**Hence error sum of squares for reduced model is usual error sum of squares $SSE$:**

$$SSE(R) = \sum_j \sum_i [Y_{ij} - (b_0 + b_1 X_j)]^2$$
$$= \sum_j \sum_i (Y_{ij} - \hat{Y}_{ij})^2 = SSE \quad \text{(3.22)}$$

---

**Degrees of freedom associated with $SSE(R)$:**

$$df_R = n - 2$$

For bank example (Table 3.4b):

$$SSE(R) = SSE = 14,741.6$$
$$df_R = 9$$

---

## 3.7.4 Test Statistic

The general linear test statistic (2.70):

$$F^* = \frac{SSE(R) - SSE(F)}{df_R - df_F} \div \frac{SSE(F)}{df_F}$$

becomes here:

$$F^* = \frac{SSE - SSPE}{(n-2) - (n-c)} \div \frac{SSPE}{n-c} \quad \text{(3.23)}$$

---

**The difference between two error sums of squares is called the **lack of fit sum of squares** here and denoted by $SSLF$:**

$$SSLF = SSE - SSPE \quad \text{(3.24)}$$

**We can then express test statistic as:**

$$F^* = \frac{SSLF}{c-2} \div \frac{SSPE}{n-c}$$
$$= \frac{MSLF}{MSPE} \quad \text{(3.25)}$$

Where:
- $MSLF$ denotes the **lack of fit mean square**
- $MSPE$ denotes the **pure error mean square**

---

**We know large values of $F^*$ lead to conclusion $H_a$ in general linear test.**

**Decision rule (2.71) here becomes:**

$$\text{If } F^* \leq F(1-\alpha; c-2, n-c), \text{ conclude } H_0$$
$$\text{If } F^* > F(1-\alpha; c-2, n-c), \text{ conclude } H_a \quad \text{(3.26)}$$

---

### 📊 Example 3.7 (continued) - Bank Example Test Statistic

For bank example, test statistic can be constructed easily from our earlier results:

$$SSPE = 1,148.0 \qquad n - c = 11 - 6 = 5$$
$$SSE = 14,741.6$$
$$SSLF = 14,741.6 - 1,148.0 = 13,593.6 \qquad c - 2 = 6 - 2 = 4$$

$$F^* = \frac{13,593.6}{4} \div \frac{1,148.0}{5} = \frac{3,398.4}{229.6} = 14.80$$

---

**Decision:** If level of significance is to be $\alpha = 0.01$, require $F(0.99; 4, 5) = 11.4$

Since $F^* = 14.80 > 11.4$, we conclude $H_a$, that regression function is **not linear** ✓

This, of course, accords with our visual impression from Figure 3.11

**$P$-value:** 0.006

---

## 3.7.5 ANOVA Table

**Definition of lack of fit sum of squares $SSLF$ in (3.24) indicates we have decomposed error sum of squares $SSE$ into two components:**

$$SSE = SSPE + SSLF \quad \text{(3.27)}$$

---

**This decomposition follows from identity:**

$$Y_{ij} - \hat{Y}_{ij} = Y_{ij} - \bar{Y}_j + \bar{Y}_j - \hat{Y}_{ij}$$

$$\underbrace{Y_{ij} - \hat{Y}_{ij}}_{\text{Error deviation}} = \underbrace{Y_{ij} - \bar{Y}_j}_{\text{Pure error deviation}} + \underbrace{\bar{Y}_j - \hat{Y}_{ij}}_{\text{Lack of fit deviation}} \quad \text{(3.28)}$$

---

**Identity shows error deviations in $SSE$ are made up of:**
1. **Pure error component** 
2. **Lack of fit component**

**Figure 3.12** illustrates this partitioning for case $Y_{13} = 160$, $X_3 = 125$ in bank example:

![Decomposition diagram](placeholder-decomposition.png)

---

**An ANOVA table can be constructed for decomposition of $SSE$:**

**Table 3.6a contains general ANOVA table**, including decomposition of $SSE$ just explained and mean squares of interest

**Table 3.6b contains ANOVA decomposition for bank example**

---

**Table 3.6a: General ANOVA Table for Testing Lack of Fit**

| Source of Variation | SS | df | MS |
|---------------------|----|----|-----|
| Regression | $SSR = \sum\sum(\hat{Y}_{ij} - \bar{Y})^2$ | 1 | $MSR = \frac{SSR}{1}$ |
| Error | $SSE = \sum\sum(Y_{ij} - \hat{Y}_{ij})^2$ | $n-2$ | $MSE = \frac{SSE}{n-2}$ |
| Lack of fit | $SSLF = \sum\sum(\bar{Y}_j - \hat{Y}_{ij})^2$ | $c-2$ | $MSLF = \frac{SSLF}{c-2}$ |
| Pure error | $SSPE = \sum\sum(Y_{ij} - \bar{Y}_j)^2$ | $n-c$ | $MSPE = \frac{SSPE}{n-c}$ |
| Total | $SSTO = \sum\sum(Y_{ij} - \bar{Y})^2$ | $n-1$ | |

---

**Table 3.6b: ANOVA Table - Bank Example**

| Source of Variation | SS | df | MS | F Ratio |
|---------------------|----|----|-----|---------|
| Regression | 5,141.3 | 1 | 5,141.3 | 3.14 |
| Error | 14,741.6 | 9 | 1,638.0 | **Prob>F** |
| Lack of fit | 13,593.6 | 4 | 3,398.4 | 14.80 |
| Pure error | 1,148.0 | 5 | 229.6 | **0.006** |
| Total | 19,882.9 | 10 | | |

---

**Note from (3.29):** We can define lack of fit sum of squares directly as:

$$SSLF = \sum_j \sum_i (\bar{Y}_j - \hat{Y}_{ij})^2 \quad \text{(3.30)}$$

**Since all $Y_{ij}$ observations at level $X_j$ have same fitted value** (which we can denote by $\hat{Y}_j$), we can express (3.30) equivalently as:

$$SSLF = \sum_j n_j(\bar{Y}_j - \hat{Y}_j)^2 \quad \text{(3.30a)}$$

---

**Formula (3.30a) indicates clearly why $SSLF$ measures lack of fit:**

If linear regression function is appropriate, means $\bar{Y}_j$ will be near fitted values $\hat{Y}_j$ calculated from estimated linear regression function and $SSLF$ will be small

On other hand, if linear regression function is not appropriate, means $\bar{Y}_j$ will not be near fitted values calculated from estimated linear regression function (as in Figure 3.11 for bank example) and $SSLF$ will be large

---

**Formula (3.30a) also indicates why $c - 2$ degrees of freedom are associated with $SSLF$:**

There are $c$ means $\bar{Y}_j$ in sum of squares, and **two degrees of freedom are lost** in estimating parameters $\beta_0$ and $\beta_1$ of linear regression function to obtain fitted values $\hat{Y}_j$

---

**An ANOVA table can be constructed for decomposition of $SSE$. Table 3.6a contains general ANOVA table** including decomposition of $SSE$ just explained and mean squares of interest, and **Table 3.6b contains ANOVA decomposition for bank example**.

---

### Comments on Lack of Fit Test

**1. Not all levels of $X$ need have repeat observations** for $F$ test for lack of fit to be applicable. **Repeat observations at only one or some levels of $X$ are sufficient.**

---

**2. It can be shown mean squares $MSPE$ and $MSLF$ have following expectations when testing whether regression function is linear:**

$$E\{MSPE\} = \sigma^2 \quad \text{(3.31)}$$

$$E\{MSLF\} = \sigma^2 + \frac{\sum n_j[\mu_j - (\beta_0 + \beta_1 X_j)]^2}{c-2} \quad \text{(3.32)}$$

---

**The reason for term "pure error":**

$MSPE$ is **always an unbiased estimator** of error term variance $\sigma^2$, no matter what is true regression function

The expected value of $MSLF$ also is $\sigma^2$ if regression function is linear, because $\mu_j = \beta_0 + \beta_1 X_j$ then and second term in (3.32) becomes zero

On other hand, if regression function is not linear, $\mu_j \neq \beta_0 + \beta_1 X_j$ and $E\{MSLF\}$ will be greater than $\sigma^2$

Hence, value of $F^*$ near 1 accords with linear regression function; large values of $F^*$ indicate regression function is not linear

---

**3. Terminology "error sum of squares" and "error mean square" is not precise when:**
- Regression function under test in $H_0$ is not true function
- Error sum of squares and error mean square then reflect effects of **both lack of fit and variability of error terms**

We continue to use terminology for consistency and now use term **"pure error"** to identify variability associated with error term only

---

**4. Suppose prior to any analysis of appropriateness of model, we had fitted:**
- Linear regression model
- Wished to test whether or not $\beta_1 = 0$ for bank example (Table 3.4b)

**Test statistic (2.60) would be:**

$$F^* = \frac{MSR}{MSE} = \frac{5,141.3}{1,638.0} = 3.14$$

For $\alpha = 0.10$, $F(0.90; 1, 9) = 3.36$, and we would conclude $H_0$, that $\beta_1 = 0$ or that there is **no linear association** between minimum deposit size (and value of gift) and number of new accounts

**Conclusion there is no relation** between these variables would be **improper**, however

**Such inference requires regression model (2.1) be appropriate**

Here, there is **definite relationship**, but regression function is **not linear**

This illustrates **importance of always examining appropriateness of model** before inferences are drawn

---

**5. General linear test approach just explained can be used to test appropriateness of other regression functions.**

Only degrees of freedom for $SSLF$ will need be modified

In general, $c - p$ degrees of freedom are associated with $SSLF$, where $p$ is number of parameters in regression function

For test of simple linear regression function, $p = 2$ because there are two parameters, $\beta_0$ and $\beta_1$, in regression function

---

**6. Alternative $H_a$ in (3.19) includes:**
- All regression functions other than linear one
- For instance, includes:
  - Quadratic regression function
  - Logarithmic one

If $H_a$ is concluded, study of residuals can be helpful in identifying appropriate function

---

**7. When we conclude employed model in $H_0$ is appropriate, usual practice is to use error mean square $MSE$ as estimator of $\sigma^2$ in preference to pure error mean square $MSPE$, since former contains more degrees of freedom**

---

**8. Observations at same level of $X$ are genuine repeats only if they involve independent trials with respect to error term.**

Suppose in regression analysis of relation between hardness ($Y$) and amount of carbon ($X$) in specimens of an alloy:
- Error term in model covers, among other things, random errors in measurement of hardness by analyst
- Effects of controlled production factors, which vary at random from specimen to specimen and affect hardness

If analyst takes two readings on hardness of specimen:
- This will **not provide genuine replication** because effects of random variation in controlled production factors are fixed in any given specimen
- For genuine replications, different specimens with same carbon content ($X$) would have to be measured by analyst so that *all* effects covered in error term could vary at random from one repeated observation to next

---

**9. When no replications are present in data set, approximate test for lack of fit can be conducted if:**
- There are some cases at adjacent $X$ levels for which mean responses are quite close to each other

Such adjacent cases are grouped together and treated as pseudoreplicates, and test for lack of fit is then carried out using these groupings of adjacent cases

**Useful summary of this and related procedures for conducting test for lack of fit when replications are not present may be found in Reference 3.8**

---

## 3.8 Overview of Remedial Measures

If simple linear regression model (2.1) is not appropriate for data set, there are **two basic choices:**

1. **Abandon** regression model (2.1) and develop and use a more appropriate model
2. **Employ some transformation** on data so that regression model (2.1) is appropriate for transformed data

---

**Each approach has advantages and disadvantages:**

**First approach** may entail more complex model that could yield better insights, but may also lead to more complex procedures for estimating parameters

**Successful use of transformations**, on other hand, leads to relatively simple methods of estimation and may involve fewer parameters than complex model

**Advantage when sample size is small**

**Yet transformations may obscure fundamental interconnections between variables**, though at other times they may illuminate them

---

**We consider use of transformations in this chapter and use of more complex models in later chapters.**

First, we provide brief overview of remedial measures:

---

### Nonlinearity of Regression Function

When regression function is not linear, direct approach is to **modify regression model (2.1) by altering nature of regression function**

For instance, quadratic regression function might be used:

$$E\{Y\} = \beta_0 + \beta_1 X + \beta_2 X^2$$

or exponential regression function:

$$E\{Y\} = \beta_0 \beta_1^X$$

**In Chapter 7:** We discuss polynomial regression functions

**In Part III:** We take up nonlinear regression functions, such as exponential regression function

---

**Transformation approach** employs transformation to linearize, at least approximately, nonlinear regression function

We discuss use of transformations to linearize regression functions in **Section 3.9**

When nature of regression function is not known, **exploratory analysis** that does not require specifying particular type of function is often useful

We discuss **exploratory regression analysis in Section 3.10**

---

### Nonconstancy of Error Variance

When error variance is not constant but varies in systematic fashion, direct approach is to **modify model to allow for this** and use method of **weighted least squares** to obtain estimators of parameters

We discuss use of weighted least squares for this purpose in **Chapter 11**

---

**Transformations can also be effective in stabilizing variance.** Some of these are discussed in **Section 3.9**

---

### Nonindependence of Error Terms

When error terms are correlated, direct remedial measure is to **work with model that calls for correlated error terms**

We discuss such model in **Chapter 12**

---

**Simple remedial transformation** that is often helpful is to work with **first differences**, topic also discussed in **Chapter 12**

---

### Nonnormality of Error Terms

**Lack of normality** and **nonconstant error variances** frequently go hand in hand

Fortunately, it's often the case that **same transformation that helps stabilize variance is also helpful in approximately normalizing error terms**

**It is therefore desirable transformation for stabilizing error variance be utilized first**, and then residuals studied to see if serious departures from normality are still present

We discuss **transformations to achieve approximate normality in Section 3.9**

---

### Omission of Important Predictor Variables

When residual analysis indicates **important predictor variable has been omitted** from model, solution is to **modify model**

In **Chapter 6 and later chapters**, we discuss **multiple regression analysis** in which two or more predictor variables are utilized

---

### Outlying Observations

When outlying observations are present, as in Figure 3.7a:
- **Use of least squares and maximum likelihood estimators (1.10) for regression model (2.1) may lead to serious distortions in estimated regression function**

When outlying observations do not represent recording errors and should not be discarded:
- May be desirable to use an **estimation procedure** that places less emphasis on such outlying observations

We discuss one such robust estimation procedure in **Chapter 11**

---

## 3.9 Transformations

We now consider in more detail use of **transformations of one or both of original variables** before carrying out regression analysis

**Simple transformations** of either:
- Response variable $Y$
- Predictor variable $X$
- Or both

are often sufficient to make simple linear regression model appropriate for transformed data

---

## 3.9.1 Transformations for Nonlinear Relation Only

We first consider transformations for **linearizing nonlinear regression relation** when:
- Distribution of error terms is reasonably close to normal distribution
- Error terms have approximately constant variance

**In this situation, transformations on $X$ should be attempted**

**Reason why transformations on $Y$ may not be desirable here:** Transformation on $Y$, such as $Y' = \sqrt{Y}$, may materially change shape of distribution of error terms from normal distribution and may also lead to substantially differing error term variances

---

**Figure 3.13 contains:**
- Some prototype nonlinear regression relations with constant error variance
- Also presents some simple transformations on $X$ that may be helpful to linearize regression relationship without affecting distributions of $Y$

**Several alternative transformations may be tried**

**Scatter plots and residual plots based on each transformation should then be prepared and analyzed, to determine which transformation is most effective**

---

### 📊 Example 3.9 - Sales Training Program

**Context:** Data from experiment on effect of number of days of training received ($X$) on performance ($Y$) in battery of simulated sales situations

**Original data:** Table 3.7, columns 1 and 2, for 10 participants in study

---

**Table 3.7: Use of Square Root Transformation of $X$ to Linearize Regression Relation**

| (1) Sales Trainee $i$ | (2) Days of Training $X_i$ | (3) Performance Score $Y_i$ | $X_i' = \sqrt{X_i}$ |
|---|---|---|---|
| 1 | 0.5 | 42.5 | 0.70711 |
| 2 | 0.5 | 50.6 | 0.70711 |
| 3 | 1.0 | 68.5 | 1.00000 |
| 4 | 1.0 | 80.7 | 1.00000 |
| 5 | 1.5 | 89.0 | 1.22474 |
| 6 | 1.5 | 99.6 | 1.22474 |
| 7 | 2.0 | 105.3 | 1.41421 |
| 8 | 2.0 | 111.8 | 1.41421 |
| 9 | 2.5 | 112.3 | 1.58114 |
| 10 | 2.5 | 125.7 | 1.58114 |

---

**Figure 3.14a: Scatter plot of original data**

![Scatter plot showing curvilinear relationship](placeholder-sales-scatter.png)

**What we see:** Regression relation appears to be curvilinear, so simple linear regression model (2.1) does not seem to be appropriate

**Since variability at different $X$ levels appears to be fairly constant:** We shall consider transformation on $X$

---

**Based on prototype plot in Figure 3.13a:** We shall consider initially **square root transformation** $X' = \sqrt{X}$

**Transformed values:** Shown in column 3 of Table 3.7

---

**Figure 3.14b: Scatter plot with transformed predictor**

![Scatter plot showing linear relationship](placeholder-sales-transformed-scatter.png)

$$Y \text{ vs. } X' = \sqrt{X}$$

**What we see:** Scatter plot now shows **reasonably linear relation**

**Variability of scatter at different $X$ levels is same as before**, since we did not make transformation on $Y$

---

**To examine further whether simple linear regression model (2.1) is appropriate now:**

We fit it to transformed $X$ data

Regression calculations with transformed $X$ data are carried out in usual fashion, except **predictor variable now is $X'$**

---

**Fitted regression function:**

$$\hat{Y} = -10.33 + 83.45X'$$

---

**Figure 3.14c: Residual plot against $X'$**

![Residual plot showing random scatter](placeholder-sales-residuals.png)

**What we see:** No evidence of lack of fit or of strongly unequal error variances

---

**Figure 3.14d: Normal probability plot**

![Normal probability plot showing linear pattern](placeholder-sales-normal.png)

**What we see:** No strong indications of substantial departures from normality indicated by this plot

**Conclusion supported by high coefficient of correlation** between ordered residuals and their expected values under normality: **0.979**

For $\alpha = 0.01$, Table B.6 shows critical value is 0.879, so observed coefficient is substantially larger and supports reasonableness of normal error terms

---

**Thus, simple linear regression model (2.1) appears to be appropriate here for transformed data**

---

**Fitted regression function in original units of $X$ can easily be obtained, if desired:**

$$\hat{Y} = -10.33 + 83.45\sqrt{X}$$

---

## 3.9.2 Transformations for Nonnormality and Unequal Error Variances

**Unequal error variances** and **nonnormality of error terms** frequently appear together

To remedy these departures from simple linear regression model (2.1), we need **transformation on $Y$**, since:
- Shapes and spreads of distributions of $Y$ need to be changed

**Such transformation on $Y$ may also at same time help to linearize curvilinear regression relation**

At other times, **simultaneous transformation on $X$ may be needed to obtain or maintain linear regression relation**

---

**Frequently, nonnormality and unequal variances departures from regression model (2.1) take form of:**
- Increasing skewness
- Increasing variability of distributions of error terms as mean response $E\{Y\}$ increases

---

**Example:** In regression of yearly household expenditures for vacations ($Y$) on household income ($X$):
- There will tend to be more variation and greater positive skewness (i.e., some very high yearly vacation expenditures) for high-income households
- Than for low-income households, who tend to consistently spend much less for vacations

---

**Figure 3.15 contains some prototype regression relations:**
- Where skewness and error variance increase with mean response $E\{Y\}$

This figure also presents:
- Some simple transformations on $Y$ that may be helpful for these cases

**Several alternative transformations on $Y$ may be tried**, as well as some simultaneous transformations on $X$

**Scatter plots and residual plots should be prepared to determine most effective transformation(s)**

---

### 📊 Example 3.10 - Plasma Levels Study

**Context:** Data on age ($X$) and plasma level of polyamine ($Y$) for portion of 25 healthy children in study

**Original data:** Table 3.8, columns 1 and 2

---

**Table 3.8: Use of Logarithmic Transformation of $Y$ to Linearize Regression Relation and Stabilize Error Variance**

| (1) Child $i$ | (2) Age $X_i$ | (3) Plasma Level $Y_i$ | $Y_i' = \log_{10} Y_i$ |
|---|---|---|---|
| 1 | 0 (newborn) | 13.44 | 1.1284 |
| 2 | 0 (newborn) | 12.84 | 1.1086 |
| 3 | 0 (newborn) | 11.91 | 1.0759 |
| 4 | 0 (newborn) | 20.09 | 1.3030 |
| 5 | 0 (newborn) | 15.60 | 1.1931 |
| 6 | 1.0 | 10.11 | 1.0048 |
| 7 | 1.0 | 11.38 | 1.0561 |
| ... | ... | ... | ... |
| 19 | 3.0 | 6.90 | 0.8388 |
| 20 | 3.0 | 6.77 | 0.8306 |
| 21 | 4.0 | 4.86 | 0.6866 |
| 22 | 4.0 | 5.10 | 0.7076 |
| 23 | 4.0 | 5.67 | 0.7536 |
| 24 | 4.0 | 5.75 | 0.7597 |
| 25 | 4.0 | 6.23 | 0.7945 |

---

**Figure 3.16a: Scatter plot of original data**

![Scatter plot showing curvilinear relationship with increasing variance](placeholder-plasma-scatter.png)

**What we see:**
- Distinct curvilinear regression relationship
- Greater variability for younger children than for older ones

---

**On basis of prototype regression pattern in Figure 3.15b:**

We shall first try **logarithmic transformation** $Y' = \log_{10} Y$

**Transformed $Y$ values:** Shown in column 3 of Table 3.8

---

**Figure 3.16b: Scatter plot with transformed response**

![Scatter plot showing more linear relationship](placeholder-plasma-transformed-scatter.png)

$$Y' = \log_{10} Y \text{ vs. } X$$

**What we see:**
- Transformation not only has led to reasonably linear regression relation
- **Variability at different levels of $X$ also has become reasonably constant**

---

**To further examine reasonableness of transformation $Y' = \log_{10} Y$:**

We fitted simple linear regression model (2.1) to transformed $Y$ data and obtained:

$$\hat{Y}' = 1.135 - 0.1023X$$

---

**Figure 3.16c: Residual plot against $X$**

![Residual plot showing random scatter](placeholder-plasma-residuals.png)

**What we see:** No serious departures from assumptions

---

**Figure 3.16d: Normal probability plot**

![Normal probability plot showing linear pattern](placeholder-plasma-normal.png)

**Coefficient of correlation between ordered residuals and their expected values under normality:** **0.981**

For $\alpha = 0.05$, Table B.6 indicates critical value is 0.959, so observed coefficient supports assumption of normality of error terms

---

**All of this evidence supports appropriateness of regression model (2.1) for transformed $Y$ data**

---

### Comments on Logarithmic Transformation

**1. At times it may be desirable to introduce constant into transformation of $Y$, such as when:**
- Some of $X$ data are near zero
- Reciprocal transformation is desired

Can shift origin by using transformation $X' = 1/(X + k)$, where $k$ is appropriately chosen constant

---

**2. When unequal error variances are present but regression relation is linear:**
- **Transformation on $Y$ may not be sufficient**
- While such transformation may stabilize error variance, it will also change linear relationship to curvilinear one

**Transformation on $X$ may therefore also be required**

This case can also be handled by using **weighted least squares**, procedure explained in **Chapter 11**

---

## 3.9.3 Box-Cox Transformations

**The Challenge:** It is often difficult to determine from diagnostic plots (such as one in Figure 3.16a for plasma levels example):
- Which transformation of $Y$ is most appropriate
- For correcting:
  - Skewness of distributions of error terms
  - Unequal error variances
  - Nonlinearity of regression function

---

**The Solution:** Box-Cox procedure (Ref. 3.9) **automatically identifies transformation from family of power transformations on $Y$**

**Family of power transformations:**

$$Y' = Y^\lambda \quad \text{(3.33)}$$

Where $\lambda$ is parameter to be determined from data

---

**Note:** This family encompasses following simple transformations:

| $\lambda$ | Transformation |
|-----------|----------------|
| $\lambda = 2$ | $Y' = Y^2$ |
| $\lambda = 0.5$ | $Y' = \sqrt{Y}$ |
| $\lambda = 0$ | $Y' = \log_e Y$ (by definition) |
| $\lambda = -0.5$ | $Y' = \frac{1}{\sqrt{Y}}$ |
| $\lambda = -1.0$ | $Y' = \frac{1}{Y}$ |

(3.34)

---

**Normal error regression model with response variable a member of family of power transformations in (3.33) becomes:**

$$Y_i^\lambda = \beta_0 + \beta_1 X_i + \varepsilon_i \quad \text{(3.35)}$$

**Note:** Regression model (3.35) includes **additional parameter, $\lambda$**, which needs to be estimated

---

**Box-Cox procedure uses method of maximum likelihood to estimate $\lambda$**, as well as other parameters $\beta_0$, $\beta_1$, and $\sigma^2$

**In this way, Box-Cox procedure identifies $\lambda$**, the **maximum likelihood estimate of $\lambda$** to use in power transformation

---

### Alternative Procedure When Software Doesn't Provide Box-Cox

**Since some statistical software packages do not automatically provide Box-Cox maximum likelihood estimate $\hat{\lambda}$ for power transformation:**

A simple procedure for obtaining $\hat{\lambda}$ using standard regression software can be employed instead

---

**This procedure involves:**
- Numerical search in range of potential $\lambda$ values
- For example, $\lambda = -2, \lambda = -1.75, \ldots, \lambda = 1.75, \lambda = 2$

**For each $\lambda$ value:**
1. $Y_i^\lambda$ observations are first standardized so that magnitude of error sum of squares does not depend on value of $\lambda$:

$$W_i = \begin{cases}
K_1(Y_i^\lambda - 1) & \lambda \neq 0 \\
K_2(\log_e Y_i) & \lambda = 0
\end{cases} \quad \text{(3.36)}$$

---

Where:

$$K_2 = \left(\prod_{i=1}^{n} Y_i\right)^{1/n} \quad \text{(3.36a)}$$

$$K_1 = \frac{1}{\lambda K_2^{\lambda-1}} \quad \text{(3.36b)}$$

**Note:** $K_2$ is **geometric mean of $Y_i$ observations**

---

2. Once standardized observations $W_i$ have been obtained for given $\lambda$ value:
   - They are regressed on predictor variable $X$
   - Error sum of squares $SSE$ is obtained

**It can be shown:** Maximum likelihood estimate $\hat{\lambda}$ is that value of $\lambda$ for which $SSE$ is minimum

---

**If desired:**
- Finer search can be conducted in neighborhood of $\lambda$ value that minimizes $SSE$

**However, Box-Cox procedure ordinarily is used only to provide guide for selecting transformation**, so **overly precise results are not needed**

---

**In any case:**
- Scatter and residual plots should be utilized to examine appropriateness of transformation identified by Box-Cox procedure

---

### 📊 Example 3.10 (continued) - Plasma Levels Box-Cox Results

**Table 3.9 contains Box-Cox results for plasma levels example**

Selected values of $\lambda$, ranging from $-1.0$ to $1.0$, were chosen, and for each chosen $\lambda$ the transformation (3.36) was made and linear regression of $W$ on $X$ was fitted

---

**For instance, for $\lambda = 0.5$:**
- Transformation $W_i = K_1(\sqrt{Y_i} - 1)$ was made
- Linear regression of $W$ on $X$ was fitted

**For this fitted linear regression:** Error sum of squares is $SSE = 48.4$

---

**The transformation that leads to smallest value of $SSE$ corresponds to $\lambda = -0.5$**, for which $SSE = 30.6$

---

**Table 3.9: Box-Cox Results - Plasma Levels Example**

| $\lambda$ | $SSE$ | $\lambda$ | $SSE$ |
|-----------|-------|-----------|-------|
| 1.0 | 78.0 | -0.1 | 33.1 |
| 0.9 | 70.4 | -0.3 | 31.2 |
| 0.7 | 57.8 | -0.4 | 30.7 |
| 0.5 | 48.4 | **-0.5** | **30.6** |
| 0.3 | 41.4 | -0.6 | 30.7 |
| 0.1 | 36.4 | -0.7 | 31.1 |
| 0 | 34.5 | -0.9 | 32.7 |
| | | -1.0 | 33.9 |

---

**Figure 3.17: SAS-JMP Box-Cox Results**

![Plot of SSE vs lambda showing minimum](placeholder-boxcox-plot.png)

**From plot:** Clear that power value near $\lambda = -0.50$ is indicated

**However, $SSE$ as function of $\lambda$ is fairly stable in range from near 0 to $-1.0$:**

So earlier choice of logarithmic transformation $Y' = \log_{10} Y$ for plasma levels example, corresponding to $\lambda = 0$, is **not unreasonable** according to Box-Cox approach

---

**One reason logarithmic transformation was chosen here:**
- Ease of interpreting it
- Use of logarithms to base 10 rather than natural logarithms does not, of course, affect appropriateness of logarithmic transformation

---

### Comments on Box-Cox Transformations

**1. At times, theoretical or a priori considerations can be utilized to help in choosing appropriate transformation**

**For example:** When shape of scatter in study of relation between:
- Price of commodity ($X$)
- Quantity demanded ($Y$)

Economists may prefer logarithmic transformations of both $Y$ and $X$ because:
- Slope of regression line for transformed variables then measures **price elasticity of demand**
- Slope then commonly interpreted as showing **percent change in quantity demanded per 1 percent change in price**, where it is understood that changes are in opposite directions

---

**2. After transformation has been tentatively selected:**
- Residual plots and other analyses described earlier need to be employed to ascertain that simple linear regression model (2.1) is appropriate for transformed data

---

**3. When transformed models are employed:**
- Estimators $b_0$ and $b_1$ obtained by least squares **have least squares properties with respect to transformed observations**, not original ones

---

**4. Maximum likelihood estimate of $\lambda$ with Box-Cox procedure is subject to sampling variability**

**In addition:**
- Error sum of squares $SSE$ is often fairly stable in neighborhood around estimate

**It is therefore often reasonable to use nearby $\lambda$ value for which:**
- Power transformation is easy to understand

**For example:**
- Use of $\lambda = 0$ instead of maximum likelihood estimate $\lambda = -0.79$ may facilitate understanding without sacrificing much in terms of effectiveness of transformation

---

**To determine reasonableness of using easier-to-understand value of $\lambda$:**
- One should examine flatness of likelihood function in neighborhood of $\lambda$ as we did in plasma levels example
- Alternatively, one may construct approximate confidence interval for $\lambda$; procedure for constructing such interval is discussed in Reference 3.10

---

**5. When Box-Cox procedure leads to $\lambda$ value near 1:**
- **No transformation of $Y$ may be needed**

---

## 3.10 Exploration of Shape of Regression Function

**Scatter plots often indicate readily nature of regression function**

**For instance:** Figure 1.3 clearly shows curvilinear nature of regression relationship between:
- Steroid level
- Age

---

**At other times, however:**
- Scatter plot is complex and it becomes difficult to see nature of regression relationship, if any, from plot

**In these cases, it is helpful to explore nature of regression relationship by:**
- Fitting smoothed curve without any constraints on regression function

**These smoothed curves are also called *nonparametric regression curves***

**They are useful not only for exploring regression relationships but also for confirming nature of regression function when scatter plot visually suggests nature of regression relationship**

---

**Many smoothing methods have been developed for obtaining smoothed curves for time series data:**
- Where $X_i$ denote time periods that are equally spaced apart

**The *method of moving averages*** uses mean of $Y$ observations for adjacent time periods to obtain smoothed values

**For example:**
- Mean of $Y$ values for first three time periods in time series might constitute first smoothed value corresponding to middle of three time periods, in other words, corresponding to time period 2

Then:
- Mean of $Y$ values for second, third, and fourth time periods would constitute second smoothed value, corresponding to middle of these three time periods, in other words, corresponding to time period 3, and so on

**Special procedures are required for obtaining smoothed values at two ends of time series**

**Larger successive neighborhoods used for obtaining smoothed values** → smoother curve will be

---

**The *method of running medians*** is similar to method of moving averages, except:
- **Median is used as average measure** in order to reduce influence of outlying observations

---

**With this method, as well as with moving average method:**
- Successive smoothing of smoothed values and other refinements may be undertaken to provide suitable smoothed curve for time series

Reference 3.11 provides good introduction to running median smoothing method

---

**Many smoothing methods have also been developed for regression data when $X$ values are not equally spaced apart**

**Simple smoothing method, *band regression***, divides data set into number of groups or "bands" consisting of adjacent cases according to their $X$ levels

For each band:
- Median $X$ value and median $Y$ value are calculated
- Points defined by pairs of these median values are then connected by straight lines

**For example, consider following simple data set divided into three groups:**

| $X$ | $Y$ | Median $X$ | Median $Y$ |
|-----|-----|------------|------------|
| 2.0 | 13.1 | 2.7 | 14.4 |
| 3.4 | 15.7 | | |
| 3.7 | 14.9 | | |
| 4.5 | 16.8 | 4.5 | 16.8 |
| 5.0 | 17.1 | | |
| 5.2 | 16.9 | | |
| 5.9 | 17.8 | 5.55 | 17.35 |

**Three pairs of medians are then plotted on scatter plot of data and connected by straight lines** as simple smoothed nonparametric regression curve

---

## 3.10.1 Lowess Method

**The *lowess method***, developed by Cleveland (Ref. 3.12), is more refined nonparametric method than band regression

It obtains smoothed curve by:
- Fitting successive linear regression functions in local neighborhoods

**The name lowess stands for *locally weighted regression scatter plot smoothing***

---

**Method is similar to moving average and running median methods** in that:
- It uses neighborhood around each $X$ value to obtain smoothed $Y$ value corresponding to that $X$ value

**It obtains smoothed $Y$ value at given $X$ level by:**
- Fitting linear regression to data in neighborhood of $X$ value
- Then using fitted value at $X$ as smoothed value

---

**To illustrate this concretely:**
- Let $(X_1, Y_1)$ denote sample case with smallest $X$ value
- $(X_2, Y_2)$ denote sample case with second smallest $X$ value, and so on

**If neighborhoods of three $X$ values are used with lowess method:**
- Then linear regression would be fitted to data:

$$(X_1, Y_1) \quad (X_2, Y_2) \quad (X_3, Y_3)$$

**Fitted value at $X_2$ would constitute smoothed value corresponding to $X_2$**

---

**Another linear regression would be fitted to data:**

$$(X_2, Y_2) \quad (X_3, Y_3) \quad (X_4, Y_4)$$

and fitted value at $X_3$ would constitute smoothed value corresponding to $X_3$, and so on

---

**Smoothed values at each end of $X$ range are also obtained by lowess procedure**

---

**Lowess method uses number of refinements in obtaining final smoothed values to improve smoothing and to make procedure robust to outlying observations:**

---

**1. Linear regression is weighted** to give cases further from middle $X$ level in each neighborhood smaller weights

---

**2. To make procedure robust to outlying observations:**
- Linear regression fitting is repeated, with weights revised so that:
  - Cases that had large residuals in first fitting receive smaller weights in second fitting

---

**3. To improve robustness of procedure further:**
- Step 2 is repeated one or more times by revising weights according to size of residuals in latest fitting

---

**To implement lowess procedure:**
- One must choose:
  - Size of successive neighborhoods to be used when fitting each linear regression
  - One must also choose **weight function** that gives less weight to neighborhood cases with $X$ values far from each center $X$ level
  - Another weight function that gives less weight to cases with large residuals

**Finally:**
- Number of iterations to make procedure robust must be chosen

---

**In practice:**
- Two iterations appear to be sufficient to provide robustness
- Also, weight functions suggested by Cleveland appear to be adequate for many circumstances

**Hence:**
- Primary choice to be made for particular application is **size of successive neighborhoods**

**Larger the size:**
- Smoother the function
- But greater the danger that smoothing will lose essential features of regression relationship

**May require some experimentation with different neighborhood sizes in order to find size that best brings out regression relationship**

---

**We explain lowess method in detail in Chapter 11 in context of multiple regression**

**Specific choices of weight functions and neighborhood sizes are discussed there**

---

### 📊 Example 3.11 - Research Quality Study

**Context:** Study of research quality at 24 research laboratories

**Variables:**
- Response variable: Measure of quality of research done at laboratory
- Explanatory variable: Measure of volume of research performed at laboratory

---

**Figure 3.18a: Scatter plot**

![Scatter plot showing complex relationship](placeholder-research-scatter.png)

**Note:** Very difficult to tell from this scatter plot whether or not relationship exists between research quality and quantity

---

**Figure 3.18b: Lowess smoothed curve**

![Lowess smoothed curve](placeholder-research-lowess.png)

**The curve suggests:**
- There might be somewhat higher research quality for medium-sized laboratories

---

**However, scatter is great so this suggested relationship should be considered only as possibility**

**Also:**
- Because any particular measures of research quality and quantity are so limited, other measures should be used to see if these corroborate relationship suggested in Figure 3.18b

---

## 3.10.2 Use of Smoothed Curves to Confirm Fitted Regression Function

**Smoothed curves are useful not only in exploratory stages** when regression model is selected, but they are also helpful in **confirming regression function chosen**

---

**Procedure for confirmation is simple:**
- Smoothed curve is plotted together with confidence band for fitted regression function

**If smoothed curve falls within confidence band:**
- We have supporting evidence of appropriateness of fitted regression function

---

### 📊 Example - Toluca Company Confirmation

**Figure 3.19a:** Repeats scatter plot for Toluca Company example from Figure 1.10a and shows lowess smoothed curve

**It appears:**
- Regression relation is linear or possibly slightly curved

---

**Figure 3.19b:** Repeats confidence band for regression line from Figure 2.6 and shows lowess smoothed curve

**We see:**
- Smoothed curve falls within confidence band for regression line
- Thereby supports appropriateness of linear regression function

---

**Figure 3.19: MINITAB Lowess Curve and Confidence Band for Regression Line - Toluca Company Example**

![Lowess curve within confidence band](placeholder-toluca-lowess.png)

---

### Comments on Smoothed Curves

**1. Smoothed curves, such as lowess curve:**
- Do NOT provide analytical expression for functional form of regression relationship
- They only suggest shape of regression curve

---

**2. Lowess procedure is not restricted to fitting linear regression functions in each neighborhood:**
- Higher-degree polynomials can also be utilized with this method

---

**3. Smoothed curves are also useful when examining residual plots to ascertain whether:**
- Residuals (or absolute or squared residuals) follow some relationship with $X$ or $\hat{Y}$

---

**4. References 3.13 and 3.14 provide good introductions to other nonparametric methods in regression analysis**

---

## 3.11 Case Example - Plutonium Measurement

**Context:** Some environmental cleanup work requires that nuclear materials, such as plutonium 238, be located and completely removed from restoration site

**Challenge:** When plutonium has become mixed with other materials in very small amounts, detecting its presence can be difficult task

---

**Detection method:** Very small amounts can be traced, however, because:
- Plutonium emits subatomic particles — **alpha particles** — that can be detected

**Devices used to detect plutonium:**
- Record **intensity of alpha particle strikes in counts per second (#/sec)**

---

**The regression relationship:**
- Between **alpha counts per second** (response variable)
- And **plutonium activity** (explanatory variable)

is then used to **estimate activity of plutonium in material under study**

---

**This use of regression relationship involves *inverse prediction*:**
- Predicting predictor variable ($X$) from observed alpha count ($Y$)
- Procedure discussed in Chapter 4

---

**The task here is to estimate regression relationship:**
- Between alpha counts per second and plutonium activity

**This relationship varies for each measurement device and must be established precisely each time a different measurement device is used**

---

**It is reasonable to assume here that:**
- Level of alpha counts **increases** with plutonium activity
- But exact nature of relationship is generally unknown

---

**In study to establish regression relationship for particular measurement device:**
- Four plutonium *standards* were used
- These standards are aluminum/plutonium rods containing fixed, known level of plutonium activity

**Levels of plutonium activity in four standards:**
- 0.0, 5.0, 10.0, and 20.0 picocuries per gram (pCi/g)

**Each standard was exposed to detection device from 4 to 10 times:**
- Rate of alpha strikes, measured as counts per second, was observed for each replication

---

**A portion of data is shown in Table 3.10**, and data are plotted as scatter plot in **Figure 3.20a**

**Notice that, as expected:**
- Strike rate tends to increase with activity level of plutonium

**Notice also:**
- Nonzero strike rates are recorded for standard containing no plutonium
- This results from background radiation
- Indicates that regression model with **intercept term is required here**

---

**Table 3.10: Basic Data - Plutonium Measurement Example**

| Case | Plutonium Activity (pCi/g) | Alpha Count Rate (#/sec) |
|------|----------------------------|--------------------------|
| 1 | 20 | 0.150 |
| 2 | 0 | 0.004 |
| 3 | 10 | 0.069 |
| ... | ... | ... |
| 22 | 0 | 0.002 |
| 23 | 5 | 0.049 |
| 24 | 0 | 0.106 |

---

### Initial Diagnostics

**As initial step to examine nature of regression relationship:**
- Lowess smoothed curve was obtained
- This curve is shown in **Figure 3.20b**

---

**Figure 3.20b: Lowess Smoothed Curve**

![Lowess curve showing slightly curved relationship](placeholder-plutonium-lowess.png)

**We see:**
- Regression relationship may be linear or slightly curvilinear in range of plutonium activity levels included in study

**We also see:**
- One of readings taken at 0.0 pCi/g (case 24) does not appear to fit with rest of observations

---

**An examination of laboratory records revealed:**
- Experimental conditions were not properly maintained for last case, and it was therefore decided that **case 24 should be discarded**

---

**Note, incidentally, how robust lowess smoothing process was here by:**
- Assigning very little weight to outlying observation

---

### Linear Regression Analysis

**Linear regression function was fitted next, based on remaining 23 cases**

**SAS-JMP regression output is shown in Figure 3.21a:**

---

**Figure 3.21a: SAS-JMP Regression Output**

| Term | Estimate | Std Error | t Ratio | Prob>\|t\| |
|------|----------|-----------|---------|------------|
| Intercept | 0.0070331 | 0.0036 | 1.95 | 0.0641 |
| Plutonium | 0.005537 | 0.00037 | 15.13 | 0.0000 |

| Source | DF | Sum of Squares | Mean Square | F Ratio |
|--------|----|--------------| ------------|---------|
| Model | 1 | 0.03619042 | 0.036190 | 228.9984 |
| Error | 21 | 0.00331880 | 0.000158 | **Prob>F** |
| C Total | 22 | 0.03950922 | | 0.0000 |

| Source | DF | Sum of Squares | Mean Square | F Ratio |
|--------|----|--------------| ------------|---------|
| Lack of Fit | 2 | 0.00016811 | 0.000084 | 0.5069 |
| Pure Error | 19 | 0.00315069 | 0.000166 | **Prob>F** |
| Total Error | 21 | 0.00331880 | | 0.6103 |

---

**Plot of residuals in Table 1.2 against predictor variable, shown in Figure 3.21b:**

![Residual plot showing some pattern](placeholder-plutonium-residuals-1.png)

---

**Normal probability plot shown in Figure 3.21c:**

![Normal probability plot with some departure](placeholder-plutonium-normal-1.png)

---

**JMP output uses label Model to denote regression component of analysis of variance**

**Label C Total stands for corrected total**

---

**We see from flared, megaphone shape of residual plot that:**
- Error variance appears to be increasing with level of plutonium activity

**Normal probability plot suggests nonnormality (heavy tails)**, but:
- Nonlinearity of plot is likely to be related (at least in part) to unequal error variances

**Existence of nonconstant variance is confirmed by lack of fit test statistic (3.25):**

$$F^* = 10.1364, \quad P\text{-value} = 0.0010$$

Of course, this result is not completely unexpected, since $Y$ was linearly related to $X$

---

### Transformation to Address Variance Issues

**To restore linear relation with transformed $Y$ variable:**
- We shall see if square root transformation of $X$ will lead to satisfactory linear fit

---

**Regression results when regressing:**
- $Y' = \sqrt{Y}$ on $X' = \sqrt{X}$

are presented in **Figure 3.23**

---

**Figure 3.23: Regression Output for Transformed Variables**

---

**Notice from residual plot in Figure 3.23b:**
- Square root transformation of predictor variable has eliminated lack of fit

**Also:**
- Normal probability plot of residuals in Figure 3.23c appears to be satisfactory
- Correlation test ($r = 0.986$) supports assumption of normally distributed error terms
  - Interpolated critical value in Table B.6 for $\alpha = 0.05$ and $n = 23$ is 0.9555

---

**However, residual plot suggests:**
- Some nonconstancy of error variance may still remain, but if so, it does not appear to be substantial

**Breusch-Pagan test statistic (3.11):**

$$X_{BP}^2 = 3.85$$

which corresponds to $P$-value of 0.05, supporting conclusion from residual plot that nonconstancy of error variance is not substantial

---

**Figure 3.23d contains SYSTAT plot of confidence band (2.40) for fitted regression line:**

$$\hat{Y}' = 0.0730 + 0.0573X'$$

---

**We see:**
- Regression line has been estimated fairly precisely

**Also plotted in this figure is lowess smoothed curve**

**This smoothed curve falls entirely within confidence band:**
- Supporting reasonableness of linear regression relation between $Y'$ and $X'$

---

**Lack of fit test statistic (3.25) now is:**

$$F^* = 1.2868 \quad (P\text{-value} = 0.2992)$$

Also supporting linearity of regression relating:
- $Y' = \sqrt{Y}$ to $X' = \sqrt{X}$

---

## 🎓 Chapter Summary

**Congratulations!** You've completed a comprehensive tour of regression diagnostics and remedial measures. Let's recap the key takeaways:

---

### **What We Learned:**

1. **Before trusting your regression model, CHECK IT!**
   - Examine predictor variable distribution
   - Analyze residuals systematically
   - Use both visual and formal diagnostic tests

2. **Residuals are your friends** - they reveal:
   - Nonlinearity
   - Nonconstant variance
   - Outliers
   - Non-independence
   - Non-normality
   - Missing variables

3. **The diagnostic toolkit:**
   - 7 essential plots (residuals vs. $X$, $\hat{Y}$, time, etc.)
   - Formal tests (Brown-Forsythe, Breusch-Pagan, F-test for lack of fit)
   - Transformations (Box-Cox, logarithmic, square root)
   - Smoothing methods (lowess curves)

4. **When things go wrong, you have options:**
   - Transform variables
   - Use weighted least squares
   - Try different regression function
   - Add omitted variables

---

### **Critical Skills Developed:**

✅ Reading and interpreting residual plots
✅ Conducting formal diagnostic tests
✅ Choosing appropriate transformations
✅ Using Box-Cox procedure
✅ Applying lowess smoothing
✅ Integrating multiple diagnostic methods

---

### **Most Important Lesson:**

> **Diagnostics are not optional!** A model that "looks good" statistically may still be completely inappropriate. Always check assumptions before making inferences.

---
