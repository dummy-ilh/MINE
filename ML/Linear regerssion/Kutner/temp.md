

# 1.1 Relations Between Variables

What it means for two variables to be “related.”?

There are two fundamentally different kinds of relationships:

1. Deterministic (Functional) relationship
2. Stochastic (Statistical) relationship



---

## 1. Functional (Deterministic) Relationship

A functional relationship is exact.

If ( X ) is given, ( Y ) is fully determined.

Mathematically:

$
Y = f(X)
$

There is **no randomness**.

For each value of ( X ), there is exactly one value of ( Y ).

---

### Example: Revenue = Price × Units Sold

Suppose a product sells at a fixed price of $2 per unit.

Let:

* ( X ) = number of units sold
* ( Y ) = revenue

Then:

$[
Y = 2X
]$

This is deterministic.

If ( X = 75 ), then:
$[
Y = 150
]$

All observed data points lie exactly on the line.

### Graphical Representation

!$[Image](https://homework.study.com/cimages/multimages/16/graph5211157392957939248.jpg)

!$[Image](https://www.jpstats.org/Regression/images/sec01_02/01_02_deterministic.png)

!$[Image](https://www.itl.nist.gov/div898/handbook/eda/section3/gif/scatplo4.gif)

!$[Image](https://www.researchgate.net/publication/360992572/figure/fig1/AS%3A11431281171723286%401688369943691/SCATTERPLOT-GRAPH-CORRELATION.ppm)

Every point lies perfectly on the line.

There is no variability.

---

### Key Property

Functional relationships have:

* Zero randomness
* Zero residuals
* No probability distributions

These belong to algebra, not statistics.

---

## 2. Statistical (Stochastic) Relationship

Real-world data almost never behave deterministically.

Instead:

For a fixed value of ( X ), multiple values of ( Y ) are possible.

This means:

$[
Y \text{ is a random variable conditional on } X
]$

Formally:

$[
Y \mid X = x \sim \text{some probability distribution}
]$

Now we are in the world of statistics.

---

### Example: Midyear vs Year-End Performance

Let:

* ( X ) = midyear performance score
* ( Y ) = year-end performance score

Empirically, higher midyear scores tend to correspond to higher year-end scores.

But not perfectly.

Two employees with the same midyear score may receive different year-end evaluations.

So:

$[
Y = f(X) + \text{random variation}
]$

---

### Scatter Plot Representation

!$[Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AFoyj8GRC5AGjUuhyPRKHSw.png)

!$[Image](https://mathbitsnotebook.com/Algebra1/StatisticsReg/residualgraph1aa.jpg)

!$[Image](https://files.planyway.com/strapi-uploads/assets/Scatter_plot_422d3c3f82.png)

!$[Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AnElSDIp1jloJiqWX7hniaQ.png)

Points cluster around a line — but do not lie exactly on it.

That “scatter” is the essence of statistics.

---

## The Modern Probabilistic View

Kutner describes regression using geometric language.


A regression model assumes:

1. For each fixed ( x ), the conditional distribution ( Y \mid X=x ) exists.
2. The conditional mean varies systematically with ( x ).

Define:

$
m(x) = \mathbb{E}$[Y \mid X = x]$
$

This function ( m(x) ) is called the **regression function**.

Graphically, it is the curve through the centers of the vertical distributions.

---

### Conceptual Picture

For each value of ( x ):

* There is a vertical distribution of possible ( Y ) values.
* The mean of that distribution lies on the regression curve.

!$[Image]$(https://www.alexejgossmann.com/assets/img/2018-08-12-conditional_distributions/conditional_densities.png)

!$[Image]$(https://blogs.sas.com/content/iml/files/2015/09/GLM_normal_identity.png)

!$[Image]$(https://bpostance.github.io/assets/images/2020-05-17-glm-fig1.png)

!$[Image]$(https://www.researchgate.net/publication/364261195/figure/fig2/AS%3A11431281375498991%401744647955135/Scatter-plots-including-a-regression-line-illustrating-the-vertical-distribution-of-the.tif)

This is the conceptual core of regression.

---

## The Two Ingredients of a Statistical Relationship

Kutner says a statistical relationship has two parts.

Let’s express this precisely:

### (1) Systematic Component

$
m(x) = \mathbb{E}$[Y \mid X = x]$
$

This captures the structured relationship.

---

### (2) Random Component

Define the error:

$
\varepsilon = Y - m(X)
$

Then:

$
Y = m(X) + \varepsilon
$

with:

$
\mathbb{E}$[\varepsilon \mid X]$ = 0
$

This decomposition is the foundation of all regression theory.

---

## Why This Matters

This decomposition separates:

* Signal (systematic pattern)
* Noise (random fluctuation)

Modern ML calls this:

Bias–Variance Decomposition
Signal–Noise Decomposition

Same concept. Older language.

---

## Linear vs Curvilinear Relationships

The regression function ( m(x) ) can take many forms:

* Linear
* Quadratic
* Exponential
* Logistic
* Arbitrary smooth function


Regression analysis is about estimating  m(x) .

---

# Deep Insight

Regression is not about fitting lines.

It is about estimating:

$[
\mathbb{E}$[Y \mid X]$
]$

Everything else is technical machinery.

---

# Connection to Modern ML

Linear regression in ML is exactly:

$[
Y = X\beta + \varepsilon
]$

Neural networks estimate:

$[
m(x) \approx \text{complex nonlinear function}
]$

But conceptually, they are doing the same thing:

Estimating the conditional expectation.

---

# Summary of Section 1.1

We move from:

Deterministic world:
$[
Y = f(X)
]$

to

Stochastic world:
$[
Y = m(X) + \varepsilon
]$

with

$[
m(x) = \mathbb{E}$[Y \mid X=x]$
]$

This is the birth of regression.

---

Ready to proceed?
