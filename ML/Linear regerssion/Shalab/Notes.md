## 📊 Chapter 1: Introduction to Regression Analysis

Linear models play a central part in modern statistical methods. They can approximate a large amount of metric data structures in their entire range of definition or at least piecewise.

---

### 1. Linear Models and Regression Analysis

Suppose the outcome of any process is denoted by a random variable $y$, called as **dependent (or study) variable**. This outcome depends on $k$ **independent (or explanatory) variables** denoted by $X_{1}, X_{2}, \dots, X_{k}$.

The behavior of $y$ can be explained by a relationship given by:
$$y=f(X_{1},X_{2},\dots,X_{k},\beta_{1},\beta_{2},\dots,\beta_{k})+\epsilon$$
* $f$ is some well-defined function.
* $\beta_{1}, \beta_{2}, \dots, \beta_{k}$ are the **parameters** which characterize the role and contribution of $X_{1}, X_{2}, \dots, X_{k}$, respectively.
* The term $\epsilon$ reflects the **stochastic nature** of the relationship and indicates that it is not exact in nature.

#### Mathematical vs. Statistical Models
* When $\epsilon=0$, the relationship is called the **mathematical model**.
* When $\epsilon \ne 0$, it is called the **statistical model**.

---

### 2. Linear vs. Nonlinear Models

A model is termed as **linear** if it is linear in parameters and **nonlinear** if it is not linear in parameters.

#### Condition for Linearity
A model is called a linear model if all the partial derivatives of $y$ with respect to each of the parameters $\beta_{1}, \beta_{2}, \dots, \beta_{k}$, are **independent of the parameters**.

If any of the partial derivatives of $y$ with respect to any of the $\beta_{1}, \beta_{2}, \dots, \beta_{k}$ is **not independent of the parameters**, then the model is called **nonlinear**.

> **Crucial Note:** The linearity or non-linearity of the model is **not described by the linearity or nonlinearity of explanatory variables** in the model.

| Example Model | Type | Reason |
| :--- | :--- | :--- |
| $y=\beta_{1}X_{1}^{2}+\beta_{2}\sqrt{X_{2}}+\beta_{3}\log X_{3}+\epsilon$ . | **Linear** | $\partial y/\partial\beta_{i}$ ($i=1,2,3$) are independent of the parameters $\beta_{i}$. |
| $y=\beta_{1}^{2}X_{1}+\beta_{2}X_{2}+\beta_{3}\log X+\epsilon$ . | **Nonlinear** | $\partial y/\partial\beta_{1}=2\beta_{1}X_{1}$ depends on $\beta_{1}$. |

---

### 3. The Linear Statistical Model and Estimation

When the function $f$ is linear in parameters, the model is a **linear model**. The general form of $f$ for a linear model is.
$$f(X_{1},X_{2},\dots,X_{k},\beta_{1},\beta_{2}\dots,\beta_{k})=\beta_{1}X_{1}+\beta_{2}X_{2}+\dots+\beta_{k}X_{k}$$

Thus, the linear model is.
$$y=\beta_{1}X_{1}+\beta_{2}X_{2}+\dots+\beta_{k}X_{k}+\epsilon$$

The knowledge of the model depends on the knowledge of the unknown parameters $\beta_{1},\beta_{2},\dots,\beta_{k}$.

**Linear statistical modeling** consists of developing approaches and tools to determine these parameters
Linear Model Example
y=β1∗X12+β2∗sqrt(X2)+β3∗log(X3)+ε
y=β1∗X1
2
+β2∗sqrt(X2)+β3∗log(X3)+ε

Partial derivatives:

∂y/∂βi=Xiterms
∂y/∂βi=Xi
t
	​

erms

They do not depend on βi.
Hence, the model is linear.

Nonlinear Model Example
y=β12∗X1+β2∗X2+ε
y=β1
2
∗X1+β2∗X2+ε
∂y/∂β1=2∗β1∗X1
∂y/∂β1=2∗β1∗X1

This depends on β1.
Hence, the model is nonlinear.
#### Estimation Procedures
Different statistical estimation procedures can be employed to estimate the parameters of the model

* **Principle of Least Squares:** Does not require any knowledge about the distribution of $y$ This is the most commonly used method.
* **Method of Moments:** Does not require any knowledge about the distribution of $y$
* **Method of Maximum Likelihood:** Needs further knowledge of the distribution of $y$

---

### 4. Regression Analysis: The "Backward Direction"

**Regression analysis** is the technique used to determine the values of the parameters given the data on $y$ and $X_{1}, X_{2}, \dots, X_{k}$

The literal meaning of regression is **"to move in the backward direction"**

#### Understanding the Backward Process

1.  **Ideal (Forward) Process:** The model exists in nature but is unknown. The pre-existing model gives rise to the data
2.  **Regression (Backward) Process:** We collect the data first, and then we move in the **backward direction**. The collected data is used to determine the parameters of the unknown model
egression analysis is the statistical technique used to estimate the unknown parameters 
β1,β2,…,βk
β
1
	​

,β
2
	​

,…,β
k
	​

.

The word regression literally means:

“To move backward”

Forward (Nature’s Process)
Model (unknown)  →  Generates Data


True relationship exists in nature

We only observe the output

Observed Data  →  Estimate Model

    We start with data

    Infer the parameters and structure

This reversal is why regression is called a backward process.
---

### 5. Steps in Regression Analysis

Regression analysis includes a sequence of steps[

| Step | Description |
| :--- | :--- |
| **1. Statement of the problem** | Specify the problem and objectives. Wrong formulation leads to erroneous statistical inferences |
| **2. Choice of relevant variables** | Select the study and explanatory variables. Correct choice determines correct statistical inferences |
| **3. Collection of data** | Collect data on relevant variables, deciding if they are **quantitative** or **qualitative**. Different methods apply (e.g., **Logistic regression** for a binary study variable; **ANOVA** if all explanatory variables are qualitative). |
| **4. Specification of model** | Determine the form of the tentative model, which depends on unknown parameters. |
| **5. Choice of method for fitting** | Choose an estimation method, typically the **Least-Squares Method**. |
| **6. Fitting of model** | Substitute parameter estimates ($\hat{\beta}_{i}$) into the equation to get a usable, fitted model, $y=f(X_{1},X_{2},\dots,X_{k},\hat{\beta}_{1},\hat{\beta}_{2},\dots,\hat{\beta}_{k})$. This is used for prediction. |
| **7. Model validation and criticism** | Check assumptions, as statistical inferences heavily depend on them being satisfied. Regression is an **iterative process** where outputs are used to diagnose, validate, criticize, and modify the inputs. |
| **8. Using the chosen model** | The ultimate objective is to determine the explicit form of the regression equation. It is used for forecasting, understanding variable interrelationships, and policy formulation. |

---

### 6. Classification of Regression Models

The classification is based on the number and type of variables and the form of the parameters.

| Type of Regression | Conditions |
| :--- | :--- |
| **Univariate** | Only one quantitative response variable. |
| **Multivariate** | Two or more quantitative response variables. |
| **Simple** | Only one explanatory variable. |
| **Multiple** | Two or more explanatory variables. |
| **Linear** | All parameters enter the equation linearly, possibly after transformation of the data. |
| **Nonlinear** | The relationship is nonlinear, or some parameters appear nonlinearly, and no transformation is possible to make them appear linearly. |
| **Analysis of Variance (ANOVA)** | All explanatory variables are qualitative variables. |
| **Analysis of Covariance (ANCOVA)** | Some explanatory variables are quantitative variables and others are qualitative variables. |
| **Logistic** | The response variable is qualitative. |

---

Would you like a more detailed explanation of the **least-squares method** for parameter estimation?
