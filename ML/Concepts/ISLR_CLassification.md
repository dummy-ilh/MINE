# Chapter 4: Classification

Classification is the process of predicting a **qualitative** (categorical) response . While we used Linear Regression for continuous values, we use Classification for discrete labels (e.g., "Default" vs. "No Default").

---

## 4.1 & 4.2 Why Not Linear Regression?

If we have a binary response (0 or 1), why not just use OLS?

1. **Out-of-bound predictions:** Linear regression can produce probabilities  or .
2. **Ordering Problems:** For  classes (e.g., Stroke, Drug Overdose, Epileptic Seizure), linear regression implies an order and a distance between classes that may not exist.

---

## 4.3 Logistic Regression

Instead of modeling  directly, logistic regression models the **probability** that  belongs to a particular category.

### 4.3.1 The Logistic Model
To keep the probability p(X) between 0 and 1, we use the Logistic Function:
p(X)=1+eβ0​+β1​Xeβ0​+β1​X​

Rearranging this gives us the Logit (or Log-Odds):
log(1−p(X)p(X)​)=β0​+β1​X

    Odds: 1−p(X)p(X)​ can take any value from 0 to ∞.

    Log-odds: Is linear in X.

### 4.3.2 Estimating Coefficients (Maximum Likelihood)

We do not use Least Squares. We use **Maximum Likelihood Estimation (MLE)**. We seek estimates for  and  such that the predicted probability  for each observation corresponds as closely as possible to the observed category.

### 4.3.4 Multiple Logistic Regression



Extending to p predictors:
log(1−p(X)p(X)​)=β0​+β1​X1​+⋯+βp​Xp​
---

## 4.4 Linear Discriminant Analysis (LDA)

Logistic regression models  directly. LDA models the distribution of  in each class separately and then uses **Bayes’ Theorem** to flip them into probabilities.

Logistic regression models P(Y=k∣X=x) directly. LDA models the distribution of X in each class separately and then uses Bayes’ Theorem to flip them into probabilities.
4.4.1 Bayes' Theorem for Classification
P(Y=k∣X=x)=∑l=1K​πl​fl​(x)πk​fk​(x)​

    πk​: Prior probability of class k.

    fk​(x): Density function of X for class k (usually assumed Normal).

4.4.2 LDA for p=1

We assume fk​(x) is Gaussian: fk​(x)=2π​σ1​exp(−2σ21​(x−μk​)2). Crucial LDA Assumption: All classes share the same variance σ2.

The Discriminant Score δk​(x) is linear:
δk​(x)=x⋅σ2μk​​−2σ2μk2​​+log(πk​)

We assign x to the class with the highest δk​(x).

### 4.4.4 Quadratic Discriminant Analysis (QDA)

**QDA Assumption:** Each class has its *own* variance .
Because the variances are different, the decision boundary becomes **quadratic** rather than linear.

---

## 4.5 Comparison of Classification Methods

| Method | Best Use Case | Decision Boundary |
| --- | --- | --- |
| **Logistic Regression** | Binary classification; low assumptions. | Linear |
| **LDA** | Classes are well-separated;  is small;  is approx. Normal. | Linear |
| **QDA** | Large training sets; very different class variances. | Quadratic |
| **KNN** | Complex, non-linear boundaries; no theoretical shape known. | Non-parametric |

---

## 🧠 FAANG "Hard" Interview Questions

### Q1: The LDA vs. Logistic Regression Debate

**Question:** *"Both LDA and Logistic Regression produce linear decision boundaries. When would you mathematically prefer LDA over Logistic Regression?"*

**Answer:** 1. **Stability:** When the classes are well-separated, Logistic Regression's coefficient estimates are surprisingly unstable. LDA does not suffer from this.
2. **Small :** If  is small and the distribution of predictors  is approximately normal in each class, LDA is more stable than Logistic Regression.
3. **Multi-class:** LDA is more naturally suited for  response classes.

### Q2: The QDA Bias-Variance Tradeoff

**Question:** *"Why would we ever use LDA if QDA is more flexible?"*

**Answer:** It's the **Bias-Variance Tradeoff**.

* **LDA** has fewer parameters to estimate ( parameters). It has higher **Bias** but lower **Variance**. Use when  is small.
* **QDA** must estimate a separate covariance matrix for each class ( parameters). This leads to higher **Variance** but lower **Bias**. Use when the training set is very large.

### Q3: LDA for High-Dimensional Data

**Question:** *"What happens to LDA if ?"*

**Answer:** LDA fails. The estimation of the covariance matrix  requires calculating its inverse. If , the sample covariance matrix is singular (not invertible). In this case, you must use **Regularized Discriminant Analysis** or **Lasso** (Chapter 6).

---
md
# 📊 A Comparison of Classification Methods — Deep, Exam-Ready Notes

We compare **Logistic Regression**, **LDA**, **QDA**, and **KNN** at a *theoretical*, *assumptions*, and *bias–variance* level, tying everything to the six scenarios.

---

## 1. Logistic Regression vs LDA: Mathematical Connection

Consider binary classification with predictor $X \in \mathbb{R}^p$.

### Logistic Regression
Assumes:
\[
\log\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = \beta_0 + \beta^T x
\]

- Discriminative model
- Directly models $P(Y|X)$
- No distributional assumption on $X$

---

### LDA Assumptions
\[
X|Y=k \sim \mathcal{N}(\mu_k, \Sigma), \quad \Sigma_1 = \Sigma_2
\]

Bayes rule yields:
\[
\log\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = c_0 + c^T x
\]

📌 **Key Result**
> Under Gaussian class-conditional densities with equal covariance, **LDA produces a linear log-odds**, identical in form to logistic regression.

---

### Why They Differ in Practice

| Aspect | Logistic | LDA |
|----|----|----|
| Estimation | MLE on $P(Y|X)$ | Plug-in generative |
| Assumptions | Weak | Gaussian + equal $\Sigma$ |
| Robust to non-normality | ✅ | ❌ |
| Small $n$ advantage | ❌ | ✅ |

📌 **Bias–Variance View**
- LDA: lower variance if assumptions correct
- Logistic: lower bias if assumptions violated

---

## 2. QDA: The Middle Ground

### QDA Assumptions
\[
X|Y=k \sim \mathcal{N}(\mu_k, \Sigma_k)
\]

Leads to:
\[
\log\frac{P(Y=1|X=x)}{P(Y=0|X=x)} = x^T A x + b^T x + c
\]

→ **Quadratic decision boundary**

---

### Bias–Variance Tradeoff

| Model | Bias | Variance |
|----|----|----|
| LDA | High | Low |
| QDA | Medium | Medium |
| KNN | Low | High |

📌 QDA works best when:
- True boundary is **quadratic**
- Sample size sufficient to estimate $\Sigma_k$

---

## 3. KNN: Fully Non-Parametric

### Prediction Rule
\[
\hat P(Y=k|X=x) = \frac{1}{K} \sum_{i \in \mathcal{N}_K(x)} \mathbf{1}(y_i = k)
\]

- No model assumptions
- Decision boundary adapts locally

---

### Bias–Variance Decomposition

- Small $K$:
  - Low bias
  - Very high variance
- Large $K$:
  - High bias
  - Low variance

📌 Optimal $K$ minimizes:
\[
\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
\]

---

## 4. Interpreting the Six Scenarios (Unified View)

### Scenarios 1–3: **Linear Bayes Boundary**

| Scenario | Data Property | Winner | Why |
|----|----|----|----|
| 1 | Gaussian, iid | LDA | Exact model match |
| 2 | Correlated Gaussian | LDA | Covariance handled |
| 3 | Heavy-tailed ($t$) | Logistic | LDA assumptions violated |

📌 QDA fails due to **overfitting covariance**

---

### Scenarios 4–5: **Quadratic Boundary**

| Scenario | Property | Winner |
|----|----|----|
| 4 | Gaussian, unequal $\Sigma$ | QDA |
| 5 | Polynomial logistic surface | QDA |

📌 Linear models fail due to **high bias**

---

### Scenario 6: **Highly Non-Linear Boundary**

| Method | Result | Reason |
|----|----|----|
| LDA / Logistic | Poor | Linear bias |
| QDA | Slight improvement | Still restricted |
| KNN-1 | Worst | Extreme variance |
| KNN-CV | Best | Proper smoothness |

📌 **Key Lesson**
> Flexibility helps *only when properly regularized*

---

## 5. Parametric vs Non-Parametric Summary

| Method | Boundary | Assumptions | Interpretability |
|----|----|----|----|
| Logistic | Linear | Minimal | High |
| LDA | Linear | Gaussian, equal $\Sigma$ | High |
| QDA | Quadratic | Gaussian | Medium |
| KNN | Arbitrary | None | None |

---

## 6. Feature Engineering as a Bridge

Adding polynomial features to logistic regression:
\[
\log\frac{P(Y=1)}{P(Y=0)} = \beta_0 + \beta_1 X_1 + \beta_2 X_1^2 + \beta_3 X_1 X_2
\]

📌 This moves:
\[
\text{Logistic} \;\rightarrow\; \text{QDA-like capacity}
\]

But:
- Bias ↓
- Variance ↑

---

## 7. Final Big-Picture Rules

1. **Linear boundary + small $n$** → LDA  
2. **Linear boundary + assumption violations** → Logistic  
3. **Moderate curvature** → QDA  
4. **Highly complex boundary** → KNN (CV-tuned)  
5. **Never trust $K=1$**

---

## 8. One-Line Exam Insight

> *Classification is not about choosing the “best” model — it is about choosing the **least wrong inductive bias** for the data at hand.*

---


