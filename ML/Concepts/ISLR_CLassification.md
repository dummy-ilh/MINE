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

**Would you like me to generate a summary table specifically for the 4.6 Lab (R/Python) commands used to run these models?**
