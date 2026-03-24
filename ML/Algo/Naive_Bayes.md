# 🧠 Naïve Bayes Classifier — Detailed Notes

---

## 🌍 1. Overview

**Naïve Bayes** is a family of **probabilistic classifiers** based on **Bayes’ theorem** with a **“naïve” independence assumption** between features.

Common applications:
- Spam filtering  
- Sentiment analysis  
- Document classification  
- Medical diagnosis  
- Real-time prediction tasks  

Despite its simplicity, Naïve Bayes often performs **surprisingly well**, especially in **text classification**.

---

## 📘 2. Bayes’ Theorem Refresher

\[
P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}
\]

- **P(H|E)** — Posterior (probability of hypothesis after seeing evidence)  
- **P(E|H)** — Likelihood (probability of evidence given hypothesis)  
- **P(H)** — Prior (belief before seeing evidence)  
- **P(E)** — Evidence (acts as a normalizing constant)

---

## 🧩 3. Applying to Classification

Let:
- \( x = (x_1, x_2, ..., x_n) \): feature vector  
- \( C_k \): class label  

\[
P(C_k | x) = \frac{P(x | C_k) \cdot P(C_k)}{P(x)}
\]

Since \( P(x) \) is constant for all classes:

\[
P(C_k | x) \propto P(C_k) \cdot P(x | C_k)
\]

---

## ⚙️ 4. The “Naïve” Independence Assumption

Assume features are **conditionally independent** given the class:

\[
P(x | C_k) = \prod_{i=1}^{n} P(x_i | C_k)
\]

Then:

\[
P(C_k | x) \propto P(C_k) \prod_{i=1}^{n} P(x_i | C_k)
\]

---

## 🧮 5. Classification Rule

Choose class with maximum posterior:

\[
\hat{C} = \arg\max_{C_k} \; P(C_k) \prod_{i=1}^{n} P(x_i | C_k)
\]

To avoid underflow, use log probabilities:

\[
\hat{C} = \arg\max_{C_k} \; \log P(C_k) + \sum_{i=1}^{n} \log P(x_i | C_k)
\]

---

## 💡 6. Types of Naïve Bayes Classifiers

| Variant | Feature Type | Probability Distribution |
|----------|---------------|--------------------------|
| **Multinomial NB** | Word counts / frequencies | Multinomial |
| **Bernoulli NB** | Binary features | Bernoulli |
| **Gaussian NB** | Continuous features | Normal (Gaussian) |
| **Categorical NB** | Discrete categorical features | Categorical |

---

## 📚 7. Example — Spam Classification

Classes:
- \( C_1 = \text{Spam} \)
- \( C_2 = \text{Not Spam} \)

Vocabulary = {win, money, free, hello}

| Word | P(word | Spam) | P(word | NotSpam) |
|------|-----------|--------------|
| win | 0.4 | 0.05 |
| money | 0.3 | 0.1 |
| free | 0.3 | 0.05 |
| hello | 0.05 | 0.5 |

Assume \( P(Spam)=0.4, P(NotSpam)=0.6 \)

Email: **“win money free”**

\[
P(Spam|email) \propto 0.4 \times 0.4 \times 0.3 \times 0.3 = 0.0144
\]
\[
P(NotSpam|email) \propto 0.6 \times 0.05 \times 0.1 \times 0.05 = 0.00015
\]

✅ Classified as **Spam**

---

## 🧮 8. Handling Zero Probabilities — Laplace Smoothing

Zero probabilities kill the product.  
Use **Laplace (add-one) smoothing**:

\[
P(x_i | C_k) = \frac{count(x_i, C_k) + 1}{\text{total words in } C_k + V}
\]
where \( V \) = vocabulary size.

---

## 📊 9. Gaussian Naïve Bayes (for continuous features)

Assume each feature \( x_i \) follows a Gaussian distribution per class:

\[
P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} 
\exp\left(-\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}\right)
\]

Estimate \( \mu_{k,i} \) and \( \sigma_{k,i} \) from training data.

---

## 🧠 10. Why It Works Well Despite Being “Naïve”

- Exact independence rarely holds, but:
  - It still ranks probabilities well for classification.
  - Correlations often cancel each other.
  - High-dimensional data (like text) dilutes dependencies.

---

## 📈 11. Complexity

| Step | Complexity |
|------|-------------|
| Training | \( O(N \cdot d) \) |
| Prediction | \( O(K \cdot d) \) |
| Memory | \( O(K \cdot d) \) |

Extremely efficient for large datasets.

---

## ⚖️ 12. Pros & Cons

### ✅ Pros
- Simple and fast  
- Works well with high-dimensional data  
- Requires little data  
- Robust to irrelevant features  
- Probabilistic output

### ❌ Cons
- Strong independence assumption  
- Poor with correlated features  
- Sensitive to zero probabilities  
- Requires distribution assumption for continuous data

---

## 🧰 13. Python Implementation (Scikit-Learn)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Sample data
docs = ["free money win", "hello friend", "free offer", "hello you"]
labels = ["spam", "ham", "spam", "ham"]

# Convert text to count vectors
cv = CountVectorizer()
X = cv.fit_transform(docs)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Predict new message
test = cv.transform(["free win"])
print(model.predict(test))


---

## 🧩 14. Comparison Table — NB vs Others

| Algorithm           | Nature         | Key Strength                |
| ------------------- | -------------- | --------------------------- |
| Naïve Bayes         | Probabilistic  | Fast, interpretable         |
| Logistic Regression | Discriminative | Handles correlated features |
| Decision Tree       | Non-parametric | Handles nonlinear patterns  |
| SVM                 | Margin-based   | Good for complex boundaries |

---

## 🧪 15. Real-World Applications

| Domain     | Use Case              |
| ---------- | --------------------- |
| Email      | Spam classification   |
| Healthcare | Disease diagnosis     |
| Sentiment  | Review classification |
| Finance    | Fraud detection       |
| NLP        | Topic tagging         |

---

## 💭 16. Common Interview Questions

1. **What is the main assumption in Naïve Bayes?**
   → Conditional independence of features given the class.

2. **How is zero probability handled?**
   → Laplace (add-one) smoothing.

3. **Difference between Multinomial and Bernoulli NB?**
   → Multinomial uses counts; Bernoulli uses binary presence.

4. **Why is Naïve Bayes fast?**
   → Only needs to count frequencies, not optimize weights.

5. **When does NB perform poorly?**
   → When features are highly correlated or continuous but non-Gaussian.

6. **What is the log probability trick?**
   → Summing log probabilities instead of multiplying raw ones to avoid underflow.

---

## 🔍 17. Quick Summary Formula Sheet

| Concept             | Formula                                                            |                    |                 |
| ------------------- | ------------------------------------------------------------------ | ------------------ | --------------- |
| Bayes’ theorem      | ( P(C                                                              | x) = \frac{P(x     | C)P(C)}{P(x)} ) |
| Naïve assumption    | ( P(x                                                              | C) = \prod_i P(x_i | C) )            |
| Decision rule       | ( \hat{C} = \arg\max_C P(C)\prod_i P(x_i                           | C) )               |                 |
| Laplace smoothing   | ( \frac{count + 1}{total + V} )                                    |                    |                 |
| Gaussian likelihood | ( \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} ) |                    |                 |

---

