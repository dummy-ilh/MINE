Excellent 👌 Here are the **Probability Axioms and Basic Rules** — clean, complete, and Markdown-friendly.

---

## 🎯 **Axioms of Probability**

Let ( S ) be the sample space and ( A, B ) be any events.

---

### **Axiom 1 — Non-Negativity**

[
P(A) \ge 0
]

> Probability of any event is never negative.

---

### **Axiom 2 — Normalization**

[
P(S) = 1
]

> The probability of the entire sample space (a certain event) is 1.

---

### **Axiom 3 — Additivity**

If two events ( A ) and ( B ) are **mutually exclusive** (disjoint),
[
A \cap B = \varnothing
]
then
[
P(A \cup B) = P(A) + P(B)
]

> For disjoint events, the probability of either occurring is the sum of their probabilities.

---

## 📘 **Derived Rules (from Axioms)**

These are direct consequences of the axioms.

---

### **1️⃣ Probability of the Impossible Event**

[
P(\varnothing) = 0
]

> The probability of an impossible event is zero.

---

### **2️⃣ Complement Rule**

[
P(A') = 1 - P(A)
]

> The probability of “not A” equals 1 minus the probability of A.

---

### **3️⃣ General Addition Rule**

For **any** two events (A) and (B) (not necessarily disjoint):
[
P(A \cup B) = P(A) + P(B) - P(A \cap B)
]

---

### **4️⃣ Subset Rule**

If ( A \subseteq B ), then
[
P(A) \le P(B)
]

---

### **5️⃣ Range Rule**

For any event (A):
[
0 \le P(A) \le 1
]

---

### **6️⃣ Conditional Probability**

[
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{if } P(B) > 0
]

> Probability of A given that B has occurred.

---

### **7️⃣ Multiplication Rule**

[
P(A \cap B) = P(A) \times P(B|A)
]

> Used when events are dependent.

If **independent**, then:
[
P(A \cap B) = P(A) \times P(B)
]

---

### **8️⃣ Law of Total Probability**

If ( A_1, A_2, \dots, A_n ) are **mutually exclusive and exhaustive** events, then:
[
P(B) = \sum_{i=1}^{n} P(B|A_i)P(A_i)
]

---

### **9️⃣ Bayes’ Theorem**

[
P(A_i|B) = \frac{P(B|A_i)P(A_i)}{\sum_{j=1}^{n} P(B|A_j)P(A_j)}
]

> Used to **reverse conditional probabilities**.

---

### **🔟 Inclusion–Exclusion for 3 Events**

[
\begin{aligned}
P(A \cup B \cup C) &= P(A) + P(B) + P(C) \
&\quad - [P(A \cap B) + P(B \cap C) + P(C \cap A)] \
&\quad + P(A \cap B \cap C)
\end{aligned}
]

---

### ✅ **Summary Table**

| Rule                | Formula                             | Applies When        |                  |              |                       |
| ------------------- | ----------------------------------- | ------------------- | ---------------- | ------------ | --------------------- |
| Complement          | (P(A') = 1 - P(A))                  | Always              |                  |              |                       |
| Addition (disjoint) | (P(A ∪ B) = P(A) + P(B))            | A and B disjoint    |                  |              |                       |
| Addition (general)  | (P(A ∪ B) = P(A) + P(B) - P(A ∩ B)) | Any A, B            |                  |              |                       |
| Multiplication      | (P(A ∩ B) = P(A)P(B                 | A))                 | General          |              |                       |
| Independence        | (P(A ∩ B) = P(A)P(B))               | If A, B independent |                  |              |                       |
| Total Probability   | (P(B) = ΣP(B                        | A_i)P(A_i))         | Exhaustive A_i   |              |                       |
| Bayes’ Theorem      | (P(A_i                              | B) = \frac{P(B      | A_i)P(A_i)}{ΣP(B | A_j)P(A_j)}) | Posterior probability |

---

Would you like me to follow this with **numerical examples** applying each of these rules (Axiom use, complement, addition, conditional, etc.) next?
