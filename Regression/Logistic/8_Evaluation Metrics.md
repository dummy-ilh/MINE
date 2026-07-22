

# Module 8 — Evaluation Metrics Deep Dive

## 1. WHY: The Accuracy Trap

Once your classification model is trained, asking **"Is this model actually good?"** cannot be answered by accuracy alone.

> **The Imbalanced Data Trap:**
> Imagine a fraud detection system where only 1 in 1,000 transactions is fraudulent ($0.1\%$ positive class). A trivial model that predicts **"not fraud"** for every transaction achieves **99.9% accuracy** while catching **0% of actual fraud**.

Accuracy treats all errors equally and assumes a balanced class distribution. In real-world scenarios (fraud detection, medical diagnosis, spam filtering), accuracy hides model failure behind dominant negative classes.

---

## 2. INTUITION: The Confusion Matrix

Every binary classification prediction falls into one of four distinct outcomes:

```
                  ACTUAL CLASS
                 Positive    Negative
PREDICTED  Pos |    TP    |    FP    |  (Type I Error / False Alarm)
CLASS      Neg |    FN    |    TN    |  (Type II Error / Miss)

```

### Real-World Analogy: Security Guard at a Club

* **True Positive (TP):** Correctly stops an underage guest *(flagged positive, was positive)*.
* **True Negative (TN):** Correctly lets in an adult *(flagged negative, was negative)*.
* **False Positive (FP) — Type I Error:** Wrongly stops an adult *(false alarm: annoys a legitimate guest)*.
* **False Negative (FN) — Type II Error:** Wrongly lets in an underage guest *(missed catch: safety/legal risk)*.

---

## 3. CORE METRICS: Precision, Recall, F1

### Metric Summary & Formulas

| Metric | Core Question | Formula | When to Prioritize |
| --- | --- | --- | --- |
| **Accuracy** | *Overall, what fraction was correct?* | $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

 | Symmetric costs & balanced classes only. |
| **Precision** | *Of everything I flagged as positive, how many were right?* | $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

 | **Costly False Positives** (e.g., spam filtering, blocking legitimate users). |
| **Recall** *(Sensitivity)* | *Of all actual positives, how many did I catch?* | $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

 | **Costly False Negatives** (e.g., medical screenings, fraud detection). |
| **F1 Score** | *What is the balanced single-number performance?* | $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

 | When both FP and FN matter under class imbalance. |

> **Why Harmonic Mean for F1?**
> A standard arithmetic mean of Precision ($1.0$) and Recall ($0.01$) yields $0.505$ (misleadingly acceptable). The **harmonic mean** heavily punishes extreme imbalances, pulling the overall score down to $0.0198$.

---

## 4. WORKED NUMERIC EXAMPLE

**Scenario:** 1,000 transactions containing **10 actual fraud cases** (1% imbalance).

* $\text{TP} = 7$ *(caught 7 of 10 real frauds)*
* $\text{FN} = 3$ *(missed 3 real frauds)*
* $\text{FP} = 20$ *(wrongly flagged 20 legitimate users)*
* $\text{TN} = 970$ *(correctly identified 970 legitimate users)*

$$\text{Accuracy} = \frac{7 + 970}{1000} = 97.7\%$$

$$\text{Precision} = \frac{7}{7 + 20} = \frac{7}{27} \approx 25.9\% \quad \text{(Only 1 in 4 alerts is real)}$$

$$\text{Recall} = \frac{7}{7 + 3} = \frac{7}{10} = 70.0\% \quad \text{(30\% of fraud slipped through)}$$

$$\text{F1 Score} = 2 \cdot \frac{0.259 \times 0.700}{0.259 + 0.700} \approx 37.8\%$$

> **Key Takeaway:** Accuracy ($97.7\%$) presents a false sense of security, whereas F1 ($37.8\%$) reveals significant operational noise (high false alarms and missed fraud).

---

## 5. ROC-AUC vs. PR-AUC

Both metrics evaluate model ranking ability across **all decision thresholds** rather than a single fixed cutoff.

```
ROC-AUC:  Plots True Positive Rate (Recall) vs False Positive Rate [FP / (FP + TN)]
PR-AUC:   Plots Precision [TP / (TP + FP)] vs Recall [TP / (TP + FN)]

```

### When to Use Which?

* **ROC-AUC:** Ideal for **balanced datasets**. Interpreted as the probability that a randomly chosen positive instance gets a higher predicted score than a randomly chosen negative instance.
* **PR-AUC:** Essential for **heavily imbalanced datasets**. ROC-AUC uses True Negatives in its False Positive Rate denominator ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). A massive $\text{TN}$ count inflates the ROC-AUC score, masking poor precision. PR-AUC excludes $\text{TN}$ entirely.

---

## 6. CALIBRATION

A model can rank instances perfectly (high AUC) while outputting inaccurate raw probabilities.

* **Definition:** A model is **well-calibrated** if a predicted probability of $0.70$ means the event actually occurs $70\%$ of the time across similar predictions.
* **Business Impact:** If business actions scale directly with predicted probability (e.g., allocating dollar amounts to retain churn-risk customers), bad calibration leads to misallocated capital—even if the ranking order is correct.
* **Diagnostic:** A **Reliability Diagram (Calibration Curve)** plots predicted probability bins against actual positive frequencies. A calibrated model follows a $45^\circ$ diagonal line.

---

## 7. FAANG L5 INTERVIEW CONTEXT & CHEAT SHEET

### Standard Interview Questions & Answers

#### Q: "Why is accuracy misleading for imbalanced datasets?"

> *"Accuracy measures overall correctness, allowing the majority class to dominate the score. Predicting the majority class exclusively yields near-perfect accuracy while completely failing to detect the minority target class."*

#### Q: "When should you prefer PR-AUC over ROC-AUC?"

> *"PR-AUC is preferred when the positive class is rare. ROC-AUC incorporates True Negatives in its denominator ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$), causing a large volume of negative cases to artificially suppress FPR and exaggerate performance."*

#### Q: "What does an AUC score of 0.3 mean?"

> *"An AUC of 0.5 indicates no ranking ability (random chance). An AUC of 0.3 indicates an informative model that is systematically inverted—flipping its predicted probability outputs yields an effective AUC of 0.7."*

#### Q: "Can a model have an AUC of 0.95 but be unsuitable for probability-based decision-making?"

> *"Yes. High AUC indicates strong ranking ability, but if predictions are poorly calibrated (e.g., outputting 0.99 for actual 0.60 risks), downstream financial and operational thresholds relying on exact probability values will fail."*

---

## 8. INTERACTIVE EVALUATION METRICS CALCULATOR

Adjust the raw Confusion Matrix counts below to observe real-time impacts on Accuracy, Precision, Recall, and F1 Score under varying levels of class imbalance.

---

## 9. PYTHON IMPLEMENTATION

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Sample confusion matrix components (Fraud Example)
TP, FN, FP, TN = 7, 3, 20, 970

# Metric computation via raw counts
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy:  {accuracy:.3f}")   # 0.977
print(f"Precision: {precision:.3f}")  # 0.259
print(f"Recall:    {recall:.3f}")     # 0.700
print(f"F1 Score:  {f1:.3f}")         # 0.378

```
