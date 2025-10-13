Here are **comprehensive, detailed notes on Isolation Forest**, structured for deep understanding — suitable for a Data Science or Machine Learning master-level study.

---

# **Isolation Forest (iForest)**

---

## **1. Overview**

**Isolation Forest** is an **anomaly detection algorithm** based on the principle that **anomalies are few and different**.
Instead of modeling the distribution of normal data points (as in density-based methods like LOF or One-Class SVM), Isolation Forest explicitly **isolates anomalies**.

It was proposed by **Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou** in 2008.

---

## **2. Core Intuition**

The main intuition:

* **Anomalies** are data points that are **rare** and **different** from most others.
* These points are **easier to isolate** because they require fewer random splits to separate them from the rest of the data.

Thus, the algorithm measures **how easy it is to isolate a point**:

* **Few splits → likely anomaly**
* **Many splits → likely normal**

This approach relies on **random partitioning** of data using **decision trees** (called *isolation trees*).

---

## **3. Working Mechanism**

### **Step-by-Step Process**

1. **Random Subsampling**
   Randomly select a small subset of the data (since anomalies are rare, large datasets make this efficient).

2. **Random Partitioning (Building Isolation Trees)**

   * Build a tree by **randomly selecting a feature**.
   * Then, **randomly selecting a split value** between the min and max values of that feature.
   * Recursively continue splitting until:

     * Each data point is isolated (i.e., in its own leaf node), or
     * The tree reaches a maximum height.

3. **Isolation Path Length**

   * The number of splits required to isolate a point is called its **path length (h(x))**.
   * Anomalies → short path length (easy to isolate).
   * Normal points → long path length (harder to isolate).

4. **Ensemble of Trees**

   * Build many isolation trees to form an **Isolation Forest**.
   * Compute the **average path length** across all trees for each data point.

5. **Anomaly Score Computation**

   * Convert average path length into an **anomaly score** ( s(x, n) ).

---

## **4. Mathematical Formulation**

### **Path Length**

Let:

* ( h(x) ): average path length of a point ( x ) over all trees
* ( c(n) ): average path length of unsuccessful searches in a Binary Search Tree of size ( n )

Then:

[
c(n) = 2H(n-1) - \frac{2(n-1)}{n}
]

where ( H(i) ) is the **harmonic number**, approximated as:

[
H(i) \approx \ln(i) + 0.5772156649 \quad (\text{Euler’s constant})
]

---

### **Anomaly Score**

The anomaly score for a point ( x ) is defined as:

[
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
]

Where:

* ( E(h(x)) ): mean path length of ( x )
* ( n ): number of samples used to build a tree

Interpretation:

* ( s(x, n) \approx 1 ): high likelihood of anomaly
* ( s(x, n) < 0.5 ): likely normal point

Thresholds can be tuned depending on the application.

---

## **5. Example (Intuitive Dry Run)**

### Example dataset:

| Point | Feature 1 | Feature 2 |
| ----- | --------- | --------- |
| A     | 1         | 2         |
| B     | 2         | 3         |
| C     | 3         | 4         |
| D     | 100       | 200       |

Here, D is clearly an outlier.

### Isolation process:

* Suppose we randomly pick `Feature 1` and split at `5`.
* D gets separated immediately (since 100 > 5).
* A, B, and C remain together (requiring more splits).

Thus:

* D has a short path (1 split) → **high anomaly score**
* A, B, C require more splits → **low anomaly score**

---

## **6. Computational Complexity**

Let:

* ( n ): number of samples
* ( t ): number of trees
* ( \psi ): subsample size per tree

Then:

* **Training time:** ( O(t \cdot \psi \cdot \log(\psi)) )
* **Testing time:** ( O(t \cdot \log(\psi)) )

Because isolation trees are constructed on **random subsamples**, Isolation Forest scales **linearly with data size**, making it efficient for large datasets.

---

## **7. Advantages**

| Feature                                  | Explanation                                                                      |
| ---------------------------------------- | -------------------------------------------------------------------------------- |
| **Efficiency**                           | Linear time complexity, good scalability to high-dimensional and large datasets. |
| **No need for distance/density metrics** | Avoids the computational cost of distance-based algorithms.                      |
| **Few assumptions**                      | Does not assume data distribution.                                               |
| **Low memory footprint**                 | Works on random subsamples.                                                      |
| **Works well in high dimensions**        | Unlike many clustering-based anomaly detectors.                                  |

---

## **8. Limitations**

| Limitation                         | Description                                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Randomness sensitivity**         | Random splits can sometimes miss anomalies if too few trees or small subsamples are used.       |
| **Performance on dense anomalies** | Not ideal when anomalies are not well-isolated (i.e., subtle contextual anomalies).             |
| **Feature scaling**                | Although less sensitive than distance-based methods, extreme feature scales can affect results. |
| **Interpretability**               | The model is somewhat opaque — it provides scores but not always intuitive explanations.        |

---

## **9. Practical Usage (Implementation Notes)**

### **Scikit-learn Implementation**

```python
from sklearn.ensemble import IsolationForest

# Initialize
iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.05,  # expected proportion of anomalies
    random_state=42
)

# Fit and predict
iso_forest.fit(X_train)
scores = iso_forest.decision_function(X_test)  # anomaly scores
labels = iso_forest.predict(X_test)  # -1 for anomalies, 1 for normal
```

### **Key Parameters**

| Parameter       | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `n_estimators`  | Number of isolation trees.                                          |
| `max_samples`   | Number of samples to draw to train each tree.                       |
| `contamination` | Expected proportion of outliers; helps set threshold automatically. |
| `max_features`  | Number of features to consider per split.                           |
| `bootstrap`     | Whether samples are drawn with replacement.                         |

---

## **10. Interpretation of Output**

* **`decision_function(X)`** → returns anomaly score (lower = more abnormal)
* **`predict(X)`** → returns labels:

  * `-1` → anomaly
  * `1` → normal

---

## **11. Comparison with Other Anomaly Detection Methods**

| Method                         | Core Idea                                         | Advantages                          | Limitations                                           |
| ------------------------------ | ------------------------------------------------- | ----------------------------------- | ----------------------------------------------------- |
| **Isolation Forest**           | Isolate anomalies using random partitions         | Fast, scalable, minimal assumptions | Less effective for subtle/contextual anomalies        |
| **One-Class SVM**              | Learn boundary around normal data                 | Works well for smooth boundaries    | Expensive for large datasets, sensitive to parameters |
| **Local Outlier Factor (LOF)** | Compare local density of a point to its neighbors | Good for local density anomalies    | Poor scalability, sensitive to neighborhood size      |
| **Autoencoder**                | Reconstruct normal patterns                       | Works with complex non-linear data  | Requires training and careful tuning                  |

---

## **12. Real-World Applications**

1. **Fraud Detection** – Credit card transactions, insurance claims.
2. **Network Intrusion Detection** – Detecting abnormal traffic patterns.
3. **Manufacturing Quality Control** – Identifying defective products.
4. **Sensor Data Analysis** – Detecting abnormal readings in IoT systems.
5. **Healthcare** – Detecting anomalous patient metrics.
6. **Finance** – Identifying irregular trading behaviors.

---

## **13. Extensions and Variants**

* **Extended Isolation Forest (EIF):**
  Uses **random hyperplanes** instead of axis-aligned splits, improving detection in high dimensions.

* **Streaming Isolation Forest:**
  Adapts to streaming data for online anomaly detection.

* **Hybrid Approaches:**
  Combine Isolation Forest with clustering or neural models for contextual anomaly detection.

---

## **14. Summary Table**

| Concept        | Key Idea                              |
| -------------- | ------------------------------------- |
| Goal           | Detect anomalies via random isolation |
| Principle      | Anomalies are easier to isolate       |
| Data Structure | Random binary trees (isolation trees) |
| Metric         | Average path length                   |
| Complexity     | O(n log n)                            |
| Output         | Anomaly score ( s(x, n) )             |
| Type           | Ensemble, unsupervised                |
| Works Best     | When anomalies are few and distinct   |

---

## **15. References**

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*.
   IEEE International Conference on Data Mining (ICDM).
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). *Isolation-based anomaly detection*.
   ACM Transactions on Knowledge Discovery from Data (TKDD).

---
Excellent — let’s go deep into the **mathematical derivation and intuition** behind the **average path length** term ( c(n) ), which is fundamental to how Isolation Forest normalizes anomaly scores.

---

# **Mathematical Derivation of Average Path Length in Isolation Forest**

---

## **1. Why We Need ( c(n) )**

When Isolation Forest calculates the **path length** ( h(x) ) for each data point, the absolute value of this path length depends on:

* The number of samples ( n ) used to build the tree, and
* How deeply the tree grows before isolating a point.

To compare path lengths across trees built on different sample sizes, we need to **normalize** ( h(x) ).
This normalization factor is ( c(n) ) — the **expected average path length** of an unsuccessful search in a **Binary Search Tree (BST)** of size ( n ).

---

## **2. Analogy with Binary Search Trees (BST)**

The structure of an Isolation Tree is mathematically similar to a **Binary Search Tree**, except:

* In BSTs, splits are determined by data ordering.
* In Isolation Forest, splits are chosen **randomly** (feature + threshold).

However, the **expected path length** for a random binary tree behaves similarly to that in a BST.
Thus, the **average path length of an unsuccessful search in a BST** is used as a theoretical baseline.

---

## **3. Expected Path Length in Random Binary Search Tree**

Let’s define:

* ( n ): number of data points (nodes)
* ( c(n) ): average path length of unsuccessful search (i.e., expected number of comparisons to isolate a point)

We can define a recursive relation for ( c(n) ).

### **Recursive relation**

When you have a BST with ( n ) nodes:

* The root divides the dataset into two subtrees.
* Each subtree contains ( i ) and ( n - i - 1 ) nodes, depending on where the split occurs.

The expected path length ( c(n) ) can be derived as:

[
c(n) = 1 + \frac{1}{n} \sum_{i=0}^{n-1} [c(i) + c(n - 1 - i)]
]

Here:

* The **1** accounts for the current comparison (root split).
* The summation averages the expected path lengths of all possible left-right splits.

---

## **4. Simplifying the Recurrence**

Using properties of harmonic numbers and expected values over random binary trees, Liu et al. (2008) derived the closed form:

[
c(n) = 2H(n-1) - \frac{2(n-1)}{n}
]

Where:
[
H(i) = \sum_{k=1}^{i} \frac{1}{k} \quad \text{(the (i^{th}) harmonic number)}
]

---

## **5. Approximation Using Logarithms**

For large ( n ), the harmonic number ( H(i) ) approximates:

[
H(i) \approx \ln(i) + \gamma
]

where ( \gamma \approx 0.5772156649 ) (Euler–Mascheroni constant).

Hence:

[
c(n) \approx 2\ln(n-1) + 2\gamma - \frac{2(n-1)}{n}
]

When ( n ) is large, the last term is negligible, and we can simplify to:

[
c(n) \approx 2\ln(n-1) + \text{constant}
]

---

## **6. Interpretation of ( c(n) )**

* ( c(n) ) represents the **average expected number of splits** needed to isolate a normal point in a dataset of size ( n ).
* Anomalies should have **path lengths significantly shorter** than ( c(n) ).

Thus, the normalized anomaly score:

[
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
]

has an **interpretable scale**:

* If ( E(h(x)) \ll c(n) ): ( s(x, n) ) → near 1 (strong anomaly)
* If ( E(h(x)) \approx c(n) ): ( s(x, n) ) → around 0.5 (normal)
* If ( E(h(x)) > c(n) ): ( s(x, n) ) → below 0.5 (very normal / dense region)

---

## **7. Step-by-Step Derivation (Expanded)**

Let’s formalize the recurrence for clarity.

### **Expected Path Length ( L(n) )**

Let ( L(n) ) denote the **average path length** for a tree of size ( n ).

When a root splits the data into two parts (of size ( i ) and ( n-i-1 )):

[
L(n) = L(i) + L(n - i - 1) + n - 1
]

Dividing by ( n ) (because each node is equally likely to be chosen as the root):

[
E[L(n)] = E[L(i)] + E[L(n - i - 1)] + (n - 1)
]

Now, we focus on **expected number of comparisons for unsuccessful search**, i.e., ( c(n) ), which leads to:

[
c(n) = c(i) + c(n - i - 1) + 2
]

Averaging over all ( i ) yields:

[
c(n) = \frac{2}{n} \sum_{i=1}^{n-1} c(i) + 2(n-1)
]

Solving this recurrence using known results from binary tree analysis leads to the harmonic number solution given earlier.

---

## **8. Verification for Small n**

Let’s check small values to confirm.

| n  | ( H(n-1) ) | ( c(n) = 2H(n-1) - 2(n-1)/n ) | Approx. Value |
| -- | ---------- | ----------------------------- | ------------- |
| 2  | 1          | ( 2(1) - 2(1)/2 = 1 )         | 1.0           |
| 3  | 1.5        | ( 3 - 4/3 = 1.6667 )          | 1.67          |
| 10 | 2.82897    | ( 5.6579 - 1.8 = 3.8579 )     | 3.86          |

As ( n ) increases, ( c(n) ) grows logarithmically — consistent with our intuition that deeper trees are needed for larger datasets.

---

## **9. Why Logarithmic Growth Matters**

The logarithmic nature of ( c(n) ) implies:

* **Normal data points** require path lengths proportional to ( \log(n) ).
* **Anomalies** have path lengths much smaller than ( \log(n) ).

Hence, the Isolation Forest effectively discriminates anomalies **without assuming any specific distribution** — it relies purely on isolation efficiency.

---

## **10. Summary of Key Insights**

| Concept       | Explanation                                                           |
| ------------- | --------------------------------------------------------------------- |
| ( c(n) )      | Expected average path length for a random binary tree of ( n ) points |
| Derived from  | Expected cost of an unsuccessful search in a BST                      |
| Formula       | ( c(n) = 2H(n-1) - \frac{2(n-1)}{n} )                                 |
| Approximation | ( c(n) \approx 2(\ln(n-1) + 0.5772) - \frac{2(n-1)}{n} )              |
| Purpose       | Normalizes path length to compute anomaly scores                      |
| Behavior      | Grows logarithmically with sample size ( n )                          |

---

## **11. Graphical Intuition (Conceptual)**

If you plot **expected path length vs. sample size (n)**:

* The curve increases **logarithmically**.
* Anomalies appear as points **far below the curve** (shorter path lengths).

This visualization confirms the principle:

> “The fewer splits needed to isolate a point, the more anomalous it is.”

---

## **12. Connection to Extended Isolation Forest (EIF)**

EIF modifies the splitting mechanism:

* Instead of axis-aligned random splits, it uses **random hyperplanes** (random slope + intercept).
* This avoids bias when features have different scales or correlations.

Mathematically, the normalization term ( c(n) ) remains the same, but path lengths become more consistent across high-dimensional, correlated data.

---
Excellent — here’s a curated set of **conceptual questions (with brief answer points)** on **Isolation Forest**, designed for **FAANG-level data science / ML interviews**.
These go from fundamental intuition → math → implementation → comparison and edge cases.

---

# **Isolation Forest – Conceptual Interview Questions**

---

## **1. Intuition & Fundamentals**

**Q1.** What is the main idea behind Isolation Forest?
**A:** Anomalies are few and different, so they can be isolated faster (i.e., require fewer random splits) than normal points.

---

**Q2.** How does Isolation Forest differ conceptually from density-based anomaly detection methods like LOF or One-Class SVM?
**A:**

* LOF/SVM try to **model normal regions** (density or boundary).
* Isolation Forest **isolates anomalies directly** without modeling normality or density.

---

**Q3.** Why is it called “Isolation” Forest?
**A:** Because it explicitly isolates each sample using random partitions — anomalies are separated (isolated) early in the partitioning process.

---

**Q4.** Why do anomalies have shorter path lengths in isolation trees?
**A:** They are rare and distant from other points, so a random split is more likely to separate them early.

---

**Q5.** Why is randomization essential in Isolation Forest?
**A:** Random feature and threshold selection ensures robustness and avoids overfitting. It also enables averaging over multiple trees for stable results.

---

## **2. Algorithmic Details**

**Q6.** How is an Isolation Tree constructed?
**A:**

* Randomly choose a feature.
* Randomly choose a split value between the feature’s min and max.
* Recurse until each instance is isolated or tree height limit is reached.

---

**Q7.** What is the meaning of “path length”?
**A:** The number of edges (splits) required to isolate a data point in a tree.

---

**Q8.** Why do we use subsampling (small subset per tree)?
**A:**

* Isolation depends more on relative position than on exact density.
* Small subsamples reduce computation and avoid biasing trees with redundant normal points.
* Empirically, subsamples of size 256 work well.

---

**Q9.** What happens if we increase the subsample size too much?
**A:**
Trees grow deeper; anomalies may require more splits → reduced sensitivity to anomalies, more computation, and possible overfitting.

---

**Q10.** How does the forest combine results from multiple trees?
**A:** The average path length of each point across all trees is used to compute an anomaly score.

---

## **3. Mathematical Concepts**

**Q11.** What does ( c(n) ) represent in the Isolation Forest algorithm?
**A:** The average path length for unsuccessful search in a Binary Search Tree of size ( n ); used to normalize path lengths.

---

**Q12.** Why does ( c(n) ) grow logarithmically with ( n )?
**A:** Because the average number of splits (comparisons) in a random binary tree increases roughly as ( O(\log n) ).

---

**Q13.** What is the formula for anomaly score?
**A:**
[
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
]

---

**Q14.** How do we interpret the anomaly score ( s(x, n) )?
**A:**

* ( s(x, n) \approx 1 ): anomaly (short path)
* ( s(x, n) \approx 0.5 ): normal point
* ( s(x, n) < 0.5 ): very normal / dense region

---

**Q15.** What does ( H(i) ) (harmonic number) signify in the formula?
**A:** It comes from the expected search cost in a binary tree, approximating the average depth of nodes.

---

## **4. Practical & Implementation-Level**

**Q16.** What are the most important hyperparameters in Isolation Forest?
**A:**

* `n_estimators` (number of trees)
* `max_samples` (subsample size per tree)
* `contamination` (expected proportion of anomalies)
* `max_features` (number of features per split)

---

**Q17.** How does the `contamination` parameter affect results?
**A:** It sets the threshold for what proportion of points are considered anomalies when converting scores to labels. Incorrect tuning can misclassify normal points as anomalies.

---

**Q18.** Is feature scaling necessary for Isolation Forest?
**A:** Usually not critical, but extremely imbalanced scales can bias splits. Normalization is recommended when features differ by several orders of magnitude.

---

**Q19.** What is the time complexity of Isolation Forest?
**A:**

* Training: ( O(t \cdot \psi \cdot \log \psi) ), where ( t ) = trees, ( \psi ) = subsample size.
* Prediction: ( O(t \cdot \log \psi) ).
  Linear with respect to dataset size — very efficient.

---

**Q20.** How does Isolation Forest handle high-dimensional data?
**A:** It handles it well compared to density-based methods, though correlations between features can still make isolation harder. Extended variants address this.

---

## **5. Theoretical & Analytical Questions**

**Q21.** Why does Isolation Forest not require distance or density measures?
**A:** It relies purely on recursive random partitioning — isolation happens by separation, not by measuring distances or densities.

---

**Q22.** Why is Isolation Forest considered ensemble-based?
**A:** It uses multiple random trees and averages their results (similar in spirit to Random Forests).

---

**Q23.** How does Isolation Forest differ from Random Forest?
**A:**

| Aspect          | Isolation Forest             | Random Forest                     |
| --------------- | ---------------------------- | --------------------------------- |
| Objective       | Anomaly detection            | Classification / Regression       |
| Split Criterion | Random feature and threshold | Information gain / Gini impurity  |
| Output          | Anomaly score                | Class label or numeric prediction |

---

**Q24.** What is the effect of correlated features on Isolation Forest?
**A:** Random axis-aligned splits may be less effective at isolating anomalies lying along correlated directions → motivates the Extended Isolation Forest (EIF).

---

**Q25.** Why can Isolation Forest fail on subtle (contextual) anomalies?
**A:** If anomalies lie within normal regions but deviate only under specific contexts (e.g., “temperature high at night”), random partitions may not expose these patterns.

---

## **6. Comparison & Extensions**

**Q26.** What are the main advantages of Isolation Forest over One-Class SVM?
**A:**

* Faster and more scalable
* Less sensitive to parameter tuning
* Works well with high-dimensional data
* Doesn’t require kernel functions

---

**Q27.** When would you prefer LOF over Isolation Forest?
**A:** When local density variations matter, e.g., in datasets with clusters of different densities where anomalies are defined relative to local neighbors.

---

**Q28.** What is the Extended Isolation Forest (EIF)?
**A:** A variant that uses **random hyperplanes** (not axis-aligned splits), improving detection for correlated or rotated data.

---

**Q29.** Does Isolation Forest handle streaming data?
**A:** The original algorithm does not, but there are **Streaming Isolation Forest** variants that adapt incrementally to new data.

---

**Q30.** Can Isolation Forest handle categorical variables?
**A:** Not directly. You need to encode them numerically (e.g., one-hot or ordinal encoding) before training.

---

## **7. Edge Cases & Interview Scenarios**

**Q31.** What happens if all points are identical?
**A:** All points will have equal and maximal path lengths — the model cannot detect anomalies (score ~0.5 for all).

---

**Q32.** What happens if you have no anomalies in your data?
**A:** The model still assigns scores, but thresholds based on `contamination` can misclassify some normal points as anomalies unless tuned properly.

---

**Q33.** What happens if anomalies form a dense cluster far from normal data?
**A:** The algorithm can still detect them because they’re isolated early — though if the cluster is large, it may start being treated as a separate “normal” region.

---

**Q34.** How does random seed affect Isolation Forest results?
**A:** Different random splits lead to slightly different trees; setting `random_state` ensures reproducibility.

---

**Q35.** Why do we use the logarithmic base 2 in the anomaly score formula?
**A:** Because binary splits halve the search space — base 2 aligns with the structure of binary trees.

---

## **8. Discussion/Advanced-Level**

**Q36.** How could you adapt Isolation Forest for semi-supervised anomaly detection?
**A:** Use labeled normal samples to bias splits or tune thresholds on validation sets containing known anomalies.

---

**Q37.** Could Isolation Forest be used for feature selection?
**A:** Indirectly — by observing feature importance from split frequencies or path-length variation per feature.

---

**Q38.** How would you visualize Isolation Forest results?
**A:**

* Scatter plot colored by anomaly score (2D case).
* Histogram of scores to identify cutoff threshold.
* Average path length vs. instance index to see outliers.

---

**Q39.** What would happen if you train Isolation Forest on data with missing values?
**A:** Most implementations (like scikit-learn) cannot handle NaNs directly — you must impute or remove missing data.

---

**Q40.** How can you validate the performance of an Isolation Forest model?
**A:**

* If labels available → use precision/recall, ROC-AUC.
* If not → visual inspection of top anomalies or synthetic anomaly injection tests.

---

# **Summary Table – Interview Focus**

| Theme         | Must-Know Points                                 |
| ------------- | ------------------------------------------------ |
| Core idea     | Anomalies are easier to isolate                  |
| Mechanism     | Random trees, random splits                      |
| Key metric    | Path length (short = anomaly)                    |
| Normalization | ( c(n) = 2H(n-1) - 2(n-1)/n )                    |
| Complexity    | Linear, scalable                                 |
| Weakness      | Subtle/contextual anomalies, correlated features |
| Extension     | Extended Isolation Forest (EIF)                  |
| Parameters    | `n_estimators`, `max_samples`, `contamination`   |

---

Here are some **useful images/diagrams** that visually explain **Isolation Forest** — their mechanisms, intuition, and comparisons — all in **Markdown format** (so you can directly embed them in notes or notebooks).

---

### **1. Isolation Forest – Intuitive Explanation (Isolation Paths)**

Shows how anomalies get isolated earlier (shorter paths) than normal points.

```markdown
![Isolation Forest Intuition](https://miro.medium.com/v2/resize:fit:1400/1*1fY4xE8U5K8T3NjtmXuzUg.png)
```

Source: *Medium – Understanding Isolation Forest Algorithm*

---

### **2. Isolation Tree Example**

Visual of a single isolation tree illustrating random splits.

```markdown
![Isolation Tree Example](https://editor.analyticsvidhya.com/uploads/82452if3.png)
```

Source: *Analytics Vidhya – Isolation Forest Explained*

---

### **3. Path Length vs Anomaly Score**

Illustrates how anomalies have shorter average path lengths (and thus higher anomaly scores).

```markdown
![Path Length Visualization](https://miro.medium.com/v2/resize:fit:1200/1*QtvIx_wfTf2ShvP4aD1zWg.png)
```

Source: *Medium – Isolation Forest Explained Visually*

---

### **4. Isolation Forest vs LOF / One-Class SVM**

Comparison plot showing decision boundaries and detected anomalies for different methods.

```markdown
![Isolation Forest vs Other Methods](https://scikit-learn.org/stable/_images/sphx_glr_plot_anomaly_comparison_001.png)
```

Source: *Scikit-learn Documentation*

---

### **5. Extended Isolation Forest (EIF) Concept**

Demonstrates random hyperplane splits instead of axis-aligned cuts.

```markdown
![Extended Isolation Forest Hyperplanes](https://miro.medium.com/v2/resize:fit:1200/1*uzmShNfvhVErVEfLG8bUgg.png)
```

Source: *Medium – Extended Isolation Forest Article*

---
