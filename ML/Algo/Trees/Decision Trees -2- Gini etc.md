# Gini vs Entropy vs Log Loss vs Information Gain
## Complete Notes with Worked Examples

---

## The Big Picture — What Are These?

These are all **impurity / uncertainty measures** used to decide how to split a node in a Decision Tree.

```
At every node, the tree asks:
"Which feature + threshold reduces impurity the MOST?"

Impurity Measures answer: "How mixed/uncertain is this set of labels?"
  → Pure node (all one class)   = impurity 0
  → Perfectly mixed (50/50)     = impurity is maximum
```

| Measure | Used As | Role |
|---|---|---|
| **Gini Impurity** | Split criterion | Measures misclassification probability |
| **Entropy** | Split criterion | Measures information / uncertainty |
| **Information Gain (IG)** | Split selector | Reduction in entropy after a split |
| **Log Loss** | Split criterion (sklearn) | Probabilistic measure of uncertainty |

> ⚠️ Information Gain is **not** a standalone impurity measure — it is the **difference** in entropy before and after a split. It's the *score* used to evaluate splits when entropy is the criterion.

---

## 1. Gini Impurity

### Formula

$$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$

- $p_i$ = fraction of samples belonging to class $i$
- $C$ = number of classes

### Properties
- Range: **[0, 0.5]** for binary; **[0, 1 − 1/C]** in general
- **0** = perfectly pure (all one class)
- **0.5** = max impurity for binary (50-50 split)
- **No logarithm** → computationally cheap
- Geometrically: measures the probability that two randomly picked samples are from **different** classes

### Intuition
> "If I randomly pick two samples from this node, what's the chance they have different labels?"  
> Gini = probability of that mismatch.

---

### ✏️ Worked Example — Gini

**Dataset:** 10 samples at a node — 6 Class A, 4 Class B

**Step 1: Compute parent Gini**
$$p_A = 0.6, \quad p_B = 0.4$$
$$\text{Gini}(S) = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = \mathbf{0.48}$$

**Step 2: Evaluate a split — Feature "Age < 30"**
- Left child (6 samples): 5A, 1B
- Right child (4 samples): 1A, 3B

$$\text{Gini}(L) = 1 - \left(\left(\frac{5}{6}\right)^2 + \left(\frac{1}{6}\right)^2\right) = 1 - (0.694 + 0.028) = \mathbf{0.278}$$

$$\text{Gini}(R) = 1 - \left(\left(\frac{1}{4}\right)^2 + \left(\frac{3}{4}\right)^2\right) = 1 - (0.0625 + 0.5625) = \mathbf{0.375}$$

**Step 3: Weighted Gini after split**
$$\text{Gini}_{split} = \frac{6}{10}(0.278) + \frac{4}{10}(0.375) = 0.167 + 0.150 = \mathbf{0.317}$$

**Step 4: Gini Reduction (equivalent of IG)**
$$\Delta\text{Gini} = 0.48 - 0.317 = \mathbf{0.163}$$

✅ Gini dropped by 0.163 — this is a good split.

---

## 2. Entropy

### Formula

$$\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

- Convention: $0 \cdot \log_2(0) = 0$
- Range: **[0, log₂(C)]** — binary: [0, 1]
- **0** = pure node
- **1** = maximum uncertainty (binary 50-50)

### Properties
- Rooted in **information theory** (Shannon, 1948)
- More **sensitive to class probability changes** than Gini (log amplifies small differences)
- Computationally **slower** due to log
- Symmetric — measures average "surprise" across all classes

### Intuition
> "How many bits of information do I need to encode the class label of a randomly chosen sample?"  
> High entropy = need many bits = very uncertain.

---

### ✏️ Worked Example — Entropy

**Same dataset:** 6A, 4B out of 10

**Step 1: Parent Entropy**
$$\text{Entropy}(S) = -(0.6 \log_2 0.6 + 0.4 \log_2 0.4)$$
$$= -(0.6 \times (-0.737) + 0.4 \times (-1.322))$$
$$= -(-0.442 - 0.529) = \mathbf{0.971 \text{ bits}}$$

**Step 2: Child Entropies (same split: Left=5A,1B | Right=1A,3B)**

$$\text{Entropy}(L) = -\left(\frac{5}{6}\log_2\frac{5}{6} + \frac{1}{6}\log_2\frac{1}{6}\right) = -(0.833 \times (-0.263) + 0.167 \times (-2.585))$$
$$= -(-0.219 - 0.431) = \mathbf{0.650}$$

$$\text{Entropy}(R) = -\left(\frac{1}{4}\log_2\frac{1}{4} + \frac{3}{4}\log_2\frac{3}{4}\right) = -(0.25 \times (-2) + 0.75 \times (-0.415))$$
$$= -(-0.500 - 0.311) = \mathbf{0.811}$$

**Step 3: Weighted Entropy after split**
$$H_{split} = \frac{6}{10}(0.650) + \frac{4}{10}(0.811) = 0.390 + 0.324 = \mathbf{0.714}$$

---

## 3. Information Gain (IG)

### Formula

$$\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in A} \frac{|S_v|}{|S|} \cdot \text{Entropy}(S_v)$$

IG is simply **how much entropy is removed** by making a split. It always uses entropy — it is **not** a separate impurity measure.

### ✏️ Worked Example — IG (continuing from above)

$$\text{IG} = \text{Entropy}(S) - H_{split} = 0.971 - 0.714 = \mathbf{0.257 \text{ bits}}$$

> The split on "Age < 30" gives us 0.257 bits of Information Gain.  
> We'd compare this against all other possible splits and pick the one with **highest IG**.

### ⚠️ Problem with IG: Bias toward High-Cardinality Features

IG tends to favour features with **many unique values** (e.g., ID, timestamp), because more branches = more ways to make pure leaves.

**Fix → Gain Ratio (used in C4.5 algorithm):**
$$\text{Gain Ratio} = \frac{\text{IG}(S, A)}{\text{SplitInfo}(A)}$$

$$\text{SplitInfo}(A) = -\sum_{v} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

Penalises features that create many small branches.

---

## 4. Log Loss (as a Split Criterion)

### Formula

$$\text{LogLoss}(S) = -\sum_{i=1}^{C} p_i \log(p_i)$$

> 🔑 **This is mathematically identical to Entropy** — just using natural log (ln) instead of log₂.

$$\text{Entropy (bits)} = -\sum p_i \log_2 p_i = \frac{-\sum p_i \ln p_i}{\ln 2} = \frac{\text{LogLoss}}{\ln 2}$$

### Why does sklearn have `criterion='log_loss'` separately?

In sklearn ≥ 1.0, `log_loss` was added as an **explicit alias** for entropy that:
- Uses **natural log** (ln) instead of log₂
- Is the **same functional form** — produces identical trees
- Is aligned with the `log_loss` evaluation metric used in probabilistic classifiers
- Makes it explicit you're optimising for **probabilistic calibration**, not just accuracy

### ✏️ Worked Example — Log Loss

**Same node: 6A, 4B → p_A = 0.6, p_B = 0.4**

$$\text{LogLoss} = -(0.6 \ln 0.6 + 0.4 \ln 0.4)$$
$$= -(0.6 \times (-0.511) + 0.4 \times (-0.916))$$
$$= -(-0.307 - 0.366) = \mathbf{0.673 \text{ nats}}$$

Compare: Entropy = 0.971 bits. Converting:
$$0.673 \div \ln 2 = 0.673 \div 0.693 = \mathbf{0.971} ✅$$

They are the same — just different units (nats vs bits).

---

## Side-by-Side Comparison on the Same Node

Node: **6A, 4B** (out of 10 samples)

| Measure | Formula Output | Value |
|---|---|---|
| Gini | $1 - (0.6^2 + 0.4^2)$ | **0.480** |
| Entropy | $-(0.6\log_2 0.6 + 0.4\log_2 0.4)$ | **0.971** |
| Log Loss | $-(0.6\ln 0.6 + 0.4\ln 0.4)$ | **0.673** |

All three peak at **p = 0.5** (50-50 split) and hit **0** at pure nodes.  
They are just **different scales** of the same underlying idea.

---

## Sensitivity Comparison

For a binary node with $p$ = fraction of class 1:

| $p$ | Gini | Entropy (scaled to 0-0.5) | Notes |
|---|---|---|---|
| 0.5 | 0.500 | 0.500 | Max impurity |
| 0.4 | 0.480 | 0.471 | Nearly same |
| 0.3 | 0.420 | 0.422 | Nearly same |
| 0.1 | 0.180 | 0.217 | Entropy more sensitive near extremes |
| 0.01 | 0.020 | 0.057 | Entropy penalises impurity more at low p |

> **Key insight:** Entropy is more sensitive to small impurities (near 0 or 1).  
> Gini is flatter — treats moderate impurity similarly across a wider range.

---

## Which is Better?

### Short answer
> **In practice: negligible difference. Default to Gini.**

### Detailed breakdown

| Criterion | When to prefer it |
|---|---|
| **Gini** | Default. Fast, no log. Works for almost everything. Less sensitive to outlier class distributions |
| **Entropy / IG** | When you need balanced splits, working with multi-class problems, or care about information-theoretic interpretation |
| **Log Loss** | When you want probabilistic output calibration; equivalent to entropy |
| **Gain Ratio** | When you have high-cardinality categoricals and IG is biased toward them (used in C4.5) |

### Research-backed verdict
Multiple studies (Raileanu & Stoffel, 2004) tested Gini vs Entropy on 32 datasets:  
- They agreed on the split **98.8% of the time**  
- When they disagreed, neither was consistently better  
- **Conclusion:** The choice barely matters — focus on `max_depth` and `min_samples_leaf` instead

### FAANG answer template
> "Gini and Entropy produce nearly identical trees — they agree on the best split ~99% of the time. I'd default to Gini for speed. I'd choose Entropy if I wanted to reason in information-theoretic terms or was working in a framework like ID3/C4.5 where IG is the native criterion. Log Loss is mathematically entropy with natural log — useful when probabilistic calibration matters."

---

## Algorithm Lineage

| Algorithm | Criterion Used |
|---|---|
| **ID3** | Information Gain (Entropy) |
| **C4.5** | Gain Ratio (fixes IG's cardinality bias) |
| **CART** | Gini (sklearn default) |
| **C5.0** | Gain Ratio (improved C4.5) |
| **sklearn** | Gini (default), Entropy, or Log Loss |

---

## Full Comparison Table

| Property | Gini | Entropy | Log Loss | Info Gain |
|---|---|---|---|---|
| Type | Impurity measure | Impurity measure | Impurity measure | Split quality score |
| Formula | $1 - \sum p_i^2$ | $-\sum p_i \log_2 p_i$ | $-\sum p_i \ln p_i$ | $H(S) - H_{after}$ |
| Range (binary) | [0, 0.5] | [0, 1] | [0, 0.693] | [0, 1] |
| Log needed? | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Computation speed | ⚡ Fastest | Medium | Medium | Medium |
| Sensitivity at extremes | Lower | Higher | Higher | N/A |
| Standalone criterion? | ✅ Yes | ✅ Yes | ✅ Yes | ❌ (derived from Entropy) |
| Tree result vs Gini | — | ~identical | ~identical | ~identical |
| sklearn `criterion=` | `'gini'` | `'entropy'` | `'log_loss'` | N/A |

---

## Summary Cheatsheet

```
All 4 answer the same question: "How impure is this node?"

Gini    →  fastest, no log, default choice
Entropy →  information theory roots, log₂, slightly more sensitive
Log Loss→  entropy with ln instead of log₂ (same thing, different units)
IG      →  not an impurity measure! It's Entropy(before) - Entropy(after)
            = how much the split improved things

Trees produced: nearly identical across all three criteria.
Focus your tuning energy on max_depth and min_samples_leaf — not criterion.
```
