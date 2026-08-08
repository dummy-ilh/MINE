# Decision Tree Splitting Criteria — Master Notes
## Gini Impurity, Entropy, Information Gain & Misclassification Error

---

## 1. The core problem: how does a tree decide where to split?

A decision tree grows by repeatedly asking yes/no questions ("is age > 30?", "is income < 50k?") and splitting the data into purer and purer subgroups. But at every node, there could be dozens of candidate questions. **How does the algorithm pick the best one?**

It needs a numeric score — an **impurity measure** — that says "how mixed-up (impure) is this group of labels?" Then, for every candidate split, it checks: *does this split reduce impurity the most?*

```
Root node (impure — mixed classes)
        │
   [pick best question]
        │
    ┌───┴───┐
  Left    Right
 (purer) (purer)
```

The three impurity measures you need to know, all doing the same job with slightly different math:

| Measure | Used by |
|---|---|
| **Gini impurity** | CART (scikit-learn's default `DecisionTreeClassifier`) |
| **Entropy / Information Gain** | ID3, C4.5 |
| **Misclassification error** | Rarely used for growing trees (explained in §7 why) |

---

## 2. Gini impurity — the formula, built from scratch

**Intuition first, formula second.** Imagine you reach into a node and pick two random data points (with replacement). Gini impurity is the probability that **they'd have different classes** if you assigned labels randomly according to the node's class proportions.

- If a node is **pure** (all one class), any two random picks always match → impurity = 0.
- If a node is **maximally mixed** (50/50 in binary case), your two random picks disagree half the time → impurity is at its max.

### The formula

For a node with `C` classes, where `pᵢ` is the proportion of class `i` in that node:

```
Gini = 1 - Σ(pᵢ²)     for i = 1 to C
```

**Where does this come from?** The probability two random draws match is `Σ(pᵢ²)` (probability both are class 1, plus both class 2, etc.). So the probability they *don't* match — impurity — is `1 - Σ(pᵢ²)`.

### For binary classification specifically

```
Gini = 1 - (p² + (1-p)²) = 2p(1-p)
```

where `p` is the proportion of the positive class.

**Range:** Gini impurity for binary classification ranges from **0** (pure node) to **0.5** (perfectly 50/50 split) — this max value shifts for more classes: with `C` classes, the max is `1 - 1/C` (at perfectly uniform class proportions).

---

## 3. Worked numerical example — computing Gini for a single node

Suppose a node contains 10 samples: 6 labeled "Yes" (will buy), 4 labeled "No" (won't buy).

```
p(Yes) = 6/10 = 0.6
p(No)  = 4/10 = 0.4

Gini = 1 - (0.6² + 0.4²)
     = 1 - (0.36 + 0.16)
     = 1 - 0.52
     = 0.48
```

This node is quite impure (close to the 0.5 max for binary). We'd like a split that produces children with lower combined impurity.

---

## 4. Choosing the best split — Gini Gain

To evaluate a candidate split, compute the **weighted average Gini impurity of the resulting children**, then compare it to the parent's Gini. The reduction is called **Gini Gain**.

```
Gini_split = (n_left/n) · Gini_left  +  (n_right/n) · Gini_right

Gini Gain = Gini_parent - Gini_split
```

The algorithm tries every possible split (every feature, every threshold) and picks the one with the **highest Gini Gain** (equivalently, lowest weighted child Gini).

### Full worked example

Continuing the 10-sample node (6 Yes, 4 No), suppose we're testing the split **"Income > 50k?"**:

```
              [10 samples: 6 Yes, 4 No]
                  Gini = 0.48
                       │
         Income > 50k? ── candidate split
                       │
        ┌──────────────┴──────────────┐
   Income > 50k                  Income ≤ 50k
   (7 samples: 5 Yes, 2 No)      (3 samples: 1 Yes, 2 No)
```

**Left child Gini:**
```
p(Yes) = 5/7 = 0.714,  p(No) = 2/7 = 0.286
Gini_left = 1 - (0.714² + 0.286²) = 1 - (0.510 + 0.0818) = 1 - 0.592 = 0.408
```

**Right child Gini:**
```
p(Yes) = 1/3 = 0.333,  p(No) = 2/3 = 0.667
Gini_right = 1 - (0.333² + 0.667²) = 1 - (0.111 + 0.445) = 1 - 0.556 = 0.444
```

**Weighted split Gini:**
```
Gini_split = (7/10)(0.408) + (3/10)(0.444)
           = 0.2856 + 0.1332
           = 0.4188
```

**Gini Gain:**
```
Gini Gain = 0.48 - 0.4188 = 0.0612
```

The tree-building algorithm computes this Gain for *every* candidate split (every feature × every threshold) and picks whichever gives the **largest Gain**. If a different split — say "Age > 30?" — gave a Gain of 0.09, the algorithm would prefer that split instead.

---

## 5. Entropy and Information Gain — the alternative measure

**Entropy**, borrowed from information theory, measures the same underlying idea (impurity/uncertainty) with a logarithmic formula instead:

```
Entropy = -Σ(pᵢ · log₂(pᵢ))     for i = 1 to C
```

For binary classification:
```
Entropy = -p·log₂(p) - (1-p)·log₂(1-p)
```

**Intuition:** entropy measures the average number of bits needed to communicate the class label, if you encoded it optimally given the node's class distribution. A pure node needs 0 bits (you already know the answer). A 50/50 node needs a full 1 bit (maximum uncertainty).

**Range for binary case:** 0 (pure) to **1.0** (50/50) — note this is a different max value and scale than Gini's 0-to-0.5 range, though both peak at the same *point* (50/50 split).

### Worked example — same node as before

```
p(Yes) = 0.6, p(No) = 0.4

Entropy = -0.6·log₂(0.6) - 0.4·log₂(0.4)
        = -0.6·(-0.737) - 0.4·(-1.322)
        = 0.442 + 0.529
        = 0.971
```

### Information Gain (the entropy equivalent of Gini Gain)

```
Information Gain = Entropy_parent - Entropy_split
```

Using the same left/right children as before:

**Left child entropy** (5 Yes, 2 No out of 7):
```
p = 5/7 = 0.714
Entropy_left = -0.714·log₂(0.714) - 0.286·log₂(0.286)
             = -0.714·(-0.486) - 0.286·(-1.807)
             = 0.347 + 0.517 = 0.864
```

**Right child entropy** (1 Yes, 2 No out of 3):
```
p = 1/3 = 0.333
Entropy_right = -0.333·log₂(0.333) - 0.667·log₂(0.667)
              = -0.333·(-1.585) - 0.667·(-0.585)
              = 0.528 + 0.390 = 0.918
```

**Weighted split entropy:**
```
Entropy_split = (7/10)(0.864) + (3/10)(0.918) = 0.6048 + 0.2754 = 0.880
```

**Information Gain:**
```
IG = 0.971 - 0.880 = 0.091
```

---

## 6. Gini vs. Entropy — side-by-side comparison

| Aspect | Gini Impurity | Entropy |
|---|---|---|
| Formula | `1 - Σpᵢ²` | `-Σpᵢ·log₂(pᵢ)` |
| Range (binary) | 0 to 0.5 | 0 to 1.0 |
| Computational cost | Cheaper — no logarithm | Slightly more expensive — requires log |
| Sensitivity | Slightly favors larger partitions / more balanced-frequency splits | Slightly more sensitive to changes in class probability, especially near extremes |
| Used by | CART, scikit-learn default | ID3, C4.5 |
| Practical difference | In practice, they agree ~95%+ of the time on which split is best | Same |

**Interview soundbite:** *"Gini and entropy almost always pick the same splits in practice — the choice rarely changes the final tree meaningfully. Gini is preferred computationally (no log calls), which is why scikit-learn defaults to it."*

### Visual comparison of the two curves (binary case)

```
Impurity
  1.0 ┤                    Entropy (peak = 1.0)
      │                 ●●●●●●●●
  0.8 ┤              ●●●        ●●●
      │            ●●    Gini (peak = 0.5)  ●●
  0.6 ┤          ●●     ▲▲▲▲▲▲▲▲▲▲▲▲          ●●
      │        ●●    ▲▲▲              ▲▲▲       ●●
  0.4 ┤      ●●    ▲▲                     ▲▲       ●●
      │    ●●   ▲▲▲                          ▲▲▲     ●●
  0.2 ┤  ●● ▲▲▲                                  ▲▲▲   ●●
      │●▲▲                                           ▲▲●
  0.0 ┼──────┬──────┬──────┬──────┬──────┬──────┬──────
      0.0   0.17   0.33   0.5    0.67   0.83   1.0
                    p (proportion of positive class)
```

Both curves are 0 at the extremes (p=0 or p=1, pure nodes) and peak at p=0.5. Entropy's curve is slightly "taller and more rounded"; Gini's is a simple parabola.

---

## 7. Misclassification error — why it's a poor splitting criterion

A third, more naive measure: just the error rate if you predicted the majority class.

```
Misclassification Error = 1 - max(pᵢ)
```

For our example node (p=0.6, 0.4): `1 - 0.6 = 0.4`.

**The problem:** misclassification error is **not sensitive enough to detect good splits** that don't change the majority class prediction. It's *piecewise linear* and less curved than Gini/entropy, so it often assigns **zero gain to splits that Gini/entropy would recognize as valuable** (e.g., a split that makes one child much purer and the other slightly less pure, while the majority class prediction stays the same in both children).

**Worked counterexample:**
Parent: 400 class A, 400 class B (800 total). Two candidate splits:

**Split 1:**
- Left: 300 A, 100 B → majority A
- Right: 100 A, 300 B → majority B
- Misclassification error: parent 0.5 → weighted children = (400/800)(0.25) + (400/800)(0.25) = 0.25 → **Gain = 0.25**

**Split 2:**
- Left: 200 A, 400 B → majority B
- Right: 200 A, 0 B → majority A (pure!)
- Misclassification error: weighted children = (600/800)(0.333) + (200/800)(0) = 0.25 → **Gain = 0.25 (identical to Split 1!)**

Misclassification error says these two splits are **equally good** — but Split 2 produced a **perfectly pure** child, which is clearly more valuable for continuing to grow a useful tree. Gini and entropy, being strictly concave functions, **do** distinguish between these — they'd assign Split 2 a higher gain. This is precisely why Gini/entropy are used for growing trees, while misclassification error is sometimes used only for *pruning* (post-hoc simplification), where this sensitivity matters less.

---

## 8. Multi-class Gini example (3 classes)

Node with 20 samples: 10 Setosa, 6 Versicolor, 4 Virginica (a classic iris-style example).

```
p(Setosa) = 10/20 = 0.5
p(Versicolor) = 6/20 = 0.3
p(Virginica) = 4/20 = 0.2

Gini = 1 - (0.5² + 0.3² + 0.2²)
     = 1 - (0.25 + 0.09 + 0.04)
     = 1 - 0.38
     = 0.62
```

Note the max possible Gini for 3 balanced classes is `1 - 1/3 = 0.667`, so this node (0.62) is close to maximally impure.

---

## 9. How this fits into the full tree-growing algorithm (CART)

```
1. At current node, for every feature:
     for every possible threshold (candidate split point):
         compute weighted Gini (or entropy) of the two children
         compute Gini Gain
2. Pick the (feature, threshold) pair with the highest Gain
3. Split the node into two children using that rule
4. Recurse into each child, repeating steps 1–3
5. Stop when a stopping condition is met:
     - node is pure (Gini = 0)
     - max_depth reached
     - min_samples_split / min_samples_leaf violated
     - Gain improvement falls below a minimum threshold
```

This greedy, node-by-node approach is why decision trees are sometimes criticized as making **locally optimal** (not globally optimal) choices — a split that looks best right now might not lead to the best *overall* tree. This is one motivation for ensemble methods (Random Forests, Gradient Boosted Trees) that average over many trees.

---

## 10. Regression trees — a quick related note (Variance / MSE as "impurity")

Gini and entropy only apply to **classification** trees. For **regression trees**, the analogous impurity measure is **variance reduction** (equivalently, weighted MSE reduction):

```
Impurity(node) = (1/n) Σ(yᵢ - ȳ)²        [variance of target in that node]

Variance Reduction = Var_parent - [ (n_left/n)·Var_left + (n_right/n)·Var_right ]
```

Same greedy logic, same tree-growing algorithm — just swap "class purity" for "how tightly clustered are the target values."

---

## 11. Summary comparison table — all four measures

| Measure | Formula (binary) | Range | Task | Convexity |
|---|---|---|---|---|
| Gini impurity | `2p(1-p)` | [0, 0.5] | Classification | Strictly concave |
| Entropy | `-p·log₂p - (1-p)·log₂(1-p)` | [0, 1.0] | Classification | Strictly concave |
| Misclassification error | `1 - max(p, 1-p)` | [0, 0.5] | Classification (pruning mainly) | Piecewise linear (not strictly concave) |
| Variance / MSE | `(1/n)Σ(yᵢ-ȳ)²` | [0, ∞) | Regression | Convex (quadratic) |

---

## 12. Interview Q&A

**Q: What is Gini impurity measuring, intuitively?**
A: The probability that two randomly drawn samples from a node (with replacement) would have different class labels, if labeled according to the node's own class proportions. 0 means pure, higher means more mixed.

**Q: Why does scikit-learn default to Gini over entropy?**
A: Gini avoids computing logarithms, so it's computationally cheaper, and in practice the two criteria almost always select the same splits — the difference in resulting trees is usually negligible.

**Q: Why isn't misclassification error used to grow trees, even though it seems like the most "direct" measure of quality?**
A: It's piecewise linear rather than strictly concave, so it can be insensitive to splits that improve class purity without changing which class is the majority in each child — it can assign identical (zero or low) gain to a clearly better split and a clearly worse one, as shown in the counterexample in §7.

**Q: What's the maximum possible Gini impurity for a node with C classes, and when is it achieved?**
A: `1 - 1/C`, achieved when all classes are equally represented (uniform distribution) in the node.

**Q: How does a decision tree pick the best split at each node?**
A: For every candidate (feature, threshold) pair, it computes the impurity of the resulting two children, weights it by the fraction of samples in each child, and compares that weighted impurity to the parent's impurity. The split with the highest impurity reduction ("Gain") is chosen. This repeats greedily/recursively down the tree.

**Q: Is this a globally optimal tree-building strategy?**
A: No — it's a greedy algorithm, choosing the locally best split at each node without looking ahead. This can lead to suboptimal overall trees; finding the globally optimal decision tree is NP-hard, so greedy heuristics (with pruning, ensembling) are used in practice instead.

**Q: How would Gini/entropy change for a regression problem?**
A: They don't apply directly — regression trees use variance (equivalently, MSE) reduction as the analogous "impurity" measure, since class proportions don't exist for continuous targets.

**Q: What's the relationship between entropy-based Information Gain and mutual information?**
A: Information Gain *is* the mutual information between the splitting feature and the target label, restricted to this particular binary split of the data — it measures how much knowing the split reduces uncertainty about the label.

**Q: A colleague says "higher Gini impurity is better." Is that right?**
A: No — it's the opposite. Lower Gini impurity means a purer, more homogeneous node, which is what you want after splitting. The tree-building algorithm seeks the split that *maximizes the reduction* in Gini (Gini Gain), i.e., minimizes the resulting weighted impurity.

---

## 13. One-paragraph summary

Decision trees pick splits by measuring how "impure" (mixed-class) a node is before and after a candidate split, then choosing whichever split reduces impurity the most. **Gini impurity** (`1 - Σpᵢ²`) measures the chance two random samples from a node disagree in class; **entropy** (`-Σpᵢ·log₂pᵢ`) measures the same idea via information-theoretic bits — both peak at maximum class mixing and hit zero at perfect purity, and in practice pick nearly identical splits, with Gini preferred for its lower computational cost. **Misclassification error** (`1 - max(pᵢ)`) is a simpler alternative but is insufficiently sensitive — it can rate a split that produces a perfectly pure child no better than one that doesn't, which is why it's used mainly for pruning rather than growing trees. For regression trees, variance/MSE reduction plays the same structural role. All of these feed the same greedy, recursive, top-down tree-building algorithm (CART).

---

## 14. Full worked dataset — all measures computed end to end

Here's one complete dataset used to compute *every* measure from this doc, so you can see them all side by side on identical numbers.

**Dataset: 14 samples, predicting "Play Tennis?" from Outlook and Humidity.**

| # | Outlook | Humidity | Play Tennis |
|---|---|---|---|
| 1 | Sunny | High | No |
| 2 | Sunny | High | No |
| 3 | Overcast | High | Yes |
| 4 | Rain | High | Yes |
| 5 | Rain | Normal | Yes |
| 6 | Rain | Normal | No |
| 7 | Overcast | Normal | Yes |
| 8 | Sunny | High | No |
| 9 | Sunny | Normal | Yes |
| 10 | Rain | Normal | Yes |
| 11 | Sunny | Normal | Yes |
| 12 | Overcast | High | Yes |
| 13 | Overcast | Normal | Yes |
| 14 | Rain | High | No |

Root node: **14 samples → 9 Yes, 5 No.**

### Step 1 — Impurity of the root node (before any split)

```
p(Yes) = 9/14 = 0.643
p(No)  = 5/14 = 0.357
```

**Gini:**
```
Gini_root = 1 - (0.643² + 0.357²) = 1 - (0.413 + 0.127) = 1 - 0.540 = 0.460
```

**Entropy:**
```
Entropy_root = -0.643·log₂(0.643) - 0.357·log₂(0.357)
             = -0.643·(-0.637) - 0.357·(-1.486)
             = 0.410 + 0.531 = 0.940
```

**Misclassification error:**
```
Error_root = 1 - max(0.643, 0.357) = 1 - 0.643 = 0.357
```

### Step 2 — Candidate split A: "Humidity" (High vs Normal)

```
Humidity = High:   7 samples → {No, No, Yes, Yes, No, Yes, No} = 3 Yes, 4 No
Humidity = Normal: 7 samples → {Yes, No, Yes, Yes, Yes, Yes, Yes} = 6 Yes, 1 No
```

**Gini of each child:**
```
Gini_High   = 1 - ((3/7)² + (4/7)²) = 1 - (0.184 + 0.327) = 1 - 0.511 = 0.489
Gini_Normal = 1 - ((6/7)² + (1/7)²) = 1 - (0.735 + 0.020) = 1 - 0.755 = 0.245
```
```
Gini_split(Humidity) = (7/14)(0.489) + (7/14)(0.245) = 0.2445 + 0.1225 = 0.367
Gini Gain = 0.460 - 0.367 = 0.093
```

**Entropy of each child:**
```
Entropy_High   = -(3/7)log₂(3/7) - (4/7)log₂(4/7) = -0.429(-1.222) - 0.571(-0.807) = 0.524+0.461 = 0.985
Entropy_Normal = -(6/7)log₂(6/7) - (1/7)log₂(1/7) = -0.857(-0.222) - 0.143(-2.807) = 0.190+0.401 = 0.591
```
```
Entropy_split(Humidity) = (7/14)(0.985) + (7/14)(0.591) = 0.4925 + 0.2955 = 0.788
Information Gain = 0.940 - 0.788 = 0.152
```

**Misclassification error of each child:**
```
Error_High   = 1 - max(3/7, 4/7) = 1 - 0.571 = 0.429
Error_Normal = 1 - max(6/7, 1/7) = 1 - 0.857 = 0.143
```
```
Error_split(Humidity) = (7/14)(0.429) + (7/14)(0.143) = 0.2145 + 0.0715 = 0.286
Error reduction = 0.357 - 0.286 = 0.071
```

### Step 3 — Candidate split B: "Outlook" (Sunny / Overcast / Rain)

```
Outlook = Sunny:    5 samples → {No, No, No, Yes, Yes} = 2 Yes, 3 No
Outlook = Overcast: 4 samples → {Yes, Yes, Yes, Yes}   = 4 Yes, 0 No   (pure!)
Outlook = Rain:     5 samples → {Yes, Yes, No, Yes, No} = 3 Yes, 2 No
```

**Gini of each child:**
```
Gini_Sunny    = 1 - ((2/5)² + (3/5)²) = 1 - (0.16+0.36) = 0.48
Gini_Overcast = 1 - ((4/4)² + (0/4)²) = 1 - 1.0 = 0.00
Gini_Rain     = 1 - ((3/5)² + (2/5)²) = 1 - (0.36+0.16) = 0.48
```
```
Gini_split(Outlook) = (5/14)(0.48) + (4/14)(0.00) + (5/14)(0.48)
                     = 0.1714 + 0 + 0.1714 = 0.343
Gini Gain = 0.460 - 0.343 = 0.117
```

**Entropy of each child:**
```
Entropy_Sunny    = -(2/5)log₂(2/5) - (3/5)log₂(3/5) = 0.529+0.442 = 0.971
Entropy_Overcast = 0  (pure node, no uncertainty)
Entropy_Rain     = -(3/5)log₂(3/5) - (2/5)log₂(2/5) = 0.442+0.529 = 0.971
```
```
Entropy_split(Outlook) = (5/14)(0.971) + (4/14)(0) + (5/14)(0.971) = 0.347+0+0.347 = 0.694
Information Gain = 0.940 - 0.694 = 0.246
```

**Misclassification error of each child:**
```
Error_Sunny    = 1 - max(2/5, 3/5) = 1 - 0.6 = 0.4
Error_Overcast = 1 - max(1, 0) = 0
Error_Rain     = 1 - max(3/5, 2/5) = 1 - 0.6 = 0.4
```
```
Error_split(Outlook) = (5/14)(0.4) + (4/14)(0) + (5/14)(0.4) = 0.143+0+0.143 = 0.286
Error reduction = 0.357 - 0.286 = 0.071
```

### Step 4 — Which split wins, and does the choice of measure matter?

| Measure | Gain: Humidity split | Gain: Outlook split | Winner |
|---|---|---|---|
| Gini Gain | 0.093 | **0.117** | Outlook |
| Information Gain (entropy) | 0.152 | **0.246** | Outlook |
| Misclassification error reduction | 0.071 | 0.071 | **Tie — can't distinguish!** |

**This is the punchline of the whole document, shown on real numbers:** Gini and entropy both correctly identify **Outlook** as the stronger split (and by a wide margin — Outlook produces a perfectly pure "Overcast" child, which both measures reward heavily). Misclassification error, however, scores both splits **identically** at 0.071 — it's blind to the fact that Outlook produced a pure child, because in every child of every split here, the majority-class error rate happens to work out the same. This is exactly the insensitivity problem described in §7, now demonstrated on a concrete dataset rather than a constructed counterexample.

The tree-building algorithm, using Gini (CART) or entropy (ID3/C4.5), would therefore split on **Outlook** first — and "Overcast" becomes an immediate leaf node (pure, 4/4 Yes), needing no further splitting.

### Step 5 — Regression-tree analog on the same idea

If instead of "Play Tennis" (Yes/No) we had a *continuous* target — say, "hours played" — the analogous calculation would replace class proportions with the **variance of hours played** in each node, and Gain would become **variance reduction**, computed exactly as in §10. The splitting logic (compute impurity, weight by child size, compare to parent, pick max reduction) is identical; only the impurity formula changes from `1-Σpᵢ²` to `Var(y)`.
