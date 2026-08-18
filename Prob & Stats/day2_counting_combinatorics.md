# Day 2 — Counting: Permutations, Combinations & The Multiplication Rule
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 1 (Sections 1.3–1.6)  
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Counting in ML?

You might wonder — why does an ML engineer need to count?

- **How many ways can you arrange a mini-batch?** → Permutations
- **How many feature subsets exist for a model with n features?** → Combinations (2ⁿ — exponential, which is why feature selection is hard)
- **Hyperparameter search:** 3 learning rates × 4 batch sizes × 2 optimizers = how many configs?
- **Probability calculation:** When outcomes are equally likely, P(A) = |A| / |Ω|. To find |A| and |Ω|, you **count**.
- **NLP:** How many unique bigrams exist in a vocabulary of size V? → V²

Counting is the bridge between sample spaces and probabilities.

---

## 2. The Fundamental Principle of Counting (Multiplication Rule)

> **If experiment 1 has m outcomes, and for each of those, experiment 2 has n outcomes, then together they have m × n outcomes.**

This generalizes:

```
If task has k steps, with n₁ choices for step 1,
n₂ choices for step 2, ..., nₖ choices for step k,

Total outcomes = n₁ × n₂ × n₃ × ... × nₖ
```

### Example 2.1

A neural network has:
- 3 choices for optimizer (SGD, Adam, RMSProp)
- 4 choices for learning rate (0.1, 0.01, 0.001, 0.0001)
- 2 choices for batch size (32, 64)

Total hyperparameter configurations = 3 × 4 × 2 = **24**

---

## 3. Factorials

Before permutations, we need the factorial:

```
n! = n × (n-1) × (n-2) × ... × 2 × 1

Special case:  0! = 1   (by definition — critical to remember)
```

| n | n! |
|---|---|
| 0 | 1 |
| 1 | 1 |
| 2 | 2 |
| 3 | 6 |
| 4 | 24 |
| 5 | 120 |
| 10 | 3,628,800 |

Factorials **explode** fast — this is why brute-force search is infeasible in ML (curse of dimensionality).

---

## 4. Permutations — Order Matters

> **A permutation is an ordered arrangement of objects.**

### 4.1 Permutations of n distinct objects (use all of them)

```
P(n, n) = n!
```

*Arrange n distinct objects in a line: n choices for position 1, (n-1) for position 2, ..., 1 for last.*

### 4.2 Permutations of n objects taken r at a time (use r of them)

```
         n!
P(n,r) = ———————
         (n-r)!
```

**Reading:** "n permute r" — choose r items from n, **order matters.**

**Formula breakdown:**
- n! = total ways to arrange all n objects
- (n-r)! = ways to arrange the leftover (n-r) objects we don't use — we divide these out

### Example 4.1

How many ways can you assign 3 distinct GPU jobs to 2 of 5 available servers (order matters — job 1 on server A ≠ job 1 on server B)?

P(5, 2) = 5! / (5-2)! = 5! / 3! = (5 × 4 × 3!) / 3! = 5 × 4 = **20**

---

### 4.3 Permutations with Repetition

If you have n objects where:
- Object type 1 repeats n₁ times
- Object type 2 repeats n₂ times
- ...
- Object type k repeats nₖ times

```
              n!
Arrangements = ——————————————————
               n₁! × n₂! × ... × nₖ!
```

This is called the **multinomial coefficient** — critical for NLP and sequence models.

---

## 5. Combinations — Order Does NOT Matter

> **A combination is a selection of objects where order doesn't matter.**

```
        n!            P(n,r)
C(n,r) = ———————— = ————————
        r!(n-r)!       r!

Also written as  ⁿCᵣ  or  (n choose r)  or  C(n,r)
```

**Why divide by r!?** Because each group of r items can be arranged r! ways — but since order doesn't matter, we divide those out.

### The Binomial Coefficient

C(n,r) is called the **binomial coefficient**, written:

```
⎛n⎞      n!
⎜ ⎟  = ————————
⎝r⎠    r!(n-r)!
```

### Key Properties

```
C(n, 0) = 1          (one way to choose nothing)
C(n, n) = 1          (one way to choose everything)
C(n, 1) = n          (n ways to choose 1 item)
C(n, r) = C(n, n-r)  (symmetry — choosing r is same as leaving out n-r)
```

### Pascal's Identity (appears in proofs often)

```
C(n, r) = C(n-1, r-1) + C(n-1, r)
```

*Intuition:* Fix one item. Either it's included (C(n-1, r-1)) or it's not (C(n-1, r)).

---

## 6. Permutation vs Combination — Decision Rule

Ask yourself: **Does the order of selection matter?**

| Situation | Order Matters? | Formula |
|---|---|---|
| Ranking top-3 models | Yes | P(n, r) |
| Selecting features for a model | No | C(n, r) |
| Arranging layers in a neural net | Yes | P(n, r) |
| Choosing a validation set | No | C(n, r) |
| Assigning labels to data points | Yes (labels are distinct positions) | P(n, r) |
| Choosing a subset of hyperparameters to tune | No | C(n, r) |

---

## 7. Sampling — With vs Without Replacement

Another key distinction, especially in ML (bootstrapping, cross-validation):

| | **Order Matters** | **Order Doesn't Matter** |
|---|---|---|
| **With replacement** | nʳ | C(n+r-1, r) |
| **Without replacement** | P(n,r) = n!/(n-r)! | C(n,r) = n!/r!(n-r)! |

**With replacement** = after picking an item, put it back (can repeat).  
**Without replacement** = once picked, it's gone (no repeats).

---

## 8. The Binomial Theorem

Deeply connected to combinations:

```
         n
(x+y)ⁿ = Σ C(n,k) · xᵏ · y^(n-k)
        k=0
```

**Why it matters in ML:** The Binomial distribution (Day 9) is built directly on this. The probability of exactly k successes in n trials is C(n,k) · pᵏ · (1-p)^(n-k).

Special case: set x = y = 1:
```
2ⁿ = C(n,0) + C(n,1) + C(n,2) + ... + C(n,n)
```
*This tells you: a dataset with n features has 2ⁿ possible feature subsets — including the empty set.*

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — Hyperparameter Grid Search

**Problem:** You're doing a grid search over:
- Learning rate: {0.001, 0.01, 0.1}
- Batch size: {16, 32, 64, 128}
- Dropout: {0.0, 0.2, 0.5}
- Layers: {2, 3, 4}

(a) How many total configurations exist?  
(b) You can only run 10 experiments. In how many ways can you choose 10 configs (order doesn't matter)?

**Solution:**

**(a)** Multiplication rule:  
Total = 3 × 4 × 3 × 4 = **144 configurations**

**(b)** Choosing 10 from 144 (order doesn't matter):  
C(144, 10) = 144! / (10! × 134!)

This is an astronomically large number (~1.6 × 10¹⁶), which is why **random search** and **Bayesian optimization** exist instead of exhaustive grid search.

---

### 🔢 Numerical 2 — Feature Selection

**Problem:** A dataset has 20 features. A data scientist wants to select exactly 5 features for a logistic regression model.

(a) How many possible feature subsets of size 5 exist?  
(b) How many subsets of **at most** 2 features exist?

**Solution:**

**(a)** C(20, 5) = 20! / (5! × 15!)  
= (20 × 19 × 18 × 17 × 16) / (5 × 4 × 3 × 2 × 1)  
= 1,860,480 / 120  
= **15,504 subsets**

**(b)** "At most 2" = 0 features + 1 feature + 2 features:  
C(20,0) + C(20,1) + C(20,2)  
= 1 + 20 + (20×19/2)  
= 1 + 20 + 190  
= **211 subsets**

---

### 🔢 Numerical 3 — Arranging Training Data (Permutations)

**Problem:** You have 8 distinct data points in a mini-batch.

(a) In how many ways can you present them to the model (order matters for SGD)?  
(b) In how many ways can you select the first 3 data points to process (order matters)?  
(c) What if you just want any subset of 3 (order doesn't matter)?

**Solution:**

**(a)** All 8 in order: 8! = **40,320 ways**

**(b)** P(8, 3) = 8! / (8-3)! = 8! / 5! = 8 × 7 × 6 = **336 ways**

**(c)** C(8, 3) = 8! / (3! × 5!) = 336 / 6 = **56 ways**

*Notice: P(8,3) = 6 × C(8,3) because 3! = 6 — the order of the 3 chosen items.*

---

### 🔢 Numerical 4 — Permutations with Repetition (NLP)

**Problem:** How many distinct arrangements exist for the word **"MISSISSIPPI"**?

**Solution:**

Total letters: 11  
M appears: 1 time  
I appears: 4 times  
S appears: 4 times  
P appears: 2 times  

```
         11!
Answer = ———————————— = 39,916,800 / (1 × 24 × 24 × 2) = **34,650**
         1! × 4! × 4! × 2!
```

**ML Connection:** This is the foundation of the **multinomial distribution** — counting word arrangements in a document. It directly underlies bag-of-words models and document generation.

---

### 🔢 Numerical 5 — Probability via Counting

**Problem:** A test dataset has 10 images: 4 cats, 3 dogs, 3 birds. You randomly select 3 images for a manual review.

(a) What is the probability all 3 are cats?  
(b) What is the probability you get exactly 2 cats and 1 dog?  
(c) What is the probability you get at least 1 cat?

**Solution:**

Total ways to choose 3 from 10:  
|Ω| = C(10, 3) = 10!/(3!×7!) = 120

**(a)** All 3 cats: choose 3 from 4 cats:  
|A| = C(4, 3) = 4  
P(all cats) = 4/120 = **1/30 ≈ 0.033**

**(b)** Exactly 2 cats AND 1 dog:  
Choose 2 cats from 4: C(4,2) = 6  
Choose 1 dog from 3: C(3,1) = 3  
|B| = 6 × 3 = 18  
P(2 cats, 1 dog) = 18/120 = **3/20 = 0.15**

**(c)** P(at least 1 cat) = 1 - P(no cats)  
P(no cats) = C(6,3)/C(10,3) = 20/120 = 1/6  
P(at least 1 cat) = 1 - 1/6 = **5/6 ≈ 0.833**

*(Complement rule + counting — always easier than summing P(1 cat) + P(2 cats) + P(3 cats))*

---

### 🔢 Numerical 6 — Bootstrapping (Sampling with Replacement)

**Problem:** You have a dataset of n = 5 samples. Bootstrap sampling draws n = 5 samples **with replacement**.

(a) How many distinct bootstrap samples exist?  
(b) What is the probability that a specific sample is **never** selected in a bootstrap draw of size n?

**Solution:**

**(a)** Each of 5 draws has 5 choices (with replacement, order matters):  
Total = 5⁵ = **3,125 distinct bootstrap samples**

**(b)** P(specific sample not chosen in one draw) = 4/5  
P(never chosen in 5 draws) = (4/5)⁵ = 1024/3125 ≈ **0.328**

**The famous limit:** As n → ∞:
```
P(not chosen) = (1 - 1/n)ⁿ → 1/e ≈ 0.368
```
So in bootstrapping, **≈36.8% of original samples are left out** on average — these become the **out-of-bag (OOB) samples** used for validation in Random Forests. This is a classic ML interview question.

---

### 🔢 Numerical 7 — Cross-Validation Splits

**Problem:** You have 100 data points and want to do 5-fold cross-validation.

In how many ways can you divide the 100 points into 5 equal groups of 20?

**Solution:**

This is a **multinomial coefficient**:

```
          100!
Ways = ——————————————————————————
        20! × 20! × 20! × 20! × 20!
```

This is an astronomically large number — it shows that there are essentially infinite valid cross-validation splits, which is why **stratified k-fold** (ensuring class balance per fold) is a meaningful constraint.

---

## 10. Common Interview Questions

| Question | Key Idea |
|---|---|
| "How many ways to split 80/20 train-test from n samples?" | C(n, 0.2n) |
| "Why is brute-force feature selection infeasible?" | 2ⁿ subsets — exponential |
| "What fraction of samples appear in a bootstrap?" | ≈ 1 - 1/e ≈ 63.2% |
| "How many bigrams exist in vocab of size V?" | V² (with replacement, order matters) |
| "What's the difference between permutation and combination?" | Order matters vs not |
| "How many unique n-grams of length k from vocab V?" | Vᵏ |

---

## 11. Key Formulas — Cheat Sheet for Day 2

```
Multiplication Rule:    n₁ × n₂ × ... × nₖ

Factorial:              n! = n × (n-1) × ... × 1,    0! = 1

Permutation:            P(n,r) = n! / (n-r)!         [order matters, no replacement]

Combination:            C(n,r) = n! / [r!(n-r)!]     [order doesn't matter, no replacement]

With replacement:       nʳ                            [order matters]

Multinomial:            n! / (n₁! × n₂! × ... × nₖ!) [repeated items]

Symmetry:               C(n,r) = C(n, n-r)
Pascal's Identity:      C(n,r) = C(n-1,r-1) + C(n-1,r)
Binomial Theorem:       (x+y)ⁿ = Σ C(n,k) xᵏ y^(n-k)
Feature subsets:        2ⁿ total subsets of n features
Bootstrap OOB:          P(not sampled) → 1/e ≈ 0.368 as n → ∞
```

---

## 12. Practice Problems (Solve Before Day 3)

1. A password must be 4 characters, each from {A–Z, 0–9} (36 options each). Repetition allowed. How many passwords are possible?

2. A team of 3 must be chosen from 7 ML engineers and 4 data scientists, such that exactly 2 are ML engineers. How many such teams exist?

3. *(Bootstrap)* For a dataset of n = 10 samples, what is the probability that a **specific** sample appears **at least once** in a bootstrap sample of size 10?

4. In how many ways can you arrange the letters in "STATISTICS"?

5. *(Interview-level)* A model is tested on 12 images. 7 are positive class, 5 are negative. If you randomly pick 4 images:
   - P(all positive)?
   - P(exactly 2 positive, 2 negative)?
   - P(at least 3 positive)?

6. *(Think hard)* You have 5 candidate models. You want to pick a top-3 ranking (1st, 2nd, 3rd place). How many rankings exist? How is this different from choosing any 3 of the 5?

---

## 13. Looking Ahead

**Day 3** — **Conditional Probability & The Chain Rule.**  
We finally answer: "Given we already know something, how does it change our probability?" This is the foundation of Bayesian thinking, Naive Bayes classifiers, and every probabilistic graphical model.

---
*End of Day 2 | Next: Day 3 — Conditional Probability & The Chain Rule*
