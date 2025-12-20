Here are markdown notes summarizing the provided document on Counting and Sets.

# Counting and Sets: Class 1, 18.05

---

## 1. Learning Goals

[cite_start]The goals for this class include[cite: 3]:
1.  [cite_start]Knowing the definitions and notation for **sets**, **intersection**, **union**, and **complement**[cite: 4].
2.  [cite_start]Being able to visualize set operations using **Venn diagrams**[cite: 5].
3.  [cite_start]Understanding how **counting** is used in computing probabilities[cite: 6].
4.  [cite_start]Being able to use the **rule of product**, **inclusion-exclusion principle**, **permutations**, and **combinations** to count elements in a set[cite: 7].

---

## 2. Counting and Probability

[cite_start]Probability for equally probable outcomes is computed by counting the elements in sets[cite: 31].

### Probability Principle (Equally Probable Outcomes)

[cite_start]If an experiment has $n$ possible outcomes, and each is equally probable, then the probability of a desirable outcome (where there are $k$ desirable outcomes) is $\frac{k}{n}$[cite: 38, 39].

[cite_start]**Example (Coin Flip):** For three fair coin flips, there are $n=8$ total outcomes[cite: 12, 13]. [cite_start]Since three outcomes have exactly one head ($k=3$), the probability of exactly one head is $3/8$[cite: 14, 15, 17].

---

## 3. Sets and Notation

[cite_start]A **set ($S$)** is a collection of elements[cite: 47].

### Definitions

| Term | Notation | Definition |
| :--- | :--- | :--- |
| **Element** | $x \in S$ | [cite_start]$x$ is in the set $S$[cite: 48]. |
| **Subset** | $A \subset S$ | [cite_start]All elements of set $A$ are also in set $S$[cite: 49]. |
| **Complement** | $A^c$ or $S-A$ | [cite_start]Elements of $S$ that are **not** in $A$[cite: 50, 72]. |
| **Union** | $A \cup B$ | [cite_start]All elements in $A$ **or** $B$ (or both)[cite: 51, 52]. |
| **Intersection** | $A \cap B$ | [cite_start]All elements in **both** $A$ and $B$[cite: 53]. |
| **Empty Set** | $\emptyset$ (or $0$) | [cite_start]The set with no elements[cite: 54]. |
| **Disjoint** | $A \cap B = \emptyset$ | [cite_start]$A$ and $B$ have no common elements[cite: 55]. |
| **Difference** | $A - B$ | [cite_start]Elements in $A$ that are **not** in $B$[cite: 56, 71]. [cite_start]Note: $A-B = A \cap B^c$[cite: 72]. |

### DeMorgan's Laws

[cite_start]The relationship between union, intersection, and complement is given by[cite: 73, 74]:
* $(A \cup B)^c = A^c \cap B^c$
* $(A \cap B)^c = A^c \cup B^c$

### Venn Diagrams

[cite_start]Venn diagrams are used to visualize set operations, where the large rectangle often represents the universal set ($S$), and circles represent subsets like $L$ and $R$[cite: 78, 79, 80]. 
### Products of Sets

[cite_start]The **product** of sets $S$ and $T$ is the set of **ordered pairs**[cite: 111]:
$$S \times T = \{(s, t) \mid s \in S, t \in T\}$$

---

## 4. Counting Principles

[cite_start]If $S$ is finite, the number of elements is denoted by $|S|$ or $\#S$[cite: 128].

### Inclusion-Exclusion Principle

[cite_start]This principle is used to count the size of a union of sets[cite: 129, 131]:
$$|A \cup B| = |A| + |B| - |A \cap B|$$
[cite_start]When you add $|A|$ and $|B|$, the elements in the intersection ($|A \cap B|$) are **double-counted**, so the intersection must be subtracted once[cite: 140].

### Rule of Product (Multiplication Rule)

[cite_start]If there are $n$ ways to perform action 1 and then $m$ ways to perform action 2, there are $n \cdot m$ ways to perform the sequence of actions[cite: 149, 150].

* [cite_start]This rule holds even if the available options for action 2 depend on action 1, as long as the *number* of ways for action 2 is constant regardless of the choice for action 1[cite: 153, 157].

---

## 5. Permutations and Combinations

These concepts define methods for selecting a number of elements from a set.

### Permutations

[cite_start]A **permutation** is a **particular ordering** (a list) of elements, where **order matters**[cite: 160, 178, 181].

* The number of permutations of a set of $k$ elements is $k! [cite_start]= k \cdot (k-1) \cdots 3 \cdot 2 \cdot 1$ (k-factorial)[cite: 166, 167].
* [cite_start]The number of permutations of $k$ distinct elements chosen from a set of size $n$ is denoted ${}_n P_k$[cite: 168, 191]:
    [cite_start]$${}_n P_k = \frac{n!}{(n-k)!} = n(n-1) \cdots (n-k+1)$$ [cite: 195]

### Combinations

[cite_start]A **combination** is a **subset** (a collection) of elements, where **order does not matter**[cite: 181, 184].

* [cite_start]The number of combinations of $k$ elements (subsets of size $k$) from a set of size $n$ is denoted ${}_n C_k$ or $\binom{n}{k}$, and is read as "**n choose k**"[cite: 191, 192, 197]:
    [cite_start]$${}_n C_k = \binom{n}{k} = \frac{n!}{k!(n-k)!} = \frac{{}_n P_k}{k!}$$ [cite: 196]

[cite_start]The combination formula arises because every subset of size $k$ can be ordered in $k!$ ways (permutations)[cite: 188, 198].

[cite_start]**Example:** The number of ways to get exactly 3 heads in 10 flips is a combination problem (choosing 3 positions out of 10 for the heads, order doesn't matter)[cite: 226, 227]:
[cite_start]$$\binom{10}{3} = 120$$ [cite: 228]
