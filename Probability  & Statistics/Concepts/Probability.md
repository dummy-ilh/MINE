#  Probability
### Applied Statistics and Probability for Engineers — Montgomery & Runger (2018)

---

## Chapter Outline
- 2.1 Sample Spaces and Events
- 2.2 Counting Techniques
- 2.3 Interpretations and Axioms of Probability
- 2.4 Unions of Events and Addition Rules
- 2.5 Conditional Probability
- 2.6 Intersections of Events and Multiplication and Total Probability Rules
- 2.7 Independence
- 2.8 Bayes' Theorem
- 2.9 Random Variables



---

## 2.1 Sample Spaces and Events

### 2.1.1 Random Experiments

> **Definition — Random Experiment:**
> An experiment that can result in different outcomes, even though it is repeated in the same manner every time, is called a **random experiment**.
---

### 2.1.2 Sample Spaces

> **Definition — Sample Space:**
> The set of all possible outcomes of a random experiment is called the **sample space** of the experiment, denoted as **S**.


#### Example 2.1 — Camera Flash

Consider an experiment selecting a cell phone camera and recording the recycle time of a flash. The time is positive, so:

| Objective | Sample Space |
|-----------|--------------|
| All positive times | S = R⁺ = {x \| x > 0} |
| Known times between 1.5 and 5 sec | S = {x \| 1.5 < x < 5} |
| Classify as low/medium/high | S = {low, medium, high} |
| Conforms to minimum specification? | S = {yes, no} |

---

> **Definition — Discrete vs. Continuous Sample Spaces:**
> - A sample space is **discrete** if it consists of a finite or countably infinite set of outcomes.
> - A sample space is **continuous** if it contains an interval (either finite or infinite) of real numbers.

- S = R⁺ is a **continuous** sample space.
- S = {yes, no} is a **discrete** sample space.

#### Example 2.2 — Camera Specifications (Two Cameras)

Recycle times of two cameras are recorded. Let S = R⁺ × R⁺ (positive quadrant of the plane).

- If we care only whether each conforms (y) or not (n): S = {yy, yn, ny, nn}
- If we only count conforming cameras in sample: S = {0, 1, 2}
- If cameras are tested until the flash recycle time fails: S = {n, yn, yyn, yyyn, yyyyn, and so forth} — a **countably infinite** discrete sample space.

---

**Tree Diagrams:** Sample spaces can be described graphically with **tree diagrams** when the experiment is conducted in steps.

#### Example 2.3 — Message Delays

Each message in a digital communication system is classified as on time or late. For three messages, possible results can be displayed by eight branches in a tree diagram.

```
Message 1 ─── On time ─── Message 2 ─── On time ─── Message 3 ─── On time
                                    │                          └─── Late
                         └─── Late  ─── Message 3 ─── On time
                                                   └─── Late
          └── Late ────── Message 2 ─── On time ─── Message 3 ─── On time
                                    │                          └─── Late
                         └─── Late  ─── Message 3 ─── On time
                                                   └─── Late
```

> **Practical Interpretation:** A tree diagram can effectively represent a sample space. Even if a tree becomes too large to construct, it can still conceptually clarify the sample space.

---

### 2.1.3 Events

> **Definition — Event:**
> An **event** is a subset of the sample space of a random experiment.

Events are formed from combinations of existing events using **basic set operations**:

| Operation | Description | Notation |
|-----------|-------------|----------|
| **Union** | All outcomes in either event | E₁ ∪ E₂ |
| **Intersection** | All outcomes in both events | E₁ ∩ E₂ |
| **Complement** | All outcomes in S but not in E | E′ (also E^C) |

#### Example 2.4 — Events

Sample space S = {yy, yn, ny, nn} from Example 2.2.

- E₁ = {yy, yn, ny} — at least one camera conforms
- E₂ = {nn} — both cameras do not conform
- E₃ = Ø (null set)
- E₄ = S (the entire sample space)
- If E₅ = {yn, ny, nn}:
  - E₁ ∪ E₅ = S
  - E₁ ∩ E₅ = {yn, ny}
  - E₁′ = {nn}

#### Example 2.5 — Camera Recycle Time (Continuous)

Sample space S = R⁺. Let:
- E₁ = {x | 10 ≤ x < 12}
- E₂ = {x | 11 < x < 15}

Then:
- E₁ ∪ E₂ = {x | 10 ≤ x < 15}
- E₁ ∩ E₂ = {x | 11 < x < 12}
- E₁′ = {x | x < 10 or 12 ≤ x}
- E₁′ ∩ E₂ = {x | 12 ≤ x < 15}

#### Example 2.6 — Hospital Emergency Visits

Visits to emergency departments at four hospitals in Arizona. Visits without seeing a physician are called **LWBS**. Data (Table):

| | Hospital 1 | Hospital 2 | Hospital 3 | Hospital 4 | Total |
|---|---|---|---|---|---|
| Total | 5292 | 6991 | 5640 | 4329 | 22,252 |
| LWBS | 195 | 270 | 246 | 242 | 953 |
| Admitted | 1277 | 1558 | 666 | 984 | 4485 |
| Not admitted | 3820 | 5163 | 4728 | 3103 | 16,814 |

- Let A = visit to hospital 1, B = visit resulting in LWBS.
- A ∩ B = 195 visits (hospital 1 that result in LWBS)
- A′ = visits to hospitals 2, 3, 4 = 6991 + 5640 + 4329 = 16,960
- A ∪ B = 5292 + 953 − 195 = 6050 (hospital 1 OR LWBS, avoiding double count)

---

**Venn Diagrams** are used to portray relationships between events.

> **Definition — Mutually Exclusive Events:**
> Two events E₁ and E₂ such that **E₁ ∩ E₂ = Ø** are said to be **mutually exclusive**.

**Additional set algebra results:**

- (E′)′ = E
- **Distributive Laws:**
  - (A ∪ B) ∩ C = (A ∩ C) ∪ (B ∩ C)
  - (A ∩ B) ∪ C = (A ∪ C) ∩ (B ∪ C)
- **DeMorgan's Laws:**
  - (A ∪ B)′ = A′ ∩ B′
  - (A ∩ B)′ = A′ ∪ B′
- A ∩ B = B ∩ A and A ∪ B = B ∪ A (commutativity)

---

## 2.2 Counting Techniques

In complex examples, the number of outcomes in a sample space or event must be counted systematically. These methods are called **counting techniques**.

> **Multiplication Rule (for counting):**
> Assume an operation can be described as a sequence of k steps, where:
> - Step 1 can be completed in n₁ ways
> - Step 2 can be completed in n₂ ways for each way of step 1
> - Step k can be completed in nₖ ways for each way of step k-1
>
> Total number of ways = **n₁ × n₂ × ··· × nₖ**

#### Example 2.7 — Web Site Design

A website design consists of 4 colors, 3 fonts, and 3 image positions.
From the multiplication rule: 4 × 3 × 3 = **36 different designs** are possible.

---

### Permutations

> **Definition — Permutation:**
> A **permutation** of elements is an **ordered sequence** of the elements.

For S = {a, b, c}, all permutations are: abc, acb, bac, bca, cab, cba.

> **Formula — Number of Permutations of n different elements:**
>
> $$n! = n \times (n-1) \times (n-2) \times \cdots \times 2 \times 1 \quad \text{(Eq. 2.1)}$$

> **Formula — Permutations of Subsets:**
> The number of permutations of subsets of r elements from a set of n different elements:
>
> $$P_r^n = n \times (n-1) \times (n-2) \times \cdots \times (n-r+1) = \frac{n!}{(n-r)!} \quad \text{(Eq. 2.2)}$$

#### Example 2.8 — Printed Circuit Board

A printed circuit board has 8 different locations. 4 different components are to be placed on the board. How many different designs?

$$P_4^8 = 8 \times 7 \times 6 \times 5 = \frac{8!}{4!} = 1680 \text{ different designs}$$

---

### Permutations of Similar Objects

> **Formula — Permutations of Similar Objects:**
> The number of permutations of n = n₁ + n₂ + ··· + nᵣ objects of which n₁ are of one type, n₂ are of a second type, ..., nᵣ are of an r-th type:
>
> $$\frac{n!}{n_1! n_2! n_3! \cdots n_r!} \quad \text{(Eq. 2.3)}$$

#### Example 2.9 — Hospital Schedule

A hospital operating room schedules 3 knee surgeries (k) and 2 hip surgeries (h) in a day.

$$\frac{5!}{2! \cdot 3!} = 10 \text{ possible sequences}$$

The 10 sequences: {kkkhh, kkhkh, kkhhk, khkkh, khkhk, khhkk, hkkkh, hkkhk, hkhkk, hhkkk}

---

### Combinations

> **Definition — Combinations:**
> The number of **combinations** (subsets of r elements, where **order does not matter**) that can be selected from a set of n elements:
>
> $$C_r^n = \binom{n}{r} = \frac{n!}{r!(n-r)!} \quad \text{(Eq. 2.4)}$$

For S = {a, b, c, d}, the subset {a, c} is indicated by marking a and c with *.

#### Example 2.10 — PCB Layout (Identical Components)

A PCB has 8 locations. 5 **identical** components are to be placed. How many different designs?

$$\binom{8}{5} = \frac{8!}{5! \cdot 3!} = 56 \text{ possible designs}$$

---

**Sampling with and without replacement:**
- **With replacement:** selected item is returned before next selection.
- **Without replacement:** selected item is NOT returned; each item can be selected only once.

#### Example 2.11 — Sampling without Replacement

A bin has 50 parts: 3 defective and 47 nondefective. A sample of 6 is selected without replacement. How many different samples contain **exactly 2 defective parts**?

- Step 1: Choose 2 defective from 3: $\binom{3}{2} = \frac{3!}{2!1!} = 3$
- Step 2: Choose 4 nondefective from 47: $\binom{47}{4} = \frac{47!}{4! \cdot 43!} = 178,365$
- By multiplication rule: $3 \times 178,365 = 535,095$

Total number of different subsets of size 6:
$$\binom{50}{6} = \frac{50!}{6! \cdot 44!} = 15,890,700$$

---

## 2.3 Interpretations and Axioms of Probability

> **Probability** is used to quantify the likelihood, or chance, that an outcome of a random experiment will occur.

Probability is a number assigned from the interval [0, 1]:
- **0** → outcome will not occur
- **1** → outcome will occur with certainty
- Higher numbers → more likely

**Two interpretations of probability:**
1. **Subjective probability (degree of belief):** Personal assessment of likelihood.
2. **Relative frequency interpretation:** Probability = limiting value of the proportion of times an outcome occurs in n repetitions as n → ∞.

### Equally Likely Outcomes

> **Definition — Equally Likely Outcomes:**
> Whenever a sample space consists of N possible outcomes that are equally likely, the probability of each outcome is **1/N**.

#### Example 2.12 — Laser Diodes

30% of laser diodes in a batch of 100 meet a customer's minimum power requirements. If one is selected randomly:
- E = subset of 30 satisfactory diodes
- P(E) = 30 × (0.01) = **0.30**

---

> **Definition — Probability of an Event:**
> For a discrete sample space, the **probability of an event E**, denoted P(E), equals the **sum of the probabilities of the outcomes in E**.

#### Example 2.13 — Probabilities of Events

A random experiment can result in outcomes {a, b, c, d} with probabilities 0.1, 0.3, 0.5, 0.1 respectively. Let:
- A = {a, b}, B = {b, c, d}, C = {d}

Then:
- P(A) = 0.1 + 0.3 = 0.4
- P(B) = 0.3 + 0.5 + 0.1 = 0.9
- P(C) = 0.1
- A ∩ B = {b}, so P(A ∩ B) = 0.3
- A ∪ B = {a, b, c, d}, so P(A ∪ B) = 1
- P(A′) = 0.6, P(B′) = 0.1, P(C′) = 0.9

#### Example 2.14 — Manufacturing Inspection

From Example 2.11: bin of 50 parts, 6 selected randomly without replacement. What is P(exactly 2 defective)?

$$P = \frac{535,095}{15,890,700} = 0.034$$

Probability of **no** defective parts:
$$P = \frac{\binom{47}{6}}{\binom{50}{6}} = \frac{10,737,573}{15,890,700} = 0.676$$

---

### Axioms of Probability

> **Axioms of Probability:**
> Probability is a number assigned to each member of a collection of events from a random experiment that satisfies:
> 1. **P(S) = 1** where S is the sample space
> 2. **0 ≤ P(E) ≤ 1** for any event E
> 3. For two events E₁ and E₂ with **E₁ ∩ E₂ = Ø**:
>    $$P(E_1 \cup E_2) = P(E_1) + P(E_2)$$

**Results derived from the axioms:**
- P(Ø) = 0
- For any event E: **P(E′) = 1 − P(E)**
- If E₁ is contained in E₂: **P(E₁) ≤ P(E₂)**

---

## 2.4 Unions of Events and Addition Rules

> **General Addition Rule (Probability of a Union):**
> $$P(A \cup B) = P(A) + P(B) - P(A \cap B) \quad \text{(Eq. 2.5)}$$

For **mutually exclusive** events (A ∩ B = Ø):
$$P(A \cup B) = P(A) + P(B) \quad \text{(Eq. 2.6)}$$

#### Example 2.15 — Semiconductor Wafers

940 wafers in a manufacturing process. H = high contamination, C = center of sputtering tool.

**Table 2.1:**

| Contamination | Center | Edge | Total |
|---|---|---|---|
| Low | 514 | 68 | 582 |
| High | 112 | 246 | 358 |
| Total | 626 | 314 | 940 |

- P(H) = 358/940
- P(C) = 626/940
- P(H ∩ C) = 112/940

$$P(H \cup C) = \frac{358}{940} + \frac{626}{940} - \frac{112}{940} = \frac{872}{940}$$

---

### Three or More Events

$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C) \quad \text{(Eq. 2.7)}$$

> **Mutually Exclusive Events (Multiple):**
> A collection E₁, E₂, ..., Eₖ is **mutually exclusive** if for all pairs: Eᵢ ∩ Eⱼ = Ø
>
> $$P(E_1 \cup E_2 \cup \cdots \cup E_k) = P(E_1) + P(E_2) + \cdots + P(E_k) \quad \text{(Eq. 2.8)}$$

#### Example 2.16 — pH

Let X denote the pH of a sample. The event {6.5 < X ≤ 7.8} can be expressed as a union of mutually exclusive events, for example:

$$P(6.5 < X \leq 7.8) = P(6.5 \leq X \leq 7.0) + P(7.0 < X \leq 7.5) + P(7.5 < X \leq 7.8)$$

or alternatively:

$$P(6.5 < X \leq 7.8) = P(6.5 < X \leq 6.6) + P(6.6 < X \leq 7.1) + P(7.1 < X \leq 7.4) + P(7.4 < X \leq 7.8)$$

The best choice depends on which probabilities are available.

---

## 2.5 Conditional Probability

Sometimes probabilities need to be revised when additional information becomes available. Incorporating this information is done by defining the **conditional probability**.

> **Definition — Conditional Probability:**
> The **conditional probability** of event B given event A, denoted P(B | A), is:
>
> $$P(B \mid A) = \frac{P(A \cap B)}{P(A)} \quad \text{for } P(A) > 0 \quad \text{(Eq. 2.9)}$$

**Interpretation:** P(B | A) = (number of outcomes in A ∩ B) / (number of outcomes in A), i.e., the relative frequency of B among the trials that produce an outcome in A.

**Digital communication channel example:**
- Error rate = 1 bit per 1000 transmitted
- If errors occur in bursts affecting many consecutive bits, then knowing the previous bit was in error increases the probability the next bit is also in error (above 1/1000).

**Manufacturing example:**
- 10% of parts have visible surface flaws
- 25% of parts with surface flaws are (functionally) defective: P(D | F) = 0.25
- Only 5% of parts without surface flaws are defective: P(D | F′) = 0.05

#### Example 2.17 — Surface Flaws and Defectives

**Table 2.2 — Parts Classified:**

| | Surface Flaw Yes (F) | Surface Flaw No | Total |
|---|---|---|---|
| Defective Yes (D) | 10 | 18 | 28 |
| No | 30 | 342 | 372 |
| Total | 40 | 360 | 400 |

- P(D | F) = 10/40 = 0.25
- P(D | F′) = 18/360 = 0.05

> **Practical Interpretation:** The probability of being defective is five times greater for parts with surface flaws, suggesting a possible link between surface flaws and functional defects.

#### Example 2.18 — Tree Diagram for Parts Classified

From Table 2.2:
$$P(D \mid F) = \frac{P(D \cap F)}{P(F)} = \frac{10/400}{40/400} = \frac{10}{40}$$

All four probabilities are different:
- P(F) = 40/400
- P(D) = 28/400
- P(F | D) = 10/28
- P(D | F) = 10/40

The tree diagram (Figure 2.12) shows two levels: Surface flaw (yes/no) then Defective (yes/no), with the appropriate conditional probabilities on each branch.

---

### Random Samples and Conditional Probability

> **Definition — Random Sample:**
> To select **randomly** implies that at each step of the sample, the items that remain in the batch are **equally likely to be selected**.

**Example:** Batch has 10 parts from tool 1 and 40 parts from tool 2. Two parts selected without replacement.

- P(E₁) = 10/50 (first part from tool 1)
- P(E₂ | E₁) = 40/49 (second from tool 2 | first from tool 1)
- P(E) = P(E₂ | E₁) × P(E₁) = (40/49)(10/50) = 8/49

---

## 2.6 Intersections of Events and Multiplication and Total Probability Rules

> **Multiplication Rule:**
> $$P(A \cap B) = P(B \mid A) P(A) = P(A \mid B) P(B) \quad \text{(Eq. 2.10)}$$

#### Example 2.19 — Machining Stages

First stage meets specifications: P(A) = 0.90
Given first stage passes, second stage passes: P(B | A) = 0.95

$$P(A \cap B) = P(B \mid A)P(A) = 0.95 \times 0.90 = 0.855$$

> **Practical Interpretation:** The probability that both stages meet specifications is ~0.85. If more stages are needed, probability decreases further, so high per-stage success is critical.

---

### Total Probability Rule

For any events A and B:
$$B = (A \cap B) \cup (A' \cap B)$$

Since A and A′ are mutually exclusive, A∩B and A′∩B are mutually exclusive.

> **Total Probability Rule (Two Events):**
> $$P(B) = P(B \cap A) + P(B \cap A') = P(B \mid A)P(A) + P(B \mid A')P(A') \quad \text{(Eq. 2.11)}$$

#### Example 2.20 — Semiconductor Contamination

| Probability of Failure | Level of Contamination | Probability of Level |
|---|---|---|
| 0.1 | High | 0.2 |
| 0.005 | Not high | 0.8 |

Let F = product fails, H = high contamination:
- P(F | H) = 0.10, P(F | H′) = 0.005
- P(H) = 0.20, P(H′) = 0.80

$$P(F) = 0.10(0.20) + 0.005(0.80) = 0.024$$

> This is the **weighted average** of the two probabilities of failure.

---

> **Total Probability Rule (Multiple Events):**
> Assume E₁, E₂, ..., Eₖ are k **mutually exclusive and exhaustive** sets. Then:
>
> $$P(B) = P(B \mid E_1)P(E_1) + P(B \mid E_2)P(E_2) + \cdots + P(B \mid E_k)P(E_k) \quad \text{(Eq. 2.12)}$$

---

## 2.7 Independence

In some cases, P(B | A) = P(B) — knowledge that A occurred does not affect the probability of B.

> **Definition — Independence (Two Events):**
> Two events A and B are **independent** if any one of the following equivalent statements is true:
> 1. P(A | B) = P(A)
> 2. P(B | A) = P(B)
> 3. **P(A ∩ B) = P(A)P(B)** ← (Eq. 2.13)

Note: **Mutually exclusive ≠ Independent.** Mutually exclusive is about outcomes (sets), while independence is about the probability model.

It follows that: P(A′ ∩ B′) = P(A′)P(B′) if A and B are independent.

#### Example 2.21 — Sampling with Replacement

Bin of 50 parts (3 defective, 47 nondefective). A part is selected, **replaced**, then a second part is selected.

- P(B | A) = 3/50 (bin is reset after replacement)
- P(A ∩ B) = P(B | A)P(A) = (3/50)(3/50) = 9/2500

Since P(B | A) = P(B), the two events are **independent**.

#### Example 2.22 — Without Replacement (Not Independent)

Same bin, 6 parts selected without replacement. A = first part defective, B = second part defective.

- P(B | A) = 2/49
- P(B) = P(B | A)P(A) + P(B | A′)P(A′) = (2/49)(3/50) + (3/49)(47/50) = 3/50

Since P(B | A) = 2/49 ≠ P(B) = 3/50, the events are **NOT independent**.

---

> **Definition — Independence (Multiple Events):**
> E₁, E₂, ..., Eₙ are **independent** if and only if for any subset of these events:
>
> $$P(E_{i_1} \cap E_{i_2} \cap \cdots \cap E_{i_k}) = P(E_{i_1}) \times P(E_{i_2}) \times \cdots \times P(E_{i_k}) \quad \text{(Eq. 2.14)}$$

#### Example 2.23 — Series Circuit

Circuit operates only if both devices function. Devices fail independently.
- P(L) = 0.8, P(R) = 0.9

$$P(\text{circuit operates}) = P(L \cap R) = P(L)P(R) = 0.80 \times 0.90 = 0.72$$

> **Practical Interpretation:** The circuit probability degrades to ~0.7 when all devices must be functional. Each device's reliability must be high in a series configuration.

#### Example 2.24 — Parallel Circuit

Circuit operates if **at least one** device functions. Two devices in parallel: P(T) = 0.95, P(B) = 0.90.

$$P(T \cup B) = 1 - P(T' \cap B') = 1 - P(T')P(B') = 1 - (0.05)(0.10) = 1 - 0.005 = 0.995$$

> **Practical Interpretation:** The parallel circuit's operating probability (0.995) is greater than either individual device. This is the advantage of parallel architecture.

#### Example 2.25 — Advanced Circuit

A more complex circuit partitioned into three columns (left: 3 parallel units with P = 0.9 each, middle: 2 parallel units with P = 0.95 each, right: 1 unit with P = 0.99).

- P(L) = 1 − 0.1³
- P(M) = 1 − 0.05²
- P(circuit) = (1 − 0.1³)(1 − 0.05²)(0.99) = **0.987**

---

## 2.8 Bayes' Theorem

Conditional probabilities commonly give the probability of an **outcome given a condition** (e.g., probability of failure given high contamination). After a random experiment generates an outcome, we are naturally interested in the **probability that a condition was present given the outcome** (e.g., probability of high contamination given a semiconductor failure).

Thomas Bayes addressed this in the 1700s.

From the definition of conditional probability:
$$P(A \cap B) = P(A \mid B)P(B) = P(B \mid A)P(A)$$

This gives:

> **Bayes' Theorem (Basic Form):**
> $$P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)} \quad \text{for } P(B) > 0 \quad \text{(Eq. 2.15)}$$

#### Example 2.26 — Semiconductor Contamination (Revisited)

Using data from Example 2.20:
- P(F | H) = 0.10, P(H) = 0.20
- P(F) = 0.024 (calculated earlier)

$$P(H \mid F) = \frac{P(F \mid H)P(H)}{P(F)} = \frac{0.10 \times 0.20}{0.024} = \frac{0.02}{0.024} = 0.83$$

---

By substituting the total probability rule for P(B):

> **Bayes' Theorem (General Form):**
> If E₁, E₂, ..., Eₖ are k mutually exclusive and exhaustive events and B is any event:
>
> $$P(E_1 \mid B) = \frac{P(B \mid E_1)P(E_1)}{P(B \mid E_1)P(E_1) + P(B \mid E_2)P(E_2) + \cdots + P(B \mid E_k)P(E_k)} \quad \text{(Eq. 2.16)}$$
>
> for P(B) > 0.

> **Note:** The numerator always equals one of the terms in the sum in the denominator.

#### Example 2.27 — Medical Diagnostic

A new medical procedure is effective in early detection of an illness. A medical screening of the population is proposed.
- **Sensitivity** (P(positive | has illness)) = 0.99
- **Specificity** (P(negative | no illness)) = 0.95
  → P(positive | no illness) = P(S | D′) = 0.05
- **Incidence** of illness = P(D) = 0.0001

You test positive. What is P(D | S)?

$$P(D \mid S) = \frac{P(S \mid D)P(D)}{P(S \mid D)P(D) + P(S \mid D')P(D')}$$

$$= \frac{0.99 \times 0.0001}{0.99(0.0001) + 0.05(1 - 0.0001)} = \frac{0.000099}{0.000099 + 0.049995} = \frac{1}{506} \approx 0.002$$

> **Practical Interpretation:** Even though the test is effective (high sensitivity and specificity), the probability of actually having the illness given a positive result is only **0.002** (0.2%) because the incidence in the general population is very low (0.01%). This illustrates the importance of prior probability in Bayesian reasoning.

---

## 2.9 Random Variables

Often the outcome from a random experiment is summarized by a single number.

> **Definition — Random Variable:**
> A **random variable** is a function that assigns a real number to each outcome in the sample space of a random experiment.

> **Notation:**
> - Random variable: denoted by an uppercase letter such as **X**
> - Measured value after the experiment: denoted by a lowercase letter such as **x = 70 milliamperes**

**Types of Random Variables:**

A measurement like current in a copper wire can assume any value in an interval — this leads to a **continuous random variable**. A count (number of defective items) takes on only integer values — this is a **discrete random variable**.

| Type | Description | Example |
|------|-------------|---------|
| **Continuous** | Can take any value in an interval | Current (mA), recycle time (sec) |
| **Discrete** | Takes on a countable set of values | Number of defects, number of errors |

---

## Summary of Key Formulas

| Formula | Equation |
|---------|----------|
| Permutations of n elements | n! |
| Permutations of r from n | P_r^n = n! / (n−r)! |
| Permutations of similar objects | n! / (n₁! n₂! ··· nᵣ!) |
| Combinations | C_r^n = n! / (r!(n−r)!) |
| Probability of a union | P(A∪B) = P(A) + P(B) − P(A∩B) |
| Mutually exclusive union | P(A∪B) = P(A) + P(B) |
| Conditional probability | P(B\|A) = P(A∩B) / P(A) |
| Multiplication rule | P(A∩B) = P(B\|A)P(A) |
| Total probability (2 events) | P(B) = P(B\|A)P(A) + P(B\|A′)P(A′) |
| Independence | P(A∩B) = P(A)P(B) |
| Bayes' theorem | P(A\|B) = P(B\|A)P(A) / P(B) |

---
# Extra problems
A class in probability theory consists of 6 men and 4 women. An examination is given,
and the students are ranked according to their performance. Assume that no two
students obtain the same score.
(a) How many different rankings are possible?
(b) If the men are ranked just among themselves and the women just among them-
selves, how many different rankings are possible?
Solution. (a) Because each ranking corresponds to a particular ordered arrangement
of the 10 people, the answer to this part is 10! = 3,628,800.
(b) Since there are 6! possible rankings of the men among themselves and 4! possi-
ble rankings of the women among themselves, it follows from the basic principle that
there are (6!)(4!) = (720)(24) = 17,280 possible rankings in this case.


Ms. Jones has 10 books that she is going to put on her bookshelf. Of these, 4 are
mathematics books, 3 are chemistry books, 2 are history books, and 1 is a language
book. Ms. Jones wants to arrange her books so that all the books dealing with the
same subject are together on the shelf. How many different arrangements are
possible?
Solution. There are 4! 3! 2! 1! arrangements such that the mathematics books are
first in line, then the chemistry books, then the history books, and then the language
book. Similarly, for each possible ordering of the subjects, there are 4! 3! 2! 1! possible
arrangements. Hence, as there are 4! possible orderings of the subjects, the desired
answer is 4! 4! 3! 2! 1! = 6912. 


How many different letter arrangements can be formed from the letters PEPPER?
Solution. We first note that there are 6! permutations of the letters P1E1P2P3E2R
when the 3P’s and the 2E’s are distinguished from each other. However, consider
any one of these permutations—for instance, P1P2E1P3E2R. If we now permute the
P’s among themselves and the E’s among themselves, then the resultant arrangement
would still be of the form PPEPER. That is, all 3! 2! permutations
P1P2E1P3E2R P1P2E2P3E1R
P1P3E1P2E2R P1P3E2P2E1R
P2P1E1P3E2R P2P1E2P3E1R
P2P3E1P1E2R P2P3E2P1E1R
P3P1E1P2E2R P3P1E2P2E1R
P3P2E1P1E2R P3P2E2P1E1R
are of the form PPEPER. Hence, there are 6!/(3! 2!) = 60 possible letter arrange-
ments of the letters PEPPER. 

A chess tournament has 10 competitors, of which 4 are Russian, 3 are from the United
States, 2 are from Great Britain, and 1 is from Brazil. If the tournament result lists just
the nationalities of the players in the order in which they placed, how many outcomes
are possible?
Solution. There are 10!
4! 3! 2! 1! = 12,60
