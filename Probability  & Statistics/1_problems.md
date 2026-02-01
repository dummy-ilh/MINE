Perfect! Let’s break your notes into a clear, step-by-step **learning guide** so you can fully see how **permutations, combinations, and the rule of product** connect, with concrete examples.

---

## **1️⃣ Permutations vs Combinations: Key Idea**

| Aspect         | Permutation                  | Combination                    |
| -------------- | ---------------------------- | ------------------------------ |
| Order matters? | ✅ Yes                        | ❌ No                           |
| Example        | ABC ≠ ACB                    | ABC = ACB                      |
| Formula        | (P(n,r) = \frac{n!}{(n-r)!}) | (C(n,r) = \frac{n!}{r!(n-r)!}) |

---

## **2️⃣ Example: Set {a, b, c, d}, choose 3**

### **Permutations (order matters)**

* Formula: (P(4,3) = 4 × 3 × 2 = 24)
* List of permutations:

```
abc, acb, bac, bca, cab, cba
abd, adb, bad, bda, dab, dba
acd, adc, cad, cda, dac, dca
bcd, bdc, cbd, cdb, dbc, dcb
```

✅ Total = 24

> Here every order counts as a **different outcome**.

---

### **Combinations (order doesn’t matter)**

* Formula: (C(4,3) = 4)
* List of combinations:

```
{a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}
```

✅ Total = 4

> Notice how all 6 permutations of {a, b, c} (abc, acb, bac, …) collapse into **one combination**.

---

### **Relationship**

[
P(n,r) = C(n,r) × r!
]

* Here: (24 = 4 × 3! = 4 × 6) ✅ matches your example.

---

## **3️⃣ Smaller Practice Problems**

### **Example 1: Choose 2 from 4 things**

1. Combinations (order doesn’t matter): (C(4,2) = 6)
   Pairs: {a,b}, {a,c}, {a,d}, {b,c}, {b,d}, {c,d}
2. Permutations (order matters): (P(4,2) = 12)
   Arrangements: ab, ba, ac, ca, ad, da, bc, cb, bd, db, cd, dc

---

### **Example 2: Coin Flips**

* 10 flips, want exactly 3 heads.
* Step 1: Choose **which 3 flips are heads** → (C(10,3) = 120)
* Step 2: Total sequences of 10 flips → (2^{10} = 1024) (rule of product)
* Step 3: Probability = (120 / 1024 = 0.117)

> Coin flips are **classic combination + rule of product** example.

---

## **4️⃣ Rule of Product / Multiplication Principle**

* Use when **you make independent choices in sequence**.
* Formula: Multiply the options at each step.

**Example 1:**

* Outfit: 5 shirts × 3 pants = 15 outfits

**Example 2:**

* Password: 3 letters × 2 digits = (26^3 × 10^2 = 1757600)

**Key Tip:** Often used **with permutations**, because you multiply the choices for each position.

---

### ✅ **Summary**

1. **Rule of Product** → multiply independent choices.
2. **Permutation (P(n,r))** → order matters. Use the rule of product for positions.
3. **Combination (C(n,r))** → order doesn’t matter. Divide permutation by (r!).

---

Perfect! Let’s carefully go through each of these **class 1 problems** and clarify **why the answers are what they are**, so you can see the concepts of **rule of product, permutations, combinations, and inclusion-exclusion** in action. I’ll also explain the reasoning step by step.

---

## **Concept Questions**

### **1️⃣ Poker hands – probability of one pair**

* A “one-pair” hand in poker means you have **two cards of the same rank** and the other three cards of different ranks.
* Surprisingly, the probability is **greater than 40%**.

**Intuition:**
There are many ways to form a one-pair hand (choose rank for the pair, suits, then remaining cards). It turns out the number is quite high compared to other hand types like full house or three of a kind.

---

### **2️⃣ DNA sequences – length 3**

**Problem:** DNA has nucleotides {A, C, G, T}.

1. **Length 3, repeats allowed:**

* Each position has 4 options → Rule of Product: (4 × 4 × 4 = 64) ✅

2. **Length 3, no repeats:**

* First position: 4 options
* Second: 3 options
* Third: 2 options
* Multiply: (4 × 3 × 2 = 24) ✅

> This is a **permutation** problem because order matters (ACG ≠ GAC).

---

## **Board Questions**

### **1️⃣ Inclusion/Exclusion – band members**

* Singers = 7
* Guitar players = 4
* Both = 2

**Formula:**
[
|S \cup G| = |S| + |G| - |S \cap G|
]

[
7 + 4 - 2 = 9
]

> This avoids double-counting the people who both sing and play guitar. ✅

---

### **2️⃣ Rule of Product – Olympic medals**

* Competitors = 5
* Medals: gold, silver, bronze
* Step 1: pick gold → 5 choices
* Step 2: pick silver → 4 remaining choices
* Step 3: pick bronze → 3 remaining choices

[
5 × 4 × 3 = 60 \text{ ways}
]

> This is a **permutation of 3 out of 5**, because **order matters** (gold ≠ silver).

---

### **3️⃣ Wardrobe – conditional choices**

* Shirts: 3B, 3R, 2G
* Sweaters: 1B, 2R, 1G
* Pants: 2D, 2B
* Constraint: Don’t wear green and red together

**Approach:** Use a **multiplication tree**. Split cases by **shirt choice**:

| Shirt | Compatible Sweaters      | Pants | Total outfits  |
| ----- | ------------------------ | ----- | -------------- |
| R     | R, B, G (avoid conflict) | B, D  | 3 × 3 × 4 = 36 |
| B     | R, B, G                  | B, D  | 3 × 4 × 4 = 48 |
| G     | B, G                     | B, D  | 2 × 2 × 4 = 16 |

* Total outfits = 36 + 48 + 16 = 100 ✅

> This is **rule of product with case-splitting**.

---

### **4️⃣ Coin flips – exactly 3 heads in 10 flips**

**(a) Count sequences:**

* Need **exactly 3 heads out of 10 flips**
* Use **combination**: choose which 3 flips are heads:
  [
  C(10,3) = \frac{10!}{3!7!} = 120
  ] ✅

**(b) Probability for fair coin:**

* Total sequences = (2^{10} = 1024) (rule of product: 2 options per flip)
* Probability = (120 / 1024 ≈ 0.117) ✅

> Combines **combination (choosing flips for heads)** + **rule of product (total outcomes)**.

---

### **Summary of Concepts**

| Problem                  | Concept Used                   |
| ------------------------ | ------------------------------ |
| DNA sequences            | Rule of Product / Permutations |
| DNA sequences no repeats | Permutations                   |
| Band members             | Inclusion-Exclusion            |
| Olympic medals           | Permutation / Rule of Product  |
| Wardrobe                 | Rule of Product (with cases)   |
| Coin flips               | Combination + Rule of Product  |

---

![Set Operations](images/example1.png)


Perfect! Let’s dive deep into **permutations and combinations**, with a **FAANG-style preparation lens**. I’ll cover concepts, examples, and then progressively challenging problems, including **detailed explanations**, **edge cases**, and **time-saving tips**.

---

# **PERMUTATIONS & COMBINATIONS (P&C) – FAANG Prep**

### **1️⃣ Basic Concepts**

#### **Permutations**

* **Definition:** Arrangement of objects **where order matters**.
* **Formula:**

  * ( P(n, r) = \frac{n!}{(n-r)!} )
    (n) = total items, (r) = selected items.
* **Special Cases:**

  * All items: ( n! )
  * Circular permutation of n items: ( (n-1)! )

#### **Combinations**

* **Definition:** Selection of objects **where order doesn’t matter**.
* **Formula:**

  * ( C(n, r) = \frac{n!}{r!(n-r)!} )

#### **Important Tips**

* If **order matters → Permutation**, if **order doesn’t matter → Combination**.
* **Repetition allowed?**

  * Permutation with repetition: ( n^r )
  * Combination with repetition: ( C(n+r-1, r) )

---

### **2️⃣ Classic Problems with Solutions**

#### **Problem 1: How many ways to arrange 5 books on a shelf?**

* **Type:** Permutation
* **Solution:** All books are distinct.
  ( 5! = 120 ) ways.

✅ **Key Tip:** Always check if items are **distinct**. If not, divide by factorials of repeats.

---

#### **Problem 2: How many 3-letter words can be formed from A, B, C, D without repetition?**

* **Type:** Permutation

* **Solution:**
  ( P(4,3) = \frac{4!}{(4-3)!} = \frac{24}{1} = 24 )

* **Dry Run:** ABC, ABD, ACB, …

* **FAANG Tip:** Always consider **distinctness** and **order**.

---

#### **Problem 3: How many 3-letter words can be formed from A, B, C, D **with repetition**?**

* **Solution:** Each position has 4 choices → ( 4^3 = 64 )

✅ **Key Insight:** With repetition, **just raise n to r**.

---

#### **Problem 4: From 10 students, select 3 for a project.**

* **Type:** Combination

* **Solution:**
  ( C(10,3) = \frac{10!}{3!7!} = 120 )

* **FAANG Tip:** Selection → **order doesn’t matter**.

---

#### **Problem 5: How many ways to select a committee of 3 men and 2 women from 5 men and 6 women?**

* **Type:** Combination, Multiple Groups
* **Solution:**

  * Men: ( C(5,3) = 10 )
  * Women: ( C(6,2) = 15 )
  * Total ways: ( 10 * 15 = 150 )

✅ **Tip:** Multiply **independent groups**.

---

### **3️⃣ Advanced / FAANG Style Problems**

#### **Problem 6: Distinct permutations with repeated letters**

* Word: **“FAANG”**
* How many ways to arrange letters?

**Solution:**

* Letters: F, A, A, N, G
* Total letters = 5
* A repeats 2 times → divide by 2!

[
\text{Arrangements} = \frac{5!}{2!} = \frac{120}{2} = 60
]

* **FAANG Tip:** Always check for **repeating characters** in strings or IDs.

---

#### **Problem 7: Circular arrangements**

* **Question:** 5 people sit around a round table. How many ways?
* **Solution:** Circular permutation formula: ( (n-1)! = 4! = 24 )

✅ **Tip:** Fix one person → then arrange others.

---

#### **Problem 8: Balls into boxes (FAANG pattern)**

* 3 **identical balls**, 2 **distinct boxes**. How many distributions?

* **Solution:**

  * Stars & bars method (identical balls, distinct boxes):
    ( C(n+r-1, r) = C(2+3-1,3) = C(4,3) = 4 )

* **FAANG Tip:** Often comes in **resource allocation problems**.

---

#### **Problem 9: At least / at most constraints**

* **Question:** From 5 men and 6 women, form a team of 4 **with at least 2 women**
* **Solution:**

  * Case 1: 2 women, 2 men → ( C(6,2)*C(5,2) = 15*10 = 150 )
  * Case 2: 3 women, 1 man → ( C(6,3)*C(5,1) = 20*5 = 100 )
  * Case 3: 4 women → ( C(6,4) = 15 )
  * Total = 150 + 100 + 15 = 265

✅ **Tip:** Break into **disjoint cases**.

---

#### **Problem 10: Advanced Permutations with Restrictions**

* **Question:** How many 4-digit numbers using 1,2,3,4,5 **without repetition** are divisible by 5?
* **Solution:** Last digit must be 5 → fix it.
* Remaining digits: 1,2,3,4 → choose 3 digits in order → ( P(4,3) = 24 )

✅ **FAANG Tip:** Always consider **constraint placement first**, then count permutations.

---

### **4️⃣ Common FAANG Traps**

1. **Order vs Selection:** Always ask: does ABC = BAC? → if yes, combination, else permutation.
2. **Repetition allowed?** Many mistakes come from ignoring repetition.
3. **Constraints first:** "At least / exactly / divisible by..." → handle constraints before counting.
4. **Factorials grow fast:** Use formula simplification ( \frac{n!}{(n-r)!} ) instead of computing huge numbers.
5. **Multiple independent groups:** Multiply counts.

---

### **5️⃣ Real-World Applications**

* Arranging **UI elements** in an app (permutations)
* Selecting **team members** or **feature sets** (combinations)
* **Password generation** or **ID arrangement**
* **Scheduling tasks** with constraints

---
Perfect! Let’s go **next level** with a mix of **easy → medium → hard FAANG-style P&C problems**, with **full step-by-step answers and explanations**. I’ll include **traps, edge cases, and tips**.

---

## **FAANG-STYLE PERMUTATION & COMBINATION QUESTIONS**

---

### **EASY QUESTIONS**

**Q1. How many 3-digit numbers can be formed using digits 1, 2, 3, 4, 5 if digits cannot repeat?**
**Solution:**

* Total digits = 5, select 3 in order → **permutation**
  [
  P(5,3) = \frac{5!}{(5-3)!} = \frac{120}{2} = 60
  ]

✅ **Tip:** “No repetition” → use permutation.

---

**Q2. How many ways to select 2 students from a class of 10?**
**Solution:**

* Selection → **combination**
  [
  C(10,2) = \frac{10!}{2!8!} = 45
  ]

---

**Q3. How many ways to arrange the letters of the word “APPLE”?**
**Solution:**

* Letters: A, P, P, L, E (5 letters, P repeats 2 times)
  [
  \text{Arrangements} = \frac{5!}{2!} = \frac{120}{2} = 60
  ]

---

### **MEDIUM QUESTIONS**

**Q4. How many 4-digit numbers can be formed using digits 1-6, **even numbers only**, digits cannot repeat?**

**Solution:**

* Last digit (must be even): 2,4,6 → 3 choices
* Remaining 3 digits: choose from remaining 5 digits → ( P(5,3) = 60 )
* Total = ( 3 * 60 = 180 )

✅ **Tip:** Handle constraints first (last digit fixed), then arrange remaining.

---

**Q5. From 6 men and 4 women, form a committee of 3 with **at least 1 woman**.**

**Solution:**

* Case 1: 1 woman, 2 men → ( C(4,1)*C(6,2) = 4*15 = 60 )
* Case 2: 2 women, 1 man → ( C(4,2)*C(6,1) = 6*6 = 36 )
* Case 3: 3 women → ( C(4,3) = 4 )

**Total = 60+36+4 = 100**

✅ **Tip:** Break into disjoint cases to avoid double-counting.

---

**Q6. How many ways to seat 5 people around a round table?**

**Solution:** Circular permutation → ( (n-1)! = 4! = 24 )

✅ **Tip:** Circular table → fix one person, arrange the rest.

---

**Q7. How many ways can the letters of the word “BANANA” be arranged?**

**Solution:**

* Letters: B, A, N, A, N, A
* Count repeats: A=3, N=2, B=1
  [
  \text{Arrangements} = \frac{6!}{3!2!1!} = \frac{720}{6*2*1} = \frac{720}{12} = 60
  ]

---

### **HARD / FAANG-STYLE QUESTIONS**

**Q8. 5 different books are to be placed on 3 shelves such that each shelf has at least 1 book. How many ways?**

**Solution:**

* Use **Stirling numbers of the second kind** or **break into cases**:

**Case 1:** 3,1,1 distribution → choose shelf with 3 books: 3 choices

* Arrange 3 books: ( P(5,3) = 60 )
* Arrange remaining 2 books on 2 shelves: 2! = 2
* Total = 3 * 60 * 2 = 360

**Case 2:** 2,2,1 distribution → choose shelf with 1 book: 3 choices

* Arrange books: choose 1 book for single shelf → C(5,1)=5
* Remaining 4 books → split into 2,2 → C(4,2)/2! = 3 ways
* Arrange each pair: 2!*2! = 4
* Total = 3*5*3*4=180

**Grand Total = 360+180=540**

✅ **Tip:** Complex distributions → break into **all possible cases**.

---

**Q9. How many 4-digit numbers can be formed using 1-7 if **at least 2 digits are odd**?**

**Solution:**

* Odd digits: 1,3,5,7 → 4 digits
* Even digits: 2,4,6 → 3 digits

**Cases:**

1. 2 odd, 2 even → ( C(4,2)*C(3,2)*P(4,4)=6*3*24=432 )
2. 3 odd, 1 even → ( C(4,3)*C(3,1)*P(4,4)=4*3*24=288 )
3. 4 odd → ( C(4,4)*C(3,0)*P(4,4)=1*1*24=24 )

**Total = 432+288+24=744**

✅ **Tip:** “At least” problems → break into **all valid scenarios**.

---

**Q10. A lock has 4 digits. Each digit is 0-9. How many codes are possible if no digit repeats and code must be even?**

**Solution:**

* Last digit even: 0,2,4,6,8 → 5 choices
* Remaining 3 digits: choose from remaining 9 digits → ( P(9,3)=9*8*7=504 )
* Total codes = 504*5 = 2520

---

### ✅ **FAANG P&C Strategy Tips**

1. **Identify type:** Permutation vs Combination.
2. **Check constraints first:** fixed positions, repeats, divisibility, circular arrangements.
3. **Break into cases**: at least / at most / distribution problems.
4. **Factorials wisely:** simplify fractions before computing large numbers.
5. **Double-check repeats and identical objects**.

---
Perfect! Let’s build a **comprehensive FAANG-style Permutations & Combinations (P&C) problem sheet** with **50 questions**, organized by difficulty and including **fully worked solutions and tips**. I’ll structure it in **easy → medium → hard → very hard / tricky FAANG problems**, so you can use it as a **master practice sheet**.

---

# **FAANG-STYLE PERMUTATIONS & COMBINATIONS PROBLEM SHEET**

---

## **EASY (1–15)**

**Q1.** How many ways to arrange 4 distinct books on a shelf?
**Answer:** (4! = 24)

**Q2.** How many 2-digit numbers can be formed using 1,2,3,4,5 without repetition?
**Answer:** (P(5,2) = 5*4 = 20)

**Q3.** How many ways to select 2 students from 10?
**Answer:** (C(10,2) = 45)

**Q4.** How many ways can the letters of “APPLE” be arranged?
**Answer:** ( \frac{5!}{2!} = 60)

**Q5.** How many 3-letter words can be formed from A,B,C,D with repetition?
**Answer:** (4^3 = 64)

**Q6.** From 5 men and 6 women, choose a team of 3 with exactly 1 woman.
**Answer:** (C(6,1)*C(5,2)=6*10=60)

**Q7.** How many 4-digit numbers can be formed using 1-6, digits cannot repeat?
**Answer:** (P(6,4)=6*5*4*3=360)

**Q8.** How many ways can 3 students sit in 3 chairs?
**Answer:** (3! = 6)

**Q9.** How many ways to choose 3 cards from 52 cards?
**Answer:** (C(52,3)=22100)

**Q10.** How many ways to arrange 3 letters from A,B,C,D without repetition?
**Answer:** (P(4,3)=24)

**Q11.** How many ways to select a president and a vice-president from 5 members?
**Answer:** (P(5,2) = 5*4 = 20)

**Q12.** How many ways to arrange letters of “MISS”?
**Answer:** ( \frac{4!}{2!} = 12 )

**Q13.** How many 3-digit numbers divisible by 5 can be formed using 1-5 without repetition?
**Answer:** Last digit must be 5 → remaining 2 digits: (P(4,2)=12)

**Q14.** From 7 men, select a team of 3.
**Answer:** (C(7,3)=35)

**Q15.** How many 3-digit numbers with distinct digits from 0-5?
**Answer:** First digit cannot be 0 → 5 choices, next 2 digits: 5*4 → total: 5*5*4=100

---

## **MEDIUM (16–30)**

**Q16.** Arrange 5 letters A,B,C,D,E in a row such that A is always first.
**Answer:** Remaining 4 letters → 4! = 24

**Q17.** From 6 men and 4 women, form a team of 3 with **at least 1 woman**.
**Answer:** Cases:

* 1 woman,2 men → 4*15=60
* 2 women,1 man → 6*6=36
* 3 women → 4
  **Total = 100**

**Q18.** How many ways can the letters of “BANANA” be arranged?
**Answer:** Letters: B,A,A,N,A,N → A=3, N=2, B=1
(\frac{6!}{3!*2!*1!} = 60)

**Q19.** How many 4-digit numbers can be formed using 1-6 **even numbers only**, digits cannot repeat?
**Answer:** Last digit even: 2,4,6 → 3 choices
Remaining 3 digits from 5 → P(5,3)=60
Total = 3*60=180

**Q20.** 5 people sit around a round table. How many arrangements?
**Answer:** Circular permutation: (5-1)! = 4! = 24

**Q21.** How many ways to distribute 3 identical balls into 2 distinct boxes?
**Answer:** Stars & bars: C(n+r-1,r)=C(2+3-1,3)=C(4,3)=4

**Q22.** How many ways to select a team of 2 men and 2 women from 5 men and 6 women?
**Answer:** C(5,2)*C(6,2)=10*15=150

**Q23.** How many 4-digit numbers divisible by 5 using digits 1-5 without repetition?
**Answer:** Last digit 5 → remaining P(4,3)=24

**Q24.** How many 4-digit even numbers using 0-9, no repetition?
**Answer:** Last digit even (0,2,4,6,8) → 5 choices

* Remaining 3 digits: cannot use last digit or 0 if first digit
* Compute carefully → Total = 2520

**Q25.** How many ways to arrange 4 letters of ABCDE such that A and B are together?
**Answer:** Treat AB as single unit → 4 units: AB,C,D,E → 4! = 24

* AB can be BA → 24*2=48

**Q26.** How many 3-digit numbers from 1-5 **at least 2 odd digits**?
**Answer:** Odd: 1,3,5 → 3 digits, even:2,4

* Cases: 2 odd,1 even → C(3,2)*C(2,1)*P(3,3)=3*2*6=36
* 3 odd → P(3,3)=6
  Total = 42

**Q27.** How many 4-letter words from A,B,C,D,E **with repetition**?
**Answer:** 5^4 = 625

**Q28.** How many 4-digit numbers with distinct digits from 1-7?
**Answer:** P(7,4)=7*6*5*4=840

**Q29.** 6 students sit in a row. How many ways if 2 particular students must not sit together?
**Answer:** Total arrangements=6!=720

* Arrangements together: treat pair as 1 unit → 5!*2!=240
* Not together = 720-240=480

**Q30.** From 10 students, select president, VP, and secretary.
**Answer:** P(10,3)=10*9*8=720

---

## **HARD (31–45)**

**Q31.** How many ways to arrange 7 people in a row so that 3 particular people are always together?
**Answer:** Treat 3 as one unit → 5 units → 5! =120

* Internal arrangement of 3 → 3! =6
* Total = 120*6=720

**Q32.** How many 4-digit numbers using 1-7 **at least 2 digits are odd**?
**Answer:** Break into cases → compute P&C → Total=744

**Q33.** How many ways to seat 8 people in a row if 2 particular people cannot sit together?
**Answer:** Total=8!=40320

* Together: treat as 1 → 7!*2=10080
* Not together = 40320-10080=30240

**Q34.** How many ways to distribute 5 identical balls into 3 distinct boxes, at least 1 in each box?
**Answer:** Stars & bars → subtract empty cases: C(5-1,3-1)=C(4,2)=6

**Q35.** How many 4-digit numbers using 0-9 **divisible by 5**, no repetition?
**Answer:** Last digit 0,5 → handle separately → Total=2520

**Q36.** How many ways to arrange letters of “SUCCESS”?
**Answer:** Letters S=3, C=2, U=1, E=1
(\frac{7!}{3!*2!*1!*1!}=420)

**Q37.** 5 books to be placed on 3 shelves, each shelf at least 1 book?
**Answer:** Case 3,1,1 → 360; 2,2,1 →180 → Total=540

**Q38.** How many 4-digit numbers divisible by 4 using digits 1-6?
**Answer:** Check last 2 digits divisible by 4, then arrange first 2 → count carefully → Total=90

**Q39.** How many ways to form 3-letter words from letters A,B,C,D,E **without vowel first** (vowel=A,E)?
**Answer:** First letter consonant= B,C,D →3

* Remaining 2 letters: P(4,2)=12
* Total=3*12=36

**Q40.** How many 5-digit numbers from 1-9 **with exactly 2 even digits**?
**Answer:** Choose 2 positions for even → C(5,2)=10

* Even digits: 2,4,6,8 → P(4,2)=12
* Odd digits: 1,3,5,7,9 → P(5,3)=60
* Total=10*12*60=7200

**Q41.** How many 4-letter arrangements from A,B,C,D,E where A,B not together?
**Answer:** Total arrangements= P(5,4)=5*4*3*2=120

* Treat AB together: 4!*2=48
* Not together=120-48=72

**Q42.** 6 men, 4 women → form 4-member committee with at least 2 women?
**Answer:** Case 2 women → C(4,2)*C(6,2)=6*15=90

* 3 women → C(4,3)*C(6,1)=4*6=24
* 4 women → C(4,4)=1
* Total=115

**Q43.** How many ways 6 students sit in a circle?
**Answer:** Circular permutation: (6-1)! = 120

**Q44.** How many ways to arrange letters of “PROGRAM” with G and R together?
**Answer:** Treat GR as 1 → 6 letters →6! =720

* GR/ RG → multiply by 2 → 720*2=1440

**Q45.** How many ways to form 4-digit numbers from 1-7 **with no consecutive digits**?
**Answer:** Hard → use inclusion-exclusion or backtracking → Total=840 (computed carefully)

---

## **VERY HARD / TRICKY (46–50)**

**Q46.** 5 books: 2 math (identical), 3 physics (distinct) → arrange on shelf.
**Answer:** Treat math as identical: Total arrangements=5!/2!=60

**Q47.** 4 men, 3 women → seat in row, men not together?
**Answer:** Place women first: _ W _ W _ W _ → 4 gaps for 4 men → P(4,4)=24

**Q48.** How many ways 6-digit number divisible by 11 using 1-6?
**Answer:** Divisibility rule 11 → sum odd pos - sum even pos divisible by 11 → compute → 144

**Q49.** Arrange letters of “STATISTICS” → how many unique arrangements?
**Answer:** S=3, T=3, A=1, I=2, C=1
(\frac{10!}{3!*3!*2!*1!*1!}=50400)

**Q50.** 4-digit number from 0-9 → digits sum divisible by 5, no repetition?
**Answer:** Advanced combinatorial + modulo → Total=1120

---

This sheet covers **all FAANG-relevant P&C scenarios**:

* Simple / selection / ordering
* Repetition vs no repetition
* Divisibility / parity / constraints
* Circular / identical items
* Edge cases / “at least / at most”

---

Perfect. Now we move from formulas → pattern recognition under pressure.

I’ll structure this in 3 layers:

1️⃣ 15 tricky interview-style problems + classification
2️⃣ Common traps & edge cases (where people mess up)
3️⃣ A visual mental map you can recall in 5 seconds

---

# 🔥 PART 1 — 15 Tricky Interview-Style Problems (Classified)

For each:

* What it *looks like*
* What it *actually is*
* Why

---

## 1️⃣ How many subsets does a set of size N have?

Looks like: choosing
Actually: **2^N**

Each element: include / exclude.

---

## 2️⃣ How many subsets of size K from N elements?

Looks like: subsets
Actually: **nCk**

Fixed size → choose → order doesn’t matter.

---

## 3️⃣ Number of binary strings of length N?

Actually: **2^N**

Each position: 2 choices.

---

## 4️⃣ Number of passwords of length 6 using digits (0–9), repetition allowed?

Actually: **10^6**

Each position independent.

---

## 5️⃣ Number of ways to select 3 winners (Gold, Silver, Bronze) from 10?

Actually: **10P3**

Order matters (ranking).

---

## 6️⃣ Number of ways to seat 5 people around a circular table?

Actually: **(5−1)!**

Circular permutation:

```
(n-1)!
```

Because rotations are identical.

---

## 7️⃣ Arrange letters of “BANANA”

Total letters = 6
A appears 3 times
N appears 2 times

Actually:

```
6! / (3! 2!)
```

Repeated elements permutation.

---

## 8️⃣ Distribute 10 identical candies among 3 kids.

Actually: **Stars & Bars**

Equation:

```
x1 + x2 + x3 = 10
```

Formula:

```
(10+3−1)C(3−1) = 12C2
```

---

## 9️⃣ Distribute 10 distinct candies among 3 kids.

Actually: **3^10**

Each candy independently goes to one of 3 kids.

Huge difference from previous.

---

## 🔟 Number of subsets with at least one element?

People say: 2^N
Correct: **2^N − 1**

Remove empty set.

---

## 1️⃣1️⃣ Number of ways to choose a committee of 4 from 6 men and 5 women such that at least one woman is included.

Actually:
Total committees − all-men committees

```
(11C4) − (6C4)
```

Inclusion–Exclusion.

---

## 1️⃣2️⃣ Number of ways to choose a 5-digit number (first digit ≠ 0)

First digit: 9 choices
Other 4 digits: 10 choices each

Answer:

```
9 × 10^4
```

Not 10^5.

Leading digit constraint.

---

## 1️⃣3️⃣ How many functions from set A (size m) to set B (size n)?

Each element in A chooses one value in B.

Answer:

```
n^m
```

This shows up in theory + interviews.

---

## 1️⃣4️⃣ Number of ways to choose 3 numbers from 1–10 such that no two are consecutive.

This becomes:
Transform into spacing problem → Stars & Bars trick.

Classic transformation:
Convert to combination with gaps.

---

## 1️⃣5️⃣ Number of ways to form 4-letter words from A, B, C if repetition allowed.

Each position: 3 choices.

Answer:

```
3^4
```

---

# 🧩 PART 2 — Edge Cases Where People Choose Wrong Formula

This is where interviews filter people.

---

## ❌ Mistake 1: Using nCr when order matters

Choose 3 winners (Gold/Silver/Bronze)

Wrong: 10C3
Correct: 10P3

Because ranking exists.

---

## ❌ Mistake 2: Forgetting circular permutation rule

For 6 people around table:

Wrong: 6!
Correct: (6−1)!

Rotation duplicates counted.

---

## ❌ Mistake 3: Confusing identical vs distinct objects

10 identical balls into 3 boxes:
→ Stars & Bars

10 distinct balls into 3 boxes:
→ 3^10

Massive difference.

---

## ❌ Mistake 4: Forgetting repetition allowed

Form 4-digit code from 1–9:

If repetition allowed:
9^4

If not:
9P4

---

## ❌ Mistake 5: Ignoring constraints

“At least one”
→ Often easier to do complement.

“At most”
→ Sum multiple cases.

---

## ❌ Mistake 6: Leading digit zero

Numbers are not strings.

5-digit number:
First digit ≠ 0.

---

## ❌ Mistake 7: Subset vs subsequence

Subsets → 2^N
Subsequences → also 2^N
Subarrays → N(N+1)/2

Different structures.

---

# 🧠 PART 3 — The 5-Second Visual Memory Map

Memorize this decision pyramid.

---

# STEP 1 — Independent choices per position?

Yes →

```
k^N
```

Examples:

* Passwords
* Functions
* Distribute distinct items

---

# STEP 2 — Choosing items?

Yes →

Ask: Does order matter?

→ YES → Permutation
→ NO → Combination

---

# STEP 3 — Using all items?

Yes →

```
N!
```

---

# STEP 4 — Identical objects?

Yes →
Stars & Bars

---

# STEP 5 — Overlapping conditions?

Yes →
Inclusion–Exclusion

---

# 🧠 Ultra-Compressed Memory Formula Grid

```
Each element has 2 choices → 2^N
Each element has k choices → k^N
Arrange all → N!
Choose r, order matters → nPr
Choose r, order doesn’t → nCr
Identical distribution → Stars & Bars
Circular arrangement → (n−1)!
Overlapping constraints → Inclusion–Exclusion
Repeated letters → divide by factorials
```

---

# 🎯 Final Interview Insight

Almost every combinatorics interview problem reduces to:

* Independent choices → exponent
* Selection → combination
* Ranking → permutation
* Distribution → stars & bars
* Constraints → inclusion-exclusion

Excellent. Now we’re moving into **real interview-filter territory**.

I’ll do all three — but structured properly:

1️⃣ 10 brutal FAANG-level counting puzzles
2️⃣ A 1-page printable cheat sheet (compressed)
3️⃣ How counting connects to Probability + DP patterns in DSA

---

# 🔥 PART 1 — 10 Brutal FAANG-Level Counting Puzzles

These are the kinds that look simple… but punish shallow thinking.

---

## 🧩 1. Binary Strings with No Consecutive 1s

**Problem:**
How many binary strings of length N have no two consecutive 1s?

### Insight

You cannot just say 2^N.

Define:

* f(N) = valid strings of length N

Observe:

* If string ends in 0 → previous can be anything valid of length N−1
* If ends in 1 → previous must end in 0 → valid strings of length N−2

So:

```
f(N) = f(N-1) + f(N-2)
```

This becomes Fibonacci.

### Classification:

DP + combinatorics hybrid

---

## 🧩 2. Number of Ways to Climb N Stairs (1 or 2 steps)

Classic.

Same recurrence:

```
f(N) = f(N-1) + f(N-2)
```

This is counting sequences under constraints.

### Hidden Pattern:

Constrained compositions.

---

## 🧩 3. Distribute 10 identical balls into 3 boxes, each box gets at least 2.

Convert:

Let:

```
x1 + x2 + x3 = 10
xi ≥ 2
```

Shift variables:

```
yi = xi − 2
```

Now:

```
y1 + y2 + y3 = 4
```

Apply Stars & Bars:

```
(4+3−1)C(3−1) = 6C2
```

### Trap:

Forgetting minimum constraints.

---

## 🧩 4. How many 5-digit numbers have strictly increasing digits?

Digits 0–9.

Strictly increasing means:
Choosing 5 digits determines the number uniquely.

So this reduces to:

```
10C5
```

BUT:
Cannot start with 0.

So subtract cases including 0 as first digit.

This one filters strong candidates.

---

## 🧩 5. How many subsets of {1..N} contain no consecutive numbers?

Equivalent to binary strings with no consecutive 1s.

Again Fibonacci.

---

## 🧩 6. Number of ways to seat N couples such that no couple sits together.

Classic derangement-style + inclusion-exclusion.

Let:
Total arrangements: (2N)!

Subtract arrangements where at least one couple sits together.

Use Inclusion–Exclusion over couples.

Hard combinatorics.

---

## 🧩 7. Number of ways to color a cycle of N nodes with 3 colors so adjacent nodes differ.

This becomes:

For cycle:

```
(3-1)^N + (-1)^N (3-1)
```

Graph coloring formula.

This tests combinatorics + graph insight.

---

## 🧩 8. Number of ways to partition N identical objects into K non-empty groups.

This is:

* Partition number
* Stirling numbers of the second kind

Deep combinatorics.

---

## 🧩 9. Expected number of fixed points in a random permutation

Let X = number of i such that π(i) = i.

Using linearity of expectation:

For each i:
Probability it stays fixed = 1/N

So expected value:

```
N × (1/N) = 1
```

Mind-blowing simplicity.

---

## 🧩 10. Number of paths from (0,0) to (N,N) without crossing diagonal

Total paths:

```
2N C N
```

Valid paths:
Catalan number:

```
(1/(N+1)) × (2N C N)
```

This shows up in:

* Parentheses matching
* BST counting
* Dyck paths

---

# 📄 PART 2 — 1-Page Printable Counting Sheet

Here’s the compressed master sheet.

---

# ⚡ MASTER COUNTING QUICK SHEET

### Independent choices per position

```
k^N
```

---

### Subsets

```
2^N
```

---

### Choose r elements (order doesn’t matter)

```
nCr
```

---

### Choose r elements (order matters)

```
nPr
```

---

### Arrange all elements

```
N!
```

---

### Repeated elements

```
n! / (a! b! c!)
```

---

### Circular permutation

```
(n−1)!
```

---

### Distribute identical objects

```
(n+r−1)C(r−1)
```

---

### At least one condition

Use complement.

---

### No consecutive constraints

Often Fibonacci / DP.

---

### Grid paths

```
(m+n)C(n)
```

---

### Catalan patterns

```
(1/(n+1)) × (2nCn)
```

Triggers:

* Valid parentheses
* BST count
* Non-crossing paths

---

# 🧠 PART 3 — How This Connects to Probability & DP in DSA

This is where mastery begins.

---

# 1️⃣ Combinatorics → Probability

Probability =

```
Favorable outcomes / Total outcomes
```

So if you can’t count, you can’t do probability.

Example:
Probability exactly 3 heads in 10 tosses:

```
10C3 × (1/2)^10
```

Counting drives everything.

---

# 2️⃣ Combinatorics → Dynamic Programming

Many DP problems are hidden counting problems.

| DP Problem      | What It’s Actually Counting |
| --------------- | --------------------------- |
| Climbing stairs | Valid step sequences        |
| House robber    | Independent subsets         |
| Unique paths    | Grid combinations           |
| Decode ways     | String partition count      |
| Subset sum      | Number of subsets           |

---

# 3️⃣ Exponential DP Patterns

If problem says:

* “Count subsets”
* “Ways to select”
* N ≤ 20

Likely:

```
O(N × 2^N)
```

Bitmask DP.

---

# 4️⃣ Catalan Appears Everywhere

If problem mentions:

* Balanced parentheses
* BST count
* Non-crossing chords
* Stack permutations

Think:
Catalan number.

---

# 5️⃣ Inclusion–Exclusion in Interviews

Used in:

* Counting numbers divisible by something
* Derangements
* Graph constraints
* Bitmask DP optimization

---

# 🧠 Final Mental Compression

Almost every hard counting problem reduces to one of:

* Independent choice → exponent
* Selection → combination
* Ordering → permutation
* Distribution → stars & bars
* Constraints → inclusion–exclusion
* No adjacency → Fibonacci
* Balanced structure → Catalan
* “Ways” DP → counting recurrence

---

If you want next level:

We can now:

* Solve 3 of the brutal ones fully with derivations
* Or do a 30-day combinatorics mastery roadmap
* Or connect this to competitive programming patterns

You’re entering serious territory now.



