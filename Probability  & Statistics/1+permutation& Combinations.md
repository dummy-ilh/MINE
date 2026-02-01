
## **1️⃣ Rule of Product (Multiplication Principle)**

**Concept:**
If a task can be done in multiple steps, and each step has a certain number of options **independent** of other steps, the total number of outcomes is the **product** of the options.

**Formula:**
If step 1 has (m) options and step 2 has (n) options, total ways = (m \times n).

**Example 1:**
You want to make a password with 2 characters:

* 1st character: 3 options (A, B, C)
* 2nd character: 4 options (1, 2, 3, 4)

**Total passwords = 3 × 4 = 12**

**Example 2:**
You have 5 shirts and 3 pants. How many different outfits can you wear?
**Answer:** 5 × 3 = 15 outfits.

✅ **Key clue to use this:** “Each choice is independent of the others” and you multiply the possibilities.

---

## **2️⃣ Permutation (Order Matters)**

**Concept:**
Permutation is used when you are **arranging or ordering** items, and **order matters**.

**Formula:**

* Choosing (r) items from (n) items and arranging them:
  $[
  P(n, r) = \frac{n!}{(n-r)!}
  ]$

* If arranging **all items**, just (n! = 1 × 2 × 3 × … × n)

**Example 1 (partial permutation):**
You have 5 books and want to put **3 on a shelf**. How many ways?

$[
P(5,3) = \frac{5!}{(5-3)!} = \frac{120}{2} = 60
]$

**Example 2 (full permutation):**
You have 4 students, and you want to line them up for a photo.
Total ways = 4! = 24 ways

✅ **Key clue:** Question asks for **order/sequence**.

---

## **3️⃣ Combination (Order Doesn’t Matter)**

**Concept:**
Combination is used when you are **choosing items** but **order doesn’t matter**.

**Formula:**
$[
C(n, r) = \frac{n!}{r!(n-r)!}
]$

**Example 1:**
From 5 students, choose 3 for a committee. Order doesn’t matter.
$[
C(5,3) = \frac{5!}{3!2!} = \frac{120}{6×2} = 10
]$

**Example 2:**
A lottery ticket has 6 numbers. You pick 6 out of 49. How many combinations?
$[
C(49,6) = \frac{49!}{6!43!} \approx 14 \text{ million ways}
]$

✅ **Key clue:** “Selection without caring about order” → Combination.

---

## **Quick Tip to Decide Which to Use**

| Scenario                                      | Use                 |
| --------------------------------------------- | ------------------- |
| Multiple independent steps (multiply options) | **Rule of Product** |
| Arranging r objects **in order**              | **Permutation**     |
| Choosing r objects **without order**          | **Combination**     |

---

## **Mixed Examples**

1. **Password problem:**
   A password has 3 letters (A-Z) and 2 digits (0-9). How many passwords?

* 3 letters: 26 options each → (26^3)
* 2 digits: 10 options each → (10^2)
* Total = (26^3 × 10^2 = 1757600) ✅ **Rule of Product**

2. **Committee from class:**
   Pick 2 presidents and 3 members from 10 students. Order matters for president but not members.

* President: P(10,2) = 10×9=90
* Members: C(8,3) = 56 (remaining 8 students)
* Total ways = 90 × 56 = 5040 ✅ **Mix of Permutation & Combination**

3. **Seating 5 people in 3 chairs:**

* Order matters → P(5,3) = 60 ✅ **Permutation**

4. **Choosing fruits:**
   Pick 2 fruits from {apple, banana, orange, mango}. Order doesn’t matter → C(4,2)=6 ✅ **Combination**

---

If you want, I can make a **visual diagram/chart** showing **Rule of Product vs Permutation vs Combination** with colors and examples. It’s super easy to remember after that.

Do you want me to make that?
