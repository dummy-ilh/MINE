Perfect! Let’s start with the **first 5 “Easy” Hash Set/Map problems**. For each, I’ll provide:

1. **Problem explanation with example**
2. **Code solution** (Python for clarity)
3. **Time & space complexity analysis**
4. **Alternative approaches or variations**

We’ll do them **one by one** so it’s clear and thorough.

---

## **1️⃣ Check for Subset**

**Problem:**
Check if array A is a subset of array B (i.e., all elements of A appear in B).

**Example:**

```
A = [1, 2, 3]
B = [1, 2, 3, 4, 5]
Output: True

A = [1, 2, 6]
B = [1, 2, 3, 4, 5]
Output: False
```

**Code (Python):**

```python
def isSubset(A, B):
    setB = set(B)
    for num in A:
        if num not in setB:
            return False
    return True

# Test
print(isSubset([1,2,3], [1,2,3,4,5]))  # True
print(isSubset([1,2,6], [1,2,3,4,5]))  # False
```

**Complexity:**

* Time: O(|B| + |A|) → build set + check each element
* Space: O(|B|) → store all elements of B in a set

**Alternative Approaches:**

* **Sorting + Two pointers:** Sort both arrays → O(|A|log|A| + |B|log|B|)
* **Hash map counting:** If duplicates matter, count frequencies.

---

## **2️⃣ Check for Disjoint**

**Problem:**
Check if two arrays have no elements in common (disjoint sets).

**Example:**

```
A = [1, 2, 3]
B = [4, 5, 6]
Output: True

A = [1, 2, 3]
B = [3, 4, 5]
Output: False
```

**Code (Python):**

```python
def areDisjoint(A, B):
    setA = set(A)
    for num in B:
        if num in setA:
            return False
    return True

# Test
print(areDisjoint([1,2,3], [4,5,6]))  # True
print(areDisjoint([1,2,3], [3,4,5]))  # False
```

**Complexity:**

* Time: O(|A| + |B|)
* Space: O(|A|)

**Alternative Approaches:**

* Sort both arrays + use two pointers → O(|A|log|A| + |B|log|B|)
* Use bit manipulation if numbers are small integers.

---

## **3️⃣ Check for Equal**

**Problem:**
Check if two arrays contain **exactly the same elements**, ignoring order (set equality).

**Example:**

```
A = [1, 2, 3]
B = [3, 2, 1]
Output: True

A = [1, 2, 3]
B = [1, 2, 4]
Output: False
```

**Code (Python):**

```python
def areEqual(A, B):
    return set(A) == set(B)

# Test
print(areEqual([1,2,3], [3,2,1]))  # True
print(areEqual([1,2,3], [1,2,4]))  # False
```

**Complexity:**

* Time: O(|A| + |B|)
* Space: O(|A| + |B|)

**Alternative Approaches:**

* Sort both arrays → compare element by element → O(|A|log|A| + |B|log|B|)
* Hash map for frequency if duplicates must match exactly.

---

## **4️⃣ Fizz Buzz**

**Problem:**
Print numbers from 1 to N with rules:

* Multiple of 3 → "Fizz"
* Multiple of 5 → "Buzz"
* Multiple of 3 & 5 → "FizzBuzz"

**Example:**

```
N = 5
Output: 1 2 Fizz 4 Buzz
```

**Code (Python):**

```python
def fizzBuzz(N):
    for i in range(1, N+1):
        if i % 15 == 0:
            print("FizzBuzz", end=" ")
        elif i % 3 == 0:
            print("Fizz", end=" ")
        elif i % 5 == 0:
            print("Buzz", end=" ")
        else:
            print(i, end=" ")
    print()

fizzBuzz(15)
```

**Complexity:**

* Time: O(N) → iterate numbers 1 to N
* Space: O(1)

**Alternative Approaches:**

* Use **precomputed arrays** for multiples to avoid repeated modulo (optimization for very large N).
* Store results in list and return instead of printing.

---

## **5️⃣ Max Distance Between Two Occurrences**

**Problem:**
Given an array, find the **maximum distance between two same elements**.

**Example:**

```
arr = [1, 2, 3, 1, 4, 2, 1]
Output: 6  # distance between first 1 and last 1
```

**Code (Python):**

```python
def maxDistance(arr):
    first_occurrence = {}
    max_dist = 0
    for i, num in enumerate(arr):
        if num not in first_occurrence:
            first_occurrence[num] = i
        else:
            max_dist = max(max_dist, i - first_occurrence[num])
    return max_dist

# Test
print(maxDistance([1,2,3,1,4,2,1]))  # 6
```

**Complexity:**

* Time: O(N) → single pass
* Space: O(N) → store first occurrence of each element

**Alternative Approaches:**

* Brute force: nested loops → O(N²)
* Use two-pointer technique if sorted (rare case)

---
Perfect! Let’s continue with **problem 6 onward** in the same structured way: explanation, code, complexity, and alternatives.

---

## **6️⃣ Duplicate Within K Distance**

**Problem:**
Given an array of integers and a number `k`, check if the array contains **duplicate elements** such that the **distance between duplicates is at most k**.

**Example:**

```
arr = [1, 2, 3, 1], k = 3
Output: True  # 1 occurs at indices 0 and 3, distance = 3 ≤ k

arr = [1, 0, 1, 1], k = 1
Output: True  # 1 occurs at indices 2 and 3, distance = 1 ≤ k
```

**Code (Python):**

```python
def containsDuplicateWithinK(arr, k):
    seen = {}
    for i, num in enumerate(arr):
        if num in seen and i - seen[num] <= k:
            return True
        seen[num] = i
    return False

# Test
print(containsDuplicateWithinK([1,2,3,1], 3))  # True
print(containsDuplicateWithinK([1,0,1,1], 1))  # True
```

**Complexity:**

* Time: O(N) → single pass
* Space: O(N) → store last seen index of each element

**Alternative Approaches:**

* Use a **sliding window set of size k**: keep k most recent elements, check if duplicate exists → O(N) time, O(k) space.

---

## **7️⃣ Intersection of Two Arrays**

**Problem:**
Find all elements that are present in both arrays (unique intersection).

**Example:**

```
A = [1, 2, 2, 3]
B = [2, 2, 3, 4]
Output: [2, 3]
```

**Code (Python):**

```python
def intersection(A, B):
    setA = set(A)
    setB = set(B)
    return list(setA & setB)  # set intersection

# Test
print(intersection([1,2,2,3], [2,2,3,4]))  # [2, 3]
```

**Complexity:**

* Time: O(|A| + |B|) → building sets
* Space: O(|A| + |B|) → store sets

**Alternative Approaches:**

* **Sorting + two pointers:** Sort both arrays, iterate → O(|A|log|A| + |B|log|B|)
* **Hash map for frequency:** If duplicates need to appear in output as many times as they appear.

---

## **8️⃣ Union of Two Arrays**

**Problem:**
Find all distinct elements present in either of the two arrays.

**Example:**

```
A = [1, 2, 3]
B = [2, 3, 4]
Output: [1, 2, 3, 4]
```

**Code (Python):**

```python
def union(A, B):
    return list(set(A) | set(B))  # set union

# Test
print(union([1,2,3], [2,3,4]))  # [1,2,3,4]
```

**Complexity:**

* Time: O(|A| + |B|)
* Space: O(|A| + |B|)

**Alternative Approaches:**

* **Sorting + merge:** Sort both arrays, merge unique elements → O(|A|log|A| + |B|log|B|)
* Use **hash map** to store frequency → output all keys.

---

## **9️⃣ Most Frequent Element**

**Problem:**
Find the element that appears most frequently in an array. If multiple, return any one.

**Example:**

```
arr = [1, 3, 2, 1, 4, 1, 3]
Output: 1
```

**Code (Python):**

```python
from collections import Counter

def mostFrequent(arr):
    count = Counter(arr)
    return max(count, key=count.get)

# Test
print(mostFrequent([1,3,2,1,4,1,3]))  # 1
```

**Complexity:**

* Time: O(N) → build frequency map + find max
* Space: O(N) → store frequencies

**Alternative Approaches:**

* Use **sorting** → sort array and count consecutive duplicates → O(NlogN), O(1) extra space
* Use **Boyer-Moore majority vote** if guaranteed that one element appears > N/2 times.

---

## **🔟 2-Sum: Find if Any Pair Exists**

**Problem:**
Check if there exists a pair of numbers in an array whose sum equals a target `k`.

**Example:**

```
arr = [1, 4, 6, 8], k = 10
Output: True  # 4+6=10

arr = [1, 2, 3], k = 7
Output: False
```

**Code (Python):**

```python
def hasPairWithSum(arr, k):
    seen = set()
    for num in arr:
        if k - num in seen:
            return True
        seen.add(num)
    return False

# Test
print(hasPairWithSum([1,4,6,8], 10))  # True
print(hasPairWithSum([1,2,3], 7))     # False
```

**Complexity:**

* Time: O(N) → single pass
* Space: O(N) → store seen elements

**Alternative Approaches:**

* **Sorting + two pointers:** O(NlogN) → iterate with left/right pointers
* Brute force nested loops → O(N²)

---
Got it 👍
Let’s **redo problems 11–15 (the next 5 Easy problems)** cleanly with full structure (problem → sample → code → complexity → alternatives).

---

# ✅ Easy Hash Problems (11–15)

---

## **11️⃣ 2-Sum – Count Pairs**

**Problem:**
Count the number of pairs `(i, j)` such that `arr[i] + arr[j] = K`.
Order doesn’t matter, i.e., `(a, b)` and `(b, a)` are the same pair.

**Example:**

```
arr = [1, 5, 7, -1, 5], K = 6
Output: 3   # (1,5), (7,-1), (1,5 again with the second 5)
```

**Code (Python):**

```python
from collections import Counter

def countPairsWithSum(arr, K):
    freq = Counter(arr)
    count = 0
    for num in arr:
        complement = K - num
        count += freq[complement]
        if complement == num:
            count -= 1  # don’t count same element pairing with itself
    return count // 2  # each pair counted twice

# Test
print(countPairsWithSum([1, 5, 7, -1, 5], 6))  # 3
```

**Complexity:**

* Time: O(N)
* Space: O(N)

**Alternatives:**

* Brute force: nested loops O(N²).
* Sorting + two pointers: O(N log N).

---

## **12️⃣ Count Pairs with Given Difference**

**Problem:**
Count pairs `(i, j)` such that `|arr[i] - arr[j]| = K`.

**Example:**

```
arr = [1, 5, 2, 2, 5, 4], K = 3
Output: 4
# Pairs: (5,2), (5,2 again), (4,1), (2,5)
```

**Code (Python):**

```python
from collections import Counter

def countPairsWithDiff(arr, K):
    freq = Counter(arr)
    count = 0
    for num in freq:
        if num + K in freq:
            count += freq[num] * freq[num + K]
    return count

# Test
print(countPairsWithDiff([1, 5, 2, 2, 5, 4], 3))  # 4
```

**Complexity:**

* Time: O(N)
* Space: O(N)

**Alternatives:**

* Brute force: O(N²).
* Sorting + two pointers: O(N log N).

---

## **13️⃣ Only Repetitive Element (1 to n-1)**

**Problem:**
An array contains numbers from `1` to `n-1` with **exactly one duplicate**. Find the duplicate.

**Example:**

```
arr = [1, 3, 4, 2, 2]
Output: 2
```

**Code (Python):**

```python
def findDuplicate(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return num
        seen.add(num)
    return -1

# Test
print(findDuplicate([1,3,4,2,2]))  # 2
```

**Complexity:**

* Time: O(N)
* Space: O(N)

**Alternatives:**

* Floyd’s cycle detection (tortoise & hare) → O(N), O(1).
* Math formula: difference between sum(arr) and expected sum.

---

## **14️⃣ Missing of a Range**

**Problem:**
Given an array and a range `[L, R]`, find all missing numbers within the range.

**Example:**

```
arr = [1, 3, 5], L = 1, R = 5
Output: [2, 4]
```

**Code (Python):**

```python
def missingInRange(arr, L, R):
    nums = set(arr)
    return [x for x in range(L, R+1) if x not in nums]

# Test
print(missingInRange([1,3,5], 1, 5))  # [2, 4]
```

**Complexity:**

* Time: O(N + (R-L))
* Space: O(N)

**Alternatives:**

* Sort array → scan for gaps → O(N log N).
* Boolean array of size R-L+1.

---

## **15️⃣ Missing from Min to Max of Array**

**Problem:**
Find all numbers missing in the range `[min(arr), max(arr)]`.

**Example:**

```
arr = [2, 4, 7, 5]
Output: [3, 6]
```

**Code (Python):**

```python
def missingFromMinToMax(arr):
    nums = set(arr)
    mn, mx = min(arr), max(arr)
    return [x for x in range(mn, mx+1) if x not in nums]

# Test
print(missingFromMinToMax([2,4,7,5]))  # [3, 6]
```

**Complexity:**

* Time: O(N + (max-min))
* Space: O(N)

**Alternatives:**

* Sort array → find gaps between consecutive elements.
* Boolean array of size (max-min+1).

---

✅ Now we’ve **fully covered Easy Problems 1–15** from your list.

Would you like me to **finish the last 3 Easy problems (16–18)** right now:

* Minimum Subsets with Distinct
* Minimum Removals for No Common
* Maximum Points on the Same Line

…so that your Easy section is 100% done before we move to **Medium problems**?


