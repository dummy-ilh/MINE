Absolutely! Below is a **Markdown-formatted version** of **LeetCode Array Problems #1–10**, complete with:

* ✅ Problem Titles and Numbers
* 🧠 Clear Explanations
* 💬 Example Inputs & Outputs
* 🛠️ All Solutions (Optimal + Alternate)
* 📘 Underlying Theory
* 🗣️ Interview-style Problem Discussions

---

# 🧩 LeetCode Array Mastery – Batch 1 (Problems #1–10)

---

## **1. [Two Sum](https://leetcode.com/problems/two-sum/) – #1**

### 🧠 Problem Explanation:

Given an array `nums` and a target integer `target`, return **indices** of the two numbers such that they add up to `target`.
**Each input has exactly one solution.**

### 💬 Example:

```text
Input: nums = [2, 7, 11, 15], target = 9  
Output: [0, 1]  // Because nums[0] + nums[1] = 2 + 7 = 9
```

---

### ✅ Optimal Solution – HashMap (O(n)):

```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

### 🔁 Alternate 1 – Brute Force (O(n²)):

```python
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
```

### 🔁 Alternate 2 – Two Pointers (Only if sorted):

```python
def twoSumSorted(nums, target):
    nums = sorted(enumerate(nums), key=lambda x: x[1])
    i, j = 0, len(nums) - 1
    while i < j:
        total = nums[i][1] + nums[j][1]
        if total == target:
            return [nums[i][0], nums[j][0]]
        elif total < target:
            i += 1
        else:
            j -= 1
```

### 📘 Theory:

* Hashing and complement
* Space-time tradeoff
* Index preservation matters

---

## **2. [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) – #88**

### 🧠 Problem Explanation:

You are given two sorted arrays `nums1` and `nums2`, where `nums1` has enough space at the end to hold `nums2`. Merge them in-place.

### 💬 Example:

```text
Input: nums1 = [1,2,3,0,0,0], m = 3; nums2 = [2,5,6], n = 3  
Output: [1,2,2,3,5,6]
```

---

### ✅ Optimal Solution – Backward Merge (O(n + m)):

```python
def merge(nums1, m, nums2, n):
    i, j, k = m-1, n-1, m+n-1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]; i -= 1
        else:
            nums1[k] = nums2[j]; j -= 1
        k -= 1
```

### 🔁 Alternate 1 – Sort after combine (violates in-place):

```python
def merge(nums1, m, nums2, n):
    nums1[:] = sorted(nums1[:m] + nums2)
```

### 🔁 Alternate 2 – Manual merge using new list:

```python
def merge(nums1, m, nums2, n):
    merged = []
    i = j = 0
    while i < m and j < n:
        if nums1[i] < nums2[j]:
            merged.append(nums1[i]); i += 1
        else:
            merged.append(nums2[j]); j += 1
    merged += nums1[i:m] + nums2[j:n]
    nums1[:m+n] = merged
```

### 📘 Theory:

* Two-pointer merge
* Backward traversal avoids overwriting

---

## **3. [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) – #121**

### 🧠 Problem Explanation:

Buy once and sell once to maximize profit, ensuring that you **sell after you buy**.

### 💬 Example:

```text
Input: prices = [7,1,5,3,6,4]  
Output: 5  // Buy at 1, sell at 6
```

---

### ✅ Optimal – Track min and profit (O(n)):

```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
```

### 🔁 Alternate 1 – Brute Force:

```python
def maxProfit(prices):
    max_profit = 0
    for i in range(len(prices)):
        for j in range(i+1, len(prices)):
            max_profit = max(max_profit, prices[j] - prices[i])
    return max_profit
```

### 🔁 Alternate 2 – Kadane-style on diffs:

```python
def maxProfit(prices):
    max_current = max_global = 0
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        max_current = max(0, max_current + diff)
        max_global = max(max_global, max_current)
    return max_global
```

### 📘 Theory:

* Greedy
* Minimum tracking
* Kadane’s on price change

---

## **4. [Single Number](https://leetcode.com/problems/single-number/) – #136**

### 🧠 Problem Explanation:

Every element appears twice except one. Find it.

### 💬 Example:

```text
Input: [4,1,2,1,2]  
Output: 4
```

---

### ✅ Optimal – XOR (O(n), O(1)):

```python
def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

### 🔁 Alternate – HashMap Count:

```python
from collections import Counter
def singleNumber(nums):
    for k, v in Counter(nums).items():
        if v == 1:
            return k
```

### 🔁 Alternate – Math Trick:

```python
def singleNumber(nums):
    return 2 * sum(set(nums)) - sum(nums)
```

### 📘 Theory:

* XOR: a^a=0, a^0=a
* Frequency hash
* Set property

---

## **5. [Majority Element](https://leetcode.com/problems/majority-element/) – #169**

### 🧠 Problem Explanation:

Return element that appears more than `n//2` times.

### 💬 Example:

```text
Input: [3,2,3]  
Output: 3
```

---

### ✅ Optimal – Boyer-Moore Voting:

```python
def majorityElement(nums):
    count = 0
    candidate = None
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate
```

### 🔁 Alternate – HashMap:

```python
from collections import Counter
def majorityElement(nums):
    return Counter(nums).most_common(1)[0][0]
```

### 🔁 Alternate – Sort and pick middle:

```python
def majorityElement(nums):
    nums.sort()
    return nums[len(nums)//2]
```

### 📘 Theory:

* Voting theory
* Frequency dominance

---

## **6. [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) – #217**

### 🧠 Problem Explanation:

Return true if any value appears at least twice.

### 💬 Example:

```text
Input: [1,2,3,1]  
Output: True
```

---

### ✅ Optimal – HashSet:

```python
def containsDuplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False
```

### 🔁 Alternate – Sort and scan:

```python
def containsDuplicate(nums):
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:
            return True
    return False
```

### 🔁 Alternate – Set length check:

```python
def containsDuplicate(nums):
    return len(nums) != len(set(nums))
```

### 📘 Theory:

* Set uniqueness
* Sorting guarantees neighbors

---

## **7. [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) – #53**

### 🧠 Problem Explanation:

Find the contiguous subarray with the largest sum.

### 💬 Example:

```text
Input: [-2,1,-3,4,-1,2,1,-5,4]  
Output: 6  // [4,-1,2,1]
```

---

### ✅ Optimal – Kadane’s Algorithm:

```python
def maxSubArray(nums):
    max_current = max_global = nums[0]
    for num in nums[1:]:
        max_current = max(num, max_current + num)
        max_global = max(max_global, max_current)
    return max_global
```

### 🔁 Alternate – DP array:

```python
def maxSubArray(nums):
    dp = [nums[0]]
    for i in range(1, len(nums)):
        dp.append(max(nums[i], dp[-1] + nums[i]))
    return max(dp)
```

### 🔁 Alternate – Divide and Conquer (O(n log n))

---

## **8. [Move Zeroes](https://leetcode.com/problems/move-zeroes/) – #283**

### 🧠 Problem Explanation:

Move all zeroes to the end while keeping the order of non-zero elements.

### 💬 Example:

```text
Input: [0,1,0,3,12]  
Output: [1,3,12,0,0]
```

---

### ✅ Optimal – Two Pointers:

```python
def moveZeroes(nums):
    insert = 0
    for num in nums:
        if num != 0:
            nums[insert] = num
            insert += 1
    for i in range(insert, len(nums)):
        nums[i] = 0
```

### 🔁 Alternate – Swap:

```python
def moveZeroes(nums):
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[i], nums[j] = nums[j], nums[i]
            j += 1
```

---

## **9. [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) – #26**

### 🧠 Problem Explanation:

Remove duplicates **in-place** and return the new length.

### 💬 Example:

```text
Input: [1,1,2]  
Output: 2 // Array becomes [1,2,_]
```

---

### ✅ Optimal – Two Pointers:

```python
def removeDuplicates(nums):
    if not nums:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if nums[i] != nums[j]:
            i += 1
            nums[i] = nums[j]
    return i + 1
```

---

## **10. [Remove Element](https://leetcode.com/problems/remove-element/) – #27**

### 🧠 Problem Explanation:

Remove all instances of `val` **in-place**, return new length.

### 💬 Example:

```text
Input: nums = [3,2,2,3], val = 3  
Output: 2 // nums becomes [2,2,_]
```

---

### ✅ Overwrite when not equal:

```python
def removeElement(nums, val):
    i = 0
    for num in nums:
        if num != val:
            nums[i] = num
            i += 1
    return i
```

### 🔁 Alternate – Swap with end:

```python
def removeElement(nums, val):
    i, n = 0, len(nums)
    while i < n:
        if nums[i] == val:
            nums[i] = nums[n-1]
            n -= 1
        else:
            i += 1
    return n
```

---

Would you like this exported as:

* 📥 A downloadable `.md` file?
* 🧠 Also include tags, difficulty, company frequency?
* 🚀 Ready for Batch 2 (11–20)?
