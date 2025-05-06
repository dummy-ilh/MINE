
# 🧠 Mastering Subarray Problems – LeetCode Guide

> **Goal**: Master all subarray-related patterns and problems in one day using templates, techniques, and problem classifications.

---

## ✅ 1. Subarray Problem Types and Patterns

| Pattern                   | Key Use Case                          | Template Style     |
|--------------------------|----------------------------------------|--------------------|
| **Brute Force**           | Small input size, check all subarrays | 3 nested loops     |
| **Sliding Window**        | Fixed-length or variable-length       | O(n)               |
| **Prefix Sum**            | Sum-related questions                 | Sum range queries  |
| **HashMap + Prefix Sum**  | Count subarrays with sum = K         | O(n)               |
| **Kadane’s Algorithm**    | Max subarray sum                     | O(n)               |
| **Two Pointers**          | Sorted arrays or non-negative input  | O(n) or O(n log n) |

---

## 🧱 2. Boilerplate Templates

### A. Kadane’s Algorithm (Max Subarray Sum)

```python
def max_subarray_sum(nums):
    max_sum = cur_sum = nums[0]
    for num in nums[1:]:
        cur_sum = max(num, cur_sum + num)
        max_sum = max(max_sum, cur_sum)
    return max_sum
```

---

### B. Count Subarrays with Sum = K (HashMap + Prefix Sum)

```python
from collections import defaultdict

def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_map = defaultdict(int)
    prefix_map[0] = 1
    
    for num in nums:
        prefix_sum += num
        count += prefix_map[prefix_sum - k]
        prefix_map[prefix_sum] += 1
    return count
```

---

### C. Sliding Window (Variable Length)

```python
def min_subarray_len(target, nums):
    left = total = 0
    res = float('inf')

    for right in range(len(nums)):
        total += nums[right]
        while total >= target:
            res = min(res, right - left + 1)
            total -= nums[left]
            left += 1
    return res if res != float('inf') else 0
```

---

## 📘 3. LeetCode Must-Do Subarray Problems by Category

### 🔹 Kadane's + Variations
- [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
- [918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/)
- [487. Max Consecutive Ones II](https://leetcode.com/problems/max-consecutive-ones-ii/)

### 🔹 Prefix Sum + HashMap
- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)
- [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)

### 🔹 Sliding Window
- [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
- [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)
- [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

### 🔹 Misc/Advanced
- [1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/)
- [325. Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)
- [862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/) ← advanced

---

## 📓 4. Learning Strategy

1. **Start with easy**: Do problems like 53, 560, 209  
2. **Level up**: Do one problem from each category  
3. **Master templates**: Practice modifying them for variations  
4. **Track patterns**: Tag problems as Kadane/prefix/hashmap/sliding window  

