I'll give you a complete sliding window masterclass in plain text. Let me build this properly.

---

# SLIDING WINDOW — COMPLETE GUIDE

---

## WHAT IS SLIDING WINDOW?

A technique where you maintain a "window" (subarray/substring) that expands or shrinks as you move through the array. Instead of recomputing from scratch every time (O(n²) or O(n³)), you add/remove one element at a time — O(n).

---

## CHEAT SHEET: WHICH PATTERN TO USE?

```
Question asks for...                        → Pattern
─────────────────────────────────────────────────────────
Fixed size subarray (sum, avg, max)         → Fixed Window
Longest subarray/substring with constraint  → Dynamic Expand + Shrink
Shortest subarray/substring with constraint → Dynamic Shrink when valid
Count of subarrays satisfying constraint    → Exactly K = AtMost(K) - AtMost(K-1)
String permutation / anagram inside string  → Fixed Window + freq map
Min window containing all chars             → Dynamic two-pointer + freq map
No-repeat longest substring                 → Dynamic + set/map
```

**Ask yourself these 3 questions:**
1. Is the window size fixed or variable? → Fixed vs Dynamic
2. Am I maximizing or minimizing? → Expand greedily vs Shrink when valid
3. Do I need count of subarrays? → Probably AtMost trick

---

## THE 5 CORE PATTERNS

---

### PATTERN 1: FIXED SIZE WINDOW

**When:** "subarray of size k", "every window of size k"

**Template:**
```python
def fixed_window(arr, k):
    window_sum = sum(arr[:k])   # build first window
    result = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i]        # add new element
        window_sum -= arr[i - k]    # remove oldest element
        result = max(result, window_sum)

    return result
```

**Key insight:** right = i, left = i - k. Window always size k.

---

### PATTERN 2: DYNAMIC WINDOW — FIND LONGEST

**When:** "longest subarray/substring where [condition]"

**Template:**
```python
def longest_window(arr):
    left = 0
    result = 0
    state = {}   # or counter, or integer

    for right in range(len(arr)):
        # 1. Add arr[right] to state
        state[arr[right]] = state.get(arr[right], 0) + 1

        # 2. Shrink from left while window is INVALID
        while window_is_invalid(state):
            state[arr[left]] -= 1
            if state[arr[left]] == 0:
                del state[arr[left]]
            left += 1

        # 3. Window is now valid — record max
        result = max(result, right - left + 1)

    return result
```

**Key insight:** Expand right always. Shrink left only when invalid. After while loop, window is valid.

---

### PATTERN 3: DYNAMIC WINDOW — FIND SHORTEST

**When:** "minimum length subarray where [condition]"

**Template:**
```python
def shortest_window(arr, target):
    left = 0
    result = float('inf')
    current = 0

    for right in range(len(arr)):
        current += arr[right]    # expand

        # Shrink from left while window is VALID (we want minimum)
        while current >= target:
            result = min(result, right - left + 1)
            current -= arr[left]
            left += 1

    return result if result != float('inf') else 0
```

**Key insight:** Opposite of longest — shrink WHILE valid, record inside the while loop.

---

### PATTERN 4: EXACTLY K → ATMOST(K) - ATMOST(K-1)

**When:** "number of subarrays with exactly k distinct / exactly k odd numbers"

**Why:** "Exactly K" is hard to count directly. But AtMost K is easy.

```
exactly(k) = atmost(k) - atmost(k-1)
```

**Template:**
```python
def count_exactly_k(arr, k):
    return at_most_k(arr, k) - at_most_k(arr, k - 1)

def at_most_k(arr, k):
    left = 0
    count = 0
    freq = {}

    for right in range(len(arr)):
        freq[arr[right]] = freq.get(arr[right], 0) + 1

        while len(freq) > k:   # or whatever condition
            freq[arr[left]] -= 1
            if freq[arr[left]] == 0:
                del freq[arr[left]]
            left += 1

        count += right - left + 1   # all subarrays ending at right

    return count
```

**Why `right - left + 1`?** Every position from left to right forms a valid subarray ending at right.

---

### PATTERN 5: MINIMUM WINDOW SUBSTRING (HARD PATTERN)

**When:** "smallest window containing all characters of t"

```python
def min_window(s, t):
    need = {}
    for c in t:
        need[c] = need.get(c, 0) + 1

    have, total = 0, len(need)   # unique chars satisfied vs needed
    window = {}
    result = ""
    result_len = float('inf')
    left = 0

    for right in range(len(s)):
        c = s[right]
        window[c] = window.get(c, 0) + 1

        if c in need and window[c] == need[c]:
            have += 1

        while have == total:   # window is valid, try to shrink
            if right - left + 1 < result_len:
                result_len = right - left + 1
                result = s[left:right+1]

            window[s[left]] -= 1
            if s[left] in need and window[s[left]] < need[s[left]]:
                have -= 1
            left += 1

    return result
```

---

## ALL LEETCODE PROBLEMS BY PATTERN

---

### PATTERN 1 — FIXED WINDOW

---

**LC 643 — Maximum Average Subarray I** (Easy)

Find max average of subarray of length k.

```python
def findMaxAverage(nums, k):
    window_sum = sum(nums[:k])
    best = window_sum

    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)

    return best / k
```

---

**LC 1343 — Number of Sub-arrays of Size K and Average >= Threshold** (Medium)

```python
def numOfSubarrays(arr, k, threshold):
    window_sum = sum(arr[:k])
    count = 1 if window_sum / k >= threshold else 0

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        if window_sum / k >= threshold:
            count += 1

    return count
```

---

**LC 567 — Permutation in String** (Medium)

Does s1's permutation exist as substring of s2?

Key insight: permutation = same frequency. Use fixed window of len(s1), compare freq maps.

```python
def checkInclusion(s1, s2):
    if len(s1) > len(s2):
        return False

    need = [0] * 26
    window = [0] * 26

    for c in s1:
        need[ord(c) - ord('a')] += 1

    for i in range(len(s1)):
        window[ord(s2[i]) - ord('a')] += 1

    if window == need:
        return True

    for i in range(len(s1), len(s2)):
        window[ord(s2[i]) - ord('a')] += 1
        window[ord(s2[i - len(s1)]) - ord('a')] -= 1
        if window == need:
            return True

    return False
```

---

**LC 438 — Find All Anagrams in a String** (Medium)

Same as 567 but collect all starting indices.

```python
def findAnagrams(s, p):
    need = [0] * 26
    window = [0] * 26
    result = []

    for c in p:
        need[ord(c) - ord('a')] += 1
    for c in s[:len(p)]:
        window[ord(c) - ord('a')] += 1

    if window == need:
        result.append(0)

    for i in range(len(p), len(s)):
        window[ord(s[i]) - ord('a')] += 1
        window[ord(s[i - len(p)]) - ord('a')] -= 1
        if window == need:
            result.append(i - len(p) + 1)

    return result
```

---

**LC 2090 — K Radius Subarray Averages** (Medium)

For each index i, average of window [i-k, i+k]. Window size = 2k+1.

```python
def getAverages(nums, k):
    n = len(nums)
    result = [-1] * n
    if 2 * k + 1 > n:
        return result

    window_sum = sum(nums[:2*k+1])
    result[k] = window_sum // (2*k+1)

    for i in range(k+1, n-k):
        window_sum += nums[i+k] - nums[i-k-1]
        result[i] = window_sum // (2*k+1)

    return result
```

---

### PATTERN 2 — DYNAMIC WINDOW (LONGEST)

---

**LC 3 — Longest Substring Without Repeating Characters** (Medium)

Classic. Window invalid when any char appears twice.

```python
def lengthOfLongestSubstring(s):
    seen = set()
    left = 0
    result = 0

    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        result = max(result, right - left + 1)

    return result
```

---

**LC 424 — Longest Repeating Character Replacement** (Medium)

Replace at most k characters. Window valid when: (window_size - max_freq) <= k

```python
def characterReplacement(s, k):
    count = {}
    left = 0
    max_freq = 0
    result = 0

    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_freq = max(max_freq, count[s[right]])

        # window_size - max_freq = chars we need to replace
        while (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1
            # Note: max_freq never decreases — this is an optimization trick

        result = max(result, right - left + 1)

    return result
```

**Tricky part:** max_freq never decreases. That's intentional — we only care about growing the result, not shrinking it.

---

**LC 1004 — Max Consecutive Ones III** (Medium)

Flip at most k zeros. Window invalid when zeros > k.

```python
def longestOnes(nums, k):
    left = 0
    zeros = 0
    result = 0

    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1

        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1

        result = max(result, right - left + 1)

    return result
```

---

**LC 1493 — Longest Subarray of 1's After Deleting One Element** (Medium)

Same as 1004 with k=1, answer is window_size - 1 (deleted element).

```python
def longestSubarray(nums):
    left = 0
    zeros = 0
    result = 0

    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        while zeros > 1:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        result = max(result, right - left)  # -1 for deleted element

    return result
```

---

**LC 340 — Longest Substring with At Most K Distinct Characters** (Medium) [Premium]

```python
def lengthOfLongestSubstringKDistinct(s, k):
    freq = {}
    left = 0
    result = 0

    for right in range(len(s)):
        freq[s[right]] = freq.get(s[right], 0) + 1

        while len(freq) > k:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1

        result = max(result, right - left + 1)

    return result
```

---

**LC 159 — Longest Substring with At Most Two Distinct Characters** (Medium) [Premium]

Same as 340 with k=2.

---

**LC 2024 — Maximize the Confusion of an Exam** (Medium)

Same as 424. Either maximize T's replaced with F, or F's replaced with T. Run 424 twice.

```python
def maxConsecutiveAnswers(answerKey, k):
    def longest(c):
        count = 0
        left = 0
        result = 0
        for right in range(len(answerKey)):
            if answerKey[right] == c:
                count += 1
            while count > k:
                if answerKey[left] == c:
                    count -= 1
                left += 1
            result = max(result, right - left + 1)
        return result

    return max(longest('T'), longest('F'))
```

---

**LC 1838 — Frequency of the Most Frequent Element** (Medium)

Sort first. Then expand window. Window valid when: max_val * size - total_sum <= k

```python
def maxFrequency(nums, k):
    nums.sort()
    left = 0
    total = 0
    result = 0

    for right in range(len(nums)):
        total += nums[right]

        # cost to make all elements equal nums[right]
        while nums[right] * (right - left + 1) - total > k:
            total -= nums[left]
            left += 1

        result = max(result, right - left + 1)

    return result
```

---

**LC 395 — Longest Substring with At Least K Repeating Characters** (Medium)

Sliding window doesn't directly apply here. Use divide and conquer OR sliding window with fixed number of unique chars (1 to 26).

```python
def longestSubstring(s, k):
    result = 0
    for unique_target in range(1, 27):
        freq = {}
        left = 0
        at_least_k = 0   # chars with freq >= k

        for right in range(len(s)):
            freq[s[right]] = freq.get(s[right], 0) + 1
            if freq[s[right]] == k:
                at_least_k += 1

            while len(freq) > unique_target:
                if freq[s[left]] == k:
                    at_least_k -= 1
                freq[s[left]] -= 1
                if freq[s[left]] == 0:
                    del freq[s[left]]
                left += 1

            if len(freq) == unique_target == at_least_k:
                result = max(result, right - left + 1)

    return result
```

---

### PATTERN 3 — DYNAMIC WINDOW (SHORTEST)

---

**LC 209 — Minimum Size Subarray Sum** (Medium)

```python
def minSubArrayLen(target, nums):
    left = 0
    total = 0
    result = float('inf')

    for right in range(len(nums)):
        total += nums[right]

        while total >= target:
            result = min(result, right - left + 1)
            total -= nums[left]
            left += 1

    return result if result != float('inf') else 0
```

---

**LC 76 — Minimum Window Substring** (Hard)

Already shown above in Pattern 5. The canonical hard sliding window problem.

---

**LC 632 — Smallest Range Covering Elements from K Lists** (Hard)

More complex — use heap + sliding window concept. Track current max, slide min up.

---

### PATTERN 4 — COUNT SUBARRAYS (ATMOST TRICK)

---

**LC 713 — Subarray Product Less Than K** (Medium)

Count subarrays with product < k. Every valid right contributes (right - left + 1) subarrays.

```python
def numSubarrayProductLessThanK(nums, k):
    if k <= 1:
        return 0

    left = 0
    product = 1
    count = 0

    for right in range(len(nums)):
        product *= nums[right]

        while product >= k:
            product //= nums[left]
            left += 1

        count += right - left + 1

    return count
```

---

**LC 992 — Subarrays with K Different Integers** (Hard)

Exactly K = AtMost(K) - AtMost(K-1)

```python
def subarraysWithKDistinct(nums, k):
    def at_most(k):
        freq = {}
        left = 0
        count = 0

        for right in range(len(nums)):
            freq[nums[right]] = freq.get(nums[right], 0) + 1

            while len(freq) > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1

            count += right - left + 1

        return count

    return at_most(k) - at_most(k - 1)
```

---

**LC 1248 — Count Number of Nice Subarrays** (Medium)

Exactly k odd numbers. Same AtMost trick.

```python
def numberOfSubarrays(nums, k):
    def at_most(k):
        left = 0
        odds = 0
        count = 0

        for right in range(len(nums)):
            if nums[right] % 2 == 1:
                odds += 1
            while odds > k:
                if nums[left] % 2 == 1:
                    odds -= 1
                left += 1
            count += right - left + 1

        return count

    return at_most(k) - at_most(k - 1)
```

---

**LC 930 — Binary Subarrays With Sum** (Medium)

Exactly k ones. AtMost trick again.

```python
def numSubarraysWithSum(nums, goal):
    def at_most(k):
        if k < 0:
            return 0
        left = 0
        total = 0
        count = 0
        for right in range(len(nums)):
            total += nums[right]
            while total > k:
                total -= nums[left]
                left += 1
            count += right - left + 1
        return count

    return at_most(goal) - at_most(goal - 1)
```

---

**LC 2461 — Maximum Sum of Distinct Subarrays With Length K** (Medium)

Fixed window + ensure all distinct.

```python
def maximumSubarraySum(nums, k):
    freq = {}
    window_sum = 0
    result = 0

    for i in range(len(nums)):
        freq[nums[i]] = freq.get(nums[i], 0) + 1
        window_sum += nums[i]

        if i >= k:
            old = nums[i - k]
            window_sum -= old
            freq[old] -= 1
            if freq[old] == 0:
                del freq[old]

        if i >= k - 1 and len(freq) == k:
            result = max(result, window_sum)

    return result
```

---

### PATTERN 5 — MINIMUM WINDOW SUBSTRING VARIANTS

---

**LC 76 — Minimum Window Substring** (Hard) — shown above

---

**LC 239 — Sliding Window Maximum** (Hard)

Fixed window but track maximum efficiently using a monotonic deque.

```python
from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()   # stores indices, decreasing values
    result = []

    for i in range(len(nums)):
        # remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**This is the deque/monotonic window pattern.** The deque stores indices in decreasing order of their values. Front is always the max.

---

**LC 480 — Sliding Window Median** (Hard)

Fixed window, track median using two heaps (max-heap for lower half, min-heap for upper half).

---

## COMMON MISTAKES & HOW TO AVOID THEM

```
Mistake                              Fix
──────────────────────────────────────────────────────────
Forgetting to shrink window         Always ask: what makes window INVALID?
Off-by-one in window size           right - left + 1 is the size
Counting subarrays wrong            += right - left + 1, not just +1
Using exactly-k directly            Use atmost(k) - atmost(k-1)
Not handling empty result           Initialize result as 0 or inf appropriately
Shrinking past left > right         Add left <= right guard if needed
```

---

## DECISION FLOWCHART

```
Is window size fixed?
├── YES → Fixed Window (Pattern 1)
└── NO
    ├── Maximize length?
    │   └── Dynamic Longest (Pattern 2)
    │       expand right always
    │       shrink left when INVALID
    │       record AFTER while loop
    │
    ├── Minimize length?
    │   └── Dynamic Shortest (Pattern 3)
    │       expand right always
    │       shrink left while VALID
    │       record INSIDE while loop
    │
    ├── Count subarrays exactly K?
    │   └── AtMost(K) - AtMost(K-1) (Pattern 4)
    │       count += right - left + 1
    │
    └── Smallest window with all targets?
        └── Min Window Substring (Pattern 5)
            track have/need counts
```

---

## COMPLEXITY

All sliding window solutions: **O(n) time, O(1) or O(k) space**

Each element is added once and removed once → 2n operations → O(n)

---

## PRACTICE ORDER (by difficulty)

```
Start here:
1. LC 643  — Max Average Subarray (warm up)
2. LC 3    — Longest Without Repeating
3. LC 209  — Min Size Subarray Sum
4. LC 567  — Permutation in String
5. LC 1004 — Max Consecutive Ones III

Then:
6. LC 424  — Longest Repeating Char Replacement
7. LC 713  — Subarray Product < K
8. LC 438  — Find All Anagrams
9. LC 930  — Binary Subarrays With Sum
10. LC 1248 — Nice Subarrays

Hard:
11. LC 992  — Subarrays K Different Integers
12. LC 76   — Minimum Window Substring
13. LC 239  — Sliding Window Maximum
```

Master these 13 and you've covered every sliding window pattern that appears in interviews.
