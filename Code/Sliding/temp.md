
---

### 219. Contains Duplicate II
**Template A5 variant** — instead of a `Counter` tracking exact multiset, we only need membership, so a `set` replaces the `Counter`. Window size is `k` (not `k+1`).
```python
def contains_nearby_duplicate(nums, k):
    window_set = set()
    for i in range(len(nums)):
        if nums[i] in window_set:
            return True
        window_set.add(nums[i])
        if len(window_set) > k:
            window_set.remove(nums[i - k])
    return False
```

---

### 594. Longest Harmonious Subsequence
**Template B3 variant** — sort first (subsequence, not subarray, so order doesn't matter). Invariant becomes `nums[right] - nums[left] <= 1` instead of "distinct count <= K".
```python
def find_lhs(nums):
    nums.sort()
    left = 0
    max_len = 0
    for right in range(len(nums)):
        while nums[right] - nums[left] > 1:
            left += 1
        if nums[right] - nums[left] == 1:
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
    return max_len
```

---

### 1652. Defuse the Bomb
**Template A1 variant** — circular indexing breaks clean remove/add, and `n <= 100`, so brute force per index is intentional here (use `% n` instead of true sliding).
```python
def decrypt(code, k):
    n = len(code)
    result = []
    for i in range(n):
        current_sum = 0
        if k > 0:
            for j in range(1, k + 1):
                current_sum += code[(i + j) % n]
        elif k < 0:
            for j in range(1, abs(k) + 1):
                current_sum += code[(i - j) % n]
        result.append(current_sum)
    return result
```

---

### 1763. Longest Nice Substring
**Template A1 variant** — "nice" is not a property you can shrink-fix from the left (non-monotonic), so this stays a nested brute enumeration rather than true two-pointer B2.
```python
def longest_nice_substring(s):
    best = ""
    for left in range(len(s)):
        lower_seen = set()
        upper_seen = set()
        for right in range(left, len(s)):
            char = s[right]
            if char.islower():
                lower_seen.add(char)
            else:
                upper_seen.add(char)
            if lower_seen == upper_seen and right - left + 1 > len(best):
                best = s[left:right + 1]
    return best
```

---

### 1876. Substrings of Size Three with Distinct Characters
**Template A6 variant** — match condition becomes `len(window_counter) == k` (all distinct) instead of `window_counter == target_counter`.
```python
from collections import Counter

def count_good_substrings(s):
    k = 3
    window_counter = Counter(s[:k])
    count = 1 if len(window_counter) == k else 0
    for i in range(len(s) - k):
        trailing_char = s[i]
        leading_char = s[i + k]
        window_counter[trailing_char] -= 1
        if window_counter[trailing_char] == 0:
            del window_counter[trailing_char]
        window_counter[leading_char] += 1
        if len(window_counter) == k:
            count += 1
    return count
```

---

### 1984. Minimum Difference Between Highest and Lowest of K Scores
**Template A2 variant** — sort first, then since the window is sorted, max/min are just the two endpoints, so no running accumulator is even needed.
```python
def minimum_difference(nums, k):
    nums.sort()
    min_diff = float("inf")
    for i in range(len(nums) - k + 1):
        current_diff = nums[i + k - 1] - nums[i]
        if current_diff < min_diff:
            min_diff = current_diff
    return min_diff
```



---

### 2379. Minimum Recolors to Get K Consecutive Black Blocks
**Template A2, unmodified shape** — `current_sum` becomes `white_count`, "add/remove nums[i]" becomes "add/remove 1 if char is 'W'".
```python
def minimum_recolors(blocks, k):
    white_count = 0
    for i in range(k):
        if blocks[i] == "W":
            white_count += 1
    min_recolors = white_count
    for i in range(len(blocks) - k):
        if blocks[i] == "W":
            white_count -= 1
        if blocks[i + k] == "W":
            white_count += 1
        if white_count < min_recolors:
            min_recolors = white_count
    return min_recolors
```

---

### 2760. Longest Even Odd Subarray With Threshold
**Template B2 variant** — instead of shrinking left by one step at a time, a single violation invalidates the *entire* prefix, so `left` jumps straight to `right` (restart pattern) rather than incrementing.
```python
def longest_alternating_subarray(nums, threshold):
    n = len(nums)
    max_len = 0
    left = 0
    while left < n:
        if nums[left] % 2 != 0 or nums[left] > threshold:
            left += 1
            continue
        right = left + 1
        while right < n and nums[right] <= threshold and nums[right] % 2 != nums[right - 1] % 2:
            right += 1
        current_len = right - left
        if current_len > max_len:
            max_len = current_len
        left = right
    return max_len
```


### 3090. Maximum Length Substring With Two Occurrences
**Template B2 variant** — identical shape to "longest unique substring", just change the shrink threshold from `> 1` to `> 2`.
```python
from collections import Counter

def maximum_length_substring(s):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(s)):
        char = s[right]
        window_counter[char] += 1
        while window_counter[char] > 2:
            left_char = s[left]
            window_counter[left_char] -= 1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

### 3206. Alternating Groups I
**Template A4 variant** — fixed window of size 3, circular (`% n`); match condition is "middle differs from both neighbors".
```python
def number_of_alternating_groups(colors):
    n = len(colors)
    count = 0
    for i in range(n):
        left_neighbor = colors[(i - 1) % n]
        right_neighbor = colors[(i + 1) % n]
        if colors[i] != left_neighbor and colors[i] != right_neighbor:
            count += 1
    return count
```

---

### 3258. Count Substrings That Satisfy K-Constraint I
**Template B6, unmodified shape** — "invalid" becomes "both the 0-count AND 1-count exceed k" (two counters instead of one running product).
```python
def count_k_constraint_substrings(s, k):
    left = 0
    count0 = 0
    count1 = 0
    total = 0
    for right in range(len(s)):
        if s[right] == "0":
            count0 += 1
        else:
            count1 += 1
        while count0 > k and count1 > k:
            if s[left] == "0":
                count0 -= 1
            else:
                count1 -= 1
            left += 1
        window_size = right - left + 1
        total += window_size
    return total
```

---

### 3364. Minimum Positive Sum Subarray
**Template A2 variant, run once per window size** — the only new idea is wrapping A2 in an outer loop over every length from `l` to `r`, since the window size itself is a range, not a single fixed `k`.
```python
def minimum_positive_sum(nums, l, r):
    best = float("inf")
    n = len(nums)
    for length in range(l, r + 1):
        current_sum = sum(nums[:length])
        if 0 < current_sum < best:
            best = current_sum
        for i in range(n - length):
            current_sum -= nums[i]
            current_sum += nums[i + length]
            if 0 < current_sum < best:
                best = current_sum
    if best == float("inf"):
        return -1
    return best
```


