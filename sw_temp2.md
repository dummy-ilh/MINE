# Sliding Window — Full Template Library
(written in your style: explicit loops, no slicing tricks, no comprehensions)

---

# PART A — FIXED-SIZE WINDOW (window size `k` is given and never changes)

### A1. Brute force baseline (for intuition only — O(n*k), don't use this)
```python
def max_subarray_sum(nums, k):
    max_sum = float("-inf")
    for i in range(len(nums) - k + 1):
        current_sum = sum(nums[i:i + k])
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

### A2. Running sum (O(n)) — this is the real Template A
**Recognition:** "subarray of size k", "average of k elements", "sum of window k"
**Invariant:** `current_sum` always equals the sum of exactly the last `k` elements seen.
```python
def max_subarray_sum(nums, k):
    current_sum = sum(nums[:k])
    max_sum = current_sum
    for i in range(len(nums) - k):
        current_sum -= nums[i]
        current_sum += nums[i + k]
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

### A3. Running product — same shape, but read the caveat
**Recognition:** "product of k elements"
**Invariant:** `current_product` equals the product of the last `k` elements.
**Caveat:** dividing only works if no element is `0`. If `0` can appear, you must detect it and recompute the window from scratch instead of dividing. Negative numbers are fine for a *fixed* window (division is still exact), they just make max/min tracking less intuitive.
```python
def max_subarray_product(nums, k):
    current_product = 1
    for i in range(k):
        current_product *= nums[i]
    max_product = current_product
    for i in range(len(nums) - k):
        current_product /= nums[i]
        current_product *= nums[i + k]
        if current_product > max_product:
            max_product = current_product
    return max_product
```

### A4. Exact count match (numeric)
**Recognition:** "how many windows of size k sum to target"
```python
def subarray_target_sum(nums, target, k):
    current_sum = 0
    for i in range(k):
        current_sum += nums[i]
    count = 1 if current_sum == target else 0
    for i in range(len(nums) - k):
        current_sum -= nums[i]
        current_sum += nums[i + k]
        if current_sum == target:
            count += 1
    return count
```

### A5. Frequency map exact match — anagram check (fixed, correct version)
**Recognition:** "permutation of", "anagram of", "same characters as"
**Why not a `set`:** a `set` only tracks *which* characters exist, not *how many* of each. `"aab"` and `"ab"` give the same set but are not anagrams. A `Counter` tracks counts, so it's the only correct tool here.
**Invariant:** `window_counter` always equals the exact letter-count multiset of the last `k` characters.
```python
from collections import Counter

def has_substring_anagram(s, anagram):
    k = len(anagram)
    anagram_counter = Counter(anagram)
    window_counter = Counter(s[:k])
    if window_counter == anagram_counter:
        return True
    for i in range(len(s) - k):
        trailing_char = s[i]
        leading_char = s[i + k]
        window_counter[trailing_char] -= 1
        window_counter[leading_char] += 1
        if window_counter == anagram_counter:
            return True
    return False
```

### A6. Frequency map — count ALL matches (your version, already correct)
```python
from collections import Counter

def count_substring_anagrams(s, anagram):
    anagram_counter = Counter(anagram)
    window_counter = Counter(s[:len(anagram)])
    num_matches = 1 if anagram_counter == window_counter else 0
    for i in range(len(s) - len(anagram)):
        trailing_char = s[i]
        leading_char = s[i + len(anagram)]
        window_counter[trailing_char] -= 1
        window_counter[leading_char] += 1
        if window_counter == anagram_counter:
            num_matches += 1
    return num_matches
```

### A7. Max/Min inside a fixed window — Monotonic Deque (the one exception to remove/add)
**Recognition:** "max of every window of size k", "sliding window maximum"
**Why it doesn't fit the normal remove/add shape:** `remove(data[i])` is only cheap if the outgoing element happens to be the current max/min. A plain running value can't answer "what's the max" in O(1) after a removal. The fix is to keep a deque of *candidate indices*, always sorted so the best candidate sits at the front.
**Invariant:** the deque holds indices in decreasing value order; the front index is always the max of the current window.
```python
from collections import deque

def max_in_each_window(nums, k):
    candidates = deque()   # stores indices, not values
    result = []
    for i in range(len(nums)):
        while candidates and nums[candidates[-1]] <= nums[i]:
            candidates.pop()
        candidates.append(i)
        if candidates[0] <= i - k:
            candidates.popleft()
        if i >= k - 1:
            result.append(nums[candidates[0]])
    return result
```

### A8. Generic fixed-window skeleton
```python
window = build_first_window()
answer = process(window)
for i in range(len(data) - k):
    remove(data[i])
    add(data[i + k])
    answer = update(answer)
return answer
```

---

# PART B — VARIABLE-SIZE WINDOW (window grows and shrinks based on a condition)

### B1. Exact target sum, return indices (your version)
**Recognition:** "two pointers", "find a subarray that sums to exactly target" (positive numbers only — shrinking only works because adding more elements only ever increases the sum)
```python
def find_target_sum_window(nums, target):
    left = 0
    current_sum = 0
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if current_sum == target:
            return [left, right]
    return None
```

### B2. Longest valid window — no repeating characters
**Recognition:** "longest substring without repeating", "longest subarray with no duplicates"
**Invariant:** every character inside the window appears at most once.
```python
from collections import Counter

def longest_unique_substring(s):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(s)):
        char = s[right]
        window_counter[char] += 1
        while window_counter[char] > 1:
            left_char = s[left]
            window_counter[left_char] -= 1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

### B3. Longest valid window with a constraint — "at most K distinct"
**Recognition:** "at most K distinct characters/types", "fruit into baskets" (K=2)
**Invariant:** `len(window_counter)` (number of distinct keys) never exceeds K.
```python
from collections import Counter

def longest_with_at_most_k_distinct(s, k):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(s)):
        char = s[right]
        window_counter[char] += 1
        while len(window_counter) > k:
            left_char = s[left]
            window_counter[left_char] -= 1
            if window_counter[left_char] == 0:
                del window_counter[left_char]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

### B4. Shortest/minimum valid window — numeric version
**Recognition:** "minimum size subarray with sum >= target", "shortest subarray that..."
**Key shape difference from B2/B3:** the `while` loop shrinks as much as possible *every time the window is valid* (not just when invalid), because you're hunting for the smallest valid window, not the largest.
```python
def min_subarray_len(nums, target):
    left = 0
    current_sum = 0
    min_len = float("inf")
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum >= target:
            window_len = right - left + 1
            if window_len < min_len:
                min_len = window_len
            current_sum -= nums[left]
            left += 1
    if min_len == float("inf"):
        return 0
    return min_len
```

### B5. Shortest/minimum valid window — frequency version (Minimum Window Substring)
**Recognition:** "smallest substring that contains all characters of...", "minimum window substring"
**Invariant:** `formed == required` means every required character currently has enough copies in the window.
```python
from collections import Counter

def min_window_substring(s, target):
    target_counter = Counter(target)
    window_counter = Counter()
    required = len(target_counter)
    formed = 0
    left = 0
    min_len = float("inf")
    min_left = 0
    for right in range(len(s)):
        char = s[right]
        window_counter[char] += 1
        if char in target_counter and window_counter[char] == target_counter[char]:
            formed += 1
        while formed == required:
            current_len = right - left + 1
            if current_len < min_len:
                min_len = current_len
                min_left = left
            left_char = s[left]
            window_counter[left_char] -= 1
            if left_char in target_counter and window_counter[left_char] < target_counter[left_char]:
                formed -= 1
            left += 1
    if min_len == float("inf"):
        return ""
    return s[min_left:min_left + min_len]
```

### B6. Count ALL valid windows — the "at most K" counting trick
**Recognition:** "number of subarrays with...", "count subarrays where product < k"
**The trick:** for every `right`, once the window is shrunk back to valid, *every* window ending at `right` and starting anywhere from `left` to `right` is valid. So you don't count one window at a time — you count `right - left + 1` of them in one shot.
```python
def count_subarrays_product_less_than_k(nums, k):
    left = 0
    current_product = 1
    count = 0
    for right in range(len(nums)):
        current_product *= nums[right]
        while left <= right and current_product >= k:
            current_product /= nums[left]
            left += 1
        window_size = right - left + 1
        count += window_size
    return count
```
**Meta-trick for "exactly K" problems:** if a problem asks for "exactly K distinct" rather than "at most K", solve it as `atMost(K) - atMost(K - 1)`. You almost never write an "exactly K" sliding window directly — you build it from two "at most K" calls.

### B7. Generic variable-window skeleton
```python
left = 0
for right in range(len(nums)):
    add(nums[right])
    while invalid():
        remove(nums[left])
        left += 1
    business_logic()
```

---

# Quick map: which template am I looking at?

| Question phrasing | Template |
|---|---|
| "window of size k", fixed | A2 / A3 / A4 |
| "anagram" / "permutation of", fixed size | A5 / A6 |
| "max/min of every window of size k" | A7 (deque) |
| "exact sum, positive numbers" | B1 |
| "longest ... without repeating / at most K distinct" | B2 / B3 |
| "shortest / minimum size subarray" | B4 / B5 |
| "number of subarrays where..." | B6 |

One thing worth internalizing: **A7 is the only template here that breaks the clean remove/add shape.** Every other template — fixed or variable — reduces to the same two moves (shrink left, extend right) plus one running piece of state (a sum, a product, or a counter). A7 needs a second structure (the deque) because "what's the max" isn't a single number you can update with `+=`/`-=` — recognizing *that* distinction in an interview is usually worth more than knowing the code by heart.
