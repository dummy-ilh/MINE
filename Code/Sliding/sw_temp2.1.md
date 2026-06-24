# Sliding Window — Master Template Reference

Two main families: **Part A** (fixed-size window — size is given and never changes) and
**Part B** (variable-size window — grows and shrinks based on a condition). **Part C** is a
deep dive into counting specifically, since "count windows where..." has its own shape and its
own confusing `k`. **Part D** covers things that look like sliding window but aren't.

**Read the `k` callout on every template before using it** — `k` means something different
almost everywhere: a window size, a max-distinct-count, an exact target count, an OR-threshold,
or sometimes nothing at all.

---

# PART A — FIXED-SIZE WINDOW
**Recognition signal for the whole part:** the problem hands you a size and it never moves —
"window of size k", "k consecutive elements", "every substring of length k".

### A1 — Brute Force (baseline only, O(n·k), don't ship this)
```python
def max_subarray_sum(nums, k):
    max_sum = float("-inf")
    for i in range(len(nums) - k + 1):
        current_sum = sum(nums[i:i + k])
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

### A2 — Running Sum (the canonical fixed-window template)
**`k` means:** the window size.
**Invariant:** `current_sum` always equals the sum of exactly the last `k` elements.
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

### A3 — Running Product
**`k` means:** the window size.
**Caveat:** dividing only works if no element is `0`. If `0` can appear, detect it and
recompute the window from scratch instead of dividing.
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

### A4 — Count Matches (sum equals an exact target, fixed size)
**`k` means:** the window size. `target` is the separate value you're matching — don't
conflate the two.
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

### A5 — Anagram Check, Boolean (full Counter comparison)
**`k` means:** `len(anagram)` — the window size is derived from the pattern, not given directly.
**Why not a `set`:** a set only tracks *which* characters exist, not *how many*. Use `Counter`.
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

### A6 — Anagram Check, Count All Matches (full Counter comparison)
Same shape as A5, just doesn't early-return.
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

### A7 — Anagram/Permutation, Fast Matched-Counter (returns all start indices)
**Why this exists alongside A5/A6:** comparing two Counters every step is O(alphabet) per
step. This version tracks one integer `matched` — how many distinct characters currently have
the *exact* required count — so each step is O(1).
**`k` means:** `len(p)` — derived from the pattern.
**Invariant:** `matched == required` exactly when the window is an anagram of `p`.
```python
def frequency_match(s, p):
    need = {}
    for ch in p:
        need[ch] = need.get(ch, 0) + 1

    window = {}
    matched = 0
    required = len(need)
    left = 0
    result = []

    for right in range(len(s)):
        ch = s[right]
        if ch in need:
            window[ch] = window.get(ch, 0) + 1
            if window[ch] == need[ch]:
                matched += 1

        if right - left + 1 > len(p):
            left_ch = s[left]
            if left_ch in need:
                if window[left_ch] == need[left_ch]:
                    matched -= 1
                window[left_ch] -= 1
            left += 1

        if matched == required and right - left + 1 == len(p):
            result.append(left)

    return result
```

### A8 — Max/Min Per Window — Monotonic Deque
**`k` means:** the window size.
**Why it doesn't fit the normal remove/add shape:** `remove(data[i])` is only cheap if the
outgoing element happens to be the current max/min. A plain running value can't answer "what's
the max" in O(1) after a removal — you need a deque of *candidate indices*, kept sorted so the
best candidate sits at the front.
**Invariant:** the deque holds indices in decreasing value order; the front is always the max
of the current window.
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
**Variant:** if validity depends on `max − min ≤ limit`, run a max-deque AND a min-deque at the
same time (dual deque).

### A9 — Circular Fixed Window
**`k` means:** the window size. The array also **wraps around** — "circular", "wraps", "last
element connects back to the first".
**Two approaches:**
1. **Double the array** (`arr + arr`) — every window is now contiguous, no modulo needed. Best
   when `k` is large or variable.
2. **Modulo indexing** — best when `k` is small and fixed (e.g. `k = 3`).
```python
# Approach 1: doubling
def circular_window_sum(code, k):
    n = len(code)
    extended = code + code

    current_sum = sum(extended[1:k + 1])   # window for index 0
    result = [current_sum]

    for i in range(1, n):
        current_sum -= extended[i]
        current_sum += extended[i + k]
        result.append(current_sum)

    return result   # only the first n entries are meaningful

# Approach 2: modulo, fixed small k=3
def circular_window_small(colors):
    n = len(colors)
    count = 0
    for i in range(n):
        prev = colors[(i - 1) % n]
        curr = colors[i]
        nxt = colors[(i + 1) % n]
        if prev != curr and curr != nxt:
            count += 1
    return count
```
**Caveat:** modulo indexing gets messy fast for variable-length windows — prefer doubling there.

### A10 — Generic skeleton
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

# PART B — VARIABLE-SIZE WINDOW
**Recognition signal for the whole part:** the window grows and shrinks based on a condition —
"longest...", "shortest...", "exact sum...".

### B1 — Exact Target Sum, Two Pointer (positive numbers only)
**`target`** is the exact sum to hit; no `k`. **Works only because all numbers are positive** —
adding more elements only ever increases the sum (monotonic).
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

### B2 — Longest, No Repeats
**No `k`** — the condition is structural ("no duplicate"), not parametrized.
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

### B3 — Longest, At Most K Distinct
**`k` means:** the maximum number of *distinct* elements/types allowed in the window — **not**
the window size. ("Fruit into baskets" is this with k=2.)
**Invariant:** `len(window_counter)` never exceeds `k`.
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
**Variant — jump-reset:** when the violation is structural and tied to the boundary itself
(not "too much of something"), jump `left = right` instead of incrementally shrinking. See B7
and the leetcode variants in Part F for examples.

### B4 — Shortest, Numeric Sum ≥ Target
**`target`** means: the threshold the running sum must reach. No `k`.
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
    return min_len if min_len != float("inf") else 0
```

### B5 — Shortest, Frequency Match (Minimum Window Substring)
**No `k`; `target`** is the pattern string whose characters must all be covered.
**Invariant:** `formed == required` means every required character currently has enough
copies in the window.
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

### B6 — Counting Windows, At Most K (the count-all-valid-windows trick)
**`k` means:** the threshold the running aggregate must stay under/at-most — **not** a window
size. This template is covered in depth in Part C since counting is its own shape.
**The trick:** once a window ending at `right` is valid (shrunk back from `left`), *every*
window ending at `right` and starting anywhere from `left` to `right` is also valid — count
`right - left + 1` of them in one shot.
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

### B7 — Longest, Non-Monotonic Validity (Expand-From-Start)
**No `k`** in general — for conditions that can **become valid again after being invalid** as
you extend right (a structural/equality condition, not "too much of X"). This breaks B2/B3
because shrinking from the left doesn't reliably fix a structural mismatch.
```python
def longest_valid_from_each_start(arr):
    best = 0
    n = len(arr)
    for i in range(n):
        state = init_state()
        for j in range(i, n):
            state = update(state, arr[j])
            if cannot_ever_recover(state):
                break
            if is_valid(state):
                current_len = j - i + 1
                if current_len > best:
                    best = current_len
    return best
```
**Common mistake:** patching B2/B3 with a `while`-shrink loop here — it won't converge, since
what broke validity isn't "too much of one element," it's structural.
**Complexity:** O(n²), Space O(1) — this is the brute-force fallback for this family.

### B8 — Shortest, Non-Invertible Aggregate (OR / AND / XOR)
**Why this exists:** B4's shrink step only works because addition is invertible (`-=` undoes
`+=`). Bitwise OR/AND are **not** invertible — once a bit is OR'd in, you can't "subtract" an
element back out. Applying the two-pointer shrink here silently gives wrong answers. Recompute
fresh from each start index instead.
```python
def shortest_window_or_at_least_k(arr, k):
    best = float("inf")
    n = len(arr)
    for i in range(n):
        agg = 0
        for j in range(i, n):
            agg |= arr[j]
            if agg >= k:
                window_len = j - i + 1
                if window_len < best:
                    best = window_len
                break
    return best if best != float("inf") else -1
```

### B9 — Generic skeleton
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

# PART C — DYNAMIC WINDOW COUNTING (counting with a condition, in depth)

This is its own shape, not a flavor of longest/shortest — and it's where `k`-confusion is
worst, so it gets its own dedicated walkthrough.

**The defining trick (same as B6):** once a window ending at `right` is valid, every window
ending at `right` starting anywhere from the current `left` to `right` is also valid. Count
`right - left + 1` in one shot rather than counting windows one at a time.

### C1 — At Most K (cross-reference to B6)
**`k` means:** the threshold the running aggregate must stay at-or-under. This **only works
for monotonic conditions** — adding an element can only make things "more invalid," never less.
See B6 above for the code (product < k). Same shape applies to "sum ≤ k", "distinct count ≤ k",
etc. — just swap what `current_product` tracks.

### C2 — Exactly K = AtMost(K) − AtMost(K−1)
**`k` means:** the exact count you want (e.g. exactly k odd numbers, exactly k distinct
elements).
**Why you basically never write "exactly K" directly:** "exactly K" isn't monotonic on its own
(adding an element can flip a window from valid → invalid → valid again), but "at most K"
always is — so build "exactly K" out of two "at most K" calls.
```python
def at_most_k(nums, k):
    if k < 0:
        return 0
    left = 0
    count_ones = 0
    total = 0
    for right in range(len(nums)):
        count_ones += nums[right]
        while count_ones > k:
            count_ones -= nums[left]
            left += 1
        total += right - left + 1
    return total

def exactly_k(nums, k):
    return at_most_k(nums, k) - at_most_k(nums, k - 1)
```
**Common mistake:** forgetting `at_most_k(nums, -1)` must return `0`, or writing two subtly
different passes instead of reusing one function.

### C3 — Fixed-Size Counting (cross-reference to A4)
"Count windows of size k that sum to target" is a counting problem where the **window size is
fixed** and the threshold is exact equality — see A4. Don't confuse this with C1/C2, where the
window size itself is what's variable and the threshold is a bound, not an exact value.

### C4 — Counting With Two Simultaneous Thresholds
**Recognition:** the invalid condition isn't a single running number but **two counters that
both have to clear a threshold** before the window is invalid (e.g. "both the 0-count AND the
1-count exceed k"). Same shrink/count shape as C1, just two tracked quantities instead of one.
See problem 3258 in Part F for a worked example.

### C5 — No-Window Pair Counting (not actually a window — see D2)
**Recognition:** the count depends on **pairs of values across the whole array**, not
contiguous ranges — e.g. "count pairs where value and value+1 both appear." There's no
`left`/`right` pointer at all. This is listed here only as a warning sign; the real template is
D2 in Part D, because it isn't sliding window.

---

# PART D — NOT ACTUALLY SLIDING WINDOW (commonly mistaken for it)

### D1 — All-Pairs Brute Force
**Recognition:** "pair (i, j)" where i and j are **not required to be adjacent or ordered** —
any two elements can form a pair. No contiguous range to slide over.
```python
def best_pair(arr):
    best = float("-inf")
    n = len(arr)
    for i in range(n):
        for j in range(n):
            if pair_condition(arr[i], arr[j]):
                candidate = combine(arr[i], arr[j])
                if candidate > best:
                    best = candidate
    return best
```
**Common mistake:** seeing the word "pair" and reaching for a window anyway.

### D2 — Frequency-Map Pairing (No Window At All)
**Recognition:** the problem is about a **subsequence** — order and contiguity don't matter,
only counts and how one value relates to *another* value (e.g. "value and value+1").
```python
from collections import Counter

def frequency_pair_best(nums):
    freq = Counter(nums)
    best = 0
    for value in freq:
        if value + 1 in freq:
            combined = freq[value] + freq[value + 1]
            if combined > best:
                best = combined
    return best
```

### D3 — Two-Pointer on Sorted Array / Binary Search on Answer
**Recognition:** input is sorted (or can be sorted), and the optimal block of k elements is
always *contiguous after sorting* — "k closest elements", "max window where max−min ≤ X after
sorting". `k` here means the number of elements to pick.
```python
def find_k_closest(arr, k, x):
    arr.sort()
    left, right = 0, len(arr) - k
    while left < right:
        mid = (left + right) // 2
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]

# Longest window with max-min <= 2k, after sorting:
def max_beauty(nums, k):
    nums.sort()
    left = 0
    best = 0
    for right in range(len(nums)):
        while nums[right] - nums[left] > 2 * k:
            left += 1
        current_len = right - left + 1
        if current_len > best:
            best = current_len
    return best
```
**Common mistake:** applying sliding window before sorting — `max - min` only collapses to
`arr[right] - arr[left]` because the array is sorted.

### D4 — DP With a Sliding Running Sum
**Recognition:** a DP recurrence of the form `dp[i] = f(dp[i-1], ..., dp[i-k])` — the current
DP value depends on a *range* of previous DP values. `k` means the width of that dependency
range (e.g. max points per turn, max jump distance) — not a window over the input array.
```python
def new_21_game(n, k, max_pts):
    if k == 0 or n >= k + max_pts:
        return 1.0
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    window_sum = 1.0
    result = 0.0
    for i in range(1, n + 1):
        dp[i] = window_sum / max_pts
        if i < k:
            window_sum += dp[i]
        else:
            result += dp[i]
        if i >= max_pts:
            window_sum -= dp[i - max_pts]
    return result
```
**Common mistake:** recomputing the window sum from scratch each step (turns O(n) into
O(n·k)) — maintain it as a running value alongside the DP fill.

---

# PART E — QUICK RECOGNITION TABLE

| Phrasing in the problem | Template | What the parameter actually means |
|---|---|---|
| "window of size k" | A2 / A3 / A4 | window size |
| "anagram of" / "permutation of" (fixed size) | A5 / A6 / A7 | len(pattern) |
| "max/min of every window of size k" | A8 | window size |
| "circular array", "wraps around" | A9 | window size (array also wraps) |
| "exact sum, positive numbers, return range" | B1 | n/a (target is the sum) |
| "longest ... without repeating" | B2 | n/a |
| "longest ... at most k distinct" | B3 | max distinct count allowed |
| "shortest / minimum size subarray, sum ≥ target" | B4 | n/a (target is the threshold) |
| "smallest substring containing all of..." | B5 | n/a (pattern defines requirement) |
| "number of subarrays where [monotonic condition] ≤ k" | B6 / C1 | threshold, not size |
| "longest" but condition can flip back valid | B7 | usually n/a |
| "shortest subarray with OR/AND/XOR ≥ k" | B8 | OR/AND/XOR threshold |
| "number of subarrays with exactly k [things]" | C2 | exact count target |
| "count windows of size k that sum to target" | C3 / A4 | window size |
| "count substrings where two separate counts both must clear k" | C4 | threshold, two counters |
| "count pairs where value relates to value±1" | C5 / D2 | n/a — not a window |
| "pair (i, j)", no adjacency required | D1 | n/a |
| "k closest elements", sorted input | D3 | number of elements to pick |
| "dp[i] depends on previous k dp values" | D4 | width of dependency range |

**One thing worth internalizing:** every Part A and Part B template reduces to the same two
moves — shrink left, extend right — plus one running piece of state (a sum, a product, or a
counter). The exceptions are the deque templates (A8 and dual-deque variants), because "what's
the max" isn't something you can update with a single `+=`/`-=`. Everything in Part D isn't
sliding window at all, even though the phrasing can sound identical.

---

# PART F — LEETCODE VARIANT GALLERY

Real problems mapped onto the templates above, grouped by which template they vary.

## Fixed-Window Variants (Part A)

### A1 variants

**1652. Defuse the Bomb** — circular indexing breaks clean remove/add, and `n <= 100`, so brute
force per index is intentional here (use `% n` instead of true sliding).
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

**1763. Longest Nice Substring** — "nice" is not a property you can shrink-fix from the left
(non-monotonic), so this stays a nested brute enumeration rather than true two-pointer B2.
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

### A2 variants

**1984. Minimum Difference Between Highest and Lowest of K Scores** — sort first, then since
the window is sorted, max/min are just the two endpoints, so no running accumulator is even
needed.
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

**2379. Minimum Recolors to Get K Consecutive Black Blocks** — unmodified shape; `current_sum`
becomes `white_count`, "add/remove nums[i]" becomes "add/remove 1 if char is 'W'".
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

**3364. Minimum Positive Sum Subarray** — run once per window size; the only new idea is
wrapping A2 in an outer loop over every length from `l` to `r`, since the window size itself is
a range, not a single fixed `k`.
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

### A4 variants

**3206. Alternating Groups I** — fixed window of size 3, circular (`% n`); match condition is
"middle differs from both neighbors".
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

### A5 variants

**219. Contains Duplicate II** — instead of a `Counter` tracking exact multiset, we only need
membership, so a `set` replaces the `Counter`. Window size is `k` (not `k+1`).
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

### A6 variants

**1876. Substrings of Size Three with Distinct Characters** — match condition becomes
`len(window_counter) == k` (all distinct) instead of `window_counter == target_counter`.
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

## Variable-Window Variants (Part B)

### B2 variants

**2760. Longest Even Odd Subarray With Threshold** — instead of shrinking left by one step at a
time, a single violation invalidates the *entire* prefix, so `left` jumps straight to `right`
(restart pattern) rather than incrementing.
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

**3090. Maximum Length Substring With Two Occurrences** — identical shape to "longest unique
substring", just change the shrink threshold from `> 1` to `> 2`.
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

### B3 variants

**594. Longest Harmonious Subsequence** — sort first (subsequence, not subarray, so order
doesn't matter). Invariant becomes `nums[right] - nums[left] <= 1` instead of "distinct count
<= K".
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

### B6 variants

**3258. Count Substrings That Satisfy K-Constraint I** — unmodified shape; "invalid" becomes
"both the 0-count AND 1-count exceed k" (two counters instead of one running product). See C4.
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