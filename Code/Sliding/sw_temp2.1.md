# Sliding Window — Master Template Reference

Naming key: every tag tells you the *shape*, and every template states explicitly what `k`
means in that template — because `k` is overloaded across this whole topic (window size?
max distinct count? exact target count? an OR-threshold? none of the above?). Don't assume;
check the callout.

```
FIX-*      fixed-size window         (k = the window size, always)
LONGEST-*  variable window, maximize length
SHORTEST-*  variable window, minimize length
EXACT-*    two-pointer exact match   (positive numbers only)
COUNT-*    counting windows         (k is usually a THRESHOLD, not a size — common confusion)
NOWIN-*    looks like sliding window, isn't (no left/right pointers)
```

---

# SECTION 1 — FIXED-SIZE WINDOW
**Recognition signal for the whole section:** the problem hands you a size and it never moves —
"window of size k", "k consecutive elements", "every substring of length k".

### FIX-1 — Brute force (baseline only, O(n·k), don't ship this)
```python
def max_subarray_sum(nums, k):
    max_sum = float("-inf")
    for i in range(len(nums) - k + 1):
        current_sum = sum(nums[i:i + k])
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

### FIX-2 — Running Sum (the real fixed-window template)
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

### FIX-3 — Running Product
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

### FIX-4 — Count Matches (sum equals an exact target, fixed size)
**`k` means:** the window size. `target` is the separate value you're matching against —
don't conflate the two.
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

### FIX-5 — Anagram Check, Boolean (full Counter comparison)
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

### FIX-6 — Anagram Check, Count All Matches (full Counter comparison)
Same shape as FIX-5, just doesn't early-return.
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

### FIX-7 — Anagram/Permutation, Fast Matched-Counter (returns all start indices)
**Why this exists alongside FIX-5/6:** comparing two Counters every step is O(alphabet) per
step. This version tracks a single integer `matched` — how many distinct characters currently
have the *exact* required count — so each step is O(1). Use this one once you've already shown
FIX-5/6 work and want the efficient version.
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

### FIX-8 — Max/Min Per Window — Monotonic Deque
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

### FIX-9 — Circular Fixed Window
**`k` means:** the window size. The array additionally **wraps around** — "circular", "last
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

### FIX-10 — Generic skeleton
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

# SECTION 2 — VARIABLE WINDOW: LONGEST
**Recognition signal for the whole section:** "longest substring/subarray such that...",
"maximum length with at most...". Expand greedily; when invalid, shrink **just enough** to
become valid again, then record.

### LONGEST-1 — No Repeats
**No `k` here** — the condition is structural ("no duplicate"), not parametrized.
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

### LONGEST-2 — At Most K Distinct
**`k` means:** the maximum number of *distinct* elements/types allowed in the window —
**not** the window size. ("Fruit into baskets" is this with k=2.)
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
(not "too much of something"), jump `left = right` instead of incrementally shrinking
(e.g. longest turbulent subarray, longest run of ordered vowels).

### LONGEST-3 — Non-Monotonic Validity (Expand-From-Each-Start)
**No `k`** in general — this is for conditions that can **become valid again after being
invalid** as you extend right (e.g. a structural/equality condition, not a "too much of X"
condition). This breaks LONGEST-1/2 because shrinking from the left doesn't fix a structural
mismatch — there's no guarantee shrinking helps at all.
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
**Common mistake:** patching LONGEST-1/2 with a `while`-shrink loop here — it won't converge,
because the thing that broke validity isn't "too much of one element," it's structural.
**Complexity:** O(n²), Space O(1) — this is the brute-force fallback for this family.

### LONGEST-4 — Generic skeleton
```python
left = 0
best = 0
for right in range(len(data)):
    add(data[right])
    while invalid():
        remove(data[left])
        left += 1
    best = max(best, right - left + 1)
```

---

# SECTION 3 — VARIABLE WINDOW: SHORTEST
**Recognition signal:** "minimum size subarray such that...", "shortest substring containing
all of...". Expand until valid, then shrink as aggressively as possible *every time it's
valid* — this is the one shape difference from LONGEST: you shrink on every valid hit, not
just to fix invalidity.

### SHORTEST-1 — Numeric Sum ≥ Target
**`k` is not used here; `target` means:** the threshold the running sum must reach.
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

### SHORTEST-2 — Frequency Match (Minimum Window Substring)
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

### SHORTEST-3 — Non-Invertible Aggregate (OR / AND / XOR) Brute Restart
**Why this exists:** SHORTEST-1's shrink step only works because addition is invertible
(`-=` undoes `+=`). Bitwise OR (and AND) are **not** invertible — once a bit is OR'd in, you
can't "subtract" an element back out. Applying the two-pointer shrink here silently gives
wrong answers. Recompute fresh from each start index instead.
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
**Complexity:** O(n²) in the naive form (can be optimized to O(n·32) by tracking, per bit
position, the nearest index contributing that bit — out of scope for this template).

### SHORTEST-4 — Generic skeleton
```python
left = 0
best = float("inf")
for right in range(len(data)):
    add(data[right])
    while valid():
        best = min(best, right - left + 1)
        remove(data[left])
        left += 1
```

---

# SECTION 4 — EXACT TARGET (two pointer, positive numbers only)
**Recognition signal:** "find a subarray that sums to exactly target" — works *only* because
all numbers are positive, so adding more elements only ever increases the sum (monotonic).

### EXACT-1 — Exact Sum, Return Indices
**`target`** is the exact sum to hit; no `k`.
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

---

# SECTION 5 — COUNTING WINDOWS
**This is its own shape, not a flavor of LONGEST/SHORTEST.** The defining trick: once a window
ending at `right` is valid (shrunk back from `left`), *every* window ending at `right` and
starting anywhere from `left` to `right` is also valid — so you count `right - left + 1` windows
in one shot instead of counting one at a time.

**The biggest source of confusion:** in this section, `k` is almost always a **threshold**
(e.g. "product < k", "at most k distinct"), not a window size. Read each callout.

### COUNT-1 — At Most K (monotonic condition)
**`k` means:** the threshold the running aggregate must stay under/at-most.
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
**Only works for monotonic conditions** (adding an element can only make things "more
invalid," never less).

### COUNT-2 — Exactly K = AtMost(K) − AtMost(K−1)
**`k` means:** the exact count you want (e.g. exactly k odd numbers, exactly k distinct).
**Why you basically never write "exactly K" directly:** "exactly K" isn't monotonic on its
own (adding an element can make a window go from valid to invalid to valid again), but
"at most K" always is — so build "exactly K" out of two "at most K" calls.
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

### COUNT-3 — Fixed-Size Exact Match
This is FIX-4 above — cross-referenced here because "count windows of size k that sum to
target" is a counting problem, just one where the window size is fixed rather than the
threshold being on an aggregate. Don't confuse it with COUNT-1/2, where the window size
itself is what's variable.

### COUNT-4 — No-Window Pair Counting
**Recognition:** the count you need depends on **pairs of values across the whole array**,
not on contiguous ranges at all — e.g. "count pairs where value and value+1 both appear."
There's no `left`/`right` pointer here; it only looks like a counting-window problem.
See NOWIN-2 below — same template, listed there since it isn't a window at all.

---

# SECTION 6 — NOT ACTUALLY SLIDING WINDOW (commonly mistaken for it)

### NOWIN-1 — All-Pairs Brute Force
**Recognition:** "pair (i, j)" where i and j are **not required to be adjacent or ordered** —
any two elements can form a pair. There's no contiguous range to slide over.
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

### NOWIN-2 — Frequency-Map Pairing (No Window At All)
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

### NOWIN-3 — Two-Pointer on Sorted Array / Binary Search on Answer
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

### NOWIN-4 — DP With a Sliding Running Sum
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
O(n·k)) — maintain it as a running value alongside the DP fill, same as any other window.

---

# Quick Recognition Table

| Phrasing in the problem | Template | What `k` actually means |
|---|---|---|
| "window of size k" | FIX-2 / FIX-3 / FIX-4 | window size |
| "anagram of" / "permutation of" (fixed size) | FIX-5 / FIX-6 / FIX-7 | len(pattern) |
| "max/min of every window of size k" | FIX-8 | window size |
| "circular array", "wraps around" | FIX-9 | window size (array also wraps) |
| "longest ... without repeating" | LONGEST-1 | n/a |
| "longest ... at most k distinct" | LONGEST-2 | max distinct count allowed |
| "longest" but condition can flip back valid | LONGEST-3 | usually n/a |
| "shortest / minimum size subarray, sum ≥ target" | SHORTEST-1 | n/a (target is the threshold) |
| "smallest substring containing all of..." | SHORTEST-2 | n/a (pattern defines requirement) |
| "shortest subarray with OR/AND/XOR ≥ k" | SHORTEST-3 | OR/AND/XOR threshold |
| "exact sum, positive numbers, return range" | EXACT-1 | n/a (target is the sum) |
| "number of subarrays where [monotonic condition] ≤ k" | COUNT-1 | threshold, not size |
| "number of subarrays with exactly k [things]" | COUNT-2 | exact count target |
| "count windows of size k that sum to target" | COUNT-3 / FIX-4 | window size |
| "count pairs where value relates to value±1" | COUNT-4 / NOWIN-2 | n/a — not a window |
| "pair (i, j)", no adjacency required | NOWIN-1 | n/a |
| "k closest elements", sorted input | NOWIN-3 | number of elements to pick |
| "dp[i] depends on previous k dp values" | NOWIN-4 | width of dependency range |

**One thing worth internalizing across all of this:** every `FIX-*`, `LONGEST-*`, `SHORTEST-*`,
and `EXACT-*` template reduces to the same two moves — shrink left, extend right — plus one
running piece of state (a sum, a product, or a counter). The exceptions are the deque
templates (FIX-8, and dual-deque variants), because "what's the max" isn't something you can
update with a single `+=`/`-=`. Everything in Section 6 isn't sliding window at all, even
though the problem phrasing can sound identical — recognizing *that* distinction is usually
worth more than knowing any individual template by heart.