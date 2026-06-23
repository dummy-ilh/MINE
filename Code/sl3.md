# Sliding Window — Full Template Reference (T1–T14)

All LeetCode Easy + Medium sliding-window problems mapped to templates.
New templates **T12–T14** added to cover patterns not in the original T1–T11.

---

# Templates T1–T7 — Original

## TEMPLATE 1 — Fixed Size Window

**Recognition signals:** "subarray of size k", "window of length k", "k consecutive elements"

**Mental model:** The window NEVER changes size. Every time you add one on the right, you must remove one on the left — they move together, locked at distance `k`.

**Generic pseudocode:**
```
for right in range(n):
    add nums[right] to window
    if right - left + 1 == k:
        record answer
        remove nums[left] from window
        left += 1
```

**Python template:**
```python
def max_subarray_sum_size_k(nums, k):
    current_sum = 0
    for i in range(k):
        current_sum += nums[i]

    max_sum = current_sum

    for i in range(len(nums) - k):
        current_sum -= nums[i]          # trailing element leaves
        current_sum += nums[i + k]      # leading element enters

        if current_sum > max_sum:
            max_sum = current_sum

    return max_sum
```

**Common mistakes:**
* Checking window size with `right - left == k` instead of `right - left + 1 == k` (off-by-one).
* Forgetting to shrink after recording — window grows unbounded.
* Sliding before the window is even full (first `k-1` steps should only expand).

**Complexity:** Time `O(n)`, Space `O(1)` (or `O(k)` if using a frequency map).

**Variants:**
* *Circular extension* — double the array (`arr + arr`) to avoid modulo arithmetic.
* *Multiple sizes* — run a standard T1 pass once per size in range `[l, r]`.
* *Sorted preprocessing* — sort first, then slide; max−min collapses to `window[-1] − window[0]`.

---

## TEMPLATE 2 — Longest Valid Window

**Recognition signals:** "longest substring/subarray such that...", "maximum length with at most...", "longest with no more than K..."

**Mental model:** Expand greedily. The moment the window becomes invalid, shrink **just enough** to make it valid again — never more. Record the max length every time the window is valid.

**Generic pseudocode:**
```
for right in range(n):
    add nums[right] to window
    while window is invalid:
        remove nums[left] from window
        left += 1
    record max(answer, right - left + 1)
```

**Python template:**
```python
def longest_valid_window(s, k):
    left = 0
    freq = {}
    best = 0

    for right in range(len(s)):
        ch = s[right]
        freq[ch] = freq.get(ch, 0) + 1

        while len(freq) > k:
            left_ch = s[left]
            freq[left_ch] -= 1
            if freq[left_ch] == 0:
                del freq[left_ch]
            left += 1

        best = max(best, right - left + 1)

    return best
```

**Common mistakes:**
* Using `if` instead of `while` for shrinking.
* Recording the answer before fixing validity.
* Forgetting to delete a key when its count hits 0.

**Complexity:** Time `O(n)`, Space `O(k)`.

**Variants:**
* *Jump-reset* — when a structural violation is tied to the boundary (not "too much of something"), jump `left = right` instead of incrementally shrinking (e.g. #2760, #978, #1839).

---

## TEMPLATE 3 — Smallest Valid Window

**Recognition signals:** "minimum window", "smallest subarray such that sum >= target", "shortest substring containing all characters"

**Mental model:** Expand until valid, then shrink as aggressively as possible while still valid, recording the minimum length at every valid point.

**Generic pseudocode:**
```
for right in range(n):
    add nums[right] to window
    while window is valid:
        record min(answer, right - left + 1)
        remove nums[left] from window
        left += 1
```

**Python template:**
```python
def smallest_valid_window(nums, target):
    left = 0
    window_sum = 0
    best = float('inf')

    for right in range(len(nums)):
        window_sum += nums[right]

        while window_sum >= target:
            best = min(best, right - left + 1)
            window_sum -= nums[left]
            left += 1

    return best if best != float('inf') else 0
```

**Common mistakes:**
* Shrinking only once instead of `while`.
* Not handling the "no valid window exists" case.

**Complexity:** Time `O(n)`, Space `O(1)` or `O(k)` with a map.

---

## TEMPLATE 4 — Counting Windows

**Recognition signals:** "number of subarrays such that...", "count subarrays with sum equal to k", "how many substrings satisfy..."

**Mental model:** For a valid window `[left, right]`, every sub-window sharing the same `right` and starting anywhere from `left` to `right` is also valid. Count a whole batch at once.

**Generic pseudocode:**
```
for right in range(n):
    add nums[right] to window
    while window is invalid:
        remove nums[left] from window
        left += 1
    answer += right - left + 1
```

**Python template:**
```python
def count_windows_at_most_k(nums, k):
    left = 0
    window_sum = 0
    count = 0

    for right in range(len(nums)):
        window_sum += nums[right]

        while window_sum > k:
            window_sum -= nums[left]
            left += 1

        count += right - left + 1

    return count
```

**Common mistakes:**
* Only works cleanly for **monotonic** conditions.
* Adding `right - left + 1` before shrinking is fully done.

**Complexity:** Time `O(n)`, Space `O(1)`.

---

## TEMPLATE 5 — AtMost(K) − AtMost(K−1)

**Recognition signals:** "exactly K distinct", "exactly K odd numbers", "exactly K of something"

**Mental model:** "Exactly K" is hard to slide directly. "At most K" IS monotonic (T4). Use:

```
Exactly(K)  =  AtMost(K)  −  AtMost(K − 1)
```

**Python template:**
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

**Common mistakes:**
* Forgetting `AtMost(-1)` should return `0`.
* Writing two separate passes with subtly different logic.

**Complexity:** Time `O(n)`, Space `O(1)`.

---

## TEMPLATE 6 — Frequency Matching

**Recognition signals:** "anagram", "permutation", "contains all characters of", "same character frequency as"

**Mental model:** Track a single integer `matched` = how many distinct characters currently have the exact required count in the window.

**Generic pseudocode:**
```
build need map from pattern
matched = 0, required = distinct chars in pattern

for right in range(n):
    add s[right] to window map
    if window[s[right]] == need[s[right]]: matched += 1

    while window is "too big":
        if window[s[left]] == need[s[left]]: matched -= 1
        remove s[left] from window
        left += 1

    if matched == required: record answer
```

**Python template:**
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

**Common mistakes:**
* Comparing full dicts every step instead of using `matched` counter.
* Mutating `need` instead of keeping it as a reference.

**Complexity:** Time `O(n)`, Space `O(distinct characters)`.

---

## TEMPLATE 7 — Monotonic Queue (Deque)

**Recognition signals:** "maximum/minimum of every window of size k", "sliding window maximum"

**Mental model:** Keep a deque of indices with values in decreasing order front-to-back. The front is always the current window's maximum.

**Generic pseudocode:**
```
deque = empty (stores indices)
for right in range(n):
    while deque not empty AND nums[deque.back] <= nums[right]:
        deque.pop_back()
    deque.push_back(right)

    if deque.front <= right - k:
        deque.pop_front()

    if right >= k - 1:
        record nums[deque.front]
```

**Python template:**
```python
from collections import deque

def sliding_window_maximum(nums, k):
    dq = deque()
    result = []

    for right in range(len(nums)):
        while dq and nums[dq[-1]] <= nums[right]:
            dq.pop()
        dq.append(right)

        if dq[0] <= right - k:
            dq.popleft()

        if right >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**Common mistakes:**
* Storing values instead of indices.
* Using `<` instead of `<=` when popping from back.
* Forgetting the `right >= k - 1` guard.

**Complexity:** Time `O(n)` amortized, Space `O(k)`.

**Variant — Dual deque:** Maintain a max-deque AND a min-deque simultaneously when validity depends on `max − min <= limit` (e.g. #1438, #2762, #3578).

---

# Templates T8–T14 — Extended

## TEMPLATE 8 — Frequency-Map Pairing (No Window At All)

**Recognition signals:** problem is about a **subsequence** — order and contiguity don't matter, only counts and how values relate to *other* values (e.g. "value and value+1").

**Mental model:** No `left`/`right` pointer. Build one frequency map of the whole array, then for each distinct value look up a related value's frequency.

**Generic pseudocode:**
```
freq = build_freq_map(nums)
best = 0
for value in freq:
    if related_value(value) in freq:
        best = combine(best, freq[value], freq[related_value(value)])
```

**Python template:**
```python
from collections import Counter

def frequency_pair_best(nums):
    freq = Counter(nums)
    best = 0
    for value in freq:
        if value + 1 in freq:
            best = max(best, freq[value] + freq[value + 1])
    return best
```

**Common mistakes:** Forcing a sliding window onto a problem where contiguity was never required.

**Complexity:** Time `O(n)`, Space `O(n)`.

---

## TEMPLATE 9 — Expand-From-Each-Start (Non-Monotonic Validity)

**Recognition signals:** "longest substring such that..." where the condition can **become valid again after being invalid** as you extend right. Breaks T2 because shrinking from the left doesn't fix a structural mismatch.

**Mental model:** Brute-force every start index. For each one expand right, stop only when this start can *never* recover. Record whenever currently valid.

**Generic pseudocode:**
```
best = 0
for i in range(n):
    state = init_state()
    for j in range(i, n):
        state = update(state, arr[j])
        if cannot_ever_recover(state):
            break
        if is_valid(state):
            best = max(best, j - i + 1)
```

**Common mistakes:** Patching T2 with a `while` shrink loop — it won't converge when validity is structural, not "too much of something."

**Complexity:** Time `O(n²)`, Space `O(1)`.

---

## TEMPLATE 10 — All-Pairs Brute Force (Not A Window Problem)

**Recognition signals:** "pair `(i, j)`" where `i` and `j` are **not required to be adjacent or ordered** — relationship between any two elements.

**Mental model:** No contiguous range to optimize. Just check every pair.

**Generic pseudocode:**
```
best = default
for i in range(n):
    for j in range(n):
        if pair_condition(arr[i], arr[j]):
            best = combine(best, arr[i], arr[j])
```

**Common mistakes:** Seeing "pair" and assuming sliding window applies.

**Complexity:** Time `O(n²)`, Space `O(1)`.

---

## TEMPLATE 11 — Non-Invertible Aggregate Brute Force

**Recognition signals:** smallest/shortest window problems (would normally be T3) where the aggregate is **bitwise OR/AND/XOR** — which cannot be "subtracted back out" when an element leaves.

**Mental model:** T3's shrink step is only valid because addition is invertible. OR is not. Recompute fresh from each start index.

**Generic pseudocode:**
```
best = infinity
for i in range(n):
    agg = identity
    for j in range(i, n):
        agg = combine(agg, arr[j])     # e.g. agg |= arr[j]
        if satisfies(agg):
            best = min(best, j - i + 1)
            break
```

**Common mistakes:** Applying T3's two-pointer shrink to an OR aggregate — silently produces wrong answers.

**Complexity:** Time `O(n²)`, Space `O(1)`.

---

## TEMPLATE 12 — Circular Array Window

**Recognition signals:** array "wraps around", "circular", window can span the end and beginning of array.

**Mental model:** Two approaches:
1. **Double the array** (`arr + arr`) — all windows are now contiguous, no modulo needed. Best when `k` is large or variable.
2. **Modulo indexing** — `arr[(i + offset) % n]`. Best when `k` is small and fixed (e.g. `k = 3`).

**Python template (doubling):**
```python
def circular_window(code, k):
    n = len(code)
    extended = code + code

    current_sum = sum(extended[1:k+1])   # window for index 0
    result = [current_sum]

    for i in range(1, n):
        current_sum -= extended[i]
        current_sum += extended[i + k]
        result.append(current_sum)

    return result
```

**Python template (modulo, k=3):**
```python
def circular_window_small(colors):
    n = len(colors)
    count = 0
    for i in range(n):
        prev = colors[(i - 1) % n]
        curr = colors[i]
        nxt  = colors[(i + 1) % n]
        if prev != curr and curr != nxt:
            count += 1
    return count
```

**Common mistakes:**
* Modulo indexing with variable-length windows gets messy — prefer doubling.
* Forgetting to restrict output to first `n` results when using doubled array.

**Complexity:** Time `O(n)`, Space `O(n)` for doubled array, `O(1)` for modulo approach.

---

## TEMPLATE 13 — Two-Pointer / Sorted + Partition

**Recognition signals:** sorted input (or can be sorted); binary search on the answer; shrink/expand from both ends; "k closest elements"; "maximum window where max−min ≤ X after sorting".

**Mental model:** After sorting, the optimal k-element subset is always a contiguous block. Slide a fixed or variable window over the sorted array. Alternatively, binary-search on the answer and count valid windows with a pointer scan.

**Python template (find k closest to x):**
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
```

**Python template (longest window max−min ≤ 2k after sort):**
```python
def max_beauty(nums, k):
    nums.sort()
    left = 0
    best = 0
    for right in range(len(nums)):
        while nums[right] - nums[left] > 2 * k:
            left += 1
        best = max(best, right - left + 1)
    return best
```

**Common mistakes:**
* Applying sliding window before sorting — only valid because sorted order makes max−min = `arr[right] − arr[left]`.
* Off-by-one in binary search boundaries.

**Complexity:** Time `O(n log n)` (sort dominates), Space `O(1)`.

---

## TEMPLATE 14 — DP + Window Optimization

**Recognition signals:** DP recurrence of the form `dp[i] = f(dp[i-1], ..., dp[i-k])` — a value that depends on a **sliding range** of prior DP values; "probability after at most k steps"; "reachability with jump range".

**Mental model:** The DP fills left to right. At each cell you need a sum (or max/min) of the previous `k` cells — maintain a running window sum/deque in parallel so you don't recompute it each time.

**Python template (sliding sum of last maxPts DP values — LC 837):**
```python
def new21Game(n, k, maxPts):
    if k == 0 or n >= k + maxPts:
        return 1.0

    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    window_sum = 1.0
    result = 0.0

    for i in range(1, n + 1):
        dp[i] = window_sum / maxPts
        if i < k:
            window_sum += dp[i]
        else:
            result += dp[i]
        if i >= maxPts:
            window_sum -= dp[i - maxPts]

    return result
```

**Python template (reachability with jump range — LC 1871):**
```python
def canReach(s, minJump, maxJump):
    n = len(s)
    dp = [False] * n
    dp[0] = True
    window = 0   # count of reachable positions in sliding window

    for i in range(1, n):
        if i >= minJump:
            window += dp[i - minJump]
        if i > maxJump:
            window -= dp[i - maxJump - 1]
        if window > 0 and s[i] == '0':
            dp[i] = True

    return dp[n - 1]
```

**Common mistakes:**
* Recomputing the window sum from scratch each step — turns `O(n)` into `O(n·k)`.
* Off-by-one on when to add/subtract from the running window sum.

**Complexity:** Time `O(n)`, Space `O(n)`.

---

# All Easy Problems

| # | Problem | Template | Variant / Notes |
|---|---------|----------|-----------------|
| 219 | Contains Duplicate II | T1 | Window size k+1; set instead of sum |
| 594 | Longest Harmonious Subsequence | T8 | Subsequence → no window; freq-map pair value+1 |
| 643 | Maximum Average Subarray I | T1 | Canonical T1 |
| 1176 | Diet Plan Performance | T1 | T1 with 3-way comparison instead of equality |
| 1652 | Defuse the Bomb | T1 + T12 | Circular; double array to avoid modulo |
| 1763 | Longest Nice Substring | T9 | Non-monotonic validity; expand-from-each-start |
| 1876 | Substrings of Size Three with Distinct Characters | T1 | k=3; set size check |
| 1984 | Minimum Difference Between Highest and Lowest of K Scores | T1 + T13 | Sort + T1; max−min = last−first in sorted window |
| 2269 | Find the K-Beauty of a Number | T1 | T1 over digit string; re-parse int each step |
| 2379 | Minimum Recolors to Get K Consecutive Black Blocks | T1 | T1; track white count in window |
| 2760 | Longest Even Odd Subarray With Threshold | T2 | Jump-reset variant |
| 2932 | Maximum Strong Pair XOR I | T10 | All-pairs; no adjacency required |
| 3090 | Maximum Length Substring With Two Occurrences | T2 | freq cap = 2 per char |
| 3095 | Shortest Subarray With OR at Least K I | T11 | OR non-invertible; brute force from each start |
| 3206 | Alternating Groups I | T1 + T12 | Circular k=3; modulo indexing |
| 3258 | Count Substrings That Satisfy K-Constraint I | T4 | zeros≤k OR ones≤k |
| 3318 | Find X-Sum of All K-Long Subarrays I | T1 | Counter window state; top-x recomputed each step |
| 3364 | Minimum Positive Sum Subarray | T1 | T1 across multiple window sizes l…r |
| 3411 | Maximum Subarray With Equal Products | T9 | Non-monotonic (prime collision); expand-from-each-start |

---

# All Medium Problems

| # | Problem | Template | Variant / Notes |
|---|---------|----------|-----------------|
| 3 | Longest Substring Without Repeating Characters | T2 | Classic T2; invalid = any char count > 1 |
| 159 | Longest Substring with At Most Two Distinct Characters | T2 | k=2 distinct chars cap |
| 187 | Repeated DNA Sequences | T1 | Fixed window k=10; set tracks seen substrings |
| 209 | Minimum Size Subarray Sum | T3 | Canonical T3 |
| 340 | Longest Substring with At Most K Distinct Characters | T2 | Parametric version of #159 |
| 395 | Longest Substring with At Least K Repeating Characters | T9 | Non-monotonic; divide & conquer or T9 |
| 413 | Arithmetic Slices | T4 | Valid window = 3+ consecutive arith diff; T4 batch count |
| 424 | Longest Repeating Character Replacement | T2 | Invalid = window_len − max_freq > k |
| 438 | Find All Anagrams in a String | T6 | T6; record all start indices |
| 487 | Max Consecutive Ones II | T2 | At most 1 zero; T2 with zero-count cap |
| 567 | Permutation in String | T6 | T6 fixed-size permutation check |
| 658 | Find K Closest Elements | T13 | Sort + binary search or T1 on sorted array |
| 713 | Subarray Product Less Than K | T4 | T4 batch count; shrink when product ≥ k |
| 718 | Maximum Length of Repeated Subarray | T9 | Common subarray; DP or T9 nested |
| 837 | New 21 Game | T14 | DP + sliding window sum over last maxPts values |
| 904 | Fruit Into Baskets | T2 | At most 2 distinct fruit types |
| 930 | Binary Subarrays With Sum | T5 | Exact sum=goal → AtMost(goal)−AtMost(goal−1) |
| 978 | Longest Turbulent Subarray | T2 | Jump-reset variant; reset left on parity break |
| 1004 | Max Consecutive Ones III | T2 | At most k zeros |
| 1016 | Binary String With Substrings Representing 1 To N | T1 | Fixed-length windows; set membership check |
| 1031 | Maximum Sum of Two Non-Overlapping Subarrays | T1 | Two T1 windows; track max of first before second |
| 1040 | Moving Stones Until Consecutive II | T13 | Sorted positions; T2 longest window fitting in spread |
| 1052 | Grumpy Bookstore Owner | T1 | Fixed window size minutes; bonus customer count |
| 1100 | Find K-Length Substrings With No Repeated Characters | T1 | T1 fixed k; validity = all distinct |
| 1151 | Minimum Swaps to Group All 1s Together | T1 | Window = total count of 1s; count 0s inside = swaps |
| 1156 | Swap For Longest Repeated Character Substring | T2 | At most 1 non-dominant char allowed |
| 1208 | Get Equal Substrings Within Budget | T2 | Cost = |s[i]−t[i]|; at most maxCost budget |
| 1234 | Replace the Substring for Balanced String | T3 | Smallest window that makes the outside balanced |
| 1248 | Count Number of Nice Subarrays | T5 | Exactly k odd numbers → T5 AtMost decomposition |
| 1297 | Maximum Number of Occurrences of a Substring | T1 | Greedy: optimal length = minSize; T1 + hash set |
| 1343 | Number of Sub-arrays of Size K and Average ≥ Threshold | T1 | Record when sum/k >= threshold |
| 1358 | Number of Substrings Containing All Three Characters | T4 | Shrink until no longer has a,b,c; batch count |
| 1423 | Maximum Points You Can Obtain from Cards | T1 | Min-sum middle window of size n−k; total − min_middle |
| 1438 | Longest Continuous Subarray With Absolute Diff ≤ Limit | T7 | Dual deque (max-deque + min-deque) |
| 1456 | Maximum Number of Vowels in a Substring of Given Length | T1 | Fixed window; count vowels |
| 1477 | Find Two Non-overlapping Sub-arrays Each With Target Sum | T3 | Two T3 passes (left→right + right→left); combine |
| 1493 | Longest Subarray of 1s After Deleting One Element | T2 | At most 1 zero (then remove it); zero-cap=1 |
| 1658 | Minimum Operations to Reduce X to Zero | T3 | Longest subarray summing to total−x; T2/T3 flip |
| 1695 | Maximum Erasure Value | T2 | Longest all-distinct subarray; T2 max-sum variant |
| 1838 | Frequency of the Most Frequent Element | T2 | Valid when (window_size × max_val − window_sum) ≤ k |
| 1839 | Longest Substring Of All Vowels in Order | T2 | Jump-reset; ordered vowels + consecutive condition |
| 1852 | Distinct Numbers in Each Subarray | T1 | Fixed window k; distinct count with freq map |
| 1871 | Jump Game VII | T14 | DP reachability; sliding window prefix sum of reachable cells |
| 1888 | Minimum Number of Flips to Make Binary String Alternating | T1 + T12 | Circular T1; double string; fixed window n with sliding flip-count |
| 1918 | Kth Smallest Subarray Sum | T3 | Binary search on answer + T3 counting windows ≤ target |
| 2024 | Maximize the Confusion of an Exam | T2 | Run twice (cap T flips, cap F flips); take max |
| 2067 | Number of Equal Count Substrings | T1 | Fixed window k = count×d for each divisor d of 26 |
| 2090 | K Radius Subarray Averages | T1 | Fixed size 2k+1; output −1 when window out of bounds |
| 2107 | Number of Unique Flavors After Sharing K Candies | T1 | Fixed window k (given away); maximize distinct outside |
| 2110 | Number of Smooth Descent Periods of a Stock | T4 | T4 batch count; reset on non-descent |
| 2134 | Minimum Swaps to Group All 1s Together II | T1 + T12 | Circular T1; double array; count 0s in window of size total_ones |
| 2260 | Minimum Consecutive Cards to Pick Up | T3 | Smallest window containing a duplicate |
| 2271 | Maximum White Tiles Covered by a Carpet | T13 | Sorted intervals; binary search + prefix sums for coverage |
| 2401 | Longest Nice Subarray | T2 | Invalid = any two elements share a bit (AND≠0); shrink until clear |
| 2411 | Smallest Subarrays With Maximum Bitwise OR | T11 | OR non-invertible; for each i find rightmost bit-contributing index |
| 2461 | Maximum Sum of Distinct Subarrays With Length K | T1 | Fixed window k; validity = all distinct (freq map) |
| 2516 | Take K of Each Character From Left and Right | T3 | Min total = flip: max middle window leaving k of each outside |
| 2537 | Count the Number of Good Subarrays | T4 | pair count += freq[nums[right]] before inserting; shrink when pairs < k |
| 2555 | Maximize Win From Two Segments | T2 | Two-pass T2; dp[i] = best segment ending ≤ i; combine two non-overlapping |
| 2653 | Sliding Subarray Beauty | T7 | Sliding window k; kth smallest negative → dual deque / sorted structure |
| 2730 | Find the Longest Semi-Repetitive Substring | T2 | At most 1 adjacent-duplicate pair |
| 2743 | Count Substrings Without Repeating Character | T4 | T4 batch count; shrink on repeated char |
| 2747 | Count Zero Request Servers | T1 | Offline; sort queries; slide server window over time range |
| 2762 | Continuous Subarrays | T7 | Dual deque (max+min); valid = max−min ≤ 2; T4 batch count |
| 2779 | Maximum Beauty of an Array After Applying Operation | T13 | Sort; T2 longest window where max−min ≤ 2k |
| 2799 | Count Complete Subarrays in an Array | T4 | Valid = distinct count == total distinct; T4 |
| 2831 | Find the Longest Equal Subarray | T2 | Invalid = (window_len − max_freq) > k deletions |
| 2841 | Maximum Sum of Almost Unique Subarray | T1 | Fixed k; valid = at least m distinct; sum when valid |
| 2875 | Minimum Size Subarray in Infinite Array | T3 + T12 | T3 on doubled (circular) array; min window sum ≡ target mod total |
| 2904 | Shortest and Lexicographically Smallest Beautiful String | T3 | Smallest window with exactly k ones; then lex-min tie-break |
| 2958 | Length of Longest Subarray With at Most K Frequency | T2 | freq cap = k per element |
| 2962 | Count Subarrays Where Max Element Appears at Least K Times | T4 | Track max count; when count ≥ k batch-add left positions |
| 2981 | Find Longest Special Substring That Occurs Thrice I | T1 | T1 over all lengths; count occurrences; check ≥ 3 |
| 2982 | Find Longest Special Substring That Occurs Thrice II | T1 | Same as 2981; bucket by char+length for large n |
| 3023 | Find Pattern in Infinite Stream I | T6 | T6 frequency matching on stream; KMP or T6 |
| 3097 | Shortest Subarray With OR at Least K II | T11 | T11 + bit-level tracking of which elements contribute each bit |
| 3135 | Equalize Strings by Adding or Removing Characters at Ends | T6 | Longest common substring via T6 / sliding hash |
| 3191 | Minimum Operations to Make Binary Array Elements Equal to One I | T1 | Fixed k=3; flip window when leading element is 0 |
| 3208 | Alternating Groups II | T1 + T12 | Circular T1 k=arbitrary; double array |
| 3254 | Find the Power of K-Size Subarrays I | T1 | Fixed k; valid = consecutive sorted with diff 1 each step |
| 3255 | Find the Power of K-Size Subarrays II | T1 | Same as 3254; larger n; same T1 approach |
| 3297 | Count Substrings That Can Be Rearranged to Contain a String I | T3 | Smallest window containing enough chars of pattern |
| 3305 | Count of Substrings Containing Every Vowel and K Consonants I | T5 | Exactly k consonants → T5; also need all 5 vowels |
| 3306 | Count of Substrings Containing Every Vowel and K Consonants II | T5 | Same as 3305; larger n |
| 3323 | Minimize Connected Groups by Inserting Interval | T1 | Sort intervals; fixed window spanning ≤ len; T1 |
| 3325 | Count Substrings With K-Frequency Characters I | T4 | Valid = some char has freq ≥ k; batch count |
| 3346 | Maximum Frequency of an Element After Performing Operations I | T2 | Sort; valid when (target×len − sum) ≤ k ops |
| 3413 | Maximum Coins From K Consecutive Bags | T1 | Fixed window k on sorted bag positions; prefix sums |
| 3422 | Minimum Operations to Make Subarray Elements Equal | T7 | Sliding median window; two heaps or T7 |
| 3439 | Reschedule Meetings for Maximum Free Time I | T1 | Fixed window k on gap array; maximize sum of k gaps |
| 3578 | Count Partitions With Max-Min Difference at Most K | T4 + T7 | Two monotonic deques for max/min; T4 batch count |
| 3589 | Count Prime-Gap Balanced Subarrays | T4 + T7 | Valid = max prime gap ≤ k in window; deque for max gap |
| 3634 | Minimum Removals to Balance Array | T2 | Track imbalance; shrink when condition broken |
| 3641 | Longest Semi-Repeating Subarray | T2 | At most 1 adjacent-pair repetition |
| 3652 | Best Time to Buy and Sell Stock using Strategy | T1 | Fixed window + prefix max; track max before window |
| 3672 | Sum of Weighted Modes in Subarrays | T1 | Fixed window; recompute mode-sum each step with freq map |
| 3679 | Minimum Discards to Balance Inventory | T2 | Shrink when imbalance exceeds threshold |
| 3694 | Distinct Points Reachable After Substring Removal | T3 | Find valid removal windows; count distinct remaining |
| 3795 | Minimum Subarray Length With Distinct Sum At Least K | T3 | Sum of distinct elements ≥ k; freq map tracks distinct sum |
| 3851 | Maximum Requests Without Violating the Limit | T2 | Shrink when resource count exceeds limit |

---

# Summary

| Template | Name | # Problems |
|----------|------|-----------|
| T1 | Fixed Size Window | ~40 |
| T2 | Longest Valid Window | ~30 |
| T3 | Smallest Valid Window | ~12 |
| T4 | Counting Windows | ~14 |
| T5 | AtMost(K) − AtMost(K−1) | 5 |
| T6 | Frequency Matching | 5 |
| T7 | Monotonic Deque | 6 |
| T8 | Frequency-Map Pairing (new) | 1 |
| T9 | Expand-From-Each-Start (new) | 4 |
| T10 | All-Pairs Brute Force (new) | 1 |
| T11 | Non-Invertible Aggregate (new) | 3 |
| T12 | Circular Array Window (new) | 6 |
| T13 | Two-Pointer / Sorted (new) | 4 |
| T14 | DP + Window Optimization (new) | 2 |

**T1 + T2 alone cover ~55% of all problems.**
T3, T4, T5, T6 cover most of the rest.
T7–T14 handle the edge cases.
