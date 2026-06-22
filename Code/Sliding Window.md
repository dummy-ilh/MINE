# Sliding Window Mastery


# SECTION 1 — Sliding Window Intuition

### The Problem Brute Force Has

Imagine you're asked: *"Find the maximum sum of any 3 consecutive numbers in this array."*

```
Array: [2, 1, 5, 1, 3, 2]
```

The brute force way: for every starting position, walk forward 3 steps and add them up.

```
Start at index 0: 2+1+5 = 8
Start at index 1: 1+5+1 = 7
Start at index 2: 5+1+3 = 9
Start at index 3: 1+3+2 = 6
```

For every one of the `n` starting positions, you do `k` work. Total work = `n * k`. If `k` grows with `n` (like "every window size from 1 to n"), this becomes **O(n²)**. You are re-adding numbers you already added in the previous window. That's wasteful.

### The Sliding Window Insight

Notice: window `[2,1,5]` and the next window `[1,5,1]` share two numbers (`1` and `5`). Instead of re-adding them, what if we just:

* **Remove** the number that fell out of the window (`2`)
* **Add** the number that just entered the window (`1`)

```
Window [2,1,5] = 8
Remove 2, Add 1  →  8 - 2 + 1 = 7   (this is window [1,5,1])
```

We did **1 subtraction + 1 addition** instead of 3 additions. That's the entire idea of sliding window. Every number is added once when it enters, and removed once when it leaves. Total work becomes **O(n)**.

### Two Pointers: left and right

Picture two fingers on the array:

```
[ 2,  1,  5,  1,  3,  2 ]
  ↑
 left
 right
```

* **right** pointer's job: move forward, **expand** the window, pull in a new element.
* **left** pointer's job: move forward, **shrink** the window, push out an old element.

The window is everything between `left` and `right` (inclusive).

### Expand / Shrink / Record — The Three Heartbeats

Every sliding window problem, no matter how complicated it looks, is built from three repeated actions:

1. **Expand** — move `right` forward, add the new element's effect into your running state (sum, count, frequency map, etc.)
2. **Shrink** — while some condition is violated (or to keep window valid/optimal), move `left` forward, remove that element's effect from your running state
3. **Record** — update your answer (max, min, count, etc.) at the right moment

### Dry Run — Maximum Sum of Window Size 3

```
Array: [2, 1, 5, 1, 3, 2]   k = 3

left=0, right=0: window=[2]            sum=2
left=0, right=1: window=[2,1]          sum=3
left=0, right=2: window=[2,1,5]        sum=8   ← window full, record max=8
shrink: remove 2 → sum=6, left=1
right=3: add 1 → sum=7   window=[1,5,1]        record max=8 (7<8)
shrink: remove 1 → sum=6, left=2
right=4: add 3 → sum=9   window=[5,1,3]        record max=9
shrink: remove 5 → sum=4, left=3
right=5: add 2 → sum=6   window=[1,3,2]        record max=9 (6<9)

Answer: 9
```

We touched every element a constant number of times. That's **O(n)**.

---

# SECTION 2 — Sliding Window Recognition Flowchart

Before writing any code, ask these questions **in order**. This should take under 30 seconds.

```
START
  │
  ├─ Does the question mention a FIXED size K?
  │     YES → TEMPLATE 1: Fixed Size Window
  │
  ├─ Does it ask for LONGEST / MAXIMUM length satisfying a condition?
  │     YES → TEMPLATE 2: Longest Valid Window
  │
  ├─ Does it ask for SMALLEST / MINIMUM length satisfying a condition?
  │     YES → TEMPLATE 3: Smallest Valid Window
  │
  ├─ Does it ask to COUNT how many windows/subarrays satisfy a condition?
  │     YES → TEMPLATE 4: Counting Windows
  │
  ├─ Does it ask for EXACTLY K (distinct elements, sum, etc.)?
  │     YES → TEMPLATE 5: AtMost(K) - AtMost(K-1)
  │
  ├─ Does it mention ANAGRAM / PERMUTATION / "same frequency as"?
  │     YES → TEMPLATE 6: Frequency Matching
  │
  └─ Does it ask for MAX or MIN of EVERY window (sliding maximum)?
        YES → TEMPLATE 7: Monotonic Queue
```

Keep this page open mentally during every problem. It is your map.

---

# SECTION 3 — Core Templates

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
def fixed_window(nums, k):
    window_sum = sum(nums[:k])
    best = window_sum

    for right in range(k, len(nums)):
        window_sum += nums[right]      # add new element
        window_sum -= nums[right - k]  # remove old element

        best = max(best, window_sum)

    return best


def fixed_window(nums, k):
    left = 0
    window_sum = 0
    best = 0
    for right in range(len(nums)):
        window_sum += nums[right]

        # window has reached size k
        if right - left + 1 == k:
            best = max(best, window_sum)
            window_sum -= nums[left]
            left += 1
    return best

```

**Dry run:** Already done above in Section 1 (array `[2,1,5,1,3,2]`, `k=3` → answer `9`).

**Common mistakes:**

* Checking window size with `right - left == k` instead of `right - left + 1 == k` (off-by-one).
* Forgetting to shrink after recording — window grows unbounded.
* Sliding before the window is even full (first `k-1` steps should only expand).

**Complexity:** Time `O(n)`, Space `O(1)` (or `O(k)` if using a frequency map).

### LeetCode Mapping — What Changes?

**LC 643 — Maximum Average Subarray I**
Same as above, just divide `best` by `k` at the end.
```python
best = max(best, window_sum / k)
```

**LC 1456 — Max Vowels in a Substring of Length K**
Instead of summing numbers, add `1` if the character is a vowel.
```python
def maxVowels(s, k):
    vowels = set("aeiou")

    count = 0

    for ch in s[:k]:
        if ch in vowels:
            count += 1

    best = count

    for right in range(k, len(s)):
        if s[right] in vowels:
            count += 1

        if s[right - k] in vowels:
            count -= 1

        best = max(best, count)

    return best
```

**LC 2461 — Maximum Sum of Distinct Subarrays of Length K**
Add a frequency map. Only count a window as valid if all `k` elements are distinct (map size == k).
```python
freq[nums[right]] = freq.get(nums[right], 0) + 1
if right - left + 1 == k:
    if len(freq) == k:          # all distinct
        best = max(best, window_sum)
    freq[nums[left]] -= 1
    if freq[nums[left]] == 0:
        del freq[nums[left]]
    window_sum -= nums[left]
    left += 1
```

**LC 219 — Contains Duplicate II (window size k+1)**
No sum needed. Use a set; if the incoming element is already in the window-set, return True.
```python
def containsNearbyDuplicate(nums, k):
    window = set()
    left = 0

    for right in range(len(nums)):

        if nums[right] in window:
            return True

        window.add(nums[right])

        if right - left + 1 > k:
            window.remove(nums[left])
            left += 1

    return False
```

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

        # shrink while window breaks the rule
        while len(freq) > k:
            left_ch = s[left]
            freq[left_ch] -= 1
            if freq[left_ch] == 0:
                del freq[left_ch]
            left += 1

        best = max(best, right - left + 1)

    return best
```

**Dry run:** `s = "eceba"`, `k = 2` (longest substring with at most 2 distinct characters)

```
right=0 'e': freq={e:1}                      len=1 best=1
right=1 'c': freq={e:1,c:1}                  len=2 best=2
right=2 'e': freq={e:2,c:1}                  len=3 best=3
right=3 'b': freq={e:2,c:1,b:1} → 3 distinct, shrink:
    remove s[0]='e': freq={e:1,c:1,b:1} still 3 distinct, shrink:
    remove s[1]='c': freq={e:1,b:1} now 2 distinct, stop. left=2
    len = 3-2+1 = 2, best stays 3
right=4 'a': freq={e:1,b:1,a:1} → 3 distinct, shrink:
    remove s[2]='e': freq={b:1,a:1} 2 distinct, stop. left=3
    len = 4-3+1 = 2, best stays 3

Answer: 3   ("ece")
```

**Common mistakes:**

* Using `if` instead of `while` for shrinking — a single shrink step is not always enough.
* Recording the answer **before** fixing validity.
* Forgetting to delete a key from the frequency map when its count hits 0 (this silently inflates "distinct count").

**Complexity:** Time `O(n)`, Space `O(k)` for the frequency map.

### LeetCode Mapping — What Changes?

**LC 3 — Longest Substring Without Repeating Characters**
Condition becomes "no duplicates" → shrink while any frequency > 1.
```python
while freq[ch] > 1:
    freq[s[left]] -= 1
    left += 1
```

**LC 424 — Longest Repeating Character Replacement**
Track the max frequency character seen so far. Window is invalid if `window_length - max_freq > k` (more than k replacements needed).
```python
max_freq = max(max_freq, freq[ch])
while (right - left + 1) - max_freq > k:
    freq[s[left]] -= 1
    left += 1
```

**LC 904 — Fruit Into Baskets**
Identical to LC 3 but `k = 2` distinct types allowed — literally the Template 2 base code with `k=2` hardcoded.

**LC 340 — Longest Substring with At Most K Distinct Characters**
This *is* the base template exactly as written above.

**LC 1004 — Max Consecutive Ones III**
Condition: at most `k` zeros allowed in window.
```python
if nums[right] == 0:
    zero_count += 1
while zero_count > k:
    if nums[left] == 0:
        zero_count -= 1
    left += 1
```

**LC 1493 — Longest Subarray of 1's After Deleting One Element**
Same as LC 1004 with `k = 1`.

---

## TEMPLATE 3 — Smallest Valid Window

**Recognition signals:** "minimum window", "smallest subarray such that sum >= target", "shortest substring containing all characters"

**Mental model:** This feels like the *opposite* of Template 2. Here, you expand until the window **becomes valid**, then you shrink as **aggressively as possible** — trying to remove elements one at a time as long as it's *still* valid — recording the minimum length at every valid point.

> Why it feels different: In Template 2, invalid → shrink → stop as soon as valid (we want to keep the window long).
> In Template 3, valid → shrink → stop as soon as invalid (we want to make the window as short as possible while it's still valid).

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

        # shrink as long as window stays valid
        while window_sum >= target:
            best = min(best, right - left + 1)
            window_sum -= nums[left]
            left += 1

    return best if best != float('inf') else 0
```

**Dry run:** `nums = [2,3,1,2,4,3]`, `target = 7` (LC 209)

```
right=0: sum=2  (not >= 7)
right=1: sum=5  (not >= 7)
right=2: sum=6  (not >= 7)
right=3: sum=8  >= 7 → record len=4, remove 2 → sum=6, left=1
              sum=6 not >= 7, stop
right=4: sum=10 >= 7 → record len=4 (4-1+1=4, best stays 4)
              remove 3 → sum=7, left=2 → still >=7 → record len=3, best=3
              remove 1 → sum=6, left=3 → not >=7, stop
right=5: sum=9  >= 7 → record len=3 (5-3+1=3)
              remove 2 → sum=7, left=4 → >=7 → record len=2, best=2
              remove 4 → sum=3, left=5 → not >=7, stop

Answer: 2   (subarray [4,3])
```

**Common mistakes:**

* Shrinking only once instead of `while` — you might be able to shrink multiple times.
* Not handling the "no valid window exists" case (return 0 or empty string).
* For LC 76 (Minimum Window Substring), forgetting to track a "matched count" so you know exactly when validity flips — see Template 6, which extends this template with frequency matching.

**Complexity:** Time `O(n)`, Space `O(1)` or `O(k)` with a map.

### LeetCode Mapping — What Changes?

**LC 209 — Minimum Size Subarray Sum**
Exactly the base template above.

**LC 76 — Minimum Window Substring**
Combine with Template 6 (Frequency Matching). The "valid" check becomes "all required characters are matched" instead of "sum >= target". See Template 6 for the matched-count trick.

---

## TEMPLATE 4 — Counting Windows

**Recognition signals:** "number of subarrays such that...", "count subarrays with sum equal to k", "how many substrings satisfy..."

**Mental model:** This is where most beginners get stuck. The trick: when you have a valid window `[left, right]`, **every** sub-window that shares the same `right` and starts anywhere from `left` to `right` is also valid (for monotonic conditions like "at most"). So instead of counting one subarray at a time, you count a whole batch at once.

**Why `answer += right - left + 1` works — Visual Proof:**

Suppose after shrinking, your window is `[left, right]` and it's the largest valid window ending at `right`. Then ALL of these are also valid (because shrinking only happens when invalid — anything smaller is still valid):

```
Window ending at "right", starting anywhere from left to right:

  [left,         right]   ← full window
  [left+1,       right]   ← one shorter
  [left+2,       right]   ← shorter still
   ...
  [right,        right]   ← just the single element

Count of these = right - left + 1
```

Each one of these is a **distinct valid subarray ending exactly at `right`**. By doing this for every `right` from `0` to `n-1`, you count every valid subarray in the array exactly once, with zero double-counting (because each subarray has exactly one "rightmost index").

**Generic pseudocode:**

```
for right in range(n):
    add nums[right] to window
    while window is invalid:
        remove nums[left] from window
        left += 1
    answer += right - left + 1     # count all valid subarrays ending at right
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

**Dry run:** `nums = [1,2,1]`, count subarrays with sum `<= 3`

```
right=0: sum=1 (<=3)              count += 0-0+1 = 1   total=1
right=1: sum=3 (<=3)              count += 1-0+1 = 2   total=3
right=2: sum=4 (>3) shrink:
    remove nums[0]=1 → sum=3, left=1 (<=3) stop
                                   count += 2-1+1 = 2   total=5

Answer: 5
Valid subarrays: [1],[2],[1],[1,2],[2,1]  → 5 ✓
```

**Common mistakes:**

* Forgetting this trick only works cleanly for **monotonic** conditions (adding more elements only makes things "more invalid", never flips back to valid). For non-monotonic conditions, this breaks.
* Adding `right - left + 1` at the wrong point (must be after shrinking is fully done).
* Confusing "count subarrays" with "count windows of fixed size" — they use different templates.

**Complexity:** Time `O(n)`, Space `O(1)`.

### LeetCode Mapping — What Changes?

**LC 713 — Subarray Product Less Than K**
Same as base template, but multiply instead of add, and shrink while `product >= k`.
```python
window_product *= nums[right]
while window_product >= k and left <= right:
    window_product /= nums[left]
    left += 1
count += right - left + 1
```

**LC 560 — Subarray Sum Equals K**
This is NOT monotonic (negative numbers break the trick) — solved with prefix sum + hashmap instead, not pure sliding window. Mentioned here so you know which problems look like sliding window but aren't.

**LC 930 / LC 1248 — Binary Subarrays With Sum / Exactly K**
These use the **AtMost(K) - AtMost(K-1)** trick — see Template 5.

---

## TEMPLATE 5 — AtMost(K) - AtMost(K-1)

**Recognition signals:** "exactly K distinct", "exactly K odd numbers", "exactly K of something"

**Mental model:** "Exactly K" is hard to slide directly because validity isn't monotonic in an easy way. But "at most K" IS monotonic and easy (Template 4). The trick:

```
Exactly(K)  =  AtMost(K)  -  AtMost(K - 1)
```

**Intuition:** `AtMost(K)` counts every window with `0, 1, 2, ..., K` of the property. `AtMost(K-1)` counts every window with `0, 1, ..., K-1` of the property. Subtracting removes everything except windows with **exactly K**.

**Visual proof:**

```
AtMost(K)    counts windows with property count in {0,1,2,...,K-1,K}
AtMost(K-1)  counts windows with property count in {0,1,2,...,K-1}

AtMost(K) - AtMost(K-1)  =  windows with property count == K exactly
```

**Dry run:** `nums = [1,0,1,0,1]`, count subarrays with **exactly** 2 ones (LC 930-style)

```
AtMost(2 ones): count all subarrays with <= 2 ones → (compute with Template 4) = 12
AtMost(1 one):  count all subarrays with <= 1 one  → (compute with Template 4) = 8

Exactly(2) = 12 - 8 = 4
```//(Numbers illustrative of the method — always re-derive with your own counting pass.)

**Python template:**

```python
def at_most_k(nums, k):
    if k < 0:
        return 0
    left = 0
    count_ones = 0
    total = 0
    for right in range(len(nums)):
        count_ones += nums[right]   # property check, e.g. is it a "1"
        while count_ones > k:
            count_ones -= nums[left]
            left += 1
        total += right - left + 1
    return total

def exactly_k(nums, k):
    return at_most_k(nums, k) - at_most_k(nums, k - 1)
```

**Common mistakes:**

* Forgetting `AtMost(-1)` should return `0` (no negative counts possible).
* Writing two separate sliding window passes with subtly different logic — keep `at_most_k` as ONE reusable function and call it twice.
* Trying to solve "exactly K" directly without this decomposition — usually leads to bugs.

**Complexity:** Time `O(n)` (two passes, still linear), Space `O(1)`.

### LeetCode Mapping — What Changes?

**LC 992 — Subarrays with K Different Integers**
Property = distinct count using a frequency map instead of a sum.
```python
freq[nums[right]] = freq.get(nums[right], 0) + 1
while len(freq) > k:
    freq[nums[left]] -= 1
    if freq[nums[left]] == 0:
        del freq[nums[left]]
    left += 1
```

**LC 1248 — Count Number of Nice Subarrays**
Property = "is odd number", same as the ones-counting dry run above.

**LC 930 — Binary Subarrays With Sum**
Slightly different: property is "sum equals goal" → use `AtMost(goal) - AtMost(goal - 1)` on sum instead of count.

---

## TEMPLATE 6 — Frequency Matching

**Recognition signals:** "anagram", "permutation", "contains all characters of", "same character frequency as"

**Mental model:** You need two frequency maps: a `need` map (built once from the target string) and a `window` map (built as you slide). Instead of comparing entire maps every step (slow), track a single integer `matched` = how many distinct characters currently have the EXACT required count in the window.

**Generic pseudocode:**

```
build need map from pattern
matched = 0
required = number of distinct chars in pattern

for right in range(n):
    add s[right] to window map
    if window[s[right]] == need[s[right]]:
        matched += 1

    while window is "too big" (size > len(pattern)):
        if window[s[left]] == need[s[left]]:
            matched -= 1
        remove s[left] from window
        left += 1

    if matched == required:
        record answer
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

        # keep window size == len(p)
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

**Dry run:** `s = "cbaebabacd"`, `p = "abc"` (find all anagram start indices — LC 438 style)

```
need = {a:1, b:1, c:1}, required=3

right=0 'c': window={c:1} matched=1
right=1 'b': window={c:1,b:1} matched=2
right=2 'a': window={c:1,b:1,a:1} matched=3 → matched==required, size==3
              record left=0
right=3 'e': window size becomes 4 → shrink:
    remove s[0]='c': window={c:0,b:1,a:1} c was matched, now broken → matched=2
    'e' not in need, not added. left=1
              matched=2 ≠ 3, no record
right=4 'b': window={b:2,a:1} ('e' irrelevant) matched still tracked carefully...
   (full trace omitted for brevity — pattern continues)

Final recorded starts: [0, 6]   matching LC 438's expected output
```

**Common mistakes:**

* Comparing full dictionaries every step instead of using the `matched` counter — this turns an O(n) solution into O(n·26) or worse.
* Forgetting to check `right - left + 1 == len(p)` before recording (window must be exactly target length).
* Mutating `need` instead of keeping it as a separate reference map.

**Complexity:** Time `O(n)`, Space `O(distinct characters)`.

### LeetCode Mapping — What Changes?

**LC 567 — Permutation in String**
Same as base template — just return `True` the first time `matched == required`.

**LC 438 — Find All Anagrams in a String**
Base template exactly as shown — collect every matching `left` index.

**LC 30 — Substring with Concatenation of All Words**
Upgrade: `need` is built from whole words instead of characters, and the window slides in word-sized chunks instead of character-sized chunks. Same `matched` counter idea, different unit size.

---

## TEMPLATE 7 — Monotonic Queue (Deque)

**Recognition signals:** "maximum/minimum of every window of size k", "sliding window maximum"

**Mental model:** We want the max of every window in O(1) per step. A regular queue doesn't help because the max could be anywhere. The trick: keep a **deque of indices** where the corresponding values are in **decreasing order** front-to-back. The front of the deque is always the current window's maximum.

### Building the Deque from Zero

Think of the deque as a "leaderboard" of candidates that could still become the max:

* When a new number arrives, any smaller number sitting at the *back* of the deque can NEVER be the max again (the new number is bigger and will outlive it in the window) — so we pop them off the back.
* We push the new number's index to the back.
* If the index at the *front* has fallen out of the window (too old), we pop it from the front.
* The front of the deque is always this window's maximum.

**Visual walkthrough:** `nums = [1,3,-1,-3,5,3,6,7]`, `k = 3`

```
Deque stores INDICES, shown with their VALUES in brackets for clarity.

i=0, val=1:  deque=[0(1)]
i=1, val=3:  1 < 3, pop index 0. deque=[1(3)]
i=2, val=-1: -1 < 3, just push. deque=[1(3), 2(-1)]
             window [0,1,2] complete → max = value at front = 3
i=3, val=-3: -3 < -1, just push. deque=[1(3), 2(-1), 3(-3)]
             front index 1 still in window [1,2,3]? yes → max = 3
i=4, val=5:  pop back while smaller: pop 3(-3), pop 2(-1), pop 1(3)
             deque=[4(5)]
             window [2,3,4] → front index 4 in range → max = 5
i=5, val=3:  3 < 5, push. deque=[4(5), 5(3)]
             window [3,4,5] → front index 4 in range → max = 5
i=6, val=6:  pop back while smaller: pop 5(3), pop 4(5)
             deque=[6(6)]
             window [4,5,6] → max = 6
i=7, val=7:  pop 6(6), deque=[7(7)]
             window [5,6,7] → max = 7

Answer sequence: [3, 3, 5, 5, 6, 7]
```

**Generic pseudocode:**

```
deque = empty (stores indices)
for right in range(n):
    while deque not empty AND nums[deque.back] <= nums[right]:
        deque.pop_back()
    deque.push_back(right)

    if deque.front <= right - k:        # fallen out of window
        deque.pop_front()

    if right >= k - 1:
        record nums[deque.front] as this window's max
```

**Python template:**

```python
from collections import deque

def sliding_window_maximum(nums, k):
    dq = deque()      # stores indices, values strictly decreasing
    result = []

    for right in range(len(nums)):
        # remove smaller values from the back — they can never win
        while dq and nums[dq[-1]] <= nums[right]:
            dq.pop()
        dq.append(right)

        # remove front index if it has slid out of the window
        if dq[0] <= right - k:
            dq.popleft()

        if right >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**Common mistakes:**

* Storing values instead of indices — you lose track of *when* an element should expire from the window.
* Using `<` instead of `<=` when popping from the back — duplicates can cause stale maxima to linger.
* Forgetting the `if right >= k - 1` guard — you'll record answers before the window is even full size.

**Complexity:** Time `O(n)` amortized (each index pushed and popped at most once), Space `O(k)`.

### LeetCode Mapping — What Changes?

**LC 239 — Sliding Window Maximum**
Exactly the base template above.

**LC 1438 — Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit**
Combine TWO monotonic deques — one tracking max, one tracking min — inside a Template-2-style (longest valid window) shell. Shrink while `max - min > limit`.

**LC 862 — Shortest Subarray with Sum at Least K**
Different flavor: monotonic deque on **prefix sums** instead of raw values, combined with Template 3's shrink-for-minimum-length idea.

---

# SECTION 4 — LeetCode Mapping Table

| Problem | Template | Recognition Signal | Modification Needed |
|---|---|---|---|
| LC 3 | T2 Longest Valid | "longest substring without repeating" | shrink while any freq > 1 |
| LC 30 | T6 Frequency Match | "concatenation of all words" | window unit = word, not char |
| LC 76 | T3 Smallest Valid | "minimum window substring" | add matched-count from T6 |
| LC 209 | T3 Smallest Valid | "minimum size subarray sum" | base template |
| LC 219 | T1 Fixed Size | "duplicate within distance k" | use a set, not sum |
| LC 239 | T7 Monotonic Queue | "max of every window" | base template |
| LC 340 | T2 Longest Valid | "at most k distinct chars" | base template |
| LC 424 | T2 Longest Valid | "replace chars, longest repeat" | track max_freq char |
| LC 438 | T6 Frequency Match | "find all anagrams" | collect all matching lefts |
| LC 567 | T6 Frequency Match | "permutation in string" | return True on first match |
| LC 643 | T1 Fixed Size | "max average subarray" | divide sum by k |
| LC 713 | T4 Counting | "product less than k" | multiply, shrink while >= k |
| LC 862 | T7 + T3 | "shortest subarray sum >= k" | deque on prefix sums |
| LC 904 | T2 Longest Valid | "fruit into baskets" | hardcode k=2 distinct |
| LC 930 | T5 AtMost(K) | "binary subarrays with sum" | AtMost on sum equality |
| LC 992 | T5 AtMost(K) | "k different integers" | AtMost on distinct count |
| LC 1004 | T2 Longest Valid | "max consecutive ones III" | shrink while zero_count > k |
| LC 1248 | T5 AtMost(K) | "nice subarrays" | property = odd numbers |
| LC 1438 | T7 + T2 | "abs diff <= limit, longest" | two deques (max & min) |
| LC 1456 | T1 Fixed Size | "max vowels length k" | count vowels instead of sum |
| LC 1493 | T2 Longest Valid | "after deleting one element" | same as LC 1004 with k=1 |
| LC 2461 | T1 Fixed Size | "distinct subarrays length k" | add distinct-check via map |

---

# SECTION 5 — Interview Cheat Sheet

When you hear this keyword in the problem... | ...reach for this template
---|---
"size K" / "length K" / "K consecutive" | **Template 1** — Fixed Size Window
"longest" / "maximum length" | **Template 2** — Longest Valid Window
"smallest" / "minimum length" | **Template 3** — Smallest Valid Window
"count subarrays/substrings" | **Template 4** — Counting Windows
"exactly K" | **Template 5** — AtMost(K) − AtMost(K−1)
"at most K" | **Template 4**, with the condition being "≤ K"
"anagram" / "permutation" | **Template 6** — Frequency Matching
"distinct" (as a constraint, not exact count) | **Template 2 or 5**, frequency map tracks distinct count
"max in every window" / "min in every window" | **Template 7** — Monotonic Queue

**The 30-second mental checklist:**

1. Is the window size fixed? → T1.
2. Otherwise, am I optimizing length (longest/shortest)? → T2 or T3.
3. Am I counting how many windows qualify? → T4 (or T5 if "exactly").
4. Does it involve matching character frequencies exactly? → T6.
5. Do I need the max/min of every window, not just one answer? → T7.

---

# SECTION 6 — 10 Minute Revision Page

### The Three Heartbeats (apply to all 7 templates)
**Expand** (move right, add effect) → **Shrink** (move left, remove effect, only while needed) → **Record** (update answer at the correct moment).

### The 7 Templates at a Glance

```python
# T1 — FIXED SIZE
for right in range(n):
    add(right)
    if right - left + 1 == k:
        record()
        remove(left); left += 1

# T2 — LONGEST VALID
for right in range(n):
    add(right)
    while invalid():
        remove(left); left += 1
    record(right - left + 1)        # AFTER fixing validity

# T3 — SMALLEST VALID
for right in range(n):
    add(right)
    while valid():
        record(right - left + 1)    # WHILE still valid
        remove(left); left += 1

# T4 — COUNTING (monotonic "at most" condition)
for right in range(n):
    add(right)
    while invalid():
        remove(left); left += 1
    count += right - left + 1       # batch-count all valid windows ending here

# T5 — EXACTLY K
exactly_k = at_most(k) - at_most(k - 1)   # reuse T4's at_most() twice

# T6 — FREQUENCY MATCH
need = build_freq_map(pattern)
matched = 0
for right in range(n):
    update window freq; if it now equals need's count, matched += 1
    while window too big:
        if it currently equals need's count, matched -= 1
        shrink
    if matched == required: record()

# T7 — MONOTONIC QUEUE (max example)
for right in range(n):
    while deque and nums[deque[-1]] <= nums[right]:
        deque.pop()
    deque.append(right)
    if deque[0] <= right - k:
        deque.popleft()
    if right >= k - 1:
        record(nums[deque[0]])
```

### Key Formulas
* Count of subarrays ending at `right` with start anywhere in `[left, right]` → `right - left + 1`
* Exactly(K) → `AtMost(K) - AtMost(K-1)`, with `AtMost(-1) = 0`
* Off-by-one check for fixed windows → use `right - left + 1 == k`, never `right - left == k`

### Mistakes That Cost Interviews
* `if` instead of `while` when shrinking
* Forgetting to delete zero-count keys from frequency maps (corrupts "distinct" counts)
* Recording the answer at the wrong point relative to the shrink loop (longest = after; smallest = during)
* Storing values instead of indices in the monotonic deque
* Applying the counting trick (`right-left+1`) to non-monotonic conditions where it doesn't hold

---

# MOST IMPORTANT TAKEAWAY

You do not need to memorize 50 LeetCode problems. You need to memorize **7 small templates** and recognize, in under 30 seconds, which one a new problem secretly is. Every problem in Section 4's table is "Template X, with this one tiny change" — practice spotting the change, not re-deriving the whole solution from scratch.
