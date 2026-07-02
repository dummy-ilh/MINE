# Two Pointers — Top Templates for Max Coverage

Distilled from the Easy/Medium/Hard solutions already covered. These ~9 templates account for the large majority of those 200+ problems. Memorize the shape, not the problem.

---

## 1. Converging ends (the classic)
`l` starts at 0, `r` at the end, move inward based on a condition. Covers anything "sorted array, find a pair/condition."

```python
def converge(arr):
    l, r = 0, len(arr) - 1
    while l < r:
        if condition(arr[l], arr[r]):
            # do something
            l += 1
            r -= 1
        elif need_more(arr[l], arr[r]):
            l += 1
        else:
            r -= 1
    return result
```
**Covers:** Two Sum II (167), Valid Palindrome (125/680), Reverse String (344), Container With Most Water (11), Squares of Sorted Array (977), Sort Array By Parity (905), Trapping Rain Water (42), Boats to Save People (881), Minimize Maximum Pair Sum (1877), Count Pairs Sum Less Than Target (2824), Backspace Compare variants, Valid Palindrome IV (2330).

---

## 2. Slow/fast write pointer (in-place compaction)
`slow` marks the next "good" write position; `fast` scans ahead and copies in when valid.

```python
def compact(nums):
    slow = 0
    for fast in range(len(nums)):
        if keep(nums[fast]):
            nums[slow] = nums[fast]
            slow += 1
    return slow  # new length
```
**Covers:** Remove Duplicates (26/80), Remove Element (27), Move Zeroes (283), Apply Operations to an Array (2460), Minimum Swaps to Move Zeros to End (3936), Limit Occurrences in Sorted Array (3940), Partition List (86).

**Variant — Dutch flag (3-way) for sort-in-one-pass:**
```python
def sort_colors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```
**Covers:** Sort Colors (75).

---

## 3. Floyd's slow/fast (cycle detection)
`slow` moves 1 step, `fast` moves 2. If they meet, there's a cycle; the meeting point also gives you the cycle start.

```python
def floyd(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True  # cycle found
    return False
```
**Covers:** Linked List Cycle (141), Linked List Cycle II (142), Happy Number (202), Find the Duplicate Number (287), Middle of Linked List (876), Palindrome Linked List (234), Circular Array Loop (457).

---

## 4. Merge two sorted sequences
Independent pointer per sequence; always advance the smaller (or matching) side.

```python
def merge(a, b):
    i, j, res = 0, 0, []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            res.append(a[i]); i += 1
        else:
            res.append(b[j]); j += 1
    res.extend(a[i:]); res.extend(b[j:])
    return res
```
**Covers:** Merge Sorted Array (88), Merge Two Sorted Lists / Sort List (148), Intersection of Two Arrays (349/350), Interval List Intersections (986), Merge Strings Alternately (1768), Merge 2D Arrays by Summing (2570), Minimum Common Value (2540), Get Max Score (1537).

---

## 5. Fixed pointer + converging inner pointers (k-Sum family)
Outer loop fixes one (or two) elements; inner two pointers solve the remaining 2-sum on the sorted rest.

```python
def k_sum(nums, target):
    nums.sort()
    res = []
    for i in range(len(nums)):
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == target:
                res.append([nums[i], nums[l], nums[r]])
                l += 1; r -= 1
            elif s < target:
                l += 1
            else:
                r -= 1
    return res
```
**Covers:** 3Sum (15), 3Sum Closest (16), 4Sum (18, add one more outer loop), 3Sum Smaller (259), 3Sum With Multiplicity (923), Valid Triangle Number (611, scan k from the right instead).

---

## 6. Two-pointer subsequence matching
One pointer per string/array; advance the "source" pointer always, the "target" pointer only on a match.

```python
def is_subsequence(s, t):
    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)
```
**Covers:** Is Subsequence (392), Longest Word in Dictionary through Deleting (524), Two Sum BSTs (1214), Camelcase Matching (1023), Sentence Similarity III (1813), Maximum Removable Characters (1898).

---

## 7. Expand-around-center (palindromes)
For every possible center, grow outward while characters match.

```python
def expand(s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return s[l + 1:r]

def longest_palindrome(s):
    res = ""
    for i in range(len(s)):
        res = max(res, expand(s, i, i), expand(s, i, i + 1), key=len)
    return res
```
**Covers:** Longest Palindromic Substring (5), Palindromic Substrings (647), Valid Palindrome II (680), First Palindromic String in Array (2108), Number of Non-overlapping Palindromes (2472) — and Manacher's (1960) is the O(n) version of this same idea.

---

## 8. Binary search the answer + two-pointer feasibility check
Guess an answer with binary search; verify it in O(n) using a two-pointer/greedy scan.

```python
def binary_search_answer(lo, hi, feasible):
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```
**Covers:** Find K-th Smallest Pair Distance (719), Successful Pairs of Spells and Potions (2300), Maximum Number of Tasks You Can Assign (2071), Minimum Time to Eat All Grains (2604), Maximum Total Beauty of the Gardens (2234), Capacity-style "minimize the max" problems generally.

---

## 9. Two pointers across two independent collections (matching/scheduling)
Like template 4, but instead of merging you're checking overlap/compatibility and advancing whichever side is "behind."

```python
def schedule(a, b):
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if compatible(a[i], b[j]):
            return a[i], b[j]   # or record/count
        if a[i] ends before b[j]:
            i += 1
        else:
            j += 1
```
**Covers:** Meeting Scheduler (1229), Advantage Shuffle (870), Maximum Distance Between a Pair of Values (1855), Checking Existence of Edge Length Limited Paths (1697, with union-find), Find the Distance Value Between Two Arrays (1385).

---

## How to use this
When you see a new problem, ask:
1. **Is it sorted (or sortable) and about pairs/sums?** → Template 1 or 5.
2. **In-place removal/compaction?** → Template 2.
3. **Linked list, "detect/find something cyclic or middle"?** → Template 3.
4. **Two sorted things to combine?** → Template 4.
5. **One sequence "contained in" another?** → Template 6.
6. **Palindrome-related?** → Template 7.
7. **"Minimize the maximum" / "maximize the minimum"?** → Template 8.
8. **Two independent lists you're matching up?** → Template 9.

If a problem doesn't fit any of these cleanly, it's probably not a pure two-pointer problem — it likely needs a stack, heap, or DP layered on top (several of the Hard ones from before are exactly this: two pointers as the *feasibility check* inside something bigger).
