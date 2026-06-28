# Greedy Algorithms — Interview Handbook

*A revision-in-a-few-hours reference for LeetCode / GFG greedy problems.*

---

## Table of Contents

1. [Quick Prerequisites](#1-quick-prerequisites)
2. [Greedy Recognition Checklist](#2-greedy-recognition-checklist)
3. [Core Greedy Patterns](#3-core-greedy-patterns)
4. [Problem Templates](#4-problem-templates)
5. [Common Interview Problems](#5-common-interview-problems)
6. [Final Reference Pages](#6-final-reference-pages)

---

## 1. Quick Prerequisites

### 1.1 Sorting with a key

Almost every greedy problem starts with **"sort by the right thing."** The hard part is never the `sort()` call — it's figuring out *what* to sort by.

```python
# Sort by a single field
intervals.sort(key=lambda x: x[0])          # by start
jobs.sort(key=lambda x: x[1])               # by deadline

# Sort by multiple fields (tie-breaker)
people.sort(key=lambda x: (x[0], -x[1]))    # by height asc, then by k desc

# Sort descending
nums.sort(key=lambda x: -x)                 # or nums.sort(reverse=True)

# Sort by a derived value (ratio, ending point, etc.)
items.sort(key=lambda x: x[1] / x[0])       # value/weight ratio (fractional knapsack)
```

**Rule of thumb:** if a problem says "interval," sort by start *or* end depending on whether
you're merging (sort by start) or selecting max count (sort by end). If it says
"deadline/profit," sort by deadline or profit depending on whether you're filtering
feasibility (deadline) or picking high-value first (profit).

### 1.2 `heapq` idioms

Python's `heapq` is a **min-heap only**. Memorize these three tricks:

```python
import heapq

# 1. Min-heap (default) — smallest pops first
heap = []
heapq.heappush(heap, 5)
heapq.heappop(heap)            # smallest

# 2. Max-heap — negate values
heapq.heappush(heap, -val)
biggest = -heapq.heappop(heap)

# 3. Heap of tuples — sorts by first element, then second (tie-break)
heapq.heappush(heap, (deadline, profit, idx))

# 4. Heapify an existing list in O(n)
heapq.heapify(nums)

# 5. Push/pop combo (slightly faster than separate calls)
heapq.heappushpop(heap, x)     # push x, then pop smallest
heapq.heapreplace(heap, x)     # pop smallest, then push x

# 6. k largest / k smallest without a full sort
heapq.nlargest(k, nums)
heapq.nsmallest(k, nums)
```

**Use a heap when** you repeatedly need the current min/max while the set of
candidates is changing (new elements arrive, old ones get consumed).

### 1.3 Custom comparators

When the ordering isn't a simple key (e.g. "which string concatenation is bigger"),
Python 3 has no `cmp=` in `sort()` — convert with `functools.cmp_to_key`:

```python
from functools import cmp_to_key

def compare(a, b):
    # return negative if a should come before b, positive if after, 0 if equal
    if a + b > b + a:
        return -1     # a before b
    elif a + b < b + a:
        return 1
    return 0

nums_str.sort(key=cmp_to_key(compare))
```

Classic use: **LC 179 Largest Number** — compare `a+b` vs `b+a` as strings.

### 1.4 Exchange argument (intuition only)

This is *the* proof technique behind nearly every greedy correctness argument, and
interviewers love hearing it even in 30 seconds:

> Take any optimal solution that disagrees with the greedy choice. Show you can
> **swap two adjacent elements** (or substitute the greedy choice for whatever the
> optimal solution did) **without making the solution worse**. Since this swap is
> always safe, the greedy choice is always at least as good — so making it first,
> repeatedly, is safe.

You don't need a rigorous proof in an interview — just say *"I can show by an
exchange argument that swapping in the greedy choice never hurts,"* and give the
one-sentence reason (e.g. "sorting by end time and picking the earliest-ending
interval never blocks more future options than any other valid first choice").

### 1.5 Greedy vs DP — the gut check

| Signal | Greedy | DP |
|---|---|---|
| "Local best choice never needs to be undone" | ✅ | |
| You need to compare *all* subsets/subsequences for a global optimum | | ✅ |
| Problem has overlapping subproblems that depend on multiple prior states | | ✅ |
| Once sorted, a single pass with no backtracking solves it | ✅ | |
| Counterexample exists where the greedy choice provably fails | | ✅ |
| Asks for **existence/count of a way** under many constraints (knapsack-like) | | ✅ |
| Asks for **min/max** achievable by always taking the "obviously best" local option | ✅ | |

**Quick test:** try to break your greedy idea with a small adversarial example
(3-4 elements) before coding. If you can't break it in under a minute, it's
probably correct. If you *can* break it, you likely need DP.

---

## 2. Greedy Recognition Checklist

### Signs a problem is Greedy
- Sorting (by value, ratio, start/end time, deadline) immediately simplifies the problem.
- The problem says "maximum number of non-overlapping…", "minimum number of…to cover…",
  "schedule", "assign", "at every step pick the best available option".
- A local optimal choice **never** needs to be revisited (no backtracking).
- There's a natural **exchange argument**: swapping the greedy choice for any other
  choice can't make things worse.
- The problem reduces to "process elements in a fixed order, maintain a running
  invariant (heap/stack/counter), make an irrevocable choice each step."

### Signs it's probably DP instead
- The greedy choice that looks "obviously best" has a counterexample with 3-4 elements.
- The problem involves a **budget/capacity that's spent across the whole array**, not just
  consumed locally (0/1 knapsack, partition into k subsets, edit distance).
- "Number of ways to…" — counting problems are essentially never plain greedy.
- The decision at index `i` depends on a **combination of earlier decisions**, not
  just one or two running aggregates.
- Problem explicitly allows revisiting/undoing choices, or the cost structure is
  *non-linear* and choices interact globally.

### Common keywords → likely pattern
| Keyword in problem | Likely approach |
|---|---|
| "maximum number of non-overlapping intervals" | Interval Scheduling |
| "merge intervals" / "free time" | Interval Merge |
| "minimum platforms / meeting rooms" | Heap or Sort+Sweep |
| "k-th largest", "top k", "closest k" | Heap |
| "schedule jobs with deadline/profit" | Heap (deadline-feasibility) |
| "candies", "ratings", "next greater" | Monotonic Stack |
| "container", "boats", "two sum to target" | Two Pointers |
| "connect ropes/sticks at min cost" | Heap (always merge two smallest) |
| "minimum spanning tree / connect all cities" | Kruskal / Prim |
| "shortest path with non-negative weights" | Dijkstra (greedy + heap) |
| "assign workers to tasks", "cookies to children" | Sort both arrays + two-pointer |
| "remove k digits", "monotonic" | Monotonic Stack |

### Decision tree (read top to bottom, stop at first match)

```
Is there an obvious "always pick the extreme value first" rule?
│
├─ YES → Can you break it with a 4-element counterexample?
│         │
│         ├─ Breaks  → reconsider: DP, or refine the sort key
│         │
│         └─ Doesn't break → GREEDY. Now ask:
│                  │
│                  ├─ Problem is about intervals?
│                  │     ├─ Count max non-overlapping → sort by END, Interval Scheduling
│                  │     └─ Merge/union of ranges      → sort by START, Interval Merge
│                  │
│                  ├─ Need running min/max while processing? → Greedy + Heap
│                  │
│                  ├─ Need "next greater/smaller so far"?    → Greedy + Stack
│                  │
│                  ├─ Two sorted sequences to pair up?       → Two Pointers
│                  │
│                  └─ Graph: connect all nodes cheaply / shortest path?
│                        ├─ MST              → Kruskal (sparse) / Prim (dense)
│                        └─ Shortest path     → Dijkstra
│
└─ NO obvious greedy rule →
          Does subproblem optimality depend on combining multiple prior states?
          ├─ YES → DP
          └─ NO  → Look harder for a sort key; it's probably still greedy
```

---

## 3. Core Greedy Patterns

> Each pattern below: **intuition → recognition cues → template → complexity →
> pitfalls → representative problems.** Full reusable skeletons live in Section 4;
> here the templates are the minimal version you'd write live in an interview.

### 3.1 Sort + Traverse

**Intuition.** Many problems become trivial once you process elements in the right
order — because then the "best" choice at each step is simply "the next one in
line." Sorting front-loads all the hard ordering logic, leaving a dumb O(n) sweep.
The exchange argument is usually: *if processing out of sorted order ever helped,
you could swap two adjacent out-of-order elements and not make things worse — so
sorted order is always at least as good.*

**Recognition cues:** "maximize/minimize sum after some pairing/negation/assignment,"
ratios, "every element must be matched to exactly one other," problems where a single
pass after sorting directly gives the answer with no extra data structure.

```python
def sort_and_traverse(arr):
    arr.sort()                  # or sort with a custom key
    result = 0
    for x in arr:
        # irrevocable, locally-optimal update
        result += x             # placeholder logic
    return result
```

**Complexity:** O(n log n) for the sort, O(n) for the traversal.

**Pitfalls:**
- Sorting ascending when you need descending (or vice versa) — always sanity check
  on a 2-element example.
- Forgetting a stable tie-break key when two elements compare equal but order matters.

**Representative problems:** LC 455 Assign Cookies, LC 1005 Maximize Sum After K
Negations, LC 561 Array Partition, GFG Min Cost to Make Array Size 1.

---

### 3.2 Interval Scheduling (max non-overlapping count)

**Intuition.** To fit the *maximum number* of non-overlapping intervals, always pick
the interval that **ends earliest** among the remaining valid candidates. Ending
early leaves the most room for future intervals — no other choice can ever leave
*more* room, so it's never worse. This is the textbook exchange-argument proof.

**Recognition cues:** "maximum number of activities/intervals/events you can attend,"
"minimum number of intervals to remove to make the rest non-overlapping" (this is
the complement: `n - max_non_overlapping`).

```python
def max_non_overlapping(intervals):
    intervals.sort(key=lambda x: x[1])     # sort by END time
    count = 0
    last_end = float('-inf')
    for start, end in intervals:
        if start >= last_end:              # no overlap with last chosen
            count += 1
            last_end = end
    return count
```

**Complexity:** O(n log n).

**Pitfalls:**
- Sorting by start instead of end — this is the #1 mistake on this pattern.
- Off-by-one on touching intervals (`start >= last_end` vs `start > last_end`,
  re-read the problem: do touching intervals count as overlapping?).

**Representative problems:** LC 435 Non-overlapping Intervals, GFG Activity
Selection Problem, LC 452 Minimum Number of Arrows to Burst Balloons (sort by end,
greedily reuse an arrow if it still hits the current balloon).

---

### 3.3 Interval Merge / Selection

**Intuition.** To merge overlapping intervals into their union, sort by **start**
time, then sweep left to right keeping a "current merged interval." Any interval
whose start is ≤ the current merged interval's end extends it; otherwise it starts
a new group. Sorting by start guarantees you never need to look backward.

**Recognition cues:** "merge overlapping intervals," "insert interval," "employee
free time," "minimum number of intervals to cover a range."

```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])     # sort by START
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:         # overlaps current group
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

**Complexity:** O(n log n).

**Pitfalls:**
- Forgetting to handle an empty input.
- Using `<` instead of `<=` when touching intervals should merge.

**Representative problems:** GFG Merge Overlapping Intervals, LC 56 (implicit via
Jump Game family), LC 1024 Video Stitching.

---

### 3.4 Earliest Finish Time / Two-Pointer Interval Matching

**Intuition.** When you have two sets that need to be matched against each other
under a feasibility constraint (e.g. "rider must finish before next rider can
start," "trainer's capacity must be ≥ player's requirement"), sort **both** sides
and walk them with two pointers, always satisfying the easiest constraint first.
This guarantees that whenever a match is possible, you find it without missing a
better hidden match later.

**Recognition cues:** "earliest finish time," "match players to trainers," "advantage
shuffle," any "given two arrays, pair elements satisfying X, maximize/minimize Y."

```python
def earliest_finish_time(starts, durations):
    order = sorted(range(len(starts)), key=lambda i: starts[i])
    finish = 0
    result = [0] * len(starts)
    for i in order:
        finish = max(finish, starts[i]) + durations[i]
        result[i] = finish
    return result
```

**Complexity:** O(n log n).

**Pitfalls:**
- Matching greedily from the wrong end (smallest-to-smallest vs smallest-to-largest)
  — depends on whether you're maximizing matches (smallest feasible) or maximizing
  sum (largest feasible).

**Representative problems:** LC 3633 / 3635 Earliest Finish Time for Rides, LC 2410
Maximum Matching of Players With Trainers, LC 870 Advantage Shuffle.

---

### 3.5 Greedy + Heap

**Intuition.** Use a heap whenever you need the **running min or max** of a
dynamically changing candidate set — new candidates appear as you scan, and you
repeatedly need "the best one available right now" without re-scanning everything.
Classic uses: always merge the two cheapest items (rope/stick connecting, Huffman
coding), always pick the highest-profit feasible job (deadline scheduling), or
maintain the top-k seen so far.

**Recognition cues:** "minimum cost to connect/merge all X," "schedule jobs with
deadline and profit, maximize profit," "kth largest," "minimum number of platforms /
meeting rooms," "maximum number of events you can attend."

```python
import heapq

def connect_at_min_cost(costs):
    """Always merge the two cheapest. (Ropes/Sticks/Huffman family.)"""
    heapq.heapify(costs)
    total = 0
    while len(costs) > 1:
        a = heapq.heappop(costs)
        b = heapq.heappop(costs)
        total += a + b
        heapq.heappush(costs, a + b)
    return total


def max_profit_with_deadlines(jobs):
    """jobs = [(deadline, profit), ...]. Process by deadline,
    keep a min-heap of profits taken so far; if heap size exceeds
    the deadline-implied slots, evict the smallest profit."""
    jobs.sort()
    heap = []
    for deadline, profit in jobs:
        heapq.heappush(heap, profit)
        if len(heap) > deadline:
            heapq.heappop(heap)        # evict the least profitable job taken
    return sum(heap)
```

**Complexity:** O(n log n) — each element pushed/popped at most O(log n) work.

**Pitfalls:**
- Forgetting `heapq.heapify` before the loop (O(n log n) once, vs n individual
  pushes which is also fine but slightly slower).
- Using a max-heap pattern but forgetting to negate on push *and* pop.
- Not re-checking heap size invariant after every push.

**Representative problems:** LC 1167 Minimum Cost to Connect Sticks, GFG Huffman
Coding, LC 502 IPO, LC 253 Meeting Rooms II, LC 1642 Furthest Building You Can Reach.

---

### 3.6 Resource Allocation (two-pointer assignment)

**Intuition.** When you must assign items from set A to set B under a feasibility
rule (A[i] can satisfy B[j] iff A[i] ≥ B[j]), sort both sets and walk with two
pointers, greedily satisfying the smallest unsatisfied demand with the smallest
sufficient resource. Using a bigger-than-necessary resource on an easy demand wastes
capacity that a harder demand might have needed — so always use the *minimum
sufficient* resource.

**Recognition cues:** "assign cookies to children," "boats to save people," "maximum
units on a truck," any bipartite greedy matching with a single feasibility inequality.

```python
def assign_resources(demand, supply):
    demand.sort()
    supply.sort()
    i = j = matched = 0
    while i < len(demand) and j < len(supply):
        if supply[j] >= demand[i]:
            matched += 1
            i += 1
        j += 1
    return matched
```

**Complexity:** O(n log n).

**Pitfalls:**
- Sorting only one side.
- Using a greedy pointer scheme when the constraint is two-sided (e.g. "boats hold
  at most 2 people AND total weight ≤ limit" — pair lightest with heaviest instead).

**Representative problems:** LC 455 Assign Cookies, LC 881 Boats to Save People,
LC 1710 Maximum Units on a Truck.

---

### 3.7 Greedy + Stack (monotonic stack)

**Intuition.** When you need to repeatedly compare the current element against the
"most recent still-relevant" earlier elements, and earlier elements become
irrelevant once a better/worse one shows up, a stack that maintains a monotonic
(increasing or decreasing) order lets you discard dominated elements in O(1)
amortized per element. The greedy insight is: an element popped off the stack can
*never* be useful again, because the new element dominates it.

**Recognition cues:** "remove k digits to make the smallest/largest number,"
"next greater element," "candy distribution," "make string non-decreasing by
removals," "monotonic stack" anywhere in your own head.

```python
def remove_k_digits_smallest(num, k):
    stack = []
    for digit in num:
        while stack and k > 0 and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    stack = stack[:len(stack) - k] if k else stack  # remove leftovers from the end
    result = ''.join(stack).lstrip('0')
    return result or '0'
```

**Complexity:** O(n) amortized (each element pushed once, popped at most once).

**Pitfalls:**
- Forgetting to trim leftover `k` removals from the **end** of the stack if the loop
  never used them all up.
- Forgetting to strip leading zeros at the end.
- Confusing "want smallest result" (pop while top > current) with "want largest
  result" (pop while top < current).

**Representative problems:** LC 402 Remove K Digits, LC 316 Remove Duplicate
Letters, LC 135 Candy, LC 738 Monotone Increasing Digits.

---

### 3.8 Greedy + Two Pointers

**Intuition.** When the search space is a sorted array (or two sorted arrays) and
you're optimizing something that monotonically trades off as one pointer moves in
versus the other, two pointers let you discard one entire side of the
possibility space per step instead of checking all pairs. The greedy step is
usually "move the pointer that can only get worse if left alone."

**Recognition cues:** "container with most water," "two sum closest to target,"
"valid palindrome with one allowed deletion," "furthest pair satisfying constraint."

```python
def max_area(heights):
    left, right = 0, len(heights) - 1
    best = 0
    while left < right:
        width = right - left
        best = max(best, width * min(heights[left], heights[right]))
        if heights[left] < heights[right]:
            left += 1     # the shorter wall is the bottleneck — move it
        else:
            right -= 1
    return best
```

**Complexity:** O(n).

**Pitfalls:**
- Moving the wrong pointer (always move the one limited by the smaller value).
- Off-by-one on the loop condition (`<` vs `<=`).

**Representative problems:** LC 11 Container With Most Water, LC 680 Valid
Palindrome II, LC 881 Boats to Save People (also resource allocation).

---

### 3.9 Graph Greedy (Kruskal, Prim, Dijkstra)

**Intuition.** All three are "greedy + the right data structure":
- **Kruskal's MST:** sort all edges by weight ascending; greedily add an edge if it
  connects two different components (checked via Union-Find). Adding the globally
  cheapest edge that doesn't create a cycle is always safe — the **cut property**
  guarantees no cheaper alternative MST exists without it.
- **Prim's MST:** grow a single tree from an arbitrary start node; at every step,
  greedily add the cheapest edge that connects the tree to a new vertex (via a
  min-heap of frontier edges).
- **Dijkstra's shortest path:** greedily finalize the unvisited vertex with the
  smallest tentative distance — since all edge weights are non-negative, no later
  relaxation can ever produce a shorter path to an already-finalized vertex.

**Recognition cues:** "minimum cost to connect all cities/points," "minimum spanning
tree," "shortest path, all weights non-negative."

```python
# --- Kruskal's MST (Union-Find) ---
def kruskal(n, edges):
    """edges = [(weight, u, v), ...]"""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    edges.sort()                  # by weight ascending
    total_cost, edges_used = 0, 0
    for w, u, v in edges:
        if union(u, v):
            total_cost += w
            edges_used += 1
            if edges_used == n - 1:
                break
    return total_cost


# --- Prim's MST (heap-based) ---
import heapq

def prim(n, adj):
    """adj[u] = list of (weight, v)"""
    visited = [False] * n
    heap = [(0, 0)]            # (weight, node) — start at node 0
    total_cost = 0
    while heap:
        w, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total_cost += w
        for weight, v in adj[u]:
            if not visited[v]:
                heapq.heappush(heap, (weight, v))
    return total_cost


# --- Dijkstra's shortest path ---
def dijkstra(n, adj, src):
    """adj[u] = list of (v, weight)"""
    dist = [float('inf')] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue            # stale entry, skip
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

**Complexity:**
- Kruskal: O(E log E) for sort + near-O(E) for Union-Find with path compression.
- Prim: O(E log V) with a binary heap.
- Dijkstra: O(E log V) with a binary heap.

**Pitfalls:**
- Kruskal: forgetting path compression / union by rank → near O(E²) worst case.
- Prim: forgetting the `if visited[u]: continue` check after popping → stale
  entries corrupt the cost.
- Dijkstra: using it on graphs with **negative edge weights** — it silently gives
  wrong answers; you need Bellman-Ford instead.
- Forgetting the stale-entry skip in Dijkstra's heap pop (same bug as Prim).

**Representative problems:** GFG Kruskal's/Prim's MST, GFG Min Cost to Connect All
Cities, GFG Dijkstra's Shortest Path Algorithm.

---

## 4. Problem Templates

These are the **literal skeletons** to copy-paste and adapt live. Each includes when
to use it, variable meanings, a dry run, and the typical 1-line tweaks that turn it
into a dozen different LeetCode problems.

### Template A — Interval Scheduling / Merge

```python
def template_interval(intervals, mode="max_count"):
    if mode == "max_count":
        intervals.sort(key=lambda x: x[1])      # sort by END
        count, last_end = 0, float('-inf')
        for s, e in intervals:
            if s >= last_end:
                count += 1
                last_end = e
        return count

    elif mode == "merge":
        intervals.sort(key=lambda x: x[0])      # sort by START
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged
```

- **`intervals`**: list of `[start, end]` pairs.
- **`last_end`**: end time of the most recently *accepted* interval.
- **`merged`**: running list of merged groups.

**Dry run** (`mode="max_count"`, intervals = `[[1,3],[2,4],[3,5],[6,8]]`):
sorted by end → `[[1,3],[2,4],[3,5],[6,8]]` → take `[1,3]` (count=1, last_end=3) →
`[2,4]` start 2 < 3, skip → `[3,5]` start 3 ≥ 3, take (count=2, last_end=5) →
`[6,8]` start 6 ≥ 5, take (count=3). **Answer: 3.**

**Typical modifications:**
- Want min removals instead of max kept → `len(intervals) - max_count`.
- Touching intervals shouldn't merge → change `<=` to `<` in merge mode.
- Need to also track *which* intervals were picked → store indices, not just count.

---

### Template B — Heap-based "always take best available"

```python
import heapq

def template_heap_greedy(items, k=None):
    """
    items: list of candidates to process in some sorted order.
    k: optional running-size cap (deadline / capacity).
    """
    items.sort()                 # sort by whatever feasibility field matters
    heap = []
    total = 0
    for value in items:           # value = whatever you're tracking (profit, cost…)
        heapq.heappush(heap, value)
        total += value
        if k is not None and len(heap) > k:
            total -= heapq.heappop(heap)   # evict the worst kept item
    return total, heap
```

- **`heap`**: running set of "kept" choices, smallest on top for easy eviction.
- **`k`**: cap derived from the problem (deadline, capacity, "at most k items").

**Dry run** (jobs by deadline `[(1,20),(2,15),(2,10)]`, evict smallest profit when
over capacity): sort by deadline → process `(1,20)`: heap=[20], total=20, len=1 ≤
deadline 1, ok. Process `(2,15)`: heap=[15,20], total=35, len=2 ≤ 2, ok. Process
`(2,10)`: heap=[10,15,20], total=45, len=3 > deadline 2 → evict 10 → total=35,
heap=[15,20]. **Answer: 35.**

**Typical modifications:**
- Max-heap instead of min → negate values on push/pop.
- Two heaps (e.g. median maintenance) → maintain a max-heap for the lower half and
  min-heap for the upper half.
- "Connect cheapest pairs" variant → pop 2, push their sum, repeat (Section 3.5).

---

### Template C — Two-Pointer Resource Matching

```python
def template_two_pointer_match(a, b, feasible):
    """
    a, b: two arrays to pair up.
    feasible(x, y): bool, whether a[i] can satisfy b[j].
    """
    a.sort()
    b.sort()
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if feasible(a[i], b[j]):
            matched += 1
            i += 1
        j += 1
    return matched
```

**Typical modifications:**
- Maximize sum of matched pairs instead of count → accumulate `min(a[i], b[j])` or
  similar at each match.
- Two-sided constraint (boats: max 2 people + weight limit) → use `left`/`right`
  pointers from opposite ends instead of both advancing forward.

---

### Template D — Monotonic Stack

```python
def template_monotonic_stack(seq, keep_smaller=True):
    """
    keep_smaller=True  -> pop while top is LARGER than current (build smallest result)
    keep_smaller=False -> pop while top is SMALLER than current (build largest result)
    """
    stack = []
    for x in seq:
        while stack and ((stack[-1] > x) if keep_smaller else (stack[-1] < x)):
            stack.pop()
        stack.append(x)
    return stack
```

**Typical modifications:**
- Add a removal budget `k` (Remove K Digits) → only pop while `k > 0`, decrement
  `k` on each pop, trim leftovers from the end if `k` remains.
- Track indices instead of values when you need positions (Next Greater Element).

---

### Template E — Union-Find (for Kruskal / connectivity greedy)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.rank[rx] += self.rank[rx] == self.rank[ry]
        return True
```

**Typical modifications:** add a `count` field tracking number of components for
"number of connected components" style problems.

---

## 5. Common Interview Problems

> Format: **one-liner** · greedy insight · template used · tweak from the generic
> template (small snippet only — full solution is just the template + the tweak).

### Sort + Traverse family — *Template: none needed, raw sort + sweep*

| Problem | Insight | Tweak |
|---|---|---|
| LC 455 Assign Cookies | Match smallest sufficient cookie to smallest greed | `Template C`, `feasible = lambda c, g: c >= g` |
| LC 1005 Max Sum After K Negations | Negate the most negative numbers first (sort ascending, flip while `k>0` and `nums[i]<0`) | After flips, if `k` is odd left over, flip the smallest absolute value once more |
| LC 561 Array Partition | Sort ascending, sum every even-indexed element | One-liner: `sum(sorted(nums)[::2])` |
| LC 942 DI String Match | Maintain `lo, hi` pointers; on `'I'` emit `lo++`, on `'D'` emit `hi--` | No heap/stack needed — pure two-pointer construction |
| LC 670 Maximum Swap | Greedily swap current digit with the **largest digit to its right** if it's bigger | Precompute "last occurrence of each digit" array first |

### Interval family — *Template A*

| Problem | Insight | Tweak |
|---|---|---|
| GFG Activity Selection | Exact Template A, `mode="max_count"` | None — direct application |
| LC 435 Non-overlapping Intervals | Answer = `n - max_count` | `return len(intervals) - template_interval(intervals, "max_count")` |
| LC 452 Min Arrows to Burst Balloons | Same as max_count, but "taking" an interval means reusing the arrow if it still overlaps | Track `arrow_pos = end` instead of discrete count comparisons; only increment arrows when `start > arrow_pos` |
| GFG Merge Overlapping Intervals | Exact Template A, `mode="merge"` | None |
| LC 763 Partition Labels | Track last occurrence of each char; extend current partition end while scanning inside it | Not interval input directly — first build `last_seen` map, then it's Template A's merge logic over implicit intervals |
| LC 1024 Video Stitching | Sort by start; greedily extend reachable end like a jump-game / interval merge hybrid | Track `current_end, next_end`; if `start > current_end` and `next_end <= current_end`, fail |

### Heap family — *Template B*

| Problem | Insight | Tweak |
|---|---|---|
| LC 1167 Connect Sticks at Min Cost | Always merge two cheapest | Direct application of Section 3.5's `connect_at_min_cost` |
| GFG Huffman Coding | Same merge-two-smallest idea, but also build the tree (store nodes, not just sums) | Push `(freq, node)` tuples; on merge, create parent node linking both children |
| LC 502 IPO | Among projects whose capital requirement ≤ current capital, take max-profit one | Sort projects by capital; maintain a max-heap (negate profit) of "affordable" projects; pop max profit `k` times, unlocking newly affordable projects after each capital increase |
| LC 253 Meeting Rooms II | Min heap of end times; if new meeting's start ≥ heap top, reuse the room (pop+push), else allocate a new room | `Template B` variant: heap holds *end times*, not values to sum |
| LC 1642 Furthest Building You Can Reach | Use ladders for the largest climbs, bricks for the rest; min-heap of "ladder-spent" climbs, swap out smallest if you run out of ladders | Track climbs that used a ladder in a min-heap of size `ladders`; if a new bigger climb appears and heap is full, evict smallest and pay bricks for it instead |
| LC 1481 Least Number of Unique Ints after K Removals | Count frequencies, sort ascending, greedily remove from smallest-frequency groups first | No heap strictly required — a sorted list of frequencies works, but a min-heap is the natural fit if frequencies stream in |

### Resource Allocation family — *Template C*

| Problem | Insight | Tweak |
|---|---|---|
| LC 881 Boats to Save People | Pair lightest with heaviest if they fit together, else heaviest goes alone | Two pointers from **opposite ends**, not both advancing forward — see Template C's two-sided note |
| LC 1710 Maximum Units on a Truck | Sort boxes by units-per-box descending, fill truck greedily | `Template A` (sort+traverse) style, not matching — take full boxes of the best type until truck or supply runs out |
| LC 826 Most Profit Assigning Work | Sort jobs by difficulty, workers by ability; for each worker, take the best profit among jobs they can do (running max as difficulty threshold rises) | Sort jobs by difficulty, precompute running-max profit array, binary-search/two-pointer per worker |

### Stack family — *Template D*

| Problem | Insight | Tweak |
|---|---|---|
| LC 402 Remove K Digits | Build smallest number, popping bigger predecessors while budget remains | Exact `Template D`, `keep_smaller=True`, with the `k`-budget addition from Template D's notes |
| LC 316 Remove Duplicate Letters | Like Remove K Digits, but can only pop a char if it reappears later, and never remove the last occurrence of any letter | Add a guard: only pop if `last_occurrence[stack[-1]] > current_index` |
| LC 135 Candy | Two passes: left→right enforce "greater rating than left neighbor gets +1 candy," right→left enforce the same from the right, take the max of both passes per index | Not a literal stack — but the "discard dominated info, keep only what's still relevant" idea is identical; implemented as two linear sweeps instead of an explicit stack |
| LC 738 Monotone Increasing Digits | Scan right to left; whenever `digits[i-1] > digits[i]`, decrement `digits[i-1]` and mark everything after as `9` | Direct stack-style pop-and-fix sweep done in place on the digit array |

### Two-Pointer family — *Template C (sub-variant) / Section 3.8*

| Problem | Insight | Tweak |
|---|---|---|
| LC 11 Container With Most Water | Move the pointer at the shorter wall inward | Exact Section 3.8 template |
| LC 680 Valid Palindrome II | On mismatch, try skipping left or right char (one allowed deletion), check both branches | Two-pointer + a single allowed "skip" — branch into two two-pointer checks |

### Graph Greedy family — *Section 3.9 templates*

| Problem | Insight | Tweak |
|---|---|---|
| GFG Kruskal's MST | Exact `kruskal()` template | None |
| GFG Prim's MST | Exact `prim()` template | None |
| GFG Dijkstra's Shortest Path | Exact `dijkstra()` template | None |
| GFG Min Cost to Connect All Cities | Build edge list from all city pairs (or given edges), run Kruskal | Edge list construction is the only addition before calling `kruskal()` |

---

## 6. Final Reference Pages

### 6.1 Pattern → When to use

| Pattern | Use when |
|---|---|
| Sort + Traverse | A single sorted pass directly yields the answer; no extra data structure needed |
| Interval Scheduling | Need max count of non-overlapping intervals |
| Interval Merge | Need the union / merged form of overlapping ranges |
| Earliest Finish Time | Matching two sequences under an ordering/feasibility constraint |
| Greedy + Heap | Need running min/max of a changing candidate set; "always take current best" |
| Resource Allocation | Bipartite matching under one feasibility inequality between two sorted arrays |
| Greedy + Stack | Earlier elements become permanently irrelevant once dominated by a later one |
| Greedy + Two Pointers | Sorted array(s), trade-off moves monotonically as pointers move |
| Graph Greedy | MST (Kruskal/Prim) or shortest path with non-negative weights (Dijkstra) |

### 6.2 Problem clue → Pattern

| Clue in problem statement | Pattern |
|---|---|
| "maximum number of non-overlapping" | Interval Scheduling |
| "merge", "free time", "cover the range" | Interval Merge |
| "top k", "kth largest/smallest", "minimum cost to connect/merge" | Greedy + Heap |
| "schedule with deadline and profit" | Greedy + Heap (deadline-feasibility) |
| "next greater/smaller", "remove k digits", "candies" | Greedy + Stack |
| "container", "two pointers", "boats" | Greedy + Two Pointers |
| "assign workers/cookies/jobs to tasks" | Resource Allocation |
| "minimum spanning tree", "connect all cities cheaply" | Kruskal / Prim |
| "shortest path, non-negative weights" | Dijkstra |
| "ratio", "negate to maximize/minimize sum" | Sort + Traverse |

### 6.3 Complexity table

| Pattern | Typical complexity |
|---|---|
| Sort + Traverse | O(n log n) |
| Interval Scheduling / Merge | O(n log n) |
| Greedy + Heap | O(n log n) |
| Resource Allocation (two-pointer) | O(n log n) |
| Greedy + Stack | O(n) amortized |
| Greedy + Two Pointers | O(n) |
| Kruskal's MST | O(E log E) |
| Prim's MST | O(E log V) |
| Dijkstra | O(E log V) |

### 6.4 Common mistakes (across all patterns)

- Sorting by the wrong key (start vs end, ascending vs descending) — always test on
  a tiny 2-3 element example before committing.
- Forgetting tie-breaking rules when two sort keys are equal.
- Using a min-heap where a max-heap (negated values) was needed, or vice versa.
- Forgetting to flush leftover budget at the end of a monotonic-stack pass
  (e.g. Remove K Digits).
- Applying Dijkstra to graphs with negative edge weights.
- Forgetting path compression / union by rank in Union-Find, tanking Kruskal's
  performance.
- Assuming a locally-greedy idea is correct without trying to break it with a small
  counterexample first.
- Off-by-one errors on `<` vs `<=` when intervals touch but don't overlap.

### 6.5 Interview checklist (read before you start coding)

1. Restate the problem; identify what's being maximized/minimized.
2. Run the **Greedy vs DP gut check** (Section 1.5) — try to break a greedy idea
   with a 3-4 element counterexample.
3. Identify the **sort key** (or confirm no sort is needed).
4. Pick the **pattern** using the decision tree in Section 2.
5. State the **exchange argument** out loud in one sentence before coding.
6. Write the loop; track exactly what invariant the heap/stack/pointer maintains.
7. Dry-run on the example given, then on an edge case (empty input, all-equal
   elements, single element).
8. State final complexity.

### 6.6 One-page Greedy decision flowchart

```
                         ┌────────────────────────┐
                         │   Read the problem.     │
                         │ What's max/min'd?        │
                         └────────────┬────────────┘
                                      │
                     Can a sort key make the choice obvious?
                                      │
                 ┌────────────────────┼────────────────────┐
                NO                                         YES
                 │                                          │
   Does optimality depend on              Try a 3-4 element counterexample.
   combining many prior states?                            │
                 │                              ┌───────────┴───────────┐
                YES                          Breaks?                 Holds?
                 │                              │                       │
               → DP                           → DP /                  → GREEDY
                                              refine sort key            │
                                                                ┌────────┴─────────┐
                                                          Intervals?          Not intervals?
                                                                │                    │
                                                  ┌─────────────┴──────┐    ┌────────┴─────────┐
                                            max count?            merge/union?   Running best of a
                                                │                      │         changing set?
                                          Interval Scheduling   Interval Merge        │
                                          (sort by END)         (sort by START)      YES → Heap
                                                                                       │
                                                                                      NO
                                                                                       │
                                                                          Earlier elements become
                                                                          permanently irrelevant?
                                                                                       │
                                                                              YES → Stack
                                                                                       │
                                                                                      NO
                                                                                       │
                                                                          Two sorted sequences to
                                                                          pair/scan together?
                                                                                       │
                                                                              YES → Two Pointers
                                                                                       │
                                                                                      NO
                                                                                       │
                                                                          Graph: connect cheaply /
                                                                          shortest path?
                                                                                       │
                                                                          MST → Kruskal/Prim
                                                                          Shortest path → Dijkstra
```

---

*End of handbook. Revise Sections 2 and 6 first if short on time — they're the
fastest path back into pattern-recognition mode before an interview.*
