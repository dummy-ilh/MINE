Alright, let's build this from the ground up. I'll cover every data structure you'll actually use for LeetCode/NeetCode, with the syntax that trips people up (enumerate, permutations, heapq, deque) explained in detail.

---

## 1. Lists (Arrays)

Lists are your default array type in Python. Dynamic size, mixed types allowed (but you'll use them as arrays).

```python
arr = [5, 3, 8, 1]

arr.append(9)          # add to end -> [5,3,8,1,9]
arr.pop()               # remove & return last -> 9
arr.pop(0)              # remove & return index 0 -> 5
arr.insert(1, 100)      # insert 100 at index 1
arr.remove(8)           # removes FIRST value equal to 8 (not index!)
arr[0]                  # access
arr[-1]                 # last element
arr[1:3]                # slice -> elements at index 1,2 (end excluded)
arr[::-1]               # reversed copy
len(arr)
arr.sort()              # in-place ascending
arr.sort(reverse=True)  # descending
sorted(arr)             # returns NEW sorted list, doesn't mutate
arr.sort(key=lambda x: -x)   # custom sort key
```

### `enumerate()` — the thing you asked about
Gives you **(index, value)** pairs while looping. Avoids manual counters.

```python
arr = ["a", "b", "c"]

for i, val in enumerate(arr):
    print(i, val)
# 0 a
# 1 b
# 2 c

for i, val in enumerate(arr, start=1):   # start counting from 1
    print(i, val)
```

Why it matters: any time you need both index and value (e.g. "find pairs that sum to target and return indices"), use `enumerate` instead of `range(len(arr))`.

### List comprehensions (used constantly)
```python
squares = [x*x for x in range(5)]                  # [0,1,4,9,16]
evens = [x for x in range(10) if x % 2 == 0]        # filter
pairs = [(i,j) for i in range(3) for j in range(3)] # nested loops
```

---

## 2. Strings

Strings are immutable — every "modification" makes a new string.

```python
s = "hello world"
s.split()            # ['hello', 'world']  (splits on whitespace by default)
s.split(",")         # split on custom delimiter
"-".join(["a","b"])  # "a-b"  (join list into string)
s.upper(); s.lower()
s.strip()            # remove leading/trailing whitespace
s[::-1]              # reverse a string
list(s)              # convert to list of chars (so you CAN mutate)
"".join(char_list)   # convert list of chars back to string
ord('a')             # 97 -> char to int
chr(97)              # 'a' -> int to char
s.isalpha(), s.isdigit(), s.isalnum()
```

---

## 3. Tuples

Immutable, ordered. Used a lot as dict keys, or to return multiple values.

```python
t = (1, 2)
x, y = t              # unpacking
d = {(1,2): "value"}   # tuples are hashable -> can be dict keys (lists can't!)
```

---

## 4. Dictionaries (Hash Maps) — the #1 tool

```python
d = {}
d["a"] = 1
d.get("a")             # 1
d.get("z", 0)           # returns 0 if key missing (no KeyError)
"a" in d                # True/False membership check — O(1)
del d["a"]
d.keys(); d.values(); d.items()
for k, v in d.items():
    print(k, v)
#create
d = {"name": "Alice", "age": 25, "score": 90}
d = dict(name="Alice", age=25)  # alternate way
d["city"] = "Delhi"        # add new key
#add update
d["age"] = 26              # update existing key
d.update({"age": 26, "city": "Delhi"})  # update multiple
#search
d["name"]            # direct access (KeyError if missing)
d.get("name")        # safe access → returns None if missing
d.get("xyz", "N/A")  # default value if key not found
"age" in d           # check if key exists → True/False
del d["age"]          # delete by key
d.pop("age")          # delete + returns value
d.pop("age", None)    # safe pop (no error if missing)
d.popitem()           # removes last inserted key-value pair
d.clear()             # empty the whole dict
#Loop
for key in d:
    print(key, d[key])

for key, val in d.items():   # best way
    print(key, val)

for key in d.keys():   print(key)
for val in d.values(): print(val)
d = {"b": 3, "a": 1, "c": 2}

# Sort by value → ascending
sorted_d = dict(sorted(d.items(), key=lambda x: x[1]))
# {'a': 1, 'c': 2, 'b': 3}

# Sort by value → descending
sorted_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
# {'b': 3, 'c': 2, 'a': 1}

# Sort by key
sorted_d = dict(sorted(d.items()))
d = {"a": 10, "b": 3, "c": 7, "d": 1}

# Keep only values > 5
filtered = {k: v for k, v in d.items() if v > 5}
# {'a': 10, 'c': 7}

# Keep only specific keys
keys_to_keep = ["a", "c"]
filtered = {k: v for k, v in d.items() if k in keys_to_keep}
len(d)              # number of keys
d.copy()            # shallow copy

# Merge two dicts (Python 3.9+)
d3 = d1 | d2

# Merge older way
d3 = {**d1, **d2}

# Set default (adds key only if not present)
d.setdefault("age", 0)
```

### `collections.defaultdict` — auto-initializes missing keys
```python
from collections import defaultdict

graph = defaultdict(list)
graph["A"].append("B")   # no KeyError even though "A" wasn't there yet

count = defaultdict(int)
count["x"] += 1          # defaults to 0, so this becomes 1
```

### `collections.Counter` — frequency counting in one line
```python
from collections import Counter

c = Counter("aabbbc")        # Counter({'b': 3, 'a': 2, 'c': 1})
c = Counter([1,1,2,3,3,3])
c.most_common(2)             # top 2 most frequent -> [(3,3),(1,2)]
c1 - c2                       # subtract counts (used in anagram diff problems)
```

---

## 5. Sets

Unordered, unique elements, O(1) membership check.

```python
s = set()
s.add(5)
s.discard(5)           # remove, no error if missing
5 in s                  # O(1) lookup
s1 | s2                 # union
s1 & s2                 # intersection
s1 - s2                 # difference
set([1,2,2,3])          # dedupe a list -> {1,2,3}
```

---

## 6. Stacks

Just use a **list** — `append()` and `pop()` are O(1) from the end.

```python
stack = []
stack.append(1)
stack.append(2)
stack.pop()      # 2 (LIFO)
stack[-1]        # peek without removing
```

---

## 7. `collections.deque` — the one you asked about

A **double-ended queue**: O(1) append/pop from BOTH ends (a plain list is O(n) for `pop(0)` / `insert(0, x)`). Use this for queues (BFS!) and sliding window problems.

```python
from collections import deque

dq = deque([1,2,3])

dq.append(4)        # add to right -> [1,2,3,4]
dq.appendleft(0)     # add to left  -> [0,1,2,3,4]
dq.pop()             # remove & return right -> 4
dq.popleft()         # remove & return left -> 0   <-- this is your queue.dequeue()

dq.rotate(1)         # rotate right by 1
```

**BFS pattern (this is 90% of why you use deque):**
```python
queue = deque([start_node])
visited = {start_node}

while queue:
    node = queue.popleft()     # FIFO: process oldest first
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

---

## 8. `heapq` — heaps / priority queues

Python only has a **min-heap** built in. It operates directly on a regular list.

```python
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)

heapq.heappop(heap)      # 1 -> always removes the SMALLEST
heap[0]                   # peek at smallest without removing

heapq.heapify(existing_list)   # convert a list into a heap in-place, O(n)
```

### Max-heap trick (memorize this — comes up constantly)
Python has no max-heap, so **negate the values**:
```python
heap = []
heapq.heappush(heap, -5)
heapq.heappush(heap, -1)
heapq.heappush(heap, -3)

largest = -heapq.heappop(heap)   # negate again -> 5
```

### Heap of tuples (e.g. for "k closest points", Dijkstra, etc.)
Heaps compare tuples element-by-element, so put the priority value first:
```python
heap = []
heapq.heappush(heap, (distance, node))   # sorts by distance automatically
dist, node = heapq.heappop(heap)          # smallest distance popped first
```

### `heapq.nlargest` / `nsmallest` — shortcut, no manual heap needed
```python
heapq.nlargest(3, arr)     # top 3 largest values
heapq.nsmallest(3, arr)
```

---

## 9. `itertools` — permutations, combinations, etc.

```python
from itertools import permutations, combinations, product, accumulate

list(permutations([1,2,3]))
# [(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)]  -- ALL orderings

list(permutations([1,2,3], 2))
# [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]  -- all orderings of length 2

list(combinations([1,2,3], 2))
# [(1,2),(1,3),(2,3)]   -- order doesn't matter, no repeats

list(product([0,1], repeat=3))
# all 8 combos of length-3 binary strings -- [(0,0,0),(0,0,1),...]

list(accumulate([1,2,3,4]))
# [1,3,6,10]  -- running prefix sum
```

> In practice you'll often just **write your own backtracking function** for permutations (NeetCode expects this for the "Permutations" problem), but `itertools.permutations` is the fast way to get the same result when you just need the data.

**Manual permutations via backtracking (what NeetCode wants you to know):**
```python
def permute(nums):
    res = []
    def backtrack(path, remaining):
        if not remaining:
            res.append(path[:])
            return
        for i in range(len(remaining)):
            backtrack(path + [remaining[i]], remaining[:i] + remaining[i+1:])
    backtrack([], nums)
    return res
```

---

## 10. Linked Lists

No built-in type — you define a node class. LeetCode gives you this class already in problems.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# traverse
curr = head
while curr:
    print(curr.val)
    curr = curr.next

# reverse a linked list (classic pattern)
def reverse(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

---

## 11. Trees

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# DFS recursive
def dfs(node):
    if not node:
        return
    dfs(node.left)
    print(node.val)
    dfs(node.right)

# BFS (level order) using deque
def bfs(root):
    q = deque([root])
    while q:
        node = q.popleft()
        print(node.val)
        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
```

---

## 12. Graphs

Usually represented as adjacency list via `defaultdict(list)`:
```python
graph = defaultdict(list)
graph[0].append(1)   # edge 0 -> 1
graph[1].append(0)   # if undirected, add both directions
```
Then run BFS (deque, above) or DFS (recursion or stack) on it.

---

## 13. `bisect` — binary search on sorted lists (bonus)

```python
import bisect

arr = [1,3,5,7]
bisect.bisect_left(arr, 5)    # index where 5 would go, leftmost -> 2
bisect.bisect_right(arr, 5)   # rightmost -> 3
bisect.insort(arr, 4)         # insert 4 keeping arr sorted
```

---

## Checking frequency dicts

```python
freq = {}
for ch in "aabbbc":
    freq[ch] = freq.get(ch, 0) + 1     # safe increment, no KeyError

# checking
"a" in freq          # True — key exists
freq["a"]             # 2 — get count (KeyError if missing)
freq.get("z")         # None if missing
freq.get("z", 0)      # 0 if missing — safest for counting logic

# checking if two freq dicts are identical (anagram check)
freq1 == freq2        # dicts support direct equality comparison

# checking if all counts satisfy a condition
all(v % 2 == 0 for v in freq.values())     # e.g. "can form palindrome" check

# find the max/min frequency key
max(freq, key=freq.get)     # key with highest count
min(freq, key=freq.get)     # key with lowest count

# using Counter (does all of the above cleaner)
from collections import Counter
c1, c2 = Counter(s1), Counter(s2)
c1 == c2                      # anagram check in one line
```

---

## Pattern-recognition ("leanings") by category

### Greedy
- Smell: "maximize/minimize X", "at each step pick the best local option", words like "minimum number of," "maximum profit."
- Core idea: sort first (usually), then make the locally optimal choice and never look back — no backtracking needed.
- Common tells: interval scheduling (sort by end time), jump game (track farthest reachable), stock problems (track min-so-far).
- Template feel:
```python
arr.sort(key=lambda x: x[1])   # sort by whatever makes greedy choice obvious
result = 0
for item in arr:
    if <locally best condition>:
        result += 1  # or update running state
```
- Gotcha: greedy only works when the problem has the "greedy choice property" — if you find yourself needing to reconsider past choices, it's not greedy, it's DP.

### Graphs
- Smell: "connections," "network," "islands," "shortest path," "can you reach," "dependencies" (course schedule = topological sort).
- Decide representation first: adjacency list (`defaultdict(list)`) almost always, adjacency matrix only if dense/small.
- Decide traversal:
  - **Unweighted shortest path / level order** → BFS with `deque`.
  - **"Does a path exist" / connected components / islands** → DFS (recursion or explicit stack) — order doesn't matter, just need to visit everything.
  - **Weighted shortest path** → Dijkstra with `heapq` (push `(dist, node)`).
  - **"Order of tasks with dependencies"** → topological sort (BFS with in-degree counting, aka Kahn's algorithm).
  - **Cycle detection** → DFS with a "visiting/visited" 3-color state, or in-degree count reaching 0 for all nodes (topo sort).
- Always track `visited = set()` — the #1 bug source is re-visiting nodes and infinite-looping.

### Backtracking
- Smell: "all possible," "all subsets/permutations/combinations," "valid arrangements" (N-Queens, Sudoku, word search).
- Template feel:
```python
def backtrack(path, choices):
    if <base case>:
        res.append(path[:])   # copy! path is mutated later
        return
    for choice in choices:
        path.append(choice)          # choose
        backtrack(path, new_choices)  # explore
        path.pop()                    # un-choose (the "backtrack")
```
- Gotcha: always undo the choice after recursing — that's the whole trick.

### Dynamic Programming
- Smell: "count the number of ways," "min/max cost to reach," overlapping subproblems, or greedy "feels wrong" because earlier choices affect later options.
- Ask: can I define `dp[i]` = answer using only first `i` elements, and express `dp[i]` in terms of `dp[i-1]`, `dp[i-2]`, etc.?
- Start with brute-force recursion + memo (`@lru_cache` or a dict), then convert to bottom-up table if needed.
```python
from functools import lru_cache

@lru_cache(None)
def dp(i):
    if <base case>: return ...
    return dp(i-1) + dp(i-2)   # or whatever the recurrence is
```
## Core algorithm templates to have memorized

### Binary Search
```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```
- Gotcha: `mid = (lo+hi)//2` can overflow in other languages, not an issue in Python. Watch off-by-one on `lo <= hi` vs `lo < hi` depending on if you want "found" vs "insertion point."
- Variant: binary search on **answer space** (e.g. "minimum capacity to ship in D days") — search over possible answers, not array indices.

### Sliding Window (variable size)
```python
def longest_substring_with_condition(s):
    left = 0
    freq = {}
    best = 0
    for right in range(len(s)):
        freq[s[right]] = freq.get(s[right], 0) + 1
        while <window invalid>:          # shrink while condition breaks
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best
```

### Prefix Sum
```python
prefix = [0] * (len(arr) + 1)
for i, x in enumerate(arr):
    prefix[i+1] = prefix[i] + x

# sum of arr[i:j] (inclusive i, exclusive j) in O(1):
range_sum = prefix[j] - prefix[i]
```
- Used for "subarray sum equals K" combined with a dict of `{prefix_sum: count}`.

### Fast & Slow Pointers (cycle detection)
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```
- Also used for "find middle of linked list" (slow ends at middle when fast hits end).

### DFS (graph/tree, iterative with explicit stack)
```python
def dfs(start):
    stack = [start]
    visited = {start}
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
```

### BFS (shortest path, unweighted)
```python
def bfs(start):
    queue = deque([(start, 0)])   # (node, distance)
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
```

### Dijkstra (weighted shortest path)
```python
def dijkstra(start, n):
    dist = {i: float('inf') for i in range(n)}
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue        # stale entry, skip
        for neighbor, weight in graph[node]:
            nd = d + weight
            if nd < dist[neighbor]:
                dist[neighbor] = nd
                heapq.heappush(heap, (nd, neighbor))
    return dist
```

### Topological Sort (Kahn's algorithm, BFS-based)
```python
def topo_sort(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:            # u -> v
        graph[u].append(v)
        indegree[v] += 1

    queue = deque([i for i in range(n) if indegree[i] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == n else []   # empty = cycle exists
```

### Union-Find / Disjoint Set (connectivity, cycle detection in undirected graphs)
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False     # already connected -> would form a cycle
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```
- Comes up in: "number of connected components," "redundant connection," Kruskal's MST.

### Quickselect (kth largest/smallest, O(n) average)
```python
import random

def quickselect(arr, k):   # kth smallest, 1-indexed
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    if k <= len(left):
        return quickselect(left, k)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))
```
- Alternative to sorting or heaps for "kth largest element" — average O(n) vs O(n log n).

### Merge Sort (stable, O(n log n), good to know for "merge intervals"/"count inversions")
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]
```

---

### One-line complexity cheat sheet (interviewers love hearing this stated out loud)

| Structure/Op | Time |
|---|---|
| list append/pop (end) | O(1) |
| list insert/pop (start/middle) | O(n) |
| dict/set get, insert, in-check | O(1) avg |
| deque append/pop (both ends) | O(1) |
| heapq push/pop | O(log n) |
| sort() | O(n log n) |
| binary search | O(log n) |
| BFS/DFS | O(V + E) |

Want the Kruskal/Prim MST templates too, or is that enough graph coverage for now?
