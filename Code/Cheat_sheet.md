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

### Quick cheat-sheet of "when to use what"

| Problem smells like... | Use |
|---|---|
| "find pair/duplicate/frequency" | dict / Counter / set |
| "process in order, need index+value" | enumerate |
| "shortest path, level order, BFS" | deque |
| "top-k, kth largest, priority" | heapq |
| "all orderings/subsets" | backtracking (or itertools) |
| "need both ends fast" | deque |
| "sorted array search" | bisect |

Want me to go deeper on any one of these (e.g. a full backtracking template, or Dijkstra with heapq), or move on to time/space complexity next?
