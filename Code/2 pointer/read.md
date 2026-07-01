### 6. LC 19 — Remove Nth Node From End of List
Remove the nth node from the end of a linked list in one pass.
```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    slow = fast = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```
### 8. LC 61 — Rotate List
Rotate a linked list to the right by k places.
```python
def rotateRight(head, k):
    if not head:
        return head
    n = 1
    tail = head
    while tail.next:
        tail = tail.next
        n += 1
    k %= n
    if k == 0:
        return head
    tail.next = head
    steps = n - k
    new_tail = head
    for _ in range(steps - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head
```

### 12. LC 86 — Partition List
Partition a linked list so all nodes < x come before nodes >= x.
```python
def partition(head, x):
    before = before_head = ListNode(0)
    after = after_head = ListNode(0)
    while head:
        if head.val < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next
    after.next = None
    before.next = after_head.next
    return before_head.next
```

### 13. LC 142 — Linked List Cycle II
Find the node where a cycle begins (or None).
```python
def detectCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None
```

### 14. LC 143 — Reorder List
Reorder a linked list into L0→Ln→L1→Ln-1→...
```python
def reorderList(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    second = slow.next
    slow.next = None
    prev = None
    while second:
        nxt = second.next
        second.next = prev
        prev = second
        second = nxt
    first, second = head, prev
    while second:
        t1, t2 = first.next, second.next
        first.next = second
        second.next = t1
        first, second = t1, t2
```

### 15. LC 148 — Sort List
Sort a linked list in O(n log n) using merge sort.
```python
def sortList(head):
    if not head or not head.next:
        return head
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None
    left = sortList(head)
    right = sortList(mid)

    dummy = ListNode()
    tail = dummy
    while left and right:
        if left.val <= right.val:
            tail.next, left = left, left.next
        else:
            tail.next, right = right, right.next
        tail = tail.next
    tail.next = left or right
    return dummy.next
```

### 22. LC 244 🔒 — Shortest Word Distance II
Design a class returning shortest distance between two words across many queries.
```python
class WordDistance:
    def __init__(self, wordsDict):
        self.locations = {}
        for i, w in enumerate(wordsDict):
            self.locations.setdefault(w, []).append(i)

    def shortest(self, word1, word2):
        loc1, loc2 = self.locations[word1], self.locations[word2]
        i, j, best = 0, 0, float('inf')
        while i < len(loc1) and j < len(loc2):
            best = min(best, abs(loc1[i] - loc2[j]))
            if loc1[i] < loc2[j]:
                i += 1
            else:
                j += 1
        return best
```

### 23. LC 251 🔒 — Flatten 2D Vector
Design an iterator that flattens a 2D vector.
```python
class Vector2D:
    def __init__(self, vec):
        self.data = [x for row in vec for x in row]
        self.index = 0

    def next(self):
        val = self.data[self.index]
        self.index += 1
        return val

    def hasNext(self):
        return self.index < len(self.data)
```

### 63. LC 1214 🔒 — Two Sum BSTs
Check if a node from each of two BSTs sums to a target.
```python
def twoSumBSTs(root1, root2, target):
    def inorder(node, arr):
        if node:
            inorder(node.left, arr)
            arr.append(node.val)
            inorder(node.right, arr)
    list1, list2 = [], []
    inorder(root1, list1)
    inorder(root2, list2)
    l, r = 0, len(list2) - 1
    while l < len(list1) and r >= 0:
        s = list1[l] + list2[r]
        if s == target:
            return True
        elif s < target:
            l += 1
        else:
            r -= 1
    return False
```

### 70. LC 1570 🔒 — Dot Product of Two Sparse Vectors
Design a class for fast dot products of sparse vectors.
```python
class SparseVector:
    def __init__(self, nums):
        self.pairs = [(i, v) for i, v in enumerate(nums) if v != 0]

    def dotProduct(self, vec):
        i, j, result = 0, 0, 0
        while i < len(self.pairs) and j < len(vec.pairs):
            if self.pairs[i][0] == vec.pairs[j][0]:
                result += self.pairs[i][1] * vec.pairs[j][1]
                i += 1
                j += 1
            elif self.pairs[i][0] < vec.pairs[j][0]:
                i += 1
            else:
                j += 1
        return result
```


### 75. LC 1650 🔒 — Lowest Common Ancestor of a Binary Tree III
Find LCA of two nodes that have a `.parent` pointer (no root given).
```python
def lowestCommonAncestor(p, q):
    a, b = p, q
    while a != b:
        a = a.parent if a.parent else q
        b = b.parent if b.parent else p
    return a
```

### 78. LC 1721 — Swapping Nodes in a Linked List
Swap the kth node from the start with the kth node from the end.
```python
def swapNodes(head, k):
    first = head
    for _ in range(k - 1):
        first = first.next
    second = head
    curr = first
    while curr.next:
        curr = curr.next
        second = second.next
    first.val, second.val = second.val, first.val
    return head
```


### 92. LC 2046 🔒 — Sort Linked List Already Sorted Using Absolute Values
A linked list sorted by absolute value — restore true ascending order.
```python
def sortLinkedList(head):
    prev = head
    curr = head.next
    while curr:
        if curr.val < 0:
            prev.next = curr.next
            curr.next = head
            head = curr
            curr = prev.next
        else:
            prev = curr
            curr = curr.next
    return head
```

### 93. LC 2095 — Delete the Middle Node of a Linked List
Delete the middle node of a linked list.
```python
def deleteMiddle(head):
    if not head or not head.next:
        return None
    slow = head
    fast = head.next.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    slow.next = slow.next.next
    return head
```

### 102. LC 2332 ⚠️ — The Latest Time to Catch a Bus
Find the latest time you (not already a passenger) can board a bus.
```python
def latestTimeCatchTheBus(buses, passengers, capacity):
    buses.sort()
    passengers.sort()
    p_idx, n, cap_left = 0, len(passengers), 0
    for bus in buses:
        cap_left = capacity
        while p_idx < n and passengers[p_idx] <= bus and cap_left > 0:
            p_idx += 1
            cap_left -= 1

    candidate = buses[-1] if cap_left > 0 else passengers[p_idx - 1]
    passenger_set = set(passengers)
    while candidate in passenger_set:
        candidate -= 1
    return candidate
```

### 103. LC 2337 — Move Pieces to Obtain a String
Check if `start` can be transformed into `target` by sliding L/R pieces over blanks.
```python
def canChange(start, target):
    s = [(c, i) for i, c in enumerate(start) if c != '_']
    t = [(c, i) for i, c in enumerate(target) if c != '_']
    if len(s) != len(t):
        return False
    for (c1, i1), (c2, i2) in zip(s, t):
        if c1 != c2:
            return False
        if c1 == 'L' and i1 < i2:
            return False
        if c1 == 'R' and i1 > i2:
            return False
    return True
```

### 104. LC 2396 — Strictly Palindromic Number
Check if n is a palindrome in every base from 2 to n-2 (trick: always False).
```python
def isStrictlyPalindromic(n):
    return False
```

### 105. LC 2406 — Divide Intervals Into Minimum Number of Groups
Find the minimum groups needed so no two intervals in the same group overlap.
```python
import heapq

def minGroups(intervals):
    intervals.sort()
    heap = []
    for start, end in intervals:
        if heap and heap[0] < start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)
```

---

**On the ⚠️ entries:** these (777, 786, 825, 1055, 1574, 1712, 1850, 1989, 2332) have subtle edge cases — multi-character runs, off-by-one boundaries, or simulation details that are easy to get slightly wrong. Treat the code as a strong starting point, but trace through the examples on the actual problem page before trusting it fully.
