
### 8. LC 170 🔒 — Two Sum III - Data Structure Design
Design a class that supports adding numbers and checking if any pair sums to a target.
```python
class TwoSum:
    def __init__(self):
        self.counts = {}

    def add(self, number):
        self.counts[number] = self.counts.get(number, 0) + 1

    def find(self, value):
        for num in self.counts:
            complement = value - num
            if complement == num:
                if self.counts[num] > 1:
                    return True
            elif complement in self.counts:
                return True
        return False
```

### 6. LC 141 — Linked List Cycle
Detect if a linked list has a cycle.
```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### 7. LC 160 — Intersection of Two Linked Lists
Find the node where two linked lists intersect.
```python
def getIntersectionNode(headA, headB):
    a, b = headA, headB
    while a is not b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a
```

### 10. LC 234 — Palindrome Linked List
Check if a linked list reads the same forward and backward.
```python
def isPalindrome(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt

    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True
```
### 28. LC 876 — Middle of the Linked List
Return the middle node of a linked list.
```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```
