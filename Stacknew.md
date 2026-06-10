# PYTHON STACKS — MASTER CHEAT SHEET
### 10 Templates · 54 LeetCode Questions · Google & Apple Focused

---

## THE BASICS

A stack is **Last In, First Out (LIFO)**.
Think of a pile of plates — you always add and remove from the TOP.

```python
stack = []

stack.append(10)    # PUSH — add to top
stack.append(20)
stack.append(30)

stack.pop()         # POP — remove from top → returns 30
stack[-1]           # PEEK — see top without removing → 20
len(stack) == 0     # IS EMPTY? → False
len(stack)          # SIZE → 2
```

| Operation | Code | Time |
|---|---|---|
| Push | `stack.append(x)` | O(1) |
| Pop | `stack.pop()` | O(1) |
| Peek | `stack[-1]` | O(1) |
| Is Empty | `len(stack) == 0` | O(1) |

---

## PATTERN RECOGNITION — FIND YOUR TEMPLATE FAST

| If the problem says... | Use |
|---|---|
| Brackets valid / matching / balanced | T4 |
| Next greater element / warmer day / stock span | T2 |
| Next smaller / final price after discount | T3 |
| Largest rectangle / trapping rain water | T3 or T7 |
| Evaluate expression / decode string | T5 |
| Get min or max in O(1) | T6 |
| Remove adjacent duplicates / backspace | T1 or T8 |
| Asteroids / collisions / directions | T9 |
| Sum of subarray mins or ranges | T10 |
| Simplify path / file system | T1 |

---

---

# TEMPLATE 1 — Push / Pop With a Condition

## The Idea
Loop through the input one character at a time.
If the current character and the top of the stack meet some condition — remove the top.
Otherwise — add the current character to the stack.
Whatever is left in the stack at the end is your answer.

## Base Template

```python
stack = []

for ch in s:
    if len(stack) != 0 and stack[-1] == ch:
        stack.pop()
    else:
        stack.append(ch)

result = ''
for ch in stack:
    result = result + ch

return result
```

## Line by Line

```python
stack = []
```
Start with an empty stack. This will hold our running result.

```python
for ch in s:
```
Go through every character in the string one at a time.

```python
    if len(stack) != 0 and stack[-1] == ch:
```
Two checks here.
First: is there anything in the stack at all? We always check this before peeking at the top.
Second: does the top of the stack equal the current character?
If both are true — we have a match — so we remove the top.

```python
        stack.pop()
```
Remove the top of the stack. Both the current char and the top cancel each other out.

```python
    else:
        stack.append(ch)
```
No match — so just add the current character to the top of the stack.

```python
result = ''
for ch in stack:
    result = result + ch
return result
```
Join everything left in the stack into a string and return it.

## Dry Run — LC 1047 `"abbaca"`

```
ch='a' → stack empty → push       → stack: ['a']
ch='b' → 'a' != 'b' → push        → stack: ['a', 'b']
ch='b' → 'b' == 'b' → pop         → stack: ['a']
ch='a' → 'a' == 'a' → pop         → stack: []
ch='c' → stack empty → push       → stack: ['c']
ch='a' → 'c' != 'a' → push        → stack: ['c', 'a']

Result: "ca" ✓
```

---

## Variations

### LC 1047 — Remove All Adjacent Duplicates
**What changes:** Nothing. This is the base template exactly.

```python
def remove_duplicates(s):
    stack = []
    for ch in s:
        if len(stack) != 0 and stack[-1] == ch:
            stack.pop()     # same char as top → cancel both
        else:
            stack.append(ch)
    result = ''
    for ch in stack:
        result = result + ch
    return result

print(remove_duplicates("abbaca"))  # "ca"
```

---

### LC 2390 — Removing Stars From a String
**What changes:** The condition. Instead of checking if top equals current, we check if current char IS a star. A star means "delete whatever is before me."

```python
def remove_stars(s):
    stack = []
    for ch in s:
        if ch == '*':
            stack.pop()       # star = delete the char before it
        else:
            stack.append(ch)  # normal char → push
    result = ''
    for ch in stack:
        result = result + ch
    return result

print(remove_stars("leet**cod*e"))  # "lecoe"
```
The condition changed from `stack[-1] == ch` to `ch == '*'`.

---

### LC 844 — Backspace String Compare
**What changes:** The trigger is `#` instead of a duplicate. Also we now do this for two strings and compare the results.

```python
def backspace_compare(s, t):

    def build(string):
        stack = []
        for ch in string:
            if ch == '#':
                if len(stack) != 0:
                    stack.pop()   # # = backspace = delete last char
            else:
                stack.append(ch)
        result = ''
        for ch in stack:
            result = result + ch
        return result

    return build(s) == build(t)

print(backspace_compare("ab#c", "ad#c"))  # True → both become "ac"
```
The condition changed from `stack[-1] == ch` to `ch == '#'`.
We also added a safety check — only pop if stack is not empty (can't backspace nothing).

---

### LC 1544 — Make The String Great
**What changes:** The condition is now two things — same letter but DIFFERENT case. Like `a` and `A` cancel each other. So we check `.lower()` equality AND that they are not the exact same character.

```python
def make_good(s):
    stack = []
    for ch in s:
        if len(stack) != 0 and stack[-1] != ch and stack[-1].lower() == ch.lower():
            stack.pop()     # same letter, different case → cancel
        else:
            stack.append(ch)
    result = ''
    for ch in stack:
        result = result + ch
    return result

print(make_good("leEeetcode"))  # "leetcode"
```
The condition changed to: top and current are same letter but different case.

---

### LC 2696 — Minimum String Length After Removing Substrings
**What changes:** The condition checks if the top of stack + current char forms a forbidden pair — either `"AB"` or `"CD"`. If yes, pop the top (the current char never gets pushed — it consumed the top).

```python
def min_length(s):
    stack = []
    for ch in s:
        if len(stack) != 0:
            pair = stack[-1] + ch
            if pair == 'AB' or pair == 'CD':
                stack.pop()       # forbidden pair found → remove top
            else:
                stack.append(ch)
        else:
            stack.append(ch)
    return len(stack)

print(min_length("ABFCACDB"))  # 2
```
The condition changed to: top + current forms a forbidden pair.

---

### LC 71 — Simplify Path ★ Google ★ Apple
**What changes:** Input is a file path, not a string of characters. We split by `/` first to get directory names. Then the condition is: if part is `..`, pop (go up one level). If part is `.` or empty, skip.

```python
def simplify_path(path):
    stack = []
    parts = path.split('/')    # split by slash first

    for part in parts:
        if part == '' or part == '.':
            continue           # skip empty parts and current dir
        elif part == '..':
            if len(stack) != 0:
                stack.pop()    # go one level up
        else:
            stack.append(part) # valid dir name → push

    return '/' + '/'.join(stack)

print(simplify_path("/home/../etc/"))   # "/etc"
print(simplify_path("/home//foo/"))     # "/home/foo"
print(simplify_path("/../"))           # "/"
```

**Dry Run → `"/home/../etc/"`**
```
split('/') → ['', 'home', '..', 'etc', '']
''    → skip
'home'→ push  → stack: ['home']
'..'  → pop   → stack: []
'etc' → push  → stack: ['etc']
''    → skip
Result: '/etc' ✓
```

---

---

# TEMPLATE 2 — Monotonic Decreasing Stack (Next Greater Element)

## The Idea
We want to find, for each element, the first element to its RIGHT that is BIGGER than it.
We keep the stack always decreasing from bottom to top.
The moment a bigger element arrives — it is the answer for everything smaller sitting in the stack.

## Base Template

```python
n = len(arr)
result = [-1] * n    # default answer is -1 (no greater element found)
stack = []           # stores INDICES

for i in range(n):
    while len(stack) != 0 and arr[stack[-1]] < arr[i]:
        index = stack[-1]
        stack.pop()
        result[index] = arr[i]
    stack.append(i)

return result
```

## Line by Line

```python
result = [-1] * n
```
Fill the result with -1. This means "no next greater element found yet." We only update when we find one.

```python
stack = []   # stores INDICES
```
Important — we store the INDEX of each element, not the element itself. This lets us update the right position in result.

```python
for i in range(n):
```
Go through each element by its index.

```python
    while len(stack) != 0 and arr[stack[-1]] < arr[i]:
```
Two checks:
Is the stack not empty? (Always check before peeking.)
Is the element at the top of the stack SMALLER than the current element?
If yes — current element `arr[i]` is the Next Greater Element for whatever is on top.

```python
        index = stack[-1]
        stack.pop()
        result[index] = arr[i]
```
Save the index at the top, remove it from the stack, then record the answer for that index.
We do this in a loop (`while`) because the current element might be the NGE for multiple elements.

```python
    stack.append(i)
```
Push the current index. It's waiting to find its own next greater element later.

> Whatever indices remain in the stack at the end have no next greater element — their result stays -1.

## Dry Run — `[4, 5, 2, 10]`

```
i=0, val=4  → stack empty → push 0        → stack: [0]
i=1, val=5  → arr[0]=4 < 5 → result[0]=5, pop
            → stack empty → push 1         → stack: [1]
i=2, val=2  → arr[1]=5 > 2 → just push    → stack: [1, 2]
i=3, val=10 → arr[2]=2 < 10 → result[2]=10, pop
            → arr[1]=5 < 10 → result[1]=10, pop
            → stack empty → push 3         → stack: [3]

End: index 3 still in stack → result[3] = -1

Result: [5, 10, 10, -1] ✓
```

---

## Variations

### LC 496 — Next Greater Element I
**What changes:** Results stored in a dict (element → NGE) instead of an array. Then look up each element of nums1 in that dict.

```python
def next_greater_element(nums1, nums2):
    stack = []
    nge = {}    # dict: element → its next greater element

    for i in range(len(nums2)):
        while len(stack) != 0 and stack[-1] < nums2[i]:
            val = stack[-1]
            stack.pop()
            nge[val] = nums2[i]       # nums2[i] is the NGE for val
        stack.append(nums2[i])        # store values not indices here

    result = []
    for num in nums1:
        if num in nge:
            result.append(nge[num])
        else:
            result.append(-1)
    return result

print(next_greater_element([4,1,2], [1,3,4,2]))  # [-1, 3, -1]
```
Changed: result is a dict. Stack stores values (not indices) since we look up by value.

---

### LC 503 — Next Greater Element II (Circular Array)
**What changes:** The array wraps around. We simulate this by looping TWICE over the array. We use `i % n` to wrap the index. We only push in the first pass.

```python
def next_greater_2(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(2 * n):          # loop twice to simulate circular
        index = i % n               # wrap around using modulo
        while len(stack) != 0 and nums[stack[-1]] < nums[index]:
            idx = stack[-1]
            stack.pop()
            result[idx] = nums[index]
        if i < n:
            stack.append(index)     # only push during the first pass

    return result

print(next_greater_2([1, 2, 1]))   # [2, -1, 2]
```
Changed: `range(2 * n)` and `index = i % n`. Push only when `i < n`.

---

### LC 739 — Daily Temperatures ★ Google
**What changes:** Instead of storing the GREATER VALUE in result, we store the NUMBER OF DAYS to wait. That is `i - index` — the gap between current index and the index in the stack.

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result = [0] * n     # 0 means no warmer day found
    stack = []

    for i in range(n):
        while len(stack) != 0 and temperatures[stack[-1]] < temperatures[i]:
            index = stack[-1]
            stack.pop()
            result[index] = i - index    # days to wait = index gap
        stack.append(i)

    return result

print(daily_temperatures([73,74,75,71,69,72,76,73]))
# [1,1,4,2,1,1,0,0]
```
Changed: `result[index] = i - index` instead of `result[index] = arr[i]`.

**Dry Run → `[73,74,75,71,69,72,76,73]`**
```
i=0,t=73 → push 0                          → stack:[0]
i=1,t=74 → 73<74 → result[0]=1-0=1, pop   → stack:[1]
i=2,t=75 → 74<75 → result[1]=2-1=1, pop   → stack:[2]
i=3,t=71 → 75>71 → push 3                 → stack:[2,3]
i=4,t=69 → 71>69 → push 4                 → stack:[2,3,4]
i=5,t=72 → pop4:result[4]=1, pop3:result[3]=2 → stack:[2,5]
i=6,t=76 → pop5:result[5]=1, pop2:result[2]=4 → stack:[6]
i=7,t=73 → 76>73 → push 7                 → stack:[6,7]
Result: [1,1,4,2,1,1,0,0] ✓
```

---

### LC 901 — Online Stock Span ★ Google ★ Apple
**What changes:** Wrapped in a class because prices come one at a time. Stack stores `(price, span)` TUPLES instead of just indices. When we pop, we ADD UP the spans — this avoids re-scanning old prices.

```python
class StockSpanner:
    def __init__(self):
        self.stack = []   # stores (price, span) pairs

    def next(self, price):
        span = 1
        while len(self.stack) != 0 and self.stack[-1][0] <= price:
            span = span + self.stack[-1][1]   # add the span of popped element
            self.stack.pop()
        self.stack.append((price, span))
        return span

s = StockSpanner()
print(s.next(100))  # 1
print(s.next(80))   # 1
print(s.next(60))   # 1
print(s.next(70))   # 2
print(s.next(75))   # 4
```
Changed: stack stores `(price, span)` tuples. We accumulate spans when popping.

---

### LC 1475 — Final Prices With Special Discount
**What changes:** We flip the comparison from `<` to `>=`. This makes it find the NEXT SMALLER OR EQUAL element (NSE) instead of NGE. The result is the original price minus the discount.

```python
def final_prices(prices):
    n = len(prices)
    result = []
    for p in prices:
        result.append(p)   # start with full prices

    stack = []

    for i in range(n):
        while len(stack) != 0 and prices[stack[-1]] >= prices[i]:
            index = stack[-1]
            stack.pop()
            result[index] = prices[index] - prices[i]   # apply discount
        stack.append(i)

    return result

print(final_prices([8,4,6,2,3]))  # [4,2,4,2,3]
```
Changed: `>=` instead of `<` (NSE instead of NGE). Result becomes `prices[index] - prices[i]`.

---

---

# TEMPLATE 3 — Monotonic Increasing Stack (Next Smaller Element)

## The Idea
Same idea as T2 but we flip the direction.
We keep the stack always INCREASING from bottom to top.
The moment a SMALLER element arrives — it is the answer for everything LARGER sitting in the stack.

## Base Template

```python
n = len(arr)
result = [-1] * n
stack = []   # stores indices

for i in range(n):
    while len(stack) != 0 and arr[stack[-1]] > arr[i]:
        index = stack[-1]
        stack.pop()
        result[index] = arr[i]    # arr[i] is the next smaller
    stack.append(i)

return result
```

## Line by Line

```python
result = [-1] * n
```
Fill with -1. We only update when we find a next smaller element.

```python
stack = []   # stores indices
```
Same as T2 — store indices so we can update the right position.

```python
    while len(stack) != 0 and arr[stack[-1]] > arr[i]:
```
The ONLY difference from T2 — we use `>` instead of `<`.
This means: pop when top is GREATER than current (looking for smaller, not greater).

```python
        result[index] = arr[i]
```
Current element is the next smaller for whatever we just popped.

> Everything else is identical to T2. The only thing that changes between T2 and T3 is the `<` vs `>` in the while condition.

## Dry Run — `[4, 2, 5, 1]`

```
i=0, val=4  → stack empty → push 0         → stack: [0]
i=1, val=2  → arr[0]=4 > 2 → result[0]=2, pop
            → stack empty → push 1          → stack: [1]
i=2, val=5  → arr[1]=2 < 5 → just push     → stack: [1, 2]
i=3, val=1  → arr[2]=5 > 1 → result[2]=1, pop
            → arr[1]=2 > 1 → result[1]=1, pop
            → push 3                         → stack: [3]

Result: [2, 1, 1, -1] ✓
```

---

## Variations

### LC 42 — Trapping Rain Water ★ Google ★ Apple
**What changes:** We don't just record the next smaller — we actually calculate water trapped. When we pop a `mid` element, we have a left wall (new top of stack) and a right wall (current `i`). Water in that pocket = (min of both walls - height of mid) × width.

```python
def trap(height):
    stack = []
    water = 0

    for i in range(len(height)):
        while len(stack) != 0 and height[stack[-1]] < height[i]:
            mid = stack[-1]
            stack.pop()
            if len(stack) != 0:           # need a left wall to trap water
                left = stack[-1]
                width = i - left - 1
                h = min(height[left], height[i]) - height[mid]
                water = water + h * width
        stack.append(i)

    return water

print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
```
Changed: After popping `mid`, calculate water using left wall, right wall, and mid height.

---

### LC 84 — Largest Rectangle in Histogram ★ Google ★ Apple
**What changes:** For each bar, we find how far LEFT and RIGHT it can extend without hitting a shorter bar. We run NSE from left to right to get `left[]`, then from right to left to get `right[]`. Area = height × (right - left).

```python
def largest_rectangle(heights):
    n = len(heights)
    stack = []
    left = [0] * n
    right = [n] * n

    # Find left boundary for each bar
    for i in range(n):
        while len(stack) != 0 and heights[stack[-1]] >= heights[i]:
            stack.pop()
        left[i] = 0 if len(stack) == 0 else stack[-1] + 1
        stack.append(i)

    stack = []

    # Find right boundary for each bar (go right to left)
    for i in range(n - 1, -1, -1):
        while len(stack) != 0 and heights[stack[-1]] >= heights[i]:
            stack.pop()
        right[i] = n if len(stack) == 0 else stack[-1]
        stack.append(i)

    max_area = 0
    for i in range(n):
        area = heights[i] * (right[i] - left[i])
        if area > max_area:
            max_area = area

    return max_area

print(largest_rectangle([2,1,5,6,2,3]))  # 10
```

**Dry Run → `[2,1,5,6,2,3]`**
```
left  = [0, 0, 2, 3, 2, 5]
right = [1, 6, 4, 4, 6, 6]
width = right - left = [1, 6, 2, 1, 4, 1]
area  = h * w = [2, 6, 10, 6, 8, 3]
MAX = 10 ✓
```

---

### LC 85 — Maximal Rectangle in Matrix
**What changes:** We treat each row of the matrix as the base of a histogram. Heights build up row by row. Then we run LC84 on each row's histogram.

```python
def maximal_rectangle(matrix):
    if len(matrix) == 0:
        return 0

    n = len(matrix[0])
    heights = [0] * n
    max_area = 0

    for row in matrix:
        for j in range(n):
            if row[j] == '1':
                heights[j] = heights[j] + 1  # build up height
            else:
                heights[j] = 0               # reset on '0'

        max_area = max(max_area, largest_rectangle(heights))  # run LC84

    return max_area
```
Changed: Build a histogram row by row, run LC84 on each row.

---

### LC 402 — Remove K Digits ★ Google
**What changes:** We want the smallest possible number after removing k digits. We pop when the top digit is LARGER than current — greedily removing big digits first. After the loop, if k removals are still left, remove from the end.

```python
def remove_k_digits(num, k):
    stack = []

    for ch in num:
        while k > 0 and len(stack) != 0 and stack[-1] > ch:
            stack.pop()
            k = k - 1        # used one removal
        stack.append(ch)

    # If k removals still remain, remove from end
    while k > 0:
        stack.pop()
        k = k - 1

    result = ''
    for ch in stack:
        result = result + ch

    result = result.lstrip('0')   # remove leading zeros
    if result == '':
        return '0'
    return result

print(remove_k_digits("1432219", 3))  # "1219"
```
Changed: Extra condition `k > 0` in the while. Also remove from end if k still remains.

---

### LC 1475 — Final Prices With Special Discount
Same code as shown in T2 variations. Just a reminder — this is NSE, not NGE.

```python
# Key line that changed from T2:
while len(stack) != 0 and prices[stack[-1]] >= prices[i]:   # >= finds NSE
    result[index] = prices[index] - prices[i]               # apply discount
```

---

---

# TEMPLATE 4 — Bracket Matching With Dictionary

## The Idea
Push open brackets onto the stack.
When you see a close bracket — look at what's on top of the stack.
If the top is the matching open bracket — they are a pair — pop the top.
If not — the string is invalid.
At the end — if stack is empty — all brackets were matched.

## Base Template

```python
stack = []
mapping = {')': '(', '}': '{', ']': '['}

for ch in s:
    if ch == '(' or ch == '{' or ch == '[':
        stack.append(ch)
    elif ch in mapping:
        if len(stack) == 0:
            return False
        top = stack[-1]
        stack.pop()
        if top != mapping[ch]:
            return False

return len(stack) == 0
```

## Line by Line

```python
mapping = {')': '(', '}': '{', ']': '['}
```
A dictionary that maps each CLOSING bracket to its expected OPENING bracket.
So `mapping[')']` gives you `'('` — the correct opening pair.

```python
    if ch == '(' or ch == '{' or ch == '[':
        stack.append(ch)
```
If it's an open bracket — push it. We'll match it later when its close bracket arrives.

```python
    elif ch in mapping:
```
If it's a close bracket (it's in our dict) — time to check if it matches.

```python
        if len(stack) == 0:
            return False
```
Stack is empty but we have a close bracket — nothing to match with. Invalid.

```python
        top = stack[-1]
        stack.pop()
        if top != mapping[ch]:
            return False
```
Look at what's on top. Remove it. Check: does it equal what the dict says should be there?
Example: `ch = ')'`, `mapping[')'] = '('`. If top is not `'('` — mismatch — invalid.

```python
return len(stack) == 0
```
If stack is empty — every open bracket was matched. Valid.
If anything is left — unmatched open brackets — invalid.

## Dry Run — `"{[()]}"`

```
mapping = {')':'(', '}':'{', ']':'['}

ch='{'  → open → push          → stack: ['{']
ch='['  → open → push          → stack: ['{', '[']
ch='('  → open → push          → stack: ['{', '[', '(']
ch=')'  → mapping[')']='(' → top='(' → match ✓ pop → stack: ['{', '[']
ch=']'  → mapping[']']='[' → top='[' → match ✓ pop → stack: ['{']
ch='}'  → mapping['}']:='{' → top='{' → match ✓ pop → stack: []

Stack empty → True ✓
```

---

## Variations

### LC 20 — Valid Parentheses ★ Apple (asked 10 times)
**What changes:** Nothing. Pure base template.

```python
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for ch in s:
        if ch in '({[':
            stack.append(ch)
        elif ch in mapping:
            if len(stack) == 0 or stack[-1] != mapping[ch]:
                return False
            stack.pop()
    return len(stack) == 0

print(is_valid("()[]{}"))  # True
print(is_valid("(]"))      # False
```

---

### LC 921 — Minimum Add to Make Valid
**What changes:** Instead of returning True/False, we COUNT how many brackets need to be added. Unmatched `)` = close with no open = count up. Unmatched `(` = whatever is left in the stack.

```python
def min_add_to_make_valid(s):
    stack = []
    unmatched_close = 0

    for ch in s:
        if ch == '(':
            stack.append(ch)
        elif ch == ')':
            if len(stack) != 0:
                stack.pop()                    # matched
            else:
                unmatched_close = unmatched_close + 1   # no open to match

    return len(stack) + unmatched_close  # unmatched opens + unmatched closes

print(min_add_to_make_valid("())"))   # 1
print(min_add_to_make_valid("((("))  # 3
```
Changed: We count instead of returning False. `len(stack)` = unmatched opens left.

---

### LC 1021 — Remove Outermost Parentheses
**What changes:** We track DEPTH using stack size. The outermost open bracket is when the stack WAS empty before the push. The outermost close bracket is when the stack BECOMES empty after the pop. We skip outermost brackets and keep everything else.

```python
def remove_outer_parentheses(s):
    stack = []
    result = ''

    for ch in s:
        if ch == '(':
            if len(stack) != 0:           # not the outermost open
                result = result + ch
            stack.append(ch)
        elif ch == ')':
            stack.pop()
            if len(stack) != 0:           # not the outermost close
                result = result + ch

    return result

print(remove_outer_parentheses("(()())(())"))  # "()()()"
```
Changed: We add to result only when stack is not empty (not the outermost bracket).

---

### LC 1614 — Maximum Nesting Depth
**What changes:** We track the maximum stack size while pushing. Stack size = current depth. We just track the highest it ever gets.

```python
def max_depth(s):
    stack = []
    max_d = 0

    for ch in s:
        if ch == '(':
            stack.append(ch)
            if len(stack) > max_d:
                max_d = len(stack)    # depth = current stack size
        elif ch == ')':
            stack.pop()

    return max_d

print(max_depth("(1+(2*3)+((8)/4))+1"))  # 3
```
Changed: Track `max_d = max(max_d, len(stack))` after each push.

---

### LC 856 — Score of Parentheses ★ Google
**What changes:** Stack holds SCORES instead of brackets. Push `0` for each `(`. On `)` — pop the top. If top was `0`, that means `()` which scores `1`. Otherwise it means `(A)` which scores `2 * top`. Add to the layer below.

```python
def score_of_parentheses(s):
    stack = [0]    # start with a base score of 0

    for ch in s:
        if ch == '(':
            stack.append(0)          # new layer, starts at 0
        elif ch == ')':
            top = stack[-1]
            stack.pop()
            if top == 0:
                stack[-1] = stack[-1] + 1           # () = 1
            else:
                stack[-1] = stack[-1] + 2 * top     # (A) = 2*A

    return stack[0]

print(score_of_parentheses("(()(()))"))  # 6
```

**Dry Run → `"(()(()))"`**
```
'(' → push 0  → stack: [0, 0]
'(' → push 0  → stack: [0, 0, 0]
')' → top=0 → () = 1 → stack: [0, 1]
'(' → push 0  → stack: [0, 1, 0]
'(' → push 0  → stack: [0, 1, 0, 0]
')' → top=0 → () = 1 → stack: [0, 1, 1]
')' → top=1 → (A) = 2*1=2 → add to below → stack: [0, 3]
')' → top=3 → (A) = 2*3=6 → stack: [6]
Result: 6 ✓
```

---

### LC 1249 — Minimum Remove to Make Valid ★ Google ★ Apple
**What changes:** Two passes. First pass — use stack to track INDICES of unmatched `(`. Any unmatched `)` gets added to a `remove` set immediately. Second pass — rebuild the string skipping all indices in `remove`.

```python
def min_remove_to_make_valid(s):
    stack = []        # stores indices of unmatched '('
    remove = set()    # indices of chars to remove

    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        elif s[i] == ')':
            if len(stack) != 0:
                stack.pop()        # matched
            else:
                remove.add(i)      # unmatched ')' → mark it

    for i in stack:
        remove.add(i)              # unmatched '(' also marked

    result = ''
    for i in range(len(s)):
        if i not in remove:
            result = result + s[i]

    return result

print(min_remove_to_make_valid("a)b(c)d"))  # "ab(c)d"
```

**Dry Run → `"a)b(c)d"`**
```
i=0 'a' → skip
i=1 ')' → stack empty → remove: {1}
i=2 'b' → skip
i=3 '(' → push 3       → stack: [3]
i=4 'c' → skip
i=5 ')' → pop 3        → stack: []
i=6 'd' → skip

remove = {1}
Rebuild: skip index 1 → "ab(c)d" ✓
```

---

---

# TEMPLATE 5 — Expression Evaluation

## The Idea
Numbers get pushed onto the stack.
When you see an operator — pop TWO numbers, apply the operator, push the result back.
For parentheses — save the current state before `(`, restore it after `)`.

## Base Template

```python
stack = []
operators = {'+', '-', '*', '/'}

for token in tokens:
    if token not in operators:
        stack.append(int(token))    # it's a number → push
    else:
        b = stack[-1]               # second number (top)
        stack.pop()
        a = stack[-1]               # first number (below top)
        stack.pop()
        if token == '+':   stack.append(a + b)
        elif token == '-': stack.append(a - b)
        elif token == '*': stack.append(a * b)
        elif token == '/': stack.append(int(a / b))

return stack[0]
```

## Line by Line

```python
operators = {'+', '-', '*', '/'}
```
A set of all operator symbols. We use it to check: is this token a number or an operator?

```python
    if token not in operators:
        stack.append(int(token))
```
If it's not an operator, it must be a number. Convert and push.

```python
        b = stack[-1]
        stack.pop()
        a = stack[-1]
        stack.pop()
```
Pop TWO numbers. Important: `b` is popped FIRST (it was on top), `a` is popped second.
Order matters for `-` and `/`. `a` is the left operand, `b` is the right operand.

```python
        if token == '+': stack.append(a + b)
```
Compute and push the result back. The next operator will use this result.

```python
return stack[0]
```
At the end, only one number remains in the stack — that's the final answer.

## Dry Run — `"2 3 1 * + 9 -"`

```
token=2  → push → stack: [2]
token=3  → push → stack: [2, 3]
token=1  → push → stack: [2, 3, 1]
token=*  → b=1, a=3 → 3*1=3 → push → stack: [2, 3]
token=+  → b=3, a=2 → 2+3=5 → push → stack: [5]
token=9  → push → stack: [5, 9]
token=-  → b=9, a=5 → 5-9=-4 → push → stack: [-4]

Result: -4 ✓
```

---

## Variations

### LC 150 — Evaluate Reverse Polish Notation ★ Google
**What changes:** Nothing. Pure base template.

```python
def eval_rpn(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}
    for token in tokens:
        if token not in operators:
            stack.append(int(token))
        else:
            b = stack[-1]; stack.pop()
            a = stack[-1]; stack.pop()
            if token == '+':   stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(int(a / b))
    return stack[0]

print(eval_rpn(["2","1","+","3","*"]))  # 9
```

---

### LC 224 — Basic Calculator ★ Google
**What changes:** Input is a string with `+`, `-`, and `()`. No `*` or `/`. When we hit `(`, we SAVE the current result and sign onto the stack and start fresh. When we hit `)`, we RESTORE by popping.

```python
def calculate(s):
    stack = []
    result = 0
    num = 0
    sign = 1     # +1 or -1

    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)      # build multi-digit numbers
        elif ch == '+':
            result = result + sign * num
            num = 0
            sign = 1
        elif ch == '-':
            result = result + sign * num
            num = 0
            sign = -1
        elif ch == '(':
            stack.append(result)           # save result before (
            stack.append(sign)             # save sign before (
            result = 0
            sign = 1
        elif ch == ')':
            result = result + sign * num
            num = 0
            result = result * stack[-1]    # multiply by sign before (
            stack.pop()
            result = result + stack[-1]    # add result before (
            stack.pop()

    result = result + sign * num
    return result

print(calculate("(1+(4+5+2)-3)+(6+8)"))  # 23
```
Changed: We save `(result, sign)` on open paren and restore on close paren.

---

### LC 227 — Basic Calculator II ★ Google
**What changes:** Now we have `*` and `/` which have HIGHER priority than `+` and `-`. So for `*` and `/`, we pop the top and compute IMMEDIATELY then push result. For `+` and `-`, we push a signed number and sum at the end.

```python
def calculate_2(s):
    stack = []
    num = 0
    sign = '+'

    for i in range(len(s)):
        ch = s[i]
        if ch.isdigit():
            num = num * 10 + int(ch)

        if (not ch.isdigit() and ch != ' ') or i == len(s) - 1:
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                top = stack[-1]; stack.pop()
                stack.append(top * num)        # compute * immediately
            elif sign == '/':
                top = stack[-1]; stack.pop()
                stack.append(int(top / num))   # compute / immediately
            sign = ch
            num = 0

    return sum(stack)

print(calculate_2("3+2*2"))    # 7
print(calculate_2("14-3/2"))   # 13
```
Changed: `*` and `/` pop and compute immediately. `+` and `-` push signed number. Final answer = `sum(stack)`.

---

### LC 394 — Decode String ★ Google
**What changes:** Stack stores `(string, count)` pairs instead of numbers. When we see `[`, save current string and count. When we see `]`, pop and repeat the inside string by the saved count.

```python
def decode_string(s):
    stack = []
    current_str = ''
    current_num = 0

    for ch in s:
        if ch.isdigit():
            current_num = current_num * 10 + int(ch)
        elif ch == '[':
            stack.append(current_str)    # save string built so far
            stack.append(current_num)    # save repeat count
            current_str = ''
            current_num = 0
        elif ch == ']':
            num = stack[-1]; stack.pop()
            prev = stack[-1]; stack.pop()
            current_str = prev + current_str * num   # expand
        else:
            current_str = current_str + ch

    return current_str

print(decode_string("3[a2[c]]"))   # "accaccacc"
```

**Dry Run → `"3[a2[c]]"`**
```
'3' → num=3
'[' → push '' and 3 → stack:['',3], str='', num=0
'a' → str='a'
'2' → num=2
'[' → push 'a' and 2 → stack:['',3,'a',2], str='', num=0
'c' → str='c'
']' → num=2, prev='a' → str = 'a' + 'c'*2 = 'acc' → stack:['',3]
']' → num=3, prev='' → str = '' + 'acc'*3 = 'accaccacc'
Result: "accaccacc" ✓
```

---

### LC 726 — Number of Atoms ★ Google
**What changes:** Stack holds DICTIONARIES of element counts. Push a new dict on `(`. On `)`, pop the dict, multiply all counts by the number after `)`, then merge back into the dict below.

```python
def count_atoms(formula):
    stack = [{}]
    i = 0
    n = len(formula)

    while i < n:
        if formula[i] == '(':
            stack.append({})
            i = i + 1
        elif formula[i] == ')':
            i = i + 1
            num_start = i
            while i < n and formula[i].isdigit():
                i = i + 1
            multiplier = int(formula[num_start:i]) if i > num_start else 1
            top = stack[-1]; stack.pop()
            for elem in top:
                if elem in stack[-1]:
                    stack[-1][elem] = stack[-1][elem] + top[elem] * multiplier
                else:
                    stack[-1][elem] = top[elem] * multiplier
        elif formula[i].isupper():
            j = i + 1
            while j < n and formula[j].islower():
                j = j + 1
            elem = formula[i:j]
            i = j
            num_start = i
            while i < n and formula[i].isdigit():
                i = i + 1
            count = int(formula[num_start:i]) if i > num_start else 1
            if elem in stack[-1]:
                stack[-1][elem] = stack[-1][elem] + count
            else:
                stack[-1][elem] = count

    final = stack[0]
    result = ''
    for elem in sorted(final):
        result = result + elem
        if final[elem] > 1:
            result = result + str(final[elem])
    return result

print(count_atoms("Mg(OH)2"))  # "H2MgO2"
```
Changed: Stack stores dicts. On `)`, multiply all counts and merge dicts.

---

---

# TEMPLATE 6 — Two Stacks (Min / Max Tracking)

## The Idea
We use TWO stacks.
The main stack does normal push/pop work.
The second (aux) stack tracks the MINIMUM (or MAXIMUM) at every state.
This lets us answer "what is the current minimum?" in O(1) at any time.

## Base Template

```python
main_stack = []
aux_stack  = []    # tracks minimum at every state

def push(x):
    main_stack.append(x)
    if len(aux_stack) == 0 or x <= aux_stack[-1]:
        aux_stack.append(x)    # only push to aux if x is new minimum

def pop():
    val = main_stack[-1]
    main_stack.pop()
    if val == aux_stack[-1]:
        aux_stack.pop()        # only pop from aux if we removed the current min

def get_min():
    return aux_stack[-1]       # min is always at the top of aux stack
```

## Line by Line

```python
main_stack = []
aux_stack  = []
```
Two separate lists. `main_stack` is the real stack. `aux_stack` only holds minimums.

```python
def push(x):
    main_stack.append(x)
```
Always push to main stack first.

```python
    if len(aux_stack) == 0 or x <= aux_stack[-1]:
        aux_stack.append(x)
```
Only push to `aux_stack` if the new value is smaller or equal to the current minimum.
We use `<=` (not `<`) so that duplicate minimums are handled correctly.

```python
def pop():
    val = main_stack[-1]
    main_stack.pop()
    if val == aux_stack[-1]:
        aux_stack.pop()
```
Always pop from main. ONLY pop from aux if the value we removed was the current minimum.
If it wasn't the minimum, aux stack doesn't need to change.

```python
def get_min():
    return aux_stack[-1]
```
The current minimum is always on TOP of the aux stack. O(1) lookup.

## Dry Run — push 5, 3, 7, 2 then pop 2

```
push 5 → main:[5]         aux:[5]       ← 5 is first, push to aux
push 3 → main:[5,3]       aux:[5,3]     ← 3 <= 5, push to aux
push 7 → main:[5,3,7]     aux:[5,3]     ← 7 > 3, skip aux
push 2 → main:[5,3,7,2]   aux:[5,3,2]   ← 2 <= 3, push to aux

get_min → aux[-1] = 2 ✓

pop 2  → main:[5,3,7]     2 == aux[-1] → aux:[5,3]
get_min → aux[-1] = 3 ✓

pop 7  → main:[5,3]       7 != aux[-1] → aux stays [5,3]
get_min → 3 ✓
```

---

## Variations

### LC 155 — Min Stack ★ Apple
**What changes:** Nothing. Wrap the base template in a class.

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if len(self.min_stack) == 0 or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack[-1]
        self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]
```

---

### LC 716 — Max Stack
**What changes:** Flip `<=` to `>=`. Now aux stack tracks the MAXIMUM instead of the minimum.

```python
class MaxStack:
    def __init__(self):
        self.stack = []
        self.max_stack = []

    def push(self, val):
        self.stack.append(val)
        if len(self.max_stack) == 0 or val >= self.max_stack[-1]:  # >= for max
            self.max_stack.append(val)

    def pop(self):
        val = self.stack[-1]
        self.stack.pop()
        if val == self.max_stack[-1]:
            self.max_stack.pop()

    def peekMax(self):
        return self.max_stack[-1]
```
Changed: `<=` → `>=`. That's literally the only change.

---

### LC 232 — Implement Queue Using Two Stacks ★ Google
**What changes:** Stack is LIFO, Queue is FIFO. Use two stacks. Push always goes to `stack1`. When we need to pop (dequeue), if `stack2` is empty, pour ALL of `stack1` into `stack2` — this REVERSES the order, giving us queue order.

```python
class MyQueue:
    def __init__(self):
        self.stack1 = []   # for push (inbox)
        self.stack2 = []   # for pop  (outbox)

    def push(self, x):
        self.stack1.append(x)

    def pop(self):
        if len(self.stack2) == 0:
            while len(self.stack1) != 0:
                self.stack2.append(self.stack1[-1])
                self.stack1.pop()        # pour stack1 → stack2 (reverses order)
        val = self.stack2[-1]
        self.stack2.pop()
        return val

    def peek(self):
        if len(self.stack2) == 0:
            while len(self.stack1) != 0:
                self.stack2.append(self.stack1[-1])
                self.stack1.pop()
        return self.stack2[-1]

    def empty(self):
        return len(self.stack1) == 0 and len(self.stack2) == 0
```
Changed: Two stacks with inbox/outbox roles. Pour from stack1 → stack2 on demand.

---

### LC 225 — Implement Stack Using Queues
**What changes:** After each push, ROTATE the queue so the newest element comes to the front. This makes `pop()` and `top()` always hit the newest element — same as a stack.

```python
from collections import deque

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for i in range(len(self.q) - 1):
            self.q.append(self.q.popleft())   # rotate: newest → front

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0
```
Changed: After push, rotate queue `n-1` times so newest item is at the front.

---

### LC 682 — Baseball Game ★ Apple
**What changes:** No aux stack here. Just one stack of scores. We handle 4 operations: number → push, `C` → pop, `D` → push double of top, `+` → push sum of top two.

```python
def cal_points(ops):
    stack = []

    for op in ops:
        if op == 'C':
            stack.pop()                                 # remove last score
        elif op == 'D':
            stack.append(stack[-1] * 2)                # double last score
        elif op == '+':
            stack.append(stack[-1] + stack[-2])        # sum of last two
        else:
            stack.append(int(op))                      # it's a number

    total = 0
    for num in stack:
        total = total + num
    return total

print(cal_points(["5","2","C","D","+"]))  # 30
```

**Dry Run → `["5","2","C","D","+"]`**
```
"5" → push 5      → stack: [5]
"2" → push 2      → stack: [5, 2]
"C" → pop         → stack: [5]
"D" → push 5*2=10 → stack: [5, 10]
"+" → push 10+5=15→ stack: [5, 10, 15]
Sum = 5+10+15 = 30 ✓
```

---

---

# TEMPLATE 7 — Stack With Index

## The Idea
Instead of storing the VALUES in the stack, we store the INDEX of each element.
Then we use index arithmetic to calculate distances, spans, and widths.
This is useful when the POSITION of an element matters, not just its value.

## Base Template

```python
stack = []   # stores INDICES, not values
result = [0] * n

for i in range(n):
    while len(stack) != 0 and arr[stack[-1]] <= arr[i]:
        stack.pop()
    if len(stack) == 0:
        result[i] = i + 1             # nothing bigger to the left
    else:
        result[i] = i - stack[-1]     # distance from last bigger element
    stack.append(i)

return result
```

## Line by Line

```python
stack = []   # stores INDICES, not values
```
We push `i` (the index), not `arr[i]` (the value). This is the key difference from T2/T3.

```python
    while len(stack) != 0 and arr[stack[-1]] <= arr[i]:
        stack.pop()
```
Pop everything that is smaller or equal to current. We use `arr[stack[-1]]` to GET the value at the stored index.

```python
    if len(stack) == 0:
        result[i] = i + 1
```
Stack is empty = no bigger element to the left. So the span goes all the way to index 0. That is `i + 1` positions.

```python
    else:
        result[i] = i - stack[-1]
```
The top of the stack is the last index with a bigger element. Distance = current index - that index.

```python
    stack.append(i)
```
Push the current INDEX (not value) for future elements to reference.

---

## Variations

### LC 901 — Online Stock Span ★ Google ★ Apple
**What changes:** Stack stores `(price, span)` tuples. Instead of recalculating distance from scratch, we ADD UP the spans of everything we pop. This saves time on re-scanning.

```python
class StockSpanner:
    def __init__(self):
        self.stack = []   # (price, span)

    def next(self, price):
        span = 1
        while len(self.stack) != 0 and self.stack[-1][0] <= price:
            span = span + self.stack[-1][1]    # accumulate span of popped element
            self.stack.pop()
        self.stack.append((price, span))
        return span

s = StockSpanner()
print(s.next(100))  # 1
print(s.next(80))   # 1
print(s.next(60))   # 1
print(s.next(70))   # 2
print(s.next(75))   # 4
```
Changed: Stack stores `(price, span)`. We add popped spans instead of computing `i - index`.

---

### LC 84 — Largest Rectangle in Histogram ★ Google ★ Apple
**What changes:** For each bar, we find NSE on both LEFT and RIGHT using index stacks. Width = `right[i] - left[i]`. Area = `height[i] * width`.

```python
def largest_rectangle(heights):
    n = len(heights)
    stack = []
    left = [0] * n
    right = [n] * n

    for i in range(n):
        while len(stack) != 0 and heights[stack[-1]] >= heights[i]:
            stack.pop()
        left[i] = 0 if len(stack) == 0 else stack[-1] + 1
        stack.append(i)

    stack = []
    for i in range(n - 1, -1, -1):
        while len(stack) != 0 and heights[stack[-1]] >= heights[i]:
            stack.pop()
        right[i] = n if len(stack) == 0 else stack[-1]
        stack.append(i)

    max_area = 0
    for i in range(n):
        area = heights[i] * (right[i] - left[i])
        if area > max_area:
            max_area = area
    return max_area

print(largest_rectangle([2,1,5,6,2,3]))  # 10
```
Changed: Two passes (left→right and right→left) to find left and right boundaries. Area = h * width.

---

### LC 42 — Trapping Rain Water ★ Google ★ Apple
**What changes:** When we pop a `mid` index, the new stack top becomes the LEFT wall, current `i` is the RIGHT wall. Water = `(min of both walls - height[mid]) * width`.

```python
def trap(height):
    stack = []
    water = 0

    for i in range(len(height)):
        while len(stack) != 0 and height[stack[-1]] < height[i]:
            mid = stack[-1]
            stack.pop()
            if len(stack) != 0:
                left = stack[-1]
                width = i - left - 1
                h = min(height[left], height[i]) - height[mid]
                water = water + h * width
        stack.append(i)

    return water

print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
```
Changed: After popping `mid`, calculate water trapped between `left` wall and current `i` wall.

---

### LC 1019 — Next Greater Node in Linked List
**What changes:** Convert the linked list to an array first. Then apply standard NGE with index stack. The conversion step is the only real addition.

```python
def next_larger_nodes(head):
    arr = []
    node = head
    while node != None:
        arr.append(node.val)
        node = node.next              # convert linked list → array first

    n = len(arr)
    result = [0] * n
    stack = []

    for i in range(n):
        while len(stack) != 0 and arr[stack[-1]] < arr[i]:
            index = stack[-1]
            stack.pop()
            result[index] = arr[i]
        stack.append(i)

    return result
```
Changed: Just the setup — linked list → array. Rest is standard NGE.

---

### LC 2104 — Sum of Subarray Ranges
**What changes:** Run the subarray sum logic TWICE — once for minimums (T10 style) and once for maximums. Subtract min total from max total. That gives sum of all ranges.

```python
def subarray_ranges(nums):
    n = len(nums)

    def sum_of_mins():
        stack = []
        total = 0
        for i in range(n + 1):
            while len(stack) != 0 and (i == n or nums[stack[-1]] >= nums[i]):
                mid = stack[-1]; stack.pop()
                left = -1 if len(stack) == 0 else stack[-1]
                total = total + nums[mid] * (mid - left) * (i - mid)
            if i < n:
                stack.append(i)
        return total

    def sum_of_maxs():
        stack = []
        total = 0
        for i in range(n + 1):
            while len(stack) != 0 and (i == n or nums[stack[-1]] <= nums[i]):
                mid = stack[-1]; stack.pop()
                left = -1 if len(stack) == 0 else stack[-1]
                total = total + nums[mid] * (mid - left) * (i - mid)
            if i < n:
                stack.append(i)
        return total

    return sum_of_maxs() - sum_of_mins()

print(subarray_ranges([1,2,3]))  # 4
```
Changed: Run two passes — one with `>=` (mins) and one with `<=` (maxs). Subtract.

---

---

# TEMPLATE 8 — Stack for String Building

## The Idea
Use the stack to BUILD a result string character by character.
Push valid characters.
Pop (remove) when a specific BAD condition is triggered.
What's left in the stack is your clean answer.

## Base Template

```python
stack = []

for ch in s:
    if len(stack) != 0 and ch == '#':
        stack.pop()         # bad condition → remove last char
    else:
        stack.append(ch)    # good char → keep it

result = ''
for ch in stack:
    result = result + ch

return result
```

## Line by Line

```python
stack = []
```
This will hold our result string character by character.

```python
for ch in s:
```
Go through every character.

```python
    if len(stack) != 0 and ch == '#':
        stack.pop()
```
If the current character is a `#` (bad) AND the stack has something — remove the last character we added.
This is the removal condition — you change this line per problem.

```python
    else:
        stack.append(ch)
```
Normal character — add it to the stack.

```python
result = ''
for ch in stack:
    result = result + ch
return result
```
Convert the stack back to a string.

---

## Variations

### LC 844 — Backspace String Compare ★ Apple
**What changes:** `#` = backspace. Build both strings using the template. Compare them.

```python
def backspace_compare(s, t):

    def build(string):
        stack = []
        for ch in string:
            if ch == '#':
                if len(stack) != 0:
                    stack.pop()     # backspace → remove last char
            else:
                stack.append(ch)
        result = ''
        for ch in stack:
            result = result + ch
        return result

    return build(s) == build(t)

print(backspace_compare("ab#c", "ad#c"))  # True → both = "ac"
```
Changed: Condition is `ch == '#'`. Do this for both strings and compare.

---

### LC 1047 — Remove Adjacent Duplicates
**What changes:** Condition is `stack[-1] == ch`. Same char on top as current = remove both.

```python
def remove_duplicates(s):
    stack = []
    for ch in s:
        if len(stack) != 0 and stack[-1] == ch:
            stack.pop()     # same char → cancel both
        else:
            stack.append(ch)
    result = ''
    for ch in stack:
        result = result + ch
    return result

print(remove_duplicates("abbaca"))  # "ca"
```
Changed: Removal condition is `stack[-1] == ch`.

---

### LC 316 — Remove Duplicate Letters ★ Google
**What changes:** Three conditions to pop: top is LARGER than current, top appears AGAIN later (we won't lose it), and current char is not already in the stack. We use a dict to track remaining counts and a dict to track what's in the stack.

```python
def remove_duplicate_letters(s):
    count = {}
    for ch in s:
        count[ch] = count.get(ch, 0) + 1    # remaining occurrences

    in_stack = {}
    stack = []

    for ch in s:
        count[ch] = count[ch] - 1           # one less remaining

        if in_stack.get(ch, False):
            continue                         # already in stack → skip

        while len(stack) != 0 and stack[-1] > ch and count[stack[-1]] > 0:
            removed = stack[-1]
            stack.pop()
            in_stack[removed] = False        # top is bigger AND appears later → pop

        stack.append(ch)
        in_stack[ch] = True

    result = ''
    for ch in stack:
        result = result + ch
    return result

print(remove_duplicate_letters("bcabc"))    # "abc"
print(remove_duplicate_letters("cbacdcbc")) # "acdb"
```
Changed: Removal condition is `stack[-1] > ch AND count[stack[-1]] > 0`. Use two dicts.

---

### LC 402 — Remove K Digits ★ Google
**What changes:** Remove a digit when the digit before it is LARGER. Do this k times max. After the loop, remove from end if k is still > 0. Strip leading zeros.

```python
def remove_k_digits(num, k):
    stack = []
    for ch in num:
        while k > 0 and len(stack) != 0 and stack[-1] > ch:
            stack.pop()
            k = k - 1           # used one removal
        stack.append(ch)

    while k > 0:                # still removals left → remove from end
        stack.pop()
        k = k - 1

    result = ''
    for ch in stack:
        result = result + ch
    result = result.lstrip('0')
    return result if result else '0'

print(remove_k_digits("1432219", 3))  # "1219"
```
Changed: Condition is `stack[-1] > ch` and we have a `k` counter. Strip leading zeros at end.

---

### LC 32 — Longest Valid Parentheses ★ Google
**What changes:** Stack stores INDICES. We start with `-1` as a base. On `)`, pop — if stack is now empty, push current index as new base. Otherwise, length = current index minus stack top.

```python
def longest_valid_parentheses(s):
    stack = [-1]    # base index
    max_len = 0

    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        else:
            stack.pop()
            if len(stack) == 0:
                stack.append(i)      # new base
            else:
                length = i - stack[-1]
                if length > max_len:
                    max_len = length

    return max_len

print(longest_valid_parentheses(")()())"))  # 4
print(longest_valid_parentheses("(()"))     # 2
```

**Dry Run → `")()()"`**
```
i=0 ')' → pop -1 → stack empty → push 0 as base → stack:[0]
i=1 '(' → push 1  → stack:[0,1]
i=2 ')' → pop 1   → stack:[0] → length = 2-0 = 2
i=3 '(' → push 3  → stack:[0,3]
i=4 ')' → pop 3   → stack:[0] → length = 4-0 = 4
max_len = 4 ✓
```

---

---

# TEMPLATE 9 — Stack for Collision / Elimination

## The Idea
Elements are moving in directions and collide when they meet.
Positive numbers = moving RIGHT. Negative numbers = moving LEFT.
A collision happens when a LEFT-moving element meets a RIGHT-moving element that is already in the stack.
The bigger one survives. Equal size = both die.

## Base Template

```python
stack = []

for val in arr:
    destroyed = False

    while len(stack) != 0 and val < 0 and stack[-1] > 0:
        if stack[-1] < abs(val):
            stack.pop()
            continue            # top was smaller → top destroyed, keep going
        elif stack[-1] == abs(val):
            stack.pop()
            destroyed = True
            break               # equal → both destroyed
        else:
            destroyed = True
            break               # top was bigger → current val destroyed

    if destroyed == False:
        stack.append(val)

return stack
```

## Line by Line

```python
destroyed = False
```
A flag to track whether the current element survived or got destroyed.

```python
    while len(stack) != 0 and val < 0 and stack[-1] > 0:
```
Three conditions for a collision:
Stack is not empty. Current element is moving LEFT (negative). Top of stack is moving RIGHT (positive).
Only when all three are true — there is a collision.

```python
        if stack[-1] < abs(val):
            stack.pop()
            continue
```
Top is smaller than current — top gets destroyed. Continue the loop — current might destroy more.

```python
        elif stack[-1] == abs(val):
            stack.pop()
            destroyed = True
            break
```
Equal size — both die. Pop the top, mark current as destroyed, stop the loop.

```python
        else:
            destroyed = True
            break
```
Top is bigger — current gets destroyed. Mark it, stop.

```python
    if destroyed == False:
        stack.append(val)
```
Only push if current element survived (or there was no collision).

## Dry Run — `[5, 10, -5]`

```
val=5  → no collision (5 > 0) → push     → stack: [5]
val=10 → no collision (10 > 0) → push    → stack: [5, 10]
val=-5 → collision! top=10, val=-5
       → 10 > |-5|=5 → -5 destroyed → stack: [5, 10]

Result: [5, 10] ✓
```

---

## Variations

### LC 735 — Asteroid Collision ★ Apple
**What changes:** Nothing. Pure base template.

```python
def asteroid_collision(asteroids):
    stack = []
    for val in asteroids:
        destroyed = False
        while len(stack) != 0 and val < 0 and stack[-1] > 0:
            if stack[-1] < abs(val):
                stack.pop(); continue
            elif stack[-1] == abs(val):
                stack.pop(); destroyed = True; break
            else:
                destroyed = True; break
        if destroyed == False:
            stack.append(val)
    return stack

print(asteroid_collision([8,-8]))       # []
print(asteroid_collision([10,2,-5]))    # [10]
```

---

### LC 2211 — Count Collisions on a Road
**What changes:** We count how many cars get destroyed. `R` moving right → push. `L` coming → pop all `R` cars it destroys (each R+L = 2 collisions). `S` stopped → pop all `R` coming at it (each R+S = 1 collision for the R).

```python
def count_collisions(directions):
    stack = []
    collisions = 0

    for d in directions:
        if d == 'R':
            stack.append(d)
        elif d == 'L':
            while len(stack) != 0 and stack[-1] == 'R':
                stack.pop()
                collisions = collisions + 2    # R destroyed + L destroyed
            if len(stack) != 0 and stack[-1] == 'S':
                collisions = collisions + 1    # only L destroyed (S stays)
        elif d == 'S':
            while len(stack) != 0 and stack[-1] == 'R':
                stack.pop()
                collisions = collisions + 1    # R hits the stopped S
            stack.append('S')

    return collisions

print(count_collisions("RLRSLL"))  # 5
```
Changed: Count collisions instead of surviving elements.

---

### LC 456 — 132 Pattern ★ Google
**What changes:** We traverse from RIGHT to LEFT. We keep a stack of candidates for the "3" (biggest). We track `third` — the best candidate for the "2" (middle value). If current element < third, we found the "1".

```python
def find_132_pattern(nums):
    stack = []
    third = float('-inf')    # best candidate for the "2" in 1-3-2

    for i in range(len(nums) - 1, -1, -1):   # traverse right to left
        if nums[i] < third:
            return True           # nums[i] is the "1", pattern found!
        while len(stack) != 0 and stack[-1] < nums[i]:
            third = stack[-1]     # popped element becomes best "2" candidate
            stack.pop()
        stack.append(nums[i])

    return False

print(find_132_pattern([3,1,4,2]))   # True
print(find_132_pattern([-1,3,2,0]))  # True
```
Changed: Traverse right to left. Track `third` as best middle value seen so far.

---

### LC 636 — Exclusive Time of Functions ★ Apple
**What changes:** Stack holds FUNCTION IDs. When a function starts, push its id. When it ends, pop and record its exclusive time. If a parent function is paused by a child, charge the parent only for the time it actually ran.

```python
def exclusive_time(n, logs):
    result = [0] * n
    stack = []
    prev_time = 0

    for log in logs:
        parts = log.split(':')
        func_id = int(parts[0])
        event   = parts[1]
        time    = int(parts[2])

        if event == 'start':
            if len(stack) != 0:
                result[stack[-1]] = result[stack[-1]] + (time - prev_time)
            stack.append(func_id)
            prev_time = time
        else:  # end
            result[stack[-1]] = result[stack[-1]] + (time - prev_time + 1)
            stack.pop()
            prev_time = time + 1

    return result

logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
print(exclusive_time(2, logs))  # [3, 4]
```

**Dry Run**
```
0:start:0 → push 0, prev=0               → stack:[0]
1:start:2 → charge func0: 2-0=2, push 1  → stack:[0,1]
1:end:5   → func1: 5-2+1=4, pop, prev=6  → stack:[0]
0:end:6   → func0: 6-6+1=1, pop          → stack:[]
result[0]=2+1=3, result[1]=4 → [3,4] ✓
```

---

### LC 2866 — Beautiful Towers II
**What changes:** We compute the sum of valid tower heights to the LEFT and RIGHT of each potential peak using a monotonic stack. Left and right sums are built incrementally using previous results stored in arrays.

```python
def maximum_sum_of_heights(maxHeights):
    n = len(maxHeights)
    left = [0] * n
    right = [0] * n
    stack = []

    for i in range(n):
        while len(stack) != 0 and maxHeights[stack[-1]] >= maxHeights[i]:
            stack.pop()
        if len(stack) == 0:
            left[i] = (i + 1) * maxHeights[i]
        else:
            j = stack[-1]
            left[i] = left[j] + (i - j) * maxHeights[i]
        stack.append(i)

    stack = []
    for i in range(n - 1, -1, -1):
        while len(stack) != 0 and maxHeights[stack[-1]] >= maxHeights[i]:
            stack.pop()
        if len(stack) == 0:
            right[i] = (n - i) * maxHeights[i]
        else:
            j = stack[-1]
            right[i] = right[j] + (j - i) * maxHeights[i]
        stack.append(i)

    result = 0
    for i in range(n):
        total = left[i] + right[i] - maxHeights[i]
        if total > result:
            result = total
    return result

print(maximum_sum_of_heights([5,3,4,1,1]))  # 13
```

---

### LC 1544 — Make The String Great
Already covered in T1, but fits here too — same letter different case = collision, both cancelled.

---

---

# TEMPLATE 10 — Stack for Subarray / Window Problems

## The Idea
For each element, count HOW MANY SUBARRAYS have it as the MINIMUM (or maximum).
That count = (distance to previous smaller on left) × (distance to next smaller on right).
Multiply that count by the element's value and sum everything up.

## Base Template

```python
stack = []
total = 0

for i in range(n + 1):    # go one extra step to flush remaining elements
    while len(stack) != 0 and (i == n or arr[stack[-1]] >= arr[i]):
        mid = stack[-1]
        stack.pop()
        left  = -1 if len(stack) == 0 else stack[-1]
        right = i
        count = (mid - left) * (right - mid)
        total = total + arr[mid] * count
    if i < n:
        stack.append(i)

return total
```

## Line by Line

```python
for i in range(n + 1):
```
We go one step PAST the array. The `i == n` case is used to flush any elements still in the stack at the end.

```python
    while len(stack) != 0 and (i == n or arr[stack[-1]] >= arr[i]):
```
Pop when: we have reached the end (`i == n`) OR the top of the stack is >= current element.
We use `>=` to handle duplicates — only one side gets counted.

```python
        mid = stack[-1]
        stack.pop()
```
`mid` is the element we are computing subarrays for. It is the minimum in all subarrays we count.

```python
        left  = -1 if len(stack) == 0 else stack[-1]
        right = i
```
`left` = the index of the PREVIOUS SMALLER element (or -1 if none exists).
`right` = current `i` = the index of the NEXT SMALLER element (or n if none exists).

```python
        count = (mid - left) * (right - mid)
```
Number of subarrays where `arr[mid]` is the minimum.
Left side has `(mid - left)` choices. Right side has `(right - mid)` choices. Multiply.

```python
        total = total + arr[mid] * count
```
Add contribution of `arr[mid]` across all those subarrays.

## Dry Run — `[3, 1, 2, 4]`

```
Subarrays and their minimums:
[3]=3  [1]=1  [2]=2  [4]=4
[3,1]=1  [1,2]=1  [2,4]=2
[3,1,2]=1  [1,2,4]=1
[3,1,2,4]=1

Sum = 3+1+2+4 + 1+1+2 + 1+1 + 1 = 17 ✓
```

---

## Variations

### LC 907 — Sum of Subarray Minimums ★ Google
**What changes:** Nothing. Pure base template. Just add MOD at the end.

```python
def sum_subarray_mins(arr):
    n = len(arr)
    stack = []
    total = 0
    MOD = 10**9 + 7

    for i in range(n + 1):
        while len(stack) != 0 and (i == n or arr[stack[-1]] >= arr[i]):
            mid = stack[-1]; stack.pop()
            left  = -1 if len(stack) == 0 else stack[-1]
            right = i
            count = (mid - left) * (right - mid)
            total = total + arr[mid] * count
        if i < n:
            stack.append(i)

    return total % MOD

print(sum_subarray_mins([3,1,2,4]))  # 17
```

---

### LC 2104 — Sum of Subarray Ranges
**What changes:** Range = max - min of each subarray. Run the template TWICE — once for minimums (`>=`), once for maximums (`<=`). Answer = sum of maxs - sum of mins.

```python
def subarray_ranges(nums):
    n = len(nums)

    def get_sum(is_min):
        stack = []
        total = 0
        for i in range(n + 1):
            while len(stack) != 0 and (i == n or
                  (nums[stack[-1]] >= nums[i] if is_min else nums[stack[-1]] <= nums[i])):
                mid = stack[-1]; stack.pop()
                left = -1 if len(stack) == 0 else stack[-1]
                total = total + nums[mid] * (mid - left) * (i - mid)
            if i < n:
                stack.append(i)
        return total

    return get_sum(False) - get_sum(True)   # maxs - mins

print(subarray_ranges([1,2,3]))  # 4
```
Changed: Run template twice with flipped comparison. Subtract.

---

### LC 84 — Largest Rectangle in Histogram (Single Pass) ★ Google ★ Apple
**What changes:** Instead of summing `arr[mid] * count`, we compute `height * width` and track the maximum area seen.

```python
def largest_rectangle(heights):
    n = len(heights)
    stack = []
    max_area = 0

    for i in range(n + 1):
        while len(stack) != 0 and (i == n or heights[stack[-1]] >= heights[i]):
            h = heights[stack[-1]]
            stack.pop()
            left = -1 if len(stack) == 0 else stack[-1]
            width = i - left - 1
            area = h * width
            if area > max_area:
                max_area = area
        if i < n:
            stack.append(i)

    return max_area

print(largest_rectangle([2,1,5,6,2,3]))  # 10
```
Changed: `area = h * width`, track `max_area` instead of summing.

---

### LC 42 — Trapping Rain Water (Single Pass) ★ Google ★ Apple
**What changes:** When popping `mid`, calculate water trapped. Need a LEFT wall (`stack[-1]` after pop) and RIGHT wall (`i`). Water = `(min of walls - height[mid]) * width`.

```python
def trap(height):
    stack = []
    water = 0

    for i in range(len(height)):
        while len(stack) != 0 and height[stack[-1]] < height[i]:
            mid = stack[-1]; stack.pop()
            if len(stack) != 0:
                left = stack[-1]
                width = i - left - 1
                h = min(height[left], height[i]) - height[mid]
                water = water + h * width
        stack.append(i)

    return water

print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
```
Changed: Compute `(min of walls - mid height) * width` instead of `arr[mid] * count`.

---

### LC 1856 — Maximum Subarray Min-Product
**What changes:** Min-product = min × sum of subarray. We need the SUM of each subarray — use a PREFIX SUM array for that. Instead of `arr[mid] * count`, compute `arr[mid] * prefix_sum_of_range`.

```python
def max_sum_min_product(nums):
    n = len(nums)
    MOD = 10**9 + 7

    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]   # build prefix sum

    stack = []
    result = 0

    for i in range(n + 1):
        while len(stack) != 0 and (i == n or nums[stack[-1]] >= nums[i]):
            mid = stack[-1]; stack.pop()
            left  = -1 if len(stack) == 0 else stack[-1]
            right = i
            subarray_sum = prefix[right] - prefix[left + 1]   # sum of range
            min_product  = nums[mid] * subarray_sum
            if min_product > result:
                result = min_product
        if i < n:
            stack.append(i)

    return result % MOD

print(max_sum_min_product([1,2,3,2]))   # 14
print(max_sum_min_product([2,3,3,1,2])) # 18
```

---

### LC 1209 — Remove All Adjacent Duplicates II ★ Apple
**What changes:** Stack stores `(char, count)` pairs instead of just characters. When the count hits k, the whole group is eliminated (just don't push it back).

```python
def remove_duplicates_k(s, k):
    stack = []   # (char, count) pairs

    for ch in s:
        if len(stack) != 0 and stack[-1][0] == ch:
            char, count = stack[-1]
            stack.pop()
            new_count = count + 1
            if new_count != k:
                stack.append((char, new_count))  # not k yet → keep
            # if new_count == k → both get dropped (don't push back)
        else:
            stack.append((ch, 1))

    result = ''
    for char, count in stack:
        result = result + char * count
    return result

print(remove_duplicates_k("deeedbbcccbdaa", 3))  # "aa"
```

**Dry Run → `"deeedbbcccbdaa"`, k=3**
```
d→(d,1)  e→(e,1)  e→(e,2)  e→count=3→drop  →stack:[(d,1)]
d→(d,2)  b→(b,1)  b→(b,2)
c→(c,1)  c→(c,2)  c→count=3→drop  →stack:[(d,2),(b,2)]
b→count=3→drop  →stack:[(d,2)]
d→count=3→drop  →stack:[]
a→(a,1)  a→(a,2)
Result: "aa" ✓
```

---

---

# ONE-PAGE QUICK REFERENCE

| Template | Core Idea | Key Line |
|---|---|---|
| T1 — Push/Pop Condition | Pop when neighbour matches a rule | Change the `if` condition |
| T2 — Monotonic Decreasing | Pop when bigger arrives (NGE) | `arr[stack[-1]] < arr[i]` |
| T3 — Monotonic Increasing | Pop when smaller arrives (NSE) | `arr[stack[-1]] > arr[i]` |
| T4 — Bracket Matching | Push open, match close via dict | `mapping = {')':'(', ...}` |
| T5 — Expression Eval | Numbers push, operators pop two | `b=pop, a=pop, push a op b` |
| T6 — Two Stacks | Second stack tracks min or max | `<=` for min, `>=` for max |
| T7 — Index Stack | Store index not value | `result[i] = i - stack[-1]` |
| T8 — String Building | Build string, pop on bad char | Change the pop condition |
| T9 — Collision | Elements collide by direction | `val<0 and stack[-1]>0` |
| T10 — Subarray Window | Count subarrays via boundaries | `(mid-left) * (right-mid)` |

---

# GOOGLE & APPLE — CONFIRMED QUESTION LIST

| LC | Problem | Template | Company |
|---|---|---|---|
| 20 | Valid Parentheses | T4 | ★ Apple (asked 10x) |
| 42 | Trapping Rain Water | T3/T7 | ★ Google ★ Apple |
| 71 | Simplify Path | T1 | ★ Google ★ Apple |
| 84 | Largest Rectangle in Histogram | T3/T7 | ★ Google ★ Apple |
| 85 | Maximal Rectangle | T3 | ★ Google |
| 150 | Evaluate RPN | T5 | ★ Google |
| 155 | Min Stack | T6 | ★ Google ★ Apple |
| 224 | Basic Calculator | T5 | ★ Google |
| 227 | Basic Calculator II | T5 | ★ Google |
| 232 | Queue Using Two Stacks | T6 | ★ Google |
| 316 | Remove Duplicate Letters | T8 | ★ Google |
| 394 | Decode String | T5 | ★ Google |
| 402 | Remove K Digits | T8 | ★ Google |
| 456 | 132 Pattern | T9 | ★ Google |
| 636 | Exclusive Time of Functions | T9 | ★ Apple |
| 682 | Baseball Game | T6 | ★ Apple |
| 726 | Number of Atoms | T5 | ★ Google |
| 739 | Daily Temperatures | T2 | ★ Google |
| 735 | Asteroid Collision | T9 | ★ Apple |
| 844 | Backspace String Compare | T8 | ★ Apple |
| 856 | Score of Parentheses | T4 | ★ Google |
| 901 | Online Stock Span | T2/T7 | ★ Google ★ Apple |
| 907 | Sum of Subarray Minimums | T10 | ★ Google |
| 1019 | Next Greater Node in Linked List | T7 | ★ Google |
| 1209 | Remove All Adjacent Duplicates II | T10 | ★ Apple |
| 1249 | Min Remove to Make Valid | T4 | ★ Google ★ Apple |
| 2866 | Beautiful Towers II | T9 | ★ Google |

---

# GOLDEN RULES

| # | Rule |
|---|---|
| 1 | Stack = LIFO. Last thing in is first thing out. Always. |
| 2 | Monotonic stack = keep stack always increasing OR always decreasing. Used for NGE, NSE, histogram, span. |
| 3 | Store INDICES when position info matters (span, width, distance). |
| 4 | Two stacks = simulate queue, or track min/max at every state. |
| 5 | Always check `len(stack) != 0` before pop or peek. Never pop blindly. |
| 6 | Postfix / prefix expressions = always stack. |
| 7 | "Nearest greater/smaller" = monotonic stack. Always. |
| 8 | Brackets problem = dict maps close→open. Clean and no repeated elifs. |
| 9 | Subarrays + min/max = Template 10. `count = (mid-left) * (right-mid)`. |
| 10 | Circular array + NGE = loop twice, use `index = i % n`. |
