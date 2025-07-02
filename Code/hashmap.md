# Mastering HashMaps in Python (Beginner to Expert)

## Table of Contents

1. What is a HashMap?
2. Real-Life Analogies
3. HashMap Internals
4. HashMap Operations
5. Hash Functions & Collisions
6. Python Implementation (`dict`)
7. Best Practices
8. Tricks & Tips
9. Visual Summary
10. LeetCode Problem Set (30 Problems)

---

## 1. What is a HashMap?

A **HashMap** is a data structure that maps **keys** to **values**, allowing efficient lookup, insertion, and deletion in **average O(1)** time.

Also known as:

* Dictionary (Python)
* Hash Table (C/C++)
* Map (Java)

### Core Idea:

It uses a **hash function** to convert the key into an **index** in an internal array (called a bucket array), storing the value at that index.

## 2. Real-Life Analogies

* **Phone Book**: Name (key) → Phone number (value)
* **Student Roll Book**: Roll number (key) → Student details (value)
* **Library Catalog**: ISBN number → Book details

## 3. HashMap Internals

### Components:

* **Key**: Unique identifier
* **Value**: Data associated with the key
* **Bucket**: Storage container for key-value pairs
* **Hash Function**: Converts a key into an index

### Collision Handling:

* **Chaining**: Linked list or list of key-value pairs at same index
* **Open Addressing**: Probes for the next available slot

## 4. HashMap Operations

| Operation | Time Complexity               |
| --------- | ----------------------------- |
| Insertion | O(1) avg, O(n) worst (resize) |
| Deletion  | O(1) avg                      |
| Lookup    | O(1) avg                      |
| Resize    | O(n)                          |

### Example:

```python
my_map = {}
my_map["name"] = "Alice"
print(my_map["name"])  # Output: Alice
```

### Basic Operations

```python
# Insertion
my_map["age"] = 25

# Update
my_map["age"] = 26

# Lookup
print(my_map.get("name"))  # Safer than my_map["name"]

# Deletion
del my_map["age"]
```

## 5. Hash Functions & Collisions

### Hash Function:

Maps keys to index values using built-in `hash()` in Python.

```python
print(hash("apple"))  # Example output: 62349782312
```

### Collision:

When two keys map to the same index. Python uses **chaining** under the hood.

## 6. Python `dict` as HashMap

### Features:

* Fast O(1) operations
* Preserves insertion order (since Python 3.7)
* Supports complex key types (tuples, strings, etc. if hashable)

```python
# Tuple as a key
coord_map = {(1, 2): "Tree", (2, 3): "House"}
```

## 7. Best Practices

* Use `dict.get(key, default)` to avoid `KeyError`
* Avoid mutable types as keys (lists, dicts)
* Use `collections.defaultdict` for automatic initialization
* Prefer `Counter` for counting items

```python
from collections import defaultdict
freq = defaultdict(int)
freq["apple"] += 1
```

## 8. Tricks & Tips

* Reverse a map: `{v: k for k, v in my_map.items()}`
* Frequency map in one line: `Counter(arr)`
* Check for duplicate: `len(set(arr)) != len(arr)`
* Grouping: Use `defaultdict(list)`
* Counting: Use `collections.Counter`

## 9. Visual Summary

```
Key     ->   Hash Function   ->   Index   ->   Bucket   ->   (Key, Value)
"name"       hash("name")         5          array[5]       ("name", "Alice")
```

---

## 10. LeetCode Problem Set (HashMap Mastery)

### Easy (10 problems)

1. Two Sum (1)
2. Valid Anagram (242)
3. Intersection of Two Arrays (349)
4. First Unique Character in a String (387)
5. Ransom Note (383)
6. Isomorphic Strings (205)
7. Contains Duplicate (217)
8. Happy Number (202)
9. Find the Difference (389)
10. Degree of an Array (697)

### Medium (15 problems)

11. Group Anagrams (49)
12. Subarray Sum Equals K (560)
13. Longest Substring Without Repeating Characters (3)
14. Top K Frequent Elements (347)
15. Longest Palindrome by Concatenating Two Letter Words (2131)
16. Word Pattern (290)
17. Minimum Window Substring (76)
18. Longest Consecutive Sequence (128)
19. Longest Repeating Character Replacement (424)
20. Binary Tree Vertical Order Traversal (314)
21. Fruit Into Baskets (904)
22. Count Good Meals (1711)
23. Number of Boomerangs (447)
24. Contiguous Array (525)
25. Find All Anagrams in a String (438)

### Hard (5 problems)

26. LRU Cache (146)
27. Alien Dictionary (269)
28. Sliding Window Maximum (239)
29. Subarrays with K Different Integers (992)
30. Number of Distinct Substrings in a String (using Rolling Hash)

Each problem will include:

* Problem summary
* Python solution with explanation
* Code patterns
* Edge case handling

---

## \[To Be Continued]: Detailed solutions for all 30 problems in upcoming sections.

few more code snippets and cheat sheets




Great! Here's an extended **HashMap Python Cheat Sheet** with **extra code snippets**, patterns, and ready-to-use logic blocks—perfect for interviews and LeetCode.

---

# 🔥 HashMap Cheat Sheet (Python)

## 🔧 1. Basic Operations

```python
my_map = {}

# Insert / Update
my_map["a"] = 1

# Lookup
val = my_map["a"]  # KeyError if not found
val = my_map.get("a", -1)  # Safe lookup

# Check if key exists
if "a" in my_map:
    print("Exists")

# Delete
del my_map["a"]

# Length
len(my_map)

# Iterate keys & values
for k, v in my_map.items():
    print(k, v)
```

---

## 🎯 2. Frequency Map

```python
from collections import Counter

# Count occurrences
freq = Counter(["a", "b", "a", "c"])
print(freq["a"])  # Output: 2
```

## 🧼 Custom Frequency Map

```python
from collections import defaultdict

freq = defaultdict(int)
for ch in "abracadabra":
    freq[ch] += 1
```

---

## 🧬 3. Grouping by Key

```python
from collections import defaultdict

group = defaultdict(list)
words = ["eat", "tea", "tan", "ate", "nat", "bat"]

for word in words:
    key = "".join(sorted(word))
    group[key].append(word)
```

---

## 🔁 4. Reverse a HashMap

```python
my_map = {"a": 1, "b": 2}
rev_map = {v: k for k, v in my_map.items()}
```

---

## 🚫 5. Remove Keys Based on Condition

```python
# Remove keys with value == 0
my_map = {k: v for k, v in my_map.items() if v != 0}
```

---

## 🧠 6. Set vs Map for Fast Lookup

```python
seen = set()
for num in nums:
    if target - num in seen:
        return True
    seen.add(num)
```

---

## 📊 7. Top K Frequent Elements

```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    freq = Counter(nums)
    return heapq.nlargest(k, freq, key=freq.get)
```

---

## 🪟 8. Sliding Window + HashMap

```python
from collections import defaultdict

def longest_unique_substring(s):
    left = 0
    seen = defaultdict(int)
    max_len = 0

    for right in range(len(s)):
        seen[s[right]] += 1

        while seen[s[right]] > 1:
            seen[s[left]] -= 1
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len
```

---

## 🧩 9. Check Isomorphism with Maps

```python
def isIsomorphic(s, t):
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))
```

---

## 📚 10. Common Interview Use-Cases

| Use-Case                   | Python Snippet                           |
| -------------------------- | ---------------------------------------- |
| Count frequency            | `Counter(arr)`                           |
| Check duplicates           | `len(set(arr)) < len(arr)`               |
| First non-repeating char   | `OrderedDict` or `Counter`               |
| Two sum pairs              | `map[target - num]`                      |
| Group by sorted key        | `defaultdict(list)`                      |
| Longest unique substring   | Sliding window + map                     |
| Window with k unique chars | Shrinking window with hashmap size check |
| Palindrome with counts     | Check odd counts in map                  |

---

## ✨ Advanced Tips

* Prefer `defaultdict(int)` over manual existence checks.
* Use `.get(k, default)` to handle missing keys gracefully.
* Use tuple keys for grids/coordinates: `visited[(x, y)] = True`
* For frequency comparison of strings: `Counter(s) == Counter(t)`

---

Would you like me to integrate this cheat sheet directly into your **Hashmap Mastery Notes** document?

Next: Proceed with **Problem 2: Valid Anagram (LeetCode 242)?**
