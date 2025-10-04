# Teaching Key Hashing Problems for Interviews

I'll guide you through each problem with:
1. Problem description
2. Hash-based solution in Python with time complexity
3. Other optimal methods if available

Let's start with the first few problems and progress step-by-step.

---## 1) Check for Subset

### Problem
Given two arrays, check if the first array is a subset of the second (all elements of first present in second).

### Hash solution
- Insert all elements of second array into a hash set (O(m))
- Check every element of first array in the set (O(n))
- Overall complexity: **O(m + n)**

```python
def is_subset(arr1, arr2):
    set_arr2 = set(arr2)
    return all(x in set_arr2 for x in arr1)
```

### Other methods
- Sorting + two pointers: O(n log n + m log m), slower.
- Counting frequency arrays if ranges known.

---## 2) Check for Disjoint Arrays

### Problem
Given two arrays, determine if they are disjoint (no common elements).

### Hash solution
- Insert all elements of one array into a set (O(n)).
- Check each element of other array if exists in set. Return False if found.
- Complexity: **O(n + m)**

```python
def are_disjoint(arr1, arr2):
    set_arr1 = set(arr1)
    for x in arr2:
        if x in set_arr1:
            return False
    return True
```

### Other methods
- Sort both arrays and use two pointers (O(n log n + m log m)).

---## 3) Check for Equality of Two Arrays

### Problem
Determine if two arrays have the same elements with the same frequency (order doesn't matter).

### Hash solution
- Use two hash maps (dicts) to count element frequencies.
- Compare the two frequency maps.
- Complexity: **O(n + m)**

```python
from collections import Counter

def are_equal(arr1, arr2):
    return Counter(arr1) == Counter(arr2)
```

### Other methods
- Sort both arrays and check equality (O(n log n + m log m)), less efficient for large unsorted arrays.

---## 4) Fizz Buzz

### Problem
Print numbers from 1 to N with the following substitutions:
- Divisible by 3: print "Fizz"
- Divisible by 5: print "Buzz"
- Divisible by both 3 and 5: print "FizzBuzz"

### Hash solution
- This is more about control structures than hashing.
- No hash needed; use conditionals.

```python
def fizz_buzz(n):
    for i in range(1, n+1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)
```

### Other methods
- Same logic; no better optimization.

---## 5) Maximum Distance Between Two Occurrences of Same Element

### Problem
Given an array, find the maximum distance (index difference) between two occurrences of the same element.

### Hash solution
- Use a dictionary to record first occurrence index of each element.
- On seeing element again, calculate distance with first index and update max.
- Complexity: **O(n)**

```python
def max_distance(arr):
    first_indices = {}
    max_dist = 0
    for i, val in enumerate(arr):
        if val not in first_indices:
            first_indices[val] = i
        else:
            dist = i - first_indices[val]
            if dist > max_dist:
                max_dist = dist
    return max_dist
```

### Other methods
- Brute force O(n²) checking all pairs, inefficient.

***
Let's continue with some important hash problems, explaining each:

***

## 6) Check for Duplicates within K Distance

### Problem
Given an array, check if any element repeats within a distance `k`.

### Hash solution
- Use a hash map to store each element's last index encountered.
- For each element, check if it exists and if the current index minus the stored index is less than or equal to `k`.
- Update the last index.

```python
def containsNearbyDuplicate(arr, k):
    index_map = {}
    for i, num in enumerate(arr):
        if num in index_map and i - index_map[num] <= k:
            return True
        index_map[num] = i
    return False
```

### Time Complexity
- O(n), since each element is visited once.

***

## 7) Intersection of Two Arrays

### Problem
Find common elements between two arrays.

### Hash solution
- Convert first array to a set for fast lookup.
- Loop through second array, collecting elements that are in the set.

```python
def intersection(arr1, arr2):
    set_arr1 = set(arr1)
    return [x for x in arr2 if x in set_arr1]
```

### Time complexity
- O(n + m), where n and m are sizes of the two arrays.

***

## 8) Union of Two Arrays

### Problem
Combine all unique elements from two arrays.

### Hash solution
```python
def union(arr1, arr2):
    return list(set(arr1) | set(arr2))
```

- Use set union operator `|` for efficiency.

### Time complexity
- O(n + m)

***

## 9) Most Frequent Element

### Problem
Find element occurring the highest number of times.

### Hash solution
- Use a hash map (Counter) to count frequency.
- Return the element with maximum count.

```python
from collections import Counter

def most_frequent(arr):
    count = Counter(arr)
    return max(count, key=count.get)
```

### Time complexity
- O(n), counting and finding max.

***

## 10) 2 Sum: Check if Any Pair Sums to Target

### Problem
Given an array, check if any pair sums to a target.

### Hash solution
- Use a set to store elements seen.
- For each element, check if `target - element` exists in set.

```python
def two_sum(arr, target):
    seen = set()
    for num in arr:
        if target - num in seen:
            return True
        seen.add(num)
    return False
```

### Time complexity
- O(n)

***

## 11) 2 Sum Count Pairs

### Problem
Count pairs that sum to target.

### Hash solution
- Use a frequency dictionary to count pairs efficiently.

```python
def count_pairs(arr, target):
    counts = {}
    count = 0
    for num in arr:
        complement = target - num
        if complement in counts:
            count += counts[complement]
        counts[num] = counts.get(num, 0) + 1
    return count
```

### Time complexity
- O(n)

***
Continuing with the next set of hashing problems:

***

## 12) Count Pairs with Given Difference

### Problem
Count pairs in array where the absolute difference equals a given value `k`.

### Hash Solution
- Use a hash set for O(1) lookups.
- For each element, check if `element + k` or `element - k` exists in the set.

```python
def count_pairs_with_diff(arr, k):
    elements = set(arr)
    count = 0
    for num in arr:
        if num + k in elements:
            count += 1
        if num - k in elements:
            count += 1
        elements.remove(num)  # avoid double counting
    return count
```

### Time Complexity
- O(n)

***

## 13) Only Repetitive Element from 1 to n-1

### Problem
Given array of n elements with values from 1 to n-1, find the repeated element.

### Hash Solution
- Use a set to record seen values while iterating through array.
- Return the first element found already in set.

```python
def find_repetitive_element(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return num
        seen.add(num)
    return -1  # no repeat found
```

### Time Complexity
- O(n)

***

## 14) Missing of a Range

### Problem
Given a range `[low, high]` and an array of numbers inside the range, find the missing numbers.

### Hash Solution
- Insert all array elements in set.
- Iterate from low to high, collecting numbers not in set.

```python
def find_missing_in_range(arr, low, high):
    s = set(arr)
    missing = [x for x in range(low, high + 1) if x not in s]
    return missing
```

### Time Complexity
- O(n + range_size)

***

## 15) Missing from Min to Max of Array

### Problem
Find missing integers between the minimum and maximum values of an array.

### Hash Solution
- Similar to problem 14, but `low = min(arr)`, `high = max(arr)`.

```python
def find_missing_min_max(arr):
    low, high = min(arr), max(arr)
    s = set(arr)
    missing = [x for x in range(low, high + 1) if x not in s]
    return missing
```

### Time Complexity
- O(n + (max - min))

***

## 16) Minimum Subsets with Distinct Elements

### Problem
Divide array into minimum number of subsets such that all elements in each subset are distinct.

### Hash Solution
- Count frequency of each element.
- Minimum subsets = maximum frequency.

```python
from collections import Counter

def min_subsets_distinct(arr):
    freq = Counter(arr)
    return max(freq.values())
```

### Time Complexity
- O(n)

***

## 17) Minimum Removals for No Common Elements

### Problem
Find the minimum number of elements to remove from two arrays so that they have no common elements.

### Hash Solution
- Find intersection elements, minimum removals = size of intersection.

```python
def min_removals_no_common(arr1, arr2):
    set1, set2 = set(arr1), set(arr2)
    intersection = set1 & set2
    return len(intersection)
```

### Time Complexity
- O(n + m)

***

## 18) Maximum Points on the Same Line

### Problem
Given points on 2D plane, find maximum points that lie on the same straight line.

### Hash Solution
- For each point, compute slope with others.
- Use hashmap to count slopes; max count slopes + duplicates give line count.

```python
from collections import defaultdict
from math import gcd

def max_points_on_line(points):
    if len(points) <= 2:
        return len(points)
    max_points = 0
    for i in range(len(points)):
        slope_count = defaultdict(int)
        duplicates = 1
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                duplicates += 1
                continue
            g = gcd(dy, dx)
            slope = (dy // g, dx // g) if dx != 0 else ('inf', 0)
            slope_count[slope] += 1
        current_max = max(slope_count.values(), default=0)
        max_points = max(max_points, current_max + duplicates)
    return max_points
```

### Time Complexity
- O(n^2), pairwise slope comparison.

***

Please let me know if you want detailed explanations or code walkthroughs on any of these problems!
