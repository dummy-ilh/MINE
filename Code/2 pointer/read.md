# LeetCode Two Pointers — Medium: Problems + Python Solutions

🔒 = Premium-locked. ⚠️ = Tricky edge cases — verify against the actual problem page before submitting.

---

### 1. LC 5 — Longest Palindromic Substring
Find the longest palindromic substring in `s`.
```python
def longestPalindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]
    res = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i + 1)
        res = max(res, odd, even, key=len)
    return res
```


### 3. LC 15 — 3Sum
Find all unique triplets that sum to zero.
```python
def threeSum(nums):
    nums.sort()
    n = len(nums)
    res = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                while l < r and nums[r] == nums[r + 1]:
                    r -= 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return res
```

### 4. LC 16 — 3Sum Closest
Find the triplet sum closest to a target.
```python
def threeSumClosest(nums, target):
    nums.sort()
    n = len(nums)
    best = nums[0] + nums[1] + nums[2]
    for i in range(n):
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if abs(s - target) < abs(best - target):
                best = s
            if s < target:
                l += 1
            elif s > target:
                r -= 1
            else:
                return s
    return best
```

### 5. LC 18 — 4Sum
Find all unique quadruplets that sum to a target.
```python
def fourSum(nums, target):
    nums.sort()
    n = len(nums)
    res = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            l, r = j + 1, n - 1
            while l < r:
                s = nums[i] + nums[j] + nums[l] + nums[r]
                if s == target:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
                elif s < target:
                    l += 1
                else:
                    r -= 1
    return res
```

### 7. LC 31 — Next Permutation
Rearrange numbers into the next lexicographically greater permutation, in place.
```python
def nextPermutation(nums):
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    l, r = i + 1, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
```


### 9. LC 75 — Sort Colors
Sort an array of 0s, 1s, 2s in place (Dutch national flag).
```python
def sortColors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```

### 10. LC 80 — Remove Duplicates from Sorted Array II
Remove duplicates in-place so each value appears at most twice.
```python
def removeDuplicates(nums):
    slow = 0
    for fast in range(len(nums)):
        if slow < 2 or nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

### 16. LC 151 — Reverse Words in a String
Reverse the order of words in a sentence.
```python
def reverseWords(s):
    return ' '.join(reversed(s.split()))
```

### 17. LC 161 🔒 — One Edit Distance
Check if two strings differ by exactly one edit (insert, delete, or replace).
```python
def isOneEditDistance(s, t):
    if abs(len(s) - len(t)) > 1:
        return False
    i = j = 0
    edited = False
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            if edited:
                return False
            edited = True
            if len(s) == len(t):
                i += 1
                j += 1
            elif len(s) < len(t):
                j += 1
            else:
                i += 1
    return edited or i < len(s) or j < len(t)
```

### 18. LC 165 — Compare Version Numbers
Compare two version strings numerically, revision by revision.
```python
def compareVersion(version1, version2):
    v1, v2 = version1.split('.'), version2.split('.')
    n = max(len(v1), len(v2))
    for i in range(n):
        x1 = int(v1[i]) if i < len(v1) else 0
        x2 = int(v2[i]) if i < len(v2) else 0
        if x1 != x2:
            return 1 if x1 > x2 else -1
    return 0
```


### 20. LC 186 🔒 — Reverse Words in a String II
Reverse word order of a char array in-place.
```python
def reverseWords(s):
    s.reverse()
    n = len(s)
    start = 0
    for i in range(n + 1):
        if i == n or s[i] == ' ':
            l, r = start, i - 1
            while l < r:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
            start = i + 1
```

### 21. LC 189 — Rotate Array
Rotate an array to the right by k steps, in place.
```python
def rotate(nums, k):
    n = len(nums)
    k %= n
    def reverse(l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
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

### 24. LC 253 🔒 — Meeting Rooms II
Find the minimum number of meeting rooms needed.
```python
def minMeetingRooms(intervals):
    starts = sorted(i[0] for i in intervals)
    ends = sorted(i[1] for i in intervals)
    i = j = rooms = max_rooms = 0
    while i < len(starts):
        if starts[i] < ends[j]:
            rooms += 1
            i += 1
            max_rooms = max(max_rooms, rooms)
        else:
            rooms -= 1
            j += 1
    return max_rooms
```

### 25. LC 259 🔒 — 3Sum Smaller
Count triplets whose sum is less than a target.
```python
def threeSumSmaller(nums, target):
    nums.sort()
    n = len(nums)
    count = 0
    for i in range(n):
        l, r = i + 1, n - 1
        while l < r:
            if nums[i] + nums[l] + nums[r] < target:
                count += r - l
                l += 1
            else:
                r -= 1
    return count
```

### 26. LC 277 🔒 — Find the Celebrity
Find the person everyone knows but who knows no one, using `knows(a, b)`.
```python
def findCelebrity(n):
    candidate = 0
    for i in range(1, n):
        if knows(candidate, i):
            candidate = i
    for i in range(n):
        if i != candidate and (knows(candidate, i) or not knows(i, candidate)):
            return -1
    return candidate
```

### 27. LC 287 — Find the Duplicate Number
Find the duplicate in an array of n+1 integers in range [1, n], O(1) space.
```python
def findDuplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow2 = nums[0]
    while slow2 != slow:
        slow2 = nums[slow2]
        slow = nums[slow]
    return slow
```

### 28. LC 360 🔒 — Sort Transformed Array
Given a sorted array and quadratic coefficients, return the transformed values sorted.
```python
def sortTransformedArray(nums, a, b, c):
    def f(x):
        return a * x * x + b * x + c
    n = len(nums)
    res = [0] * n
    l, r = 0, n - 1
    idx = n - 1 if a >= 0 else 0
    while l <= r:
        fl, fr = f(nums[l]), f(nums[r])
        if a >= 0:
            if fl > fr:
                res[idx] = fl
                l += 1
            else:
                res[idx] = fr
                r -= 1
            idx -= 1
        else:
            if fl < fr:
                res[idx] = fl
                l += 1
            else:
                res[idx] = fr
                r -= 1
            idx += 1
    return res
```

### 29. LC 443 — String Compression
Compress a char array in-place using counts for repeated chars.
```python
def compress(chars):
    write = read = 0
    n = len(chars)
    while read < n:
        char = chars[read]
        count = 0
        while read < n and chars[read] == char:
            read += 1
            count += 1
        chars[write] = char
        write += 1
        if count > 1:
            for digit in str(count):
                chars[write] = digit
                write += 1
    return write
```

### 30. LC 457 — Circular Array Loop
Detect a cycle in a circular array where movement direction must stay consistent.
```python
def circularArrayLoop(nums):
    n = len(nums)
    def next_idx(i):
        return (i + nums[i]) % n
    for i in range(n):
        slow, fast = i, i
        while nums[slow] * nums[next_idx(fast)] > 0 and nums[slow] * nums[next_idx(next_idx(fast))] > 0:
            slow = next_idx(slow)
            fast = next_idx(next_idx(fast))
            if slow == fast:
                if slow != next_idx(slow):
                    return True
                break
    return False
```

### 31. LC 475 — Heaters
Find the minimum heater radius to warm all houses.
```python
import bisect

def findRadius(houses, heaters):
    heaters.sort()
    res = 0
    for h in houses:
        idx = bisect.bisect_left(heaters, h)
        left = heaters[idx - 1] if idx > 0 else float('-inf')
        right = heaters[idx] if idx < len(heaters) else float('inf')
        res = max(res, min(h - left, right - h))
    return res
```

### 32. LC 481 — Magical String
Count occurrences of '1' in the first n characters of the self-describing magical string.
```python
def magicalString(n):
    if n == 0:
        return 0
    s = [1, 2, 2]
    i = 2
    while len(s) < n:
        next_val = 3 - s[-1]
        s += [next_val] * s[i]
        i += 1
    return s[:n].count(1)
```

### 33. LC 522 — Longest Uncommon Subsequence II
Find the longest string that isn't a subsequence of any other string in the array.
```python
def findLUSlength(strs):
    def is_subsequence(a, b):
        it = iter(b)
        return all(c in it for c in a)
    n = len(strs)
    res = -1
    for i in range(n):
        if all(i == j or not is_subsequence(strs[i], strs[j]) for j in range(n)):
            res = max(res, len(strs[i]))
    return res
```

### 34. LC 524 — Longest Word in Dictionary through Deleting
Find the longest dictionary word formable by deleting characters from `s`.
```python
def findLongestWord(s, dictionary):
    def is_subsequence(word):
        it = iter(s)
        return all(c in it for c in word)
    best = ""
    for word in dictionary:
        if is_subsequence(word):
            if len(word) > len(best) or (len(word) == len(best) and word < best):
                best = word
    return best
```

### 35. LC 532 — K-diff Pairs in an Array
Count unique pairs with an absolute difference of k.
```python
def findPairs(nums, k):
    nums.sort()
    n = len(nums)
    l, r, count = 0, 1, 0
    while r < n:
        if l == r or nums[r] - nums[l] < k:
            r += 1
        elif nums[r] - nums[l] > k:
            l += 1
        else:
            count += 1
            l += 1
            while l < n and nums[l] == nums[l - 1]:
                l += 1
    return count
```

### 36. LC 556 — Next Greater Element III
Find the smallest integer greater than n using the same digits.
```python
def nextGreaterElement(n):
    digits = list(str(n))
    i = len(digits) - 2
    while i >= 0 and digits[i] >= digits[i + 1]:
        i -= 1
    if i < 0:
        return -1
    j = len(digits) - 1
    while digits[j] <= digits[i]:
        j -= 1
    digits[i], digits[j] = digits[j], digits[i]
    digits[i + 1:] = reversed(digits[i + 1:])
    result = int(''.join(digits))
    return result if result <= 2**31 - 1 else -1
```

### 37. LC 567 — Permutation in String
Check if s2 contains a permutation of s1 as a substring.
```python
from collections import Counter

def checkInclusion(s1, s2):
    n, m = len(s1), len(s2)
    if n > m:
        return False
    need = Counter(s1)
    window = Counter(s2[:n])
    if window == need:
        return True
    for i in range(n, m):
        window[s2[i]] += 1
        window[s2[i - n]] -= 1
        if window[s2[i - n]] == 0:
            del window[s2[i - n]]
        if window == need:
            return True
    return False
```

### 38. LC 581 — Shortest Unsorted Continuous Subarray
Find the shortest subarray that, if sorted, makes the whole array sorted.
```python
def findUnsortedSubarray(nums):
    n = len(nums)
    sorted_nums = sorted(nums)
    l, r = 0, n - 1
    while l < n and nums[l] == sorted_nums[l]:
        l += 1
    while r > l and nums[r] == sorted_nums[r]:
        r -= 1
    return r - l + 1 if l < r else 0
```

### 39. LC 611 — Valid Triangle Number
Count triplets that can form a valid triangle.
```python
def triangleNumber(nums):
    nums.sort()
    n = len(nums)
    count = 0
    for k in range(n - 1, 1, -1):
        l, r = 0, k - 1
        while l < r:
            if nums[l] + nums[r] > nums[k]:
                count += r - l
                r -= 1
            else:
                l += 1
    return count
```

### 40. LC 633 — Sum of Square Numbers
Check if c can be expressed as the sum of two squares.
```python
import math

def judgeSquareSum(c):
    l, r = 0, int(math.sqrt(c))
    while l <= r:
        s = l * l + r * r
        if s == c:
            return True
        elif s < c:
            l += 1
        else:
            r -= 1
    return False
```

### 41. LC 647 — Palindromic Substrings
Count all palindromic substrings.
```python
def countSubstrings(s):
    n = len(s)
    count = 0
    def expand(l, r):
        nonlocal count
        while l >= 0 and r < n and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
    for i in range(n):
        expand(i, i)
        expand(i, i + 1)
    return count
```

### 42. LC 658 — Find K Closest Elements
Find the k closest elements to x in a sorted array.
```python
def findClosestElements(arr, k, x):
    l, r = 0, len(arr) - k
    while l < r:
        mid = (l + r) // 2
        if x - arr[mid] > arr[mid + k] - x:
            l = mid + 1
        else:
            r = mid
    return arr[l:l + k]
```

### 43. LC 723 🔒 — Candy Crush
Simulate crushing groups of 3+ same-value candies until stable.
```python
def candyCrush(board):
    rows, cols = len(board), len(board[0])
    changed = True
    while changed:
        changed = False
        crush = set()
        for r in range(rows):
            for c in range(cols - 2):
                v = abs(board[r][c])
                if v and v == abs(board[r][c + 1]) == abs(board[r][c + 2]):
                    crush.update([(r, c), (r, c + 1), (r, c + 2)])
        for c in range(cols):
            for r in range(rows - 2):
                v = abs(board[r][c])
                if v and v == abs(board[r + 1][c]) == abs(board[r + 2][c]):
                    crush.update([(r, c), (r + 1, c), (r + 2, c)])
        if crush:
            changed = True
            for r, c in crush:
                board[r][c] = -abs(board[r][c])
            for c in range(cols):
                write = rows - 1
                for r in range(rows - 1, -1, -1):
                    if board[r][c] > 0:
                        board[write][c] = board[r][c]
                        write -= 1
                for r in range(write, -1, -1):
                    board[r][c] = 0
    return board
```

### 44. LC 763 — Partition Labels
Partition a string into the max number of parts so each letter appears in only one part.
```python
def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    res = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            res.append(end - start + 1)
            start = i + 1
    return res
```

### 45. LC 777 🔒 ⚠️ — Swap Adjacent in LR String
Check if `start` can become `end` using L/R adjacent swap rules.
```python
def canTransform(start, end):
    if len(start) != len(end):
        return False
    s = [(c, i) for i, c in enumerate(start) if c != 'X']
    e = [(c, i) for i, c in enumerate(end) if c != 'X']
    if len(s) != len(e):
        return False
    for (c1, i1), (c2, i2) in zip(s, e):
        if c1 != c2:
            return False
        if c1 == 'L' and i1 < i2:
            return False
        if c1 == 'R' and i1 > i2:
            return False
    return True
```

### 46. LC 786 🔒 ⚠️ — K-th Smallest Prime Fraction
Find the kth smallest fraction `arr[i]/arr[j]` for i < j.
```python
def kthSmallestPrimeFraction(arr, k):
    n = len(arr)
    l, r = 0.0, 1.0
    while True:
        mid = (l + r) / 2
        count, j = 0, 1
        best = (0, 1)
        for i in range(n):
            while j < n and arr[i] >= mid * arr[j]:
                j += 1
            count += n - j
            if j < n and arr[i] * best[1] > best[0] * arr[j]:
                best = (arr[i], arr[j])
        if count == k:
            return [best[0], best[1]]
        elif count < k:
            l = mid
        else:
            r = mid
```

### 47. LC 795 — Number of Subarrays with Bounded Maximum
Count subarrays where the max element is within [left, right].
```python
def numSubarrayBoundedMax(nums, left, right):
    def count(bound):
        res = run = 0
        for x in nums:
            run = run + 1 if x <= bound else 0
            res += run
        return res
    return count(right) - count(left - 1)
```

### 48. LC 809 — Expressive Words
Count words that can stretch into a given target string via repeated chars (3+ rule).
```python
def expressiveWords(s, words):
    def get_groups(string):
        groups, i, n = [], 0, len(string)
        while i < n:
            j = i
            while j < n and string[j] == string[i]:
                j += 1
            groups.append((string[i], j - i))
            i = j
        return groups

    s_groups = get_groups(s)
    count = 0
    for word in words:
        w_groups = get_groups(word)
        if len(w_groups) != len(s_groups):
            continue
        ok = True
        for (c1, n1), (c2, n2) in zip(s_groups, w_groups):
            if c1 != c2 or n1 < n2 or (n1 != n2 and n1 < 3):
                ok = False
                break
        if ok:
            count += 1
    return count
```

### 49. LC 825 ⚠️ — Friends Of Appropriate Ages
Count friend requests under the age-appropriateness rule.
```python
def numFriendRequests(ages):
    count = [0] * 121
    for age in ages:
        count[age] += 1
    prefix = [0] * 122
    for i in range(1, 121):
        prefix[i + 1] = prefix[i] + count[i]
    res = 0
    for ageA in range(1, 121):
        if count[ageA] == 0:
            continue
        lower = ageA // 2 + 7
        if lower >= ageA:
            continue
        cnt = prefix[ageA + 1] - prefix[lower + 1]
        res += cnt * count[ageA] - count[ageA]
    return res
```

### 50. LC 826 — Most Profit Assigning Work
Assign workers to jobs (within their ability) to maximize total profit.
```python
def maxProfitAssignment(difficulty, profit, worker):
    jobs = sorted(zip(difficulty, profit))
    worker.sort()
    i, best, total = 0, 0, 0
    for w in worker:
        while i < len(jobs) and jobs[i][0] <= w:
            best = max(best, jobs[i][1])
            i += 1
        total += best
    return total
```

### 51. LC 838 — Push Dominoes
Simulate dominoes falling left/right or staying upright.
```python
def pushDominoes(dominoes):
    s = list(dominoes)
    n = len(s)
    prev, prev_char = -1, 'L'
    for i in range(n + 1):
        if i == n or s[i] != '.':
            curr_char = s[i] if i < n else 'R'
            if prev_char == 'L' and curr_char == 'L':
                for k in range(prev + 1, i):
                    s[k] = 'L'
            elif prev_char == 'R' and curr_char == 'R':
                for k in range(prev + 1, i):
                    s[k] = 'R'
            elif prev_char == 'R' and curr_char == 'L':
                lo, hi = prev + 1, i - 1
                while lo < hi:
                    s[lo], s[hi] = 'R', 'L'
                    lo += 1
                    hi -= 1
            prev, prev_char = i, curr_char
    return ''.join(s)
```

### 52. LC 845 — Longest Mountain in Array
Find the length of the longest "mountain" subarray (strictly up then strictly down).
```python
def longestMountain(arr):
    n = len(arr)
    best, i = 0, 1
    while i < n - 1:
        if arr[i - 1] < arr[i] > arr[i + 1]:
            l = i - 1
            while l > 0 and arr[l - 1] < arr[l]:
                l -= 1
            r = i + 1
            while r < n - 1 and arr[r] > arr[r + 1]:
                r += 1
            best = max(best, r - l + 1)
            i = r
        else:
            i += 1
    return best
```

### 53. LC 870 — Advantage Shuffle
Rearrange nums1 to maximize the count of elements that beat the corresponding nums2 element.
```python
def advantageCount(nums1, nums2):
    sorted1 = sorted(nums1)
    sorted2_idx = sorted(range(len(nums2)), key=lambda i: nums2[i])
    res = [0] * len(nums1)
    l, r = 0, len(sorted1) - 1
    for idx in reversed(sorted2_idx):
        if sorted1[r] > nums2[idx]:
            res[idx] = sorted1[r]
            r -= 1
        else:
            res[idx] = sorted1[l]
            l += 1
    return res
```

### 54. LC 881 — Boats to Save People
Find the minimum boats needed, each carrying at most 2 people within a weight limit.
```python
def numRescueBoats(people, limit):
    people.sort()
    l, r, boats = 0, len(people) - 1, 0
    while l <= r:
        if people[l] + people[r] <= limit:
            l += 1
        r -= 1
        boats += 1
    return boats
```

### 55. LC 923 — 3Sum With Multiplicity
Count triplets (with repeats allowed) summing to a target, mod 1e9+7.
```python
def threeSumMulti(arr, target):
    MOD = 10**9 + 7
    arr.sort()
    n = len(arr)
    res = 0
    for i in range(n):
        l, r = i + 1, n - 1
        t = target - arr[i]
        while l < r:
            if arr[l] + arr[r] < t:
                l += 1
            elif arr[l] + arr[r] > t:
                r -= 1
            elif arr[l] != arr[r]:
                lc = rc = 1
                while l + 1 < r and arr[l] == arr[l + 1]:
                    lc += 1
                    l += 1
                while r - 1 > l and arr[r] == arr[r - 1]:
                    rc += 1
                    r -= 1
                res += lc * rc
                l += 1
                r -= 1
            else:
                count = r - l + 1
                res += count * (count - 1) // 2
                break
    return res % MOD
```

### 56. LC 948 — Bag of Tokens
Maximize score by trading tokens for power (gain score) or score for power (lose score, gain power).
```python
def bagOfTokensScore(tokens, power):
    tokens.sort()
    l, r = 0, len(tokens) - 1
    score = best = 0
    while l <= r:
        if power >= tokens[l]:
            power -= tokens[l]
            score += 1
            best = max(best, score)
            l += 1
        elif score > 0:
            power += tokens[r]
            score -= 1
            r -= 1
        else:
            break
    return best
```

### 57. LC 962 — Maximum Width Ramp
Find the max width j-i such that nums[i] <= nums[j] and i < j.
```python
def maxWidthRamp(nums):
    stack = []
    best = 0
    for i, v in enumerate(nums):
        if not stack or nums[stack[-1]] > v:
            stack.append(i)
        else:
            while stack and nums[stack[-1]] <= v:
                best = max(best, i - stack[-1])
                stack.pop()
    return best
```

### 58. LC 969 — Pancake Sorting
Sort an array using only "pancake flip" prefix-reversal operations.
```python
def pancakeSort(arr):
    res = []
    n = len(arr)
    for size in range(n, 1, -1):
        idx = arr.index(size)
        if idx != size - 1:
            res.append(idx + 1)
            arr[:idx + 1] = reversed(arr[:idx + 1])
            res.append(size)
            arr[:size] = reversed(arr[:size])
    return res
```

### 59. LC 986 — Interval List Intersections
Find the intersection of two lists of disjoint, sorted intervals.
```python
def intervalIntersection(firstList, secondList):
    i, j, res = 0, 0, []
    while i < len(firstList) and j < len(secondList):
        lo = max(firstList[i][0], secondList[j][0])
        hi = min(firstList[i][1], secondList[j][1])
        if lo <= hi:
            res.append([lo, hi])
        if firstList[i][1] < secondList[j][1]:
            i += 1
        else:
            j += 1
    return res
```

### 60. LC 1023 — Camelcase Matching
Check which queries match a pattern (each uppercase letter must appear in order; lowercase can be inserted freely).
```python
def camelMatch(queries, pattern):
    def matches(q):
        i = 0
        for c in q:
            if i < len(pattern) and c == pattern[i]:
                i += 1
            elif c.isupper():
                return False
        return i == len(pattern)
    return [matches(q) for q in queries]
```

### 61. LC 1048 — Longest String Chain
Find the longest chain of words where each is the previous plus one character.
```python
def longestStrChain(words):
    words.sort(key=len)
    dp = {}
    best = 1
    for word in words:
        dp[word] = 1
        for i in range(len(word)):
            pred = word[:i] + word[i + 1:]
            if pred in dp:
                dp[word] = max(dp[word], dp[pred] + 1)
        best = max(best, dp[word])
    return best
```

### 62. LC 1055 🔒 ⚠️ — Shortest Way to Form String
Find the minimum number of subsequences of `source` needed to form `target`.
```python
def shortestWay(source, target):
    i = 0
    count = 0
    while i < len(target):
        start = i
        for c in source:
            if i < len(target) and c == target[i]:
                i += 1
        if i == start:
            return -1
        count += 1
    return count
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

### 64. LC 1229 🔒 — Meeting Scheduler
Find the earliest common free slot of given duration between two schedules.
```python
def minAvailableDuration(slots1, slots2, duration):
    slots1.sort()
    slots2.sort()
    i = j = 0
    while i < len(slots1) and j < len(slots2):
        start = max(slots1[i][0], slots2[j][0])
        end = min(slots1[i][1], slots2[j][1])
        if end - start >= duration:
            return [start, start + duration]
        if slots1[i][1] < slots2[j][1]:
            i += 1
        else:
            j += 1
    return []
```

### 65. LC 1237 — Find Positive Integer Solution for a Given Equation
Find all (x, y) pairs satisfying a monotonic custom function equal to z.
```python
def findSolution(customfunction, z):
    res = []
    x, y = 1, 1000
    while x <= 1000 and y >= 1:
        val = customfunction.f(x, y)
        if val == z:
            res.append([x, y])
            x += 1
            y -= 1
        elif val < z:
            x += 1
        else:
            y -= 1
    return res
```

### 66. LC 1265 🔒 — Print Immutable Linked List in Reverse
Print an immutable linked list's values in reverse order using only its limited API.
```python
def printLinkedListInReverse(head):
    if head:
        printLinkedListInReverse(head.getNext())
        head.printValue()
```

### 67. LC 1471 — The k Strongest Values in an Array
Return the k "strongest" values, defined by distance from the median (ties favor larger value).
```python
def getStrongest(arr, k):
    arr.sort()
    median = arr[(len(arr) - 1) // 2]
    arr.sort(key=lambda x: (-abs(x - median), -x))
    return arr[:k]
```

### 68. LC 1498 — Number of Subsequences That Satisfy the Given Sum Condition
Count subsequences where min + max <= target, mod 1e9+7.
```python
def numSubseq(nums, target):
    MOD = 10**9 + 7
    nums.sort()
    n = len(nums)
    l, r, res = 0, n - 1, 0
    while l <= r:
        if nums[l] + nums[r] > target:
            r -= 1
        else:
            res += pow(2, r - l, MOD)
            l += 1
    return res % MOD
```

### 69. LC 1508 — Range Sum of Sorted Subarray Sums
Sum the sorted list of all subarray sums, restricted to a given index range.
```python
def rangeSum(nums, n, left, right):
    MOD = 10**9 + 7
    sums = []
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += nums[j]
            sums.append(total)
    sums.sort()
    return sum(sums[left - 1:right]) % MOD
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

### 71. LC 1574 ⚠️ — Shortest Subarray to be Removed to Make Array Sorted
Find the length of the shortest subarray to remove so the rest is non-decreasing.
```python
def findLengthOfShortestSubarray(arr):
    n = len(arr)
    l, r = 0, n - 1
    while l < n - 1 and arr[l] <= arr[l + 1]:
        l += 1
    if l == n - 1:
        return 0
    while r > 0 and arr[r - 1] <= arr[r]:
        r -= 1
    res = min(n - l - 1, r)
    i, j = 0, r
    while i <= l and j < n:
        if arr[i] <= arr[j]:
            res = min(res, j - i - 1)
            i += 1
        else:
            j += 1
    return res
```

### 72. LC 1577 — Number of Ways Where Square of Number Is Equal to Product of Two Numbers
Count triplets where one array's value squared equals the product of two values in the other.
```python
from collections import Counter

def numTriplets(nums1, nums2):
    def count_triplets(a, b):
        squares = Counter(x * x for x in a)
        res, n = 0, len(b)
        for i in range(n):
            for j in range(i + 1, n):
                res += squares.get(b[i] * b[j], 0)
        return res
    return count_triplets(nums1, nums2) + count_triplets(nums2, nums1)
```

### 73. LC 1616 — Split Two Strings to Make Palindrome
Check if splitting and swapping prefixes/suffixes of two strings can form a palindrome.
```python
def checkPalindromeFormation(a, b):
    def is_pal(s, l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

    def check(a, b):
        l, r = 0, len(a) - 1
        while l < r and a[l] == b[r]:
            l += 1
            r -= 1
        return is_pal(a, l, r) or is_pal(b, l, r)

    return check(a, b) or check(b, a)
```

### 74. LC 1634 🔒 — Add Two Polynomials Represented as Linked Lists
Add two polynomials represented as sorted linked lists of (coefficient, power).
```python
def addPoly(poly1, poly2):
    dummy = curr = PolyNode()
    p1, p2 = poly1, poly2
    while p1 and p2:
        if p1.power > p2.power:
            curr.next = PolyNode(p1.coefficient, p1.power)
            p1 = p1.next
        elif p1.power < p2.power:
            curr.next = PolyNode(p2.coefficient, p2.power)
            p2 = p2.next
        else:
            coeff = p1.coefficient + p2.coefficient
            if coeff != 0:
                curr.next = PolyNode(coeff, p1.power)
            p1, p2 = p1.next, p2.next
        if curr.next:
            curr = curr.next
    curr.next = p1 or p2
    return dummy.next
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

### 76. LC 1679 — Max Number of K-Sum Pairs
Find the max number of disjoint pairs summing to k.
```python
def maxOperations(nums, k):
    nums.sort()
    l, r, count = 0, len(nums) - 1, 0
    while l < r:
        s = nums[l] + nums[r]
        if s == k:
            count += 1
            l += 1
            r -= 1
        elif s < k:
            l += 1
        else:
            r -= 1
    return count
```

### 77. LC 1712 ⚠️ — Ways to Split Array Into Three Subarrays
Count ways to split into 3 contiguous parts where each part's sum is <= the next.
```python
import bisect

def waysToSplit(nums):
    MOD = 10**9 + 7
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    total = prefix[n]
    res = 0
    for i in range(1, n - 1):
        left = prefix[i]
        if left > total - left:
            break
        lo = bisect.bisect_left(prefix, 2 * left, i + 1, n)
        hi = bisect.bisect_right(prefix, (total + left) / 2, i + 1, n)
        res += max(0, hi - lo)
    return res % MOD
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

### 79. LC 1750 — Minimum Length of String After Deleting Similar Ends
Repeatedly delete matching prefix/suffix blocks; return the final length.
```python
def minimumLength(s):
    l, r = 0, len(s) - 1
    while l < r and s[l] == s[r]:
        c = s[l]
        while l <= r and s[l] == c:
            l += 1
        while l <= r and s[r] == c:
            r -= 1
    return r - l + 1
```

### 80. LC 1754 — Largest Merge Of Two Strings
Greedily merge two strings to form the lexicographically largest result.
```python
def largestMerge(word1, word2):
    res, i, j = [], 0, 0
    while i < len(word1) and j < len(word2):
        if word1[i:] > word2[j:]:
            res.append(word1[i])
            i += 1
        else:
            res.append(word2[j])
            j += 1
    res.append(word1[i:])
    res.append(word2[j:])
    return ''.join(res)
```

### 81. LC 1764 — Form Array by Concatenating Subarrays of Another Array
Check if `groups` can be found as non-overlapping, in-order subarrays of `nums`.
```python
def canChoose(groups, nums):
    i = 0
    for g in groups:
        found = False
        while i + len(g) <= len(nums):
            if nums[i:i + len(g)] == g:
                found = True
                i += len(g)
                break
            i += 1
        if not found:
            return False
    return True
```

### 82. LC 1813 — Sentence Similarity III
Check if one sentence can be formed by inserting a block of words into the other.
```python
def areSentencesSimilar(sentence1, sentence2):
    w1, w2 = sentence1.split(), sentence2.split()
    if len(w1) > len(w2):
        w1, w2 = w2, w1
    i = 0
    while i < len(w1) and w1[i] == w2[i]:
        i += 1
    j = 0
    while j < len(w1) - i and w1[-1 - j] == w2[-1 - j]:
        j += 1
    return i + j >= len(w1)
```

### 83. LC 1850 ⚠️ — Minimum Adjacent Swaps to Reach the Kth Smallest Number
Find min adjacent digit swaps to reach the kth next permutation of a number's digits.
```python
def getMinSwaps(num, k):
    def next_permutation(arr):
        n = len(arr)
        i = n - 2
        while i >= 0 and arr[i] >= arr[i + 1]:
            i -= 1
        if i >= 0:
            j = n - 1
            while arr[j] <= arr[i]:
                j -= 1
            arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1:] = reversed(arr[i + 1:])

    target = list(num)
    for _ in range(k):
        next_permutation(target)

    digits = list(num)
    swaps = 0
    for i in range(len(digits)):
        if digits[i] != target[i]:
            j = i
            while digits[j] != target[i]:
                j += 1
            while j > i:
                digits[j], digits[j - 1] = digits[j - 1], digits[j]
                j -= 1
                swaps += 1
    return swaps
```

### 84. LC 1855 — Maximum Distance Between a Pair of Values
Find max j - i where nums1[i] <= nums2[j], i <= j, across two non-increasing arrays.
```python
def maxDistance(nums1, nums2):
    i = j = res = 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] > nums2[j]:
            i += 1
            if i > j:
                j = i
        else:
            res = max(res, j - i)
            j += 1
    return res
```

### 85. LC 1861 — Rotating the Box
Simulate gravity pulling stones ('#') right in each row, then rotate the box 90° clockwise.
```python
def rotateTheBox(box):
    for row in box:
        write = len(row) - 1
        for c in range(len(row) - 1, -1, -1):
            if row[c] == '*':
                write = c - 1
            elif row[c] == '#':
                row[c] = '.'
                row[write] = '#'
                write -= 1
    return [list(row) for row in zip(*box[::-1])]
```

### 86. LC 1868 🔒 — Product of Two Run-Length Encoded Arrays
Multiply two run-length encoded arrays elementwise, returning the RLE result.
```python
def findRLEArray(encoded1, encoded2):
    res, i, j = [], 0, 0
    while i < len(encoded1) and j < len(encoded2):
        val1, freq1 = encoded1[i]
        val2, freq2 = encoded2[j]
        product = val1 * val2
        freq = min(freq1, freq2)
        if res and res[-1][0] == product:
            res[-1][1] += freq
        else:
            res.append([product, freq])
        encoded1[i][1] -= freq
        encoded2[j][1] -= freq
        if encoded1[i][1] == 0:
            i += 1
        if encoded2[j][1] == 0:
            j += 1
    return res
```

### 87. LC 1877 — Minimize Maximum Pair Sum in Array
Pair up elements to minimize the maximum pair sum.
```python
def minPairSum(nums):
    nums.sort()
    l, r, best = 0, len(nums) - 1, 0
    while l < r:
        best = max(best, nums[l] + nums[r])
        l += 1
        r -= 1
    return best
```

### 88. LC 1885 🔒 — Count Pairs in Two Arrays
Count pairs (i, j), i < j, where nums1[i]+nums1[j] > nums2[i]+nums2[j].
```python
def countPairs(nums1, nums2):
    n = len(nums1)
    diff = sorted(nums1[i] - nums2[i] for i in range(n))
    l, r, count = 0, n - 1, 0
    while l < r:
        if diff[l] + diff[r] > 0:
            count += r - l
            r -= 1
        else:
            l += 1
    return count
```

### 89. LC 1898 — Maximum Number of Removable Characters
Find the max prefix of `removable` indices you can remove while `p` stays a subsequence of `s`.
```python
def maximumRemovals(s, p, removable):
    def is_subsequence(k):
        removed = set(removable[:k])
        i = 0
        for j, c in enumerate(s):
            if j in removed:
                continue
            if i < len(p) and c == p[i]:
                i += 1
        return i == len(p)

    l, r = 0, len(removable)
    while l < r:
        mid = (l + r + 1) // 2
        if is_subsequence(mid):
            l = mid
        else:
            r = mid - 1
    return l
```

### 90. LC 1963 — Minimum Number of Swaps to Make the String Balanced
Find the minimum adjacent swaps to balance a bracket string.
```python
def minSwaps(s):
    balance = 0
    max_imbalance = 0
    for c in s:
        balance += 1 if c == '[' else -1
        max_imbalance = max(max_imbalance, -balance)
    return (max_imbalance + 1) // 2
```

### 91. LC 1989 🔒 ⚠️ — Maximum Number of People That Can Be Caught in Tag
Maximize the number of "it" players catching non-"it" players within a distance.
```python
def catchMaximumAmountofPeople(team, dist):
    n = len(team)
    res = 0
    i = j = 0
    while i < n and j < n:
        if team[i] == 0:
            i += 1
        elif team[j] == 1:
            j += 1
        elif abs(i - j) <= dist:
            res += 1
            i += 1
            j += 1
        elif i < j:
            i += 1
        else:
            j += 1
    return res
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

### 94. LC 2105 — Watering Plants II
Two people water plants from opposite ends with separate cans; count refills.
```python
def minimumRefill(plants, capacityA, capacityB):
    i, j = 0, len(plants) - 1
    a, b = capacityA, capacityB
    refills = 0
    while i < j:
        if a < plants[i]:
            refills += 1
            a = capacityA
        a -= plants[i]
        if b < plants[j]:
            refills += 1
            b = capacityB
        b -= plants[j]
        i += 1
        j -= 1
    if i == j and max(a, b) < plants[i]:
        refills += 1
    return refills
```

### 95. LC 2109 — Adding Spaces to a String
Insert spaces into a string at given sorted indices.
```python
def addSpaces(s, spaces):
    res, j = [], 0
    for i, c in enumerate(s):
        if j < len(spaces) and spaces[j] == i:
            res.append(' ')
            j += 1
        res.append(c)
    return ''.join(res)
```

### 96. LC 2110 — Number of Smooth Descent Periods of a Stock
Count subarrays where each price is exactly 1 less than the previous.
```python
def getDescentPeriods(prices):
    n = len(prices)
    res = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and prices[j + 1] == prices[j] - 1:
            j += 1
        length = j - i + 1
        res += length * (length + 1) // 2
        i = j + 1
    return res
```

### 97. LC 2130 — Maximum Twin Sum of a Linked List
Find the max sum of "twin" nodes (i-th from start + i-th from end) in an even-length list.
```python
def pairSum(head):
    vals = []
    while head:
        vals.append(head.val)
        head = head.next
    l, r, best = 0, len(vals) - 1, 0
    while l < r:
        best = max(best, vals[l] + vals[r])
        l += 1
        r -= 1
    return best
```

### 98. LC 2149 — Rearrange Array Elements by Sign
Rearrange so positive and negative numbers alternate, starting with positive.
```python
def rearrangeArray(nums):
    pos = [x for x in nums if x > 0]
    neg = [x for x in nums if x < 0]
    res = []
    for p, n in zip(pos, neg):
        res.append(p)
        res.append(n)
    return res
```

### 99. LC 2161 — Partition Array According to Given Pivot
Rearrange so elements < pivot come first, then == pivot, then > pivot (stable order).
```python
def pivotArray(nums, pivot):
    less = [x for x in nums if x < pivot]
    equal = [x for x in nums if x == pivot]
    greater = [x for x in nums if x > pivot]
    return less + equal + greater
```

### 100. LC 2300 — Successful Pairs of Spells and Potions
For each spell, count potions whose product with it meets a success threshold.
```python
def successfulPairs(spells, potions, success):
    potions.sort()
    n = len(potions)
    res = []
    for s in spells:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if s * potions[mid] >= success:
                hi = mid
            else:
                lo = mid + 1
        res.append(n - lo)
    return res
```

### 101. LC 2330 🔒 — Valid Palindrome IV
Check if a string can become a palindrome by changing at most 2 characters.
```python
def makePalindrome(s):
    l, r, diff = 0, len(s) - 1, 0
    while l < r:
        if s[l] != s[r]:
            diff += 1
        l += 1
        r -= 1
    return diff <= 2
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

### Container With Most Water
``
def maxArea(height):
    l, r = 0, len(height) - 1;    best = 0
    while l < r:
        best = max(best, (r - l) * min(height[l], height[r]))
        if height[l] < height[r]:            l += 1
        else:            r -= 1
```

Reverse the order of words in a sentence.
```python
def reverseWords(s):
    return ' '.join(reversed(s.split()))

        words = s.split()        res = []
or
        for i in range(len(words) - 1, -1, -1):
            res.append(words[i])
            if i != 0:        res.append(" ")

        return "".join(res)
or
        while left < right:
            word[left], word[right] = word[right], word[left]
            left += 1
            right -=1

        return " ".join(word) 
