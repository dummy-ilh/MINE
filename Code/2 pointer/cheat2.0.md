# Two Pointers — Templates + Full Code, Merged

Every problem from your Easy list, filed under the right template, with
the actual code and a one-line note on how it bends the base shape.

---

## 1. Converging ends

Base shape:
```python
def converge(arr):
    l, r = 0, len(arr) - 1
    while l < r:
        if condition(arr[l], arr[r]):
            l += 1; r -= 1
        elif need_more(arr[l], arr[r]):
            l += 1
        else:
            r -= 1
    return result
```

### 344. Reverse String (base case)
```python
def reverseString(s):
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1; r -= 1
```

### 125. Valid Palindrome (base case)
```python
def isPalindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum(): l += 1
        while l < r and not s[r].isalnum(): r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1; r -= 1
    return True
```

### 977. Squares of a Sorted Array (base case — builds result outside-in)
```python
def sortedSquares(nums):
    n = len(nums)
    res = [0] * n
    l, r = 0, n - 1
    for i in range(n - 1, -1, -1):
        if abs(nums[l]) > abs(nums[r]):
            res[i] = nums[l] * nums[l]; l += 1
        else:
            res[i] = nums[r] * nums[r]; r -= 1
    return res
```

### 2824. Count Pairs Whose Sum is Less than Target (base case)
```python
def countPairs(nums, target):
    nums.sort()
    l, r = 0, len(nums) - 1
    count = 0
    while l < r:
        if nums[l] + nums[r] < target:
            count += r - l
            l += 1
        else:
            r -= 1
    return count
```

### 345. Reverse Vowels — skip-and-swap, gated on vowel membership
```python
def reverseVowels(s):
    vowels = set('aeiouAEIOU')
    chars = list(s)
    l, r = 0, len(chars) - 1
    while l < r:
        if chars[l] not in vowels: l += 1
        elif chars[r] not in vowels: r -= 1
        else:
            chars[l], chars[r] = chars[r], chars[l]
            l += 1; r -= 1
    return ''.join(chars)
```

### 917. Reverse Only Letters — same skip-and-swap, gated on `isalpha()`
```python
def reverseOnlyLetters(s):
    chars = list(s)
    l, r = 0, len(chars) - 1
    while l < r:
        if not chars[l].isalpha(): l += 1
        elif not chars[r].isalpha(): r -= 1
        else:
            chars[l], chars[r] = chars[r], chars[l]
            l += 1; r -= 1
    return ''.join(chars)
```

### 3823. Reverse Letters, Keep Specials — identical mechanic to 917
```python
def reverseLettersOnly(s):
    chars = list(s)
    l, r = 0, len(chars) - 1
    while l < r:
        if not chars[l].isalpha(): l += 1
        elif not chars[r].isalpha(): r -= 1
        else:
            chars[l], chars[r] = chars[r], chars[l]
            l += 1; r -= 1
    return ''.join(chars)
```

### 680. Valid Palindrome II — converge, branch into two checks on first mismatch
```python
def validPalindrome(s):
    def is_pal(l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1; r -= 1
        return True

    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return is_pal(l + 1, r) or is_pal(l, r - 1)
        l += 1; r -= 1
    return True
```

### 832. Flipping an Image — swap + XOR fused, `l <= r` self-flips the middle
```python
def flipAndInvertImage(image):
    for row in image:
        l, r = 0, len(row) - 1
        while l <= r:
            row[l], row[r] = row[r] ^ 1, row[l] ^ 1
            l += 1; r -= 1
    return image
```

### 905. Sort Array By Parity — one-directional Lomuto-style partition
```python
def sortArrayByParity(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        if nums[l] % 2 == 0:
            l += 1
        else:
            nums[l], nums[r] = nums[r], nums[l]
            r -= 1
    return nums
```

### 942. DI String Match — virtual `lo`/`hi` bounds, no input array to converge on
```python
def diStringMatch(s):
    lo, hi = 0, len(s)
    res = []
    for c in s:
        if c == 'I':
            res.append(lo); lo += 1
        else:
            res.append(hi); hi -= 1
    res.append(lo)
    return res
```

### 1332. Remove Palindromic Subsequences — converge just to answer yes/no
```python
def removePalindromeSub(s):
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return 2
        l += 1; r -= 1
    return 1
```

### 2000. Reverse Prefix of Word — no loop, slice-reverse instead
```python
def reversePrefix(word, ch):
    if ch not in word:
        return word
    i = word.index(ch)
    return word[:i + 1][::-1] + word[i + 1:]
```

### 2465. Number of Distinct Averages — converge, dedup via set
```python
def distinctAverages(nums):
    nums.sort()
    l, r = 0, len(nums) - 1
    averages = set()
    while l < r:
        averages.add(nums[l] + nums[r])
        l += 1; r -= 1
    return len(averages)
```

### 2511. Maximum Enemy Forts Captured — single forward pointer + `prev` marker (gap-measuring, not converging)
```python
def captureForts(forts):
    res = 0
    prev = -1
    for i, v in enumerate(forts):
        if v != 0:
            if prev != -1 and forts[prev] != v:
                res = max(res, i - prev - 1)
            prev = i
    return res
```

### 2697. Lexicographically Smallest Palindrome — converge, but mutates both chars
```python
def makeSmallestPalindrome(s):
    chars = list(s)
    l, r = 0, len(chars) - 1
    while l < r:
        c = min(chars[l], chars[r])
        chars[l] = chars[r] = c
        l += 1; r -= 1
    return ''.join(chars)
```

### 3643. Flip Square Submatrix Vertically — converge lifted to 2D (row swaps)
```python
def reverseSubmatrix(grid, x, y, k):
    top, bottom = x, x + k - 1
    while top < bottom:
        for c in range(y, y + k):
            grid[top][c], grid[bottom][c] = grid[bottom][c], grid[top][c]
        top += 1; bottom -= 1
    return grid
```

### 3750. Minimum Flips to Reverse Binary String — converge + mismatch counter
```python
def minFlipsToPalindrome(s):
    l, r = 0, len(s) - 1
    flips = 0
    while l < r:
        if s[l] != s[r]:
            flips += 1
        l += 1; r -= 1
    return flips
```

### 3794. Reverse String Prefix — trivial slice, no explicit pointers
```python
def reverseStringPrefix(word, k):
    return word[:k][::-1] + word[k:]
```

### 246. Strobogrammatic Number — converge, equality via dict lookup
```python
def isStrobogrammatic(num):
    mapping = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
    l, r = 0, len(num) - 1
    while l <= r:
        if num[l] not in mapping or mapping[num[l]] != num[r]:
            return False
        l += 1; r -= 1
    return True
```

### 541. Reverse String II — template 1 repeated inside chunks
```python
def reverseStr(s, k):
    chars = list(s)
    for start in range(0, len(chars), 2 * k):
        l, r = start, min(start + k, len(chars)) - 1
        while l < r:
            chars[l], chars[r] = chars[r], chars[l]
            l += 1; r -= 1
    return ''.join(chars)
```

### 557. Reverse Words in a String III — "reverse" goal, solved word-by-word via slicing
```python
def reverseWords(s):
    return ' '.join(word[::-1] for word in s.split(' '))
```

---

## 2. Slow/fast write pointer (in-place compaction)

Base shape:
```python
def compact(nums):
    slow = 0
    for fast in range(len(nums)):
        if keep(nums[fast]):
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

### 26. Remove Duplicates from Sorted Array (base case)
```python
def removeDuplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

### 27. Remove Element (base case)
```python
def removeElement(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

### 283. Move Zeroes (base case)
```python
def moveZeroes(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```

### 922. Sort Array By Parity II — two write pointers, each stepping by 2
```python
def sortArrayByParityII(nums):
    n = len(nums)
    even, odd = 0, 1
    while even < n and odd < n:
        if nums[even] % 2 == 0:
            even += 2
        elif nums[odd] % 2 == 1:
            odd += 2
        else:
            nums[even], nums[odd] = nums[odd], nums[even]
    return nums
```

### 1089. Duplicate Zeros — reverse compaction, runs backward from the end
```python
def duplicateZeros(arr):
    n = len(arr)
    zeros = arr.count(0)
    i = n - 1
    j = n + zeros - 1
    while i >= 0:
        if j < n:
            arr[j] = arr[i]
        if arr[i] == 0:
            j -= 1
            if j < n:
                arr[j] = 0
        i -= 1; j -= 1
```

### 2460. Apply Operations to an Array — merge pass + standard compaction pass
```python
def applyOperations(nums):
    n = len(nums)
    for i in range(n - 1):
        if nums[i] == nums[i + 1]:
            nums[i] *= 2
            nums[i + 1] = 0
    slow = 0
    for fast in range(n):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
    return nums
```

### 3936. Minimum Swaps to Move Zeros to End — counts swaps, skips no-op self-swaps
```python
def minSwapsToMoveZeros(nums):
    slow = 0
    swaps = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            if fast != slow:
                swaps += 1
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
    return swaps
```

### 3940. Limit Occurrences in Sorted Array — lookback by `limit` instead of 1
```python
def limitOccurrences(nums, limit):
    slow = 0
    for fast in range(len(nums)):
        if slow < limit or nums[fast] != nums[slow - limit]:
            nums[slow] = nums[fast]
            slow += 1
    return nums[:slow]
```

---

## 3. Floyd's slow/fast (cycle detection)

No new easy-list problems this round — none of your failed problems were
linked-list/cycle-shaped. Base template stands as-is:
```python
def floyd(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

---

## 4. Merge two sorted sequences

Base shape:
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

### 88. Merge Sorted Array (base case — merges from the back, in place)
```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]; i -= 1
        else:
            nums1[k] = nums2[j]; j -= 1
        k -= 1
```

### 1768. Merge Strings Alternately — no comparison, unconditional alternation
```python
def mergeAlternately(word1, word2):
    i, j = 0, 0
    res = []
    while i < len(word1) or j < len(word2):
        if i < len(word1):
            res.append(word1[i]); i += 1
        if j < len(word2):
            res.append(word2[j]); j += 1
    return ''.join(res)
```

### 2570. Merge Two 2D Arrays by Summing Values — genuine third branch for equal keys
```python
def mergeArrays(nums1, nums2):
    i, j = 0, 0
    res = []
    while i < len(nums1) and j < len(nums2):
        if nums1[i][0] == nums2[j][0]:
            res.append([nums1[i][0], nums1[i][1] + nums2[j][1]])
            i += 1; j += 1
        elif nums1[i][0] < nums2[j][0]:
            res.append(nums1[i]); i += 1
        else:
            res.append(nums2[j]); j += 1
    res.extend(nums1[i:]); res.extend(nums2[j:])
    return res
```

### 1961. Check If String Is a Prefix of Array — one side grows a string, no output array
```python
def isPrefixString(s, words):
    prefix = ''
    for w in words:
        prefix += w
        if prefix == s:
            return True
        if len(prefix) > len(s):
            return False
    return False
```

---

## 5. Fixed pointer + converging inner pointers (k-Sum family)

Base shape:
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

**These are the same problem family (pair/triplet-sum) but solved with a
hash set instead of pointers — filed here so you remember both tools
exist, not because the code below is two-pointer.**

### 653. Two Sum IV — Input is a BST — hash set + DFS instead of pointers
```python
def findTarget(root, k):
    seen = set()
    def dfs(node):
        if not node:
            return False
        if k - node.val in seen:
            return True
        seen.add(node.val)
        return dfs(node.left) or dfs(node.right)
    return dfs(root)
```

### 1346. Check If N and Its Double Exist — single pass, hash set
```python
def checkIfExist(arr):
    seen = set()
    for x in arr:
        if 2 * x in seen or (x % 2 == 0 and x // 2 in seen):
            return True
        seen.add(x)
    return False
```

### 2367. Number of Arithmetic Triplets — sorted+distinct lets set-lookup replace the inner pointer scan
```python
def arithmeticTriplets(nums, diff):
    s = set(nums)
    count = 0
    for x in nums:
        if x + diff in s and x + 2 * diff in s:
            count += 1
    return count
```

### 2903. Find Indices With Index and Value Difference I — plain O(n²), no optimization applied
```python
def findIndices(nums, indexDifference, valueDifference):
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if abs(i - j) >= indexDifference and abs(nums[i] - nums[j]) >= valueDifference:
                return [i, j]
    return [-1, -1]
```

---

## 6. Two-pointer subsequence matching

Base shape:
```python
def is_subsequence(s, t):
    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)
```

### 392. Is Subsequence (base case)
```python
def isSubsequence(s, t):
    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)
```

### 925. Long Pressed Name — extra branch: `j` can advance without matching on a repeat
```python
def isLongPressedName(name, typed):
    i = j = 0
    while j < len(typed):
        if i < len(name) and name[i] == typed[j]:
            i += 1; j += 1
        elif j > 0 and typed[j] == typed[j - 1]:
            j += 1
        else:
            return False
    return i == len(name)
```

### 455. Assign Cookies — same shape, roles renamed (greedy matching)
```python
def findContentChildren(g, s):
    g.sort()
    s.sort()
    i = j = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            i += 1
        j += 1
    return i
```

**Not actually this template — filed here because it's the "find a
substring" cousin, but it's brute-force slicing, no pointer state:**

### 28. Find the Index of the First Occurrence in a String
```python
def strStr(haystack, needle):
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1
```

---

## 7. Expand-around-center (palindromes)

Base shape:
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

### 2108. First Palindromic String in Array — direct check, no expansion, same theme only
```python
def firstPalindrome(words):
    for w in words:
        if w == w[::-1]:
            return w
    return ""
```

---

## 8. Binary search the answer + two-pointer feasibility check

Base shape:
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

### 1385. Find the Distance Value Between Two Arrays — binary search per element (really a template-9 problem solved this way)
```python
import bisect

def findTheDistanceValue(arr1, arr2, d):
    arr2.sort()
    count = 0
    for x in arr1:
        lo = bisect.bisect_left(arr2, x - d)
        if lo == len(arr2) or arr2[lo] > x + d:
            count += 1
    return count
```

---

## 9. Two pointers across two independent collections

Base shape:
```python
def schedule(a, b):
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if compatible(a[i], b[j]):
            return a[i], b[j]
        if a[i] ends before b[j]:
            i += 1
        else:
            j += 1
```

### 3633. Earliest Finish Time (Land and Water Rides I) — brute-force O(n·m) double loop, not a pointer sweep
```python
def earliestFinishTime(landStartTime, landDuration, waterStartTime, waterDuration):
    res = float('inf')
    n, m = len(landStartTime), len(waterStartTime)
    for i in range(n):
        land_finish = landStartTime[i] + landDuration[i]
        for j in range(m):
            water_finish = waterStartTime[j] + waterDuration[j]
            start = max(land_finish, waterStartTime[j])
            res = min(res, start + waterDuration[j])
            start2 = max(water_finish, landStartTime[i])
            res = min(res, start2 + landDuration[i])
    return res
```

---

## Bucket 0 — Doesn't fit any template (the real decoys)

Filed here because these *feel* like two-pointer problems but the given
solution uses a completely different technique.

### 696. Count Binary Substrings — run-length group counting, not positions
```python
def countBinarySubstrings(s):
    prev, curr, res = 0, 1, 0
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            curr += 1
        else:
            res += min(prev, curr)
            prev, curr = curr, 1
    res += min(prev, curr)
    return res
```

### 821. Shortest Distance to a Character — two separate one-directional sweeps
```python
def shortestToChar(s, c):
    n = len(s)
    res = [0] * n
    prev = float('-inf')
    for i in range(n):
        if s[i] == c:
            prev = i
        res[i] = i - prev
    prev = float('inf')
    for i in range(n - 1, -1, -1):
        if s[i] == c:
            prev = i
        res[i] = min(res[i], prev - i)
    return res
```

### 844. Backspace String Compare — stack-building, not pointer-from-the-end
```python
def backspaceCompare(s, t):
    def build(string):
        result = []
        for ch in string:
            if ch != '#':
                result.append(ch)
            elif result:
                result.pop()
        return result
    return build(s) == build(t)
```
*(Note: the classic **optimal** solution for this one really is two
pointers walking both strings from the end, skipping backspaces on the
fly — worth redoing as a Template 1 exercise since this version uses
O(n) extra space unnecessarily.)*

### 1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence — plain iteration
```python
def isPrefixOfWord(sentence, searchWord):
    for i, w in enumerate(sentence.split(' '), 1):
        if w.startswith(searchWord):
            return i
    return -1
```

### 2200. Find All K-Distant Indices in an Array — brute-force nested range scan
```python
def findKDistantIndices(nums, key, k):
    n = len(nums)
    key_indices = [i for i, v in enumerate(nums) if v == key]
    res = set()
    for ki in key_indices:
        for i in range(max(0, ki - k), min(n, ki + k + 1)):
            res.add(i)
    return sorted(res)
```

### 2970. Count the Number of Incremovable Subarrays I — brute-force O(n²)
```python
def incremovableSubarrayCount(nums):
    n = len(nums)
    def is_increasing(arr):
        return all(arr[k] < arr[k + 1] for k in range(len(arr) - 1))
    count = 0
    for i in range(n):
        for j in range(i, n):
            remaining = nums[:i] + nums[j + 1:]
            if is_increasing(remaining):
                count += 1
    return count
```

---

## Updated "how to use this" checklist

# ==========================
# Python DSA Quick Tips
# ==========================

# 1. Intersection of two arrays (unique common elements)
list(set(nums1) & set(nums2))

# 2. Sort a list (ascending)
nums.sort()          # In-place
sorted_nums = sorted(nums)   # Returns a new sorted list

# 3. Create a set
seen = set()

# 4. Add an element to a set
seen.add(value)

# 5. Check if an element exists in a set
if value in seen:
    ...

# 6. Size of a set
len(seen)

# 7. Enumerate (index + value)
for i, num in enumerate(nums):
    print(i, num)

# 8. Combine two digits into one number
total += int(str(nums[l]) + str(nums[r]))

# Example:
nums = [7, 5, 2, 4]
# nums[0] and nums[3] -> "7" + "4" = "74"
# int("74") = 74

# 9. Convert number to string
str(num)

# 10. Convert string to integer
int(s)
