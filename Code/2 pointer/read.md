# LeetCode Two Pointers — Easy: Problems + Python Solutions





### 20. LC 541 — Reverse String II
Reverse the first k characters of every 2k-sized chunk.
```python
def reverseStr(s, k):
    chars = list(s)
    for start in range(0, len(chars), 2 * k):
        l, r = start, min(start + k, len(chars)) - 1
        while l < r:
            chars[l], chars[r] = chars[r], chars[l]
            l += 1
            r -= 1
    return ''.join(chars)
```

### 21. LC 557 — Reverse Words in a String III
Reverse each word in a sentence while keeping word order.
```python
def reverseWords(s):
    return ' '.join(word[::-1] for word in s.split(' '))
```

### 22. LC 653 — Two Sum IV - Input is a BST
Check if any two nodes in a BST sum to a target value.
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



### 24. LC 696 — Count Binary Substrings
Count substrings with equal numbers of consecutive 0s and 1s.
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

### 25. LC 821 — Shortest Distance to a Character
For each index, find the distance to the nearest occurrence of character `c`.
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

### 26. LC 832 — Flipping an Image
Flip each row horizontally, then invert all bits.
```python
def flipAndInvertImage(image):
    for row in image:
        l, r = 0, len(row) - 1
        while l <= r:
            row[l], row[r] = row[r] ^ 1, row[l] ^ 1
            l += 1
            r -= 1
    return image
```

### 27. LC 844 — Backspace String Compare
Check if two strings are equal after applying their backspace characters (`#`).
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


### 29. LC 905 — Sort Array By Parity
Rearrange so all even numbers come before all odd numbers.
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


```

### 31. LC 922 — Sort Array By Parity II
Rearrange so even-indexed positions hold even values and odd-indexed hold odd values.
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

### 32. LC 925 — Long Pressed Name
Check if `typed` could result from long-pressing keys while typing `name`.
```python
def isLongPressedName(name, typed):
    i = j = 0
    while j < len(typed):
        if i < len(name) and name[i] == typed[j]:
            i += 1
            j += 1
        elif j > 0 and typed[j] == typed[j - 1]:
            j += 1
        else:
            return False
    return i == len(name)
```

### 33. LC 942 — DI String Match
Build a permutation matching a pattern of 'I' (increase) and 'D' (decrease).
```python
def diStringMatch(s):
    lo, hi = 0, len(s)
    res = []
    for c in s:
        if c == 'I':
            res.append(lo)
            lo += 1
        else:
            res.append(hi)
            hi -= 1
    res.append(lo)
    return res
```

### 34. LC 977 — Squares of a Sorted Array
Return squares of array elements, sorted ascending.
```python
def sortedSquares(nums):
    n = len(nums)
    res = [0] * n
    l, r = 0, n - 1
    for i in range(n - 1, -1, -1):
        if abs(nums[l]) > abs(nums[r]):
            res[i] = nums[l] * nums[l]
            l += 1
        else:
            res[i] = nums[r] * nums[r]
            r -= 1
    return res
```

### 35. LC 1089 — Duplicate Zeros
Duplicate every zero in-place, shifting remaining elements right, keeping array length fixed.
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
        i -= 1
        j -= 1
```

### 37. LC 1332 — Remove Palindromic Subsequences
String has only 'a' and 'b' — return min removals to delete the whole string as palindromic subsequences.
```python
def removePalindromeSub(s):
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return 2
        l += 1
        r -= 1
    return 1
```

### 38. LC 1346 — Check If N and Its Double Exist
Check if there exist i, j such that `arr[i] == 2 * arr[j]`.
```python
def checkIfExist(arr):
    seen = set()
    for x in arr:
        if 2 * x in seen or (x % 2 == 0 and x // 2 in seen):
            return True
        seen.add(x)
    return False
```

### 39. LC 1385 — Find the Distance Value Between Two Arrays
Count elements in arr1 with no element in arr2 within distance d.
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

### 40. LC 1455 — Check If a Word Occurs As a Prefix of Any Word in a Sentence
Return the 1-indexed position of the first word that has `searchWord` as a prefix.
```python
def isPrefixOfWord(sentence, searchWord):
    for i, w in enumerate(sentence.split(' '), 1):
        if w.startswith(searchWord):
            return i
    return -1
```

### 41. LC 1768 — Merge Strings Alternately
Merge two strings by alternating characters.
```python
def mergeAlternately(word1, word2):
    i, j = 0, 0;    res = [];    while i < len(word1) or j < len(word2):
        if i < len(word1):            res.append(word1[i])            i += 1
        if j < len(word2):            res.append(word2[j])            j += 1
    return ''.join(res)
```



### 43. LC 1961 — Check If String Is a Prefix of Array
Check if `s` equals the concatenation of some prefix of the words array.
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

### 44. LC 2000 — Reverse Prefix of Word
Reverse the substring from index 0 to the first occurrence of `ch`.
```python
def reversePrefix(word, ch):
    if ch not in word:
        return word
    i = word.index(ch)
    return word[:i + 1][::-1] + word[i + 1:]
```

### 45. LC 2108 — Find First Palindromic String in the Array
Return the first word that is a palindrome, or "".
```python
def firstPalindrome(words):
    for w in words:
        if w == w[::-1]:
            return w
    return ""
```

### 46. LC 2200 — Find All K-Distant Indices in an Array
Find all indices within k of any index holding `key`.
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

### 47. LC 2367 — Number of Arithmetic Triplets
Count triplets (i, j, k) in a sorted distinct array forming an arithmetic sequence with common difference `diff`.
```python
def arithmeticTriplets(nums, diff):
    s = set(nums)
    count = 0
    for x in nums:
        if x + diff in s and x + 2 * diff in s:
            count += 1
    return count
```

### 49. LC 2460 — Apply Operations to an Array
Double adjacent equal pairs (zeroing the second), then push all zeros to the end.
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

### 50. LC 2465 — Number of Distinct Averages
Repeatedly remove the min and max, recording their average; count distinct averages.
```python
def distinctAverages(nums):
    nums.sort()
    l, r = 0, len(nums) - 1
    averages = set()
    while l < r:
        averages.add(nums[l] + nums[r])
        l += 1
        r -= 1
    return len(averages)
```

### 51. LC 2511 — Maximum Enemy Forts That Can Be Captured
Find the max number of empty positions capturable between an enemy fort (1) and your fort (-1).
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


### 54. LC 2570 — Merge Two 2D Arrays by Summing Values
Merge two sorted [id, value] arrays, summing values when ids match.
```python
def mergeArrays(nums1, nums2):
    i, j = 0, 0
    res = []
    while i < len(nums1) and j < len(nums2):
        if nums1[i][0] == nums2[j][0]:
            res.append([nums1[i][0], nums1[i][1] + nums2[j][1]])
            i += 1
            j += 1
        elif nums1[i][0] < nums2[j][0]:
            res.append(nums1[i])
            i += 1
        else:
            res.append(nums2[j])
            j += 1
    res.extend(nums1[i:])
    res.extend(nums2[j:])
    return res
```

### 55. LC 2697 — Lexicographically Smallest Palindrome
Make a string a palindrome with minimum changes, choosing the lexicographically smallest result.
```python
def makeSmallestPalindrome(s):
    chars = list(s)
    l, r = 0, len(chars) - 1
    while l < r:
        c = min(chars[l], chars[r])
        chars[l] = chars[r] = c
        l += 1
        r -= 1
    return ''.join(chars)
```


### 57. LC 2903 — Find Indices With Index and Value Difference I
Find indices i, j satisfying minimum index gap and value gap requirements.
```python
def findIndices(nums, indexDifference, valueDifference):
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if abs(i - j) >= indexDifference and abs(nums[i] - nums[j]) >= valueDifference:
                return [i, j]
    return [-1, -1]
```

### 58. LC 2970 — Count the Number of Incremovable Subarrays I
Count subarrays whose removal leaves the rest of the array strictly increasing.
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


### 60. LC 3633 ⚠️ — Earliest Finish Time for Land and Water Rides I
Find the earliest time you can finish one land ride followed by one water ride, or vice versa.
```python
def earliestFinishTime(landStartTime, landDuration, waterStartTime, waterDuration):
    res = float('inf')
    n, m = len(landStartTime), len(waterStartTime)
    for i in range(n):
        land_finish = landStartTime[i] + landDuration[i]
        for j in range(m):
            water_finish = waterStartTime[j] + waterDuration[j]
            # land then water
            start = max(land_finish, waterStartTime[j])
            res = min(res, start + waterDuration[j])
            # water then land
            start2 = max(water_finish, landStartTime[i])
            res = min(res, start2 + landDuration[i])
    return res
```

### 61. LC 3643 ⚠️ — Flip Square Submatrix Vertically
Flip a k×k submatrix of a grid vertically (top-to-bottom), in place.
```python
def reverseSubmatrix(grid, x, y, k):
    top, bottom = x, x + k - 1
    while top < bottom:
        for c in range(y, y + k):
            grid[top][c], grid[bottom][c] = grid[bottom][c], grid[top][c]
        top += 1
        bottom -= 1
    return grid
```


### 63. LC 3750 ⚠️ — Minimum Number of Flips to Reverse Binary String
Find the minimum number of bit flips needed to make a binary string equal to its own reverse.
```python
def minFlipsToPalindrome(s):
    l, r = 0, len(s) - 1
    flips = 0
    while l < r:
        if s[l] != s[r]:
            flips += 1
        l += 1
        r -= 1
    return flips
```

### 64. LC 3794 ⚠️ — Reverse String Prefix
Reverse the first k characters of a string.
```python
def reverseStringPrefix(word, k):
    return word[:k][::-1] + word[k:]
```

### 65. LC 3823 ⚠️ — Reverse Letters Then Special Characters in a String
Reverse only the letters in the string, ignoring non-letter positions.
```python
def reverseLettersOnly(s):
    chars = list(s)
    l, r = 0, len(chars) - 1
    while l < r:
        if not chars[l].isalpha():
            l += 1
        elif not chars[r].isalpha():
            r -= 1
        else:
            chars[l], chars[r] = chars[r], chars[l]
            l += 1
            r -= 1
    return ''.join(chars)
```


### 67. LC 3936 ⚠️ — Minimum Swaps to Move Zeros to End
Move all zeros to the end of the array, returning the minimum number of swaps used.
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

### 68. LC 3940 ⚠️ — Limit Occurrences in Sorted Array
Given a sorted array, keep at most `limit` occurrences of each value, in place.
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


list(set(nums1) & set(nums2))
sorting nums is  nums.sort().. nothting else
seen=set() add to set seen.add(data) -- len(seen)
for i, j in enumerate(nums): bracket

total += int(str(nums[l]) + str(nums[r])) #[7,52,2,4] =74


### 55. LC 2697 — Lexicographically Smallest Palindrome
Make a string a palindrome with minimum changes, choosing the lexicographically smallest result.
```python
def makeSmallestPalindrome(s):
    chars = list(s)
    l, r = 0, len(chars) - 1
    while l < r:
        c = min(chars[l], chars[r])
        chars[l] = chars[r] = c
        l += 1
        r -= 1
    return ''.join(chars)
```

Tempalte 2

### 1. LC 26 — Remove Duplicates from Sorted Array
Given a sorted array, remove duplicates in-place so each element appears once; return the new length.
```python
def removeDuplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):  # important 
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]  # should point to new slow position aka 1
    return slow + 1
```###  for removing a number    for fast in range(len(nums)):   if nums[fast] != val:
```
### 3. LC 28 — Find the Index of the First Occurrence in a String
    n, m = len(haystack), len(needle);    for i in range(n - m + 1):     if haystack[i:i + m] == needle:
```















Template 6 6. Two-pointer subsequence matching

### 17. LC 392 — (i.e., "ace" is a subsequence of "abcde" while "aec" is not).Check whether `s` is a subsequence of `t`.
```python
def isSubsequence(s, t):
    i = j = 0;
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)

s = "ace"
t = "abcde"

i=0,j=0: a=a ✓ → i=1,j=1
i=1,j=1: c≠b ✗ → j=2
i=1,j=2: c=c ✓ → i=2,j=3
i=2,j=3: e≠d ✗ → j=4
i=2,j=4: e=e ✓ → i=3,j=5

i == len(s) (3) → True ✅
```

19. LC 455 — Assign Cookies
Maximize the number of content children given greed factors and cookie sizes.

def findContentChildren(g, s):
    g.sort()
    s.sort()
    i = j = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            i += 1
        j += 1
    return i

Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.

 

Example 1:

Input: g = [1,2,3], s = [1,1]
Output: 1
Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3.



Check if a number looks the same when rotated 180 degrees.
```python
def isStrobogrammatic(num):
    mapping = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
    l, r = 0, len(num) - 1
    while l <= r:
        if num[l] not in mapping or mapping[num[l]] != num[r]:
            return False
        l += 1
        r -= 1
    return True



    
### 4. LC 88 — Merge Sorted Array
Merge `nums2` into `nums1` in-place; `nums1` has extra space at the end (size m+n).
```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
```

### 5. LC 125 — Valid Palindrome
Check if a string is a palindrome, ignoring non-alphanumeric chars and case.
```python
def isPalindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while l < r and not s[r].isalnum():
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1
        r -= 1
    return True
```




### 12. LC 283 — Move Zeroes
Move all zeros to the end of the array in-place, keeping relative order of non-zeros.
```python
def moveZeroes(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```
