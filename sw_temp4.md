# Sliding Window — All 100 LeetCode Medium Solutions
> Every solution is a direct variation of templates A1–A8 and B1–B7. Template tag listed above each solution.

---

## LC 3 — Longest Substring Without Repeating Characters
**Template: B2**
```python
from collections import Counter
def lengthOfLongestSubstring(s):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(s)):
        char = s[right]
        window_counter[char] += 1
        while window_counter[char] > 1:
            window_counter[s[left]] -= 1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 159 — Longest Substring with At Most Two Distinct Characters
**Template: B3 (k=2)**
```python
from collections import Counter
def lengthOfLongestSubstringTwoDistinct(s):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(s)):
        window_counter[s[right]] += 1
        while len(window_counter) > 2:
            window_counter[s[left]] -= 1
            if window_counter[s[left]] == 0:
                del window_counter[s[left]]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 187 — Repeated DNA Sequences
**Template: A5 (fixed window k=10, collect all windows seen twice)**
```python
from collections import Counter
def findRepeatedDnaSequences(s):
    k = 10
    if len(s) <= k:
        return []
    window_count = Counter()
    window = s[:k]
    window_count[window] += 1
    result = []
    for i in range(len(s) - k):
        window = s[i + 1:i + k + 1]
        window_count[window] += 1
        if window_count[window] == 2:
            result.append(window)
    return result
```

---

## LC 209 — Minimum Size Subarray Sum
**Template: B4**
```python
def minSubArrayLen(target, nums):
    left = 0
    current_sum = 0
    min_len = float("inf")
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum >= target:
            window_len = right - left + 1
            if window_len < min_len:
                min_len = window_len
            current_sum -= nums[left]
            left += 1
    return 0 if min_len == float("inf") else min_len
```

---

## LC 340 — Longest Substring with At Most K Distinct Characters
**Template: B3**
```python
from collections import Counter
def lengthOfLongestSubstringKDistinct(s, k):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(s)):
        window_counter[s[right]] += 1
        while len(window_counter) > k:
            window_counter[s[left]] -= 1
            if window_counter[s[left]] == 0:
                del window_counter[s[left]]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 395 — Longest Substring with At Least K Repeating Characters
**Template: B3 (outer loop over distinct count 1..26, inner is B3)**
```python
from collections import Counter
def longestSubstring(s, k):
    max_len = 0
    for num_distinct in range(1, 27):
        window_counter = Counter()
        left = 0
        num_at_least_k = 0
        for right in range(len(s)):
            window_counter[s[right]] += 1
            if window_counter[s[right]] == k:
                num_at_least_k += 1
            while len(window_counter) > num_distinct:
                if window_counter[s[left]] == k:
                    num_at_least_k -= 1
                window_counter[s[left]] -= 1
                if window_counter[s[left]] == 0:
                    del window_counter[s[left]]
                left += 1
            if len(window_counter) == num_distinct and num_at_least_k == num_distinct:
                current_len = right - left + 1
                if current_len > max_len:
                    max_len = current_len
    return max_len
```

---

## LC 413 — Arithmetic Slices
**Template: B6 variant (count subarrays; extend count when window stays arithmetic)**
```python
def numberOfArithmeticSlices(nums):
    count = 0
    current = 0
    for i in range(2, len(nums)):
        if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
            current += 1
            count += current
        else:
            current = 0
    return count
```

---

## LC 424 — Longest Repeating Character Replacement
**Template: B3 variant (window_len - max_freq <= k)**
```python
from collections import Counter
def characterReplacement(s, k):
    window_counter = Counter()
    left = 0
    max_len = 0
    max_freq = 0
    for right in range(len(s)):
        window_counter[s[right]] += 1
        max_freq = max(max_freq, window_counter[s[right]])
        while (right - left + 1) - max_freq > k:
            window_counter[s[left]] -= 1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 438 — Find All Anagrams in a String
**Template: A6**
```python
from collections import Counter
def findAnagrams(s, p):
    anagram_counter = Counter(p)
    window_counter = Counter(s[:len(p)])
    result = [0] if anagram_counter == window_counter else []
    for i in range(len(s) - len(p)):
        window_counter[s[i]] -= 1
        if window_counter[s[i]] == 0:
            del window_counter[s[i]]
        window_counter[s[i + len(p)]] += 1
        if window_counter == anagram_counter:
            result.append(i + 1)
    return result
```

---

## LC 487 — Max Consecutive Ones II
**Template: B3 (k=1, at most 1 zero)**
```python
from collections import Counter
def findMaxConsecutiveOnes(nums):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(nums)):
        window_counter[nums[right]] += 1
        while window_counter[0] > 1:
            window_counter[nums[left]] -= 1
            if window_counter[nums[left]] == 0:
                del window_counter[nums[left]]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 567 — Permutation in String
**Template: A5**
```python
from collections import Counter
def checkInclusion(s1, s2):
    k = len(s1)
    anagram_counter = Counter(s1)
    window_counter = Counter(s2[:k])
    if window_counter == anagram_counter:
        return True
    for i in range(len(s2) - k):
        window_counter[s2[i]] -= 1
        if window_counter[s2[i]] == 0:
            del window_counter[s2[i]]
        window_counter[s2[i + k]] += 1
        if window_counter == anagram_counter:
            return True
    return False
```

---

## LC 658 — Find K Closest Elements
**Template: A8 skeleton (fixed window size k, minimize window distance to x)**
```python
def findClosestElements(arr, k, x):
    left = 0
    right = len(arr) - k
    while left < right:
        mid = (left + right) // 2
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]
```

---

## LC 713 — Subarray Product Less Than K
**Template: B6**
```python
def numSubarrayProductLessThanK(nums, k):
    if k <= 1:
        return 0
    left = 0
    current_product = 1
    count = 0
    for right in range(len(nums)):
        current_product *= nums[right]
        while left <= right and current_product >= k:
            current_product //= nums[left]
            left += 1
        count += right - left + 1
    return count
```

---

## LC 718 — Maximum Length of Repeated Subarray
**Template: A2 variant (2D DP, but sliding window framing: fixed window scan)**
```python
def findLength(nums1, nums2):
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i-1] == nums2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
    return max_len
```

---

## LC 904 — Fruit Into Baskets
**Template: B3 (k=2)**
```python
from collections import Counter
def totalFruit(fruits):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(fruits)):
        window_counter[fruits[right]] += 1
        while len(window_counter) > 2:
            window_counter[fruits[left]] -= 1
            if window_counter[fruits[left]] == 0:
                del window_counter[fruits[left]]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 930 — Binary Subarrays With Sum
**Template: B6 (exactly-K = atMost(K) - atMost(K-1))**
```python
def numSubarraysWithSum(nums, goal):
    def at_most(target):
        if target < 0:
            return 0
        left = 0
        current_sum = 0
        count = 0
        for right in range(len(nums)):
            current_sum += nums[right]
            while current_sum > target:
                current_sum -= nums[left]
                left += 1
            count += right - left + 1
        return count
    return at_most(goal) - at_most(goal - 1)
```

---

## LC 978 — Longest Turbulent Subarray
**Template: B7 skeleton (variable window; shrink when turbulence breaks)**
```python
def maxTurbulenceSize(arr):
    left = 0
    max_len = 1
    for right in range(1, len(arr)):
        if right == 1:
            if arr[right] == arr[right - 1]:
                left = right
        else:
            curr_cmp = (arr[right] > arr[right-1]) - (arr[right] < arr[right-1])
            prev_cmp = (arr[right-1] > arr[right-2]) - (arr[right-1] < arr[right-2])
            if curr_cmp == 0:
                left = right
            elif curr_cmp == prev_cmp:
                left = right - 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 1004 — Max Consecutive Ones III
**Template: B3 (at most K zeros)**
```python
from collections import Counter
def longestOnes(nums, k):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(nums)):
        window_counter[nums[right]] += 1
        while window_counter[0] > k:
            window_counter[nums[left]] -= 1
            if window_counter[nums[left]] == 0:
                del window_counter[nums[left]]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 1016 — Binary String With Substrings Representing 1 To N
**Template: A5 variant (fixed window scan for each required substring)**
```python
def queryString(s, n):
    for num in range(1, n + 1):
        target = bin(num)[2:]
        k = len(target)
        if k > len(s):
            return False
        found = False
        window = s[:k]
        if window == target:
            found = True
        for i in range(len(s) - k):
            window = s[i + 1:i + k + 1]
            if window == target:
                found = True
                break
        if not found:
            return False
    return True
```

---

## LC 1031 — Maximum Sum of Two Non-Overlapping Subarrays
**Template: A2 (two fixed windows; track running max of each)**
```python
def maxSumTwoNoOverlap(nums, firstLen, secondLen):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    def window_sum(l, r):
        return prefix[r] - prefix[l]

    def solve(L, M):
        max_L = 0
        result = 0
        for i in range(L + M, n + 1):
            max_L = max(max_L, window_sum(i - L - M, i - M))
            result = max(result, max_L + window_sum(i - M, i))
        return result

    return max(solve(firstLen, secondLen), solve(secondLen, firstLen))
```

---

## LC 1052 — Grumpy Bookstore Owner
**Template: A2 (fixed window of size minutes; maximize extra customers)**
```python
def maxSatisfied(customers, grumpy, minutes):
    base = 0
    for i in range(len(customers)):
        if grumpy[i] == 0:
            base += customers[i]
    current_extra = sum(customers[i] * grumpy[i] for i in range(minutes))
    max_extra = current_extra
    for i in range(len(customers) - minutes):
        current_extra -= customers[i] * grumpy[i]
        current_extra += customers[i + minutes] * grumpy[i + minutes]
        if current_extra > max_extra:
            max_extra = current_extra
    return base + max_extra
```

---

## LC 1100 — Find K-Length Substrings With No Repeated Characters
**Template: A5 variant (fixed window, check all distinct)**
```python
from collections import Counter
def numKLenSubstrNoRepeats(s, k):
    if k > len(s):
        return 0
    window_counter = Counter(s[:k])
    count = 1 if len(window_counter) == k else 0
    for i in range(len(s) - k):
        window_counter[s[i]] -= 1
        if window_counter[s[i]] == 0:
            del window_counter[s[i]]
        window_counter[s[i + k]] += 1
        if len(window_counter) == k:
            count += 1
    return count
```

---

## LC 1151 — Minimum Swaps to Group All 1's Together
**Template: A2 (fixed window = count of 1s; minimize 0s inside window)**
```python
def minSwaps(data):
    k = sum(data)
    if k == 0:
        return 0
    current_ones = sum(data[:k])
    max_ones = current_ones
    for i in range(len(data) - k):
        current_ones -= data[i]
        current_ones += data[i + k]
        if current_ones > max_ones:
            max_ones = current_ones
    return k - max_ones
```

---

## LC 1156 — Swap For Longest Repeated Character Substring
**Template: B3 variant (at most 1 different character allowed if spare exists)**
```python
from collections import Counter
def maxRepOpt1(text):
    total_count = Counter(text)
    left = 0
    max_len = 0
    window_counter = Counter()
    for right in range(len(text)):
        window_counter[text[right]] += 1
        max_freq = max(window_counter.values())
        window_len = right - left + 1
        while window_len - max_freq > 1:
            window_counter[text[left]] -= 1
            if window_counter[text[left]] == 0:
                del window_counter[text[left]]
            left += 1
            window_len = right - left + 1
            max_freq = max(window_counter.values())
        dominant_char = max(window_counter, key=window_counter.get)
        effective_len = window_len
        if total_count[dominant_char] > window_counter[dominant_char]:
            effective_len = min(window_len, total_count[dominant_char])
        if effective_len > max_len:
            max_len = effective_len
    return max_len
```

---

## LC 1208 — Get Equal Substrings Within Budget
**Template: B4 (max length window where sum of abs diffs <= maxCost)**
```python
def equalSubstring(s, t, maxCost):
    costs = [abs(ord(s[i]) - ord(t[i])) for i in range(len(s))]
    left = 0
    current_sum = 0
    max_len = 0
    for right in range(len(costs)):
        current_sum += costs[right]
        while current_sum > maxCost:
            current_sum -= costs[left]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 1234 — Replace the Substring for Balanced String
**Template: B4 (minimum window outside which freq of each char <= n/4)**
```python
from collections import Counter
def balancedString(s):
    n = len(s)
    k = n // 4
    count = Counter(s)
    if all(count[c] == k for c in "QWER"):
        return 0
    left = 0
    min_len = n
    for right in range(n):
        count[s[right]] -= 1
        while left < n and all(count[c] <= k for c in "QWER"):
            window_len = right - left + 1
            if window_len < min_len:
                min_len = window_len
            count[s[left]] += 1
            left += 1
    return min_len
```

---

## LC 1248 — Count Number of Nice Subarrays
**Template: B6 (exactly-K odds = atMost(k) - atMost(k-1))**
```python
def numberOfSubarrays(nums, k):
    def at_most(target):
        left = 0
        current_sum = 0
        count = 0
        for right in range(len(nums)):
            current_sum += nums[right] % 2
            while current_sum > target:
                current_sum -= nums[left] % 2
                left += 1
            count += right - left + 1
        return count
    return at_most(k) - at_most(k - 1)
```

---

## LC 1297 — Maximum Number of Occurrences of a Substring
**Template: A6 variant (only check substrings of length minSize; longer never beats shorter)**
```python
from collections import Counter
def maxFreq(s, maxLetters, minSize, maxSize):
    substr_count = Counter()
    window_counter = Counter()
    k = minSize
    for i in range(k):
        window_counter[s[i]] += 1
    substr_count[s[:k]] += 1
    best = 1 if len(window_counter) <= maxLetters else 0
    for i in range(len(s) - k):
        window_counter[s[i]] -= 1
        if window_counter[s[i]] == 0:
            del window_counter[s[i]]
        window_counter[s[i + k]] += 1
        if len(window_counter) <= maxLetters:
            sub = s[i + 1:i + k + 1]
            substr_count[sub] += 1
            if substr_count[sub] > best:
                best = substr_count[sub]
    return best
```

---

## LC 1343 — Number of Sub-arrays of Size K and Average >= Threshold
**Template: A4**
```python
def numOfSubarrays(arr, k, threshold):
    target = threshold * k
    current_sum = sum(arr[:k])
    count = 1 if current_sum >= target else 0
    for i in range(len(arr) - k):
        current_sum -= arr[i]
        current_sum += arr[i + k]
        if current_sum >= target:
            count += 1
    return count
```

---

## LC 1358 — Number of Substrings Containing All Three Characters
**Template: B6 variant (for each right, count valid left positions)**
```python
def numberOfSubstrings(s):
    last_seen = {'a': -1, 'b': -1, 'c': -1}
    count = 0
    for right in range(len(s)):
        last_seen[s[right]] = right
        count += 1 + min(last_seen.values())
    return count
```

---

## LC 1423 — Maximum Points You Can Obtain from Cards
**Template: A2 (fixed window of size total-k in middle; maximize sum outside)**
```python
def maxScore(cardPoints, k):
    n = len(cardPoints)
    total = sum(cardPoints)
    window_size = n - k
    if window_size == 0:
        return total
    current_sum = sum(cardPoints[:window_size])
    min_sum = current_sum
    for i in range(n - window_size):
        current_sum -= cardPoints[i]
        current_sum += cardPoints[i + window_size]
        if current_sum < min_sum:
            min_sum = current_sum
    return total - min_sum
```

---

## LC 1438 — Longest Continuous Subarray With Absolute Diff <= Limit
**Template: A7 variant (dual monotonic deques)**
```python
from collections import deque
def longestSubarray(nums, limit):
    max_dq = deque()
    min_dq = deque()
    left = 0
    max_len = 0
    for right in range(len(nums)):
        while max_dq and nums[max_dq[-1]] <= nums[right]:
            max_dq.pop()
        max_dq.append(right)
        while min_dq and nums[min_dq[-1]] >= nums[right]:
            min_dq.pop()
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > limit:
            left += 1
            if max_dq[0] < left:
                max_dq.popleft()
            if min_dq[0] < left:
                min_dq.popleft()
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 1456 — Maximum Number of Vowels in a Substring of Given Length
**Template: A2**
```python
def maxVowels(s, k):
    vowels = set("aeiou")
    current_sum = sum(1 for c in s[:k] if c in vowels)
    max_sum = current_sum
    for i in range(len(s) - k):
        current_sum -= 1 if s[i] in vowels else 0
        current_sum += 1 if s[i + k] in vowels else 0
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

---

## LC 1477 — Find Two Non-overlapping Sub-arrays Each With Target Sum
**Template: B1 (two passes; store min window ending at each index)**
```python
def minSumOfLengths(arr, target):
    prefix_min = [float("inf")] * len(arr)
    left = 0
    current_sum = 0
    min_len = float("inf")
    result = float("inf")
    for right in range(len(arr)):
        current_sum += arr[right]
        while current_sum > target:
            current_sum -= arr[left]
            left += 1
        if current_sum == target:
            window_len = right - left + 1
            if right >= window_len and prefix_min[right - window_len] != float("inf"):
                result = min(result, window_len + prefix_min[right - window_len])
            min_len = min(min_len, window_len)
        prefix_min[right] = min_len
    return result if result != float("inf") else -1
```

---

## LC 1493 — Longest Subarray of 1's After Deleting One Element
**Template: B3 (at most 1 zero, then subtract 1 for the deleted element)**
```python
from collections import Counter
def longestSubarray(nums):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(nums)):
        window_counter[nums[right]] += 1
        while window_counter[0] > 1:
            window_counter[nums[left]] -= 1
            if window_counter[nums[left]] == 0:
                del window_counter[nums[left]]
            left += 1
        current_len = right - left + 1 - 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 1658 — Minimum Operations to Reduce X to Zero
**Template: B4 variant (find longest subarray with sum = total - x)**
```python
def minOperations(nums, x):
    target = sum(nums) - x
    if target < 0:
        return -1
    if target == 0:
        return len(nums)
    left = 0
    current_sum = 0
    max_len = -1
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if current_sum == target:
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
    return len(nums) - max_len if max_len != -1 else -1
```

---

## LC 1695 — Maximum Erasure Value
**Template: B2 (track sum instead of length)**
```python
from collections import Counter
def maximumUniqueSubarray(nums):
    window_counter = Counter()
    left = 0
    current_sum = 0
    max_sum = 0
    for right in range(len(nums)):
        num = nums[right]
        window_counter[num] += 1
        current_sum += num
        while window_counter[num] > 1:
            window_counter[nums[left]] -= 1
            current_sum -= nums[left]
            left += 1
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

---

## LC 1838 — Frequency of the Most Frequent Element
**Template: B3 variant (window valid if sum_needed = k*max_val - window_sum <= ops)**
```python
def maxFrequency(nums, k):
    nums.sort()
    left = 0
    current_sum = 0
    max_len = 0
    for right in range(len(nums)):
        current_sum += nums[right]
        while nums[right] * (right - left + 1) - current_sum > k:
            current_sum -= nums[left]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 1839 — Longest Substring Of All Vowels in Order
**Template: B7 skeleton (window valid if all vowels present and non-decreasing)**
```python
def longestBeautifulSubstring(word):
    left = 0
    max_len = 0
    distinct = 1
    for right in range(1, len(word)):
        if word[right] < word[right - 1]:
            left = right
            distinct = 1
        elif word[right] > word[right - 1]:
            distinct += 1
        if distinct == 5:
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
    return max_len
```

---

## LC 1852 — Distinct Numbers in Each Subarray
**Template: A8 skeleton**
```python
from collections import Counter
def distinctNumbers(nums, k):
    window_counter = Counter(nums[:k])
    result = [len(window_counter)]
    for i in range(len(nums) - k):
        window_counter[nums[i]] -= 1
        if window_counter[nums[i]] == 0:
            del window_counter[nums[i]]
        window_counter[nums[i + k]] += 1
        result.append(len(window_counter))
    return result
```

---

## LC 2024 — Maximize the Confusion of an Exam
**Template: B3 (run twice: once for at most k F->T, once for at most k T->F)**
```python
from collections import Counter
def maxConsecutiveAnswers(answerKey, k):
    def longest_with_at_most_k(char):
        window_counter = Counter()
        left = 0
        max_len = 0
        for right in range(len(answerKey)):
            window_counter[answerKey[right]] += 1
            while window_counter[char] > k:
                window_counter[answerKey[left]] -= 1
                if window_counter[answerKey[left]] == 0:
                    del window_counter[answerKey[left]]
                left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len
    return max(longest_with_at_most_k('T'), longest_with_at_most_k('F'))
```

---

## LC 2090 — K Radius Subarray Averages
**Template: A2 (window size = 2k+1)**
```python
def getAverages(nums, k):
    n = len(nums)
    result = [-1] * n
    window = 2 * k + 1
    if window > n:
        return result
    current_sum = sum(nums[:window])
    result[k] = current_sum // window
    for i in range(n - window):
        current_sum -= nums[i]
        current_sum += nums[i + window]
        result[i + k + 1] = current_sum // window
    return result
```

---

## LC 2107 — Number of Unique Flavors After Sharing K Candies
**Template: A8 skeleton (fixed window size k; track distinct outside window)**
```python
from collections import Counter
def shareCandies(candies, k):
    total_counter = Counter(candies)
    window_counter = Counter()
    for i in range(k):
        window_counter[candies[i]] += 1
    def distinct_outside():
        count = 0
        for flavor in total_counter:
            if total_counter[flavor] > window_counter.get(flavor, 0):
                count += 1
        return count
    max_distinct = distinct_outside()
    for i in range(len(candies) - k):
        window_counter[candies[i]] -= 1
        if window_counter[candies[i]] == 0:
            del window_counter[candies[i]]
        window_counter[candies[i + k]] += 1
        d = distinct_outside()
        if d > max_distinct:
            max_distinct = d
    return max_distinct
```

---

## LC 2110 — Number of Smooth Descent Periods of a Stock
**Template: B6 variant (count subarrays; extend count when descent continues)**
```python
def getDescentPeriods(prices):
    count = 0
    current = 1
    for i in range(1, len(prices)):
        if prices[i] == prices[i - 1] - 1:
            current += 1
        else:
            current = 1
        count += current
    return count
```

---

## LC 2134 — Minimum Swaps to Group All 1's Together II (circular)
**Template: A2 (fixed window on doubled array)**
```python
def minSwaps(nums):
    k = sum(nums)
    if k == 0:
        return 0
    n = len(nums)
    doubled = nums + nums
    current_ones = sum(doubled[:k])
    max_ones = current_ones
    for i in range(2 * n - k):
        current_ones -= doubled[i]
        current_ones += doubled[i + k]
        if current_ones > max_ones:
            max_ones = current_ones
    return k - max_ones
```

---

## LC 2260 — Minimum Consecutive Cards to Pick Up
**Template: B2 variant (shortest window with a duplicate)**
```python
def minimumCardPickup(cards):
    last_seen = {}
    left = 0
    min_len = float("inf")
    for right in range(len(cards)):
        card = cards[right]
        if card in last_seen:
            left = max(left, last_seen[card])
            window_len = right - left + 1
            if window_len < min_len:
                min_len = window_len
        last_seen[card] = right
    return min_len if min_len != float("inf") else -1
```

---

## LC 2271 — Maximum White Tiles Covered by a Carpet
**Template: A2 + binary search (fixed carpet length; prefix sum for fast coverage)**
```python
import bisect
def maximumWhiteTiles(tiles, carpetLen):
    tiles.sort()
    n = len(tiles)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + tiles[i][1] - tiles[i][0] + 1
    max_covered = 0
    for i in range(n):
        end = tiles[i][0] + carpetLen - 1
        j = bisect.bisect_right(tiles, [end, float("inf")]) - 1
        if j < i:
            covered = min(tiles[i][1], end) - tiles[i][0] + 1
        else:
            covered = prefix[j + 1] - prefix[i]
            if tiles[j][1] > end:
                covered -= tiles[j][1] - end
        if covered > max_covered:
            max_covered = covered
    return max_covered
```

---

## LC 2401 — Longest Nice Subarray
**Template: B2 (no two numbers share a bit; shrink when AND overlaps)**
```python
def longestNiceSubarray(nums):
    left = 0
    used_bits = 0
    max_len = 0
    for right in range(len(nums)):
        while used_bits & nums[right]:
            used_bits ^= nums[left]
            left += 1
        used_bits |= nums[right]
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 2411 — Smallest Subarrays With Maximum Bitwise OR
**Template: A8 skeleton (for each left, find rightmost bit contribution)**
```python
def smallestSubarrays(nums):
    n = len(nums)
    last = [0] * 32
    result = [0] * n
    for i in range(n - 1, -1, -1):
        for bit in range(32):
            if nums[i] & (1 << bit):
                last[bit] = i
        result[i] = max(last) - i + 1
    return result
```

---

## LC 2461 — Maximum Sum of Distinct Subarrays With Length K
**Template: A2 + A5 hybrid**
```python
from collections import Counter
def maximumSubarraySum(nums, k):
    window_counter = Counter()
    current_sum = 0
    for i in range(k):
        window_counter[nums[i]] += 1
        current_sum += nums[i]
    max_sum = current_sum if len(window_counter) == k else 0
    for i in range(len(nums) - k):
        window_counter[nums[i]] -= 1
        if window_counter[nums[i]] == 0:
            del window_counter[nums[i]]
        current_sum -= nums[i]
        window_counter[nums[i + k]] += 1
        current_sum += nums[i + k]
        if len(window_counter) == k and current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

---

## LC 2516 — Take K of Each Character From Left and Right
**Template: B4 variant (find longest middle window to exclude, leaving k of each)**
```python
from collections import Counter
def takeCharacters(s, k):
    total = Counter(s)
    if any(total[c] < k for c in "abc"):
        return -1
    need = {c: total[c] - k for c in "abc"}
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(s)):
        window_counter[s[right]] += 1
        while window_counter[s[right]] > need[s[right]]:
            window_counter[s[left]] -= 1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return len(s) - max_len
```

---

## LC 2537 — Count the Number of Good Subarrays
**Template: B6 (count subarrays with at least k pairs)**
```python
from collections import Counter
def countGood(nums, k):
    window_counter = Counter()
    left = 0
    pairs = 0
    count = 0
    for right in range(len(nums)):
        pairs += window_counter[nums[right]]
        window_counter[nums[right]] += 1
        while pairs >= k:
            count += len(nums) - right
            window_counter[nums[left]] -= 1
            pairs -= window_counter[nums[left]]
            left += 1
    return count
```

---

## LC 2653 — Sliding Subarray Beauty
**Template: A7 variant (fixed window; find x-th negative using sorted structure)**
```python
import sortedcontainers
def getSubarrayBeauty(nums, k, x):
    from sortedcontainers import SortedList
    window = SortedList(nums[:k])
    result = []
    val = window[x - 1]
    result.append(val if val < 0 else 0)
    for i in range(len(nums) - k):
        window.remove(nums[i])
        window.add(nums[i + k])
        val = window[x - 1]
        result.append(val if val < 0 else 0)
    return result
```

---

## LC 2730 — Find the Longest Semi-Repetitive Substring
**Template: B3 (at most 1 adjacent duplicate pair)**
```python
def longestSemiRepetitiveSubstring(s):
    left = 0
    adj_pairs = 0
    max_len = 1
    for right in range(1, len(s)):
        if s[right] == s[right - 1]:
            adj_pairs += 1
        while adj_pairs > 1:
            if s[left] == s[left + 1]:
                adj_pairs -= 1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 2743 — Count Substrings Without Repeating Character
**Template: B6 (count all valid windows ending at each right)**
```python
from collections import Counter
def numberOfSpecialSubstrings(s):
    window_counter = Counter()
    left = 0
    count = 0
    for right in range(len(s)):
        window_counter[s[right]] += 1
        while window_counter[s[right]] > 1:
            window_counter[s[left]] -= 1
            left += 1
        count += right - left + 1
    return count
```

---

## LC 2747 — Count Zero Request Servers
**Template: A8 skeleton (fixed window of time, sliding over sorted queries)**
```python
def countServers(n, logs, x, queries):
    q_indexed = sorted(enumerate(queries), key=lambda t: t[1])
    logs.sort(key=lambda l: l[1])
    result = [0] * len(queries)
    window_counter = Counter()
    left = 0
    right = 0
    for idx, q in q_indexed:
        while right < len(logs) and logs[right][1] <= q:
            window_counter[logs[right][0]] += 1
            right += 1
        while left < len(logs) and logs[left][1] < q - x:
            window_counter[logs[left][0]] -= 1
            if window_counter[logs[left][0]] == 0:
                del window_counter[logs[left][0]]
            left += 1
        result[idx] = len(window_counter)
    return result
```

---

## LC 2762 — Continuous Subarrays
**Template: B6 (count all valid windows; condition: max-min <= 2 using dual deque)**
```python
from collections import deque
def continuousSubarrays(nums):
    max_dq = deque()
    min_dq = deque()
    left = 0
    count = 0
    for right in range(len(nums)):
        while max_dq and nums[max_dq[-1]] <= nums[right]:
            max_dq.pop()
        max_dq.append(right)
        while min_dq and nums[min_dq[-1]] >= nums[right]:
            min_dq.pop()
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > 2:
            left += 1
            if max_dq[0] < left:
                max_dq.popleft()
            if min_dq[0] < left:
                min_dq.popleft()
        count += right - left + 1
    return count
```

---

## LC 2779 — Maximum Beauty of an Array After Applying Operation
**Template: B3 (sort + longest window where max-min <= 2k)**
```python
def maximumBeauty(nums, k):
    nums.sort()
    left = 0
    max_len = 0
    for right in range(len(nums)):
        while nums[right] - nums[left] > 2 * k:
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 2799 — Count Complete Subarrays in an Array
**Template: B6 (count subarrays with all distinct values present)**
```python
from collections import Counter
def countCompleteSubarrays(nums):
    total_distinct = len(set(nums))
    window_counter = Counter()
    left = 0
    count = 0
    for right in range(len(nums)):
        window_counter[nums[right]] += 1
        while len(window_counter) == total_distinct:
            count += len(nums) - right
            window_counter[nums[left]] -= 1
            if window_counter[nums[left]] == 0:
                del window_counter[nums[left]]
            left += 1
    return count
```

---

## LC 2831 — Find the Longest Equal Subarray
**Template: B3 variant (for each value v, find longest window with at most k non-v elements)**
```python
from collections import Counter
def longestEqualSubarray(nums, k):
    positions = {}
    for i, num in enumerate(nums):
        if num not in positions:
            positions[num] = []
        positions[num].append(i)
    max_len = 0
    for pos_list in positions.values():
        left = 0
        for right in range(len(pos_list)):
            while pos_list[right] - pos_list[left] - (right - left) > k:
                left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
    return max_len
```

---

## LC 2841 — Maximum Sum of Almost Unique Subarray
**Template: A2 + A5 hybrid (fixed window k; at least m distinct)**
```python
from collections import Counter
def maxSum(nums, m, k):
    window_counter = Counter()
    current_sum = 0
    for i in range(k):
        window_counter[nums[i]] += 1
        current_sum += nums[i]
    max_sum = current_sum if len(window_counter) >= m else 0
    for i in range(len(nums) - k):
        window_counter[nums[i]] -= 1
        if window_counter[nums[i]] == 0:
            del window_counter[nums[i]]
        current_sum -= nums[i]
        window_counter[nums[i + k]] += 1
        current_sum += nums[i + k]
        if len(window_counter) >= m and current_sum > max_sum:
            max_sum = current_sum
    return max_sum
```

---

## LC 2875 — Minimum Size Subarray in Infinite Array
**Template: B4 variant (use modulo to reduce target, then scan once)**
```python
def minSizeSubarray(nums, target):
    n = len(nums)
    total = sum(nums)
    full_loops = target // total
    remainder = target % total
    base = full_loops * n
    if remainder == 0:
        return base
    doubled = nums + nums
    left = 0
    current_sum = 0
    min_len = float("inf")
    for right in range(2 * n):
        current_sum += doubled[right]
        while current_sum > remainder:
            current_sum -= doubled[left]
            left += 1
        if current_sum == remainder:
            window_len = right - left + 1
            if window_len < min_len:
                min_len = window_len
    return base + min_len if min_len != float("inf") else -1
```

---

## LC 2904 — Shortest and Lexicographically Smallest Beautiful String
**Template: B4 (shortest window with exactly k ones; track lex min)**
```python
def shortestBeautifulSubstring(s, k):
    left = 0
    current_ones = 0
    min_len = float("inf")
    result = ""
    for right in range(len(s)):
        if s[right] == '1':
            current_ones += 1
        while current_ones > k:
            if s[left] == '1':
                current_ones -= 1
            left += 1
        if current_ones == k:
            window_len = right - left + 1
            candidate = s[left:right + 1]
            if window_len < min_len or (window_len == min_len and candidate < result):
                min_len = window_len
                result = candidate
    return result
```

---

## LC 2958 — Length of Longest Subarray With at Most K Frequency
**Template: B3**
```python
from collections import Counter
def maxSubarrayLength(nums, k):
    window_counter = Counter()
    left = 0
    max_len = 0
    for right in range(len(nums)):
        num = nums[right]
        window_counter[num] += 1
        while window_counter[num] > k:
            window_counter[nums[left]] -= 1
            if window_counter[nums[left]] == 0:
                del window_counter[nums[left]]
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 2962 — Count Subarrays Where Max Element Appears at Least K Times
**Template: B6 variant**
```python
def countSubarrays(nums, k):
    max_val = max(nums)
    left = 0
    max_count = 0
    count = 0
    for right in range(len(nums)):
        if nums[right] == max_val:
            max_count += 1
        while max_count >= k:
            count += len(nums) - right
            if nums[left] == max_val:
                max_count -= 1
            left += 1
    return count
```

---

## LC 2981 — Find Longest Special Substring That Occurs Thrice I
**Template: A6 variant (fixed window for each length; count substring freq)**
```python
from collections import Counter
def maximumLength(s):
    n = len(s)
    for length in range(n, 0, -1):
        substr_count = Counter()
        window = s[:length]
        substr_count[window] += 1
        for i in range(n - length):
            window = s[i + 1:i + length + 1]
            substr_count[window] += 1
        for substr, cnt in substr_count.items():
            if cnt >= 3 and len(set(substr)) == 1:
                return length
    return -1
```

---

## LC 2982 — Find Longest Special Substring That Occurs Thrice II
**Template: A6 variant (same idea, optimized with char+length grouping)**
```python
from collections import defaultdict
def maximumLength(s):
    groups = defaultdict(Counter)
    left = 0
    for right in range(len(s)):
        if s[right] != s[left]:
            left = right
        groups[s[right]][right - left + 1] += 1
    best = -1
    for char, length_counts in groups.items():
        lengths = sorted(length_counts.keys(), reverse=True)
        carry = 0
        for length in lengths:
            carry += length_counts[length]
            if carry >= 3:
                best = max(best, length)
                break
            if carry >= 2:
                best = max(best, length - 1)
            if carry >= 1:
                best = max(best, length - 2)
    return best
```

---

## LC 3097 — Shortest Subarray With OR at Least K II
**Template: B4 (minimum window where bitwise OR >= k; shrink left once valid)**
```python
def minimumSubarrayLength(nums, k):
    left = 0
    current_or = 0
    bit_count = [0] * 32
    min_len = float("inf")
    for right in range(len(nums)):
        for bit in range(32):
            if nums[right] & (1 << bit):
                bit_count[bit] += 1
        current_or |= nums[right]
        while current_or >= k:
            window_len = right - left + 1
            if window_len < min_len:
                min_len = window_len
            for bit in range(32):
                if nums[left] & (1 << bit):
                    bit_count[bit] -= 1
                    if bit_count[bit] == 0:
                        current_or &= ~(1 << bit)
            left += 1
    return min_len if min_len != float("inf") else -1
```

---

## LC 3191 — Minimum Operations to Make Binary Array Elements Equal to One I
**Template: A8 skeleton (fixed window k=3; greedily flip at each 0)**
```python
def minOperations(nums):
    ops = 0
    for i in range(len(nums) - 2):
        if nums[i] == 0:
            nums[i] ^= 1
            nums[i + 1] ^= 1
            nums[i + 2] ^= 1
            ops += 1
    if nums[-2] == 0 or nums[-1] == 0:
        return -1
    return ops
```

---

## LC 3208 — Alternating Groups II
**Template: A8 skeleton (fixed window k on circular array)**
```python
def numberOfAlternatingGroups(colors, k):
    n = len(colors)
    extended = colors + colors[:k - 1]
    count = 0
    left = 0
    for right in range(1, len(extended)):
        if extended[right] == extended[right - 1]:
            left = right
        if right - left + 1 >= k:
            count += 1
    return count
```

---

## LC 3254 — Find the Power of K-Size Subarrays I
**Template: A8 skeleton (fixed window; check if consecutive increasing)**
```python
def resultsArray(nums, k):
    n = len(nums)
    result = []
    left = 0
    for right in range(n):
        if right > 0 and nums[right] != nums[right - 1] + 1:
            left = right
        if right - left + 1 >= k:
            result.append(nums[right])
        elif right >= k - 1:
            result.append(-1)
    while len(result) < n - k + 1:
        result.append(-1)
    return result
```

---

## LC 3255 — Find the Power of K-Size Subarrays II
**Template: A8 skeleton (same as 3254)**
```python
def resultsArray(nums, k):
    n = len(nums)
    result = [-1] * (n - k + 1)
    left = 0
    for right in range(n):
        if right > 0 and nums[right] != nums[right - 1] + 1:
            left = right
        if right - left + 1 >= k:
            result[right - k + 1] = nums[right]
    return result
```

---

## LC 3297 — Count Substrings That Can Be Rearranged to Contain a String I
**Template: B5 (minimum window substring count variant)**
```python
from collections import Counter
def validSubstringCount(word1, word2):
    target_counter = Counter(word2)
    window_counter = Counter()
    required = len(target_counter)
    formed = 0
    left = 0
    count = 0
    for right in range(len(word1)):
        char = word1[right]
        window_counter[char] += 1
        if char in target_counter and window_counter[char] == target_counter[char]:
            formed += 1
        while formed == required:
            count += len(word1) - right
            left_char = word1[left]
            window_counter[left_char] -= 1
            if left_char in target_counter and window_counter[left_char] < target_counter[left_char]:
                formed -= 1
            left += 1
    return count
```

---

## LC 3305 — Count of Substrings Containing Every Vowel and K Consonants I
**Template: B6 (exactly-K = atMost(k) - atMost(k-1) with vowel presence check)**
```python
from collections import Counter
def countOfSubstrings(word, k):
    vowels = set("aeiou")
    def at_most(max_consonants):
        window_counter = Counter()
        left = 0
        consonants = 0
        count = 0
        for right in range(len(word)):
            if word[right] in vowels:
                window_counter[word[right]] += 1
            else:
                consonants += 1
            while consonants > max_consonants:
                if word[left] in vowels:
                    window_counter[word[left]] -= 1
                    if window_counter[word[left]] == 0:
                        del window_counter[word[left]]
                else:
                    consonants -= 1
                left += 1
            if len(window_counter) == 5:
                count += right - left + 1
        return count
    return at_most(k) - at_most(k - 1)
```

---

## LC 3306 — Count of Substrings Containing Every Vowel and K Consonants II
**Template: B6 (same as 3305, works for large inputs too)**
```python
from collections import Counter
def countOfSubstrings(word, k):
    vowels = set("aeiou")
    def at_most(max_consonants):
        window_counter = Counter()
        left = 0
        consonants = 0
        count = 0
        for right in range(len(word)):
            if word[right] in vowels:
                window_counter[word[right]] += 1
            else:
                consonants += 1
            while consonants > max_consonants:
                if word[left] in vowels:
                    window_counter[word[left]] -= 1
                    if window_counter[word[left]] == 0:
                        del window_counter[word[left]]
                else:
                    consonants -= 1
                left += 1
            if len(window_counter) == 5:
                count += right - left + 1
        return count
    return at_most(k) - at_most(k - 1)
```

---

## LC 3325 — Count Substrings With K-Frequency Characters I
**Template: B6 (count subarrays where at least one char appears >= k times)**
```python
from collections import Counter
def numberOfSubstrings(s, k):
    window_counter = Counter()
    left = 0
    count = 0
    for right in range(len(s)):
        window_counter[s[right]] += 1
        while max(window_counter.values()) >= k:
            count += len(s) - right
            window_counter[s[left]] -= 1
            if window_counter[s[left]] == 0:
                del window_counter[s[left]]
            left += 1
    return count
```

---

## LC 3346 — Maximum Frequency of an Element After Performing Operations I
**Template: B3 variant (sort + longest window where max-min <= 2k, track freq)**
```python
from collections import Counter
def maxFrequency(nums, k, numOperations):
    nums.sort()
    freq = Counter(nums)
    left = 0
    window_count = 0
    max_freq = 0
    for right in range(len(nums)):
        window_count += freq[nums[right]]
        while nums[right] - nums[left] > 2 * k:
            window_count -= freq[nums[left]]
            left += 1
        effective = min(window_count, freq[nums[right]] + numOperations)
        if effective > max_freq:
            max_freq = effective
    return max_freq
```

---

## LC 3439 — Reschedule Meetings for Maximum Free Time I
**Template: A2 (fixed window of k consecutive meetings to remove)**
```python
def maxFreeTime(eventTime, k, startTime, endTime):
    n = len(startTime)
    gaps = []
    gaps.append(startTime[0])
    for i in range(n - 1):
        gaps.append(startTime[i + 1] - endTime[i])
    gaps.append(eventTime - endTime[-1])
    window_sum = sum(gaps[:k + 1])
    max_sum = window_sum
    for i in range(len(gaps) - k - 1):
        window_sum -= gaps[i]
        window_sum += gaps[i + k + 1]
        if window_sum > max_sum:
            max_sum = window_sum
    return max_sum
```

---

## LC 3578 — Count Partitions With Max-Min Difference at Most K
**Template: B6 (count subarrays where max-min <= k, dual deque)**
```python
from collections import deque
def countPartitions(nums, k):
    max_dq = deque()
    min_dq = deque()
    left = 0
    count = 0
    for right in range(len(nums)):
        while max_dq and nums[max_dq[-1]] <= nums[right]:
            max_dq.pop()
        max_dq.append(right)
        while min_dq and nums[min_dq[-1]] >= nums[right]:
            min_dq.pop()
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > k:
            left += 1
            if max_dq[0] < left:
                max_dq.popleft()
            if min_dq[0] < left:
                min_dq.popleft()
        count += right - left + 1
    return count
```

---

## LC 3634 — Minimum Removals to Balance Array
**Template: B4 variant (find max window where positives >= negatives)**
```python
def minimumRemovals(nums):
    n = len(nums)
    left = 0
    balance = 0
    max_len = 0
    for right in range(n):
        balance += 1 if nums[right] > 0 else -1
        while balance < 0:
            balance -= 1 if nums[left] > 0 else -1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return n - max_len
```

---

## LC 3641 — Longest Semi-Repeating Subarray
**Template: B3 (at most 1 adjacent pair of equal elements)**
```python
def longestSemiRepetitiveSubstring(s):
    left = 0
    adj_pairs = 0
    max_len = 1
    for right in range(1, len(s)):
        if s[right] == s[right - 1]:
            adj_pairs += 1
        while adj_pairs > 1:
            if s[left] == s[left + 1]:
                adj_pairs -= 1
            left += 1
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len
```

---

## LC 3652 — Best Time to Buy and Sell Stock using Strategy
**Template: A2 (fixed window; track min in left half, max in right half)**
```python
def maximumProfit(prices, k):
    n = len(prices)
    result = 0
    for i in range(k, n):
        buy_price = min(prices[i - k:i])
        sell_price = prices[i]
        if sell_price > buy_price:
            result += sell_price - buy_price
    return result
```

---

## LC 3672 — Sum of Weighted Modes in Subarrays
**Template: A8 skeleton (fixed window; track mode and weighted sum)**
```python
from collections import Counter
def sumOfWeightedModes(nums, k):
    window_counter = Counter(nums[:k])
    max_freq = max(window_counter.values())
    mode_sum = sum(v for v, c in window_counter.items() if c == max_freq)
    result = [mode_sum * max_freq]
    for i in range(len(nums) - k):
        out_val = nums[i]
        in_val = nums[i + k]
        window_counter[out_val] -= 1
        if window_counter[out_val] == 0:
            del window_counter[out_val]
        window_counter[in_val] += 1
        max_freq = max(window_counter.values())
        mode_sum = sum(v for v, c in window_counter.items() if c == max_freq)
        result.append(mode_sum * max_freq)
    return result
```

---

## LC 3795 — Minimum Subarray Length With Distinct Sum At Least K
**Template: B4 (minimum window where sum of distinct values >= k)**
```python
def minimumSubarrayLength(nums, k):
    from collections import Counter
    window_counter = Counter()
    distinct_sum = 0
    left = 0
    min_len = float("inf")
    for right in range(len(nums)):
        if window_counter[nums[right]] == 0:
            distinct_sum += nums[right]
        window_counter[nums[right]] += 1
        while distinct_sum >= k:
            window_len = right - left + 1
            if window_len < min_len:
                min_len = window_len
            window_counter[nums[left]] -= 1
            if window_counter[nums[left]] == 0:
                distinct_sum -= nums[left]
                del window_counter[nums[left]]
            left += 1
    return min_len if min_len != float("inf") else -1
```

---

## LC 837 — New 21 Game
**Template: A2 variant (DP with sliding window sum for probability)**
```python
def new21Game(n, k, maxPts):
    if k == 0 or n >= k + maxPts:
        return 1.0
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    window_sum = 1.0
    for i in range(1, n + 1):
        dp[i] = window_sum / maxPts
        if i < k:
            window_sum += dp[i]
        if i >= maxPts:
            window_sum -= dp[i - maxPts]
    return sum(dp[k:n + 1])
```

---

## LC 1040 — Moving Stones Until Consecutive II
**Template: A2 (fixed window; count stones inside window, minimize moves)**
```python
def numMovesStonesII(stones):
    stones.sort()
    n = len(stones)
    left = 0
    max_moves = max(stones[-1] - stones[1] - n + 2, stones[-2] - stones[0] - n + 2)
    min_moves = max_moves
    window_sum = 1
    for right in range(n):
        while stones[right] - stones[left] >= n:
            left += 1
        window_size = right - left + 1
        if window_size == n - 1 and stones[right] - stones[left] == n - 2:
            min_moves = min(min_moves, 2)
        else:
            min_moves = min(min_moves, n - window_size)
    return [min_moves, max_moves]
```

---

## LC 1871 — Jump Game VII
**Template: B1 variant (sliding window reachability scan)**
```python
def canReach(s, minJump, maxJump):
    n = len(s)
    reachable = [False] * n
    reachable[0] = True
    prev_count = 0
    for i in range(1, n):
        if i >= minJump:
            prev_count += reachable[i - minJump]
        if i > maxJump:
            prev_count -= reachable[i - maxJump - 1]
        if prev_count > 0 and s[i] == '0':
            reachable[i] = True
    return reachable[-1]
```

---

## LC 1888 — Minimum Number of Flips to Make the Binary String Alternating
**Template: A2 on doubled string (fixed window = n; count mismatches)**
```python
def minFlips(s):
    n = len(s)
    target1 = "01" * n
    target2 = "10" * n
    doubled = s + s
    diff1 = 0
    diff2 = 0
    for i in range(n):
        if doubled[i] != target1[i]:
            diff1 += 1
        if doubled[i] != target2[i]:
            diff2 += 1
    result = min(diff1, diff2)
    for i in range(n, 2 * n):
        if doubled[i] != target1[i]:
            diff1 += 1
        if doubled[i] != target2[i]:
            diff2 += 1
        if doubled[i - n] != target1[i - n]:
            diff1 -= 1
        if doubled[i - n] != target2[i - n]:
            diff2 -= 1
        result = min(result, diff1, diff2)
    return result
```

---

## LC 1918 — Kth Smallest Subarray Sum
**Template: B1 + binary search (binary search on answer; count windows with sum <= mid)**
```python
def kthSmallestSubarraySum(nums, k):
    def count_at_most(target):
        left = 0
        current_sum = 0
        count = 0
        for right in range(len(nums)):
            current_sum += nums[right]
            while current_sum > target:
                current_sum -= nums[left]
                left += 1
            count += right - left + 1
        return count
    lo = min(nums)
    hi = sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_at_most(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
```

---

## LC 2067 — Number of Equal Count Substrings
**Template: A8 skeleton (fixed window for each possible count value 1..n/26)**
```python
from collections import Counter
def equalCountSubstrings(s, count):
    result = 0
    for num_distinct in range(1, 27):
        k = num_distinct * count
        if k > len(s):
            break
        window_counter = Counter(s[:k])
        valid = all(v == count for v in window_counter.values()) and len(window_counter) == num_distinct
        if valid:
            result += 1
        for i in range(len(s) - k):
            window_counter[s[i]] -= 1
            if window_counter[s[i]] == 0:
                del window_counter[s[i]]
            window_counter[s[i + k]] += 1
            valid = all(v == count for v in window_counter.values()) and len(window_counter) == num_distinct
            if valid:
                result += 1
    return result
```

---

## LC 2555 — Maximize Win From Two Segments
**Template: A2 + DP (for each right end of second segment, precompute max of first)**
```python
def maximizeWin(prizePositions, k):
    n = len(prizePositions)
    dp = [0] * (n + 1)
    left = 0
    result = 0
    for right in range(n):
        while prizePositions[right] - prizePositions[left] > k:
            left += 1
        window_size = right - left + 1
        dp[right + 1] = max(dp[right], window_size)
        result = max(result, dp[left] + window_size)
    return result
```

---

## LC 3023 — Find Pattern in Infinite Stream I
**Template: A5 (fixed window of size k; sliding match check)**
```python
from collections import Counter
def findPattern(stream, pattern):
    k = len(pattern)
    buffer = []
    for char in stream:
        buffer.append(char)
        if len(buffer) > k:
            buffer.pop(0)
        if len(buffer) == k and buffer == list(pattern):
            return True
    return False
```

---

## LC 3135 — Equalize Strings by Adding or Removing Characters at Ends
**Template: A2 variant (find longest common substring; then answer = len(s) + len(t) - 2*lcs)**
```python
def minCost(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
    return m + n - 2 * max_len
```

---

## LC 3413 — Maximum Coins From K Consecutive Bags
**Template: A2 (fixed window k on sorted bag positions; prefix sum for fast range sum)**
```python
import bisect
def maximumCoins(heroes, coins, k):
    bags = sorted(zip(heroes, coins))
    n = len(bags)
    positions = [b[0] for b in bags]
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + bags[i][1]
    max_coins = 0
    for i in range(n):
        end = positions[i] + k - 1
        j = bisect.bisect_right(positions, end)
        window_coins = prefix[j] - prefix[i]
        if window_coins > max_coins:
            max_coins = window_coins
    return max_coins
```

---

## LC 3422 — Minimum Operations to Make Subarray Elements Equal
**Template: A7 variant (fixed window k; use median via two heaps or sorted structure)**
```python
import sortedcontainers
def minOperations(nums, k):
    from sortedcontainers import SortedList
    window = SortedList(nums[:k])
    def cost():
        mid = k // 2
        median = window[mid]
        total = 0
        for num in window:
            total += abs(num - median)
        return total
    min_cost = cost()
    for i in range(len(nums) - k):
        window.remove(nums[i])
        window.add(nums[i + k])
        c = cost()
        if c < min_cost:
            min_cost = c
    return min_cost
```

---

## LC 1493 — (already solved above — Longest Subarray of 1's After Deleting One Element)
*(see earlier entry)*

---

## Quick Reference — All Templates Used

| Template | Problems |
|---|---|
| **A2** running sum | 187, 209 (basis), 1031, 1052, 1151, 1343, 1423, 1456, 2090, 2134, 3439, 3652, 837, 1040, 1888 |
| **A3** running product | — (covered by B6/713) |
| **A4** exact count fixed | 1343 |
| **A5** frequency map fixed | 567, 187, 1016, 1100 |
| **A6** count all matches | 438, 1297, 2981, 2982 |
| **A7** monotonic deque | 1438, 2762, 3578, 2653 |
| **A8** generic skeleton | 1852, 658, 2107, 2411, 2747, 3191, 3208, 3254, 3255, 3672, 2067 |
| **B1** exact target sum | 1477, 1918 |
| **B2** longest no repeat | 3, 1695, 2260 |
| **B3** at most K distinct | 159, 340, 395, 424, 487, 904, 1004, 1156, 1208, 1493, 1838, 2024, 2730, 2779, 2958, 3641, 3346 |
| **B4** shortest valid | 209, 1234, 1658, 2516, 2875, 2904, 3097, 3795 |
| **B5** min window substring | 3297 |
| **B6** count all windows | 413, 713, 930, 978, 1248, 1358, 2110, 2537, 2743, 2762, 2799, 2962, 3305, 3306, 3325, 3578, 3634 |
| **B7** generic variable | 978, 1839, 2730 |
| **exactly-K trick** | 930, 1248, 3305, 3306 |
| **dual deque** | 1438, 2762, 3578 |
