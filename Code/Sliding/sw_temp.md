Tempalte 1:
def max_subarray_sum(nums, k):
    max_sum = float("-inf")

    for i in range(len(nums) - k + 1):
        current_sum = sum(nums[i:i + k])

        if current_sum > max_sum:
            max_sum = current_sum

    return max_sum
----------------------------------------------------------
def max_subarray_sum(nums, k):
    current_sum = sum(nums[:k])
    max_sum = current_sum

    for i in range(len(nums) - k):
        current_sum -= nums[i]
        current_sum += nums[i + k]

        if current_sum > max_sum:
            max_sum = current_sum

    return max_sum
----------------------------------------------------------
  def max_subarray_product(nums, k):

    current_product = 1

    for i in range(k):
        current_product *= nums[i]

    max_product = current_product

    for i in range(len(nums) - k):

        current_product /= nums[i]
        current_product *= nums[i + k]

        if current_product > max_product:
            max_product = current_product

    return max_product
 ----------------------------------------------------------

 def subarray_target_sum(nums, target, k):

    current_sum = 0

    for i in range(k):
        current_sum += nums[i]

    count = 1 if current_sum == target else 0

    for i in range(len(nums) - k):

        current_sum -= nums[i]
        current_sum += nums[i + k]

        if current_sum == target:
            count += 1

    return count
----------------------------------------------------------
def has_substring_anagram(s, anagram):

    k = len(anagram)

    window_set = set(s[:k])
    anagram_set = set(anagram)

    if window_set == anagram_set:
        return True

    for i in range(len(s) - k):

        window_set.remove(s[i])

        window_set.add(
            s[i + k]
        )

        if window_set == anagram_set:
            return True

    return False
----------------------------------------------------------
from collections import Counter

def count_substring_anagrams(s, anagram):

    anagram_counter = Counter(anagram)

    window_counter = Counter(
        s[:len(anagram)]
    )

    num_matches = (
        1
        if anagram_counter == window_counter
        else 0
    )

    for i in range(
        len(s) - len(anagram)
    ):

        trailing_char = s[i]

        leading_char = s[
            i + len(anagram)
        ]

        window_counter[
            trailing_char
        ] -= 1

        window_counter[
            leading_char
        ] += 1

        if (
            window_counter
            == anagram_counter
        ):
            num_matches += 1

    return num_matches
----------------------------------------------------------
# Process first K elements

window = build_first_window()

answer = process(window)

for i in range(len(data) - k):

    remove(
        data[i]
    )

    add(
        data[i + k]
    )

    answer = update(
        answer
    )

return answer
----------------------------------------------------------
left = 0
current_sum = 0

for right in range(len(nums)):

    current_sum += nums[right]

    while current_sum > target:

        current_sum -= nums[left]
        left += 1

    if current_sum == target:
        return [left, right]

return None
----------------------------------------------------------
Fixed Window
process first K

for i in range(len(nums)-k):

    remove outgoing

    add incoming

    business logic
Variable Window
left = 0

for right in range(len(nums)):

    add right

    while invalid:

        remove left
        left += 1

    business logic
