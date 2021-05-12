# LC300. Longest Increasing Subsequence

# LC525. Contiguous Array
def findMaxLength(self, nums: List[int]) -> int:
    c2i = {} # store value to index, cache
    maxlen = count = 0
    for i, n in enumerate(nums):
        count += 1 if n else -1
        if count == 0: maxlen = max(maxlen, i+1) # balanced from 0 to i
        if count in c2i: maxlen = max(maxlen, i - c2i[count])
        else: c2i[count] = i
    return maxlen

# LC53. Maximum Subarray
def maxSubArray(self, nums: List[int]) -> int:
    total = max_total = nums[0]
    for i in range(1, len(nums)):
        total += nums[i]
        # if the total is not worth to keep, start a new total
        # we can also add code to keep track the start index.
        total = max(total, nums[i])
        max_total = max(max_total, total)  # this is our goal.
    return max_total

# LC128. Longest Consecutive Sequence, top100 - use cache
def longestConsecutive(self, nums: List[int]) -> int:  # O(n)
    hashed = set(nums)
    maxc = 0
    for i in nums:
        if i-1 in hashed: continue# find new start of a series
        j = 1
        while i + j in hashed: j += 1
        maxc = max(maxc, j)
    return maxc

# LC1498. Number of Subsequences That Satisfy the Given Sum Condition
def numSubseq(self, nums: List[int], target: int) -> int:
    nums.sort() # min and max ignores orders
    l, r = 0, len(nums) - 1
    res = 0
    mod = 10**9 + 7
    while l <= r:  # [2,3,3,4,6,7] 12 counter example for =
        if nums[l] + nums[r] > target: r -= 1
        else:
            res += pow(2, r - l, mod)  # count a[i+1] to a[j] all sub sequence
            l += 1
    return res % mod

# LC152. Maximum Product Subarray
def maxProduct(self, nums: List[int]) -> int:
    if not nums: return 0
    max_all = curr_max = curr_min = nums[0]
    for i in range(1, len(nums)):
        e = nums[i]
        emx = e * curr_max
        emn = e * curr_min
        curr_max = max([e, emx, emn]) # we track both for +/-
        curr_min = min([e, emx, emn])
        max_all = max(max_all, curr_max)
    return max_all

# Given an array containing only positive integers, return if you can pick two integers from the array which cuts
# the array into three pieces such that the sum of elements in all pieces is equal.
# Example 1:
# Input: array = [2, 4, 5, 3, 3, 9, 2, 2, 2]
# Output: true
# Explanation: choosing the number 5 and 9 results in three pieces [2, 4], [3, 3] and [2, 2, 2]. Sum = 6.
#
# Example 2:
# Input: array =[1, 1, 1, 1],
# Output: false
