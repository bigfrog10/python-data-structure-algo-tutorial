
# LC674. Longest Continuous Increasing Subsequence
def findLengthOfLCIS(self, nums: List[int]) -> int:
    ans = anchor = 0
    for i in range(len(nums)):
        if i and nums[i-1] >= nums[i]: anchor = i  # reset anchor
        ans = max(ans, i - anchor + 1)
    return ans

# LC1027. Longest Arithmetic Subsequence
def longestArithSeqLength(self, A: List[int]) -> int:
    n = len(A)
    dp = {}  # dp[index][diff] the length of arithmetic sequence at index with difference diff.
    for i in range(n):
        for j in range(i + 1, n):  # O(n^2)
            diff = A[j] - A[i]  # accumulate counts for each diff over all indices
            dp[j, diff] = dp.get((i, diff), 1) + 1  # A[i], A[j] have length 2
    return max(dp.values())

# LC1143. Longest Common Subsequence
def longestCommonSubsequence(self, text1: str, text2: str) -> int: # cached recursion
    @lru_cache(None)
    def solve(i, j):
        if i == len(text1) or j == len(text2): return 0
        if text1[i] == text2[j]: return 1 + solve(i+1, j+1)
        else: return max(solve(i+1, j), solve(i, j+1))
    print(solve.cache_info())
    return solve(0, 0)

# LC1498. Number of Subsequences That Satisfy the Given Sum Condition
def numSubseq(self, nums: List[int], target: int) -> int:
    nums.sort()  # min and max ignores orders, O(nlogn)
    l, r = 0, len(nums) - 1
    res = 0
    mod = 10**9 + 7
    while l <= r:  # [2,3,3,4,6,7] 12 counter example for =
        if nums[l] + nums[r] > target: r -= 1
        else:
            res += pow(2, r - l, mod)  # count a[i+1] to a[j] all sub sequence
            l += 1
    return res % mod

# LC581. Shortest Unsorted Continuous Subarray
def findUnsortedSubarray(self, nums: List[int]) -> int:
    n = len(nums)
    maxv, minv = nums[0], nums[n-1]
    begin, end = -1, -2  # so that end - begin + 1 = 0 if sorted
    for i in range(1, n):
        maxv = max(maxv, nums[i])
        minv = min(minv, nums[~i])
        if nums[i] < maxv: end = i  # last seen smaller value with index i
        if nums[~i] > minv: begin = n-1 - i # last seen larger value from right side
    return end - begin + 1

# LC334. Increasing Triplet Subsequence
def increasingTriplet(self, nums: List[int]) -> bool:
    first_num, second_num = float("inf"), float("inf")
    for n in nums:
        if n <= first_num: first_num = n
        # even first updates later, this ensures there is smaller before this
        elif n <= second_num: second_num = n
        else: return True
    return False

# LC1218. Longest Arithmetic Subsequence of Given Difference
def longestSubsequence(self, arr: List[int], diff: int) -> int:
    res = {}  # array value -> longest length of arithmetic seq, fast
    for num in arr:
        res[num] = res[num - diff] + 1 if (num - diff) in res else 1
    return max(res.values())

# LC300. Longest Increasing Subsequence
def lengthOfLIS(self, nums: List[int]) -> int:
    hist = []  # hist[i] smallest ending element in strictly increasing sequence of length i
    for e in nums:
        if not hist: hist.append(e)
        elif e > hist[-1]: hist.append(e)
        else:
            # find index for smallest n such that n >= e
            idx = bisect.bisect_left(hist, e)
            hist[idx] = e  # replace it with e
    return len(hist)

# LC128. Longest Consecutive Sequence
def longestConsecutive(self, nums: List[int]) -> int:  # O(n)
    hashed = set(nums)  # overall O(n)
    maxc = 0
    for i in nums:  # O(n)
        if i-1 in hashed: continue  # find new start of a series
        j = 1
        while i + j in hashed: j += 1  # this builds only once for the for loop
        maxc = max(maxc, j)
    return maxc

# LC525. Contiguous Array
def findMaxLength(self, nums: List[int]) -> int:
    c2i = {} # store value to index, cache
    maxlen = count = 0
    for i, n in enumerate(nums):
        count += 1 if n else -1
        if count == 0: maxlen = max(maxlen, i+1)  # balanced from 0 to i
        if count in c2i: maxlen = max(maxlen, i - c2i[count])
        else: c2i[count] = i
    return maxlen

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
