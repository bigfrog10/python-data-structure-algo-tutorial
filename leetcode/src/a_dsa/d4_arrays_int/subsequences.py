
# LC673. Number of Longest Increasing Subsequence  LIS number
def findNumberOfLIS(self, nums: List[int]) -> int:
    if not nums: return 0  # O(n^2) time and O(n) space
    n = len(nums)
    m, dp, cnt = 0, [1] * n, [1] * n  # dp[i] = length of LIS at i.
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                if dp[i] < dp[j]+1: dp[i], cnt[i] = dp[j]+1, cnt[j]
                elif dp[i] == dp[j]+1: cnt[i] += cnt[j] # add nums[i]
        m = max(m, dp[i])  # m is the max length of longest increasing sequences
    return sum(c for l, c in zip(dp, cnt) if l == m)

# LC3201. Find the Maximum Length of Valid Subsequence I
def maximumLength(self, nums: List[int]) -> int:  # subsequence is not continuous subarray.
    count, end = [0, 0], [0, 0]  # O(1) space and O(n) time
    for n in nums:
        count[n % 2] += 1  # all 0's and all 1's
        end[n % 2] = end[1 - n % 2] + 1  # all 10's and 01's
    return max(max(count), max(end))

# 3202. Find the Maximum Length of Valid Subsequence II
def maximumLength(self, nums: List[int], k: int) -> int:
    res = 0
    for v in range(k):  # Enumearet the sub sum v = (sub[0] + sub[1]) % k
        dp = [0] * k  # dp[a % k] means the longest of subarray ending with a % k
        for a in nums:
            dp[a % k] = dp[(v - a) % k] + 1  # a can be the next element after (v - a) % k
        res = max(res, max(dp))
    return res

# LC416. Partition Equal Subset Sum  - Knapsack
def canPartition(self, nums: List[int]) -> bool:  # sequence, not continuous subset
    n, total = len(nums), sum(nums)  # O(n * total)
    if total % 2 != 0: return False
    @lru_cache(maxsize=None)
    def dfs(idx: int, subset_sum: int) -> bool:
        if subset_sum == 0: return True
        if idx == n-1 or subset_sum < 0: return False
        # include this element, or skip this element
        result = dfs(idx + 1, subset_sum - nums[idx + 1]) or dfs(idx + 1, subset_sum)
        return result
    return dfs(0, total // 2)

# LC1218. Longest Arithmetic Subsequence of Given Difference
def longestSubsequence(self, arr: List[int], diff: int) -> int:
    res = {}  # array value -> longest length of arithmetic seq, fast
    for num in arr:
        res[num] = res[num - diff] + 1 if (num - diff) in res else 1
    return max(res.values())

# LC674. Longest Continuous Increasing Subsequence - this is really continuous subarray, not sequences
def findLengthOfLCIS(self, nums: List[int]) -> int:
    ans = anchor = 0
    for i in range(len(nums)):
        if i and nums[i-1] >= nums[i]: anchor = i  # reset anchor
        ans = max(ans, i - anchor + 1)
    return ans

# LC1027. Longest Arithmetic Subsequence - las
def longestArithSeqLength(self, A: List[int]) -> int:
    n = len(A)
    dp = {}  # dp[index][diff] the length of arithmetic sequence at index with difference diff.
    for i in range(n):
        for j in range(i + 1, n):  # O(n^2)
            diff = A[j] - A[i]  # accumulate counts for each diff over all indices
            dp[j, diff] = dp.get((i, diff), 1) + 1  # A[i], A[j] have length 2
    return max(dp.values())

# LC1143. Longest Common Subsequence - lcs
def longestCommonSubsequence(self, text1: str, text2: str) -> int: # cached recursion
    @lru_cache(None)
    def solve(i, j):
        if i == len(text1) or j == len(text2): return 0
        if text1[i] == text2[j]: return 1 + solve(i+1, j+1)
        else: return max(solve(i+1, j), solve(i, j+1))
    print(solve.cache_info())
    return solve(0, 0)

# LC1498. Number of Subsequences That Satisfy the Given Sum Condition - max + min <= target
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

# LC334. Increasing Triplet Subsequence
def increasingTriplet(self, nums: List[int]) -> bool:
    first_num, second_num = float("inf"), float("inf")
    for n in nums:
        if n <= first_num: first_num = n
        # even first updates later, this ensures there is smaller before this
        elif n <= second_num: second_num = n
        else: return True
    return False

# LC300. Longest Increasing Subsequence  LIS length
def lengthOfLIS(self, nums: List[int]) -> int:  # O(nlogn) runtime, O(n) space
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


