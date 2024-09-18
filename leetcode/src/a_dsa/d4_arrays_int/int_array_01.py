
# LC629. K inverse Pairs Array
def kInversePairs(self, n: int, k: int) -> int:
    # for k=0, dp=1, just 1..n    O(nk) time in loops, O(k) space
    dp, mod = [1]+[0] * k, 1_000_000_007
    # dp[j] is the number of permutations with j inverse pairs, for i numbers
    for i in range(n):  # add a new number
        tmp, sm = [], 0
        for j in range(k + 1):
            sm += dp[j]  # cumulative sum
            if j >= i+1: sm -= dp[j-i-1]  # minus invalid pairs
            sm %= mod
            tmp.append(sm)
        dp = tmp
    return dp[k]
def kInversePairs(self, n: int, k: int) -> int:
    @cache  # works but time out  O(n * k)
    def dp(n, k):  # dp(n, k) means arrays with length n and k inverse pairs
        if k == 0: return 1
        if n == 1 or k < 0: return 0
        # dp(n, k-1): any k-1 pair, move n to left, is a k pair
        # dp(n-1, k): put n at the end
        # dp(n-1, k-n): overcount from #1, n at the beginning. dp(n - 1, k-1 - (n-1) )
        return (dp(n, k-1) + dp(n-1, k) - dp(n-1, k-n)) % (10**9+7)
    return dp(n, k)

# LC2419. Longest Subarray With Maximum Bitwise AND    max bits subarray
def longestSubarray(self, nums: List[int]) -> int:  # O(n) time, O(1) space
    # max(nums) is the asked number. So we look for subarrays of max * m
    j, max_val, res = - 1, max(nums), 0
    for i, num in enumerate(nums):
        if num != max_val: j = i
        res = max(res, i - j)
    return res

# LC486. Predict the Winner
def predictTheWinner(self, nums: List[int]) -> bool:  # O(n^2) time and O(n) space
    n = len(nums)
    @cache  # (O(n^2)
    def maxDiff(left, right):  # he maximum score difference the current player can achieve.
        if left == right: return nums[left]
        score_by_left = nums[left] - maxDiff(left + 1, right)
        score_by_right = nums[right] - maxDiff(left, right - 1)
        return max(score_by_left, score_by_right)
    return maxDiff(0, n - 1) >= 0

# LC2210. Count Hills and Valleys in an Array
def countHillValley(self, nums: List[int]) -> int:  # O(n) time, O(1) space
    count, trend = 0, 0  # trend 1 for up, 0 flat, -1 down
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            if trend == -1:  # previously down
                count += 1  # valley
            trend = 1  # now change to up
        elif nums[i] < nums[i-1]:
            if trend == 1:  # previously up
                count += 1
            trend = -1
    return count

# LC2303. Calculate Amount Paid in Taxes     tax calc
def calculateTax(self, brackets: List[List[int]], income: int) -> float:
    ans = prev = 0
    for hi, pct in brackets:
        hi = min(hi, income)
        ans += (hi - prev) * pct / 100
        prev = hi
    return ans

# LC1151. Minimum Swaps to Group All 1's Together
def minSwaps(self, data: List[int]) -> int:  # O(n) and O(1)
    ones = sum(data)
    cnt_one = max_one = 0
    left = right = 0
    while right < len(data):
        # updating the number of 1's by adding the new element
        cnt_one += data[right]
        right += 1
        # maintain the length of the window to ones
        if right - left > ones:
            # updating the number of 1's by removing the oldest element
            cnt_one -= data[left]
            left += 1
        # record the maximum number of 1's in the window
        max_one = max(max_one, cnt_one)
    return ones - max_one

# LC525. Contiguous Array - longest subarray with equal # of 1 and 0, 01 array 01 equal
def findMaxLength(self, nums: List[int]) -> int:
    val2idx = {}  # O(n) time and space
    cumu = max_len = 0
    for i, num in enumerate(nums):
        cumu += 1 if num == 1 else -1  # so we look for 0 when #0 = #1
        if cumu == 0: max_len = i + 1  # balanced from 0 to i
        elif cumu in val2idx: max_len = max(max_len, i - val2idx[cumu])
        else: val2idx[cumu] = i  # do this on first occurrence for longest cases
    return max_len

# 330. Patching Array
def minPatches(self, nums: List[int], n: int) -> int:  # O(m + logn), m = times to incr index
    miss, added, index = 1, 0, 0
    while miss <= n:  #O(n)  # O(logn) since we doulbe miss
        if index < len(nums) and nums[index] <= miss:
            miss += nums[index]  # cover (1, miss) with new miss
            index += 1
        else:
            miss += miss  # cover (1, 2*miss)
            added += 1  # need new number
    return added

# same as above
# LC2952. Minimum Number of Coins to be Added
def minimumAddedCoins(self, coins: List[int], target: int) -> int:
    coins = sorted(coins)
    miss, added, index = 1, 0, 0
    while miss <= target:  # O(logn) since we doulbe miss
        if index < len(coins) and coins[index] <= miss:  # miss is covered
            miss += coins[index]  # cover (1, miss) with new miss
            index += 1
        else:  # patch miss to the array
            miss += miss  # cover (1, 2*miss)
            added += 1  # need new number, patch
    return added

# LC2625. Flatten Deeply Nested Array        flatten nested
def flatten(arr: list, n):
    if n == 0: return arr
    res = []
    for a in arr:
        if isinstance(a, list):
            b = flatten(a, n-1)
            res.extend(b)
        else: res.append(a)
    return res

# LC565 Array Nesting   interloop
def arrayNesting(self, nums: List[int]) -> int: # O(n) tinme and O(1) space
    res = 0
    for i in range(len(nums)):
        cnt = 0
        while nums[i] >= 0:  # 2 loops still check every element once.
            nums[i], cnt, i = -nums[i] - 1, cnt + 1, nums[i]
        res = max(res, cnt)
        if res > len(nums): return res # works without this too, optimization
    return res

















