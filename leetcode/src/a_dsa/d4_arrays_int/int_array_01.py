
# LC1497. Check If Array Pairs Are Divisible by k
def canArrange(self, arr: List[int], k: int) -> bool:
    count = Counter()
    for i in arr:
        count[i % k] += 1
    if count[0] % 2 == 1: return False
    for key, v in count.items():
        if key != 0 and count[key] != count[k - key]: return False
    return True

# LC517. Super Washing Machines
def findMinMoves(self, machines: List[int]) -> int:
    n = len(machines)
    dress_total = sum(machines)
    if dress_total % n != 0: return -1

    dress_per_machine = dress_total // n
    for i in range(n):
        # Change the number of dresses in the machines to
        # the number of dresses to be removed from this machine
        # (could be negative)
        machines[i] -= dress_per_machine
    # curr_sum is the number of dresses to move at this point,
    # max_sum is the max number of dresses to move at this point or before,
    # m is the number of dresses to move out from the current machine.
    curr_sum = max_sum = res = 0
    for m in machines:
        curr_sum += m
        max_sum = max(max_sum, abs(curr_sum))
        res = max(res, max_sum, m)
    return res

# LC2340. Minimum adjacent swaps to make a valid array
def minimumSwaps(self, nums: List[int]) -> int:
    mx, mn = -inf, inf
    midx = nidx = 0
    for i, n in enumerate(nums):
        if n >= mx:
            mx = n
            midx = i  # last index
        if n < mn:
            mn = n
            nidx = i # 1st index
    midx = len(nums) - midx - 1
    if midx + nidx > len(nums) - 1:
        return midx + nidx - 1
    else:
        return midx + nidx

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
    max_val, res = max(nums), 0
    j = -1
    for i, num in enumerate(nums):
        if num != max_val: j = i
        res = max(res, i - j)
    return res

# LC486. Predict the Winner  player 1 win  player array sides
def predictTheWinner(self, nums: List[int]) -> bool:  # O(n^2) time and O(n) space
    n = len(nums)  # O(n^2) time and space
    @cache
    def maxDiff(left, right):  # score diff between 2 players
        if left == right: return nums[left]
        score_by_left = nums[left] - maxDiff(left + 1, right)
        score_by_right = nums[right] - maxDiff(left, right - 1)
        return max(score_by_left, score_by_right)
    return maxDiff(0, n - 1) >= 0

# LC2210. Count Hills and Valleys in an Array hill and valley
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
            miss += miss  # cover (1, 2*miss), add coin with value miss
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

# LC565 Array Nesting   interloop array loop
def arrayNesting(self, nums: List[int]) -> int: # O(n) tinme and O(1) space
    res = 0
    for i in range(len(nums)):
        cnt = 0
        while nums[i] >= 0:  # 2 loops still check every element once.
            nums[i], cnt, i = -nums[i] - 1, cnt + 1, nums[i]
        res = max(res, cnt)
        if res > len(nums): return res # works without this too, optimization
    return res

# LC1913. Maximum Product Difference Between Two Pairs  diff 2 pairs  prod diff
def maxProductDifference(self, nums: List[int]) -> int:
    big1 = big2 = 0  # avoid sorting, so this is O(n)
    small1 = small2 = inf
    for num in nums:
        if num > big1: big1, big2 = num, big1
        else: big2 = max(big2, num)
        if num < small1: small1, small2 = num, small1
        else: small2 = min(small2, num)
    return big1 * big2 - small1 * small2


# LC2439. Minimize Maximum of Array  min max array ops
def minimizeArrayValue(self, nums: List[int]) -> int:
    cumu = maxm = 0
    for i, num in enumerate(nums, start=1):
        cumu += num
        # At each step, we can try to minimize the element by evenly placing
        # the excess between the previous elements.
        maxm = max(ceil(cumu / i), maxm)
    return maxm
    # https://leetcode.com/problems/minimize-maximum-of-array/solutions/2706521/JavaC++Python-Prefix-Sum-Average-O(n)/

# LC413. Arithmetic Slices  number of subarrays
def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    ans = count = 0
    for i in range(2, len(nums)):
        if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
            count += 1
        else:  # 1 + 2 +...+ count, 1 n-length subarray, 2 (n-1)-length...
            ans += count * (count + 1) // 2  # subarrays
            count = 0
    return ans + count * (count + 1) // 2

#LC2256. Minimum Average Difference
def minimumAverageDifference(self, nums: List[int]) -> int:
    n, total = len(nums), sum(nums)
    ans, mind, presum = -1, inf, 0
    for i in range(n):
        presum += nums[i]
        left_avg = presum // (i + 1)
        right_avg = (total - presum) // (n - i - 1) if i != n-1 else 0
        diff = abs(left_avg - right_avg)
        if diff < mind:
            mind = diff
            ans = i
    return ans







