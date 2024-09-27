
#LC256. Paint House
def minCost(self, costs: List[List[int]]) -> int:
    costs = costs.copy()  # O(n) space and time
    for n in reversed(range(len(costs) - 1)):
        # Total cost of painting nth house red.
        costs[n][0] += min(costs[n + 1][1], costs[n + 1][2])
        # Total cost of painting nth house green.
        costs[n][1] += min(costs[n + 1][0], costs[n + 1][2])
        # Total cost of painting nth house blue.
        costs[n][2] += min(costs[n + 1][0], costs[n + 1][1])

    if len(costs) == 0: return 0
    return min(costs[0]) # Return the minimum in the first row.

# LC265. Paint House II
def minCostII(self, costs: List[List[int]]) -> int:  # O(nk) time, O(1) space
    def two_mins(values):  # values is a generator
        ai = a = bi = b = None
        for i, x in values:
            if ai == None or x <= a:
                bi, b = ai, a
                ai, a = i, x
            elif bi == None or x <= b:
                bi, b = i, x
        return ai, a, bi, b  # a smallest, b 2nd smallest
    n, k = len(costs), len(costs[0])
    ai, a, bi, b = two_mins(enumerate(costs[0]))
    for i in range(1, n):
        _ai, _a, _bi, _b = ai, a, bi, b
        vs = ((j, costs[i][j] + (_a if j != _ai else _b)) for j in range(k))
        ai, a, bi, b = two_mins(vs)
    return a
# https://leetcode.com/problems/paint-house-ii/solutions/2971844/python-clean-optimal-space-beats-97/


# 2 state recursion
# LC198. House Robber
def rob(self, nums):  # O(n)
    prevRob , currRob  = 0, 0 # rob current house, or skip current
    for num in nums:
        prevRob , currRob  = currRob , max(prevRob  + num, currRob)
    return currRob
def rob(self, nums: List[int]) -> int:
    if not nums: return 0
    n = len(nums)
    @lru_cache(None)
    def robbing(idx):
        if idx == n: return 0
        if idx == n-1: return nums[idx]
        rob_first = robbing(idx+2)  # rob first, skip 2nd
        rob_sec = robbing(idx+1)
        return max(nums[idx] + rob_first, rob_sec)
    ret = robbing(0)
    print(robbing.cache_info())
    return ret

# LC213. House Robber II
def rob(self, nums: List[int]) -> int:  # O(n)
    def rob_street(nums):
        prevRob , currRob  = 0, 0 # prev loot and current loot
        for num in nums: # rob current house, or skip current
            prevRob , currRob  = currRob , max(prevRob  + num, currRob)
        return currRob
    return max(nums[0] + rob_street(nums[2:-1]), rob_street(nums[1:]))

# LC740. Delete and Earn  delete earn
def deleteAndEarn(self, nums: List[int]) -> int:  # O(n)
    if not nums: return 0
    c = collections.Counter(nums)
    m, M = min(nums), max(nums)
    prev = curr = 0 # prev = skip current, curr = take current
    for n in range(m, M+1):  # count sorting
        prev, curr = curr, max(prev + n*c[n], curr)
    return curr

# LC70. Climbing Stairs   climb stairs
def climbStairs(self, n: int) -> int:
    # a is the number of ways to reach the current step, b is the number of ways to reach the next step
    a = b = 1  # step 0, ground, backtrack from the formula below for step 1 and 2
    for _ in range(n):  # step 1 to n
        a, b = b, a + b
    return a
from functools import lru_cache
def climbStairs(self, n: int) -> int:
    @lru_cache(None)
    def climb(i, n):
        if i > n: return 0
        if i == n: return 1
        return climb(i+1, n) + climb(i+2, n)
    return climb(0, n)



# LC1629. Slowest Key
def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
    k, t = keysPressed[0], releaseTimes[0]
    for i in range(1, len(keysPressed)):
        time = releaseTimes[i] - releaseTimes[i-1]
        if time > t or (time == t and keysPressed[i] > k):
            t = time
            k = keysPressed[i]
    return k

# LC1155. Number of Dice Rolls With Target Sum
def numRollsToTarget(self, d: int, f: int, target: int) -> int:
    @lru_cache(None)
    def dfs(t, d):
        if t == 0 and d == 0: return 1
        if d <= 0 or t <= 0: return 0
        return sum(dfs(t-i, d-1) for i in range(1, f+1))
    return dfs(target, d) % int(1e9 + 7)

# These 3 problems are same logic
# LC1478. Allocate Mailboxes
def minDistance(self, houses: List[int], k: int) -> int:
    houses.sort()  # O(n^2 * k)
    @lru_cache(None)
    def dfs(i, j, k): # place k boxes between i and j houses
        if k == j - i + 1: return 0 # 1 box for every house
        if k > j - i + 1: return inf # overly conservative, have extra.
        if k == 1:
            m = houses[(i+j)//2] # median
            return sum(abs(x-m) for x in houses[i:j+1])
        # try all possibilities for earch group
        return min(dfs(i, x, 1) + dfs(x+1, j, k-1) for x in range(i, j) or [inf])
    return dfs(0, len(houses)-1, k)

# LC312. Burst Balloons - return max coins collected
def maxCoins(self, nums: List[int]) -> int:  # O(n^3) time, O(n^2) space
    nums = [1] + nums + [1]
    @lru_cache(maxsize=None)
    def dp(left, right): # max coins within (left, right), exclusive,  O(n^2)
        if left + 1 == right: return 0
        # add each balloon on the interval and return the maximum score
        return max(dp(left, i) + nums[left] * nums[i] * nums[right] +  # O(n)
                   dp(i, right) for i in range(left+1, right))
    res = dp(0, len(nums)-1)
    print(dp.cache_info())
    return res

# LC1000. Minimum Cost to Merge Stones
def mergeStones(self, stones: List[int], K: int) -> int:
    if not stones: return 0
    if K == 0: return 0
    n = len(stones)
    if (n - 1) % (K - 1) != 0: return -1
    cusum = [0] + list(accumulate(stones))
    @lru_cache(None)
    def dp(i, j):  # min cost of merging stones[i:j+1]
        minc = float('inf')
        for m in range(j, i, -K+1):  # from high to low reduce K-1 piles at a time.
            t = dp(i, m-1) + dp(m, j)
            minc = min(minc, t)
        # we only add cost when it's doable
        if (j - i) % (K-1) == 0: minc += cusum[j+1] - cusum[i]
        return 0 if minc == float('inf') else minc
    return dp(0, n-1)

# LC1335. Minimum Difficulty of a Job Schedule
def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:  # O(nd)
    n = len(jobDifficulty)
    @lru_cache(None)
    def dp(days, jidx, maxd):
        if jidx < days: return float('inf') # we need at least 1 job per day
        if days == 0: return max(jobDifficulty[:jidx+1])
        nmax = max(maxd, jobDifficulty[jidx]) # if we fit jidx .. n-1 in one day
        a = dp(days-1, jidx-1, 0) + nmax  # last day has last new job
        b = dp(days, jidx-1, nmax)  # merge last new job into existing partitions
        return min(a, b)
    r = dp(d-1, n-1, 0)  # we use d-1 is to accommodate jidx < days condition
    return -1 if r == float('inf') else r



