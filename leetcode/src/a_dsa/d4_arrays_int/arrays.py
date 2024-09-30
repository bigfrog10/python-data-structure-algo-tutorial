

# LC2022. Convert 1D Array Into 2D Array   1d to 2d
def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
    if m * n != len(original): return []
    return [original[i:i+n] for i in range(0, m*n, n)]

# LC2553. Separate the Digits in an Array  separate digits
def separateDigits(self, nums: List[int]) -> List[int]:
    return [int(i) for n in nums for i in str(n)]

# LC554. Brick Wall
def leastBricks(self, wall: List[List[int]]) -> int:
    edge_counts = defaultdict(int)  # O(nm) time, O(m) space
    for row in wall:
        edge_pos = 0
        for brick_width in row[:-1]:
            edge_pos += brick_width
            edge_counts[edge_pos] += 1
    return len(wall) - max(edge_counts.values(), default=0)

# LC1033. Moving Stones Until Consecutive  move stone
def numMovesStones(self, a: int, b: int, c: int) -> List[int]:
    x, y, z = sorted([a, b, c])
    if x + 1 == y == z - 1: min_steps = 0
    elif y - x > 2 and z - y > 2: min_steps = 2
    else: min_steps = 1  # insert between x y or y z.
    max_steps = z - x - 2
    return [min_steps, max_steps]

# LC1040. Moving Stones Until Consecutive II move stone
def numMovesStonesII(self, stones: List[int]) -> List[int]:
    stones.sort()  # O(nlogn) time and O(1) space
    i, n, low = 0, len(stones), len(stones)
    high = max(stones[-1] - stones[1] - n + 2, stones[-2] - stones[0] - n + 2)
    for j in range(n):
        while stones[j] - stones[i] >= n: i += 1  # sliding window is O(n)
        # there is only 1 stone outside the sliding window, or stones are consecutive
        if j - i + 1 == n - 1 and stones[j] - stones[i] == n - 2: low = 2
        # j - i + 1 is already stored stones in the sliding window
        else: low = min(low, n - (j - i + 1))
    return [low, high]
# https://leetcode.com/problems/moving-stones-until-consecutive-ii/solutions/1488487/python-sliding-window-with-detailed-expalanation/?envType=company&envId=facebook&favoriteSlug=facebook-three-months
# https://leetcode.com/problems/moving-stones-until-consecutive-ii/?envType=company&envId=facebook&favoriteSlug=facebook-three-months
# https://leetcode.com/problems/moving-stones-until-consecutive-ii/solutions/3071375/clean-python-8-lines-high-speed-o-n-time-o-1-space-beats-96-9-with-explanation/?envType=company&envId=facebook&favoriteSlug=facebook-three-months
# https://leetcode.com/problems/moving-stones-until-consecutive-ii/?envType=company&envId=facebook&favoriteSlug=facebook-three-months


# LC1043. Partition Array for Maximum Sum
def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
    n = len(arr)
    @cache
    def f(i):
        if i == n: return 0
        maxSum = mx = -inf
        for j in range(i, min(i+k, n)):
            mx = max(mx, arr[j])  # much faster than max(arr[i:j+1])
            summ = f(j+1) + mx * (j-i+1)
            maxSum = max(maxSum, summ)
        return maxSum
    return f(0)

# LC1052. Grumpy Bookstore Owner
# https://leetcode.com/problems/grumpy-bookstore-owner/description/?envType=company&envId=facebook&favoriteSlug=facebook-three-months
# https://leetcode.com/problems/grumpy-bookstore-owner/solutions/5343974/sliding-windows-10ms-beats-99-47/?envType=company&envId=facebook&favoriteSlug=facebook-three-months
def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
    n = len(customers)
    satisfied = sum(customers[:minutes])
    satisfied += sum( (1-grumpy[i])*customers[i] for i in range(minutes, n))
    max_satisfied=satisfied
    for i in range(minutes, n):
        satisfied -= grumpy[i-minutes] * customers[i-minutes]
        satisfied += grumpy[i] * customers[i]
        max_satisfied = max(max_satisfied, satisfied)
    return max_satisfied

# LC539. Minimum Time Difference  min time diff
def findMinDifference(self, timePoints: List[str]) -> int:  # O(nlogn)
    t = sorted(int(t[:2]) * 60 + int(t[-2:]) for t in timePoints)
    t.append(t[0] + 1440)  # 1440 = 24 * 60, handle circular 1st and last
    return min(b - a for a, b in zip(t, t[1:]))
def findMinDifference(self, timePoints: List[str]) -> int:
    day = 24 * 60  # 1440  # bucket sort, O(n)
    bukt = [0] * day
    for t in timePoints:
        t = int(t[:2]) * 60 + int(t[-2:])
        if bukt[t]: return 0  # same time before, so diff = 0
        bukt[t] = 1
    prev, first, ans=-1, -1, inf
    for i in range(day):
        if bukt[i] == 0: continue
        if prev==-1: first=i
        else: ans = min(ans, i-prev)
        prev=i
    ans = min(ans, day - (prev - first))  # to deal with circular, last - first
    return ans