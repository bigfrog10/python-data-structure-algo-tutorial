
# LC1891. Cutting Ribbons
def maxLength(self, ribbons: List[int], k: int) -> int:  # O(log(min(totl // k, maxl)))
    totl, maxl = sum(ribbons), max(ribbons)
    if k > totl: return 0
    lo, hi = max(1, maxl // k), min(totl // k, maxl)
    while lo < hi:  # binary search on desired length
        mid = (lo + hi + 1) // 2  # pattern for max
        if sum(x // mid for x in ribbons) >= k: lo = mid
        else: hi = mid - 1
    return lo

# LC1011. Capacity To Ship Packages Within D Days
def shipWithinDays(self, weights: List[int], D: int) -> int:  # O(nlog(sum - max))
    left, right = max(weights), sum(weights)
    while left < right:  # O(log(right - left)
        midw, days, currw = (left + right) // 2, 1, 0
        for w in weights:  # O(n)
            if currw + w > midw:
                days += 1
                currw = 0
            currw += w
        if days > D: left = midw + 1
        else: right = midw
    return left

# LC1231. Divide Chocolate
def maximizeSweetness(self, A, K):
    left, right = 1, sum(A) // (K + 1)
    while left < right:
        mid = (left + right + 1) // 2
        cur = cuts = 0
        for a in A: ## doing cuts now
            cur += a
            if cur >= mid:
                cuts += 1
                cur = 0
        if cuts > K: left = mid
        else: right = mid - 1
    return right

# LC162. Find Peak Element - return peak index, array peak element
def findPeakElement(self, nums: List[int]) -> int: # logn
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid+1]: right = mid  # decreasing, so peak is on the other side
        else: left = mid + 1
    return left

# LC410. Split Array Largest Sum
def splitArray(self, nums: List[int], m: int) -> int:
    l, r = max(nums), sum(nums)
    while l < r:
        mid = (l + r) // 2
        count, cur = 1, 0
        for n in nums:
            cur += n
            if cur > mid:
                count += 1
                cur = n
        if count > m: l = mid + 1
        else: r = mid
    return l

# LC719. Find K-th Smallest Pair Distance - kth pair distance
def smallestDistancePair(self, nums: List[int], k: int) -> int:
    def less_than(x: int) -> int:
        i, pairs = 0, 0 # sliding window starts from i, ends in j
        for j in range(len(nums)):  # O(2n)
            # ignore dist > x pairs. increase i to find pairs <= x
            while nums[j] - nums[i] > x: i += 1
            pairs += j - i # since sorted, so we count in all pairs in between
        return pairs # number of pairs with dist <= x
    nums.sort()  # O(nlogn)
    low, high = 0, nums[-1] - nums[0]
    while low < high: # binary search on differences so we have exactly k pairs, log(D)
        mid = low + (high - low) // 2
        if less_than(mid) < k: low = mid + 1
        else: high = mid
    return low

# LC875. Koko Eating Bananas
def minEatingSpeed(self, piles: List[int], h: int) -> int:
    left, right = 1, max(piles)  # we start 1 because we want min value
    while left < right:
        mid = left + (right - left) // 2
        # ceiling = (x - 1) // q + 1
        counts = sum((p-1) // mid + 1 for p in piles)
        if counts > h: left = mid + 1  # we split too much, so try to split less
        else: right = mid  # we reduce this to get min
    return left

# LC1395. Count Number of Teams - team of 3 solders, increasing or descreasing ratings
from sortedcontainers import SortedList
def numTeams(self, rating: List[int]) -> int:  # nlogn
    def count_low_high(sl, x):
        lo = sl.bisect_left(x)
        hi = len(sl) - lo
        return lo, hi

    result = 0
    left = SortedList()
    right = SortedList(rating)
    for x in rating:  # O(n)
        right.remove(x)  # logn
        loL, hiL = count_low_high(left ,x)
        loR, hiR = count_low_high(right,x)
        result += loL * hiR + hiL * loR
        left.add(x)
    return result
