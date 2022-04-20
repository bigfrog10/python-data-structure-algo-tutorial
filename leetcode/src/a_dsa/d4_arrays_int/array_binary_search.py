
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

# LC719. Find K-th Smallest Pair Distance - kth pair
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

# LC162. Find Peak Element - return peak index, array peak element
def findPeakElement(self, nums: List[int]) -> int: # logn
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid+1]: right = mid  # decreasing, so peak is on the other side
        else: left = mid + 1
    return left
