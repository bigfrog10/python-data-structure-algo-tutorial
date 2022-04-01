
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
