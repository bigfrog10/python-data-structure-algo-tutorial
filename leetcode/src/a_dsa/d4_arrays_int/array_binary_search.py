
# LC1011. Capacity To Ship Packages Within D Days   ship capacity conveyor
def shipWithinDays(self, weights: List[int], D: int) -> int: # O(nlog(sum - max))
    left, right = max(weights), sum(weights)
    while left < right:  # O(log(right - left)
        midw = (left + right) // 2
        days, currw = 1, 0
        for w in weights:  # O(n)
            if currw + w > midw:
                days += 1
                currw = 0
            currw += w
        if days > D: left = midw + 1
        else: right = midw
    return left

# LC875. Koko Eating Bananas
def minEatingSpeed(self, piles: List[int], h: int) -> int:
    left, right = 1, max(piles)  # we start 1 because we want min value
    while left < right:
        mid = (right + left) // 2  # eating speed k
        counts = sum((p-1) // mid + 1 for p in piles)  # ceiling = (x - 1) // q + 1
        if counts > h: left = mid + 1  # we split too much, so try to split less
        else: right = mid  # we reduce this to get min
    return left

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

# LC162. Find Peak Element - return peak index, array peak element
def findPeakElement(self, nums: List[int]) -> int: # logn
    left, right = 0, len(nums)-1  # need -1 to ensure below +1 inbound
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid+1]: right = mid  # decreasing, so peak is on the other side
        else: left = mid + 1
    return left

# LC2187. Minimum Time to Complete Trips
def minimumTime(self, time: List[int], totalTrips: int) -> int:
    m = min(time) * totalTrips  ## O(logm)
    return bisect_left(range(1, m), totalTrips, key= lambda x: sum(x // t for t in time)) + 1

# LC410. Split Array Largest Sum - split into m parts  split largest sum split sum k split k
def splitArray(self, nums: List[int], m: int) -> int: # O(len(nums) * log(sum(nums))) time, O(1) space
    l, r = max(nums), sum(nums)  # bisect on count of subarrays, which sum < mid
    while l < r:
        mid = (l + r) // 2
        count, cur = 1, 0  # 1 is for last part
        for n in nums:  # O(len(nums))
            cur += n
            if cur > mid:
                count += 1
                cur = n
        if count > m: l = mid + 1
        else: r = mid
    return l

# LC2071. Maximum Number of Tasks You Can Assign - with pills
def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
    from sortedcontainers import SortedList  # O(nlogn * logn)
    tasks.sort()  # sort once, small to large
    workers.sort()
    def check_valid(ans):  # can finish "ans" tasks or not
        _tasks = SortedList(tasks[:ans])  # weakest tasks
        _workers = workers[-ans:]  # strongest workers
        remain_pills = pills
        for worker in _workers:  # O(n)
            task = _tasks[0]
            # the worker can finish the min task without pill, just move on
            if worker >= task: _tasks.pop(0)  # log(n)
            elif remain_pills and worker + strength >= task:
                # the worker cannot finish the min task without pill, but can solve it with pill
                # remove the max task that the strengthened worker can finish instead
                remove_task_idx = _tasks.bisect_right(worker + strength)
                _tasks.pop(remove_task_idx - 1)
                remain_pills -= 1
            else: return False
        return True
    lo, hi = 0, min(len(workers), len(tasks))  #  O(logn)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if check_valid(mid): lo = mid
        else: hi = mid - 1
    return lo

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
