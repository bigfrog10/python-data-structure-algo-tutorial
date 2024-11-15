
# 2 types of search


###############################################################################
# we know the target
# LC69. Sqrt(x)
def mySqrt(self, x: int) -> int:
    left, right = 0, x
    while left <= right:
        mid = (left + right) // 2
        if mid * mid < x: left = mid + 1
        elif mid * mid > x: right = mid -1
        else: return mid
    return right  # since right < left and the problem requires round down.

# LC35. Search Insert Position
def searchInsert(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        pivot = (left + right) // 2
        if nums[pivot] == target: return pivot
        if target < nums[pivot]: right = pivot - 1
        else: left = pivot + 1
    return left
# return bisect.bisect_left(nums, target)

# another pattern
###############################################################################
# LC69. Sqrt(x)
def mySqrt(self, x: int) -> int:  # This is a new pattern
    left, right = 0, x  # O(logn)
    while left < right:
        mid = (left + right + 1) // 2  # works only for integers
        if mid * mid > x: right = mid - 1
        else: left = mid
    return left
# LC34. Find First and Last Position of Element in Sorted Array  start and end position start end position
def searchRange(self, nums: List[int], target: int) -> List[int]:
    if not nums or target < nums[0] or target > nums[-1]: return [-1, -1]
    left, right = 0, len(nums) - 1
    while left < right:  # search left
        mid = (right + left) // 2
        if nums[mid] < target: left = mid+1
        else: right = mid  # we keep right side >= target
    if nums[left] != target: return [-1, -1]
    left1, right = left, len(nums) - 1  # search right
    while left1 < right:
        mid = (right + left1 + 1) // 2
        if nums[mid] > target: right = mid - 1
        else: left1 = mid
    return [left, right]
def searchRange1(self, nums: List[int], target: int) -> List[int]:
    if not nums: return [-1, -1]
    left = bisect.bisect_left(nums, target)  # the index of leftmost target
    left = left if left < len(nums) and nums[left] == target else -1
    if left == -1: return [-1, -1]
    right = bisect.bisect(nums, target)  # the index right after target
    return [left, right - 1]
###############################################################################


###############################################################################
# we dont know the target
# 1062
left, right = 0, len(s)
while left < right:
    mid = (left + right) // 2
    if valid(mid): left = mid + 1
    else: right = mid
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

# LC658. Find K Closest Elements - arround x in the array  k closest integer
def findClosestElements(self, A, k, x): # O(logn + k)
    left, right = 0, len(A) - k
    while left < right:
        mid = (left + right) // 2
        if x - A[mid] > A[mid + k] - x: left = mid + 1  # if A[mid + k] is closer to x than A[mid]
        else: right = mid  # smaller element always wins when there is a tie
    return A[left:left + k]  # left = right

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

# LC162. Find Peak Element - return peak index, array peak element
def findPeakElement(self, nums: List[int]) -> int: # logn
    left, right = 0, len(nums)-1  # need -1 to ensure below +1 inbound
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid+1]: right = mid  # decreasing, so peak is on the other side
        else: left = mid + 1
    return left

# LC852. Peak Index in a Mountain Array  peak mount
def peakIndexInMountainArray(self, arr: List[int]) -> int:
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mi = (lo + hi) // 2
        if arr[mi] < arr[mi + 1]: lo = mi + 1
        else: hi = mi
    return lo

# LC540. Single Element in a Sorted Array - of double elements, find it in log time single elem sorted single sorted
def singleNonDuplicate(self, nums: List[int]) -> int:  # simplest and fast  O(logn)
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (hi + lo) // 2
        if mid % 2 == 1: mid -= 1  # move to even case
        if nums[mid] == nums[mid + 1]:  # means we have even numbers of left
            lo = mid + 2  # so go to right to find the odd/single
        else: hi = mid  # otherwise move to left.
    return nums[lo]  # because hi is not equal

# LC278. First Bad Version
def firstBadVersion(self, n):
    start, end = 1, n
    while start < end:  # start == end after loop
        mid = (end + start) // 2
        if isBadVersion(mid): end = mid  # keep end side is bad
        else: start = mid + 1  # start side is after good
    return start