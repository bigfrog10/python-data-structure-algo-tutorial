import bisect
from typing import List

# LC1539. Kth Missing Positive Number - from 1
def findKthPositive(self, arr, k):  # O(logn)
    beg, end = 0, len(arr)
    while beg < end:
        mid = (beg + end) // 2  # arr[mid] - mid - 1 is number of missings at mid.
        if arr[mid] - mid - 1 < k: beg = mid + 1
        else: end = mid
    return end + k  # # end-1 is the largest index < k: arr[end-1] + k - (arr[end-1] - (end-1) - 1) = k + end
# l is the first index that gives at least k missing numbers. It may have more missing numbers than we need, so we are actually interested in index l - 1.
# At index l - 1, we have A[l-1] - (l-1) - 1 missing numbers
# so after index l - 1 , we need to find k - (A[l-1] - (l-1) - 1) missing numbers, i.e. k - A[l-1] + l missing numbers
# At index l - 1, our number is A[l-1]. Add them up, the target number will be A[l-1] + k - A[l-1] + l, i.e. k + l;

# LC1060. Missing Element in Sorted Array - kth missing from the left start value
def missingElement(self, nums: List[int], k: int) -> int:  # logn
    # Return how many numbers are missing until nums[idx]
    missing = lambda idx: nums[idx] - nums[0] - idx
    n = len(nums)
    # If kth missing number is larger than  the last element of the array
    if k > missing(n - 1): return nums[-1] + k - missing(n - 1)
    left, right = 0, n - 1  # O(logn), binary search
    # find left = right index such that missing(left - 1) < k <= missing(left)
    while left < right:
        pivot = left + (right - left) // 2
        if missing(pivot) < k: left = pivot + 1
        else: right = pivot
    # kth missing number is greater than nums[left - 1] and less than nums[left]
    return nums[left - 1] + k - missing(left - 1) # k - missing(left-1) is the diff

# LC658. Find K Closest Elements - arround x in the array
def findClosestElements(self, A, k, x): # O(logn + k)
    left, right = 0, len(A) - k
    while left < right:
        mid = (left + right) // 2
        if x - A[mid] > A[mid + k] - x: left = mid + 1  # if A[mid + k] is closer to x than A[mid]
        else: right = mid  # smaller element always wins when there is a tie
    return A[left:left + k]  # left = right

# LC34. Find First and Last Position of Element in Sorted Array
def searchRange(self, nums: List[int], target: int) -> List[int]:
    if not nums or target < nums[0] or target > nums[-1]: return [-1, -1]
    left, right = 0, len(nums) - 1
    while left < right:  # search left
        mid = left + (right - left) // 2
        if nums[mid] < target: left = mid+1
        else: right = mid  # we keep right side >= target
    if nums[left] != target: return [-1, -1]
    left1, right = left, len(nums) - 1  # search right
    while left1 < right:
        mid = left1 + (right - left1 + 1) // 2
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



# LC278. First Bad Version
def firstBadVersion(self, n):
    start, end = 1, n
    while start < end:  # start == end after loop
        mid = start + (end - start) // 2
        if isBadVersion(mid): end = mid  # keep end side is bad
        else: start = mid + 1  # start side is after good
    return start

# LC540. Single Element in a Sorted Array - find it in log time
def singleNonDuplicate(self, nums: List[int]) -> int:  # simplest and fast
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mid % 2 == 1: mid -= 1  # move to even case
        if nums[mid] == nums[mid + 1]:  # means we have even numbers of left
            lo = mid + 2  # so go to right to find the odd/single
        else: hi = mid  # otherwise move to left.
    return nums[lo]  # because hi is not equal

# LC702. Search in a Sorted Array of Unknown Size
def search(self, reader: 'ArrayReader', target: int) -> int:
    hi = 1
    while reader.get(hi) < target: hi <<= 1
    lo = hi >> 1
    while lo <= hi:
        mid = lo + hi >> 1
        if reader.get(mid) < target: lo = mid + 1
        elif reader.get(mid) > target: hi = mid - 1
        else: return mid
    return -1

# LC852. Peak Index in a Mountain Array
def peakIndexInMountainArray(self, arr: List[int]) -> int:
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mi = (lo + hi) // 2
        if arr[mi] < arr[mi + 1]: lo = mi + 1
        else: hi = mi
    return lo

# LC1095. Find in Mountain Array
def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
    A = mountain_arr
    n = A.length()
    # find index of peak
    l, r = 0, n - 1
    while l < r:
        m = (l + r) // 2
        if A.get(m) < A.get(m + 1):
            l = peak = m + 1
        else: r = m
    # find target in the left of peak
    l, r = 0, peak
    while l <= r:
        m = (l + r) // 2
        if A.get(m) < target: l = m + 1
        elif A.get(m) > target: r = m - 1
        else: return m
    # find target in the right of peak
    l, r = peak, n - 1
    while l <= r:
        m = (l + r) // 2
        if A.get(m) > target: l = m + 1
        elif A.get(m) < target: r = m - 1
        else: return m
    return -1

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

# LC704. Binary Search
def search(self, nums: List[int], target: int) -> int:
    b, e = 0, len(nums)-1
    while b <= e:
        m = b + (e - b) // 2
        if nums[m] == target: return m
        elif nums[m] > target: e = m-1
        else: b = m+1
    return -1

# LC362. Design Hit Counter
class HitCounter:
    def __init__(self):
        self.data = []
    def hit(self, timestamp: int) -> None:
        self.data.append(timestamp)
    def getHits(self, timestamp: int) -> int:
        last = timestamp - 300
        if last <= 0: return len(self.data)
        idx = bisect.bisect(self.data, last)
        ret = len(self.data) - idx
        self.data = self.data[idx:]
        return ret


def closest(arr: list, target: int):
    if not arr:
        return None

    left, right = 0, len(arr)-1
    while left + 1 < right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return target
        elif arr[mid] < target:
            left = mid
        else:
            right = mid

    ld = abs(arr[left] - target)
    rd = abs(arr[right] - target)
    return arr[left] if ld < rd else arr[right]

print(closest([1, 2, 3], 2))  # 2
print(closest([1, 4, 6], 3))  # 4
print(closest([1, 4, 6], 5))  # 6
print(closest([1, 3, 3, 4], 2))  # 3

print(closest([1, 2, 4, 5, 6, 20, 26, 30], 11))  # 6
