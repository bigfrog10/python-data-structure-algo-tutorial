import bisect
from typing import List

# LC1539. Kth Missing Positive Number
def findKthPositive(self, arr, k):  # O(logn)
    beg, end = 0, len(arr)
    while beg < end:
        mid = (beg + end) // 2  # arr[mid] - mid - 1 is number of missings at mid.
        if arr[mid] - mid - 1 < k: beg = mid + 1
        else: end = mid
    return end + k
# l is the first index that gives at least k missing numbers. It may have more missing numbers than we need, so we are actually interested in index l - 1.
# At index l - 1, we have A[l-1] - (l-1) - 1 missing numbers
# so after index l - 1 , we need to find k - (A[l-1] - (l-1) - 1) missing numbers, i.e. k - A[l-1] + l missing numbers
# At index l - 1, our number is A[l-1]. Add them up, the target number will be A[l-1] + k - A[l-1] + l, i.e. k + l;

# LC162. Find Peak Element
def findPeakElement(self, nums: List[int]) -> int: # logn
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid+1]: right = mid  # decreasing, so peak is on the other side
        else: left = mid + 1
    return left

# LC658. Find K Closest Elements
def findClosestElements(self, A, k, x): # O(logn + k)
    left, right = 0, len(A) - k
    while left < right:
        mid = (left + right) // 2
        if x - A[mid] > A[mid + k] - x: left = mid + 1  # if A[mid + k] is closer to x than A[mid]
        else: right = mid  # smaller element always wins when there is a tie
    return A[left:left + k]  # left = right

# LC1891. Cutting Ribbons
def maxLength(self, ribbons: List[int], k: int) -> int:
    totl, maxl = sum(ribbons), max(ribbons)
    if k > totl: return 0
    lo, hi = max(1, maxl // k), min(totl // k, maxl)
    while lo < hi:  # binary search on desired length
        mid = (lo + hi + 1) // 2
        if sum(x // mid for x in ribbons) >= k: lo = mid
        else: hi = mid - 1
    return lo

# LC1060. Missing Element in Sorted Array
def missingElement(self, nums: List[int], k: int) -> int:
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

# LC378. Kth Smallest Element in a Sorted Matrix
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:  # best solution
    n = len(matrix)  # O(nlognlog(max-min))
    l, r = matrix[0][0], matrix[n - 1][n - 1]
    while l < r:  # log(max-min)
        mid = (l+r) // 2
        count = sum(bisect.bisect(row, mid) for row in matrix)
        if count < k: l = mid+1
        else: r = mid
    return l

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

# LC278. First Bad Version
def firstBadVersion(self, n):
    start, end = 1, n
    while start < end:  # start == end after loop
        mid = start + (end - start) // 2
        if isBadVersion(mid): end = mid  # keep end side is bad
        else: start = mid + 1  # start side is after good
    return start

# LC33. Search in Rotated Sorted Array, top100
def search(self, nums: List[int], target: int) -> int:
    start, end = 0, len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target: return mid
        elif nums[mid] >= nums[start]:
            if nums[start] <= target < nums[mid]: end = mid - 1
            else: start = mid + 1  # cover 2 cases, target <> mid
        else:
            if nums[mid] < target <= nums[end]: start = mid + 1
            else: end = mid - 1
    return -1

# LC81. Search in Rotated Sorted Array II
def search(self, nums: List[int], target: int) -> bool:
    start, end = 0, len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target: return True
        elif nums[mid] > nums[start]:
            if nums[start] <= target < nums[mid]: end = mid - 1
            else: start = mid + 1  # cover 2 cases, target <> mid
        elif nums[mid] < nums[start]:
            if nums[mid] < target <= nums[end]: start = mid + 1
            else: end = mid - 1
        else:  # equal and not target, could use a while. This is O(n)
            start += 1  # can't use mid, since it could jump out of range due to rotation
    return False

# LC153. Find Minimum in Rotated Sorted Array
def findMin(self, nums: List[int]) -> int:
    start, end = 0, len(nums) - 1
    while start < end:
        mid = (start + end) // 2
        if nums[mid] < nums[end]: end = mid
        else: start = mid + 1
    return nums[start]

# LC154. Find Minimum in Rotated Sorted Array II
def findMin(self, nums: List[int]) -> int:
    low, high = 0, len(nums)-1
    while high > low:
        pivot = low + (high - low) // 2
        if nums[pivot] < nums[high]: high = pivot  # we want eliminate higher values
        elif nums[pivot] > nums[high]: low = pivot + 1  # pivot is on left higher values
        else: high -= 1  # have to go slowly since min could be between
    # the 'low' and 'high' index converge to the inflection point.
    return nums[low]

# LC34. Find First and Last Position of Element in Sorted Array
def searchRange(self, nums: List[int], target: int) -> List[int]:  # O(logn)
    if not nums or target < nums[0] or target > nums[-1]: return [-1, -1]
    left, right = 0, len(nums) - 1
    while left < right:  # search left
        if nums[left] == target: break
        mid = left + (right - left) // 2
        if nums[mid] < target: left = mid+1
        else: right = mid  # we keep right side >= target
    if nums[left] != target: return [-1, -1]
    ret = [left]
    right = len(nums) - 1  # search right
    while left < right:
        if nums[right] == target: break
        mid = left + (right - left) // 2 + 1
        if nums[mid] > target: right = mid - 1
        else: left = mid
    ret.append(right)
    return ret

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

# LC719. Find K-th Smallest Pair Distance
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
