# LC278. First Bad Version
def firstBadVersion(self, n):
    start, end = 1, n
    while start < end: # start == end after loop
        mid = start + (end - start) // 2
        if isBadVersion(mid): end = mid  # keep end side is bad
        else: start = mid + 1  # start side is after good

    return start

# LC1539. Kth Missing Positive Number
def findKthPositive(self, arr, k):
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

# LC1060. Missing Element in Sorted Array
def missingElement(self, nums: List[int], k: int) -> int:
    if not nums or k == 0: return 0
    diff = nums[-1] - nums[0] + 1 # complete length
    missing = diff - len(nums) # complete length - real length
    # num[i] - i - num[0] is the number of missing values before num[i]
    if k > missing: return nums[-1] + k - missing
    left, right = 0, len(nums) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        missing = nums[mid] - nums[left] - (mid - left)
        if missing < k:
            left = mid
# KEY: move left forward, we need to minus the missing words of this range
            k -= missing
        else: right = mid
    return nums[left] + k

# LC658. Find K Closest Elements
def findClosestElements(self, A, k, x):
    left, right = 0, len(A) - k
    while left < right:
        mid = (left + right) // 2
        if x - A[mid] > A[mid + k] - x: left = mid + 1
        else: right = mid
    return A[left:left + k] # left = right

# LC378. Kth Smallest Element in a Sorted Matrix
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:  # best solution
    n = len(matrix)
    l, r = matrix[0][0], matrix[n - 1][n - 1]
    while l < r:
        mid = (l+r) // 2
        count = sum(bisect.bisect(row, mid) for row in matrix)
        if count < k: l = mid+1
        else: r = mid
    return l

# LC34. Find First and Last Position of Element in Sorted Array
def searchRange(self, nums: List[int], target: int) -> List[int]:
    if not nums or target < nums[0] or target > nums[-1]: return [-1, -1]
    left, right = 0, len(nums) - 1
    ret = []
    while left < right:# search left
        if nums[left] == target: break
        mid = left + (right - left) // 2  # +1 here results infinite loop, [5,7,7,8,8,10] 6
        if nums[mid] < target: left = mid+1
        else: right = mid # we keep right side > target
    if nums[left] != target: return [-1, -1]
    ret.append(left)
    right = len(nums) - 1  # search right
    while left < right:
        if nums[right] == target: break
        mid = left + (right - left) // 2 + 1  # without +1, infinite loop [5,7,7,8,8,10] 8
        if nums[mid] > target: right = mid-1
        else: left = mid
    ret.append(right)
    return ret

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

# LC875. Koko Eating Bananas
def minEatingSpeed(self, piles: List[int], h: int) -> int:
    left, right = 1, max(piles)  # we start 1 because we want min value
    while left < right:
        mid = left + (right - left) // 2
        # ceiling = (x - 1) // q + 1
        counts = sum( (p-1) // mid + 1 for p in piles)
        if counts > h: left = mid + 1 # we split too much, so try to split less
        else: right = mid # we reduce this to get min
    return left

# LC162. Find Peak Element
def findPeakElement(self, nums: List[int]) -> int: # logn
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid+1]: right = mid # decreasing, so peak is on the other side
        else: left = mid + 1
    return left

# LC300. Longest Increasing Subsequence
def lengthOfLIS(self, nums: List[int]) -> int:
    hist = []  # hist[i] smallest ending element in strictly increasing sequence of length i
    for e in nums:
        if not hist: hist.append(e)
        elif e > hist[-1]: hist.append(e)
        else:
            # find index for smallest n such that n >= e
            idx = bisect.bisect_left(hist, e)
            # print(idx)
            hist[idx] = e  # replace it with e
    return len(hist)

# LC1268. Search Suggestions System
def suggestedProducts(self, A, word):
    A.sort()
    res, prefix, i = [], '', 0
    for c in word:
        prefix += c
        i = bisect.bisect_left(A, prefix, i)
        res.append([w for w in A[i:i + 3] if w.startswith(prefix)])
    return res

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
