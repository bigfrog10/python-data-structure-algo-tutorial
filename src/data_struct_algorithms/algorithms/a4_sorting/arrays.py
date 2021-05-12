from typing import List
from heapq import *
from collections import Counter

# LC88. Merge Sorted Array
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    i, j, k = m-1, n-1, m+n-1  # start from backward
    while i > -1 and j > -1:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    if j > -1: nums1[0:j+1] = nums2[0:j+1]

# LC825. Friends Of Appropriate Ages
def numFriendRequests(self, ages: List[int]) -> int:
    if not ages: return 0
    n = len(ages)
    counts = Counter(ages)
    keys = sorted(counts.keys())
    ret = 0
    for i in range(len(keys)): # B
        age_b = keys[i]
        count_b = counts[age_b]
        for j in range(i, len(keys)): # A
            age_a = keys[j]
            if age_b * 2 <= age_a + 14: break # keys is sorted
            # if age_a < 100 < age_b: continue # This is no use
            ret += counts[age_a] * (count_b - (age_a == age_b))
    return ret

# LC295. Find Median from Data Stream, top100
class MedianFinder:
    def __init__(self):
        self.heaps = [], []  # max from small heap, min from large heap
    def addNum(self, num):
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(large) < len(small): heappush(large, -heappop(small))
    def findMedian(self):
        small, large = self.heaps
        if len(large) > len(small): return float(large[0])
        return (large[0] - small[0]) / 2.0

# LC1213. Intersection of Three Sorted Arrays
def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
    ans = []
    # prepare three pointers to iterate through three arrays
    # p1, p2, and p3 point to the beginning of arr1, arr2, and arr3 accordingly
    p1 = p2 = p3 = 0
    while p1 < len(arr1) and p2 < len(arr2) and p3 < len(arr3):
        if arr1[p1] == arr2[p2] == arr3[p3]:
            ans.append(arr1[p1])
            p1 += 1
            p2 += 1
            p3 += 1
        else: # move smallest pointer
            if arr1[p1] < arr2[p2]: p1 += 1
            elif arr2[p2] < arr3[p3]: p2 += 1
            else: p3 += 1
    return ans

# LC1439. Find the Kth Smallest Sum of a Matrix With Sorted Rows
def kthSmallest(self, mat: List[List[int]], k: int) -> int: # 300ms
    dp = mat[0][:min(k,len(mat[0]))]
    for l in mat[1:]:
        tmp = []
        for i in dp:
            for j in l:
                tmp += [i+j]
        dp = sorted(tmp)[:min(k,len(tmp))]
    return dp[-1]

# LC969. Pancake Sorting
def pancakeSort(self, arr: List[int]) -> List[int]:
    res = []
    for x in range(len(arr), 1, -1):
        i = arr.index(x)
        # i+1 to flip x to the front, x to flip x to the back
        if i > 0: res.extend([i + 1, x])
        else: res.extend([x])
        # arr[i] is unchanged with 2 flips, arr[:i:-1](from i+1 to end) flipped only once.
        # arr[:i:-1] = arr[i+1:][::-1]
        arr = arr[:i:-1] + arr[:i]
    return res

# LC347. Top K Frequent Elements, top100
import heapq
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = Counter(nums)
    arr = [(f, k) for k, f in count.items()]
    def partition(arr, left, right):
        target = arr[right][0]
        i = left
        for j in range(left, right):
            if arr[j][0] <= target:
                if i != j: arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[right] = arr[right], arr[i]
        return i
    def quick_select(arr, left, right, k):
        mid = partition(arr, left, right)
        if mid == k: return arr[k:]
        elif mid < k: return quick_select(arr, mid+1, right, k)
        else: return quick_select(arr, left, mid-1, k)
    n = len(arr)
    sorteda = quick_select(arr, 0, n-1, n-k)
    return [x[1] for x in sorteda][::-1]
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    if k == len(nums): return nums
    count = Counter(nums)
    # return sorted(count.keys(), key=lambda x: count[x])[-k:][::-1]
    return heapq.nlargest(k, count.keys(), key=count.get) # O(N log k) time

# LC4. Median of Two Sorted Arrays, top100
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    N1, N2 = len(nums1), len(nums2)
    if N1 < N2: nums1, N1, nums2, N2 = nums2, N2, nums1, N1
    l, r = 0, N2*2
    while l <= r:
        j = (l + r) // 2
        i = N1 + N2 - j
        # inf comes from nonoverlap cases
        L1 = float('-inf') if i == 0 else nums1[(i-1) // 2]
        R1 = float('inf') if i == 2*N1 else nums1[i // 2]
        L2 = float('-inf') if j == 0 else nums2[(j-1) // 2]
        R2 = float('inf') if j == 2*N2 else nums2[j // 2]
        if L1 > R2: l = j + 1
        elif L2 > R1: r = j - 1
        else: return (max(L1, L2) + min(R1, R2))/2.0

# LC240. Search a 2D Matrix II
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix: return False
    h, w = len(matrix), len(matrix[0])
    row, col = h - 1, 0
    while row >= 0 and col < w:
        if target == matrix[row][col]: return True
        elif target < matrix[row][col]:  row -= 1
        else: col += 1
    return False

# LC26. Remove Duplicates from Sorted Array
def removeDuplicates(self, nums: List[int]) -> int:
    i = 0 # 2 pointers
    for j in range(1, len(nums) ):
        if nums[j] != nums[i]:  # if equal, we keep going without doing anything.
            i += 1
            nums[i] = nums[j]
    return i+1

# LC324. Wiggle Sort II
def wiggleSort(self, nums):
    nums.sort()  # O(nlogn)
    half = len(nums[::2])
    nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1]

# LC334. Increasing Triplet Subsequence
def increasingTriplet(self, nums: List[int]) -> bool:
    first_num, second_num = float("inf"), float("inf")
    for n in nums:
        if n <= first_num: first_num = n
        # even first updates later, this ensures there is smaller before this
        elif n <= second_num: second_num = n
        else: return True
    return False

# LC937. Reorder Data in Log Files
def reorderLogFiles(self, logs: List[str]) -> List[str]:
    if not logs: return logs
    la, na = [], []  # letter, number arrays
    for s in logs:
        sidx = s.index(' ') + 1 # first char after space
        if s[sidx].isnumeric(): na.append(s)
        else: la.append(s)
    la.sort(key=lambda s: (s[s.index(' ')+1:], s[:s.index(' ')]))
    return la + na


