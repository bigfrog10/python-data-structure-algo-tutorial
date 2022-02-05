
# LC215. Kth Largest Element in an Array, top100
def findKthLargest(self, nums: List[int], k: int) -> int:
    # on average, we reduce by n/2, so add up to 1*n
    # worst case, we reduce by 1, so n-1, n-2, ..., so add up to n^2.
    def partition(nums, left, right):  # bisect right
        target = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] <= target:
                if i != j: nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i # smaller on left, larger on right
    def quick_select(nums, left, right, k):
        mid = partition(nums, left, right)
        if mid == k: return nums[k]
        elif mid < k: return quick_select(nums, mid+1, right, k)
        else: return quick_select(nums, left, mid-1, k)
    n = len(nums)
    return quick_select(nums, 0, n-1, n-k)
# top k elements: https://www.geeksforgeeks.org/k-largestor-smallest-elements-in-an-array/

def findKthLargest(self, nums: List[int], k: int) -> int:  # long solution
    def partition(a: list, pivot, start=0, end=None):
        # O(n), break list a to 2, right is larger, left is equal or smaller
        if end is None: end = len(a)
        i, j = start, end-1
        while i < j:
            while i < end and a[i] <= pivot: i += 1
            while j >= 0 and a[j] > pivot: j -= 1
            if i < j: a[i], a[j] = a[j], a[i]
        if j == end-1: return start + (end - start) // 2
        return j

    def bucket_medians(nums: list, start, end, bucket_size=5):
        medians, i, bucket = [], start, []
        while i < end:
            bucket = []
            while len(bucket) < bucket_size:
                bucket.append(nums[i])
                i += 1
                if i >= end: break
            bucket.sort()
            j = len(bucket) // 2
            medians.append(bucket[j])
        return medians

    def kth_smallest(nums: list, start, end, kth, bucket_size=5):  # O(n)
        if end - start <= bucket_size:  # recursion base case
            a = sorted(nums[start:end])
            return a[kth-1-start]
        medians = bucket_medians(nums, start, end, bucket_size)
        mlen = len(medians)
        median = kth_smallest(medians, 0, mlen, mlen // 2)  # median of medians
        part_idx = partition(nums, median, start, end)
        if kth == part_idx+1:
            a = sorted(nums[start:kth+1])
            return a[kth-1-start]
        elif kth < part_idx+1: return kth_smallest(nums, start, part_idx+1, kth, bucket_size)
        else: return kth_smallest(nums, part_idx+1, end, kth, bucket_size)
    n1 = [-x for x in nums]
    return -kth_smallest(n1, 0, len(n1), k)

# LC973. K Closest Points to Origin
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    # copied from 215
    dist = [(-x*x - y*y, x, y) for x, y in points]
    heapq.heapify(dist)
    ret = heapq.nlargest(k, dist)
    ret1 = [ (x, y) for _, x, y in ret]
    return ret1
def kClosest1(self, points: List[List[int]], K: int) -> List[List[int]]:
    # O(NlogK)
    return heapq.nsmallest(K, points, lambda p: p[0] * p[0] + p[1] * p[1])

# LC347. Top K Frequent Elements, top100
import heapq
import collections
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = collections.Counter(nums)
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
    count = collections.Counter(nums)
    # return sorted(count.keys(), key=lambda x: count[x])[-k:][::-1]
    return heapq.nlargest(k, count.keys(), key=count.get)  # O(N log k) time
