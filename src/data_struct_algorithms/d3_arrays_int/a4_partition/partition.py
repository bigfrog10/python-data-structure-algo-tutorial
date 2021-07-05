
# LC75. Sort Colors   Dutch National Flag
def sortColors(self, nums: List[int]) -> None:
    if not nums: return
    lnth = len(nums)
    # https://en.wikipedia.org/wiki/Dutch_national_flag_problem
    lidx, hidx = 0, lnth - 1
    mid_h = 0
    color = 1 # among 0, 1, 2 different colors
    while mid_h <= hidx:  #O(n)
        if nums[mid_h] < color:
            nums[lidx], nums[mid_h] = nums[mid_h], nums[lidx]
            lidx += 1
            mid_h += 1
        elif nums[mid_h] > color:
            nums[mid_h], nums[hidx] = nums[hidx], nums[mid_h]
            hidx -= 1
        else: mid_h += 1

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



# LC973. K Closest Points to Origin
def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
    # copied from 215
    dist = [ (-x*x - y*y, x, y) for x, y in points]
    heapq.heapify(dist)
    ret = heapq.nlargest(K, dist)
    ret1 = [ (x, y) for _, x, y in ret]
    return ret1

# LC692. Top K Frequent Words
def topKFrequent(self, words: List[str], k: int) -> List[str]:
    freqs = Counter(words)
    return heapq.nsmallest(k, freqs.keys(), lambda w: (-freqs.get(w), w))

