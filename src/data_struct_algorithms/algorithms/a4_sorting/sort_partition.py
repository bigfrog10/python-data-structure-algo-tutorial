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

# Merge Sort
def merge(a,b):
    c = []
    while len(a) != 0 and len(b) != 0:
        if a[0] < b[0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])
    if len(a) == 0: c += b
    else: c += a
    return c
def mergesort(x):
    if len(x) == 0 or len(x) == 1: return x
    else:
        middle = len(x)//2
        a = mergesort(x[:middle])
        b = mergesort(x[middle:])
        return merge(a,b)
