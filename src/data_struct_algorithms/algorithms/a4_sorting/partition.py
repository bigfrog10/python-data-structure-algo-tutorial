# LC75. Sort Colors
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
