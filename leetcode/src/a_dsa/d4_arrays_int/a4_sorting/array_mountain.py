
# LC852. Peak Index in a Mountain Array  peak mount
def peakIndexInMountainArray(self, arr: List[int]) -> int:
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mi = (lo + hi) // 2
        if arr[mi] < arr[mi + 1]: lo = mi + 1
        else: hi = mi
    return lo

# LC1095. Find in Mountain Array  for given target    find target index    mount find target  mount peak
def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
    A = mountain_arr  # O(logn) time and O(1) space
    n = A.length()
    l, r = 0, n - 1
    while l < r: # find index of peak
        m = (l + r) // 2
        if A.get(m) < A.get(m + 1):
            l = m + 1
        else: r = m
    peak = l
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
