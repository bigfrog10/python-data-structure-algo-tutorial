from typing import List
from collections import Counter

# LC1877. Minimize Maximum Pair Sum in Array
def minPairSum(self, nums: List[int]) -> int:
    nums.sort()
    return max(nums[i] + nums[~i] for i in range(len(nums) // 2))

# LC540. Single Element in a Sorted Array
def singleNonDuplicate(self, nums: List[int]) -> int:  # simplest and fast
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mid % 2 == 1: mid -= 1 # move to even case
        if nums[mid] == nums[mid + 1]: # means we have even numbers of left
            lo = mid + 2 # so go to right to find the odd/single
        else: hi = mid # otherwise move to left.
    return nums[lo]  # because hi is not equal

# LC1122. Relative Sort Array
def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
    aord = {v: i for i, v in enumerate(arr2)}
    maxv = max(arr2)
    arr1.sort(key=lambda x: aord.get(x, maxv+x))
    return arr1

# LC665. Non-decreasing Array
def checkPossibility(self, nums: List[int]) -> bool:
    removed = False
    nums = [0] + nums + [1001] # min-1, max + 1 of this array
    for i in range(len(nums) - 3):
        if nums[i + 1] > nums[i + 2]:
            if removed: return False
            if nums[i] > nums[i + 2] and nums[i + 1] > nums[i + 3]:
                return False
            removed = True
    return True

# LC1909. Remove One Element to Make the Array Strictly Increasing
def canBeIncreasing(self, nums: List[int]) -> bool:
    removed = False
    nums = [0] + nums + [1001] # min-1, max + 1 of this array
    for i in range(len(nums) - 3):
        if nums[i + 1] >= nums[i + 2]:
            if removed: return False
            if nums[i] >= nums[i + 2] and nums[i + 1] >= nums[i + 3]:
                return False
            removed = True
    return True

# LC977. Squares of a Sorted Array
def sortedSquares(self, nums: List[int]) -> List[int]: # O(n)
    idx = bisect.bisect(nums, 0)
    n1, n2 = nums[:idx], nums[idx:]
    n22 = [i*i for i in n2]
    n11 = [i*i for i in n1]
    n11.reverse()
    ret = []
    i = j = 0
    while i< len(n11) and j < len(n22):
        if n11[i] <= n22[j]:
            ret.append(n11[i])
            i += 1
        else:
            ret.append(n22[j])
            j += 1
    for i1 in range(i, len(n11)): ret.append(n11[i1])
    for j1 in range(j, len(n22)): ret.append(n22[j1])
    return ret

# LC1636. Sort Array by Increasing Frequency
def frequencySort(self, nums: List[int]) -> List[int]:
    count = collections.Counter(nums)
    return sorted(nums, key=lambda x: (count[x], -x))

# LC26. Remove Duplicates from Sorted Array
def removeDuplicates(self, nums: List[int]) -> int:
    i = 0 # 2 pointers
    for j in range(1, len(nums) ):
        if nums[j] != nums[i]:  # if equal, we keep going without doing anything.
            i += 1
            nums[i] = nums[j]
    return i+1

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




# LC1387. Sort Integers by The Power Value
def getKth(self, lo: int, hi: int, k: int) -> int:
    dic = {1:0}
    def power(n):
        if n in dic: return dic[n]
        if n % 2: dic[n] = power(3 * n + 1) + 1
        else: dic[n] = power(n // 2) + 1
        return dic[n]
    for i in range(lo,hi+1): power(i)
    lst = [(dic[i], i) for i in range(lo, hi+1)]
    heapq.heapify(lst)
    for i in range(k): ans = heapq.heappop(lst)
    return ans[1]

# LC4. Median of Two Sorted Arrays, top100
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    A, B = nums1, nums2
    m, n = len(A), len(B)
    if m > n: A, B, m, n = B, A, n, m
    if n == 0: raise ValueError
    imin, imax, half_len = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2
        j = half_len - i  # condition 1: i+j = half_len
        if i < m and B[j-1] > A[i]:
            imin = i + 1  # i is too small, must increase it
        elif i > 0 and A[i-1] > B[j]:
            imax = i - 1  # i is too big, must decrease it
        else: # i is perfect,
            # condition 2: B[j-1] <= A[i] and A[i-1] <= B[j], so max(left) <= min(right)
            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])
            if (m + n) % 2 == 1: return max_of_left
            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])
            return (max_of_left + min_of_right) / 2.0



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

# LC163. Missing Ranges
def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
    res = []
    if not nums:
        if lower == upper: res.append(str(lower))
        else: res.append(f'{lower}->{upper}')
        return res
    left = lower
    for i in nums:
        if i - left == 1: res.append(str(left))
        elif i - left > 1: res.append(f'{left}->{i-1}')
        left = i+1
    i = nums[-1]
    if upper - i == 1: res.append(str(upper))
    elif upper - i > 1: res.append(f'{i+1}->{upper}')
    return res

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



# LC324. Wiggle Sort II
def wiggleSort(self, nums):
    nums.sort()  # O(nlogn)
    half = len(nums[::2])
    nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1]

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



# LC1481. Least Number of Unique Integers after K Removals
def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
    counts = Counter(arr)
    counts = sorted(counts.items(), key=lambda x: x[1])
    removals = set()
    for key, v in counts:
        if v <= k:
            removals.add(key)
            k = k - v
        else: break
    return len(counts) - len(removals)

# LC896. Monotonic Array
def isMonotonic(self, A: List[int]) -> bool:
    increasing = decreasing = True # so all equals is true
    for i in range(len(A)-1):
        if A[i] > A[i+1]: increasing = False
        elif A[i] < A[i+1]: decreasing = False
        if not increasing and not decreasing: return False
    return True
