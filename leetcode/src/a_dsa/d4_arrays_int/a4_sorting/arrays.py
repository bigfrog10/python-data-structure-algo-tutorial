from typing import List
from collections import Counter

# LC88. Merge Sorted Array - merge n2 to n1
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

# LC977. Squares of a Sorted Array
def sortedSquares(self, nums: List[int]) -> List[int]: # O(n)
    n = len(nums)
    result = [0] * n
    left, right = 0, n-1
    for i in range(n)[::-1]:
        if abs(nums[left]) < abs(nums[right]):
            square = nums[right]
            right -= 1
        else:
            square = nums[left]
            left += 1
        result[i] = square * square
    return result

# LC26. Remove Duplicates from Sorted Array - in place change
def removeDuplicates(self, nums: List[int]) -> int:
    i = 0  # 2 pointers
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:  # if equal, we keep going without doing anything.
            i += 1
            nums[i] = nums[j]
    return i+1

# LC80. Remove Duplicates from Sorted Array II - keep 2 same values, in place change
def removeDuplicates(self, nums: List[int]) -> int:
    j, count = 1, 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]: count += 1
        else: count = 1

        if count <= 2:
            nums[j] = nums[i]
            j += 1
    return j

# LC896. Monotonic Array
def isMonotonic(self, A: List[int]) -> bool:
    increasing = decreasing = True  # so all equals is true
    for i in range(len(A)-1):
        if A[i] > A[i+1]: increasing = False
        elif A[i] < A[i+1]: decreasing = False
        if not increasing and not decreasing: return False
    return True

# LC2071. Maximum Number of Tasks You Can Assign - with pills
def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
    from sortedcontainers import SortedList  # O(nlogn * logn)
    tasks.sort()  # sort once, small to large
    workers.sort()
    def check_valid(ans):  # can finish "ans" tasks or not
        _tasks = SortedList(tasks[:ans])  # weakest tasks
        _workers = workers[-ans:]  # strongest workers
        remain_pills = pills
        for worker in _workers:  # O(n)
            task = _tasks[0]
            # the worker can finish the min task without pill, just move on
            if worker >= task: _tasks.pop(0)  # log(n)
            elif remain_pills and worker + strength >= task:
                # the worker cannot finish the min task without pill, but can solve it with pill
                # remove the max task that the strengthened worker can finish instead
                remove_task_idx = _tasks.bisect_right(worker + strength)
                _tasks.pop(remove_task_idx - 1)
                remain_pills -= 1
            else: return False
        return True
    lo, hi = 0, min(len(workers), len(tasks))  #  O(logn)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if check_valid(mid): lo = mid
        else: hi = mid - 1
    return lo

# LC1213. Intersection of Three Sorted Arrays - 3 sorted array
def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
    ans = []
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

# LC825. Friends Of Appropriate Ages
def numFriendRequests(self, ages: List[int]) -> int:  # O(n), prefix sum problem
    buckets = [0] * 121  # 120 is given
    for a in ages: buckets[a] += 1  # bucket count
    res = 0
    for i in range(1, len(buckets)):
        cnt = buckets[i]
        buckets[i] += buckets[i-1]  # cumu sum
        if not cnt: continue
        mid = i // 2 + 7  # requirement 1
        if mid >= i: continue
        res += cnt * (buckets[i] - buckets[mid] - 1)  # minus self
    return res

# LC163. Missing Ranges
def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:  # O(n), O(1)
    def formatRange(lower, upper):
        if lower == upper: return str(lower)
        else: return str(lower) + "->" + str(upper)
    res = []
    prev = lower - 1
    for i in range(len(nums) + 1):
        curr = nums[i] if i < len(nums) else upper + 1
        if prev + 1 <= curr - 1: # there is a gap
            res.append(formatRange(prev + 1, curr - 1))
        prev = curr
    return res

# LC1636. Sort Array by Increasing Frequency
def frequencySort(self, nums: List[int]) -> List[int]:
    count = collections.Counter(nums)
    return sorted(nums, key=lambda x: (count[x], -x))

# LC1122. Relative Sort Array
def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
    aord = {v: i for i, v in enumerate(arr2)}
    maxv = max(arr2)
    arr1.sort(key=lambda x: aord.get(x, maxv+x))
    return arr1

# LC905. Sort Array By Parity - move even to front, odd to back
def sortArrayByParity(self, nums: List[int]) -> List[int]:
    i, j = 0, len(nums) - 1
    while i < j:
        if nums[i] % 2 > nums[j] % 2: nums[i], nums[j] = nums[j], nums[i]
        if nums[i] % 2 == 0: i += 1
        if nums[j] % 2 == 1: j -= 1
    return nums

# LC1331. Rank Transform of an Array
def arrayRankTransform(self, arr: List[int]) -> List[int]:  # O(nlogn)
    rank = {}
    for a in sorted(arr):
        rank.setdefault(a, len(rank) + 1)  # if not there, use this value
    return map(rank.get, arr)

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
    cnt = 0
    for i in range(1, len(nums)):
        if nums[i] <= nums[i-1]:
            if cnt > 0: return False
            cnt += 1
            if i > 1 and nums[i] <= nums[i-2]:
                nums[i] = nums[i-1]  # case like [1,6,7,5,8]
    return True

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
def findMedianSortedArrays(self, A, B):  # O(log(m+n))
    l = len(A) + len(B)
    if l % 2 == 1: return self.kth(A, B, l // 2)
    else: return (self.kth(A, B, l // 2) + self.kth(A, B, l // 2 - 1)) / 2.
def kth(self, a, b, k):
    if not a: return b[k]
    if not b: return a[k]
    ia, ib = len(a) // 2 , len(b) // 2
    ma, mb = a[ia], b[ib]

    if ia + ib < k:  # when k is bigger than the sum of a and b's median indices
        # if a's median is bigger than b's, b's first half doesn't include k
        if ma > mb: return self.kth(a, b[ib + 1:], k - ib - 1)
        else: return self.kth(a[ia + 1:], b, k - ia - 1)
    else:  # when k is smaller than the sum of a and b's indices
        # if a's median is bigger than b's, a's second half doesn't include k
        if ma > mb: return self.kth(a[:ia], b, k)
        else: return self.kth(a, b[:ib], k)
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    # https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2481/Share-my-O(log(min(mn)))-solution-with-explanation
    A, B = nums1, nums2  # O(log(min(m, n)))
    m, n = len(A), len(B)
    if m > n: A, B, m, n = B, A, n, m
    if n == 0: raise ValueError
    imin, imax, half_len = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2  # binary search
        j = half_len - i  # condition 1: i+j = half_len, so break into 2 equal parts
        if i < m and B[j-1] > A[i]: imin = i + 1  # i is too small, must increase it
        elif i > 0 and A[i-1] > B[j]: imax = i - 1  # i is too big, must decrease it
        else:  # i is perfect,
            # condition 2: B[j-1] <= A[i] and A[i-1] <= B[j], so max(left) <= min(right)
            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])
            if (m + n) % 2 == 1: return max_of_left  # the middle one
            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])
            return (max_of_left + min_of_right) / 2.0  # no middle, so take average

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

# Merge 3 sorted arrays and remove duplicates
