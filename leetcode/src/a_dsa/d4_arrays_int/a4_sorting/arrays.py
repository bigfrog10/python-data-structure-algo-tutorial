from typing import List
from collections import Counter

# LC1287. Element Appearing More Than 25% In Sorted Array
def findSpecialInteger(self, arr: List[int]) -> int:
    size = len(arr) // 4
    for i in range(len(arr) - size):
        if arr[i] == arr[i + size]:  # because arr is sorted
            return arr[i]
    return -1

# LC2191. Sort the Jumbled Numbers
def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
    def m(x):
        if x == 0: return mapping[0]
        res, mul = 0, 1
        while x > 0:
            res = res + mapping[x % 10] * mul
            x = x // 10
            mul *= 10
        return res
    return sorted(nums, key=m)

# LC4. Median of Two Sorted Arrays, top100   median 2 sorted arrays  median of 2 sorted array
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    m, n = len(nums1), len(nums2)  # O(log(min(m,n))) time and O(1) space
    if m > n: return self.findMedianSortedArrays(nums2, nums1)
    left, right = 0, m  # bisect on smaller array
    while left <= right:  # search for pA so that
        pA = (left + right) // 2  # so right side may have 1 extra element
        pB = (m + n + 1) // 2 - pA
        maxLeftA = float("-inf") if pA == 0 else nums1[pA - 1]
        minRightA = float("inf") if pA == m else nums1[pA]
        maxLeftB = float("-inf") if pB == 0 else nums2[pB - 1]
        minRightB = float("inf") if pB == n else nums2[pB]
        if maxLeftA <= minRightB and maxLeftB <= minRightA:
            if (m + n) % 2 == 0:
                return (max(maxLeftA, maxLeftB) + min(minRightA, minRightB)) / 2
            else:
                return max(maxLeftA, maxLeftB)
        elif maxLeftA > minRightB: right = pA - 1  # cut the smaller side
        else: left = pA + 1

# LC881. Boats to Save People
def numRescueBoats(self, people: List[int], limit: int) -> int:  # nlogn
    people.sort()
    i, j = 0, len(people) - 1
    ans = 0
    while i <= j:
        ans += 1
        if people[i] + people[j] <= limit:
            i += 1
        j -= 1
    return ans

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

# LC88. Merge Sorted Array - merge n2 to n1  merge array
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


# LC977. Squares of a Sorted Array   square sort array  square of sort  square array
def sortedSquares(self, nums: List[int]) -> List[int]:  # O(n) time, O(1) space
    n, res = len(nums), []
    left, right = 0, n-1
    while left <= right:
        if abs(nums[left]) < abs(nums[right]):
            square = nums[right]
            right -= 1
        else:
            square = nums[left]
            left += 1
        res.append(square * square)
    return reversed(res)

# LC26. Remove Duplicates from Sorted Array - in place change  remove dupe from sorted array  remove dupe array
def removeDuplicates(self, nums: List[int]) -> int:
    i = 0  # 2 pointers
    for j in range(len(nums)):
        if nums[j] != nums[i]:  # if equal, we keep going without doing anything.
            i += 1
            nums[i] = nums[j]
    return i+1

# LC80. Remove Duplicates from Sorted Array II - keep 2 same values, in place change  remove dupe 2
def removeDuplicates(self, nums: List[int]) -> int:
    i = count = 1
    for j in range(1, len(nums)):
        if nums[j] == nums[j - 1]: count += 1
        else: count = 1
        if count <= 2:
            nums[i] = nums[j]
            i += 1
    return i

# LC896. Monotonic Array        mono array
def isMonotonic(self, A: List[int]) -> bool:
    increasing = decreasing = True  # so all equals is true
    for i in range(len(A)-1):
        if A[i] > A[i+1]: increasing = False
        elif A[i] < A[i+1]: decreasing = False
        if not increasing and not decreasing: return False
    return True

# LC1213. Intersection of Three Sorted Arrays - 3 sorted array intersect 3 sorted intersection 3
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
    buckets = [0] * 121  # age 120 is given
    for a in ages: buckets[a] += 1  # bucket count
    res = 0
    for i in range(1, len(buckets)):  # i is age
        cnt = buckets[i]  # number of people with aga i
        buckets[i] += buckets[i-1]  # cumu sum
        if not cnt: continue  # conditions are: other age > i's age > other age // 2 + 7
        mid = i // 2 + 7  # requirement 1
        if mid >= i: continue
        res += cnt * (buckets[i] - buckets[mid] - 1)  # minus self
    return res

# LC163. Missing Ranges
def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:  # O(n), O(1)
    res=[]
    for a in nums+[upper+1]:
        if a > lower:
            res.append([lower, a-1])
        lower = a+1
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

# LC905. Sort Array By Parity - move even to front, odd to back   sort even odd  sort odd even
def sortArrayByParity(self, nums: List[int]) -> List[int]:
    i = 0
    for j in range(len(nums)):
        if nums[j] % 2 == 0:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    return nums

# LC1331. Rank Transform of an Array rank array
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

# LC1439. Find the Kth Smallest Sum of a Matrix With Sorted Rows
def kthSmallest(self, mat: List[List[int]], k: int) -> int:  # O(nklogk) time
    def select(nums1, nums2):
        res, visited = [], set()
        heap = [(nums1[0] + nums2[0], 0, 0)]
        while heap:
            total, i, j = heapq.heappop(heap)
            res.append(total)
            if len(res) == k: break
            if i+1 < len(nums1) and (i+1, j) not in visited:
                heapq.heappush(heap, (nums1[i+1] + nums2[j], i+1, j))
                visited.add((i+1, j))
            if j+1 < len(nums2) and (i,j+1) not in visited:
                heapq.heappush(heap, (nums1[i] + nums2[j+1], i, j+1))
                visited.add((i, j+1))
        return res
    result = mat[0]
    for row in mat[1:]: result = select(result, row)
    return result[-1]

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
    def get_key(log):
        _id, rest = log.split(" ", maxsplit=1)
        return (0, rest, _id) if rest[0].isalpha() else (1, )
    return sorted(logs, key=get_key)

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
def merge(arr1, arr2, arr3):
    import math
    res = []
    i = j = k = 0
    while i < len(arr1) or j < len(arr2) or k < len(arr3):
        a = arr1[i] if i < len(arr1) else math.inf
        b = arr2[j] if j < len(arr2) else math.inf
        c = arr3[k] if k < len(arr3) else math.inf
        mn = min(a, b, c)
        if not res or res[-1] != mn: res.append(mn)
        while i < len(arr1) and arr1[i] == mn: i += 1
        while j < len(arr2) and arr2[j] == mn: j += 1
        while k < len(arr3) and arr3[k] == mn: k += 1
    return res

print(merge([1, 3], [2, 4], [1, 5]))
print(merge([1, 3], [2, 3, 4], [1, 2, 3, 4, 5]))
