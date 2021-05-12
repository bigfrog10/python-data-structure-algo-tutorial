from typing import List
from functools import lru_cache
import bisect

# LC724. Find Pivot Index
def pivotIndex(self, nums: List[int]) -> int:
    S = sum(nums)
    leftsum = 0
    for i, x in enumerate(nums):
        if leftsum == (S - leftsum - x): return i
        leftsum += x
    return -1

# LC189. Rotate Array
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k = k % n
    tmp = nums[n-k:]
    nums[k:] = nums[:n-k]
    nums[:k] = tmp

# LC91. Decode Ways, top100
def numDecodings(self, s: str) -> int:  # Best, fast and short
    @lru_cache(maxsize=None)
    def walk(idx):
        if idx == len(s) : return 1
        if s[idx] == '0': return 0
        if idx == len(s) - 1: return 1  # This has to be after above check, case: '0'
        ret = walk(idx + 1)
        if int(s[idx: idx+2]) <= 26: ret += walk(idx + 2)
        return ret
    ret = walk(0)
    return ret

# LC41. First Missing Positive, top100
def firstMissingPositive(self, nums: List[int]) -> int:
    # missing is in [1 ..., len(nums)] and we care only positives
    positives = set(x for x in nums if 0 < x <= len(nums))
    n = len(positives)
    if n == 0: return 1 # if all nums are 0, then next is 1 and 1 is missing
    for i in range(1, n+1): # this order honors smallest missing
        if i not in positives: return i
    return n + 1

# LC412. Fizz Buzz
def fizzBuzz(self, n: int) -> List[str]:
    res = []
    for i in range(1, n+1):
        if i % 3 == 0 and i % 5 == 0: res.append('FizzBuzz')
        elif i % 3 == 0: res.append('Fizz')
        elif i % 5 == 0: res.append('Buzz')
        else: res.append(str(i))
    return res

# LC287. Find the Duplicate Number
def findDuplicate(self, nums: List[int]) -> int:  # use +/- sing, O(n) time and O(1) space
    for num in nums:
        if nums[ abs(num) ] < 0:
            ans = abs(num)
            break
        nums[ abs(num) ] = -nums[ abs(num) ]
    # restore nums
    for i in range(len(nums)):
        nums[i] = abs(nums[i])
    return ans

# LC315. Count of Smaller Numbers After Self
def countSmaller(self, nums):
    def merge_sort(idxnums):
        half = len(idxnums) // 2
        if half:
            left, right = merge_sort(idxnums[:half]), merge_sort(idxnums[half:])
            for i in range(len(idxnums))[::-1]:
                # merge - The smaller numbers on the right of a number are exactly
                # those that jump from its right to its left during a stable sort.
                # So I do mergesort with added tracking of those right-to-left jumps.
                if not right or left and left[-1][1] > right[-1][1]:
                    smaller[left[-1][0]] += len(right) # record current jumps
                    idxnums[i] = left.pop() # normal merge
                else:
                    idxnums[i] = right.pop()
        return idxnums
    smaller = [0] * len(nums)
    merge_sort(list(enumerate(nums))) # idx, value pair
    return smaller

# LC198. House Robber
def rob(self, nums: List[int]) -> int:
    if not nums: return 0
    n = len(nums)
    @lru_cache(None)
    def robbing(idx):
        if idx == n: return 0
        if idx == n-1: return nums[idx]
        rob_first = robbing(idx+2)  # rob first, skip 2nd
        rob_sec = robbing(idx+1)
        return max(nums[idx] + rob_first, rob_sec)
    ret = robbing(0)
    print(robbing.cache_info())
    return ret

# LC740. Delete and Earn
def deleteAndEarn(self, nums):
    if not nums: return 0
    c = collections.Counter(nums)
    m, M = min(nums), max(nums)
    prev = curr = 0 # prev = skip current, curr = take current
    for n in range(m, M+1):
        prev, curr = curr, max(prev + n*c[n], curr)
    return curr

# LC169. Majority Element
def majorityElement(self, nums: List[int]) -> int:
    import collections
    counts = collections.Counter(nums)
    return max(counts.keys(), key=counts.get)

# LC350. Intersection of Two Arrays II
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    import collections
    counts1 = collections.Counter(nums1)
    counts2 = collections.Counter(nums2)
    # or we may just loop nums2 and manipulate counts.
    res = []
    for i, v in counts1.items():
        v1 = counts2[i]
        c = min(v, v1)
        for k in range(c): res.append(i)
    return res

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

# LC1335. Minimum Difficulty of a Job Schedule
def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
    n = len(jobDifficulty) # DP, remove loop k, fastest
    @lru_cache(None)
    def dp(days, jidx, maxd):
        if jidx < days: return float('inf') # we need at least 1 job per day
        if days == 0: return max(jobDifficulty[:jidx+1])
        nmax = max(maxd, jobDifficulty[jidx]) # if we fit jidx .. n-1 in one day
        a = dp(days-1, jidx-1, 0) + nmax  # last day has last new job
        b = dp(days, jidx-1, nmax)  # merge last new job into existing partitions
        #print(f'day={day}, jidx={jidx}, nmax={nmax}')
        return min(a, b)
    r = dp(d-1, n-1, 0)  # we use d-1 is to accommodate jidx < days condition
    return -1 if r == float('inf') else r


# LC1629. Slowest Key
def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
    k, t = keysPressed[0], releaseTimes[0]
    for i in range(1, len(keysPressed)):
        time = releaseTimes[i] - releaseTimes[i-1]
        if time > t or (time == t and keysPressed[i] > k):
            t = time
            k = keysPressed[i]
    return k

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

# LC1762. Buildings With an Ocean View - increasing stack
def findBuildings(self, heights: List[int]) -> List[int]:
    res = [len(heights) - 1]
    for i in range(len(heights) - 2, -1, -1):
        if heights[i] > heights[res[-1]]:
            res.append(i)
    res.reverse()
    return res

# LC238. Product of Array Except Self, top100
def productExceptSelf(self, nums: List[int]) -> List[int]:
    length = len(nums)
    ret = [0] * length
    ret[0] = 1
    for i in range(1, length): ret[i] = nums[i - 1] * ret[i - 1]
    tmp = 1
    for i in reversed(range(length)):
        ret[i] = ret[i] * tmp
        tmp *= nums[i]
    return ret

# LC896. Monotonic Array
def isMonotonic(self, A: List[int]) -> bool:
    increasing = decreasing = True # so all equals is true
    for i in range(len(A)-1):
        if A[i] > A[i+1]: increasing = False
        elif A[i] < A[i+1]: decreasing = False
        if not increasing and not decreasing: return False
    return True

# LC66. Plus One
def plusOne(self, digits: List[int]) -> List[int]:
    for i in reversed(range(len(digits))):
        if digits[i] == 9: digits[i] = 0
        else:
            digits[i] += 1
            return digits # when we don't have carry
    return [1] + digits # when we have carry

# LC674. Longest Continuous Increasing Subsequence
