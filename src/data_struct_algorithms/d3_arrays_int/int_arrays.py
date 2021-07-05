from typing import List
# LC287. Find the Duplicate Number  Floyd's Tortoise and Hare (Cycle Detection)
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

# LC1295. Find Numbers with Even Number of Digits
def findNumbers(self, nums: List[int]) -> int:
    return sum(~len(str(x)) & 1 for x in nums)
def findNumbers1(self, nums: List[int]) -> int:
    return sum(int(math.log10(n)) % 2 for n in nums) # log10(n) + 1 is the # of digits.

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

# LC846. Hand of Straights
def isNStraightHand(self, hand: List[int], W: int) -> bool:
        if len(hand) % W != 0: return False
        c = collections.Counter(hand)
        for i in sorted(c): # O(nlogn), order is no concern.
            if c[i] == 0: continue
            cnt = c[i]
            for j in range(W):
                c[i + j] -= cnt
                if c[i + j] < 0: return False
        return True

# LC945. Minimum Increment to Make Array Unique
def minIncrementForUnique(self, nums: List[int]) -> int: # O(n)
    if not nums: return 0
    count = collections.Counter(nums)
    maxv = max(nums)
    ans, dups = 0, []
    # we may have dups after singles, so use this range
    for x in range(maxv + len(nums)): # This is the largest range
        if count[x] >= 2: # save dups
            dups.extend([x] * (count[x] - 1))
        elif dups and count[x] == 0: # empty slot found
            ans += x - dups.pop() # get dup fill in this slot
    return ans

# LC179. Largest Number
def largestNumber(self, nums: List[int]) -> str:
    num = [str(x) for x in nums]
    # cmp(x, y) can be replaced by (x > y) - (x < y)
    # in this problem x is (a+b) and y is (b+a)
    cmp = lambda b, a: ((a+b)>(b+a))-((a+b)<(b+a))
    num.sort(key = functools.cmp_to_key(cmp))
    return ''.join(num).lstrip('0') or '0'

# LC41. First Missing Positive, top100
def firstMissingPositive(self, nums: List[int]) -> int:
    # missing is in [1 ..., len(nums)] and we care only positives
    positives = set(x for x in nums if 0 < x <= len(nums))
    n = len(positives)
    if n == 0: return 1 # if all nums are 0, then next is 1 and 1 is missing
    for i in range(1, n+1): # this order honors smallest missing
        if i not in positives: return i
    return n + 1
def firstMissingPositive(self, nums: List[int]) -> int:
    n = len(nums)
    for i in range(n): # clean up so range is [1, n]
        if nums[i] < 1 or nums[i] > n: nums[i] = 0
    for i in range(n):
        if nums[i] > 0: # 1 <= nums[i] % (n + 1) <= n:
            ind = nums[i] % (n + 1) - 1  # ensure ind in bound.
            nums[ind] += n + 1 # seen == a > n
    for i in range(n):
        if nums[i] <= n: return i + 1 # not seen == a <= n
    return n + 1

# LC189. Rotate Array
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k = k % n
    nums[k:], nums[:k] = nums[:n-k], nums[n-k:]

# LC66. Plus One
def plusOne(self, digits: List[int]) -> List[int]:
    for i in reversed(range(len(digits))):
        if digits[i] == 9: digits[i] = 0
        else:
            digits[i] += 1
            return digits # when we don't have carry
    return [1] + digits # when we have carry

# LC268. Missing Number
def missingNumber(self, nums: List[int]) -> int:
    s = sum(nums)
    n = len(nums)
    t = n * (n + 1) // 2
    return t - s
def missingNumber(self, nums):
    missing = len(nums)
    for i, num in enumerate(nums):
        missing ^= i ^ num
    return missing




# LC775. Global and Local Inversions
def isIdealPermutation(self, A):
    return all(abs(i-x) <= 1 for i,x in enumerate(A))

# LC679. 24 Game
def judgePoint24(self, nums: List[int]) -> bool:
    if len(nums) == 1: return math.isclose(nums[0], 24)
    return any(self.judgePoint24([x] + rest)
               for a, b, *rest in itertools.permutations(nums)
               for x in {a+b, a-b, a*b, b and a/b})

# LC1583. Count Unhappy Friends
def unhappyFriends(self, n: int, preferences: List[List[int]], pairs: List[List[int]]) -> int:
    d, p = {}, preferences
    for x, y in pairs:
        d[x] = p[x][:p[x].index(y)] # prefer persons before y
        d[y] = p[y][:p[y].index(x)]
    res = 0
    for x in d:
        for u in d[x]:
            if x in d[u]: # x is in prefer list of u before his pair.
                res += 1
                break
    return res

# LC611. Valid Triangle Number
def triangleNumber(self, nums: List[int]) -> int:
    n = len(nums)
    nums.sort()
    count = 0
    for i in range(n-1,1,-1):
        hi, lo = i - 1, 0
        while lo < hi:
            if nums[hi]+nums[lo] > nums[i]:
                count += hi - lo
                hi -= 1
            else: lo += 1
    return count

# LC349. Intersection of Two Arrays
def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    set1 = set(nums1)
    set2 = set(nums2)
    return list(set2 & set1)



# LC448. Find All Numbers Disappeared in an Array
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        nums[index] = - abs(nums[index])
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]




# LC228. Summary Ranges
def summaryRanges(self, nums: List[int]) -> List[str]:
    ans = []
    pointer = 0
    for i, n in enumerate(nums):
        if i == len(nums) - 1 or nums[i+1] - nums[i] > 1:
            ans.append(str(nums[pointer]) + '->' + str(n) if nums[pointer] != n else str(n))
            pointer = i+1
    return ans

# LC315. Count of Smaller Numbers After Self  # BBG hard
def countSmaller(self, nums: List[int]) -> List[int]:
    sorted_arr = [] # O(nlogn)
    rst = []
    for num in nums[::-1]:
        idx = bisect_left(sorted_arr, num)
        rst.append(idx)
        sorted_arr.insert(idx, num)  # this is o(n)
    return rst[::-1]

import sortedcontainers
class Solution:
    def countSmaller(self, nums):
        s = sortedcontainers.SortedList()
        ans = []
        for i in reversed(range(len(nums))):
            ans.append(s.bisect_left(nums[i]))
            s.add(nums[i])
        return reversed(ans)

# LC493. Reverse Pairs
from sortedcontainers import SortedList
class Solution:
    def reversePairs(self, arr: List[int]) -> int:
        brr = SortedList(arr)
        count = 0
        # anything smaller before larger is discarded.
        for i in range(len(arr)): # O(nlogn), loop is n, logn inside
            brr.discard(arr[i])
            k = brr.bisect_left((arr[i]+1)//2)
            count += k
        return count

# LC1470. Shuffle the Array
def shuffle(self, nums: List[int], n: int) -> List[int]:
    res = [0] * (2*n)
    res[::2] = nums[:n]
    res[1::2] = nums[n:]
    return res
def shuffle1(self, nums: List[int], n: int) -> List[int]:
    return [num for t in zip(nums[:n],nums[n:]) for num in t]

# LC412. Fizz Buzz
def fizzBuzz(self, n: int) -> List[str]:
    res = []
    for i in range(1, n+1):
        if i % 3 == 0 and i % 5 == 0: res.append('FizzBuzz')
        elif i % 3 == 0: res.append('Fizz')
        elif i % 5 == 0: res.append('Buzz')
        else: res.append(str(i))
    return res

# LC697. Degree of an Array
def findShortestSubArray(self, nums: List[int]) -> int:
    dt = collections.defaultdict(list)
    for i, v in enumerate(nums): dt[v].append(i)
    degree = max(len(v) for v in dt.values())
    return min(dt[k][-1] - dt[k][0] + 1 for k in dt if len(dt[k]) == degree)

# LC169. Majority Element
def majorityElement(self, nums: List[int]) -> int:
    import collections
    counts = collections.Counter(nums)
    return max(counts.keys(), key=counts.get)





# LC217. Contains Duplicate
def containsDuplicate(self, nums: List[int]) -> bool:
    showed = set()
    for i in nums:
        if i in showed: return True
        showed.add(i)
    return False
def containsDuplicate(self, nums):
    return len(set(nums)) < len(nums)



# LC1762. Buildings With an Ocean View - increasing stack
def findBuildings(self, heights: List[int]) -> List[int]:
    res = [len(heights) - 1]
    for i in range(len(heights) - 2, -1, -1):
        if heights[i] > heights[res[-1]]:
            res.append(i)
    res.reverse()
    return res

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

# LC1578. Minimum Deletion Cost to Avoid Repeating Letters
def minCost(self, s: str, cost: List[int]) -> int:
    res = max_cost = 0 # O(n)
    for i in range(len(s)): # For each group of continuous same characters,
        if i > 0 and s[i] != s[i - 1]: # min cost = sum_cost(group) - max_cost(group)
            max_cost = 0 # preset this to prevent max added to res
        res += min(max_cost, cost[i]) # add min first, then update max next line
        max_cost = max(max_cost, cost[i]) # this ensures max is not added
    return res




