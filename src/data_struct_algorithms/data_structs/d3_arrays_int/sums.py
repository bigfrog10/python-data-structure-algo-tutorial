# LC560. Subarray Sum Equals K
from collections import defaultdict
from typing import List
def subarraySum(self, nums: List[int], k: int) -> int:
    # approach 4,
    count = cusum = 0
    counts = defaultdict(int)
    for i in range(len(nums)):
        cusum += nums[i]
        if cusum == k: count += 1
        if cusum - k in counts: count += counts[cusum - k]
        counts[cusum] += 1

    return count

# LC523. Continuous Subarray Sum
def checkSubarraySum(self, nums: List[int], k: int) -> bool:
    if not nums: return False
    summ, sd = 0, {0: -1}  # [1,1,1] 3, cumu is divisible by k
    for i, n in enumerate(nums):
        summ += n
        if k != 0: summ = summ % k
        if summ in sd:
            if i - sd[summ] > 1: return True  # [0] 0 if we have =, it returns true but answer is false.
        else: sd[summ] = i
    return False

# LC689. Maximum Sum of 3 Non-Overlapping Subarrays
def maxSumOfThreeSubarrays(self,nums, k):  # best solution
    sum1, sum2, sum3 = sum(nums[:k]), sum(nums[k:2*k]), sum(nums[2*k:3*k])
    cumu1, cumu2, cumu3 = sum1, sum1 + sum2, sum1 + sum2 + sum3
    mw1index, mw2index, mw3index = [0], [0,k], [0,k,2*k]#mw1,mw2,mw3's index.
    for i in range(1,len(nums)-3*k+1):#last index for w1 window will be n-3k
        sum1 += nums[i-1+k] - nums[i-1]
        if sum1 > cumu1: cumu1, mw1index = sum1, [i]
        sum2 += nums[i-1+2*k] - nums[i-1+k]
        if cumu1 + sum2 > cumu2:
            cumu2, mw2index = cumu1 + sum2, mw1index + [i+k]
        sum3 += nums[i-1+3*k] - nums[i-1+2*k]
        if cumu2 + sum3 > cumu3:
            cumu3, mw3index = cumu2 + sum3, mw2index + [i+2*k]
    return mw3index

# LC548. Split Array with Equal Sum
def splitArray(self, nums):
    for  i in range(3, len(nums)-3):
        if nums[i]==0 and nums[i+1]==0: continue
        left = self.canDivide(nums[:i]) # return half sum if divideable
        if len(left) == 0:  continue
        right = self.canDivide(nums[i+1:])
        if len(left & right) > 0: return True  # intersection
    return False
def canDivide(self, nums):
    total = sum(nums)
    res, sumup = set(), nums[0]
    for i in range(1, len(nums)-1):
        if (total-nums[i]) % 2 == 0 and sumup == (total-nums[i]) // 2:
            res.add(sumup)
        sumup += nums[i]
    return res

# LC1. Two Sum, top100
def twoSum(self, nums, target):
    cache = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in cache: return [cache[diff], i]
        else: cache[num] = i
    return None

# LC167. Two Sum II - Input array is sorted
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    # There is no O(logn) solution, so we shoot for O(n)
    # The purpose is to use O(1) space, so take out the hashmap.
    if not numbers or len(numbers) < 2: return None
    lnth = len(numbers)
    i, j = 0, lnth - 1
    while i < j:
        a, b = numbers[i], numbers[j]
        if a + b < target: i += 1
        elif a + b == target: return [i+1, j+1]
        else: j -= 1
    return None

# LC15. 3Sum top100
def threeSum(self, nums):  # shorter and quicker 90%
    n = len(nums)
    nums.sort()
    res = []
    for i in range(n-2):
        if nums[i] > 0: break # sorted, then all 3 are > 0 and sum > 0, so can't be 0
        if i > 0 and nums[i] == nums[i-1]: continue # since we did in previous i-1
        l, r = i+1, n-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0: l +=1
            elif s > 0: r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]: l += 1 # prevent dups
                while l < r and nums[r] == nums[r-1]: r -= 1
                l += 1
                r -= 1
    return res

def threeSum(self, nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()
    for i in range(len(nums)):
        if nums[i] > 0: break
        if i == 0 or nums[i - 1] != nums[i]:
            self.twoSum(nums, i, res)
    return res

def twoSum(self, nums: List[int], i: int, res: List[List[int]]):
    seen = set()
    j = i + 1
    while j < len(nums):
        complement = -nums[i] - nums[j]
        if complement in seen:
            res.append([nums[i], nums[j], complement])
            while j + 1 < len(nums) and nums[j] == nums[j + 1]:
                j += 1
        seen.add(nums[j])
        j += 1

# LC454. 4Sum II
def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
    counts = collections.Counter()
    for i, j in itertools.product(A, B): counts[i+j] += 1
    ret = 0
    for i, j in itertools.product(C, D): ret += counts[-i-j]
    return ret

# LC1010. Pairs of Songs With Total Durations Divisible by 60
def numPairsDivisibleBy60(self, time: List[int]) -> int:
    counts = [0] * 60
    ret = 0
    for t in time:
        t %= 60
        if t == 0: ret += counts[0]
        else: ret += counts[60-t]
        counts[t] += 1
    return ret

# LC1011. Capacity To Ship Packages Within D Days
def shipWithinDays(self, weights: List[int], D: int) -> int:
    left, right = max(weights), sum(weights)
    while left < right:
        midw, days, currw = (left + right) // 2, 1, 0
        for w in weights:
            if currw + w > midw:
                days += 1
                currw = 0
            currw += w
        if days > D: left = midw + 1
        else: right = midw
    return left
