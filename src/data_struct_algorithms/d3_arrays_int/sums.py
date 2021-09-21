
# LC53. Maximum Subarray        - max sum amount all subarrays
def maxSubArray(self, nums: List[int]) -> int:
    total = max_total = nums[0]
    for i in range(1, len(nums)):
        total += nums[i]
        # if the total is not worth to keep, start a new total
        # we can also add code to keep track the start index.
        total = max(total, nums[i])
        max_total = max(max_total, total)  # this is our goal.
    return max_total

# LC39. Combination Sum  # O(n^(target/min))
def combinationSum(self, candidates, target):  # fastest
    candidates = sorted(candidates) # small to large, cut branches
    result = [] # O(n^(target/min cand), power is tree depth
    def dfs(remain, stack): # stack is history
        if remain == 0:
            result.append(stack)
            return
        for item in candidates:
            if item > remain: break
            if stack and item < stack[-1]: continue # smallers are done already.
            else: dfs(remain - item, stack + [item])
    dfs(target, [])
    return result

# LC40. Combination Sum II
def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    result = [] # O(2^n)
    def combine_sum_2(start, path, target):
        if not target:
            result.append(path)
            return
        for i in range(start, len(candidates)):
            # ignore duplicates
            if i > start and candidates[i] == candidates[i - 1]: continue
            if candidates[i] > target: break # so sum > target, not a solution
            # we used i here, so next search starting from i+1
            combine_sum_2(i + 1, path + [candidates[i]], target - candidates[i])
    candidates.sort()
    combine_sum_2(0, [], target)
    return result

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

# LC416. Partition Equal Subset Sum  - backpack
def canPartition(self, nums: List[int]) -> bool:
    n = len(nums)
    total_sum = sum(nums)
    if total_sum % 2 != 0: return False
    @lru_cache(maxsize=None)  # O(n * subset_sum)
    def dfs(idx: int, subset_sum: int) -> bool:
        if subset_sum == 0: return True
        if idx == n-1 or subset_sum < 0: return False
        # include this element, or skip this element
        result = dfs(idx + 1, subset_sum - nums[idx + 1]) or dfs(idx + 1, subset_sum)
        return result
    subset_sum = total_sum // 2
    return dfs(0, subset_sum)

# LC698. Partition to K Equal Sum Subsets
def canPartitionKSubsets(self, nums, k):  # best solution
    nums = [n for n in nums if n != 0] # O(2^n)
    ASum = sum(nums)
    if ASum % k != 0: return False
    buckets = [0]*k
    subsum = ASum // k
    nums.sort(reverse=True)  # larger number first
    lnth = len(nums)
    def walk(i):  #DFS determines which bucket to put the 'current element' (nums[idx] ) into
        if i == lnth: # this means we allocate all elements
            return True # len(set(buckets)) == 1 # , this works too
        for j in range(k): # For each bucket
            buckets[j] += nums[i]
            if buckets[j] <= subsum and walk(i+1): # if we successfully place ith element
                return True
            buckets[j] -= nums[i] # back out
            # if we put ith into one empty bucket unsucessfully,
            # no need to try other empty buckets. simply return False at the end.
            if buckets[j] == 0: break
        return False
    return walk(0)

# LC209. Minimum Size Subarray Sum
def minSubArrayLen(self, s: int, nums: List[int]) -> int:
    total = left = 0 # since all numbers are positive, this works.
    result = len(nums) + 1
    for right, n in enumerate(nums):
        total += n
        while total >= s:
            result = min(result, right - left + 1)
            total -= nums[left]
            left += 1
    return result if result <= len(nums) else 0

# LC494. Target Sum
def findTargetSumWays(self, nums: List[int], S: int) -> int:
    n = len(nums)
    @lru_cache(None)  # O(n * S)
    def dp(i, s):  # index and sum, how many ways to compose a[0], ..., a[i-1] to have sum s.
        if i == n: return s == S # 1 or 0
        add = dp(i+1, s + nums[i])
        sub = dp(i+1, s - nums[i])
        return add + sub
    ret = dp(0, 0)
    print(dp.cache_info())
    return ret

# LC713. Subarray Product Less Than K
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    if k <= 1: return 0 # nums are positive, so there is no solution
    prod = 1
    ans = left = 0
    for right, val in enumerate(nums):
        prod *= val
        while prod >= k:
            prod //= nums[left]
            left += 1
        ans += right - left + 1
    return ans

# LC724. Find Pivot Index
def pivotIndex(self, nums: List[int]) -> int:
    S = sum(nums)
    leftsum = 0
    for i, x in enumerate(nums):
        if leftsum == (S - leftsum - x): return i
        leftsum += x
    return -1

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

# LC1480. Running Sum of 1d Array
def runningSum(self, nums: List[int]) -> List[int]:
    output = [nums[0]]
    for i in range (1, len(nums)):
        nums[i] += nums[i - 1]
        output.append(nums[i])
    return output

# LC1588. Sum of All Odd Length Subarrays
def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    n = len(arr)
    res = 0
    for j in range(1, n + 1, 2):  # O(n^3)
        for i in range(n - j + 1):
            res += sum(arr[i:i + j])
    return res
def sumOddLengthSubarrays(self, A):  # O(n)
    # for any i there are i+1 options on left, n-i options on right.
    # [a[i]] is counted on left, all a's counted on right.
    # for length k arrays, there are (k+1) // 2 odd length arrays
    res, n = 0, len(A)
    for i, a in enumerate(A):
        res += ((i + 1) * (n - i) + 1) // 2 * a  # +1 for ceiling
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
        if nums[i] > 0: break # then all 3 are > 0 and sum > 0, so can't be 0
        if i > 0 and nums[i] == nums[i-1]: continue
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
    res, dups = set(), set()
    seen = {}
    for i, val1 in enumerate(nums):
        if val1 not in dups:
            dups.add(val1)
            for j, val2 in enumerate(nums[i+1:]):
                complement = -val1 - val2
                if complement in seen and seen[complement] == i:
                    res.add(tuple(sorted((val1, val2, complement))))
                seen[val2] = i
    return res

# LC16. 3Sum Closest
def threeSumClosest(self, nums: List[int], target: int) -> int:
    diff = float('inf')
    nums.sort()  # O(nlogn), required by 2 pointers
    for i in range(len(nums)):
        lo, hi = i + 1, len(nums) - 1
        while (lo < hi):
            sum1 = nums[i] + nums[lo] + nums[hi]
            if abs(target - sum1) < abs(diff):
                diff = target - sum1
            if sum1 < target: lo += 1
            else: hi -= 1
        if diff == 0: break
    return target - diff

# LC259. 3Sum Smaller
def threeSumSmaller(self, nums: List[int], target: int) -> int:
    if not nums or len(nums) < 3: return 0
    def sum2smaller(nums, start, target):
        lnth = len(nums)
        counts = 0
        lo, hi = start, lnth-1
        while lo < hi:
            if nums[lo] + nums[hi] < target:
                counts += hi - lo  # ordered array
                lo += 1
            else: hi -= 1
        return counts
    nums.sort()
    counts = 0
    for i in range(len(nums)-2):
        counts += sum2smaller(nums, i+1, target - nums[i])
    return counts

# LC18. 4Sum
def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
    pairs = collections.defaultdict(list)
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            pairs[nums[i]+nums[j]].append((i, j))
    res = set()
    for s, ps in pairs.items():
        for a, b in ps:
            if target - s in pairs:
                for k,v in pairs[target-s]:
                    if a != k and a != v and b != k and b != v:
                        ans = sorted([nums[i] for i in [a,b,k,v]])
                        res.add(tuple(ans))
    return res

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
