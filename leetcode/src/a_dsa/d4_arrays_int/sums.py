
# LC16. 3Sum Closest
def threeSumClosest(self, nums: List[int], target: int) -> int:  # O(n^2)
    diff = float('inf')
    nums.sort()  # O(nlogn), required by 2 pointers
    for i in range(len(nums)):  # O(n)
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:  # O(n)
            sum1 = nums[i] + nums[lo] + nums[hi]
            if abs(target - sum1) < abs(diff):
                diff = target - sum1
            if sum1 < target: lo += 1
            else: hi -= 1
        if diff == 0: break
    return target - diff
# https://leetcode.com/problems/3sum-closest/discuss/778177/Python3-%3A-Runtime%3A-52-ms-faster-than-99.77

# LC1011. Capacity To Ship Packages Within D Days
def shipWithinDays(self, weights: List[int], D: int) -> int:
    left, right = max(weights), sum(weights)
    while left < right:  # O(log(right - left)
        midw, days, currw = (left + right) // 2, 1, 0
        for w in weights:  # O(n)
            if currw + w > midw:
                days += 1
                currw = 0
            currw += w
        if days > D: left = midw + 1
        else: right = midw
    return left

# LC724. Find Pivot Index
def pivotIndex(self, nums: List[int]) -> int:
    S, leftsum = sum(nums), 0
    for i, x in enumerate(nums):
        if leftsum == (S - leftsum - x): return i
        leftsum += x
    return -1

# LC39. Combination Sum  # O(n^(target/min))
def combinationSum(self, candidates, target):  # fastest
    candidates = sorted(candidates)  # small to large, cut branches
    result = []  # O(n^(target/min cand), power is tree depth
    def dfs(remain, stack): # stack is history
        if remain == 0:
            result.append(stack)
            return
        for item in candidates:
            if item > remain: break
            if stack and item < stack[-1]: continue  # smallers are done already.
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

# LC377. Combination Sum IV
def combinationSum4(self, nums: List[int], target: int) -> int:  # O(T * N)
    @functools.lru_cache(maxsize = None)
    def combs(remain):  # O(target)
        if remain == 0: return 1
        result = 0
        for num in nums:  # O(n)
            if remain - num >= 0: result += combs(remain - num)
        return result
    return combs(target)

# LC494. Target Sum
def findTargetSumWays(self, nums: List[int], S: int) -> int:
    n = len(nums)
    @lru_cache(None)  # O(n * S)
    def dp(i, s):  # index and sum, how many ways to compose a[0], ..., a[i-1] to have sum s.
        if i == n: return s == S  # 1 or 0
        add = dp(i+1, s + nums[i])
        sub = dp(i+1, s - nums[i])
        return add + sub
    return dp(0, 0)

# LC416. Partition Equal Subset Sum  - backpack
def canPartition(self, nums: List[int]) -> bool:  # sequence, not continuous subset
    n, total = len(nums), sum(nums)  # O(n * total)
    if total % 2 != 0: return False
    @lru_cache(maxsize=None)
    def dfs(idx: int, subset_sum: int) -> bool:
        if subset_sum == 0: return True
        if idx == n-1 or subset_sum < 0: return False
        # include this element, or skip this element
        result = dfs(idx + 1, subset_sum - nums[idx + 1]) or dfs(idx + 1, subset_sum)
        return result
    return dfs(0, total // 2)

# LC698. Partition to K Equal Sum Subsets
def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
    nums, ASum = [n for n in nums if n != 0], sum(nums) # O(2^n)
    if ASum % k != 0: return False
    subsum, buckets = ASum // k, [0] * k
    nums.sort(reverse=True)  # larger number first
    lnth = len(nums)
    def dfs(i):  #DFS determines which bucket to put the 'current element' (nums[idx] ) into
        if i == lnth: return True
        for j in range(k): # For each bucket
            buckets[j] += nums[i]
            if buckets[j] <= subsum and dfs(i+1): # if we successfully place ith element
                return True
            buckets[j] -= nums[i] # back out
            # if we put ith into one empty bucket unsucessfully,
            # no need to try other empty buckets. simply return False at the end.
            if buckets[j] == 0: break
        return False
    return dfs(0)

# LC1. Two Sum, top100
def twoSum(self, nums, target):
    cache = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in cache: return [cache[diff], i]
        else: cache[num] = i
    return None

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
def threeSum(self, nums: List[int]) -> List[List[int]]:
    def twoSum(i: int, res: List[List[int]]):
        seen = set()
        j = i + 1
        while j < len(nums):
            complement = -nums[i] - nums[j]
            if complement in seen:
                res.append([nums[i], nums[j], complement])
                while j + 1 < len(nums) and nums[j] == nums[j + 1]: j += 1
            seen.add(nums[j])
            j += 1
    res = []
    nums.sort()
    for i in range(len(nums)):
        if nums[i] > 0: break
        if i == 0 or nums[i - 1] != nums[i]: twoSum(i, res)
    return res

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
def fourSum(self, nums: List[int], target: int) -> List[List[int]]:  # O(n^(k-1))
    def kSum(nums: List[int], target: int, k: int) -> List[List[int]]:
        res = []
        if not nums: return res
        average_value = target // k
        if average_value < nums[0] or nums[-1] < average_value: return res
        if k == 2: return twoSum(nums, target)
        for i in range(len(nums)):
            if i == 0 or nums[i - 1] != nums[i]:  # to avoid dupes
                for subset in kSum(nums[i + 1:], target - nums[i], k - 1):
                    res.append([nums[i]] + subset)
        return res
    def twoSum(nums: List[int], target: int) -> List[List[int]]:
        res = []  # solution for sorted array
        lo, hi = 0, len(nums) - 1
        while (lo < hi):
            curr_sum = nums[lo] + nums[hi]
            if curr_sum < target or (lo > 0 and nums[lo] == nums[lo - 1]): lo += 1
            elif curr_sum > target or (hi < len(nums) - 1 and nums[hi] == nums[hi + 1]):
                hi -= 1
            else:
                res.append([nums[lo], nums[hi]])
                lo += 1
                hi -= 1
        return res
    nums.sort()
    return kSum(nums, target, 4)

# LC454. 4Sum II
def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
    counts = collections.Counter()  # O(n^2)
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
