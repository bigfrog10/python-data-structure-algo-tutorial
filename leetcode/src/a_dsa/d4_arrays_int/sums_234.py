
# LC1. Two Sum, top100 - 2sum
def twoSum(self, nums, target):
    cache = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in cache: return [cache[diff], i]
        else: cache[num] = i
    return None

# LC167. Two Sum II - Input array is sorted 2sum
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

# LC1099. Two Sum Less Than K - 2sum, max sum < K returned
def twoSumLessThanK(self, nums: List[int], k: int) -> int:
    nums.sort()  # O(nlogn)
    answer, left, right = -1, 0, len(nums) -1
    while left < right:  # O(n)
        sum = nums[left] + nums[right]
        if sum < k:
            answer = max(answer, sum)
            left += 1
        else: right -= 1
    return answer
def twoSumLessThanK(self, nums: List[int], k: int) -> int:  # O(n), counting sort
    count = [0] * 1001  # get min and max to set up this array.
    for num in nums: count[num] += 1  # this assumes nums are >= 0
    answer = -1
    lo, hi =1,  1000
    while lo <= hi:
        if lo + hi >= k or count[hi] == 0: hi -= 1
        else:
            if count[lo] > (0 if lo < hi else 1):  # if lo = hi, need 2 for lo + hi
                answer = max(answer, lo + hi)
            lo += 1
    return answer

# LC1877. Minimize Maximum Pair Sum in Array
def minPairSum(self, nums: List[int]) -> int:
    nums.sort()
    return max(nums[i] + nums[~i] for i in range(len(nums) // 2))

# LC16. 3Sum Closest - find closest sum
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

# LC15. 3Sum - return all distinct tuples
def threeSum(self, nums):  # shorter and quicker 90%, O(n^2)
    n = len(nums)
    nums.sort()
    res = []
    for i in range(n-2):
        if nums[i] > 0: break  # then all 3 are > 0 and sum > 0, so can't be 0
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

# LC259. 3Sum Smaller - num of triples with sum < target
def threeSumSmaller(self, nums: List[int], target: int) -> int:  # O(n^2)
    if not nums or len(nums) < 3: return 0
    lnth = len(nums)
    def sum2smaller(nums, start, target):
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

# LC18. 4Sum - return all quadruplets sum to target
def fourSum(self, nums: List[int], target: int) -> List[List[int]]:  # O(n^(k-1))
    def kSum(nums: List[int], target: int, k: int) -> List[List[int]]:
        res = []
        if not nums: return res
        average_value = target // k
        if average_value < nums[0] or nums[-1] < average_value: return res
        if k == 2: return twoSum(nums, target)
        for i in range(len(nums)):
            if i == 0 or nums[i - 1] != nums[i]:  # to avoid dupes
                for subset in kSum(nums[i + 1:], target - nums[i], k - 1):  # we have k-2 n-loops
                    res.append([nums[i]] + subset)
        return res
    def twoSum(nums: List[int], target: int) -> List[List[int]]:  # O(n)
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

# LC454. 4Sum II - number of ways to add to target
def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
    counts = collections.Counter()  # O(n^2)
    for i, j in itertools.product(A, B): counts[i+j] += 1
    ret = 0
    for i, j in itertools.product(C, D): ret += counts[-i-j]
    return ret
