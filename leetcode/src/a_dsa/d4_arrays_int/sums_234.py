
# LC1. Two Sum, top100 - 2sum
def twoSum(self, nums, target):  # O(n) time and space
    cache = {}  # value -> index, which is returned
    for i, num in enumerate(nums):
        diff = target - num
        if diff in cache: return [cache[diff], i]
        else: cache[num] = i
    return None

# LC167. Two Sum II - Input array is sorted 2sum sorted  2sum ii
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    # O(n) time and O(1) space(take out cache)- there is no O(logn) solution
    i, j = 0, len(numbers) - 1
    while i < j:
        s = numbers[i] + numbers[j]
        if s < target: i += 1
        elif s == target: return [i+1, j+1]
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

# LC170. Two Sum III - Data structure design   2sum3
class TwoSum:
    def __init__(self):
        self.counters = Counter()
    def add(self, number: int) -> None:
        self.counters[number] += 1
    def find(self, value: int) -> bool:
        for w in self.counters.keys():
            d = value - w
            if d != w and d in self.counters: return True
            if d == w and self.counters[w] > 1: return True
        return False

# LC1877. Minimize Maximum Pair Sum in Array
def minPairSum(self, nums: List[int]) -> int:
    nums.sort()
    return max(nums[i] + nums[~i] for i in range(len(nums) // 2))

# LC16. 3Sum Closest - find closest sum
def threeSumClosest(self, nums: List[int], target: int) -> int:  # O(n^2)
    diff = float('inf')  # O(n^2) time, O(1) space
    nums.sort()  # O(nlogn), required by 2 pointers
    for i in range(len(nums)):  # O(n)
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:  # O(n)
            sum1 = nums[i] + nums[lo] + nums[hi]
            if abs(target - sum1) < abs(diff):
                diff = target - sum1
            if sum1 < target: lo += 1
            else: hi -= 1  # after while, hi = lo
        if diff == 0: break
    return target - diff
# https://leetcode.com/problems/3sum-closest/discuss/778177/Python3-%3A-Runtime%3A-52-ms-faster-than-99.77

# LC15. 3Sum - return all distinct tuples
def threeSum(self, nums):  # shorter and quicker 90%, O(n^2) in time, O(n) in space
    nums.sort()  # need to copy this, O(n)
    n, res = len(nums), []
    for i in range(n-2):
        if nums[i] > 0: break  # then all 3 are > 0 and sum > 0, so can't be 0
        if i > 0 and nums[i] == nums[i-1]: continue  # dupes
        l, r = i+1, n-1
        while l < r:  # need this loop after 1st result, -1, 0, 0, 0, 0, 1
            s = nums[i] + nums[l] + nums[r]
            if s < 0: l +=1
            elif s > 0: r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]: l += 1 # prevent dups
                while l < r and nums[r] == nums[r-1]: r -= 1
                l, r = l+1, r-1
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
def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
    pairs = collections.defaultdict(list)  # O(n^3) time and O(n^2) space
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            pairs[nums[i]+nums[j]].append((i, j))
    res = set()
    for s, ps in pairs.items():  # 3 loops here, so O(n^3)
        for a, b in ps:
            if target - s in pairs:
                for k,v in pairs[target-s]:
                    if a != k and a != v and b != k and b != v:
                        ans = sorted([nums[i] for i in [a,b,k,v]])
                        res.add(tuple(ans))
    return res
def fourSum(self, nums: List[int], target: int) -> List[List[int]]:  # O(n^(k-1)) k = 4 time, O(n^2) space
    def kSum(nums: List[int], target: int, k: int) -> List[List[int]]:
        res = []  # O(n^(k-1)) time, O(n^2) space in n recursive calls, each has O(n)
        if not nums: return res
        average_value = target // k  # works without this, but faster
        if average_value < nums[0] or nums[-1] < average_value: return res
        if k == 2: return twoSum(nums, target)
        for i in range(len(nums)):
            if i and nums[i - 1] == nums[i]: continue  # to avoid dupes
            for subset in kSum(nums[i + 1:], target - nums[i], k - 1):  # we have k-2 n-loops
                res.append([nums[i]] + subset)
        return res
    def twoSum(nums: List[int], target: int) -> List[List[int]]:  # O(n) time, O(1) space
        res = []  # solution for sorted array
        lo, hi = 0, len(nums) - 1
        while (lo < hi):
            curr_sum = nums[lo] + nums[hi]
            # to avoid dupes
            if curr_sum < target or (lo > 0 and nums[lo] == nums[lo - 1]): lo += 1
            elif curr_sum > target or (hi < len(nums) - 1 and nums[hi] == nums[hi + 1]):
                hi -= 1
            else:
                res.append([nums[lo], nums[hi]])
                lo, hi = lo + 1, hi - 1
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


















