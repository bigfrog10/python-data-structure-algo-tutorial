
# LC1099. Two Sum Less Than K
def twoSumLessThanK(self, nums: List[int], k: int) -> int:
    nums.sort()  # O(nlogn)
    answer = -1
    for i in range(len(nums)):
        j = bisect_left(nums, k - nums[i], i + 1) - 1
        if j > i: answer = max(answer, nums[i] + nums[j])
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
