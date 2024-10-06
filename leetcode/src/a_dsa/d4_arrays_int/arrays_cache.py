
# LC1590. Make Sum Divisible by P sum div by p sum div p div sum p div sum by p
def minSubarray(self, nums: List[int], p: int) -> int:
    totalSum = sum(nums)
    target = totalSum % p
    if target == 0: return 0
    cache = {0: -1}
    cumu = 0
    res = len(nums)
    for i, num in enumerate(nums):
        cumu += num
        rem = cumu % p
        diff = (rem - target) % p
        if diff in cache:
            res = min(res, i - cache[diff])
        cache[rem] = i
    return res if res < len(nums) else -1

