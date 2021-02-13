################################################################################
# Given a non-empty array of integers nums, every element appears twice except
# for one. Find that single one.
#
# Follow up: Could you implement a solution with a linear runtime complexity and
# without using extra memory?
# EASY
################################################################################
# x ^ x = 0
# x ^ 0 =x
# x ^ y = y ^ x
#     => x ^ y ^ x = x ^ x ^ y = 0 ^ y = y, so x is eliminated.
def find_single(nums: list) -> int:
    res = 0
    for i in nums:
        res ^= i

    return res
