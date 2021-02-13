################################################################################
# Given an array of integers nums and an integer target, return indices of the
# two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may
# not use the same element twice.
# You can return the answer in any order.
# EASY
################################################################################

from collections import Counter


def find_sum_pair(nums: list, target: int):
    bag = Counter(nums)  # element -> count, to handle duplicated values
    idx_map = {num: idx for idx, num in enumerate(nums)}

    for idx, num in enumerate(nums):
        other = target - num
        if other in bag:
            if (other == num and bag[other] > 1) or (other != num and bag[other] > 0):
                return idx, idx_map[target - num]

    return None


print(find_sum_pair([2, 7, 11, 15], 9))  # (0, 1)
print(find_sum_pair([3, 2, 4], 6))  # (1, 2)
print(find_sum_pair([3, 3], 6))  # (0, 1)


# To extends a little bit: Given an array of integers, print out pairs of
# elements such that the sum of the pair is the given target.
#
# For example, given target 8 and the array [1, 1, 6, 7, 7, 2, 5, 4], we should
# should print out
# (1, 7), (1, 7), (6, 2)
