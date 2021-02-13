################################################################################
# Given an integer array nums, find the contiguous subarray (containing at least
# one number) which has the largest sum and return its sum.
# Follow up: If you have figured out the O(n) solution, try coding another
# solution using the divide and conquer approach, which is more subtle.
# EASY
################################################################################
# This is greedy: local max = global max
def max_sub_array(nums: list) -> int:
    total = max_total = nums[0]
    for i in range(1, len(nums)):
        total += nums[i]
        # if the total is not worth to keep, start a new total
        # we can also add code to keep track the start index.
        total = max(total, nums[i])
        max_total = max(max_total, total)  # this is our goal.

    return max_total


print(max_sub_array([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
