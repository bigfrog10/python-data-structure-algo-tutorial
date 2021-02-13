################################################################################
# Given an array of size n, find the majority element. The majority element is
# the element that appears more than ⌊ n/2 ⌋ times.
# You may assume that the array is non-empty and the majority element always
# exist in the array.
# EASY
################################################################################
import collections


def find_majority(nums):
    counts = collections.Counter(nums)
    return max(counts.keys(), key=counts.get)


print(find_majority([2,2,1,1,1,2,2]))
