################################################################################
# Given a sorted array nums, remove the duplicates in-place such that each
# element appears only once and returns the new length.
#
# Do not allocate extra space for another array, you must do this by modifying
# the input array in-place with O(1) extra memory.
# EASY
################################################################################

def remove_dups(nums: list) -> int:
    temp = None
    for idx in range(len(nums) - 1, -1, -1):  # to loop from n-1, n-2, ..., 0
        if temp is not None:
            if nums[idx] == temp:
                del nums[idx]
            else:
                temp = nums[idx]
        else:
            temp = nums[idx]

    print(nums)
    return len(nums)


print(remove_dups([1, 1, 2, 2, 3, 3]))
print(remove_dups([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
