################################################################################
# Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one
# sorted array.
# Note:
#     The number of elements initialized in nums1 and nums2 are m and n
#     respectively.
#     You may assume that nums1 has enough space (size that is equal to m + n)
#     to hold additional elements from nums2.
# EASY
################################################################################

def merge_array(nums1: list, m: int, nums2: list, n: int) -> None:
    # m = len(nums1)
    # n = len(nums2)
    # for i in range(n):
    #     nums1.append(0)  # just to simulate enough space.

    i, j, k = m-1, n-1, m+n-1  # start from backward
    while i > -1 and j > -1:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1

    if i == -1:
        while j > -1:
            nums1[j] = nums2[j]
            j -= 1


a = [0]
merge_array(a, 0, [1], 1)
print(a)

a = [1,2,3,0,0,0]
merge_array(a, 3, [2,5,6], 3)
print(a)
