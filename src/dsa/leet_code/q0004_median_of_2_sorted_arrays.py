################################################################################
# There are two sorted arrays nums1 and nums2 of size m and n respectively. Find
# the median of the two sorted arrays. The overall run time complexity should be
# O(log (m+n)). You may assume nums1 and nums2 cannot be both empty.
# Example 1: nums1 = [1, 3], nums2 = [2], the median is 2.0
# Example 2: nums1 = [1, 2], nums2 = [3, 4], the median is (2 + 3)/2 = 2.5
# HARD
################################################################################

# http://www.drdobbs.com/parallel/finding-the-median-of-two-sorted-arrays/240169222
# https://zxi.mytechroad.com/blog/algorithms/binary-search/leetcode-4-median-of-two-sorted-arrays/

# If we merge these 2 arrays, it takes O(m + n).
# O(log) indicates we have to use binary search somehow.
def find_median_sorted(num_list1: list, num_list2: list) -> float:
    len1 = len(num_list1)
    len2 = len(num_list2)
    mid = (len1 + len2) // 2

    if (len1 + len2) % 2 == 0:  # even
        mid1 = find_kth_smallest(num_list1, num_list2, mid, 0, len1-1, 0, len2-1)
        mid2 = find_kth_smallest(num_list1, num_list2, mid - 1, 0, len1-1, 0, len2-1)
        return (mid1 + mid2) / 2
    else:  # odd
        return find_kth_smallest(num_list1, num_list2, mid, 0, len1-1, 0, len2-1)


def find_kth_smallest(num_list1: list, num_list2: list, k: int,
                      start1: int, end1: int, start2: int, end2: int) -> float:
    len1 = end1 - start1 + 1  # inclusive
    len2 = end2 - start2 + 1

    if len1 <= 0:
        return num_list2[start2 + k]
    if len2 <= 0:
        return num_list1[start1 + k]
    if k == 0:  # do this after len check, since the index may be out of range by now.
        return num_list1[start1] if num_list1[start1] < num_list2[start2] else num_list2[start2]

    # try to eliminate impossibles to narrow down the scope
    kth1 = k * len1 // (len1 + len2)  #
    kth2 = k - kth1 - 1  # kth1 + kth2 = k - 1

    idx1 = start1 + kth1
    idx2 = start2 + kth2
    if num_list1[idx1] > num_list2[idx2]:
        k = k - kth2 - 1
        end1 = idx1
        start2 = idx2 + 1
    else:
        k = k - kth1 - 1
        end2 = idx2
        start1 = idx1 + 1

    return find_kth_smallest(num_list1, num_list2, k, start1, end1, start2, end2)


print(find_median_sorted([1, 3, 5, 7], []))  # (3 + 5) / 2 = 4
print(find_median_sorted([], [1, 3, 5, 7]))  # (3 + 5) / 2 = 4
print(find_median_sorted([1, 3, 5, 7, 9], []))  # 5
print(find_median_sorted([], [1, 3, 5, 7, 9]))  # 5

print(find_kth_smallest([1, 2, 3], [4, 5, 6], 0, 0, 2, 0, 2))  # 1, 2, 3, 4, 5, 6
print(find_kth_smallest([1, 2, 3], [4, 5, 6], 1, 0, 2, 0, 2))
print(find_kth_smallest([1, 2, 3], [4, 5, 6], 2, 0, 2, 0, 2))
print(find_kth_smallest([1, 2, 3], [4, 5, 6], 3, 0, 2, 0, 2))
print(find_kth_smallest([1, 2, 3], [4, 5, 6], 4, 0, 2, 0, 2))
print(find_kth_smallest([1, 2, 3], [4, 5, 6], 5, 0, 2, 0, 2))

print(find_median_sorted([1, 2, 3], [4, 5, 6]))  # (3 + 4) / 2 = 3.5
print(find_median_sorted([4, 5, 6], [7, 8, 9]))  # (6 + 7) / 2 = 6.5
print(find_median_sorted([1, 2, 3, 4], [5, 6, 7, 8]))  # 4.5
print(find_median_sorted([5, 6, 7, 8], [9, 10, 11, 12]))  # 8.5

print(find_median_sorted([-37, 4, 7, 13, 29, 49, 50], [-24, -14, 32]))  # 10
print(find_median_sorted([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))  # 3.5
print(find_median_sorted([1, 2, 4, 7, 9], [3, 5, 6, 8]))  # 5
print(find_median_sorted([-50, -47, -36, -35, 0, 13, 14, 16], [-31, 1, 9, 23, 30, 39]))  # 5
print(find_median_sorted([2, 3, 4], [1]))  # 2.5
