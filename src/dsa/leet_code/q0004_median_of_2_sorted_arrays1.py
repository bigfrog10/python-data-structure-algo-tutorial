# https://zxi.mytechroad.com/blog/algorithms/binary-search/leetcode-4-median-of-two-sorted-arrays/
# https://www.algorithmsandme.com/find-kth-smallest-element-in-two-sorted-arrays/
# https://www.cnblogs.com/grandyang/p/4465932.html
#
# for a size n sorted list A, indexed 0, ..., n-1, the median is
#     A[n / 2], if n is odd
#     (A[(n-1) / 2] + A[n / 2]) / 2, if n is even
# A uniform formula is just the second one:
#     (A[(n-1) / 2] + A[n / 2]) / 2
# If n is odd, these 2 elements are the same, so addition and division are
# cancelled each other.
#
# Note that on the left side of (n - 1) / 2, there are half of list, excluding
# median elements (1 or 2). We will use this fact in the following.
#
# Now if we have 2 sorted lists, size n1 and n2
#     A0, A1, ..., A(i-1), Ai, ..., An
#     B0, B1, ..., B(j-1), Bj, ..., Bm
# after merging,
#     C0, C1, ..., C(k-1), Ck, ..., C(n + m - 1)
# where k = i + j. If k-1 is the index such that half of C's on the left
# (excluding median elements). Then
#     k - 1 = (n + m - 1) / 2 => k = (n + m + 1) / 2
#
# Here is the trick: if we can find i and j such that
#     Ai >= B(j-1) and Bj >= A(i-1)
# then there are i elements on the left of Ai on A's and j elements on the left
# of Bj on B's. Then after the merging, max(Ai, Bj) will be the kth element in
# C's, where k = i + j.
# So we are going to run binary search on the index of shorter list to find
# i and j to satisfy the above.
# If we can't find them, it means one of the following cases:
#     Ai < B(j-1): then we need to increase i. If we can't increase anymore,
#         it means all A's are smaller than B(j-1), then B(j-1) is the answer.
#     Bj < A(i-1): then we need to decrease i. If we can't decrease anymore,
#         it means all A's are larger than Bj, then Bj is the answer.
# We just need to search on the first condition.
import sys


def find_median_sorted(num_list1: list, num_list2: list) -> float:
    len1, len2 = len(num_list1), len(num_list2)
    if len1 > len2:
        return find_median_sorted(num_list2, num_list1)
    # so now we assume len1 <= len2

    k = (len1 + len2 + 1) // 2
    left, right = 0, len1  # binary search
    while left < right:
        i = left + (right - left) // 2
        j = k - i
        if num_list1[i] < num_list2[j - 1]:  # binary search constraint
            left = i + 1
        else:
            right = i

    i, j = left, k - left
    ai_1 = num_list1[i-1] if i > 0 else -sys.maxsize  # means using the other.
    bj_1 = num_list2[j-1] if j > 0 else -sys.maxsize
    ck_1 = max(ai_1, bj_1)

    if (len1 + len2) % 2 != 0:  # odd
        return ck_1

    ai = num_list1[i] if i < len1 else sys.maxsize
    bj = num_list2[j] if j < len2 else sys.maxsize
    ck = min(ai, bj)

    return (ck_1 + ck) / 2


print(find_median_sorted([1, 2, 3], [4, 5, 6, 7]))  # 4
print(find_median_sorted([7, 8, 9], [1, 2, 3, 4]))  # 4
print(find_median_sorted([1, 3, 5, 9], [4, 8, 11, 22]))  # 6.5

print(find_median_sorted([1, 2, 3], [4, 5, 6]))  # 3.5
print(find_median_sorted([7, 8, 9], [1, 2, 3]))  # 5
print(find_median_sorted([1, 3, 5, 9], [4, 8, 10, 11, 22]))  # 8

print(find_median_sorted([-37, 4, 7, 13, 29, 49, 50], [-24, -14, 32]))  # 10
print(find_median_sorted([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))  # 3.5
print(find_median_sorted([1, 2, 4, 7, 9], [3, 5, 6, 8]))  # 5
print(find_median_sorted([-50, -47, -36, -35, 0, 13, 14, 16], [-31, 1, 9, 23, 30, 39]))  # 5
print(find_median_sorted([2, 3, 4], [1]))  # 2.5
