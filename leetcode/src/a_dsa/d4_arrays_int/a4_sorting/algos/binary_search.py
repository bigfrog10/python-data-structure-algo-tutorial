# Facebook
# Given a sorted array of integers with duplicates, count the number of unique values.
# Example: [1, 1, 1, 1, 2, 2, 2, 2, 5, 5,  5,  7,  7,  8,  8, 10] -> 6
#           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
# Follow up 1: Space O(1)
# Follow up 2: Runtime O(logn) if the number of unique elements, K, is much less than the size of the array, N

# https://codereview.stackexchange.com/questions/114073/count-the-number-of-unique-elements-in-a-sorted-array

# O(n) in runtime and space, regardless the sorted nature
def count_uniq(arr):
    res = set()
    for i in arr: res.add(i)
    return len(res)


print(count_uniq([1, 2, 2, 3]))  # 3

def count_uniqs(arr):  # arr is sorted, O(n) in time and O(1) in space
    count = i = 0  # 2 pointers
    for j in range(1, len(arr)):
        if arr[i] != arr[j]:  # do this only when values change
            count += 1
            i = j
    return count + 1

print(count_uniqs([1, 1, 2, 2, 3, 3]))

def count_uniqsj(arr):  # if number of unique values, k, is << N.  O(klogn) time and O(1) space
    def jump_at(arr, idx):
        v = arr[idx]
        lo, hi = idx, len(arr)
        while lo < hi:  # find least index i s.t. arr[i] > v
            mid = (lo + hi) // 2
            if arr[mid] == v: lo = mid + 1
            else: hi = mid
        return lo
    count = i = 0  # 2 pointers
    while i < len(arr):
        i = jump_at(arr, i)
        count += 1
    return count

print(count_uniqsj([1, 1, 2, 2, 3, 3]))

import bisect
def count_uniqsjb(arr):
    count = i = 0
    while i < len(arr):
        i = bisect.bisect(arr, arr[i]) # bisect right so that new i is afer arr[i]
        count += 1
    return count

print(count_uniqsjb([1, 1, 1, 1, 2, 2, 2, 2, 5, 5,  5,  7,  7,  8,  8, 10]))
