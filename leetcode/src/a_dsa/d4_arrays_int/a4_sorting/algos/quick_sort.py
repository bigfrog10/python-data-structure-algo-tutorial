# https://stackabuse.com/quicksort-in-python/
# divide and conquer, in place, unstable

def quick_sort_rec(arr: list): # not in place
    if not arr or len(arr) < 2: return arr
    n = len(arr)
    idx = 0 # index used to partition array
    for i in range(1, n):
        if arr[i] <= arr[0]:
            idx += 1
            arr[i], arr[idx] = arr[idx], arr[i]
    arr[0], arr[idx] = arr[idx], arr[0]
    left = quick_sort_rec(arr[:idx])
    right = quick_sort_rec(arr[idx+1:])
    return left + [arr[idx]] + right

# ary = [4, 2, 7, 3, 1, 6]
# print(quick_sort_rec(ary))

def quick_sort_rec(arr: list): # in place sorting  quicksort
    def partition(nums, left, right):  # bisect right
        target = nums[right]  # we may randomize this target
        i = left
        for j in range(left, right):
            if nums[j] <= target:
                if i != j: nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i # smaller on left, larger on right
    def sort(nums, left, right):
        if left >= right: return
        idx = partition(nums, left, right)
        sort(nums, left, idx-1)   # recursion is subject to limit
        sort(nums, idx+1, right)

    n = len(arr)
    return sort(arr, 0, n-1)

# ary = [4, 2, 7, 3, 1, 6]
# quick_sort_rec(ary)
# print(ary)

def quick_sort_iter(arr: list): # in place, iterative. scalable
    def partition(nums, left, right):  # bisect right
        target = nums[right]  # we may randomize this target
        i = left
        for j in range(left, right):
            if nums[j] <= target:
                if i != j: nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i # smaller on left, larger on right
    stack = [(0, len(arr)-1)]
    while stack:
        left, right = stack.pop()
        pidx = partition(arr, left, right)
        if pidx - 1 > left: stack.append((left, pidx-1))
        if pidx + 1 < right: stack.append((pidx+1, right))
    return arr

ary = [4, 2, 7, 3, 1, 6]
print(quick_sort_rec(ary))

import random
big_num = 2**15
l = list(range(big_num))
random.shuffle(l)
assert(quick_sort_iter(l) == list(range(big_num)))
print("It sorts!")

