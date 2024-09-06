
# Merge Sort
def merge(a,b):
    c = []
    while len(a) != 0 and len(b) != 0:
        if a[0] < b[0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])
    if len(a) == 0: c += b
    else: c += a
    return c
def mergesort(x):
    if len(x) == 0 or len(x) == 1: return x
    else:
        middle = len(x)//2
        a = mergesort(x[:middle])
        b = mergesort(x[middle:])
        return merge(a,b)

def merge_sort_rec(arr: list): # in place recursive
    if len(arr) <= 1: return
    mid = len(arr) // 2 # O(nlogn)
    left, right = arr[:mid], arr[mid:]
    merge_sort_rec(left)
    merge_sort_rec(right)

    i = j = k = 0 # merge
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
    return arr

# ary =  [12, 11, 13, 5, 6, 7]
# print(merge_sort_rec(ary))

def merge_sort_iter(arr: list):
    pass

ary =  [12, 11, 13, 5, 6, 7]
print(merge_sort_iter(ary))

import random
big_num = 2**15
l = list(range(big_num))
random.shuffle(l)
assert(merge_sort_iter(l) == list(range(big_num)))
print("It sorts!")
