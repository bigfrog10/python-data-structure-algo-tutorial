

def merge_sort_in_place(alist): # best solution
    """Inplace merge sort of array without recursive. The basic idea
    is to avoid the recursive call while using iterative solution.
    The algorithm first merge chunk of length of 2, then merge chunks
    of length 4, then 8, 16, .... , until 2^k where 2^k is large than
    the length of the array
    """
    i = 1
    while i <= len(alist):
        j = 0
        for j in range(0, len(alist), i * 2):
            left, right = j, min(len(alist), j + 2 * i)
            mid = j + i
            p, q = left, mid
            while p < mid and q < right:
                if alist[p] <= alist[q]:  # already sorted, = for stable
                    p += 1  # ... skip to next pair
                else: # need to swap...
                    temp = alist[q] # store temp value...
                    alist[p + 1: q + 1] = alist[p:q] # ... shift to the right...
                    alist[p] = temp # update value
                    p, mid, q = p + 1, mid + 1, q + 1  # ... go to next pair
        i *= 2
    return alist

# print(merge_sort_in_place([12, 11, 13, 5, 6, 7]))
# import random
#
# big_num = 2**15
# l = list(range(big_num))
# random.shuffle(l)
#
# assert(merge_sort_in_place(l) == list(range(big_num)))
# print("It sorts!")

###########################################################################
def merge_sort(array: list):
    # It turns out in place sort is a big deal, so we skip that
    # https://stackoverflow.com/questions/2571049/how-to-sort-in-place-using-the-merge-sort-algorithm/15657134#15657134
    # so we use O(n) space
    # https://stackoverflow.com/questions/18761766/mergesort-with-python
    if not array or len(array) == 1:
        return array
    n = len(array)

    def _merge(left, mid, right):
        la, ra = array[left:mid], array[mid:right]
        i = j = k = 0  # left, right, and original counters
        while i < len(la) and j < len(ra):
            if la[i] <= ra[j]:  # otherwise, no action, so this is stable sort
                array[left + k] = la[i]
                i += 1
            else:
                array[left + k] = ra[j]
                j += 1
            k += 1
        while i < len(la):
            array[left + k] = la[i]
            i += 1
            k += 1
        while j < len(ra):
            array[mid+j] = ra[j]
            j += 1
            k += 1

    def _sort(left, right):
        if left + 1 < right:
            mid = left + (right - left) // 2
            _sort(mid, right)
            _sort(left, mid)
            _merge(left, mid, right)
        else:
            if array[left] > array[right-1]:
                array[left], array[right-1] = array[right-1], array[left]
    _sort(0, n)
    return array

print(merge_sort([12, 11, 13, 5, 6, 7]))
import random

big_num = 2**15
l = list(range(big_num))
random.shuffle(l)

assert(merge_sort(l) == list(range(big_num)))
print("It sorts!")

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
print(mergesort([12, 11, 13, 5, 6, 7]))

# heap sort use space O(1), time O(nlogn)
# but not stable.
# new algorithms are wikisort and grailsort
# they are block sort based: https://en.wikipedia.org/wiki/Block_sort
# converge to O(n) for near sorted lists.
# https://www.programiz.com/dsa/heap-sort
