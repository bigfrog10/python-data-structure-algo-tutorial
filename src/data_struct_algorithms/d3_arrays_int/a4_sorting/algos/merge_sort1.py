
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


# https://stackoverflow.com/questions/46158537/how-to-modify-merge-sort-to-be-in-place
# working fine, but we are kind of cheat since we copy sub arrays
# the c version does swap too. Now the question is whether these swaps affect
# runtime by too much. Seems this is indeed the case because this method is noticably
# slower than others.
def merge_sort_ip(alist):
    i = 1
    while i <= len(alist):
        for j in range(0, len(alist), i * 2):
            left, right = j, min(len(alist), j + 2 * i)
            mid = j + i
            p, q = left, mid
            while p < mid and q < right:
                if alist[p] < alist[q]:  # already sorted...
                    p += 1  # ... skip to next pair
                else:  # need to swap...
                    temp = alist[q]  # store temp value...
                    alist[p + 1: q + 1] = alist[p:q]  # shift to the right... thhis is kind of cheating, so it's slow.
                    alist[p] = temp  # update value
                    p, mid, q = p + 1, mid + 1, q + 1  # ... go to next pair
        i *= 2
    return alist


print(merge_sort([3, 2, 1]))
print(merge_sort([5, 2, 4, 1]))
print(merge_sort([5, 2, 4, 7, 1, 3, 2, 6]))

arr = list(range(2 ** 15))
import random
random.shuffle(arr)
assert(merge_sort(arr) == list(range(2 ** 15)))
