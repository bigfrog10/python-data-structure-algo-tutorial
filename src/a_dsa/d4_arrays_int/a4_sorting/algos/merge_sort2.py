# https://stackoverflow.com/questions/19085450/merge-sort-in-place-for-python-cant-find-what-is-wrong
# https://github.com/Orangefish/algo/blob/master/sorting_and_search/sort_merge.py

# This is an O(1) space and O(nlogn) time implementation

def sort_imerge(Seq, l=0, u=None):  # upper and lower
    u = len(Seq) if u is None else u
    if  u - l > 1:
        m = l + (u - l) // 2
        w = l + u - m  # this is not exactly m, could be off by 1.
        wsort(Seq, l, m, w)
        while w - l > 2:
            n = w
            w = l + (n - l + 1) // 2
            wsort(Seq, w, n, l)
            wmerge(Seq, l, l + n - w, n, u, w)
        n = w
        while n > l:  # fallback to insert sort
            for m in range(n, u):
                if Seq[m-1] > Seq[m]:
                    Seq[m-1], Seq[m] = Seq[m], Seq[m-1]
            n -= 1

    return Seq


def wmerge(Seq, i, m, j, n, w):
    """Merge subarrays [i, m) and [j, n) into work area w.
    All indexes point into Seq.
    The space after w must be enough to fit both subarrays.
    """
    while i < m and j < n:
        if Seq[i] < Seq[j]:
            Seq[i], Seq[w] = Seq[w], Seq[i]
            i += 1
        else:
            Seq[j], Seq[w] = Seq[w], Seq[j]
            j += 1
        w += 1
    while i < m:
        Seq[i], Seq[w] = Seq[w], Seq[i]
        i += 1
        w += 1
    while j < n:
        Seq[j], Seq[w] = Seq[w], Seq[j]
        j += 1
        w += 1


def wsort(Seq, l, u, w):
    """
    Sort subarray [l, u) and put reuslt into work area w.
    All indexes point into Seq.
    """
    if  u - l > 1:
        m = l + (u - l) // 2
        sort_imerge(Seq, l, m)
        sort_imerge(Seq, m, u)
        wmerge(Seq, l, m, m, u, w)
    else:
        while l < u:
            Seq[l], Seq[w] = Seq[w], Seq[l]
            l += 1
            w += 1


print(sort_imerge([2, 3, 1]))
print(sort_imerge([5, 2, 4, 1]))
print(sort_imerge([5, 2, 4, 7, 1, 3, 2, 6]))

arr = list(range(2 ** 15))
import random
random.shuffle(arr)
assert(sort_imerge(arr) == list(range(2 ** 15)))

