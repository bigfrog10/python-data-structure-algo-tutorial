from collections import Counter
from functools import lru_cache


def opt_weights(arr: list):
    counts = Counter(arr)
    n = len(arr)
    half = sum(arr) / 2

    weights = list(set(arr))
    weights.sort(reverse=True)
    res = []

    @lru_cache
    def knapsack(idx, weight, count, indices):
        if weight > half:
            return count, weight
        if idx >= n:
            return count, weight

        # we use idx
        w = weights[idx]
        c = counts[w]
        nw = weight + w * c
        nc = count + c
        if nw < half:
            nc, nw = knapsack(idx+1, nw, nc, indices + [idx])

        # we don't use this idx
        count1, weight1 = knapsack(idx+1, weight, count, indices)
        if weight1 > half:
            if nc < count1:
                return nc, nw
            elif nc == count1:
                if nc >= weight1:
                    return nc, nw
                else:
                    return count1, weight1
            else:
                return count1, weight1

    idxs = []
    c, w = knapsack(0, 0, 0, idxs)
    print(f'count={c}, weight={w}')



