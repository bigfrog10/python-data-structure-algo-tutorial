from collections import Counter
from functools import lru_cache


# https://leetcode.com/discuss/interview-question/1021441/Amazon-OA-or-optimizating-Box-Weight
# we may ask with same counts, what's the max weight we could get
def opt_weights(arr: list):
    counts = Counter(arr)
    half = sum(arr) / 2

    weights = list(set(arr))
    weights.sort(reverse=True)
    n = len(weights)

    res = []
    cands = []
    minc = float('inf')
    maxw = float('-inf')

    @lru_cache(None)
    def knapsack(idx, weight, count):
        nonlocal minc, maxw
        if count > minc:
            return weight, count
        if count == minc:
            if weight < maxw:
                return weight, count
        if weight > half:
            cands.append((count, weight, res[:]))
            minc = min(minc, count)
            maxw = max(maxw, weight)
            return weight, count
        if idx >= n:
            return weight, count

        # we use idx
        w = weights[idx]
        c = counts[w]

        res.append(idx)
        nw, nc = knapsack(idx+1, weight + w * c, count + c)
        res.pop()

        # we don't use this idx
        weight1, count1 = knapsack(idx+1, weight, count)

        # now compute return result
        if nw > half and weight1 > half:  # if both ways exceed 1/2 of total weight
            if nc < count1:
                return nw, nc
            elif nc == count1:
                if nw >= weight1:
                    return nw, nc
                else:
                    return weight1, count1
            else:
                return weight1, count1
        elif nw > half:  # if only the first way exceeds
            return nw, nc
        elif weight1 > half:  # if only the 2nd way exceeds
            return weight1, count1
        else:  # if none of them exeeds, lean on weight so we could reach 1/2 as fast as we could
            if nw >= weight1:
                return nw, nc
            else:
                return weight1, count1

    wf, cf = knapsack(0, 0, 0)
    print(knapsack.cache_info())
    # print(f'count={cf}, weight={wf}')
    print(cands)

    minc, maxw, res = float('inf'), float('-inf'), None
    for c, w, i in cands:
        if c < minc:
            minc, maxw, res = c, w, i
        elif c == minc:
            if w > maxw:
                minc, maxw, res = c, w, i

    ret = []
    for i in res:
        ret.extend([weights[i]] * counts[weights[i]])
    # print(f'sol={ret}')
    return ret


print(opt_weights([2, 1, 1, 1]))  # [1, 1, 1]
print(opt_weights([1, 1, 1, 1]))  # [1, 1, 1, 1]
print(opt_weights([15, 20, 20, 20, 50]))  # [50, 15]
print(opt_weights([9, 20, 20, 20, 50]))  # [20, 20, 20]
print(opt_weights([4, 5, 4, 1]))  # [4, 4]
print(opt_weights([1, 2, 3, 4, 5]))  # [5, 4]
print(opt_weights([20, 10, 1, 1]))  # [20]
print(opt_weights([i for i in range(1, 100)]))
