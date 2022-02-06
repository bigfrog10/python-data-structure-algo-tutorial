from heapq import *

def findMedian(arr):
    ret, smallh, largeh = [], [], []
    for x in arr:
        # add to leaps
        heappush(smallh, -heappushpop(largeh, x))
        if len(largeh) < len(smallh):
            heappush(largeh, -heappop(smallh))
        # get median
        if len(largeh) > len(smallh):
            ret.append(largeh[0])
        else:
            ret.append((largeh[0] - smallh[0]) // 2)

    return ret

print(findMedian([5, 15, 1, 3]))  # [5, 10, 5, 4]
print(findMedian([2, 4, 7, 1, 5, 3]))  # [2, 3, 4, 3, 4, 3]


