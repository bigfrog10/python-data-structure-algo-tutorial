import functools
import math


@functools.lru_cache(maxsize=None)
def min_square(n):

    u = math.floor(math.sqrt(n))  # don't use ceil because it breaks perfect square case
    res = n  # n = 1 + 1 + 1 ... which is the worst solution
    for i in range(1, u+1):
        res = min(res, min_square(n - i * i) + 1)

    return res


class Solution:
    def numSquares(self, n: int) -> int:
        res = min_square(n)
        print(min_square.cache_info())
        return res


print(Solution().numSquares(43))
