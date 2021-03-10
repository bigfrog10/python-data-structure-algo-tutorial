from collections import Counter
from functools import lru_cache
# There is one condition not quite clear
# The intersection of A and B is null: can same weight different items be counted as same or different.


def opt_weights(arr: list):
    num_count = Counter(arr)
    nums = sorted(num_count.keys(), reverse=True)
    n = len(nums)
    target = sum(arr) / 2
    ans = [[0] * (len(arr) + 1), 0]

    @lru_cache(None)
    def dfs(pos, weight, cur):
        nonlocal ans
        if weight > target and len(cur) <= len(ans[0]):
            if len(cur) < len(ans[0]) or weight > ans[1]:
                ans[0] = cur
                ans[1] = weight
            return
        if pos >= n:
            return

        for i in range(pos, len(nums)):
            num = nums[i]
            count = num_count[num]
            dfs(i + 1, weight + num * count, cur + [num] * count)

    dfs(0, 0, [])
    print(dfs.cache_info())
    return ans[0]


print(opt_weights([2, 1, 1, 1]))  # [1, 1, 1]
print(opt_weights([1, 1, 1, 1]))  # [1, 1, 1, 1]
print(opt_weights([15, 20, 20, 20, 50]))  # [50, 15]
print(opt_weights([9, 20, 20, 20, 50]))  # [20, 20, 20]
print(opt_weights([4, 5, 4, 1]))  # [4, 4]
print(opt_weights([1, 2, 3, 4, 5]))  # [5, 4]
print(opt_weights([20, 10, 1, 1]))  # [20]
print(opt_weights([i for i in range(1, 100)]))
