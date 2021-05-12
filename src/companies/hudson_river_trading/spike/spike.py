from collections import Counter


def spike(nums: list) -> int:
    if not nums:
        return 0

    # we could run these in one loop, O(n)
    max_v = max(nums)
    counts = Counter(nums)

    ret = 1  # single peak
    for num, count in counts.items():  # O(n)
        if num != max_v:  # for every off peak value
            ret += min(2, count)  # either 2 sides of peak or 1 side if there is only 1.

    return ret


print(spike([]))  # 0
print(spike([1]))  # 1
print(spike([1, 2]))  # 2 -> 1, 2
print(spike([2, 1]))  # 2 -> 2, 1
print(spike([1, 1]))  # 1 -> 1
print(spike([1, 2, 3]))  # 3 -> 1, 2, 3
print(spike([2, 3, 1]))  # 3 -> 2, 3, 1
print(spike([2, 2, 3]))  # 3 -> 2, 3, 2
print(spike([2, 2, 2]))  # 1 -> 2
print(spike([1, 1, 2, 2, 3, 3]))  # 5 -> 1, 2, 3, 2, 1
