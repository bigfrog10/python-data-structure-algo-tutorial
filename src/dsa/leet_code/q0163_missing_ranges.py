################################################################################
# You are given an inclusive range [lower, upper] and a sorted unique integer
# array nums, where all elements are in the inclusive range.
# A number x is considered missing if x is in the range [lower, upper] and x is
# not in nums.
# Return the smallest sorted list of ranges that cover every missing number
# exactly. That is, no element of nums is in any of the ranges, and each missing
# number is in one of the ranges.
# EASY
################################################################################

def find_missing_ranges(nums: list, lower: int, upper: int) -> list:
    res = []

    if not nums:
        if lower == upper:
            res.append(str(lower))
            return res
        else:
            res.append(f'{lower}->{upper}')
            return res

    left, right = lower, upper
    for i in nums:
        if i - left == 1:
            res.append(str(left))
        elif i - left > 1:
            res.append(f'{left}->{i-1}')
        left = i+1

    i = nums[-1]
    if right - i == 1:
        res.append(str(right))
    elif right - i > 1:
        res.append(f'{i+1}->{right}')
    return res


print(find_missing_ranges([0,1,3,50,75], 0, 99))
print(find_missing_ranges([], 1, 1))
print(find_missing_ranges([], -3, -1))
print(find_missing_ranges([-1], -1, -1))
print(find_missing_ranges([-1], -2, -1))
