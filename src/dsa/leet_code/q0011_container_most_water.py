################################################################################
# Given n non-negative integers a1, a2, ..., an , where each represents a point
# at coordinate (i, ai). n vertical lines are drawn such that the two endpoints
# of the line i is at (i, ai) and (i, 0). Find two lines, which, together with
# the x-axis forms a container, such that the container contains the most water.
#
# Notice that you may not slant the container.
# MEDIUM
################################################################################

# O(n^2)
def max_area(num_list: list, last_idx: int) -> (float, int, int):
    if last_idx == 1:
        return min(num_list[0], num_list[1]), 0, 1

    res, x, y = max_area(num_list, last_idx - 1)

    for i in range(last_idx):
        v = min(num_list[i], num_list[last_idx]) * (last_idx - i)
        if v > res:
            res, x, y = v, i, last_idx

    return res, x, y


print(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7], 8))


# O(n)
def max_area1(heights: list) -> float:
    res, b, e = 0, 0, len(heights) - 1
    while b < e:
        res = max(res, min(heights[b], heights[e]) * (e - b))
        if heights[b] < heights[e]:
            b += 1
        else:
            e -= 1
    
    return res


print(max_area1([1, 8, 6, 2, 5, 4, 8, 3, 7]))
