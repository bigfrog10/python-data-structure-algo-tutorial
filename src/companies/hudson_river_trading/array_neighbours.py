# https://www.includehelp.com/algorithms/find-nearest-greatest-neighbours-of-each-element-in-an-array.aspx
# for each element in an array of integers, find its nearest larger neighbour.
# if 2 neighbours have same distance, pick larger neighbour.

def array_neighbours(arr):
    # trivial solution is for each element, search right and left until a larger element found.
    # this is O(n^2)
    # below is O(n) solution,
    ret = []
    stack = []
    for i, e in enumerate(arr):
        if i == 0:  # push first element into stack
            stack.append(e)
        else:  # we try to keep a decreasing stack
            while stack and e > stack[-1]:  # if it's not decreasing, we found the larger neighbour
                stack.pop()
                ret.append(e)
            stack.append(e)

    if stack:  # values are decreasing now.
        ret.append(None)  # The top element in stack has no neighbour since it's the largest
        ret.extend(stack[:-1])  # the rest has neighbour to its left

    return ret


print(array_neighbours([2, 1, 3, 4, 5]))

print(array_neighbours([2, 1, 5, 4, 3]))
