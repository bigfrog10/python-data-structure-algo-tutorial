# https://www.includehelp.com/algorithms/find-nearest-greatest-neighbours-of-each-element-in-an-array.aspx
# for each element in an array of integers, find its nearest larger neighbour.
# if 2 neighbours have same distance, pick larger neighbour.

def array_neighbours(arr):
    # trivial solution is for each element, search right and left until a larger element found.
    # this is O(n^2)
    # below is O(n) solution,
    ret = [None] * len(arr)
    stack = []
    for i, e in enumerate(arr):
        if i == 0: stack.append((i, e)) # push first element into stack
        else:  # we try to keep a decreasing stack
            while stack and e > stack[-1][1]:  # if it's not decreasing, we found the larger neighbour
                idx, v = stack.pop()
                ret[idx] = e
            stack.append((i, e))
    b = None
    for idx, v in stack:
        ret[idx] = b
        b = v
    return ret

def array_neighbours1(arr):
    stack, ret = [], []
    for a in arr:
        bigger = a
        if stack:
            e = stack.pop()
            while e < bigger:
                ret.append(e)
                if not stack: break
                e = stack.pop()
            if e > bigger:
                stack.append(e)
        stack.append(a)
    while stack:
        ret.append(stack.pop())

print(array_neighbours([2, 1, 3, 4, 5]))

print(array_neighbours([2, 1, 5, 4, 3]))

print(array_neighbours([5, 1, 3, 4, 2]))  # this is wrong
