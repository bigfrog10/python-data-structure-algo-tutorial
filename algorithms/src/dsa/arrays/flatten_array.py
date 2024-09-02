
def flatten(arr: list, n):
    if n == 0: return arr
    res = []
    for a in arr:
        if isinstance(a, list):
            b = flatten(a, n-1)
            res.extend(b)
        else:
            res.append(a)
    return res

m = [1, 2, 3, [4, 5, 6], [7, 8, [9, 10, 11], 12], [13, 14, 15]]
print(flatten(m, 0))
print(flatten(m, 1))
print(flatten(m, 2))