# merge 3 sorted arrays without dupe

def merge(arr1, arr2, arr3):
    import math
    res = []
    i = j = k = 0
    while i < len(arr1) or j < len(arr2) or k < len(arr3):
        a = arr1[i] if i < len(arr1) else math.inf
        b = arr2[j] if j < len(arr2) else math.inf
        c = arr3[k] if k < len(arr3) else math.inf
        mn = min(a, b, c)
        # if not res or res[-1] != mn:
        res.append(mn)
        while i < len(arr1) and arr1[i] == mn: i += 1
        while j < len(arr2) and arr2[j] == mn: j += 1
        while k < len(arr3) and arr3[k] == mn: k += 1
    return res

print(merge([1, 3], [2, 4], [1, 5]))
print(merge([1, 3], [2, 3, 4], [1, 2, 3, 4, 5]))
