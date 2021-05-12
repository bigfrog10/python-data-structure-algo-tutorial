# Given an array of integers and another integer K, After we remove K continuous elements
# from the array, we can compute amplitude for the rest of the array as max - min.
# Find the min amplitude.

def cumu_pre(arr):
    n = len(arr)
    maxa, mina = [arr[0]] + [float('-inf')] * (n-1), [arr[0]] + [float('inf')] * (n - 1)
    for i in range(1, n):
        maxa[i] = max(arr[i], maxa[i-1])
        mina[i] = min(arr[i], mina[i-1])
    return maxa, mina

def cumu_post(arr):
    n = len(arr)
    maxa, mina = [float('-inf')] * (n-1) + [arr[n-1]], [float('inf')] * (n - 1) + [arr[n-1]]
    for i in range(n-2, -1, -1):
        maxa[i] = max(arr[i], maxa[i+1])
        mina[i] = min(arr[i], mina[i+1])
    return maxa, mina

def min_ampl(arr, k):
    if not arr: return
    n = len(arr)
    pre_cumu_max, pre_cumu_min = cumu_pre(arr)
    post_cumu_max, post_cumu_min = cumu_post(arr)
    ret = float('inf')
    for i in range(n-k+1):
        if i == 0:
            ret = min(ret, post_cumu_max[k] - post_cumu_min[k])
        elif i == n - k:
            ret = min(ret, pre_cumu_max[i] - pre_cumu_min[i])
        else:
            minv = min(pre_cumu_min[i-1], post_cumu_min[i+k])
            maxv = max(pre_cumu_max[i-1], post_cumu_max[i+k])
            ret = min(ret, maxv - minv)
    return ret


print(min_ampl([5, 3, 6, 1, 3], 2))
