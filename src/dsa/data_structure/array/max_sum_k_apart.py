

# return max sum of 2 elements in the array with indices apart from at least k.
# trivial solution takes O(n^2), this takes O(n), one for loop
# https://www.quora.com/Is-it-possible-to-find-maximum-sum-of-two-numbers-in-an-unsorted-array-such-that-the-numbers-are-not-adjacent-to-each-other-in-O-N
# DP is working too: https://www.geeksforgeeks.org/maximum-sum-subsequence-least-k-distant-elements/
def max_sum_k_apart(nums, k):
    if not nums:
        return 0

    n = len(nums)
    if n <= k:
        return 0

    emax = nums[0]
    smax = nums[0] + nums[k]
    for i in range(k, n):
        emax = max(emax, nums[i-k])  # track partner's max
        smax = max(smax, emax + nums[i])

    return smax


n = [1, 3, 4, 3, 1, 1, 1]
print(max_sum_k_apart(n, 2))
