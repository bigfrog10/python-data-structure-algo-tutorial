# You are given an array arr of N integers. For each index i, you are required to
# determine the number of contiguous subarrays that fulfill the following conditions:
#
#     The value at index i must be the maximum element in the contiguous subarrays, and
#     These contiguous subarrays must either start from or end on index i.
#
# Signature
# int[] countSubarrays(int[] arr)
# Input
#
#     Array arr is a non-empty list of unique integers that range between 1 to 1,000,000,000
#     Size N is between 1 and 1,000,000
#
# Output
# An array where each index i contains an integer denoting the maximum number of contiguous subarrays of arr[i]
# Example:
# arr = [3, 4, 1, 6, 2]
# output = [1, 3, 1, 5, 1]
# Explanation:
#
#     For index 0 - [3] is the only contiguous subarray that starts (or ends) with 3, and the maximum value in this subarray is 3.
#     For index 1 - [4], [3, 4], [4, 1]
#     For index 2 - [1]
#     For index 3 - [6], [6, 2], [1, 6], [4, 1, 6], [3, 4, 1, 6]
#     For index 4 - [2]
#
# So, the answer for the above input is [1, 3, 1, 5, 1]
def count_subarrays(arr): # O(n) square
    # Write your code here
    n = len(arr)
    ret = [0] * n
    for i in range(n):  # O(n^2)
        j = 1
        while i - j >= 0:
            if arr[i - j] > arr[i]:
                break
            j += 1
        ret[i] += 1 + j - 1  # 1 for i itself and j-1 for left neighbours

        j = 1
        while i + j < n:
            if arr[i + j] > arr[i]:
                break
            j += 1
        ret[i] += j - 1

    return ret

def count_subarrays(arr):
  # Write your code here
  n = len(arr)
  ret = [0] * n
  for i in range(n):  # O(n^2)
    j = 1
    while i - j >= 0: # this is O(n^2)
      if arr[i-j] > arr[i]:
        break
      j += 1
    ret[i] += 1 + j - 1  # 1 for i, j-1 for left neighbours

    j = 1
    while i + j < n:
      if arr[i+j] > arr[i]:
        break
      j += 1
    ret[i] += j - 1

  return ret

# print(count_subarrays([3, 4, 1, 6, 2]))  # [1, 3, 1, 5, 1]
# print(count_subarrays([2, 4, 7, 1, 5, 3])) # [1, 2, 6, 1, 3, 1]

# https://stackoverflow.com/questions/31918666/number-of-contiguous-subarrays-in-which-element-of-array-is-max
# O(n) using Stack. Basic idea is to move from left to right keeping track of sub arrays
# ending at "i" and then right to left keeping track of sub arrays starting from "i":
def count_subarrays1(arr):
    stack = []
    res = [0] * len(arr)
    for i in range(len(arr)):
      while stack and arr[i] > arr[stack[-1]]:
        res[i] += res[stack.pop()]
      stack.append(i)
      res[i] += 1

    stack = []
    tmp = [0] * len(arr)
    for i in range(len(arr))[::-1]:
      while stack and arr[i] > arr[stack[-1]]:
        idx = stack.pop()
        res[i] += tmp[idx]
        tmp[i] += tmp[idx]
      stack.append(i)
      tmp[i] += 1

    return res

# print(count_subarrays1([3, 4, 1, 6, 2]))  # [1, 3, 1, 5, 1]
# print(count_subarrays1([2, 4, 7, 1, 5, 3])) # [1, 2, 6, 1, 3, 1]

# basically we build decreasing stack. The higher values help us forget cases before it.
def count_subarrays1(arr):
    n = len(arr)
    stack = [] # maintain indices for decreasing values
    res1 = [0] * n  # number of values on left less than current value
    for i in range(n):
        while stack and arr[i] > arr[stack[-1]]:
            res1[i] += res1[stack.pop()] # add previous counts for lower values
            res1[i] += 1 # this is corresponding to previous value
        stack.append(i)
    print(res1) # we leave with empty or decreasing stack.

    stack = []  # this time is for right side of current, we use
    res2 = [0] * n  # number of values on right less than current value
    for i in range(n)[::-1]:
        while stack and arr[i] > arr[stack[-1]]:
            res2[i] += res2[stack.pop()] # add previous counts for lower values
            res2[i] += 1 # this is corresponding to previous value
        stack.append(i)
    print(res2)  # we leave with empty or decreasing stack.

    res = [r1 + r2 + 1 for r1, r2 in zip(res1, res2)]
    return res

print(count_subarrays1([3, 4, 1, 6, 2]))  # [1, 3, 1, 5, 1]
print(count_subarrays1([2, 4, 7, 1, 5, 3])) # [1, 2, 6, 1, 3, 1]
