# Suppose we have a list of N numbers, and repeat the following operation until we're left with only a single number: Choose any two numbers and replace them with their sum. Moreover, we associate a penalty with each operation equal to the value of the new number, and call the penalty for the entire list as the sum of the penalties of each operation.
# For example, given the list [1, 2, 3, 4, 5], we could choose 2 and 3 for the first operation, which would transform the list into [1, 5, 4, 5] and incur a penalty of 5. The goal in this problem is to find the worst possible penalty for a given input.
# Signature:
# int getTotalTime(int[] arr)
# Input:
# An array arr containing N integers, denoting the numbers in the list.
# Output format:
# An int representing the worst possible total penalty.
# Constraints:
# 1 ≤ N ≤ 10^6
# 1 ≤ Ai ≤ 10^7, where *Ai denotes the ith initial element of an array.
# The sum of values of N over all test cases will not exceed 5 * 10^6.
# Example
# arr = [4, 2, 1, 3]
# output = 26
# First, add 4 + 3 for a penalty of 7. Now the array is [7, 2, 1]
# Add 7 + 2 for a penalty of 9. Now the array is [9, 1]
# Add 9 + 1 for a penalty of 10. The penalties sum to 26.
# if adjcent required, see LC1000
def getTotalTime(arr):
  # Write your code here
  arr1 = sorted(arr, reverse=True)
  tmp = arr1[0]
  total = 0
  for i in range(1, len(arr1)):
    tmp += arr1[i]
    total += tmp
  return total

from heapq import *
def getTotalTime(arr):
  total = 0
  while len(arr) > 1:
    idx, v = 0, float('-inf')
    for i in range(len(arr)-1):
      summ = arr[i] + arr[i+1]
      if summ > v:
        idx = i
        v = summ
    total += v
    arr = arr[:idx] + [v] + arr[idx+2:]
  return total

print(getTotalTime([2, 3, 9, 8, 4]))
print(getTotalTime([4, 2, 1, 3]))
