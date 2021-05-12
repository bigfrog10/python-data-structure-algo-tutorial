# Given an array of integers (which may include repeated integers), determine if there's a way to split the array into two subsequences A and B such that the sum of the integers in both arrays is the same, and all of the integers in A are strictly smaller than all of the integers in B.
# Note: Strictly smaller denotes that every integer in A must be less than, and not equal to, every integer in B.
# Signature
# bool balancedSplitExists(int[] arr)
# Input
# All integers in array are in the range [0, 1,000,000,000].
# Output
# Return true if such a split is possible, and false otherwise.
# Example 1
# arr = [1, 5, 7, 1]
# output = true
# We can split the array into A = [1, 1, 5] and B = [7].
# Example 2
# arr = [12, 7, 6, 7, 6]
# output = false
# We can't split the array into A = [6, 6, 7] and B = [7, 12] since this doesn't satisfy the requirement that all integers in A are smaller than all integers in B.

# O(n): https://leetcode.com/discuss/interview-question/718692/facebook-training-balanced-split
def balancedSplitExists(arr):
  # Write your code here
  summ = sum(arr)
  if summ % 2 != 0:
    return False
  arr1 = sorted(arr)

  cumu = list(accumulate(arr1))
  n = len(cumu)
  t = summ // 2
  idx = bisect.bisect(cumu, t) # from right
  #print('idx=' + str(idx))
  #print('sum=' + str(cumu[idx-1]))
  if cumu[idx-1] != t:
    return False
  # if idx+1 < n and arr1[idx] == arr1[idx+1]:
  #  return False
  if idx > 0 and arr1[idx] == arr1[idx-1]:
    return False
  return True
