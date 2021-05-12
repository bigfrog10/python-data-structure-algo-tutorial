# Given a sequence of n integers arr, determine the lexicographically smallest sequence which may be obtained from it after performing at most k element swaps, each involving a pair of consecutive elements in the sequence.
# Note: A list x is lexicographically smaller than a different equal-length list y if and only if, for the earliest index at which the two lists differ, x's element at that index is smaller than y's element at that index.
# Signature
# int[] findMinArray(int[] arr, int k)
# Input
# n is in the range [1, 1000].
# Each element of arr is in the range [1, 1,000,000].
# k is in the range [1, 1000].
# Output
# Return an array of n integers output, the lexicographically smallest sequence achievable after at most k swaps.
# Example 1
# n = 3
# k = 2
# arr = [5, 3, 1]
# output = [1, 5, 3]
# We can swap the 2nd and 3rd elements, followed by the 1st and 2nd elements, to end up with the sequence [1, 5, 3]. This is the lexicographically smallest sequence achievable after at most 2 swaps.
# Example 2
# n = 5
# k = 3
# arr = [8, 9, 11, 2, 1]
# output = [2, 8, 9, 11, 1]
# We can swap [11, 2], followed by [9, 2], then [8, 2].
# https://leetcode.com/discuss/interview-question/848430/element-swapping-facebook-coding-practice-2020
def findMinArray(arr, k):
  # Write your code here
  ret = []
  while k > 0 and arr:
    mi = 0 # 2nd swapping index
    for i in range(1, min(k+1, len(arr))):
      if arr[i] < arr[mi]:
        mi = i
    k -= mi # swap mi times to move mi to front
    ret.append(arr[mi]) # move to front
    arr = arr[0:mi] + arr[mi+1:]
  return ret + arr

print(findMinArray([5, 3, 1], 2))
