# Given two arrays A and B of length N, determine if there is a way to make A equal
# to B by reversing any subarrays from array B any number of times.
# Signature
# bool areTheyEqual(int[] arr_a, int[] arr_b)
# Input
# All integers in array are in the range [0, 1,000,000,000].
# Output
# Return true if B can be made equal to A, return false otherwise.
# Example
# A = [1, 2, 3, 4]
# B = [1, 4, 3, 2]
# output = true
# After reversing the subarray of B from indices 1 to 3, array B will equal array A.

def are_they_equal(array_a, array_b):
  # Write your code here
  n = len(array_a)
  start = walk = 0
  diff = False
  while walk < n:
    if array_a[walk] == array_b[walk] and not diff:
      walk += 1
      start += 1
      continue

    diff = True
    if array_a[start] == array_b[walk] and array_a[walk] == array_b[start]:
      if array_a[start:walk+1] == array_b[start:walk+1][::-1]:
        walk += 1 # flip
        start = walk
        continue
    else:
      walk += 1

  return start == n

print(are_they_equal([1, 2, 3, 4], [1, 4, 3, 2]))
print(are_they_equal([1, 2, 3, 4], [1, 2, 3, 5]))
print(are_they_equal([1, 2, 3, 4], [1, 4, 3, 5]))

print(are_they_equal([1, 2, 3, 4, 5], [2, 1, 5, 4, 3]))

# https://www.geeksforgeeks.org/check-if-two-arrays-can-be-made-equal-by-reversing-any-subarray-once/
# https://www.1point3acres.com/bbs/interview/facebook-software-engineer-666815.html
