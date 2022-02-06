# Given an integer N, and a permutation, P of the integers from 1 to N, denoted as
# (a_1, a_2, ..., a_N), rearrange the elements of the permutation into increasing order,
# repeatedly making the following operation:
# Select a sub-portion of the permutation, (a_i, ..., a_j), and reverse its order.
#
# Your goal is to compute the minimum number of such operations required to return the
# permutation to increasing order.
# Example
#
# If N = 3, and P = (3, 1, 2), we can do the following operations:
# Select (1, 2) and reverse it: P = (3, 2, 1).
# Select (3, 2, 1) and reverse it: P = (1, 2, 3).
# output = 2
#
# Since we are looking min, BFS is the way to go
from collections import deque

def minOperations(arr):
  # Write your code here
  n = len(arr)
  visited, queue = {tuple(arr)}, deque([arr])
  ret, target = 0, sorted(arr)
  while queue: # BFS
    for _ in range(len(queue)):
      curr = queue.popleft()
      if curr == target:
          return ret

      for left in range(n):
        for right in range(left+1, n):
          nc = list(curr)
          nc[left:right+1] = nc[left:right+1][::-1]
          nct = tuple(nc)
          if nct not in visited:
            queue.append(nc)
            visited.add(nct)
    ret += 1
  return -1

print(minOperations([3, 1, 2]))
