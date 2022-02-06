from typing import List
# 2020-10-09 2 arrays in sorted order, no duplicates. need find the max continuous length AP
# [arithmetic prog] in the first array by adding 0 or more elements from the second array.
# Example: A= [4,8,13], B=[0,9,12] ANSWER = [0,4,8,12] , 4
# we can just merge A and B and LC1027, runtime O(n^2)
# or we bisect insert and check

# LC1027. Longest Arithmetic Subsequence, not subarray
def longestArithSeqLength(self, A: List[int]) -> int:
    n = len(A)
    dp = {} # dp[index][diff] the length of arithmetic sequence at index with difference diff.
    for i in range(n):
        for j in range(i + 1, n):
            diff = A[j] - A[i]
            dp[j, diff] = dp.get((i, diff), 1) + 1  # A[i], A[j] have length 2

    return max(dp.values())

# Given an array and a number k , then check for every number if it is duplicated in
# previous k or next k numbers
def find_dup_k_distance(array,k):
    cache = set()
    for i,elem in enumerate(array):
        if elem in cache: return True
        cache.add(elem)
        if i >= k: cache.remove(array[i-k])

    return False

#print(find_dup_k_distance([10, 5, 3, 4, 3, 5, 6],3))
#print(find_dup_k_distance([10, 5, 3, 4,7,8,9, 3, 5, 6],3))

# https://leetcode.com/discuss/interview-question/345744
# Given a list of k sorted iterators. Implement MergingIterator to merge them.
from heapq import *
class MergingIterator:
    def __init__(self, iterators):
        self.min_heap = []
        for iterator in iterators:  # O(klogk) time and space
            if iterator.hasNext():
                heappush(self.min_heap, (iterator.next(), iterator))

    def hasNext(self):
        return len(self.min_heap) > 0

    def next(self):
        num, idx, iterator = heappop(self.min_heap)
        if iterator.hasNext():  # O(k)
            heappush(self.min_heap, (iterator.next(), iterator))
        return num

# Given a two-dimensional array of strings, return all possible combination of words.
# Example: [['grey','black'], ['fox','dog'], ['jumped','ran','growled']
# Return: grey fox jumped, grey fox ran, grey fox growled, black fox jumped
# ... black dog growled
lists = [['grey','black'], ['fox','dog'], ['jumped','ran','growled'],]
#print(lists)
from itertools import product
print(list(product(*lists)))  # works fine

ans = [[]]
for l in lists:
    new_ans = []
    for a in ans:
        for w in l:
            new_ans.append(a + [w])
    ans = new_ans

for a in ans:
    print(' '.join(a))
