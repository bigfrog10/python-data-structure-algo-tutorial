## Arrays

Array are index based list. Normally, they are the fastest block of data and 
need to pre-allocate memory by declaring its size, such as in C++ and Java.
Python is a little different.

## Basic Operations
[Python Wiki](https://wiki.python.org/moin/TimeComplexity) outlines the time
for operations:
- get, such as ```a[i]```
- set, such as ```a[i] = 5```


## Partition Selection
- Find both min/max in one scan: O(N)  
  [Maximum and minimum of an array using minimum number of comparisons](https://www.geeksforgeeks.org/maximum-and-minimum-in-an-array/?ref=leftbar-rightbar)
  use pair comparison to break out n/2 if n is even or (n-1)/2 comparison and 
  (n-1)/2 + 1 elements in each subgroup.
  each subgroup needs direct comparisons, n/2 - 1 times. So for n even, we have 3 * n / 2 - 2.
  For n odd, 2 * ((n-1) / 2 ) + (n-1) / 2 = n-1 + (n-1) / 2 = 3(n - 1) / 2
- Find first K largest element: O(N)  
  This is partition selection.  
  [LeetCode 215](https://leetcode.com/problems/kth-largest-element-in-an-array/)
  and [LeetCode 973](https://leetcode.com/problems/k-closest-points-to-origin/)  
  Two good references are [Medians](http://staff.ustc.edu.cn/~csli/graduate/algorithms/book6/chap10.htm)
  and [Selection](https://web.bowdoin.edu/~ltoma/teaching/cs231/fall09/Lectures/7-selection/selection.pdf)
  https://rcoh.me/posts/linear-time-median-finding/
  https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
  
## Sorting
https://en.wikipedia.org/wiki/Sorting_algorithm

There are many sorting algorithms, best time is O(NlogN).

is this sort stable?

https://realpython.com/sorting-algorithms-python/

[Minimum number of swaps required to sort an array](https://www.geeksforgeeks.org/minimum-number-swaps-required-sort-array/?ref=leftbar-rightbar)

## Ordering
use stack or queue


Given an array:
- find the max sum of 2 elements: just pick 2 largest
- find the max diff of 2 elements: pointers
- find the max sum of 2 elements that are k apart:
  https://www.quora.com/Is-it-possible-to-find-maximum-sum-of-two-numbers-in-an-unsorted-array-such-that-the-numbers-are-not-adjacent-to-each-other-in-O-N
  https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/
  DP or pointers
  see src folder
  
we may ask same sum questions for 3 elements, or m elements (m is constant, 
independent of array size).
https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/discuss/1027055/A-generic-solution-for-m-subarrays

[LC 239]
max in sliding windows, is O(n) but much harder
monotonic queue
sum in sliding window is easy, O(n)

LC 560
find num of subarrays whose sum is equal to a given target.
prefix sum

Presum

LC 485. Max Consecutive Ones

39. Combination Sum
1031. Maximum Sum of Two Non-Overlapping Subarrays

300. Longest Increasing Subsequence
491. Increasing Subsequences
