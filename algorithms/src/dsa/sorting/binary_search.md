

## Binary Search to find extremes

### Minimums
```python
def binary_search(array) -> int:
    def condition(value) -> bool:
        pass

    left, right = 0, len(array)
    while left < right:
        mid = left + (right - left) // 2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    return left  # after exiting the while loop, left is the minimal k satisfying the condition function;
```

### Maximums
```python
def mySqrt(self, x: int) -> int:
        left, right = 0, x
        while left < right:
            mid = (left + right + 1) // 2
            if mid * mid > x: right = mid - 1
            else: left = mid
        return left 
```
without `+1` we have infinite loop (x=8 for example).
after loop left is the max


- @1482. Minimum Number of Days to Make m Bouquets
- @1283. Find the Smallest Divisor Given a Threshold
- @1231. Divide Chocolate
- @1201. Ugly Number III 
- @1011. Capacity To Ship Packages In N Days
- @875. Koko Eating Bananas
- @774. Minimize Max Distance to Gas Station
- @719. Find K-th Smallest Pair Distance 
- @668. Kth Smallest Number in Multiplication Table
- @410. Split Array Largest Sum
- @278. First Bad Version 
- @69. Sqrt(x)
- @35. Search Insert Position
- @34. Find First and Last Position of Element in Sorted Array

https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/discuss/769698/Python-Clear-explanation-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems.
https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/discuss/256729/JavaC%2B%2BPython-Binary-Search
