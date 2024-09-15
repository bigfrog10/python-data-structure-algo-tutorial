from typing import List

# LC977. Squares of a Sorted Array
def sortedSquares(self, nums: List[int]) -> List[int]:  # O(n) time, O(1) space
    n = len(nums)
    result = [0] * n
    left, right = 0, n-1
    for i in reversed(range(n)):
        if abs(nums[left]) < abs(nums[right]):
            square = nums[right]
            right -= 1
        else:
            square = nums[left]
            left += 1
        result[i] = square * square
    return result

# LC992. Subarrays with K Different Integers - good subarrays
# https://leetcode.com/problems/subarrays-with-k-different-integers/discuss/523136/JavaC%2B%2BPython-Sliding-Window
def subarraysWithKDistinct(self, A: List[int], K: int) -> int:  # O(n) time and O(k) space
    def atMostK(A, K): # we demand K diff ints
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)): # move right
            if count[A[j]] == 0: K -= 1 # we don't have this char anymore
            count[A[j]] += 1
            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0: K += 1 # we need 1 more
                i += 1  # left pointer move right
            res += j - i + 1 # when k >= 0 # all substring starting j
        return res
    return atMostK(A, K) - atMostK(A, K - 1)

# LC11. Container With Most Water  container water
def maxArea(self, height: List[int]) -> int:  # O(n) time and O(1) space
    i, j, water = 0, len(height) - 1, 0
    while i < j:
        water = max(water, (j - i) * min(height[i], height[j]))
        if height[i] < height[j]: i += 1
        else: j -= 1
    return water

# LC27. Remove Element - remove given value  remove value
def removeElement(self, nums: List[int], val: int) -> int:
    i = 0
    for j in range(len(nums)):
        if nums[j] != val:
            nums[i] = nums[j]
            i += 1
    return i

# LC283. Move Zeroes - to end
def moveZeroes(self, nums: List[int]) -> None:
    i = 0  # index for last nonzero from left, or move together
    for j in range(len(nums)):  # we move nonzeor to back actually
        if nums[j] != 0:  # if value is zero, j moves but i stays.
            nums[i] = nums[j]
            if i != j: nums[j] = 0
            i += 1

# LC1004. Max Consecutive Ones III - can flip <=k bits.
def longestOnes(self, nums: List[int], k: int) -> int:
    left = zero_count =max_length = 0
    for right in range(len(nums)):
        if nums[right] == 0: zero_count += 1
        while zero_count > k:
            if nums[left] == 0: zero_count -= 1
            left += 1
        max_length = max(max_length, right - left + 1)
    return max_length
def longestOnes(self, nums: List[int], K: int) -> int:
    left = 0
    for right in range(len(nums)):
        K -= 1 - nums[right]  # only when it's 0, k is changed.
        if K < 0:
            K += 1 - nums[left]
            left += 1
    return len(nums) - left  # len(nums) = right + 1
# https://leetcode.com/problems/max-consecutive-ones-iii/discuss/247564/javacpython-sliding-window/379427?page=3

# LC480. Sliding Window Median
def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:  # O(nk)
    window = sorted(nums[:k])
    medians = []
    for a, b in zip(nums, nums[k:] + [0]):  # add 0 to add last value
        medians.append((window[k//2] + window[~(k//2)]) / 2.)
        window.remove(a)  # O(k)
        bisect.insort(window, b) # O(k)
    return medians

# LC42. Trapping Rain Water, top100
def trap(self, heights: List[int]) -> int: # 2 pointers, O(n)
    if not heights: return 0
    left, right = 0, len(heights)-1
    lmax = rmax = ret = 0  # maintain history
    while left < right:
        if heights[left] < heights[right]:
            lmax = max(lmax, heights[left])
            ret += lmax - heights[left]  # water trapped
            left += 1
        else:
            rmax = max(rmax, heights[right])
            ret += rmax - heights[right]
            right -= 1
    return ret

# LC2090. K Radius Subarray Averages
def getAverages(self, nums: List[int], k: int) -> List[int]:
    ans = [-1]*len(nums)
    rsm, r = 0, 2*k+1 # range sum, 2k elements on both sides and center
    for i, x in enumerate(nums):
        rsm += x
        if i >= r: rsm -= nums[i-r]  # i is 0 index based
        if i+1 >= r: ans[i-k] = rsm // r  # i is 0 index based, i+1 is no. of elements
    return ans

# LC1248. Count Number of Nice Subarrays
def numberOfSubarrays(self, nums: List[int], k: int) -> int:
    left = count = res = 0
    for right in range(len(nums)):
        if nums[right] & 1:
            k -= 1
            count = 0
        while k == 0:
            k += nums[left] & 1
            left += 1 # move left pointer
            count += 1 # count this subarray
        res += count
    return res



# LC611. Valid Triangle Number
def triangleNumber(self, nums: List[int]) -> int:
    n = len(nums)
    nums.sort()
    count = 0
    for i in range(n-1,1,-1):
        hi, lo = i - 1, 0
        while lo < hi:
            if nums[hi]+nums[lo] > nums[i]:
                count += hi - lo
                hi -= 1
            else: lo += 1
    return count
