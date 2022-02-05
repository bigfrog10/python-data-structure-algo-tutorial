from typing import List

# LC1004. Max Consecutive Ones III
def longestOnes(self, A: List[int], K: int) -> int:
    left = 0
    for right in range(len(A)):
        K -= 1 - A[right]
        if K < 0:
            K += 1 - A[left]
            left += 1
    return right - left + 1  # len(A) - left, include both left and right
# https://leetcode.com/problems/max-consecutive-ones-iii/discuss/247564/javacpython-sliding-window/379427?page=3

# LC480. Sliding Window Median
def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:  # O(nk)
    window = sorted(nums[:k])
    medians = []
    for a, b in zip(nums, nums[k:] + [0]):
        medians.append((window[k//2] + window[~(k//2)]) / 2.)
        window.remove(a)
        bisect.insort(window, b)
    return medians

# LC283. Move Zeroes
def moveZeroes(self, nums: List[int]) -> None:
    i = 0  # index for last nonzero from left, or move together
    for j in range(len(nums)):  # we move nonzeor to back actually
        if nums[j] != 0:  # if value is zero, j moves but i stays.
            nums[i] = nums[j]
            if i != j: nums[j] = 0
            i += 1

# LC42. Trapping Rain Water, top100
def trap(self, heights: List[int]) -> int: # 2 pointers
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

# LC11. Container With Most Water
def maxArea(self, height: List[int]) -> int:
    i, j = 0, len(height) - 1
    water = 0
    while i < j:
        water = max(water, (j - i) * min(height[i], height[j]))
        if height[i] < height[j]: i += 1
        else: j -= 1
    return water



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

# LC992. Subarrays with K Different Integers
def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
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




