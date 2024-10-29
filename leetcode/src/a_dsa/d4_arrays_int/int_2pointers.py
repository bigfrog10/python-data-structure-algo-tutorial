from typing import List

# LC2134. Minimum Swaps to Group All 1's Together II min swap group binary circular
def minSwaps(self, nums: List[int]) -> int:
    n, w = len(nums), sum(nums)  # w window size
    ones = max_ones = sum(nums[:w])
    for i in range(n-1):
        ones += nums[(i + w) % n] - nums[i]
        max_ones = max(ones, max_ones)  # max of 1's in window of size w
    return w - max_ones

# LC992. Subarrays with K Different Integers - good subarrays  k diff integers
# https://leetcode.com/problems/subarrays-with-k-different-integers/discuss/523136/JavaC%2B%2BPython-Sliding-Window
def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:  # O(n) time and space
    def at_most(distinctK: int) -> int:
        freq_map = defaultdict(int)
        res = left = 0
        for right in range(len(nums)):
            freq_map[nums[right]] += 1
            while len(freq_map) > distinctK:
                freq_map[nums[left]] -= 1
                if freq_map[nums[left]] == 0:
                    del freq_map[nums[left]]
                left += 1
            res += right - left + 1
        return res
    return at_most(k) - at_most(k - 1)

# LC11. Container With Most Water  container water
def maxArea(self, height: List[int]) -> int:  # O(n) time and O(1) space
    i, j, water = 0, len(height) - 1, 0
    while i < j:
        water = max(water, (j - i) * min(height[i], height[j]))
        if height[i] < height[j]: i += 1
        else: j -= 1
    return water

# LC27. Remove Element - remove given value  remove value delete value remove in place
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
    left = zeros = res = 0
    for right in range(len(nums)):
        if nums[right] == 0: zeros += 1
        while zeros > k:
            if nums[left] == 0: zeros -= 1
            left += 1
        res = max(res, right - left + 1)
    return res
def longestOnes(self, nums: List[int], K: int) -> int:
    left = 0
    for right in range(len(nums)):
        K -= 1 - nums[right]  # only when it's 0, k is changed.
        if K < 0:
            K += 1 - nums[left]
            left += 1
    return len(nums) - left  # len(nums) = right + 1
# https://leetcode.com/problems/max-consecutive-ones-iii/discuss/247564/javacpython-sliding-window/379427?page=3

# LC480. Sliding Window Median  slide win median
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

# LC1838. Frequency of the Most Frequent Element k operations
def maxFrequency(self, nums: List[int], k: int) -> int:
    nums.sort()  # O(nlogn) time and O(1) space
    ans = left = total = 0  # cumulative sum
    for right in range(len(nums)):
        target = nums[right]
        total += target
        while (right - left + 1) * target - total > k:
            total -= nums[left]
            left += 1
        ans = max(ans, right - left + 1)
    return ans
def maxFrequency(self, nums: List[int], k: int) -> int:
    nums.sort()  # O(nlogn) time and O(1) space
    left = total = 0  # cumulative sum
    for right in range(len(nums)):
        target = nums[right]
        total += target
        if (right - left + 1) * target - total > k:
            total -= nums[left]
            left += 1  # this window never decrease, best
    return len(nums) - left  # so we return best window size












