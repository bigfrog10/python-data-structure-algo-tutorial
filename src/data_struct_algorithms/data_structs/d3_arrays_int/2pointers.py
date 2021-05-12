# LC1004. Max Consecutive Ones III
def longestOnes(self, A: List[int], K: int) -> int:
    left = 0
    for right in range(len(A)):
        K -= 1 - A[right]
        if K < 0:
            K += 1 - A[left]
            left += 1
    return right - left + 1 # len(A) - left, include both left and right

# LC283. Move Zeroes
def moveZeroes(self, nums: List[int]) -> None:
    i = 0  # index for last nonzero from left, or move together
    for j in range(len(nums)): # we move nonzeor to back actually
        if nums[j] != 0: # if value is zero, j moves but i stays.
            nums[i] = nums[j]
            if i != j: nums[j] = 0
            i += 1

# LC163. Missing Ranges
def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
    res = []
    if not nums:
        if lower == upper: res.append(str(lower))
        else: res.append(f'{lower}->{upper}')
        return res
    left = lower
    for i in nums:
        if i - left == 1: res.append(str(left))
        elif i - left > 1: res.append(f'{left}->{i-1}')
        left = i+1
    i = nums[-1]
    if upper - i == 1: res.append(str(upper))
    elif upper - i > 1: res.append(f'{i+1}->{upper}')
    return res

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
    res = 0
    b, e = 0, len(height) - 1
    while b < e:
        res = max(res, min(height[b], height[e]) * (e - b))
        if height[b] < height[e]: b += 1 # move lower height
        else: e -= 1
    return res

# LC1248. Count Number of Nice Subarrays
def numberOfSubarrays(self, nums: List[int], k: int) -> int:
    left = count = res = 0  # count for even numbers.
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


