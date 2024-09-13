
# LC2210. Count Hills and Valleys in an Array
def countHillValley(self, nums: List[int]) -> int:
    count, trend = 0, 0  # trend 1 for up, 0 flat, -1 down
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            if trend == -1:  # previously down
                count += 1  # valley
            trend = 1  # now change to up
        elif nums[i] < nums[i-1]:
            if trend == 1:  # previously up
                count += 1
            trend = -1
    return count

# LC2303. Calculate Amount Paid in Taxes
def calculateTax(self, brackets: List[List[int]], income: int) -> float:
    ans = prev = 0
    for hi, pct in brackets:
        hi = min(hi, income)
        ans += (hi - prev) * pct / 100
        prev = hi
    return ans

# LC1151. Minimum Swaps to Group All 1's Together
def minSwaps(self, data: List[int]) -> int:  # O(n) and O(1)
    ones = sum(data)
    cnt_one = max_one = 0
    left = right = 0
    while right < len(data):
        # updating the number of 1's by adding the new element
        cnt_one += data[right]
        right += 1
        # maintain the length of the window to ones
        if right - left > ones:
            # updating the number of 1's by removing the oldest element
            cnt_one -= data[left]
            left += 1
        # record the maximum number of 1's in the window
        max_one = max(max_one, cnt_one)
    return ones - max_one

# LC525. Contiguous Array - longest subarray with equal # of 1 and 0, 01 array
def findMaxLength(self, nums: List[int]) -> int:
    val2idx = {}  # O(n) time and space
    cumu = max_len = 0
    for i, num in enumerate(nums):
        cumu += 1 if num == 1 else -1  # so we look for 0 when #0 = #1
        if cumu == 0: max_len = i + 1  # balanced from 0 to i
        elif cumu in val2idx: max_len = max(max_len, i - val2idx[cumu])
        else: val2idx[cumu] = i  # do this on first occurrence for longest cases
    return max_len

# 330. Patching Array
def minPatches(self, nums: List[int], n: int) -> int:
    miss, added, index = 1, 0, 0
    while miss <= n:  #O(n)
        if index < len(nums) and nums[index] <= miss:
            miss += nums[index]  # cover (1, miss) with new miss
            index += 1
        else:
            miss += miss  # cover (1, 2*miss)
            added += 1  # need new number
    return added


# LC2625. Flatten Deeply Nested Array        flatten nested
def flatten(arr: list, n):
    if n == 0: return arr
    res = []
    for a in arr:
        if isinstance(a, list):
            b = flatten(a, n-1)
            res.extend(b)
        else: res.append(a)
    return res

# LC565 Array Nesting   interloop
def arrayNesting(self, nums: List[int]) -> int: # O(n) tinme and O(1) space
    res = 0
    for i in range(len(nums)):
        cnt = 0
        while nums[i] >= 0:  # 2 loops still check every element once.
            nums[i], cnt, i = -nums[i] - 1, cnt + 1, nums[i]
        res = max(res, cnt)
        if res > len(nums): return res # works without this too, optimization
    return res

















