
# LC2289. Steps to Make Array Non-decreasing  array number of removes to increase  non decreasing
def totalSteps(self, nums: List[int]) -> int:
    res, stack = 0, []  # O(n) time space
    for i in range(len(nums)-1,-1,-1):
        cur = 0  # number of rounds to remove nums continuously
        while stack and nums[stack[-1][0]] < nums[i]:
            _, v = stack.pop()  # eat this number
            cur = max(cur+1, v)
        res = max(res, cur)
        stack.append([i, cur])
    return res

# LC769. Max Chunks To Make Sorted
def maxChunksToSorted(self, arr: List[int]) -> int:
    max_so_far = arr[0]
    count = 0
    for i in range(len(arr)):
        if max_so_far < arr[i]: max_so_far = arr[i]
        if max_so_far == i: count += 1
    return count
# https://leetcode.com/problems/max-chunks-to-make-sorted/solutions/1225805/python-easy-sol-28ms-with-detailed-explanation/

# LC768. Max Chunks To Make Sorted II
def maxChunksToSorted(self, arr: List[int]) -> int:
    stack = []
    for a in arr:  # O(n) time and space
        _max = a  # for-while = stack push and pop -> O(2n)
        while stack and stack[-1] > a:
            _max = max(_max, stack.pop())
        stack.append(_max)  # these can be sorted in 1 block
    return len(stack)

# LC1642. Furthest Building You Can Reach - furthest jump furthest building  reach building tallest building tall building
def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
    heap = []   ## O(NlogK) time and O(k) space, k=len(ladders)
    for i in range(len(heights) - 1):
        d = heights[i + 1] - heights[i]
        if d > 0:
            heapq.heappush(heap, d)
        if len(heap) > ladders:  # ladders is number of ladders that can be used.
            bricks -= heapq.heappop(heap)
        if bricks < 0:
            return i
    return len(heights) - 1

# https://1e9.medium.com/monotonic-queue-notes-980a019d5793
# LC456. 132 Pattern
def find132pattern(self, nums: List[int]) -> bool:  # O(n) time and space
    stack = []  # decreasing
    mid = float('-inf')
    for n in nums[::-1]:  # from end!!
        if n < mid: return True # this is smallest
        while stack and n > stack[-1]:
            mid = stack.pop() # this n is largest
        stack.append(n)  # decreasing stack
    return False

# LC84. Largest Rectangle in Histogram   large rect histo
def largestRectangleArea(self, heights: List[int]) -> int:  # O(n) runtime and space
    heights.append(0)  # append 0 so heights[-1] is this 0 when stack has -1
    ans, stack = 0, [-1]  # stack for increasing height index
    for i in range(len(heights)):  # last 0 will pop stack leftovers
        while heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i - stack[-1] - 1  # last pop use i always
            ans = max(ans, h * w)
        stack.append(i)
    return ans

# LC85. Maximal Rectangle  max rectangle
def maximalRectangle(self, matrix):  # O(mn) time and O(n) space
    # https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/28917/AC-Python-clean-solution-using-stack-76ms
    if not matrix or not matrix[0]: return 0
    n = len(matrix[0])
    ans, height = 0, [0] * (n + 1)  # last 1 is to trigger below while loop
    for row in matrix:
        for i in range(n): height[i] = height[i] + 1 if row[i] == '1' else 0
        stack = [-1]  # now update target
        for i in range(n + 1):
            while height[i] < height[stack[-1]]: # maintain increasing height in stack
                h = height[stack.pop()]  # pop out larger one
                w = i - 1 - stack[-1]  # i-1: -1 is for (n+1) 0
                ans = max(ans, h * w)
            stack.append(i)
    return ans

# LC1944. Number of Visible People in a Queue
def canSeePersonsCount(self, heights: List[int]) -> List[int]:  # O(n)
    stack, res = [], [0] * len(heights)  # decreasing mono stack
    for i, v in enumerate(heights):
        while stack and v >= heights[stack[-1]]: res[stack.pop()] += 1  # pop() can see v
        if stack: res[stack[-1]] += 1  # -1 can see v
        stack.append(i)
    return res

# LC739. Daily Temperatures   next greater element
def dailyTemperatures(self, T: List[int]) -> List[int]:
    n = len(T)
    ans, hottest = [0] * n, 0  # O(n) runtime, O(1) space
    for currday in range(n)[::-1]:
        tempr = T[currday]
        if tempr >= hottest: # do nothing for increasing from right
            hottest = tempr
            continue  # the ans is 0, no need to go further.
        days = 1
        while tempr >= T[currday + days]:
            # this is the "days" jump to shrink the while total to N for the for loop
            days += ans[currday + days]  # accu through valley
        ans[currday] = days
    return ans
def dailyTemperatures(self, T: List[int]) -> List[int]:
    ret, stack = [0] * len(T), []  # monotonic stack, decreasing  O(n) time and space
    for curr_day, curr_temp in enumerate(T):
        while stack and curr_temp > T[stack[-1]]:
            prev_day = stack.pop()
            ret[prev_day] = curr_day - prev_day
        stack.append(curr_day)
    return ret

# LC735. Asteroid Collision
def asteroidCollision(self, asteroids):  # O(n) time and space
    ans = []  # decreasing stack
    for new in asteroids:
        while ans and new < 0 < ans[-1]:
            if ans[-1] < -new:
                ans.pop()
                continue
            elif ans[-1] == -new: ans.pop()
            break
        else: ans.append(new)
    return ans

# LC1950. Maximum of Minimum Values in All Subarrays - minmax of all subarrays
def findMaximums(self, nums: List[int]) -> List[int]:
    stack = []
    nums = [-1] + nums + [0]
    n = len(nums)
    res = [0] * (n - 2)
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] >= num:
            j = stack.pop()
            k = i - 1 - (stack[-1] + 1)  # k is the range where nums[j] is max, array length
            res[k] = max(res[k], nums[j])
        stack.append(i)
    for i in range(n - 3, 0, -1):  # example, 5, 1, 1, 5
        res[i - 1] = max(res[i], res[i - 1])
    return res

# LC2104. Sum of Subarray Ranges - see LC828   sum sub range of a subarray range
def subArrayRanges(self, nums: List[int]) -> int:  # O(n) time and space
    A = [-math.inf] + nums + [-math.inf]  # make sure stack has 1 elem
    res, s = 0, []
    for i, x in enumerate(A):
        while s and x < A[s[-1]]: # stack increasing
            j = s.pop()  # A[j] is smallest between i and s[-1]
            res -= A[j] * (i - j) * (j - s[-1])
        s.append(i)
    A = [inf] + nums + [inf]  # use inf instead of -inf
    s = []
    for i, x in enumerate(A):
        while s and x > A[s[-1]]:  # stack decreasing
            j = s.pop()
            res += A[j] * (i - j) * (j - s[-1])
        s.append(i)
    return res

# LC907. Sum of Subarray Minimums    sum sub min sum min
def sumSubarrayMins(self, arr: List[int]) -> int:
    res, stack = 0, []  # O(n) time and space
    arr = [float('-inf')] + arr + [float('-inf')]
    for i, n in enumerate(arr):
        while stack and n < arr[stack[-1]]:  #  non-decreasing stack
            cur = stack.pop()
            # num of subarrays with A[cur] as minimum, right * left
            res += arr[cur] * (i - cur) * (cur - stack[-1])
            res = res % (10**9 + 7)
        stack.append(i)
    return res

# LC962. Maximum Width Ramp max width ramp
def maxWidthRamp(self, nums: List[int]) -> int:
    res, st = 0, []
    for i, a in enumerate(nums):
        if not st or a < nums[st[-1]]: st.append(i) # decreasing stack
    for j in range(len(nums))[::-1]:
        while st and nums[j] >= nums[st[-1]]: res = max(res, j - st.pop())
    return res

# LC496. Next Greater Element I - first greater on right, 2 arrays  1st greater
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    st, d = [], {}  # O(len(nums2)) runtime and space
    for n in nums2:
        while st and st[-1] < n: d[st.pop()] = n  # maintain increasing stack
        st.append(n)
    return [d.get(x, -1) for x in nums1]

# LC503. Next Greater Element II - circular array
def nextGreaterElements(self, nums):  # O(n) time and space
    n = len(nums)
    ret = [-1] * n
    stack = nums[::-1]  # reverse this for comparison
    for i in range(n)[::-1]:
        while stack and stack[-1] <= nums[i]: stack.pop()  # maintain increasing order
        if stack: ret[i] = stack[-1]  # this -1 number is larger than current
        stack.append(nums[i])  # in case this is the next greater
    return ret

