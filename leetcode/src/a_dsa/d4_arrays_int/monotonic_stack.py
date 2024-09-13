
# LC2289. Steps to Make Array Non-decreasing
def totalSteps(self, nums: List[int]) -> int:
    res,stack = 0, []
    for i in range(len(nums)-1,-1,-1):
        cur = 0  # number of rounds to remove nums
        while stack and nums[stack[-1][0]]<nums[i]:
            _,v = stack.pop()  # eat this number
            cur=max(cur+1,v)
        res = max(res,cur)
        stack.append([i,cur])
    return res

# LC768. Max Chunks To Make Sorted II
def maxChunksToSorted(self, arr: List[int]) -> int:
    stack = []
    for a in arr:  # O(n) time and space
        _max = a  # for-while = stack push and pop -> O(2n)
        while stack and stack[-1] > a:
            _max = max(_max, stack.pop())
        stack.append(_max)  # these can be sorted in 1 block
    return len(stack)

# LC1642. Furthest Building You Can Reach - furthest jump furthest building
def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
    heap = []   ## O(NlogK) time and O(k) space, k=len(ladders)
    for i in range(len(heights) - 1):
        d = heights[i + 1] - heights[i]
        if d > 0:
            heapq.heappush(heap, d)
        if len(heap) > ladders:
            bricks -= heapq.heappop(heap)
        if bricks < 0:
            return i
    return len(heights) - 1

# https://1e9.medium.com/monotonic-queue-notes-980a019d5793
# LC456. 132 Pattern
def find132pattern(self, nums: List[int]) -> bool:  # O(n) time and space
    stack = []
    mid = float('-inf')
    for n in nums[::-1]:
        if n < mid: return True # this is smallest
        while stack and stack[-1] < n:
            mid = stack.pop() # this n is largest
        stack.append(n)
    return False

# LC84. Largest Rectangle in Histogram
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

# LC739. Daily Temperatures
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    n = len(temperatures)
    answer, hottest = [0] * n, 0  # O(n) runtime, O(1) space
    for curr_day in range(n)[::-1]:
        current_temp = temperatures[curr_day]
        if current_temp >= hottest:
            hottest = current_temp
            continue  # the answer is 0, no need to go further.
        days = 1
        while current_temp >= temperatures[curr_day + days]:
            # this is the "days" jump to shrink the while total to N for the for loop
            days += answer[curr_day + days]  # accu through valley
        answer[curr_day] = days
    return answer
def dailyTemperatures(self, T: List[int]) -> List[int]:
    ret, stack = [0] * len(T), []  # monotonic stack, decreasing  O(n) time and space
    for i in reversed(range(len(T))):
        while stack and T[i] >= T[stack[-1]]: stack.pop()
        if stack: ret[i] = stack[-1] - i
        stack.append(i)
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

# LC2104. Sum of Subarray Ranges - see LC828
def subArrayRanges(self, nums: List[int]) -> int:  # O(n) time and space
    res = 0
    inf = float('inf')
    A = [-inf] + nums + [-inf]
    s = []
    for i, x in enumerate(A):
        while s and A[s[-1]] > x:
            j = s.pop()
            k = s[-1]
            res -= A[j] * (i - j) * (j - k)
        s.append(i)

    A = [inf] + nums + [inf]
    s = []
    for i, x in enumerate(A):
        while s and A[s[-1]] < x:
            j = s.pop()
            k = s[-1]
            res += A[j] * (i - j) * (j - k)
        s.append(i)
    return res

# LC907. Sum of Subarray Minimums
def sumSubarrayMins(self, arr: List[int]) -> int:  # O(n)
    res, stack = 0, []  #  non-decreasing
    A = [float('-inf')] + arr + [float('-inf')]
    for i, n in enumerate(A):
        while stack and A[stack[-1]] > n:
            cur = stack.pop()
            res += A[cur] * (i - cur) * (cur - stack[-1])  # right * left
        stack.append(i)
    return res % (10**9 + 7)

# LC962. Maximum Width Ramp
def maxWidthRamp(self, A):
    res, st = 0, []
    for i, a in enumerate(A):
        if not st or a < A[st[-1]]: st.append(i) # decreasing stack
    for j in range(len(A))[::-1]:
        while st and A[j] >= A[st[-1]]: res = max(res, j - st.pop())
    return res

# LC84. Largest Rectangle in Histogram
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

# LC496. Next Greater Element I - first greater on right, 2 arrays  1st greater
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    st, d = [], {}  # O(len(nums2)) runtime and space
    for n in nums2:
        while st and st[-1] < n: d[st.pop()] = n  # maintain decreasing stack
        st.append(n)
    return [d.get(x, -1) for x in nums1]

# LC503. Next Greater Element II - circular array
def nextGreaterElements(self, nums):  # O(n) time and space
    n = len(nums)
    ret = [-1] * n
    stack = nums[::-1]  # reverse this for comparison
    for i in range(n)[::-1]:
        while stack and stack[-1] <= nums[i]: stack.pop()  # maintain decreasing order
        if stack: ret[i] = stack[-1]  # this -1 number is larger than current
        stack.append(nums[i])  # in case this is the next greater
    return ret
