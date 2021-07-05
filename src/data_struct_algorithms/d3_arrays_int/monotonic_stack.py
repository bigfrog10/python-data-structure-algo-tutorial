# LC962. Maximum Width Ramp
def maxWidthRamp(self, A):
    res, st = 0, []
    for i, a in enumerate(A):
        if not st or a < A[st[-1]]: st.append(i) # decreasing stack
    for j in range(len(A))[::-1]:
        while st and A[j] >= A[st[-1]]: res = max(res, j - st.pop())
    return res

# LC84. Largest Rectangle in Histogram
def largestRectangleArea(self, height: List[int]) -> int:  # short but not fast
    height.append(0)
    stack = [-1]
    ans = 0
    for i in range(len(height)):
        while height[i] < height[stack[-1]]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1
            ans = max(ans, h * w)
        stack.append(i)
    height.pop()
    return ans

# LC85. Maximal Rectangle
def maximalRectangle(self, matrix):
    # https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/28917/AC-Python-clean-solution-using-stack-76ms
    if not matrix or not matrix[0]: return 0
    n = len(matrix[0])
    height = [0] * (n + 1)
    ans = 0
    for row in matrix:
        for i in range(n):
            height[i] = height[i] + 1 if row[i] == '1' else 0
        stack = [-1]
        for i in range(n + 1):
            while height[i] < height[stack[-1]]: # maintain increasing height in stack
                h = height[stack.pop()]
                w = i - 1 - stack[-1]
                ans = max(ans, h * w)
            stack.append(i)
    return ans




# LC496. Next Greater Element I
def nextGreaterElement(self, findNums, nums):
    st, d = [], {}
    for v in nums:
        while st and st[-1] < v: # maintain decreasing stack
            d[st.pop()] = v
        st.append(v)
    return map(lambda x: d.get(x, -1), findNums)

# LC503. Next Greater Element II
def nextGreaterElements(self, nums):  # best solution
    n = len(nums)
    ret = [-1] * n
    stack = nums[::-1] # reverse this because append() below, append is O(1)
    # for i in range(n - 1, -1, -1):
    for i in range(n)[::-1]:
        while stack and stack[-1] <= nums[i]: # maintain decreasing order
            stack.pop()
        if stack: ret[i] = stack[-1] # this -1 number is larger than current
        stack.append(nums[i]) # in case this is the next greater
    return ret

# LC739. Daily Temperatures
def dailyTemperatures(self, T: List[int]) -> List[int]:
    stack = []  # monotonic stack, decreasing
    ret = [0] * len(T)
    for i in reversed(range(len(T))):
        while stack and T[i] >= T[stack[-1]]: stack.pop()
        if stack: ret[i] = stack[-1]-i
        stack.append(i)
    return ret

# LC735. Asteroid Collision
def asteroidCollision(self, asteroids):
    ans = [] # decreasing stack
    for new in asteroids:
        while ans and new < 0 < ans[-1]:
            if ans[-1] < -new:
                ans.pop()
                continue
            elif ans[-1] == -new: ans.pop()
            break
        else: ans.append(new)
    return ans

# https://1e9.medium.com/monotonic-queue-notes-980a019d5793
# LC456. 132 Pattern
def find132pattern(self, nums: List[int]) -> bool:
    stack = []
    s3 = float('-inf')
    for n in nums[::-1]:
        if n < s3: return True # this n is 1st, we found it, so return.
        while stack and stack[-1] < n:
            s3 = stack.pop() # find second as s3, 3rd as n
        stack.append(n)
    return False
