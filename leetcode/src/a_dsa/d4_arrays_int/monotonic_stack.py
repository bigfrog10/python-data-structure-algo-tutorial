
# LC739. Daily Temperatures
def dailyTemperatures(self, T: List[int]) -> List[int]:
    ret, stack = [0] * len(T), []  # monotonic stack, decreasing
    for i in reversed(range(len(T))):
        while stack and T[i] >= T[stack[-1]]: stack.pop()
        if stack: ret[i] = stack[-1] - i
        stack.append(i)
    return ret
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    n = len(temperatures)
    answer, hottest = [0] * n, 0  # O(n) runtime, O(1) space
    for curr_day in range(n - 1, -1, -1):
        current_temp = temperatures[curr_day]
        if current_temp >= hottest:
            hottest = current_temp
            continue  # the answer is 0, no need to go further.
        days = 1
        while current_temp >= temperatures[curr_day + days]:
            # this is the "days" jump to shrink the while total to N for the for loop
            days += answer[curr_day + days]
        answer[curr_day] = days
    return answer

# LC1944. Number of Visible People in a Queue
def canSeePersonsCount(self, heights: List[int]) -> List[int]:  # O(n)
    stack, res = [], [0] * len(heights)  # decreasing mono stack
    for i, v in enumerate(heights):
        while stack and v >= heights[stack[-1]]: res[stack.pop()] += 1
        if stack: res[stack[-1]] += 1  # if there is a higher one, count 1
        stack.append(i)
    return res

# LC735. Asteroid Collision
def asteroidCollision(self, asteroids):
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

# LC85. Maximal Rectangle
def maximalRectangle(self, matrix):
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

# LC496. Next Greater Element I
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    st, d = [], {}  # O(len(nums2)) runtime and space
    for n in nums2:
        while st and st[-1] < n: d[st.pop()] = n  # maintain decreasing stack
        st.append(n)
    return [d.get(x, -1) for x in nums1]

# LC503. Next Greater Element II
def nextGreaterElements(self, nums):  # best solution
    n = len(nums)
    ret = [-1] * n
    stack = nums[::-1]  # reverse this for comparison
    for i in range(n)[::-1]:
        while stack and stack[-1] <= nums[i]: stack.pop()  # maintain decreasing order
        if stack: ret[i] = stack[-1]  # this -1 number is larger than current
        stack.append(nums[i])  # in case this is the next greater
    return ret

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
