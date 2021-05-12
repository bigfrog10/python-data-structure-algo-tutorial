# LC84. Largest Rectangle in Histogram
def largestRectangleArea(self, heights: List[int]) -> int:  # short but not fast
    if not heights: return 0
    # first 0 is to jumpstart, last 0 is to ensure loop goes to end to pump stack
    h = [0] + heights + [0]
    n = len(h)
    ret, stack = 0, [0] # indices, maintain increasing
    for i in range(1, n):
        while h[i] < h[stack[-1]]: # pump larger and compare
            ret = max(ret, h[stack.pop()] * (i - 1 - stack[-1]))
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

# LC739. Daily Temperatures
def dailyTemperatures(self, T: List[int]) -> List[int]:
    stack = []  # monotonic stack, decreasing
    ret = [0] * len(T)
    for i in reversed(range(len(T))):
        while stack and T[i] >= T[stack[-1]]: stack.pop()
        if stack: ret[i] = stack[-1]-i
        stack.append(i)
    return ret

# LC496. Next Greater Element I
def nextGreaterElement(self, findNums, nums):
    st, d = [], {}
    for v in nums:
        while st and st[-1] < v: # maintain decreasing stack
            d[st.pop()] = v
        st.append(v)

    return map(lambda x: d.get(x, -1), findNums)
