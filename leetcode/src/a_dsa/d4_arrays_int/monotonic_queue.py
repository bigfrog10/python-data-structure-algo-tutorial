
# LC1696. Jump Game VI
def maxResult(self, nums: List[int], k: int) -> int:  # O(n) time and O(k) space
    n, score = len(nums), nums[0]
    dq = deque([(0, score)])
    for i in range(1, n):
        while dq and dq[0][0] < i-k: dq.popleft()  # pop out the old index
        score = dq[0][1] + nums[i]
        while dq and dq[-1][1] <= score: dq.pop()  # pop out the smaller value
        dq.append((i, score))
    return score

# LC1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
# https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/discuss/609771/JavaC%2B%2BPython-Deques-O(N)
def longestSubarray(self, nums: List[int], limit: int) -> int:  # O(n)
    maxd = collections.deque()
    mind = collections.deque()
    i = 0
    for a in nums:
        while len(maxd) and a > maxd[-1]: maxd.pop()
        while len(mind) and a < mind[-1]: mind.pop()
        maxd.append(a)
        mind.append(a)
        if maxd[0] - mind[0] > limit:
            if maxd[0] == nums[i]: maxd.popleft()
            if mind[0] == nums[i]: mind.popleft()
            i += 1
    return len(nums) - i

# LC239. Sliding Window Maximum
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    res =[]   # O(n) time, O(k) space
    deq = collections.deque()  # store idxs, nums mono decreasing
    for i in range(len(nums)):
        if deq and deq[0] == i - k: deq.popleft()  # sliding out
        while deq and nums[i] > nums[deq[-1]]: deq.pop() # decreasing queue
        deq.append(i)
        if i >= k-1: res.append(nums[deq[0]])  # range is [0, k-1]
    return res

# LC1425. Constrained Subsequence Sum
def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
    q = collections.deque() # O(n)
    for i in range(len(nums)): # nums are overwritten as max sum so far
        if q and nums[q[0]] > 0: nums[i] += nums[q[0]] # add prev sum
        while q and nums[q[-1]] < nums[i]: q.pop() # if curr > prev sum, drop prev sum
        q.append(i)
        if i - q[0] == k: q.popleft() # keep deque size to k
    return max(nums)


