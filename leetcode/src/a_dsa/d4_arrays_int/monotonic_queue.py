
# LC1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
def longestSubarray(self, nums: List[int], limit: int) -> int:
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

# LC239. Sliding Window Maximum, top100
def maxSlidingWindow(self, nums: 'List[int]', k: 'int') -> 'List[int]':  # O(n)
    n = len(nums)
    if n == 0 or k == 0: return []
    if k == 1: return nums
    deq = collections.deque()
    def clean_deque(i):
        if deq and deq[0] == i - k: deq.popleft()  # remove indexes of elements not from sliding window
        # remove from deq indexes of all elements
        # which are smaller than current element nums[i]
        while deq and nums[i] > nums[deq[-1]]: deq.pop() # decreasing queue
    output =[]
    for i in range(n):
        clean_deque(i)
        deq.append(i)
        if i >= k-1: output.append(nums[deq[0]])
    return output

# LC1425. Constrained Subsequence Sum
def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
    q = collections.deque() # O(n)
    for i in range(len(nums)): # nums are overwritten as max sum so far
        if q and nums[q[0]] > 0: nums[i] += nums[q[0]] # add prev sum
        while q and nums[q[-1]] < nums[i]: q.pop() # if curr > prev sum, drop prev sum
        q.append(i)
        if i - q[0] == k: q.popleft() # keep deque size to k
    return max(nums)


