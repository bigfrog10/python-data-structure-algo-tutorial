# 995. Minimum Number of K Consecutive Bit Flips  k flip
def minKBitFlips(self, nums: List[int], k: int) -> int:
    n = len(nums)  # Length of the input list
    flip_queue = collections.deque()  # Queue to keep track of flips
    flipped = 0  # Current flip state
    result = 0  # Total number of flips
    for i, num in enumerate(nums):
        if i >= k: # Remove the effect of the oldest flip if it's out of the current window
            flipped ^= flip_queue[0]
        if flipped == nums[i]:  # If the current bit is 0 (i.e., it needs to be flipped), 1==1 0 == 0
            if i + k > n: # If we cannot flip a subarray starting at index i
                return -1
            flip_queue.append(1) # Add a flip at this position
            flipped ^= 1  # Toggle the flipped state
            result += 1  # Increment the flip count
        else:
            flip_queue.append(0)

        if len(flip_queue) > k: # Remove the oldest flip effect if the queue is longer than k
            flip_queue.popleft()
    return result

# LC134. Gas Station - on a circle
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:  # O(n)
    if not gas or not cost: return -1
    total = current = 0
    start_station = 0
    for i, (g, c) in enumerate(zip(gas, cost)):
        exp = g - c
        total += exp  # accumulate total diff of gas - cost
        current += exp
        if current < 0:
            current = 0
            start_station = i+1
    return start_station if total >= 0 else -1

# LC1710. Maximum Units on a Truck
def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    boxes = sorted(boxTypes, key=lambda x: x[1], reverse=True)  # O(nlogn)
    total, counts = 0, 0 # greedy: box count 1 with max units
    for c, u in boxes:
        if c <= truckSize - counts:
            counts += c
            total += c * u
        else:
            total += (truckSize - counts) * u
            counts = truckSize
            break
    return total

# LC1648. Sell Diminishing-Valued Colored Balls
def maxProfit(self, inv: List[int], orders: int) -> int:
    # https://leetcode.com/problems/sell-diminishing-valued-colored-balls/discuss/927522/Python-n-log-n-690-ms
    arr=sorted(Counter(inv).items(), reverse=True)+[(0,0)]
    ans, ind, width = 0, 0, 0

    while orders>0: # constraint: sum(inv) >= orders
        width += arr[ind][1] # number of ball
        sell = min(orders, width * (arr[ind][0] - arr[ind+1][0])) # sell diff to flatten
        whole, remainder= divmod(sell, width)
        price_w = width * whole * (arr[ind][0] + arr[ind][0] - (whole-1)) // 2
        price_r = remainder * (arr[ind][0] - whole)
        ans += price_w + price_r
        orders -= sell
        ind += 1
    return ans % 1_000_000_007

# LC1326. Minimum Number of Taps to Open to Water a Garden
def minTaps(self, n: int, ranges: List[int]) -> int:
    jumps = [0]*(n+1)
    for x, r in enumerate(ranges):
        l = max(0, x-r)
        jumps[l] = max(jumps[l], x+r)
    step = start = end = 0
    while end < n:
        start, end = end+1, max(jumps[start:end+1])  # greedy on max
        if start > end: return -1
        step += 1
    return step



# LC1306. Jump Game III  O(n)
def canReach(self, arr: List[int], start: int) -> bool:  # O(n) runtime and space
    if 0 <= start < len(arr) and arr[start] >= 0:  # DFS
        if arr[start] == 0: return True
        arr[start] = - arr[start]  # visited
        return self.canReach(arr, start + arr[start]) or self.canReach(arr, start - arr[start])
    return False  # existing array can be restored back
def canReach(self, arr: List[int], start: int) -> bool:  # O(n) runtime and space
    q, n = [start], len(arr)  # BFS
    while q:
        node = q.pop(0)
        if arr[node] == 0: return True
        if arr[node] < 0: continue  # visited
        arr[node] = -arr[node]  # mark as visited
        for i in [node + arr[node], node - arr[node]]:
            if 0 <= i < n: q.append(i)
    return False

# LC55. Jump Game
def canJump(self, nums: List[int]) -> bool:  # greedy, O(n)
    target = len(nums) - 1 # we start from end go backward to front.
    for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= target: target = i # find earliest i such that
        # don't check "else" because even this is not true, i-1 could be true
    return target == 0 # we can reach target from i, and repeat this process to 1st.

# LC45. Jump Game II
def jump(self, nums: List[int]) -> int:
    farthest = current_jump_end = jumps = 0
    for i in range(len(nums) - 1):
        # we continuously find the how far we can reach in the current jump
        farthest = max(farthest, i + nums[i])
        # if we have come to the end of the current jump,
        # we need to make another jump
        if i == current_jump_end:
            jumps += 1
            current_jump_end = farthest
    return jumps

# LC818. Race Car
def racecar(self, target):  # O(t logt)
    @lru_cache(None)
    def dp(t):
        n = t.bit_length()
        p1= 2 ** (n-1)
        p2 = p1 * 2  # 2 ** n

        if p2 - 1 == t: return n  # nA

        res = dp(p2 - 1 - t) + n + 1  # nA R and then dp(p2-1-t) backward, i.e., we pass target and come back
        for m in range(n-1):  # (n-1)A R and then m steps
            res = min(res, dp(t - p1 + 2 ** m) + n + m + 1)
        return res
    return dp(target)




