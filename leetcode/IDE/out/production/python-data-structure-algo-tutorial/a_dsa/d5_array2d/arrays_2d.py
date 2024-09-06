
# LC2397. Maximum Rows Covered by Columns
def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:
    n,m = len(matrix),len(matrix[0])
    ans = 0
    def check(state,row,rowIncludedCount):
        if sum(state) > numSelect: return
        if row==n:
            nonlocal ans
            ans = max(ans, rowIncludedCount)
            return
        check(state[::], row+1, rowIncludedCount)  # skip this row
        for j in range(m):
            if matrix[row][j]==1: state[j] = 1
        check(state, row+1, rowIncludedCount+1)  # count in this row
    check([0]*m, 0, 0)
    return ans

# LC373. Find K Pairs with Smallest Sums - similar to 378
def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    min_heap, res = [], []  # O(klogk)
    for i in range(min(len(nums1), k)): heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))
    while min_heap and len(res) < k:
        _, i, j = heapq.heappop(min_heap)
        res.append([nums1[i], nums2[j]])
        # all rows first elem are in heap, j+1 results shifting to right
        if j + 1 < len(nums2): heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))
    return res
# https://leetcode.com/problems/find-k-pairs-with-smallest-sums/discuss/84577/O(k)-solution

# LC1868. Product of Two Run-Length Encoded Arrays
def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
    res, l, r = [], 0, 0   # O(n + m), counts of unique numbers
    while encoded1[-1][-1] != 0:
        prod = encoded1[l][0] * encoded2[r][0]
        low = min(encoded1[l][1], encoded2[r][1])
        if res and res[-1][0] == prod: res[-1][1] += low  # extend freq if same value
        else: res.append([prod, low])
        encoded1[l][1] -= low  # minus the finished range
        encoded2[r][1] -= low
        if encoded1[l][1] == 0: l += 1
        if encoded2[r][1] == 0: r += 1
    return res

# LC815. Bus Routes
def numBusesToDestination(self, routes, S, T): # BFS on stops
    stop2routes = collections.defaultdict(set)
    for i, route in enumerate(routes):
        for j in route: stop2routes[j].add(i)  # label stop with bus route #
    queue, seen = [(S, 0)], {S}
    for stop, bus in queue:  # BFS for min stops
        if stop == T: return bus
        for route in stop2routes[stop]:  # for each bus in this stop
            for st in routes[route]:  # for each stop with this bus
                if st not in seen:
                    queue.append((st, bus + 1))
                    seen.add(st)
            routes[route] = []  # ***neat trick: seen route, cut buses for min
    return -1

# LC1743. Restore the Array From Adjacent Pairs
def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
    adj, ans, n = defaultdict(list), [], len(adjacentPairs) + 1
    for a, b in adjacentPairs:
        adj[a] += [b]
        adj[b] += [a]
    prev = -math.inf
    for k, v in adj.items():
        if len(v) == 1:
            ans += [k]
            break
    while len(ans) < n:
        for next in adj.pop(ans[-1]):
            if next != prev:
                prev = ans[-1]
                ans += [next]
                break
    return ans

# LC983. Minimum Cost For Tickets - ticket cost
def mincostTickets(self, days, costs):   # O(len(days))
    durations = [1, 7, 30]  # 2, 7, 5
    N = len(days)
    @lru_cache(None)
    def dp(i): # How much money to do days[i]+
        if i >= N: return 0
        ans, j = float('inf'), i
        for c, d in zip(costs, durations):
            while j < N and days[j] < days[i] + d: j += 1
            ans = min(ans, dp(j) + c)
        return ans
    return dp(0)
def mincostTickets(self, days: List[int], costs: List[int]) -> int:  # O(max(durs))
    durations = [1, 7, 30]  # this should be in parameter passed in
    day_set = set(days)
    @lru_cache(None)
    def dp(i):  # BFS on days
        if i > days[-1]: return 0
        elif i in day_set: # if we do travel today
            return min(dp(i + d) + c for c, d in zip(costs, durations))
        else: return dp(i + 1) # wait for next day if we don't travel today
    # print(dp.cache_info())
    return dp(days[0])

# LC406. Queue Reconstruction by Height - people height
def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:  # O(nlogn)
    from sortedcontainers import SortedList
    people.sort(key=lambda x: (x[0], -x[1]))
    ans = [None] * len(people)
    sl = SortedList(list(range(len(people))))
    for p in people:
        idx = sl[p[1]]
        ans[idx] = p
        sl.remove(idx)  # logn
    return ans
def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:  # O(n^2)
    res = []
    for p in sorted((-x[0], x[1]) for x in people): # from largest to smallest
        res.insert(p[1], [-p[0], p[1]]) # insert only relevant to larger values
    return res

# LC2092. Find All People With Secret - x and y meet at time t
def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
    sm = sorted(meetings, key=lambda x: x[2])  # O(mlogm), m = # of meetings.
    can = {0, firstPerson}
    for _, grp in itertools.groupby(sm, key=lambda x: x[2]):  # O(m)
        queue, graph = set(), defaultdict(list)
        for x, y, _ in grp:  # connect all people in the group
            graph[x].append(y)
            graph[y].append(x)
            if x in can: queue.add(x)
            if y in can: queue.add(y)
        queue = deque(queue)
        while queue:
            x = queue.popleft()
            for y in graph[x]:
                if y not in can:
                    can.add(y)
                    queue.append(y)  # This ensures no order dependency
    return can

# LC946. Validate Stack Sequences
def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
    stack, i = [], 0
    for x in pushed:
        stack.append(x)
        while stack and stack[-1] == popped[i]:
            i += 1
            stack.pop()
    return len(stack) == 0

# LC1168. Optimize Water Distribution in a Village
def minCostToSupplyWater(self, n: int, wells, pipes) -> int:  # O((N+M) * log(N+M)), because of heap
    # build graph, Prim's algo for minimum spanning tree.  O(N+M) space
    graph = collections.defaultdict(list)
    for u, v, w in pipes:
        graph[u].append([w, u, v])
        graph[v].append([w, v, u])
    for i in range(n):
        graph[0].append([wells[i], 0, i+1])  # No need to point back to 0 since we start from 0.
    edges = graph[0]
    heapq.heapify(edges)
    res, visited = 0, {0}
    while len(visited) < n+1 and edges:
        w, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            res += w
            for edge in graph[v]:
                if edge[2] not in visited: heapq.heappush(edges, edge)
    return res

# LC1834. Single-Threaded CPU - single cpu
def getOrder(self, tasks: List[List[int]]) -> List[int]:  # O(nlogn)
    tasks = sorted([(t[0], t[1], i) for i, t in enumerate(tasks)])
    res, h = [], []  # heap
    i, n, time = 0, len(tasks), tasks[0][0]
    while len(res) < n:
        while i < n and tasks[i][0] <= time:  # available task for now
            heapq.heappush(h, (tasks[i][1], tasks[i][2])) # (processing_time, original_index)
            i += 1
        if h:
            t_diff, original_index = heapq.heappop(h)
            time += t_diff
            res.append(original_index)
        elif i < n: time = tasks[i][0]
    return res



# LC1376. Time Needed to Inform All Employees
def numOfMinutes(self, n, headID, manager, informTime):
    reports = defaultdict(list)
    for i, m in enumerate(manager):
        if m >= 0: reports[m].append(i)

    def dfs(i):
        return max([dfs(j) for j in reports[i]] or [0]) + informTime[i]
    return dfs(headID)

# LC1640. Check Array Formation Through Concatenation - chain arrays, concat array
def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:  # nlogn
    n, p_len = len(arr), len(pieces)
    pieces.sort()  # O(nlogn)
    i = 0
    while i < n:
        left, right = 0, p_len - 1
        found = -1
        while left <= right:  # use binary search to find target piece:
            mid = (left + right)//2
            if pieces[mid][0] == arr[i]:
                found = mid
                break
            elif pieces[mid][0] > arr[i]: right = mid - 1
            else: left = mid + 1
        if found == -1: return False
        # check target piece
        target_piece = pieces[found]
        for x in target_piece:
            if x != arr[i]: return False
            i += 1
    return True

# LC223. Rectangle Area - intersection of 2 rectangles
def computeArea(self, A, B, C, D, E, F, G, H):  # AB lower left, CD upper right
    # overlap lower left is [max(A, E), max(B, F)], upper right is [min(C, G), min(D, H)]
    overlap = max(min(C,G) - max(A,E), 0) * max(min(D,H) - max(B,F), 0)
    return (A-C)*(B-D) + (E-G)*(F-H) - overlap

# LC836. Rectangle Overlap - intersection of 2 rectangles
def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
    A, B, C, D = rec1[0], rec1[1], rec1[2], rec1[3]
    E, F, G, H = rec2[0], rec2[1], rec2[2], rec2[3]
    x1 = max(A, E)
    y1 = max(B, F)
    x2 = min(C, G)
    y2 = min(D, H)
    return x1 < x2 and y1 < y2
