
# LC983. Minimum Cost For Tickets
def mincostTickets(self, days, costs):   # O(len(days))
    durations = [1, 7, 30]
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

# LC2092. Find All People With Secret
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

# LC1319. Number of Operations to Make Network Connected
def makeConnected(self, n: int, connections: List[List[int]]) -> int:  # O(m) m = len(conns)
    if len(connections) < n - 1: return -1
    G = [set() for i in range(n)]
    for i, j in connections:  # space could be n^2, for fully connected net
        G[i].add(j)
        G[j].add(i)
    seen = [0] * n
    def dfs(i):
        if seen[i]: return 0
        seen[i] = 1
        for j in G[i]: dfs(j)
        return 1
    # the number of connected networks - 1 is what we need to do to connect them
    return sum(dfs(i) for i in range(n)) - 1

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

# LC1834. Single-Threaded CPU
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

# LC1376. Time Needed to Inform All Employees
def numOfMinutes(self, n, headID, manager, informTime):
    reports = defaultdict(list)
    for i, m in enumerate(manager):
        if m >= 0: reports[m].append(i)

    def dfs(i):
        return max([dfs(j) for j in reports[i]] or [0]) + informTime[i]
    return dfs(headID)
