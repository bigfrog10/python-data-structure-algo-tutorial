
# LC2092. Find All People With Secret
def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
    can = {0, firstPerson}  # O(mlogm), m = # of meetings.
    for _, grp in itertools.groupby(sorted(meetings, key=lambda x: x[2]), key=lambda x: x[2]):
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
def makeConnected(self, n: int, connections: List[List[int]]) -> int:
    if len(connections) < n - 1: return -1
    G = [set() for i in range(n)]
    for i, j in connections:
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
def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
    uf = {i: i for i in range(n + 1)}  # O((N+M) * log(N+M)), because of sorting
    def find(x):
        if x != uf[x]: uf[x] = find(uf[x])
        return uf[x]
    # think well cost is the pipe cost from this house to house 0
    w = [[c, 0, i] for i, c in enumerate(wells, 1)]
    p = [[c, i, j] for i, j, c in pipes]
    res = 0
    for c, x, y in sorted(w + p):  # greedy, from small to large costs
        x, y = find(x), find(y)
        if x != y:
            uf[find(x)] = find(y)
            res += c
            n -= 1
        if n == 0: return res

# LC1834. Single-Threaded CPU
def getOrder(self, tasks: List[List[int]]) -> List[int]:
    tasks = sorted([(t[0], t[1], i) for i, t in enumerate(tasks)])
    i = 0
    res, h = [], []  # heap
    time = tasks[0][0]
    while len(res) < len(tasks):
        while i < len(tasks) and tasks[i][0] <= time:  # available task for now
            heapq.heappush(h, (tasks[i][1], tasks[i][2])) # (processing_time, original_index)
            i += 1
        if h:
            t_diff, original_index = heapq.heappop(h)
            time += t_diff
            res.append(original_index)
        elif i < len(tasks): time = tasks[i][0]
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
