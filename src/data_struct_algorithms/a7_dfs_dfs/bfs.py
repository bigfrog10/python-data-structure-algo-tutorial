# BFS for extremes

# LC322. Coin Change
def coinChange(self, coins: List[int], amount: int) -> int:
    if amount == 0:  return 0 # 680 ms, it also is O(c^(amount / min(coins)))
    coins.sort(reverse=True) # we try to put larger coins to reduce numbers of coins
    queue, visited = deque([(0, 0)]), {0}
    while queue:
        for _ in range(len(queue)): # BFS
            amt, count = queue.popleft()
            for coin in coins:
                namt, nc = amt + coin, count + 1
                if namt == amount: return nc
                elif namt < amount and namt not in visited:
                    visited.add(namt)
                    queue.append((namt, nc))
    return -1





# LC518. Coin Change 2
def change(self, amount: int, coins: List[int]) -> int:
    if not coins: return 1 if amount == 0 else 0
    n = len(coins)
    dp = [1] + [0] * amount
    for c in coins:
        for i in range(c, amount+1): dp[i] += dp[i-c]
    return dp[amount]

# LC752. Open the Lock
def openLock(self, deadends: List[str], target: str) -> int:
    def nbs(digit): # neighbours
        d = int(digit)
        d1 = d - 1 if d > 0 else 9
        d2 = d + 1 if d < 9 else 0
        return str(d1), str(d2)
    terminals = set(deadends)
    queue, seen = deque([('0000', 0)]), {'0000'}  # BFS, num of turns
    while queue:
        state, level = queue.popleft()
        if state == target: return level
        if state in terminals: continue  # deadend, don't go further
        for i, s in enumerate(state):
            for nb in nbs(s):
                nstate = state[:i] + nb + state[i+1:]
                if nstate not in seen:
                    seen.add(nstate)
                    queue.append([nstate, level+1])
    return -1





# LC317. Shortest Distance from All Buildings
from itertools import product
def shortestDistance(self, grid):
    if not grid or not grid[0]: return -1
    n, m = len(grid), len(grid[0])
    # 0 for distance, 1 for counts/buildings
    matrix = [[[0,0] for _ in range(m)] for _ in range(n)]
    def bfs(start, blds):
        q = [(start, 0)] # 0 is the distance.
        while q:
            po, distance = q.pop(0)
            for dp in (-1,0), (1,0), (0,1), (0,-1):
                i, j = po[0]+dp[0], po[1]+dp[1]
                if 0 <= i <n and 0 <= j < m and matrix[i][j][1] == blds:
                    if grid[i][j] == 0:
                        matrix[i][j][0] += distance + 1
                        matrix[i][j][1] = blds + 1
                        q.append(([i,j], distance+1))
    blds = 0    # count how many building we have visited
    for i, j in product(range(n), range(m)):
        if grid[i][j] == 1: # loop through buildings
            bfs([i,j], blds)
            blds += 1
    res = float('inf')
    for i, j in product(range(len(matrix)), range(len(matrix[0]))):
        if matrix[i][j][1] == blds: res = min(res, matrix[i][j][0])
    return res if res!=float('inf') else -1

# LC286. Walls and Gates
def wallsAndGates(self, rooms: List[List[int]]) -> None:
    if not rooms:  return
    n, m = len(rooms), len(rooms[0])
    WALL, GATE, EMPTY = -1, 0, 2147483647
    gates = []  # find all gates
    for i, j in product(range(n), range(m)):
        if rooms[i][j] == GATE: gates.append((i, j))
    queue, steps = deque(gates), 0  # BFS
    while queue:
        print(queue)
        currl = len(queue)
        for k in range(currl):
            i, j = queue.popleft()
            if rooms[i][j] == EMPTY: rooms[i][j] = steps
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = i + dx, j + dy
                if 0 <= x < n and 0 <= y < m and rooms[x][y] == EMPTY:
                    queue.append((x, y))
        steps += 1

# LC983. Minimum Cost For Tickets
def mincostTickets(self, days: List[int], costs: List[int]) -> int:
    durations = [1, 7, 30]  # this should be in parameter passed in
    day_set = set(days)
    @lru_cache(None)
    def dp(i):
        if i > days[-1]: return 0
        elif i in day_set: # if we do travel today
            return min(dp(i + d) + c for c, d in zip(costs, durations))
        else:
            return dp(i + 1) # wait for next day if we don't travel today

    print(dp.cache_info())
    return dp(days[0])

# LC909. Snakes and Ladders
def snakesAndLadders(self, board: List[List[int]]) -> int:
    n = len(board)
    def coord(order):
        q, r = divmod(order-1, n)
        x = n - 1 - q
        y = r if q % 2 == 0 else n-1-r  # even and odd rows
        return x, y
    queue, visited = deque([(1, 0)]), set()  # order, steps
    maxs = n * n
    while queue: # BFS to get min
        x, s = queue.popleft()
        if x == maxs: return s
        if x in visited: continue
        visited.add(x)
        for i in range(6):
            move = x + i + 1
            if move > maxs: continue
            x1, y1 = coord(move)
            if board[x1][y1] != -1: move = board[x1][y1]
            if move not in visited:
                queue.append((move, s+1))
    return -1

# LC815. Bus Routes
def numBusesToDestination(self, routes, S, T): # BFS on stops
    stop2routes = collections.defaultdict(set)
    for i, route in enumerate(routes):
        for j in route: stop2routes[j].add(i) # label stop with bus route #
    queue, seen = [(S, 0)], {S}
    for stop, bus in queue:  # BFS for min stops
        if stop == T: return bus
        for route in stop2routes[stop]: # for each bus in this stop
            for st in routes[route]: # for each stop with this bus
                if st not in seen:
                    queue.append((st, bus + 1))
                    seen.add(st)
            routes[route] = []  # ***neat trick: seen route, cut buses for min
    return -1

# LC691. Stickers to Spell Word, effectively this is bfs since we look for min.
def minStickers(self, stickers: List[str], target: str) -> int:
    counters = [Counter(s) for s in stickers]
    @lru_cache(None)
    def dfs(target): # target string to child nodes by apply stickers
        res = float('inf')
        for sticker in counters: # DFS on stickers/neighbours
            if target[0] not in sticker: continue # to cut search branches
            targetnew = target
            for s in sticker: targetnew = targetnew.replace(s,'', sticker[s])
            if targetnew == '':
                res = 1
                break
            elif targetnew != target: res = min(res, 1 + dfs(targetnew))
        return res
    res = dfs(target)
    return -1 if res == float('inf') else res

# LC1129. Shortest Path with Alternating Colors
def shortestAlternatingPaths(self, n: int, red_edges: List[List[int]], blue_edges: List[List[int]]) -> List[int]:
    graph = defaultdict(lambda : defaultdict(set))
    red, blue = 0, 1
    for st, end in red_edges: graph[st][red].add(end)
    for st, end in blue_edges: graph[st][blue].add(end)
    res = [math.inf] * n
    q = deque([(0,red), (0,blue)])
    level = -1
    while q:
        level += 1
        size = len(q)
        for i in range(size):
            node, color = q.popleft()
            opp_color = color^1
            res[node] = min(level, res[node])
            neighbors = graph[node][opp_color]
            for child in list(neighbors):
                graph[node][opp_color].remove(child)
                q.append((child, opp_color))
    return [r if r != math.inf else -1 for r in res]


