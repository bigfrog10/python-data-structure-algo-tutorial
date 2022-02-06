# BFS for extremes







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
