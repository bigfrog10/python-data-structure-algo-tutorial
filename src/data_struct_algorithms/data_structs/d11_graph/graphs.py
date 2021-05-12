# LC133. Clone Graph
def cloneGraph(self, node: 'Node') -> 'Node':
    if not node: return None
    v2n = dict() # v is unique, new node references
    def dfs(node):
        if node.val in v2n:
            nn = v2n[node.val]
            return nn
        nn = Node(node.val)
        v2n[node.val] = nn
        for ne in node.neighbors:
            nne = dfs(ne)
            nn.neighbors.append(nne)
        return nn
    nr = dfs(node)
    return nr

# LC785. Is Graph Bipartite?
def isBipartite(self, graph: List[List[int]]) -> bool:
    color = {}  # like seen in other cases
    for node in range(len(graph)): # go through each node
        if node in color: continue
        stack = [node] # DFS
        color[node] = 0 # paint color, component start
        while stack:
            node = stack.pop()
            for nei in graph[node]:
                if nei not in color:
                    stack.append(nei)
                    color[nei] = color[node] ^ 1
                # if child and parent have same color,
                elif color[nei] == color[node]: return False
    return True

# LC323. Number of Connected Components in an Undirected Graph
def countComponents(self, n, edges):
    graph = defaultdict(set) #{i: set() for i in range(n)}
    for v1, v2 in edges:
        graph[v1].add(v2)
        graph[v2].add(v1)
    seen = set()
    def bfs(node):  # get seen populated
        queue = [node]
        for n in queue:
            for nei in graph[n]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append(nei)
    count = 0
    for i in range(n):
        if i not in seen:
            bfs(i)
            count += 1
    return count

# LC1192. Critical Connections in a Network
def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
    network = [[] for _ in range(n)]
    for a, b in connections:
        network[a].append(b)
        network[b].append(a)
    disc, low = [-1]*n, [-1]*n  # discovery time and low link
    ret = []
    def tarjan(prev, node, time): # DFS on low
        # base case, we already visited this node, so don't need to touch it again.
        if disc[node] != -1: return disc[node]
        disc[node] = low[node] = time
        for x in network[node]:
            if x != prev:  # so we don't go back to parent/prev
                time += 1
                newLow = tarjan(node, x, time)
                low[node] = min(low[node], newLow)
                if low[x] > disc[node]: ret.append([node, x])  # find bridge, x has no circle back to node
        return low[node]
    tarjan(0, 0, 0)
    return ret

# LC863. All Nodes Distance K in Binary Tree
def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
    adj = collections.defaultdict(list)  # create graph
    def dfs(node):
        if node.left:
            adj[node].append(node.left)
            adj[node.left].append(node)
            dfs(node.left)
        if node.right:
            adj[node].append(node.right)
            adj[node.right].append(node)
            dfs(node.right)
    dfs(root)
    res, visited = [], set() # DFS with distance
    def dfs2(node, d):
        if d < K:
            visited.add(node)
            for v in adj[node]:
                if v not in visited: dfs2(v, d + 1)
        else: res.append(node.val)
    dfs2(target, 0)
    return res

# LC787. Cheapest Flights Within K Stops - weighted graph + heap
def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
    graph = defaultdict(list)  # Dijkstra
    for source,dest,cost in flights: graph[source].append((cost,dest))
    heap = []
    heapq.heappush(heap, (0, src, -1))  # add src to kick things off
    while heap:
        nextCost, nextDest, currSteps = heapq.heappop(heap)
        if currSteps > K: continue
        if nextDest == dst: return nextCost
        for nc,nd in graph[nextDest]:  # this is not optimal as dijkstra
            heapq.heappush(heap, (nc + nextCost, nd, currSteps + 1))
    return -1

# LC1135. Connecting Cities With Minimum Cost
def minimumCost(self, N: int, connections: List[List[int]]) -> int:
    G = collections.defaultdict(list)
    for city1, city2, cost in connections:  # O(E)
        G[city1].append((cost, city2))
        G[city2].append((cost, city1))
    queue, visited = [(0, N)], set()  # [1] Arbitrary starting point N costs 0.
    total = 0
    while queue: # [3] Exit if all cities are visited. O(VlogV)
        cost, city = heapq.heappop(queue)
        if city not in visited:
            visited.add(city)
            total += cost # [2] Grow tree by one edge.
            for edge in G[city]:
                heapq.heappush(queue, edge)
    return total if len(visited) == N else -1

# LC62. Unique Paths - no blocks
def uniquePaths(self, m: int, n: int) -> int:
    return math.comb(m+n-2, n-1) # select n-1 from m+n-2

def uniquePaths(self, m, n):
    if not m or not n: return 0
    cur = [1] * n  # first row
    for i in range(1, m):  # previous row + current row, 1st cell is always 1.
        for j in range(1, n): cur[j] += cur[j-1]  # previous cur[j] + current cur[j-1]
    return cur[-1]

# LC63. Unique Paths II - has blocks, from upper left to lower right
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    og = obstacleGrid
    n, m = len(og), len(og[0])
    if not m or not n: return 0
    idx = -1
    for i, c in enumerate(og[0]): # mark first row with blocks
        if c == 1:
            idx = i
            break
    idx = m if idx == -1 else idx
    cur = [1] * idx + [0] * (m - idx) # first there is a stone, block all rest
    for i in range(1, n): # start from 2nd row
        for j in range(0, m):
            if j == 0: # if previously not blocked, check now
                cur[j] = int(cur[j] != 0 and og[i][j] == 0)
            else:
                if og[i][j] == 1: cur[j] = 0
                else: cur[j] += cur[j-1]  # previous cur[j] + current cur[j-1]
    return cur[-1]

# LC980. Unique Paths III - has blocks, arbitrary start and end
def uniquePathsIII(self, A): # O(3^n)
    m, n, empty = len(A), len(A[0]), 1
    for i in range(m):  # find start x, y and count empty cells
        for j in range(n):
            if A[i][j] == 1: x, y = (i, j) # find start
            elif A[i][j] == 0: empty += 1 # count empty
    self.res = 0
    def dfs(x, y, empty): # DFS on cells and empty cell count
        if not (0 <= x < m and 0 <= y < n and A[x][y] >= 0):
            return # obstacles
        if A[x][y] == 2:
            self.res += empty == 0 # reach goal and touch all
            return
        A[x][y] = -2 # mark visited
        for i, j in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
            dfs(i, j, empty - 1)
        A[x][y] = 0  # backout, since we need check history fills all empty cells
    dfs(x, y, empty)
    return self.res



# Cycle Detection
# LC684. Redundant Connection - undirected graph
def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
    graph = collections.defaultdict(set)
    def dfs(source, target): # detect cycle
        if source not in seen:
            seen.add(source)
            if source == target: return True
            return any(dfs(nei, target) for nei in graph[source])
    for u, v in edges:
        seen = set()
        if u in graph and v in graph and dfs(u, v): return u, v
        graph[u].add(v)
        graph[v].add(u)

# LC685. Redundant Connection II - Directed graph
def findRedundantDirectedConnection(self, edges):  # [[2,3],[3,1],[3,4],[4,2]]
    parents = {}
    def find(u):  # union find
        if p[u] != u: p[u] = find(p[u])
        return p[u]
    def detect_cycle(edge):  # go from u to v (forms a cycle) along parents
        u, v = edge
        while u != v and u in parents: u = parents[u]
        return u == v
    candidates = []  # stores two edges from the vertex where it has two parents
    for u, v in edges:
        if v not in parents: parents[v] = u
        else:
            candidates.append((parents[v], v))
            candidates.append((u, v))
    if candidates:  # case 2 & case 3 where one vertex has two parents
        return candidates[0] if detect_cycle(candidates[0]) else candidates[1]
    # case 1, we just perform a standard union find, same as redundant-connection
    p = list(range(len(edges)+1)) # cycle
    for edge in edges:
        u, v = map(find, edge)
        if u == v: return edge
        p[u] = p[v]

# LC1559. Detect Cycles in 2D Grid
def containsCycle(self, grid: List[List[str]]) -> bool:  # best solution
    n, m = len(grid), len(grid[0])
    def dfs(i, j, seen):
        if (i, j) in seen: return True # we found a loop
        seen.add((i, j))
        for x, y in [(i+1, j), (i, j+1)]:
            if 0 <= x < n and 0 <= y < m and grid[x][y] == grid[i][j]:
                loop = dfs(x, y, seen)
                if loop: return True
        return False
    visited = set()
    for i, j in product(range(n), range(m)):
        if (i, j) not in visited:
            seen = set()
            loop = dfs(i, j, seen)
            if loop: return True
            visited.update(seen)
    return False

# LC1059. All Paths from Source Lead to Destination
def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    g = defaultdict(set)
    for [x,y] in edges: g[x].add(y)
    visited = defaultdict(int)
    def dfs(node):
        if visited[node] == 1: return True
        elif visited[node] == -1: return False
        elif len(g[node]) == 0: return node == destination
        else:
            visited[node] = -1 # not reach to dest
            for child in g[node]:
                if not dfs(child): return False
            visited[node] = 1 # reach dest
            return True
    return dfs(source)
# https://www.geeksforgeeks.org/eulerian-path-and-circuit/
