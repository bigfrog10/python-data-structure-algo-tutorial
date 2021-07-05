
# LC399. Evaluate Division
def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    graph = defaultdict(dict)
    for (a, b), v in zip(equations, values):
        graph[a][b] = v
        graph[b][a] = 1.0 / v
    def dfs(x, y, w, visited):  # use dfs to find query values.
        visited.add(x)
        ns = graph[x]
        for var, weight in ns.items():
            if var == y: return w * weight
            if var not in visited:
                s = dfs(var, y, w*weight, visited)
                if s != 0: return s
        return 0
    ret = []
    for a, b in queries:
        r = dfs(a, b, 1, set())
        if r == 0: ret.append(-1)
        else: ret.append(r)
    return ret

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




# LC1059. All Paths from Source Lead to Destination # BBG
def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    g = defaultdict(set) # O(V)
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

# LC797. All Paths From Source to Target
# To return all solutions, we need to dfs with path  # BBG
def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    results, target = [], len(graph) - 1
    def backtrack(currNode, path): # we need all paths, so backtrack on path.
        if currNode == target:
            results.append(list(path)) # new path
            return
        for nextNode in graph[currNode]: # neighbours
            path.append(nextNode)
            backtrack(nextNode, path)
            path.pop() # backout
    backtrack(0, [0]) # 0 is starting point, [0] is current path
    return results

def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    target, res = len(graph) - 1, []
    def dfs(node, path):
        if node == target: res.append(path)
        else:
            for nei in graph[node]: dfs(nei, path + [nei])  # O(2^n)
    dfs(0, [0])
    return res

# LC133. Clone Graph
def cloneGraph(self, node: 'Node') -> 'Node':
    if not node: return None
    v2n = dict() # v is unique, new node references
    def dfs(node):
        if node.val in v2n: return v2n[node.val]
        nn = Node(node.val)
        v2n[node.val] = nn
        for ne in node.neighbors:
            nn.neighbors.append(dfs(ne))
        return nn
    nr = dfs(node)
    return nr



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


