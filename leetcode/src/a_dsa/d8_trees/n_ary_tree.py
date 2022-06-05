
# LC1522. Diameter of N-Ary Tree - path = edges
def diameter(self, root: 'Node') -> int:
    ret = 0  # root itself, in case there is no child
    def dfs(node):  # return number of edges
        nonlocal ret  # edges
        if not node: return 0
        max1 = max2 = 0
        for child in node.children:
            depth = dfs(child)
            if max1 < depth: max1, max2 = depth, max1
            elif max2 < depth: max2 = depth
        ret = max(ret, max1 + max2)  # edges
        return max(max1, max2) + 1  # 1 is this node-to-child edge
    dfs(root)
    return ret

# LC582. Kill Process
def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
    d = collections.defaultdict(list)
    for c, p in zip(pid, ppid): d[p].append(c)
    bfs = [kill]
    for i in bfs: bfs.extend(d.get(i, []))
    return bfs

# LC427. Construct Quad Tree
def construct(self, grid: List[List[int]]) -> 'Node':
    def build(r1, c1, r2, c2):
        if r1 > r2 or c1 > c2: return None
        # leaf if all cells have same value
        isLeaf = all(grid[i][j] == grid[r1][c1] for i in range(r1, r2+1) for j in range(c1, c2+1))
        if isLeaf: return Node(grid[r1][c1], True, None, None, None, None)

        rowMid, colMid = (r1 + r2) // 2, (c1 + c2) // 2
        return Node(False, False,
            build(r1, c1, rowMid, colMid),           # top left
            build(r1, colMid + 1, rowMid, c2),       # top right
            build(rowMid + 1, c1, r2, colMid),       # bottom left
            build(rowMid + 1, colMid + 1, r2, c2))  # bottom right

    return build(0, 0, len(grid)-1, len(grid[0]) - 1)

# LC428. Serialize and Deserialize N-ary Tree
class Codec:
    def serialize(self, root: 'Node') -> str:
        if not root: return ''
        def node_to_json(node):
            ret = f'"{node.val}":['
            for child in node.children:
                ret += node_to_json(child) + ','
            if ret[-1] == ',': ret = ret[:-1]
            ret += ']'
            ret = '{' + ret + '}'
            return ret
        ret = node_to_json(root)
        return ret
    def deserialize(self, data: str) -> 'Node':
        if not data: return None
        def dict_to_node(kvs): #DFS
            if kvs is None: return None  # base case
            for k, v in kvs.items():
                tn = Node(int(k), [])
                for value in v:
                    tn.children.append(dict_to_node(value))
                return tn
        kv = json.loads(data)
        return dict_to_node(kv)



# LC1245. Tree Diameter - given graph edges
def treeDiameter(self, edges: List[List[int]]) -> int:
    graph = defaultdict(set)  # topological sort, n-ary tree
    for edge in edges:
        u, v = edge
        graph[u].add(v)
        graph[v].add(u)
    vertex_left = len(graph)
    leaves = [i for i in range(vertex_left) if len(graph[i]) == 1]
    layers = 0
    while vertex_left > 2:
        vertex_left -= len(leaves)
        next_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            if len(graph[neighbor]) == 1: next_leaves.append(neighbor)
        layers += 1
        leaves = next_leaves
    return layers * 2 + (1 if vertex_left == 2 else 0)



# LC261. Graph Valid Tree
def validTree(self, n: int, edges: List[List[int]]) -> bool:
    if not edges: return n <= 1
    if n-1 != len(edges): return False  # a tree has to have only n-1 edges
    graph = defaultdict(list)
    for e in edges:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[0])
    stack, seen = [0], {0}  # start from node 0
    while stack: # DFS to check is fully connected
        e = stack.pop()
        for ne in graph[e]:
            if ne not in seen:
                seen.add(ne)
                stack.append(ne)
    return len(seen) == n

# LC1490. Clone N-ary Tree
def cloneTree(self, root: 'Node') -> 'Node':
    if root is None: return None
    return Node(root.val, [self.cloneTree(child) for child in root.children])

# LC310. Minimum Height Trees - n-ary tree
def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
    if n <= 2: return [i for i in range(n)] # topological sort, n-ary tree
    neighbors = defaultdict(set)
    for start, end in edges:
        neighbors[start].add(end)
        neighbors[end].add(start)
    leaves = [i for i in range(n) if len(neighbors[i]) == 1]
    while n > 2:  # topologic sort, 3 [[0,1],[0,2]] --> [0]
        n -= len(leaves)
        new_leaves = []
        for lv in leaves:
            nb = neighbors[lv].pop() # leave has only 1 neighbour - parent
            neighbors[nb].remove(lv)
            if len(neighbors[nb]) == 1: new_leaves.append(nb)
        leaves = new_leaves
    return leaves  # last leaves, could be 1 or 2 leaves
