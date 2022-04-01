

# LC987. Vertical Order Traversal of a Binary Tree - sort in same position
def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:  # O(nlog(n/w)
    res = defaultdict(list) # column number to (row number, val)
    min_col = max_col = 0  # track column range
    def preorder(node, i, j):
        nonlocal min_col, max_col
        if not node: return
        res[j].append((i, node.val))
        min_col = min(min_col, j)
        max_col = max(max_col, j)
        preorder(node.left, i+1, j-1)
        preorder(node.right, i+1, j+1)
    preorder(root, 0, 0)
    # sort within cell
    ret = [[n[1] for n in sorted(res[k])] for k in range(min_col, max_col + 1)]
    return ret

# LC958. Check Completeness of a Binary Tree
def isCompleteTree(self, root):  # O(N) time and O(H) space
    def dfs(root):
        if not root: return 0
        l, r = dfs(root.left), dfs(root.right)
        if l & (l + 1) == 0 and l / 2 <= r <= l:
            return l + r + 1
        if r & (r + 1) == 0 and r <= l <= r * 2 + 1:
            return l + r + 1
        return -1
    return dfs(root) > 0
def isCompleteTree(self, root: Optional[TreeNode]) -> bool:  # O(N) time and space
    bfs, i = [root], 0
    while bfs[i]:  # on exit, i is the first None we see.
        bfs.extend([bfs[i].left, bfs[i].right])
        i += 1
    return not any(bfs[i:])  # we shouldn't have any non None after i

# LC545. Boundary of Binary Tree
def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:  # O(n) runtime, O(n) space(stack recursion)
    def left_bound(root):
        if not root: return
        if root.left or root.right: ans.append(root.val) # condition 1
        if root.left: left_bound(root.left) # condition 2
        else: left_bound(root.right) # condition 3
    def leaf(root):
        if not root: return
        leaf(root.left)
        if not root.left and not root.right: ans.append(root.val)
        leaf(root.right)
    def right_bound(root):  ## Reversed order
        if not root: return
        if root.right: right_bound(root.right)  # condition 2
        else: right_bound(root.left)  # condition 3
        if root.left or root.right: ans.append(root.val) # condition 1,
    if not root: return []
    ans = []
    ans.append(root.val)
    left_bound(root.left)
    leaf(root.left)
    leaf(root.right)
    right_bound(root.right)
    return ans

# LC742. Closest Leaf in a Binary Tree
def findClosestLeaf(self, root: TreeNode, k: int) -> int:  # O(n)
    graph = collections.defaultdict(list)
    knode = None
    def dfs(node, par = None): # convert to graph
        nonlocal knode
        if node:
            if node.val == k: knode = node
            graph[node].append(par)
            graph[par].append(node)
            dfs(node.left, node)
            dfs(node.right, node)
    dfs(root)
    queue, seen = collections.deque([knode]), {knode}
    while queue:  # BFS for shortest
        node = queue.popleft()
        if node:
            if len(graph[node]) <= 1: return node.val  # leaf
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append(nei)

# LC863. All Nodes Distance K in Binary Tree
def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:  # O(n) time
    adj = collections.defaultdict(list)  # create graph, O(V) space
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
    res, visited = [], set()  # DFS with distance
    def dfs2(node, d):
        visited.add(node)
        if d < k:
            for v in adj[node]:
                if v not in visited: dfs2(v, d + 1)
        else: res.append(node.val)  # ==k, no more recursion, so won't > k
    dfs2(target, 0)
    return res
