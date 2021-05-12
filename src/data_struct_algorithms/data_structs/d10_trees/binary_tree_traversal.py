# LC199. Binary Tree Right Side View
def rightSideView(self, root: TreeNode) -> List[int]:
    if not root: return []
    ret = []
    def traverse(node, depth):
        if depth == len(ret): ret.append(node.val)
        for n in [node.right, node.left]:
            if n: traverse(n, depth+1)
    traverse(root, 0)
    return ret

# LC314. Binary Tree Vertical Order Traversal
def verticalOrder(self, root: TreeNode) -> List[List[int]]:
    columnTable = defaultdict(list)
    queue = deque([(root, 0)])
    while queue:
        node, column = queue.popleft()
        if node is not None:
            columnTable[column].append(node.val)
            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))
    return [columnTable[x] for x in sorted(columnTable.keys())] # O(nlogn) sort

# LC987. Vertical Order Traversal of a Binary Tree
def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
    res = defaultdict(list) # column number to (row number, val)
    def preorder(node, i, j):
        if not node: return
        res[j].append((i, node.val))
        preorder(node.left, i+1, j-1)
        preorder(node.right, i+1, j+1)
    preorder(root, 0, 0)
    cols = sorted(res.keys())
    ret = [[n[1] for n in sorted(res[k])] for k in cols]
    return ret

# LC102. Binary Tree Level Order Traversal
def levelOrder(self, root):
    ans, level = [], [root]
    while root and level:
        ans.append([node.val for node in level])
        level = [kid for n in level for kid in (n.left, n.right) if kid]
    return ans

# LC103. Binary Tree Zigzag Level Order Traversal
def zigzagLevelOrder(self, root):
    ans, row, drxn = [], [root], 1
    while any(row):
        ans += [node.val for node in row][::drxn],
        row = [child for node in row for child in (node.left, node.right) if child]
        drxn *= -1
    return ans

# LC116. Populating Next Right Pointers in Each Node
def connect(self, root: 'Node') -> 'Node':  # This is better written
    if not root: return root
    Q = deque([root])
    while Q: # BFS
        size = len(Q)
        for i in range(size):  # BFS
            node = Q.popleft()
            if i < size - 1: node.next = Q[0] # assign to next
            if node.left: Q.append(node.left)
            if node.right: Q.append(node.right)
    return root

# LC1367. Linked List in Binary Tree
def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
    def dfs(head, root):
        if not head: return True
        if not root: return False
        return root.val == head.val and (dfs(head.next, root.left) or dfs(head.next, root.right))
    if not head: return True
    if not root: return False
    return dfs(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)

# LC742. Closest Leaf in a Binary Tree
def findClosestLeaf(self, root: TreeNode, k: int) -> int:
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
    queue, seen = collections.deque([knode]), set([knode])
    while queue: # BFS
        node = queue.popleft()
        if node:
            if len(graph[node]) <= 1: return node.val # leaf
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append(nei)

# LC545. Boundary of Binary Tree
def boundaryOfBinaryTree(self, root: TreeNode) -> List[int]:
    # 4th condition is useless
    if not root: return []
    boundary = [root.val]
    def dfs(node, left_bd, right_bd):
        if not node: return
        # append when going down from the left or at leaf node
        if (not node.left and not node.right) or left_bd:
            boundary.append(node.val)
        if node.left and node.right:
            dfs(node.left, left_bd, False)
            dfs(node.right, False, right_bd)
        else:
            dfs(node.left,  left_bd, right_bd)
            dfs(node.right, left_bd, right_bd)
        # append to boundary when coming up from the right
        # use and here to avoid duplicates - leaf and right side
        if (node.left or node.right) and right_bd:
            boundary.append(node.val)
    dfs(root.left, True, False)
    dfs(root.right, False, True)
    return boundary
