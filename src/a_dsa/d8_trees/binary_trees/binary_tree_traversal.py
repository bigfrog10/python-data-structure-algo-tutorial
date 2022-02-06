
# LC314. Binary Tree Vertical Order Traversal
def verticalOrder(self, root: TreeNode) -> List[List[int]]:
    if root is None: return []  # O(n)
    queue, columnTable = collections.deque([(root, 0)]), collections.defaultdict(list)
    min_column = max_column = 0  # track column range for last line
    while queue:
        node, column = queue.popleft()
        columnTable[column].append(node.val)
        if node.left: queue.append((node.left, column - 1))
        if node.right: queue.append((node.right, column + 1))
        min_column = min(min_column, column)
        max_column = max(max_column, column)
    return [columnTable[x] for x in range(min_column, max_column + 1)]

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

# LC543. Diameter of Binary Tree
def diameterOfBinaryTree(self, root: TreeNode) -> int:
    diameter = 0
    def path_max(node):  # DFS
        nonlocal diameter
        if not node: return 0
        left = path_max(node.left)
        right = path_max(node.right)
        join = left + right  # path means edges, not nodes
        diameter = max(diameter, join)
        return 1 + max(left, right)  # plus this node
    path_max(root)
    return diameter

# LC103. Binary Tree Zigzag Level Order Traversal
def zigzagLevelOrder(self, root):
    ans, row, drxn = [], [root], 1
    while any(row):
        ans += [node.val for node in row][::drxn],
        row = [child for node in row for child in (node.left, node.right) if child]
        drxn *= -1
    return ans

# LC111. Minimum Depth of Binary Tree
def minDepth(self, root: Optional[TreeNode]) -> int:
    res, row = 0, [root]  # BFS for min
    while any(row):
        res += 1
        if any(not(child.left or child.right) for child in row): return res
        row = [child for node in row for child in (node.left, node.right) if child]
    return res

# LC545. Boundary of Binary Tree
def boundaryOfBinaryTree(self, root: TreeNode) -> List[int]:
    def left_bound(root): # O(n), loop all node once
       if not root: return
       if root.left or root.right: ans.append(root.val) # condition 4
       if root.left: left_bound(root.left) # condition 2
       else: left_bound(root.right) # condition 3
    def leaf(root):
       if not root: return
       leaf(root.left)
       if not root.left and not root.right: ans.append(root.val)
       leaf(root.right)
    def right_bound(root):
       if not root: return
       if not root.right: right_bound(root.left)
       else: right_bound(root.right)
       if root.left or root.right: ans.append(root.val) ## Reverse
    if not root: return []
    ans = []
    ans.append(root.val)
    left_bound(root.left)
    leaf(root.left)
    leaf(root.right)
    right_bound(root.right)
    return ans

# LC102. Binary Tree Level Order Traversal
def levelOrder(self, root):
    ans, level = [], [root]
    while root and level:
        ans.append([node.val for node in level])
        level = [kid for n in level for kid in (n.left, n.right) if kid]
    return ans

# LC1973. Count Nodes Equal to Sum of Descendants
def equalToDescendants(self, root: Optional[TreeNode]) -> int:
    output = 0
    def traverse(node):
        nonlocal output
        if node is None: return 0
        val_left = traverse(node.left)
        val_right = traverse(node.right)
        if node.val == val_left + val_right: output += 1
        return node.val + val_left + val_right
    traverse(root)
    return output

# LC101. Symmetric Tree
def isSymmetric(self, root: TreeNode) -> bool:
    def is_mirror(n1, n2):
        if n1 is None and n2 is None: return True
        if n1 is None or n2 is None: return False
        return n1.val == n2.val and is_mirror(n1.left, n2.right)\
            and is_mirror(n1.right, n2.left)
    return is_mirror(root, root)

# LC257. Binary Tree Paths
def binaryTreePaths(self, root: TreeNode) -> List[str]:
    ret = []
    def dfs(node, path):
        if not node: return
        path = path + '->' + str(node.val) if path else str(node.val)
        if not node.left and not node.right: # leaf
            ret.append(path)
            return
        dfs(node.left, path)
        dfs(node.right, path)
    dfs(root, '')
    return ret

# LC951. Flip Equivalent Binary Trees
def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool: # O(min(#nodes))
    if not root1 and not root2: return True
    if not root1 or not root2: return False
    if root1.val != root2.val: return False
    return (self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right) or
            self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left))

# LC662. Maximum Width of Binary Tree
def widthOfBinaryTree(self, root: TreeNode) -> int:
    width = 0
    level = [(1, root)]
    while level:
        width = max(width, level[-1][0] - level[0][0] + 1)
        level = [kid
                 for number, node in level
                 for kid in enumerate((node.left, node.right), 2 * number)
                 if kid[1]]
    return width

# LC655. Print Binary Tree - O(n)
def printTree(self, root: TreeNode) -> List[List[str]]:
    def get_height(node):
        return 0 if not node else 1 + max(get_height(node.left), get_height(node.right))
    def update_output(node, row, left, right):
        if not node: return
        mid = (left + right) // 2
        self.output[row][mid] = str(node.val)
        update_output(node.left, row + 1 , left, mid - 1)
        update_output(node.right, row + 1 , mid + 1, right)
    height = get_height(root)
    width = 2 ** height - 1
    self.output = [[''] * width for i in range(height)]
    update_output(node=root, row=0, left=0, right=width - 1)
    return self.output

# LC250. Count Univalue Subtrees
def countUnivalSubtrees(self, root: TreeNode) -> int:
    def dfs(node):
        if not node: return 0, True
        if not node.left and not node.right: # leaf
            return 1, True
        ln, lv = dfs(node.left)
        rn, rv = dfs(node.right)
        uni = (not node.left or node.left.val == node.val)
        uni = uni and (not node.right or node.right.val == node.val)
        if lv and rv and uni: cnts = ln + rn + 1
        else: cnts = ln + rn # ignore this node since val different
        return cnts, uni
    return dfs(root)[0]

# LC968. Binary Tree Cameras
def minCameraCover(self, root: TreeNode) -> int:
    # 0: not covered: 1: covered but not has a camera on it; 2: a camera on it.
    self.sum=0
    def dfs(node):
        if not node: return 1
        l = dfs(node.left)
        r = dfs(node.right)
        if l==0 or r==0:
            self.sum += 1 # need a camera to cover
            return 2
        elif l==2 or r==2: return 1
        else: return 0
    if dfs(root) == 0: self.sum += 1
    return self.sum

# LC366. Find Leaves of Binary Tree
def findLeaves(self, root: TreeNode) -> List[List[int]]:
    out = []  # DFS O(n) since we loop all node once
    def dfs(node):  # dfs on height - # number of edges from this node to deepest leaf.
        if not node: return -1 # so that leaf height is 0
        i = 1 + max(dfs(node.left), dfs(node.right))
        if i == len(out): out.append([])
        out[i].append(node.val)
        return i
    dfs(root)
    return out

# LC1740. Find Distance in a Binary Tree
def findDistance(self, root: TreeNode, p: int, q: int) -> int:
    ans = 0
    def fn(node):  # Traverse the tree post-order.
        nonlocal ans
        if not node: return False, -inf
        ltf, lv = fn(node.left)
        rtf, rv = fn(node.right)
        if node.val in (p, q) or ltf and rtf:
            if ltf: ans += lv + 1
            if rtf: ans += rv + 1
            return True, 0
        return ltf or rtf, max(lv, rv) + 1
    fn(root)
    return ans

# LC662. Maximum Width of Binary Tree
def widthOfBinaryTree(self, root: TreeNode) -> int:
    width = 0
    level = [(1, root)]
    while level:
        width = max(width, level[-1][0] - level[0][0] + 1)
        level = [kid
                 for number, node in level
                 for kid in enumerate((node.left, node.right), 2 * number)
                 if kid[1]]
    return width



# LC1315. Sum of Nodes with Even-Valued Grandparent
def sumEvenGrandparent(self, root: TreeNode) -> int:
    total = 0
    def dfs(node, parent): # carry parent is the key step
        nonlocal total
        if node.left:
            if parent and parent.val % 2 == 0: total += node.left.val
            dfs(node.left, node)
        if node.right:
            if parent and parent.val % 2 == 0: total += node.right.val
            dfs(node.right, node)
    dfs(root, None)
    return total

# LC104. Maximum Depth of Binary Tree
def maxDepth(self, root: TreeNode) -> int:
    if not root: return 0
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
def maxDepth(self, root):
    depth, stack = 0, [(1, root)]
    while stack != []:
        current_depth, root = stack.pop()
        if root:
            depth = max(depth, current_depth)
            stack.append((current_depth + 1, root.left))
            stack.append((current_depth + 1, root.right))
    return depth

# LC1448. Count Good Nodes in Binary Tree
def goodNodes(self, root: TreeNode) -> int:
    res = 0
    def dfs(node, vmax):
        nonlocal res
        if not node: return
        if node.val >= vmax: res +=1
        nmax = max(node.val, vmax)
        dfs(node.left, nmax)
        dfs(node.right, nmax)
    dfs(root, root.val)
    return res

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
    queue, seen = collections.deque([knode]), {knode}
    while queue: # BFS
        node = queue.popleft()
        if node:
            if len(graph[node]) <= 1: return node.val # leaf
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append(nei)

# LC863. All Nodes Distance K in Binary Tree
def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:  # O(n)
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
        visited.add(node)
        if d < k:
            for v in adj[node]:
                if v not in visited: dfs2(v, d + 1)
        else: res.append(node.val)  # ==k, no more recursion, so won't > k
    dfs2(target, 0)
    return res
