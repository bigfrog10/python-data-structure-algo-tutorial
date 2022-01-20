from collections import deque, defaultdict, Counter
from typing import List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# LC1650. Lowest Common Ancestor of a Binary Tree III
def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
    seen = {p, q}
    while p or q:
        if p:
            p = p.parent
            if p in seen: return p
            elif p: seen.add(p)
        if q:
            q = q.parent
            if q in seen: return q
            elif q: seen.add(q)
    return None
def lowestCommonAncestor1(self, p: 'Node', q: 'Node') -> 'Node':
    p1, p2 = p, q  # O(1) space, but goes a + b + c, still O(n)
    while p1 != p2:
        p1 = p1.parent if p1.parent else q
        p2 = p2.parent if p2.parent else p
    return p1

# LC236. Lowest Common Ancestor of a Binary Tree, top100
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def dfs(node, p, q):
        if not node: return None
        if node.val == p.val or node.val == q.val: return node
        left = dfs(node.left, p, q)
        right = dfs(node.right, p, q)
        if left and right: return node # both are not null, then this is LCA
        return left or right # carry the not None node
    return dfs(root, p, q)





# LC226. Invert Binary Tree
def invertTree(self, root):
    if root is None: return None
    root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
    return root

# LC572. Subtree of Another Tree
def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
    def preorder(node):
        if not node: return 'none'
        return f'[root={node.val},l={preorder(node.left)},r={preorder(node.right)}]'
    s1 = preorder(s)
    s2 = preorder(t)
    return s1.find(s2) > -1




# LC222. Count Complete Tree Nodes
def countNodes(self, root): # O((logn)^2)
    @lru_cache(None)
    def left_depth(root): # called logn times
        if not root: return 0
        return 1 + left_depth(root.left)
    if not root: return 0
    leftDepth = left_depth(root.left) # O(logn)
    rightDepth = left_depth(root.right)
    if leftDepth == rightDepth: # left is complete
        return pow(2, leftDepth) + self.countNodes(root.right)
    else: # right is complete
        return pow(2, rightDepth) + self.countNodes(root.left)

# LC513. Find Bottom Left Tree Value
def findBottomLeftValue(self, root: TreeNode) -> int: # O(n)
    queue = [root]
    for node in queue: # go right then left, so last node is left.
        queue += filter(None, (node.right, node.left))
    return node.val

# LC993. Cousins in Binary Tree
def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
    stk = [root]
    while stk:
        p = {c.val:node.val for node in stk for c in (node.left,node.right) if c}
        if x in p and y in p and p[x] != p[y]: return True
        stk = [child for node in stk for child in (node.left,node.right) if child]
    return False



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

# LC117. Populating Next Right Pointers in Each Node II
def connect(self, root: 'Node') -> 'Node':
    res = root
    while root:
        cur = leftmost = Node(0)
        while root:
            if root.left: # level travel at child level
                cur.next = root.left # connect to right at child level
                cur = root.left # level travel at child level
            if root.right:
                cur.next = root.right
                cur = root.right
            root = root.next # level travel at parent level
        root = leftmost.next # next level starting point
    return res

# LC1361. Validate Binary Tree Nodes
def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool: # slower
    indegree = [0] * n  # Topological sort
    for left, right in zip(leftChild, rightChild):
        if left > -1: indegree[left] += 1
        if right > -1: indegree[right] += 1
        # only 1 parent
        if indegree[left] > 1 or indegree[right] > 1: return False
    queue = deque(i for i, d in enumerate(indegree) if d == 0)
    if len(queue) > 1: return False # only 1 root
    while queue: # start from root, remove root to generate new roots
        node = queue.popleft()
        for child in leftChild[node], rightChild[node]:
            if child == -1: continue
            indegree[child] -= 1
            if indegree[child] == 0: queue.append(child)
    return sum(indegree) == 0

# LC100. Same Tree
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    def inorder(node):
        if not node: return 'none'
        return f'{node.val}: [{node.left}, {node.right}]'
    s, t = inorder(p), inorder(q)
    return s == t





# LC958. Check Completeness of a Binary Tree
def isCompleteTree(self, root):
    bfs = [root]
    i = 0
    while bfs[i]: # on exit, i is the first None we see.
        bfs.append(bfs[i].left)
        bfs.append(bfs[i].right)
        i += 1
    return not any(bfs[i:]) # we shouldn't have any non None after i

# LC110. Balanced Binary Tree
def isBalanced(self, root: TreeNode) -> bool:
    def dfs(node):
        if not node: return True, 0  # is balanced, depth
        lb, lp = dfs(node.left)
        rb, rp = dfs(node.right)
        return lb and rb and abs(lp-rp) <= 1, max(lp, rp)+1
    bal, depth = dfs(root)
    return bal



# LC617. Merge Two Binary Trees
def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
    if not t1: return t2
    if not t2: return t1
    t1.val += t2.val
    t1.left = self.mergeTrees(t1.left, t2.left)
    t1.right = self.mergeTrees(t1.right, t2.right)
    return t1

# LC814. Binary Tree Pruning
def pruneTree(self, root: TreeNode) -> TreeNode:
    def prune(node): # return true if all zeros
        if not node: return True
        lz = prune(node.left)
        rz = prune(node.right)
        if lz: node.left = None
        if rz: node.right = None
        return lz and rz and node.val == 0
    z = prune(root)
    return None if z else root

# LC654. Maximum Binary Tree
def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
    def dfs(arr):
        if not arr: return None
        if len(arr) == 1: return TreeNode(arr[0])
        mx, imx = float('-inf'), -1
        for i, x in enumerate(arr):
            if x > mx: mx, imx = x, i
        n = TreeNode(mx)
        n.left = dfs(arr[:imx])
        n.right = dfs(arr[imx+1:])
        return n
    return dfs(nums)

# LC998. Maximum Binary Tree II
def insertIntoMaxTree(self, root, val):
    if root and root.val > val:
        root.right = self.insertIntoMaxTree(root.right, val)
        return root
    node = TreeNode(val)
    node.left = root
    return node


