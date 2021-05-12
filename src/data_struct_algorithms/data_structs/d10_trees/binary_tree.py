from collections import deque, defaultdict, Counter
from typing import List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# LC1361. Validate Binary Tree Nodes
def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool: # slower
    indegree = [0] * n  # Topological sort
    for left, right in zip(leftChild, rightChild):
        if left > -1: indegree[left] += 1
        if right > -1: indegree[right] += 1
        if indegree[left] > 1 or indegree[right] > 1: return False
    queue = collections.deque(i for i, d in enumerate(indegree) if d == 0)
    if len(queue) > 1: return False # this is root
    while queue: # start from root, remove root to generate new roots
        node = queue.popleft()
        for child in leftChild[node], rightChild[node]:
            if child == -1: continue
            indegree[child] -= 1
            if indegree[child] == 0: queue.append(child)
    return sum(indegree) == 0

# LC236. Lowest Common Ancestor of a Binary Tree, top100
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def dfs(node, p, q):
        if not node: return None
        left = dfs(node.left, p, q)
        right = dfs(node.right, p, q)
        if left and right: return node
        if node.val == p.val or node.val == q.val: return node
        return left or right

    return dfs(root, p, q)

# LC1650. Lowest Common Ancestor of a Binary Tree III
def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':  # follow parent
    cached = set([p, q])
    while p is not None or q is not None:
        if p is not None:
            p = p.parent
            if p in cached: return p
            elif p: cached.add(p)
        if q is not None:
            q = q.parent
            if q in cached: return q
            elif q: cached.add(q)
    return None
# LC536. Construct Binary Tree from String
def str2tree(self, s: str) -> TreeNode: # recursion
    idx = 0  # input is preorder traversal
    def scan(s):
        nonlocal idx
        if idx == len(s): return None
        j = idx
        while j < len(s) and s[j] != '(' and s[j] != ')': j += 1
        res = TreeNode(int(s[idx:j]))
        idx = j
        if idx < len(s) and s[idx] == '(':
            idx += 1 # skip (
            res.left = scan(s)
            idx += 1 # skip )
        if idx < len(s) and s[idx] == '(':
            idx += 1
            res.right = scan(s)
            idx += 1
        return res
    return scan(s)


# LC100. Same Tree
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    def inorder(node):
        if not node: return 'none'
        return f'{node.val}: [{node.left}, {node.right}]'
    s, t = inorder(p), inorder(q)
    return s == t

# LC572. Subtree of Another Tree
def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
    def preorder(node):
        if not node: return 'none'
        return f'[root={node.val},l={preorder(node.left)},r={preorder(node.right)}]'
    s1 = preorder(s)
    s2 = preorder(t)
    return s1.find(s2) > -1

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
        if not node: return True, 0 # is balanced, depth
        lb, lp = dfs(node.left)
        rb, rp = dfs(node.right)
        return lb and rb and abs(lp-rp) <= 1, max(lp, rp)+1
    bal, depth = dfs(root)
    return bal

# LC993. Cousins in Binary Tree
def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
    stk = root and [root]
    while stk:
        p = {c.val:node.val for node in stk for c in (node.left,node.right) if c}
        if x in p and y in p and p[x] != p[y]: return True
        stk = [child for node in stk for child in (node.left,node.right) if child]
    return False

# LC101. Symmetric Tree
def isSymmetric(self, root: TreeNode) -> bool:
    def is_mirror(n1, n2):
        if n1 is None and n2 is None: return True
        if n1 is None or n2 is None: return False
        return n1.val == n2.val and is_mirror(n1.left, n2.right)\
            and is_mirror(n1.right, n2.left)
    return is_mirror(root, root)

# LC226. Invert Binary Tree
def invertTree(self, root):
    if root is None: return None
    root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
    return root

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

# LC998. Maximum Binary Tree II
def insertIntoMaxTree(self, root, val):
    if root and root.val > val:
        root.right = self.insertIntoMaxTree(root.right, val)
        return root
    node = TreeNode(val)
    node.left = root
    return node

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

# LC1485. Clone Binary Tree With Random Pointer
def copyRandomBinaryTree(self, root: 'Node') -> 'NodeCopy':
    nodeArr = {}

    def dfs(root):
        if not root: return None
        if root in nodeArr: return nodeArr[root]
        nRoot = NodeCopy(root.val)
        nodeArr[root] = nRoot
        nRoot.left = dfs(root.left)
        nRoot.right = dfs(root.right)
        nRoot.random = dfs(root.random)
        return nRoot

    return dfs(root)

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
