from collections import deque, defaultdict, Counter
from typing import List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# LC94. Binary Tree Inorder Traversal
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    def inorder(root):  # recursive
        return inorder(root.left) + [root.val] + inorder(root.right) if root else []
    def preorder(root):
        return [root.val] + preorder(root.left) + preorder(root.right) if root else []
    def postorder(root):
        return postorder(root.left) + postorder(root.right) + [root.val] if root else []
    return inorder(root)
def inorderTraversal(self, root):
    ans, stack = [], []
    while stack or root:  # iterative
        if root:  # deal with left
            stack.append(root)
            root = root.left
        else:
            node = stack.pop()
            ans.append(node.val)  # inorder, add after all left children
            root = node.right  # deal with right
    return ans

# LC145. Binary Tree Postorder Traversal

# LC144. Binary Tree Preorder Traversal

# LC1110. Delete Nodes And Return Forest
def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
    res, to_delete_set = [], set(to_delete)
    def dfs(root, is_root):
        if not root: return None
        root_deleted = root.val in to_delete_set
        if is_root and not root_deleted: res.append(root)

        root.left = dfs(root.left, root_deleted) # if root is deleted, then left and right are roots.
        root.right = dfs(root.right, root_deleted)
        return None if root_deleted else root
    dfs(root, True)
    return res

# LC1650. Lowest Common Ancestor of a Binary Tree III
def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
    path = set()  # store the parents (path) from root to p, and then check q's path
    while p:  # O(h)
        path.add(p)
        p = p.parent
    while q not in path: q = q.parent
    return q
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
        if left and right: return node  # both are not null, then this is LCA
        return left or right  # carry the not None node
    return dfs(root, p, q)

# LC1644. Lowest Common Ancestor of a Binary Tree II
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    res = None
    def dfs(node):  # return how many matches
        nonlocal res
        if not node: return 0
        cur = node == p or node == q # count of matches
        left = dfs(node.left)
        right = dfs(node.right)
        if cur+left+right == 2 and not res: res = node # and condition ensures lowest
        return cur+left+right
    dfs(root)
    return res

# LC1123. Lowest Common Ancestor of Deepest Leaves
def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    def dfs(node):  #  O(n)
        if not node: return 0, None
        h1, lca1 = dfs(node.left)
        h2, lca2 = dfs(node.right)
        if h1 > h2: return h1 + 1, lca1
        if h1 < h2: return h2 + 1, lca2
        return h1 + 1, node
    return dfs(root)[1]

# LC1361. Validate Binary Tree Nodes
def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool: # slower
    indegree = [0] * n  # Topological sort  O(n)
    for left, right in zip(leftChild, rightChild):
        if left > -1: indegree[left] += 1
        if right > -1: indegree[right] += 1
        if indegree[left] > 1 or indegree[right] > 1: return False  # only 1 parent
    queue = deque(i for i, d in enumerate(indegree) if d == 0)
    if len(queue) > 1: return False  # only 1 root
    while queue:  # start from root, remove root to generate new roots
        node = queue.popleft()
        for child in leftChild[node], rightChild[node]:
            if child == -1: continue  # no child
            indegree[child] -= 1
            if indegree[child] == 0: queue.append(child)
    return sum(indegree) == 0

# LC572. Subtree of Another Tree
def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
    def preorder(node):
        if not node: return 'none'
        return f'[root={node.val},l={preorder(node.left)},r={preorder(node.right)}]'
    s1 = preorder(s)
    s2 = preorder(t)
    return s1.find(s2) > -1

# LC814. Binary Tree Pruning
def pruneTree(self, root: TreeNode) -> TreeNode:
    def prune(node):  # return true if all zeros
        if not node: return True
        lz = prune(node.left)
        rz = prune(node.right)
        if lz: node.left = None
        if rz: node.right = None
        return lz and rz and node.val == 0
    z = prune(root)
    return None if z else root

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

# LC110. Balanced Binary Tree
def isBalanced(self, root: TreeNode) -> bool:
    def dfs(node):
        if not node: return True, 0  # is balanced, depth
        lb, lp = dfs(node.left)
        rb, rp = dfs(node.right)
        return lb and rb and abs(lp-rp) <= 1, max(lp, rp)+1
    bal, depth = dfs(root)
    return bal

# LC337. House Robber III
def rob(self, root: Optional[TreeNode]) -> int:
    def helper(node):
        if not node: return (0, 0)  # return [rob this node, not rob this node]
        left = helper(node.left)
        right = helper(node.right)
        # if we rob this node, we cannot rob its children
        rob = node.val + left[1] + right[1]
        # else we could choose to either rob its children or not
        not_rob = max(left) + max(right)
        return [rob, not_rob]
    return max(helper(root))

# LC100. Same Tree
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    def inorder(node):
        if not node: return 'none'
        return f'{node.val}: [{inorder(node.left)}, {inorder(node.right)}]'
    s, t = inorder(p), inorder(q)
    return s == t
def isSameTree(self, p, q):
    if not p and not q: return True
    if not q or not p: return False
    if p.val != q.val: return False
    return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)

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



# LC958. Check Completeness of a Binary Tree
def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
    bfs, i = [root], 0
    while bfs[i]:  # on exit, i is the first None we see.
        bfs.extend([bfs[i].left, bfs[i].right])
        i += 1
    return not any(bfs[i:])  # we shouldn't have any non None after i

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
