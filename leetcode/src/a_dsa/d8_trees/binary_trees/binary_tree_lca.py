
# LC1650. Lowest Common Ancestor of a Binary Tree III - has parent, given 2 nodes. tree LCA
def lowestCommonAncestor1(self, p: 'Node', q: 'Node') -> 'Node':
    p1, p2 = p, q  # O(1) space, but goes a + b + c, still O(2h)
    while p1 != p2:
        p1 = p1.parent if p1.parent else q
        p2 = p2.parent if p2.parent else p
    return p1

# LC236. Lowest Common Ancestor of a Binary Tree - normal node, in the tree  LCA  lca bt
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def dfs(node):  ## O(n) time and space
        if not node: return None
        if node == p or node == q: return node
        left, right = dfs(node.left), dfs(node.right)
        if left and right: return node  # both are not null, then this is LCA
        return left or right  # carry the not None node
    return dfs(root)

# LC1644. Lowest Common Ancestor of a Binary Tree II, normal node, might not be in tree  lca bt
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    res = None
    def dfs(node):  # return how many matches
        nonlocal res
        if res: return 0
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

# LC865. Smallest Subtree with all the Deepest Nodes
def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
    def deep(root):
        if not root: return 0, None  # depth, node
        l, r = deep(root.left), deep(root.right)
        if l[0] > r[0]: return l[0] + 1, l[1]
        elif l[0] < r[0]: return r[0] + 1, r[1]
        else: return l[0] + 1, root
    return deep(root)[1]

# LC2096. Step-By-Step Directions From a Binary Tree Node to Another - LR direction
def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
    def find(n: TreeNode, val: int, path: List[str]) -> bool:  # root to this node
        if n.val == val:
            return True
        if n.left and find(n.left, val, path):
            path += "L"
        elif n.right and find(n.right, val, path):
            path += "R"
        return path

    s, d = [], []
    find(root, startValue, s)
    find(root, destValue, d)
    while len(s) and len(d) and s[-1] == d[-1]:  # clean up for LCA
        s.pop()
        d.pop()
    return "".join("U" * len(s)) + "".join(reversed(d))
