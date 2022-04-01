
# LC1650. Lowest Common Ancestor of a Binary Tree III - has parent, given 2 nodes
def lowestCommonAncestor1(self, p: 'Node', q: 'Node') -> 'Node':
    p1, p2 = p, q  # O(1) space, but goes a + b + c, still O(2h)
    while p1 != p2:
        p1 = p1.parent if p1.parent else q
        p2 = p2.parent if p2.parent else p
    return p1

# LC236. Lowest Common Ancestor of a Binary Tree - normal node, in the tree
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def dfs(node, p, q):  ## O(n) time and space
        if not node: return None
        if node == p or node == q: return node
        left = dfs(node.left, p, q)
        right = dfs(node.right, p, q)
        if left and right: return node  # both are not null, then this is LCA
        return left or right  # carry the not None node
    return dfs(root, p, q)

# LC1644. Lowest Common Ancestor of a Binary Tree II, normal node, might not in tree
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

# LC2096. Step-By-Step Directions From a Binary Tree Node to Another
def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
    def find(n: TreeNode, val: int, path: List[str]) -> bool:
        if n.val == val:
            return True
        if n.left and find(n.left, val, path):
            path += "L"
        elif n.right and find(n.right, val, path):
            path += "R"
        return path
    ######################################################
    def lca(node):
        """Return lowest common ancestor of start and dest nodes."""
        if not node or node.val in (startValue , destValue): return node
        left, right = lca(node.left), lca(node.right)
        return node if left and right else left or right

    root = lca(root) # only this sub-tree matters
    ######################################################
    s, d = [], []
    find(root, startValue, s)
    find(root, destValue, d)
    while len(s) and len(d) and s[-1] == d[-1]:  # clean up for LCA
        s.pop()
        d.pop()
    return "".join("U" * len(s)) + "".join(reversed(d))
