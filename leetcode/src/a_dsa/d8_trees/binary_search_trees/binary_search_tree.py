
# LC938. Range Sum of BST - works for count, average as well
def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
    ret = 0  # O(n)
    def dfs(node):
        nonlocal ret
        if not node: return
        if low <= node.val <= high: ret += node.val
        # if v < low, then all left children under v < low
        if node.val > low: dfs(node.left)
        if node.val < high: dfs(node.right)
    dfs(root)
    return ret

# LC173. Binary Search Tree Iterator
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        self._leftmost_inorder(root)
    def _leftmost_inorder(self, root):
        while root:
            self.stack.append(root)
            root = root.left
    def next(self) -> int:  # amortized O(1)
        topmost_node = self.stack.pop()
        if topmost_node.right: self._leftmost_inorder(topmost_node.right)
        return topmost_node.val
    def hasNext(self) -> bool:
        return len(self.stack) > 0

# LC270. Closest Binary Search Tree Value
def closestValue(self, root: Optional[TreeNode], target: float) -> int: # O(H)
    closest = root.val
    while root:
        closest = min(root.val, closest, key=lambda x: abs(target - x))
        # if target < root.val, then target to right side distance > target to root distance
        root = root.left if target < root.val else root.right
    return closest

# LC1305. All Elements in Two Binary Search Trees, merge trees
def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
    stack1, stack2, output = [], [], []  # O(m + n)
    while root1 or root2 or stack1 or stack2:
        while root1:
            stack1.append(root1)
            root1 = root1.left  # find smallest
        while root2:
            stack2.append(root2)  # find smallest
            root2 = root2.left
        if not stack2 or stack1 and stack1[-1].val <= stack2[-1].val:
            root1 = stack1.pop()
            output.append(root1.val)
            root1 = root1.right
        else:
            root2 = stack2.pop()
            output.append(root2.val)
            root2 = root2.right
    return output

# LC333. Largest BST Subtree
def largestBSTSubtree(self, root: TreeNode) -> int:
    highest = 0
    def find(node):  # post order, bottom up
        nonlocal highest
        if not node: return float('inf'), float('-inf'), 0
        lmin, lmax, lnum = find(node.left)
        rmin, rmax, rnum = find(node.right)
        n = float('-inf')  # to indicate this is not a bst
        if lmax < node.val < rmin:  # this is a bst
            n = lnum + rnum + 1
            highest = max(n, highest)
        return min(node.val, lmin), max(node.val, rmax), n
    find(root)
    return highest

# LC98. Validate Binary Search Tree
def isValidBST(self, root: TreeNode) -> bool:
    def validate(node, low=-math.inf, high=math.inf):
        if not node: return True
        if node.val <= low or node.val >= high: return False
        return validate(node.right, node.val, high) and\
               validate(node.left, low, node.val)
    return validate(root)

# LC1038. Binary Search Tree to Greater Sum Tree
def bstToGst(self, root: TreeNode) -> TreeNode:
    post_sum = 0
    def dfs(root):
        nonlocal post_sum
        if root:
            dfs(root.right)
            post_sum += root.val
            root.val = post_sum
            dfs(root.left)
    dfs(root)
    return root

# LC230. Kth Smallest Element in a BST
def kthSmallest(self, root: TreeNode, k: int) -> int:  # iterative
    stack = []  # O(H) space
    while True:  # O(H + k)
        while root:  # go all the way to left to smallest.
            stack.append(root)
            root = root.left
        root = stack.pop()  # from left bottom
        k -= 1
        if not k: return root.val
        root = root.right
    return None
def kthSmallest(self, root, k):  # O(n)
    def inorder(r):
        return inorder(r.left) + [r.val] + inorder(r.right) if r else []
    return inorder(root)[k - 1]

# LC285. Inorder Successor in BST
def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
    successor = None
    while root:
        if p.val >= root.val:
            root = root.right
        else:
            successor = root
            root = root.left
    return successor

# LC510. Inorder Successor in BST II - has parent
def inorderSuccessor(self, node: 'Node') -> 'Node':
    # the successor is somewhere lower in the right subtree
    if node.right:
        node = node.right
        while node.left: node = node.left
        return node
    # the successor is somewhere upper in the left tree
    while node.parent and node == node.parent.right: node = node.parent
    return node.parent  # first left

# LC235. Lowest Common Ancestor of a Binary Search Tree - lca
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    while root:
        if p.val < root.val > q.val:
            root = root.left
        elif p.val > root.val < q.val:
            root = root.right
        else:
            return root

# LC897. Increasing Order Search Tree
def increasingBST(self, root: TreeNode) -> TreeNode:  # O(n) runtime, O(H) space
    def rearrange(node, tail):
        if not node: return tail
        res = rearrange(node.left, node)  # left tree -> link list + root
        node.left = None
        node.right = rearrange(node.right, tail)  # right tree -> link list + tail
        return res
    return rearrange(root, None)

# LC95. Unique Binary Search Trees II
def generateTrees(self, n: int) -> List[Optional[TreeNode]]:  # O(n * catalan number), O(4^n)
    def trees(first, last):  # g(n) = sum [g(i-1) * g(n-i)] i=1..n, g(0) = g(1) = 1
        return [TreeNode(root, left, right)
                for root in range(first, last+1)
                for left in trees(first, root-1)
                for right in trees(root+1, last)] or [None]
    return trees(1, n)

# LC669. Trim a Binary Search Tree
def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
    def trim(node):  # O(n) time and space
        if not node: return None
        elif node.val > high: return trim(node.left)
        elif node.val < low: return trim(node.right)
        else:
            node.left = trim(node.left)
            node.right = trim(node.right)
            return node
    return trim(root)

# LC96. Unique Binary Search Trees  # BBG
def numTrees(self, n: int) -> int:  # O(4^n / n^(3/2)), Catalan numbers G_n
    G = [1, 1] + [0]*(n-1)  # g(i) number of unique trees with length i
    for i in range(2, n+1):  # O(n^2)
        # use j as top root, left options * right options
        for j in range(1, i+1): G[i] += G[j-1] * G[i-j]
    return G[n]

# LC450. Delete Node in a BST
def deleteNode(self, root, key):
    def DFS(node, target):
        if not node: return None
        if node.val == target:
            if node.left != None:
                l = node.left # go left
                while l and l.right: l = l.right  # find predecessor
                l.right = node.right  # predecessor next to node.right
                return node.left
            else: return node.right  # node is removed
        elif target > node.val: node.right = DFS(node.right, target)
        else: node.left = DFS(node.left, target)
        return node
    return DFS(root, key)

# LC701. Insert into a Binary Search Tree
def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode: # recursive
    if not root: return TreeNode(val)
    if val < root.val: root.left = self.insertIntoBST(root.left, val)
    else: root.right = self.insertIntoBST(root.right, val)
    return root

def inorderPredecessor(self, root, p):
    pred = None
    while root:
        if p.val >= root.val:
            pred = root
            root = root.right
        else:
            root = root.left
    return pred

# LC653. Two Sum IV - Input is a BST
def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
    if not root: return False  # O(n), BST is not used
    bfs, s = [root], set()
    for i in bfs:
        if k - i.val in s: return True
        s.add(i.val)
        if i.left:  bfs.append(i.left)  # we may have i from left and j from right with i+j=k
        if i.right: bfs.append(i.right)  # so we need to search both side.
    return False

# LC700. Search in a Binary Search Tree
def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    walker = root
    while walker:  # O(H)
        if walker.val == val: return walker
        elif walker.val > val: walker = walker.left
        else: walker = walker.right # <
    return walker  # None

# LC776. Split BST
def splitBST(self, root: TreeNode, V: int) -> List[TreeNode]:
    if not root: return None, None
    if root.val <= V:
        smaller, larger = self.splitBST(root.right, V)
        root.right = smaller
        return root, larger
    else:
        smaller, larger = self.splitBST(root.left, V)
        root.left = larger
        return smaller, root

# LC1902. Depth of BST Given Insertion Order
def maxDepthBST(self, order: List[int]) -> int:  # O(nlogn)
    depths = sortedcontainers.SortedDict()
    depths[-math.inf] = 0 # add dummy bounds to avoid extra ifs
    #depths[math.inf] = 0
    for x in order:
        i = depths.bisect_left(x)
        depths[x] = 1 + max(depths.values()[i - 1:i + 1])  # left and right subtrees
    return max(depths.values())

# LC1932. Merge BSTs to Create Single BST
def canMerge(self, trees: List[TreeNode]) -> Optional[TreeNode]:
    nodes = {}
    indeg = collections.defaultdict(int)
    for t in trees:
        if t.val not in indeg: indeg[t.val] = 0  # select 0 below in sources
        if t.left:
            indeg[t.left.val] += 1
            if t.left.val not in nodes: nodes[t.left.val] = t.left
        if t.right:
            indeg[t.right.val] += 1
            if t.right.val not in nodes: nodes[t.right.val] = t.right
        nodes[t.val] = t
    sources = [k for k, v in indeg.items() if v == 0]
    if len(sources) != 1: return None  # check single root

    self.cur = float('-inf')
    self.is_invalid = False
    seen = set()
    def inorder(val):
        seen.add(val)
        node = nodes[val]
        if node.left: node.left = inorder(node.left.val)
        # check inorder increasing
        if val <= self.cur:
            self.is_invalid = True
            return
        self.cur = val
        if node.right: node.right = inorder(node.right.val)
        return node

    root = inorder(sources[0])
    if len(seen) != len(nodes) or self.is_invalid:
        return None  # check full traversal
    return root

# LC783. Minimum Distance Between BST Nodes - min diff of values in nodes
def minDiffInBST(self, root: Optional[TreeNode]) -> int:
    def dfs(node):
        if node:
            dfs(node.left)
            self.ans = min(self.ans, node.val - self.prev)
            self.prev = node.val
            dfs(node.right)

    self.prev = float('-inf')
    self.ans = float('inf')
    dfs(root)
    return self.ans

# LC1214. Two Sum BSTs
def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
    stack, seen = [], set()  # O(n1 + n2) runtime, O(n1) space
    while stack or root1:
        while root1:
            stack.append(root1)
            root1 = root1.left
        root1 = stack.pop()
        seen.add(root1.val)
        root1 = root1.right
    while stack or root2:
        while root2:
            stack.append(root2)
            root2 = root2.left
        root2 = stack.pop()
        if target - root2.val in seen: return True
        root2 = root2.right
    return False

# LC449. Serialize and Deserialize BST
class Codec:
    def serialize(self, root):
        def preorder(node):
            if node:
                vals.append(str(node.val))
                preorder(node.left)
                preorder(node.right)
        vals = []
        preorder(root)
        return ' '.join(vals)
    def deserialize(self, data):
        preorder = list(map(int, data.split()))
        inorder = sorted(preorder)
        return self.buildTree(preorder, inorder)
    def buildTree(self, preorder, inorder):
        def build(stop):
            if inorder and inorder[-1] != stop:
                root = TreeNode(preorder.pop())
                root.left = build(root.val)
                inorder.pop()
                root.right = build(stop)
                return root
        preorder.reverse()
        inorder.reverse()
        return build(None)

# LC99. Recover Binary Search Tree
def recoverTree(self, root: Optional[TreeNode]) -> None:
    x = y = pred = None
    stack = []
    while stack or root: # we visited each node twice, push & pop, so O(N)
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop() # check
        if pred and root.val < pred.val:
            x = root
            if y is None: y = pred  # [1,3,null,null,2] to go further
            else: break # here we find them
        pred = root
        root = root.right # check right side
    x.val, y.val = y.val, x.val
