# LC938. Range Sum of BST
def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
    ret = 0
    def traverse(node):
        nonlocal ret
        if not node: return
        # if v < low, then all left children under v < low
        if node.val > low: traverse(node.left)
        if low <= node.val <= high: ret += node.val
        if node.val < high: traverse(node.right)
    traverse(root)
    return ret

# LC270. Closest Binary Search Tree Value
def closestValue(self, root: TreeNode, target: float) -> int:  # Iterative
    closest, nv = float('inf'), None
    stack = [root]
    while stack:
        node = stack.pop()
        if not node: continue
        if node.val == target: return node.val
        diff = abs(node.val - target)
        if diff < closest:
            closest = min(closest, diff)
            nv = node.val
        if node.val < target: stack.append(node.right)
        else: stack.append(node.left)
    return nv

# LC98. Validate Binary Search Tree
def isValidBST(self, root: TreeNode) -> bool:
    queue = deque([(root, float('-inf'), float('inf'))])
    while queue:
        node, minv, maxv = queue.popleft()
        if node.left:
            if node.left.val >= node.val: return False
            if node.left.val <= minv: return False
            queue.append((node.left, minv, node.val))
        if node.right:
            if node.right.val <= node.val: return False
            if node.right.val >= maxv: return False
            queue.append((node.right, node.val, maxv))
    return True

# LC700. Search in a Binary Search Tree
def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    walker = root
    while walker:  # O(H)
        if walker.val == val: return walker
        elif walker.val > val: walker = walker.left
        else: walker = walker.right # <
    return walker  # None

# LC333. Largest BST Subtree
def largestBSTSubtree(self, root: TreeNode) -> int:
    highest = 0
    def find(node): # post order, bottom up
        nonlocal highest
        if not node:  return float('inf'), float('-inf'), 0
        lmin, lmax, lnum = find(node.left)
        rmin, rmax, rnum = find(node.right)
        n = float('-inf')  # to indicate this is not a bst
        if lmax < node.val < rmin:  # this is a bst
            n = lnum + rnum + 1
            highest = max(n, highest)
        return min(node.val, lmin), max(node.val, rmax), n
    find(root)
    return highest

# LC1382. Balance a Binary Search Tree
def balanceBST(self, root: TreeNode) -> TreeNode:
    def inorder(node, li):
        if not node: return
        inorder(node.left, li)
        li.append(node)
        inorder(node.right, li)
    def buildBst(li, start, end):
        if not li or start > end: return None
        mid = start + (end - start) // 2
        root = li[mid]
        root.left = buildBst(li, start, mid - 1)
        root.right = buildBst(li, mid + 1, end)
        return root
    if not root: return None
    nodes = []
    inorder(root, nodes)
    return buildBst(nodes, 0, len(nodes) - 1)

def inorderPredecessor(self, root, p):
    pred = None
    while root:
        if p.val >= root.val:
            pred = root
            root = root.right
        else:
            root = root.left
    return pred
# LC285. Inorder Successor in BST
def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
    successor = None
    while root:
        if p.val >= root.val: root = root.right
        else:
            successor = root
            root = root.left
    return successor

# LC510. Inorder Successor in BST II
def inorderSuccessor(self, node: 'Node') -> 'Node':
    # the successor is somewhere lower in the right subtree
    if node.right:
        node = node.right
        while node.left: node = node.left
        return node
    # the successor is somewhere upper in the tree
    while node.parent and node == node.parent.right: node = node.parent
    return node.parent  # first left

# LC108. Convert Sorted Array to Binary Search Tree
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    def to_bst(nums, b, e) -> TreeNode:
        if b > e: return None
        mid = b + (e - b) // 2  # if len even, use 2nd or 1st
        root = TreeNode(nums[mid], None, None)
        root.left = to_bst(nums, b, mid-1)
        root.right = to_bst(nums, mid + 1, e)
        return root
    return to_bst(nums, 0, len(nums)-1)

# LC426. Convert Binary Search Tree to Sorted Doubly Linked List
def treeToDoublyList(self, root: 'Node') -> 'Node':  # best solution
    if not root: return None
    prev = dummy = Node(-1)
    def inorder(node):
        nonlocal prev
        if not node: return
        inorder(node.left)
        prev.right = node
        node.left = prev
        prev = node  # this is the trick
        inorder(node.right)
    inorder(root)

    prev.right = dummy.right
    dummy.right.left = prev
    return dummy.right

# LC173. Binary Search Tree Iterator
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        self._leftmost_inorder(root)
    def _leftmost_inorder(self, root):
        while root:
            self.stack.append(root)
            root = root.left
    def next(self) -> int:
        topmost_node = self.stack.pop()
        if topmost_node.right: self._leftmost_inorder(topmost_node.right)
        return topmost_node.val
    def hasNext(self) -> bool:
        return len(self.stack) > 0

# LC230. Kth Smallest Element in a BST
def kthSmallest(self, root: TreeNode, k: int) -> int:  # iterative
    stack = []
    while True:
        while root: # go all the way to left to smallest.
            stack.append(root)
            root = root.left
        root = stack.pop() # from left bottom
        k -= 1
        if not k: return root.val
        root = root.right
    return None

# LC450. Delete Node in a BST
def deleteNode(self, root, key):
    def DFS(node, target):
        if not node: return None
        if node.val == target:
            l = node.left # go left
            while l and l.right: l = l.right # find predecessor
            if node.left != None:
                l.right = node.right # predecessor next to node.right
                return node.left
            else: return node.right # node is removed
        elif target > node.val: node.right = DFS(node.right, target)  #use property of BST
        else: node.left = DFS(node.left, target)
        return node
    return DFS(root, key)

# LC701. Insert into a Binary Search Tree
def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode: # recursive
    if not root: return TreeNode(val)
    if val < root.val: root.left = self.insertIntoBST(root.left, val)
    else: root.right = self.insertIntoBST(root.right, val)
    return root

# LC776. Split BST
def splitBST(self, root: TreeNode, V: int) -> List[TreeNode]:
    if not root: return None, None
    elif root.val <= V:
        bns = self.splitBST(root.right, V)
        root.right = bns[0]
        return root, bns[1]
    else:
        bns = self.splitBST(root.left, V)
        root.left = bns[1]
        return bns[0], root

# LC96. Unique Binary Search Trees
def numTrees(self, n: int) -> int:
    G = [0]*(n+1)
    G[0], G[1] = 1, 1
    for i in range(2, n+1): # number of BST with i as root
        for j in range(1, i+1): G[i] += G[j-1] * G[i-j]
    return G[n]
