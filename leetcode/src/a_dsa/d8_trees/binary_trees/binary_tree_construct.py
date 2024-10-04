
# LC894. All Possible Full Binary Trees
def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
    @functools.cache
    def fbt(m):
        if m == 1: return [TreeNode(0)]
        res = []
        for x in range(m):
            y = m - 1 - x
            for left in fbt(x):
                for right in fbt(y):
                    res.append(TreeNode(0, left, right))
        return res
    return fbt(n)

# LC114. Flatten Binary Tree to Linked List - pre-order
def flatten(self, root: Optional[TreeNode]) -> None:  # O(n), O(1)
    node = root
    while node:
        if node.left:
            rightmost = node.left
            while rightmost.right:  # Find the rightmost node
                rightmost = rightmost.right
            rightmost.right = node.right  # predecessor
            node.right, node.left = node.left, None
        node = node.right  # This is original node.left

# LC536. Construct Binary Tree from String  max array
def str2tree(self, s: str) -> Optional[TreeNode]:  # O(n)
    def build(iteri) -> TreeNode:  # works for pattern "num(a)(b))"
        num = ''
        while (nxt := next(iteri)) not in '()': num += nxt
        node = TreeNode(int(num))
        if nxt == '(':
            node.left = build(iteri)
            if next(iteri) == '(':
                node.right = build(iteri)
                next(iteri)  # skip tail ')', need below extra )
        return node
    return build(iter(s + ')')) if s else None

# LC297. Serialize and Deserialize Binary Tree  bt serial bt
class Codec:
    def serialize(self, root):
        def po(node):  # preorder
            if node is None: return "#"
            return f"{node.val},{po(node.left)},{po(node.right)}"
        return po(root)
    def deserialize(self, data):
        def helper(it):
            node_val = next(it)
            if node_val == "#": return None  # cut recursion
            node = TreeNode(int(node_val))
            node.left = helper(it)
            node.right = helper(it)
            return node
        node_vals = data.split(",")
        print(data)
        it = iter(node_vals)
        return helper(it)

# LC1110. Delete Nodes And Return Forest - delete tree nodes
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

# LC919. Complete Binary Tree Inserter
class CBTInserter:
    # Store tree nodes to a list self.tree in bfs order.
    # Node tree[i] has left child tree[2 * i + 1] and tree[2 * i + 2]
    # So when insert the Nth node (0-indexed), we push it into the list.
    # we can find its parent tree[(N - 1) / 2] directly.
    def __init__(self, root: Optional[TreeNode]):
        self.tree = [root]
        for i in self.tree:
            if i.left: self.tree.append(i.left)
            if i.right: self.tree.append(i.right)
    def insert(self, val: int) -> int: # O(1)
        N = len(self.tree)
        self.tree.append(TreeNode(val))
        if N % 2: self.tree[(N - 1) // 2].left = self.tree[-1]
        else: self.tree[(N - 1) // 2].right = self.tree[-1]
        return self.tree[(N - 1) // 2].val
    def get_root(self) -> Optional[TreeNode]:
        return self.tree[0]



# LC105. Construct Binary Tree from Preorder and Inorder Traversal
def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
    if not preorder: return None
    inorder_idxs = {v: i for i, v in enumerate(inorder)}
    def rec(leftpt, rightpt):
        if leftpt > rightpt: return
        root = TreeNode(preorder.pop(0))
        mid = inorder_idxs[root.val]
        root.left = rec(leftpt, mid-1)
        root.right = rec(mid+1, rightpt)
        return root
    return rec(0, len(inorder)-1)

# LC106. Construct Binary Tree from Inorder and Postorder Traversal  # BBG
def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
    inorder_idxs={v: i for i, v in enumerate(inorder)}
    def rec(leftpt, rightpt):
        if leftpt > rightpt: return
        root = TreeNode(postorder.pop()) # pre-order pop from front
        mid = inorder_idxs[root.val]
        root.right = rec(mid+1, rightpt) # flip these 2 lines for preorder
        root.left = rec(leftpt, mid-1)
        return root
    return rec(0, len(inorder)-1)

# LC889. Construct Binary Tree from Preorder and Postorder Traversal
def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
    preIndex, posIndex = 0, 0
    def construct(pre, post):  # O(n) since we loop preindex once
        nonlocal preIndex, posIndex
        root = TreeNode(pre[preIndex])
        preIndex += 1
        if (root.val != post[posIndex]): root.left = construct(pre, post)
        if (root.val != post[posIndex]): root.right = construct(pre, post)
        posIndex += 1
        return root
    return construct(pre, post)

# LC1367. Linked List in Binary Tree  bt linked
def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
    def dfs(head, root):
        if not head: return True
        if not root: return False
        return root.val == head.val and (dfs(head.next, root.left) or dfs(head.next, root.right))
    if not head: return True
    if not root: return False
    return dfs(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)

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

# LC663. Equal Tree Partition
def checkEqualTree(self, root: TreeNode) -> bool:
    seen = []
    def sum_(node):
        if not node: return 0
        seen.append(sum_(node.left) + sum_(node.right) + node.val)
        return seen[-1]
    total = sum_(root)
    seen.pop()  # pop out root sum
    return total / 2.0 in seen
