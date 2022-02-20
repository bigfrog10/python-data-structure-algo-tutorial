
# LC297. Serialize and Deserialize Binary Tree
class Codec:
    def serialize(self, root):
        if not root: return ''
        def node_to_json(node):
            ret = f'"{node.val}":['
            if node.left: ret += node_to_json(node.left) + ','
            else: ret += 'null,'
            if node.right: ret += node_to_json(node.right)
            else: ret += 'null'
            ret = '{' + ret + ']}'
            return ret  # or we convert to dict and then str(dict)
        ret = node_to_json(root)
        return ret
    def deserialize(self, data):
        if not data: return []
        def dict_to_node(kvs):
            if kvs is None: return None
            for k, v in kvs.items():
                tn = TreeNode(int(k))
                tn.left = dict_to_node(v[0])
                tn.right = dict_to_node(v[1])
                return tn
        kv = json.loads(data)
        return dict_to_node(kv)

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

# LC114. Flatten Binary Tree to Linked List
def flatten(self, root: TreeNode) -> None: # preorder
    if not root: return None
    node = root
    while node:
        if node.left:
            rightmost = node.left
            while rightmost.right:  # Find the rightmost node
                rightmost = rightmost.right
            rightmost.right = node.right  # predecessor
            node.right = node.left
            node.left = None  # single linked list
        node = node.right  # This is original node.left

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

# LC1367. Linked List in Binary Tree
def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
    def dfs(head, root):
        if not head: return True
        if not root: return False
        return root.val == head.val and (dfs(head.next, root.left) or dfs(head.next, root.right))
    if not head: return True
    if not root: return False
    return dfs(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)

# LC536. Construct Binary Tree from String
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
    seen.pop() # pop out root sum
    return total / 2.0 in seen


# LC449. Serialize and Deserialize BST
class Codec:  # json: node -> {1: [{2:[None, None]}, {3:[None, None]}]}
    def serialize(self, root):
        if not root: return ''  # base case
        def node_to_json(node): # DFS
            ret = f'"{node.val}":['
            if node.left: ret += node_to_json(node.left) + ','
            else: ret += 'null,'
            if node.right: ret += node_to_json(node.right)
            else: ret += 'null'
            ret += ']'
            ret = '{' + ret + '}'
            return ret
        ret = node_to_json(root)
        return ret

    def deserialize(self, data):
        if not data: return []
        def dict_to_node(kvs): #DFS
            if kvs is None: return None  # base case
            for k, v in kvs.items():
                tn = TreeNode(int(k))
                tn.left = dict_to_node(v[0])
                tn.right = dict_to_node(v[1])
                return tn
        kv = json.loads(data)
        return dict_to_node(kv)
