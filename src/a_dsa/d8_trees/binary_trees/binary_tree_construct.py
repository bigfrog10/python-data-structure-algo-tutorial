
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
            node.left = None # single linked list
        node = node.right  # This is original node.left

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
    def recurse(pre, post):  # O(n) since we loop preindex once
        nonlocal preIndex, posIndex
        root = TreeNode(pre[preIndex])
        preIndex += 1
        if (root.val != post[posIndex]):
            root.left = recurse(pre, post)
        if (root.val != post[posIndex]):
            root.right = recurse(pre, post)
        posIndex += 1
        return root
    return recurse(pre, post)

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

# LC297. Serialize and Deserialize Binary Tree
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
