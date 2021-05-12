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

# LC106. Construct Binary Tree from Inorder and Postorder Traversal
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
    if not pre:  return None
    root = TreeNode(pre[0])
    if len(pre) == 1: return root
    L = post.index(pre[1]) + 1
    root.left = self.constructFromPrePost(pre[1:L+1], post[:L])
    root.right = self.constructFromPrePost(pre[L+1:], post[L:-1])
    return root

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
