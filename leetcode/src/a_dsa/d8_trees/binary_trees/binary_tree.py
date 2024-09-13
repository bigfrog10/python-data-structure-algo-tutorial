from collections import deque, defaultdict, Counter
from typing import List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 2265. Count Nodes Equal to Average of Subtree
def averageOfSubtree(self, root: TreeNode) -> int:
    res = 0
    def dfs(node, total, num_nodes):
        if not node: return 0, 0
        left_total, left_nn = dfs(node.left, total, num_nodes)
        right_total, right_nn = dfs(node.right, total, num_nodes)
        this_total = left_total + right_total + node.val
        this_nn = left_nn + right_nn + 1
        if this_total // this_nn == node.val:
            nonlocal res
            res += 1
        return this_total, this_nn
    dfs(root, 0, 0)
    return res

# LC814. Binary Tree Pruning
def pruneTree(self, root: TreeNode) -> TreeNode:  # O(n) time and O(h) space
    def prune(node):  # return true if all zeros
        if not node: return True
        lz = prune(node.left)
        rz = prune(node.right)
        if lz: node.left = None
        if rz: node.right = None
        return lz and rz and node.val == 0
    z = prune(root)
    return None if z else root

# LC958. Check Completeness of a Binary Tree    bt complete
def isCompleteTree(self, root):  # O(N) time and O(H) space
    def dfs(root):
        if not root: return 0
        l, r = dfs(root.left), dfs(root.right)
        if l & (l + 1) == 0 and l / 2 <= r <= l:
            return l + r + 1
        if r & (r + 1) == 0 and r <= l <= r * 2 + 1:
            return l + r + 1
        return -1
    return dfs(root) > 0
def isCompleteTree(self, root: Optional[TreeNode]) -> bool:  # O(N) time and space
    bfs, i = [root], 0
    while bfs[i]:  # on exit, i is the first None we see.
        bfs.extend([bfs[i].left, bfs[i].right])
        i += 1
    return not any(bfs[i:])  # we shouldn't have any non None after i

# LC545. Boundary of Binary Tree
def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:  # O(n) runtime, O(n) space(stack recursion)
    def left_bound(root):
        if not root: return
        if root.left or root.right: ans.append(root.val) # condition 1
        if root.left: left_bound(root.left) # condition 2
        else: left_bound(root.right) # condition 3
    def leaf(root):
        if not root: return
        leaf(root.left)
        if not root.left and not root.right: ans.append(root.val)
        leaf(root.right)
    def right_bound(root):  ## Reversed order
        if not root: return
        if root.right: right_bound(root.right)  # condition 2
        else: right_bound(root.left)  # condition 3
        if root.left or root.right: ans.append(root.val) # condition 1,
    if not root: return []
    ans = []
    ans.append(root.val)
    left_bound(root.left)
    leaf(root.left)
    leaf(root.right)
    right_bound(root.right)
    return ans

# LC987. Vertical Order Traversal of a Binary Tree - sort in same position
def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:  # O(nlog(n/w)
    res = defaultdict(list) # column number to (row number, val)
    min_col = max_col = 0  # track column range
    def preorder(node, i, j):
        nonlocal min_col, max_col
        if not node: return
        res[j].append((i, node.val))  # keep same cell values together
        min_col = min(min_col, j)
        max_col = max(max_col, j)
        preorder(node.left, i+1, j-1)
        preorder(node.right, i+1, j+1)
    preorder(root, 0, 0)
    # sort within cell
    ret = [[n[1] for n in sorted(res[k])] for k in range(min_col, max_col + 1)]
    return ret




# LC145. Binary Tree Postorder Traversal  bt postorder
def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    nums=[]  # O(n) time, O(h) space
    def dfs(root):
        if root==None: return None
        dfs(root.left)
        dfs(root.right)
        nums.append(root.val)
    dfs(root)
    return nums
def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:  # Morris
    res = []
    while root:
        if not root.right:
            res.append(root.val)
            root = root.left
        else:
            prev = root.right
            while prev.left and prev.left is not root: prev = prev.left
            if prev.left is root:
                prev.left = None
                root = root.left
            else:
                res.append(root.val)
                prev.left = root
                root = root.right
    res.reverse()
    return res

# LC144. Binary Tree Preorder Traversal
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    ret = []
    def dfs(node):
        if not node: return
        ret.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ret
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:  # morris
    node, output = root, []
    while node:
        if not node.left:
            output.append(node.val)
            node = node.right
        else:
            predecessor = node.left
            while predecessor.right and predecessor.right is not node:
                predecessor = predecessor.right
            if not predecessor.right:
                output.append(node.val)
                predecessor.right = node
                node = node.left
            else:
                predecessor.right = None
                node = node.right
    return output

# LC94. Binary Tree Inorder Traversal
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    res = []
    def dfs(currentNode):
        if not currentNode: return
        dfs(currentNode.left)
        res.append(currentNode.val)
        dfs(currentNode.right)
    dfs(root)
    return res
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
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:  # O(n) time and O(1) space, Morris Traversal
    curr = root
    output =[]
    while(curr):
        if not curr.left:
            output.append(curr.val)
            curr = curr.right
        else:
            predecessor = curr.left  # find the inorder predecessor of the current node
            while predecessor.right !=None and predecessor.right != curr:
                predecessor = predecessor.right # go as right as possible
            # Now check which out of 2 above condition it reached
            if predecessor.right == None:
                predecessor.right = curr
                curr = curr.left
            else:
                # left subtree is already visited , so delete the link and then go to right subtree
                predecessor.right = None
                output.append(curr.val)
                curr = curr.right
    return output


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
def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:  # O(mn)
    def preorder(node):
        if not node: return 'none'
        return f'[root={node.val},l={preorder(node.left)},r={preorder(node.right)}]'
    s1 = preorder(root)
    s2 = preorder(subRoot)
    return s1.find(s2) > -1
def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:  # O(mn)
    def is_same(node1, node2):
        if not node1 or not node2: return node1 == node2
        if node1.val != node2.val: return False
        return is_same(node1.left, node2.left) and is_same(node1.right, node2.right)
    if not root: return False
    if is_same(root, subRoot): return True
    return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

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

# LC110. Balanced Binary Tree - check whether it's balanced
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
def isSameTree(self, p, q):  # O(n) and O(h)
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

# LC654. Maximum Binary Tree - root is max among children
def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
    stack = []
    for x in nums:
        n = TreeNode(x)
        while stack and x > stack[-1].val:
            n.left = stack.pop()
        if stack:
            stack[-1].right = n
        stack.append(n)
    return stack[0]

# LC998. Maximum Binary Tree II
def insertIntoMaxTree(self, root, val):
    if root and root.val > val:
        root.right = self.insertIntoMaxTree(root.right, val)
        return root
    node = TreeNode(val)
    node.left = root
    return node

# LC742. Closest Leaf in a Binary Tree
def findClosestLeaf(self, root: TreeNode, k: int) -> int:  # O(n)
    graph = collections.defaultdict(list)
    knode = None
    def dfs(node, par = None): # convert to graph
        nonlocal knode
        if node:
            if node.val == k: knode = node
            graph[node].append(par)
            graph[par].append(node)
            dfs(node.left, node)
            dfs(node.right, node)
    dfs(root)
    queue, seen = collections.deque([knode]), {knode}
    while queue:  # BFS for shortest
        node = queue.popleft()
        if node:
            if len(graph[node]) <= 1: return node.val  # leaf
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append(nei)

# LC863. All Nodes Distance K in Binary Tree
def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:  # O(n) time
    adj = collections.defaultdict(list)  # create graph, O(V) space
    def dfs(node):
        if node.left:
            adj[node].append(node.left)
            adj[node.left].append(node)
            dfs(node.left)
        if node.right:
            adj[node].append(node.right)
            adj[node.right].append(node)
            dfs(node.right)
    dfs(root)
    res, visited = [], set()  # DFS with distance
    def dfs2(node, d):
        visited.add(node)
        if d < k:
            for v in adj[node]:
                if v not in visited: dfs2(v, d + 1)
        else: res.append(node.val)  # ==k, no more recursion, so won't > k
    dfs2(target, 0)
    return res


