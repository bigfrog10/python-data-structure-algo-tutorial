
# LC1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree  bt clone
def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    self.ans = None
    def dfs(o: TreeNode, c: TreeNode):
        if self.ans: return
        if o:
            if o is target:
                self.ans = c
                return
            dfs(o.left, c.left)
            dfs(o.right, c.right)
    dfs(original, cloned)
    return self.ans

# LC404. Sum of Left Leaves
def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
    def dfs(node, isLeft):
        if node:
            if isLeft and not node.left and not node.right:
                return node.val
            return dfs(node.left, True) + dfs(node.right, False)
        return 0

    return dfs(root, False)

# LC508. Most Frequent Subtree Sum
def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:  # O(n) time and space
    if root is None: return []
    count = collections.Counter()
    def dfs(node):
        if node is None: return 0
        s = node.val + dfs(node.left) + dfs(node.right)
        count[s] += 1
        return s
    dfs(root)
    maxCount = max(count.values())
    return [s for s in count if count[s] == maxCount]

# LC690. Employee Importance
def getImportance(self, employees: List['Employee'], query_id: int) -> int:
    emap = {e.id: e for e in employees}  ## O(n) time and space
    def dfs(eid):
        e = emap[eid]
        return e.importance + sum(dfs(sid) for sid in e.subordinates)
    return dfs(query_id)

# LC543. Diameter of Binary Tree - binary tree diameter  bt diameter
def diameterOfBinaryTree(self, root: TreeNode) -> int:
    diameter = 0
    def path_max(node):  # DFS, return depth
        nonlocal diameter
        if not node: return 0
        left = path_max(node.left)
        right = path_max(node.right)
        join = left + right  # path means edges, not nodes
        diameter = max(diameter, join)
        return 1 + max(left, right)  # plus this node
    path_max(root)
    return diameter

# LC1973. Count Nodes Equal to Sum of Descendants - subnodes sum
def equalToDescendants(self, root: Optional[TreeNode]) -> int:
    output = 0
    def traverse(node):  # return sum of this subtree
        nonlocal output
        if node is None: return 0
        val_left = traverse(node.left)
        val_right = traverse(node.right)
        if node.val == val_left + val_right: output += 1
        return node.val + val_left + val_right
    traverse(root)
    return output

# LC1740. Find Distance in a Binary Tree - distance in binary tree
def findDistance(self, root: TreeNode, p: int, q: int) -> int:
    ans = 0
    def fn(node):  # Traverse the tree post-order.
        nonlocal ans
        if not node: return False, -inf
        ltf, lv = fn(node.left)
        rtf, rv = fn(node.right)
        if node.val in (p, q) or ltf and rtf:
            if ltf: ans += lv + 1
            if rtf: ans += rv + 1
            return True, 0
        return ltf or rtf, max(lv, rv) + 1
    fn(root)
    return ans

# LC655. Print Binary Tree - O(n)
def printTree(self, root: TreeNode) -> List[List[str]]:
    def get_height(node):
        return 0 if not node else 1 + max(get_height(node.left), get_height(node.right))
    def update_output(node, row, left, right):
        if not node: return
        mid = (left + right) // 2
        self.output[row][mid] = str(node.val)
        update_output(node.left, row + 1 , left, mid - 1)
        update_output(node.right, row + 1 , mid + 1, right)
    height = get_height(root)
    width = 2 ** height - 1
    self.output = [[''] * width for i in range(height)]
    update_output(node=root, row=0, left=0, right=width - 1)
    return self.output

# LC101. Symmetric Tree - mirror tree, tree mirror
def isSymmetric(self, root: TreeNode) -> bool:
    def is_mirror(n1, n2):  # return if mirrored
        if n1 is None and n2 is None: return True
        if n1 is None or n2 is None: return False
        return n1.val == n2.val and is_mirror(n1.left, n2.right) and is_mirror(n1.right, n2.left)
    return is_mirror(root, root)

# LC1443. Minimum Time to Collect All Apples in a Tree
def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
    adj = [[] for _ in range(n)]  # graph DFS
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()
    def dfs(node):
        if node in visited: return 0
        visited.add(node)
        secs = 0
        for child in adj[node]: secs += dfs(child)
        if secs > 0: return secs + 2  # otherwise, no need to count this
        else: return 2 if hasApple[node] else 0
    return max(dfs(0) - 2, 0)

# LC951. Flip Equivalent Binary Trees
def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool: # O(min(#nodes))
    if not root1 and not root2: return True
    if not root1 or not root2: return False
    if root1.val != root2.val: return False
    return (self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right) or
            self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left))

# LC257. Binary Tree Paths - return all root-leaf paths  bt paths
def binaryTreePaths(self, root: TreeNode) -> List[str]:
    ret = []
    def dfs(node, path):
        if not node: return
        path = path + '->' + str(node.val) if path else str(node.val)
        if not node.left and not node.right: # leaf
            ret.append(path)
            return
        dfs(node.left, path)
        dfs(node.right, path)
    dfs(root, '')
    return ret



# LC104. Maximum Depth of Binary Tree  bt max depth
def maxDepth(self, root: TreeNode) -> int:
    if not root: return 0
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
def maxDepth(self, root):
    depth, stack = 0, [(1, root)]
    while stack != []:
        current_depth, root = stack.pop()
        if root:
            depth = max(depth, current_depth)
            stack.append((current_depth + 1, root.left))
            stack.append((current_depth + 1, root.right))
    return depth

# LC250. Count Univalue Subtrees
def countUnivalSubtrees(self, root: TreeNode) -> int:
    def dfs(node):
        if not node: return 0, True
        if not node.left and not node.right: # leaf
            return 1, True
        ln, lv = dfs(node.left)
        rn, rv = dfs(node.right)
        uni = (not node.left or node.left.val == node.val)
        uni = uni and (not node.right or node.right.val == node.val)
        if lv and rv and uni: cnts = ln + rn + 1
        else: cnts = ln + rn # ignore this node since val different
        return cnts, uni
    return dfs(root)[0]

# LC968. Binary Tree Cameras
def minCameraCover(self, root: TreeNode) -> int:
    # 0: not covered: 1: covered but not has a camera on it; 2: a camera on it.
    self.sum=0
    def dfs(node):
        if not node: return 1
        l = dfs(node.left)
        r = dfs(node.right)
        if l==0 or r==0:
            self.sum += 1 # need a camera to cover
            return 2
        elif l==2 or r==2: return 1
        else: return 0
    if dfs(root) == 0: self.sum += 1
    return self.sum

# LC366. Find Leaves of Binary Tree
def findLeaves(self, root: TreeNode) -> List[List[int]]:
    out = []  # DFS O(n) since we loop all node once
    def dfs(node):  # dfs on height - # number of edges from this node to deepest leaf.
        if not node: return -1  # so that leaf height is 0
        i = 1 + max(dfs(node.left), dfs(node.right))
        if i == len(out): out.append([])
        out[i].append(node.val)
        return i
    dfs(root)
    return out

# LC1315. Sum of Nodes with Even-Valued Grandparent
def sumEvenGrandparent(self, root: TreeNode) -> int:
    total = 0
    def dfs(node, parent): # carry parent is the key step
        nonlocal total
        if node.left:
            if parent and parent.val % 2 == 0: total += node.left.val
            dfs(node.left, node)
        if node.right:
            if parent and parent.val % 2 == 0: total += node.right.val
            dfs(node.right, node)
    dfs(root, None)
    return total

# LC1448. Count Good Nodes in Binary Tree
def goodNodes(self, root: TreeNode) -> int:
    res = 0
    def dfs(node, vmax):
        nonlocal res
        if not node: return
        if node.val >= vmax: res +=1
        nmax = max(node.val, vmax)
        dfs(node.left, nmax)
        dfs(node.right, nmax)
    dfs(root, root.val)
    return res
