
# LC515. Find Largest Value in Each Tree Row
def largestValues(self, root: TreeNode) -> List[int]:
    row, maxes = [root], []
    while any(row):  # To deal with None in the row
        maxes.append(max(node.val for node in row if node))
        row = [kid for node in row for kid in (node.left, node.right) if kid]
    return maxes

# LC513. Find Bottom Left Tree Value
def findBottomLeftValue(self, root: TreeNode) -> int: # O(n)
    queue = [root]
    for node in queue:  # go right then left, so last node is left.
        queue += filter(None, (node.right, node.left))
    return node.val

# LC1161. Maximum Level Sum of a Binary Tree
def maxLevelSum(self, root: TreeNode) -> int:
    ans, q, depth = (-math.inf, 0), [root], -1
    while q:
        ans = max(ans, (sum(node.val for node in q), depth))
        q = [kid for node in q for kid in (node.left, node.right) if kid]
        depth -= 1
    return -ans[1]

# LC116. Populating Next Right Pointers in Each Node - Perfect tree
def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':  # O(n) runtime and O(1) space
    res = root
    while root and root.left:
        next1 = root.left  # save next level
        while root:  # loop this level
            root.left.next = root.right  # set horizontal pointer for left
            # set horizontal pointer for right, later if both exist or earlier
            root.right.next = root.next and root.next.left
            root = root.next  # go to next node at this level
        root = next1  # go to next level
    return res
def connect(self, root: 'Node') -> 'Node':  # This is better written
    if not root: return root
    Q = collections.deque([root])
    while Q:
        size = len(Q)
        for i in range(size):  # BFS
            node = Q.popleft()
            if i < size - 1: node.next = Q[0]  # assign to next
            if node.left: Q.append(node.left)
            if node.right: Q.append(node.right)
    return root

# LC117. Populating Next Right Pointers in Each Node II
def connect(self, root: 'Node') -> 'Node':
    res = root
    while root:
        cur = leftmost = Node(0)
        while root:
            if root.left:  # level travel at child level
                cur.next = root.left
                cur = root.left  # level travel at child level
            if root.right:
                cur.next = root.right  # connect to right at child level
                cur = root.right
            root = root.next  # level travel at parent level
        root = leftmost.next  # next level starting point
    return res

# LC637. Average of Levels in Binary Tree
def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
    level, averages = [root], []
    while level:
        averages.append(sum(node.val for node in level) / len(level))
        level = [kid for node in level for kid in (node.left, node.right) if kid]
    return averages

# LC199. Binary Tree Right Side View
def rightSideView(self, root: TreeNode) -> List[int]:
    if not root: return []
    ret = []
    def traverse(node, depth):
        if depth == len(ret): ret.append(node.val)
        for n in [node.right, node.left]:
            if n: traverse(n, depth+1)
    traverse(root, 0)
    return ret

# LC103. Binary Tree Zigzag Level Order Traversal
def zigzagLevelOrder(self, root):
    ans, row, drxn = [], [root], 1
    while any(row):
        ans += [node.val for node in row][::drxn],
        row = [child for node in row for child in (node.left, node.right) if child]
        drxn *= -1
    return ans

# LC111. Minimum Depth of Binary Tree
def minDepth(self, root: Optional[TreeNode]) -> int:
    res, row = 0, [root]  # BFS for min
    while any(row):
        res += 1
        if any(not(child.left or child.right) for child in row): return res
        row = [child for node in row for child in (node.left, node.right) if child]
    return res

# LC102. Binary Tree Level Order Traversal
def levelOrder(self, root):
    if not root: return []
    ans, level = [], [root]
    while level:
        ans.append([node.val for node in level])
        level = [kid for n in level for kid in (n.left, n.right) if kid]
    return ans

# LC1104. Path In Zigzag Labelled Binary Tree
def pathInZigZagTree(self, label: int) -> List[int]:
    level, node_count, res = 1, 1, [] # O(log n) space
    while label >= node_count*2: # O(log n) time, Determine level of the label, root level is 1
        node_count *= 2
        level += 1
    while label > 0: # O(log n) time, Iterate from the target label to the root
        res.append(label)
        level_min, level_max = 2 ** (level-1), 2 ** level - 1
        # it's really max - (label - min)
        label = (level_max + level_min - label) // 2  # label for next level
        level -= 1
    return res[::-1] # O(n) time

# LC993. Cousins in Binary Tree
def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
    stk = [root]
    while stk:
        p = {c.val: node.val for node in stk for c in (node.left, node.right) if c}
        if x in p and y in p and p[x] != p[y]: return True
        stk = [child for node in stk for child in (node.left,node.right) if child]
    return False
