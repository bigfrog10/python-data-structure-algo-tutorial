# LC124. Binary Tree Maximum Path Sum
def maxPathSum(self, root: TreeNode) -> int:
    max_sum = float('-inf')
    def path_max(node):
        nonlocal max_sum
        if not node: return 0
        left = max(path_max(node.left), 0)  # 0 means we don't chose this path
        right = max(path_max(node.right), 0)
        join = left + node.val + right
        max_sum = max(max_sum, join)
        return node.val + max(left, right)
    path_max(root)
    return max_sum

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

# LC112. Path Sum
def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
    def dfs(node, target): # DFS since we are asked if exists a path
        if not node: return False
        if node.val == target:
            if node.left is None and node.right is None: # This is to check leaf
                return True # Terminal step
        return dfs(node.left, target - node.val) or\
               dfs(node.right, target - node.val)
    return dfs(root, targetSum)

# LC113. Path Sum II - return all paths
def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
    res = [] # since we need historic info, we backtrack
    def dfs(node, target, path):
        if not node: return False
        if node.val == target:
            if not node.left and not node.right:
                res.append(path + [node.val]) # this is a copy
        target -= node.val
        path.append(node.val)
        dfs(node.left, target, path)
        dfs(node.right, target, path)
        path.pop() # backtrack
    dfs(root, targetSum, [])
    return res

# LC437. Path Sum III - any path sum=target count
def pathSum(self, root: TreeNode, target: int) -> int:
    count, cusum_counts = 0, defaultdict(int)
    def path_sum_count(node, curr_sum):
        nonlocal count
        if not node: return
        curr_sum += node.val # accumulate along the path
        # this is the case starting from root
        if curr_sum == target: count += 1
        # this is the case starting from middle, could be 0.
        count += cusum_counts[curr_sum - target]
        cusum_counts[curr_sum] += 1 # now check children, and backout
        path_sum_count(node.left, curr_sum)
        path_sum_count(node.right, curr_sum)
        cusum_counts[curr_sum] -= 1 # backout for other tree paths.
    path_sum_count(root, 0) # accumu is 0 at start
    return count

# LC666. Path Sum IV - encode sum
def pathSum(self, nums: List[int]) -> int:
    total, counter = 0, Counter()
    # to count occurrance of this value for all paths
    for n in reversed(nums): # from bottom up
        depth, pos, value = n // 100, (n % 100) // 10, n % 10
        # if counter is 0, it's a new path; else existing path.
        total += value * counter[(depth, pos)] or value
        # to increment parent counter with path numbers
        counter[(depth-1, (pos+1) // 2)] += counter[(depth, pos)] or 1
    return total

# LC1120. Maximum Average Subtree
def maximumAverageSubtree(self, root: TreeNode) -> float:
    max_t = float('-inf') # segment tree
    def dfs(tree): # O(n)
        nonlocal max_t
        ls = lc = rs = rc = 0
        if tree.left: ls, lc = dfs(tree.left)
        if tree.right: rs, rc = dfs(tree.right)
        ts, tc = tree.val + ls + rs, 1 + lc + rc
        max_t = max(max_t, ts / tc)
        return ts, tc # sum and count
    dfs(root)
    return max_t

# LC1026. Maximum Difference Between Node and Ancestor
def maxAncestorDiff(self, root: TreeNode) -> int:
    md = 0  # along each treepath, we get min and max
    def dfs(node, minv, maxv):
        nonlocal md
        if not node: return
        maxv = max(maxv, node.val)
        minv = min(minv, node.val)
        md = max(md, abs(maxv - minv))
        dfs(node.left, minv, maxv)
        dfs(node.right, minv, maxv)
    dfs(root, root.val, root.val)
    return md

# LC104. Maximum Depth of Binary Tree
def maxDepth(self, root: TreeNode) -> int:
    if not root: return 0
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

# LC543. Diameter of Binary Tree
def diameterOfBinaryTree(self, root: TreeNode) -> int:
    diameter = 0
    def path_max(node):
        nonlocal diameter
        if not node: return 0
        left = max(path_max(node.left), 0)  # 0 means we don't chose this path
        right = max(path_max(node.right), 0)
        join = left + right  # path means edges, not nodes
        diameter = max(diameter, join)
        return 1 + max(left, right)
    path_max(root)
    return diameter

# LC515. Find Largest Value in Each Tree Row
def largestValues(self, root: TreeNode) -> List[int]:
    row, maxes = [root], []
    while any(row):  # To deal with None in the row
        maxes.append(max(node.val for node in row if node))
        row = [kid for node in row for kid in (node.left, node.right) if kid]
    return maxes

# LC129. Sum Root to Leaf Numbers
def sumNumbers(self, root: TreeNode) -> int:
    total = 0
    def dfs(node, path_total): # path related indicates DFS
        nonlocal total
        if not node: return
        path_total = node.val + path_total * 10
        if not node.left and not node.right: total += path_total
        else:
            dfs(node.left, path_total)
            dfs(node.right, path_total)
    dfs(root, 0)
    return total

# LC1740. Find Distance in a Binary Tree
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

# LC872. Leaf-Similar Trees
def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
    def dfs(node, res):
        if not node: return res
        if not node.left and not node.right:
            res.append(node.val)
            return res
        dfs(node.left, res)
        dfs(node.right, res)
        return res
    seq1 = dfs(root1, [])
    seq2 = dfs(root2, [])
    return seq1 == seq2

# LC979. Distribute Coins in Binary Tree
def distributeCoins(self, root: TreeNode) -> int:
    self.ans = 0
    def dfs(node): # the excess number of coins in the subtree at or below this node
        if not node: return 0
        L, R = dfs(node.left), dfs(node.right)
        self.ans += abs(L) + abs(R) # extra from children
        return node.val + L + R - 1 # total extra
    dfs(root)
    return self.ans

# LC538. Convert BST to Greater Tree
def convertBST(self, root: TreeNode) -> TreeNode:
    post_sum = 0
    def dfs(root):
        nonlocal post_sum
        if root:
            dfs(root.right)
            post_sum += root.val
            root.val = post_sum
            dfs(root.left)
        return root
    dfs(root)
    return root
