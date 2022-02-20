
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
