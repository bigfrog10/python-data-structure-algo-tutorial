# DFS to explore all possibilities and collect stats, return 1 or all results.
from collections import Counter, defaultdict
from typing import List
# LC339. Nested List Weight Sum
def depthSum(self, nestedList: List[NestedInteger]) -> int:
    def dfs(nested_list, depth):
        total = 0
        for nested in nested_list:
            if nested.isInteger(): total += nested.getInteger() * depth
            else: total += dfs(nested.getList(), depth + 1)
        return total
    return dfs(nestedList, 1)

# LC341. Flatten Nested List Iterator
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = list(reversed(nestedList)) # so pop() O(1)
    def next(self) -> int:
        self.make_stack_top_an_integer()
        return self.stack.pop().getInteger()
    def hasNext(self) -> bool:
        self.make_stack_top_an_integer()
        return len(self.stack) > 0
    def make_stack_top_an_integer(self):
        # While the stack contains a nested list at the top...
        while self.stack and not self.stack[-1].isInteger():
            # Unpack the list at the top by putting its items onto
            # the stack in reverse order.
            self.stack.extend(reversed(self.stack.pop().getList()))

# LC39. Combination Sum
def combinationSum(self, candidates, target):  # fastest
    candidates = sorted(candidates) # small to large, cut branches
    result = []
    def dfs(remain, stack): # stack is history
        if remain == 0:
            result.append(stack)
            return
        for item in candidates:
            if item > remain: break
            if stack and item < stack[-1]: continue # smallers are done already.
            else: dfs(remain - item, stack + [item])
    dfs(target, [])
    return result

# LC839. Similar String Groups
def numSimilarGroups(self, strs: List[str]) -> int:
    def similar(s, t):
        diff = 0
        for c1, c2 in zip(s, t):
            diff += 1 if c1 != c2 else 0
            if diff > 2: return False
        return True
    visited = set()
    def dfs(st, graph):
        for nb in graph[st]:
            if nb not in visited:
                visited.add(nb)
                dfs(nb, graph)

    graph = defaultdict(list)
    for s1 in strs:  #  O(n^2)
        for s2 in strs:
            if similar(s1, s2): graph[s1].append(s2)
    ret = 0
    for s1 in strs:
        if s1 not in visited:
            dfs(s1, graph)
            ret += 1

    return ret

# LC721. Accounts Merge
def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
    graph = defaultdict(list)  # build graph
    for acct in accounts:
        for email in acct[2:]:
            graph[acct[1]].append(email)
            graph[email].append(acct[1])
    seen = set()
    def dfs(i):
        tmp = {i}
        for j in graph[i]:
            if j not in seen:
                seen.add(j)
                tmp |= dfs(j)
        return tmp
    ret = []
    for acct in accounts:
        for email in acct[1:]:
            if email not in seen:
                seen.add(email)
                eg = dfs(email)
                ret.append([acct[0]] + sorted(eg))
    return ret

# LC489. Robot Room Cleaner
def cleanRoom(self, robot):
    def backout(robot):
        robot.turnRight()
        robot.turnRight()  # turn back
        robot.move()
        robot.turnRight()
        robot.turnRight()  # turn to original dir
    visited = set()
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
    def dfs(i, j, cf, robot):
        visited.add((i, j))
        robot.clean()
        for d in range(4):
            cf = (cf + d) % 4  # e.g., facing right needs to start from 2nd index
            x, y = i + dirs[cf][0], j + dirs[cf][1]
            if (x, y) not in visited and robot.move():
                dfs(x, y, cf, robot)
                backout(robot)
            robot.turnRight()  # as d increment
    dfs(0, 0, 0, robot)
    print(visited)

# LC935. Knight Dialer
def knightDialer(self, n: int) -> int:
    MOD = 10**9 + 7
    moves = [[4,6],[6,8],[7,9],[4,8],[3,9,0],[],[1,7,0],[2,6],[1,3],[2,4]]
    dp = [1] * 10 # counts for current hop and current digit
    for hops in range(n-1):
        dp2 = [0] * 10
        for node, count in enumerate(dp): # loop all digits 0-9
            for nei in moves[node]:  # loop all jumps
                dp2[nei] += count
                dp2[nei] %= MOD
        dp = dp2
    return sum(dp) % MOD

# LC417. Pacific Atlantic Water Flow
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
    if not matrix: return []
    rows, cols = len(matrix), len(matrix[0])
    p_visited, a_visited = set(), set()
    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    def traverse(i, j, visited):
        if (i, j) in visited: return
        visited.add((i, j))
        for direction in directions: # Traverse neighbors.
            next_i, next_j = i + direction[0], j + direction[1]
            if 0 <= next_i < rows and 0 <= next_j < cols:
                if matrix[next_i][next_j] >= matrix[i][j]:
                    traverse(next_i, next_j, visited)
    for row in range(rows):
        traverse(row, 0, p_visited)
        traverse(row, cols - 1, a_visited)
    for col in range(cols):
        traverse(0, col, p_visited)
        traverse(rows - 1, col, a_visited)
    return list(p_visited & a_visited)
