


# DFS to explore all possibilities and collect stats, return 1 or all results.
from collections import Counter, defaultdict
from typing import List

# LC1376. Time Needed to Inform All Employees
def numOfMinutes(self, n, headID, manager, informTime):
    reports = defaultdict(list)
    for i, m in enumerate(manager):
        if m >= 0: reports[m].append(i)

    def dfs(i):
        return max([dfs(j) for j in reports[i]] or [0]) + informTime[i]
    return dfs(headID)

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





# LC339. Nested List Weight Sum
def depthSum(self, nestedList: List[NestedInteger]) -> int:
    def dfs(nested_list, depth):
        total = 0
        for nested in nested_list:
            if nested.isInteger(): total += nested.getInteger() * depth
            else: total += dfs(nested.getList(), depth + 1)
        return total
    return dfs(nestedList, 1)

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

# LC377. Combination Sum IV
def combinationSum4(self, nums: List[int], target: int) -> int:
    nums.sort()
    @functools.lru_cache(maxsize = None)
    def combs(remain):
        if remain == 0: return 1
        result = 0
        for num in nums:
            if remain - num >= 0:
                result += combs(remain - num)
            else: break
        return result
    return combs(target)

# LC216. Combination Sum III
from itertools import combinations
def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    return [c for c in combinations(range(1, 10), k) if sum(c) == n]

def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    self.ans = []
    def dfs(start, sol, k, n):
        if k == 0 and n == 0:
            self.ans.append(sol)
        if start > 9 or start > n or k <= 0:
            return
        dfs(start+1, sol+[start], k-1, n-start)
        dfs(start+1, sol, k, n)
    dfs(1, [], k, n)
    return self.ans



# LC472. Concatenated Words
def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    word_set = set(words)
    def check(word):
        n = len(word)
        for i in range(1, n):
            if word[i:] not in word_set: continue
            # so now word[i:] in word_set
            if word[:i] in word_set: return True # so both part are words
            if check(word[:i]): return True # recursion check
        return False
    res = []
    for w in words:
        if check(w): res.append(w)
    return res
