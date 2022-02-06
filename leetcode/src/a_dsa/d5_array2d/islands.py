
# LC827. Making A Large Island
def largestIsland(self, grid: List[List[int]]) -> int: # recursion
    if not grid or not grid[0]: return 0
    n = len(grid)
    def dfs(r, c, gid):  # get island size
        ret, grid[r][c] = 1, gid  # mark islands by numbers
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < n and 0 <= y < n and grid[x][y] == 1:
                ret += dfs(x, y, gid)
        return ret
    islands, gid = {}, 2  # 0 for water, 1 for island, so we start with 2
    for r, c in product(range(n), range(n)):  # find each island size
        if grid[r][c] == 1:  # unexplored island
            islands[gid] = dfs(r, c, gid)
            gid += 1
    ret = max(islands.values() or [0]) # in case all land no water
    for r, c in product(range(n), range(n)):
        if grid[r][c] == 0:  # go through each water
            seen = set()  # this is to filter out repetitive islands from differnt landings
            for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= x < n and 0 <= y < n and grid[x][y] > 1:
                    seen.add(grid[x][y])
            expand = sum(islands[p] for p in seen) + 1
            ret = max(ret, expand)
    return ret

# LC934. Shortest Bridge
def shortestBridge(self, A: List[List[int]]) -> int:
    m, n = len(A), len(A[0])
    i, j = next((i, j) for i in range(m) for j in range(n) if A[i][j])
    stack = [(i, j)]
    seen = set(stack)
    while stack: # dfs on first island
        i, j = stack.pop()
        seen.add((i, j)) # mark as visited
        for ii, jj in (i-1, j), (i, j-1), (i, j+1), (i+1, j):
            if 0 <= ii < m and 0 <= jj < n and A[ii][jj] and (ii, jj) not in seen:
                stack.append((ii, jj))
                seen.add((ii, jj))
    ans = 0
    queue = list(seen)
    while queue:  # bfs on second island
        newq = []
        for i, j in queue:
            for ii, jj in (i-1, j), (i, j-1), (i, j+1), (i+1, j):
                if 0 <= ii < m and 0 <= jj < n and (ii, jj) not in seen:
                    if A[ii][jj] == 1: return ans
                    newq.append((ii, jj))
                    seen.add((ii, jj))
        queue = newq
        ans += 1

# LC695. Max Area of Island
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    if not grid: return 0  # O(rows * columns)
    n, m = len(grid), len(grid[0])
    def dfs(i, j):
        if grid[i][j] == 0 or grid[i][j] == 2: return 0
        grid[i][j] = 2
        amax = 1
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            a, b = i + d[0], j + d[1]
            if 0 <= a < n and 0 <= b < m and grid[a][b] != 0 and grid[a][b] != 2:
                amax += dfs(a, b)
        return amax
    amax = 0
    for i in range(n):
        for j in range(m): amax = max(amax, dfs(i, j))
    return amax

# LC463. Island Perimeter
def islandPerimeter(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    result = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                up = 0 if r == 0 else grid[r-1][c]
                down = 0 if r == rows-1 else grid[r+1][c]
                left = 0 if c == 0 else grid[r][c-1]
                right = 0 if c == cols-1 else grid[r][c+1]
                result += 4-(up+left+right+down)
    return result

# LC305. Number of Islands II
def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
    pa = {}  # parents
    def find(x):
        while x in pa:
            if pa[x] in pa: pa[x]=pa[pa[x]]  # path compress
            x = pa[x]
        return x
    def union(x,y):
        pax,pay = find(x), find(y)
        if pax == pay: return False # has been unioned.
        pa[pax] = pay  # no rank in this implementation
        return True
    seen, res, count = set(), [], 0
    for x, y in positions:  #connect with neighbor val==1,if union success,means one island disappear.
        if (x, y) not in seen:
            seen.add((x, y))
            count += 1
            for i, j in (x+1, y), (x-1, y), (x, y+1), (x, y-1):
                if (i, j) in seen and union((i, j), (x, y)):
                    count -= 1
        res.append(count)
    return res

# LC200. Number of Islands, top100
from itertools import product
def numIslands(self, board: List[List[str]]) -> int:
    if not board: return 0 # O(MN)
    rows, cols = len(board), len(board[0])
    seen = set()
    def dfs(i, j):
        seen.add((i, j))
        for dx, dy in (1, 0), (-1, 0), (0, 1), (0, -1):
            x, y = i + dx, j + dy
            if 0 <= x < rows and 0 <= y < cols and board[i][j] == '1' and (x, y) not in seen:
                dfs(x, y)
    count = 0
    for i, j in product(range(rows), range(cols)):
        if board[i][j] == '1' and (i, j) not in seen:
            count += 1
            dfs(i, j)
    return count

# LC694. Number of Distinct Islands - transformations
def numDistinctIslands(self, grid: List[List[int]]) -> int:
    if not grid: return 0 # close to # 200
    n, m = len(grid), len(grid[0])
    seen = set()
    def explore(r, c, xo, yo, shape):
        if 0 <= r < n and 0 <= c < m and (r, c) not in seen and grid[r][c] != 0:
            seen.add((r, c))
            shape.add((r-xo, c-yo))
            for x, y in (r+1, c), (r-1, c), (r, c+1), (r, c-1):
                explore(x, y, xo, yo, shape)
    shapes = set()
    for i, j in product(range(n), range(m)):
        if grid[i][j] == 1 and (i, j) not in seen:
            shape = set()  # collection all island coordinates
            explore(i, j, i, j, shape)
            if shape: shapes.add(frozenset(shape))  # hash
    return len(shapes)




# LC130. Surrounded Regions
def solve(self, board):
    if not any(board): return
    m, n = len(board), len(board[0])
    save = [ij for k in range(max(m,n)) for ij in ((0, k), (m-1, k), (k, 0), (k, n-1))]
    while save:
        i, j = save.pop()
        if 0 <= i < m and 0 <= j < n and board[i][j] == 'O':
            board[i][j] = 'S' # we explore from boundaries
            save += (i, j-1), (i, j+1), (i-1, j), (i+1, j)
    board[:] = [['XO'[c == 'S'] for c in row] for row in board]






# LC547. Number of Provinces
def findCircleNum(self, A): # O(rows * columns)
    N = len(A)
    seen = set()
    def dfs(node):
        for nei, adj in enumerate(A[node]):
            if adj and nei not in seen:
                seen.add(nei)
                dfs(nei)
    ans = 0
    for i in xrange(N):
        if i not in seen:
            dfs(i)
            ans += 1
    return ans




# LC711. Number of Distinct Islands II

# LC1254. Number of Closed Islands
def closedIsland(self, grid: List[List[int]]) -> int:
    if not grid or not grid[0]: return 0
    m, n = len(grid), len(grid[0])
    seen, stack = set(), []
    def dfs(i, j):
        if grid[i][j] != 0 or (i, j) in seen: return
        seen.add((i, j))
        for x, y in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
            if 0 <= x < m and 0 <= y < n: dfs(x, y)
    for i in range(m): # explore boundaries so they don't get counted.
        dfs(i, 0)
        dfs(i, n-1)
    for j in range(n):
        dfs(0, j)
        dfs(m-1, j)
    islands = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0 and (i, j) not in seen:
                dfs(i, j)
                islands += 1
    return islands






