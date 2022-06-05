
# LC827. Making A Large Island
def largestIsland(self, grid: List[List[int]]) -> int: # O(n^2) runtime and space
    if not grid or not grid[0]: return 0
    n = len(grid)
    def dfs(r, c, gid):  # get island size
        ret, grid[r][c] = 1, gid  # mark islands by numbers
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < n and 0 <= y < n and grid[x][y] == 1: ret += dfs(x, y, gid)
        return ret
    islands, gid = {}, 2  # 0 for water, 1 for island, so we start with 2
    for r, c in product(range(n), range(n)):  # find each island size
        if grid[r][c] == 1:  # unexplored island
            islands[gid] = dfs(r, c, gid)
            gid += 1
    ret = max(islands.values() or [0])  # in case all land no water
    for r, c in product(range(n), range(n)):
        if grid[r][c] == 0:  # go through each water
            seen = set()  # this is to filter out repetitive islands from differnt landings
            for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= x < n and 0 <= y < n and grid[x][y] > 1: seen.add(grid[x][y])
            expand = sum(islands[p] for p in seen) + 1
            ret = max(ret, expand)
    return ret

# LC200. Number of Islands
from itertools import product
def numIslands(self, board: List[List[str]]) -> int:
    if not board: return 0  # O(MN)
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

# LC305. Number of Islands II - add land one by one - union find
def numIslands2(self, m, n, positions):
    parent, rank = {}, {}  # tree depth logn
    def find(x):
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):  #  M union or find sequences on N objects takes O(N + M * log*N)
        x, y = find(x), find(y)
        if x == y: return 0  # no new island
        if rank[x] < rank[y]: x, y = y, x
        parent[y] = x
        rank[y] += rank[x]
        return 1  # new island
    counts, count = [], 0  # let N = m*n, union is O(N + Nlog*N)
    for i, j in positions:  # L operations, so O(N + L* log*N)
        if (i, j) not in parent:
            x = parent[x] = i, j
            rank[x] = 1
            count += 1  # x is a new island
            for y in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
                if y in parent: count -= union(x, y)
        counts.append(count)
    return counts

# LC463. Island Perimeter
def islandPerimeter(self, grid: List[List[int]]) -> int:  # O(mn), O(1)
    rows, cols = len(grid), len(grid[0])
    result = 0
    for r, c in itertools.product(range(rows), range(cols)):
        if grid[r][c] == 1:
            result += 4
            if r > 0 and grid[r-1][c] == 1: result -= 2  # remove borders between
            if c > 0 and grid[r][c-1] == 1: result -= 2
    return result

# LC934. Shortest Bridge
def shortestBridge(self, A: List[List[int]]) -> int:
    m, n = len(A), len(A[0])
    i, j = next((i, j) for i in range(m) for j in range(n) if A[i][j])
    stack, seen = [(i, j)], set()
    while stack:  # dfs on first island to populate seen
        i, j = stack.pop()
        seen.add((i, j)) # mark as visited
        for ii, jj in (i-1, j), (i, j-1), (i, j+1), (i+1, j):
            if 0 <= ii < m and 0 <= jj < n and A[ii][jj] and (ii, jj) not in seen:
                stack.append((ii, jj))
                seen.add((ii, jj))
    ans = 0  # shortest distance, also levels to BFS
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
    seen = set()
    def dfs(i, j):
        if grid[i][j] == 0 or (i, j) in seen: return 0
        seen.add((i, j))
        amax = 1
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            a, b = i + d[0], j + d[1]
            if 0 <= a < n and 0 <= b < m and grid[a][b] != 0 and (a, b) not in seen:
                amax += dfs(a, b)
        return amax
    amax = 0
    for i in range(n):
        for j in range(m): amax = max(amax, dfs(i, j))
    return amax







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

# LC711. Number of Distinct Islands II - transformers
def numDistinctIslands2(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])

    # _abs: absolute coordinate
    def DFS(row: int, col: int, _abs: List[Tuple[int, int]]) -> None:
        if not 0 <= row < m or not 0 <= col < n or grid[row][col] == 0:
            return
        grid[row][col] = 0 # Visit by marking as 0
        _abs.append((row, col))
        DFS(row, col + 1, _abs)
        DFS(row + 1, col, _abs)
        DFS(row, col - 1, _abs)
        DFS(row - 1, col, _abs)

    # Normalize by relative coordinate w.r.t. "top-left": (min(rows), min(cols))
    # Use frozenset (immutable version of set) of coordinates to ignore cell traversal order
    def normalize(_abs: List[Tuple[int, int]]) -> FrozenSet[List[Tuple[int, int]]]:
        rows, cols = zip(*_abs) # Unzip _abs by calling zip(*_abs)
        r_min, c_min = min(rows), min(cols)
        return frozenset([(r - r_min, c - c_min) for r, c in _abs])

    distinct_islands = set()
    for row in range(m):
        for col in range(n):
            if grid[row][col] == 1:
                _abs = [] # Absolute coordinates
                DFS(row, col, _abs)
                # 8 transformations
                transformations = []
                transformations.append(normalize([(+r, +c) for r, c in _abs]))
                transformations.append(normalize([(+r, -c) for r, c in _abs]))
                transformations.append(normalize([(-r, +c) for r, c in _abs]))
                transformations.append(normalize([(-r, -c) for r, c in _abs]))
                transformations.append(normalize([(+c, +r) for r, c in _abs]))
                transformations.append(normalize([(+c, -r) for r, c in _abs]))
                transformations.append(normalize([(-c, +r) for r, c in _abs]))
                transformations.append(normalize([(-c, -r) for r, c in _abs]))
                # frozenset(transformations) removes duplicated transformed coordinates
                # distinct_island.add() removes islands of duplicated transformations as before
                distinct_islands.add(frozenset(transformations))
    return len(distinct_islands)

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

# LC1905. Count Sub Islands
def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
    n, m = len(grid2), len(grid2[0])
    def dfs(i, j):
        if not (0 <= i < n and 0 <= j < m and grid2[i][j]): return 1
        grid2[i][j] = 0  # visited
        res = grid1[i][j]
        for di, dj in [0, 1], [1, 0], [-1, 0], [0, -1]:
            res &= dfs(i + di, j + dj)
        return res  # 0 or 1
    return sum(dfs(i, j) for i in range(n) for j in range(m) if grid2[i][j])
