
# LC1463. Cherry Pickup II
def cherryPickup(self, grid: List[List[int]]) -> int:  # O(m n^2) time and space
    m, n = len(grid), len(grid[0])
    @lru_cache(None)
    def dp(row, col1, col2):  # O(m n^2)
        if col1 < 0 or col1 >= n or col2 < 0 or col2 >= n: return -inf
        result = grid[row][col1]
        if col1 != col2: result += grid[row][col2]
        if row != m-1:
            result += max(dp(row+1, new_col1, new_col2)
                          for new_col1 in [col1, col1+1, col1-1]
                          for new_col2 in [col2, col2+1, col2-1])
        return result
    return dp(0, 0, n-1)

# LC1091. Shortest Path in Binary Matrix - 01 matrix  top left low right
def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:  # O(n) runtime and space
    if not grid or grid[0][0] != 0: return -1
    n, que, visited = len(grid), deque([(0, 0, 1)]), set()  # x, y, steps
    while que:  # BFS
        i, j, steps = que.popleft()
        if i == n-1 and j == n-1: return steps
        if (i, j) in visited: continue
        visited.add((i, j))
        for x, y in (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1):
            ni, nj = i + x, j + y
            if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 0 and (ni, nj) not in visited:
                que.append((ni, nj, steps+1))
    return -1

# LC317. Shortest Distance from All Buildings
def shortestDistance(self, grid):
    if not grid or not grid[0]: return -1  # O(n^2 * m^2)
    m, n = len(grid), len(grid[0])
    totalB = sum(grid[i][j]==1 for i in range(m) for j in range(n))  # number of buildings
    ## do BFS from each building, and decrement all empty place for every building visit
    ## when grid[i][j] == -totalB, it means that grid[i][j] are already visited from all buildings
    dist = [[0]*n for i in range(m)]  # distances from buildings
    def bfs(i, j, bIndex):  # mark distance from this building
        queue = collections.deque([(i, j, 0)])
        while queue:
            i, j, d = queue.popleft()
            for x, y in (i+1, j), (i-1, j), (i, j-1), (i, j+1):
                if 0 <= x < m and 0 <= y < n and grid[x][y] == bIndex:
                    dist[x][y] += d+1
                    grid[x][y] -= 1
                    queue.append((x, y, d+1))
    bIndex = 0  # count how many building we have visited
    for i, j in itertools.product(range(m), range(n)):
        if grid[i][j]==1:  # loop through buildings
            bfs(i, j, bIndex)
            bIndex -= 1
    res = [dist[i][j] for i in range(m) for j in range(n) if grid[i][j]+totalB==0]
    return min(res) if res else -1

# LC778. Swim in Rising Water
def swimInWater(self, grid: List[List[int]]) -> int:  # O(N^2log(N^2)) time, O(N^2) space
    N = len(grid)
    pq, seen = [(grid[0][0], 0, 0)], {(0, 0)}
    ans = 0
    while pq: # DFS, O(N^2)
        d, r, c = heapq.heappop(pq)
        ans = max(ans, d)
        if r == c == N-1: return ans
        for cr, cc in (r-1, c), (r+1, c), (r, c-1), (r, c+1):
            if N > cr >= 0 <= cc < N and (cr, cc) not in seen:
                heapq.heappush(pq, (grid[cr][cc], cr, cc))  # log
                seen.add((cr, cc))

# LC286. Walls and Gates - distance from gates
def wallsAndGates(self, rooms: List[List[int]]) -> None:  # O(mn)
    WALL, GATE, EMPTY = -1, 0, 2147483647  # given
    q = [(i, j) for i, row in enumerate(rooms) for j, r in enumerate(row) if r == GATE]  # all gates
    for i, j in q:  # leve travel
        for I, J in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
            if 0 <= I < len(rooms) and 0 <= J < len(rooms[0]) and rooms[I][J] == EMPTY:
                rooms[I][J] = rooms[i][j] + 1
                q.append((I, J))

# LC1559. Detect Cycles in 2D Grid
def containsCycle(self, grid: List[List[str]]) -> bool:  # O(mn)
    m, n = len(grid), len(grid[0])
    visited = set()
    def dfs(node, parent):
        if node in visited: return True
        visited.add(node)
        nx,ny = node
        for cx, cy in [nx+1,ny], [nx-1, ny],[nx,ny+1], [nx,ny-1]:
            if m > cx >= 0 <= cy < n and grid[cx][cy] == grid[nx][ny] and (cx,cy) != parent:
                if dfs((cx, cy), node): return True
        return False
    return any((i,j) not in visited and dfs((i, j), (i, j)) for i, j in product(range(m), range(n)))

# LC1102. Path With Maximum Minimum Value - minmax search Dijkstra
def maximumMinimumPath(self, A: List[List[int]]) -> int:  # Time: O(MN log MN), space O(MN)
    R, C = len(A), len(A[0])  # Dijkstra
    maxHeap = [(-A[0][0], 0, 0)]
    seen = [[0 for _ in range(C)] for _ in range(R)]
    seen[0][0] = 1
    while maxHeap:  # some low level point touched but not expanded
        val, x, y = heapq.heappop(maxHeap)
        if x == R - 1 and y == C - 1:  return -val
        for dx, dy in (0, 1), (1, 0), (0, -1), (-1, 0):
            nx, ny = x + dx, y + dy
            if 0 <= nx < R and 0 <= ny < C and not seen[nx][ny]:
                seen[nx][ny] = 1
                heapq.heappush(maxHeap, (max(val, -A[nx][ny]), nx, ny))
    return -1

# LC1263. Minimum Moves to Move a Box to Their Target Location - move box
def minPushBox(self, grid: List[List[str]]) -> int:  # faster BFS, O((mn)^2)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "T": target = (i,j)
            if grid[i][j] == "B": box = (i,j)
            if grid[i][j] == "S": person = (i,j)
    def empty(x, y): # O(1) verify
        return 0 <= x <len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != '#'
    def reachable(curr, dest, box):  #BFS to check dest is reachable from curr
        que, v = deque([curr]), set()
        while que:
            pos = que.popleft()
            if pos == dest: return True
            for x, y in (pos[0]+1,pos[1]), (pos[0]-1,pos[1]), (pos[0],pos[1]+1), (pos[0],pos[1]-1):
                if empty(x, y) and (x, y) != box and (x, y) not in v:
                    v.add((x,y))
                    que.append((x,y))
        return False
    q, visited = deque([(0, box, person)]), {box + person}
    while q:  # main BFS
        pushes, box, person = q.popleft()
        if box == target: return pushes
        b_coord = [(box[0]+1,box[1]),(box[0]-1,box[1]),(box[0],box[1]+1),(box[0],box[1]-1)]
        p_coord = [(box[0]-1,box[1]),(box[0]+1,box[1]),(box[0],box[1]-1),(box[0],box[1]+1)]
        for new_box, new_person in zip(b_coord,p_coord):
            if empty(*new_box) and new_box + box not in visited:
                if empty(*new_person) and reachable(person, new_person, box):
                    visited.add(new_box+box)
                    q.append((pushes + 1,new_box,box))
    return -1



# LC329. Longest Increasing Path in a Matrix  long incr paths long inc path  long incr path matrix lip
import functools
def longestIncreasingPath(self, matrix):
    if not matrix or not matrix[0]: return 0  # O(mn) space and time due to cache
    M, N = len(matrix), len(matrix[0])
    @functools.lru_cache(maxsize=None)
    def dfs(i, j):
        ret = 0
        for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
            if 0 <= x < M and 0 <= y < N and matrix[i][j] < matrix[x][y]:
                ret = max(ret, dfs(x, y))
        return ret + 1  # add this cell
    return max(dfs(x, y) for x in range(M) for y in range(N))

# LC296. Best Meeting Point
def minTotalDistance(self, grid: List[List[int]]) -> int:  # O(mn)
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    rows, cols = [], []
    for i in range(m):  # loop m and then n
        for j in range(n):
            if grid[i][j] == 1: rows.append(i)
    for j in range(n):  # loop n and then m
        for i in range(m):
            if grid[i][j] == 1: cols.append(j)
    def min_dist(locations: List[int]) -> int:  # 1 dimensional case
        i, j, sumd = 0, len(locations)-1, 0
        while i < j:
            sumd += locations[j] - locations[i]
            i, j = i+1, j-1
        return sumd
    return min_dist(rows) + min_dist(cols)  # Manhattan distance





# LC417. Pacific Atlantic Water Flow
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:  # O(nm)
    if not matrix: return []
    rows, cols = len(matrix), len(matrix[0])
    p_visited, a_visited = set(), set()
    def traverse(i, j, visited):  # goes up
        if (i, j) in visited: return
        visited.add((i, j))
        for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0): # Traverse neighbors.
            ni, nj = i + dx, j + dy
            if rows > ni >= 0 <= nj < cols and matrix[ni][nj] >= matrix[i][j]:
                traverse(ni, nj, visited)
    for row in range(rows):  # along border, go inland
        traverse(row, 0, p_visited)
        traverse(row, cols - 1, a_visited)
    for col in range(cols):
        traverse(0, col, p_visited)
        traverse(rows - 1, col, a_visited)
    return list(p_visited & a_visited)

# 1293. Shortest Path in a Grid with Obstacles Elimination at most k obstacles
def shortestPath(self, grid: List[List[int]], k: int) -> int:  # O(Nk), N is # of cells
    rows, cols = len(grid), len(grid[0])
    if k >= rows + cols - 2: return rows + cols - 2
    state = (0, 0, k)  # (row, col, remaining quota to eliminate obstacles)
    queue, seen = deque([(0, state)]), set([state])  # (steps, state)
    while queue:
        steps, (row, col, k) = queue.popleft()
        if (row, col) == (rows - 1, cols - 1): return steps
        for x, y in (row, col + 1), (row + 1, col), (row, col - 1), (row - 1, col):
            if 0 <= x < rows and 0 <= y < cols:
                nk = k - grid[x][y]
                new_state = x, y, nk
                if nk >= 0 and new_state not in seen:
                    seen.add(new_state)
                    queue.append((steps + 1, new_state))
    return -1
def shortestPath(self, grid: List[List[int]], k: int) -> int:  # best, A*, O(Nklog(Nk))
    m, n = len(grid), len(grid[0])
    state = m-1, n-1, k
    queue, seen = [(m+n-2, 0, state)], {state}  # manhattan distance
    while queue:
        _, steps, (i, j, k) = heapq.heappop(queue)  # _ is for sorting
        if k >= i + j - 1: return steps + i + j  # free walk with no obstacle
        for x, y in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
            if m > x >= 0 <= y < n:
                state = x, y, k - grid[x][y]
                if state not in seen and state[2] >= 0:
                    heapq.heappush(queue, (x+y+steps+1, steps+1, state))
                    seen.add(state)
    return -1

# LC64. Minimum Path Sum - min path sum matrix
def minPathSum(self, grid: List[List[int]]) -> int:  # O(mn) time, O(n) space
    M, N = len(grid), len(grid[0])
    dp = [0] + [float('inf')] * (N-1)
    for i in range(M):
        dp[0] += grid[i][0]
        for j in range(1, N):
            dp[j] = min(dp[j-1], dp[j]) + grid[i][j]
    return dp[-1]

# LC62. Unique Paths - no blocks
def uniquePaths(self, m: int, n: int) -> int:
    return math.comb(m+n-2, n-1)  # select n-1 from m+n-2
def uniquePaths(self, m, n):
    cur = [1] * n  # first row, only 1 way to each cell along this row
    for i in range(1, m):  # previous row + current row, 1st cell is always 1.
        for j in range(1, n): cur[j] += cur[j-1]  # previous cur[j] + current cur[j-1]
    return cur[-1]

# LC63. Unique Paths II - has blocks, from upper left to lower right
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    og = obstacleGrid
    n, m = len(og), len(og[0])
    if not m or not n: return 0
    idx = m
    for i, c in enumerate(og[0]):
        if c == 1:
            idx = i
            break
    cur = [1] * idx + [0] * (m - idx)
    for i in range(1, n):
        for j in range(0, m):
            if j == 0: # if previously not blocked, check now
                cur[j] = int(cur[j] != 0 and og[i][j] == 0)
            else:
                if og[i][j] == 1: cur[j] = 0
                else: cur[j] += cur[j-1]  # previous cur[j] + current cur[j-1]
    return cur[-1]

# LC980. Unique Paths III - has blocks, arbitrary start and end, obstacle
def uniquePathsIII(self, A): # O(3^n)
    m, n, empty = len(A), len(A[0]), 1
    for i in range(m):  # find start x, y and count empty cells
        for j in range(n):
            if A[i][j] == 1: x, y = (i, j) # find start
            elif A[i][j] == 0: empty += 1 # count empty
    self.res = 0
    def dfs(x, y, empty): # DFS on cells and empty cell count
        if not (0 <= x < m and 0 <= y < n and A[x][y] >= 0):
            return # obstacles
        if A[x][y] == 2:
            self.res += empty == 0 # reach goal and touch all
            return
        A[x][y] = -2 # mark visited
        for i, j in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
            dfs(i, j, empty - 1)
        A[x][y] = 0  # backout, since we need check history fills all empty cells
    dfs(x, y, empty)
    return self.res

# LC741. Cherry Pickup
def cherryPickup(self, grid: List[List[int]]) -> int:
    # greedy is not working on separate walkers. Have to consider all walkers as whole state.
    if not grid: return 0
    n, m = len(grid), len(grid[0])
    @lru_cache(None)  # dp
    def dp(r1, c1, c2):
        r2 = r1 + c1 - c2 # go with diagonal levels, so that shortest steps reach this level first
        if r1 < 0 or r2 < 0 or c1 < 0 or c2 < 0 or grid[r1][c1] == -1 or grid[r2][c2] == -1:
            return float('-inf')
        # we ignore r2, c2 since r1, c1 already picks up the cherry here, if any
        if r1 == 0 and c1 == 0: return grid[r1][c1] # baseline
        cherry = grid[r1][c1] + (c1 != c2) * grid[r2][c2]
        total = max(dp(r1, c1-1, c2-1),  # left, left
                    dp(r1, c1-1, c2),    # left, up, r2 = r1 + c1 -1 -c2 = curr r2 - 1
                    dp(r1-1, c1, c2),    # up, up
                    dp(r1-1, c1, c2-1))  # up, left
        return cherry + total
    ret = max(0, dp(n-1, m-1, m-1))
    return ret

# LC490. The Maze - soccer ball
def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:  # O(mn)
    q, visited = deque([start]), set()  # BFS super
    while q:
        i, j = q.popleft()
        if [i, j] == destination: return True
        for di,dj in [-1, 0], [0, -1], [1, 0], [0, 1]:
            ni, nj = i + di, j + dj
            while 0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and maze[ni][nj] == 0:
                ni, nj = ni + di, nj + dj
            ni, nj = ni - di, nj - dj  # need to backout 1 step
            if (ni,nj) not in visited:
                visited.add((ni,nj))
                q.append((ni,nj))
    return False

# LC120. Triangle - min sum tree path
def minimumTotal(self, triangle: List[List[int]]) -> int:  # O(n^2) time and O(n) space
    f = [0] * (len(triangle) + 1)  # extra 0 - just to unify below min()
    for row in triangle[::-1]:  # dp from back to front
        for i in range(len(row)):
            f[i] = row[i] + min(f[i], f[i + 1])
    return f[0]

# LC931. Minimum Falling Path Sum
def minFallingPathSum(self, A):  # O(n^2) time and O(n) space
    dp = A[0]
    for row in A[1:]:
        dp = [value + min(dp[c], dp[max(c - 1, 0)], dp[min(len(A) - 1, c + 1)]) for c, value in enumerate(row)]
    return min(dp)
