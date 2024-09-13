from collections import deque

# LC688. Knight Probability in Chessboard   nite prob
def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
    moves = ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
    @cache  # O(k * n^2) time and space
    def dp(m, i, j):  # prob at (i, j) on board at step m
        if not (0 <= i < n and 0 <= j < n): return 0
        if m == 0: return 1
        return sum(dp(m - 1, i + _i, j + _j) for _i, _j in moves) / 8
    return dp(k, row, column)

# LC576. Out of Boundary Paths
def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
    MOD = 10**9 + 7
    @cache  # O(mn * maxMove) time, O(mn) time. N:
    def f(i, j, c):
        if not (0 <= i < m and 0 <= j < n): return 1
        elif c:
            return sum(f(i+q,j,c-1) + f(i,j+q,c-1) for q in (-1,1))%MOD
        return 0  # if c=0, return 0
    return f(startRow, startColumn, maxMove)

# LC2596. Check Knight Tour Configuration
def checkValidGrid(self, grid: List[List[int]]) -> bool:
    n = len(grid)
    arr = [None] * (n*n)
    for row in range(n):
        for col in range(n): arr[grid[row][col]] = (row, col)
    pos = arr[0]
    if pos != (0, 0): return False
    for i in range(1, n*n):
        a = abs(pos[0] - arr[i][0])
        b = abs(pos[1] - arr[i][1])
        if a == 1 and b == 2 or a == 2 and b == 1: pos = arr[i]
        else: return False
    return True

# 1778. Shortest Path in a Hidden Grid
def findShortestPath(self, master: 'GridMaster') -> int:  # O(area)
    can_move = set() # need to build graph first, for bfs. Otherwise TLE.
    directions = {'U':(-1, 0, 'D'), 'D':(1, 0, 'U'), 'L':(0, 1, 'R'), 'R':(0, -1, 'L')}
    target = None
    def build_graph(x, y):
        if master.isTarget():
            nonlocal target
            target = (x, y)
            return
        can_move.add((x, y))
        for direction in directions:
            dx, dy, revert_direction = directions[direction]
            nx, ny = x + dx, y + dy
            if (nx, ny) not in can_move and master.canMove(direction):
                master.move(direction)
                build_graph(nx, ny)
                master.move(revert_direction)
    build_graph(0, 0)
    if target is None: return -1

    queue, level = [(0, 0)], 0 # bfs
    visited = set(queue)
    while queue:
        next_queue = queue
        queue = []
        level += 1
        for x, y in next_queue:
            for next_node in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if next_node == target: return level
                if next_node not in can_move or next_node in visited: continue
                visited.add(next_node)
                queue.append(next_node)
    return -1


# LC489. Robot Room Cleaner
def cleanRoom(self, robot):  # O(open cells)
    def go_back():  # backout one step
        robot.turnRight()
        robot.turnRight()  # turn back
        robot.move()
        robot.turnRight()
        robot.turnRight()  # turn to original dir
    def clean_cell(cell, cf):
        visited.add(cell)
        robot.clean()
        for i in range(4):
            new_cf = (cf + i) % 4  # e.g., facing right needs to start from 2nd index
            new_d = directions[new_cf]
            new_cell = (cell[0] + new_d[0], cell[1] + new_d[1])
            if not new_cell in visited and robot.move():
                clean_cell(new_cell, new_cf)
                go_back()
            robot.turnRight()  # turn the robot following chosen direction : clockwise
    # going clockwise : 0: 'up', 1: 'right', 2: 'down', 3: 'left'
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = set()
    clean_cell((0, 0), 0)

# LC1730. Shortest Path to Get Food
def getFood(self, grid: List[List[str]]) -> int:  # O(mn)
    rows, cols = len(grid), len(grid[0])
    for r, c in product(range(rows), range(cols)):
        if grid[r][c] == '*':
                start = (r,c)
                break
    directions = [(0, -1), (0, 1), (1, 0), (-1,0)]
    deque, seen = collections.deque([start]), set()
    res = 0
    while deque:
        for _ in range(len(deque)):
            currR, currC = deque.popleft()
            seen.add((currR, currC))
            if grid[currR][currC] == '#': return res
            for dx, dy in directions:
                x, y = currR + dx, currC + dy
                if 0 <= x < rows and 0 <= y < cols and grid[x][y] != 'X' and (x,y) not in seen:
                    deque.append((x,y))
                    seen.add((x,y))
        res += 1
    return -1

# LC994. Rotting Oranges
def orangesRotting(self, grid: List[List[int]]) -> int:  # O(rows * cols)
    rows, cols = len(grid), len(grid[0])
    rotten, fresh = set(), set()
    for i, j in product(range(rows), range(cols)):
        if grid[i][j] == 2: rotten.add((i, j))
        if grid[i][j] == 1: fresh.add((i, j))
    timer = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while fresh:  # BFS
        if not rotten: return -1
        rotten = {(i+di, j+dj) for i, j in rotten for di, dj in dirs if (i+di, j+dj) in fresh}
        fresh -= rotten
        timer += 1
    return timer

# LC174. Dungeon Game - knight princess
def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:  # O(n) space
    m, n = len(dungeon), len(dungeon[0])
    dp = [float('inf')] * (n+1)
    dp[n-1] = 1
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            cost = min(dp[j+1], dp[j]) - dungeon[i][j]
            dp[j] =  max(cost, 1)
    return dp[0]
def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:  # O(mn) time and space
    m, n = len(dungeon), len(dungeon[0])
    @functools.lru_cache(None)
    def dp(i,j):  # min health needed from i, j to end
        if (i,j) in ((m-1,n),(m,n-1)): return 1  # pass end cell
        if i == m or j == n: return math.inf  # go out of boundary
        ncost = min(dp(i+1,j), dp(i,j+1)) - dungeon[i][j]
        return max(ncost, 1)  # if < 1, it means dungeon has more orbs
    return dp(0,0)

# LC773. Sliding Puzzle - to 123,450
def slidingPuzzle(self, board: List[List[int]]) -> int:
    swaps = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]
    goal = [1,2,3, 4,5,0]
    board = board[0] + board[1]
    i = board.index(0)
    q = collections.deque([(board, 0, i)])
    visited = set([tuple(board)])
    while q:
        board, moves, i = q.popleft()
        if board == goal: return moves
        for j in swaps[i]:
            nxt = board[:]
            nxt[i], nxt[j] = nxt[j], nxt[i]
            if tuple(nxt) not in visited:
                visited.add(tuple(nxt))
                q.append((nxt, moves + 1, j))
    return -1

# LC529. Minesweeper
def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
    n, m = len(board), len(board[0])  ## O(mn)
    dirs = ((-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1))
    def dfs(i, j):  # don't need visited since we alter values already
        if board[i][j] == 'M': board[i][j] = 'X'  # we should get here, i.e., don't step on mine.
        elif board[i][j] == 'E':
            # neighbour mines
            nm = sum(board[i+dx][j+dy] == 'M' for dx, dy in dirs if 0 <= i+dx < n and 0 <= j+dy < m)
            board[i][j] = str(nm or 'B')
            if not nm: # no mine nearby
                for dx, dy in dirs:
                    if 0 <= i+dx < n and 0 <= j+dy < m: dfs(i + dx, j + dy)
            # the logic with mine nearby is complicated, with this logic missing
            # there could be cells not visited, e.g., a mine at (1, 1).
    dfs(*click)
    return board

# LC361. Bomb Enemy
def maxKilledEnemies(self, grid):  # O(mn) time and O(n) space
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    result = 0
    colhits = [0] * n
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if j == 0 or row[j-1] == 'W':
                rowhits = 0
                k = j
                while k < n and row[k] != 'W':
                    rowhits += row[k] == 'E'
                    k += 1
            if i == 0 or grid[i-1][j] == 'W':
                colhits[j] = 0
                k = i
                while k < m and grid[k][j] != 'W':
                    colhits[j] += grid[k][j] == 'E'
                    k += 1
            if cell == '0':
                result = max(result, rowhits + colhits[j])
    return result


# LC1197. Minimum Knight Moves  min knight move
def minKnightMoves(self, x: int, y: int) -> int:  # O(x*y)
    @lru_cache(None)
    def dp(x,y):  # O(x*y) in cache
        if x + y == 0: return 0  # (0, 0)
        elif x + y == 2: return 2  # (1, 1), (0, 2), (2, 0)
        return min(dp(abs(x-1), abs(y-2)), dp(abs(x-2), abs(y-1))) + 1
    return dp(abs(x), abs(y))  # first quardrant due to symmetry

# LC 2056. Number of Valid Move Combinations On Chessboard
def countCombinations(self, pieces, positions):
    D = ((), ((-1,0),(1,0),(0,-1),(0,1)),((-1,-1),(1,-1),(-1,1),(1,1)))
    P = {"rook":1, "bishop":2, "queen":3}
    n = len(pieces)
    M = (1<<8)-1  # 011111111, The kth bit of T[x][y] from right indicates a piece is at (x, y) at time k
    T = [[M] * 9 for _ in range(9)]  # 1 means empty, 0 means occupied
    def dfs(i): # search from ith piece
        if i == n: return 1
        p, (x0, y0) = pieces[i], positions[i]
        cnt = 0
        if T[x0][y0] == M:  # stay at its original position
            t = T[x0][y0]
            T[x0][y0] = 0  # set all bits to 0, (x0, y0) stays here all the time
            cnt += dfs(i+1)
            T[x0][y0] = t  # restore bits
        for dx, dy in D[P[p]&1]+D[P[p]&2]:  # move from its original position
            x, y, b = x0+dx, y0+dy, 1  # b is the time to move
            while 0 < x < 9 and 0 < y < 9 and (T[x][y] >> b & 1):  # rule 1
                T[x][y] ^= 1 << b  # rule 1a, move to (x, y), set bth bit to 0
                if (T[x][y] >> b+1) == (M >> b+1):  # rule 2a, if all future states are free to move
                    t = T[x][y]
                    T[x][y] &= (1 << b) - 1  # rule 2b, keep all future states to 0
                    cnt += dfs(i+1)
                    T[x][y] = t  # restore
                x, y, b = x+dx, y+dy, b+1
            while b > 1: T[(x:=x-dx)][(y:=y-dy)] |= (1 << (b:=b-1)) # restore
        return cnt
    return dfs(0)

# LC935. Knight Dialer
def knightDialer(self, n: int) -> int:  # O(n) time and O(1) space
    MOD = 10**9 + 7
    moves = [[4,6],[6,8],[7,9],[4,8],[3,9,0],[],[1,7,0],[2,6],[1,3],[2,4]]
    dp = [1] * 10  # counts for current hop and current digit
    for hops in range(n-1):
        dp2 = [0] * 10
        for node, count in enumerate(dp):  # loop all digits 0-9
            for nei in moves[node]:  # loop all jumps
                dp2[nei] += count
                dp2[nei] %= MOD
        dp = dp2
    return sum(dp) % MOD
import numpy as np
def knightDialer(self, N):  # O(logn)
    mod = 10**9 + 7
    if N == 1: return 10
    M = np.matrix([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                   [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]])
    res, N = 1, N - 1
    while N:
        if N % 2: res = res * M % mod
        M = M * M % mod
        N //= 2
    return int(np.sum(res)) % mod

# LC419. Battleships in a Board
def countBattleships(self, board: List[List[str]]) -> int:
    total = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'X':
                flag = 1
                if j > 0 and board[i][j-1] == 'X': flag = 0  # ignore double count
                if i > 0 and board[i-1][j] == 'X': flag = 0
                total += flag
    return total


# LC51. N-Queens - n queens
def solveNQueens(self, n: int) -> List[List[str]]:
    res, board = [], [] # O(n!)
    cols, diag, off_diag = set(), set(), set()
    def backtrack(i):  # recursion on rows
        if i == n:
            res.append(list(board))
            return
        for j in range(n):
            if j not in cols and j-i not in diag and j+i not in off_diag:
                cols.add(j)  # order is not significant, these 4 steps are independent.
                diag.add(j-i)
                off_diag.add(j+i)
                board.append(j)
                backtrack(i+1)  # recursion
                board.pop()  # backout
                off_diag.remove(j+i)
                diag.remove(j-i)
                cols.remove(j)
    backtrack(0)
    res1 = [['.' * col + 'Q' + '.'*(n - col - 1) for col in board] for board in res]
    return res1



# LC1034. Coloring A Border
def colorBorder(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
    seen, m, n = set(), len(grid), len(grid[0])
    def dfs(x, y):  # return: is the right component or not
        if (x, y) in seen: return True
        if not (0 <= x < m and 0 <= y < n and grid[x][y] == grid[row][col]): return False
        seen.add((x, y)) # now it's in the same component
        if dfs(x + 1, y) + dfs(x - 1, y) + dfs(x, y + 1) + dfs(x, y - 1) < 4:
            grid[x][y] = color  # this is a border
        return True
    dfs(row, col)
    return grid

# LC723. Candy Crush
def candyCrush(self, board):  # TIME : O((M * N)^2)
    R, C = len(board), len(board[0])
    crushed = False
    for i in range(R):  # mark crushed cells by negative numbers
        for j in range(C):
            if board[i][j] == 0: continue
            v = abs(board[i][j])
            # for vertical crush
            if i < R - 2 and v == abs(board[i+1][j]) == abs(board[i+2][j]):
                board[i][j] = board[i+1][j] = board[i+2][j] = -v # This is the key thought
                crushed = True
            # for horizontal crush
            if j < C - 2 and v == abs(board[i][j+1]) == abs(board[i][j+2]):
                board[i][j] = board[i][j+1] = board[i][j+2] = -v
                crushed = True
    if crushed:
        for j in range(C): # for each column, crush from bottom in code.
            row_idx = R - 1 # but logically crush from top.
            for i in range(R-1, -1, -1):
                if board[i][j] > 0:  # if not crushed, move down
                    board[row_idx][j] = board[i][j]
                    row_idx -= 1
            while row_idx >= 0:  # zero out above
                board[row_idx][j] = 0
                row_idx -= 1
    return self.candyCrush(board) if crushed else board

# LC1926. Nearest Exit from Entrance in Maze -
def nearestExit(self, maze: List[List[str]], start: List[int]) -> int:  # O(MN), BFS
    M, N = len(maze), len(maze[0])
    isExit = lambda i, j: not i or i == M - 1 or not j or j == N - 1
    que, seen, level = deque([[*start]]), {tuple(start)}, 0
    while que:
        for _ in range(len(que)):
            i, j = que.popleft()
            if isExit(i, j) and level: return level
            for u, v in [i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]:
                if M > u >= 0 <= v < N and maze[u][v] == '.' and (u, v) not in seen:
                    que.append([u, v])
                    seen.add((u, v))
        level += 1
    return -1

# LC289. Game of Life
def gameOfLife(self, board: List[List[int]]) -> None: # 1 bit for old, 1 bit for new.
    m, n = len(board), len(board[0])
    for i in range(m): # use 2 bits for this life and future life
        for j in range(n):
            # find neighbours
            nbs = [board[r][c] & 1 for r in range(i-1, i+2) for c in range(j-1, j+2) if 0 <= r < m and 0 <=c < n]
            s = board[i][j]
            lp = sum(nbs) - s  # life support
            if s == 0 and lp == 3:  # LIVE
                board[i][j] = 2 + board[i][j]  # move new life state to 2nd bit. 2 = LIVE << 1
            elif s == 1 and (lp == 2 or lp == 3): board[i][j] = 2 + board[i][j]
            # else: # DEAD, don't need to do anything since 2nd bit is 0 already
    for i in range(m):  # we shift 2nd bit back to 1st bit, move from left to right
        for j in range(n):
            board[i][j] >>= 1



# LC909. Snakes and Ladders  snake ladder  snake ladder
def snakesAndLadders(self, board: List[List[int]]) -> int:
    n = len(board)
    def coord(order):
        q, r = divmod(order-1, n)
        x = n - 1 - q
        y = r if q % 2 == 0 else n-1-r  # even and odd rows
        return x, y
    queue, visited = deque([(1, 0)]), set()  # order, steps
    maxs = n * n
    while queue:  # BFS to get min
        x, s = queue.popleft()
        if x == maxs: return s
        if x in visited: continue
        visited.add(x)
        for i in range(6):
            move = x + i + 1
            if move > maxs: continue
            x1, y1 = coord(move)
            if board[x1][y1] != -1: move = board[x1][y1]
            if move not in visited:
                queue.append((move, s+1))
    return -1
