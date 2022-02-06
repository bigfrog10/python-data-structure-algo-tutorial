from collections import deque

# LC1263. Minimum Moves to Move a Box to Their Target Location
def minPushBox(self, grid: List[List[str]]) -> int:  # faster BFS, O((mn)^2)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "T": target = (i,j)
            if grid[i][j] == "B": box = (i,j)
            if grid[i][j] == "S": person = (i,j)
    def empty(x,y): # O(1) verify
        return 0 <= x <len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != '#'
    def reachable(curr, dest, box):  #BFS to check dest is reachable from curr
        que, v = deque([curr]), set()
        while que:
            pos = que.popleft()
            if pos == dest: return True
            for x,y in [(pos[0]+1,pos[1]), (pos[0]-1,pos[1]), (pos[0],pos[1]+1), (pos[0],pos[1]-1)]:
                if empty(x,y) and (x,y) not in v and (x,y) != box:
                    v.add((x,y))
                    que.append((x,y))
        return False
    q, visited = deque([(0, box, person)]), {box + person}
    while q:  # main BFS
        dist, box, person = q.popleft()
        if box == target: return dist
        b_coord = [(box[0]+1,box[1]),(box[0]-1,box[1]),(box[0],box[1]+1),(box[0],box[1]-1)]
        p_coord = [(box[0]-1,box[1]),(box[0]+1,box[1]),(box[0],box[1]-1),(box[0],box[1]+1)]
        for new_box,new_person in zip(b_coord,p_coord):
            if empty(*new_box) and new_box+box not in visited:
                if empty(*new_person) and reachable(person,new_person,box):
                    visited.add(new_box+box)
                    q.append((dist+1,new_box,box))
    return -1

# LC1197. Minimum Knight Moves
def minKnightMoves(self, x: int, y: int) -> int:
    @lru_cache(None)
    def dp(x,y):
        if x + y == 0: return 0  # (0, 0)
        elif x + y == 2: return 2  # (1, 1), (0, 2), (2, 0)
        return min(dp(abs(x-1), abs(y-2)), dp(abs(x-2), abs(y-1))) + 1
    return dp(abs(x),abs(y))  # first quardrant due to symmetry

# LC348. Design Tic-Tac-Toe
class TicTacToe:
    def __init__(self, n: int):
        self.row, self.col = [0] * n, [0] * n
        self.diag, self.anti_diag = 0, 0
        self.n = n
    def move(self, row: int, col: int, player: int) -> int:
        offset = player * 2 - 3  # either 1 or -1 for player 1 or 2
        self.row[row] += offset  # mark this player position with -1 or 1
        self.col[col] += offset
        if row == col: self.diag += offset
        if row + col == self.n - 1: self.anti_diag += offset
        if offset * self.n in [self.row[row], self.col[col], self.diag, self.anti_diag]:
            return player
        return 0

# LC794. Valid Tic-Tac-Toe State
def validTicTacToe(self, board):
    FIRST, SECOND = 'XO'
    x_count = sum(row.count(FIRST) for row in board)
    o_count = sum(row.count(SECOND) for row in board)
    def win(board, player):
        for i in range(3):
            if all(board[i][j] == player for j in range(3)): return True
            if all(board[j][i] == player for j in range(3)): return True
        return (player == board[1][1] == board[0][0] == board[2][2] or
                player == board[1][1] == board[0][2] == board[2][0])
    if o_count not in {x_count-1, x_count}: return False
    if win(board, FIRST) and x_count-1 != o_count: return False
    if win(board, SECOND) and x_count != o_count: return False
    return True

# LC1275. Find Winner on a Tic Tac Toe Game
def tictactoe(self, moves: List[List[int]]) -> str:
    row, col, diag = [0] * 3, [0] * 3, [0] * 2
    player = 1
    for i, j in moves:
        row[i] += player
        col[j] += player
        if i == j: diag[0] += player
        if i + j == 2: diag[1] += player
        if row[i] == 3 or col[j] == 3 or 3 in diag: return "A"
        if row[i] == -3 or col[j] == -3 or -3 in diag: return "B"
        player = -player
    return "Draw" if len(moves) == 9 else "Pending"

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
def knightDialer(self, n: int) -> int:
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

# LC489. Robot Room Cleaner
def cleanRoom(self, robot):
    def go_back():
        robot.turnRight()
        robot.turnRight()  # turn back
        robot.move()
        robot.turnRight()
        robot.turnRight()  # turn to original dir
    def backtrack(cell, cf):
        visited.add(cell)
        robot.clean()
        for i in range(4):
            new_d = (cf + i) % 4  # e.g., facing right needs to start from 2nd index
            new_cell = (cell[0] + directions[new_d][0], cell[1] + directions[new_d][1])
            if not new_cell in visited and robot.move():
                backtrack(new_cell, new_d)
                go_back()
            # turn the robot following chosen direction : clockwise
            robot.turnRight()  # as
    # going clockwise : 0: 'up', 1: 'right', 2: 'down', 3: 'left'
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = set()
    backtrack((0, 0), 0)

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

# LC529. Minesweeper
def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
    n, m = len(board), len(board[0])  # https://leetcode.com/problems/minesweeper/discuss/99897/10-line-python-solution
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

# LC51. N-Queens
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

# LC1102. Path With Maximum Minimum Value
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

# LC994. Rotting Oranges
def orangesRotting(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    rotten, fresh = set(), set()
    for i, j in product(range(rows), range(cols)):
        if grid[i][j] == 2: rotten.add((i, j))
        if grid[i][j] == 1: fresh.add((i, j))
    timer = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while fresh: # BFS
        if not rotten: return -1
        rotten = {(i+di, j+dj) for i, j in rotten for di, dj in dirs if (i+di, j+dj) in fresh}
        fresh -= rotten
        timer += 1
    return timer

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
def candyCrush(self, board):  # TIME : O(M * N)
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
def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
    q, visited = deque([start]), set()  # BFS super
    while q:
        i,j = q.popleft()
        if [i, j] == destination: return True
        for di,dj in [-1,0],[0,-1],[1,0],[0,1]:
            ni, nj = i + di  , j + dj
            while 0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and maze[ni][nj] == 0 :
                ni, nj  = ni + di , nj + dj
            ni, nj  = ni - di, nj - dj # need to backout 1 step
            if (ni,nj) not in visited:
                visited.add((ni,nj))
                q.append((ni,nj))
    return False

# LC36. Valid Sudoku
def isValidSudoku(self, board: List[List[str]]) -> bool:  # one pass, faster, cache encoded positions
    digits = set('123456789')
    visited = defaultdict(set)  # 'row' + i, 'col' + j, 'sqr' + i-j as 3 keys to hold visited
    for i in range(9):
        row = board[i]
        for j in range(9):
            c = row[j]
            if c == '.': continue
            if c not in digits: return False
            showed = visited['row' + str(i)]
            if c in showed: return False
            showed.add(c)
            showed = visited['col' + str(j)]
            if c in showed: return False
            showed.add(c)
            showed = visited['sqr' + str(i // 3) + '-' + str(j // 3)]
            if c in showed: return False
            showed.add(c)
    return True

# LC37. Sudoku Solver
def solveSudoku(self, board): # fast, since no n so O(1)
    rows, cols = collections.defaultdict(set), collections.defaultdict(set)
    triples, empties = collections.defaultdict(set), collections.deque()
    for r in range(9):
        for c in range(9):
            if board[r][c] == ".": empties.append((r, c))
            else:
                rows[r].add(board[r][c])
                cols[c].add(board[r][c])
                triples[(r // 3, c // 3)].add(board[r][c])
    def dfs():
        if not empties: return True
        r, c = empties[0]
        t = (r // 3, c // 3)
        for dig in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            if dig not in rows[r] and dig not in cols[c] and dig not in triples[t]:
                board[r][c] = dig
                rows[r].add(dig)
                cols[c].add(dig)
                triples[t].add(dig)
                empties.popleft()
                if dfs(): return True
                else:  # backout
                    board[r][c] = "."
                    rows[r].discard(dig)
                    cols[c].discard(dig)
                    triples[t].discard(dig)
                    empties.appendleft((r, c))
        return False
    dfs()

# 1293. Shortest Path in a Grid with Obstacles Elimination
def shortestPath(self, grid: List[List[int]], k: int) -> int:  # best, A*
    m, n = len(grid), len(grid[0])
    state = m-1, n-1, k
    queue, seen = [(m+n-2, 0, state)], {state}  # manhattan distance
    while queue:
        _, steps, (i, j, k) = heapq.heappop(queue)
        if k >= i + j - 1: return steps + i + j  # free walk with no obstacle
        for x, y in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
            if m > x >= 0 <= y < n:
                state = x, y, k - grid[x][y]
                if state not in seen and state[2] >= 0:
                    heapq.heappush(queue, (x+y+steps+1, steps+1, state))
                    seen.add(state)
    return -1

# LC62. Unique Paths - no blocks
def uniquePaths(self, m: int, n: int) -> int:
    return math.comb(m+n-2, n-1)  # select n-1 from m+n-2
def uniquePaths(self, m, n):
    if not m or not n: return 0
    cur = [1] * n  # first row
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

# LC980. Unique Paths III - has blocks, arbitrary start and end
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

# LC361. Bomb Enemy
def maxKilledEnemies(self, grid):
    maxEnemy = 0
    tgrid = [list(i) for i in zip(*grid)]
    for i in range(len(grid)): # for each row, we duplicate scan for each seg(sep by walls)
        for j in range(len(grid[0])):
            if grid[i][j] == '0':
                maxEnemy = max(maxEnemy,
                               self.countEInRow(j, grid[i]) + self.countEInRow(i, tgrid[j]))
    return maxEnemy
def countEInRow(self, i, row):
    #if len(row) == 1: return 0
    tempE = 0
    for j in range(i+1, len(row)): # move right
        if row[j] == 'E': tempE += 1
        if row[j] == 'W': break
    for j in range(i-1,-1,-1): # move left
        if row[j] == 'E': tempE += 1
        if row[j] == 'W': break
    return tempE
