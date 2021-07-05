
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

# LC62. Unique Paths - no blocks
def uniquePaths(self, m: int, n: int) -> int:
    return math.comb(m+n-2, n-1) # select n-1 from m+n-2

def uniquePaths(self, m, n):
    if not m or not n: return 0
    cur = [1] * n  # first row
    for i in range(1, m):  # previous row + current row, 1st cell is always 1.
        for j in range(1, n): cur[j] += cur[j-1]  # previous cur[j] + current cur[j-1]
    return cur[-1]


# LC51. N-Queens
def solveNQueens(self, n: int) -> List[List[str]]:
    res, board = [], [] # O(n!)
    cols, diag, off_diag = set(), set(), set()
    def backtrack(i):  # recursion on rows
        if i == n: res.append(list(board))
        for j in range(n):
            if j not in cols and j-i not in diag and j+i not in off_diag:
                cols.add(j)  # order is not significant, these 4 steps are independent.
                diag.add(j-i)
                off_diag.add(j+i)
                board.append(j)
                backtrack(i+1) # recursion
                board.pop() # backout
                off_diag.remove(j+i)
                diag.remove(j-i)
                cols.remove(j)
    backtrack(0)
    res1 = [['.' * col + 'Q' + '.'*(n - col - 1) for col in board] for board in res]
    return res1

# LC348. Design Tic-Tac-Toe
class TicTacToe(object):  # 83%
    def __init__(self, n):
        self.row, self.col = [0] * n, [0] * n
        self.diag, self.anti_diag = 0, 0
        self.n = n
    def move(self, row, col, player):
        offset = player * 2 - 3 # either 1 or -1 for player 1 or 2
        self.row[row] += offset
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

# LC723. Candy Crush
class Solution:
    def candyCrush(self, board): # TIME : O((M * N))
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

# LC36. Valid Sudoku
def isValidSudoku(self, board: List[List[str]]) -> bool:  # one pass, faster, cache encoded positions
    digits = set('123456789')
    # 'row' + i, 'col' + j, 'sqr' + i-j as 3 keys to hold visited
    visited = defaultdict(set)
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
            lp = sum(nbs) - s # life support
            if s == 0 and lp == 3: # LIVE
                board[i][j] = 2 + board[i][j]  # move new life state to 2nd bit. 2 = LIVE << 1
            elif s == 1 and (lp == 2 or lp == 3): board[i][j] = 2 + board[i][j]
            # else: # DEAD, don't need to do anything since 2nd bit is 0 already
    for i in range(m): # we shift 2nd bit back to 1st bit, move from left to right
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
