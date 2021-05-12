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
