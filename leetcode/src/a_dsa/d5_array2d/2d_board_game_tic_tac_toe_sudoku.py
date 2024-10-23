
# LC348. Design Tic-Tac-Toe   tic tac toe
class TicTacToe:
    def __init__(self, n: int):
        self.rows, self.cols = [0] * n, [0] * n
        self.diag, self.anti_diag = 0, 0
        self.n = n
    def move(self, row: int, col: int, player: int) -> int:
        offset = player * 2 - 3  # either 1 or -1 for player 1 or 2
        self.rows[row] += offset  # mark this player position with -1 or 1
        self.cols[col] += offset
        if row == col: self.diag += offset
        if row + col == self.n - 1: self.anti_diag += offset
        if offset * self.n in [self.rows[row], self.cols[col], self.diag, self.anti_diag]:
            return player
        return 0

# LC1275. Find Winner on a Tic Tac Toe Game
def tictactoe(self, moves: List[List[int]]) -> str:
    row, col, diag = [0] * 3, [0] * 3, [0] * 2  # results
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
    visited = defaultdict(set)  # 'row' + i, 'col' + j, 'sqr' + i-j as 3 keys to hold visited
    for i in range(9):
        row = board[i]
        for j in range(9):
            c = row[j]
            if c == '.': continue
            if c not in '123456789': return False
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
        r, c = empties.popleft()
        t = (r // 3, c // 3)
        for dig in "123456789":
            if dig not in rows[r] and dig not in cols[c] and dig not in triples[t]:
                board[r][c] = dig
                rows[r].add(dig)
                cols[c].add(dig)
                triples[t].add(dig)
                if dfs(): return True
                else:  # backout
                    board[r][c] = "."
                    rows[r].discard(dig)
                    cols[c].discard(dig)
                    triples[t].discard(dig)
        empties.appendleft((r, c))  # backout, try other digits
        return False
    dfs()





