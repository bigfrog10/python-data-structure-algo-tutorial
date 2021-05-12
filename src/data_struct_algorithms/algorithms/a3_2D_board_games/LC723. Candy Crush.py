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
                    if board[i][j] > 0:  # if not crushed, move up
                        board[row_idx][j] = board[i][j]
                        row_idx -= 1
                while row_idx >= 0: # zero out above
                    board[row_idx][j] = 0
                    row_idx -= 1
        return self.candyCrush(board) if crushed else board
