# 给一个NxM的棋盘，每个格子有不同的权重，放两个rook（象棋的車）上去，怎样让两个rook站的位置获得最大的权重，条件限制是他们不能互相吃掉。

class SolutionTwoRook:
    def maxTwoWeight(self, board):  # O(mn)
        m, n = len(board), len(board[0])
        tl = [[float('-inf')] * n for _ in range(m)]
        tr = [[float('-inf')] * n for _ in range(m)]
        bl = [[float('-inf')] * n for _ in range(m)]
        br = [[float('-inf')] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    tl[i][j] = board[i][j]
                elif i == 0:
                    tl[i][j] = max(board[i][j], tl[i][j - 1])
                elif j == 0:
                    tl[i][j] = max(board[i][j], tl[i - 1][j])
                else:
                    tl[i][j] = max(board[i][j], tl[i][j - 1], tl[i - 1][j])

        for i in range(m - 1, -1, -1):
            for j in range(n):
                if i == m - 1 and j == 0:
                    bl[i][j] = board[i][j]
                elif i == m - 1:
                    bl[i][j] = max(board[i][j], bl[i][j - 1])
                elif j == 0:
                    bl[i][j] = max(board[i][j], bl[i + 1][j])
                else:
                    bl[i][j] = max(board[i][j], bl[i][j - 1], bl[i + 1][j])

        for i in range(m):
            for j in range(n - 1, -1, -1):
                if i == 0 and j == n - 1:
                    tr[i][j] = board[i][j]
                elif i == 0:
                    tr[i][j] = max(board[i][j], tr[i][j + 1])
                elif j == n - 1:
                    tr[i][j] = max(board[i][j], tr[i - 1][j])
                else:
                    tr[i][j] = max(board[i][j], tr[i][j + 1], tr[i - 1][j])

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i == m - 1 and j == n - 1:
                    br[i][j] = board[i][j]
                elif i == m - 1:
                    br[i][j] = max(board[i][j], br[i][j + 1])
                elif j == n - 1:
                    br[i][j] = max(board[i][j], br[i + 1][j])
                else:
                    br[i][j] = max(board[i][j], br[i][j + 1], br[i + 1][j])
        res = 0
        for i in range(m):
            for j in range(n):
                curr = board[i][j]
                tlMax = tl[i - 1][j - 1] if i - 1 >= 0 and j - 1 >= 0 else 0
                trMax = tr[i - 1][j + 1] if i - 1 >= 0 and j + 1 < n else 0
                blMax = bl[i + 1][j - 1] if i + 1 < m and j - 1 >= 0 else 0
                brMax = br[i + 1][j + 1] if i + 1 < m and j + 1 < n else 0

                res = max(res, curr + max(tlMax, trMax, blMax, brMax))

        return res
