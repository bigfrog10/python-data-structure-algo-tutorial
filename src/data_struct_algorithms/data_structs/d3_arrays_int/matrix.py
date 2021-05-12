# LC498. Diagonal Traverse
def findDiagonalOrder(self, matrix):
    if not matrix: return []
    m, n = len(matrix), len(matrix[0])
    ret = []
    row = col = 0
    for _ in range(m * n):
        ret.append(matrix[row][col])
        if (row + col) % 2 == 0: # start from row, move up
            if col == n - 1: row += 1 # hit right side for next sum = row + col
            elif row == 0: col += 1
            else:
                row -= 1
                col += 1
        else: # start from col, move down
            if row == m - 1: col += 1 # hit bottom for next sum = row + col
            elif col == 0: row += 1
            else:
                row += 1
                col -= 1
    return ret

# LC311. Sparse Matrix Multiplication
def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    if not A or not A[0] or not B or not B[0]:
        return [[]]
    sparse_A = self.get_none_zero(A)
    sparse_B = self.get_none_zero(B)
    n, m, k = len(A), len(A[0]), len(B[0])
    C = [[0] * k for _ in range(n)]
    for i, j, val_A in sparse_A:
        for x, y, val_B in sparse_B:
            if j == x:
                C[i][y] += val_A * val_B
    return C
def get_none_zero(self, A):
    res = []
    n, m = len(A), len(A[0])
    for i in range(n):
        for j in range(m):
            if A[i][j] == 0:
                continue
            res.append((i, j, A[i][j]))  # we should model sparse matrix like this
    return res

# LC1570. Dot Product of Two Sparse Vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.sparse = {}
        for i, n in enumerate(nums):
            if n != 0: self.sparse[i] = n
    def dotProduct(self, vec: 'SparseVector') -> int:
        dot = 0
        for i, n in self.sparse.items():
            if i in vec.sparse: dot += n * vec.sparse[i]
        return dot

# LC54. Spiral Matrix, top100
def spiralOrder(self, matrix):
    res = []
    while matrix:
        res.extend(matrix.pop(0))
        # zip rows to columns, flattern each column, reverse order
        matrix = [*zip(*matrix)][::-1]
    return res
# [[1,2,3],[4,5,6],[7,8,9]] ->  [(6, 9), (5, 8), (4, 7)] -.  [(8, 7), (5, 4)]
# -> [(4,), (5,)] -> [(5,)]

# LC59. Spiral Matrix II
def generateMatrix(self, n): # this is python 3 version
    res, lo = [[n*n]], n*n
    while lo > 1:
        lo, hi = lo - len(res), lo
        print('res:', res)
        res = [[i for i in range(lo, hi)]] + [list(j) for j in zip(*res[::-1])]
    return res
# [[9]] -> [[8], [9]] -> [[6, 7], [9, 8]] -> [[4, 5], [9, 6], [8, 7]]
# -> [[1,2,3],[8,9,4],[7,6,5]]

# LC885. Spiral Matrix III
def spiralMatrixIII(self, R, C, r0, c0):
    i, j = r0, c0
    coordinates = [[r0, c0]]
    step_size, sign = 1, 1
    while len(coordinates) < R*C:
        for _ in range(step_size):
            j += sign # follow row
            if 0 <= i < R and 0 <= j < C: coordinates.append([i, j])
        for _ in range(step_size):
            i += sign # follow column
            if 0 <= i < R and 0 <= j < C: coordinates.append([i, j])
        step_size += 1
        sign *= -1
    return coordinates

# LC48. Rotate Image
def rotate(self, A):
    A[:] = zip(*A[::-1])

def rotate(self, matrix: List[List[int]]) -> None:
    # divide and conquer, outer borders, one layer at a time
    def border_rotate(matrix, pi):  # i is the upper right corner, pivot index
        h = len(matrix) - 2 * pi
        if h == 0: return # False  # signal time to stop
        hi = h - 1  # last index from pivot to the other corner
        last = pi + hi # last index from pivot to the other corner - 1
        for i in range(hi):  # we stop before the corner
            tmp = matrix[pi][pi+i]  # upper left corner to tmp
            matrix[pi][pi+i] = matrix[last - i][pi]  # lower left corner to upper left
            matrix[last - i][pi] = matrix[last][last-i]  # lower right to lower left
            matrix[last][last-i] = matrix[pi+i][last]  # upper right to lower right
            matrix[pi+i][last] = tmp  # upper left, tmp, to upper right
    for i in range(len(matrix) // 2):
        border_rotate(matrix, i)

# LC329. Longest Increasing Path in a Matrix
import functools # 100%
def longestIncreasingPath(self, matrix):
    if not matrix or not matrix[0]: return 0
    M, N = len(matrix), len(matrix[0])
    @functools.lru_cache(maxsize=None)
    def dfs(i, j):
        ret, val = 0, matrix[i][j]
        for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
            if 0 <= x < M and 0 <= y < N and val < matrix[x][y]:
                ret = max(ret, dfs(x, y))
        return ret + 1  # add this cell
    return max(dfs(x, y) for x in range(M) for y in range(N))

# LC73. Set Matrix Zeroes
def setZeroes(self, matrix):
    # First row has zero?
    m, n = len(matrix), len(matrix[0])
    firstRowHasZero = not all(matrix[0])
    # Use first row/column as marker, scan the matrix
    for i in range(1, m):
        for j in range(n):
            if matrix[i][j] == 0: matrix[0][j] = matrix[i][0] = 0
    # Set the zeros
    for i in range(1, m):
        for j in range(n - 1, -1, -1):
            if matrix[i][0] == 0 or matrix[0][j] == 0: matrix[i][j] = 0
    # Set the zeros for the first row
    if firstRowHasZero: matrix[0] = [0] * n

# LC251. Flatten 2D Vector
class Vector2D:
    def __init__(self, v: List[List[int]]):
        self.nums = []
        for inner_list in v:
            self.nums.extend(inner_list)
        self.position = -1
    def next(self) -> int:
        self.position += 1
        return self.nums[self.position]
    def hasNext(self) -> bool:
        return self.position + 1 < len(self.nums)

# LC1428. Leftmost Column with at Least a One
def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
    if not binaryMatrix: return -1
    rs, cs = binaryMatrix.dimensions()
    i, j = 0, cs - 1
    ret = -1
    while i < min(rs, cs) and j >= 0:
        e = binaryMatrix.get(i, j)
        if e == 1:
            ret = j
            j -= 1
        elif e == 0: i += 1
    return ret

# LC149. Max Points on a Line
def maxPoints(self, points: List[List[int]]) -> int:
    def helper(currentPoint, points):
        slopes,duplicates,ans = {},0,0
        x1, y1 = currentPoint
        for x2, y2 in points:
            if x1 == x2 and y1 == y2: duplicates += 1
            else: # else find the slop and add in dic
                slope = (x2 - x1) / (y2 - y1) if y2 != y1 else 'inf'
                count = slopes.get(slope, 0) + 1
                slopes[slope] = count
                ans = max(ans, count)
        return ans + 1 + duplicates
    ans = 0
    while points:
        currentPoint = points.pop()
        ans = max(ans, helper(currentPoint, points))
    return ans
