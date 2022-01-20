from typing import List

# LC 2056. Number of Valid Move Combinations On Chessboard

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
# LC64. Minimum Path Sum
def minPathSum(self, grid: List[List[int]]) -> int:
    M, N = len(grid), len(grid[0])
    dp = [0] + [float('inf')] * (N-1)
    for i in range(M):
        dp[0] = dp[0] + grid[i][0]
        for j in range(1, N):
            dp[j] = min(dp[j-1], dp[j]) + grid[i][j]
    return dp[-1]

# LC73. Set Matrix Zeroes
def setZeroes(self, matrix):
    m, n = len(matrix), len(matrix[0])
    firstRowHasZero = not all(matrix[0])  # First row has zero?
    for i in range(1, m): # Use first row/column as marker, scan the matrix
        for j in range(n):
            if matrix[i][j] == 0: matrix[0][j] = matrix[i][0] = 0
    for i in range(1, m): # Set the zeros
        for j in range(n - 1, -1, -1):
            if matrix[i][0] == 0 or matrix[0][j] == 0: matrix[i][j] = 0
    # Set the zeros for the first row
    if firstRowHasZero: matrix[0] = [0] * n

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
def spiralOrder(self, matrix):
    result = []
    while matrix and matrix[0]:
        if matrix[0]: result += matrix.pop(0)
        if matrix and matrix[0]:
            for row in matrix: result.append(row.pop())
        if matrix and matrix[-1]: result += matrix.pop()[::-1]
        if matrix and matrix[0]:
            for row in matrix[::-1]: result.append(row.pop(0))
    return result

# LC632. Smallest Range Covering Elements from K Lists
def smallestRange(self, A): # O(nlogk)
    pq = [(row[0], i, 0) for i, row in enumerate(A)] # push 1st element from each list
    heapq.heapify(pq) # (value, row, column)
    ans = -1e9, 1e9
    right = max(row[0] for row in A)
    while pq:
        left, i, j = heapq.heappop(pq)
        if right - left < ans[1] - ans[0]: ans = left, right # track ans
        if j + 1 == len(A[i]): return ans # the min row reached end
        v = A[i][j+1] # replace minimal value with next one in same list
        right = max(right, v)
        heapq.heappush(pq, (v, i, j+1))

# LC221. Maximal Square
from collections import defaultdict
from itertools import product
def maximalSquare(self, matrix: List[List[str]]) -> int: # DP
    if not matrix: return 0
    rows, cols = len(matrix), len(matrix[0])
    # DP(i, j) is the largest side of all squares ended at (i, j)
    dp = defaultdict(int)
    max_len = 0 # track this
    for i, j in product(range(rows), range(cols)):
        if matrix[i][j] == '1':
            dp[i+1, j+1] = min([dp[i+1, j], dp[i, j+1], dp[i, j]]) + 1
            max_len = max(max_len, dp[i+1, j+1])
    return max_len ** 2

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

# LC1424. Diagonal Traverse II
def findDiagonalOrder(self, A):
    res = defaultdict(list)
    for i, r in enumerate(A):
        for j, a in enumerate(r): res[i + j].append(a)
    return [a for _, r in res.items() for a in reversed(r)]

# LC74. Search a 2D Matrix
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix: return False  # O(log(mn)
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1  # binary search
    while left <= right:
        pivot_idx = (left + right) // 2
        pivot_element = matrix[pivot_idx // n][pivot_idx % n]
        if target == pivot_element: return True
        else:
            if target < pivot_element: right = pivot_idx - 1
            else: left = pivot_idx + 1
    return False

# LC240. Search a 2D Matrix II
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix: return False
    h, w = len(matrix), len(matrix[0])
    row, col = h - 1, 0
    while row >= 0 and col < w:
        if target == matrix[row][col]: return True
        elif target < matrix[row][col]:  row -= 1
        else: col += 1
    return False

# LC48. Rotate Image
def rotate(self, A):
    A[:] = zip(*A[::-1])




# LC1314. Matrix Block Sum
def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
    m, n = len(mat), len(mat[0])
    rangeSum = [[0] * (n + 1) for _ in range(m + 1)] # 0 row and col are dummy
    for i in range(m):
        for j in range(n):
            rangeSum[i + 1][j + 1] = rangeSum[i + 1][j] + rangeSum[i][j + 1] - rangeSum[i][j] + mat[i][j]
    ans = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            r1, c1, r2, c2 = max(0, i - K), max(0, j - K), min(m, i + K + 1), min(n, j + K + 1)
            ans[i][j] = rangeSum[r2][c2] - rangeSum[r1][c2] - rangeSum[r2][c1] + rangeSum[r1][c1]
    return ans

# LC1274. Number of Ships in a Rectangle
def countShips(self, sea, P, Q):  # P - topRight, Q - bottomLeft
    res = 0
    if P.x >= Q.x and P.y >= Q.y and sea.hasShips(P, Q):
        if P.x == Q.x and P.y == Q.y: return 1
        mx, my = (P.x + Q.x) // 2, (P.y + Q.y) // 2
        # upper right
        res += self.countShips(sea, P, Point(mx + 1, my + 1))
        # upper left
        res += self.countShips(sea, Point(mx, P.y), Point(Q.x, my + 1))
        # lower left
        res += self.countShips(sea, Point(mx, my), Q)
        # lower right
        res += self.countShips(sea, Point(P.x, my), Point(mx + 1, Q.y))
    return res



# LC329. Longest Increasing Path in a Matrix
import functools # 100%
def longestIncreasingPath(self, matrix):
    if not matrix or not matrix[0]: return 0
    M, N = len(matrix), len(matrix[0])
    @functools.lru_cache(maxsize=None)
    def dfs(i, j):
        ret, val = 0, matrix[i][j]
        for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
            if 0 <= x < M and 0 <= y < N and val > matrix[x][y]:
                ret = max(ret, dfs(x, y))
        return ret + 1  # add this cell
    return max(dfs(x, y) for x in range(M) for y in range(N))



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
