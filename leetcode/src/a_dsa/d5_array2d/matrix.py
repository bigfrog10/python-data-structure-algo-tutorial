from typing import List
import itertools

# LC48. Rotate Image
def rotate(self, A):
    A[:] = zip(*A[::-1])
def rotate(self, matrix: List[List[int]]) -> None:
    n = len(matrix)
    for i in range(n//2): matrix[i], matrix[~i] = matrix[~i], matrix[i]
    for i,j in itertools.combinations(range(n), 2):  # flip around diagonal
        matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
def rotate(self, matrix: List[List[int]]) -> None:  # right clockwise by 90 degrees
    n = len(matrix[0])
    for i in range(n // 2 + n % 2):
        for j in range(n // 2):
            tmp = matrix[n - 1 - j][i]
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
            matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i]
            matrix[j][n - 1 - i] = matrix[i][j]
            matrix[i][j] = tmp
def transpose(self, matrix):  # flip along diagonal
    return zip(*matrix)
def reflect(self, matrix):  # flip from left to right
    return [matrix[i][::-1] for i in range(len(matrix))]

# LC867. Transpose Matrix
def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
    return zip(*matrix)

# LC149. Max Points on a Line
def maxPoints(self, points: List[List[int]]) -> int:  # O(n^2)
    points.sort()
    slope, res = defaultdict(int), 0
    for i, (x1, y1) in enumerate(points):
        slope.clear()
        for x2, y2 in points[i + 1:]:
            dx, dy = x2 - x1, y2 - y1  # this is where we need sort
            g = gcd(dx, dy)
            m = (dx // g, dy // g)
            slope[m] += 1
            res = max(res, slope[m])
    return res + 1  # plus the 1st point

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

# LC498. Diagonal Traverse
def findDiagonalOrder(self, matrix):  # O(mn)
    if not matrix: return []
    m, n = len(matrix), len(matrix[0])
    ret = []
    row = col = 0
    for _ in range(m * n):
        ret.append(matrix[row][col])
        if (row + col) % 2 == 0:  # start from row, move up
            if col == n - 1: row += 1  # hit right, move down
            elif row == 0: col += 1  # hit top, move right
            else:  # the order of if-else check is significant
                row -= 1
                col += 1
        else:  # start from col, move down
            if row == m - 1: col += 1  # hit bottom, move right
            elif col == 0: row += 1  # hit left, move down
            else:
                row += 1
                col -= 1
    return ret

# LC1428. Leftmost Column with at Least a One - sorted 01 matrix
def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:  # O(n + m), diagonal
    rows, cols = binaryMatrix.dimensions()
    row, col = 0, cols - 1  # upper right corner
    while row < rows and col >= 0:
        if binaryMatrix.get(row, col) == 0: row += 1  # move down
        else: col -= 1  # move left
    # If we never left the last column, it must have been all 0's.
    return col + 1 if col != cols - 1 else -1

# LC766. Toeplitz Matrix
def isToeplitzMatrix(self, matrix):  # O(mn) runtime, O(1) space, has follow ups
    return all(r == 0 or c == 0 or matrix[r-1][c-1] == val
               for r, row in enumerate(matrix)
               for c, val in enumerate(row))
def isToeplitzMatrix(self, m):
    return all(r1[:-1] == r2[1:] for r1,r2 in zip(m, m[1:]))

# LC1424. Diagonal Traverse II
def findDiagonalOrder(self, A):
    res = defaultdict(list)
    for i, r in enumerate(A):
        for j, a in enumerate(r): res[i + j].append(a)
    return [a for _, r in res.items() for a in reversed(r)]

# LC311. Sparse Matrix Multiplication
def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    if not A or not A[0] or not B or not B[0]: return [[]]
    def get_none_zero(A):
        n, m, res = len(A), len(A[0]), []
        for i, j in itertools.product(range(n), range(m)):
            if A[i][j] != 0: res.append((i, j, A[i][j]))  # we should model sparse matrix like this
        return res  # this list should use smaller space than the matrix
    sparse_A, sparse_B = get_none_zero(A), get_none_zero(B)
    n, m, k = len(A), len(A[0]), len(B[0])
    C = [[0] * k for _ in range(n)]
    for i, j, val_A in sparse_A:
        for x, y, val_B in sparse_B:
            if j == x: C[i][y] += val_A * val_B
    return C

# LC378. Kth Smallest Element in a Sorted Matrix
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:  # O(klogk) time and O(k) space
    m, n = len(matrix), len(matrix[0])  # For general, the matrix need not be a square
    minHeap = []  # val, r, c
    for r in range(min(k, m)): heappush(minHeap, (matrix[r][0], r, 0))
    ans = -1  # any dummy value
    for i in range(k):
        ans, r, c = heappop(minHeap)
        if c+1 < n: heappush(minHeap, (matrix[r][c + 1], r, c + 1))
    return ans
def kthSmallest(self, matrix, k):
    m, n = len(matrix), len(matrix[0])  # For general, the matrix need not be a square
    def countLessOrEqual(x):
        cnt = 0
        c = n - 1  # start with the rightmost column
        for r in range(m):
            while c >= 0 and matrix[r][c] > x: c -= 1  # decrease column until matrix[r][c] <= x
            cnt += (c + 1)
        return cnt
    left, right = matrix[0][0], matrix[-1][-1]
    ans = -1
    while left <= right:
        mid = (left + right) // 2
        if countLessOrEqual(mid) >= k:
            ans = mid
            right = mid - 1  # try to looking for a smaller value in the left side
        else:
            left = mid + 1  # try to looking for a bigger value in the right side
    return ans

# LC2033. Minimum Operations to Make a Uni-Value Grid - uni value, univalue
def minOperations(self, grid: List[List[int]], x: int) -> int:
    vals = list(itertools.chain(*grid))  # flatting matrix to array - [[2,4],[6,8]] ->  [2,4,6,8]
    if len(set(val % x for val in vals)) > 1: return -1  # if we have 2 diff residues, can't do it.
    median = heapq.nlargest((len(vals)+1) // 2, vals)[-1]  # O(N) possible via "quick select", return 6 for 8, 6
    return sum(abs(val - median)//x for val in vals)

# LC1861. Rotating the Box - stones, obstacles
def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
    for row in box:
        bottom = len(row) - 1            # initialize with the last position in row
        for j in range(len(row))[::-1]:  # iterate from the end of the row
            if row[j] == "*":            # we cannot move stones behind obstacles,
                bottom = j - 1           # so update move position to the first before obstacle
            elif row[j] == "#":          # if stone, move it to the "move_position"
                row[bottom], row[j] = row[j], row[bottom]
                bottom -= 1
    return zip(*box[::-1])               # rotate array, or list(...)

# LC1878. Get Biggest Three Rhombus Sums in a Grid
def getBiggestThree(self, grid):  # O(C), C number of cells
    m, n, heap = len(grid), len(grid[0]), []

    def update(heap, num):
        if num not in heap:
            heappush(heap, num)
            if len(heap) > 3: heappop(heap)
        return heap

    for num in chain(*grid): update(heap, num)

    @lru_cache(None)
    def dp(i, j, dr):
        if not 0 <= i < n or not 0 <= j < m: return 0
        return dp(i-1, j+dr, dr) + grid[j][i]

    for q in range(1, (1 + min(m, n))//2):  # q is center to point length in the square case
        for i in range(q, n - q):
            for j in range(q, m - q):
                p1 = dp(i + q, j, -1) - dp(i, j - q, -1)  # upper right edge without upper point
                p2 = dp(i - 1, j + q - 1, -1) - dp(i - q - 1, j - 1, -1)  # lower left edge without lower point
                p3 = dp(i, j - q, 1) - dp(i - q, j, 1)  # upper left edge without left point
                p4 = dp(i + q - 1, j + 1, 1) - dp(i - 1, j + q + 1, 1)  # lower right edge without right point
                update(heap, p1 + p2 + p3 + p4)

    return sorted(heap)[::-1]

# LC764. Largest Plus Sign
def orderOfLargestPlusSign(self, N: int, mines: List[List[int]]) -> int:  # O(N^2)
    grid = [[N] * N for i in range(N)]

    for m in mines: grid[m[0]][m[1]] = 0

    for i in range(N):
        l, r, u, d = 0, 0, 0, 0
        for j, k in zip(range(N), reversed(range(N))):
            l = l + 1 if grid[i][j] != 0 else 0
            if l < grid[i][j]:
                grid[i][j] = l
            r = r + 1 if grid[i][k] != 0 else 0
            if r < grid[i][k]:
                grid[i][k] = r
            u = u + 1 if grid[j][i] != 0 else 0
            if u < grid[j][i]:
                grid[j][i] = u
            d = d + 1 if grid[k][i] != 0 else 0
            if d < grid[k][i]:
                grid[k][i] = d
    res = 0
    for i in range(N):
        for j in range(N):
            if res < grid[i][j]: res = grid[i][j]
    return res

# LC74. Search a 2D Matrix - matrix binary search elem in matrix, matrix bs, search 2d search matrix
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:  # O(log(mn))
    if not matrix: return False
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1 # binary search
    while left <= right:
        pivot_idx = (left + right) // 2
        pivot_element = matrix[pivot_idx // n][pivot_idx % n]
        if target == pivot_element: return True
        elif target < pivot_element: right = pivot_idx - 1
        else: left = pivot_idx + 1
    return False

# LC1074. Number of Submatrices That Sum to Target - area sum to target
def numSubmatrixSumTarget(self, A, target):
    m, n = len(A), len(A[0])
    for row in A:
        for i in range(n - 1):
            row[i + 1] += row[i]  # sum up each row
    res = 0
    for i in range(n):  # loop 2 columns
        for j in range(i, n):  # O(mnn) runtime and O(m) space
            c = collections.defaultdict(int)
            cur, c[0] = 0, 1
            for k in range(m):  # 560. Subarray Sum Equals K, 1D case
                cur += A[k][j] - (A[k][i - 1] if i > 0 else 0)
                res += c[cur - target]
                c[cur] += 1
    return res



# LC542. 01 Matrix - distance to 0
def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:  # O(rc)
    m, n = len(mat), len(mat[0])

    for r in range(m):
        for c in range(n):
            if mat[r][c] > 0:
                top = mat[r - 1][c] if r > 0 else math.inf
                left = mat[r][c - 1] if c > 0 else math.inf
                mat[r][c] = min(top, left) + 1

    for r in range(m - 1, -1, -1):
        for c in range(n - 1, -1, -1):
            if mat[r][c] > 0:
                bottom = mat[r + 1][c] if r < m - 1 else math.inf
                right = mat[r][c + 1] if c < n - 1 else math.inf
                mat[r][c] = min(mat[r][c], bottom + 1, right + 1)

    return mat

# LC73. Set Matrix Zeroes
def setZeroes(self, matrix):
    m, n = len(matrix), len(matrix[0])
    firstRowHasZero = not all(matrix[0])  # First row has zero?
    for i in range(1, m):  # Use first row/column as marker, scan the matrix
        for j in range(n):
            if matrix[i][j] == 0: matrix[0][j] = matrix[i][0] = 0
    for i in range(1, m): # Set the zeros
        for j in range(n - 1, -1, -1):
            if matrix[i][0] == 0 or matrix[0][j] == 0: matrix[i][j] = 0
    # Set the zeros for the first row
    if firstRowHasZero: matrix[0] = [0] * n

# LC661. Image Smoother
def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:  # O(mn)
    m, n = len(img), len(img[0])
    res = copy.deepcopy(img)
    for x in range(m):
        for y in range(n):
            neighbors = [img[_x][_y] for _x in (x-1, x, x+1) for _y in (y-1, y, y+1)
                                     if 0 <= _x < m and 0 <= _y < n]
            res[x][y] = sum(neighbors) // len(neighbors)
    return res



# LC1329. Sort the Matrix Diagonally
def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
    n, m = len(mat), len(mat[0])
    d = collections.defaultdict(list)
    for i in range(n):
        for j in range(m): d[i - j].append(mat[i][j])
    for k in d: d[k].sort(reverse=1)
    for i in range(n):
        for j in range(m): mat[i][j] = d[i - j].pop()
    return mat

# LC1901. Find a Peak Element II
def findPeakGrid(self, mat: List[List[int]]) -> List[int]:  # O(mlogn)
    top, bottom = 0, len(mat)-1
    while bottom > top:  # find row max
        mid = (top + bottom) // 2
        if max(mat[mid]) > max(mat[mid+1]): bottom = mid
        else: top = mid+1
    return [bottom, mat[bottom].index(max(mat[bottom]))]
def findPeakGrid(self, mat: List[List[int]]) -> List[int]:  # O(m + n)
    m, n = len(mat), len(mat[0])
    def quad_search(s0, e0, s1, e1):
        m0, m1 = (s0 + e0) // 2, (s1 + e1) // 2
        i, j = m0, m1
        for jj in range(s1, e1):  # find max along middle lines
            if mat[m0][jj] > mat[m0][j]: j = jj
        for ii in range(s0, e0):
            if mat[ii][m1] > mat[i][j]: i, j = ii, m1
        cur = mat[i][j]  # compare with 4 sides
        up = mat[i-1][j] if i > 0 else -1
        down = mat[i+1][j] if i < m - 1 else -1
        left = mat[i][j-1] if j > 0 else -1
        right = mat[i][j+1] if j < n - 1 else - 1
        if cur > up and cur > down and cur > left and cur > right:
            return i, j
        if i < m0 or (i == m0 and cur < up): e0 = m0  # move interval boundaries
        else: s0 = m0 + 1
        if j < m1 or (j == m1 and cur < left): e1 = m1
        else: s1 = m1 + 1
        return quad_search(s0, e0, s1, e1)  # drill down
    return quad_search(0, m, 0, n)



# LC221. Maximal Square  max square
def maximalSquare(self, matrix: List[List[str]]) -> int: # DP
    if not matrix: return 0
    rows, cols = len(matrix), len(matrix[0])
    # DP(i, j) is the largest side of all squares ended at (i, j)
    dp = collections.defaultdict(int)  # O(mn)
    max_len = 0 # track this
    for i, j in itertools.product(range(rows), range(cols)):
        if matrix[i][j] == '1':
            dp[i+1, j+1] = min([dp[i+1, j], dp[i, j+1], dp[i, j]]) + 1  # weakest link
            max_len = max(max_len, dp[i+1, j+1])
    return max_len ** 2
# https://leetcode.com/problems/maximal-square/?envType=company&envId=apple&favoriteSlug=apple-six-months

# LC240. Search a 2D Matrix II - zigzag search 2d matrix
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix: return False
    h, w = len(matrix), len(matrix[0])
    row, col = h - 1, 0
    while row >= 0 and col < w:
        if target == matrix[row][col]: return True
        elif target < matrix[row][col]:  row -= 1
        else: col += 1
    return False

# LC1351. Count Negative Numbers in a Sorted Matrix - zigzag
def countNegatives(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])  # O(m + n)
    r, c, cnt = m - 1, 0, 0
    while r >= 0 and c < n:
        if grid[r][c] < 0:
            cnt += n - c
            r -= 1
        else: c += 1
    return cnt

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

# LC54. Spiral Matrix, top100 - return elems in spiral
def spiralOrder(self, matrix):
    res = []
    while matrix:
        res.extend(matrix.pop(0))
        # zip rows to columns, flattern each column, reverse order
        matrix = [*zip(*matrix)][::-1]
    return res
# [[1,2,3],[4,5,6],[7,8,9]] ->  [(6, 9), (5, 8), (4, 7)] ->  [(8, 7), (5, 4)] -> [(4,), (5,)] -> [(5,)]
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

# LC59. Spiral Matrix II  generate
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
