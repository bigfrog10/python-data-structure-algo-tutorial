from typing import List
import itertools
# LC1428. Leftmost Column with at Least a One
def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:  # O(n + m), diagonal
    rows, cols = binaryMatrix.dimensions()
    row, col = 0, cols - 1  # upper right corner
    while row < rows and col >= 0:
        if binaryMatrix.get(row, col) == 0: row += 1  # move down
        else: col -= 1  # move left
    # If we never left the last column, it must have been all 0's.
    return col + 1 if col != cols - 1 else -1

# LC317. Shortest Distance from All Buildings
def shortestDistance(self, grid):
    if not grid or not grid[0]: return -1  # O(n^2 * m^2)
    n, m = len(grid), len(grid[0])
    # 0 for distance, 1 for counts/buildings
    matrix = [[[0, 0] for _ in range(m)] for _ in range(n)]
    def bfs(start, blds):
        q = [(start, 0)]  # 0 is the distance.
        while q:  # BFS for distance
            po, distance = q.pop(0)
            for dp in (-1, 0), (1, 0), (0, 1), (0, -1):
                i, j = po[0] + dp[0], po[1] + dp[1]
                if 0 <= i <n and 0 <= j < m and matrix[i][j][1] == blds:
                    if grid[i][j] == 0:
                        matrix[i][j][0] += distance + 1
                        matrix[i][j][1] = blds + 1  # reachable to all blds
                        q.append(([i, j], distance+1))
    blds = 0  # count how many building we have visited
    for i, j in product(range(n), range(m)):  # O(mn)
        if grid[i][j] == 1:  # loop through buildings
            bfs([i, j], blds)
            blds += 1
    res = float('inf')
    for i, j in product(range(len(matrix)), range(len(matrix[0]))):
        if matrix[i][j][1] == blds: res = min(res, matrix[i][j][0])
    return res if res != float('inf') else -1

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

# LC1424. Diagonal Traverse II
def findDiagonalOrder(self, A):
    res = defaultdict(list)
    for i, r in enumerate(A):
        for j, a in enumerate(r): res[i + j].append(a)
    return [a for _, r in res.items() for a in reversed(r)]

# LC766. Toeplitz Matrix
def isToeplitzMatrix(self, matrix):  # O(mn) runtime, O(1) space, has follow ups
    return all(r == 0 or c == 0 or matrix[r-1][c-1] == val
               for r, row in enumerate(matrix)
               for c, val in enumerate(row))

# LC286. Walls and Gates
def wallsAndGates(self, rooms: List[List[int]]) -> None:  # O(mn)
    WALL, GATE, EMPTY = -1, 0, 2147483647  # given
    q = [(i, j) for i, row in enumerate(rooms) for j, r in enumerate(row) if r == GATE]  # all gates
    for i, j in q:
        for I, J in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
            if 0 <= I < len(rooms) and 0 <= J < len(rooms[0]) and rooms[I][J] == EMPTY:
                rooms[I][J] = rooms[i][j] + 1
                q.append((I, J))

# LC939. Minimum Area Rectangle
def minAreaRect(self, points: List[List[int]]) -> int:  # O(n^2)
    res, seen = float('inf'), set()
    for x1, y1 in points:
        for x2, y2 in seen:
            if (x1, y2) in seen and (x2, y1) in seen:
                area = abs(x1 - x2) * abs(y1 - y2)
                if area: res = min(area, res)
        seen.add((x1, y1))
    return res if res < float('inf') else 0

# LC963. Minimum Area Rectangle II
def minAreaFreeRect(self, points: List[List[int]]) -> float:  # O(n^2)
        def distSquare(x1,y1,x2,y2): return (x1-x2)**2 + (y1-y2)**2
        def dist(x1,y1,x2,y2): return sqrt((x1-x2)**2 + (y1-y2)**2)
        def midPos(x1,y1,x2,y2): return ((x1+x2)/2,(y1+y2)/2)

        linesMap = defaultdict(list) # (len, mid of p1 and p2) => [(p1,p2)], grouping
        N = len(points)
        for i in range(N):
            for j in range(i + 1, N):
                l = distSquare(*points[i], *points[j])
                m = midPos(*points[i], *points[j])
                linesMap[(l, m)].append((i,j))
        minArea = float("inf")
        for lines in linesMap.values():
            if len(lines) < 2: continue
            M = len(lines)
            for i in range(M): # try all pairs of lines
                for j in range(i + 1, M):
                    p1, p2, p3 = points[lines[i][0]], points[lines[j][0]], points[lines[j][1]]
                    d1, d2 = dist(*p1, *p2), dist(*p1, *p3)
                    minArea = min(minArea, d1 * d2)
                    print(p1, points[lines[i][1]],  p2, p3, minArea)
        return minArea if minArea != float("inf") else 0

# LC1091. Shortest Path in Binary Matrix
def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:  # O(n) runtime and space
    if not grid or grid[0][0] != 0: return -1
    n, que, visited = len(grid), deque([(0, 0, 1)]), set()  # x, y, steps
    while que:  # BFS
        i, j, steps = que.popleft()
        if i == n-1 and j == n-1: return steps
        if (i, j) in visited: continue
        visited.add((i, j))
        for x, y in (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1):
            ni, nj = i + x, j + y
            if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 0 and (ni, nj) not in visited:
                que.append((ni, nj, steps+1))
    return -1

# LC1074. Number of Submatrices That Sum to Target
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

# LC778. Swim in Rising Water
def swimInWater(self, grid: List[List[int]]) -> int:  # O(N^2log(N^2)) time, O(N^2) time
    N = len(grid)
    pq, seen = [(grid[0][0], 0, 0)], {(0, 0)}
    ans = 0
    while pq: # DFS, O(N^2)
        d, r, c = heapq.heappop(pq)
        ans = max(ans, d)
        if r == c == N-1: return ans
        for cr, cc in (r-1, c), (r+1, c), (r, c-1), (r, c+1):
            if N > cr >= 0 <= cc < N and (cr, cc) not in seen:
                heapq.heappush(pq, (grid[cr][cc], cr, cc))  # log
                seen.add((cr, cc))

# LC406. Queue Reconstruction by Height
def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:  # O(nlogn)
    from sortedcontainers import SortedList
    people.sort(key=lambda x: (x[0], -x[1]))
    ans = [None] * len(people)
    sl = SortedList(list(range(len(people))))
    for p in people:
        idx = sl[p[1]]
        ans[idx] = p
        sl.remove(idx)  # logn
    return ans
def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:  # O(n^2)
    res = []
    for p in sorted((-x[0], x[1]) for x in people): # from largest to smallest
        res.insert(p[1], [-p[0], p[1]]) # insert only relevant to larger values
    return res

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

# LC2033. Minimum Operations to Make a Uni-Value Grid
def minOperations(self, grid: List[List[int]], x: int) -> int:
    # https://asvrada.github.io/blog/median-shortest-distance-sum/
    vals = list(itertools.chain(*grid))  # flatting matrix to array
    if len(set(val % x for val in vals)) > 1: return -1  # 1 is for min element
    median = sorted(vals)[len(vals) // 2]  # O(N) possible via "quick select"
    return sum(abs(val - median)//x for val in vals)

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
    for i in range(1, m):  # Use first row/column as marker, scan the matrix
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

# LC48. Rotate Image
def rotate(self, A):
    A[:] = zip(*A[::-1])

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

# LC417. Pacific Atlantic Water Flow
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:  # O(nm)
    if not matrix: return []
    rows, cols = len(matrix), len(matrix[0])
    p_visited, a_visited = set(), set()
    def traverse(i, j, visited):  # goes up
        if (i, j) in visited: return
        visited.add((i, j))
        for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0): # Traverse neighbors.
            ni, nj = i + dx, j + dy
            if rows > ni >= 0 <= nj < cols and matrix[ni][nj] >= matrix[i][j]:
                traverse(ni, nj, visited)
    for row in range(rows):  # along border, go inland
        traverse(row, 0, p_visited)
        traverse(row, cols - 1, a_visited)
    for col in range(cols):
        traverse(0, col, p_visited)
        traverse(rows - 1, col, a_visited)
    return list(p_visited & a_visited)

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
def maximalSquare(self, matrix: List[List[str]]) -> int: # DP
    if not matrix: return 0
    rows, cols = len(matrix), len(matrix[0])
    # DP(i, j) is the largest side of all squares ended at (i, j)
    dp = collections.defaultdict(int)
    max_len = 0 # track this
    for i, j in itertools.product(range(rows), range(cols)):
        if matrix[i][j] == '1':
            dp[i+1, j+1] = min([dp[i+1, j], dp[i, j+1], dp[i, j]]) + 1
            max_len = max(max_len, dp[i+1, j+1])
    return max_len ** 2

# LC378. Kth Smallest Element in a Sorted Matrix
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    n = len(matrix)  # O(nlognlog(max-min))
    l, r = matrix[0][0], matrix[n - 1][n - 1]
    while l < r:  # log(max-min)
        mid = (l+r) // 2
        count = sum(bisect.bisect(row, mid) for row in matrix)
        if count < k: l = mid+1
        else: r = mid
    return l

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

# LC1351. Count Negative Numbers in a Sorted Matrix
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
