# LC303. Range Sum Query - Immutable
from itertools import accumulate
class NumArray:
    def __init__(self, nums: List[int]):
        self.accumu = [0] + list(accumulate(nums))
        print(self.accumu)
    def sumRange(self, left: int, right: int) -> int:
        return self.accumu[right+1] - self.accumu[left]

# LC304. Range Sum Query 2D - Immutable
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        n, m = len(matrix), len(matrix[0])
        self.sums = [ [0 for j in range(m+1)] for i in range(n+1) ]
        for i in range(1, n+1):
            for j in range(1, m+1):
                self.sums[i][j] = matrix[i-1][j-1] + self.sums[i][j-1] + \
                                  self.sums[i-1][j] - self.sums[i-1][j-1]
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        row2, col2 = row2+1, col2+1
        return self.sums[row2][col2] + self.sums[row1][col1] \
             - self.sums[row2][col1] - self.sums[row1][col2]

# LC307. Range Sum Query - Mutable
class NumArray:
    # https://leetcode.com/problems/range-sum-query-mutable/discuss/75753/Java-using-Binary-Indexed-Tree-with-clear-explanation
    # https://leetcode.com/problems/range-sum-query-mutable/discuss/954063/Python-144ms-80-Fenwick-Tree
    def __init__(self, nums: List[int]):
        self._nums = [0] * len(nums)
        self._ftree = [0] * (len(nums) + 1)  # Fenwich tree start with index 1
        for i in range(len(nums)):
            self.update(i, nums[i])
    def update(self, i: int, val: int) -> None:  # O(logn)
        if i >= len(self._nums) or i < 0: return
        delta = val - self._nums[i]  # get the delta
        self._nums[i] = val
        i += 1  # Fenwich tree start with index 1
        while i <= len(self._nums):
            self._ftree[i] += delta
            # go to a larger range, which contains current range
            i += i & (-i)  # i = i + LowestSignificentBit(i)
    def sumRange(self, i: int, j: int) -> int:
        return self._getSum(j) - self._getSum(i-1)
    def _getSum(self, i:int) -> int:  # log(n)
        if i < 0: return 0
        i += 1 # Fenwich tree start with index 1
        ans = 0
        while i > 0:
            ans += self._ftree[i]
            # go to a new range, which is immediately before current range
            i -= i & (-i)  # i = i - LowestSignificentBit(i)
        return ans

# LC308. Range Sum Query 2D - Mutable
class NumMatrix:
    # https://cs.stackexchange.com/questions/10538/bit-what-is-the-intuition-behind-a-binary-indexed-tree-and-how-was-it-thought-a
    def __init__(self, matrix: List[List[int]]):
        if not matrix: return
        if len(matrix[0]) == 0: return
        self.row_num = len(matrix) + 1
        self.col_num = len(matrix[0]) + 1
        self.bit = [ [0] * self.col_num for _ in range(self.row_num) ]
        for i in range(1, self.row_num):
            for j in range(1, self.col_num): self._bit_update(i, j, matrix[i-1][j-1])
    def _bit_update(self, r, c, val):
        while r < self.row_num:
            j = c
            row = self.bit[r]
            while j < self.col_num:
                row[j] += val
                j += j & (-j)
            r += r & (-r)
    def _bit_query(self, r, c):
        sumv = 0
        while r > 0:
            j = c
            while j > 0:
                sumv += self.bit[r][j]
                j -= j & (-j)
            r -= r & (-r)
        return sumv
    def update(self, row: int, col: int, val: int) -> None:
        old_v = self.sumRegion(row, col, row, col)
        diff = val - old_v
        self._bit_update(row+1, col+1, diff)
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        sum1 = self._bit_query(row2+1, col2+1) + self._bit_query(row1, col1)
        sum2 = self._bit_query(row2+1, col1) + self._bit_query(row1, col2+1)
        return sum1 - sum2


