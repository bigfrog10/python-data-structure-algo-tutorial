# copied from https://www.geeksforgeeks.org/segment-tree-efficient-implementation/,
# which is from here: https://codeforces.com/blog/entry/18051
# This is a non-recursion/iterative implementation
# The array is the level traversal of the tree with leading None

class SegmentTree:
    def __init__(self, array: list, agg_f):
        self._size = len(array)
        self._agg_f = agg_f  # used by parent to aggregate left/right children
        self._build(array)

    def _build(self, array):  # O(n)
        n = self._size
        self._tree = [None] * (2 * n)
        for i in range(n):  # originals stored in the 2nd half
            self._tree[n+i] = array[i]  # O(n)

        for i in range(n-1, 0, -1):  # parents, O(n), i -> 2i and 2i+1
            self._tree[i] = self._agg_f(self._tree[i << 1], self._tree[i << 1 | 1])

    def update(self, idx, value):  # O(logN) because of last line
        idx = self._size + idx
        self._tree[idx] = value  # update array itself

        i = idx
        while i > 1:  # update all parents, i ^ 1 is i's neighbour(left or right depends on even or odd
            self._tree[i >> 1] = self._agg_f(self._tree[i], self._tree[i ^ 1])
            i >>= 1

    def query(self, left, right):
        left, right = left + self._size, right + self._size
        res = None
        while left < right:  # O(logN) because of last 2 lines
            if left & 1:
                res = self._agg_f(res, self._tree[left])
                left += 1

            if right & 1:
                right -= 1
                res = self._agg_f(res, self._tree[right])

            left >>= 1
            right >>= 1

        return res


arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]

st = SegmentTree(arr, lambda x, y: x + y if x else y)
print(st._tree)  # [[None, 45, 23, 22, 18, 5, 9, 13, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(st.query(3, 6))  # 15 = 4 + 5 + 6

st.update(4, 10)  # change 5 to 10 -> so diff is 5
print(st._tree)   # [None, 50, 23, 27, 18, 5, 14, 13, 17, 1, 2, 3, 4, 10, 6, 7, 8, 9]
print(st.query(3, 6))  # so 15 is changed to 20

print(st.query(0, 3))  # 6
