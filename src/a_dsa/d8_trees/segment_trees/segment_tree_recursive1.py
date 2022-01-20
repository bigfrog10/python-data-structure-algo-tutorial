# https://dev.to/curingartur/segment-tree-3hpe

class SegmentTree:
    def __init__(self, nums: list, agg_f):
        if not nums:
            raise Exception('Empty data')

        self.n = len(nums)
        self.nums = nums
        self.agg_f = agg_f
        self.arr = [0] * (4 * self.n)
        self.build(1, 0, self.n - 1)

    def build(self, v, tl, tr ):
        if tl == tr:
            self.arr[v] = self.nums[tl]
        else:
            tm = (tl + tr) // 2
            self.build(v*2, tl, tm)
            self.build(v*2+1, tm+1, tr)
            self.arr[v] = self.agg_f(self.arr[v*2], self.arr[v*2+1])

    def _update(self, v, tl, tr, pos, val):
        if tl == tr:
            self.arr[v] = val
        else:
            mid = (tl + tr) // 2
            if pos <= mid:
                self._update(v * 2, tl, mid, pos, val)
            else:
                self._update(v * 2 + 1, mid + 1, tr, pos, val)

            self.arr[v] = self.arr[v * 2] + self.arr[v * 2 + 1]

    def update(self, i: int, val: int) -> None:
        self._update(1, 0, self.n -1, i, val)

    # This is related to LC 732
    def _sum(self, v, tl, tr, l, r):
        if (l > r):
            return 0
        if l == tl and r == tr:
            return self.arr[v]

        tm = (tl + tr) // 2
        return self._sum(v*2, tl, tm, l, min(r, tm)) + self._sum(v*2+1, tm+1, tr, max(l, tm+1), r)

    def sumRange(self, i: int, j: int) -> int:
        return self._sum(1, 0, self.n - 1,   i, j)
