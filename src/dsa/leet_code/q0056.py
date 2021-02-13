from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals or len(intervals) == 1:
            return intervals

        intervals.sort(key=lambda x: x[0])
        ret = []
        a, b = 0, 0  # 2 pointers
        for i in range(1, len(intervals)):
            if intervals[b][1] >= intervals[i][0]:  # first end > 2nd start, merge
                if intervals[b][1] <= intervals[i][1]:
                    b = i
            else:
                ret.append([intervals[a][0], max(intervals[a][1], intervals[b][1])])
                a = b = i  # start a new search

        # last interval
        ret.append([intervals[a][0], max(intervals[a][1], intervals[b][1])])
        return ret


print(Solution().merge([[1,3],[2,6],[8,10],[15,18]]))
print(Solution().merge([[1,4],[4,5]]))
print(Solution().merge([[1,4],[2,3]]))
print(Solution().merge([[2,3],[4,5],[6,7],[8,9],[1,10]]))
