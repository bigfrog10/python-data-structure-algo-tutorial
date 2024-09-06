
# LC1266. Minimum Time Visiting All Points
def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
    n, res = len(points), 0
    for i in range(n-1):
        dx = abs(points[i+1][0]-points[i][0])
        dy = abs(points[i+1][1]-points[i][1])
        res += max(dx,dy)
    return res

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
