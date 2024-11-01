
# LC1105. Filling Bookcase Shelves  bookshelf
def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
    @cache
    def place(book_idx, currw, currh):  # return total height
        if book_idx == len(books): return 0
        bookw, bookh = books[book_idx]
        ans = bookh + place(book_idx + 1, bookw, bookh)  # new shelf
        if currw + bookw <= shelfWidth:   # same shelf
            ans1 = place(book_idx + 1, currw + bookw, max(bookh, currh))
            delta = max(0, bookh - currh)
            ans = min(ans, delta + ans1)
        return ans
    return place(0, 0, 0)  # height after placing ith book to nth book
def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
    n = len(books)  # O(n*shelfWidth) time, O(n) space
    f = [0] * (n + 1)  # f(i) shelf height up to book i
    for i, (w, h) in enumerate(books, 1):
        f[i] = f[i - 1] + h  # new row
        for j in range(i - 1, 0, -1):
            w += books[j - 1][0]  # minus 1 because i shift up by 1 in for loop
            if w > shelfWidth: break
            h = max(h, books[j - 1][1])
            f[i] = min(f[i], f[j - 1] + h)
    return f[n]
    # https://leetcode.com/problems/filling-bookcase-shelves/solutions/5561109/pure-dp-code-98-margin-no-memoization-followed/?envType=company&envId=amazon&favoriteSlug=amazon-three-months


# LC1947. Maximum Compatibility Score Sum  max compat score sum student compat
def maxCompatibilitySum(self, students: List[List[int]], mentors: List[List[int]]) -> int:
    m, n = len(students), len(students[0])  # O(m^2 * 2^m * n) time, O(m * 2^m) space
    @cache
    def dp(student: int, mentor: int) -> int:  # max score sum
        if student == m: return 0  # all students assigned,
        res = 0
        for i in range(m):  # iterate all mentors
            if mentor & (1 << i): continue  # mentor already assigned.
            score = sum(1 for j in range(n) if students[student][j] == mentors[i][j])
            res = max(res, score + dp(student + 1, mentor | (1 << i)))
        return res
    return dp(0, 0)
# https://leetcode.com/problems/maximum-compatibility-score-sum/solutions/5776782/python3-bitmask-dp-detailed-explanation/?envType=company&envId=facebook&favoriteSlug=facebook-three-months

# LC750. Number Of Corner Rectangles
def countCornerRectangles(self, grid: List[List[int]]) -> int:
    def count(row1, row2):  # one number, one mask, how many common 1's
        num_ones = (row1 & row2).bit_count()
        return num_ones * (num_ones - 1) // 2  # combinations of 2 1's(columns)
    rows = [int(''.join(map(str, row)), 2) for row in grid]
    return sum(count(row1, rows[j]) for i, row1 in enumerate(rows) for j in range(i + 1, len(rows)))

# LC3030. Find the Grid of Region Average  image avg
def resultGrid(self, image: List[List[int]], threshold: int) -> List[List[int]]:
    m, n = len(image), len(image[0])  # O(n^2) time and space
    grid = [[[0, 0] for _ in range(n)] for _ in range(m)]  # sum & count, for average later
    for i, j in product(range(m-2), range(n-2)):  # O(mn)
        s = 0
        for ii, jj in product(range(i, i+3), range(j, j+3)):
            if (ii < i+2 and abs(image[ii+1][jj] - image[ii][jj]) > threshold or
                    jj < j+2 and abs(image[ii][jj+1] - image[ii][jj]) > threshold):
                break  # not a region
            s += image[ii][jj]
        else:  # without break
            s //= 9  # region 3 X 3 average
            for ii, jj in product(range(i, i+3), range(j, j+3)):
                grid[ii][jj][0] += s
                grid[ii][jj][1] += 1

    for i, j in product(range(m), range(n)):
        if grid[i][j][1]:
            image[i][j] = grid[i][j][0] // grid[i][j][1]
    return image

# LC2397. Maximum Rows Covered by Columns  max rows  max row covered  cover column
def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:  # O(m) space, O(2^n * m) time
    n,m = len(matrix),len(matrix[0])
    ans = 0
    def check(state,row,rowIncludedCount):
        if sum(state) > numSelect: return
        if row==n:
            nonlocal ans
            ans = max(ans, rowIncludedCount)
            return
        check(state[:], row+1, rowIncludedCount)  # skip this row
        for j in range(m):
            if matrix[row][j]==1: state[j] = 1
        check(state, row+1, rowIncludedCount+1)  # count in this row
    check([0]*m, 0, 0)
    return ans

# LC373. Find K Pairs with Smallest Sums - similar to 378 k-pair
def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    min_heap, res = [], []  # O(klogk)
    for i in range(min(len(nums1), k)): heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))
    while min_heap and len(res) < k:
        _, i, j = heapq.heappop(min_heap)
        res.append([nums1[i], nums2[j]])
        # all rows first elem are in heap, j+1 results shifting to right
        if j + 1 < len(nums2): heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))
    return res
# https://leetcode.com/problems/find-k-pairs-with-smallest-sums/discuss/84577/O(k)-solution

# LC1868. Product of Two Run-Length Encoded Arrays  run length
def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
    res, l, r = [], 0, 0   # O(n + m), counts of unique numbers
    while encoded1[-1][-1] != 0:
        prod = encoded1[l][0] * encoded2[r][0]
        low = min(encoded1[l][1], encoded2[r][1])
        if res and res[-1][0] == prod: res[-1][1] += low  # extend freq if same value
        else: res.append([prod, low])
        encoded1[l][1] -= low  # minus the finished range
        encoded2[r][1] -= low
        if encoded1[l][1] == 0: l += 1
        if encoded2[r][1] == 0: r += 1
    return res

# LC815. Bus Routes
def numBusesToDestination(self, routes, S, T): # BFS on stops
    stop2routes = collections.defaultdict(set)
    for i, route in enumerate(routes):
        for j in route: stop2routes[j].add(i)  # label stop with bus route #
    queue, seen = [(S, 0)], {S}
    for stop, bus in queue:  # BFS for min stops
        if stop == T: return bus
        for route in stop2routes[stop]:  # for each bus in this stop
            for st in routes[route]:  # for each stop with this bus
                if st not in seen:
                    queue.append((st, bus + 1))
                    seen.add(st)
            routes[route] = []  # ***neat trick: seen route, cut buses for min
    return -1

# LC1743. Restore the Array From Adjacent Pairs
def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
    adj, ans, n = defaultdict(list), [], len(adjacentPairs) + 1
    for a, b in adjacentPairs:
        adj[a] += [b]
        adj[b] += [a]
    prev = -math.inf
    for k, v in adj.items():
        if len(v) == 1:
            ans += [k]
            break
    while len(ans) < n:
        for next in adj.pop(ans[-1]):
            if next != prev:
                prev = ans[-1]
                ans += [next]
                break
    return ans

# LC983. Minimum Cost For Tickets - ticket cost min cost min ticket
def mincostTickets(self, days, costs):   # O(len(days) * len(costs))
    durations = [1, 7, 30]  # 2, 7, 5
    N = len(days)
    @lru_cache(None)
    def dp(i): # How much money to do days[i]+
        if i >= N: return 0
        ans, j = float('inf'), i
        for c, d in zip(costs, durations):
            while j < N and days[j] < days[i] + d: j += 1
            ans = min(ans, dp(j) + c)
        return ans
    return dp(0)
def mincostTickets(self, days: List[int], costs: List[int]) -> int:  # O(max(durs))
    durations = [1, 7, 30]  # this should be in parameter passed in
    day_set = set(days)
    @lru_cache(None)
    def dp(i):  # BFS on days
        if i > days[-1]: return 0
        elif i in day_set: # if we do travel today
            return min(dp(i + d) + c for c, d in zip(costs, durations))
        else: return dp(i + 1) # wait for next day if we don't travel today
    # print(dp.cache_info())
    return dp(days[0])

# LC406. Queue Reconstruction by Height - people height
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

# LC2092. Find All People With Secret - x and y meet at time t
def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
    sm = sorted(meetings, key=lambda x: x[2])  # O(mlogm), m = # of meetings.
    can = {0, firstPerson}
    for _, grp in itertools.groupby(sm, key=lambda x: x[2]):  # O(m)
        queue, graph = set(), defaultdict(list)
        for x, y, _ in grp:  # connect all people in the group
            graph[x].append(y)
            graph[y].append(x)
            if x in can: queue.add(x)
            if y in can: queue.add(y)
        queue = deque(queue)
        while queue:
            x = queue.popleft()
            for y in graph[x]:
                if y not in can:
                    can.add(y)
                    queue.append(y)  # This ensures no order dependency
    return can

# LC946. Validate Stack Sequences
def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
    stack, i = [], 0
    for x in pushed:
        stack.append(x)
        while stack and stack[-1] == popped[i]:
            i += 1
            stack.pop()
    return len(stack) == 0

# LC1168. Optimize Water Distribution in a Village
def minCostToSupplyWater(self, n: int, wells, pipes) -> int:  # O((N+M) * log(N+M)), because of heap
    # build graph, Prim's algo for minimum spanning tree.  O(N+M) space
    graph = collections.defaultdict(list)
    for u, v, w in pipes:
        graph[u].append([w, u, v])
        graph[v].append([w, v, u])
    for i in range(n):
        graph[0].append([wells[i], 0, i+1])  # No need to point back to 0 since we start from 0.
    edges = graph[0]
    heapq.heapify(edges)
    res, visited = 0, {0}
    while len(visited) < n+1 and edges:
        w, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            res += w
            for edge in graph[v]:
                if edge[2] not in visited: heapq.heappush(edges, edge)
    return res

# LC1834. Single-Threaded CPU - single cpu
def getOrder(self, tasks: List[List[int]]) -> List[int]:  # O(nlogn)
    tasks = sorted([(t[0], t[1], i) for i, t in enumerate(tasks)])
    res, h = [], []  # heap
    i, n, time = 0, len(tasks), tasks[0][0]
    while len(res) < n:
        while i < n and tasks[i][0] <= time:  # available task for now
            heapq.heappush(h, (tasks[i][1], tasks[i][2])) # (processing_time, original_index)
            i += 1
        if h:
            t_diff, original_index = heapq.heappop(h)
            time += t_diff
            res.append(original_index)
        elif i < n: time = tasks[i][0]
    return res



# LC1376. Time Needed to Inform All Employees
def numOfMinutes(self, n, headID, manager, informTime):
    reports = defaultdict(list)
    for i, m in enumerate(manager):
        if m >= 0: reports[m].append(i)

    def dfs(i):
        return max([dfs(j) for j in reports[i]] or [0]) + informTime[i]
    return dfs(headID)

# LC1640. Check Array Formation Through Concatenation - chain arrays, concat array
def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:  # nlogn
    n, p_len = len(arr), len(pieces)
    pieces.sort()  # O(nlogn)
    i = 0
    while i < n:
        left, right = 0, p_len - 1
        found = -1
        while left <= right:  # use binary search to find target piece:
            mid = (left + right)//2
            if pieces[mid][0] == arr[i]:
                found = mid
                break
            elif pieces[mid][0] > arr[i]: right = mid - 1
            else: left = mid + 1
        if found == -1: return False
        # check target piece
        target_piece = pieces[found]
        for x in target_piece:
            if x != arr[i]: return False
            i += 1
    return True

# LC223. Rectangle Area - intersection of 2 rectangles  rect area
def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
    overlap = max(min(ax2, bx2) - max(ax1, bx1), 0) * max(min(ay2, by2) - max(ay1, by1), 0)
    return (ax1 - ax2) * (ay1 - ay2) + (bx1 - bx2) * (by1 - by2) - overlap
def computeArea(self, A, B, C, D, E, F, G, H):  # AB lower left, CD upper right
    # overlap lower left is [max(A, E), max(B, F)], upper right is [min(C, G), min(D, H)]
    overlap = max(min(C,G) - max(A,E), 0) * max(min(D,H) - max(B,F), 0)
    return (A-C)*(B-D) + (E-G)*(F-H) - overlap

# LC836. Rectangle Overlap - intersection of 2 rectangles
def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
    A, B, C, D = rec1[0], rec1[1], rec1[2], rec1[3]
    E, F, G, H = rec2[0], rec2[1], rec2[2], rec2[3]
    x1 = max(A, E)
    y1 = max(B, F)
    x2 = min(C, G)
    y2 = min(D, H)
    return x1 < x2 and y1 < y2

# LC853. Car Fleet
def carFleet(self, target: int, pos: List[int], speed: List[int]) -> int:  # O(nlogn)
    time = [float(target - p) / s for p, s in sorted(zip(pos, speed))]
    res = slowest = 0  # fleets are counted by slowest cars
    for t in time[::-1]:
        if t > slowest:
            res += 1
            slowest = t  # in the same group, diff from prev slowest group
    return res
# https://leetcode.com/problems/car-fleet/solutions/139850/c-java-python-straight-forward

























