
# LC1162. As Far from Land as Possible
def maxDistance(self, grid: List[List[int]]) -> int:  # O(n^2) time space
    n=len(grid)  # # iterate the array twice for the grids
    for i in range(0,n):  # should copy the grid
        for j in range(0,n):  # Just BFS
            if(grid[i][j] == 1): continue  # skip all the lands
            top_dist = 2*n if i==0 else grid[i-1][j]+1
            left_dist = 2*n if j==0 else grid[i][j-1]+1
            grid[i][j] = min(top_dist, left_dist)
    for i in range(n-1,-1,-1):
        for j in range(n-1,-1,-1):
            if(grid[i][j]==1): continue  # skip all the lands
            bottom_dist = 2*n if i == n-1 else grid[i+1][j]+1
            right_dist = 2*n if j==n-1 else grid[i][j+1]+1
            grid[i][j] = min(grid[i][j], bottom_dist, right_dist)
    result = max(max(row) for row in grid)
    # land 1 + another 1 = 2, but we are just 1 step away from land, so -1
    return result-1 if result-1 != 2*n and result-1 != 0 else -1
def maxDistance(self, grid: List[List[int]]) -> int:
    ROWS, COLS = len(grid), len(grid[0])  # standard bfs
    queue = deque()
    for r,c in product(range(ROWS), range(COLS)):
        if grid[r][c]: queue.append((r,c))
    if not queue or ROWS * COLS == len(queue): # no land or all land
        return -1
    visited = set()
    level = 0
    while queue: # count number of BFS layers
        for _ in range(len(queue)):
            r,c = queue.popleft()
            for dr,dc in (0,1),(1,0),(0,-1),(-1,0):
                nr,nc = r+dr,c+dc
                if ROWS > nr >= 0 <= nc < COLS and not grid[nr][nc] and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        level += 1
    return level - 1