

# LC733. Flood Fill image coloring
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    if not image: return []  # DFS fill this island, O(pixels) time and space(recursion)
    c = image[sr][sc]
    if c == color: return image
    n, m = len(image), len(image[0])
    def dfs(i, j):
        if image[i][j] == c:
            image[i][j] = color
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = i + dx, j + dy
                if 0 <= x < n and 0 <= y < m:
                    dfs(x, y)
    dfs(sr, sc)
    return image

# See maze images
# The maze is represented as an n * m grid of cells, where each cell is either
# empty(denoted by 0), or contains an obstacle(denoted by 1). HackerMan is
# currently standing at cell (0, 0) and wishes to reach the cell (n-1, m-1).
# For a jump parameter denoted by k, in one move, HackerMan can move to any of
# the following cells:
# 1. (i+x, j) where 1 <= x <= k, provided cell (i+x, j) lies in the maze and
#    there are no cells containing obstacles in the range (i, j) -> (i+x, j).
# 2. (i, j+x) where 1 <= x <= k, provided cell (i, j+x) lies in the maze and
#    there are no cells containing obstacles in the range (i, j) -> (i, j+x).
# 3. (i-x, j) where 1 <= x <= k, provided cell (i+x, j) lies in the maze and
#    there are no cells containing obstacles in the range (i, j) -> (i-x, j).
# 4. (i, j-x) where 1 <= x <= k, provided cell (i, j-x) lies in the maze and
#    there are no cells containing obstacles in the range (i, j) -> (i, j-x).
# Find the minimum number of moves in which HackerMan can reach the cell
# (n-1, m-1) starting from (0, 0), or -1 if it is impossible to reach that
# cell.
def walk(maze, k: int) -> int:
    if not maze: return -1
    n, m = len(maze), len(maze[0])
    queue = deque([0, 0, 0])
    visited = set()
    while queue:
        i, j, steps = queue.popleft()
        if i == n-1 and j == m-1: return steps
        if (i, j) in visited: continue
        visited.add((i, j))
        up = down = left = right = True  # can walk without obstacles
        for x in range(1, k+1):
            down = down & (maze[i+x][j] == 0)
            if i+x < n and down and (i+x, j) not in visited:
                queue.append((i+x, j, steps+1))
            # do it for up, left and right)
    return -1
# run time O(mnk), space O(mn)

