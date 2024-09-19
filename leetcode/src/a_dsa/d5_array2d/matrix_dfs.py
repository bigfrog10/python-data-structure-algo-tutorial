

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