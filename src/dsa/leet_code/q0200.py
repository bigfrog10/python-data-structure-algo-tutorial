WATER = '0'
LAND = '1'


class Solution:
    def numIslands(self, grid: list) -> int:
        def n1(i: int, j: int, res: int, g: list):
            if len(g) == 0:
                return 0

            rs = len(g)  # number of rows
            cs = len(g[0])  # number of columns

            if g[i][j] == 1:
                if i+1 < rs and j+1 < cs and g[i+1][j] == 0 and g[i][j+1] == 0:
                    res += 1  # it's an isolated island

            if i == rs - 1 and j == cs - 1:  # last cell
                res += g[i][j]
            elif i == rs - 1:
                # use 2 pointers
                a, b = j, j
                while a < rs:
                    while g[i][a] == WATER:
                        a += 1
                    if a == rs:  # no 1's found
                        return res

                    b = a
                    while g[i][b] == LAND:
                        b += 1
                    # now we have all 1's between a and b
                    if b == cs:  # all 1's
                        res += 1
                        return res

                    a = b  # a point 0 again
            elif j == cs - 1:  # same logic above
                a, b = i, i
                while a < cs:
                    while g[a][j] == WATER:
                        a += 1
                    if a == cs:  # no 1's found
                        return res

                    b = a
                    while g[b][j] == LAND:
                        b += 1
                    if b == rs:
                        res += 1
                        return res

                    a = b  # a point 0 again
            else:
                # now we scan 2 rows, and then 2 columns on boundaries
                a, b = i+1, i+1  # scan column
                while a < rs:
                    while a < rs and g[a][j] == WATER:
                        a += 1
                    # a points to 1
                    b = a
                    while b < rs and g[b][j] == LAND:
                        b += 1
                    # b points to 0, we have 1's between a, b
                    extended = False
                    for k in range(a, b):
                        if g[k][j+1] == LAND:
                            extended = True
                            break  # island extends to right, no action
                    if not extended and b > a:
                        res += 1

                    a = b  # a point 0 again

                a, b = j+1, j+1  # scan rows
                while a < cs:
                    while a < cs and g[i][a] == WATER:
                        a += 1
                    # a points to 1
                    b = a
                    while b < cs and g[i][b] == LAND:
                        b += 1
                    # b points to 0, we have 1's between a, b
                    extended = False
                    for k in range(a, b):
                        if g[i+1][k] == LAND:
                            extended = True
                            break  # island extends to right, no action
                    if not extended and b > a:
                        res += 1

                    a = b  # a point 0 again

            return n1(i+1, j+1, res, g)

        return n1(0, 0, 0, grid)


# O(mn), we visit each cell 4 times from 4 directions at most.
def count_islands_dfs(g: list) -> int:  # g is number of rows
    if not g:  # either None or empty
        return 0

    # the downside is that we modify input data, no good. If we make copy, then space is O(MN)
    def dfs(i, j, g):  # start from (i, j) to go dfs
        if 0 <= i < len(g) and 0 <= j < len(g[0]) and g[i][j] == LAND:
            g[i][j] = WATER  # sink it so we don't come back here later on.
            dfs(i+1, j, g)
            dfs(i-1, j, g)
            dfs(i, j+1, g)
            dfs(i, j-1, g)

    count = 0
    for i in range(len(g)):
        for j in range(len(g[0])):
            if g[i][j] == LAND:  # find a island and start exploration
                count += 1
                dfs(i, j, g)

    return count


def count_islands_bfs(g: list) -> int:  # g is number of rows
    if not g:  # either None or empty
        return 0

    # the downside is that we modify input data, no good. If we make copy, then space is O(MN)
    def bfs(i, j, g):  # start from (i, j) to go dfs
        if 0 <= i < len(g) and 0 <= j < len(g[0]) and g[i][j] == LAND:
            g[i][j] = WATER  # sink it so we don't come back here later on.
            bfs(i+1, j, g)
            bfs(i-1, j, g)
            bfs(i, j+1, g)
            bfs(i, j-1, g)

    count = 0
    for i in range(len(g)):
        for j in range(len(g[0])):
            if g[i][j] == LAND:  # find a island and start exploration
                count += 1
                bfs(i, j, g)

    return count

g = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]

g1 = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

# this proves that we need all 4 directions
g2 = [
    ["1","1","1"],
    ["0","1","0"],
    ["1","1","1"]]

# print(Solution().numIslands(g))
print(count_islands_dfs(g2))
