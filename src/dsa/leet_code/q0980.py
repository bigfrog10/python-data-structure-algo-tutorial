
def uniquePathsIII(grid: list) -> int:
    if not grid:
        return 0

    h = len(grid)
    w = len(grid[0])
    c = 0

    def is_finished(grid):
        for i in range(len(grid)):
            row = grid[i]
            for j in range(len(row)):
                if row[j] == 0:  # unoccupied
                    return False

        return True

    def move(grid, i, j, p):
        nonlocal c
        if grid[i][j] == 1 or grid[i][j] == -1 or grid[i][j] == 3:
            return False
        elif grid[i][j] == 2:
            if is_finished(grid):
                c += 1
                print(p)
                return True
            else:
                return False
        elif grid[i][j] == 0:
            grid[i][j] = 3
            p.append((i, j))
            res = backtrack(grid, i, j, p)
            p.pop()
            grid[i][j] = 0 # restore

            return res

    def backtrack(grid, i, j, p):
        if i+1 < h:
            move(grid, i+1, j, p)
        if i-1 >= 0:
            move(grid, i-1, j, p)
        if j+1 < w:
            move(grid, i, j+1, p)
        if j-1 >= 0:
            move(grid, i, j-1, p)

        return False

    # find start
    si, sj = 0, 0
    for i in range(len(grid)):
        row = grid[i]
        found = False
        for j in range(len(row)):
            if row[j] == 1:  # unoccupied
                si, sj = i, j
                found = True
                break
        if found:
            break

    p = [(si, sj)]
    backtrack(grid, si, sj, p)
    return c

p = [[1,0,0,0],[0,0,0,0],[0,0,2,-1]]
print(uniquePathsIII(p))
