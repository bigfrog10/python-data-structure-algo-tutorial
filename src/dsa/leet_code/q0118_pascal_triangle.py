################################################################################
# Given a non-negative integer numRows, generate the first numRows of Pascal's
# triangle.
# EASY
################################################################################

def pascal_triangle(numRows: int):
    res = []
    if numRows == 0:
        return res

    row = [1]
    res.append(row)
    for i in range(numRows-1):
        nr = [1]
        for j in range(len(row)-1):
            nr.append(row[j] + row[j+1])

        nr.append(1)
        res.append(nr)
        row = nr

    return res


for i in range(3, 10):
    print(pascal_triangle(i))
