################################################################################
# The string "PAYPALISHIRING" is written in a zigzag pattern on a given number
# of rows like this: (you may want to display this pattern in a fixed font for
# better legibility)
# P   A   H   N
# A P L S I I G
# Y   I   R
# And then read line by line: "PAHNAPLSIIGYIR".
# If we go with 4 rows,
# P     I    N
# A   L S  I G
# Y A   H R
# P     I
# If we go with 2 rows,
# PYAIHRN
# APLSIIG
#
# MEDIUM
################################################################################
# Zigzag vertically.
# Vertical line has size of #rows, between vertical lines has size of #rows - 2
# (minus first and last row). So first row chars follow this.
# For 2nd row, L is 2 * #rows - 2 - 2 away from A, I is same away from S.
# Generally, this number 2 * # rows - 2 - 2 * i, for row i.

def to_zigzag_line(s: str, rows: int) -> str:
    if rows < 2:
        return s

    s_len = len(s)
    res = ''
    dist = 2 * rows - 2
    for i in range(rows):
        for j in range(i, s_len, dist):  # loop over 2 vertical lines
            res += s[j]

            if 0 < i < rows - 1:  # not first or last row
                k = j + dist - 2 * i
                if k < s_len:
                    res += s[k]

    return res


print(to_zigzag_line('PAYPALISHIRING', 1))
print(to_zigzag_line('PAYPALISHIRING', 2))
print(to_zigzag_line('PAYPALISHIRING', 3))
print(to_zigzag_line('PAYPALISHIRING', 4))
print(to_zigzag_line('PAYPALISHIRING', 15))


def to_zigzag(s: str, rows: int) -> str:
    if rows < 2:
        return s

    s_len = len(s)
    res = ''
    dist = 2 * rows - 2
    for i in range(rows):
        for j in range(i, s_len, dist):  # loop over 2 vertical lines
            res += s[j]
            if i == 0 or i == rows - 1:
                for k in range(rows-2):
                    res += ' '
            else:
                k = j + dist - 2 * i
                for m in range(rows-2-i):
                    res += ' '
                if k < s_len:
                    res += s[k]
                for m in range(rows-2-i, rows-3):  # extra -1 since s[k] is filled.
                    res += ' '
        res += '\n'
    return res


print(to_zigzag('PAYPALISHIRING', 1))
print(to_zigzag('PAYPALISHIRING', 2))
print(to_zigzag('PAYPALISHIRING', 3))
print(to_zigzag('PAYPALISHIRING', 4))
print(to_zigzag('PAYPALISHIRING', 15))
