################################################################################
# Roman numerals are represented by seven different symbols: I, V, X, L, C, D
# and M.
# Symbol       Value
# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000
# For example, 2 is written as II in Roman numeral, just two one's added
# together. 12 is written as XII, which is simply X + II. The number 27 is
# written as XXVII, which is XX + V + II.
#
# Roman numerals are usually written largest to smallest from left to right.
# However, the numeral for four is not IIII. Instead, the number four is written
# as IV. Because the one is before the five we subtract it making four. The same
# principle applies to the number nine, which is written as IX. There are six
# instances where subtraction is used:
#    I can be placed before V (5) and X (10) to make 4 and 9.
#    X can be placed before L (50) and C (100) to make 40 and 90.
#    C can be placed before D (500) and M (1000) to make 400 and 900.
# Given an integer, convert it to a roman numeral. 1 <= num <= 3999
# MEDIUM
################################################################################

symbols = {1: "I",  4: 'IV', 5: "V", 9: 'IX',
           10: 'X', 40: 'XL', 50: 'L', 90: 'XC',
           100: 'C',  400: 'CD', 500: 'D', 900: 'CM',
           1000: 'M',
           }


def int2roman(base10: int):
    thousands = base10 // 1000
    res = base10 % 1000
    hundreds = res // 100
    res = res % 100
    tens = res // 10
    singles = res % 10

    return _th2r(thousands) + _digit2r(hundreds, 100) + _digit2r(tens, 10) + _digit2r(singles, 1)


def _digit2r(i, unit):
    if i == 0:
        return ''
    if i < 4:
        return symbols[1 * unit] * i
    if i == 4:
        return symbols[4 * unit]
    if i == 5:
        return symbols[5 * unit]
    if i < 9:
        return symbols[5 * unit] + symbols[1 * unit] * (i - 5)
    if i == 9:
        return symbols[9 * unit]


def _th2r(i: int):
    return symbols[1000] * i


for j in range(1, 100):
    print(int2roman(j))

print('-' * 80)
print(int2roman(1994))
