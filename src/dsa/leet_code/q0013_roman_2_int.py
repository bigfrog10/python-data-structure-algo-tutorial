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
# Roman numerals are usually written largest to smallest from left to right.
# However, the numeral for four is not IIII. Instead, the number four is written
# as IV. Because the one is before the five we subtract it making four. The same
# principle applies to the number nine, which is written as IX. There are six
# instances where subtraction is used:
#    I can be placed before V (5) and X (10) to make 4 and 9.
#    X can be placed before L (50) and C (100) to make 40 and 90.
#    C can be placed before D (500) and M (1000) to make 400 and 900.
# Given a roman numeral, convert it to an integer.
# EASY
################################################################################
base10 = {"I": 1, 'IV': 4, "V": 5, 'IX': 9,
               'X': 10, 'XL': 40, 'L': 50, 'XC': 90,
               'C': 100,  'CD': 400, 'D': 500, 'CM': 900,
               'M': 1000,
           }


def roman2int(s: str) -> int:
    ret = 0
    idx = 0
    while idx < len(s):
        v, idx = _roman_digit(s, idx)
        ret += v

    return ret


def _roman_digit(s: str, start: int):
    v = base10.get(s[start: start+2], None)
    if v is not None:
        return v, start + 2

    if s.startswith('III'):
        return 3, start + 3
    elif s.startswith('II'):
        return 2, start + 2
    else:
        v = base10.get(s[start], None)
        if v is None:
            raise ValueError('unknown char: ' + s[start])

        return v, start + 1


for k in base10.keys():
    print(roman2int(k))

print(roman2int('II'))
print(roman2int('III'))

print(roman2int('LVIII'))
print(roman2int('MCMXCIV'))
