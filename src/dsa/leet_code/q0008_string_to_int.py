################################################################################
# Implement atoi which converts a string to an integer.
# The function first discards as many whitespace characters as necessary until
# the first non-whitespace character is found. Then, starting from this
# character takes an optional initial plus or minus sign followed by as many
# numerical digits as possible, and interprets them as a numerical value.
#
# The string can contain additional characters after those that form the
# integral number, which are ignored and have no effect on the behavior of this
# function.
#
# If the first sequence of non-whitespace characters in str is not a valid
# integral number, or if no such sequence exists because either str is empty or
# it contains only whitespace characters, no conversion is performed.
#
# If no valid conversion could be performed, a zero value is returned.
#
# Note:
#     - Only the space character ' ' is considered a whitespace character.
#     - Assume we are dealing with an environment that could only store integers
#       within the 32-bit signed integer range: [−2^31,  2^31 − 1]. If the
#       numerical value is out of the range of representable values, 2^31 − 1 or
#       −2^31 is returned.
# MEDIUM
################################################################################

def atoi(s: str) -> int:
    s = s.strip()
    sign = '+'
    if s[0] == '-':
        sign = '-'
        s = s[1:]

    digits = '1234567890'
    end = len(s)
    for idx, c in enumerate(s):
        if c not in digits:
            end = idx
            break
    s = s[:end]

    if end == 0:
        return 0

    res = int(s)
    if sign == '-':
        res = -res

    if res < -2147483648:  # -2^31
        return -2147483648
    if res > 2147483647:  # 2^31 - 1
        return 2147483647

    return res


print(atoi('42'))
print(atoi('    -42'))
print(atoi('4193 with words'))
print(atoi('words and 987'))
print(atoi('-91283472332'))
