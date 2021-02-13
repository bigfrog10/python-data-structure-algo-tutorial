################################################################################
# Given a 32-bit signed integer, reverse digits of an integer.
# Note:
# Assume we are dealing with an environment that could only store integers
# within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of
# this problem, assume that your function returns 0 when the reversed integer
# overflows.
# EASY
################################################################################
# Python does have this restriction

def reverse_digit(i: int) -> int:
    if i >= 0:
        r = int(str(i)[::-1])
    else:
        r = -int(str(-i)[::-1])

    if r > 2147483647 or r < -2147483648:
        return 0
    else:
        return r


print(reverse_digit(0))
print(reverse_digit(123))
print(reverse_digit(-123))
print(reverse_digit(120))
print(reverse_digit(12345678901234567890))
print(reverse_digit(1534236469))


def reverse_digit1(x: int) -> int:
    pos = x if x >= 0 else -x

    res = 0
    while pos > 0:
        r = pos % 10

        # other language may need to catch this
        res = res * 10 + r
        if res > 2147483647 or res < -2147483648:
            return 0

        pos = pos // 10

    return res if x > 0 else -res


print(reverse_digit1(0))
print(reverse_digit1(123))
print(reverse_digit1(-123))
print(reverse_digit1(120))
print(reverse_digit1(12345678901234567890))
print(reverse_digit1(1534236469))
