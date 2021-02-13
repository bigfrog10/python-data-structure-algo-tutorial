################################################################################
# Given a non-negative integer x, compute and return the square root of x.
# Since the return type is an integer, the decimal digits are truncated, and
# only the integer part of the result is returned.
# EASY
################################################################################

def my_sqrt(x: int) -> int:
    if x == 0:
        return 0
    if x < 4:  # to ensure sqrt(x) < x / 2
        return 1

    left, right = 2, x // 2  # first 2 is sqrt(4)
    while left <= right:
        middle = left + (right - left) // 2
        sqr = middle * middle
        if sqr > x:
            right = middle - 1  # middle is tested in sqr
        elif sqr < x:
            left = middle + 1
        else:
            return middle

    return right  # close to sqrt(x)


for i in range(1, 26):
    print(f'i={i}, sqrt={my_sqrt(i)}')
