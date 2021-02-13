################################################################################
# Determine whether an integer is a palindrome. An integer is a palindrome when
# it reads the same backward as forward.
#
# Follow up: Could you solve it without converting the integer to a string?
# EASY
################################################################################
# q0005 can be used here, if we convert it to string
# q0007 can be used here, if we reverse digits and check equal. If run out of
# int range, then it's not.

def is_palindrome(x: int) -> bool:
    if x < 0 or (x > 0 and x % 10 == 0):
        return False

    revert = 0
    while x > revert:
        revert = revert * 10 + x % 10
        x = x // 10

    return x == revert or x == revert // 10


print(is_palindrome(121))
print(is_palindrome(1221))
print(is_palindrome(1231))
