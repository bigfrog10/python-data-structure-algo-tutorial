################################################################################
# Given a non-empty array of decimal digits representing a non-negative integer,
# increment one to the integer.
# The digits are stored such that the most significant digit is at the head of
# the list, and each element in the array contains a single digit.
# You may assume the integer does not contain any leading zero, except the
# number 0 itself.
# EASY
################################################################################

def plus_one(digits: list) -> list:
    carry = 0
    res = []  # we could modify the list directly
    for idx, i in enumerate(digits[::-1]):
        if idx == 0:
            s = i + 1
            carry = s // 10
            s = s % 10
            res.append(s)
        else:
            s = i + carry
            carry = s // 10
            s = s % 10
            res.append(s)

    if carry > 0:
        res.append(carry)

    return res[::-1]


print(plus_one([1, 2, 3]))
print(plus_one([9, 9]))
print(plus_one([0, 2]))
print(plus_one([0, 9]))
print(plus_one([0]))
