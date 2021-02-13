################################################################################
# You are given two non-empty linked lists representing two non-negative
# integers. The digits are stored in reverse order, and each of their nodes
# contains a single digit. Add the two numbers and return the sum as a linked
# list. You may assume the two numbers do not contain any leading zero, except
# the number 0 itself.
# Example:
#     2 -> 4 -> 3
#     5 -> 6 -> 4
# -----------------
#     7 -> 0 -> 8
# MEDIUM
################################################################################

# Two list may not have the same length.
# test cases:
#    2 4 + 5 6 4
#    2 4 3 + 5 6 4
#    2 4 3 + 5 6
#    5 + 5
#    9 9 9 + 1
#    1 + 9 9 9

from dataclasses import dataclass


@dataclass()
class ListNode:
    value: int
    next: 'ListNode'


def sum_digits(list_int1: ListNode, list_int2: ListNode) -> ListNode:
    # list is already started from last digit.
    result = None
    head = None
    carry = 0
    current1 = list_int1
    current2 = list_int2
    while current1:  # loop first list
        if current2:  # if second list has digit too
            sum1 = current1.value + current2.value + carry
            current2 = current2.next
        else:  # if second list ends
            sum1 = current1.value + carry

        carry = sum1 // 10
        residue = sum1 % 10

        if result:
            result.next = ListNode(residue, None)
            result = result.next
        else:
            result = ListNode(residue, None)
            head = result

        current1 = current1.next

    while current2:  # in case list 2 is longer, we continue
        sum1 = current2.value + carry

        carry = sum1 // 10
        residue = sum1 % 10

        if result:
            result.next = ListNode(residue, None)
            result = result.next
        else:
            result = ListNode(residue, None)
            head = result

        current2 = current2.next

    if carry:
        result.next = ListNode(carry, None)

    return head


# Convenient interfaces
def _array2list(*args):  # convert a list of digits to node list (reverse order)
    result = None
    for d in reversed(args):
        result = ListNode(d, result)

    return result


def _num2list(num: int):  # convert a number to node list (reverse order)
    result = None
    ret = None
    while num:
        digit = num % 10
        if not result:
            result = ListNode(digit, None)
            ret = result
        else:
            result.next = ListNode(digit, None)
            result = result.next
        num = num // 10

    return ret


def _list2num(num_list: ListNode):  # convert a list of nodes back to number
    nl = num_list
    ret = 0
    power = 0
    while nl:
        ret = nl.value * 10 ** power + ret
        nl = nl.next
        power += 1

    return ret


print(sum_digits(_array2list(3, 4, 2), _array2list(4, 6, 5)))
print(_list2num(sum_digits(_num2list(342), _num2list(465))) == (342 + 465))
print(_list2num(sum_digits(_num2list(342), _num2list(65))) == (342 + 65))
print(_list2num(sum_digits(_num2list(42), _num2list(465))) == (42 + 465))
print(_list2num(sum_digits(_num2list(5), _num2list(5))) == (5 + 5))
print(_list2num(sum_digits(_num2list(999), _num2list(1))) == 1000)
print(_list2num(sum_digits(_num2list(1), _num2list(999))) == 1000)
