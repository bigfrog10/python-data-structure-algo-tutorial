################################################################################
# Given head, the head of a linked list, determine if the linked list has a
# cycle in it.
#
# There is a cycle in a linked list if there is some node in the list that can
# be reached again by continuously following the next pointer. Internally, pos
# is used to denote the index of the node that tail's next pointer is connected
# to. Note that pos is not passed as a parameter.
#
# Return true if there is a cycle in the linked list. Otherwise, return false.
# EASY
################################################################################
# Hare and Turtule

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def has_cycle(head: ListNode):
    if not head:
        return False

    hare = head
    turtle = head
    while True:
        turtle = turtle.next
        hare = hare.next
        if hare is None:
            return False
        hare = hare.next  # double speed
        if hare is None:
            return False

        if hare == turtle:
            return True
