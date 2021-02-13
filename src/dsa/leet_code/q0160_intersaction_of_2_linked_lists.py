################################################################################
# Write a program to find the node at which the intersection of two singly
# linked lists begins.
# EASY
################################################################################
# traverse both lists once to get the sizes, then traverse the longer list for
# the difference steps so that now we have same length lists.
# Then traverse both one step at a time at the same time for comparison.
#
# Another way is below. walk at the same time on both list. If the end is
# reached, then point to the other head. Comparison both end nodes to check
# whether there is an intersaction.
# after both walkers walk M + N - Common length, they meet at the starting point
# of the common segment.

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None

        walkerA, walkerB = headA, headB
        endA, endB = None, None
        while True:
            # order is important
            if walkerA.next == None:
                endA = walkerA  # remember last node for checking intersection
            if walkerB.next == None:
                endB = walkerB
            if endA and endB and endA != endB:
                return None  # this is to test whether there is an intersection

            # So we have intersection
            if walkerA == walkerB:  # while walking, if there is a match return it.
                return walkerA
            else:  # no match, then walk
                walkerA, walkerB = walkerA.next, walkerB.next
            if walkerA is None:
                walkerA = headB
            if walkerB is None:
                walkerB = headA
