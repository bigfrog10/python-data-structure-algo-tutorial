################################################################################
# Merge two sorted linked lists and return it as a new sorted list. The new list
# should be made by splicing together the nodes of the first two lists.
# EASY
################################################################################
from dataclasses import dataclass

@dataclass()
class ListNode:
    val: int
    next: 'ListNode'


# sorted ascending
def merge_sorted_list(list1: ListNode, list2: ListNode) -> ListNode:
    walker = None
    head = None
    while list1 or list2:
        if list1 is None:
            if walker:
                walker.next = list2
                walker = walker.next
            else:
                walker = list2
                head = walker
            list2 = list2.next
        elif list2 is None:
            if walker:
                walker.next = list1
                walker = walker.next
            else:
                walker = list1
                head = walker
            list1 = list1.next
        else:
            v1 = list1.val
            v2 = list2.val  # save these before the pointers move
            if walker:
                walker.next = list1 if v1 <= v2 else list2
                walker = walker.next
            else:
                walker = list1 if v1 <= v2 else list2
                head = walker
            list1 = list1.next if v1 <= v2 else list1
            list2 = list2.next if v1 > v2 else list2

    return head


# Convenient interfaces
def _array2list(*args) -> ListNode:  # convert a list of digits to node list (reverse order)
    result = None
    for d in reversed(args):
        result = ListNode(d, result)

    return result


def _list2array(node: ListNode) -> list:
    ret = []
    while node:
        ret.append(node.val)
        node = node.next

    return ret


res = merge_sorted_list(_array2list(1, 2, 4), _array2list(1, 3, 4))
print(_list2array(res))
print(merge_sorted_list(_array2list(), _array2list()))
print(merge_sorted_list(_array2list(), _array2list(0)))
print(merge_sorted_list(_array2list(-9, 3), _array2list(5, 7)))
