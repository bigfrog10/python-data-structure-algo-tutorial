from dataclasses import dataclass

@dataclass()
class ListNode:
    val: int
    next: 'ListNode'


def merge_sorted_list(list1: ListNode, list2: ListNode) -> ListNode:
    head = ListNode(0, None)
    walker = head

    while list1 and list2:
        if list1.val <= list2.val:
            walker.next = list1
            list1 = list1.next
        else:
            walker.next = list2
            list2 = list2.next

        walker = walker.next

    walker.next = list1 if list1 else list2

    return head.next


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
