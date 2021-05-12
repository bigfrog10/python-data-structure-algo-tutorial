# LC25. Reverse Nodes in k-Group
def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
    n, curr = 0, head # find size
    while curr:
        n += 1
        curr = curr.next
    dummy = nhead = ListNode() # the new list to insert to
    ntail = None  # store temp head for next
    for _ in range(n//k):
        ntail = head  # the save the position for later jumping to
        for _ in range(k):
            head.next, nhead.next, head = nhead.next, head, head.next  # insert at nhead
        nhead = ntail  # move insertion point to the right by k steps
    ntail.next = head  # for last remainder connection
    return dummy.next

# LC206. Reverse Linked List, top100
def reverseList(self, head: ListNode) -> ListNode:
    prev, curr = None, head
    while curr is not None:
        new_curr, curr.next = curr.next, prev # save next, reverse next
        prev, curr = curr, new_curr # move prev and current forward
    return prev

# LC92. Reverse Linked List II
def reverseBetween(self, head, m, n):
    dummy = pre = ListNode(0)
    dummy.next = head
    for _ in range(m-1): pre = pre.next # move the  position before start
    cur= pre.next # start
    node = None # point to pre, right before cur
    for _ in range(n-m+1): # reverse the defined part
        nxt = cur.next  # save next
        cur.next = node # point next to prev node
        node = cur # move forward, node, curr
        cur= nxt
    pre.next.next = cur # point old start to tail
    pre.next = node # point pre to new start/old end
    return dummy.next

##### DELETE ################################################################
# LC203. Remove Linked List Elements
def removeElements(self, head: ListNode, val: int) -> ListNode:
    sentinel = ListNode(0, head)
    prev, curr = sentinel, head
    while curr:
        if curr.val == val: prev.next = curr.next
        else: prev = curr
        curr = curr.next
    return sentinel.next

# LC237. Delete Node in a Linked List
def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next

# LC19. Remove Nth Node From End of List
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    first = second = dummy
    for i in range(n+1): first = first.next # to get nth parent
    while first: first, second = first.next, second.next # loop to end
    second.next = second.next.next
    return dummy.next

# LC82. Remove Duplicates from Sorted List II - keep distinct
def deleteDuplicates(self, head: ListNode) -> ListNode:
    pred = sentinel = ListNode(0, head)
    while head:
        # if it's a beginning of duplicates sublist
        # skip all duplicates
        if head.next and head.val == head.next.val:
            # move till the end of duplicates sublist
            while head.next and head.val == head.next.val:
                head = head.next
            pred.next = head.next # skip all duplicates
        # otherwise, move predecessor
        else: pred = pred.next
        head = head.next
    return sentinel.next

# LC83. Remove Duplicates from Sorted List
def deleteDuplicates(self, head: ListNode) -> ListNode:
    pred = sentinel = ListNode(0, head)
    while head:
        # if it's a beginning of duplicates sublist
        # skip all duplicates
        if head.next and head.val == head.next.val:
            # move till the end of duplicates sublist
            pred = pred.next
            while head.next and head.val == head.next.val:
                head = head.next
            pred.next = head.next # skip all duplicates
        # otherwise, move predecessor
        else: pred = pred.next
        head = head.next
    return sentinel.next
