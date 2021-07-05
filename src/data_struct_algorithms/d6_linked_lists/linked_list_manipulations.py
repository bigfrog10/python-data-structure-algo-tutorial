
# LC21. Merge Two Sorted Lists
def mergeTwoLists(self, l1, l2):
    curr = head = ListNode(-1)
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 if l1 else l2  # left over
    return head.next

# LC23. Merge k Sorted Lists, top100
def mergeKLists(self, lists): # Every node is touched logk times, so O(Nlogk)
    amount = len(lists)
    interval = 1
    while interval < amount:
        for i in range(0, amount - interval, interval * 2):
            lists[i] = self.merge2Lists(lists[i], lists[i + interval])
        interval *= 2
    return lists[0] if amount > 0 else None

def merge2Lists(self, l1, l2):
    head = point = ListNode(0)  # dummy node
    while l1 and l2:
        if l1.val <= l2.val:
            point.next = l1
            l1 = l1.next
        else:
            point.next = l2
            l2 = l2.next
        point = point.next
    if not l1: point.next=l2
    else: point.next=l1
    return head.next

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

# LC24. Swap Nodes in Pairs
def swapPairs(self, head: ListNode) -> ListNode:
    prev_node = dummy = ListNode(-1, head)
    while head and head.next:
        first_node, second_node = head, head.next # Nodes to be swapped
        # relink to down the road
        prev_node.next, first_node.next = second_node, second_node.next
        second_node.next = first_node # reverse between current and next
        # Reinitializing the head and prev_node for next swap
        prev_node, head = first_node, first_node.next
    return dummy.next

# LC206. Reverse Linked List, top100
def reverseList(self, head: ListNode) -> ListNode:
    prev, curr = None, head
    while curr is not None:
        curr.next, prev, curr = prev, curr, curr.next
    return prev

# LC143. Reorder List
def reorderList(self, head: ListNode) -> None:
    if not head: return
    slow = fast = head # in 1->2->3->4->5->6 find 4
    while fast and fast.next: # slow is the middle node
        slow, fast = slow.next, fast.next.next
    prev, curr = None, slow # reverse the second half, prev is new head
    while curr: # convert 1->2->3->4->5->6 into 1->2->3->4 and 6->5->4
        curr.next, prev, curr = prev, curr, curr.next
    first, second = head, prev # merge two sorted linked lists
    while second.next: # merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
        first.next, first = second, first.next
        second.next, second = first, second.next

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

# LC82. Remove Duplicates from Sorted List II  # BBG
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

# LC148. Sort List
def sortList(self, head):
    if not head or not head.next: return head
    fast, slow = head.next, head
    while fast and fast.next:
        fast, slow = fast.next.next, slow.next
    start = slow.next
    slow.next = None
    l, r = self.sortList(head), self.sortList(start)
    return self.merge(l, r)
def merge(self, l, r):
    dummy = p = ListNode(0)
    while l and r:
        if l.val < r.val:
            p.next = l
            l = l.next
        else:
            p.next = r
            r = r.next
        p = p.next
    p.next = l or r
    return dummy.next

# LC92. Reverse Linked List II
def reverseBetween(self, head, m, n):
    dummy = start = ListNode(0, head)
    for _ in range(m-1): start = start.next # move the  position before m
    pre, curr = None, start.next # point to pre, right before cur
    for _ in range(n-m+1): # reverse the defined part
        curr.next, pre, curr = pre, curr, curr.next
    start.next.next = curr # point old start to tail: curr = n+1
    start.next = pre # point start to new head
    return dummy.next





# LC19. Remove Nth Node From End of List
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    first = second = dummy
    for i in range(n+1): first = first.next # to get nth parent
    while first: first, second = first.next, second.next # loop to end
    second.next = second.next.next
    return dummy.next

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
