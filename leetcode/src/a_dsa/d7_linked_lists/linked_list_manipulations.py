
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
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    lstlen = len(lists)  # O(Nlogk)
    jump = 1
    while jump < lstlen:
        for i in range(0, lstlen - jump, jump * 2):
            lists[i] = self.mergeTwoLists(lists[i], lists[i + jump])  ## use above method
        jump *= 2
    return lists[0] if lstlen > 0 else None

# LC148. Sort List
def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
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
            p.next, l = l, l.next
        else:
            p.next, r = r, r.next
        p = p.next
    p.next = l or r
    return dummy.next

# LC19. Remove Nth Node From End of List
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)  # handle k=0
    first = second = dummy
    for i in range(n+1): first = first.next  # +1 for dummy, after nth node in front.
    while first: first, second = first.next, second.next  # parent of nth node in back.
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

# LC83. Remove Duplicates from Sorted List
def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
    curr = head
    while curr and curr.next:
        if curr.val == curr.next.val: curr.next = curr.next.next
        else: curr = curr.next
    return head

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

# LC237. Delete Node in a Linked List
def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next
