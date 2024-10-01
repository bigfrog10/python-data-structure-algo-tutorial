
# LC148. Sort List  merge sort list
def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next: return head  # [4,2,1,3]
    fast, slow = head.next, head  # need to get prev to 2nd head
    while fast and fast.next:  # after while fast=[3], slow=[2,1,3]
        fast, slow = fast.next.next, slow.next
    start, slow.next = slow.next, None  # start=[1,3] head=[4,2]
    l, r = self.sortList(head), self.sortList(start)  # merge sort
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

# LC21. Merge Two Sorted Lists   merge 2 sorted lists  merge sorted  merge list
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
    curr.next = l1 or l2  # left over
    return head.next

# LC23. Merge k Sorted Lists, top100
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    lstlen, jump = len(lists), 1  ## O(Nlogk)
    while jump < lstlen:  # logk
        for i in range(0, lstlen - jump, jump * 2):   # O(all nodes)
            lists[i] = self.mergeTwoLists(lists[i], lists[i + jump])  ## use above method
        jump *= 2
    return lists[0] if lstlen > 0 else None














