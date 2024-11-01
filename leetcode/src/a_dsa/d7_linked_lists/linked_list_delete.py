

# LC3217. Delete Nodes From Linked List Present in Array set
def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
    s = set(nums)
    dummy = curr = ListNode(0, head)
    while curr and curr.next:
        if curr.next.val in s:
            curr.next = curr.next.next
        else: curr = curr.next
    return dummy.next

# LC19. Remove Nth Node From End of List
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    first = second = dummy = ListNode(0, head)  # handle k=0
    for i in range(n+1): first = first.next  # +1 for dummy, after nth node in front.
    while first: first, second = first.next, second.next  # parent of nth node in back.
    second.next = second.next.next
    return dummy.next

# LC203. Remove Linked List Elements - by value remove node by value delete node value
def removeElements(self, head: ListNode, val: int) -> ListNode:
    dummy = curr = ListNode(0, head)
    while curr and curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else: curr = curr.next
    return dummy.next

# LC83. Remove Duplicates from Sorted List - keep one dupe copy  remove dupe delete node dupe
def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
    curr = head
    while curr and curr.next:
        if curr.val == curr.next.val: curr.next = curr.next.next
        else: curr = curr.next
    return head

# LC82. Remove Duplicates from Sorted List II - remove all dupes  remove dupe delete node dupe
def deleteDuplicates(self, head: ListNode) -> ListNode:  # O(n)
    prev = dummy = ListNode(-1, head)
    while head:
        while head.next and head.val == head.next.val:
            head = head.next  # at the end, head is last dupe elem
        if prev.next == head: prev = head  # no dupe
        else:prev.next = head.next  # has dupe, skip them
        head = head.next
    return dummy.next

# LC237. Delete Node in a Linked List
def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next

# LC2487. Remove Nodes From Linked List
def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if head is None or head.next is None:
        return head
    next_node = self.removeNodes(head.next)
    # If the next node has greater value than head, delete the head
    # Return next node, which removes the current head and
    # makes next the new head
    if head.val < next_node.val:
        return next_node
    head.next = next_node  # Keep the head
    return head




