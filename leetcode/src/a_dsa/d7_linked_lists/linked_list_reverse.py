
# LC206. Reverse Linked List, top100  reverse list
def reverseList(self, head: ListNode) -> ListNode:
    prev, curr = None, head
    while curr is not None:
        curr.next, prev, curr = prev, curr, curr.next
    return prev

# LC92. Reverse Linked List II - reverse between 2 nodes  reverse list
def reverseBetween(self, head, m, n):
    dummy = start = ListNode(0, head)  # 1 2 3 4 5, m = 2, n = 4
    for _ in range(m-1): start = start.next  # move the  position before m
    pre, curr = None, start.next  # curr = 2, start = 1
    for _ in range(n-m+1):  # reverse the defined part
        curr.next, pre, curr = pre, curr, curr.next
    start.next.next = curr  # point old start to tail: curr = n+1 5
    start.next = pre  # point start to new head, pre = 4 3 2
    return dummy.next

# LC24. Swap Nodes in Pairs - reverse pair
def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
    prev = dummy = ListNode(None, head)
    while head and head.next:
        prev.next, head.next = head.next, head.next.next  # point to next new
        prev.next.next = head  # reverse next
        prev, head = head, head.next  # move forward
    return dummy.next

# LC25. Reverse Nodes in k-Group  k at a time   reverse k-group  reverse k group
def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
    n, curr = 0, head  # find size
    while curr:
        n += 1
        curr = curr.next
    dummy = nhead = ListNode()  # the new list to insert to, nhead is previous
    for _ in range(n//k):
        ntail = head  # the save the position for later jumping to
        for _ in range(k):  # these are a circle
            head.next, nhead.next, head = nhead.next, head, head.next  # insert at nhead
        nhead = ntail  # move insertion point to the right by k steps
    ntail.next = head  # for last remainder connection
    return dummy.next

# LC2130. Maximum Twin Sum of a Linked List
def pairSum(self, head: Optional[ListNode]) -> int:
    fast, rev = head, None # use head as slow
    while fast and fast.next:  # 1, 2, 3, 2, 1    1, 2, 3, 4, 2, 1
        fast = fast.next.next
        rev, rev.next, head = head, rev, head.next
    # head = 3 2 1, tail = 2 1   head = 3 2 1, tail = 4 2 1
    tail = head.next if fast else head  # fast none when list is even
    res = 0
    while rev:
        res = max(res, rev.val + tail.val)
        head, head.next, rev = rev, head, rev.next  # restore head to original
        tail = tail.next  # 2, 3, 2, 1, then 1, 2,3,2,1
    return res

# LC234. Palindrome Linked List  pali linked list
def isPalindrome(self, head: Optional[ListNode]) -> bool:  # O(n) time, O(1) space
    fast, rev = head, None  # use head as slow
    while fast and fast.next:  # 1, 2, 3, 2, 1    1, 2, 3, 4, 2, 1
        fast = fast.next.next
        rev, head, rev.next = head, head.next, rev
    # head = 3 2 1, rev = 2 1   rev = 3 2 1, head = 4 2 1, fast=last or None
    tail = head.next if fast else head  # fast none when list is even
    isPali = True
    while rev and isPali:
        isPali = isPali and rev.val == tail.val
        head, head.next, rev = rev, head, rev.next  # restore head to original
        tail = tail.next  # 2, 3, 2, 1, then 1, 2,3,2,1
    return isPali
def isPalindrome(self, head: ListNode) -> bool:
    vals = []
    current_node = head
    while current_node is not None:
        vals.append(current_node.val)
        current_node = current_node.next
    return vals == vals[::-1]

# LC143. Reorder List - odd + <even reverse> odd even reverse singly
def reorderList(self, head: ListNode) -> None:
    if not head: return
    slow = fast = head  # in 1->2->3->4->5->6 find 4
    while fast and fast.next:  # slow 4 5 6
        slow, fast = slow.next, fast.next.next
    prev, curr = None, slow  # reverse
    while curr:  # convert 4 5 6 to 6->5->4
        curr.next, prev, curr = prev, curr, curr.next
    first, second = head, prev  # prev: 6 5 4, head: 1 2 3 4
    while second.next:  # merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
        first.next, first = second, first.next
        second.next, second = first, second.next

















