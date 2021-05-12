class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# LC21. Merge Two Sorted Lists
def mergeTwoLists(self, l1, l2):
    prehead = ListNode(-1)
    prev = prehead
    while l1 and l2:
        if l1.val <= l2.val:
            prev.next = l1
            l1 = l1.next
        else:
            prev.next = l2
            l2 = l2.next
        prev = prev.next
    prev.next = l1 if l1 else l2
    return prehead.next

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

# LC160. Intersection of Two Linked Lists
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    pA = headA
    pB = headB
    while pA != pB:
        pA = headB if pA is None else pA.next
        pB = headA if pB is None else pB.next
    return pA

# LC817. Linked List Components
def numComponents(self, head: ListNode, nums: List[int]) -> int:
    setG = set(nums)
    res = 0
    while head:
        if head.val in setG and (head.next == None or head.next.val not in setG):
            res += 1
        head = head.next
    return res

# LC369. Plus One Linked List
def plusOne(self, head: ListNode) -> ListNode:
    node, start = head, None
    while node:
        if node.val < 9: start = node
        node = node.next
    if start:
        start.val += 1
        node = start.next
    else:
        new = ListNode(1)
        new.next = head
        node = head
        head = new
    while node:
        node.val = 0
        node = node.next
    return head

# LC1721. Swapping Nodes in a Linked List
def swapNodes(self, head: ListNode, k: int) -> ListNode:
    slow, fast = head, head
    for _ in range(k - 1): fast = fast.next
    first = fast # save first, the kth from head
    while fast.next:
        slow, fast = slow.next, fast.next
    first.val, slow.val = slow.val, first.val
    return head
# LC876. Middle of the Linked List
def middleNode(self, head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
# LC138. Copy List with Random Pointer
def copyRandomList(self, head: 'ListNode') -> 'ListNode':
    if not head: return None
    walker = head  # weave in copies
    while walker:
        nn = ListNode(walker.val, walker.next)
        walker.next = nn
        walker = nn.next
    walker = head # copy random pointers
    while walker:
        if walker.random: walker.next.random = walker.random.next
        walker = walker.next.next
    new_head = nwalker = head.next  # seperate old and new
    walker = head
    while nwalker:
        walker.next = nwalker.next
        walker = walker.next
        nwalker.next = walker.next if walker else None
        nwalker = nwalker.next
    return new_head

# LC2. Add Two Numbers, top100
def addTwoNumbers(self, l1, l2):
    result = result_tail = ListNode(0)
    carry = 0
    while l1 or l2 or carry:  # pattern
        val1  = (l1.val if l1 else 0)  # pattern
        val2  = (l2.val if l2 else 0)  # pattern
        carry, out = divmod(val1+val2 + carry, 10)  # pattern
        result_tail.next = ListNode(out)
        result_tail = result_tail.next
        l1 = l1.next if l1 else None  # pattern
        l2 = l2.next if l2 else None  # pattern
    return result.next

# LC234. Palindrome Linked List
def isPalindrome(self, head):
    fast, rev = head, None # use head as slow
    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, head = head, rev, head.next # reverse list
    tail = head.next if fast else head
    isPali = True
    while rev:
        isPali = isPali and rev.val == tail.val
        head, head.next, rev = rev, head, rev.next # reverse again
        tail = tail.next
    return isPali

# LC328. Odd Even Linked List
def oddEvenList(self, head: ListNode) -> ListNode:
    if not head: return head
    odd_walker = head # condition leans on even, so code is simpler.
    even_head = even_walker = head.next
    while even_walker and even_walker.next:
        odd_walker.next = even_walker.next  # jump next node to next next
        odd_walker = odd_walker.next
        even_walker.next = odd_walker.next
        even_walker = even_walker.next
    odd_walker.next = even_head
    return head

# LC61. Rotate List
def rotateRight(self, head: ListNode, k: int) -> ListNode:
    if not head or not k: return head
    last, L = head, 1
    while last.next:
        last = last.next
        L += 1
    last.next = head
    for _ in range(L - k % L): last = last.next
    dummy = last.next # new head
    last.next = None
    return dummy
