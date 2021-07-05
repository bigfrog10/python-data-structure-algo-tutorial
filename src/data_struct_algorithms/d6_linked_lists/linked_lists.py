class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# LC138. Copy List with Random Pointer  # BBG
def copyRandomList(self, head: 'Node') -> 'Node':
    if not head: return None
    walker = head  # weave in copies
    while walker:
        walker.next = Node(walker.val, walker.next)
        walker = walker.next.next
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

# LC445. Add Two Numbers II
def addTwoNumbers(self, l1, l2):
    def reverse_list(lst):
        prev, curr = None, lst
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        return prev
    rl1, rl2 = reverse_list(l1), reverse_list(l2)
    res = curr = ListNode(0)
    carry = 0
    while rl1 or rl2 or carry:
        v1 = rl1.val if rl1 else 0
        v2 = rl2.val if rl2 else 0
        carry, digit = divmod(v1 + v2 + carry, 10)
        curr.next = ListNode(digit)
        curr = curr.next
        rl1 = rl1.next if rl1 else None
        rl2 = rl2.next if rl2 else None
    return reverse_list(res.next)

# LC160. Intersection of Two Linked Lists
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    pA = headA
    pB = headB
    while pA != pB:
        pA = headB if pA is None else pA.next
        pB = headA if pB is None else pB.next
    return pA

# LC234. Palindrome Linked List
def isPalindrome(self, head):
    fast, rev = head, None # use head as slow
    while fast and fast.next: # 1, 2, 3, 2, 1
        fast = fast.next.next
        rev, rev.next, head = head, rev, head.next
    tail = head.next if fast else head # fast none when list is even
    isPali = True # head is 3, 2, 1
    while rev: # rev is 2, 1
        isPali = isPali and rev.val == tail.val
        head, head.next, rev = rev, head, rev.next # 2, 3, 2, 1, then 1, 2,3,2,1
        tail = tail.next
    print(head)
    return isPali





# LC1290. Convert Binary Number in a Linked List to Integer
def getDecimalValue(self, head: ListNode) -> int:
    num = head.val
    while head.next:
        num = num * 2 + head.next.val
        head = head.next
    return num

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



# LC430. Flatten a Multilevel Doubly Linked List
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
class Solution:
    def flatten(self, head: 'Node') -> 'Node': # DFS
        if not head: return None
        prev = pseudoHead = Node(0,None,head,None)
        stack = [head]
        while stack: # DFS
            curr = stack.pop()
            prev.next = curr  # link prev and current, double direction
            curr.prev = prev
            if curr.next: # push current to stack so we deal with child first
                stack.append(curr.next)
            if curr.child: # deal with child before next
                stack.append(curr.child)
                curr.child = None # don't forget to remove all child pointers.
            prev = curr
        # detach the pseudo head node from the result.
        pseudoHead.next.prev = None
        return pseudoHead.next


# LC817. Linked List Components
def numComponents(self, head: ListNode, nums: List[int]) -> int:
    setG = set(nums)
    res = 0
    while head:
        if head.val in setG and (head.next == None or head.next.val not in setG):
            res += 1
        head = head.next
    return res

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
