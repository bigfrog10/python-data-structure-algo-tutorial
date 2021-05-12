# LC708. Insert into a Sorted Circular Linked List
def insert(self, head: 'Node', insertVal: int) -> 'Node':
    if not head:
        head = Node(insertVal, head)
        head.next = head
        return head
    curr = head
    while True:
        if curr.val <= insertVal <= curr.next.val: break
        if curr.val > curr.next.val: # last element
            if curr.val <= insertVal >= curr.next.val: break
            elif curr.val >= insertVal <= curr.next.val: break
        curr = curr.next
        if curr == head: break  # all elements are equal

    curr.next = Node(insertVal, curr.next)
    return head

# LC141. Linked List Cycle
def hasCycle(self, head: ListNode) -> bool:
    if not head: return False
    hare = turtle = head
    while True:
        hare, turtle = hare.next, turtle.next
        if hare is None: return False
        hare = hare.next  # double speed
        if hare is None: return False
        if hare == turtle: return True

# LC142. Linked List Cycle II
def detectCycle(self, head: ListNode) -> ListNode:
    hare = turtle = head
    while hare and hare.next:  # travel n cycles plus arc C1
        hare = hare.next.next # faster
        turtle = turtle.next # slower
        if turtle == hare: break
    if not hare or not hare.next: return None # reach end, no cycle
    hare = head # leave turtle there and restart hare with speed 1
    while True:
        if hare == turtle: return hare
        hare = hare.next
        turtle = turtle.next
