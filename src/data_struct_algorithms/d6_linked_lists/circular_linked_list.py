

# LC141. Linked List Cycle
def hasCycle(self, head: ListNode) -> bool:
    hare = turtle = head
    while hare and hare.next:
        turtle, hare = turtle.next, hare.next.next
        if hare == turtle: return True
    return False

# LC142. Linked List Cycle II
def detectCycle(self, head: ListNode) -> ListNode:
    hare = turtle = head
    while hare and hare.next:  # travel n cycles plus arc C1
        turtle, hare = turtle.next, hare.next.next # slower, faster
        if turtle == hare: break
    if not hare or not hare.next: return None # reach end, no cycle
    hare = head # leave turtle there and restart hare with speed 1
    while True:
        if hare == turtle: return hare
        turtle, hare = turtle.next, hare.next

# LC708. Insert into a Sorted Circular Linked List
def insert(self, head: 'Node', insertVal: int) -> 'Node':
    if not head: # corner case
        head = Node(insertVal)
        head.next = head
        return head
    curr = head
    while True:
        if curr.val <= insertVal <= curr.next.val: break
        if curr.val > curr.next.val: # max element
            if curr.val <= insertVal >= curr.next.val: break # > max
            elif curr.val >= insertVal <= curr.next.val: break # < min
        curr = curr.next
        if curr == head: break # all elements are equal
    curr.next = Node(insertVal, curr.next)
    return head
