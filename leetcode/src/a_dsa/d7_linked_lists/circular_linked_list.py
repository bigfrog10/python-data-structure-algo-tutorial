
# LC708. Insert into a Sorted Circular Linked List  circular sorted
def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
    if not head:  # corner case
        head = Node(insertVal)
        head.next = head  # make it circular
        return head
    curr = head
    while True:  # O(n)
        if curr.val <= insertVal <= curr.next.val: break  # right place
        if curr.val > curr.next.val: # max element
            if insertVal >= curr.val: break # > max
            elif insertVal <= curr.next.val: break # < min
        curr = curr.next
        if curr == head: break # one node list
    curr.next = Node(insertVal, curr.next)  # insert between curr and curr.next
    return head

# LC141. Linked List Cycle  # Floyd turtle hare turtle
def hasCycle(self, head: ListNode) -> bool:
    hare = turtle = head
    while hare and hare.next:
        turtle, hare = turtle.next, hare.next.next
        if hare == turtle: return True
    return False

# LC142. Linked List Cycle II - cycle start point   turtle hare turtle
def detectCycle(self, head: ListNode) -> ListNode:
    hare = turtle = head
    while hare and hare.next:  # travel n cycles plus arc C1
        turtle, hare = turtle.next, hare.next.next # slower, faster
        if turtle == hare: break
    if not hare or not hare.next: return None # reach end, no cycle
    hare = head # leave turtle there and restart hare with speed 1
    while hare != turtle:
        # if hare == turtle: return hare
        turtle, hare = turtle.next, hare.next
    return hare
