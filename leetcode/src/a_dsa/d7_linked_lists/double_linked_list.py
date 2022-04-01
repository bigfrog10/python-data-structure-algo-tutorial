
# LC430. Flatten a Multilevel Doubly Linked List
def flatten(self, head: 'Node') -> 'Node': # DFS
    if not head: return None
    prev = dummy = Node(0, None, head, None)
    stack = [head]
    while stack: # DFS
        curr = stack.pop()
        prev.next, curr.prev = curr, prev  # link prev and current, double direction
        if curr.next: # push current to stack so we deal with child first
            stack.append(curr.next)
        if curr.child: # deal with child before next
            stack.append(curr.child)
            curr.child = None # don't forget to remove all child pointers.
        prev = curr
    # detach the pseudo head node from the result.
    dummy.next.prev = None
    return dummy.next

