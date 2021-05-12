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
