################################################################################
# Design a stack that supports push, pop, top, and retrieving the minimum
# element in constant time.
#    push(x) -- Push element x onto stack.
#    pop() -- Removes the element on top of the stack.
#    top() -- Get the top element.
#    getMin() -- Retrieve the minimum element in the stack.
# EASY
################################################################################
# need to maintain 2 orders, push order and value order.
# we may use tuple (value, min) with one stack, or use 2 structures.
# tuple appraoch is simple.

class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.values = []  # list end is stack top

    def push(self, x: int) -> None:
        if self.values:
            t = self.values[-1]
            mv = (x, min(x, t[1]))
        else:
            mv = (x, x)
        self.values.append(mv)

    def pop(self) -> None:
        self.values.pop()

    def top(self) -> int:
        t = self.values[-1]
        return t[0]

    def getMin(self) -> int:
        t = self.values[-1]
        return t[1]
