# LC1209. Remove All Adjacent Duplicates in String II
def removeDuplicates(self, s, k):
    stack = [['#', 0]]
    for c in s:
        if stack[-1][0] == c:
            stack[-1][1] += 1
            if stack[-1][1] == k: stack.pop() # remove this group
        else: stack.append([c, 1])
    return ''.join(c * k for c, k in stack)
