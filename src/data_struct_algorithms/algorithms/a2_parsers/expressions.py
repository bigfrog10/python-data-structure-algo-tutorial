# LC1106. Parsing A Boolean Expression
def parseBoolExpr(self, expression: str) -> bool:
    func = {'&' : all, '|' : any, '!' : lambda x : not x[0]}
    stack = []
    for c in expression:
        if c == 't': stack.append(True)
        elif c == 'f': stack.append(False)
        elif c in func: stack.append(func[c])
        elif c == '(': stack.append('(')
        elif c == ')':
            ss = []
            while stack[-1] != '(': ss.append(stack.pop())
            stack.pop() # skip (
            stack.append(stack.pop()(ss)) # operator
    return stack.pop()

# LC224. Basic Calculator
def calculate(self, s):
    stack, sign, num = [], '+', 0  # stack for () and sign in front
    for i, c in enumerate(s + '+'):
        if c.isdigit(): num = num * 10 + int(c)
        elif c == '(':
            stack.append(sign) # save history
            stack.append('(')
            sign = '+' # start new sign
        elif c in '+-)':
            if sign == '+': stack.append(num)
            elif sign == '-': stack.append(-num)
            if c == ')':
                num, item = 0, stack.pop()
                while item != '(':
                    num += item
                    item = stack.pop()
                sign = stack.pop()
            else: sign, num = c, 0
    return sum(stack)

# LC227. Basic Calculator II
def calculate(self, s: str) -> int:
    stack, sign, num = [], '+', 0  # stack for () and sign in front
    for i, c in enumerate(s + '+'):
        if c.isdigit(): num = num * 10 + int(c)
        elif c in '+-*/)':
            if sign == '+': stack.append(num)
            elif sign == '-': stack.append(-num)
            elif sign == '*': stack.append(stack.pop() * num)
            elif sign == '/': stack.append(int(stack.pop() / num))
            sign, num = c, 0 # this is for +-*/
    return sum(stack)

# LC772. Basic Calculator III
def calculate(self, s: str) -> int:
    stack, sign, num = [], '+', 0  # stack for () and sign in front
    for i, c in enumerate(s + '+'):
        if c.isdigit(): num = num * 10 + int(c)
        elif c == '(':
            stack.append(sign) # save history
            stack.append(c)
            sign = '+'
        elif c in '+-*/)':
            if sign == '+': stack.append(num)
            elif sign == '-': stack.append(-num)
            elif sign == '*': stack.append(stack.pop() * num)
            elif sign == '/': stack.append(int(stack.pop() / num))
            if c == ')':
                num, item = 0, stack.pop()  # num is used afterward
                while item != '(':
                    num += item
                    item = stack.pop()
                sign = stack.pop()
            else: sign, num = c, 0 # this is for +-*/
    return sum(stack)

# LC1597. Build Binary Expression Tree From Infix Expression
def expTree(self, s: str) -> 'Node':
    n = len(s)
    if n == 1: return Node(s) # number
    fstOpIdx, bal = None, 0
    for i in range(n-1, 0, -1): # find operator index from back
        if s[i] == ")": bal += 1
        elif s[i] == "(": bal -= 1
        elif bal == 0: # handle outside of parenthesis first
            if s[i] in "+-":
                fstOpIdx = i
                break # break out for +-, this is the root.
            elif s[i] in "*/" and fstOpIdx is None:
                fstOpIdx = i # stay on rightmost value
    if fstOpIdx is None:
        return self.expTree(s[1:-1]) # remove parenthesis
    node = Node(s[fstOpIdx]) # operator
    node.left = self.expTree(s[:fstOpIdx])
    node.right = self.expTree(s[fstOpIdx+1:])
    return node

# LC1612. Check If Two Expression Trees are Equivalent
def checkEquivalence(self, root1: 'Node', root2: 'Node') -> bool:
    def inorder(root):
        return inorder(root.left) + [root.val] + inorder(root.right) if root else []
    r1 = inorder(root1)
    r2 = inorder(root2)
    return collections.Counter(r1) == collections.Counter(r2)

# LC150. Evaluate Reverse Polish Notation
def evalRPN(self, tokens: List[str]) -> int:
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "/": lambda a, b: int(a / b),
        "*": lambda a, b: a * b
    }
    stack = []
    for token in tokens:
        if token in operations:
            number_2 = stack.pop()
            number_1 = stack.pop()
            operation = operations[token]
            stack.append(operation(number_1, number_2))
        else: stack.append(int(token))
    return stack.pop()

# LC1628. Design an Expression Tree With Evaluate Function
class VNode(Node):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def evaluate(self) -> int:
        if self.val.isnumeric(): return int(self.val)
        else:
            v1 = self.left.evaluate()
            v2 = self.right.evaluate()
            if self.val == '+': return v1 + v2
            elif self.val == '-': return v1 - v2
            elif self.val == '*': return v1 * v2
            elif self.val == '/': return v1 // v2
            return None
class TreeBuilder(object):
    def buildTree(self, postfix: List[str]) -> 'Node':
        oprnd_stack = []
        for s in postfix:
            if s.isnumeric(): oprnd_stack.append(s)
            else:
                s1 = oprnd_stack.pop()
                s2 = oprnd_stack.pop()
                tn1 = VNode(s1) if isinstance(s1, str) else s1
                tn2 = VNode(s2) if isinstance(s2, str) else s2
                tn = VNode(s, tn2, tn1)
                oprnd_stack.append(tn)
        return oprnd_stack.pop()
