
# LC772. Basic Calculator III - +-*/()
def calculate(self, s: str) -> int:  # O(n) time and space
    stack, sign, num = [], '+', 0  # stack for () and sign in front
    for i, c in enumerate(s + '+'):
        if c.isdigit(): num = num * 10 + int(c)
        elif c == '(':
            stack.append(sign)  # save history
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

# LC282. Expression Add Operators  - return all results *** 4 cards to 24 game  exp add
def addOperators(self, num: str, target: int) -> List[str]:
    n, res = len(num), []  # O(4^n) (3 operators plus no-op)
    def dfs(idx, expr, cur, last):  # cur is the current value, last is last value
        if idx == n:
            if cur == target: res.append(expr)
            return
        for i in range(idx + 1, n + 1):  # n+1 because we have num[idx:i]
            s, x = num[idx:i], int(num[idx:i])  # s could '0'
            if last is None: dfs(i, s, x, x)
            else:
                dfs(i, expr + "+" + s, cur + x, x)
                dfs(i, expr + "-" + s, cur - x, -x)
                # This is to handle 1 + 2 * 3, we need to backout 2 and add 2 * 3.
                dfs(i, expr + "*" + s, cur - last + last*x, last*x)
            if num[idx] == '0': break  # after idx+1 we break out otherwise we have 05
    dfs(0, '', 0, None)
    return res

# LC227. Basic Calculator II  # only +-*/, no parenth
def calculate(self, s: str) -> int:  # O(n) runtime but O(1) space,
    val = res = 0
    n, i, op = len(s), 0, '+'
    while i < n:
        if s[i].isdigit():
            num = 0
            while i < n and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            if op == '+': val = num  # previous operator
            elif op == "-": val = -num
            elif op == "*": val *= num  # it's a factor, not a term
            elif op == "/": val = int(val / num)
        elif s[i] in '+-*/':
            if s[i] in '+-':  # it's a term, not a factor for */
                res += val
                val = 0
            op = s[i]
            i += 1
        else: i += 1  # skip space
    return res + val
def calculate(self, s: str) -> int:  # O(n) runtime and space
    num, op = 0, "+"
    stack = []
    for i, c in enumerate(s):
        if c.isdigit(): num = num * 10 + int(c)
        if c in "+-*/" or i == len(s) - 1:
            if op == "+": stack.append(num) # previous operation, not current
            elif op == "-": stack.append(-num)
            elif op == "*": stack.append(stack.pop() * num)
            else: stack.append(int(stack.pop() / num))  # use int for negative
            num, op = 0, c
    return sum(stack)

# LC224. Basic Calculator - with +-()
def calculate(self, s):  # O(n) time and space
    res, stack = 0, []
    num, sign = 0, 1
    for ss in s:
        if ss.isdigit():
            num = 10 * num + int(ss)
        elif ss in ["-", "+"]:
            res += sign * num  # finish previous term
            num = 0
            sign = 1 if ss == '+' else -1
        elif ss == "(":
            stack.append(res)
            stack.append(sign)
            sign, res = 1, 0
        elif ss == ")":
            res += sign * num   # expr inside ')'
            res *= stack.pop()  # old sign before '(', -(3) shows we can't combine res and sign first
            res += stack.pop()  # old res before '('
            num = 0
    return res + num * sign



# LC494. Target Sum - with plus minus +- operators - ints expression built for target  add + - sign
def findTargetSumWays(self, nums: List[int], S: int) -> int:
    n = len(nums)
    @lru_cache(None)  # O(n * S)
    def dp(i, s):  # index and sum, how many ways to compose a[0], ..., a[i-1] to have sum s.
        if i == n: return s == S  # 1 or 0
        add = dp(i+1, s + nums[i])
        sub = dp(i+1, s - nums[i])
        return add + sub
    return dp(0, 0)
def findTargetSumWays(self, nums: List[int], target: int) -> int:
    count = collections.Counter({0: 1}) # Iterative, DP
    for x in nums:
        step = collections.Counter()
        for y in count:
            step[y + x] += count[y]
            step[y - x] += count[y]
        count = step
    return count[target]

# LC2019. The Score of Students Solving Math Expression
def scoreOfStudents(self, s: str, answers: List[int]) -> int:  # O(n^3 * ???)
    @functools.lru_cache(None)
    def dp(i, j):  # all possible result for the substring from s[i] to s[j], O(n^2)
        if i == j: return {int(s[i])}  # base case
        res = {}
        for m in range(i + 1, j, 2):  # O(n) ways to break substrings
            for a in dp(i, m - 1):  # truncate to 1000
                for b in dp(m + 1, j): # truncate to 1000
                    cur = a * b if s[m] == '*' else a + b
                    if cur <= 1000:  # opt with 0 <= answers[i] <= 1000
                        res[cur] = 2
        return res # truncate to 1000
    res = {**dp(0, len(s) - 1), **{eval(s): 5}}
    return sum(res.get(a, 0) for a in answers)

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
