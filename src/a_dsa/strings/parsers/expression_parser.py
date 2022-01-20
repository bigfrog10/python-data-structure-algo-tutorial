import queue
import re

# https://blog.csdn.net/iteye_15612/article/details/81725858
# https://www.chris-j.co.uk/parsing.php
# https://www.cs.cmu.edu/~adamchik/15-121/lectures/Stacks%20and%20Queues/Stacks%20and%20Queues.html
# Read in the tokens one at a time
# If a token is an integer, write it into the output
# If a token is an operator, push it to the stack, if the stack is empty. If
#     the stack is not empty, you pop entries with higher or equal priority and
#     only then you push that token to the stack.
# If a token is a left parentheses '(', push it to the stack
# If a token is a right parentheses ')', you pop entries until you meet '('.
# When you finish reading the string, you pop up all tokens which are left there.
# Arithmetic precedence is in increasing order: '+', '-', '*', '/';

number_or_symbol = re.compile(r'(\d+|[^ 0-9])')


def eval_str_expr(str_expr: str):
    tokens = re.findall(number_or_symbol, str_expr)
    expr_queue = _process_tokens(tokens)
    return _eval_postfix(expr_queue)


def _process_tokens(tokens):
    stack = []
    output_queue = queue.Queue()

    for token in tokens:
        if token.isnumeric():
            output_queue.put(token)
        elif token in '+-*/^':  # operators
            while stack:
                item = stack.pop()
                if item in '+-*/^' and _same_or_higher_priority(item, token):
                    output_queue.put(item)
                else:  # put it back into stack
                    stack.append(item)
                    break

            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            # t := stack.pop() <- walrus operator in python 3.8
            t = stack.pop()
            while t != '(':
                output_queue.put(t)
                t = stack.pop()
    # else such as ' ' will be ignored

    while stack:
        output_queue.put(stack.pop())

    return output_queue


def _same_or_higher_priority(op1, op2):  # op1 is same or higher order than op2
    if op2 in '+-':
        return True
    elif op2 in '*/':
        return op1 in '*/^'
    else:
        return op2 == '^' and op1 == '^'


def _eval_postfix(postfix_queue):
    operand_stack = []
    while not postfix_queue.empty():
        item = postfix_queue.get()
        if item.isnumeric():
            operand_stack.append(int(item))
        elif item in '+-*/^':
            first = operand_stack.pop()
            second = operand_stack.pop()

            if item == '+':
                operand_stack.append(second + first)
            elif item == '-':
                operand_stack.append(second - first)
            elif item == '*':
                operand_stack.append(second * first)
            elif item == '/':
                operand_stack.append(second / first)
            elif item == '^':
                operand_stack.append(second ** first)

    return operand_stack.pop()


# print(eval_str_expr('(81 * 6) / 42 + (3 - 1)'))
# print(eval('(81 * 6) / 42 + (3 - 1)'))
# print(eval_str_expr('3+2*2'))
# print(eval_str_expr('3-(2-(1+2))'))
# print(eval_str_expr('3+2*2'))
print(eval_str_expr('3*2+2'))

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
