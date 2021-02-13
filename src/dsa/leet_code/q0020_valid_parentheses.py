################################################################################
# Given a string s containing just the characters '(', ')', '{', '}', '[' and
# ']', determine if the input string is valid.
# An input string is valid if:
#     Open brackets must be closed by the same type of brackets.
#     Open brackets must be closed in the correct order.
# EASY
################################################################################

def check_match(s: str) -> bool:  # s is a lit of parentheses
    # string is a list, which is a stack
    stack = []  # to store unprocessed
    PAIRS = {'(': ')', '{': '}', '[': ']'}

    for idx, c in enumerate(s):
        if c in PAIRS.keys():  # open parentheses
            stack.append(c)
        else:  # closed parentheses
            if len(stack) == 0:  # empty
                return False

            if PAIRS[stack[-1]] == c:  # matched
                stack.pop()
            else:
                return False

    return len(stack) == 0


print(check_match('()'))  # true
print(check_match('()[]{}'))  # true
print(check_match('(]'))  # false
print(check_match('([)]'))  # false
print(check_match('{[]}'))  # true
