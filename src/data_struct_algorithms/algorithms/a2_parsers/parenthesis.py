# LC1249. Minimum Remove to Make Valid Parentheses
def minRemoveToMakeValid(self, s: str) -> str: # return one modified string
    stack, remove = [], [] # find all indices to remove
    for i, c in enumerate(s):
        if c not in '()': continue  # others, keep it
        elif c == '(': stack.append(i)  # record it
        elif not stack: remove.append(i)  # mo match ( for ), add to remove
        else: stack.pop()  # matched, remove old record
    remove.extend(stack)  # whatever left is no match
    idx, ret = 0, ''
    for i in remove:  # remove those
        ret += s[idx:i]  # up to removed
        idx = i + 1  # skip removed
    ret += s[idx:]  # add leftover
    return ret

# 301. Remove Invalid Parentheses
def removeInvalidParentheses(self, s): # return all possible results
    def isvalid(s):
        ctr = 0
        for c in s:
            if c == '(': ctr += 1
            elif c == ')': ctr -= 1
            if ctr < 0: return False
        return ctr == 0
    level = {s}
    while True:
        valid = list(filter(isvalid, level))
        if valid: return valid
        level = {s[:i] + s[i+1:] for s in level for i in range(len(s))}

# LC921. Minimum Add to Make Parentheses Valid
def minAddToMakeValid(self, S: str) -> int:
    left = bal = 0 # bal = num of ( minus mn of )
    for s in S:
        bal += 1 if s == '(' else -1 # cancel out balanced
        if bal == -1: # there is ) unmatched
            left += 1 # expected ( needs 1 more
            bal += 1  # diff needs +1 too
    return left + bal # "(((" -> ret=0, bal=3

# LC678. Valid Parenthesis String - with *
def checkValidString(self, s):  # greedy
    lo = hi = 0  # smallest and largest possible number of open left brackets
    for c in s:
        lo += 1 if c == '(' else -1
        hi += 1 if c != ')' else -1
        if hi < 0: break  # too many )
        lo = max(lo, 0) # if low is minus, replace it with 0
    return lo == 0

# LC20. Valid Parentheses, top100
def isValid(self, s: str) -> bool:
    PAIRS = {'(': ')', '{': '}', '[': ']'}
    stack = []  # to store unprocessed
    for idx, c in enumerate(s):
        if c in PAIRS.keys(): stack.append(c) # open parentheses
        else:  # closed parentheses
            if len(stack) == 0: return False # ) has no (
            if PAIRS[stack[-1]] == c: stack.pop() # matched
            else: return False # no suppose to have other chars
    return len(stack) == 0

# LC32. Longest Valid Parentheses
def longestValidParentheses(self, s):
    stack, longest = [0], 0 # track current length and its max
    for c in s:
        if c == "(": stack.append(0)
        else: # )
            if len(stack) > 1:
                val = stack.pop()
                stack[-1] += val + 2
                longest = max(longest, stack[-1])
            else: stack = [0] # get unmatched ), restart
    return longest
def longestValidParentheses(self, s):
    l, r = 0, 0
    max_ = 0
    for i in range(len(s)):
        if s[i] == '(': l +=1
        else: r +=1
        if l == r: max_ = max(max_, 2 * r)
        else:
            if r >= l: l = r = 0
    l,r = 0,0
    for i in reversed(range(len(s))):
        if s[i] == '(': l += 1
        else: r += 1
        if l == r: max_ = max(max_, 2 * l)
        else:
            if l >= r: l = r = 0
    return max_

# LC22. Generate Parentheses
def generateParenthesis(self, n: int) -> List[str]:
    def generate_more(p: set):  # insert () into everywhere
        ret = set()
        for p1 in p:
            for i in range(len(p1)): ret.add(p1[:i] + '()' + p1[i:])
        return ret
    ret = {'()'}
    for _ in range(n-1): ret = generate_more(ret)
    return ret
