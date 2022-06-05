
# LC1249. Minimum Remove to Make Valid Parentheses - with letters, return one string result
def minRemoveToMakeValid(self, s: str) -> str:  # O(n) runtime and space
    stack, remove = [], []  # find all indices to remove
    for i, c in enumerate(s):
        if c not in '()': continue  # others, keep it
        elif c == '(': stack.append(i)  # record it
        # now c == ')'
        elif not stack: remove.append(i)  # mo matching ( for ), add to remove
        else: stack.pop()  # matched, remove old record
    remove.extend(stack)  # whatever left is no match
    idx, ret = 0, ''
    for i in remove:  # remove those
        ret += s[idx:i]  # up to removed
        idx = i + 1  # skip removed
    ret += s[idx:]  # add leftover
    return ret

# LC921. Minimum Add to Make Parentheses Valid - return num of operations
def minAddToMakeValid(self, S: str) -> int:  # O(n)
    left = bal = 0  # bal = # of '(' - # of ')'
    for s in S:
        bal += 1 if s == '(' else -1  # cancel out balanced
        if bal == -1:  # there is ) unmatched
            left += 1  # expected ( needs 1 more
            bal += 1  # diff needs +1 too
    return left + bal  # "(((" -> left=0, bal=3

# LC301. Remove Invalid Parentheses - with letters, return all results
def removeInvalidParentheses(self, s): # O(2^n) return all possible results
    def isvalid(s):  # O(n)
        ctr = 0
        for c in s:
            if c == '(': ctr += 1
            elif c == ')': ctr -= 1
            if ctr < 0: return False
        return ctr == 0
    level = {s}
    while True:  # O(2^(n-1)), no need to check single char sets.
        valid = list(filter(isvalid, level))
        if valid: return valid  # we want min num of removals
        level = {s[:i] + s[i+1:] for s in level for i in range(len(s)) if s[i] in '()'}

# LC1614. Maximum Nesting Depth of the Parentheses
def maxDepth(self, s: str) -> int:
    # https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/discuss/888949/JavaC%2B%2BPython-Parentheses-Problem-Foundation
    res = cur = 0
    for c in s:
        if c == '(':
            cur += 1
            res = max(res, cur)
        if c == ')': cur -= 1
    return res

# LC22. Generate Parentheses
def generateParenthesis(self, n: int) -> List[str]:
    ret = {'()'}
    for _ in range(n-1):
        ret = {s[:i] + '()' + s[i:] for s in ret for i in range(len(s))}
    return ret

# LC1541. Minimum Insertions to Balance a Parentheses String - unbalanced ( to ))
def minInsertions(self, s: str) -> int:
    open_missing = close_missing = close_needed = 0  # miss (, miss ), and miss ))
    for c in s:
        if c == '(':
            if close_needed % 2:  # if we have odd ) in the last step
                close_missing += 1  # for previous invalid, have to add one closing bracket to make it valid
                close_needed -= 1  # no need to track the last invalid case
            close_needed += 2
        else:  # it's )
            close_needed -= 1
            if close_needed < 0: # previously we have ")))"
                open_missing += 1  # so increment this for ( to balance )))
                close_needed += 2  # no need to track the last invalid
    return open_missing + close_missing + close_needed

# LC20. Valid Parentheses - {} () []
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

# LC1963. Minimum Number of Swaps to Make the String Balanced - parenthesis '[] balance'
def minSwaps(self, s: str) -> int:
    balance = max_bal = 0
    for c in s:
        balance += -1 if c == '[' else 1
        max_bal = max(max_bal, balance)
    return (max_bal + 1) // 2  # ceiling

# LC856. Score of Parentheses
def scoreOfParentheses(self, S: str) -> int:
    ans = bal = 0
    for i, x in enumerate(s):
        if x == '(': bal += 1
        else:  # )
            bal -= 1
            if s[i-1] == '(': ans += 1 << bal # only when seeing ()
    return ans

# LC241. Different Ways to Add Parentheses
def diffWaysToCompute(self, expression: str) -> List[int]:
    # runtime is: http://people.math.sc.edu/howard/Classes/554b/catalan.pdf
    # runtime is C_(n-1) = (select n-1 from 2(n-1)) / n, n = len(expr)
    # pn = sum(p_i * p_(n-i)) for i in 1 to n-1
    @lru_cache(None)
    def diff_ways(expr: str):
        res = []
        if expr.isdigit(): res.append(int(expr))  # base case
        for i, v in enumerate(expr):
            if v in '+-*':
                pre = diff_ways(expr[0: i])
                post = diff_ways(expr[i + 1:])
                res.extend(a+b if v=='+' else a-b if v=='-' else a*b for a in pre for b in post)
        return res
    return diff_ways(expression)

# LC32. Longest Valid Parentheses - longest substring
def longestValidParentheses(self, s: str) -> int:  # O(n) time and space
    stack, longest = [0], 0  # track current length and its max
    for c in s:
        if c == "(": stack.append(0)
        else:  # )
            if len(stack) > 1:
                val = stack.pop()
                stack[-1] += val + 2
                longest = max(longest, stack[-1])
            else: stack = [0]  # get unmatched ), restart
    return longest
def longestValidParentheses(self, s: str) -> int:  # O(n) time and O(1) space
    n = len(s)
    maxl = left = right = 0
    for c in s:  # from left to right
        if c == '(': left += 1
        else: right += 1
        if left == right: maxl = max(maxl, 2 * right)
        elif left < right: left = right = 0  # start over with new
    left = right = 0
    for c in s[::-1]:
        if c == '(': left += 1
        else: right += 1
        if left == right: maxl = max(maxl, 2 * left)
        elif left >= right: left = right = 0
    return maxl

# LC678. Valid Parenthesis String - paretnh with *, par with *, par *
def checkValidString(self, s):  # greedy
    cmin = cmax = 0  # smallest and largest possible number of (, or how many ) expected
    for i in s:
        cmax += - 1 if i == ")" else cmax + 1  # treat * as (
        cmin = cmin + 1 if i == '(' else max(cmin - 1, 0)  # treat * as )
        if cmax < 0: return False  # too many (
    return cmin == 0
