# DP
def reg_exp_match(string: str, pattern: str) -> bool:
    memo = {}

    def dp(i, j):
        if (i, j) not in memo:
            if j == len(pattern):
                ans = i == len(string)
            else:
                first_match = i < len(string) and pattern[j] in {string[i], '.'}
                if j+1 < len(pattern) and pattern[j+1] == '*':
                    ans = dp(i, j+2) or first_match and dp(i+1, j)
                else:
                    ans = first_match and dp(i+1, j+1)

            memo[i, j] = ans
        return memo[i, j]

    return dp(0, 0)


print(reg_exp_match('aa', 'a'))  # False
print(reg_exp_match('aa', 'a*'))  # True
print(reg_exp_match('abcd', 'd*'))  # False
print(reg_exp_match('ab', '.*'))  # True
print(reg_exp_match('ab', '.*c'))  # False
print(reg_exp_match('aab', 'c*a*b'))  # True
print(reg_exp_match('aaa', 'aaaa'))  # False
print(reg_exp_match('aaa', 'aaa'))  # True

print(reg_exp_match('aaa', 'a*a'))  # True
print(reg_exp_match('aaaaaa', 'a*aaab'))  # False??
print(reg_exp_match('aaaaaa', 'a*aaa'))  # True
print(reg_exp_match('aaaaaa', '.*aaab'))  # False
print(reg_exp_match('aaaaaa', '.*aaa'))  # True
print(reg_exp_match('mississippi', 'mis*is*p*.'))  # False
print(reg_exp_match('mississippi', 'mis*is*ip*.'))  # True
