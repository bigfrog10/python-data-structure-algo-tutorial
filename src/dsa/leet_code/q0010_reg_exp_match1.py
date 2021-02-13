
def reg_exp_match(string: str, pattern: str) -> bool:
        if not pattern:
            return not string

        first_match = bool(string) and pattern[0] in {string[0], '.'}

        if len(pattern) >= 2 and pattern[1] == '*':
            return (reg_exp_match(string, pattern[2:]) or
                    first_match and reg_exp_match(string[1:], pattern))
        else:
            return first_match and reg_exp_match(string[1:], pattern[1:])


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
