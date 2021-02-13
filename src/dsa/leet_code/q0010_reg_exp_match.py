################################################################################
# Given an input string (s) and a pattern (p), implement regular expression
# matching with support for '.' and '*' where:
#     '.' Matches any single character.
#     '*' Matches zero or more of the preceding element.
# The matching should cover the *entire* input string (not partial).
# HARD
################################################################################

def reg_exp_match(string: str, pattern: str) -> bool:
    p_idx = 0
    s_prev = None
    p_prev = None
    for idx, c in enumerate(string):
        if p_idx >= len(pattern):
            return False

        if pattern[p_idx] == '.':
            p_prev = '.'
            p_idx += 1
            continue
        elif pattern[p_idx] == '*':
            if p_prev == '.':
                continue
            else:  # a*b match b, ab, aab
                if c != s_prev:
                    p_idx += 1
                    if pattern[p_idx] != '.' and pattern[p_idx] != c:
                        return False
                    p_idx += 1
                    if p_idx < len(pattern):
                        p_prev = pattern[p_idx]
        else:  # other chars
            while True:
                if pattern[p_idx] != c:
                    if p_idx+1 >= len(pattern) or pattern[p_idx+1] != '*':
                        return False
                    else:
                        p_idx += 2
                        p_prev = pattern[p_idx-1]
                        if p_idx >= len(pattern):  # nothing after to compare to c
                            break
                else:
                    break

            if p_idx < len(pattern):
                p_prev = pattern[p_idx]
            p_idx += 1

        s_prev = c

    if p_idx < len(pattern)-1:
        if pattern[p_idx] != '*':
            return False

    return True


# print(reg_exp_match('aa', 'a'))  # False
# print(reg_exp_match('aa', 'a*'))  # True
# print(reg_exp_match('abcd', 'd*'))  # False
# print(reg_exp_match('ab', '.*'))  # True
# print(reg_exp_match('ab', '.*c'))  # False
# print(reg_exp_match('aab', 'c*a*b'))  # True
# print(reg_exp_match('aaa', 'aaaa'))  # False
# print(reg_exp_match('aaa', 'aaa'))  # True
print(reg_exp_match('aaa', 'a*a'))  # True
print(reg_exp_match('aaaaaa', 'a*aaab'))  # False??
print(reg_exp_match('aaaaaa', 'a*aaa'))  # True
print(reg_exp_match('aaaaaa', '.*aaab'))  # False
print(reg_exp_match('aaaaaa', '.*aaa'))  # True
print(reg_exp_match('mississippi', 'mis*is*p*.'))  # False
print(reg_exp_match('mississippi', 'mis*is*ip*.'))  # True
