



# https://leetcode.com/discuss/interview-question/380650/Bloomberg-or-Phone-Screen-or-Candy-Crush-1D
# Candy Crush 1D
# "aaabbbc" -> c, "aabbbacd" -> cd, "aabbccddeeedcba" -> ""
def remove_dupes(s):
    stack = [['#', 0]]
    for c in s:
        if stack[-1][0] == c: stack[-1][1] += 1
        else:
            if stack[-1][1] >= 3: stack.pop()
            if stack[-1][0] == c: stack[-1][1] += 1 # check prev prev after pop
            else: stack.append([c,1])
    return ''.join(c * k for c, k in stack)
