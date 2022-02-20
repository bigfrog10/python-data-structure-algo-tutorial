
# LC316. Remove Duplicate Letters
def removeDuplicateLetters(self, s: str) -> str:
    last_idx = {c: i for i, c in enumerate(s)}
    stack, seen = [], set()
    for i, c in enumerate(s): # O(n)
        if c not in seen:  # keep only one inside
            # if stack's char is larger than current and it's not the last
            # we drop this char and wait for the last
            while stack and c < stack[-1] and i < last_idx[stack[-1]]:
                seen.discard(stack.pop()) # increasing stack
            seen.add(c)
            stack.append(c)
    return ''.join(stack)

# LC1081. Smallest Subsequence of Distinct Characters
def removeDuplicateLetters(self, s: str) -> str:
    last_occurrence = {c: i for i, c in enumerate(s)}
    stack, seen = [], set()  # O(n)
    for i, c in enumerate(s):
        if c not in seen:  # keep only one inside
            # if stack's char is larger than current and it's not the last
            # we drop this char and wait for the last
            while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
                seen.discard(stack.pop())  # increasing stack
            seen.add(c)
            stack.append(c)
    return ''.join(stack)

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
