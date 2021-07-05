
# LC316. Remove Duplicate Letters
# LC1081. Smallest Subsequence of Distinct Characters
def removeDuplicateLetters(self, s: str) -> str:
    last_occurrence = {c: i for i, c in enumerate(s)}
    stack, seen = [], set() # O(n)
    for i, c in enumerate(s):
        if c not in seen:  # keep only one inside
            # if stack's char is larger than current and it's not the last
            # we drop this char and wait for the last
            while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
                seen.discard(stack.pop()) # increasing stack
            seen.add(c)
            stack.append(c)
    return ''.join(stack)

# LC1047. Remove All Adjacent Duplicates In String
def removeDuplicates(self, S: str) -> str: # O(n)
    output = []
    for ch in S:
        if output and ch == output[-1]: output.pop()
        else: output.append(ch)
    return ''.join(output)

# LC1209. Remove All Adjacent Duplicates in String II
def removeDuplicates(self, s, k):  # O(n)
    stack = [['#', 0]] # 0 for ignoring when joining at the last
    for c in s:
        if stack[-1][0] == c:
            stack[-1][1] += 1
            if stack[-1][1] == k: stack.pop() # remove this group
        else: stack.append([c, 1])
    return ''.join(c * cnt for c, cnt in stack)

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
