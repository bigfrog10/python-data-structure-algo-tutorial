################################################################################
# Given a non-empty string s and a dictionary wordDict containing a list of
# non-empty words, determine if s can be segmented into a space-separated
# sequence of one or more dictionary words.
# Note:
#     - The same word in the dictionary may be reused multiple times in the
#       segmentation.
#     - You may assume the dictionary does not contain duplicate words.
#
################################################################################
# working but slow, so add cache
# Complexity is T(s) = wT(s-1) + s
# w is how many words in dict, s is the size of string.
# +s is because startswith() call
cached = {}  # need to clear this between calls


def word_breakable(s: str, words: list) -> bool:
    if not s:
        return True

    if s in cached:
        return False

    for w in words:
        if s.startswith(w):
            s1 = s[len(w):]
            if s1 not in cached:
                is_breakable = word_breakable(s1, words)
                if is_breakable:
                    return True

                cached[s1] = is_breakable

    cached[s] = False
    return False


def word_breakable1(s: str, words: list) -> bool:
    pass


print(word_breakable('leetcode', ["leet", "code"]))
print(word_breakable('applepenapple', ["apple", "pen"]))
print(word_breakable('catsandog', ["cats", "dog", "sand", "and", "cat"]))
print(word_breakable('applepie', ["pie", "pear", "apple", "peach"]))
print(word_breakable('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab',
                     ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]))

