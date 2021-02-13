################################################################################
# Given a non-empty string s and a dictionary wordDict containing a list of
# non-empty words, add spaces in s to construct a sentence where each word is a
# valid dictionary word. Return all such possible sentences.
# Note:
#     - The same word in the dictionary may be reused multiple times in the
#       segmentation.
#     - You may assume the dictionary does not contain duplicate words.
# HARD
################################################################################
cached = {}  # string -> list of list of words broken from string, many break ways.


def break_by_word(s: str, words: list) -> list:
    if not s:
        return []

    if s in cached:
        return cached[s]

    for w in words:
        if s.startswith(w):
            s1 = s[len(w):]
            if s1 == '':
                cached[s] = [[s]]
                continue

            if s1 not in cached:
                break_by_word(s1, words)

            wds_list = cached[s1]
            wds_list1 = cached[s] if s in cached else []
            for wds in wds_list:
                wds1 = wds.copy()
                wds1.insert(0, w)
                wds_list1.append(wds1)
            cached[s] = wds_list1  # in case it's init from []

    if s not in cached:
        cached[s] = []

    return cached[s]


print(break_by_word('catsanddog', ["cat", "cats", "and", "sand", "dog"]))
cached = {}
print(break_by_word('pineapplepenapple', ["apple", "pen", "applepen", "pine", "pineapple"]))
cached = {}
print(break_by_word('catsandog', ["cats", "dog", "sand", "and", "cat"]))
