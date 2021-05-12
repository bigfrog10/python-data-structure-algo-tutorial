# LC14. Longest Common Prefix
def longestCommonPrefix(self, strs):
    if not strs: return ""
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch: return shortest[:i]
    return shortest

# LC249. Group Shifted Strings
def groupStrings(self, strings: List[str]) -> List[List[str]]:
    def shash(s):
        if not s: return -1
        if len(s) == 1: return 1 # single char string can always be shifted
        d = ord(s[0]) - ord('a') # d is the shift
        ret = []
        for c in s:
            h = (ord(c) - d) % 26 # now shift all chars by the same
            ret.append(h)
        return tuple(ret)

    groups = defaultdict(list)
    for s in strings: groups[shash(s)].append(s)
    return groups.values()

# LC734. Sentence Similarity
def areSentencesSimilar(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
    if len(sentence1) != len(sentence2): return False
    sets = set((a, b) for a, b in similarPairs)
    for w, v in zip(sentence1, sentence2):
        if w != v and (w,v) not in sets and (v,w) not in sets: return False
    return True
