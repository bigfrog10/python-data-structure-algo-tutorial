# LC140. Word Break II
def wordBreak(s: str, wordDict: List[str]) -> List[str]:
    word_set = set(wordDict)  # return all possible answer
    def dfs(s):
        output = []
        if s in word_set: output.append(s)  # one of solutions
        for w in word_set:
            if s.startswith(w):
                tmp = dfs(s[len(w):])
                for x in tmp: output.append(' '.join([w] + [x]))
        return output
    return dfs(s)

# LC139. Word Break, top100
def wordBreak(self, s, words):
    max_len = max(map(len,words))  # branch cut
    words = set(words)
    ok = [True]  # cache
    for i in range(1, len(s)+1): # comma change this to a tuple
        ok += any(ok[j] and s[j:i] in words for j in range(max(0, i-max_len), i)),
    return ok[-1]

def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    wds = set(wordDict)
    @lru_cache(None)
    def dfs(s):
        if s in wds: return True
        for w in wordDict:
            if s.startswith(w) and dfs(s[len(w):]): return True
        return False
    return dfs(s)

# LC127. Word Ladder, top100
def ladderLength(self, beginWord, endWord, wordList):
    wordList = set(wordList)
    queue = collections.deque([[beginWord, 1]])

    while queue:  # BFS since we look for shortest
        word, length = queue.popleft()
        if word == endWord: return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word in wordList:
                    wordList.remove(next_word) # this is the key, BFS doesn't look back
                    queue.append([next_word, length + 1])
    return 0

# LC126. Word Ladder II, top100
def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    wd = set(wordList)
    if endWord not in wd: return []
    chars = 'abcdefghijklmnopqrstuvwxyz'
    found = False
    visited = {beginWord}
    queue = deque([[beginWord]])
    res = []
    while queue and not found:# a neat trick to go through this level
        lv = set()  # visited at this level
        for _ in range(len(queue)):
            w = queue.popleft()
            word = w[-1]
            for i in range(len(word)):
                for c in chars:
                    nw = word[:i] + c + word[i+1:]
                    if nw == endWord:
                        res.append(w + [nw])
                        found = True
                    elif nw in wd and nw not in visited:
                        lv.add(nw)
                        queue.append(w + [nw])
        visited.update(lv)
    return res

# LC720. Longest Word in Dictionary
def longestWord(self, words: List[str]) -> str:
    wds = set(words) # O(n)
    handled = {}
    ret, maxl = '', 0
    for wd in words:
        changeable = True
        if wd not in handled: # make sure all prefixes are in
            for i in range(len(wd)):
                handled[wd[:i+1]] = wd[:i+1] in wds
                if not handled[wd[:i+1]]:
                    changeable = False
                    break
        else: changeable = handled[wd]
        if changeable:
            if len(wd) > maxl or len(wd) == maxl and wd < ret:
                ret = wd
                maxl = len(wd)
    return ret

# LC472. Concatenated Words
def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    word_set = set(words)
    def check(word):
        n = len(word)
        for i in range(1, n):
            if word[i:] not in word_set: continue
            # so now word[i:] in word_set
            if word[:i] in word_set: return True # so both part are words
            if check(word[:i]): return True # recursion check
        return False
    res = []
    for w in words:
        if check(w): res.append(w)
    return res
