
# LC676. Implement Magic Dictionary
class MagicDictionary:
    def __init__(self):
        self.trie = {}
    def buildDict(self, dictionary: List[str]) -> None:
        for word in dictionary:
            node = self.trie
            for letter in word: node = node.setdefault(letter, {})
            node[None] = None
    def search(self, word: str) -> bool:
        def find(node, i, mistakeAllowed):
            if i == len(word):
                return None in node and not mistakeAllowed
            if word[i] not in node:
                return any(find(node[letter], i+1, False) for letter in node if letter) if mistakeAllowed else False
            if mistakeAllowed:
                return find(node[word[i]], i+1, True) or any(find(node[letter], i+1, False) for letter in node if letter and letter != word[i])
            return find(node[word[i]], i+1, False)
        return find(self.trie, 0, True)

# LC139. Word Break, top100
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    wds = set(wordDict)
    @lru_cache(None)
    def dfs(s):
        if s in wds: return True
        for w in wordDict:
            if s.startswith(w) and dfs(s[len(w):]): return True
        return False
    return dfs(s)
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    @lru_cache # O(n^3) n = len(s) + O(m), m is size of dict
    def wordBreakMemo(s: str, start: int):
        if start == len(s): return True
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in wordDict and wordBreakMemo(s, end): return True
        return False
    return wordBreakMemo(s, 0)

# LC140. Word Break II
def wordBreak(s: str, wordDict: List[str]) -> List[str]:
    word_set = set(wordDict)  # return all possible answer
    def dfs(s):
        output = []
        if s in word_set: output.append(s)  # one of solutions
        for w in word_set: # or we loop s with prefix
            if s.startswith(w):
                tmp = dfs(s[len(w):])
                for x in tmp: output.append(' '.join([w] + [x]))
        return output
    return dfs(s)

# LC68. Text Justification
def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    ans = []
    line, width = [], 0
    for word in words:
        if width + len(line) + len(word) > maxWidth:  # len(line) spaces
            n, k = divmod(maxWidth - width, max(1, len(line)-1))
            for i in range(max(1, len(line)-1)):
                line[i] += " " * (n + (i < k)) # add space after word
            ans.append("".join(line))
            line, width = [], 0
        line.append(word)
        width += len(word)
    ans.append(" ".join(line).ljust(maxWidth))
    return ans

# LC79. Word Search, top100 - search in 2d matrix
def exist(self, board: List[List[str]], word: str) -> bool:
    if not board or not board[0]: return False
    h, w, wl = len(board), len(board[0]), len(word)
    def dfs(i, j, wi):
        if board[i][j] != word[wi]: return False
        board[i][j] = ord(board[i][j]) ^ 256
        exist = wi+1 == wl
        for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if 0 <= x < h and 0 <= y < w: exist = exist or dfs(x, y, wi+1)
        board[i][j] = chr(board[i][j] ^ 256) # backout
        return exist
    bls = set() # precheck board has all letters from word,
    for row in board: bls.update(row) # this makes the test faster 5% -> 74%
    wls = set(word)
    if len(wls - bls) > 0: return False
    for i in range(h):
        for j in range(w):
            if dfs(i, j, 0): return True
    return False

# LC212. Word Search II, top100
def findWords(self, board: List[List[str]], words: List[str]) -> List[str]: # This is fast
    trie, WORD_KEY = {}, '$'
    for word in words:
        node = trie
        for letter in word: node = node.setdefault(letter, {})
        node[WORD_KEY] = word  # save word at the end
    rowNum, colNum = len(board), len(board[0])
    matchedWords = []
    def dfs(row, col, parent):
        letter = board[row][col]
        currNode = parent[letter]
        word_match = currNode.pop(WORD_KEY, None)  # check end, cut branches
        if word_match: matchedWords.append(word_match)
        board[row][col] = '#' # Before the EXPLORATION, mark the cell as visited
        # Explore the neighbors in 4 directions, i.e. up, right, down, left
        for (dx, dy) in (-1, 0), (0, 1), (1, 0), (0, -1): # O(3^max(words))
            nx, ny = row + dx, col + dy
            if nx < 0 or nx >= rowNum or ny < 0 or ny >= colNum: continue
            if not board[nx][ny] in currNode: continue
            dfs(nx, ny, currNode)
        board[row][col] = letter # End of EXPLORATION, we restore the cell
        # Optimization: incrementally remove the matched leaf node in Trie.
        if not currNode: parent.pop(letter) # we pop'd WORD_KEY before
    for row in range(rowNum): # O(nm)
        for col in range(colNum):# starting from each of the cells
            if board[row][col] in trie: dfs(row, col, trie)
    return matchedWords

# LC127. Word Ladder, top100
def ladderLength(self, beginWord, endWord, wordList):  # BFS
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

# LC126. Word Ladder II, top100 - Same for find all paths in DAG
def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    wd = set(wordList) # BFS
    if endWord not in wd: return []
    chars = 'abcdefghijklmnopqrstuvwxyz'
    found = False # once we found one, stop this level for shortest ladders.
    visited = {beginWord}
    queue = deque([[beginWord]]) # queue on ladder/path
    res = []
    while queue and not found:# a neat trick to go through this level
        lv = set()  # visited at this level
        for _ in range(len(queue)):
            w = queue.popleft()
            word = w[-1] # get last word in ladder and go forward
            for i in range(len(word)):
                for c in chars:
                    nw = word[:i] + c + word[i+1:]
                    if nw == endWord:
                        res.append(w + [nw])
                        found = True # keep on this level to find all others.
                    elif nw in wd and nw not in visited:
                        lv.add(nw)
                        queue.append(w + [nw])
        visited.update(lv)
    return res

# LC609. Find Duplicate File in System
def findDuplicate(self, paths: List[str]) -> List[List[str]]:
    groups = defaultdict(list)
    for s in paths:
        segs = s.split(' ')
        p = segs[0]
        for i in range(1, len(segs)):
            seg = segs[i]
            psegs = seg.split('(')
            c = psegs[1] # [:-1]  # remove ) at the end
            groups[c].append(p + '/' + psegs[0])
    ret = [gl for gl in groups.values() if len(gl) > 1]
    return ret

# LC14. Longest Common Prefix
def longestCommonPrefix(self, strs):
    if not strs: return ""
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch: return shortest[:i]
    return shortest

# LC1268. Search Suggestions System
def suggestedProducts(self, A, word):
    A.sort()
    res, prefix, i = [], '', 0
    for c in word:
        prefix += c
        i = bisect.bisect_left(A, prefix, i)
        res.append([w for w in A[i:i + 3] if w.startswith(prefix)])
    return res



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

# LC524. Longest Word in Dictionary through Deleting - not only counts, but also order
def findLongestWord(self, s, d):
    for x in sorted(d, key=lambda x: (-len(x), x)):
        it = iter(s) # call 'in' moves to next char
        if all(c in it for c in x):
            return x
    return ''





# LC734. Sentence Similarity
def areSentencesSimilar(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
    if len(sentence1) != len(sentence2): return False
    sets = set((a, b) for a, b in similarPairs)
    for w, v in zip(sentence1, sentence2):
        if w != v and (w,v) not in sets and (v,w) not in sets: return False
    return True

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

# LC1487. Making File Names Unique
def getFolderNames(self, names: List[str]) -> List[str]:
    used = set()
    counter = defaultdict(int)
    result = []
    for name in names:
        count = counter[name]
        candidate = name
        while candidate in used:
            count += 1
            candidate = f'{name}({count})'
        counter[name] = count
        result.append(candidate)
        used.add(candidate)
    return result

# LC1048. Longest String Chain
def longestStrChain(self, words: List[str]) -> int:
    word_set = set(words)
    @lru_cache(None)
    def dfs(word): # O(N) or O(L^2 * N) where L is max len of words
        t_max =1
        for i in range(len(word)):
            tmp = word[:i]+word[i+1:]
            if tmp in word_set: t_max = max(t_max, dfs(tmp)+1)
        return t_max
    result = 0
    for word in word_set: result = max(dfs(word),result)
    return result
