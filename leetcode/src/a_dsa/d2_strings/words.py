
# LC140. Word Break II
def wordBreak(s: str, wordDict: List[str]) -> List[str]:
    word_set = set(wordDict)  # O(2^len(words) + W), return all possible answer
    def dfs(s):
        output = []
        if s in word_set: output.append(s)  # one of solutions
        for w in word_set:  # or we loop s with prefix
            if s.startswith(w):
                tmp = dfs(s[len(w):])
                for x in tmp: output.append(w + ' ' + x)
        return output
    return dfs(s)
def wordBreak(self, s, wordDict):  # O(2^len(wordDict) + W)
    memo, wordDict = {len(s): ['']}, set(wordDict)
    def sentences(i):  # returns list of all sentences built from the suffix s[i:]
        if i not in memo:
            memo[i] = [s[i:j] + (tail and ' ' + tail)  # ' '+tail if tail else tail
                       for j in range(i+1, len(s)+1) if s[i:j] in wordDict
                       for tail in sentences(j)]
        return memo[i]
    return sentences(0)

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

# LC249. Group Shifted Strings
def groupStrings(self, strings: List[str]) -> List[List[str]]:
    def shash(s):
        if not s: return -1
        if len(s) == 1: return 1  # single char string can always be shifted
        d = ord(s[0]) - ord('a')  # d is the shift
        ret = []
        for c in s:
            h = (ord(c) - d) % 26  # now shift all chars by the same to get hash
            ret.append(h)
        return tuple(ret)
    groups = collections.defaultdict(list)
    for s in strings: groups[shash(s)].append(s)
    return groups.values()

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
                    wordList.remove(next_word)  # this is the key, BFS doesn't look back
                    queue.append([next_word, length + 1])
    return 0

# LC126. Word Ladder II, top100 - Same for find all paths in DAG
def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    wordList = set(wordList)
    res, layer = [], {beginWord: [[beginWord]]}
    while layer:
        newlayer = collections.defaultdict(list)
        for w in layer:
            if w == endWord: res.extend(k for k in layer[w])
            else:
                for i in range(len(w)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        neww = w[:i] + c + w[i+1:]
                        if neww in wordList:
                            newlayer[neww] += [j + [neww] for j in layer[w]]
        wordList -= set(newlayer.keys())
        layer = newlayer
    return res

# LC691. Stickers to Spell Word, effectively this is bfs since we look for min.
def minStickers(self, stickers: List[str], target: str) -> int:
    counters = [Counter(s) for s in stickers]
    @lru_cache(None)
    def dfs(target):  # target string to child nodes by apply stickers
        res = float('inf')
        for sticker in counters:  # DFS on stickers/neighbours
            if target[0] not in sticker: continue  # to cut search branches
            targetnew = target
            for s in sticker: targetnew = targetnew.replace(s, '', sticker[s])
            if targetnew == '':
                res = 1
                break
            elif targetnew != target: res = min(res, 1 + dfs(targetnew))
        return res
    res = dfs(target)
    return -1 if res == float('inf') else res

# LC824 Goat Latin
def toGoatLatin(self, sentence: str) -> str:
    vowels = set(['a', 'e', 'i', 'o', 'u'])
    words = sentence.split(' ')
    ret = []
    for i, w in enumerate(words):
        if w[0].lower() in vowels: w += 'ma'
        else: w = w[1:] + w[0] + 'ma'
        w += 'a' * (i+1)
        ret.append(w)
    return ' '.join(ret)

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
        board[i][j] = chr(board[i][j] ^ 256)  # backout
        return exist
    bls = set()  # precheck board has all letters from word,
    for row in board: bls.update(row)  # this makes the test faster 5% -> 74%
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
        board[row][col] = '#' # Before the EXPLORATION, mark the cell as visited, backtracking
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

# LC14. Longest Common Prefix
def longestCommonPrefix(self, strs):
    if not strs: return ""
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch: return shortest[:i]
    return shortest

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
            if i == len(word): return None in node and not mistakeAllowed
            if word[i] not in node:
                return any(find(node[letter], i+1, False) for letter in node if letter) if mistakeAllowed else False
            if mistakeAllowed:
                return find(node[word[i]], i+1, True) or \
                       any(find(node[letter], i+1, False) for letter in node if letter and letter != word[i])
            return find(node[word[i]], i+1, False)
        return find(self.trie, 0, True)

# LC68. Text Justification
def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    ans = []
    line, width = [], 0
    for word in words:
        if width + len(line) + len(word) > maxWidth:  # len(line) spaces
            n, k = divmod(maxWidth - width, max(1, len(line)-1))
            for i in range(max(1, len(line)-1)):
                line[i] += " " * (n + (i < k))  # add space after word
            ans.append("".join(line))
            line, width = [], 0
        line.append(word)
        width += len(word)
    ans.append(" ".join(line).ljust(maxWidth))
    return ans

# LC30. Substring with Concatenation of All Words
def findSubstring(self, s: str, words: List[str]) -> List[int]:  # O(numWords * len(s))
    wordBag = Counter(words)   # count the freq of each word
    wordLen, numWords = len(words[0]), len(words)
    totalLen, res = wordLen*numWords, []
    for i in range(len(s)-totalLen+1):   # scan through s
        # For each i, determine if s[i:i+totalLen] is valid
        seen = defaultdict(int)   # reset for each i
        for j in range(i, i+totalLen, wordLen):
            currWord = s[j:j+wordLen]
            if currWord in wordBag:
                seen[currWord] += 1
                if seen[currWord] > wordBag[currWord]: break
            else: break  # if not in wordBag
        if seen == wordBag: res.append(i)  # store result
    return res



# LC692. Top K Frequent Words
def topKFrequent(self, words: List[str], k: int) -> List[str]:  # O(n)
    freqs = Counter(words)
    return heapq.nsmallest(k, freqs.keys(), lambda w: (-freqs[w], w))

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

# LC1554. Strings Differ by One Character
def differByOne(self, dict: List[str]) -> bool:  # O(mn)
    n, m = len(dict), len(dict[0])
    for j in range(m):
        seen = set()
        for i in range(n):
            new_w = dict[i][:j] + '*' +dict[i][j+1:]
            if new_w in seen: return True
            seen.add(new_w)
    return False

# LC839. Similar String Groups
def numSimilarGroups(self, strs: List[str]) -> int:
    def similar(s, t):
        diff = 0
        for c1, c2 in zip(s, t):
            diff += 1 if c1 != c2 else 0
            if diff > 2: return False
        return True
    graph = defaultdict(list)
    for s1 in strs:  #  O(n^2)
        for s2 in strs:
            if similar(s1, s2): graph[s1].append(s2)
    visited = set()
    def dfs(st, graph):
        for nb in graph[st]:
            if nb not in visited:
                visited.add(nb)
                dfs(nb, graph)
    ret = 0
    for s1 in strs:
        if s1 not in visited:
            dfs(s1, graph)
            ret += 1
    return ret

# LC734. Sentence Similarity
def areSentencesSimilar(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
    if len(sentence1) != len(sentence2): return False
    sets = set((a, b) for a, b in similarPairs)
    for w, v in zip(sentence1, sentence2):
        if w != v and (w,v) not in sets and (v,w) not in sets: return False
    return True

# LC151. Reverse Words in a String
def reverseWords(self, s: str) -> str:
    return " ".join(reversed(s.split()))

# LC809. Expressive Words
def expressiveWords(self, s: str, words: List[str]) -> int:  # O(n * max(all words and s))
    def check(S, W):  # 2 pointers
        i, j, n, m = 0, 0, len(S), len(W)
        for i in range(n):
            if j < m and S[i] == W[j]: j += 1
            elif S[i - 1:i + 2] != S[i] * 3 != S[i - 2:i + 1]: return False
        return j == m  # not like S = "abc" W="abcd"
    return sum(check(s, W) for W in words)

# LC1160. Find Words That Can Be Formed by Characters
def countCharacters(self, words: List[str], chars: str) -> int:
    suml = 0
    chars_counter = collections.Counter(chars)
    for word in words:
        word_counter = collections.Counter(word)
        if all(word_counter[c] <= chars_counter[c] for c in word_counter):
            suml += len(word)
    return suml

# LC1048. Longest String Chain
def longestStrChain(self, words: List[str]) -> int:
    word_set = set(words)
    @lru_cache(None)
    def dfs(word):  # O(N) or O(L^2 * N) where L is max len of words
        t_max =1
        for i in range(len(word)):
            tmp = word[:i] + word[i+1:]  # delete a letter, instead of inserting
            if tmp in word_set: t_max = max(t_max, dfs(tmp)+1)
        return t_max
    result = 0
    for word in word_set: result = max(dfs(word), result)
    return result

# LC1268. Search Suggestions System
def suggestedProducts(self, A, word):
    A.sort()
    res, prefix, i = [], '', 0
    for c in word:
        prefix += c
        i = bisect.bisect_left(A, prefix, i)
        res.append([w for w in A[i:i + 3] if w.startswith(prefix)])
    return res

# LC524. Longest Word in Dictionary through Deleting - not only counts, but also order
def findLongestWord(self, s, d):
    for x in sorted(d, key=lambda x: (-len(x), x)):
        it = iter(s) # call 'in' moves to next char
        if all(c in it for c in x):
            return x
    return ''

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
