
# LC609. Find Duplicate File in System
def findDuplicate(self, paths: List[str]) -> List[List[str]]:  # O(all chars) time and space
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

# LC14. Longest Common Prefix - lcp
def longestCommonPrefix(self, strs):  # O(sum(len(str)))
    if not strs: return ""
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch: return shortest[:i]
    return shortest

# LC2042. Check if Numbers Are Ascending in a Sentence
def areNumbersAscending(self, s: str) -> bool:
    nums = [int(w) for w in s.split() if w.isdigit()]
    return all(nums[i-1] < nums[i] for i in range(1, len(nums)))

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

# LC151. Reverse Words in a String
def reverseWords(self, s: str) -> str:
    return " ".join(reversed(s.split()))

# LC1048. Longest String Chain
def longestStrChain(self, words: List[str]) -> int:  # O(L^2 * N) where L is max len of words, O(N) space
    word_set = set(words)
    @lru_cache(None)
    def dfs(word):  # O(N)
        t_max =1
        for i in range(len(word)):  # L^2
            tmp = word[:i] + word[i+1:]  # delete a letter, instead of inserting
            if tmp in word_set: t_max = max(t_max, dfs(tmp)+1)
        return t_max
    result = 0
    for word in word_set: result = max(dfs(word), result)
    return result

# LC249. Group Shifted Strings
def groupStrings(self, strings: List[str]) -> List[List[str]]:
    def shash(s):
        if not s: return -1
        d = ord(s[0]) - ord('a')  # d is the shift
        ret = []
        for c in s:
            h = (ord(c) - d) % 26  # now shift all chars by the same to get hash
            ret.append(h)
        return tuple(ret)
    groups = collections.defaultdict(list)
    for s in strings: groups[shash(s)].append(s)
    return groups.values()

# LC140. Word Break II - return all possible answer
def wordBreak(self, s: str, wordDict):  #  we may have O(2^n) solutions, n=len(s)
    word_set = set(wordDict)   ## O(2^n + m) time and space, m = len(wordDict)
    @cache
    def dfs(s):
        if s == '': return []
        output = []
        if s in word_set: output.append(s) # one of solutions
        for i in range(len(s) - 1):
            if s[:i+1] in word_set:  # substring takes n
                tmp = dfs(s[i+1:])  # tmp is like ['a b', 'ab']
                for x in tmp: output.append(s[:i+1] + ' ' + x)
        return output
    res = dfs(s)
    return res
def wordBreak(s: str, wordDict: List[str]) -> List[str]:
    word_set = set(wordDict)  # O(2^len(s) + len(worddict),
    def dfs(s):
        output = []
        if s in word_set: output.append(s)  # one of solutions
        for w in word_set:  # or we loop s with prefix, O(n^2)
            if s.startswith(w):
                tmp = dfs(s[len(w):])
                for x in tmp: output.append(w + ' ' + x)
        return output
    return dfs(s)

# LC139. Word Break, top100 - return breakable or not
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    wordset = set(wordDict)  ## O(n^3) n = len(s)     ### + O(m), m is size of dict
    @lru_cache  # O(n)
    def break_words(start: int):
        if start == len(s): return True
        for end in range(start + 1, len(s)+1):  # O(n), need +1 for next line [start:end]
            if s[start:end] in wordset and break_words(end): return True  # s[start:end] is O(n)
        return False
    return break_words(0)
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    wds = set(wordDict)
    @lru_cache(None)
    def dfs(s):
        if s in wds: return True
        for w in wordDict:
            if s.startswith(w) and dfs(s[len(w):]): return True
        return False
    return dfs(s)

# LC472. Concatenated Words   concat words
def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    word_set = set(words)  # O(N * M^2) time
    def check(word):  # could use cache here
        n = len(word)  # Check if current word can be made from smaller words
        for i in range(1, n):  # O(m^2), longest word
            if word[i:] not in word_set: continue
            # so now word[i:] in word_set
            if word[:i] in word_set: return True # so both part are words
            if check(word[:i]): return True # recursion check
        return False
    res = []
    for w in words:  # O(n) - number of words
        if check(w): res.append(w)
    return res

# LC127. Word Ladder, return min # of words to transform
def ladderLength(self, beginWord, endWord, wordList):  # BFS, O(m^2 * len(wordList))
    wordList = set(wordList)  # m = max word length
    queue = collections.deque([[beginWord, 1]])
    while queue:  # BFS since we look for shortest
        word, length = queue.popleft()
        if word == endWord: return length
        for i in range(len(word)):  # O(m)
            for c in string.ascii_lowercase:  # 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]  # O(m)
                if next_word in wordList:
                    wordList.remove(next_word)  # this is the key, BFS doesn't look back
                    queue.append([next_word, length + 1])
    return 0

# LC126. Word Ladder II, top100 - Same for find all paths in DAG
def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    wordList = set(wordList)  # O(n * m^2)
    res, layer = [], {beginWord: [[beginWord]]}  # end word to history
    while layer:  # O(len(words)), BFS to find min
        newlayer = collections.defaultdict(list)
        for w in layer:
            if w == endWord: res.extend(k for k in layer[w])
            else:
                for i in range(len(w)):   # O(m)
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        neww = w[:i] + c + w[i+1:]   # O(m)
                        if neww in wordList:
                            newlayer[neww] += [j + [neww] for j in layer[w]]
        wordList -= set(newlayer.keys())
        layer = newlayer
    return res

# LC79. Word Search, top100 - search in 2d matrix
def exist(self, board: List[List[str]], word: str) -> bool:  # O(h * w * 3^wl), wl = len(word), space O(wl)
    if not board or not board[0]: return False
    h, w, wl = len(board), len(board[0]), len(word)
    def dfs(i, j, wi):
        if board[i][j] != word[wi]: return False
        if wi+1 == wl: return True
        board[i][j] = ord(board[i][j]) ^ 256
        for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if 0 <= x < h and 0 <= y < w:
                if dfs(x, y, wi+1): return True
        board[i][j] = chr(board[i][j] ^ 256)  # backout
        return False
    # bls = set()  # precheck board has all letters from word,
    # for row in board: bls.update(row)  # this makes the test faster 5% -> 74%
    # wls = set(word)
    # if len(wls - bls) > 0: return False
    for i in range(h):
        for j in range(w):
            if dfs(i, j, 0): return True
    return False

# LC212. Word Search II - return all words
def findWords(self, board: List[List[str]], words: List[str]) -> List[str]: # This is fast
    trie, WORD_KEY = {}, '$'  # O(M * 4 * 3^(L-1)), M=cells, L=max(len(word) for words)
    for word in words:  # space O(number of letters in trie)
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
            if rowNum > nx >= 0 <= ny < colNum and board[nx][ny] in currNode:
                dfs(nx, ny, currNode)
        board[row][col] = letter # End of EXPLORATION, we restore the cell
        # Optimization: incrementally remove the matched leaf node in Trie.
        if not currNode: parent.pop(letter) # we pop'd WORD_KEY before
    for row in range(rowNum): # O(nm)
        for col in range(colNum):# starting from each of the cells
            if board[row][col] in trie: dfs(row, col, trie)
    return matchedWords

# LC68. Text Justification
def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    res, line, width = [], [], 0  # O(len(words) * avg(len(w))) time, O(maxWidth) space
    for w in words:
        if width + len(line) + len(w) > maxWidth:  # len(line) spaces between words
            m = len(line) - 1 or 1 # if only 1 word in this line
            for i in range(maxWidth - width): line[i % m] += ' '  # add ' ' to each word
            res.append(''.join(line))
            line, width = [], 0
        line.append(w)
        width += len(w)
    res.append(" ".join(line).ljust(maxWidth))  # last line
    return res

# LC616. Add Bold Tag in String, same as LC758.
def addBoldTag(self, s: str, dict1: List[str]) -> str:  # O(len(s) * len(words)) time and O(len(s)) space
    if not dict1: return s

    words = set(dict1)
    idxs = []
    for i in range(len(s)):  # O(len(s) * len(words) * max(words len)
        suffix = s[i:]
        for w in words:
            if suffix.startswith(w): idxs.append([i, i + len(w)])

    if not idxs: return s  # idx length is O(len(s) * len(words))
    intvs = []
    for t in idxs:
        if len(intvs) == 0 or intvs[-1][1] < t[0]: intvs.append(t)
        else: intvs[-1][1] = max(t[1], intvs[-1][1])

    res = []
    start = end = 0
    for a in intvs:  # intvs length is O(len(s) * len(words))
        end = a[0]
        res.append(s[start:end])
        res.append(f"<b>{s[a[0]:a[1]]}</b>")
        start = a[1]
    res.append(s[start:])
    return ''.join(res)
def addBoldTag(self, s: str, words: List[str]) -> str:
    status = [False] * len(s)
    for word in words:  # O(len(words))
        start, last = s.find(word), len(word)  # O(len(s) * O(maxlen(words)))
        while start != -1: # this word appears multiple places
            for i in range(start, last+start): status[i] = True
            start = s.find(word, start+1)
    i, final = 0, ""
    while i < len(s):  # O(len(s))
        if status[i]:
            final += "<b>"
            while i < len(s) and status[i]:
                final += s[i]
                i += 1
            final += "</b>"
        else:
            final += s[i]
            i += 1
    return final
def addBoldTag(self, s: str, words: List[str]) -> str:  # long solution with kmp
    def kmp_pi(s: str):  # the length of the longest *proper* prefix of s which is also a suffix
        m = len(s)
        pi = [0] * m
        fmm_idx = 0  # index of first mismatch
        for i in range(1, m):
            # when there is a mismatch, rollback to previous suffix.
            while fmm_idx and s[i] != s[fmm_idx]: fmm_idx = pi[fmm_idx-1]
            # if fmm_idx = 0, continue with pi[i] = 0
            if s[i] == s[fmm_idx]:  # if matched, move forward
                fmm_idx += 1
                pi[i] = fmm_idx
        return pi
    def kmp_search(text, pattern):  # return first indices of all occurrences of pattern in text
        matches, pi = [], kmp_pi(pattern)
        n, m = len(text), len(pattern)
        k = 0  # pattern index
        for i in range(n):
            while k and text[i] != pattern[k]: k = pi[k - 1]  # if k = 0, continue
            if text[i] == pattern[k]:
                if k == m - 1:
                    matches.append(i - k)
                    k = pi[k]
                else: k += 1
        return matches
    status = [False] * len(s)  # compromise to flags
    for word in words:  # O(len(words) * len(s))
        idxs = kmp_search(s, word)
        for i in idxs: status[i:i+len(word)] = [True] * len(word)
    i, res = 0, ""
    while i < len(s):  # O(len(s))
        if status[i]:
            res += "<b>"
            while i < len(s) and status[i]:
                res += s[i]
                i += 1
            res += "</b>"
        else:
            res += s[i]
            i += 1
    return res

# LC758. Bold Words in String
def addBoldTag(self, s, dict):
    status = [False]*len(s)
    for word in dict:
        start, last = s.find(word), len(word)
        while start != -1: # this word appears multiple places
            for i in range(start, last+start): status[i] = True
            start = s.find(word, start+1)
    i, final = 0, ""
    while i < len(s):
        if status[i]:
            final += "<b>"
            while i < len(s) and status[i]:
                final += s[i]
                i += 1
            final += "</b>"
        else:
            final += s[i]
            i += 1
    return final

# LC691. Stickers to Spell Word - effectively this is bfs since we look for min.
def minStickers(self, stickers: List[str], target: str) -> int:
    counters = [Counter(s) for s in stickers]  # O(S^T with lowers)
    @lru_cache(None)
    def dfs(target):  # target string to child nodes by apply stickers
        res = float('inf')
        for sticker in counters:  # DFS on stickers/neighbours
            if target[0] not in sticker: continue  # to cut search branches
            targetnew = target
            for c in sticker: targetnew = targetnew.replace(c, '', sticker[c])
            if not targetnew: return 1
            if len(targetnew) < len(target):
                res = min(res, 1 + dfs(targetnew))
        return res
    res = dfs(target)
    return -1 if res == float('inf') else res

# LC1002. Find Common Characters
def commonChars(self, words: List[str]) -> List[str]:  # O(all chars in words)
    res = collections.Counter(words[0])
    for a in words: res &= collections.Counter(a)
    return list(res.elements())

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

# LC2023. Number of Pairs of Strings With Concatenation Equal to Target concat to target  pair concat equal
def numOfPairs(self, nums: List[str], target: str) -> int:
    freq = Counter(nums)
    ans = 0
    for k, v in freq.items():
        if target.startswith(k):
            suffix = target[len(k):]
            ans += v * freq[suffix]  # num of k * num of suffix
            if k == suffix: ans -= v  # together, n^2 - n when prefix = suffix
    return ans

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


# LC557. Reverse Words in a String III
def reverseWords(self, s: str) -> str:
    words = s.split(' ')
    res = ' '.join(w[::-1] for w in words)
    return res

# LC809. Expressive Words - stretchy word
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



# LC1268. Search Suggestions System
def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:  # O(nlog(n))+O(mlog(n))
    products.sort()  # O(nlogn)
    res, prefix, i = [], '', 0
    for c in searchWord:  # O(m)
        prefix += c
        i = bisect.bisect_left(products, prefix)#, i)  # O(logm)
        res.append([w for w in products[i:i + 3] if w.startswith(prefix)])
    return res
# Or Trie

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



# LC243. Shortest Word Distance - distance between 2 given words in the array
def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
    out = first = second = float('inf')
    for i,word in enumerate(wordsDict):
        if word == word1: first = i
        elif word == word2: second = i
        out = min(abs(first - second), out)
    return out
