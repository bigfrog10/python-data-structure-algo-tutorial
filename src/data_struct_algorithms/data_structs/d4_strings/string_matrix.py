# LC79. Word Search, top100
def exist(self, board: List[List[str]], word: str) -> bool:
    if not board or not board[0]: return False
    h, w, wl = len(board), len(board[0]), len(word)
    def dfs(i, j, wi):
        if wi == wl: return True
        if board[i][j] != word[wi]: return False
        board[i][j] = ord(board[i][j]) ^ 256
        exist = wi+1 == wl
        for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if 0 <= x < h and 0 <= y < w:
                exist = exist or dfs(x, y, wi+1)
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
