# 211. Design Add and Search Words Data Structure
class WordDictionary: # much faster
    def __init__(self): self.trie = {}
    def addWord(self, word: str) -> None:
        node = self.trie
        for ch in word: node = node.setdefault(ch, {})
        node['$'] = True  # mark end of word
    def search(self, word: str) -> bool:
        def search_in_node(word, node) -> bool: # recursion on dot
            for i, ch in enumerate(word):
                if ch in node: node = node[ch] # char found, go down
                else:
                    if ch == '.':  # we need to check all but $
                        for x in node:
                            if x != '$' and search_in_node(word[i + 1:], node[x]):
                                return True
                    return False
            return '$' in node
        return search_in_node(word, self.trie)

# LC1233. Remove Sub-Folders from the Filesystem
def removeSubfolders(self, folder: List[str]) -> List[str]:
    trie = {}
    for path in folder: # build trie from input
        dirs = path[1:].split('/')
        node = trie
        for di in dirs: node = node.setdefault(di, {})
        node['end'] = path # save path to end
    print(trie)
    ret = [] # collect top folders
    def dfs(node):
        if 'end' in node:
            ret.append(node['end']) # earliest end is top folder
            return
        for d in node: dfs(node[d])
    dfs(trie)
    return ret

# LC642. Design Search Autocomplete System
from collections import defaultdict
class AutocompleteSystem:
    def __init__(self, sentences: List[str], times: List[int]):
        self.counts = defaultdict(int) # or use trie
        for s, c in zip(sentences, times): self.counts[s] = c
        self.ui = '' # user input
        self.matches = []
    def input(self, c: str) -> List[str]:
        if c == '#':
            self.counts[self.ui] += 1
            self.ui = ''
            self.matches = []
            return []
        if not self.ui: # new input
            self.matches = [(-ct, s) for s, ct in self.counts.items() if s[0] == c]  # O(n)
            self.matches.sort()  # O(nlogn)
            self.matches = [s for _, s in self.matches]
        else:
            n = len(self.ui)
            self.matches = [m for m in self.matches if len(m) > n and m[n] == c]
        self.ui += c
        return self.matches[:3]

# LC208. Implement Trie (Prefix Tree)
class TrieNode:
    def __init__(self, value):
        self.value = value
        self.children = dict()
        self.is_end = False
    def add_child(self, value):
        self.children[value] = TrieNode(value)
    def get_child(self, key):
        return self.children[key]
    def has_child(self, key):
        return key in self.children
class Trie:
    def __init__(self):
        self.root = TrieNode(None)
    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if not node.has_child(c): node.add_child(c)
            node = node.get_child(c)
        node.is_end = True
    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            if not node.has_child(c): return False
            node = node.get_child(c)
        return node.is_end
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if not node.has_child(c): return False
            node = node.get_child(c)
        return True
