# https://www.baeldung.com/cs/tries-prefix-trees

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
        """
        Initialize your data structure here.
        """
        self.root = TrieNode(None)

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for c in word:
            if not node.has_child(c):
                node.add_child(c)
            node = node.get_child(c)
        node.is_end = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for c in word:
            if not node.has_child(c):
                return False
            node = node.get_child(c)

        return node.is_end


    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for c in prefix:
            if not node.has_child(c):
                return False
            node = node.get_child(c)

        return True
