# LC1166. Design File System
class FileSystem:
    def __init__(self):
        self.root = {}
        self.values = {}
    def createPath(self, path: str, value: int) -> bool:
        segs = path.split('/')
        walker = self.root
        for i in range(1, len(segs)-1):  # skip leading "", last is parent, used to create last folder
            if segs[i] not in walker: return False # parent is not here
            walker = walker[segs[i]]
        if segs[-1] in walker: return False # existed already
        else: walker[segs[-1]] = {}
        self.values[path] = value
        return True
    def get(self, path: str) -> int:
        if path not in self.values: return -1
        return self.values[path]

# LC588. Design In-Memory File System
from collections import defaultdict
class Node:
    def __init__(self):
        self.child=defaultdict(Node)
        self.content=""
class FileSystem(object):
    def __init__(self):
        self.root=Node()
    def find(self,path):#find and return node at path.
        curr=self.root
        if len(path)==1: return self.root # /
        for word in path.split("/")[1:]: curr=curr.child[word]
        return curr
    def ls(self, path):
        curr=self.find(path)
        if curr.content: return [path.split('/')[-1]]#file path,return file name
        else: return sorted(curr.child.keys())
    def mkdir(self, path):
        self.find(path)
    def addContentToFile(self, filePath, content):
        curr=self.find(filePath)
        curr.content += content
    def readContentFromFile(self, filePath):
        curr = self.find(filePath)
        return curr.content

# LC588. Design In-Memory File System
from collections import defaultdict
class Node:
    def __init__(self):
        self.child=defaultdict(Node)
        self.content=""
class FileSystem(object):
    def __init__(self):
        self.root=Node()
    def find(self,path):#find and return node at path.
        curr=self.root
        if len(path) == 1: return self.root # /
        for word in path.split("/")[1:]: curr = curr.child[word]
        return curr
    def ls(self, path):
        curr=self.find(path)
        if curr.content: return [path.split('/')[-1]]#file path,return file name
        else: return sorted(curr.child.keys())
    def mkdir(self, path):
        self.find(path)
    def addContentToFile(self, filePath, content):
        curr=self.find(filePath)
        curr.content += content
    def readContentFromFile(self, filePath):
        curr = self.find(filePath)
        return curr.content
