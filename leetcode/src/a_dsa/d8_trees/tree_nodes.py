class BinaryTreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class NaryTreeNode:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children if children else []
