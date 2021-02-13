################################################################################
# Given the root of a binary tree, return its maximum depth.
# A binary tree's maximum depth is the number of nodes along the longest path
# from the root node down to the farthest leaf node.
# EASY
################################################################################
from dataclasses import  dataclass

@dataclass()
class TreeNode:
    val: int
    left: 'TreeNode'
    right: 'TreeNode'


def binary_tree_max_depth(root: TreeNode) -> int:
    if not root:
        return 0

    return 1 + max(binary_tree_max_depth(root.left),
                   binary_tree_max_depth(root.right))


# The nth level starts from 2^n - 1, and length 2^n
def array2tree(array: list) -> TreeNode:
    root = None
    parents = None
    level = 0
    loop = True
    while loop:
        nodes = []
        for i in range(2 ** level - 1, 2 ** (level + 1) - 1):
            if i >= len(array):
                loop = False
                break

            node = TreeNode(array[i], None, None)
            nodes.append(node)

            if parents:
                j = (i - (2 ** level - 1)) // 2
                k = i % 2
                if k == 0:
                    parents[j].right = node
                else:
                    parents[j].left = node
            else:
                root = node

        parents = nodes
        level += 1

    return root


def tree2array(root: TreeNode) -> list:
    def t2a(nodes: list, array: list):
        children = []
        for n in nodes:
            array.append(n.val)
            if n.left:
                children.append(n.left)
            if n.right:
                children.append(n.right)

        return children

    res = []
    nodes = [root]
    while nodes:
        nodes = t2a(nodes, res)

    return res


a = array2tree([1, 2, 2, 3, 4, 4, 3])
print(a)
print(binary_tree_max_depth(a))
