################################################################################
# Given a binary tree, check whether it is a mirror of itself (ie, symmetric
# around its center).
# For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3
# But the following [1,2,2,null,3,null,3] is not:
#     1
#    / \
#   2   2
#    \   \
#     3   3
# EASY
################################################################################
from dataclasses import  dataclass

@dataclass()
class TreeNode:
    val: int
    left: 'TreeNode'
    right: 'TreeNode'


def is_tree_symmetric(root: TreeNode) -> bool:
    def is_mirror(n1, n2):
        if n1 is None and n2 is None:
            return True
        if n1 is None or n2 is None:
            return False

        return n1.val == n2.val and is_mirror(n1.left, n2.right)\
            and is_mirror(n1.right, n2.left)

    return is_mirror(root, root)


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
print(is_tree_symmetric(a))
print(tree2array(a))
