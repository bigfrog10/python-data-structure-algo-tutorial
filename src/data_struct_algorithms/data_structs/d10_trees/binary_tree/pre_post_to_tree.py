
class TreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        return f'[val={self.data}, left={self.left}, right={self.right}]'


# modified from: https://www.techiedelight.com/construct-full-binary-tree-from-preorder-postorder-sequence/
# https://www.tutorialspoint.com/construct-binary-tree-from-preorder-and-postorder-traversal-in-python
# With pre- and post- order traversals, the binary tree is not uniquely determined.
# However, if we know the tree is a full binary tree, i.e., every node has 0 or 2 child nodes, then
# the binary tree is unique.
def const_tree_from_pre_post(pre_order: list, post_order: list):
    po_idxs = {e: i for i, e in enumerate(post_order)}

    def build(idx, left, right):
        root = TreeNode(pre_order[idx])

        idx += 1
        if idx == len(pre_order):
            return root, idx

        nidx = po_idxs[pre_order[idx]]
        if left <= nidx <= right - 2:  # right - 1 is root
            root.left, idx = build(idx, left, nidx)
            root.right, idx = build(idx, nidx+1, right-1)

        return root, idx

    root, idx = build(0, 0, len(pre_order) - 1)
    return root


pre = [1,2,4,5,3,6,7]
post = [4,5,2,6,7,3,1]
print(const_tree_from_pre_post(pre, post))  # 1, 2, 4, 5, 3, 6, 7

pre = [1, 2, 4, 5, 3, 6, 8, 9, 7]
post = [4, 5, 2, 8, 9, 6, 7, 3, 1]
print(const_tree_from_pre_post(pre, post)) # 1, 2, 4, 5, 3, 6, 8, 9, 7
