
# LC426. Convert Binary Search Tree to Sorted Doubly Linked List, BST
def treeToDoublyList(self, root: 'Node') -> 'Node':  # O(n) time and O(1) space, Morris Traversal
    if not root: return root
    prev = head = Node()
    curr = root
    while curr:
        if curr.left:
            rightmost = curr.left # find predecessor
            while rightmost.right: rightmost = rightmost.right
            rightmost.right = curr  # predecessor to curr
            tmp, curr = curr, curr.left
            tmp.left = None  # predecessor points to curr/tmp now, next else blcok set both dirs.
        else:
            curr.left, prev.right = prev, curr  # point to each other
            prev, curr = curr, curr.right
    prev.right = head.right
    head.right.left = prev
    return head.right
def treeToDoublyList(self, root: 'Node') -> 'Node':  # O(n) runtime and space
    if not root: return None
    prev = dummy = Node(-1)  # dummy is head, stay there
    def inorder(node):  # recursion stack is O(n)
        nonlocal prev
        if not node: return
        inorder(node.left)  # all the way to 1st node 1
        prev.right, node.left = node, prev  # point to each other
        prev = node  # this is the trick: move prev to 1
        inorder(node.right)
    inorder(root)
    prev.right = dummy.right  # fix head and tail
    dummy.right.left = prev
    return dummy.right

# LC108. Convert Sorted Array to Binary Search Tree
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    def to_bst(nums, b, e) -> TreeNode:
        if b > e: return None
        mid = b + (e - b) // 2  # if len even, use 2nd or 1st
        root = TreeNode(nums[mid], None, None)
        root.left = to_bst(nums, b, mid-1)
        root.right = to_bst(nums, mid + 1, e)
        return root
    return to_bst(nums, 0, len(nums)-1)

# LC109. Convert Sorted List to Binary Search Tree, similar logic as in LC108
def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:  # O(n) time, O(h) space
    def findSize(head):
        ptr, c = head, 0
        while ptr:
            ptr = ptr.next
            c += 1
        return c
    size = findSize(head)
    def convert(l, r):  # Recursively form a BST out of linked list from l --> r
        nonlocal head
        if l > r: return None
        mid = (l + r) // 2
        left = convert(l, mid - 1)  # order is significant!
        node = TreeNode(head.val)
        node.left = left
        head = head.next
        node.right = convert(mid + 1, r)
        return node
    return convert(0, size - 1)

# LC897. Increasing Order Search Tree - tree bst to linked list
def increasingBST(self, root: TreeNode) -> TreeNode:  # O(n) runtime, O(H) space
    def rearrange(node, tail):
        if not node: return tail
        res = rearrange(node.left, node)  # left tree -> link list + root
        node.left = None
        node.right = rearrange(node.right, tail)  # right tree -> link list + tail
        return res
    return rearrange(root, None)

# LC1008. Construct Binary Search Tree from Preorder Traversal
def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
    def buildTree(A, bound):
        if not A or A[-1] > bound: return None
        node = TreeNode(A.pop())
        node.left = buildTree(A, node.val)
        node.right = buildTree(A, bound)
        return node
    return buildTree(preorder[::-1], float('inf'))

def from_postorder(nodes: Sequence[int]) -> BinaryTree[int]:
    def build_subtree(subtree_nodes: Sequence[int]) -> BinaryTree[int]:
        if not subtree_nodes:
            return None

        n = len(subtree_nodes)
        # Locates the insertion point for x to maintain sorted order.
        # This is the first element greater than root.
        x = bisect.bisect_left(subtree_nodes, subtree_nodes[-1], hi=n - 1)

        root = BinaryTree(subtree_nodes[-1])
        root.left = build_subtree(subtree_nodes[:x])
        # slice returns empty list if end is <= start
        root.right = build_subtree(subtree_nodes[x:n - 1])

        return root

    return build_subtree(nodes)



# LC1382. Balance a Binary Search Tree - balance bst
# http://www.smunlisted.com/day-stout-warren-dsw-algorithm.html
# https://csactor.blogspot.com/2018/08/dsw-day-stout-warren-algorithm-dsw.html
def balanceBST(self, root: TreeNode) -> TreeNode:  # O(n) runtime, O(1) space
    # DSW, no recursion
    def to_vine(node):  # to a single linked list, input is the dummy node
        n, cnt = node.right, 0  ## right rotation
        while n:
            if n.left:
                old_n, n  = n, n.left  # move
                old_n.left, n.right = n.right, old_n  # make list
                node.right = n  # track header
            else:
                node, n = n, n.right
                cnt += 1
        return cnt
    dummy = TreeNode(-1, right=root)
    size = to_vine(dummy)
    def compress(node, size: int):  ## left rotation
        n = node.right
        for _ in range(size):
            old_n, n = n, n.right  # move
            old_n.right, n.left = n.left, old_n  # make tree
            node.right, node = n, n  # track header
            n = n.right
    h = 2 ** int(math.log2(size+1)) - 1
    compress(dummy, size - h)
    while h > 1:
        h = h // 2
        compress(dummy, h)
    return dummy.right
def balanceBST(self, root: TreeNode) -> TreeNode:  # O(n) runtime and O(n) space
    def inorder(node, li):  # O(N), collect all nodes to a list
        if not node: return
        inorder(node.left, li)
        li.append(node)
        inorder(node.right, li)
    def buildBst(li, start, end):  # O(N)
        if not li or start > end: return None
        mid = start + (end - start) // 2
        root = li[mid]
        root.left = buildBst(li, start, mid - 1)
        root.right = buildBst(li, mid + 1, end)
        return root
    if not root: return None
    nodes = []
    inorder(root, nodes)
    return buildBst(nodes, 0, len(nodes) - 1)
