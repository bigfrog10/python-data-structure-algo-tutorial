# You are given a singly-linked list that contains N integers. A subpart of the list is a contiguous set of even elements, bordered either by either end of the list or an odd element. For example, if the list is [1, 2, 8, 9, 12, 16], the subparts of the list are [2, 8] and [12, 16].
# Then, for each subpart, the order of the elements is reversed. In the example, this would result in the new list, [1, 8, 2, 9, 16, 12].
# The goal of this question is: given a resulting list, determine the original order of the elements.
# Implementation detail:
# You must use the following definition for elements in the linked list:
#
# class Node {
#     int data;
#     Node next;
# }
#
# Signature
# Node reverse(Node head)
# Constraints
# 1 <= N <= 1000, where N is the size of the list
# 1 <= Li <= 10^9, where Li is the ith element of the list
# Example
# Input:
# N = 6
# list = [1, 2, 8, 9, 12, 16]
# Output:
# [1, 8, 2, 9, 16, 12]
class Node:
  def __init__(self, x):
    self.data = x
    self.next = None

def reverse(head):
  # Write your code here
  def rev_ll(start, end):
    if not head: return None
    prev, curr = end, start.next
    while curr != end and curr.data % 2 == 0:
      cn = curr.next
      curr.next = prev
      prev, curr = curr, cn
    start.next = prev
  def printll(ll):
    ret = []
    while ll:
      ret.append(ll.data)
      ll = ll.next
    print(ret)

  printll(head)
  dummy = Node('dummy')
  dummy.next = head
  prev, curr = None, head
  st, end = None, None
  while curr:
    if not prev and curr.data % 2 == 0:
      st = dummy
    elif prev and prev.data % 2 != 0 and curr and curr.data % 2 == 0:
      st = prev
    elif prev and prev.data % 2 == 0 and curr and curr.data % 2 != 0:
      end = curr
      if st.next != prev: # not single even number
        rev_ll(st, end)
    prev = curr
    curr = curr.next
  if prev and prev.data % 2 == 0:
    end = curr
    if st.next != prev: # not single even number
      rev_ll(st, end)
  printll(dummy.next)
  return dummy.next

def createll(arr):
    dummy = Node('dummy')
    w = dummy
    for a in arr:
        w.next = Node(a)
        w = w.next
    return dummy.next

print(reverse(createll([1, 2, 3])))
print(reverse(createll([1, 2, 4, 3])))
print(reverse(createll([1, 2, 4, 6, 3])))
print(reverse(createll([2, 4, 6, 3])))
print(reverse(createll([1, 2, 4, 6])))
