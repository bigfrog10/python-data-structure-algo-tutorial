
# LC895. Maximum Frequency Stack - max freq stack  mfs
class FreqStack(object): # This is O(1) operations
    def __init__(self):
        self.maxf = 0 # This is the trick to keep O(1)
        self.v2f = collections.defaultdict(int)
        self.f2v = collections.defaultdict(list)
    def push(self, x):
        # we don't delete x from current freq, this maintains the stack order.
        f = self.v2f[x] + 1
        self.v2f[x] = f
        self.f2v[f].append(x)
        self.maxf = max(self.maxf, f)
    def pop(self):
        x = self.f2v[self.maxf].pop()
        if not self.f2v[self.maxf]:  #if that's the only max element in frequency decrease by 1
            self.maxf -= 1 # this is true because of the line below, i.e., freq dec by 1
        # we don't need to reinsert x with lower frequency, it's already there.
        self.v2f[x] -= 1
        return x
# LC622. Design Circular Queue - RingBuffer
class MyCircularQueue: # array based, O(1) for all ops
    def __init__(self, k: int):  # double linked list
        self.capacity, self.queue = k, [0]*k
        self.count = 0  # how many elements in the queue
        self.headIndex = 0  # front element index
        # tail element index = (head + count - 1) % capa
    def enQueue(self, value: int) -> bool:  # tail
        if self.count == self.capacity: return False
        tail = (self.headIndex + self.count) % self.capacity
        self.queue[tail] = value
        self.count += 1
        return True
    def deQueue(self) -> bool:  # head
        if self.count == 0: return False
        self.headIndex = (self.headIndex + 1) % self.capacity
        self.count -= 1
        return True
    def Front(self) -> int: # just return the element, don't remove it
        if self.count == 0: return -1
        return self.queue[self.headIndex]
    def Rear(self) -> int: # just return the element, don't remove it
        if self.count == 0: return -1
        tail = (self.headIndex + self.count - 1) % self.capacity
        return self.queue[tail]
    def isEmpty(self) -> bool: return self.count == 0
    def isFull(self) -> bool: return self.count == self.capacity

# LC232. Implement Queue using Stacks   queue from stack to queue stack queue
class MyQueue:
    def __init__(self):  # amortized O(1) - O(1) average for n operations
        self.stack1 = []
        self.stack2 = []
    def push(self, x: int) -> None:
        self.stack1.append(x)
    def pop(self) -> int:
        if not self.stack2:
            while self.stack1: self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
    def peek(self) -> int:
        if not self.stack2:
            while self.stack1: self.stack2.append(self.stack1.pop())
        return self.stack2[-1]
    def empty(self) -> bool:
        return len(self.stack1) == 0 and len(self.stack2) == 0

# LC155. Min Stack, top100
class MinStack:
    def __init__(self):
        self.values = []  # list end is stack top
    def push(self, x: int) -> None:
        if self.values: mv = (x, min(x, self.values[-1][1]))  # maintain current min
        else:  mv = (x, x)
        self.values.append(mv)
    def pop(self) -> None: self.values.pop()
    def top(self) -> int: return self.values[-1][0]
    def getMin(self) -> int: return self.values[-1][1]

# LC716. Max Stack
class Dllist:
    def __init__(self, value, left=None, right=None):
        self.val = value
        self.left = left
        self.right = right
    def __repr__(self):
        return f'[{self.val}: left={self.left.val if self.left else None}, right={self.right.val if self.right else None}]'
import sortedcontainers
class MaxStack:  # This is O(logn) solution with extra space needed
    def __init__(self):
        self.head = Dllist('head')
        self.tail = Dllist('tail')
        self.head.right = self.tail
        self.tail.left = self.head
        self.values = sortedcontainers.SortedDict()  # pointers to double linked list, value to nodes
        self.size = 0
    def push(self, x: int) -> None:
        node = Dllist(x, left=self.tail.left, right=self.tail)
        self.tail.left.right, self.tail.left = node, node
        if x in self.values: self.values[x].append(node)
        else: self.values[x] = [node]
        self.size += 1
    def pop(self) -> int:
        if self.size == 0: return -1
        node = self.tail.left
        node.left.right, self.tail.left = self.tail, node.left
        self.size -= 1
        self.values[node.val].pop()
        if not self.values[node.val]: del self.values[node.val]
        return node.val
    def top(self) -> int: return self.tail.left.val  # O(1)
    def peekMax(self) -> int:
        key, nodes = self.values.peekitem()
        return key
    def popMax(self) -> int:  # O(n)
        if self.size == 0: return -1
        key, nodes = self.values.peekitem()
        node = nodes.pop()
        if not nodes: del self.values[key]
        self.size -= 1
        node.left.right, node.right.left = node.right, node.left
        return key
class MaxStack:  # use a stack and sorted container: all ops are at most log(n)
    def __init__(self): self.values = []  # list end is stack top
    def push(self, x: int) -> None:
        if self.values:
            t = self.values[-1]
            mv = (x, max(x, t[1])) # maintain current min
        else:  mv = (x, x)
        self.values.append(mv)
    def pop(self) -> int: return self.values.pop()[0]
    def top(self) -> int: return self.values[-1][0]
    def peekMax(self) -> int: return self.values[-1][1]
    def popMax(self) -> int:   # O(n)
        stack = []
        maxv = self.values[-1][1]
        removed = False # remove only first max
        while self.values and self.values[-1][1] == maxv:
            t = self.values.pop() # pop out all entries with this max
            if t[0] == maxv and not removed:  removed = True
            else: stack.append(t)
        oldmax = self.values[-1][1] if self.values else float('-inf')
        while stack:
            v, m = stack.pop()
            oldmax = max(oldmax, v) # add back with new max
            self.values.append((v, oldmax))
        return maxv



# LC225. Implement Stack using Queues
import collections
class Stack:
    def __init__(self):
        self._queue = collections.deque()
    def push(self, x):
        q = self._queue
        q.append(x)
        for _ in range(len(q) - 1): q.append(q.popleft())
    def pop(self):
        return self._queue.popleft()
    def top(self):
        return self._queue[0]
    def empty(self):
        return not len(self._queue)

# LC641. Design Circular Deque
class MyCircularDeque:
    def __init__(self, k: int):
        self.k = k
        self.arr = [-1] * k  # -1 is required by the problem
        self.front = 0 # current empty
        self.rear = 1 # current empty
        self._size = 0
    def insertFront(self, value: int) -> bool:
        if self.isFull(): return False
        self.arr[self.front] = value
        self.front = (self.front - 1) % self.k
        self._size += 1
        return True
    def insertLast(self, value: int) -> bool:
        if self.isFull(): return False
        self.arr[self.rear] = value
        self.rear = (self.rear + 1) % self.k
        self._size += 1
        return True
    def deleteFront(self) -> bool:
        if self.isEmpty(): return False
        self.front = (self.front + 1) % self.k
        self.arr[self.front] = -1 # -1 is required by the problem
        self._size -= 1
        return True
    def deleteLast(self) -> bool:
        if self.isEmpty(): return False
        self.rear = (self.rear - 1) % self.k
        self.arr[self.rear] = -1 # -1 is required by the problem
        self._size -= 1
        return True
    def getFront(self) -> int:
        f = (self.front + 1) % self.k
        return self.arr[f]
    def getRear(self) -> int:
        r = (self.rear - 1 + self.k) % self.k
        return self.arr[r]
    def isEmpty(self) -> bool: return self._size == 0
    def isFull(self) -> bool: return self._size == self.k

# LC353. Design Snake Game
class SnakeGame:
    def __init__(self, width, height, food):
        self.width, self.height = width, height
        self.snake = deque([[0,0]])  # snake head is at the front (height, width),
        self.food = deque(food) # (0, 0) is upper left, going down positive
        self.dirs = {'U': [-1, 0], 'L': [0, -1], 'R': [0, 1], 'D': [1, 0]}
    def move(self, direction):
        newHead = [self.snake[0][0]+self.dirs[direction][0], self.snake[0][1]+self.dirs[direction][1]]
        # notice that the newHead can be equal to self.snake[-1]
        if (newHead[0] < 0 or newHead[0] >= self.height) or (newHead[1] < 0 or newHead[1] >= self.width)\
            or (newHead in self.snake and newHead != self.snake[-1]): return -1
        self.snake.appendleft(newHead) # just make the food be part of snake, or move
        if self.food and self.food[0] == newHead:  # eat food
            self.food.popleft()   # delete the food that's already eaten
        else: self.snake.pop() # not eating food: append head and delete tail
        return len(self.snake)-1

# LC1429. First Unique Number
class FirstUnique:
    def __init__(self, nums: List[int]):
        self.deque = deque(nums)
        self.count = Counter(nums)
    def showFirstUnique(self) -> int:  # amortized O(1)
        while self.deque and self.count[self.deque[0]] > 1:
            self.deque.popleft()
        return self.deque[0] if self.deque else -1
    def add(self, value: int) -> None:  # O(1)
        if value not in self.count:
            self.deque.append(value)
        self.count[value] += 1  # we may have values inside queue.
