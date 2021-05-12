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
    def showFirstUnique(self) -> int:
        while self.deque and self.count[self.deque[0]] > 1:
            self.deque.popleft()
        return self.deque[0] if self.deque else -1
    def add(self, value: int) -> None:
        if value not in self.count:
            self.deque.append(value)
        self.count[value] += 1  # we may have values inside queue.

# LC622. Design Circular Queue
class MyCircularQueue: # array based
    def __init__(self, k: int):  # double linked list
        self.capacity = k
        self.count = 0  # how many elements in the queue
        self.queue = [0]*k
        self.headIndex = 0  # front element index
        # tail element index = (head + count - 1) % capa
    def enQueue(self, value: int) -> bool:
        if self.count == self.capacity: return False
        tail = (self.headIndex + self.count) % self.capacity
        self.queue[tail] = value
        self.count += 1
        return True
    def deQueue(self) -> bool:
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
    def isEmpty(self) -> bool:
        return self.count == 0
    def isFull(self) -> bool:
        return self.count == self.capacity
