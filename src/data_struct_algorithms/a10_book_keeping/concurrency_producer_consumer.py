# LC1188. Design Bounded Blocking Queue
import threading
from collections import deque
from threading import Lock

class BoundedBlockingQueue(object):
    def __init__(self, capacity: int):
        self.en, self.de = Lock(), Lock()
        self.q = deque()
        self.capacity = capacity
        self.de.acquire()
    def enqueue(self, element: int) -> None:
        self.en.acquire()
        self.q.append(element)
        if len(self.q) < self.capacity: self.en.release()
        if self.de.locked(): self.de.release()
    def dequeue(self) -> int:
        self.de.acquire()
        val = self.q.popleft()
        if len(self.q): self.de.release()
        if val and self.en.locked(): self.en.release()
        return val
    def size(self) -> int:
        return len(self.q)

import random
import time
bbq = BoundedBlockingQueue(10)
def produce(bbq):
    for i in range(10):
        r = random.randint(1, 3)
        time.sleep(r)
        bbq.enqueue(r)
        print(f'produce: {i} - {r}')

def consume(bbq):
    for i in range(10):
        r = random.randint(1, 3)
        time.sleep(r)
        q = bbq.dequeue()
        print(f'consume: {i} - {q}')

t1 = threading.Thread(target=produce, args=(bbq,))
t2 = threading.Thread(target=consume, args=(bbq,))
t1.start()
t2.start()
t1.join()
t2.join()
print(f'{bbq.size()}')
