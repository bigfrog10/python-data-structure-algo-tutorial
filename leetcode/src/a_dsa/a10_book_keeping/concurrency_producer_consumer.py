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


# LC1242. Web Crawler Multithreaded
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
class Solution:
    def __init__(self):
        self.queue = Queue()
        self.visited = set()
        self.result = []
        self.lock = RLock()
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        self.visited = set([startUrl])
        self.queue.put(startUrl)
        with ThreadPoolExecutor() as e:
            for _ in range(8):
                e.submit(self.run, htmlParser)  # start workers
        return self.result
    def run(self, htmlParser):
        try:
            while True:
                # use timeout to quit, or pass None here. in seconds
                item = self.queue.get(block=True, timeout=0.05)
                if not item: break
                self.result.append(item)
                for url in htmlParser.getUrls(item):
                    with self.lock:  # lock don't do anything due to GIL
                        if self.hostname(url) != self.hostname(item):
                            continue
                        if url in self.visited: continue
                        self.visited.add(url)
                        self.queue.put(url)
        except: pass
    def hostname(self, url):
        return url.split('//')[1].split('/')[0]

class Solution {
    private Set<String> visited = ConcurrentHashMap.newKeySet();
    private String hostName;
    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        hostName = getHostname(startUrl);
        traverse(startUrl, htmlParser);
        return new ArrayList(visited);
    }
    private static String getHostname(String url) {
        return url.split("/")[2];
    }
    private void traverse(String startUrl, HtmlParser htmlParser) {
        visited.add(startUrl);
        htmlParser.getUrls(startUrl)
        .parallelStream()
        .filter(url -> getHostname(url).equals(hostName))
        .filter(url -> !visited.contains(url))
        .forEach(url -> traverse(url, htmlParser));
    }
}













