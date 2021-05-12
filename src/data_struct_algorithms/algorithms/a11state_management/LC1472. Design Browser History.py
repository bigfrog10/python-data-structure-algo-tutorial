# LC1472. Design Browser History
class BrowserHistory:
    def __init__(self, homepage: str):
        self.hist = [homepage]
        self.idx = 0
    def visit(self, url: str) -> None:
        self.hist = self.hist[:self.idx+1] # clear forward hist
        self.hist.append(url)
        self.idx = len(self.hist) - 1
    def back(self, steps: int) -> str:
        self.idx -= steps
        if self.idx < 0: self.idx = 0
        return self.hist[self.idx]
    def forward(self, steps: int) -> str:
        self.idx += steps
        if self.idx >= len(self.hist):
            self.idx = len(self.hist) - 1
        return self.hist[self.idx]
