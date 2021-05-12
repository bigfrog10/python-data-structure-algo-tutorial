# (a*ba*ba*b)*
# (a*ba*b)*

class FSM:
    def __init__(self):
        self.cs = 1  # 1, 2, 3

    def process(self, s):
        for c in s:
            if c == 'a':
                pass # not change to state
            elif c == 'b':
                self.cs = self.cs + 1
                if self.cs > 3:
                    self.cs = self.cs % 3
            else:
                return False


