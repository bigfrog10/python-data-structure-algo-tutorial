# LC277. Find the Celebrity
def findCelebrity(self, n: int) -> int:
    # question: does a know b? if yes, then rule out a, if no, then rule out b
    keep = 0
    for i in range(1, n):
        if knows(keep, i): keep = i
        # else leave keep as is.
    for i in range(n):
        if not knows(i, keep) or (i != keep and knows(keep, i)):
            return -1
    return keep



# LC1041. Robot Bounded In Circle
def isRobotBounded(self, instructions: str) -> bool:
    # north = 0, east = 1, south = 2, west = 3
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    x = y = 0 # Initial position is in the center
    face = 0 # facing north
    for i in instructions:
        if i == "L": face = (face + 3) % 4
        elif i == "R": face = (face + 1) % 4
        else:
            x += dirs[face][0]
            y += dirs[face][1]
    return (x == 0 and y == 0) or face != 0
