# https://stackoverflow.com/questions/35272592/how-are-finite-automata-implemented-in-code


# General solution
def process(s: str, init_state, transitions: dict, terminal_states: set):
    state = init_state
    for c in s:
        state = transitions[state][c]

    return state in terminal_states


tr = {
    1: {'a': 1, 'b': 2},
    2: {'a': 2, 'b': 3},
    3: {'a': 3, 'b': 1},
}

tm = {3}

print(process('abab', 1, tr, tm))  # True
print(process('ab', 1, tr, tm))  # False


# shortcut solution
def process(s):
    state = 1
    for c in s:
        if c == 'a':
            pass
        elif c == 'b':
            state += 1
            if state > 3:
                state = 1
        else:
            return False

    return state == 3


print(process('abab'))  # True
print(process('ab'))  # False
