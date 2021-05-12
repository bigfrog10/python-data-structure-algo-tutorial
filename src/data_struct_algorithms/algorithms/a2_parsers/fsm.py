# 65. Valid Number
def isNumber1(self, s):
        s = s.strip()
        met_dot = met_e = met_digit = False
        for i, char in enumerate(s):
            if char in '+-':
                if i > 0 and s[i-1] not in 'Ee': return False
            elif char == '.':
                if met_dot or met_e: return False
                met_dot = True
            elif char in 'Ee':
                if met_e or not met_digit: return False
                met_e, met_digit = True, False
            elif char.isdigit(): met_digit = True
            else: return False
        return met_digit # must've see digits, others are optional.

def isNumber(self, s: str) -> bool:
    import re
    #Example:               +-     1 or 1. or 1.2 or .2   e +- 1
    # ? - 0 or 1 match, +: 1 or more, *: 0 or more
    engine = re.compile(r"^[+-]?((\d+\.?\d*)|(\d*\.?\d+))(e[+-]?\d+)?$")
    return engine.match(s.strip())  # i prefer this over putting more things (\S*) in regex

def isNumber(self, s: str) -> bool:
    transitions = {
        'start': {'.': 'dot', '+': 'sign1', '-': 'sign1', 'd': 'D1'},
        'sign1': {'.': 'dot', 'd': 'D1'},
        'dot': {'d': 'D2'},
        'D1': { 'd': 'D1', '.': 'D2', 'E': 'Exp', 'e': 'Exp'},
        'D2': {'d': 'D2', 'E': 'Exp', 'e': 'Exp'},
        'Exp': {'+': 'sign2', '-': 'sign2', 'd': 'D3'},
        'sign2': {'d': 'D3'},
        'D3': {'d': 'D3'}
    }
    terminals = {'D1', 'D2', 'D3'}

    state = 'start'
    for c in s:
        if c.isdigit(): c = 'd'
        tr = transitions[state]
        if c not in tr: return False
        else: state = tr[c]
    return state in terminals

