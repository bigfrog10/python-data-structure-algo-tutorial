################################################################################
# Given a column title as appear in an Excel sheet, return its corresponding
# column number.
#For example:
#    A -> 1
#    B -> 2
#    C -> 3
#    ...
#    Z -> 26
#    AA -> 27
#    AB -> 28
#    ...
# EASY
################################################################################

def excel_title_to_num(s: str) -> int:
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    converter = {x: idx+1 for idx, x in enumerate(labels)}

    res = 0
    for c in s:
        res = res * 26 + converter[c]

    return res


print(excel_title_to_num('A'))
print(excel_title_to_num('Z'))
print(excel_title_to_num('AA'))
print(excel_title_to_num('AB'))  # 28
print(excel_title_to_num('ZY'))  # 701
