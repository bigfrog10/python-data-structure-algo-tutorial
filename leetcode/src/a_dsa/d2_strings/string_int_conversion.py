
# LC12. Integer to Roman
def intToRoman(self, num: int) -> str:
    integer = { 1000: "M", 900: "CM", 500: "D", 400: "CD", 100: "C",
        90: "XC", 50: "L", 40: "XL", 10: "X", 9: "IX", 5: "V", 4: "IV", 1: "I"}
    s = ""
    for k, v in integer.items():  # insertion order
        d, num = divmod(num, k)
        s += d*v
    return s

# LC8. String to Integer (atoi)
def myAtoi(self, s: str) -> int:
    s = s.strip()
    if len(s) == 0: return 0
    sign = -1 if s[0] == '-' else 1
    if s[0] in ['-', '+']: s = s[1:]
    ret, i = 0, 0
    while i < len(s) and s[i].isdigit():
        ret = ret*10 + ord(s[i]) - ord('0')
        i += 1
    return max(-2**31, min(sign * ret, 2**31-1))

# LC13. Roman to Integer
def romanToInt(self, s: str) -> int:
    values = { "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = values.get(s[-1])
    for i in reversed(range(len(s) - 1)):
        if values[s[i]] < values[s[i + 1]]: total -= values[s[i]]
        else: total += values[s[i]]  # right most is always add
    return total

# LC273. Integer to English Words
def numberToWords(self, num):
    lessThan20 = ["","One","Two","Three","Four","Five","Six","Seven","Eight","Nine", "Ten",
                  "Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen", "Seventeen","Eighteen","Nineteen"]
    tens = ["","","Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]
    zero3 = ["","Thousand","Million","Billion"]

    def hundreds(num):
        if num == 0: return ""  # need this to prevent extra space in next line
        elif num < 20: return lessThan20[num] + " "
        elif num < 100: return tens[num // 10] + " " + hundreds(num % 10)
        else: return lessThan20[num // 100] + " Hundred " + hundreds(num % 100)

    if num == 0: return "Zero"
    ret = ''
    for i in range(len(zero3)):
        if num % 1000 != 0: ret = hundreds(num % 1000) + zero3[i] + ' ' + ret
        num = num // 1000
    return ret.strip()

# LC171. Excel Sheet Column Number
def titleToNumber(self, s: str) -> int:
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    converter = {x: idx+1 for idx, x in enumerate(labels)}
    res = 0
    for c in s: res = res * 26 + converter[c]
    return res
