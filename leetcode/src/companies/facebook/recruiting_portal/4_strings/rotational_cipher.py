# Rotational Cipher
# One simple way to encrypt a string is to "rotate" every alphanumeric character by a certain amount. Rotating a character means replacing it with another character that is a certain number of steps away in normal alphabetic or numerical order.
# For example, if the string "Zebra-493?" is rotated 3 places, the resulting string is "Cheud-726?". Every alphabetic character is replaced with the character 3 letters higher (wrapping around from Z to A), and every numeric character replaced with the character 3 digits higher (wrapping around from 9 to 0). Note that the non-alphanumeric characters remain unchanged.
# Given a string and a rotation factor, return an encrypted string.
# Signature
# string rotationalCipher(string input, int rotationFactor)
# Input
# 1 <= |input| <= 1,000,000
# 0 <= rotationFactor <= 1,000,000
# Output
# Return the result of rotating input a number of times equal to rotationFactor.
# Example 1
# input = Zebra-493?
# rotationFactor = 3
# output = Cheud-726?
# Example 2
# input = abcdefghijklmNOPQRSTUVWXYZ0123456789
# rotationFactor = 39
# output = nopqrstuvwxyzABCDEFGHIJKLM9012345678
def rotationalCipher(input, rotation_factor):
  # Write your code here
  lower = 'abcdefghijklmnopqrstuvwxyz'
  ls = set(lower)
  upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  us = set(upper)
  digit = '0123456789'
  ds = set(digit)

  text = input
  k = rotation_factor
  ret = []
  for c in text:  # O(n)
    if c in ls:
      i = ord(c) - ord('a')
      i = (i + k) % 26
      ret.append(lower[i])
    elif c in us:
      i = ord(c) - ord('A')
      i = (i + k) % 26
      ret.append(upper[i])
    elif c in ds:
      i = int(c)
      i = (i + k) % 10
      ret.append(str(i))
    else:
      ret.append(c)

  return ''.join(ret)

print(rotationalCipher("All-convoYs-9-be:Alert1.", 4)) # "Epp-gsrzsCw-3-fi:Epivx5."
print(rotationalCipher("abcdZXYzxy-999.@", 200)) # "stuvRPQrpq-999.@"
