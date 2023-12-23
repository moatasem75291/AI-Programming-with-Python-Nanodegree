import numpy as np

print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
      first task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)


# 1-function to print the sum, largest number, smallest number, average of the numbers in a list
def some_calculations(numbers):
    numbers = np.array(numbers)
    sum = numbers.sum()
    min = numbers.min()
    max = numbers.max()
    mean = numbers.mean()
    return f"Sum: {sum}\nMin: {min}\nMax: {max}\nMean: {mean}"


print(some_calculations([1, 2, 3, 4, 5]))
print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
      Second task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)


# 2-function that takes two lists and returns True if they have at least one common member
def common_names(firstNames, secondNames):
    firstNamesSet = set(firstNames)
    secondNamesSet = set(secondNames)

    result = firstNamesSet.intersection(secondNamesSet)
    if result:
        return True
    else:
        return False


print(common_names(["Tom", "Bob", "Sue", "Rachel"], ["Bob", "Susan", "Roger", "Mike"]))
print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
      Third task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)


# 3-function to combine two dictionary adding values for common keys
from collections import Counter


def combine_dictionaries(dic1, dic2):
    return dict(Counter(dic1) + Counter(dic2))


d1 = {"a": 100, "b": 200, "c": 300}
d2 = {"a": 300, "b": 200, "d": 400}
print(combine_dictionaries(d1, d2))
print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
      Fourth task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)

# 4- function which iterates the integers from 1 to 100. For multiples of three print "Fizz" instead of the number and for the multiples of five print "Buzz". For numbers  which are multiples of both three and five print "FizzBuzz"


def print_numbers():
    for i in range(1, 101):
        if i % 3 == 0 and i % 5 == 0:
            print(f"{i} FizzBuzz")
        elif i % 3 == 0:
            print(f"{i} Fizz")
        elif i % 5 == 0:
            print(f"{i} Buzz")
        else:
            print(i)


print_numbers()
print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
      Fifth task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)
# 5- function to check the validity of password input by users.
# Validation :
# • At least 1 letter between [a-z] and 1 letter between [A-Z].
# • At least 1 number between [0-9].
# • At least 1 character from [$#@].
# • Minimum length 6 characters.
# • Maximum length 16 characters
import re


def check_password(password):
    if len(password) < 6 or len(password) > 16:  # the password
        return False
    if not re.search("[a-z]", password):  # at least one lowercase letter
        return False
    if not re.search("[A-Z]", password):  # at least one uppercase letter
        return False
    if not re.search("[0-9]", password):  # at least one digit
        return False
    if not re.search("[$#@]", password):  # at least one special character
        return False

    return True


# Test the function
print(f"The result for this Password: Test123! , is: {check_password('Test123!')}")
print(f"The result for this Password: TEST123! , is: {check_password('TEST123!')}")
print(f"The result for this Password: Test123$ , is: {check_password('Test123$')}")
print(f"The result for this Password: te@123 , is: {check_password('te@123')}")
print(f"The result for this Password: Test1234 , is: {check_password('Test1234')}")
print(f"The result for this Password: T$4 , is: {check_password('T$4')}")

print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
      Sixth task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)


# 6- function that takes a list and returns a new list with unique elements of the first list
def unique_list(numbers):
    return list(set(numbers))


print(unique_list([1, 2, 3, 3, 3, 3, 4, 5]))

print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
      Seventh task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)


# 7-function which calculates the length of a hypotenuse of a right angle triangle given height and base. Hint: Pythagorean theorem
def calculate_hypotenuse(height, base):
    return (height**2 + base**2) ** 0.5


print(calculate_hypotenuse(5, 8))
print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
    Eighth task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)


# 8- Scrabble score:Given a word, compute the Scrabble score for that word where the values are as follows
# Letter                           Value
# A, E, I, O, U, L, N, R, S, T       1
# D, G                               2
# B, C, M, P                         3
# F, H, V, W, Y                      4
# K                                  5
# J, X                               8
# Q, Z                               10
# eg. “Cabbage” is worth 14 points
# • 3 points for C
# • 1 point for A, twice
# • 3 points for B, twice
# • 2 points for G
# • 1 point for E
def compute_scrabble_score(word):
    letter_score = {
        "A": 1,
        "E": 1,
        "I": 1,
        "O": 1,
        "U": 1,
        "L": 1,
        "N": 1,
        "R": 1,
        "S": 1,
        "T": 1,
        "D": 2,
        "G": 2,
        "B": 3,
        "C": 3,
        "M": 3,
        "P": 3,
        "F": 4,
        "H": 4,
        "V": 4,
        "W": 4,
        "Y": 4,
        "K": 5,
        "J": 8,
        "X": 8,
        "Q": 10,
        "Z": 10,
    }
    word = word.upper()
    score = sum(letter_score[letter] for letter in word)

    return score


word = "Cabbage"
print(f"{word} is worth {compute_scrabble_score(word)} points")
print(
    """
#$%#^%$&&$#%&%$&$%&$%&$&%^*$#$&
    Ninth task
#$^&$*(*$%#$^&%&^#%@^%$&^**&$$%)
"""
)


def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


# example usage
num1 = 48
num2 = 18

print(f"The GCD of {num1} and {num2} is: {gcd(num1, num2)}")
