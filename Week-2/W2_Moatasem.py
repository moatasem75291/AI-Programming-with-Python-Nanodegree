import sys


def main(inputOne, inputTwo):
    try:
        inputOne, inputTwo = float(inputOne), float(inputTwo)
        print(
            inputOne + inputTwo,
            inputOne - inputTwo,
            inputOne * inputTwo,
            end=", ",
            sep=", ",
        )
        try:
            print(f"{int((inputOne / inputTwo) * 100)}%")
        except ZeroDivisionError:
            print("Error: Division by zero, try again.")
    except:
        checkInputOne, checkInputTwo = (
            isinstance(inputOne, str) and not inputOne.isdigit(),
            isinstance(inputTwo, str) and not inputTwo.isdigit(),
        )  # Check if the inputs are strings but not numbers.
        if checkInputOne and checkInputTwo:
            print(f"{inputOne} {inputTwo}, Other operations are not applicable.")
        else:
            if not checkInputOne:
                print(f"{inputOne}{inputTwo}, {int(inputOne)*inputTwo}")
            else:
                print(f"{inputTwo}{inputOne}, {inputOne*int(inputTwo)}")


print("Enter two inputs: ", "Enter c to exit.")
while True:
    inputOne, inputTwo = input(), input()

    if inputOne == "c" or inputTwo == "c":
        sys.exit()

    main(inputOne, inputTwo)
