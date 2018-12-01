
from random import randint


def guessingGame():
    maxNum = int(input('Until what number would you like to guess (please enter a positive number? '))
    while maxNum < 1:
        maxNum = int(input('Please enter a positive number: '))
    comp = (randint(1, maxNum))
    guess = int(input('What is your first guess? '))
    while guess != comp:
        if guess < comp:
            guess = int(input('Guess higher: '))
        elif guess > comp:
            guess = int(input('Guess lower: '))
    print('Congrats, you guessed right!')

    ans = input('Would you like to play again? (y/n) ')
    if ans == 'y':
        guessingGame()
    else:
        print('Thanks for playing!')


guessingGame()
