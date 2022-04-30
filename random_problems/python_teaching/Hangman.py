realWord = input('player 1 enter a word: ').lower()
progress = '-' * len(realWord)
print('\n'*10000)
print(progress)
lives = 12
usedLetters = []

while progress != realWord and lives > 0:
  guess = input('enter a letter or a word (%s lives left)' % lives)
  old = progress
  if len(guess) == 1:
    progress = ''.join([x if guess == x else y for x, y in zip(realWord, progress)])
  elif guess == realWord:
    progress = realWord
  
  if old == progress and guess not in usedLetters and len(guess) > 0:
    lives -= 1
  print(progress) 
  usedLetters.append(guess)
  
if lives > 0: print('You won!')
else: print('Game over, the word was: ' + realWord)