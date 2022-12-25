# The Best Hangman Word 

The top 10 best hangman words, according to my analysis, are:

01. jump
02. fuzzy
03. buck
04. fuck
05. bulk
06. july
07. flux
08. judy
09. luck
10. quiz

My definition of a good hangman word is as follows:

01.  It needs to be a word most people know and understand in english.
02.  It needs to contain at least 4 letters.
03.  It would take the longest to guess.

Lets tackle these one by one.

## Getting a Word List

We derive our main word list from the following [Google's Trillion Word Corpus](https://books.google.com/ngrams/info). This corpus compiles the 10k most common english words by usage frequency. We will use this subset as our definition of _words most people know and understand in english_. If this requirement is not included we would end up with words like "pneumonoultramicroscopicsilicovolcanoconiosis". Picking these words leads to nobody wanting to play with you again. This would be a sad and undesirable outcome. You want people to fail at guessing your word, but have them think they should have been able to guess it if they had played better.

## Filtering Short Words

Most letters are considered complete words by themselves. This means that the hardest words are just single letters as you have the smallest chance of guessing it correctly before your lives run out. In order to preserve the spirit of the game a word is only considered valid if it contains 4 or more unique letters.

## Guessing Strategy

How hard a word is depends on the strategy that the guesser is using. In order to keep this analysis simple we will assume:
* The guesser will approximately guess letters in order of most to least occurring.
* The guesser does not know our approach of how to choose the best hangman word.

## Scoring Words 

Given our requirements, we need a method for ranking which words are better than others. The way we do this is my summing the letter frequency of all the unique letters in a word. The words with the lowest score will be the best words. This ensures that:
* Words that contain uncommon letters will be scored as better.
* Words that use the least amounts of letters are scored as better.

The latter point is important as adding more letters just gives the guesser more opportunities to guess correctly. 

If we take all of these points into account we get the list mentioned at the beginning. Please find the code [here](https://github.com/loicrw/Loic-Roldan-Waals/blob/master/random_problems/best_hangman_word/Best%20Hangman%20Word.py)