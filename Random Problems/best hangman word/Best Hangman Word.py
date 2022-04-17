import pandas as pd


# #define file names
# working directory where everything is saved
wd = "C:/Users/lroldanwaa001/Desktop/Data Analytics/Random Problems/best hangman word/"


# name of files
wordsText = "google words.txt"
freqText = "letter frequency.csv"
minWordLength = 3
# importing the words and letter frequency datasets
words = pd.read_csv(wd + wordsText, sep=";")
freq = pd.read_csv(wd + freqText, sep=";")


# convert all letters to lower case
freq['letter'] = freq['letter'].str.lower()

# turn into dictionairy
freq.set_index('letter', inplace=True)
scores = freq.to_dict()['frequency']

# get the unique letters in all words
words['unique'] = ["".join(set(x)) for x in words['words'].astype(str)]

# generate a column for the length of all unique letters in each word
words['length unique'] = words['unique'].str.len()

# filter out short words
words = words[words['length unique'] >= minWordLength]

# #freq score is the sum of the frequency score
# calculate the score per word


def wordScore(word):
    return sum(scores[letter] for letter in word)


# apply the function for each word
words['freq score'] = [wordScore(x) for x in words['words'].astype(str)]

# testing other metrics
words['avg score per letter'] = words['freq score'] / words['length unique']

# sort the values of the dataframe with the smallest score first
words.sort_values(by='freq score', inplace=True)
