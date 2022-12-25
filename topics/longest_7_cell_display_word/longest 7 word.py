# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:29:11 2018

@author: lroldanwaa001
"""
# ## dataframe method
import pandas as pd

# import the words and turn it into a dataframe
words = pd.read_csv("C:/Users/lroldanwaa001/Desktop/Data Analytics/longest 7 cell display word/words.txt",  sep=',')

# create list of forbidden signs
forbidden = ['k', 'm', 's', 'v', 'w', 'x', 'y', 'z', 'ethane', 'amine', 'ical',
            'phic', 'tion', 'ing', 'e']

# go through each word and check if it is longer and if it contains any forbidden characters
cont = words['a'].str.contains('|'.join(forbidden))
cont = [not i for i in cont]
words = words[cont]
words['word length'] = words['a'].str.len()
words.sort_values(by = 'word length', ascending = False, inplace = True)



# ##list method
"""
Created on Tue Oct 16 11:29:11 2018

@author: lroldanwaa001

import pandas as pd

#import the words and turn it into a list
words = list(pd.read_csv("C:/Users/lroldanwaa001/Desktop/Data Analytics/longest 7 cell display word/words.txt",  sep=',')['a'])

#create forbidden signs
forbidden = ['k', 'm', 's', 'v', 'w', 'x', 'y', 'z', 'ethane', 'amine', 'ical', 'phic', 'tion', 'ing']
allowedwords = []

#go through each word and check if it is longer and if it contains any forbidden characters
for x in words:
    if any(elem in str(x) for elem in forbidden):
        continue
    else:
        allowedwords.insert(0, x)

df = pd.DataFrame(data={'words': allowedwords})
df['word length'] = df['words'].str.len()
df.sort_values(by = 'word length', axis = 0, ascending = False, inplace = True)
"""

###list method only longest word
"""
import pandas as pd

#import the words and turn it into a list
words = list(pd.read_csv("C:/Users/lroldanwaa001/Desktop/Data Analytics/longest 7 cell display word/words.txt",  sep=',')['a'])

#create forbidden signs
forbidden = ['k', 'm', 's', 'v', 'w', 'x', 'y', 'z', 'ethane', 'amine', 'ical', 'phic', 'tion', 'ing']
longestword = " "

#go through each word and check if it is longer and if it contains any forbidden characters
for x in words:
    if len(longestword) < len(str(x)):
        if any(elem in str(x) for elem in forbidden):
            continue
        else:
            longestword = str(x)
    else:
        continue
"""
