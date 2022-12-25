# -*- coding: utf-8 -*-
# import all packages
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np


# set the working directory so for easier file loading and saving
wd = "C:/Users/lroldanwaa001/Desktop/Data Analytics/Random Problems/Scraping/"

# search method copied from the web and adapted for this use case
class google:
    @classmethod
    #get the movie urls
    def search(self, search):
        page = requests.get("http://www.google.nl/search?q="+search)
        soup = BeautifulSoup(page.content, "html.parser")
        links = soup.find_all("a",href=re.compile("(?<=/url\?q=)(htt.*://.*)"))
        urls = [re.split(":(?=http)",link["href"].replace("/url?q=","")) for link in links]
        return [url for url in urls if 'webcache' not in url]

# scrape the page for various attributes of the movie
def movieAttributes(adres):
    #go to the page and extract the info
    page = requests.get(adres)
    soup = BeautifulSoup(page.content, "html.parser")
        
    #find and clean genres
    info = str(soup.find('script', attrs={'type': 'application/ld+json'}))
    try:
        genre = (info.split('"genre": ')[1]).split(']')[0].strip()
        genre = [i for i in genre.split('"') if i.isalpha()]
        genre = ','.join(str(x) for x in genre)
    except:
        genre = ''
    
    #get the rating count and the imdb rating
    try:
        ratingCount = int((info.split('"ratingCount": ')[1]).split(',')[0].strip())
    except:
        ratingCount = np.nan
    
    try:
        ratingImdb = float((info.split('"ratingValue": "')[1]).split('"')[0].strip())
    except:
        ratingImdb = np.nan
    
    
    #get the metascore for good, mixed and bad reviews
    try:
        metascore = str(soup.find('div', attrs={'class': 'metacriticScore score_mixed titleReviewBarSubItem'}))
        metascore = int(metascore.split('<span>')[1].split('</span>')[0])
    except:
        metascore = np.nan
    
    if np.isnan(metascore):         
        try:
            metascore = str(soup.find('div', attrs={'class': 'metacriticScore score_favorable titleReviewBarSubItem'}))
            metascore = int(metascore.split('<span>')[1].split('</span>')[0])
        except:
            metascore = np.nan
           
    if np.isnan(metascore):         
        try:
            #get the metascore and number of reviews
            metascore = str(soup.find('div', attrs={'class': 'metacriticScore score_unfavorable titleReviewBarSubItem'}))
            metascore = int(metascore.split('<span>')[1].split('</span>')[0])
        except:
            metascore = np.nan
         
    #get year of release + name 
    try: 
        year  = int(str(soup.find('title')).split('- IMDb')[0].split('(')[1][0:4])
    except:
        year = np.nan
    name = str(soup.find('title')).split('(')[0].split('<title>')[1]
    results = [name, genre, ratingCount, ratingImdb, metascore, year]
    return results

    
# prapare the movies for the google search
def preptxt(x):
    x = x.replace(' ','+') + '+imdb' 
    x = x.replace('++', '+')
    return x

# get the urls of the list of all the movies
def getUrls(lst):
    urlList = []
    i=0
    
    for title in lst:
        i+=1
        # add a counter to view progress
        print(str(i) + ' out of ' + str(len(lst)) + ' movies done')
        urlList.append(google.search(title)[0][0].split('&sa=')[0])
    return urlList

# get the attributes for the list of all the movies
def getAttibutes(lst):
    attributes = [[]]
    i = 0
    
    for url in lst:
        attributes.append(movieAttributes(url))
        print(attributes[i])
        i +=1
        # add a counter to view progress
        if i % 5 == 0:
            print(str(i) + ' out of ' + str(len(lst)) + ' movies done')
    del attributes[0]
    return pd.DataFrame(attributes, columns=['IMDb name', 'genre', 'IMDb rating count', 'IMDb rating', 'metascore', 'year'])




''' use this for timing chuncks of code
import timeit
start_time = timeit.default_timer()
timeit.default_timer() - start_time
'''

# importing the list of movies that we want
movies = pd.read_excel(wd + "Ratings.xlsx", sheet='Sheet 1')

#prepare for looking them up on google
movieList = list(movies['Movie name'])
movieList = [preptxt(title) for title in movieList]

# add the links to our dataframe
movies['url'] = getUrls(movieList)

# retrieve all the attributes
allAttributes = getAttibutes(movies['url'])
movies = pd.concat([movies, allAttributes], axis=1)


# get all the genres as dummies
genreDummies = movies['genre'].str.get_dummies(sep=',')
movies = pd.concat([movies, genreDummies], axis=1)
# also get rid of the useless/incorrect columns
movies.drop(movies.columns[31:35], axis=1, inplace=True)
movies.drop(['Person'], axis = 1, inplace = True)
test = movies.drop(movies.iloc[:, 31:35], axis=0)


# save the hard work
movies.to_csv(wd + 'complete movie database.csv', sep = ';')




