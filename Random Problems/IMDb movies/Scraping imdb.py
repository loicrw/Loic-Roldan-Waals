# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np



wd = "C:/Users/lroldanwaa001/Desktop/Data Analytics/Random Problems/Scraping/"

# copied from the webs
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
    
    
    #get the metascore and number of reviews
    try:
        metascore = str(soup.find('div', attrs={'class': 'metacriticScore score_mixed titleReviewBarSubItem'}))
        metascore = int(metascore.split('<span>')[1].split('</span>')[0])
    except:
        metascore = np.nan
    
    if np.isnan(metascore):         
        try:
            #get the metascore and number of reviews
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

# get the urls of the list of movies
def getUrls(lst):
    urlList = []
    i=0
    for title in lst:
        i+=1
        print(str(i) + ' out of ' + str(len(lst)) + ' movies done')
        urlList.append(google.search(title)[0][0].split('&sa=')[0])
    return urlList

# get the attributes for the list of movies
def getAttibutes(lst):
    attributes = [[]]
    i = 0
    
    for url in lst:
        attributes.append(movieAttributes(url))
        print(attributes[i])
        i +=1
        if i % 5 == 0:
            print(str(i) + ' out of ' + str(len(lst)) + ' movies done')
    del attributes[0]
    return pd.DataFrame(attributes, columns=['IMDb name', 'genre', 'IMDb rating count', 'IMDb rating', 'metascore', 'year'])




'''
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








### now the machine learning part
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

#get the scraping data
wd = "C:/Users/lroldanwaa001/Desktop/Data Analytics/Random Problems/Scraping/"
movies = pd.read_csv(wd + "complete movie database.csv", sep = ';').drop(['Unnamed: 0'], axis = 1)

#split into attributes and target value
X = movies[movies.columns[5:]]
y = movies['Rating']

#deal with missing values by indicating them with a column
X['Missing metascore'] = list(X['metascore'].isna())
X['Missing IMDb score'] = list(X['IMDb rating'].isna())
X['Missing year'] = list(X['year'].isna())

#impute the missing values with the mean values
colMissing = ['metascore', 'IMDb rating', 'year']
for i in colMissing:
    X[i].fillna(np.mean(X[i]), inplace = True)



X = X[['IMDb rating','year','metascore','Missing year', 'Missing metascore', 'Action', 'Animation', 'Mystery', 'PG', 'R']]

# determining the best parameters for the ML models
parameters = {'gamma': [0.001, 0.0001, 0.0005, 0.003, 0.005], 'C': [0.01, 0.03, 0.001, 0.003], 'kernel': ['linear', 'rbf']}
sm = svm.SVR()
gridsm = RandomizedSearchCV(sm, param_distributions = parameters, n_iter = 40, cv = 3, scoring = 'r2')
gridsm.fit(X, y)
gridsm.best_params_
scoresm = gridsm.best_score_


parameters = {'n_iter': [10,1,5,15,12]}
nb = BayesianRidge()
gridnb = RandomizedSearchCV(nb, param_distributions = parameters, n_iter = 5, cv = 3, scoring = 'r2')
gridnb.fit(X,y)
scorenb = gridnb.best_score_
gridnb.best_params_
scoresnb = (cross_val_score(nb, X, y, cv = 3, scoring = 'r2')).mean()


parameters = {'fit_intercept': [1,0], 'n_jobs': [1,5,10,50,100]}
lr = LinearRegression()
gridlr = RandomizedSearchCV(lr, param_distributions = parameters, n_iter = 10, cv = 3, scoring = 'r2')
gridlr.fit(X,y)
scorelr = gridlr.best_score_
gridlr.best_params_
scoreslr = (cross_val_score(lr, X, y, cv = 3, scoring = 'r2')).mean()


parameters = {'n_estimators': [1,2,3,5,4,6,7], 'max_depth': [1], 'max_features': [None], 'max_leaf_nodes':[5,4,6]}
rf = RandomForestRegressor()
gridrf = RandomizedSearchCV(rf, param_distributions = parameters, n_iter = 21, cv = 3, scoring = 'r2')
gridrf.fit(X,y)
scorerf = gridrf.best_score_
gridrf.best_params_
scoresrf = (cross_val_score(rf, X, y, cv = 3, scoring = 'r2')).mean()


parameters = {'learning_rate': [0.03,0.05,0.07], 'n_estimators': [30,50,70], 'max_features': [None], 'max_leaf_nodes':[5,4,6], 'max_depth': [1,2,5,10]}
gb = GradientBoostingRegressor()
gridgb = RandomizedSearchCV(gb, param_distributions = parameters, n_iter = 108, cv = 3, scoring = 'r2')
gridgb.fit(X,y)
scoregb = gridgb.best_score_
gridgb.best_params_
gb = GradientBoostingRegressor(learning_rate = 0.01, n_estimators = 75)
scoresgb = (cross_val_score(gb, X, y, cv = 3, scoring = 'r2')).mean()


def multipleAgg(X, y):
    aggscores = []
    for i in range(300):
        xtrain, xtest, ytrain, ytest= train_test_split(X, y, test_size=0.3, random_state=i)
        sm = svm.SVR(kernel='linear',gamma=0.001, C=0.03).fit(xtrain, ytrain)
        #nb = BayesianRidge(n_iter=1).fit(xtrain, ytrain)
        lr = LinearRegression(n_jobs=1, fit_intercept=1).fit(xtrain, ytrain)
        #rf = RandomForestRegressor(n_estimators = 200, max_depth = 1, max_leaf_nodes=6, max_features=None).fit(xtrain, ytrain)
        #gb = GradientBoostingRegressor(learning_rate = 0.05, n_estimators = 30, max_leaf_nodes = 5, max_features = None).fit(xtrain, ytrain)
        
        predsm = sm.predict(xtest)
        #prednb = nb.predict(xtest)
        predlr = lr.predict(xtest)
        #predrf = rf.predict(xtest)
        #predgb = gb.predict(xtest)
        
        aggpred = [sum(e)/len(e) for e in zip(*[predsm, predlr])]
        aggscores.append(r2_score(ytest, aggpred))
    return np.mean(aggscores)

ascoreAgg = multipleAgg(X, y)


testingRandState = multipleAgg(X, y)


from matplotlib import pyplot as plt
data = [x for x in testingRandState if x > 0]
np.median(data)
np.mean(data)

# fixed bin size
bins = np.arange(-100, 100, 0.03) # fixed bin size

plt.xlim([min(data)-0.05, max(data)+0.05])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Distribution of r2 scores for multipleAgg')
plt.xlabel('r2 score')
plt.ylabel('count')



(movies['metascore'].corr(movies['Rating']))**2

movies['metascore']
r2_score(y, X['metascore'])
y.corr(X['IMDb rating'])**2


plt.scatter(y,X['IMDb rating'])
plt.scatter(y,X['metascore'])
plt.scatter(X['IMDb rating'], X['metascore'])
X['metascore'].corr(X['IMDb rating'])
'''    
xtrain, xtest, ytrain, ytest= train_test_split(X, y, test_size=0.33, random_state=3)
sm = svm.SVR(gamma=0.01, C=0.1).fit(xtrain, ytrain)
nb = BayesianRidge().fit(xtrain, ytrain)
rf = RandomForestRegressor(n_estimators = 100, max_depth = 3).fit(xtrain, ytrain)

predsm = sm.predict(xtest)
prednb = nb.predict(xtest)
predrf = rf.predict(xtest)

aggpred = [sum(e)/len(e) for e in zip(*[predsm, prednb, predrf])]
aggscore = r2_score(ytest, aggpred)


(movies['IMDb rating'].corr(movies['Rating']))**2
'''
















