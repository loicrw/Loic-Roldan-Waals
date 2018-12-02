# -*- coding: utf-8 -*-

# Imports
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
    


#get the scraped data
wd = "C:/Users/lroldanwaa001/Desktop/Data Analytics/Random Problems/IMDb movies/"
movies = pd.read_csv(wd + "complete movie database.csv", sep = ';').drop(['Unnamed: 0'], axis = 1)

#split into attributes and target value
X = movies[movies.columns[5:]]
y = movies['Rating']

#deal with missing values by indicating them with another column
X['Missing metascore'] = list(X['metascore'].isna())
X['Missing IMDb score'] = list(X['IMDb rating'].isna())
X['Missing year'] = list(X['year'].isna())

#impute the missing values with the mean values
colMissing = ['metascore', 'IMDb rating', 'year']
for i in colMissing:
    X[i].fillna(np.mean(X[i]), inplace = True)

'''
### Scaling decreases performance and is thus not commented out
# scale the data
scaler = MinMaxScaler()
scaler.fit(X)
# save the columns
cols = X.columns
X = scaler.transform(X)
X = pd.DataFrame(X, columns = cols)
'''

# after manually checking only these variables seem to have an impact on my rating of the movies, thus only these are selected
X = X[['IMDb rating','year','metascore','Missing year', 'Missing metascore', 'Action', 'Animation', 'Mystery', 'PG', 'R']]

# Determining optimal parameters for SVR; conclusion: kernel='linear',gamma=0.001, C=0.03
parameters = {'gamma': [0.001, 0.0001, 0.0005, 0.003, 0.005], 'C': [0.01, 0.03, 0.001, 0.003], 'kernel': ['linear', 'rbf']}
sm = svm.SVR()
gridsm = RandomizedSearchCV(sm, param_distributions =parameters, n_iter = 40, cv=3, scoring='r2')
gridsm.fit(X, y)
gridsm.best_params_
scoresm = gridsm.best_score_

# Determining optimal parameters for NB; conclusion: n_iter=1
parameters = {'n_iter': [10,1,5,15,12]}
nb = BayesianRidge()
gridnb = RandomizedSearchCV(nb, param_distributions =parameters, n_iter = 5, cv=3, scoring='r2')
gridnb.fit(X,y)
scorenb = gridnb.best_score_
gridnb.best_params_
scoresnb = (cross_val_score(nb, X, y, cv=3, scoring='r2')).mean()

# Determining optimal parameters for linear regression; conclusion: n_jobs=1, fit_intercept=1
parameters = {'fit_intercept': [1,0], 'n_jobs': [1,5,10,50,100]}
lr = LinearRegression()
gridlr = RandomizedSearchCV(lr, param_distributions =parameters, n_iter = 10, cv=3, scoring='r2')
gridlr.fit(X,y)
scorelr = gridlr.best_score_
gridlr.best_params_
scoreslr = (cross_val_score(lr, X, y, cv=3, scoring='r2')).mean()

# Determining optimal parameters for random forest; conclusion: n_estimators = 4, max_depth = 1, max_leaf_nodes=6, max_features=None
parameters = {'n_estimators': [1,2,3,5,4,6,7], 'max_depth': [1, 3, 10, 15, 20], 'max_features': [None], 'max_leaf_nodes':[5,4,6, 10]}
rf = RandomForestRegressor()
gridrf = RandomizedSearchCV(rf, param_distributions =parameters, n_iter = 21, cv=3, scoring='r2')
gridrf.fit(X,y)
scorerf = gridrf.best_score_
gridrf.best_params_
scoresrf = (cross_val_score(rf, X, y, cv=3, scoring='r2')).mean()

# Determining optimal parameters for gradient boosting; conclusion: learning_rate = 0.05, n_estimators = 30, max_leaf_nodes = 5, max_features = None
parameters = {'learning_rate': [0.03,0.05,0.07], 'n_estimators': [30,50,70], 'max_features': [None], 'max_leaf_nodes':[5,4,6], 'max_depth': [1,2,5,10]}
gb = GradientBoostingRegressor()
gridgb = RandomizedSearchCV(gb, param_distributions =parameters, n_iter = 108, cv=3, scoring='r2')
gridgb.fit(X,y)
scoregb = gridgb.best_score_
gridgb.best_params_
gb = GradientBoostingRegressor(learning_rate = 0.01, n_estimators = 75)
scoresgb = (cross_val_score(gb, X, y, cv=3, scoring='r2')).mean()


# determine the score when multiple models are combined and which combination of models leads to optimal performance; conclusion: sm and lr
def multipleAgg(X, y):
    aggscores = []
    
    # run the aggregated model 300 times with different random states
    for i in range(300):
        xtrain, xtest, ytrain, ytest= train_test_split(X, y, test_size=0.3, random_state=i)
        sm = svm.SVR(kernel='linear',gamma=0.001, C=0.03).fit(xtrain, ytrain)
        #nb = BayesianRidge(n_iter=1).fit(xtrain, ytrain)
        lr = LinearRegression(n_jobs=1, fit_intercept=1).fit(xtrain, ytrain)
        #rf = RandomForestRegressor(n_estimators = 4, max_depth = 1, max_leaf_nodes=6, max_features=None).fit(xtrain, ytrain)
        #gb = GradientBoostingRegressor(learning_rate = 0.05, n_estimators = 30, max_leaf_nodes = 5, max_features = None).fit(xtrain, ytrain)
        
        predsm = sm.predict(xtest)
        #prednb = nb.predict(xtest)
        predlr = lr.predict(xtest)
        #predrf = rf.predict(xtest)
        #predgb = gb.predict(xtest)
        
        aggpred = [sum(e)/len(e) for e in zip(*[predsm, predlr])]
        aggscores.append(r2_score(ytest, aggpred))
    return np.mean(aggscores), aggscores

ascoreAgg, aggscores = multipleAgg(X, y)


# visualisation for data exploration
# label all r2 scores under 0 as outliers and exclude them (less then 30 outliers
data = [x for x in aggscores if x > 0]
np.median(data)
np.mean(data)

# show a histogram of fixed bin size of the distribution of the r2 scores
bins = np.arange(-100, 100, 0.03) # fixed bin size
plt.xlim([min(data)-0.05, max(data)+0.05])
plt.hist(data, bins=bins, alpha=0.5)
plt.title('Distribution of r2 scores for multipleAgg')
plt.xlabel('r2 score')
plt.ylabel('count')


#Q what is the correlation coefficient between the metascore, the IMDb rating and my rating?
(movies['metascore'].corr(movies['Rating']))**2
(movies['IMDb rating'].corr(movies['Rating']))**2

#Q what do the scatter plots look like?
plt.scatter(y,X['IMDb rating'])
plt.scatter(y,X['metascore'])
plt.scatter(X['IMDb rating'], X['metascore'])

# 3d scatter where the color is my score (lighter is higher)
plt.scatter(X['metascore'], X['IMDb rating'], c=y)
plt.show()

''' MISCELANEOUS TRIAL VERSIONS  
xtrain, xtest, ytrain, ytest= train_test_split(X, y, test_size=0.33, random_state=3)
sm = svm.SVR(gamma=0.01, C=0.1).fit(xtrain, ytrain)
nb = BayesianRidge().fit(xtrain, ytrain)
rf = RandomForestRegressor(n_estimators = 100, max_depth = 3).fit(xtrain, ytrain)

predsm = sm.predict(xtest)
prednb = nb.predict(xtest)
predrf = rf.predict(xtest)

aggpred = [sum(e)/len(e) for e in zip(*[predsm, prednb, predrf])]
aggscore = r2_score(ytest, aggpred)
'''
















