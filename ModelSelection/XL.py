
# coding: utf-8

# In[84]:

import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pydotplus
import math
import sys
from random import sample
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sklearn
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.learning_curve import learning_curve
from __future__ import division # ensures that default division is real number division
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'matplotlib')
mpl.rc('figure', figsize=[10,6]) 


# In[56]:

df = pd.read_csv('wdbc.data.csv',header = None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df.head()


# In[80]:

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[81]:

X = df.values[:,2:]
y = df['class']
title = "Learning Curves (Logistic Regression)"
estimator = linear_model.LogisticRegression()

plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=5)

plt.show()


# In[87]:

#find best max_depth for decision tree
best = 0
for i in range(25):
    kf = cross_validation.KFold(n=X.shape[0],n_folds=5)
    score = []
    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtree.fit(X_train,y_train)
        score.append(dtree.score(X_test, y_test))
    if(np.mean(score)>best):
        best = np.mean(score)
        bestDepth = i+1
print bestDepth

    
#find better criterion
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=bestDepth)
kf = cross_validation.KFold(n=X.shape[0],n_folds=5)
score = []
for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtree.fit(X_train,y_train)
        score.append(dtree.score(X_test, y_test))


dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=bestDepth)
kf = cross_validation.KFold(n=X.shape[0],n_folds=5)
score1 = []
for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtree.fit(X_train,y_train)
        score1.append(dtree.score(X_test, y_test))

if(np.mean(score)>np.mean(score1)):
    bestCriterion = 'entropy'
else:
    bestCriterion = 'gini'
print bestCriterion


# In[88]:

title = "Learning Curves (Decision Tree)"
estimator = tree.DecisionTreeClassifier(criterion=bestCriterion, max_depth=bestDepth)

plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=5)

plt.show()


# In[89]:

#find best n_neighbors for k-Nearest neighbors
best = 0
for i in range(20):
    kf = cross_validation.KFold(n=X.shape[0],n_folds=5)
    score = []
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh.fit(X_train,y_train)
        score.append(neigh.score(X_test, y_test))
    if(np.mean(score)>best):
        best = np.mean(score)
        bestNeighbors = i+1
print bestNeighbors

    
#find better p for k-Nearest neighbors
best=0
for j in range(5):
    kf = cross_validation.KFold(n=X.shape[0],n_folds=5)
    score = []
    neigh = KNeighborsClassifier(n_neighbors=bestNeighbors,p = j+1)
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh.fit(X_train,y_train)
        score.append(neigh.score(X_test, y_test))
    if(np.mean(score)>best):
        best = np.mean(score)
        bestP = j+1
print bestP


# In[92]:

title = "Learning Curves (k-Nearest Neighbors)"
estimator = KNeighborsClassifier(n_neighbors=bestNeighbors,p = bestP)

plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01), cv=5)

plt.show()


# In[ ]:



