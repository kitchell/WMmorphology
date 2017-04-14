# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:34:50 2017

@author: lindseykitchell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:01:21 2017

@author: lindseykitchell
"""

# Compare Algorithms
from parse_ev_files import parse_ev
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset

path = '/Users/lindseykitchell/Box Sync/fiberVolumes/vol_norm_ev_files/'
#path = '/Users/lindseykitchell/Box Sync/fiberVolumes/normalized_ev_values/'
#path = '/Users/lindseykitchell/Box Sync/fiberVolumes/ev_files/'

# parse the .ev files
shapeDNA_df, Eigenvalues, labels = parse_ev(path,50)
X = Eigenvalues.tolist()
Y = labels

# prepare configuration for cross validation test harness
seed = 5
# prepare models
models = []
models.append(('LR', LogisticRegression(C=.5, solver='newton-cg', fit_intercept=True, multi_class='ovr', warm_start=True)))
models.append(('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto',n_components=2)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=5, algorithm='ball_tree', p=1)))
models.append(('CART', DecisionTreeClassifier(max_features = 'log2', splitter='best', criterion = 'entropy', min_samples_leaf = 10)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf', C=3, probability=True, decision_function_shape='ovo')))
models.append(('RF', RandomForestClassifier(n_estimators=50, criterion='entropy', max_features='log2', min_samples_leaf=10)))
models.append(('ABC', AdaBoostClassifier(n_estimators=50)))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print cv_results
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()