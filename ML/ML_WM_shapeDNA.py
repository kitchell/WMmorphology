# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:46:24 2017

@author: lindseykitchell
"""

from parse_ev_files import parse_ev
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np


# get path to files
path = '/Users/lindseykitchell/Box Sync/fiberVolumes/vol_norm_ev_files/'
#path = '/Users/lindseykitchell/Box Sync/fiberVolumes/normalized_ev_values/'

# parse the .ev files
shapeDNA_df, Eigenvalues, labels = parse_ev(path, 50)
Eigenvalues = Eigenvalues.tolist()


# testing RandomForestClassifier

# first use GridSearchCV to get best parameters

kf = KFold(len(labels), 10, shuffle=True)
 
parameters = {'n_estimators':[50,75,100,500], 'criterion': ('gini', 'entropy'), 'max_features': ('auto', 'log2', None), 'min_samples_leaf':[1,5,10,50,100,200,500]}
svr = RandomForestClassifier()
clf = GridSearchCV(svr, parameters, cv=kf, n_jobs=-1)
clf.fit(Eigenvalues, labels)

print clf.best_params_
print clf.best_score_
#print clf.grid_scores_


# Then test again with the best parameters

clf = RandomForestClassifier(n_estimators=60, min_samples_split=2, criterion='entropy', max_features='log2')
kf = KFold(len(labels), 10, shuffle = True)
acc = []

for train_indices, test_indices in kf:
    data_train = [Eigenvalues[ii] for ii in train_indices]
    data_test = [Eigenvalues[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    clf.fit(data_train, labels_train)
    pred = clf.predict(data_test)
    acc.append(clf.score(data_test, labels_test))
    print labels_test

print np.mean(acc)




# testing AdaBoost

# first use GridSearchCV to get best parameters



kf = KFold(len(labels), 10, shuffle=True)
 
parameters = {'n_estimators':[25,50,75,100,500]}

svr = AdaBoostClassifier()
clf = GridSearchCV(svr, parameters, cv=kf,)
clf.fit(Eigenvalues, labels)

print clf.best_params_
print clf.best_score_
#print clf.grid_scores_


# Then test again with the best parameters

clf = AdaBoostClassifier(n_estimators=50 )
kf = KFold(len(labels), 10, shuffle = True)
acc = []

for train_indices, test_indices in kf:
    data_train = [Eigenvalues[ii] for ii in train_indices]
    data_test = [Eigenvalues[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    clf.fit(data_train, labels_train)
    pred = clf.predict(data_test)
    acc.append(clf.score(data_test, labels_test))

print np.mean(acc)





# testing SVM

kf = KFold(len(labels), 10, shuffle=True)
 
parameters = {'C':[1,2,3,4,5,10], 'probability':(True,False), 'decision_function_shape':('ovo','ovr',None)}

svr = SVC(kernel='rbf')
clf = GridSearchCV(svr, parameters, cv=kf,)
clf.fit(Eigenvalues, labels)

print clf.best_params_
print clf.best_score_
#print clf.grid_scores_


# Then test again with the best parameters

clf = SVC(kernel='rbf', C=3, probability=True, decision_function_shape='ovo')
kf = KFold(len(labels), 10, shuffle = True)
acc = []

for train_indices, test_indices in kf:
    data_train = [Eigenvalues[ii] for ii in train_indices]
    data_test = [Eigenvalues[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    clf.fit(data_train, labels_train)
    pred = clf.predict(data_test)
    acc.append(clf.score(data_test, labels_test))

print np.mean(acc)


# testing KNN

kf = KFold(len(labels), 10, shuffle=True)
 
parameters = {'n_neighbors':[2,3,4,5], 'weights':('uniform', 'distance'), 'algorithm':('ball_tree','kd_tree','brute'),'leaf_size':[5,6,7,8],'p':[1,2]}

svr = KNeighborsClassifier()
clf = GridSearchCV(svr, parameters, cv=kf, n_jobs=-1)
clf.fit(Eigenvalues, labels)

print clf.best_params_
print clf.best_score_
#print clf.grid_scores_


# Then test again with the best parameters

clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=5, algorithm='ball_tree', p=1)
kf = KFold(len(labels), 10, shuffle = True)
acc = []

for train_indices, test_indices in kf:
    data_train = [Eigenvalues[ii] for ii in train_indices]
    data_test = [Eigenvalues[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    clf.fit(data_train, labels_train)
    pred = clf.predict(data_test)
    acc.append(clf.score(data_test, labels_test))

print np.mean(acc)


#testing CART

kf = KFold(len(labels), 10, shuffle=True)
 
parameters = {'criterion':('gini', 'entropy'), 'splitter': ('best', 'random'), 'max_features':('auto', 'sqrt', 'log2', None), 'min_samples_leaf':[1,5,7,10,20]}

svr = DecisionTreeClassifier()
clf = GridSearchCV(svr, parameters, cv=kf)
clf.fit(Eigenvalues, labels)

print clf.best_params_
print clf.best_score_
#print clf.grid_scores_


# Then test again with the best parameters

clf = DecisionTreeClassifier(max_features = 'log2', splitter='best', criterion = 'gini', min_samples_leaf = 10)
kf = KFold(len(labels), 10, shuffle = True)
acc = []

for train_indices, test_indices in kf:
    data_train = [Eigenvalues[ii] for ii in train_indices]
    data_test = [Eigenvalues[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    clf.fit(data_train, labels_train)
    pred = clf.predict(data_test)
    acc.append(clf.score(data_test, labels_test))

print np.mean(acc)


# testing LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


kf = KFold(len(labels), 10, shuffle=True)
 
parameters = {'shrinkage':('auto',None), 'n_components':[2,3,4,5,6,7,10,20,45], }


svr = LinearDiscriminantAnalysis(solver='lsqr')
clf = GridSearchCV(svr, parameters, cv=kf)
clf.fit(Eigenvalues, labels)

print clf.best_params_
print clf.best_score_
#print clf.grid_scores_


# Then test again with the best parameters

clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', n_components=2)
kf = KFold(len(labels), 10, shuffle = True)
acc = []

for train_indices, test_indices in kf:
    data_train = [Eigenvalues[ii] for ii in train_indices]
    data_test = [Eigenvalues[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    clf.fit(data_train, labels_train)
    pred = clf.predict(data_test)
    acc.append(clf.score(data_test, labels_test))

print np.mean(acc)



# testing logistic regression

from sklearn.linear_model import LogisticRegression

kf = KFold(len(labels), 10, shuffle=True)
 
parameters = { 'C':[.2,.3,.5,.6,.7], 'fit_intercept':(True, False),'solver': ('newton-cg', 'lbfgs', 'sag'), 'multi_class':('ovr','multinomial'), 'warm_start':(True,False)}

#'penalty':('l1','l2'),'dual':(True, False),'multi_class':('ovr','multinomial')

svr = LogisticRegression()
clf = GridSearchCV(svr, parameters, cv=kf)
clf.fit(Eigenvalues, labels)

print clf.best_params_
print clf.best_score_
#print clf.grid_scores_


# Then test again with the best parameters

clf = LogisticRegression(C=.5, solver='newton-cg', fit_intercept=True, multi_class='ovr', warm_start=True)
kf = KFold(len(labels), 10, shuffle = True)
acc = []

for train_indices, test_indices in kf:
    data_train = [Eigenvalues[ii] for ii in train_indices]
    data_test = [Eigenvalues[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    clf.fit(data_train, labels_train)
    pred = clf.predict(data_test)
    acc.append(clf.score(data_test, labels_test))

print np.mean(acc)





from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
tsne_transf = model.fit_transform(Eigenvalues)



for i in range(len(labels)):
    if labels[i] == 1:
        plt.plot(tsne_transf[i][0],tsne_transf[i][1],'ro')
    else:
        plt.plot(tsne_transf[i][0],tsne_transf[i][1],'bo')
plt.show()

















# cross validation one way
data_train, data_test, labels_train, labels_test = train_test_split(
    Eigenvalues, labels, test_size=0.2, random_state=0)


from sklearn.grid_search import GridSearchCV

kf = KFold(len(labels), 10, shuffle = True)
 
parameters = {'n_estimators':[10,15,20,30], 'criterion': ('gini', 'entropy'), 'max_features': ('auto', 'log2', None)}
svr = RandomForestClassifier()
clf = GridSearchCV(svr, parameters, cv=kf)
clf.fit(Eigenvalues, labels)









clf = DecisionTreeClassifier()
clf.fit(data_train, labels_train)
clf.predict(data_test)

clf.score(data_test, labels_test)




clf = RandomForestClassifier()
clf.fit(data_train, labels_train)
clf.predict(data_test)

clf.score(data_test, labels_test)



clf = AdaBoostClassifier()
clf.fit(data_train, labels_train)
clf.predict(data_test)

clf.score(data_test, labels_test)





# using PCA

#
pca = PCA()
pca = pca.fit(data_train)
train_transf = pca.transform(data_train)

for e in range(5):
    for i in range(len(train_transf)):
        plt.plot(train_transf[i][e],train_transf[i][e+1],'ro')
    plt.show()


test_transf = pca.transform(data_test)

clf = DecisionTreeClassifier()
clf.fit(train_transf, labels_train)
clf.predict(test_transf)

clf.score(test_transf, labels_test)




clf = RandomForestClassifier()
clf.fit(train_transf, labels_train)
clf.predict(test_transf)

clf.score(test_transf, labels_test)



clf = AdaBoostClassifier()
clf.fit(train_transf, labels_train)
clf.predict(test_transf)

clf.score(test_transf, labels_test)
