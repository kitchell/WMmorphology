# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:20:24 2017

@author: lindseykitchell
"""

import matplotlib.pyplot as plt
from parse_ev_files import parse_ev

path = '/Users/lindseykitchell/Box Sync/fiberVolumes/vol_norm_ev_files/'
#path = '/Users/lindseykitchell/Box Sync/fiberVolumes/normalized_ev_values/'
#path = '/Users/lindseykitchell/Box Sync/fiberVolumes/ev_files/'

# parse the .ev files
shapeDNA_df, Eigenvalues, labels = parse_ev(path,50)
X = Eigenvalues.tolist()
Y = labels

for e in range(10):
    print e
    for i in range(len(Y)):
        if i == 47:
            plt.plot(X[i][e], X[i][e+1], 'go')
        elif i == 20:
            plt.plot(X[i][e], X[i][e+1], 'go')
        elif i ==6: 
            plt.plot(X[i][e], X[i][e+1], 'go')
        elif Y[i] == 0:
            plt.plot(X[i][e], X[i][e+1], 'ro')
        else: 
            plt.plot(X[i][e], X[i][e+1], 'bo')
    plt.show()    
            
        

min_v = 10
      
for i in range(len(Y)):
    if X[i][0] < min_v:
        min_v = X[i][0]
        
for i in range(len(Y)):
    if min_v in X[i]:
        print shapeDNA_df.File[i], i
        

max_v = 10
      
for i in range(len(Y)):
    if X[i][1] < min_v:
        max_v = X[i][1]
        
for i in range(len(Y)):
    if max_v in X[i]:
        print shapeDNA_df.File[i], i
        
#
#plt.plot(X[1], X[2], 'o')
#plt.plot(X[2], X[3], 'o')
#plt.plot(X[3], X[4], 'o')
#plt.plot(X[3], X[5], 'o')
#plt.plot(X[5], X[6], 'o')
#
#
#xv= 1
#yv= 3
#for i in range(len(Y)):
#    if i == 47:
#        plt.plot(X[i][xv], X[i][yv], 'go')
#    elif i == 20:
#        plt.plot(X[i][xv], X[i][yv], 'go')
#    elif i ==6: 
#        plt.plot(X[i][xv], X[i][yv], 'go')
#    elif Y[i] == 0:
#        plt.plot(X[i][xv], X[i][yv], 'ro')
#    else: 
#        plt.plot(X[i][xv], X[i][yv], 'bo')
#plt.show() 
#
#
#for i in range(len(Y)):
#    if X[i][1] > 0.2:
#        print shapeDNA_df.File[i], i