# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:33:45 2017

@author: lindseykitchell
"""

# This is a function to parse the .ev ASCII files from shape-DNA.
# It takes the path to the folder the .ev files are in as input and
# outputs a pandas dataframe with the information from the top of the file,
# a list with all the eigevavlues, and the labels (PROB (0) or STREAM (1))


def parse_ev(path, n_ev):
    import os, sys
    import pandas as pd
    import numpy as np
    file_list = []
    shape_DNA_info = []
    for file in os.listdir(path):
        if file.endswith('.ev'):
            file_list.append(file)
    all_ev = np.zeros((len(file_list), n_ev))
    labels = np.zeros(len(file_list))
    for f_n in range(len(file_list)):
        f = [i for i in open(path+file_list[f_n]) if i[:-1]]
        d = dict((line.strip().split(': ') for line in f[0:16]))
        eigenvalues = f[18::]
        ev = []
        for e in range(len(eigenvalues)):
            eigenvalues[e] = eigenvalues[e].strip('{}\n ')
            eigenvalues[e] = eigenvalues[e].strip(' ')
            eigenvalues[e] = eigenvalues[e][0:-1]
            lt = eigenvalues[e].split(' ; ')
            for i in range(len(lt)):
                ev.append(float(lt[i]))
        shape_DNA_info.append(d)
        all_ev[f_n] = ev[0:n_ev]
        if 'STREAM' in file_list[f_n]:
            labels[f_n] = 1
    df = pd.DataFrame(shape_DNA_info)
    return df, all_ev, labels
