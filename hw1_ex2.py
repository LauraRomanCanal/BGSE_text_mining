#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Laura
"""

import re
import matplotlib.pylab as plt
import string
import pandas as pd
import numpy as np
import pickle
import os
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import porter

os.chdir('/Users/Laura/Desktop/hw_1')

import sys
sys.version

data = pd.read_table("speech_data_extend.txt",encoding="utf-8")
speeches = data['speech']

''' Exercise 1 '''
#1. tokenize
sp_tkn = [tokenize.word_tokenize(speech) for speech in speeches]
#2. keep only ascii_letters
for i in range(len(sp_tkn)):
    sp_tkn[i] = [j for j in sp_tkn[i] if j[0] in set(string.ascii_letters)]
#3. remove stopwords
stop = set(stopwords.words('english'))
for i in range(len(sp_tkn)):
    sp_tkn[i] = [j.lower() for j in sp_tkn[i] if j.lower() not in stop]

#4. stemmize
stemmer = porter.PorterStemmer()
stemmed = []
for i in range(len(sp_tkn)):
    stemmed.append([stemmer.stem(word) for word in sp_tkn[i]])

''' Exercise 2'''

#topic dictionaries: positive, negative, uncertainty words
with open('dict_pos_mp.txt', 'r') as myfile:
    s_p=myfile.read().replace('\n', ' ')
with open('dict_neg_mp.txt', 'r') as myfile:
    s_n=myfile.read().replace('\n', ' ')
with open('dict_uncertainty.txt', 'r') as myfile:
    s_u=myfile.read().replace('\n', ' ')

#alternative way to open dictionaries
#pp = open("dict_pos_mp.txt", "r")
#a = [columns for columns in ( raw.strip().split() for raw in pp )]  
#dim_p = len(a)
#nn = open("dict_neg_mp.txt", "r")
#b = [columns for columns in ( raw.strip().split() for raw in nn )]  
#dim_n = len(b)
#uu = open("dict_neg_mp.txt", "r")
#c = [columns for columns in ( raw.strip().split() for raw in uu )]  
#dim_u = len(c) 
#dim_clas = dim_p+dim_n+dim_u

# compute topic% for each doc 
pos= [0]*len(stemmed);neg= [0]*len(stemmed); unc= [0]*len(stemmed)

for i in range(len(stemmed)):
    words = set(stemmed[i])
    p=0; n=0; u=0; t=0
    for j in words: 
        if (' ' + j + ' ') in (' ' + s_p + ' '):
            p +=1
        if (' ' + j + ' ') in (' ' + s_n + ' '):
            n +=1
        if (' ' + j + ' ') in (' ' + s_u + ' '):
            u +=1
    if (len(words) > 0):
        pos[i] = p*100/len(words)
        neg[i] = n*100/len(words)
        unc[i] = u*100/len(words)  
          
index_top_pos = sorted(pos,reverse=True)[:10]
#pos.index(20.0)
sorted(neg,reverse=True)[:10]
sorted(unc,reverse=True)[:10]
     
#compare specific topic words vs all topics words
pp= [0]*len(stemmed); nn= [0]*len(stemmed); uu= [0]*len(stemmed)

for i in range(len(stemmed)):
    words = set(stemmed[i])
    p=0; n=0; u=0; t=0
    for j in words: 
        if (' ' + j + ' ') in (' ' + s_p + ' '):
            p +=1
        if (' ' + j + ' ') in (' ' + s_n + ' '):
            n +=1
        if (' ' + j + ' ') in (' ' + s_u + ' '):
            u +=1
    t = u+n+p
    if (t > 0):
        pp[i] = 100*p/t
        nn[i] = 100*n/t
        uu[i] = 100*u/t  


def frequencyDistribution(data):
    return {i: data.count(i) for i in data}   
pos = (frequencyDistribution(pp))
print (frequencyDistribution(nn))
print (frequencyDistribution(uu))


from collections import Counter

labels, values = zip(*Counter(pp).items())
indexes = np.arange(len(labels))
width = 0.5
plt.bar(indexes, values, width)
plt.xticks(rotation=90)
plt.title("positive words vs classified words")
plt.xticks(indexes + width * 0.5, labels)
plt.show()
 
labels, values = zip(*Counter(nn).items())
indexes = np.arange(len(labels))
width = 0.5
plt.bar(indexes, values, width)
plt.xticks(rotation=90)
plt.title("negative words vs classified words")
plt.xticks(indexes + width * 0.5, labels)
plt.show()

labels, values = zip(*Counter(uu).items())
indexes = np.arange(len(labels))
width = 0.5
plt.bar(indexes, values, width)
plt.xticks(rotation=90)
plt.title("uncertainty words vs classified words")
plt.xticks(indexes + width * 0.5, labels)
plt.show()


