#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:55:06 2017

@author: Euan, Laura
"""


import numpy as np
import pandas as pd
import nltk
import os
import scipy.sparse as ssp
import time
import matplotlib
from numpy.random import dirichlet
from collections import Counter
from matplotlib import pyplot


import topicmodels


os.chdir('/Users/Laura/Desktop/text_mining_hw2/ex1')
from utils import data_processing, get_vocab, make_count
os.chdir('/Users/Laura/Desktop/text_mining_hw2')
data = pd.read_table("speech_data_extend.txt",encoding="utf-8")
data_post1945 = data.loc[data.year >= 1945]
%time stemmed, processed_data = data_processing(data_post1945)

ldaobj = topicmodels.LDA.LDAGibbs(stemmed, 10)


ldaobj.sample(0, 20, 75)


perp_2 = ldaobj.perplexity()

os.chdir('/Users/Laura/Desktop/text_mining_hw2/data')
perp_1 = pd.read_csv('./perp.csv') 


plt.plot(perp_1,   lw = 1., label = 'uncollapsed')
plt.plot(perp_2, lw = 1., label = 'collapsed')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('perplexity')
plt.savefig('perplexities_collapsed.png', bbox_inches='tight')


# 2.2

ldaobj = topicmodels.LDA.LDAGibbs(stemmed, 10)
ldaobj.sample(1000, 5, 100)




k = ldaobj.K
alpha =  ldaobj.alpha

docterms = ldaobj.dt_avg()
nm = np.array([len(doc) for doc in stemmed])
nm = nm.reshape((10252,1))
nmz = nm*docterms
nmz

theta = (nmz+alpha)/(nm+k*alpha)
theta.shape


# idea 1: compare the sum
theta.sum(axis=0)

os.chdir('/Users/Laura/Desktop/def')
theta_uncoll = pd.read_csv('./theta_uncollapsed.csv') 
theta_uncoll.drop( 'Unnamed: 0',axis=1,inplace=True)
theta_uncoll.sum(axis=0)

theta_uncoll= np.array(theta_uncoll)

# idea 2: compar mean and sd

m1 = theta.mean(axis=0)
st1= theta.std(axis=0)

m2 = theta_uncoll.mean(axis=0)
st2 = theta_uncoll.std(axis=0)
data_compare = pd.DataFrame([m1,m2,st1,st2])

# idea 3: compare the histograms/plots

t1 = theta[:,0]
t2 = theta[:,1]
t3 = theta[:,2]
t4 = theta[:,3]
t5 = theta[:,4]
t6 = theta[:,5]
t7 = theta[:,6]
t8 = theta[:,7]
t9 = theta[:,8]
t10 = theta[:,9]

ut1 = theta_uncoll[:,0]
ut2 = theta_uncoll[:,1]
ut3 = theta_uncoll[:,2]
ut4 = theta_uncoll[:,3]
ut5 = theta_uncoll[:,4]
ut6 = theta_uncoll[:,5]
ut7 = theta_uncoll[:,6]
ut8 = theta_uncoll[:,7]
ut9 = theta_uncoll[:,8]
ut10 = theta_uncoll[:,9]


#T1-UT2
#T2-UT5


bins = np.linspace(0.03, 0.225, 70)

pyplot.hist(t8, bins, alpha=0.5, label='collapsed')
pyplot.hist(ut1, bins, alpha=0.5, label='uncoll, t1')
pyplot.hist(ut2, bins, alpha=0.5, label='uncoll, t2')
pyplot.hist(ut3, bins, alpha=0.5, label='uncoll, t3')
pyplot.hist(ut4, bins, alpha=0.5, label='uncoll, t4')
pyplot.hist(ut5, bins, alpha=0.5, label='uncoll, t5')
pyplot.hist(ut6, bins, alpha=0.5, label='uncoll, t6')
pyplot.hist(ut7, bins, alpha=0.5, label='uncoll, t7')
pyplot.hist(ut8, bins, alpha=0.5, label='uncoll, t8')
pyplot.hist(ut9, bins, alpha=0.5, label='uncoll, t9')
pyplot.legend(loc='upper right')
pyplot.show()
