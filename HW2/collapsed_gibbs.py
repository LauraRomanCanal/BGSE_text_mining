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

import matplotlib.pyplot as plt
from numpy.random import dirichlet
from collections import Counter
from matplotlib import pyplot


import topicmodels

#2.1 collapsed gibbs sampling perplexity vs. uncollapsed perplexity.

os.chdir('/Users/Laura/Desktop/text_mining_hw2/ex1')
from utils import data_processing, get_vocab, make_count
os.chdir('/Users/Laura/Desktop/text_mining_hw2')
data = pd.read_table("speech_data_extend.txt",encoding="utf-8")
data_post1945 = data.loc[data.year >= 1945]
%time stemmed, processed_data = data_processing(data_post1945)

ldaobj = topicmodels.LDA.LDAGibbs(stemmed, 10)


ldaobj.sample(0, 20, 75)


perp2 = ldaobj.perplexity()
perp_2 =  pd.DataFrame(perp_2)
pd.DataFrame.to_csv(perp_2,path_or_buf='perp2.csv',index=False)

os.chdir('/Users/Laura/Desktop/text_mining_hw2/data')
perp1 = pd.read_csv('./perplexity.csv') 
perp1 = np.array(perp1)
perp1 = perp1[:,1]

plt.plot(perp1,   lw = 1., label = 'uncollapsed')
plt.plot(perp2, lw = 1., label = 'collapsed')
plt.legend(loc='center left', bbox_to_anchor=(0.65, 0.8))
plt.ylabel('perplexity')
plt.xlabel('samples')
plt.savefig('perplexities.png', bbox_inches='tight')


# 2.2 

ldaobj = topicmodels.LDA.LDAGibbs(stemmed, 10)
ldaobj.sample(1000, 5, 100)


k = ldaobj.K
alpha =  ldaobj.alpha
docterms = ldaobj.dt_avg()
nm = np.array([len(doc) for doc in stemmed])
nm = nm.reshape((10252,1))
nmz = nm*docterms

theta_collapsed = (nmz+alpha)/(nm+k*alpha)

theta_uncoll = pd.read_csv('./theta_uncollapsed.csv') 
theta_uncoll.drop( 'Unnamed: 0',axis=1,inplace=True)
theta_uncoll = np.array(theta_uncoll)


theta_uncoll.shape
# idea 1: compare the sum
a =theta_collapsed.sum(axis=0)

b = theta_uncoll.sum(axis=0)
thetas_sum = pd.DataFrame(np.vstack((a,b)).T)
thetas_sum.columns = ['collapsed','uncollapsed']
thetas_sum


# idea 2: compar mean, sd, var
theta_collapsed.mean()
theta_uncoll.mean()
m1 = np.array(theta_collapsed.mean(axis=0))
st1= np.array(theta_collapsed.std(axis=0))
var1 = np.array(np.var(theta_collapsed, axis=0))

m2 = np.array(theta_uncoll.mean(axis=0))
st2 = np.array(theta_uncoll.std(axis=0))
var2 = np.array(np.var(theta_uncoll, axis=0))

basic_stats = pd.DataFrame(np.vstack((m1,m2,st1,st2,var1,var2)).T)
basic_stats.columns = ['mean_c','mean_unc','sd_c','sd_unc','var_c','var_u']
basic_stats

# idea 3: compare the histograms/plots

t1 = theta_collapsed[:,0]
t2 = theta_collapsed[:,1]
t3 = theta_collapsed[:,2]
t4 = theta_collapsed[:,3]
t5 = theta_collapsed[:,4]
t6 = theta_collapsed[:,5]
t7 = theta_collapsed[:,6]
t8 = theta_collapsed[:,7]
t9 = theta_collapsed[:,8]
t10 = theta_collapsed[:,9]

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


bins = np.linspace(0.03, 0.225, 70)

pyplot.hist(t1, bins, alpha=0.5, label='topic 1')
pyplot.hist(t2, bins, alpha=0.5, label='topic 2')
pyplot.hist(t3, bins, alpha=0.5, label='topic 3')
pyplot.hist(t4, bins, alpha=0.5, label='topic 4')
pyplot.hist(t5, bins, alpha=0.5, label='topic 5')
pyplot.hist(t6, bins, alpha=0.5, label='topic 6')
pyplot.hist(t7, bins, alpha=0.5, label='topic 7')
pyplot.hist(t8, bins, alpha=0.5, label='topic 8')
pyplot.hist(t9, bins, alpha=0.5, label='topic 9')
pyplot.legend(loc='upper right')
pyplot.show()

pyplot.hist(ut1, bins, alpha=0.5, label='topic 1')
pyplot.hist(ut2, bins, alpha=0.5, label='topic 2')
pyplot.hist(ut3, bins, alpha=0.5, label='topic 3')
pyplot.hist(ut4, bins, alpha=0.5, label='topic 4')
pyplot.hist(ut5, bins, alpha=0.5, label='topic 5')
pyplot.hist(ut6, bins, alpha=0.5, label='topic 6')
pyplot.hist(ut7, bins, alpha=0.5, label='topic 7')
pyplot.hist(ut8, bins, alpha=0.5, label='topic 8')
pyplot.hist(ut9, bins, alpha=0.5, label='topic 9')
pyplot.legend(loc='upper right')
pyplot.show()

