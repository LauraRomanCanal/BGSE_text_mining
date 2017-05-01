import numpy as np
import pandas as pd
import nltk
import os
import scipy.sparse as ssp
import time
import matplotlib
%matplotlib inline
from numpy.random import dirichlet
from utils import data_processing, get_vocab, make_count
from collections import Counter

os.chdir('/home/euan/documents/text-mining/BGSE_text_mining/')

data = pd.read_table("HW1/speech_data_extend.txt",encoding="utf-8")
data_post1945 = data.loc[data.year >= 1945]
%time stemmed, processed_data = data_processing(data_post1945)

def Gibbs_sampling_LDA(stemmed, K, alpha = None, eta = None, m=3, iters = 200, burnin = 500, perplexity = False):
    '''
    Gibbs sampler for LDA model
    '''

    def Z_class_1(Beta, Theta):
        Z = [np.ndarray.tolist( np.argmax( Beta[:,[idx[word] for word in stemmed[i]]] * Theta[i,:].reshape((K, 1)), axis = 0) ) for i in range(Theta.shape[0] )]
        return Z

    #%time Z = Z_class(Beta, Theta)

    def Beta_sample(eta, Z):
        z_s = [z for sublist in Z for z in sublist ]
        M = np.zeros(shape=(K,V))
        for k in range(K):
            words = [s[i] for i in range(len(z_s)) if z_s[i] == k]
            counts = Counter(words)
            for word in set(words):
                M[k,idx[word]] = counts[word]
        Beta = [dirichlet(alpha = eta + M[i],size = 1)[0] for i in range(K)]
        return np.array(Beta)

    def Theta_sample(alpha, Z):
        N   = np.zeros(shape=(D,K))
        for i in range(D):
            counts   = Counter(Z[i])
            for j in set(counts.keys()):
                N[i,j]  = counts[j]
        Theta = [dirichlet(alpha = alpha + N[i],size = 1)[0] for i in range(D)]
        return np.array(Theta)

    def onehotencode(Z):
        '''
        Create function to one-hot encode topic allocation
        '''
        a       = np.array([i for sublist in Z for i in sublist ])
        b       = np.zeros((a.size, a.max()+1))
        b[np.arange(a.size),a] = 1
        return(b)

    def perplexity(Theta, Beta, count_matrix):
        '''
        Calculate perplexity for given sample
        '''
        ltb     = np.log(Theta.dot(Beta))
        num     = np.sum(count_matrix.multiply(ltb))
        denom   = len(s)
        return np.exp(-num/denom)

    # Get params needed for passing to sampling functions
    s       = [i for sublist in stemmed for i in sublist ]
    vocab   = get_vocab(stemmed)
    D       = len(stemmed)
    V       = len(vocab)
    idx     = dict(zip(vocab,range(len(vocab))))
    count_matrix = make_count(stemmed, idx)
    perp   = []

    # Initialise params
    if eta == None:
        eta = 200/V
    if alpha == None:
        alpha = 50/K

    Theta   = dirichlet(alpha = [alpha]*K, size = D)
    Beta    = dirichlet(alpha = [eta]*V, size = K)
    Z       = Z_class_1(Beta, Theta)
    labels  = ssp.coo_matrix(onehotencode(Z))

    # SAMPLING
    print('TIME:', time.strftime("%H:%M:%S", time.gmtime()))
    for i in range(burnin):
        Z       = Z_class_1(Beta, Theta)
        Beta    = Beta_sample(eta, Z)
        Theta   = Theta_sample(alpha, Z)
        if i%20 == 0:
            if perplexity:
                perp.append(perplexity(Theta, Beta, count_matrix))
            print('Burnin iteration {}'.format(i))

    print('TIME:', time.strftime("%H:%M:%S", time.gmtime()))
    for i in range(iters):
        Z       = Z_class_1(Beta, Theta)
        Beta    = Beta_sample(eta, Z)
        Theta   = Theta_sample(alpha, Z)

        # Add every m-th sample to output
        if i%m == 0:
            labels  += ssp.coo_matrix(onehotencode(Z))
        if i%20 == 0:
            if perplexity:
                perp.append(perplexity(Theta, Beta, count_matrix))
            print( "Iteration {}".format(i))

    return (labels, perp)

%time LDA_labels, perp = Gibbs_sampling_LDA(stemmed, K = 10, iters = 2000, perplexity=True, burnin = 1000)

LDA_labels = pd.DataFrame(LDA_labels.toarray())
LDA_labels.index = [i for sublist in stemmed for i in sublist ]
pd.DataFrame.to_csv(LDA_labels,path_or_buf='LDA_labels.csv',index=True)

perp =  pd.DataFrame(perp)
pd.DataFrame.to_csv(perp,path_or_buf='perp.csv',index=False)


##############################################################
# Using the lda package
##############################################################

import lda
from matplotlib import pyplot as plt

idx = dict(zip(get_vocab(stemmed),range(len(get_vocab(stemmed)))))
X   = make_count(stemmed, idx)
X   = X.astype(int)
model = lda.LDA(n_topics= 10, n_iter=2500, alpha = 200/len(get_vocab(stemmed)), eta = 50/10)
%time model.fit(X)
