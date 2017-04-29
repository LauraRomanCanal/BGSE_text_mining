import numpy as np
import pandas as pd
import nltk
import os
import scipy.sparse as ssp
import time

from numpy.random import dirichlet
from utils import data_processing, get_vocab, make_count
from collections import Counter

os.chdir('/home/euan/documents/text-mining/BGSE_text_mining/')

data = pd.read_table("HW1/speech_data_extend.txt",encoding="utf-8")
data_post1900 = data.loc[data.year >= 1900]
stemmed, processed_data = data_processing(data_post1900)

def Gibbs_sampling_LDA(stemmed, K, alpha = None, eta = None, m=3, iters = 200, burnin = 500, perplexity = False):
    '''
    Gibbs sampler for LDA model
    '''

    #def Z_probs(Beta, Theta):
    #    denoms = Theta.dot(Beta)
    #    Z = [ [Theta[i,:]*Beta[:,idx[word]]/denoms[i,idx[word]] for word in stemmed[i] ] for i in range(Theta.shape[0])]
    #    return Z

    def Z_class(Beta, Theta):
        Z = [[ np.argmax(Theta[i,:]*Beta[:,idx[word]]) for word in stemmed[i] ] for i in range(Theta.shape[0])]
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

    #%time  Beta = Beta_sample(eta, Z)

    def Theta_sample(alpha, Z):
        N   = np.zeros(shape=(D,K))
        for i in range(D):
            counts   = Counter(Z[i])
            for j in set(counts.keys()):
                N[i,j]  = counts[j]
        Theta = [dirichlet(alpha = alpha + N[i],size = 1)[0] for i in range(D)]
        return np.array(Theta)

    #%time Theta =  Theta_sample(alpha, Z)

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

    #%time perp = perplexity(Theta,Beta,count_matrix)

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
    Z       = Z_class(Beta, Theta)
    labels  = ssp.coo_matrix(onehotencode(Z))

    # SAMPLING
    for i in range(burnin):
        Z       = Z_class(Beta, Theta)
        Beta    = Beta_sample(eta, Z)
        Theta   = Theta_sample(alpha, Z)
        if i%20 == 0:
            if perplexity:
                perp.append(perplexity(Theta, Beta, count_matrix))
    for i in range(iters):
        Z       = Z_class(Beta, Theta)
        Beta    = Beta_sample(eta, Z)
        Theta   = Theta_sample(alpha, Z)
        # Add every m-th sample to output
        if i%m == 0:
            labels  += ssp.coo_matrix(onehotencode(Z))
        if i%20 == 0:
            if perplexity:
                perp.append(perplexity(Theta, Beta, count_matrix))

    return (labels, perp)

%time LDA_labels, perp = Gibbs_sampling_LDA(stemmed, K = 10, iters = 10, perplexity=True, burnin = 1)

LDA_labels = pd.DataFrame(LDA_labels.toarray())
pd.DataFrame.to_csv(LDA_labels,path_or_buf='LDA_labels.csv',index=False)

perp =  pd.DataFrame(perp)
pd.DataFrame.to_csv(perp,path_or_buf='perp.csv',index=False)



##############################################################
# Using the lda package
##############################################################

import lda

idx = dict(zip(get_vocab(stemmed),range(len(get_vocab(stemmed)))))
X   = make_count(stemmed, idx)
X   = X.astype(int)
model = lda.LDA(n_topics= 10, n_iter=1000)
%time model.fit(X)
