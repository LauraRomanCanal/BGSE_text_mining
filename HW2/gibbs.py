import numpy as np
import pandas as pd
import nltk
import datetime
import os
import scipy.sparse as ssp
from numpy.random import dirichlet
from utils import data_processing, get_vocab, make_count
from collections import Counter

os.chdir('/home/euan/documents/text-mining/BGSE_text_mining')
data = pd.read_table("HW1/speech_data_extend.txt",encoding="utf-8")
data_post1950 = data.loc[data.year >= 1950]
stemmed, processed_data = data_processing(data_post1950)

def Gibbs_sampling_LDA(stemmed, K, alpha, eta, m=5, iters = 200):
    '''
    Gibbs sampler for LDA model
    '''
    #def Z_probs(Beta, Theta):
    #    Z = [[Theta[i,:]*Beta[:,idx[word]]/Theta[i,:].dot(Beta[:,idx[word]]) for word in stemmed[i] ] for i in range(Theta.shape[0])]
    #    return Z

    def Z_probs(Beta, Theta):
        denoms = Theta.dot(Beta)
        Z = [ [Theta[i,:]*Beta[:,idx[word]]/denoms[i,idx[word]] for word in stemmed[i] ] for i in range(Theta.shape[0])]
        return Z
        i = 0; word = 'hous'

    def Z_class(Beta, Theta):
        Z = [[ np.argmax(Theta[i,:]*Beta[:,idx[word]])  for word in stemmed[i] ] for i in range(Theta.shape[0])]
        return Z

    def Beta_sample(eta, Z):
        z_s = [np.argmax(i) for sublist in Z for i in sublist ]
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
            Z_s     = [np.argmax(i) for i in Z[i]]
            counts  = Counter(Z_s)
            for j in set(counts.keys()):
                N[i,j]  = counts[j]
        Theta = [dirichlet(alpha = alpha + N[i],size = 1)[0] for i in range(D)]
        return np.array(Theta)

    def perplexity(Theta, Beta, count_matrix):
        ltb     = np.log(Theta.dot(Beta))
        num     = np.sum(count_matrix.multiply(ltb))
        denom   = len(s)
        return np.exp(-num/denom)

    # Get params needed for passing to sampling functions
    s       = [i for sublist in stemmed for i in sublist ] # list of all words in corpus (not unique)
    vocab   = get_vocab(stemmed)
    D       = len(stemmed)
    V       = len(vocab)
    idx     = dict(zip(vocab,range(len(vocab))))
    count_matrix = make_count(stemmed, idx)
    Z_Output  = []
    Perp   = []

    # Initialise params
    Theta   = dirichlet(alpha = [alpha]*K, size = D)
    Beta    = dirichlet(alpha = [eta]*V, size = K)
    Z       = Z_probs(Beta, Theta)

    # SAMPLING
    for i in range(iters):
        Z       = Z_class(Beta, Theta)
        Beta    = Beta_sample(eta, Z)
        Theta   = Theta_sample(alpha, Z)
        Z_s     = [[np.argmax(j) for j in Z[i]] for i in range(len(Z))]

        # Add every m-th sample to output
        if i%m == 0:
            Z_Output.append(Z_s)
            Perp.append(perplexity(Theta,Beta,count_matrix))

    return (Z_Output, Perp)

t1 = datetime.datetime.now().time()
LDA_samples = Gibbs_sampling_LDA(stemmed, K = 10, alpha = 1, eta = 1, m = 3, iters = 1000)
t2 = datetime.datetime.now().time()
print((t2.minute - t1.minute)*60 + (t2.second - t1.second) + (t2.microsecond - t1.microsecond)/1e6)

t1 = datetime.datetime.now().time()
for i in range(10):
    Z = Z_probs(Beta, Theta)
t2 = datetime.datetime.now().time()
print((t2.minute - t1.minute)*60 + (t2.second - t1.second) + (t2.microsecond - t1.microsecond)/1e6)
