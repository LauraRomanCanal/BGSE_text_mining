import numpy as np
import pandas as pd
import nltk
import os
from numpy.random import dirichlet
from utils import data_processing, get_vocab
from collections import Counter

os.chdir('/home/euan/documents/text-mining/BGSE_text_mining')
data = pd.read_table("HW1/speech_data_extend.txt",encoding="utf-8")
stemmed, processed_data = data_processing(data)

def Gibbs_sampling_LDA(stemmed, K, alpha, eta, m=5, iters = 200):

    def Z_probs(Beta, Theta):
        Z = [[Theta[i,:]*Beta[:,idx[word]]/Theta[i,:].dot(Beta[:,idx[word]]) for word in stemmed[i] ] for i in range(Theta.shape[0])]
        return Z

    def Beta_sample(eta, Z):
        K   = len(Z[0][0])
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
        K   = len(Z[0][0])
        N   = np.zeros(shape=(D,K))
        for i in range(D):
            Z_s     = [np.argmax(i) for i in Z[i]]
            counts  = Counter(Z_s)
            for j in set(counts.keys()):
                N[i,j]  = counts[j]
        Theta = [dirichlet(alpha = alpha + N[i],size = 1)[0] for i in range(D)]
        return np.array(Theta)

    s       = [i for sublist in stemmed for i in sublist ] # list of all words in corpus (not unique)
    vocab   = get_vocab(stemmed)
    D       = len(stemmed)
    V       = len(vocab)
    idx     = dict(zip(vocab,range(len(vocab))))
    Output  = []

    #Initialise params
    Theta   = dirichlet(alpha = [alpha]*K, size = D)
    Beta    = dirichlet(alpha = [eta]*V, size = K)
    Z       = Z_probs(Beta, Theta)

    # SAMPLING
    for i in range(iters):
        Z       = Z_probs(Beta, Theta)
        Beta    = Beta_sample(eta, Z)
        Theta   = Theta_sample(alpha, Z)
        Z_s     = [[np.argmax(j) for j in Z[i]] for i in range(len(Z))]
        if i%m == 0:
            Output.append(Z_s)

    return Output

z = Gibbs_sampling_LDA(stemmed, K = 10, alpha = 1, eta = 1, m = 5, iters = 200)
