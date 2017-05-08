import numpy as np
import pandas as pd
import nltk
import os
import sys
import scipy.sparse as ssp
import time
import matplotlib
import tqdm
from tqdm import tqdm
from numpy.random import dirichlet
from collections import Counter
from utils import data_processing, get_vocab, make_count


os.chdir('/home/euan/documents/text-mining/BGSE_text_mining/HW2')
sys.path.append(os.getcwd())
from utils import data_processing, get_vocab, make_count

%matplotlib inline

data = pd.read_table("../HW1/speech_data_extend.txt",encoding="utf-8")
data_post1945 = data.loc[data.year >= 1945]
%time stemmed, processed_data = data_processing(data_post1945)

def Gibbs_sampling_LDA(stemmed, K, alpha = None, eta = None, m=3, n_samples = 200, burnin = 500, perplexity = False):
    '''
    Gibbs sampler for LDA model
    '''

    def Z_class_1(Beta, Theta):
        Z = [np.ndarray.tolist( np.argmax( Beta[:,[idx[word] for word in stemmed[i]]] * \
        Theta[i,:].reshape((K, 1)), axis = 0) ) for i in range(Theta.shape[0] )]
        return Z

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
    labels  = np.zeros((n_samples, len(s)))

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
    for i in range(m*n_samples):
        Z       = Z_class_1(Beta, Theta)
        Beta    = Beta_sample(eta, Z)
        Theta   = Theta_sample(alpha, Z)

        # Add every m-th sample to output
        if i%m == 0:
            Z_s = [i for sublist in Z for i in sublist ]
            j = np.int(i/m)
            labels[j, :] = Z_s
        if i%20 == 0:
            if perplexity:
                perp.append(perplexity(Theta, Beta, count_matrix))
            print( "Iteration {}".format(i))

    return (labels, perp)

LDA_labels, perp = Gibbs_sampling_LDA(stemmed, K = 10, n_samples = 100,
                                            perplexity=True, burnin = 1000)

doc_label = [[i]*len(stemmed[i]) for i in range(len(stemmed))]
doc_label = [i for sublist in doc_label for i in sublist]
labels = LDA_labels[0]

def dt_matrix(labels, doc_label):
    dt = np.zeros((len(set(doc_label)), K))
    label_dict = dict(zip(range(labels.shape[0]), labels))
    doc_dict = dict(zip(range(len(doc_label)), doc_label))
    for i in range(len(doc_label)):
        dt[doc_dict[i], label_dict[i]] += 1
    return dt

labels = LDA_labels[0]

%time dt1 = dt_matrix(labels, doc_label)
