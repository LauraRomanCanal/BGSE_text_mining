import numpy as np
import pandas as pd
import nltk
import os
from numpy.random import dirichlet
from utils import data_processing, get_vocab

os.chdir('/home/euan/documents/text-mining/BGSE_text_mining')
data = pd.read_table("HW1/speech_data_extend.txt",encoding="utf-8")
stemmed, processed_data = data_processing(data)

s = [i for sublist in stemmed for i in sublist ]
vocab = get_vocab(stemmed)
word_idx = dict(zip(vocab,range(len(vocab))))

K = 5
V = len(vocab)
D = len(stemmed)

Theta = dirichlet(alpha = [1]*K, size = D)
Beta = dirichlet(alpha = [1]*V, size = K)
beta = Beta[:,0]
theta = Theta[0,:]
doc = stemmed[0]
Theta.shape[0]

def Z_sample(Beta, Theta):
    Z = [[Theta[i,:]*Beta[:,word_idx[word]]/Theta[i,:].dot(Beta[:,word_idx[word]]) for word in stemmed[i] ] for i in range(Theta.shape[0])]
    return Z

def Beta_sample(eta, Z):
    z_s = [np.argmax(i) for sublist in Z for i in sublist ]
    M = np.zeros(shape=(K,V))
    for k in range(K):
        words = [s[i] for i in range(len(z_s)) if z_s[i] == k]
        for word in set(words):
            M[k,word_idx[word]] = words.count(word)



Z = Z_sample(Beta,Theta)
z_s = [np.argmax(i) for sublist in Z for i in sublist ]

M = np.zeros(shape=(K,V))

for k in range(K):
    words = [s[i] for i in range(len(z_s)) if z_s[i] == k]
    for word in set(words):
        M[k,word_idx[word]] = words.count(word)
