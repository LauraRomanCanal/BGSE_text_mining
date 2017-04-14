#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:14:29 2017
@authors: Euan,Laura
"""
import re
import string
import pandas as pd
import numpy as np
import pickle
import os
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import porter
import matplotlib
from matplotlib import pyplot as plt
from numpy.linalg import svd
from scipy.misc import logsumexp
from nltk.tokenize import RegexpTokenizer

#os.chdir('/Users/Laura/Desktop/text_mining_hw1/try3')

#PRE-PROCESSING DATA
###############################################################################
def my_tokeniser(speeches):
    # Tokenize speeches
    tokenizer = RegexpTokenizer(r'\w+')
    sp_tkn = [tokenizer.tokenize(speech) for speech in speeches]
    return sp_tkn

def remove_nonalph(sp_tkn):
    # Remove non-alphabetic tokens
    for i in range(len(sp_tkn)):
        sp_tkn[i] = [j for j in sp_tkn[i] if j[0] in set(string.ascii_letters)]
    return sp_tkn

def stopword_del(sp_tkn):
    # Remove stopwords
    stop = set(stopwords.words('english'))
    for i in range(len(sp_tkn)):
        sp_tkn[i] = [j.lower() for j in sp_tkn[i] if j.lower() not in stop]
    return sp_tkn

def my_stem(sp_tkn):
    # Stem words in documents
    stemmer = porter.PorterStemmer()
    stemmed = [[stemmer.stem(word) for word in doc] for doc in sp_tkn]
    return stemmed

def remove_zerolen_strings(stemmed, data):
    idx = [i for i in range(len(stemmed)) if len(stemmed[i]) == 0]
    stemmed = [i for i in stemmed if len(i) > 0]
    data = data.drop(data.index[idx])
    data = data.reset_index(drop=True)
    return [stemmed, data]

def data_processing(data):
    '''
    Put together all steps in data processing. NOTE data must have column 'speech'
    '''
    speeches = data.speech
    sp_tkn = my_tokeniser(speeches)
    sp_tkn = remove_nonalph(sp_tkn)
    sp_tkn = stopword_del(sp_tkn)
    stemmed = my_stem(sp_tkn)
    stemmed, data = remove_zerolen_strings(stemmed, data)
    return [stemmed, data]
###############################################################################

# CALCULATING TF-IDF SCORES
###############################################################################
def get_vocab(stemmed_data):
    # extracts corpus vocabulary from list of documents
    vocab = list(set().union(*stemmed_data))
    return vocab

def doc_count(stemmed,vocab):
    # counts how many documents each word appears in
    df = dict(zip(vocab,[0]*len(vocab)))
    for i in range(len(stemmed)):
        words = set(stemmed[i])
        for j in words:
            df[j] = df[j]+1
    return df

def make_IDF(stemmed,vocab):
    # Calculates IDF factor for each word in vocabulary
    D   = len(stemmed)
    n   = len(get_vocab(stemmed))
    df  = doc_count(stemmed,vocab)
    IDF = [np.log(D/d) for d in df.values()]
    IDF_dict = dict(zip(vocab,IDF))
    return IDF_dict

def make_count(stemmed):
    vocab = get_vocab(stemmed)
    D = len(stemmed)
    n = len(vocab)
    idx = dict(zip(vocab,range(len(vocab))))
    count_matrix = np.ndarray(shape=(D,n))

    for i in range(len(stemmed)):
        for j in set(stemmed[i]):
            count_matrix[i,idx[j]] = stemmed[i].count(j)
    return count_matrix

def corpus_tf(stemmed):
    # Calculate corpus-level TF scores
    count_matrix = make_count(stemmed)
    tf = 1 +  np.log(np.sum(count_matrix, axis = 0))
    return tf

def corpus_tf_idf(stemmed):
    # Calculate corpus-level TF-IDF scores
    count_matrix = make_count(stemmed)
    vocab = get_vocab(stemmed)
    idf = list(make_IDF(stemmed, vocab).values())
    tf = 1 +  np.log(np.sum(count_matrix, axis = 0))
    tf_idf = tf * idf
    return tf_idf

def custom_stopword_del(stemmed, our_stopwords):
    for i in range(len(stemmed)):
        stemmed[i] = [j.lower() for j in stemmed[i] if j.lower() not in our_stopwords]
    return stemmed
###############################################################################

# Read in data
# documents defined at the paragraph level
data = pd.read_table("speech_data_extend.txt",encoding="utf-8")

# PROCESS THE DATA
stemmed, processed_data = data_processing(data)

#tf scores
vocab = get_vocab(stemmed)
tf_scores = corpus_tf(stemmed)

sort_tf = sorted(tf_scores,reverse=True)
ind_tf = sorted(range(len(tf_scores)), key=lambda k: tf_scores[k],reverse=True)
vocab_s = [vocab[i] for i in ind_tf]

term_sorttf = pd.DataFrame(
    {'term': vocab_s,
    'tf': sort_tf
    })

#tf-idf scores
tf_idf_scores = corpus_tf_idf(stemmed)

sort_tfidf = sorted(tf_idf_scores,reverse=True)
ind_tfidf = sorted(range(len(tf_idf_scores)), key=lambda k: tf_idf_scores[k],reverse=True)
vocab_sidf = [vocab[i] for i in ind_tfidf]
#sorted tf_idf

term_sortfidf = pd.DataFrame(
    {'term': vocab_sidf,
    'tf-idf': sort_tfidf
    })


# Remove context-specific stopwords
our_stopwords = set(vocab_sidf[0:2000])
stemmed = custom_stopword_del(stemmed, our_stopwords)
stemmed, processed_data = remove_zerolen_strings(stemmed, processed_data)

'''
 QUESTION 2
'''

from nltk import PorterStemmer

# DICTIONARY METHODS
###############################################################################
def read_dictionary(path):
    '''
    Read in and format and stem dictionary
    output: list of stemmed words
    '''
    file_handle = open(path)
    file_content = file_handle.read()
    file_content = file_content.lower()
    stripped_text = re.sub(r'[^a-z\s]',"",file_content)
    stripped_text = stripped_text.split("\n")

    #remove the last entry
    del stripped_text[-1]

    # remove duplicates
    stripped_text = list(set(stripped_text))

    # we need to stem it
    stemmed = [PorterStemmer().stem(i) for i in stripped_text]

    return(stemmed)

def calculate_sentiment_for_word_list(sentiment_dictionary, words):
    """
    description: calculate the sentiment of a word list based on a provided
                 sentiment dictionary
    input: words: the list of words to calculate the sentiment of
    """
    recognized_word_count = 0

    words_list = []
    for word in words:
        if word in sentiment_dictionary:
            recognized_word_count += 1
            words_list.append(word)

    return recognized_word_count, words_list
###############################################################################
ethic_dict = read_dictionary('./dictionaries/ethics.csv')
politic_dict = read_dictionary('./dictionaries/politics.csv')
negative_dict = read_dictionary('./dictionaries/negative.csv')
positive_dict = read_dictionary('./dictionaries/positive.csv')
passive_dict = read_dictionary('./dictionaries/passive.csv')
econ_dict = read_dictionary('./dictionaries/econ.csv')
military_dict = read_dictionary('./dictionaries/military.csv')
uncert_dict = read_dictionary('./dictionaries/uncertainty.csv')
dim_dict = [len(positive_dict),
len(negative_dict),
len(econ_dict),
len(politic_dict),
len(passive_dict),
len(military_dict),
len(ethic_dict),
len(uncert_dict)]
n_dict = 9
counts = np.ndarray(shape=(len(stemmed),n_dict))
for j in range(len(stemmed)):
    words = []
    words = set(stemmed[j])
    counts[j,0] = calculate_sentiment_for_word_list(positive_dict,words)[0]
    counts[j,1] = calculate_sentiment_for_word_list(negative_dict,words)[0]
    counts[j,2] = calculate_sentiment_for_word_list(uncert_dict,words)[0]
    counts[j,3] = calculate_sentiment_for_word_list(passive_dict,words)[0]
    counts[j,4] = calculate_sentiment_for_word_list(ethic_dict,words)[0]
    counts[j,5] = calculate_sentiment_for_word_list(politic_dict,words)[0]
    counts[j,6] = calculate_sentiment_for_word_list(econ_dict,words)[0]
    counts[j,7] = calculate_sentiment_for_word_list(military_dict,words)[0]

    #also we can keep track on classif words with per document and dictionary with
    #pos_words = calculate_sentiment_for_word_list(positive_dict,words)[1]

#determine topic of each doc
tot = []
for j in range(len(stemmed)):
    c = 0
    for i in range(7):
        c += counts[j,i]
    counts[j,8] = c

cc = pd.DataFrame(counts, columns=['pos', 'neg', 'unc', 'passive', 'ethic', 'polit', 'econ', 'milit', 'total'])

#determine weight for each topic across all docs
all_docs = cc.sum(axis=0)
perc = np.ndarray(shape=(9,))
for i in range(9):
    perc[i]=100*all_docs[i]/all_docs[8]
perc = pd.DataFrame(perc)
perc.columns = ['%']
perc.index =    ['positive', 'negative', 'uncertainty', 'passive', 'ethic', 'politics', 'economy', 'military', 'total']

perc.sort_values(by='%', ascending=0)
#so speeches are mostly positive and about politics and economy


#let's study the evolution of speeches topics over years
dd= pd.DataFrame(data)
data_by_years = dd.groupby('year', sort=False, as_index=True)['speech'].apply(' '.join)
df3 = data_by_years.reset_index()
df3.shape

stemmed_y, processed_data_y = data_processing(df3)
len(stemmed_y)
#we now have 224 docs (one per year)

counts_y = np.ndarray(shape=(len(stemmed_y),n_dict))
for j in range(len(stemmed_y)):
    words = []
    words = set(stemmed_y[j])
    counts_y[j,0] = calculate_sentiment_for_word_list(positive_dict,words)[0]
    counts_y[j,1] = calculate_sentiment_for_word_list(negative_dict,words)[0]
    counts_y[j,2] = calculate_sentiment_for_word_list(uncert_dict,words)[0]
    counts_y[j,3] = calculate_sentiment_for_word_list(passive_dict,words)[0]
    counts_y[j,4] = calculate_sentiment_for_word_list(ethic_dict,words)[0]
    counts_y[j,5] = calculate_sentiment_for_word_list(politic_dict,words)[0]
    counts_y[j,6] = calculate_sentiment_for_word_list(econ_dict,words)[0]
    counts_y[j,7] = calculate_sentiment_for_word_list(military_dict,words)[0]

len(negative_dict)
#determine topic of each doc
tot = []
for j in range(len(stemmed_y)):
    c = 0
    for i in range(7):
        c += counts_y[j,i]
    counts_y[j,8] = c

cc_y = pd.DataFrame(counts_y, columns=['pos', 'neg', 'unc', 'passive', 'ethic', 'polit', 'econ', 'milit', 'total'])
cc_y['pos']=100*cc_y['pos']/cc_y['total']
cc_y['neg']=100*cc_y['neg']/cc_y['total']
cc_y['unc']=100*cc_y['unc']/cc_y['total']
cc_y['passive']=100*cc_y['passive']/cc_y['total']
cc_y['ethic']=100*cc_y['ethic']/cc_y['total']
cc_y['polit']=100*cc_y['polit']/cc_y['total']
cc_y['econ']=100*cc_y['econ']/cc_y['total']
cc_y['milit']=100*cc_y['milit']/cc_y['total']
cc_y['year'] =   df3['year']


cc_y.plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))

#every 32 years
cc_y[0:32].plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))
cc_y[33:65].plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))
cc_y[66:98].plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))
cc_y[99:131].plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))
cc_y[132:164].plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))
cc_y[165:197].plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))
cc_y[197:230].plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="line",title="Speeches topics evolution").legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

#barplot
#df.plot(x="year", y=["pos", "neg", "unc", "passive","ethic","polit","econ","milit"], kind="bar").legend(loc='center left', bbox_to_anchor=(1, 0.5))
#cc_y.sort(columns='neg',axis=0, ascending=False)


'''
QUESTION 3
'''

import sklearn
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
###############################################################################

def make_TF_IDF(stemmed):
    # Calculates TF-IDF matrix
    vocab = get_vocab(stemmed)
    D = len(stemmed)
    idx = dict(zip(vocab,range(len(vocab))))
    IDF_dict = make_IDF(stemmed,vocab)
    tf_idf = np.ndarray(shape=(D,len(vocab)))

    for i in range(len(stemmed)):
        for j in set(stemmed[i]):
            tf_idf[i,idx[j]] = stemmed[i].count(j)*IDF_dict[j]
    return tf_idf
###############################################################################

# Comparison of parties post 1860

# First collect names and assign parties to all presidents after first Republican president elected
pres    = sorted(list ( set(data.loc[data.year > 1860].president)))
party   = ['rep']*3 + ['dem']*3 + ['rep']*8 + ['dem']*3 + ['rep']*3 + ['dem']*1 + ['rep']*2 + ['dem'] + ['rep'] + ['dem']*2

pres_party = dict(zip(pres, party))

data_post1860 = processed_data.loc[processed_data.year > 1860]
data_post1860 = data_post1860.reset_index(drop=True)
parties = [pres_party[i] for i in data_post1860.president]
data_post1860 = data_post1860.assign(party=parties)

stemmed_post1860, processed_post1860 = data_processing(data_post1860)

speeches = data_post1860.speech
sp_tkn = my_tokeniser(speeches)
sp_tkn = remove_nonalph(sp_tkn)
sp_tkn = stopword_del(sp_tkn)
stemmed = my_stem(sp_tkn)
stemmed, data = remove_zerolen_strings(stemmed, data)

stemmed_post1860 = custom_stopword_del(stemmed_post1860, our_stopwords)
stemmed_post1860, processed_post1860 = remove_zerolen_strings(stemmed_post1860, processed_post1860)

parties_post1860 = [i for i in processed_post1860.party]
dem_idx = [i for i in range(len(parties_post1860)) if parties_post1860[i] == 'dem']
rep_idx = [i for i in range(len(parties_post1860)) if parties_post1860[i] == 'rep']

tf_idf_post1860 = make_TF_IDF(stemmed_post1860)

cos_sim = cosine_similarity(tf_idf_post1860)

similarity_within_dem = cos_sim[dem_idx,:][:,dem_idx]
similarity_within_rep = cos_sim[rep_idx,:][:,rep_idx]
similarity_between_parties = cos_sim[dem_idx,:][:,rep_idx]

print(np.mean(similarity_within_dem))
print(np.mean(similarity_within_rep))
print(np.mean(similarity_between_parties))

'''
Now do singular value decomposition
'''

U, S, V = svds(tf_idf_post1860, k = 200)

low_rank_approx = U.dot(np.diag(S)).dot(V)

low_rank_cos_sim = cosine_similarity(low_rank_approx)

low_rank_similarity_within_dem = low_rank_cos_sim[dem_idx,:][:,dem_idx]
low_rank_similarity_within_rep = low_rank_cos_sim[rep_idx,:][:,rep_idx]
low_rank_similarity_between_parties = low_rank_cos_sim[dem_idx,:][:,rep_idx]

print(np.mean(low_rank_similarity_within_dem))
print(np.mean(low_rank_similarity_within_rep))
print(np.mean(low_rank_similarity_between_parties))

'''
QUESTION 4
'''
# EM
###############################################################################
def E_step(rho_i, B_i, count_matrix):
    L =  np.log(rho_i) + count_matrix.dot(np.log(B_i.T))
    z_hat = np.exp((L.T - logsumexp(L, axis=1)).T)
    return z_hat

def rho_update(z_hat, count_matrix):
    D = np.shape(count_matrix)[0]
    rho_i = np.sum(z_hat, axis = 0) / D
    return rho_i

def beta_update(z_hat, count_matrix, N_d):
    lower_bound =  np.finfo('float').max**(-1)
    B_i = (count_matrix.T.dot(z_hat) / np.sum(z_hat.T * N_d, axis=1)).T
    B_i[B_i == 0.0] = lower_bound
    return B_i

def MM_loglik(rho_i, B_i, count_matrix):
    # Calculate log-likelihood of Multinomial Mixture Model
    L =  np.log(rho_i) + count_matrix.dot(np.log(B_i.T))
    L[L <= -500] = -500
    L =  np.exp(L)
    ll = np.sum(L, axis = 1)
    if ll.min() == 0.0:
        ll[ll==0.0] = np.finfo('float').max**(-1)
    ll = np.sum(np.log(ll))
    return(ll)

def Multinom_Mixt_EM(data, k, max_iters = 100, eps = 10^(-3)):
    count_matrix = make_count(data)
    vocab = get_vocab(data)
    D = len(data)
    n = len(vocab)
    N_d = [len(x) for x in data]

    # Initialise params
    rho_i   = [1/k]*k
    B_i     = np.random.dirichlet([1]*n, size=k)
    loglik_seq = [MM_loglik(rho_i, B_i, count_matrix)]

    for i in range(max_iters):

        # E step
        z_hat   = E_step(rho_i, B_i, count_matrix)

        # M step
        rho_i   = rho_update(z_hat, count_matrix)
        B_i     = beta_update(z_hat,count_matrix, N_d)
        loglik_seq.append(MM_loglik(rho_i, B_i, count_matrix))

        # Early stopping criterion
        if (loglik_seq[len(loglik_seq) - 1] - loglik_seq[len(loglik_seq) - 2]) <= eps:
            return [z_hat, rho_i, B_i, loglik_seq]

    return [z_hat, rho_i, B_i, loglik_seq]
###############################################################################

z_hat, rho_i, B_i, loglik_seq = Multinom_Mixt_EM(stemmed, k=3, max_iters = 100)
