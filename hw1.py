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
os.chdir('/home/euan/documents/text-mining/BGSE_text_mining')
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
    #return [stemmed, data]
    return (stemmed, data)
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
    #return [stemmed, data] doesn't work to me
    return (stemmed, data)
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

# Read in data
# documents defined at the paragraph level
data = pd.read_table("speech_data_extend.txt",encoding="utf-8")

# PROCESS THE DATA
stemmed, processed_data = data_processing(data)

#tf scores

def custom_stopword_del(stemmed, data):
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
    stemmed, processed_data = remove_zerolen_strings(stemmed, data)

# Remove context-specific stopwords
our_stopwords = set(vocab_sidf[0:2000])
stemmed = custom_stopword_del(stemmed, our_stopwords)
stemmed, processed_data = remove_zerolen_strings(stemmed, processed_data

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

def count_on_dict(sentiment_dictionary, words):
    """
    description: calculate counts of a word list based on a dictionary
    """
    recognized_word_count = 0

    words_list = []
    for word in words:
        if word in sentiment_dictionary:
            recognized_word_count += 1
            words_list.append(word)
    return recognized_word_count, words_list

def docs_dict_matrix(stem,positive_dict,negative_dict,ethic_dict,politic_dict,econ_dict,military_dict,uncert_dict,passive_dict ):
    '''
    description: computes docs-topics matrix with data arranged by year or by docs
    '''    
    counts = np.ndarray(shape=(len(stem),8))
    for j in range(len(stem)):
        words = []
        words = stem[j]
        counts[j,0] = count_on_dict(positive_dict,words)[0]
        counts[j,1] = count_on_dict(negative_dict,words)[0]
        counts[j,2] = count_on_dict(uncert_dict,words)[0]
        counts[j,3] = count_on_dict(passive_dict,words)[0]
        counts[j,4] = count_on_dict(ethic_dict,words)[0]
        counts[j,5] = count_on_dict(politic_dict,words)[0]
        counts[j,6] = count_on_dict(econ_dict,words)[0]
        counts[j,7] = count_on_dict(military_dict,words)[0]
        #pos_words = calculate_sentiment_for_word_list(positive_dict,words)[1] # classif words
    
    counts = pd.DataFrame(counts, columns=['pos', 'neg', 'unc', 'passive', 'ethic', 'polit', 'econ', 'milit'])
    counts['total'] = counts.sum(axis=1)
    return counts
###############################################################################
### 2.a), 2.b)
'''
Docs - Topics
'''
#Dictionaries/topics of interest preprocessed
positive_dict = read_dictionary('./dictionaries/positive.csv'); negative_dict = read_dictionary('./dictionaries/negative.csv')
ethic_dict = read_dictionary('./dictionaries/ethics.csv'); politic_dict = read_dictionary('./dictionaries/politics.csv')
econ_dict = read_dictionary('./dictionaries/econ.csv'); military_dict = read_dictionary('./dictionaries/military.csv')
uncert_dict = read_dictionary('./dictionaries/uncertainty.csv'); passive_dict = read_dictionary('./dictionaries/passive.csv')

''' 
execute dict_cleaning.py file!!!!!!!!!!!!!!!!
'''

#docs-topics matrix
dtm = docs_dict_matrix(stemmed,positive_dict,negative_dict,ethic_dict,politic_dict,econ_dict,military_dict,uncert_dict,passive_dict )
dtm.shape #22968 docs

# % topic across all documents
sum_odocs = dtm.sum(axis=0)
perc = np.ndarray(shape=(8,))
for i in range(8):
    perc[i]=100*sum_odocs[i]/sum_odocs[8]
perc = pd.DataFrame(perc)
perc.columns = ['%']
perc.index =    ['positive', 'negative', 'uncertainty', 'passive', 'ethic', 'politics', 'economy', 'military']
perc.sort_values(by='%', ascending=0)

'''
Years - Topics
'''
#yearly data processing
data_by_years= pd.DataFrame(processed_data)
data_by_years = data_by_years.groupby('year', sort=False, as_index=True)['speech'].apply(' '.join)
data_by_years = data_by_years.reset_index()
stemmed_y, processed_data_y = data_processing(data_by_years)

#year-topics matrix
yt = docs_dict_matrix(stemmed_y,positive_dict,negative_dict,ethic_dict,politic_dict,econ_dict,military_dict,uncert_dict,passive_dict )
yt['year'] =data_by_years['year']
yt.shape #224 years
# in %
ytp=yt
ytp['pos']=100*ytp['pos']/ytp['total']; ytp['neg']=100*ytp['neg']/ytp['total']
ytp['unc']=100*ytp['unc']/ytp['total']; ytp['passive']=100*ytp['passive']/ytp['total']
ytp['ethic']=100*ytp['ethic']/ytp['total']; ytp['polit']=100*ytp['polit']/ytp['total']
ytp['econ']=100*ytp['econ']/ytp['total']; ytp['milit']=100*ytp['milit']/ytp['total']
ytp['total']=100; ytp['year'] =data_by_years['year']


''' 
yearly topics evolution - plots
'''
# whole dataset
###############################################################################
us_dates = [1812,1861,1865,1914,1918,1929,1941,1945, 1974,1991,2001,2008]
us_dates_exp = ['War on Britain','Civil War',
                '',
                'WWI','', 'Black Thursday','WWII','',
                'Watergate scandal','Iraq attacks','WTC attack', 'Great Recession']

X = ytp['year']
Y1 = ytp['pos'];Y2= ytp['neg'];Y3 = ytp['unc'];Y4= ytp['passive']
Y5 = ytp['ethic'];Y6= ytp['polit'];Y7 = ytp['econ']; Y8= ytp['milit']
plt.plot(X, Y1,   lw = 1., label = 'positive')
plt.plot(X, Y2, lw = 1., label = 'negative')
plt.plot(X, Y3, lw = 1., label = 'uncertainty')
plt.plot(X, Y4, lw = 1., label = 'passive')
plt.plot(X, Y5, lw = 1., label = 'ethic')
plt.plot(X, Y6,  lw = 1., label = 'politics')
plt.plot(X, Y7,  lw = 1., label = 'economics')
plt.plot(X, Y8,  lw = 1., label = 'military')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for i in range(len(us_dates)):
    plt.axvline(us_dates[i],linestyle="dashed", color="black", lw=0.6)
    plt.text(us_dates[i],-8,us_dates_exp[i],rotation=90)
plt.ylabel('%')
plt.title('Speeches topics evolution', y=1.08)
plt.savefig('./figures/evolution.png', bbox_inches='tight')

#more on detail
###############################################################################
us_dates = [1812]
us_dates_exp = ['War on Britain']
plt.plot(X[0:33], Y1[0:33],   lw = 1.5, label = 'positive')
plt.plot(X[0:33], Y2[0:33], lw = 1.5, label = 'negative')
#plt.plot(X[0:33], Y3[0:33], lw = 1.5, label = 'uncertainty')
#plt.plot(X[0:33], Y4[0:33], lw = 1.5, label = 'passive')
#plt.plot(X[0:33], Y5[0:33], lw = 1.5, label = 'ethics')
plt.plot(X[0:33], Y6[0:33],  lw = 1.5, label = 'politics')
plt.plot(X[0:33], Y7[0:33],  lw = 1.5, label = 'economics')
plt.plot(X[0:33], Y8[0:33],  lw = 1.5, label = 'military')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for i in range(len(us_dates)):
    plt.axvline(us_dates[i], linestyle="dashed", color="black", lw=0.6)
    plt.text(us_dates[i],-3,us_dates_exp[i],rotation=90)
plt.ylabel('%')
plt.title('Speeches topics evolution', y=1.08)
plt.savefig('./figures/1820s.png', bbox_inches='tight')


us_dates = [1861,1865]
us_dates_exp = ['Civil War','Civil War ends']
plt.plot(X[60:93], Y1[60:93],   lw = 1.5, label = 'positive')
plt.plot(X[60:93], Y2[60:93], lw = 1.5, label = 'negative')
#plt.plot(X[60:93], Y3[60:93], lw = 1.5, label = 'uncertainty')
#plt.plot(X[60:93], Y4[60:93], lw = 1.5, label = 'passive')
#plt.plot(X[60:93], Y5[60:93], lw = 1.5, label = 'ethics')
plt.plot(X[60:93], Y6[60:93],  lw = 1.5, label = 'politics')
plt.plot(X[60:93], Y7[60:93],  lw = 1.5, label = 'economics')
plt.plot(X[60:93], Y8[60:93],  lw = 1.5, label = 'military')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for i in range(len(us_dates)):
    plt.axvline(us_dates[i], linestyle="dashed", color="black", lw=0.6)
    plt.text(us_dates[i],-5,us_dates_exp[i],rotation=90)
plt.ylabel('%')
plt.title('Speeches topics evolution', y=1.08)
plt.savefig('./figures/1850s.png', bbox_inches='tight')


us_dates = [ 1914,1918,1929,1941,1945]
us_dates_exp = ['WWI','WWI ends', 'Black Thursday','WWII','WWII ends']
plt.plot(X[120:170], Y1[120:170],   lw = 1.5, label = 'positive')
plt.plot(X[120:170], Y2[120:170], lw = 1.5, label = 'negative')
#plt.plot(X[120:170], Y3[120:170], lw = 1.5, label = 'uncertainty')
#plt.plot(X[120:170], Y4[120:170], lw = 1.5, label = 'passive')
#plt.plot(X[120:170], Y5[120:170], lw = 1.5, label = 'ethics')
plt.plot(X[120:170], Y6[120:170],  lw = 1.5, label = 'politics')
plt.plot(X[120:170], Y7[120:170],  lw = 1.5, label = 'economics')
plt.plot(X[120:170], Y8[120:170], lw = 1.5, label = 'military')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for i in range(len(us_dates)):
    plt.axvline(us_dates[i], linestyle="dashed", color="black", lw=0.6)
    plt.text(us_dates[i],-10,us_dates_exp[i],rotation=90)
plt.ylabel('%')
plt.title('Speeches topics evolution', y=1.08)
plt.savefig('./figures/interest1900s.png', bbox_inches='tight')

us_dates = [ 1974,1991,2001,2008]
us_dates_exp = ['Watergate scandal','Iraq attacks','WTC attack', 'Great Recession']
plt.plot(X[180:220], Y1[180:220],   lw = 1.5, label = 'positive')
plt.plot(X[180:220], Y2[180:220], lw = 1.5, label = 'negative')
#plt.plot(X[180:220], Y3[180:220], lw = 1.5, label = 'uncertainty')
#plt.plot(X[180:220], Y4[180:220], lw = 1.5, label = 'passive')
#plt.plot(X[180:220], Y5[180:220], lw = 1.5, label = 'ethics')
plt.plot(X[180:220], Y6[180:220],  lw = 1.5, label = 'politics')
plt.plot(X[180:220], Y7[180:220],  lw = 1.5, label = 'economics')
plt.plot(X[180:220], Y8[180:220], lw = 1.5, label = 'military')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for i in range(len(us_dates)):
    plt.axvline(us_dates[i], linestyle="dashed", color="black", lw=0.6)
    plt.text(us_dates[i],-8,us_dates_exp[i],rotation=90)
plt.ylabel('%')
plt.title('Speeches topics evolution', y=1.08)
plt.savefig('./figures/2000s.png', bbox_inches='tight')


###############################################################################
### 2.c) Time serie from 1948 (data available)
from scipy.stats.stats import pearsonr

uncert= ytp[ytp.year >= 1948].unc.reset_index(drop=True).values; posit= ytp[ytp.year >= 1948].pos.reset_index(drop=True).values
negat= ytp[ytp.year >= 1948].neg.reset_index(drop=True).values; passive= ytp[ytp.year >= 1948].passive.reset_index(drop=True).values
econ= ytp[ytp.year >= 1948].econ.reset_index(drop=True).values; polit= ytp[ytp.year >= 1948].polit.reset_index(drop=True).values
milit= ytp[ytp.year >= 1948].milit.reset_index(drop=True).values

''' unemployment'''
file = pd.read_table("./timeseries/annual_unemployment.txt",header=None)
#file = pd.read_table("./timeseries/jan_unempl.txt",header=None); unempl = pd.DataFrame(file[1])
unempl = pd.DataFrame(file[1]); unempl = unempl[1].values

corr_unempl = [pearsonr(unempl, uncert), pearsonr(unempl, posit),pearsonr(unempl, negat),
                pearsonr(unempl, passive) ,pearsonr(unempl, econ),pearsonr(unempl, polit),pearsonr(unempl, milit) ] 
corr_unempl = pd.DataFrame(corr_unempl)
corr_unempl[2] = ['uncertainty', 'positive', 'negative', 'passive', 'economy', 'politics', 'military']; corr_unempl.columns = ('unempl corr', 'p-val','topic')


'''inflation rate'''
file2 = pd.read_table("./timeseries/inflation_rate.txt",header=None)
infl = file2[file2[0]>=1948].reset_index(drop=True); infl = pd.DataFrame(infl[1]); infl = infl[1].values

corr_infl = [pearsonr(infl, uncert), pearsonr(infl, posit),pearsonr(infl, negat),
                pearsonr(infl, passive) ,pearsonr(infl, econ),pearsonr(infl, polit),pearsonr(infl, milit) ] 
corr_infl = pd.DataFrame(corr_infl)
corr_infl[2] = ['uncertainty', 'positive', 'negative', 'passive', 'economy', 'politics', 'military']; corr_infl.columns = ('infl corr', 'p-val','topic')

### 2.d) topics on years according to tf-idf

###############################################################################

def ranking(stemmed,data,dictionary, use_tf_idf, n):  
    vocab = get_vocab(stemmed)
    dt_matrix = make_count(stemmed)
    tfidf_matrix = make_TF_IDF(stemmed)

    if (use_tf_idf):
        dtm = tfidf_matrix
    else:
        dtm = dt_matrix

# Get rid of words in the document term matrix not in the dictionary
    dict_tokens_set = set(item for item in dictionary)
    intersection = list(set(dict_tokens_set) & set(vocab))
    vec_positions = [int(token in intersection) for token in vocab] 

# Get the score of each document
    sums = np.zeros(len(dtm))
    for j in range(len(dtm)):
        sums[j] = sum([a * b for a, b in zip(dtm[j], vec_positions)])

# Order them and return the n top documents
    order = sorted(range(len(sums)), key = lambda k: sums[k], reverse=True)
    ordered_year_data_n = [None] * len(dtm)
    ordered_sums = np.zeros(len(dtm))

    counter = 0        
    for num in order:
        ordered_year_data_n[counter] = data.year[num]
        ordered_sums[counter] = sums[num]
        counter += 1

    return list((ordered_year_data_n[0:n], ordered_sums[0:n]))

def tf_idf_dict(stemmed,data,dictionary,n ):
    
    sorted_years,tf_score = ranking(stemmed,data, dictionary, False, n) 
    print ("The highest ranked documents using DTM are:")
    for i in range(len(sorted_years)):
        print ("{0} {1}".format(sorted_years[i], tf_score[i]), set(data.loc[data.year == sorted_years[i]].president)) 

    #TF-IDF
    sorted_years2, tfidf_score = ranking(stemmed,data, dictionary, True, n)  
    print ("The highest ranked documents using TF-IDF are:")
    for i in range(len(sorted_years2)):
        print ("{0} {1}  ".format(sorted_years2[i], tfidf_score[i]), set(data.loc[data.year == sorted_years2[i]].president)) 
        
    return 
###############################################################################
from collections import OrderedDict

####### dt score
# for a given dictionary

#tf_idf_dict(stemmed_y,processed_data_y, positive_dict, 10)

pos_sorted_years,pos_tf_score = ranking(stemmed_y, processed_data_y, positive_dict, False, 10) 
neg_sorted_years,neg_tf_score = ranking(stemmed_y, processed_data_y, negative_dict, False, 10) 
et_sorted_years,et_tf_score = ranking(stemmed_y, processed_data_y, ethic_dict, False, 10) 
pol_sorted_years,pol_tf_score = ranking(stemmed_y, processed_data_y, politic_dict, False, 10) 
ec_sorted_years,ec_tf_score = ranking(stemmed_y, processed_data_y, econ_dict, False, 10) 
mil_sorted_years,mil_tf_score = ranking(stemmed_y, processed_data_y, military_dict, False, 10) 
unc_sorted_years,unc_tf_score = ranking(stemmed_y, processed_data_y, uncert_dict, False, 10) 
pas_sorted_years,pas_tf_score = ranking(stemmed_y, processed_data_y, passive_dict, False, 10) 

DT_score = pd.DataFrame(OrderedDict({'year_p':pos_sorted_years, 'positive':pos_tf_score,
'year_n':neg_sorted_years ,'negative':neg_tf_score, 'y_et':et_sorted_years, 'ethic':et_tf_score, 
'y_pol':pol_sorted_years, 'politics':pol_tf_score, 'y_ec':ec_sorted_years, 'econ':ec_tf_score, 
'y_mil':mil_sorted_years, 'military':mil_tf_score, 'y_u':unc_sorted_years, 'uncertainty':unc_tf_score, 'y_pas':pas_sorted_years, 'passive':pas_tf_score} ))

#check with 2b

yt.sort(columns='pos',axis=0, ascending=False)['year'][:5]
yt.sort(columns='neg',axis=0, ascending=False)['year'][:5]
yt.sort(columns='ethic',axis=0, ascending=False)['year'][:5]
yt.sort(columns='polit',axis=0, ascending=False)['year'][:5]
yt.sort(columns='econ',axis=0, ascending=False)['year'][:5]
yt.sort(columns='milit',axis=0, ascending=False)['year'][:5] 
yt.sort(columns='unc',axis=0, ascending=False)['year'][:5]
yt.sort(columns='passive',axis=0, ascending=False)['year'][:5]
#YAAAAAAAASSSSS WE GET THE SAME

##### tf-idf score
ipos_sorted_years,ipos_tf_score = ranking(stemmed_y, processed_data_y, positive_dict, True, 10) 
ineg_sorted_years,ineg_tf_score = ranking(stemmed_y, processed_data_y, negative_dict, True, 10) 
iet_sorted_years,iet_tf_score = ranking(stemmed_y, processed_data_y, ethic_dict, True, 10) 
ipol_sorted_years,ipol_tf_score = ranking(stemmed_y, processed_data_y, politic_dict, True, 10) 
iec_sorted_years,iec_tf_score = ranking(stemmed_y, processed_data_y, econ_dict, True, 10) 
imil_sorted_years,imil_tf_score = ranking(stemmed_y, processed_data_y, military_dict, True, 10) 
iunc_sorted_years,iunc_tf_score = ranking(stemmed_y, processed_data_y, uncert_dict, True, 10) 
ipas_sorted_years,ipas_tf_score = ranking(stemmed_y, processed_data_y, passive_dict, True, 10) 

TFIDF_score = pd.DataFrame(OrderedDict({'year_p':ipos_sorted_years, 'positive':ipos_tf_score,
'year_n':ineg_sorted_years ,'negative':ineg_tf_score, 'y_et':iet_sorted_years, 'ethic':iet_tf_score, 
'y_pol':ipol_sorted_years, 'politics':ipol_tf_score, 'y_ec':iec_sorted_years, 'econ':iec_tf_score, 
'y_mil':imil_sorted_years, 'military':imil_tf_score, 'y_u':iunc_sorted_years, 'uncertainty':iunc_tf_score, 'y_pas':ipas_sorted_years, 'passive':ipas_tf_score} ))


'''
QUESTION 3
'''

import sklearn
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Comparison of parties post 1860

# First collect names and assign parties to all presidents after first Republican president elected
pres    = sorted(list ( set(data.loc[data.year > 1860].president)))
party   = ['rep']*3 + ['dem']*3 + ['rep']*8 + ['dem']*3 + ['rep']*3 + ['dem']*1 + ['rep']*2 + ['dem'] + ['rep'] + ['dem']*2

pres_party = dict(zip(pres, party))

stemmed, processed_data = data_processing(data)

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
Now we will do the same analysis for presidents post 1965. We choose 1965 because the civil rights act represents a turning point in the ideology of the democratic party, and the start of the 'southern strategy' of republican presidential candidates. This means that (hopefully), both democrats and republicans will be more homogeneous in this analysis.
'''

data_post1965 = processed_data.loc[processed_data.year >= 1965]
data_post1965 = data_post1965.reset_index(drop=True)
parties = [pres_party[i] for i in data_post1965.president]
data_post1965 = data_post1965.assign(party=parties)

stemmed_post1965, processed_post1965 = data_processing(data_post1965)

stemmed_post1965 = custom_stopword_del(stemmed_post1965, our_stopwords)
stemmed_post1965, processed_post1965 = remove_zerolen_strings(stemmed_post1965, processed_post1965)

parties_post1965 = [i for i in processed_post1965.party]
dem_idx = [i for i in range(len(parties_post1965)) if parties_post1965[i] == 'dem']
rep_idx = [i for i in range(len(parties_post1965)) if parties_post1965[i] == 'rep']

tf_idf_post1965 = make_TF_IDF(stemmed_post1965)

cos_sim = cosine_similarity(tf_idf_post1965)

similarity_within_dem = cos_sim[dem_idx,:][:,dem_idx]
similarity_within_rep = cos_sim[rep_idx,:][:,rep_idx]
similarity_between_parties = cos_sim[dem_idx,:][:,rep_idx]

print(np.mean(similarity_within_dem))
print(np.mean(similarity_within_rep))
print(np.mean(similarity_between_parties))

np.mean(similarity_within_dem)/np.mean(similarity_between_parties)
np.mean(similarity_within_rep)/np.mean(similarity_between_parties)

'''
Now do singular value decomposition
'''

U, S, V = svds(tf_idf_post1965, k = 200)

low_rank_approx = U.dot(np.diag(S)).dot(V)

low_rank_cos_sim = cosine_similarity(low_rank_approx)

low_rank_similarity_within_dem = low_rank_cos_sim[dem_idx,:][:,dem_idx]
low_rank_similarity_within_rep = low_rank_cos_sim[rep_idx,:][:,rep_idx]
low_rank_similarity_between_parties = low_rank_cos_sim[dem_idx,:][:,rep_idx]

print(np.mean(low_rank_similarity_within_dem))
print(np.mean(low_rank_similarity_within_rep))
print(np.mean(low_rank_similarity_between_parties))

np.mean(low_rank_similarity_within_dem)/np.mean(low_rank_similarity_between_parties)
np.mean(low_rank_similarity_within_rep)/np.mean(low_rank_similarity_between_parties)

'''
We get here that without latent semantic analysis democrats are on average 1.5% more similar to each other than they are to republicans, and republicans are 8.2% more similar to each other than they are to democrats. However, when we peform LSA we get that democrats are 0.11% more similar to each other than they are to republicans and republicans are 8.2% more similar to each other than they are to democrats. Therefore, LSA does not give us a sharper distinction between parties, even when we concentrate on the last 50 years.
'''

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
