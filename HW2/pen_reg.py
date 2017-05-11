import numpy as np
import pandas as pd
import nltk
import os
import sys
import scipy.sparse as ssp
import time
import matplotlib
import tqdm
import lda
from lda import LDA
from tqdm import tqdm
from numpy.random import dirichlet
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import data_processing, get_vocab, make_count

data = pd.read_table("../HW1/speech_data_extend.txt",encoding="utf-8")
data_post1945 = data.loc[data.year >= 1945]
%time stemmed, processed_data = data_processing(data_post1945)

pres = ['BushI', 'BushII', 'Carter', 'Clinton', 'Eisenhower', 'Ford', 'JohnsonII', 'Kennedy',\
'Nixon', 'Nixon', 'Obama', 'Reagan', 'RooseveltII', 'Truman']
party = ['R','R','D','D','R','R','D','D','R','R','D','R','D','D']
pres_party = dict(zip(pres, party))

parties = [pres_party[i] for i in processed_data.president]

vocab   = get_vocab(stemmed)
D       = len(stemmed)
V       = len(vocab)
idx     = dict(zip(vocab,range(len(vocab))))

X = make_count(stemmed, idx).toarray()

###############
# LASSO
###############

from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, parties)

model = LogisticRegression(penalty = 'l1')

model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy_score(y_test, preds)
confusion_matrix(y_test, preds)

######################
# Ridge
######################

model = LogisticRegression(penalty = 'l2')

model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy_score(y_test, preds)
confusion_matrix(y_test, preds)

###########################
# Using topic assignments
###########################

lda_labels = LDA(n_topics = 10)
lda_labels.fit(X)
dt = lda_labels.doc_topic_

X_train, X_test, y_train, y_test = train_test_split(dt, parties)

model = LogisticRegression(penalty = 'l1')
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy_score(y_test, preds)
confusion_matrix(y_test, preds)

model = LogisticRegression(penalty = 'l2')
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy_score(y_test, preds)
confusion_matrix(y_test, preds)
