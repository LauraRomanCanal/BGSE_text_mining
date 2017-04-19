''' method 1'''
##########################################################################################

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
        words = set(stem[j])
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


''' method 2'''
################################################################################################################################################
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


def dt_matrix(stemmed,dictionary,dtm):
    vocab = get_vocab(stemmed)
   
# Get rid of words in the document term matrix not in the dictionary
    dict_tokens_set = set(item for item in dictionary)
    intersection = list(set(dict_tokens_set) & set(vocab))
    vec_positions = [int(token in intersection) for token in vocab] 

# Get the score of each document
    sums = np.zeros(len(dtm))

    for j in range(len(dtm)):
        sums[j] = sum([a * b for a, b in zip(dtm[j], vec_positions)])
    return sums

##########################################################################################

data_by_years= pd.DataFrame(data)
data_by_years = data_by_years.groupby('year', sort=False, as_index=True)['speech'].apply(' '.join)
data_by_years = data_by_years.reset_index()
stemmed_y, processed_data_y = data_processing(data_by_years)

positive_dict = read_dictionary('./dictionaries/positive.csv'); negative_dict = read_dictionary('./dictionaries/negative.csv')
ethic_dict = read_dictionary('./dictionaries/ethics.csv'); politic_dict = read_dictionary('./dictionaries/politics.csv')
econ_dict = read_dictionary('./dictionaries/econ.csv'); military_dict = read_dictionary('./dictionaries/military.csv')
uncert_dict = read_dictionary('./dictionaries/uncertainty.csv'); passive_dict = read_dictionary('./dictionaries/passive.csv')

#method 1
method1 = docs_dict_matrix(stemmed_y,positive_dict,negative_dict,ethic_dict,politic_dict,econ_dict,military_dict,uncert_dict,passive_dict )


#method2
dtm = make_count(stemmed_y)
# or use
#dtm = make_TF_IDF(stemmed_y)

pos_r = dt_matrix(stemmed_y,positive_dict, dtm)
neg_r = dt_matrix(stemmed_y,negative_dict,dtm)
unc_r = dt_matrix(stemmed_y, uncert_dict,dtm)
pa_r = dt_matrix(stemmed_y, passive_dict,dtm)
eth_r = dt_matrix(stemmed_y, ethic_dict,dtm)
pol_r = dt_matrix(stemmed_y, politic_dict,dtm)
ec_r  = dt_matrix(stemmed_y, econ_dict,dtm)
mil_r = dt_matrix(stemmed_y, military_dict,dtm)

method2 = np.transpose([pos_r,neg_r, unc_r,pa_r, eth_r, pol_r, ec_r, mil_r ])
m2 = pd.DataFrame(method2, columns=['pos', 'neg', 'unc', 'passive', 'ethic', 'polit', 'econ', 'milit'])
m2['total'] = m2.sum(axis=1)

method1



  