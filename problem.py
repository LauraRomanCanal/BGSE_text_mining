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

method1 = docs_dict_matrix(stemmed_y,positive_dict,negative_dict,ethic_dict,politic_dict,econ_dict,military_dict,uncert_dict,passive_dict )

def ranking(stemmed,data,dictionary, use_tf_idf, n):  


    if (use_tf_idf):
        dtm = make_TF_IDF(stemmed)
    else:
        dtm = make_count(stemmed)

# Scoring of each document for a given dictionary and method (count or tfidf)
    sums = dt_matrix(stemmed,dictionary,dtm)
    
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
  
def dt_matrix(stemmed,dictionary):
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

pos_r = dt_matrix(stemmed_y,positive_dict)
neg_r = dt_matrix(stemmed_y,negative_dict)
unc_r = dt_matrix(stemmed_y, uncert_dict)
eth_r = dt_matrix(stemmed_y, ethic_dict)
pol_r = dt_matrix(stemmed_y, politic_dict)
ec_r  = dt_matrix(stemmed_y, econ_dict)
mil_r = dt_matrix(stemmed_y, military_dict)

method2 = np.transpose([pos_r,neg_r, unc_r,eth_r, pol_r, ec_r, mil_r ])
a.sum