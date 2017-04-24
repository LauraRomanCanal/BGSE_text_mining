#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:29:18 2017

@author: Laura
"""
from nltk import PorterStemmer
import re

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

positive_dict = read_dictionary('./dictionaries/positive.csv'); negative_dict = read_dictionary('./dictionaries/negative.csv'); uncert_dict = read_dictionary('./dictionaries/uncertainty.csv')
ethic_dict = read_dictionary('./dictionaries/ethics.csv'); politic_dict = read_dictionary('./dictionaries/politics.csv')
econ_dict = read_dictionary('./dictionaries/econ.csv'); military_dict = read_dictionary('./dictionaries/military.csv')
passive_dict = read_dictionary('./dictionaries/passive.csv')


dict1 = set(item for item in positive_dict)
dict2 = set(item for item in negative_dict)
dict3 = set(item for item in uncert_dict)
dict4 = set(item for item in ethic_dict)
dict5 = set(item for item in politic_dict)
dict6 = set(item for item in econ_dict)
dict7 = set(item for item in military_dict)
dict8 = set(item for item in passive_dict)


i12 = list(set(dict1) & set(dict2)  )
i13 = list(set(dict1) & set(dict3)  )
i14 = list(set(dict1) & set(dict4)  )
i15 = list(set(dict1) & set(dict5)  )
i16 = list(set(dict1) & set(dict6)  )
i17 = list(set(dict1) & set(dict7)  )
i18 = list(set(dict1) & set(dict8)  )
positive_dict = list(set(positive_dict)-set(i12)-set(i13)-set(i14)-set(i15)-set(i16)-set(i17)-set(i18))
dict1 = set(item for item in positive_dict)

i23 = list(set(dict2) & set(dict3)  )
i24 = list(set(dict2) & set(dict4)  )
i25 = list(set(dict2) & set(dict5)  )
i26 = list(set(dict2) & set(dict6)  )
i27 = list(set(dict2) & set(dict7)  )
i28 = list(set(dict2) & set(dict8)  )
negative_dict = list(set(negative_dict)-set(i23)-set(i24)-set(i25)-set(i26)- set(i27)-set(i28))
dict2 = set(item for item in negative_dict)

i35 = list(set(dict3) & set(dict5)  )
i38 = list(set(dict3) & set(dict8)  )
uncert_dict = list(set(uncert_dict)-set(i35)-set(i38))
dict3 = set(item for item in uncert_dict)

i45 = list(set(dict4) & set(dict5)  )
i46 = list(set(dict4) & set(dict6)  )
i48 = list(set(dict4) & set(dict8)  )
ethic_dict = list(set(ethic_dict)-set(i45)-set(i46)-set(i48))
dict4 = set(item for item in ethic_dict)

i56 = list(set(dict5) & set(dict6)  )
i57 = list(set(dict5) & set(dict7)  )
i58 = list(set(dict5) & set(dict8)  )
politic_dict = list(set(politic_dict)-set(i56)-set(i57)-set(i58))
dict5 = set(item for item in politic_dict)

i67 = list(set(dict6) & set(dict7)  )
i68 = list(set(dict6) & set(dict8)  )
econ_dict = list(set(econ_dict)-set(i67)-set(i68))
dict6 = set(item for item in econ_dict)

i78 = list(set(dict7) & set(dict8) )
passive_dict = list(set(passive_dict)-set(i78))
