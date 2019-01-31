# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 12:28:09 2018

@author: Neehar
"""

#file_path = 'E:\TAMU\Sem 2\656\Python SAS\HW8'

import os
#os.chdir(file_path)
import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  
print(files)

adoc=[]
z=[1,2,3,4,5,6,7,8]
for d in z:
    with open ("E:\TAMU\Sem 2\656\Python SAS\HW8\T%x.txt" %d, "r") as text_file:
        adoc.append(text_file.read())


# Convert to all lower case - required
a_text=[]
#a_text = ("%s" %data[0:1]).lower()
for d in range(8):
    c= ("%s" %adoc[d]).lower()
    c = c.replace('-', ' ')
    c = c.replace('_', ' ')
    c = c.replace(',', ' ')
    c = c.replace("'nt", " not")
    c = c.replace("n't", " not")
    a_text.append(c)


# Tokenize
tokens=[]
for d in range(8):
    t = word_tokenize(a_text[d])
    t = [word.replace(',', '') for word in t]
    t = [word for word in t if ('*' not in word) and word != "''" and \
              word !="``"]    
# Remove punctuation
    for word in t:
        word = re.sub(r'[^\w\d\s]+','',word)
    print("\nDocument %x contains a total of" %d, len(t), " terms.")
    tokens.append(t)


# POS Tagging
tagged_tokens=[]
for d in range(8):
    tt = nltk.pos_tag(tokens[d])
    pos_list = [word[1] for word in tt if word[1] != ":" and \
                word[1] != "."]
    pos_dist = FreqDist(pos_list)
    pos_dist.plot(title="Parts of Speech")
    for pos, frequency in pos_dist.most_common(pos_dist.N()):
        print('{:<15s}:{:>4d}'.format(pos, frequency))
    tagged_tokens.append(tt)
    


# Remove stop words
stop = stopwords.words('english') + list(string.punctuation)
stop_tokens=[]
for d in range(8):
    st = [word for word in tagged_tokens[d] if word[0] not in stop]
    # Remove single character words and simple punctuation
    st = [word for word in st if len(word[0]) > 1]
    # Remove numbers and possive "'s"
    st = [word for word in st \
                   if (not word[0].replace('.','',1).isnumeric()) and \
                   word[0]!="'s" ]
    print("\nDocument %x contains" %d, len(st), \
                      " terms after removing stop words.\n")
    token_dist = FreqDist(st)
    for word, frequency in token_dist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word[0], frequency))
    stop_tokens.append(st)


# Lemmatization - Stemming with POS
# WordNet Lematization Stems using POS
stemmer = SnowballStemmer("english")
wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
wnl = WordNetLemmatizer()
final_stemmed_tokens=[]
for d in range(8):
    stemmed_tokens = []
    for token in stop_tokens[d]:
        term = token[0]
        pos  = token[1]
        pos  = pos[0]
        try:
            pos   = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))   
    print("Document %x contains" %d, len(stemmed_tokens), "terms after stemming.\n")
    final_stemmed_tokens.append(stemmed_tokens)


df=[]
for d in range(8):
    # Word distribution
    #fdist = FreqDist(word for word in stemmed_tokens)
    fdist = FreqDist(final_stemmed_tokens[d]).most_common(20)
    df1=pd.DataFrame(fdist,columns=['Word','Freq-Doc%x' %d])
    df.append(df1)
    

Td_Matrix = df[0].merge(df[1], how='outer').merge(df[2], how='outer').merge(df[3], how='outer').merge(df[4], how='outer').merge(df[5], how='outer').merge(df[6], how='outer').merge(df[7], how='outer').fillna(0)
total_cnt = Td_Matrix.iloc[:,1:9].sum(axis=1)
# Td_Matrix = pd.DataFrame(Td_Matrix, total_cnt)
total_cnt= pd.DataFrame({'total':total_cnt})
# Td_Matrix.merge(total_cnt, how='outer', on=index)
Td_Matrix = Td_Matrix.join(total_cnt)
Td_Matrix=Td_Matrix.sort_values(by='total', ascending=False)[0:19]