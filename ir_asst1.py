"""
Domain Specific Search Retrival System by :

S.Devendra Dheeraj Gupta  -  2017B5A70670H
K.Srinivas  -  2017B3A70746H
Abhirath Singh -  2018A7PS0521H
Ritvik C -  2018A7PS0180H
"""

import glob
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from collections import Counter
import numpy as np
from collections import OrderedDict

ps = PorterStemmer()
word_in_doc = {}  # Maintains a dict of terms for every doc
count = {}        # Maintains the count of terms across docs


def read_files(path):
    """
    this function takes input absolute path of the dataset
    and returns a dictionary dict which has docs name as key and
    the value as the text of the docs.
    """

    # Reads data from .txt files and store them in a dict
    dict = {}
    files = glob.glob(path)
    for file in files:
        name = file.split('/')[-1]
        print("Reading file: ",name)
        with open(file, 'r', errors='ignore') as f:
            data = f.read()
        dict[name] = data
    return dict

def invertedIndex(dict):
    """
    In this function we remove stop words and also
    stem the words using ntlk PorterStemmer library.
    This function takes the dict as input and returns
    list of words which are not in stop words. Also we precompute
    word_in_doc and count dictionary to be use in other function.
    """
    # Tokenizes, removes stop words and stems words to construct index.
    # string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    stop = stopwords.words('english') + list(string.punctuation) + ['\n']
    wordList = []

    for doc_id in dict.keys():
        word_in_doc[doc_id] = {}
        count[doc_id] = {}

    for doc_id, doc in dict.items():
        for word in word_tokenize(doc.lower()):
            word = ps.stem(word) # Stemming
            word_in_doc[doc_id][word] = 1
            try:
                count[doc_id][word] += 1
            except:
                count[doc_id][word] = 1
            if not word in stop:
                wordList.append(word)
    print("Done with wordlist")
    return wordList

def termFrequency(vocab, dict):
    """
    Term frequency is how many times a term is present in the doc.
    We use the count dictionary that we computed aleady.
    """

    # Calucates tf by searching across all docs using count dict.
    tf = {}
    for doc_id in dict.keys():
        tf[doc_id] = {}
    for word in vocab:
        for doc_id,doc in dict.items():
            try:
                tf[doc_id][word] = count[doc_id][word]
            except:
                tf[doc_id][word] = 0
    print("Done with termFrequency")
    return tf

def docFrequency(vocab, dict):
    """
    The number of docs containing the word.
    We use count dictionary to return the df dictionary.
    """

    # Calculates docfrequency of a term.
    df = {}
    for word in vocab:
        cnt = 0
        for doc in dict.keys():
            try:
                if word_in_doc[doc][word] == 1:
                    cnt += 1
            except KeyError:
                pass
        df[word] = cnt
    print("Done with docFrequency")
    return df

def inverseDocFreq(vocab,doc_fre,length):
    """
    returns idf dictionary.
    """

    # Calculates inverseDocFreq
    idf= {}
    for word in vocab:
        idf[word] = np.log10((length) / doc_fre[word])
    print("Done with inverseDocFreq")
    return idf

def tfidf(vocab,tf,idf_scr,doc_dict):
    """
    Calculate tf-idf to be used in calculating the score of query.
    """
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            try:
                tf_idf_scr[doc_id][word] = (np.log10(1 + tf[doc_id][word])) * idf_scr[word]
                # tf_idf_scr[doc_id][word] = (tf[doc_id][word]) * idf_scr[word]
            except:
                tf_idf_scr[doc_id][word] = 0
    print("Done with tf_idf")
    return tf_idf_scr

def vectorSpaceModel(query, doc_dict,tfidf_scr, vocab):
    """
    Calculates relevant documents related to the query.
    """

    # Removing punctuation symbols from query and tokenizing the query.
    query_vocab_stripped = list(set([x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query.lower().split()]))
    query_list = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query.lower().split()]

    # Storing the count of words to calculate relevance score.
    query_wc = {}
    for word in query_vocab_stripped:
        query_wc[word] = query_list.count(word)

    matching_scores = {}
    for doc_id in doc_dict.keys():
        score = 0
        for word in query_vocab_stripped:
            cnt = query_wc[word]
            word = ps.stem(word)
            if word not in vocab:
                continue
            else:
                # Calculating score for a doc related to query.
                score += cnt * tf_idf[doc_id][word]
        matching_scores[doc_id] = score
    sorted_value = OrderedDict(sorted(matching_scores.items(), key=lambda x: x[1], reverse = True))
    best10 = {k: sorted_value[k] for k in list(sorted_value)[:10]}
    return best10 # Getting top 10 results.

if __name__  == "__main__":
   path_to_folder = 'D:/4-1/IR/Assignments/1/Testdocs'
   path = path_to_folder + '/*.txt'
   print("Preprocessing dataset. Pleasie wait...")
   start = time.time()
   docs = read_files(path)                       #returns a dictionary of all docs
   M = len(docs)                                 #number of files in dataset
   print(M,"files read from the destination folder.")
   wordList = invertedIndex(docs)                #returns a list of tokenized words
   vocab = list(set(wordList))                   #returns a list of unique words
   print(len(vocab),"unique words in the Index.")
   tf_dict = termFrequency(vocab, docs)          #returns term frequency
   df_dict = docFrequency(vocab, docs)           #returns document frequencies
   idf_dict = inverseDocFreq(vocab,df_dict,M)    #returns idf scores
   tf_idf = tfidf(vocab,tf_dict,idf_dict,docs)   #returns tf-idf socres

   print("Preprocessing Time :","--- %s seconds---" % (time.time() - start))
   while True:
       x = input("Press 0 to exit or press 1 to give another query\n")
       if x == "0":
           break
       print("Enter a query:")
       lines = []
       while True:
           line = input()
           if line:
               lines.append(line)
           else:
               break
       query = '\n'.join(lines)
       query_start = time.time()
       ans = vectorSpaceModel(query, docs, tf_idf, vocab)
       first_key = list(ans.keys())[0]
       if(ans[first_key] == 0):
           print("ERROR: No matching documents found..!!!")
       else:
           print("The top 10 documents for your query are:")
           for key, val in ans.items():
               print(key + " :  " + str(val))
       print("Query Retrieval Time :","--- %s seconds---" % (time.time() - query_start))
       print("\n")
   print("Thank You...")
