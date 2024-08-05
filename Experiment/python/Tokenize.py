import re
import string
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.chunk import ne_chunk
from nltk.collocations import *
from sklearn.preprocessing import normalize
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Define function named `tokenize` that takes a list of documents and optional parameters
def tokenize(docs, lemmatized=True, remove_stopword=True, bigram=True):
    # Initialize an empty list
    tokenized_docs = []
    # Loop through each document in the list of documents
    for doc in docs:
        # Define a regex
        pattern=r'\w[\w\',-]*\w'                        
        tokens=nltk.regexp_tokenize(doc, pattern)
        # Convert all tokens to lowercase 
        tokens_list = list(map(str.lower,tokens))
        # Generate bigrams from the tokens using Gensim's `Phrases` and `Phraser` classes, if `bigram` is True
        if bigram==True: 
            phrases = Phrases([tokens_list], min_count=2, threshold=0.8, \
                          scoring='npmi')
            bigram = Phraser(phrases)
            text = ' '.join(tokens_list)
            tokens_list = bigram[text.split()]
        # Lemmatize the tokens using NLTK's `WordNetLemmatizer` class, if `lemmatized` is True
        if lemmatized==True:
            lemma = WordNetLemmatizer()
            lem = map(lemma.lemmatize, tokens_list)
            tokens_list = list(lem)
        # Remove stop words from the tokens using NLTK's `stopwords.words()` function, if `remove_stopword` is True
        if remove_stopword==True:
            tokens_list = [word for word in tokens_list if not word in stopwords.words("english")]
        # Append the list of tokens to `tokenized_docs`
        tokenized_docs.append(tokens_list)
            
    # Return the list of tokenized documents
    return tokenized_docs

def compute_tfidf(tokenized_docs):
    smoothed_tf_idf = None # initialize the variable
    # Create a dictionary of document frequencies, using the tokenized_docs
    docs_tokens={idx:nltk.FreqDist(doc) \
             for idx,doc in enumerate(tokenized_docs)}
    # Create a Document-Term-Matrix (DTM) from the dictionary
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index")
    # Fill any missing values in the DTM with 0
    dtm=dtm.fillna(0)
    # Sort the DTM by index
    dtm = dtm.sort_index(axis = 0)
    # Get the term frequency matrix
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    # Divide the DTM by the length of the document to get normalized term frequency
    tf=np.divide(tf, doc_len[:,None])
    # Get the document frequency matrix
    df=np.where(tf>0,1,0)
    # Smooth the IDF with log transform and apply +1 smoothing
    smoothed_idf=np.log(np.divide(len(tokenized_docs)+1, np.sum(df, axis=0)+1))+1
    # Compute smoothed TF-IDF
    smoothed_tf_idf = normalize(tf*smoothed_idf)
    
    return smoothed_tf_idf
# Calculate similarrity by the traditional way
def assess_similarity(question_tokens, answer_tokens, pair_QA, label_feature = None):
    # Initialize variables
    result = None
    number = len(question_tokens)
    # Combine question and answer tokens to create a corpus
    corpus = question_tokens + answer_tokens
    # Calculate the TF-IDF score for the corpus
    tf_idf_corpus = compute_tfidf(corpus)
    # Split the TF-IDF score matrix into question and answer matrices
    tf_idf_ques = tf_idf_corpus[:number,:]
    tf_idf_ans = tf_idf_corpus[number:,:]
    # Initialize an empty list to store similarity scores
    simi_que_ans = []
    # Calculate cosine similarity between each question and answer pair
    for i in range(number):
        each_ans = tf_idf_ans[i].reshape(1, -1)
        each_ques = tf_idf_ques[i].reshape(1, -1)
        each_simi = cosine_similarity(each_ques, each_ans)[0][0]
        simi_que_ans.append(each_simi)
    # Create a DataFrame to store similarity scores and pair of questions and answers
    if label_feature:
        result = pd.DataFrame({
            "question_anwser_sim": simi_que_ans,
            "Pair QA": pair_QA})
    else:
        result = pd.DataFrame({
            "question_anwser_sim": simi_que_ans})
    return result

