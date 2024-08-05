import re
import string
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import *
from sklearn.preprocessing import normalize
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

## Test when retrieve a query with Top 1, 3, or 5 responses:
def test_retrive(questions, top =1,remove_stopword=False,bigram = False, change_para=True):
    score = [0]*len(questions)
    index = 0
    for text in questions:
        test_text = tokenize([text], lemmatized=True, remove_stopword=True, bigram=False)
        test_text = tfidf.transform(test_text)
        y_pred = model_ran.predict(test_text)
        new_answer = list(dataset[dataset.Category == name_cate[y_pred[0]]]["Answers"])
        new_ques = text*len(new_answer)
        new_table = pd.DataFrame({"Questions":new_ques,
                                 "Answers": new_answer})
        if remove_stopword==True and bigram == False:
            question_new = tokenize(new_table["Questions"],lemmatized=True,\
                                    remove_stopword=True, bigram = False)
            answer_new = tokenize(new_table["Answers"], lemmatized=True,\
                                  remove_stopword=True, bigram = False)
            simi_new= assess_similarity(question_new, answer_new, pair_QA = None,\
                                        label_feature = None)
            tokenized_test = tokenize(new_table["Questions"]+new_table["Answers"],\
                                       lemmatized=True, remove_stopword=True, bigram = False)
            
        elif remove_stopword==True and bigram == True:
            question_new = tokenize(new_table["Questions"],lemmatized=True,\
                                    remove_stopword=True, bigram = True)
            answer_new = tokenize(new_table["Answers"], lemmatized=True, \
                                  remove_stopword=True, bigram = True)
            simi_new= assess_similarity(question_new, answer_new, pair_QA = None,\
                                        label_feature = None)
            tokenized_test = tokenize(new_table["Questions"]+new_table["Answers"],\
                                       lemmatized=True, remove_stopword=True, bigram = True)
            
        elif remove_stopword==False and bigram == False:
            question_new = tokenize(new_table["Questions"],lemmatized=False,\
                                    remove_stopword=False, bigram = False)
            answer_new = tokenize(new_table["Answers"], lemmatized=False, \
                                  remove_stopword=False, bigram = False)
            simi_new= assess_similarity(question_new, answer_new, pair_QA = None,\
                                        label_feature = None)
            tokenized_test = tokenize(new_table["Questions"]+new_table["Answers"],\
                                       lemmatized=False, remove_stopword=False, bigram = False)

       if change_para:    
            new_test = tfidf_vector.transform(tokenized_test)
            new_model = scipy.sparse.hstack([new_test,np.array(simi_new["question_anwser_sim"]).reshape(-1,1)])
            pro = clf.predict_proba(new_model)
        else:
            new_test = tfidf_vector_1.transform(tokenized_test)
            new_model = scipy.sparse.hstack([new_test,np.array(simi_new["question_anwser_sim"]).reshape(-1,1)])
            pro = clf_1.predict_proba(new_model)
        
        if top==1:
            top_1 = new_table["Answers"][np.argmax(pro[:,1])]
            if final_dataset["Answers"][index] in [top_1]:
                score[index]=1
        elif top==3:
            top_3 = new_table["Answers"][pro[:,1].argsort()[::-1][:3]]
            if final_dataset["Answers"][index] in list(top_3):
                score[index]=1
        elif top==5:
            top_5 = new_table["Answers"][pro[:,1].argsort()[::-1][:5]]
            if final_dataset["Answers"][index] in list(top_5):
                score[index]=1  
        index += 1
    return score

## Visualize retrieve for a query with Top 3 responses
def retrive_doc(text):
    test_text = tokenize([text], lemmatized=True, remove_stopword=True, bigram=True)
    test_text = tfidf.transform(test_text)
    y_pred = model_ran.predict(test_text)
    new_answer = list(dataset[dataset.Category == name_cate[y_pred[0]]]["Answers"])
    new_ques = text*len(new_answer)
    new_table = pd.DataFrame({"Questions":new_ques,
                             "Answers": new_answer})
    question_new = tokenize(new_table["Questions"],lemmatized=True, remove_stopword=True, bigram = False)
    answer_new = tokenize(new_table["Answers"], lemmatized=True, remove_stopword=True, bigram = False)
    simi_new= assess_similarity(question_new, answer_new, pair_QA = None,label_feature = None)
    tokenized_test = tokenize(new_table["Questions"]+new_table["Answers"],\
                               lemmatized=True, remove_stopword=True, bigram = False)
    new_test = tfidf_vector.transform(tokenized_test)
    new_model = scipy.sparse.hstack([new_test,np.array(simi_new["question_anwser_sim"]).reshape(-1,1)])
    # distance = clf.decision_function(new_model)
    pro = clf.predict_proba(new_model)
    top_3 = new_table["Answers"][pro[:,1].argsort()[::-1][:3]]
    for i in range(len(list(top_3))):
        print(f"{(i+1)}. {list(top_3)[i]}\n")
 