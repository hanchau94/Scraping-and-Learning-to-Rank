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

## Train the model to predict the Category for Question
data1=pd.read_csv("Replace_letter.csv")
data2=pd.read_csv("Insert_letter.csv")
data3=pd.read_csv("Delete_word.csv")

Category_feature = list(dataset["Category"])*4
Question_feature = list(dataset["Questions"]) + list(data1["Questions"]) \
+ list(data2["Questions"])+list(data3["Questions"])

clustering_data = pd.DataFrame({"Category": Category_feature,
                         "Questions":Question_feature})
clustering_data = clustering_data.dropna()

encoder = LabelEncoder()
clustering_data['Category'] = encoder.fit_transform(clustering_data['Category'])

train, test = train_test_split(clustering_data, test_size=0.2, random_state=0)
train_X = tokenize(train["Questions"], lemmatized=True, remove_stopword=True, bigram=True)
test_X = tokenize(test["Questions"], lemmatized=True, remove_stopword=True, bigram=True)
def identity_tokenizer(text):
    return text
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer,lowercase=False)    
xtrain = tfidf.fit_transform(train_X)
xtest = tfidf.transform(test_X)

model_tree = tree.DecisionTreeClassifier()
# Evaluate model performance using 10-fold cross-validation
scores_tree = cross_val_score(model_tree, xtrain, train["Category"], cv=10)

model_ran = RandomForestClassifier()
scores_ran = cross_val_score(model_ran, xtrain, train["Category"], cv=10)

model_log = LogisticRegression()
scores_log = cross_val_score(model_log, xtrain, train["Category"], cv=10)

## Model 1: Using only one feature, the consine similarity, to train
# new_data is taken from Pre-processing
train_data, test_data = train_test_split(new_data , test_size=0.2,\
                    random_state=42)
X_train = train_data[["Category","Questions","Answers"]]
Y_train = train_data["Label"]
X_test = test_data[["Category","Questions","Answers"]]
Y_test = test_data["Label"]

question_tokens = tokenize(X_train["Questions"],lemmatized=True, remove_stopword=True, bigram = True)
answer_tokens = tokenize(X_train["Answers"], lemmatized=True, remove_stopword=True, bigram = True)
result_train= assess_similarity(question_tokens, answer_tokens, pair_QA = None,label_feature = None)

question_tokens = tokenize(X_test["Questions"],lemmatized=True, remove_stopword=True, bigram = True)
answer_tokens = tokenize(X_test["Answers"], lemmatized=True, remove_stopword=True, bigram = True)
result_test= assess_similarity(question_tokens, answer_tokens, pair_QA = None,label_feature = None)


data_train = np.array(result_train["question_anwser_sim"]).reshape(-1,1)
data_test = np.array(result_test["question_anwser_sim"]).reshape(-1,1)

# to train SVM model with the cosine similarity
clf = svm.LinearSVC().fit(data_train, Y_train)
predict_p =clf.decision_function(data_test)
y_pred =clf.predict(data_test)


fpr, tpr, thresholds = roc_curve(Y_test, predict_p, pos_label=1)
precision, recall, thresholds = precision_recall_curve(Y_test, predict_p, pos_label=1)

## Method 2: Using TF-IDF matrix and the cosine similarity to train
def svm_model(X_train,Y_train,X_test,Y_test,remove_stopword=True,bigram = True):
    # set several situations with changing 2 parameters.
    if remove_stopword==True and bigram == False:
        question_tokens = tokenize(X_train["Questions"],lemmatized=True, remove_stopword=True, bigram = False)
        answer_tokens = tokenize(X_train["Answers"], lemmatized=True, remove_stopword=True, bigram = False)
        result_train= assess_similarity(question_tokens, answer_tokens, pair_QA = None,label_feature = None)

        question_tokens = tokenize(X_test["Questions"],lemmatized=True, remove_stopword=True, bigram = False)
        answer_tokens = tokenize(X_test["Answers"], lemmatized=True, remove_stopword=True, bigram = False)
        result_test= assess_similarity(question_tokens, answer_tokens, pair_QA = None,label_feature = None)
        
        
        # to tokenize the question and answer
        tokenized_train = tokenize(X_train["Questions"]+" "+X_train["Answers"],\
                                   lemmatized=True, remove_stopword=True, bigram = False)
        tokenized_test = tokenize(X_test["Questions"]+" "+X_test["Answers"],\
                                   lemmatized=True, remove_stopword=True, bigram = False)
    
    
    elif remove_stopword==True and bigram == True:
        question_tokens = tokenize(X_train["Questions"],lemmatized=True,\
                                   remove_stopword=True, bigram = True)
        answer_tokens = tokenize(X_train["Answers"], lemmatized=True, \
                                 remove_stopword=True, bigram = True)
        result_train= assess_similarity(question_tokens, answer_tokens,\
                                        pair_QA = None,label_feature = None)

        question_tokens = tokenize(X_test["Questions"],lemmatized=True,\
                                   remove_stopword=True, bigram = True)
        answer_tokens = tokenize(X_test["Answers"], lemmatized=True, \
                                 remove_stopword=True, bigram = True)
        result_test= assess_similarity(question_tokens, answer_tokens, \
                                       pair_QA = None,label_feature = None)
        
        tokenized_train = tokenize(X_train["Questions"]+" "+X_train["Answers"],\
                                   lemmatized=True, remove_stopword=True, bigram = True)
        tokenized_test = tokenize(X_test["Questions"]+" "+X_test["Answers"],\
                                   lemmatized=True, remove_stopword=True, bigram = True)
        
    elif remove_stopword==False and bigram == False:
        question_tokens = tokenize(X_train["Questions"],lemmatized=False,\
                                   remove_stopword=False, bigram = False)
        answer_tokens = tokenize(X_train["Answers"], lemmatized=False, \
                                 remove_stopword=False, bigram = False)
        result_train= assess_similarity(question_tokens, answer_tokens,\
                                        pair_QA = None,label_feature = None)

        question_tokens = tokenize(X_test["Questions"],lemmatized=False,\
                                   remove_stopword=False, bigram = False)
        answer_tokens = tokenize(X_test["Answers"], lemmatized=False, \
                                 remove_stopword=False, bigram = False)
        result_test= assess_similarity(question_tokens, answer_tokens, \
                                       pair_QA = None,label_feature = None)
        
        # to tokenize the question and answer
        tokenized_train = tokenize(X_train["Questions"]+" "+X_train["Answers"],\
                                   lemmatized=False, remove_stopword=False, bigram = False)
        tokenized_test = tokenize(X_test["Questions"]+" "+X_test["Answers"],\
                                   lemmatized=False, remove_stopword=False, bigram = False) 
      
    elif remove_stopword==False and bigram == True:
        question_tokens = tokenize(X_train["Questions"],lemmatized=False,\
                                   remove_stopword=False, bigram = True)
        answer_tokens = tokenize(X_train["Answers"], lemmatized=False, \
                                 remove_stopword=False, bigram = True)
        result_train= assess_similarity(question_tokens, answer_tokens,\
                                        pair_QA = None,label_feature = None)

        question_tokens = tokenize(X_test["Questions"],lemmatized=False,\
                                   remove_stopword=False, bigram = True)
        answer_tokens = tokenize(X_test["Answers"], lemmatized=False, \
                                 remove_stopword=False, bigram = True)
        result_test= assess_similarity(question_tokens, answer_tokens, \
                                       pair_QA = None,label_feature = None)
        
        # to tokenize the question and answer
        tokenized_train = tokenize(X_train["Questions"]+" "+X_train["Answers"],\
                                   lemmatized=False, remove_stopword=False, bigram = True)
        tokenized_test = tokenize(X_test["Questions"]+" "+X_test["Answers"],\
                                   lemmatized=False, remove_stopword=False, bigram = True)   
    # generate tfidf matrix
    tfidf_vector = TfidfVectorizer(tokenizer=identity_tokenizer,lowercase=False)    
    xtrain = tfidf_vector.fit_transform(tokenized_train)
    xtest = tfidf_vector.transform(tokenized_test)
    # to create the new dataset
    xtrain_new = scipy.sparse.hstack([xtrain,np.array(result_train["question_anwser_sim"]).reshape(-1,1)])
    xtest_new = scipy.sparse.hstack([xtest,np.array(result_test["question_anwser_sim"]).reshape(-1,1)])
    # train the SVM model
    model = svm.LinearSVC()
    clf = CalibratedClassifierCV(model).fit(xtrain_new, Y_train)
    # predict_p =clf.decision_function(xtest_new)
    predict_p =clf.predict_proba(xtest_new)[:,1]
    y_pred =clf.predict(xtest_new)

    print(classification_report(Y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(Y_test, predict_p, \
                                     pos_label=1)
    precision, recall, thresholds = precision_recall_curve(Y_test, predict_p, pos_label=1)
    # calculate auc and prc
    print("AUC: {:.2%}".format(auc(fpr, tpr)),", PRC: {:.2%}".format(auc(recall, precision)))
    return tfidf_vector, clf