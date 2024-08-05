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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Load data
data = new_data

# Preprocessing
stop_words = nltk.corpus.stopwords.words('english')

def clean_text(text, apply_cleaning=True):
    if apply_cleaning:
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

filtered_data['Question'] = filtered_data['Question'].apply(clean_text)
filtered_data['Answer'] = filtered_data['Answer'].apply(lambda x: clean_text(x, apply_cleaning=False))

# Encoding labels
label_encoder = LabelEncoder()
filtered_data['Label'] = label_encoder.fit_transform(filtered_data['Label'])

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_data['Question'])
y = filtered_data['Label']

# BiLSTM Model
model = Sequential()
model.add(Embedding(input_dim=len(vectorizer.vocabulary_), output_dim=128))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train model
model.fit(X.toarray(), y, validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stopping])

# Prediction
def predict_answer_model2(question_cat):
    question = clean_text(question_cat)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X)
    ranked_indices = np.argsort(similarities)[0][::-1]
    
    perfect_answer_found = False
    closest_answers = []
    
    for index in ranked_indices:
        if filtered_data['Label'][index] == 1:
            perfect_answer_found = True
            return filtered_data['Answer'][index]
        else:
            closest_answers.append(filtered_data['Answer'][index])
            
        if len(closest_answers) == 3:
            break
    
    if not perfect_answer_found:
        return closest_answers
    
    return "Sorry, I don't have an answer to that."
    