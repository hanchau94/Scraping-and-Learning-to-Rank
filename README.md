# Learning to Rank Chatbot for University Customer Support
Welcome to our project, aimed at revolutionizing university customer support with the implementation of a Learning to Rank Chatbot. In today's digital age, universities face increasing demands for efficient and effective support services. Recognizing this challenge, our team has developed a cutting-edge chatbot system that leverages Learning to Rank algorithms to prioritize and deliver personalized responses to user queries.

By integrating machine learning techniques with natural language processing, our chatbot learns from user interactions, continuously improving its ability to provide relevant and helpful assistance. This project not only streamlines the support process for students, faculty, and staff but also enhances overall user satisfaction and productivity within the university community.

We invite you to explore our codebase.

## Table of Contents
- [Web Scraping](#web-scraping)
- [Model Traning](#model-training)
- [Installation](#installation)
- [Data](#data)
- [Evaluation](#evaluation)

## Web Scraping
- Data was scraped from various FAQ sections of Stevens' website, such as Housing and Dining, OPT, New Student Orientation, and more.
- <a herf="https://github.com/hanchau94/Scraping-and-Learning-to-Rank/tree/main/Experiment/python">Functions</a> were created to handle different HTML structures for accurate data extraction.
- The scraped data was saved into a CSV file named Dataset_Scraping.csv, organized with an additional "Category" column.

## Model Training
- Text data was tokenized and vectorized using TfidfVectorizer.
  - <a herf="https://github.com/hanchau94/Scraping-and-Learning-to-Rank/tree/main/Experiment/python"> **tokenize()**</a> function to extract the unigram (more than 2 letters) or bigram token. The unigram still keep font after remove punctual such as “I-20”, “can’t”, or “Steven’s”
  - The cosine similarity between the questions and answers from <a herf="https://github.com/hanchau94/Scraping-and-Learning-to-Rank/tree/main/Experiment/python">**assess_similarity()**</a> function.
  - We find the tf-idf matrix by TfidfVectorizer() from scikit-learn.
- Machine learning models including <a herf="https://github.com/hanchau94/Scraping-and-Learning-to-Rank/tree/main/Experiment/python"> SVM</a> and <a herf=https://github.com/hanchau94/Scraping-and-Learning-to-Rank/tree/main/Experiment/python>BiLSTM</a> were trained using the <a herf = "https://en.wikipedia.org/wiki/Learning_to_rank">'Learning to Rank - Pointwise'</a> approach. We combine these models with a Random Forest model to classify the query's category and retrieve the top response from that category.

## Installation
Install the following required libraries:
- BeautifulSoup, requests, and <a href="https://selenium-python.readthedocs.io/installation.html"> webdriver </a> (for web scraping).
- ```re```, ```gensim```, ```ntlk```, ```spicy```, and ```naw```.
- Scikit-learn
- <a href = "https://www.tensorflow.org/install"> TensorFlow </a>

## Data
- To enhance the size of our dataset and improve the accuracy of our model, we employed <a herf = "https://github.com/hanchau94/Scraping-and-Learning-to-Rank/tree/main/Experiment/python"> data augmentation techniques </a>. Specifically, including synonym replacements and random insertions to generate additional variations of existing question-answer pairs.
- A new dataset was created to predict if a question and answer are paired, with each question having three answers: one correct (label 1) and two randomly selected (label 0).

## Evaluation
- We evaluate the models using AUC and PRC scores.
- The performance of retrieved documents is assessed based on top-1, top-3, and top-5 answers for cases with and without stop-word removal. Removing stop-words consistently provides better performance.
