# Learning to Rank Chatbot for University Customer Support
Welcome to our project, aimed at revolutionizing university customer support with the implementation of a Learning to Rank Chatbot. In today's digital age, universities face increasing demands for efficient and effective support services. Recognizing this challenge, our team has developed a cutting-edge chatbot system that leverages Learning to Rank algorithms to prioritize and deliver personalized responses to user queries.

By integrating machine learning techniques with natural language processing, our chatbot learns from user interactions, continuously improving its ability to provide relevant and helpful assistance. This project not only streamlines the support process for students, faculty, and staff but also enhances overall user satisfaction and productivity within the university community.

We invite you to explore our codebase.

## Table of Contents
- [Web Scraping](#web-scraping)
- [Model Traning](#model-training)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)

## Web Scraping
- Data was scraped from various FAQ sections of Stevens' website, such as Housing and Dining, OPT, New Student Orientation, and more.
- Functions were created to handle different HTML structures for accurate data extraction.
- The scraped data was saved into a CSV file named Dataset_Scraping.csv, organized with an additional "Category" column.

## Model Training

- Text data was tokenized and vectorized using TfidfVectorizer.
- Machine learning models including SVM and BLSTM were trained using the 'Learning to Rank - Pointwise' approach. We will combine to the random forest model in 4 machine learning models to classify where the query's category is belong and then we retrieve top reponse from that category.

## Installation
Install the following required libraries:
- BeautifulSoup, requests, and <a href="https://selenium-python.readthedocs.io/installation.html"> webdriver </a> (for Web Scraping).
- re, gensim, ntlk, spicy, and naw.
- Scikit-learn
- <a href = "https://www.tensorflow.org/install"> TensorFlow </a>

## Data
- To enhance the size of our dataset and improve the accuracy of our model, we employed data augmenta- tion techniques. Specifically, we utilized methods such as synonym replacements and random insertions to generate additional variations of the existing question-answer pairs.
- Overall, data augmentation proved to be a beneficial approach in enhancing the size and diversity of our dataset, which in turn improved the performance of our chatbot.
- We created a new dataset to predict if the question and answer are paired, where each question has three answers: one answer with a corresponding label of 1 and two randomly selected answers with corresponding label of 0.
