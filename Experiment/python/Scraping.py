import requests
from bs4 import BeautifulSoup  
from selenium import webdriver
import pandas as pd
import time
import re
import string
import numpy as np 
import random
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Define a function called getFAQ that takes in two arguments, page_url and driver
def getFAQ(page_url, driver):
    # Initialize the result variable to None
    result = None
    # Use the driver to get the webpage at the specified URL
    driver.get(page_url)
    # Store the HTML code of the webpage in the page variable
    page = driver.page_source
    # Use BeautifulSoup to parse the HTML code
    soup = BeautifulSoup(page, 'html.parser')
    # Initialize empty lists 
    Questions = []
    Answers = []
    # Use soup.select to extract all elements with class name from website
    data_each = soup.select("div.accordion__item")
    # Loop through each element in data_each
    for each in data_each:
        question = each.select("div.accordion__button")
        Questions.append(question[0].get_text())
        answer = each.select("div.rich-text-base_c--rich-text-base__TWX_T")
        Answers.append(answer[0].get_text())
    # Combine Questions and Answers into a list called list_data
    list_data = [Questions, Answers]
    # Create a list of column names for the result DataFrame
    columns=['Questions','Answers']
    # Create a dictionary with keys from columns and values from list_data
    data_1 = dict(zip(columns,list_data))
    # Convert it into a pandas DataFrame called result
    result = pd.DataFrame(data_1)
    # Return the result DataFrame
    return result

def getFAQ1(page_url, driver):
    result = None
# Load the web page
    driver.get(page_url)
# Get the page source and parse it with BeautifulSoup
    page = driver.page_source
    soup = BeautifulSoup(page, 'html.parser')
# Initialize empty lists 
    Questions = []
    Answers = []
    data_each = soup.select("div.rich-text-base_c--rich-text-base__TWX_T")
    tag_list = []
    ques_ans = []
# Loop through each div and its child tags
    for data in data_each:
        tag_name = data.children
        for tag in tag_name:
            # Append the text and tag name to separate lists
            ques_ans.append(tag.get_text())
            tag_list.append(tag.name)
            # If the tag is an h3 or h4, append the text to the Questions list
            if tag.name == "h4" or tag.name == "h3":
                Questions.append(tag.get_text())
# Remove the first tag if it doesn't contain a question
    index = 0        
    while index < len(ques_ans):
        if "?" not in ques_ans[0]:
            del ques_ans[0]
            del tag_list[0]
            index +=1
        else: 
            break
# Combine answers that are split across multiple tags
    i = 0
    while i < len(tag_list):
        if tag_list[i] == "p":
            new_str = ques_ans[i]
            i += 1
            while i < len(tag_list) and (tag_list[i] == "p" or tag_list[i] == "ul" or tag_list[i] == "ol"):
                new_str += ques_ans[i]
                i += 1
            Answers.append(new_str)
        else:
            i += 1
# Combine the Questions and Answers lists into a DataFrame
    list_data = [Questions, Answers]
    columns=['Questions','Answers']
    data_1 = dict(zip(columns,list_data))
    result = pd.DataFrame(data_1)
    return result

def getFAQ2(page_url, driver):
    result = None
    driver.get(page_url)  # load the web pag
    page = driver.page_source  # get the HTML source of the pag
    soup = BeautifulSoup(page, 'html.parser')  # create a Beautiful_soup object to parse the html
    Questions = []
    Answers = []
# select all the div elemnts with class
    data_each = soup.select("div.rich-text-base_c--rich-text-base__TWX_T")
    tag_list = []  # list to store the tag names
    ques_ans = []  # list to store the text of each tag
    for data in data_each:
        tag_name = data.children
        for tag in tag_name:
            ques_ans.append(tag.get_text())
            tag_list.append(tag.name)
            # the h2 tag is also a question
            if tag.name == "h4" or tag.name == "h3" or tag.name == "h2":
                Questions.append(tag.get_text())
# remove the first tag if it is not a question
    index = 0        
    while index < len(ques_ans):
        if "?" not in ques_ans[0]:
            del ques_ans[0]
            del tag_list[0]
            index +=1
        else: 
            break
    i = 0
    while i < len(tag_list):
        if tag_list[i] == "p":
            new_str = ques_ans[i]
            i += 1
            while i < len(tag_list) and (tag_list[i] == "p" or tag_list[i] == "ul" or tag_list[i] == "ol"):
                new_str += ques_ans[i]
                i += 1
            Answers.append(new_str)
        else:
            i += 1

    list_data = [Questions, Answers]  
    columns=['Questions','Answers']  
    data_1 = dict(zip(columns,list_data)) 
    result = pd.DataFrame(data_1)  
    
    return result 

def getFAQ4(page_url, driver):
    result = None
    driver.get(page_url)
    page = driver.page_source
    soup = BeautifulSoup(page, 'html.parser')
#inItialize
    Questions = []
    Answers = []
# Select all paragraphs since we cant get individual class and separate it later
    data_each = soup.select("div.rich-text-base_c--rich-text-base__TWX_T p")
    ques_ans = []
    for data in data_each:
        # Append each paragraph to ques_ans list
        ques_ans.append(data.get_text())
# Remove the first paragraph if it's not a question
    i = 0        
    while i < len(ques_ans):
        if "?" not in ques_ans[0]:
            del ques_ans[0]
            i+=1
        else: 
            ques_ans = ques_ans
            break
    new_variable = []    
    for each in ques_ans:
# If a paragraph ends with a question mark, add it to the Questions list and mark it with "q"
        if "?" in each:
            Questions.append(each)
            new_variable.append("q")
        # Otherwise mark it with "a"
        else:
            new_variable.append("a")
# Combine adjacent paragraphs marked with "a" into a single string answer
    j = 0
    while j < len(ques_ans):
        if new_variable[j] == "a":
            new_str = ques_ans[j]
            j += 1
            while j < len(ques_ans) and new_variable[j] == "a":
                new_str += ques_ans[j]
                j += 1
            Answers.append(new_str)
        else:
            j += 1
# Create a dictionary with Questions and Answers lists, then convert it to a pandas dataframe
    list_data = [Questions, Answers]
    columns=['Questions','Answers']
    data_1 = dict(zip(columns,list_data))
    result = pd.DataFrame(data_1)
    return result


# Stevens URL links.
page_urls = [
    "https://www.stevens.edu/housing-dining-frequently-asked-questions",
    "https://www.stevens.edu/opt-faqs",
    "https://www.stevens.edu/admission-aid/tuition-financial-aid/frequently-asked-questions/frequently-asked-questions-financial-aid",
    "https://www.stevens.edu/frequently-asked-questions-for-newly-admitted-students",
    "https://www.stevens.edu/cpt-frequently-asked-questions",
    "https://www.stevens.edu/change-of-program",
    "https://www.stevens.edu/applying-for-a-social-security-number",
    "https://www.stevens.edu/admission-aid/undergraduate-admissions/accepted-students/orientation",
    "https://www.stevens.edu/stevens-online/frequently-asked-questions",
    "https://www.stevens.edu/admission-aid/undergraduate-admissions/accepted-students/pre-orientation",
    "https://www.stevens.edu/counseling-pshychological-services/about-student-counseling-and-psychological-services",
    "https://www.stevens.edu/development-alumni-engagement/give-to-stevens/giving-faq",
    "https://www.stevens.edu/interlibrary-loan-and-document-delivery-services/interlibrary-loan-and-document-delivery-services-faqs",
    "https://www.stevens.edu/services/stevens-wi-fi",
    "https://www.stevens.edu/page-basic/innovation-expo-2023-tools",
    "https://www.stevens.edu/admission-aid/undergraduate-admissions/transfer-students",
    "https://www.stevens.edu/counseling-pshychological-services/seeking-help-off-campus",
    "https://www.stevens.edu/office-of-student-accounts/1098-t-tuition-statement-information",
    "https://www.stevens.edu/school-systems-enterprises-graduate-program-faqs",
    "https://www.stevens.edu/school-systems-enterprises-undergraduate-program-faqs",
    "https://www.stevens.edu/istem/faq",
    "https://www.stevens.edu/governance-and-policy/european-union-general-data-protection-regulation-gdpr-faq",
    "https://www.stevens.edu/hr/workday-faqs",
    "https://www.stevens.edu/disability-services/faq"
]

# Corresponding list of getFAQ functions for each URL
getFAQ_functions = [
    getFAQ, getFAQ, getFAQ, getFAQ, getFAQ, getFAQ, 
    getFAQ, getFAQ, getFAQ, getFAQ, getFAQ, getFAQ, 
    getFAQ, getFAQ, getFAQ, getFAQ, getFAQ, getFAQ, 
    getFAQ1, getFAQ1, getFAQ1, getFAQ2, getFAQ3, getFAQ4
]

# List of categories
list_cates = [
    "Housing and Dining", "OPT", "Tuition Financial Aid", "Newly Admitted Students", 
    "CPT", "Change of Program", "Applying for a Social Security Number", 
    "New Student Orientation", "Online Programs", "New Student Pre-Orientation", 
    "Counseling and Psychological Services CAPS", "Giving", 
    "Interlibrary Loan and Document Delivery Services", "Stevens Wi-Fi", 
    "Innovation Expo", "Transfer Applicant", "Seeking Help Off-Campus", 
    "1098-T Tuition Statement", "Enterprises Graduate Program", 
    "Enterprises Undergraduate Program", "iSTEM@Stevens", "GDPR", 
    "Workday", "Disability Services"
]

# Using Chromedriver
executable_path = '/usr/bin/chromedriver'
driver = webdriver.Chrome(executable_path=executable_path)

# Initialize an empty list to store results
results = []

# Loop through URLs and getFAQ functions to fetch results and insert categories
for i, (url, getFAQ_func, category) in enumerate(zip(page_urls, getFAQ_functions, list_cates)):
    result = getFAQ_func(url, driver)
    result.insert(0, "Category")
    result.insert(1, category)
    results.append(result)

# Quit the driver
driver.quit()

# Combine all results into a single dataset
dataset = pd.concat(results).reset_index(drop=True)

#Save in CSV format
dataset.to_csv("Dataset_Scraping.csv", index=False)

