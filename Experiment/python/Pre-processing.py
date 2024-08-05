import pandas as pd
import nlpaug.augmenter.word as naw
import random
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder


## Method 1: using naw to increase the size
# Creating the new dataset including Duplicated Questions.
dataset = pd.read_csv("Dataset_Scraping.csv")

aug = naw.SynonymAug(aug_src='wordnet')

# Augment the sentence for questions and answers
aug_question = [aug.augment(dataset["Questions"][i])[0] for i in range(len(dataset["Questions"]))]
aug_answer = [aug.augment(dataset["Answers"][i])[0] for i in range(len(dataset["Answers"]))]

C_new = []
Q_new = []
Q_1_new = []
A_new = []
A_1_new = []
output = []
number_sample = 1000
number_que =  number_sample/len(dataset["Answers"])

for i in range(len(dataset["Questions"])):
    index = 0
    for j in range(int(number_que)):
        # To take the pair of question and answer
        if index == 0:
            C_new.append(dataset["Category"][i])
            Q_new.append(dataset["Questions"][i])
            Q_1_new.append(aug_question[i])
            A_new.append(dataset["Answers"][i])
            A_1_new.append(aug_answer[i])
            output.append(1)
        # to take 2 answers randomly for each question
        else:
            my_list = list(range(len(dataset["Answers"])))
            random_number = random.choice(my_list)
            while dataset["Category"][random_number] == dataset["Category"][i]:
                random_number = random.choice(my_list)
            C_new.append(dataset["Category"][i])
            Q_new.append(dataset["Questions"][i])
            Q_1_new.append(aug_question[i])
            A_new.append(dataset["Answers"][random_number])
            A_1_new.append(aug_answer[random_number])
            output.append(0)
        index +=1

Category = C_new*3
Questions = Q_new + Q_1_new + Q_new
Answers = A_new + A_1_new + A_1_new
Label = output*3
new_data = pd.DataFrame({"Category": Category,
                         "Questions":Questions,
                         "Answers": Answers,
                         "Label": Label})

labelencoder = LabelEncoder()
new_data['Category'] = labelencoder.fit_transform(new_data['Category'])
train_data, test_data = train_test_split(new_data , test_size=0.2,\
                    random_state=42)

## Method 2: using functions to augment the dataset
aug_data = pd.DataFrame(columns=data.columns)

def delete_word(sentence):
    words = nltk.word_tokenize(sentence)  
    index = random.randint(0, len(words)-1)
    words.pop(index)
    return ' '.join(words)


# delete a random word in the question
new_question = delete_word(row["Questions"])
aug_data = aug_data.append({"Category": row["Category"], \
                            "Questions": new_question, "Answers": row["Answers"]}, ignore_index=True)


aug_data.to_csv("Delete_word.csv", index=False)

# create an empty dataframe to store the augmented data
aug_data = pd.DataFrame(columns=data.columns)

def insert_word(sentence):
    words = nltk.word_tokenize(sentence)
    index = random.randint(0, len(words)-1)
    word = replace_word(words[index])
    words.insert(index, word)
    return ' '.join(words)

# loop through each row in the dataset and apply the data augmentation techniques
for index, row in data.iterrows():
    # insert a random word in the question
    new_question = insert_word(row["Questions"])
    aug_data = aug_data.append({"Category": row["Category"], "Questions": new_question, "Answers": row["Answers"]}, ignore_index=True)
    
# save the augmented data to a CSV file
aug_data.to_csv("Insert_word.csv", index=False)


def replace_word(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    if synonyms:
        return random.choice(synonyms)
    else:
        return word

    # replace words in the question
    new_question = replace_word(row["Questions"])
    aug_data = aug_data.append({"Category": row["Category"], "Questions": new_question, "Answers": row["Answers"]}, ignore_index=True)
  
    
aug_data.to_csv("Replace_word.csv", index=False)

data1=pd.read_csv("Replace_letter.csv")
data2=pd.read_csv("Insert_letter.csv")
data3=pd.read_csv("Delete_word.csv")

C_new = []
Q_new = []
Q_1_new = []
Q_2_new = []
Q_3_new = []
A_new = []
A_1_new = []
A_2_new = []
A_3_new = []
output = []
number_sample = 1000
number_que =  number_sample/len(dataset["Answers"])

for i in range(len(dataset["Questions"])):
    index = 0
    for j in range(int(number_que)):
        # To take the pair of question and answer
        if index == 0:
            C_new.append(dataset["Category"][i])
            Q_new.append(dataset["Questions"][i])
            Q_1_new.append(data1["Questions"][i])
            Q_2_new.append(data2["Questions"][i])
            Q_3_new.append(data3["Questions"][i])
            A_new.append(dataset["Answers"][i])
            A_1_new.append(data1["Answers"][i])
            A_2_new.append(data2["Answers"][i])
            A_3_new.append(data3["Answers"][i])
            output.append(1)
        # to take 2 answers randomly for each question
        else:
            my_list = list(range(len(dataset["Answers"])))
            random_number = random.choice(my_list)
            while dataset["Category"][random_number] == dataset["Category"][i]:
                random_number = random.choice(my_list)
            C_new.append(dataset["Category"][i])
            Q_new.append(dataset["Questions"][i])
            Q_1_new.append(data1["Questions"][i])
            Q_2_new.append(data2["Questions"][i])
            Q_3_new.append(data3["Questions"][i])
            A_new.append(dataset["Answers"][random_number])
            A_1_new.append(data1["Answers"][random_number])
            A_2_new.append(data2["Answers"][random_number])
            A_3_new.append(data3["Answers"][random_number])
            output.append(0)
        index +=1

Category = C_new*4
Questions = Q_new + Q_1_new + Q_2_new + Q_3_new
Answers = A_new + A_2_new + A_3_new + A_1_new 
Label = output*4

new_data = pd.DataFrame({"Category": Category,
                         "Questions":Questions,
                         "Answers": Answers,
                       "Label": Label})
new_data = new_data.dropna()
new_data   