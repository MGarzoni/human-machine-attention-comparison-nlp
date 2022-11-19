import pandas as pd
import string

# merge datasets into one
original_data = pd.concat(
    map(pd.read_csv, ['original_data/ham_part1(50words).csv', 'original_data/ham_part3.csv', 'original_data/ham_part4.csv', 'original_data/ham_part5.csv',
                      'original_data/ham_part6(100words).csv', 'original_data/ham_part7.csv', 'original_data/ham_part8(200words).csv']), ignore_index=True)

# make new version of dataset by discarding every second and third row
# by doing this we preserve only one HAM for each review (discarding the other two)
final_data = pd.DataFrame(original_data[::3]).reset_index().drop(['index'], axis=1)

# preprocess texts: lowercase, remove punctuation and newline characters
final_data['Input.text'] = pd.Series(
    [x.lower().translate(str.maketrans('', '', string.punctuation + '\n'))
     for x in final_data['Input.text']])

# save final data to new csv file
final_data.to_csv('formatted_data/final_data.csv')

# read in dataset and only keep the text of reviews
import pandas as pd
data = pd.read_csv('formatted_data/final_data.csv').drop(['Unnamed: 0', 'Answer.Q1Answer', 'Answer.html_output'], axis=1)
# labels = data['Input.label']

# split into train and val
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, 
                                                  train_size=0.8, test_size=0.2, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, 
                                                    train_size=0.875, test_size=0.125, random_state=0)

# save new splits to .csv
# labels.to_csv('formatted_data/labels.csv')
X_train.to_csv('formatted_data/final_train.csv')
X_val.to_csv('formatted_data/final_val.csv')
X_val.to_csv('formatted_data/final_test.csv')

# check how many the labels and answers distribution
# from collections import Counter
# labels_balance = Counter(final_data['Input.label'])
# answers_balance = Counter(final_data['Answer.Q1Answer'])
# print(labels_balance, answers_balance)

# below steps of removing stopwords and lemmatizing the input texts
# should not be done because this was not done when the data was labelled by humans
# the problem would be that, for each review, we would pass to the model a string
# that does not match the string that the human has seen

# remove stopwords
# from nltk.corpus import stopwords
# stop = stopwords.words('english') + ['\n']
# pre_processed_inputs = pre_processed_inputs.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# print(pre_processed_inputs[N])
# print()

# lemmatize inputs
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# lemmatized = []
# for x in pre_processed_inputs:
#     lemmas = []
#     for word in x.split():
#         # print(f"ACTUAL {word} ------ LEMMATIZED {lemmatizer.lemmatize(word)}")
#         lemmas.append(lemmatizer.lemmatize(word))
#     lemmatized.append(" ".join(lemmas))
# pre_processed_inputs = pd.Series(lemmatized)
# print(pre_processed_inputs[N])


