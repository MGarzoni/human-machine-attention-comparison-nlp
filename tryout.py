import pandas as pd 
import numpy as np
import re
from collections import Counter


# this function parses the input review in html format
# and outputs a binarized attention map relative to that review
# each value in the list refers to a word
# the value is 1 if it refers to an important word, 0 otherwise
def generate_binary_human_attention_vector(html, num_words_in_review):

    p = re.compile('<span(.*?)/span>')
    all_span_items = p.findall(html)
    
    if html == '{}':
        print('Empty human annotation - This should never print')
        return [0] * num_words_in_review
    
    if (len(all_span_items) == num_words_in_review + 1):
        if ((all_span_items[num_words_in_review] == '><') or (all_span_items[num_words_in_review] == ' data-vivaldi-spatnav-clickable="1"><')):
            
            binarized_human_attention = [0] * num_words_in_review
            for i in range(0, len(all_span_items) - 1):
                if 'class="active"' in all_span_items[i]:
                    binarized_human_attention[i] = 1

        else:
            print('This should never print.')
    else:
        print('This should never print.')

        
    return binarized_human_attention

data = pd.read_csv('data_formatted/final_data.csv')
# print(data.head())
# print(data.describe())

# labels_balance = Counter(data['Input.label'])
# answers_balance = Counter(data['Answer.Q1Answer'])
# print(labels_balance, answers_balance)

# inputs = data['Input.text']
# labels = data['Input.label']
# print(inputs[10], labels[10])


html = data['Answer.html_output'][16]
num_words_in_review = len(data['Input.text'][16].split())
binarized_human_attention = generate_binary_human_attention_vector(html, num_words_in_review)

def num_words_highlited_per_review(reviews):
    
    output = {}
    
    for i, review in enumerate(reviews):
        
        p = re.compile('<span(.*?)/span>')
        text_review = " ".join([x[1:-1] for x in p.findall(review)])
        
        output[i] = [text_review, review.count('class="active"')]
    
    return output

num_words_highlited_per_review = num_words_highlited_per_review(data['Answer.html_output'])
num_words_highlited_per_review[16]

num_highlighted = html.count('class="active"')
print("\nNumber of words highlighted in this review:", num_highlighted)
print("\nOriginal annotation:", html)
print("\nBinarized attention map:", binarized_human_attention)

text = inputs = data['Input.text'][5]
words_index = {}
words_count = Counter()
for i, word in enumerate(text.split(' ')):
    words_index[word] = i
    words_count[word] += 1

print(words_index)
print(words_count.most_common())

vocab_size = len(words_count) 
print(vocab_size)
layer_0 = np.zeros((1, vocab_size))
print(layer_0)

for index, key  in words_index.items():
    print(index, key)

