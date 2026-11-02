#!/usr/bin/env python
# coding: utf-8

# Practical NO.01  
# Name - Sakshi Vijaykumar Muthiyan  
# Roll No. -  
# Title - Write a program for pre-processing of a text document such as stop word removal, stemming. 

# In[1]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') 
print(stopwords.words('english'))


# In[3]:


import nltk
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = """This is a sample sentence,
showing off the stop words filtration."""
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)
# converts the words in word_tokens to lower case and then checks whether 
#they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words] 
#with no lower case conversion
filtered_sentence = [] 
for w in word_tokens:
    if w not in stop_words: 
        filtered_sentence.append(w)
print(word_tokens) 
print(filtered_sentence)


# In[5]:


with open("text.txt", "w") as f:
    f.write("This is a sample text file for information retrieval laboratory.")

import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
# word_tokenize accepts
# a string as an input, not a file.
stop_words = set(stopwords.words('english')) 
file1 = open("text.txt")
# Use this to read file content as a stream:
line = file1.read() 
words = line.split() 
for r in words:
    if not r in stop_words:
        appendFile = open('filteredtext.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close()


# In[6]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
ps = PorterStemmer()
# choose some words to be stemmed
words = ["program", "programs", "programmer", "programming", "programmers"]
for w in words:
    print(w, " : ", ps.stem(w))


# In[7]:


# importing modules
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
ps = PorterStemmer()
sentence = "Programmers program with programming languages" 
words = word_tokenize(sentence)
for w in words:
    print(w, " : ", ps.stem(w))

