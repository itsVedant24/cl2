# IR 1


# linux
# sudo apt install python3-venv
# python3 -m venv venv
# source venv/bin/activate
# pip install nltk

# window 
# python -m venv venv
# venv\Scripts\activate
# pip install nltk


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Input text
text = "The students are playing cricket and studying different subjects."

# Convert text into lowercase
text = text.lower()


words = word_tokenize(text)

# Stop word removal
stop_words = set(stopwords.words('english'))
filtered_words = []

for word in words:
    if word.isalpha() and word not in stop_words:
        filtered_words.append(word)

# Stemming
ps = PorterStemmer()
stemmed_words = []

for word in filtered_words:
    stemmed_words.append(ps.stem(word))

# Display results
print("Original Text:", text)
print("After Stop Word Removal:", filtered_words)
print("After Stemming:", stemmed_words)



# 1. Text Document Preprocessing
# Text preprocessing is the process of cleaning and preparing text for NLP and Information Retrieval. It includes tokenization,
#  stop word removal and stemming. Tokenization divides text into individual words, stop word removal eliminates common words 
# like is, the, are, and stemming converts words into their root forms, such as playing into play. Python's NLTK library is used 
# for this practical.

