import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec

train_data = pd.read_csv('train.csv', index_col=0)
sentences = train_data['discourse_text'].tolist()


def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tt = TweetTokenizer()
    text = tt.tokenize(text)
    for i in range(len(text) - 1, -1, -1):
        contain_alphabets = False
        for j in range(len(text[i])):
            if 97 <= ord(text[i][j]) <= 122:
                contain_alphabets = True
                break
        if not contain_alphabets:
            text.pop(i)

    for word in text:
        if word in stop_words:
            text.remove(word)

    return text


for k in range(len(sentences)):
    sentences[k] = preprocess(sentences[k])

model = Word2Vec(sentences, vector_size=64)
model.save("word2vec.model")

