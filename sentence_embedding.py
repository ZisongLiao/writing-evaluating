import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential, Model
from keras import Input, layers
from keras.layers import LSTM, Masking, Bidirectional, MaxPooling1D, Dense, Lambda
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import tensorflow as tf


train_data = pd.read_csv('train.csv', index_col=0)
sentences = train_data['discourse_text'].tolist()
split_sentences = []
train_labels = []
end_punc = ['.', '!', '?']
for sentence in sentences:
    sentence = sentence.strip()
    if sentence[-1] not in end_punc:
        sentence += '.'
    start = 0
    for end in range(len(sentence)):
        if sentence[end] in end_punc:
            split_sentences.append(sentence[start:end + 1])
            if end + 1 < len(sentence):
                train_labels.append(0)
            else:
                train_labels.append(1)
            start = end + 1


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


train_sentences = []
for k in range(len(split_sentences)):
    train_sentences.append(preprocess(split_sentences[k]))

for m in range(len(train_sentences) - 1, -1, -1):
    if len(train_sentences[m]) == 0:
        train_sentences.pop(m)
        train_labels.pop(m)


w2v_model = Word2Vec.load('word2vec.model')
input1 = []
padding = []
for i in range(64):
    padding.append(-1)

for sentence in train_sentences:
    temp = []
    for w in range(20):
        if w < len(sentence):
            try:
                temp.append(w2v_model.wv[sentence[w]].tolist())
            except KeyError:
                temp.append(padding)
        else:
            temp.append(padding)
    input1.append(temp[:])
train_x = np.array(input1[:3000])
train_y = np.array(train_labels[:3000])
validation_x = np.array(input1[3000:3300])
validation_y = np.array(train_labels[3000:3300])

def merge_sentences(x, l1, l2, train_labels, records):
    padding = tf.fill([1, 1, 64], -1.)
    tensors_out = []
    n = 0
    flag = True
    if len(records) != 0:
        flag = False
    for dis in range(l1):
        tensors_in = []
        while n < l2:
            tensors_in.append(x[0:1, :])
            if train_labels[n] == 1 or n == l2 - 1:
                if flag:
                    records.append(len(tensors_in))
                while len(tensors_in) < 50:
                    tensors_in.append(padding)
                tensors_out.append(tf.concat(tensors_in, axis=1))
                n += 1
                break
            n += 1
    return tf.concat(tensors_out, axis=0)


def split_sentences(x, records):
    tensors = []
    for dis in range(len(records)):
        tensors.append(x[dis, 0:records[dis], :])
    return tf.concat(tensors, axis=0)


padding_records = []
model = Sequential()
model.add(Masking(mask_value=-1., input_shape=(20, 64)))
model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(20, 64)))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(MaxPooling1D(pool_size=20))
model.add(Lambda(merge_sentences, output_shape=(50, 64), arguments={'l1': 1500, 'l2': 3000,
                                                                    'train_labels': train_labels, 'records': padding_records}))
model.add(Masking(mask_value=-1, input_shape=(50, 64)))
model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(50, 64)))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dense(2, activation='softmax'))
model.add(Lambda(split_sentences, output_shape=(64,), arguments={'records': padding_records}))
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()

model.fit(train_x, train_y, batch_size=3000, epochs=5, validation_data=(validation_x, validation_y))

