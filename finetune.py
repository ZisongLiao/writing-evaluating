from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.layers import LSTM, Masking, Bidirectional, MaxPooling1D, Dense, Lambda, Dropout
import keras
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import tensorflow as tf
from sklearn.model_selection import train_test_split

df_raw = pd.read_csv('train.csv', usecols=[4, 5])
df_label = pd.DataFrame({'discourse_type': ['Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence',
                                            'Concluding Statement'], 'y': list(range(7))})
df_raw = pd.merge(df_raw, df_label, on='discourse_type', how='left')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,
                                 max_length=512,
                                 pad_to_max_length=True,
                                 return_attention_mask=True,
                                 truncation=True
                                 )


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


def encode_examples(ds, limit=-1):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if limit > 0:
        ds = ds.take(limit)

    for index, row in ds.iterrows():
        review = row["discourse_text"]
        label = row["y"]
        bert_input = convert_example_to_feature(review)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


def split_dataset(df):
    train_set, x = train_test_split(df,
                                    stratify=df['discourse_type'],
                                    test_size=0.1,
                                    random_state=42)
    val_set, test_set = train_test_split(x,
                                         stratify=x['discourse_type'],
                                         test_size=0.5,
                                         random_state=43)

    return train_set, val_set, test_set


train_data, val_data, test_data = split_dataset(df_raw)
ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(1)
ds_val_encoded = encode_examples(val_data).batch(1)
ds_test_encoded = encode_examples(test_data).batch(1)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)

learning_rate = 2e-5
number_of_epochs = 2

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)
model.evaluate(ds_test_encoded)

path = 'D:\Study\kaggle\writing-evaluating\writing-evaluating\evaluation_bert_model'
model.save_pretrained(path)
