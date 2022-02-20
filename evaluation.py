from transformers import TFBertPreTrainedModel, TFBertMainLayer
from tokenizers import BertWordPieceTokenizer
from keras.layers import LSTM, Masking, Bidirectional, MaxPooling1D, Dense, Lambda, Dropout
import keras
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import tensorflow as tf

tokenizer = BertWordPieceTokenizer("f:/evaluating/input/train.txt", lowercase=True, add_special_tokens=False)

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


train_x = np.array(train_sentences[:3000])
train_y = np.array(train_labels[:3000])
validation_x = np.array(train_sentences[3000:3300])
validation_y = np.array(train_labels[3000:3300])


def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,
                                 max_length=max_length,
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
    if (limit > 0):
        ds = ds.take(limit)

    for (i, row) in enumerate(ds.values):
        review = row[1]
        label = list(row[2:])
        bert_input = convert_example_to_feature(review)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(label)
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

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


from transformers.modeling_tf_utils import (
    TFQuestionAnsweringLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    shape_list,
)


class TFBertForMultilabelClassification(TFBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultilabelClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name='bert')
        self.pooling = tf.keras.layers.MaxPooling1D(pool_size=config.max_length)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier',
                                                activation='softmax')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        pooled_output = self.popoling(outputs[0])
        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)
        return logits


model = TFBertForMultilabelClassification.from_pretrained('bert-based-uncased', num_labels=6)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1)
loss = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.CategoricalAccuracy()
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)
