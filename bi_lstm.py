import os
import re
import numpy as np
import pandas as pd

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Bidirectional, Embedding, concatenate, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, GRU
from keras.models import Model
from keras import optimizers
from keras.callbacks import Callback
from keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from symspellpy.symspellpy import SymSpell, Verbosity

DATA_PATH = './data'
SOLUTION_PATH = './solutions'
WEIGHTS_PATH = './model_weights'
EMBEDDING_FILE = 'glove.twitter.27B.200d.txt'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
DICT_ENG_FREQ = 'frequency_dictionary_en_82_765.txt'

embed_size = 200 
max_features = 300000 
maxlen = 300

num_folds = 5
feature_size = 3
num_filters = 100
dropout = 0.5
lr = 0.001
batch_size = 32
num_epoch = 100

train = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILE))
test = pd.read_csv(os.path.join(DATA_PATH, TEST_FILE))

def feature_eng(df):
    special_characters = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]', re.IGNORECASE)
    df['comment_text'] = df['comment_text'].str.replace(special_characters, '')
    # ratio of capital letters to all letters
    df['capitals_ratio'] = df['comment_text'].apply(lambda comment: sum(1 for word in comment if word.isupper())) \
    / df['comment_text'].apply(len)
    df['capitals_ratio'] = df['capitals_ratio'].fillna(0)
    # ratio of unique word to all words
    df['unique_ratio'] = df['comment_text'].apply(lambda comment: len(set(word for word in comment.split()))) \
    / df['comment_text'].str.count('\S+')
    df['unique_ratio'] = df['unique_ratio'].fillna(0)
    # ratio of lenght of comment to average one
    mean = 60
    df['lenght_ratio'] = df['comment_text'].str.count('\S+') / mean

    return df

train = feature_eng(train)
test = feature_eng(test)

train_features = train[['capitals_ratio', 'unique_ratio', 'lenght_ratio']]
test_features = test[['capitals_ratio', 'unique_ratio', 'lenght_ratio']]

scaler = StandardScaler()
scaler.fit(np.vstack((train_features, test_features)))
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

X_train = train['comment_text'].fillna('something').values
X_test = test['comment_text'].fillna('something').values

list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[list_classes].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train_sequence = tokenizer.texts_to_sequences(X_train)
X_test_sequence = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train_sequence, maxlen=maxlen)
X_test = pad_sequences(X_test_sequence, maxlen=maxlen)

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

# dict with embeddings
embeddings_index = dict(get_coefs(*line.strip().split()) for line in open(os.path.join(DATA_PATH, EMBEDDING_FILE)))

# spell corrector
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=20)
dictionary_path = os.path.join(DATA_PATH, DICT_ENG_FREQ)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1);

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size + 2))

# token for unknown words last dimension for caps word and mistake signal
something = np.zeros((embed_size + 2,))
something[:embed_size,] = embeddings_index.get('something')

def all_caps(word):
    'return 1 if all leters in word are capital else 0'
    return int(len(word) > 1 and word.isupper())

def embed_word(embedding_matrix, i, word, unkwown=False):
    'Impute embedding vector at row i in embedding matrix'
    if not unkwown:
        embedding_vector = embeddings_index.get(word)
    else:
        embedding_vector = something[:embed_size]
        embedding_matrix[i, embed_size + 1] = 1
    
    embedding_matrix[i, :embed_size] = embedding_vector
    embedding_matrix[i, embed_size] = all_caps(word)
            
for word, i in word_index.items():
    # skip if word too rare
    if i >= max_features: 
        continue
        
    if embeddings_index.get(word) is not None:
        embed_word(embedding_matrix, i, word)
    else:
        # skip too long unnknow words
        if len(word) > 20:
            embed_word(embedding_matrix, i, word, unkwown=True)
        # use spell correction
        else:
            corrected_word = sym_spell.lookup(word, verbosity=Verbosity.TOP, max_edit_distance=2)
            # if there is corrected word
            if corrected_word:
                word2 = corrected_word[0].term
                if embeddings_index.get(word2) is not None:
                    embed_word(embedding_matrix, i, word2)
                # for corrected_word there is no embedding vector
                else:
                    embed_word(embedding_matrix, i, word, unkwown=True)
            # if there is no fixed word with edit distance 2
            else: 
                embed_word(embedding_matrix, i, word, unkwown=True)
                
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()

        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, verbose=0)
        score = roc_auc_score(self.y_val, y_pred)
        if score > self.max_score:
            model.save_weights(os.path.join(WEIGHTS_PATH, 'bi_lstm_weights.h5'))
            self.max_score=score
            self.not_better_count = 0
        else:
            self.not_better_count += 1
            if self.not_better_count > 3:
                self.model.stop_training = True
                
                
def get_model(num_filters=num_filters, dropout=dropout, lr=lr):
    features_input = Input(shape=(feature_size,))
    inp = Input(shape=(maxlen,))
    
    x = Embedding(max_features, embed_size+2, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(dropout)(x)
    x, x_h, x_c = Bidirectional(GRU(num_filters, return_sequences=True, return_state=True))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    # A concatenation of the maximum pool, average pool, last state and additional features
    x = concatenate([max_pool, avg_pool, x_h, features_input])
    
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=[inp, features_input], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])
    return model

predict = np.zeros((test.shape[0], len(list_classes)))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=30)
for train_index, test_index in kf.split(X_train):
    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
    X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
    X_features_fold, X_val_features_fold = train_features[train_index], train_features[test_index]
    
    K.clear_session()
    model = get_model()
    
    ra_val = RocAucEvaluation(validation_data=([X_val_fold, X_val_features_fold], y_val_fold))
    
    model.fit([X_train_fold, X_features_fold], y_train_fold, 
              batch_size=batch_size, epochs=num_epoch, verbose=0, callbacks=[ra_val])
    
    # load best model
    model.load_weights(os.path.join(WEIGHTS_PATH, 'bi_lstm_weights.h5'))
    
    predict += model.predict([X_test, test_features], batch_size=batch_size, verbose=0) / num_folds 


sample_submission = pd.read_csv(os.path.join(SOLUTION_PATH, 'sample_submission.csv'))
sample_submission[list_classes] = predict
sample_submission.to_csv('bi_lstm_submission.csv', index=False)