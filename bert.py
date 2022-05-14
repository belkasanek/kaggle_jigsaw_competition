import os
import re
import gc

import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from keras.callbacks import Callback, EarlyStopping
from keras import backend as K

from transformers import AutoTokenizer, TFBertModel

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold

DATA_PATH = './data'
SOLUTION_PATH = './solutions'
WEIGHTS_PATH = './model_weights'

train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\n', ' ', text)
    
    text = re.sub(r"thats", " that is ", text)
    text = re.sub(r"youre", " you are ", text)
    text = re.sub(r"cant", " can not ", text)
    text = re.sub(r"didnt", " did not ", text)
    text = re.sub(r"dont", " do not ", text)
    text = re.sub(r"doesnt", " does not ", text)
    text = re.sub(r"ive", " i have ", text)
    text = re.sub(r"youve", " you have ", text)
    text = re.sub(r"wont", " will not ", text)
    text = re.sub(r"hes", " he is ", text)
    text = re.sub(r"isnt", " is not ", text)
    text = re.sub(r"havent", " have not ", text)
    text = re.sub(r"arent", " are not ", text)
    text = re.sub(r"whats", " what is ", text)
    text = re.sub(r"wasnt", " was not ", text)
    text = re.sub(r"theres", " there is ", text)
    text = re.sub(r"youll", " you will ", text)
    text = re.sub(r"wouldnt", " would not ", text)
    text = re.sub(r"shouldnt", " should not ", text)
    text = re.sub(r"theyre", " they are ", text)

    text = re.sub(r"buttsecks", " butt sex ", text)
    text = re.sub(r"fggt", " faggot ", text)
    text = re.sub(r"niggas", " nigger ", text)
    text = re.sub(r"nigga", " nigger ", text)
    text = re.sub(r"cunts", " cunt ", text)
    text = re.sub(r"fck", " fuck ", text)

    text = re.sub(r"bitchfuck", " bitch fuck ", text)
    text = re.sub(r"youfuck", " you fuck ", text)
    text = re.sub(r"sexsex", " sex sex ", text)
    text = re.sub(r"fack", " fuck ", text)
    text = re.sub(r"dicks", " dick ", text)
    
    for word in ['retard', 'nigger', 'cunt', 'twat', 'moron', 'wanker', 'faggot', 'cocksucker']:
        text = re.sub(word, " bastard ", text) 

    text = text.strip(' ')
    return text


train['comment_text'] = train['comment_text'].map(lambda x : clean_text(x))
test['comment_text'] = test['comment_text'].map(lambda x : clean_text(x))

# create hash column to remove duplicates
train['comment_text'] = train['comment_text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)
train['hash'] = train['comment_text'].str[:200].apply(hash)

test['comment_text'] = test['comment_text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)
test['hash'] = test['comment_text'].str[:200].apply(hash)

train = train.drop_duplicates(subset=['hash'])
test = test.drop_duplicates(subset=['hash'])

train = train[train.columns[:-1]]
test = test[test.columns[:-1]]

max_length = 120
model_name = 'bert-base-uncased'

bert_model = TFBertModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
auto_tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_emb_model():
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    output = bert_model(inputs)
    
    y = concatenate([GlobalAveragePooling1D()(output['hidden_states'][i]) for i in range(-1, -4, -1)])

    model = Model(inputs=inputs, outputs=y)

    return model

model = get_emb_model()

# save embedding of train and test comments
x = tokenizer(
    text=list(train['comment_text'].values),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding='max_length',
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

output = model.predict(x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
                       batch_size=32)

np.save(os.path.join(DATA_PATH, 'train_data_embeddings_bert_125'), output)
np.save('train_label', train[list_classes].values)

x = tokenizer(
    text=list(test['comment_text'].values),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding='max_length',
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

output = model.predict(x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
                       batch_size=32)

np.save(os.path.join(DATA_PATH, 'test_data_embeddings_bert_125'), output)

train = np.load(os.path.join(DATA_PATH, 'train_data_embeddings_bert_125.npy'), allow_pickle=True)
y = np.load(os.path.join(DATA_PATH, 'train_label.npy'), allow_pickle=True)
test = np.load(os.path.join(DATA_PATH, 'test_data_embeddings_bert_125.npy'), allow_pickle=True)

def get_final_model():
    input_vec = Input(shape=(768*3,), name='polled_embedding', dtype='float32')
         
    x = Dense(64, activation='relu', name='hidden')(input_vec)
    x = Dropout(0.2)(x)
    y = Dense(len(list_classes), activation='sigmoid', name='outputs')(x)

    model = Model(inputs=input_vec, outputs=y)
    
    optimizer = Adam()
    loss = BinaryCrossentropy()
    metrics = AUC()
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model

ffn = get_final_model()

# custom callback
class RocAucEvaluation(Callback):
    def __init__(self, patience, validation_data=()):
        super(Callback, self).__init__()
        
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0
        self.patience = patience

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, verbose=0)
        score = roc_auc_score(self.y_val, y_pred)
        
        print(f'epoch: {epoch}\nroc_auc: {round(score, 4)}')
        logs['roc_auc'] = score
        
        if score > self.max_score:
            self.max_score = score
            self.not_better_count = 0
            self.model.save(os.path.join(WEIGHTS_PATH, 'ffn'))
        else:
            self.not_better_count += 1
            if self.not_better_count > self.patience:
                self.model.stop_training = True
                print()
                print(f'Best roc_auc score: {round(self.max_score, 4)}')
                print('Early Sropping triggered.')

# Train
num_folds = 5
epochs = 100
batch_size = 64
kf = KFold(n_splits=num_folds, shuffle=True, random_state=30)

predict = np.zeros((test.shape[0], len(list_classes)))

for train_index, test_index in kf.split(train):
    y_train_fold, y_val_fold = y[train_index], y[test_index]
    X_train_fold, X_val_fold = train[train_index], train[test_index]
    
    K.clear_session()
    ffn = get_final_model()
    
    rocauc_early_stopping = RocAucEvaluation(patience=5,
                                             validation_data=(X_val_fold.astype('float32'), 
                                                              y_val_fold.astype('int8')))
    ffn.fit(x=X_train_fold.astype('float32'), 
            y=y_train_fold.astype('int8'),
            callbacks=[rocauc_early_stopping],
            batch_size=batch_size,
            epochs=epochs, 
            verbose=0)
    
    # load best model
    reconstructed_model = tf.keras.models.load_model(os.path.join(WEIGHTS_PATH, 'ffn'))
    
    predict += reconstructed_model.predict(test.astype('float32')) / num_folds

sample_submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
sample_submission[list_classes] = predict
sample_submission.to_csv(os.path.join(SOLUTION_PATH, 'bert_125_5fold.csv'), index=False)