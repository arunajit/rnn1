import tensorflow as tf
import theano
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine import Input, Model, InputSpec
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.utils import class_weight
import os
import pydot
import graphviz

EPCOHS = 2 #Increase epochs (2 set for demo)
BATCH_SIZE = 500 
INPUT_DIM = 4 
OUTPUT_DIM = 50 
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.2
MAXLEN = 150 
checkpoint_dir ='dataset'
os.path.exists(checkpoint_dir)

input_file = 'dataset/data.csv'

def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

def load_data(test_split = 0.1, maxlen = MAXLEN):    
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    ts = int(len(df) * (1 - test_split))
    Xtr = df['sequence'].values[:ts]
    ytr = np.array(df['target'].values[:ts])
    Xts = np.array(df['sequence'].values[ts:])
    yts = np.array(df['target'].values[ts:])
    print('Avg. train sequence length: {}'.format(np.mean(list(map(len, Xtr)), dtype=int)))
    print('Avg. test sequence length: {}'.format(np.mean(list(map(len, Xts)), dtype=int)))
    return pad_sequences(Xtr, maxlen=maxlen), ytr, pad_sequences(Xts, maxlen=maxlen), yts

def lstm(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    Xtr, ytr, Xts, yts = load_data()    
    model = lstm(len(Xtr[0])) 

    fpath= checkpoint_dir + "/weights.hdf5"
    checkpoint = ModelCheckpoint(fpath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print ('Fitting model...')
    class_weight = class_weight.compute_class_weight('balanced', np.unique(ytr), ytr)
    print(class_weight)
    history = model.fit(Xtr, ytr, batch_size=BATCH_SIZE, class_weight=class_weight,epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)   
    score, acc = model.evaluate(Xts, yts, batch_size=BATCH_SIZE)
    print('Validation score:', score)
    print('Validation accuracy:', acc)
