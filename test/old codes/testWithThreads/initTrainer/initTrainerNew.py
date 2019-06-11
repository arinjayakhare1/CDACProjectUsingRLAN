import os
import csv
import math
import numpy as np 
import nltk
from nltk.corpus import stopwords
import collections
import string
import re
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import time
import pickle
from nltk.corpus import words
import sys 
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from sklearn.metrics import precision_recall_fscore_support
import gc
from keras import backend
import tensorflow   

cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]


import time
from contextlib import contextmanager



# def preprocess(text):
#     # lowercase
#     text = text[3:]
#     text=text.lower()
#     text = ' '.join([word for word in text.split() if word not in cachedStopWords])
#     text = ' '.join([word for word in text.split() if( not word.startswith("@") and not word.startswith("http") and not word.startswith("\\")) ])
#     text = ' '.join([word for word in text.split() if word in allEnglishWords])
#     #text =  re.sub("[_]","",text)
#     #remove tags
#     text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
#     # remove special characters and digits
#     text=re.sub("(\\d|\\W)+"," ",text)
#     if(text.startswith("rt ") or text.startswith(" rt")):
#         text = text[3:]
#     if(text == "rt"):
#         text = ""
#     while(text != "" and text[0] == ' '):
#         text = text[1:]
#     return text

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("\n\n" + name + ' done in ' + str(round(time.time() - t0)) + 's \n')



def clean(s):
    transalator = str.maketrans("","",string.punctuation)
    return s.translate(transalator)

def preprocess(text):
    text = clean(text).lower()
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return text

with timer("Reading data"):
    x = []
    y = []
    radical = []
    radicalOne = 0
    with open("input.csv",'r', encoding="utf8") as csvFile:
        reader = csv.reader(csvFile)
        p = 0
        for row in reader:
            if(len(row) == 2):
                s = row[0]
                x.append(preprocess(s))
                if(row[1] != '0'):
                    radicalOne += 1 
                radical.append(0 if row[1] == '0' else 1)
            p = p + 1   
    csvFile.close           

    print("Size of x:",len(x)," Size of y:",len(radical)," Positive : ",radicalOne)
    X = []
    for t in x:
        t = re.sub(r'[^\w\s]',' ',t)
        t = ' '.join([word for word in t.split() if word != " "])
        t = t.lower()
        t = ' '.join([word for word in t.split() if word not in cachedStopWords])
        X.append(t)


with timer("making Tokeniser"):
    print("Type of X:",type(X)) 
    Features = X
    Radical = radical

    kf = KFold(n_splits=10)

    vocabSize = len(allEnglishWords)
    tokenizer = Tokenizer(num_words= vocabSize)
    tokenised = tokenizer.fit_on_texts(allEnglishWords)

    with open('../models/tokenizer/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



with timer("trying to clear GPU memory initially "):
    backend.clear_session()

    for i in range(20):
        gc.collect()

with timer("Making label vector"):
    Radical = np.zeros([len(Radical),3],dtype = int)
    for x in range(0, len(radical)):
        Radical[x,radical[x]] = 1

  


with timer("Training the model"):
    with timer("Making Embedding_index dict"):
        embeddings_index = dict()
        f = open('glove.twitter.27B/glove.twitter.27B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))

    with timer("Making Embedding Matrix"):
        embedding_matrix = np.zeros((vocabSize, 100))
        for word, index in tokenizer.word_index.items():
            if index > vocabSize - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector

    
    tokenisedTrain = tokenizer.texts_to_sequences(Features)

    max_review_length = 180
    X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post') 

    #Radical
    radicalModel = Sequential()
    radicalModel.add(Embedding(vocabSize, 100, input_length=max_review_length,weights=[embedding_matrix]))
    radicalModel.add(Dropout(0.2))
    radicalModel.add(Conv1D(64, 5, activation='relu'))
    radicalModel.add(MaxPooling1D(pool_size=4))
    radicalModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    radicalModel.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    radicalModel.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
    radicalModel.add(Dense(3, activation='sigmoid'))
    radicalModel.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
    epochs = 1
    fitHistory = radicalModel.fit(X_Train, Radical, epochs = 1, batch_size = 150)
    trainingAccuracy = fitHistory.history['acc']
    while(trainingAccuracy[0] < 0.99 or epochs < 15):
        epochs += 1
        print("\nTraining until accuracy improves for epoch = ", epochs, "for part 2.1 for iteration : ", iteration)
        fitHistory = radicalModel.fit(X_Train, Radical, epochs = 1, batch_size = 150)
        trainingAccuracy = fitHistory.history['acc']
        if(epochs == 50):
            break

    if(os.path.isfile("../models/radical/radicalModel.h5")):
            os.remove("../models/radical/radicalModel.h5")

    model.save("../models/radical/radicalModel.h5")              