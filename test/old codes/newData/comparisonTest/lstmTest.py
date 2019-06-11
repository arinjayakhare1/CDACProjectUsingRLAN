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

cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]


import time
from contextlib import contextmanager

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
    iteration = 0
    gRadicalAccu = 0
    gPrecision = [0,0]
    gRecall = [0,0]
    gFScore = [0,0]

    vocabSize = len(allEnglishWords)
    tokenizer = Tokenizer(num_words= vocabSize)
    tokenised = tokenizer.fit_on_texts(allEnglishWords)


    gPositivePredRadical = 0


with timer("Cross Validation"):

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
        
    for train_index, test_index in kf.split(Features):
        with timer("Making nueral network for iteration : " + str(iteration + 1)):
            iteration += 1
            print("\n\n\n\nMaking nueral Network for iteration:",iteration)

            #Making Training and Testing Data
            X_Train = [Features[x] for x in train_index]
            X_Test = [Features[x] for x in test_index]
            radicalTrain = [Radical[x] for x in train_index]
            radicalTest = [Radical[x] for x in test_index]

            tokenisedTrain = tokenizer.texts_to_sequences(X_Train)
            tokenisedTest = tokenizer.texts_to_sequences(X_Test)

            max_review_length = 180
            X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')
            X_Test = sequence.pad_sequences(tokenisedTest, maxlen=max_review_length,padding='post')

            #Radical
            radicalModel = Sequential()
            radicalModel.add(Embedding(vocabSize, 100, input_length=max_review_length,weights=[embedding_matrix], trainable=False))
            radicalModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            radicalModel.add(Dense(1, activation='sigmoid'))
            radicalModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            radicalModel.fit(X_Train,radicalTrain,epochs=10, batch_size=100)
            radicalScore = radicalModel.evaluate(X_Test,radicalTest,verbose = 100)
            accuRadicalLstm = radicalScore[1]
            print("\nRadical Training Done for Iteration",iteration)
            positiveRadical = [x for x in radicalTest if x == 1]
            predictRadical = radicalModel.predict_classes(X_Test, verbose = 1)
            positivePredRadical = [x for x in predictRadical if x > 0]
            prec, recall, fscore, support = precision_recall_fscore_support(radicalTest, predictRadical)
            print("Number of positive Examples : ",len(positiveRadical),  "\nratio : ", (len(positiveRadical) / len(radicalTest)), "\nPositive Predicted : ", len(positivePredRadical), "\naccuracy : ", accuRadicalLstm, "\nwrongness : ", 1 - accuRadicalLstm,"\n\nPrecision : ",prec,"\nRecall : ", recall, "\nf1Score : ", fscore, "\nsupport : ", support )


            gRadicalAccu += accuRadicalLstm
            gPositivePredRadical += len(positivePredRadical)
            gPrecision[0] += prec[0]
            gPrecision[1] += prec[1]
            gRecall[0] += recall[0]
            gRecall[1] += recall[1]
            gFScore[0] += fscore[0]
            gFScore[1] += fscore[1]


with timer("final Output"):

    gRadicalAccu /= 10
    gPrecision = [x / 10 for x in gPrecision]
    gRecall = [x / 10 for x in gRecall]
    gFScore = [x / 10 for x in gFScore]


    print("\n\n\n\nOverall AccuracyScores for LSTM :","\nRadical: ",gRadicalAccu)
    print("Precision : ", gPrecision)
    print("Recal : ", gRecall)
    print("FScore : ", gFScore)
    print("Positive Predicitions in total : ")
    print("Radical : ", gPositivePredRadical)
    print("Positive Predicitions in average : ")
    print("Radical : ", gPositivePredRadical / 10)                  
