import time
from contextlib import contextmanager
import gc
gc.collect()

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("\n\n" + name + ' done in ' + str(round(time.time() - t0)) + 's \n')

print("\n\nStarting\n\n")

with timer("Importing and setting up libraries"):
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
    from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GRU
    from sklearn.metrics import precision_recall_fscore_support
    from keras import backend
    import tensorflow

    cachedStopWords = stopwords.words("english")
    allEnglishWords = words.words()
    allEnglishWords[:] = [x.lower() for x in allEnglishWords]

    config = tensorflow.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    sess = tensorflow.Session(config=config) 
    backend.set_session(sess)

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
    gEpochs = 0


    vocabSize = len(allEnglishWords)
    tokenizer = Tokenizer(num_words= vocabSize)
    tokenised = tokenizer.fit_on_texts(allEnglishWords)


    gPositivePredRadical = 0

with timer("trying to clear GPU memory initially "):
    backend.clear_session()

    for i in range(20):
        gc.collect()


with timer("Making label vector"):
    Y = np.zeros([len(Radical),3],dtype = int)
    for x in range(0, len(Radical)):
        Y[x,Radical[x]] = 1

with timer("Making Embedding_index dict"):
        embeddings_index = dict()
        f = open('glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")
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

with timer("Cross Validation"):
    for train_index, test_index in kf.split(Features):
        with timer("Making nueral network for iteration : " + str(iteration + 1)):
            iteration += 1
            print("\n\n\n\nMaking nueral Network for iteration:",iteration," for part 1")


            

            #Making Training and Testing Data
            X_Train = [Features[x] for x in train_index]
            X_Test = [Features[x] for x in test_index]
            radicalTrain = Y[train_index]
            radicalTest = Y[test_index]
            radicalTest1 = [Radical[x] for x in test_index]

            tokenisedTrain = tokenizer.texts_to_sequences(X_Train)
            tokenisedTest = tokenizer.texts_to_sequences(X_Test)


            max_review_length = 180
            X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')
            X_Test = sequence.pad_sequences(tokenisedTest, maxlen=max_review_length,padding='post')

            #Radical
            radicalModel = Sequential()
            radicalModel.add(Embedding(vocabSize, 100, input_length=max_review_length,weights=[embedding_matrix]))
            radicalModel.add(Dropout(0.2))
            radicalModel.add(Conv1D(64, 5, activation='relu'))
            radicalModel.add(MaxPooling1D(pool_size=4))
            radicalModel.add(GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            radicalModel.add(GRU(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            radicalModel.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
            radicalModel.add(Dense(3, activation='sigmoid'))
            radicalModel.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
            epochs = 1
            print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration)
            fitHistory = radicalModel.fit(X_Train, radicalTrain, epochs = 1, batch_size = 150)
            trainingAccuracy = fitHistory.history['acc']
            while(trainingAccuracy[0] < 0.99 or epochs < 15):
                epochs += 1
                print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration)
                fitHistory = radicalModel.fit(X_Train, radicalTrain, epochs = 1, batch_size = 150)
                trainingAccuracy = fitHistory.history['acc']
                if(epochs == 50):
                    break
            radicalScore = radicalModel.evaluate(X_Test,radicalTest,verbose = 100)
            accuRadicalLstm1 = radicalScore[1]
            print("\nRadical Training Done for Iteration ",iteration, " for part 1 with epochs : ", epochs)
            positiveRadical = [x for x in radicalTest if x[0] == 0]
            predictRadical = radicalModel.predict_classes(X_Test, verbose = 1)
            positivePredRadical = [x for x in predictRadical if x > 0]
            prec1, recall1, fscore1, support1 = precision_recall_fscore_support(radicalTest1, predictRadical)
            print("Number of positive Examples : ",len(positiveRadical),  "\nratio : ", (len(positiveRadical) / len(radicalTest)), "\nPositive Predicted : ", len(positivePredRadical), "\naccuracy : ", accuRadicalLstm1, "\nwrongness : ", 1 - accuRadicalLstm1,"\n\nPrecision : ",prec1,"\nRecall : ", recall1, "\nf1Score : ", fscore1, "\nsupport : ", support1 )


            gRadicalAccu += accuRadicalLstm1
            gPositivePredRadical += len(positivePredRadical)
            gPrecision[0] += prec1[0]
            gPrecision[1] += prec1[1]
            gRecall[0] += recall1[0]
            gRecall[1] += recall1[1]
            gFScore[0] += fscore1[0]
            gFScore[1] += fscore1[1]
            gEpochs += epochs


            with timer("trying to clear GPU memory"):
                del radicalModel

                backend.clear_session()

                for i in range(20):
                    gc.collect()


            

with timer("final Output"):

    gRadicalAccu /= 10
    gPrecision = [x / 10 for x in gPrecision]
    gRecall = [x / 10 for x in gRecall]
    gFScore = [x / 10 for x in gFScore]


    print("\n\n\n\nOverall AccuracyScores for LSTM :","\nRadical : ",gRadicalAccu)
    print("Precision : ", gPrecision)
    print("Recal : ", gRecall)
    print("FScore : ", gFScore)
    print("Positive Predicitions in total : ")
    print("Radical : ", gPositivePredRadical)
    print("Positive Predicitions in average : ")
    print("Radical : ", gPositivePredRadical / 20)   
    print("Total epochs : ", gEpochs)
    print("Average epochs : ", (gEpochs / 20))


    