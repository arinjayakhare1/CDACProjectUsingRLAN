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
        text = text.split(",")[-1]
        text = clean(text).lower()
        text=text.lower()
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])
        text = ' '.join([word for word in text.split() if( not word.startswith("@") and not word.startswith("http") and not word.startswith("\\")) ])
        text = ' '.join([word for word in text.split() if word in allEnglishWords])
        #text =  re.sub("[_]","",text)
        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)
        if(text.startswith("rt ") or text.startswith(" rt")):
            text = text[3:]
        if(text == "rt"):
            text = ""
        while(text != "" and text[0] == ' '):
            text = text[1:]
        return text

with timer("Reading data"):
    x = []
    y = []
    radical = []
    violentExtremism = []
    nonViolentExtremism = []
    radicalViolence = []
    nonRadicalViolence = []
    posViolentExtremism = 0
    posNonViolentExtremism = 0
    posRadicalViolence = 0
    posNonRadicalViolence = 0
    with open("input.csv",'r', encoding="utf8") as csvFile:
        reader = csv.reader(csvFile)
        p = 0
        for row in reader:

            #To ignore header
            if(p == 0):
                p = p + 1
                continue
            if(len(row) >= 2):
                if(row[1] == "" or row[2] == "" or row[3] == "" or row[4] == ""):
                    continue
                s = row[0].split(',', 1)[1]
                x.append(preprocess(s))

                if(row[1] == "0.0"):
                    violentExtremism.append(0)
                else:
                    posViolentExtremism += 1
                    violentExtremism.append(1)

                if(row[2] == "0.0"):
                    nonViolentExtremism.append(0)
                else:
                    posNonViolentExtremism += 1
                    nonViolentExtremism.append(1)

                if(row[3] == "0.0"):
                    radicalViolence.append(0)
                else:
                    posRadicalViolence += 1
                    radicalViolence.append(1)

                if(row[4] == "0.0"):
                    nonRadicalViolence.append(0)
                else:
                    posNonRadicalViolence += 1
                    nonRadicalViolence.append(1)

                
            p = p + 1   
    csvFile.close           

    print("Size of x:",len(x))
    print("Size of violentExtremism : ", len(violentExtremism), "\t positive : ", posViolentExtremism)
    print("Size of nonViolentExtremism : ", len(nonViolentExtremism), "\t positive : ", posNonViolentExtremism)
    print("Size of radicalViolence : ", len(radicalViolence), "\t positive : ", posRadicalViolence)
    print("Size of nonRadicalViolence : ", len(nonRadicalViolence), "\t positive : ", posNonRadicalViolence)

    # print(violentExtremism)
    # print(nonViolentExtremism)
    # print(radicalViolence)
    # print(nonRadicalViolence)
    X = []
    for t in x:
        t = re.sub(r'[^\w\s]',' ',t)
        t = ' '.join([word for word in t.split() if word != " "])
        t = t.lower()
        t = ' '.join([word for word in t.split() if word not in cachedStopWords])
        X.append(t)


with timer("Making tokeniser"):
    vocabSize = len(allEnglishWords)
    tokenizer = Tokenizer(num_words= vocabSize)
    tokenised = tokenizer.fit_on_texts(allEnglishWords)


    kf = KFold(n_splits=10)
    Features = X

with timer("Making Variables"):
    gViolentExtremismAccu = 0
    gViolentExtremismPrecision = [0,0, 0]
    gViolentExtremismRecall = [0,0, 0]
    gViolentExtremismFScore = [0,0, 0]
    gViolentExtremismEpochs = 0


    gNonViolentExtremismAccu = 0
    gNonViolentExtremismPrecision = [0,0, 0]
    gNonViolentExtremismRecall = [0,0, 0]
    gNonViolentExtremismFScore = [0,0, 0]
    gNonViolentExtremismEpochs = 0


    gRadicalViolenceAccu = 0
    gRadicalViolencePrecision = [0,0, 0]
    gRadicalViolenceRecall = [0,0, 0]
    gRadicalViolenceFScore = [0,0, 0]
    gRadicalViolenceEpochs = 0


   
    gNonRadicalViolenceAccu = 0
    gNonRadicalViolencePrecision = [0,0, 0]
    gNonRadicalViolenceRecall = [0,0, 0]
    gNonRadicalViolenceFScore = [0,0, 0]
    gNonRadicalViolenceEpochs = 0


with timer("trying to clear GPU memory initially "):
    backend.clear_session()

    for i in range(20):
        gc.collect()


with timer("Making label vector"):

    YViolentExtremism = np.zeros([len(violentExtremism),3],dtype = int)
    for x in range(0, len(violentExtremism)):
        YViolentExtremism[x,violentExtremism[x]] = 1


    YNonViolentExtremism = np.zeros([len(nonViolentExtremism),3],dtype = int)
    for x in range(0, len(nonViolentExtremism)):
        YNonViolentExtremism[x,nonViolentExtremism[x]] = 1


    YRadicalViolence = np.zeros([len(radicalViolence),3],dtype = int)
    for x in range(0, len(radicalViolence)):
        YRadicalViolence[x,radicalViolence[x]] = 1


    YNonRadicalViolence = np.zeros([len(nonRadicalViolence),3],dtype = int)
    for x in range(0, len(nonRadicalViolence)):
        YNonRadicalViolence[x,nonRadicalViolence[x]] = 1

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


with timer("Cross Validation for Violent Extremism"):
    iteration = 0
    for train_index, test_index in kf.split(Features):
        with timer("Making nueral network for iteration : " + str(iteration + 1) + "   for Violent Extremism"):
            print("\n\n\n\nMaking nueral Network for iteration:",iteration, "   for Violent Extremism")
            iteration += 1


            

            #Making Training and Testing Data
            X_Train = [Features[x] for x in train_index]
            X_Test = [Features[x] for x in test_index]
            violentExtremismTrain = YViolentExtremism[train_index]
            violentExtremismTest = YViolentExtremism[test_index]
            violentExtremismTest1 = [violentExtremism[x] for x in test_index]

            tokenisedTrain = tokenizer.texts_to_sequences(X_Train)
            tokenisedTest = tokenizer.texts_to_sequences(X_Test)


            max_review_length = 180
            X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')
            X_Test = sequence.pad_sequences(tokenisedTest, maxlen=max_review_length,padding='post')

            #Radical
            violentExtremismModel = Sequential()
            violentExtremismModel.add(Embedding(vocabSize, 100, input_length=max_review_length,weights=[embedding_matrix]))
            violentExtremismModel.add(Dropout(0.2))
            violentExtremismModel.add(Conv1D(64, 5, activation='relu'))
            violentExtremismModel.add(MaxPooling1D(pool_size=4))
            violentExtremismModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            violentExtremismModel.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            violentExtremismModel.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
            violentExtremismModel.add(Dense(3, activation='sigmoid'))
            violentExtremismModel.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
            epochs = 1
            print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Violent Extremism")
            fitHistory = violentExtremismModel.fit(X_Train, violentExtremismTrain, epochs = 1, batch_size = 150)
            trainingAccuracy = fitHistory.history['acc']
            while(trainingAccuracy[0] < 0.99 or epochs < 15):
                epochs += 1
                print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Violent Extremism")
                fitHistory = violentExtremismModel.fit(X_Train, violentExtremismTrain, epochs = 1, batch_size = 150)
                trainingAccuracy = fitHistory.history['acc']
                if(epochs == 100):
                    break
            violentExtremismScore = violentExtremismModel.evaluate(X_Test,violentExtremismTest,verbose = 100)
            accuViolentExtremismLstm1 = violentExtremismScore[1]
            print("\nViolence Extremism training Done for Iteration ",iteration, " for part 1 with epochs : ", epochs)
            positiveViolentExtremism = [x for x in violentExtremismTest if x[0] == 1]
            predictViolentExtremism = violentExtremismModel.predict_classes(X_Test, verbose = 1)
            positivePredViolentExtremism = [x for x in predictViolentExtremism if x > 0]
            prec1, recall1, fscore1, support1 = precision_recall_fscore_support(violentExtremismTest1, predictViolentExtremism)
            print("Number of positive Examples : ",len(positiveViolentExtremism),  "\nratio : ", (len(positiveViolentExtremism) / len(violentExtremismTest)), "\nPositive Predicted : ", len(positivePredViolentExtremism), "\naccuracy : ", accuViolentExtremismLstm1, "\nwrongness : ", 1 - accuViolentExtremismLstm1,"\n\nPrecision : ",prec1,"\nRecall : ", recall1, "\nf1Score : ", fscore1, "\nsupport : ", support1 )


            gViolentExtremismAccu += accuViolentExtremismLstm1
            gViolentExtremismPrecision[0] += prec1[0]
            try:
                gViolentExtremismPrecision[1] += prec1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gViolentExtremismPrecision[2] += prec1[2]
            except:
                print("LALALALA")
            gViolentExtremismRecall[0] += recall1[0]
            try:
                gViolentExtremismRecall[1] += recall1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gViolentExtremismRecall[2] += recall1[2]
            except:
                print("LALALALA")
            gViolentExtremismFScore[0] += fscore1[0]
            try:
                gViolentExtremismFScore[1] += fscore1[1]
            except:
                #doNothing
                print("LAla")

            try:
                gViolentExtremismFScore[2] += fscore1[2]

            except:
                print("LALALALA")
            gViolentExtremismEpochs += epochs


            with timer("trying to clear GPU memory"):
                del violentExtremismModel

                backend.clear_session()

                for i in range(20):
                    gc.collect()


with timer("Cross Validation for Non Violent Extremism"):
    iteration = 0
    for train_index, test_index in kf.split(Features):
        with timer("Making nueral network for iteration : " + str(iteration + 1) + "   for Non Violent Extremism"):
            print("\n\n\n\nMaking nueral Network for iteration:",iteration, "   for Non Violent Extremism")
            iteration += 1


            

            #Making Training and Testing Data
            X_Train = [Features[x] for x in train_index]
            X_Test = [Features[x] for x in test_index]
            nonViolentExtremismTrain = YNonViolentExtremism[train_index]
            nonViolentExtremismTest = YNonViolentExtremism[test_index]
            nonViolentExtremismTest1 = [nonViolentExtremism[x] for x in test_index]

            tokenisedTrain = tokenizer.texts_to_sequences(X_Train)
            tokenisedTest = tokenizer.texts_to_sequences(X_Test)


            max_review_length = 180
            X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')
            X_Test = sequence.pad_sequences(tokenisedTest, maxlen=max_review_length,padding='post')

            #Radical
            nonViolentExtremismModel = Sequential()
            nonViolentExtremismModel.add(Embedding(vocabSize, 100, input_length=max_review_length,weights=[embedding_matrix]))
            nonViolentExtremismModel.add(Dropout(0.2))
            nonViolentExtremismModel.add(Conv1D(64, 5, activation='relu'))
            nonViolentExtremismModel.add(MaxPooling1D(pool_size=4))
            nonViolentExtremismModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            nonViolentExtremismModel.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            nonViolentExtremismModel.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
            nonViolentExtremismModel.add(Dense(3, activation='sigmoid'))
            nonViolentExtremismModel.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
            epochs = 1
            print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Non Violent Extremism")
            fitHistory = nonViolentExtremismModel.fit(X_Train, nonViolentExtremismTrain, epochs = 1, batch_size = 150)
            trainingAccuracy = fitHistory.history['acc']
            while(trainingAccuracy[0] < 0.99 or epochs < 15):
                epochs += 1
                print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Non Violent Extremism")
                fitHistory = nonViolentExtremismModel.fit(X_Train, nonViolentExtremismTrain, epochs = 1, batch_size = 150)
                trainingAccuracy = fitHistory.history['acc']
                if(epochs == 100):
                    break
            nonViolentExtremismScore = nonViolentExtremismModel.evaluate(X_Test,nonViolentExtremismTest,verbose = 100)
            accuNonViolentExtremismLstm1 = nonViolentExtremismScore[1]
            print("\nNon Violent Extremism Training Done for Iteration ",iteration, " for part 1 with epochs : ", epochs)
            positiveViolentExtremism = [x for x in nonViolentExtremismTest if x[0] == 1]
            predictNonViolentExtremism = nonViolentExtremismModel.predict_classes(X_Test, verbose = 1)
            positivePredNonViolentExtremism = [x for x in predictNonViolentExtremism if x > 0]
            prec1, recall1, fscore1, support1 = precision_recall_fscore_support(nonViolentExtremismTest1, predictNonViolentExtremism)
            print("Number of positive Examples : ",len(positiveViolentExtremism),  "\nratio : ", (len(positiveViolentExtremism) / len(nonViolentExtremismTest)), "\nPositive Predicted : ", len(positivePredNonViolentExtremism), "\naccuracy : ", accuNonViolentExtremismLstm1, "\nwrongness : ", 1 - accuNonViolentExtremismLstm1,"\n\nPrecision : ",prec1,"\nRecall : ", recall1, "\nf1Score : ", fscore1, "\nsupport : ", support1 )


            gNonViolentExtremismAccu += accuNonViolentExtremismLstm1
            gNonViolentExtremismPrecision[0] += prec1[0]
            try:
                gNonViolentExtremismPrecision[1] += prec1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gNonViolentExtremismPrecision[2] += prec1[2]
            except:
                print("LALALALA")
            gNonViolentExtremismRecall[0] += recall1[0]
            try:
                gNonViolentExtremismRecall[1] += recall1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gNonViolentExtremismRecall[2] += recall1[2]
            except:
                print("LALALALA")
            gNonViolentExtremismFScore[0] += fscore1[0]
            try:
                gNonViolentExtremismFScore[1] += fscore1[1]
            except:
                #doNothing
                print("LAla")

            try:
                gNonViolentExtremismFScore[2] += fscore1[2]

            except:
                print("LALALALA")
            gNonViolentExtremismEpochs += epochs


            with timer("trying to clear GPU memory"):
                del nonViolentExtremismModel

                backend.clear_session()

                for i in range(20):
                    gc.collect()




#######################################################################################################################
with timer("Cross Validation for Radical Violence"):
    iteration = 0
    for train_index, test_index in kf.split(Features):
        with timer("Making nueral network for iteration : " + str(iteration + 1) + "   for Radical Violence"):
            print("\n\n\n\nMaking nueral Network for iteration:",iteration, "   for Radical Violence")
            iteration += 1


            

            #Making Training and Testing Data
            X_Train = [Features[x] for x in train_index]
            X_Test = [Features[x] for x in test_index]
            radicalViolenceTrain = YRadicalViolence[train_index]
            radicalViolenceTest = YRadicalViolence[test_index]
            radicalViolenceTest1 = [radicalViolence[x] for x in test_index]

            tokenisedTrain = tokenizer.texts_to_sequences(X_Train)
            tokenisedTest = tokenizer.texts_to_sequences(X_Test)


            max_review_length = 180
            X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')
            X_Test = sequence.pad_sequences(tokenisedTest, maxlen=max_review_length,padding='post')

            #Radical
            radicalViolenceModel = Sequential()
            radicalViolenceModel.add(Embedding(vocabSize, 100, input_length=max_review_length,weights=[embedding_matrix]))
            radicalViolenceModel.add(Dropout(0.2))
            radicalViolenceModel.add(Conv1D(64, 5, activation='relu'))
            radicalViolenceModel.add(MaxPooling1D(pool_size=4))
            radicalViolenceModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            radicalViolenceModel.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            radicalViolenceModel.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
            radicalViolenceModel.add(Dense(3, activation='sigmoid'))
            radicalViolenceModel.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
            epochs = 1
            print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Violent Extremism")
            fitHistory = radicalViolenceModel.fit(X_Train, radicalViolenceTrain, epochs = 1, batch_size = 150)
            trainingAccuracy = fitHistory.history['acc']
            while(trainingAccuracy[0] < 0.99 or epochs < 15):
                epochs += 1
                print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Violent Extremism")
                fitHistory = radicalViolenceModel.fit(X_Train, radicalViolenceTrain, epochs = 1, batch_size = 150)
                trainingAccuracy = fitHistory.history['acc']
                if(epochs == 100):
                    break
            radicalViolenceScore = radicalViolenceModel.evaluate(X_Test,radicalViolenceTest,verbose = 100)
            accuRadicalViolenceLstm1 = radicalViolenceScore[1]
            print("\nRadical Violence Training Done for Iteration ",iteration, " for part 1 with epochs : ", epochs)
            positiveRadicalViolence = [x for x in radicalViolenceTest if x[0] == 1]
            predictRadicalViolence = radicalViolenceModel.predict_classes(X_Test, verbose = 1)
            positivePredRadicalViolence = [x for x in predictRadicalViolence if x > 0]
            prec1, recall1, fscore1, support1 = precision_recall_fscore_support(radicalViolenceTest1, predictRadicalViolence)
            print("Number of positive Examples : ",len(positiveRadicalViolence),  "\nratio : ", (len(positiveRadicalViolence) / len(radicalViolenceTest)), "\nPositive Predicted : ", len(positivePredRadicalViolence), "\naccuracy : ", accuRadicalViolenceLstm1, "\nwrongness : ", 1 - accuRadicalViolenceLstm1,"\n\nPrecision : ",prec1,"\nRecall : ", recall1, "\nf1Score : ", fscore1, "\nsupport : ", support1 )


            gRadicalViolenceAccu += accuRadicalViolenceLstm1
            gRadicalViolencePrecision[0] += prec1[0]
            try:
                gRadicalViolencePrecision[1] += prec1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gRadicalViolencePrecision[2] += prec1[2]
            except:
                print("LALALALA")
            gRadicalViolenceRecall[0] += recall1[0]
            try:
                gRadicalViolenceRecall[1] += recall1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gRadicalViolenceRecall[2] += recall1[2]
            except:
                print("LALALALA")
            gRadicalViolenceFScore[0] += fscore1[0]
            try:
                gRadicalViolenceFScore[1] += fscore1[1]
            except:
                #doNothing
                print("LAla")

            try:
                gRadicalViolenceFScore[2] += fscore1[2]

            except:
                print("LALALALA")
            gRadicalViolenceEpochs += epochs


            with timer("trying to clear GPU memory"):
                del radicalViolenceModel

                backend.clear_session()

                for i in range(20):
                    gc.collect()


with timer("Cross Validation for Non Radical Violence"):
    iteration = 0
    for train_index, test_index in kf.split(Features):
        with timer("Making nueral network for iteration : " + str(iteration + 1) + "   for Non Radical Violence"):
            print("\n\n\n\nMaking nueral Network for iteration:",iteration, "   for Non Radical Violence")
            iteration += 1


            

            #Making Training and Testing Data
            X_Train = [Features[x] for x in train_index]
            X_Test = [Features[x] for x in test_index]
            nonRadicalViolenceTrain = YNonRadicalViolence[train_index]
            nonRadicalViolenceTest = YNonRadicalViolence[test_index]
            nonRadicalViolenceTest1 = [nonRadicalViolence[x] for x in test_index]

            tokenisedTrain = tokenizer.texts_to_sequences(X_Train)
            tokenisedTest = tokenizer.texts_to_sequences(X_Test)


            max_review_length = 180
            X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')
            X_Test = sequence.pad_sequences(tokenisedTest, maxlen=max_review_length,padding='post')

            #Radical
            nonRadicalViolenceModel = Sequential()
            nonRadicalViolenceModel.add(Embedding(vocabSize, 100, input_length=max_review_length,weights=[embedding_matrix]))
            nonRadicalViolenceModel.add(Dropout(0.2))
            nonRadicalViolenceModel.add(Conv1D(64, 5, activation='relu'))
            nonRadicalViolenceModel.add(MaxPooling1D(pool_size=4))
            nonRadicalViolenceModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            nonRadicalViolenceModel.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            nonRadicalViolenceModel.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
            nonRadicalViolenceModel.add(Dense(3, activation='sigmoid'))
            nonRadicalViolenceModel.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
            epochs = 1
            print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Non Violent Extremism")
            fitHistory = nonRadicalViolenceModel.fit(X_Train, nonRadicalViolenceTrain, epochs = 1, batch_size = 150)
            trainingAccuracy = fitHistory.history['acc']
            while(trainingAccuracy[0] < 0.99 or epochs < 15):
                epochs += 1
                print("\nTraining until accuracy improves for epoch = ", epochs, " for iteration : ", iteration, "   for Non Violent Extremism")
                fitHistory = nonRadicalViolenceModel.fit(X_Train, nonRadicalViolenceTrain, epochs = 1, batch_size = 150)
                trainingAccuracy = fitHistory.history['acc']
                if(epochs == 100):
                    break
            nonRadicalViolenceScore = nonRadicalViolenceModel.evaluate(X_Test,nonRadicalViolenceTest,verbose = 100)
            accuNonRadicalViolenceLstm1 = nonRadicalViolenceScore[1]
            print("\nNon Radical Violence Training Done for Iteration ",iteration, " for part 1 with epochs : ", epochs)
            positiveNonRadicalViolence = [x for x in nonRadicalViolenceTest if x[0] == 1]
            predictNonRadicalViolence = nonRadicalViolenceModel.predict_classes(X_Test, verbose = 1)
            positivePredNonRadicalViolence = [x for x in predictNonRadicalViolence if x > 0]
            prec1, recall1, fscore1, support1 = precision_recall_fscore_support(nonRadicalViolenceTest1, predictNonRadicalViolence)
            print("Number of positive Examples : ",len(positiveNonRadicalViolence),  "\nratio : ", (len(positiveNonRadicalViolence) / len(nonRadicalViolenceTest)), "\nPositive Predicted : ", len(positivePredNonRadicalViolence), "\naccuracy : ", accuNonRadicalViolenceLstm1, "\nwrongness : ", 1 - accuNonRadicalViolenceLstm1,"\n\nPrecision : ",prec1,"\nRecall : ", recall1, "\nf1Score : ", fscore1, "\nsupport : ", support1 )


            gNonRadicalViolenceAccu += accuNonRadicalViolenceLstm1
            gNonRadicalViolencePrecision[0] += prec1[0]
            try:
                gNonRadicalViolencePrecision[1] += prec1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gNonRadicalViolencePrecision[2] += prec1[2]
            except:
                print("LALALALA")
            gNonRadicalViolenceRecall[0] += recall1[0]
            try:
                gNonRadicalViolenceRecall[1] += recall1[1]
            except:
                #doNothing
                print("LAla")
            try:
                gNonRadicalViolenceRecall[2] += recall1[2]
            except:
                print("LALALALA")
            gNonRadicalViolenceFScore[0] += fscore1[0]
            try:
                gNonRadicalViolenceFScore[1] += fscore1[1]
            except:
                #doNothing
                print("LAla")

            try:
                gNonRadicalViolenceFScore[2] += fscore1[2]

            except:
                print("LALALALA")
            gNonRadicalViolenceEpochs += epochs


            with timer("trying to clear GPU memory"):
                del nonRadicalViolenceModel

                backend.clear_session()

                for i in range(20):
                    gc.collect()



with timer("Final Output"):


    gViolentExtremismAccu /= 10
    gViolentExtremismEpochs /= 10
    gViolentExtremismPrecision = [x / 10 for x in gViolentExtremismPrecision]
    gViolentExtremismRecall = [x / 10 for x in gViolentExtremismRecall]
    gViolentExtremismFScore = [x / 10 for x in gViolentExtremismFScore]


    gNonViolentExtremismAccu /= 10
    gNonViolentExtremismEpochs /= 10
    gNonViolentExtremismPrecision = [x /10 for x in gNonViolentExtremismPrecision]
    gNonViolentExtremismRecall = [x / 10 for x in gNonViolentExtremismRecall]
    gNonViolentExtremismFScore = [x / 10 for x in gNonViolentExtremismFScore]


    gRadicalViolenceAccu /= 10
    gRadicalViolenceEpochs /= 10
    gRadicalViolencePrecision = [x / 10 for x in gRadicalViolencePrecision]
    gRadicalViolenceRecall = [x / 10 for x in gRadicalViolenceRecall]
    gRadicalViolenceFScore = [x / 10 for x in gRadicalViolenceFScore]


    gNonRadicalViolenceAccu /= 10
    gNonRadicalViolenceEpochs /= 10
    gNonRadicalViolencePrecision = [x / 10 for x in gNonRadicalViolencePrecision]
    gNonRadicalViolenceRecall = [x / 10 for x in gNonRadicalViolenceRecall]
    gNonRadicalViolenceFScore = [x / 10 for x in gNonRadicalViolenceFScore]


    print("\n\n\n\n")
    print("Score for Violent Extremism : \n", "accuracy : ", gViolentExtremismAccu, "\nPrecision : ", gViolentExtremismPrecision, "\nRecall : ", gViolentExtremismRecall, "\nFScore : ", gViolentExtremismFScore, "\nAverageEpochs : ", gViolentExtremismEpochs)
    
    print("\n\n\n\n")
    print("Score for Non Violent Extremism : \n", "accuracy : ", gNonViolentExtremismAccu, "\nPrecision : ", gNonViolentExtremismPrecision, "\nRecall : ", gNonViolentExtremismRecall, "\nFScore : ", gNonViolentExtremismFScore, "\nAverageEpochs : ", gNonViolentExtremismEpochs)

    print("\n\n\n\n")
    print("Score for Radical Violence : \n", "accuracy : ", gRadicalViolenceAccu, "\nPrecision : ", gRadicalViolencePrecision, "\nRecall : ", gRadicalViolenceRecall, "\nFScore : ", gRadicalViolenceFScore, "\nAverageEpochs : ", gRadicalViolenceEpochs)
    
    print("\n\n\n\n")
    print("Score for Non Radical Violence : \n", "accuracy : ", gNonRadicalViolenceAccu, "\nPrecision : ", gNonRadicalViolencePrecision, "\nRecall : ", gNonRadicalViolenceRecall, "\nFScore : ", gNonRadicalViolenceFScore, "\nAverageEpochs : ", gNonRadicalViolenceEpochs)
    
