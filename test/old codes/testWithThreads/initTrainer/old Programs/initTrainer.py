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

cachedStopWords = stopwords.words("english")


if __name__ == '__main__':
	print("Starting Main")
	startTime = time.time()
	x = []
	y = []
	fraud = []
	radical = []
	violence = []
	with open("annotPart1.csv",'r', encoding="utf8") as csvFile:
		reader = csv.reader(csvFile)
		p = 0
		for row in reader:
			if(len(row) == 5 and p != 0):
				x.append(row[1])
				temp = []
				temp.append(0 if row[2] == '0' else 1)
				temp.append(0 if row[3] == '0' else 1)
				temp.append(0 if row[4] == '0' else 1)
				fraud.append(0 if row[2] == '0' else 1)
				radical.append(0 if row[3] == '0' else 1)
				violence.append(0 if row[4] == '0' else 1)
				y.append(temp)
			p = p + 1	
	csvFile.close			
	with open("annot_part2.csv",'r', encoding="utf8") as csvFile:
		reader = csv.reader(csvFile)
		p = 0
		for row in reader:
			if(len(row) == 5 and p != 0):
				x.append(row[1])
				temp = []
				temp.append(0 if row[2] == '0' else 1)
				temp.append(0 if row[3] == '0' else 1)
				temp.append(0 if row[4] == '0' else 1)
				fraud.append(0 if row[2] == '0' else 1)
				radical.append(0 if row[3] == '0' else 1)
				violence.append(0 if row[4] == '0' else 1)
				y.append(temp)
			p = p + 1	
	csvFile.close			

	print("Size of x:",len(x)," Size of y:",len(y))
	X = []
	for t in x:
		t = re.sub(r'[^\w\s]',' ',t)
		t = ' '.join([word for word in t.split() if word != " "])
		t = t.lower()
		t = ' '.join([word for word in t.split() if word not in cachedStopWords])
		X.append(t)
	print("Type of X:",type(X))	
	Features = X
	Fraud = fraud
	Radical = radical
	Violence = violence

	kf = KFold(n_splits=10)
	iteration = 0
	gFraudAccu = 0
	gRadicalAccu = 0
	gViolenceAccu = 0
	gTotalAccu = 0
	
	vocabSize = 50000
	tokenizer = Tokenizer(num_words= vocabSize)
	tokenised = tokenizer.fit_on_texts(X)
	
	for train_index, test_index in kf.split(Features):
		iteration += 1
		print("\n\n\n\nMaking nueral Network for iteration:",iteration)
		iterStart = time.time()

		#Making Training and Testing Data
		X_Train = [Features[x] for x in train_index]
		X_Test = [Features[x] for x in test_index]
		fraudTrain = [Fraud[x] for x in train_index]
		fraudTest = [Fraud[x] for x in test_index]
		radicalTrain = [Radical[x] for x in train_index]
		radicalTest = [Radical[x] for x in test_index]
		violenceTrain = [Violence[x] for x in train_index]
		violenceTest = [Violence[x] for x in test_index]

		

		tokenisedTrain = tokenizer.texts_to_sequences(X_Train)
		tokenisedTest = tokenizer.texts_to_sequences(X_Test)

		max_review_length = 180
		X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')
		X_Test = sequence.pad_sequences(tokenisedTest, maxlen=max_review_length,padding='post')



		#Fraud
		fraudModel = Sequential()
		fraudModel.add(Embedding(50000, 100, input_length=max_review_length))
		fraudModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
		fraudModel.add(Dense(1, activation='sigmoid'))
		fraudModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		fraudModel.fit(X_Train,fraudTrain,epochs=10, batch_size=100)
		fraudScore = fraudModel.evaluate(X_Test,fraudTest,verbose = 100)
		accuFraudLstm = fraudScore[1]
		fraudEndTime = time.time()
		print("\nFraud Training Done for Iteration",iteration,"\nTime:",fraudEndTime - iterStart)
		positiveFraud = [x for x in fraudTest if x == 1]
		print("Number of positive Examples : ",len(positiveFraud),  "  ratio : ", (len(positiveFraud) / len(fraudTest)) )

		#Radical
		radicalModel = Sequential()
		radicalModel.add(Embedding(50000, 100, input_length=max_review_length))
		radicalModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
		radicalModel.add(Dense(1, activation='sigmoid'))
		radicalModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		radicalModel.fit(X_Train,radicalTrain,epochs=10, batch_size=100)
		radicalScore = radicalModel.evaluate(X_Test,radicalTest,verbose = 100)
		accuRadicalLstm = radicalScore[1]
		radicalEndTime = time.time()
		print("\nRadical Training Done for Iteration",iteration,"\nTime:",radicalEndTime - fraudEndTime)
		positiveRadical = [x for x in radicalTest if x == 1]
		print("Number of positive Examples : ",len(positiveRadical),  "  ratio : ", (len(positiveRadical) / len(radicalTest)) )


		#Violence
		violenceModel = Sequential()
		violenceModel.add(Embedding(50000, 100, input_length=max_review_length))
		violenceModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
		violenceModel.add(Dense(1, activation='sigmoid'))
		violenceModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		violenceModel.fit(X_Train,violenceTrain,epochs=10, batch_size=100)
		violenceScore = violenceModel.evaluate(X_Test,violenceTest,verbose = 100)
		accuViolenceLstm = violenceScore[1]
		violenceEndTime = time.time()
		print("\nViolence Training Done for Iteration",iteration,"\nTime:",violenceEndTime - radicalEndTime)
		positiveViolence = [x for x in violenceTest if x == 1]
		print("Number of positive Examples : ",len(positiveViolence),  "  ratio : ", (len(positiveViolence) / len(violenceTest)) )

		totalAccu = (accuViolenceLstm + accuRadicalLstm + accuFraudLstm) / 3

		gFraudAccu += accuFraudLstm
		gViolenceAccu += accuViolenceLstm
		gRadicalAccu += accuRadicalLstm
		gTotalAccu += totalAccu

		iterEndTime = time.time()
		print("\n\nAccuracyScores for LSTM Iteration:",iteration,"\nFraud: ",accuFraudLstm,"\nRadical: ",accuRadicalLstm,"\nViolence: ",accuViolenceLstm,"\nTotal Accuracy:",totalAccu,"\nTotal Time:",iterEndTime - iterStart)

	gFraudAccu /= 10
	gViolenceAccu /= 10
	gRadicalAccu /= 10
	gTotalAccu /= 10
	endTime = time.time()
	
	print("\n\n\n\nOverall AccuracyScores for LSTM :","\nFraud: ",gFraudAccu,"\nRadical: ",gRadicalAccu,"\nViolence: ",gViolenceAccu,"\nTotal Accuracy:",gTotalAccu,"\nTime:",endTime - startTime)



		
		