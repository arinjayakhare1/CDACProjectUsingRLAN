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

cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]


def clean(s):
	transalator = str.maketrans("","",string.punctuation)
	return s.translate(transalator)

def preprocess(text):
	text = clean(text).lower()
	text = ' '.join([word for word in text.split() if word not in cachedStopWords])
	return text


if __name__ == '__main__':
	print("Starting Main")
	startTime = time.time()
	x = []
	y = []
	fraud = []
	radical = []
	violence = []
	fraudOne = 0
	radicalOne = 0
	violenceOne = 0
	with open("annotPart1.csv",'r', encoding="utf8") as csvFile:
		reader = csv.reader(csvFile)
		p = 0
		for row in reader:
			if(len(row) == 5 and p != 0):
				s = row[1]
				x.append(preprocess(s))
				temp = []
				temp.append(1 if row[2] == '0' else 0)
				temp.append(1 if row[3] == '0' else 0)
				temp.append(1 if row[4] == '0' else 0)
				fraud.append(1 if row[2] == '0' else 0)
				radical.append(1 if row[3] == '0' else 0)
				violence.append(1 if row[4] == '0' else 0)
				if(row[2] != '0'):
					fraudOne += 1
				if(row[3] != '0'):
					radicalOne += 1
				if(row[4] != '0'):
					violenceOne += 1		
				y.append(temp)
			p = p + 1	
	csvFile.close			
	with open("annot_part2.csv",'r', encoding="utf8") as csvFile:
		reader = csv.reader(csvFile)
		p = 0
		for row in reader:
			if(len(row) == 5 and p != 0):
				s = row[1]
				x.append(preprocess(s))
				temp = []
				temp.append(1 if row[2] == '0' else 0)
				temp.append(1 if row[3] == '0' else 0)
				temp.append(1 if row[4] == '0' else 0)
				fraud.append(1 if row[2] == '0' else 0)
				radical.append(1 if row[3] == '0' else 0)
				violence.append(1 if row[4] == '0' else 0)
				if(row[2] != '0'):
					fraudOne += 1
				if(row[3] != '0'):
					radicalOne += 1
				if(row[4] != '0'):
					violenceOne += 1
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
	fraudTrain = fraud
	radicalTrain = radical
	violenceTrain = violence
	print("Positive Frauds : ", fraudOne, "\nPositive radical : ", radicalOne, "\nPositive Violence : ", violenceOne)

	vocabSize = 500000
	tokenizer = Tokenizer(num_words= vocabSize)
	tokenised = tokenizer.fit_on_texts(allEnglishWords)

	tokenisedTrain = tokenizer.texts_to_sequences(X)
	max_review_length = 180
	X_Train = sequence.pad_sequences(tokenisedTrain, maxlen=max_review_length,padding='post')	
	
	with open('../models/tokenizer/tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)




	print("Starting training")
	#Fraud
	fraudModel = Sequential()
	fraudModel.add(Embedding(500000, 100, input_length=max_review_length))
	fraudModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	fraudModel.add(Dense(1, activation='sigmoid'))
	fraudModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	fraudModel.fit(X_Train,fraudTrain,epochs=10, batch_size=100)	
	print("Fraud Training Done")

	if(os.path.isfile("../models/fraud/fraudModel.h5")):
		os.remove("../models/fraud/fraudModel.h5")

	fraudModel.save("../models/fraud/fraudModel.h5")	
	print("Saved Fraud Model in json")

	#Radical
	radicalModel = Sequential()
	radicalModel.add(Embedding(500000, 100, input_length=max_review_length))
	radicalModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	radicalModel.add(Dense(1, activation='sigmoid'))
	radicalModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	radicalModel.fit(X_Train,radicalTrain,epochs=10, batch_size=100)
	print("Radical Training Done")

	if(os.path.isfile("../models/radical/radicalModel.h5")):
		os.remove("../models/radical/radicalModel.h5")

	radicalModel.save("../models/radical/radicalModel.h5")
	print("Saved radical Model in json")


	#Violence
	violenceModel = Sequential()
	violenceModel.add(Embedding(500000, 100, input_length=max_review_length))
	violenceModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	violenceModel.add(Dense(1, activation='sigmoid'))
	violenceModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	violenceModel.fit(X_Train,violenceTrain,epochs=10, batch_size=100)
	print("Violence Training Done")	

	if(os.path.isfile("../models/violence/violenceModel.h5")):
		os.remove("../models/violence/violenceModel.h5")

	violenceModel.save("../models/violence/violenceModel.h5")	
	print("Saved violence Model in json")