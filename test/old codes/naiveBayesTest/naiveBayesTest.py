import csv
import os
import re
import string
import math
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import collections 
import numpy as np

from sklearn.model_selection import KFold
cachedStopWords = stopwords.words("english")

class stringProcesserNB(object):

    """Implementation of Naive Bayes for binary classification"""
    def predict(self,X):
    	result = []
    	for x in X:
    		counts = self.get_word_counts(self.tokenize(x))
    		feature_score = 0
    		normal_score = 0
    		for word, _ in counts.items():
    			if word not in self.vocab: continue
    			#add Laplace smoothing
    			log_w_given_feature = math.log((self.word_counts['feature'].get(word,0.0) + 1) / (sum(self.word_counts['feature'].values()) + len(self.vocab)))
    			log_w_given_normal = math.log((self.word_counts['normal'].get(word,0.0) + 1) / (sum(self.word_counts['normal'].values()) + len(self.vocab)))
    			feature_score += log_w_given_feature
    			normal_score += log_w_given_normal
    		feature_score += self.log_class_priors['feature']
    		normal_score += self.log_class_priors['normal']

    		if feature_score > normal_score:
    			result.append(1)
    		else:
    			result.append(0)
    	return result		 	

    def Fit(self,X,Y):
    	self.log_class_priors = {}
    	self.word_counts = {}
    	self.vocab = set()

    	n = len(X)
    	sumFeature = 0.0
    	sumNormal = 0.0
    	for label in Y:
    		if (label == 1):
    			sumFeature = sumFeature + 1
    		else:
    			sumNormal = sumNormal + 1
    	#print("sum1:",sumFeature," sum2:",sumNormal," n:",n," fractionf:", (sumFeature / n)," fractionn:",(sumNormal / n))		
    			
    	self.log_class_priors['feature'] = math.log(sumFeature / n)
    	self.log_class_priors['normal'] = math.log(sumNormal / n)
    	self.word_counts['feature'] = {}
    	self.word_counts['normal'] = {}


    	for x,y in zip(X, Y):
    		c = 'feature' if y == 1 else 'normal'
    		counts = self.get_word_counts(self.tokenize(x))
    		for word,count in counts.items():
    			if word not in self.vocab:
    				self.vocab.add(word)
    			if word not in self.word_counts[c]:
    				self.word_counts[c][word] = 0.0
    			self.word_counts[c][word] += count

    def get_word_counts(self, words):
    	word_counts = {}
    	for word in words:
    		word_counts[word] = word_counts.get(word, 0.0) + 1.0
    	return word_counts	

    def clean(self,s):
    	transalator = str.maketrans("","",string.punctuation)
    	return s.translate(transalator)

    def tokenize(self, text):
    	text = self.clean(text).lower()
    	text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    	return re.split("\W+",text)		




	
 


if __name__ == '__main__':
	print("Starting Main")
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

	kf = KFold(n_splits=20)
	iteration = 0

	gFraudAccu = 0
	gRadicalAccu = 0
	gViolenceAccu = 0
	gTotalAccu = 0

	for train_index, test_index in kf.split(x):
		iteration += 1
		print("\n\n\nNaive Bayes for iteration : ",iteration)
		trainX = [x[i] for i in train_index]
		testX = [x[i] for i in test_index]
		trainFraud = [fraud[i] for i in train_index]
		testFraud = [fraud[i] for i in test_index]
		trainRadical = [radical[i] for i in train_index]
		testRadical = [radical[i] for i in test_index]
		trainViolence = [violence[i] for i in train_index]
		testViolence = [violence[i] for i in test_index]


		#Training models for all 3 types

		#SVM
		print("Training with Naive Bayes")

		#Training for Fraud
		fraudLearnNb = stringProcesserNB()
		fraudLearnNb.Fit(trainX, trainFraud)

		#Training for Radicalism
		radicalLearnNb = stringProcesserNB()
		radicalLearnNb.Fit(trainX,trainRadical)

		#Training for Violence
		violenceLearnNb = stringProcesserNB()
		violenceLearnNb.Fit(trainX,trainViolence)

		#Testing Accuracy of svm
		print("Testing Accuracy of Naive Bayes")

		#Testing for Fraud
		PredFraudNb = fraudLearnNb.predict(testX)

		#Testing for Radicalism
		PredRadicalNb = radicalLearnNb.predict(testX)

		#TestingForViolence
		PredViolenceNb = violenceLearnNb.predict(testX)

		#FindingAccuracyScore
		AccuFraudNb = accuracy_score(testFraud, PredFraudNb)
		AccuRadicalNb = accuracy_score(testRadical, PredRadicalNb)
		AccuViolenceNb = accuracy_score(testViolence, PredViolenceNb)
		totalAccuNb = (AccuFraudNb + AccuViolenceNb + AccuRadicalNb) / 3

		positiveFraud = [x for x in testFraud if x == 1]
		positiveRadical = [x for x in testRadical if x == 1]
		positiveViolence = [x for x in testViolence if x == 1]
		positivePredFraud = [x for x in PredFraudNb if x == 1]
		positivePredRadical = [x for x in PredRadicalNb if x == 1]
		positivePredViolence = [x for x in PredViolenceNb if x == 1]
		print("\nNumber of positive Examples for fraud : ",len(positiveFraud),  "  ratio : ", (len(positiveFraud) / len(testFraud)), "  Poitivepredictions : ", len(positivePredFraud) , "   accuracy : ", AccuFraudNb, "  wrongness : ", 1 - AccuFraudNb )
		print("\nNumber of positive Examples for radical : ",len(positiveRadical),  "  ratio : ", (len(positiveRadical) / len(testRadical)), "  Poitivepredictions : ", len(positivePredRadical) , "   accuracy : ", AccuRadicalNb, "  wrongness : ", 1 - AccuRadicalNb )
		print("\nNumber of positive Examples for violence : ",len(positiveViolence),  "  ratio : ", (len(positiveViolence) / len(testViolence)), "  Poitivepredictions : ", len(positivePredViolence) , "   accuracy : ", AccuViolenceNb, "  wrongness : ", 1 - AccuViolenceNb )



		print("\n\nAccuracyScores for Naive Bayes Iteration:",iteration,"\nFraud: ",AccuFraudNb,"\nRadical: ",AccuRadicalNb,"\nViolence: ",AccuViolenceNb,"\nTotal Accuracy:",totalAccuNb)
		gFraudAccu += AccuFraudNb
		gRadicalAccu += AccuRadicalNb
		gViolenceAccu += AccuViolenceNb
		gTotalAccu += totalAccuNb


	gFraudAccu /= 20
	gRadicalAccu /= 20
	gViolenceAccu /= 20
	gTotalAccu /= 20	
	
	print("\n\n\noverall AccuracyScores for Naive Bayes :","\nFraud: ",gFraudAccu,"\nRadical: ",gRadicalAccu,"\nViolence: ",gViolenceAccu,"\nTotal Accuracy:",gTotalAccu)
	




