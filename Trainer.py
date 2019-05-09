import os
import math
import numpy as np 
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
import collections
import string
import re
import tweepy
import csv
import pandas as pd
import pymongo
from mongoDBUtils import mongoDBUtils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.models import load_model
import time
import threading
import pickle
from keras import backend as K
import gc
import datetime

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json

cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]

class Trainer:
	def __init__(self, counter):
		self.mongoClient = mongoDBUtils()
		self.db = self.mongoClient.db
		self.tweetIdTextCollection = self.mongoClient.tweetIdTextCollection
		self.tweetIdStatusCollection = self.mongoClient.tweetIdStatusCollection
		self.tweetIdScoreCollection = self.mongoClient.tweetIdScoreCollection
		self.model = "models/model/model.h5"
		self.counter = counter

		K.clear_session()

		for i in range(20):
			gc.collect()

	
	def loadModel(self, h5FileName):
		loadedModel = load_model(h5FileName)
		return loadedModel

		
	def getUntrainedTweets(self):
		paramDict = {}
		paramDict["trained"] = 0
		paramDict["classified"] = 1
		untrainedTweetIds = self.mongoClient.find(self.tweetIdStatusCollection, paramDict)

		classifiedTweetData = []
		classifiedTweetIds = []

		for x in untrainedTweetIds:
			if(self.mongoClient.find(self.tweetIdScoreCollection,{"tweet_id":x["tweet_id"]}).count() == 0):
				continue

			tempParamDict = {}
			tempParamDict["tweet_id"] = x["tweet_id"]
			temp = []
			temp.append(x["tweet_id"])
			classifiedTweetText = self.mongoClient.find(self.tweetIdTextCollection, tempParamDict)[0]["tweet_text"]
			temp.append(classifiedTweetText)
			classifiedScores = self.mongoClient.find(self.tweetIdScoreCollection, tempParamDict)
			classifiedScore = {}
			for y in classifiedScores:
				classifiedScore = y
				break
			Y = []
			for i in range(0, len(self.mongoClient.classes)):
				Y.append(int(classifiedScore[self.mongoClient.classes[i]]))
			temp.append(Y)

			classifiedTweetData.append(temp)
			classifiedTweetIds.append(temp[0])
		return classifiedTweetData, classifiedTweetIds	
	
	def fitDataOnModel(self, model, X, Y):
		epochs = 1
		fitHistory = model.fit(X, Y, epochs = 1, batch_size = 150)
		trainingAccuracy = fitHistory.history['acc']
		while(trainingAccuracy[0] < 0.99 or epochs < 15):
			epochs += 1
			print("Fitting for epoch : ",epochs)
			fitHistory = model.fit(X, Y, epochs = 1, batch_size = 150)
			trainingAccuracy = fitHistory.history['acc']
			if(epochs == 50):
				break
		return model

	def saveModel(self, model, h5FileName):
		if(os.path.isfile(h5FileName)):
			os.remove(h5FileName)

		model.save(h5FileName)	

	def returnLabelFromArray(self, scoreArray):
		p = 5
		for i in range(0, len(scoreArray)):
			if(scoreArray[i] > 0):
				return i


	def updateTrained(self, trainedTweetIds, typeScores):
		for i in range(0, len(trainedTweetIds)):
			self.mongoClient.update(self.tweetIdStatusCollection, {"tweet_id" : trainedTweetIds[i]}, {"$set" : {"trained" : 1}})

			self.counter.trainedTweets[self.mongoClient.classes[self.returnLabelFromArray(typeScores[i])]] += 1				

	def trainModels(self):
		untrainedTweets, untrainedTweetIds = self.getUntrainedTweets()
		print("Got ", len(untrainedTweets), " to train on")
		if(len(untrainedTweets) == 0):
			return
		model = load_model(self.model)


		tweetTexts = []
		typeScores = []

		for x in untrainedTweets:
			t = re.sub(r'[^\w\s]',' ',x[1])
			t = ' '.join([word for word in t.split() if word != " "])
			t = t.lower()
			t = ' '.join([word for word in t.split() if word not in cachedStopWords])
			t = ' '.join([word for word in t.split() if word in allEnglishWords])
			tweetTexts.append(t)
			typeScores.append(x[2])

		print("Type Scores : ", typeScores)
		Scores = typeScores
		print("Scores : ", Scores)

		if(len(tweetTexts) == 0):
			return

		with open('models/tokenizer/tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)
		
		tokenisedTrainTexts = tokenizer.texts_to_sequences(tweetTexts)
		max_review_length = 180
		trainTweetTexts = sequence.pad_sequences(tokenisedTrainTexts, maxlen=max_review_length,padding='post')	


		newModel = self.fitDataOnModel(model, trainTweetTexts, Scores)


		self.saveModel(newModel, self.model)	


		self.updateTrained(untrainedTweetIds, typeScores)

		del model
		del newModel
		

	def end(self):
		K.clear_session()

		for i in range(20):
			gc.collect()
		self.mongoClient.disconnect()