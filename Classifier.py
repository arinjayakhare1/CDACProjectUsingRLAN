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

class Classifier:
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


	def getUnclassifiedTweets(self):
		paramDict = {}
		paramDict["classified"] = 0
		print(paramDict)
		unclassifedTweetIds = self.mongoClient.find(self.tweetIdStatusCollection, paramDict)

		unclassifiedTweetTuples = []
		for x in unclassifedTweetIds:
			tempParamDict = {}
			tempParamDict["tweet_id"] = x["tweet_id"]
			unclassifiedTweetText = self.mongoClient.find(self.tweetIdTextCollection, tempParamDict)[0]["tweet_text"]
			tempPair = [x["tweet_id"], unclassifiedTweetText]
			unclassifiedTweetTuples.append(tempPair)
		return unclassifiedTweetTuples	

	def loadModel(self, h5FileName):
		loadedModel = load_model(h5FileName)
		return loadedModel
			
		
	def updateClassifed(self, tweetIds, scores):
		for i in range(0, len(tweetIds)):
			self.mongoClient.update(self.tweetIdStatusCollection, {"tweet_id" : tweetIds[i]}, {"$set" : {"classified" : 1}})
			self.counter.classifiedTweets[self.mongoClient.classes[scores[i]]] += 1



	def classifyUnclassifedData(self):
		endTime2 = time.time() + 86400
		print("Trying to get Unclassified tweetIds")
		unclassifiedTweets = self.getUnclassifiedTweets()
		if(len(unclassifiedTweets) == 0):
			return
		print("Got ", len(unclassifiedTweets)," to classify")
		model = self.loadModel(self.model)

		with open('models/tokenizer/tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)	

		tweetTexts = []
		classifiedTweetIds = []
		for x in unclassifiedTweets:
			t = re.sub(r'[^\w\s]',' ',x[1])
			t = ' '.join([word for word in t.split() if word != " "])
			t = t.lower()
			t = ' '.join([word for word in t.split() if word not in cachedStopWords])
			t = ' '.join([word for word in t.split() if word in allEnglishWords])
			tweetTexts.append(t)
			classifiedTweetIds.append(x[0])	
		
		tokenisedTrainTexts = tokenizer.texts_to_sequences(tweetTexts)
		max_review_length = 180
		trainTweetTexts = sequence.pad_sequences(tokenisedTrainTexts, maxlen=max_review_length,padding='post')


		scores = model.predict_classes(np.array(trainTweetTexts), verbose = 1)

		classifiedDictArray = []
		for i in range(0, len(tweetTexts)):			
			Y = np.zeros(len(self.mongoClient.classes))
			Y[scores[i]] = 1
			paramDict = {}
			paramDict["tweet_id"] = classifiedTweetIds[i]
			paramDict["TTL"] = endTime2
			for i in range(0, len(self.mongoClient.classes)):
				paramDict[self.mongoClient.classes[i]] = Y[i] 
			self.mongoClient.insert(self.tweetIdScoreCollection, paramDict)


		self.updateClassifed(classifiedTweetIds, scores)

		del model

	def end(self):
		K.clear_session()

		for i in range(20):
			gc.collect()
		self.mongoClient.disconnect()	
