import os
import math
import numpy as np 
import nltk
from nltk.corpus import stopwords
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


cachedStopWords = stopwords.words("english")
		
lock = threading.Lock()		

# def getNewTweets(mongoClient, tweetIdTextCollection, tweetIdClassifiedTrainedCollection):
# 	global cachedStopWords

# 	consumer_key = '1O2IXOW1UlzKwld3EzWZvTKes'
# 	consumer_secret = 'wxyUZgd6N9sjhNBCHzekpbt4tMhLjzlTkSG3OUGCAxrRucHwJw'
# 	access_token = '730995663647858689-C7uvohzEW06FKqgDC7G1Z5RhjDZc6Es'
# 	access_token_secret = 'eVd2CT8mFiMGkmltXha50Y9TlNf5f0wEPechjOs8XIVsq'

# 	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# 	auth.set_access_token(access_token, access_token_secret)
# 	api = tweepy.API(auth)



# 	for tweet in tweepy.Cursor(api.search,q="#violence",count=100,lang="en",since="2017-04-03").items(1000):		
# 		param1Dict = {}
# 		param1Dict["tweet_id"] = tweet.id
# 		tweetsFound = mongoClient.find(tweetIdTextCollection, param1Dict)
# 		tweetCount = 0

# 		for x in tweetsFound:
# 			tweetCount += 1
# 			if(tweetCount == 1):
# 				break

# 		if(tweetCount == 0):
# 			t = re.sub(r'[^\w\s]',' ',tweet.text)
# 			t = ' '.join([word for word in t.split() if word != " "])
# 			t = t.lower()
# 			t = ' '.join([word for word in t.split() if word not in cachedStopWords])
# 			param1Dict["tweet_text"] = t
# 			mongoClient.insert(tweetIdTextCollection, param1Dict)

# 		param2Dict = {}
# 		param2Dict["tweet_id"] = tweet.id
# 		tweetsFound = mongoClient.find(tweetIdClassifiedTrainedCollection, param2Dict)
# 		tweetCount = 0

# 		for x in tweetsFound:
# 			tweetCount += 1
# 			if(tweetCount == 1):
# 				break

# 		if(tweetCount == 0):
# 			param2Dict["classified"] = 0
# 			param2Dict["trained"] = 0
# 			mongoClient.insert(tweetIdClassifiedTrainedCollection, param2Dict)				

class Classifier:
	def __init__(self):
		self.mongoClient = mongoDBUtils()
		self.db = self.mongoClient.getDB("tweets")
		self.tweetIdTextCollection = self.mongoClient.getCollection(self.db, "tweetIdText")
		self.tweetIdClassifiedTrainedCollection = self.mongoClient.getCollection(self.db, "tweetIdClassifiedTrained")
		self.tweetIdClassifyCollection = self.mongoClient.getCollection(self.db, "tweetIdClassify")

	def getNewData(self):
		getNewTweets(self.mongoClient, self.tweetIdTextCollection, self.tweetIdClassifiedTrainedCollection)

	def getUnclassifiedTweets(self):
		paramDict = {}
		paramDict["classified"] = 0
		unclassifedTweetIds = self.mongoClient.find(self.tweetIdClassifiedTrainedCollection, paramDict)

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
			
		
	def updateClassifed(self, tweetIds):
		for id in tweetIds:
			self.mongoClient.update(self.tweetIdClassifiedTrainedCollection, {"tweet_id" : id}, {"$set" : {"classified" : 1}})



	def classifyUnclassifedData(self):
		unclassifedTweets = self.getUnclassifiedTweets()
		fraudModel = self.loadModel("models/fraud/fraudModel.h5")
		violenceModel = self.loadModel("models/violence/violenceModel.h5")
		radicalModel = self.loadModel("models/radical/radicalModel.h5")

		with open('models/tokenizer/tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)	

		tweetTexts = []
		classifiedTweetIds = []
		for x in unclassifedTweets:
			tweetTexts.append(x[1])
			classifiedTweetIds.append(x[0])	
		
		tokenisedTrainTexts = tokenizer.texts_to_sequences(tweetTexts)
		max_review_length = 180
		trainTweetTexts = sequence.pad_sequences(tokenisedTrainTexts, maxlen=max_review_length,padding='post')


		fraudScores = fraudModel.predict_classes(np.array(trainTweetTexts), verbose = 1)
		radicalScores = radicalModel.predict_classes(np.array(trainTweetTexts), verbose = 1)
		violenceScores = violenceModel.predict_classes(np.array(trainTweetTexts), verbose = 1)

		classifiedDictArray = []
		for i in range(0, len(tweetTexts)):
			tempDict = {}
			tempDict["tweet_id"] = classifiedTweetIds[i]
			tempDict["fraud"] = int(fraudScores[i][0])
			tempDict["violence"] = int(violenceScores[i][0])
			tempDict["radical"] = int(radicalScores[i][0])
			classifiedDictArray.append(tempDict)


		self.mongoClient.insertMany(self.tweetIdClassifyCollection, classifiedDictArray)
		self.updateClassifed(classifiedTweetIds)

	def end(self):
		K.clear_session()
		self.mongoClient.disconnect()	


class Trainer:
	def __init__(self):
		self.mongoClient = mongoDBUtils()
		self.db = self.mongoClient.getDB("tweets")
		self.tweetIdTextCollection = self.mongoClient.getCollection(self.db, "tweetIdText")
		self.tweetIdClassifiedTrainedCollection = self.mongoClient.getCollection(self.db, "tweetIdClassifiedTrained")
		self.tweetIdClassifyCollection = self.mongoClient.getCollection(self.db, "tweetIdClassify")

	
	def loadModel(self, h5FileName):
		loadedModel = load_model(h5FileName)
		return loadedModel

		
	def getUntrainedTweets(self):
		paramDict = {}
		paramDict["trained"] = 0
		paramDict["classified"] = 1
		untrainedTweetIds = self.mongoClient.find(self.tweetIdClassifiedTrainedCollection, paramDict)

		classifiedTweetData = []
		classifiedTweetIds = []

		for x in untrainedTweetIds:
			if(self.mongoClient.find(self.tweetIdClassifyCollection, {"tweet_id" : x["tweet_id"]}).count() == 0):
				continue

			tempParamDict = {}
			tempParamDict["tweet_id"] = x["tweet_id"]
			temp = []
			temp.append(x["tweet_id"])
			classifiedTweetText = self.mongoClient.find(self.tweetIdTextCollection, tempParamDict)[0]["tweet_text"]
			temp.append(classifiedTweetText)
			classifiedScores = self.mongoClient.find(self.tweetIdClassifyCollection, tempParamDict)[0]
			temp.append(classifiedScores["fraud"])
			temp.append(classifiedScores["radical"])
			temp.append(classifiedScores["violence"])

			classifiedTweetData.append(temp)
			classifiedTweetIds.append(temp[0])
		return classifiedTweetData, classifiedTweetIds	
	
	def fitDataOnModel(self, model, X, Y):
		model.fit(X, Y, epochs=10, batch_size=32,verbose = 1)	
		return model

	def saveModel(self, model, h5FileName):
		if(os.path.isfile(h5FileName)):
			os.remove(h5FileName)

		model.save(h5FileName)	

	def updateTrained(self, trainedTweetIds):
		for id in trainedTweetIds:
			self.mongoClient.update(self.tweetIdClassifiedTrainedCollection, {"tweet_id" : id}, {"$set" : {"trained" : 1}})				

	def trainModels(self):
		untrainedTweets, untrainedTweetIds = self.getUntrainedTweets()
		if(len(untrainedTweets) == 0):
			return
		fraudModel = load_model("models/fraud/fraudModel.h5")
		violenceModel = load_model("models/violence/violenceModel.h5")
		radicalModel = load_model("models/radical/radicalModel.h5")


		tweetTexts = []
		fraudScores = []
		violenceScores = []
		radicalScores = []

		for x in untrainedTweets:
			tweetTexts.append(x[1])
			fraudScores.append(x[2])
			radicalScores.append(x[3])
			violenceScores.append(x[4])

		if(len(tweetTexts) == 0):
			return

		with open('models/tokenizer/tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)
		
		tokenisedTrainTexts = tokenizer.texts_to_sequences(tweetTexts)
		max_review_length = 180
		trainTweetTexts = sequence.pad_sequences(tokenisedTrainTexts, maxlen=max_review_length,padding='post')	

		newFraudModel = self.fitDataOnModel(fraudModel, trainTweetTexts, fraudScores)

		newRadicalModel = self.fitDataOnModel(radicalModel, trainTweetTexts, radicalScores)

		newViolenceModel = self.fitDataOnModel(violenceModel, trainTweetTexts, violenceScores)

		self.saveModel(newFraudModel, "models/fraud/fraudModel.h5")

		self.saveModel(newRadicalModel, "models/radical/radicalModel.h5")	

		self.saveModel(newViolenceModel, "models/violence/violenceModel.h5")

		self.updateTrained(untrainedTweetIds)
		


	def end(self):
		K.clear_session()
		self.mongoClient.disconnect()	


def classifierThreadFunction():
	global lock
	while(True):
		lock.acquire()
		print("Lock given to classifier thread")
		classifier = Classifier()
		classifier.getNewData()
		classifier.classifyUnclassifedData()
		classifier.end()
		print("Lock released by classifier thread")
		lock.release()
		time.sleep(1800)


def trainerThreadFunction():
	global lock
	while(True):
		lock.acquire()		
		print("Trainer Thread Starting")
		print("Lock Given to trainer Thread")
		trainer = Trainer()
		trainer.trainModels()
		trainer.end()
		print("Lock released by trainer thread")
		lock.release()
		time.sleep(10800)



def main():
	classifierThread = threading.Thread(target = classifierThreadFunction)
	classifierThread.start()
	trainerThread = threading.Thread(target = trainerThreadFunction)
	trainerThread.start()





if __name__=="__main__":
	main()



	