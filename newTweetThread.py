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

from counter import counter

from mongoDBUtils import mongoDBUtils


lock2 = threading.Lock()

counter = counter()

cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]

#Global keys
consumer_key = 'bXDXqTMPU4916wdoCcP0YWhPG'
consumer_secret = '5ZZRhWNAPEeA2pZ6sxFbsNs3YJrRL7Hi3JkFBopYScsS6aufN6'
access_token = '917327613009346561-M3IzKqJbUwz4AquFWvxezmkSufC8KA6'
access_token_secret = 'iRx4wnZstUg2Bg88BNw8ETJHZWzNokSHul8GloRDMaFXB'

# consumer_key = '1O2IXOW1UlzKwld3EzWZvTKes'
# consumer_secret = 'wxyUZgd6N9sjhNBCHzekpbt4tMhLjzlTkSG3OUGCAxrRucHwJw'
# access_token = '730995663647858689-C7uvohzEW06FKqgDC7G1Z5RhjDZc6Es'
# access_token_secret = 'eVd2CT8mFiMGkmltXha50Y9TlNf5f0wEPechjOs8XIVsq'

class tweetStreamlistener(StreamListener):

	def __init__(self):
		self.mongoClient = mongoDBUtils()
		self.db = self.mongoClient.db
		self.tweetIdTextCollection = self.mongoClient.tweetIdTextCollection
		self.tweetIdStatusCollection = self.mongoClient.tweetIdStatusCollection
		self.tweetIdScoreCollection = self.mongoClient.tweetIdScoreCollection
		self.twitterKeysCollection = self.mongoClient.twitterKeysCollection


	def checkRequiredKeys(self, keys):
		requiredKeys = ["id", "text"]

		if("limit" in keys):
			return False
		
		res = [reqKey in keys for reqKey in requiredKeys]

		return res

	def tweetStreamSaverThreadFunction(self, id, text, geo, created_at, entities, extended_entities, retweeted_status):
		global lock2, counter
		lock2.acquire()
		# print("Lock2 with tweetStreamSaverThreadFunction")
		endTime2 = time.time() + 86400
		endTime2 = datetime.datetime.fromtimestamp(endTime2, None)
		param1Dict = {}
		param1Dict["tweet_id"] = id
		tweetsFound = self.mongoClient.find(self.tweetIdTextCollection, param1Dict)
		tweetCount = 0

		for x in tweetsFound:
			tweetCount += 1
			if(tweetCount == 1):
				break

		if(tweetCount == 0):
			param1Dict["tweet_text"] = text

			param1Dict["TTL"] = endTime2
			param1Dict["created_at"] = created_at
			param1Dict["geo"] = geo
			param1Dict["entities"] = entities
			param1Dict["extended_entities"] = extended_entities
			param1Dict["retweeted_status"] = retweeted_status
				
			self.mongoClient.insert(self.tweetIdTextCollection, param1Dict)

		param2Dict = {}
		param2Dict["tweet_id"] = id

		param3Dict = {}
		param3Dict["tweet_id"] = id
		tweetsFound = self.mongoClient.find(self.tweetIdStatusCollection, param2Dict)
		tweetCount = 0

		for x in tweetsFound:
			tweetCount += 1
			if(tweetCount == 1):
				break

		if(tweetCount == 0):
			param2Dict["classified"] = 0
			param2Dict["trained"] = 0
			param2Dict["viralTrained"] = 0
			param2Dict["clusterAnalysis"] = 0
			param2Dict["clusterAnalysisAllClass"] = 0
			param2Dict["TTL"] = endTime2
			self.mongoClient.insert(self.tweetIdStatusCollection, param2Dict)

		counter.tweetsAdded += 1
		lock2.release()
		# print("lock2 released by tweetStreamSaverThreadFunction")

	def on_data(self, data):
		if(data.startswith("{'limit'")):
			return False
		
		tweet = json.loads(data)

		keys = tweet.keys()

		if(self.checkRequiredKeys(keys)):
			param1Dict = {}
			param1Dict["created_at"] = ""
			param1Dict["geo"] = ""
			param1Dict["entities"] = ""
			param1Dict["extended_entities"] = ""
			param1Dict["retweeted_status"] = {}

			if("created_at" in keys):
				param1Dict["created_at"] = tweet["created_at"]

			if("geo" in keys):
				param1Dict["geo"] = tweet["geo"]

			if("entities" in keys):
				param1Dict["entities"] = tweet["entities"]

			if("extended_entities" in keys):
				param1Dict["extended_entities"] = tweet["extended_entities"]

			if("retweeted_status" in keys):
				param1Dict["retweeted_status"] = tweet["retweeted_status"]

			newTweetStreamSaverThread = threading.Thread(target = self.tweetStreamSaverThreadFunction,
														 args = (tweet["id"],
														 		 tweet["text"], 
														 		 param1Dict["geo"], 
														 		 param1Dict["created_at"], 
														 		 param1Dict["entities"], 
														 		 param1Dict["extended_entities"],
														 		 param1Dict["retweeted_status"]))
			newTweetStreamSaverThread.start()
		else:
			return False

		return(True)

	def on_error(self, status):
		print("Error in retrieving tweets : ", status)
		self.getNewKeys()
		return False

	def getNewKeys(self):
		global consumer_key, consumer_secret, access_token, access_token_secret
		newKeysCursor = self.twitterKeysCollection.find({}).sort("lastUsedTime", pymongo.ASCENDING)

		newKeys = {}

		for x in newKeysCursor:
			newKeys = x
			break

		consumer_key = newKeys["consumer_key"]
		consumer_secret = newKeys["consumer_secret"]
		access_token = newKeys["access_token"]
		access_token_secret = newKeys["access_token_secret"]

		lastUsedTime = time.time()
		lastUsedTime = datetime.datetime.fromtimestamp(lastUsedTime, None)


		self.mongoClient.update(self.twitterKeysCollection, {"consumer_key" : consumer_key}, {"$set" : {"lastUsedTime" : lastUsedTime}})





	def end(self):
		self.mongoClient.disconnect()


def getNewTweets(pCounter):
	global consumer_key, consumer_secret, access_token, access_token_secret, counter

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	counter = pCounter

	keywordsPath = 'keywords.txt'
	twitterStream = Stream(auth, tweetStreamlistener())
	twitterStream.filter(track=open(keywordsPath, 'r'), languages=['en'])



