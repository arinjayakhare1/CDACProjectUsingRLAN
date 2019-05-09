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


from mongoDBUtils import mongoDBUtils
from Classifier import Classifier
from Trainer import Trainer
from newTweetThread import getNewTweets
from counter import counter


cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]
		
lock = threading.Lock()
lock2 = threading.Lock()


counter = counter()

def displayStatistics():
	global counter

	print("\n\n\n\n\n\n\n\n")
	print("Tweets added till now : ", counter.tweetsAdded)
	print("\nTweets classified : ")
	print("violentExtremism : ", counter.classifiedTweets["violentExtremism"])
	print("nonViolentExtremism : ", counter.classifiedTweets["nonViolentExtremism"])
	print("radicalViolence : ", counter.classifiedTweets["radicalViolence"])
	print("nonRadicalViolence : ", counter.classifiedTweets["nonRadicalViolence"])
	print("\nTweets trained : ")
	print("violentExtremism : ", counter.trainedTweets["violentExtremism"])
	print("nonViolentExtremism : ", counter.trainedTweets["nonViolentExtremism"])
	print("radicalViolence : ", counter.trainedTweets["radicalViolence"])
	print("nonRadicalViolence : ", counter.trainedTweets["nonRadicalViolence"])
	print("\n\n\n\n")



def newTweetStreamThreadFunction():
	global lock, counter
	#time.sleep(10)
	while(True):

		getNewTweets(counter)
		# print("Ended getNewTweet Function")
		# time.sleep(900)

def trainerThreadFunction():
	global lock, counter
	time.sleep(1)
	while(True):
		lock.acquire()
		print("Lock given to Trainer Thread")
		trainer = Trainer(counter)
		trainer.trainModels()
		trainer.end()
		print("Lock Released by Trainer Thread")
		displayStatistics()
		lock.release()
		time.sleep(10800)

def classifierThreadFunction():
	global lock, counter
	while(True):
		lock.acquire()
		print("Lock given to classifier thread")
		classifier = Classifier(counter)
		classifier.classifyUnclassifedData()
		classifier.end()
		print("Lock Released by Classifier Thread")
		displayStatistics()
		lock.release()
		time.sleep(1800)



def main():
	print("Starting")

	classifierThread = threading.Thread(target = classifierThreadFunction)
	classifierThread.start()

	trainerThread = threading.Thread(target = trainerThreadFunction)
	trainerThread.start()

	newTweetStreamThread = threading.Thread(target = newTweetStreamThreadFunction)
	newTweetStreamThread.start()





if __name__=="__main__":
	main()



