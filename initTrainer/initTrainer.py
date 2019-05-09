from nltk.corpus import words, stopwords
from keras.preprocessing.text import Tokenizer
import re, string
import numpy as np
from sklearn.utils import shuffle
from contextlib import contextmanager
from keras.models import Model
import time
import gc
gc.collect()
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GRU, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from nltk.corpus import words, stopwords
from keras.preprocessing.text import Tokenizer
import numpy as np
import numpy
import csv
import string
import re
import keras
import math
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as k
import tensorflow
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
import os
import pickle


@contextmanager
def timer(name):
	print("\n\nStarting to do : ", name, "\n\n")
	t0 = time.time()
	yield
	print("\n\n" + name + ' done in ' + str(round(time.time() - t0)) + 's \n')


cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]
vocabSize = len(allEnglishWords)
tokenizer = Tokenizer(num_words=vocabSize)
tokenised = tokenizer.fit_on_texts(allEnglishWords)


if(os.path.isfile('../models/tokenizer/tokenizer.pickle')):
		os.remove('../models/tokenizer/tokenizer.pickle')

with open('../models/tokenizer/tokenizer.pickle', 'wb') as handle:
	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


class RLAN:
	def __init__(self):
		with timer("Making Predictor Model"):
			self.predictorModel = self.createPredictorModel()
			print(self.predictorModel)
		with timer("Making Judge Model"):
			self.judgeModel = self.createJudgeNetwork()
			print(self.judgeModel)
		self.cachedStopWords = stopwords.words('english')
		self.allEnglishWords = words.words()
		self.allEnglishWords[:] = [x.lower() for x in self.allEnglishWords]
		self.vocabSize = len(self.allEnglishWords)
		self.tokenizer = Tokenizer(num_words = self.vocabSize)
		self.tokenizer.fit_on_texts(self.allEnglishWords)
		self.learningRate = 1
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())

	def exit(self):
		del self.predictorModel
		del self.judgeModel
		for i in range(1, 20):
			gc.collect()
		keras.backend.clear_session()

	def fit(self, labelledTweets, labelledTweetScores, unlabelledTweets):
		with timer("Pre training the predictor model"):
		
			predIn = []
			predOut = []
			scores = [0, 0, 0, 0, 0]

			for i in labelledTweetScores:
				scores[i] += 1

			minScore = min(scores)
			scores = []
			for i in range(0, 5):
				scores.append(minScore)

			for i in range(0, len(labelledTweets)):
				if(scores[labelledTweetScores[i]] > 0):
					predIn.append(labelledTweets[i])
					predOut.append(labelledTweetScores[i])
					scores[labelledTweetScores[i]] -= 1

			with timer("Making label vector"):
				Y = np.zeros([len(predOut), 5], dtype=int)
				for x in range(0, len(predOut)):
						Y[x, predOut[x]] = 1

			predOut = np.array(Y)
			tokenisedTrain = tokenizer.texts_to_sequences(predIn)
			predIn = sequence.pad_sequences(tokenisedTrain, maxlen=180, padding='post')

			epochs = 0
			trainingAccuracy = [0]
			while(epochs < 30):
				epochs += 1
				print("epochs : ", epochs)
				fitHistory = self.predictorModel.fit(predIn, predOut, epochs=1, batch_size=32)
				trainingAccuracy = fitHistory.history['acc']

		with timer("Prepredicting the labels of unlabelled data"):
			tokenisedForPredicting = tokenizer.texts_to_sequences(unlabelledTweets)
			X_Unlabelled = sequence.pad_sequences(tokenisedForPredicting, maxlen = 180, padding = 'post')
			predScores = self.predictorModel.predict_classes(X_Unlabelled, verbose = 1)

		pred = [0, 0, 0, 0, 0]
		for i in predScores:
			pred[i] += 1

		print("\n\n\n\n", pred, "\n\n\n\n")


		with timer("Making predicted label vector"):
			Z = np.zeros([len(predScores), 5], dtype=int)
			for x in range(0, len(predScores)):
					Z[x, predScores[x]] = 1
		Z = np.array(Z)

		with timer("Pretraining the judge model"):

			tokenisedForTraining = tokenizer.texts_to_sequences(labelledTweets)
			X_Labelled = sequence.pad_sequences(tokenisedForTraining, maxlen = 180, padding = 'post')
			Y = np.zeros([len(labelledTweetScores), 5], dtype=int)
			for x in range(0, len(labelledTweetScores)):
					Y[x, labelledTweetScores[x]] = 1
			judgeIn = []
			judgeAuxIn = []
			judgeOut = []
			for i in X_Labelled:
				judgeIn.append(i)
			for i in X_Unlabelled:
				judgeIn.append(i)
			for i in Y:
				judgeAuxIn.append(i)
			for i in Z:
				judgeAuxIn.append(i)
			for i in Y:
				judgeOut.append(1)
			for i in Z:
				judgeOut.append(0)

			judgeIn, judgeAuxIn, judgeOut = shuffle(judgeIn, judgeAuxIn, judgeOut, random_state = 0)

			# print(judgeIn)
			# print(judgeAuxIn)
			# print(judgeOut)

			epochs = 0
			trainingAccuracy = [0]
			while(epochs < 15):
				epochs += 1
				print("epochs : ", epochs)
				fitHistory = self.judgeModel.fit([judgeIn, judgeAuxIn], judgeOut, epochs = 1, batch_size = 150)
				trainingAccuracy = fitHistory.history['acc']

		with timer("Using reinforcement learning"):
			with timer("Generating Constants"):
				numberOfTrainingIterations = 300
				m = 1000
			

			while(numberOfTrainingIterations > 0):
				numberOfTrainingIterations -= 1


				

				print("\nIteration : ", 300 - numberOfTrainingIterations)
				with timer("Making label vector"):
					Y = np.zeros([len(labelledTweetScores), 5], dtype=int)
					for x in range(0, len(labelledTweetScores)):
							Y[x, labelledTweetScores[x]] = 1

				Y = np.array(Y)
				trainLabelledData, trainLabels, trainUnlabelledData = labelledTweets, Y, unlabelledTweets
				trainLabelledData, trainLabels = shuffle(trainLabelledData, trainLabels, random_state = 0)
				trainUnlabelledData = shuffle(trainUnlabelledData, random_state = 0)
				trainLabelledData2, trainLabels2, trainUnlabelledData2 = [], [], []
				for i in range(0, m):
					trainLabelledData2.append(trainLabelledData[i])
					trainLabels2.append(trainLabels[i])
					trainUnlabelledData2.append(trainUnlabelledData[i])
				trainLabelledData, trainLabels, trainUnlabelledData = trainLabelledData2, trainLabels2, trainUnlabelledData2
				temp1, temp2, temp3 = trainLabelledData2, trainLabels2, trainUnlabelledData2

				tokenisedForPredicting = tokenizer.texts_to_sequences(trainLabelledData)
				trainLabelledData = sequence.pad_sequences(tokenisedForPredicting, maxlen = 180, padding = 'post')

				tokenisedForPredicting = tokenizer.texts_to_sequences(trainUnlabelledData)
				trainUnlabelledData = sequence.pad_sequences(tokenisedForPredicting, maxlen = 180, padding = 'post')

				predtrainUnlabbeledData = self.predictorModel.predict_classes(trainUnlabelledData, verbose = 1)
				with timer("Making predicted label vector"):
					W = np.zeros([len(predtrainUnlabbeledData), 5], dtype=int)
					for x in range(0, len(predtrainUnlabbeledData)):
							W[x, predtrainUnlabbeledData[x]] = 1
				W = np.array(W)
				predtrainUnlabbeledData = W

				judgeIn = []
				judgeAuxIn = []
				judgeOut = []
				indexes = []
				judgeIn2 = []
				for i in temp1:
					judgeIn2.append(i)
				for i in temp3:
					judgeIn2.append(i)
				for i in trainLabelledData:
					judgeIn.append(i)
				for i in trainUnlabelledData:
					judgeIn.append(i)
				for i in trainLabels:
					judgeAuxIn.append(i)
				for i in predtrainUnlabbeledData:
					judgeAuxIn.append(i)
				for i in trainLabels:
					judgeOut.append(1)
				for i in predtrainUnlabbeledData:
					judgeOut.append(0)
				for i in range(0, len(judgeIn)):
					indexes.append(i)


				# print(judgeIn)
				# print("\n\n",  judgeAuxIn)
				# print("\n\n", judgeOut)


				judgeIn, judgeAuxIn, judgeOut, indexes, judgeIn2 = shuffle(judgeIn, judgeAuxIn, judgeOut, indexes, judgeIn2, random_state = 0)

				predictedProbabilities = self.predictorModel.predict(np.array(judgeIn), verbose = 1)

				k = 3
				while(k > 0):
					k -= 1
					with timer("Updating the judge model"):
						
						epochs = 0
						trainingAccuracy = [0]
						while(epochs < 15):
							epochs += 1
							print("\nIteration : ", 200 - numberOfTrainingIterations, "\tsteps : ", 3  - k)
							print("epochs for updating judje model : ", epochs)
							fitHistory = self.judgeModel.fit([judgeIn, judgeAuxIn], judgeOut, epochs = 1, batch_size = 64)
							trainingAccuracy = fitHistory.history['acc']

					with timer("Calculating reward"):

						with timer("Making gradient generator function"):
							weights = self.predictorModel.trainable_weights
							gradients = self.predictorModel.optimizer.get_gradients(self.predictorModel.total_loss, weights)

							inputTensor = [
												self.predictorModel.inputs[0], 
												self.predictorModel.sample_weights[0],
												self.predictorModel.targets[0],
												keras.backend.learning_phase()
										  ]
							get_gradients = keras.backend.function(inputs = inputTensor, outputs = gradients)

						reward = []
						reward.append(np.zeros((5, 100, 64)))
						reward.append(np.zeros(64))
						reward.append(np.zeros((64, 400)))
						reward.append(np.zeros((100, 400)))
						reward.append(np.zeros(400))
						reward.append(np.zeros((100, 128)))
						reward.append(np.zeros((32, 128)))
						reward.append(np.zeros(128))
						reward.append(np.zeros((32, 64)))
						reward.append(np.zeros((16, 64)))
						reward.append(np.zeros(64))
						reward.append(np.zeros((16, 5)))
						reward.append(np.zeros(5))



						print(predictedProbabilities)

						for i in range(0, len(indexes)):
							# print("Calculating reward for ", i)
							if(judgeOut[i] == 1):
								rewardTemp = self.getRewardLabelled(judgeIn[i], judgeAuxIn[i], predictedProbabilities[i], get_gradients)
								# print("\n\n\n\nLabelled tweet ka : ", rewardTemp)
							else:
								rewardTemp = self.getRewardUnlabelled(judgeIn[i], predictedProbabilities[i], get_gradients)
								# print("\n\n\n\nUNLabelled tweet ka : ", rewardTemp)

							for j in range(0, len(reward)):
								reward[j] = np.add(reward[j], rewardTemp[j])
							# print("\n\n\n\n", i, " tak ka reward : ", reward)

						for i in range(0, len(reward)):
							reward[i] = reward[i] / (2 * m)
							reward[i] = reward[i] * self.learningRate

					with timer("Updating Predictor model"):
						modelWeights = self.predictorModel.get_weights()

						print("\n\n\n\nold : ")
						for i in modelWeights:
							print(type(i))
							print(i.shape)
							print(i)

						reward[1] = reward[1][0]
						reward[4] = reward[4][0]
						reward[7] = reward[7][0]
						reward[10] = reward[10][0]
						reward[12] = reward[12][0]

						print("\n\n\n\nreward : ")
						for i in reward:
							print(type(i))
							print(i.shape)
							print(i)


						for i in range(1, len(modelWeights)):
							modelWeights[i] = np.add(modelWeights[i], reward[i - 1])

						print("\n\n\n\nnew : ")
						for i in modelWeights:
							print(type(i))
							print(i.shape)
							print(i)



						self.predictorModel.set_weights(modelWeights)
					with timer("Garbage colection"):
						for i in range(0, 20):
							gc.collect()
					


	def getLabelFromArray(self, Y):
		for i in Y:
			if(i == 1):
				return i


	def evaluate_gradient(self, input, output, get_gradients):

		sw = np.ones(180)
		for i in range(15, 180):
			sw[i] = 0
		tensorInput = [
							[input], 
							sw,
							[output],
							0
					  ]
		return get_gradients(tensorInput)

	def getRewardLabelled(self, labelledTweet, labelledScore, predictedProbabilities, get_gradients):
		reward = []

		evaluatedGradient = self.evaluate_gradient(labelledTweet, labelledScore, get_gradients)
		# print("LabelledTweet : ", predictedProbabilities)
		p =  predictedProbabilities[self.getLabelFromArray(labelledScore)]
		if(p == 0):
			p = 1
		for i in range(0, len(evaluatedGradient)):
			if(i != 0):
				reward.append((evaluatedGradient[i] / p) )
		return reward







	def getRewardUnlabelled(self, unlabelledTweet, predictedProbabilities, get_gradients):
		reward = []
		reward.append(np.zeros((5, 100, 64)))
		reward.append(np.zeros(64))
		reward.append(np.zeros((64, 400)))
		reward.append(np.zeros((100, 400)))
		reward.append(np.zeros(400))
		reward.append(np.zeros((100, 128)))
		reward.append(np.zeros((32, 128)))
		reward.append(np.zeros(128))
		reward.append(np.zeros((32, 64)))
		reward.append(np.zeros((16, 64)))
		reward.append(np.zeros(64))
		reward.append(np.zeros((16, 5)))
		reward.append(np.zeros(5))



		evaluatedGradient0 = self.evaluate_gradient(unlabelledTweet, [1, 0, 0, 0, 0], get_gradients)
		evaluatedGradient1 = self.evaluate_gradient(unlabelledTweet, [0, 1, 0, 0, 0], get_gradients)
		evaluatedGradient2 = self.evaluate_gradient(unlabelledTweet, [0, 0, 1, 0, 0], get_gradients)
		evaluatedGradient3 = self.evaluate_gradient(unlabelledTweet, [0, 0, 0, 1, 0], get_gradients)
		evaluatedGradient4 = self.evaluate_gradient(unlabelledTweet, [0, 0, 0, 0, 1], get_gradients)

		v0 = self.judgeModel.predict([[unlabelledTweet], [[1, 0, 0, 0, 0]]], verbose = 0)
		v1 = self.judgeModel.predict([[unlabelledTweet], [[0, 1, 0, 0, 0]]], verbose = 0)
		v2 = self.judgeModel.predict([[unlabelledTweet], [[0, 0, 1, 0, 0]]], verbose = 0)
		v3 = self.judgeModel.predict([[unlabelledTweet], [[0, 0, 0, 1, 0]]], verbose = 0)
		v4 = self.judgeModel.predict([[unlabelledTweet], [[0, 0, 0, 0, 1]]], verbose = 0)

		v = []
		v.append(v0)
		v.append(v1)
		v.append(v2)
		v.append(v3)
		v.append(v4)

		evaluatedGradients = []
		evaluatedGradients.append(evaluatedGradient0)
		evaluatedGradients.append(evaluatedGradient1)
		evaluatedGradients.append(evaluatedGradient2)
		evaluatedGradients.append(evaluatedGradient3)
		evaluatedGradients.append(evaluatedGradient4)	

		# print("UnlabelledTweet : ", predictedProbabilities)



		for i in range(0, len(reward)):
			for j in range(0, len(predictedProbabilities)):
				p = predictedProbabilities[j]
				if(p == 0):
					p = 1
				reward[i] = np.add(reward[i], (evaluatedGradients[j][i + 1] * v[j] / p))

		return reward


	def createPredictorModel(self):
		vocabSize = len(allEnglishWords)
		tokenizer = Tokenizer(num_words= vocabSize)
		tokenised = tokenizer.fit_on_texts(allEnglishWords)
		model = Sequential()
		
		with timer("Making embedding index dict"):
			embeddings_index = dict()
			f = open('glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")
			for line in f:
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs
			f.close()
			print('Loaded %s word vectors.' % len(embeddings_index))


		with timer("Making Embedding matrix"):
			embedding_matrix = np.zeros((vocabSize, 100))
			for word, index in tokenizer.word_index.items():
				if index > vocabSize - 1:
					break
				else:
					embedding_vector = embeddings_index.get(word)
					if embedding_vector is not None:
						embedding_matrix[index] = embedding_vector

		with timer("Creating predictor model"): 
			model.add(Embedding(vocabSize, 100, input_length=180, weights=[embedding_matrix]))
			model.add(Dropout(0.2))
			model.add(Conv1D(64, 5, activation='relu'))
			model.add(MaxPooling1D(pool_size=4))
			model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
			model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
			model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
			model.add(Dense(5, activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])

		return model

	def createJudgeNetwork(self):
		with timer("Making embedding index dict"):
			embeddings_index = dict()
			f = open('glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")
			for line in f:
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs
			f.close()
			print('Loaded %s word vectors.' % len(embeddings_index))


		with timer("Making Embedding matrix"):
			embedding_matrix = np.zeros((vocabSize, 100))
			for word, index in tokenizer.word_index.items():
				if index > vocabSize - 1:
					break
				else:
					embedding_vector = embeddings_index.get(word)
					if embedding_vector is not None:
						embedding_matrix[index] = embedding_vector
						
		with timer("Making JUDGE Model"):
			main_input = Input(shape = (180,), dtype = 'int32', name = 'main_input')
			x = Embedding(vocabSize, 100, input_length=180, weights=[embedding_matrix])(main_input)
			# x = Embedding(output_dim=100, input_dim=vocabSize, input_length=180)(main_input)
			lstm_out = LSTM(180)(x)

			auxilary_input = Input(shape = (5,), name = 'aux_input')
			x = keras.layers.concatenate([lstm_out, auxilary_input])

			x = Dense(64, activation = 'relu')(x)

			main_output = Dense(1, activation = 'sigmoid', name = 'main_output')(x)

			model = Model(inputs = [main_input, auxilary_input], outputs = [main_output])
			model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1], metrics = ['accuracy'])

		return model

# test = RLAN()

print("\n\nSab chal gaya\n\n")
# print(test)
# for i in test.predictorModel.get_weights():
# 	print(i.shape)
# 	print(type(i))

# weights = test.predictorModel.get_weights()
# reward = 1


def clean(s):
	transalator = str.maketrans("", "", string.punctuation)
	return s.translate(transalator)



def preprocess(text):
	text = text.split(",")[-1]
	text = clean(text).lower()
	text = text.lower()
	text = ' '.join([word for word in text.split()
									if word not in cachedStopWords])
	text = ' '.join([word for word in text.split() if(not word.startswith(
		"@") and not word.startswith("http") and not word.startswith("\\"))])
	# text = ' '.join([word for word in text.split()
	# 								if word in allEnglishWords])
	#text =  re.sub("[_]","",text)
	#remove tags
	text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
	# remove special characters and digits
	text = re.sub("(\\d|\\W)+", " ", text)
	if(text.startswith("rt ") or text.startswith(" rt")):
		text = text[3:]
	if(text == "rt"):
		text = ""
	while(text != "" and text[0] == ' '):
		text = text[1:]
	return text


with timer("Reading labelled data"):
	columnToRead = 1
	x = []
	y = []
	x0 = []
	x1 = []
	x2 = []
	label = []
	typeLabel = [0, 0, 0, 0, 0]
	with open("labelledInput.csv", 'r', encoding="utf8") as csvFile:
		reader = csv.reader(csvFile)
		p = 0
		for row in reader:
			if(p == 0):
				p = p + 1
				continue
			if(len(row) >= 2):
				s = row[0].split(",")[1]
				# print(s)
				x.append(preprocess(s))
				sc = 4
				for i in range(1, 5):
					if(row[i] == "1.0" or row[i] == "2.0"):
						sc = i - 1
						break
				label.append(sc)
				typeLabel[sc] += 1
			p = p + 1
	csvFile.close

print("text : ", len(x), "\t scores = ", typeLabel[0] + typeLabel[1] + typeLabel[2] + typeLabel[3] + typeLabel[4])
print("Types : \n", typeLabel)

X = []
for t in x:
		t = re.sub(r'[^\w\s]', ' ', t)
		t = ' '.join([word for word in t.split() if word != " "])
		t = t.lower()
		t = ' '.join([word for word in t.split()
									if word not in cachedStopWords])
		X.append(t)

with timer("Reading unlabelled input"):
	unlabelledTweets = []
	counter = 0
	with open('unlabelledInput.csv', 'r', encoding = "ISO-8859-1") as csvFile:
		reader = csv.reader(csvFile)
		for row in reader:
			# if(counter == 1000):
			# 	break
			counter += 1
			unlabelledTweets.append(preprocess(row[0]))
			if(len(unlabelledTweets) % 1000 == 0):
				print(len(unlabelledTweets))

with timer("Creating model"):
	model = RLAN()
	model.fit(X, label, unlabelledTweets)


	if(os.path.isfile("../models/model/model.h5")):
			os.remove("../models/model/model.h5")


	model.predictorModel.save("../models/model/model.h5")