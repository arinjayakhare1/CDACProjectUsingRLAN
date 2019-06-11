from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from nltk.corpus import words, stopwords
from keras.preprocessing.text import Tokenizer
import numpy as np
import csv
import string
import re
import numpy as np
import tensorflow as tf
import keras.backend as k
import keras
import time
from contextlib import contextmanager
import gc
gc.collect()

k.clear_session()

@contextmanager
def timer(name):
	print("\n\nStarting to do : ", name, "\n\n")
	t0 = time.time()
	yield
	print("\n\n" + name + ' done in ' + str(round(time.time() - t0)) + 's \n')


print("\n\nStarting\n\n")
cachedStopWords = stopwords.words("english")
allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]
vocabSize = len(allEnglishWords)
tokenizer = Tokenizer(num_words=vocabSize)
tokenised = tokenizer.fit_on_texts(allEnglishWords)

class Predictor():
	def __init__(self):
		self.predictor = self.createPredictorModel()

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


model = Predictor().predictor
print(model)
print("\n\n\n\nWeights matrix : ")
oldWeights = model.get_weights()
for i in oldWeights:
	print(type(i))
	print(i.shape)


listOfVariableTensors = model.trainable_weights
for i in listOfVariableTensors:
	print(i)

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
	text = ' '.join([word for word in text.split()
									if word in allEnglishWords])
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
	with open("newData.csv", 'r', encoding="utf8") as csvFile:
		reader = csv.reader(csvFile)
		p = 0
		for row in reader:
			if(p == 0):
				p = p + 1
				continue
			if(len(row) >= 2):
				s = row[0].split(",")[1]
				# print(s)
				x.append(s)
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



tokenisedTrain = tokenizer.texts_to_sequences(X)
X_Labelled = sequence.pad_sequences(
	tokenisedTrain, maxlen=180, padding='post')

labelledTweetScores = label
with timer("Making label vector"):
	Y = np.zeros([len(labelledTweetScores), 5], dtype=int)
	for x in range(0, len(labelledTweetScores)):
			Y[x, labelledTweetScores[x]] = 1


Y = np.array(Y)
with timer("Pre training the model"):
	epochs = 0
	trainingAccuracy = [0]
	while(trainingAccuracy[0] < 0.99 and epochs < 1):
		epochs += 1
		print("epochs : ", epochs)
		fitHistory = model.fit(X_Labelled, Y, epochs=1, batch_size=150)
		trainingAccuracy = fitHistory.history['acc']

print(model.total_loss)

weights = model.trainable_weights
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

# print("\n\n\n\nweights : ", weights)
for i in weights:
	print(type(i))
	print(i.shape)

print("\n\n\n\n")

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 keras.backend.learning_phase(), # train or test mode
]

get_gradients = keras.backend.function(inputs=input_tensors, outputs=gradients)

output = np.zeros(5)
output[2] = 1
inputs = [
			[X_Labelled[0]], # X
			np.ones(180),
          	[output], # y
          	0 # learning phase in TEST mode
]

# print("\n\n\n\nweights : ", model.get_weights())
# print("\n\n\n\ngradients : ", get_gradients(inputs))

for i in get_gradients(inputs):
	print(type(i))
	print(i.shape)

for i in get_gradients(inputs):
	print(i)
	break