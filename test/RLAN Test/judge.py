from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GRU, Input
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from nltk.corpus import words, stopwords
from keras.preprocessing.text import Tokenizer
import numpy as np
import csv
import keras
import re
import random

import time
import string
from contextlib import contextmanager
import gc
gc.collect()

@contextmanager
def timer(name):
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


class Judge():
	def __init__(self):
		judge = self.createJudgeNetwork()

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

			auxilary_input = Input(shape = (3,), name = 'aux_input')
			x = keras.layers.concatenate([lstm_out, auxilary_input])

			x = Dense(64, activation = 'relu')(x)

			main_output = Dense(1, activation = 'sigmoid', name = 'main_output')(x)

			model = Model(inputs = [main_input, auxilary_input], outputs = [main_output])
			model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1], metrics = ['accuracy'])

		return model



	


judgeModel = Judge().judge
print("\n\nModel Made : ", judgeModel, "\n\n")


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

with timer("Reading data"):
    x = []
    y = []
    radical = []
    radicalOne = 0
    with open("input.csv", 'r', encoding="utf8") as csvFile:
        reader = csv.reader(csvFile)
        p = 0
        for row in reader:
            if(p == 0):
                p = p + 1
                continue
            if(len(row) >= 2):
                s = row[0]
                x.append(preprocess(s))
                if(row[2] != '0.0'):
                    radicalOne += 1
                    if(row[2] != '1.0' and row[2] != '2.0'):
                        print("Chutiya annotator tha : ", row[2], " row : ", p)
                        radicalOne -= 1
                s = 0
                if(row[2] == '1.0'):
                    s = 1
                if(row[2] == '2.0'):
                    s = 2
                radical.append(s)
            p = p + 1
    csvFile.close

X = []
for t in x:
    t = re.sub(r'[^\w\s]', ' ', t)
    t = ' '.join([word for word in t.split() if word != " "])
    t = t.lower()
    t = ' '.join([word for word in t.split()
                  if word not in cachedStopWords])
    X.append(t)

tokenisedTest = tokenizer.texts_to_sequences(X)
X_Test = sequence.pad_sequences(
	tokenisedTest, maxlen=180, padding='post')


aux_input = []
output = []

for i in range(0, len(X_Test)):
	a = random.randint(1, 100)
	b = random.randint(1, 100)

	aux_input.append(a % 2)
	output.append(b % 2)
	

aux_input  = np.array(radical)

print("Auxilary Input = ", aux_input)
print("Output = ", output)

with timer('Fitting the model'):
	epochs = 1
	print("epochs : ", epochs)
	fitHistory = judgeModel.fit(
		[X_Test, aux_input], [output], epochs=1, batch_size=200)
	trainingAccuracy = fitHistory.history['acc']
	while(trainingAccuracy[0] < 0.9):
		epochs += 1
		print("epochs : ", epochs)
		fitHistory = judgeModel.fit([X_Test, aux_input], [output], epochs=1, batch_size=200)
		trainingAccuracy = fitHistory.history['acc']
		
		




