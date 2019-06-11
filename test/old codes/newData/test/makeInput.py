import random
import os



import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("\n\n" + name + ' done in ' + str(round(time.time() - t0)) + 's \n')
    

texts = []
scores = []
count = 0
with timer("Reading file 1"):
	with open("radicalTexts.txt","r") as f:
		for line in f:
			if(line.endswith("\n")):
				line = line[:-1]
			texts.append(line)
			scores.append(1)
			count += 1
			if(count == 50000):
				break

with timer("Reading file 2"):
	count = 0
	with open("normalTexts.txt","r") as f:
		for line in f:
			if(line.endswith("\n")):
				line = line[:-1]
			texts.append(line)
			scores.append(0)
			count += 1
			if(count == 50000):
				break

with timer("Reshuffling the array"):
	array = []
	for x in range(0, len(texts)):
		array.append(x)
	random.shuffle(array)
	print(array)
	print(type(array))

consecutiveZeroes = 0
consecutiveOnes = 0

with timer("Saving to file"):
	newTexts = []
	newScores = []
	consecutiveZeroes = 0
	consecutiveOnes = 1

	for x in array:
		newTexts.append(texts[x])
		newScores.append(scores[x])
		if(len(newScores) > 1):
			t = len(newScores)
			if(newScores[t - 1] == 0 and newScores[t - 2] == 0):
				consecutiveZeroes += 1
			if(newScores[t - 1] == 1 and newScores[t - 2] == 1):
				consecutiveOnes += 1

	if(os.path.isfile("input.csv")):
		os.remove("input.csv")

	f = open("input.csv","a+")

	for x in range(0,len(newTexts)):
		s = newTexts[x] + "," + str(newScores[x]) + "\n"
		f.write(s)
	f.close()
	
	print("consecutiveZeroes : ", consecutiveZeroes, "\nconsecutiveOnes : ", consecutiveOnes)	