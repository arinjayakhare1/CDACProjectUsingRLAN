import os
import time
import re
from nltk.corpus import stopwords

from nltk.corpus import words


import time
from contextlib import contextmanager
cachedStopWords = stopwords.words("english")

allEnglishWords = words.words()
allEnglishWords[:] = [x.lower() for x in allEnglishWords]

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("\n\n" + name + ' done in ' + str(round(time.time() - t0)) + 's \n')


def preprocess(text):
	# lowercase
	text = text[3:]
	text=text.lower()
	text = ' '.join([word for word in text.split() if word not in cachedStopWords])
	text = ' '.join([word for word in text.split() if( not word.startswith("@") and not word.startswith("http") and not word.startswith("\\")) ])
	text = ' '.join([word for word in text.split() if word in allEnglishWords])
	#text =  re.sub("[_]","",text)
	#remove tags
	text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
	# remove special characters and digits
	text=re.sub("(\\d|\\W)+"," ",text)
	if(text.startswith("rt ") or text.startswith(" rt")):
		text = text[3:]
	if(text == "rt"):
		text = ""
	while(text != "" and text[0] == ' '):
		text = text[1:]
	return text


if(os.path.isfile("texts.txt")):
	os.remove("texts.txt")

ids = []
texts = []

with  timer("Reading file 1"):
	print("Starting 99941")

	with open("AnnotationData1.csv","r",encoding='utf-8', errors='ignore') as csvFile:
		for line in csvFile:
			ind = line.split(',')[0]
			ind = ind[1:]
			s = line.split(',')[1]
			s = s[0:len(s) - 1]
			s = preprocess(s)
			if(s == ""):
				continue
			s = s + "\n"
			if( ind not in ids):
				texts.append(s)
				ids.append(ind)
				print("1 : ", len(texts))

    p = len(texts)				

	print("Got data 1")

with timer("Reading file 2"):
	with open("AnnotationData2.csv","r",encoding='utf-8', errors='ignore') as csvFile:
		for line in csvFile:
			ind = line.split(',')[0]
			ind = ind[1:]
			s = line.split(',')[1]
			s = s[0:len(s) - 1]
			s = preprocess(s)
			if(s == ""):
				continue
			s = s + "\n"
			if(ind not in ids):
				texts.append(s)
				ids.append(ind)
				print("2 : ", len(texts) - p)


	print("Got data 2")

print(len(ids))
print(len(texts))	

with timer("Saving data in file"):
	print("Saving data in file")


	if(os.path.isfile("texts.txt")):
		os.remove("texts.txt")

	f = open("texts.txt","a+")

	for text in texts:
		f.write(text)

	time2 = time.time()
	print(time2 - time1)
	f.close()		