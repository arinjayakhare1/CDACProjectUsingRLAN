import re
import nltk
from nltk.corpus import stopwords
import os
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

with timer("Reading and preprocessing the tweets"):
	texts = []

	with open("normalTweets","r") as f:
		for x in f:
			if(preprocess(x) != ""):
				texts.append(preprocess(x))
				print(len(texts))

	print(len(texts))
	texts.sort(key = lambda s : len(s), reverse = True)

with timer("Saving to file"):		
	print("Saving to file")
	p = ""

	if(os.path.isfile("normalTexts.txt")):
		os.remove("normalTexts.txt")

	f = open("normalTexts.txt","a+")

	count = 0
	for line in texts:
		if(line == p):
			continue
		p = line	
		count += 1
		s = line + "\n"
		f.write(s)
		
	f.close()	
	print("Saved to file")

with timer("Checking the file"):
	count = 0
	with open("normalTexts.txt","r") as f:
		for line in f:
			count += 1
	print("Numbr of lines = ",count)		