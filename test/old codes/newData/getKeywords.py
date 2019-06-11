import re
import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os

cachedStopWords = stopwords.words("english")


 
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    print("2 : ", len(sorted_items))
    print("2 : ", topn)
    sorted_items1 = []
    for x in range(0, topn):
    	sorted_items1.append(sorted_items[x])
    #use only topn items from vector
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items1:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def preprocess(text):
	# lowercase
	text=text.lower()
	#remove tags
	text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
	# remove special characters and digits
	text=re.sub("(\\d|\\W)+"," ",text)
	text = ' '.join([word for word in text.split() if word not in cachedStopWords])
	return text


if __name__ == "__main__":
	print("Starting main")

	texts = []

	with open("texts.txt","r",encoding = "utf8") as textFile:
		for row in textFile:
			texts.append(row)

	print(len(texts))		

	cv=CountVectorizer()
	word_count_vector=cv.fit_transform(texts)
	feature_names=cv.get_feature_names()

	tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
	tfidf_transformer.fit(word_count_vector)
	tf_idf_vector=tfidf_transformer.transform(cv.transform(texts))

	sorted_items=sort_coo(tf_idf_vector.tocoo())
	print("1 : ",len(sorted_items))
	#extract only the top n; n here is 100
	keywords=extract_topn_from_vector(feature_names,sorted_items,topn=500)

	print("\n===Keywords===")
	for k in keywords:
	    print(k,keywords[k])


	if(os.path.isfile("keywordsTemp.txt")):
		os.remove("keywordsTemp.txt")

	f = open("keywordsTemp.txt","a+")

	for k in keywords:
		f.write(k)
		f.write("\n")

