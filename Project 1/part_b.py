import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

# Stemmer Class for Tokenizing later
nltk.download('punkt')
nltk.download('wordnet')
stop_words = text.ENGLISH_STOP_WORDS
class LemmaTokenizer(object):
	def __init__(self):
	    self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		new_doc = re.sub('[,.-:/()?{}*$#&]', ' ', doc) # Remove symbols
		new_doc = ''.join([ch for ch in new_doc if ch not in string.punctuation])  # remove all punctuation
		new_doc = "".join(ch for ch in new_doc if ord(ch) < 128)  # remove all non-ascii characters
		new_doc = new_doc.lower() # convert to lowercase
		return [self.wnl.lemmatize(t) for t in word_tokenize(new_doc)] # stemmed


# Load the eight category
categories_8 = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 'rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

eight_train = fetch_20newsgroups(subset='train', categories=categories_8, shuffle=True, random_state=42)
eight_test = fetch_20newsgroups(subset='test', categories=categories_8, shuffle=True, random_state=42)

# Tokenize each document into words
# Gets rid of stop words, and stemmed version of word
# Ignores words appearing in less then 5 (or 2 if min_df = 2) documents 
vectorizer = CountVectorizer(min_df=5, stop_words= stop_words, tokenizer=LemmaTokenizer() )
X_train_counts = vectorizer.fit_transform(eight_train.data)
X_test_counts = vectorizer.transform(eight_test.data)

# TFIDF
# We set smooth_idf = false so we use the equation idf(d, t) = log [ n / df(d, t) ] + 1
tfidf_transformer = TfidfTransformer(smooth_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Print the number of documents and term 
print X_train_tfidf.shape
print X_test_tfidf.shape