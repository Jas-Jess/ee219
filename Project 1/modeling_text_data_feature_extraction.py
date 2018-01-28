import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer

categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 'rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# Tokenize each document into words
# Gets rid of stop words, and stemmed version of word
stop_words = text.ENGLISH_STOP_WORDS
vectorizer = CountVectorizer(min_df=2, stop_words = stop_words)
X_train_counts = vectorizer.fit_transform(twenty_train.data)
X_test_counts = vectorizer.transform(twenty_test.data)

print X_train_counts.shape
print X_test_counts.shape

# TFIDF (NOT FINISHED)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
