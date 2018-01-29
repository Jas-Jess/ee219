import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer

'''##############
     Part B
###############'''
categories_8 = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 'rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

eight_train = fetch_20newsgroups(subset='train', categories=categories_8, shuffle=True, random_state=42)
eight_test = fetch_20newsgroups(subset='test', categories=categories_8, shuffle=True, random_state=42)

# Tokenize each document into words
# Gets rid of stop words, and stemmed version of word
stop_words = text.ENGLISH_STOP_WORDS
vectorizer = CountVectorizer(min_df=2, stop_words = stop_words)
X_train_counts = vectorizer.fit_transform(eight_train.data)
X_test_counts = vectorizer.transform(eight_test.data)

# TFIDF (NOT FINISHED)
# We set smooth_idf = false so we use the equation idf(d, t) = log [ n / df(d, t) ] + 1
tfidf_transformer = TfidfTransformer(smooth_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Print the number of documents and term 
print X_train_tfidf.shape
print X_test_tfidf.shape

'''##############
     Part C
###############'''
categories_20 = ['alt.atheism', 'comp.graphics',  'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

twenty_train = fetch_20newsgroups(subset='train', categories=categories_20, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories_20, shuffle=True, random_state=42)

