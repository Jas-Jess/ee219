import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.decomposition import TruncatedSVD, NMF
import sklearn.metrics as smet
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

def print_stats(actual, predicted):
    
    print "Accuracy is ", smet.accuracy_score(actual, predicted) * 100
    print "Recall is ", smet.recall_score(actual, predicted, average='macro') * 100
    print "Precision is ", smet.precision_score(actual, predicted, average='macro') * 100    

    print "Confusion Matrix is ", smet.confusion_matrix(actual, predicted)  

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
categories_4 = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']

four_train = fetch_20newsgroups(subset='train', categories=categories_4, shuffle=True, random_state=42)
four_test = fetch_20newsgroups(subset='test', categories=categories_4, shuffle=True, random_state=42)

# Tokenize each document into words
# Gets rid of stop words, and stemmed version of word
# Ignores words appearing in less then 5 (or 2 if min_df = 2) documents 
vectorizer = CountVectorizer(min_df=2, stop_words= stop_words, tokenizer=LemmaTokenizer() )
X_train_counts = vectorizer.fit_transform(four_train.data)
X_test_counts = vectorizer.transform(four_test.data)

# TFIDF
# We set smooth_idf = false so we use the equation idf(d, t) = log [ n / df(d, t) ] + 1
tfidf_transformer = TfidfTransformer(smooth_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# NMF
nmf = NMF(n_components=50, init='random', random_state=0)
X_train_nmf = nmf.fit_transform(X_train_tfidf)
X_test_nmf = nmf.transform(X_test_tfidf)

# Naive Multiclass 
print "One Vs One Classification using Naive Bayes"
one_vs_one = OneVsOneClassifier(GaussianNB()).fit(X_train_nmf, four_train.target)
predicted = one_vs_one.predict(X_test_nmf)
print_stats(four_test.target, predicted)

print "One Vs Rest Classifciation using Naive Bayes"
one_vs_rest = OneVsRestClassifier(GaussianNB()).fit(X_train_nmf, four_train.target)
predicted = one_vs_rest.predict(X_test_nmf)
print_stats(four_test.target, predicted)
