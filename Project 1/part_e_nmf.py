
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
from sklearn import svm
import sklearn.metrics as smet
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def print_stats(actual, predicted):
	
	print "Accuracy is ", smet.accuracy_score(actual, predicted) * 100
	print "Recall is ", smet.recall_score(actual, predicted, average='macro') * 100
	print "Precision is ", smet.precision_score(actual, predicted, average='macro') * 100

	print "Confusion Matrix is ", smet.confusion_matrix(actual, predicted)	

def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.show(ax)

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
vectorizer = CountVectorizer(min_df=2, stop_words= stop_words, tokenizer=LemmaTokenizer() )
X_train_counts = vectorizer.fit_transform(eight_train.data)
X_test_counts = vectorizer.transform(eight_test.data)

# TFIDF
# We set smooth_idf = false so we use the equation idf(d, t) = log [ n / df(d, t) ] + 1
tfidf_transformer = TfidfTransformer(smooth_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# NMF
nmf = NMF(n_components=50, init='random', random_state=0)
X_train_nmf = nmf.fit_transform(X_train_tfidf)
X_test_nmf = nmf.transform(X_test_tfidf)

# separate into two groups(Computer Tech & Recreation)
train_target_group = [ int(x / 4) for x in eight_train.target]
test_actual= [ int(x / 4) for x in eight_test.target]

###### HARD SVD #######
# SVM
hard_svc = svm.SVC(kernel='linear', probability=True, C=1000)
hard_svc.fit(X_train_nmf, train_target_group)

predicted = hard_svc.predict(X_test_nmf)
predicted_probs = hard_svc.predict_proba(X_test_nmf)
fpr, tpr, _ = roc_curve(test_actual, predicted_probs[:,1])

# Stats
print "Testing NMF SVM Hard Margin"
print_stats (test_actual, predicted)
plot_roc(fpr, tpr)


###### Soft SVD #######

# SVM
soft_svc = svm.SVC(kernel='linear', probability=True, C=0.0001)
soft_svc.fit(X_train_nmf, train_target_group)

predicted = soft_svc.predict(X_test_nmf)
predicted_probs = soft_svc.predict_proba(X_test_nmf)

fpr, tpr, _ = roc_curve(test_actual, predicted_probs[:,1])

# Stats
print "Testing NMF SVM Soft Margin"
print_stats (test_actual, predicted)
plot_roc(fpr, tpr)