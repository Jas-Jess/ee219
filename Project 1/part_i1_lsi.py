
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

def print_stats(actual, predicted):
	
	print "Accuracy is ", smet.accuracy_score(actual, predicted) * 100
	print "Precision is ", smet.precision_score(actual, predicted, average='macro') * 100

	print "Recall is ", smet.recall_score(actual, predicted, average='macro') * 100

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
		new_doc = "".join(ch for ch in new_doc if ord(ch) < 128)  # remove all n5n-ascii characters
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

# 'arpack' for the ARPACK wrapper in SciPy (scipy.sparse.linalg.svds)
svd = TruncatedSVD(n_components=50, algorithm='arpack')
X_train_lsi = svd.fit_transform(X_train_tfidf)
X_test_lsi = svd.transform(X_test_tfidf)

# separate into two groups(Computer Tech & Recreation)
train_target_group = [ int(x / 4) for x in eight_train.target]
test_actual= [ int(x / 4) for x in eight_test.target]

params = [-3, -2, -1, 0, 1, 2, 3]
l1_accuracies = []
l2_accuracies = []

l1_coeficients = []
l2_coeficients = []

for param in params:
    l1_log_reg = LogisticRegression( penalty = 'l1', C = 10 ** param, solver = 'liblinear')
    l1_log_reg.fit(X_train_lsi, train_target_group)
    l1_predicted = l1_log_reg.predict(X_test_lsi)
    l1_accuracies.append(100 - smet.accuracy_score(test_actual, l1_predicted) * 100)
    l1_coeficients.append(np.mean(l1_log_reg.coef_))


    l2_log_reg = LogisticRegression(penalty = 'l2', C = 10 ** param, solver = 'liblinear')
    l2_log_reg.fit(X_train_lsi, train_target_group)
    l2_predicted = l2_log_reg.predict(X_test_lsi)
    l2_accuracies.append(100 - smet.accuracy_score(test_actual, l2_predicted) * 100)
    l2_coeficients.append(np.mean(l2_log_reg.coef_))

min_err_l1 = 100
min_param_l1 = 0

min_err_l2 = 100
min_param_l2 = 0

for i, param in enumerate(params):
	print 'L1 Regularization'
	print ('Regularization parameter: ', param)
	print ('Training Error: ', l1_accuracies[i])
	print ('Mean of coefficients: ', l1_coeficients[i])
	if l1_accuracies[i] < min_err_l1:
		min_err_l1 = l1_accuracies[1]
		min_param_l1 = param

	print 'L2 Regularization'
	print ('Regularization parameter: ', param)
	print ('Training Error: ', l2_accuracies[i])
	print ('Mean of coefficients: ', l2_coeficients[i])
	if l2_accuracies[i] < min_err_l2:
		min_err_l2 = l2_accuracies[i]
		min_param_l2 = param 

	print ''


plt.plot(l1_accuracies)
plt.title('Training Error of L1 Regulization ')
plt.xlabel('Regularization Parameter')
plt.ylabel('Testing error')
plt.xticks(range(6), [10 ** param for param in params])
plt.show()
plt.clf()

plt.plot(l2_accuracies)
plt.title('Training Error of L2 Regulization ')
plt.xlabel('Regularization Parameter')
plt.ylabel('Testing error')
plt.xticks(range(6), [10 ** param for param in params])
plt.show()

# Plot Best ROC Graph
l1_log_reg = LogisticRegression( penalty = 'l1', C = 10 ** param, solver = 'liblinear')
l1_log_reg.fit(X_train_lsi, train_target_group)
l1_predicted = l1_log_reg.predict(X_test_lsi)
predicted_probs = l1_log_reg.predict_proba(X_test_lsi)
fpr, tpr, _ = roc_curve(test_actual, predicted_probs[:,1])
print "Best L1 fit "
print_stats (test_actual, l1_predicted)
plot_roc(fpr, tpr)


l2_log_reg = LogisticRegression(penalty = 'l2', C = 10 ** param, solver = 'liblinear')
l2_log_reg.fit(X_train_lsi, train_target_group)
l2_predicted = l2_log_reg.predict(X_test_lsi)
predicted_probs = l2_log_reg.predict_proba(X_test_lsi)
fpr, tpr, _ = roc_curve(test_actual, predicted_probs[:,1])
print "Best L1 fit "
print_stats (test_actual, l2_predicted)
plot_roc(fpr, tpr)
