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
import math
import operator

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


# Load the all 20 category
categories_20 = ['alt.atheism', 'comp.graphics',  'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

# Load the documents per category
twenty_train_per_category = []
for category in categories_20:
	categories = [category]
	cat_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42).data
	words = ""
	for doc in cat_data:
		words = words +" " + doc
	twenty_train_per_category.append(words)

# Tokenize each document into words
# Gets rid of stop words, and stemmed version of word
# Ignores words appearing in less then 2 documents
vectorizer = CountVectorizer(min_df=2, stop_words= stop_words, tokenizer=LemmaTokenizer() )
X_train_counts = vectorizer.fit_transform(twenty_train_per_category)

######################
###### TF-ICF ########
######################

# Get sizing for variables
max_term_freq = [0]*X_train_counts.shape[0] #Maximum term frequency per category
cat_count = [0]*vectorized_newsgroups_train.shape[1] # Category count per term

# get each max freq in each of the twenty categories
for i in range(0, X_train_counts[0], 1):
	max_term_freq[i] = np.amax(X_train_counts[i,:])

# counts the number of terms 
for i in range(0, X_train_counts[1], 1):
	for j in range(0, X_train_counts[0], 1):
		if X_train_counts[j,i] != 0:
			cat_count[i] += 1;
		# Else cat_count += 0

# Preallocate tf_icf
X_train_tficf = np.zeros((len(vectorizer.get_feature_names()), X_train_counts.shape[1])

# Calculating tf-icf
for i in range(X_train_counts[1]):
	freq = X_train_counts[:,i].toarray()

	for j in range(X_train_counts[0]):
		max_freq = max_term_freq[j]
		len_cat = len(categories_20)

		# Formula Porvided 
		X_train_tficf[i][j] = ((0.5+(0.5*(freq/float(max_freq))))*math.log10(len_cat/float(1+cat_count[i])))

# Print out the 10 Signficant Terms for the Class
# comp.sys.ibm.pc.hardware (index = 3), comp.sys.mac.hardware (index = 4), misc.forsale(index = 6), soc.religion.christian (index = 15)
cat_index = [3, 4, 6, 15]
for category in cat_index:
	cur_tficf = {} # Consider the tficf for current class
	term_index = 0;
	for term in vectorizer.get_feature_names():
		cur_tficf[term] = X_train_tficf[term][category]
		term_index += 1
	top_10_sig_terms = dict(sorted(cur_tficf.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
	print categories_20[category]
	print top_10_sig_terms.keys()
