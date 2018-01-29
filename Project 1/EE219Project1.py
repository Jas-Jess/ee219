# ##### Part A: Dataset and Problem Statement
# #Goal: plot	a historgram of the number of training documents per class to check if they are evenly distributed

# import numpy as np 
# import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction import text


# # #computer technology
# # categories = ['comp.graphics']
# # graphics_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # graphics_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
# # categories = ['comp.os.ms-windows.misc']
# # misc_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # misc_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
# # categories = ['comp.sys.ibm.pc.hardware']
# # pc_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # pc_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
# # categories = ['comp.sys.mac.hardware']
# # mac_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # mac_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# # #recreational activity
# # categories = ['rec.autos']
# # autos_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # autos_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
# # categories = ['rec.motorcycles']
# # motorcycles_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # motorcycles_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
# # categories = ['rec.sport.baseball']
# # baseball_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # baseball_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
# # categories = ['rec.sport.hockey']
# # hockey_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# # hockey_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# # docsPerClass = [len(graphics_train.filenames[:]), len(misc_train.filenames[:]), len(pc_train.filenames[:]), len(mac_train.filenames[:]),
# # 				len(autos_train.filenames[:]), len(motorcycles_train.filenames[:]), len(baseball_train.filenames[:]), len(hockey_train.filenames[:])]

# # plt.bar(['graphics','misc','pc','mac','autos','motorcycles','baseball','hockey'],docsPerClass)
# # plt.ylabel('Training Files')
# # plt.xlabel('Class')
# # plt.title('Training Documents Per Class')

# # plt.show()

# ##### Part B: Modeling Text Data and Feature Extraction
# # Goal: Perform the following on the documents of balanced data of 8 classes, to convert them into
# # numerical feature vectors. First tokenize each document into words. Then, excluding the stop words,
# # punctuations, and using stemmed version of words, create a TFxIDF vector representations.

# categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
# 				'rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
# eight_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# eight_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# stop_words = text.ENGLISH_STOP_WORDS
# vectorizer = CountVectorizer(min_df=1,stop_words=stop_words)
# print vectorizer

# # print eight_train.data[0]
# X = vectorizer.fit_transform(eight_train.data)
# # print eight_train.data[3]

# vectorizer.get_feature_names()
# X.toarray()

# Y = vectorizer.transform(eight_test.data)
# print Y.toarray()


from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem.snowball import SnowballStemmer
import a
import string
import re

# Uncomment if the machine is missing punkt, wordnet or stopwords modules.
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')



# RegExpTokenizer reduces term count from 29k to 25k
class StemTokenizer(object):
    def __init__(self):
        self.snowball_stemmer = SnowballStemmer("english")

    def __call__(self, doc):
        doc = re.sub('[,.-:/()?{}*$#&]', ' ', doc)
        doc = ''.join(ch for ch in doc if ch not in string.punctuation)
        doc = ''.join(ch for ch in doc if ord(ch) < 128)
        doc = doc.lower()
        words = doc.split()
        words = [word for word in words if word not in text.ENGLISH_STOP_WORDS]

        return [
            self.snowball_stemmer.stem(word) for word in words
        ]

def get_vectorizer():
    return CountVectorizer(
        tokenizer=StemTokenizer(),
        lowercase=True,
        min_df = 2,
        max_df = 0.99
    )

def get_tfid_transformer():
    return TfidfTransformer(
        norm='l2',
        sublinear_tf=True
    )

if __name__ == "__main__":
    categories=[
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey'
    ]

    pipeline = Pipeline(
        [
            ('vectorize', get_vectorizer()),
            ('tf-idf', get_tfid_transformer())
        ]
    )

    train = a.fetch_train(categories)

    print("%d documents" % len(train.filenames))
    print("%d categories" % len(train.target_names))

    train_idf = pipeline.fit_transform(train.data)
    print "Number of terms in TF-IDF representation:",train_idf.shape[1]