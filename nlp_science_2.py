from sklearn.ensemble import RandomForestClassifier
import bz2
from tqdm import tqdm
import json 
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk import ngrams
import numpy
nltk.download('punkt')
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer


source_dir = "./after_process/"
texts = []
resp_array = []
#for file_bz2 in listdir(source_dir):
with bz2.BZ2File(source_dir+"banks_responses_1670583357.json.bz2", 'r') as thefile:
    for row in tqdm(thefile):
        resp = json.loads(row)
        resp_array.append(resp)
        if (len(resp['text'].split()) > 0):
            texts.append(resp['text'])


df = pd.DataFrame(resp_array)
df = df.dropna()
df = df.reset_index(drop=True) 


df = df.loc[(df['rating_grade'] == 1) | (df['rating_grade'] == 5)]
y = df.rating_grade

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
X = vectorizer.fit_transform(df.text).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X2 = tfidfconverter.fit_transform(X).toarray()

x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)

rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
predictions_rfc = rfc.predict(x_test)
print('RandomForestClassifier metrics:\n', classification_report(y_test, predictions_rfc))


'''
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.fit_transform(x_test)
x_train_cv.shape
x_test_cv.shape



'''