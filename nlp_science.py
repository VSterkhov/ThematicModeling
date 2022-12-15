from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim import similarities
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
from sklearn.linear_model import LogisticRegression

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

tokenizer = RegexpTokenizer(r"\w+")

tokenize_words = [tokenizer.tokenize(text.lower()) for text in tqdm(texts)]
concat_array = numpy.concatenate(tokenize_words, axis=0)

freq_dict = Counter(concat_array)
most_com = freq_dict.most_common(10)

for idx in most_com:
    print(idx)
    #print(tokenize_words[idx[1]])



freqs = list(freq_dict.values())
freqs = sorted(freqs, reverse = True)

fig, ax = plt.subplots()
ax.plot(freqs[:300], range(300))
plt.show()



cnt = Counter()
n_words = []
n_tokens = []
tokens = []
for text in tqdm(texts):
    tokens = tokenizer.tokenize(text.lower())
    cnt.update([token for token in tokens])
    n_words.append(len(cnt))
    n_tokens.append(sum(cnt.values()))


fig, ax = plt.subplots()
ax.plot(n_tokens, n_words)
plt.show()


#dictionary = Dictionary(vector)
#corpus = [dictionary.doc2bow(y) for y in tqdm(vector)]


df = pd.DataFrame(resp_array)
df = df.dropna()
df = df.reset_index(drop=True) 

x_train, x_test, y_train, y_test = train_test_split(df.text, df.rating_grade)

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
bow = vec.fit_transform(x_train)
clf = LogisticRegression(random_state=42, solver='liblinear')
clf.fit(bow, y_train)
pred = clf.predict(vec.transform(x_test))
print(pred, y_test)




from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim import similarities


texts = [text.split() for text in df.text]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

index = similarities.MatrixSimilarity(corpus_tfidf)
sims = index[corpus_tfidf]

plt.figure(figsize = (10,10))
sns.heatmap(data=sims, cmap = 'Spectral').set(xticklabels=[],yticklabels=[])
plt.title("Матрица близости")
plt.show()


from gensim.models import lsimodel
lsi = lsimodel.LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=30)
print(lsi.show_topics(5))


corpus_lsi = lsi[corpus]
index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[corpus_lsi]
sims  = (sims + 1)/2.
plt.figure(figsize = (10,10))
sns.heatmap(data=sims, cmap = 'Spectral').set(xticklabels=[], yticklabels=[])
plt.title("Матрица близости")
plt.show()


'''
# поизучаем, что здесь происходит
# corpus


tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

index = similarities.MatrixSimilarity(corpus_tfidf)
sims = index[corpus_tfidf]

#sims.shape
    

plt.figure(figsize = (10,10))
sns.heatmap(data=sims, cmap = 'Spectral').set(xticklabels=[],yticklabels=[])
plt.title("Матрица близости")
plt.show()
'''