import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
import re
import nltk
from nltk.corpus import stopwords
import bz2
import json
from tqdm import tqdm
from pymystem3 import Mystem
import calendar
import time
import dis 

warnings.filterwarnings('ignore')
m = Mystem()

nltk.download('stopwords')

mystopwords = stopwords.words('russian') + [
    'это', 'наш' , 'тыс', 'млн', 'млрд', 'также',  'т', 'д',
    'который','прошлый','сей', 'свой', 'наш', 'мочь', 'такой'
]
ru_words = re.compile("[А-Яа-я]+")

def words_only(text):
    return " ".join(ru_words.findall(text))


def lemmatize(text, mystem=m):
    try:
        return "".join(m.lemmatize(text)).strip()  
    except:
        return " "


def remove_stopwords(text, mystopwords = mystopwords):
    try:
        return " ".join([token for token in text.split() if not token in mystopwords])
    except:
        return ""

    
def preprocess(text):
    return remove_stopwords(lemmatize(words_only(text.lower())))


rows_max = 10000
it_counter = 0
json_str = ""

def createResponsesFile(json_str):
    current_GMT = time.gmtime()
    timestamp = calendar.timegm(current_GMT)
    with bz2.open("./after_process/banks_responses_"+str(timestamp)+".json.bz2", "w") as writing_file:
        json_bytes = json_str.encode('utf-8')
        writing_file.write(json_bytes)
        writing_file.close()


with bz2.BZ2File('./input/banki_responses.json.bz2', 'r') as thefile:
    for row in tqdm(thefile):
        resp = json.loads(row)
        if not resp['rating_not_checked'] and (len(resp['text'].split()) > 0):
            it_counter+=1
            resp['text'] = preprocess(resp['text'])
            json_str = json_str + json.dumps(resp) + "\n"
            if (it_counter % rows_max) == 0:
                createResponsesFile(json_str)
                json_str = "" 
    createResponsesFile(json_str)