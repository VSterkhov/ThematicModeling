import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
import bz2
from tqdm import tqdm
import json

df_cities = pd.read_json('./dicts/russian-cities.json')
cities = df_cities['name'].values

city_probabilities = {}

for city in cities:
    city = re.sub("[^А-Яа-я]","",city)
    city = re.sub("[Ё]","Е",city)
    city = city.upper()
    
    RUSSIAN = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    POS = {l: i for i, l in enumerate(RUSSIAN)}
    probabilities = np.zeros((len(RUSSIAN), len(RUSSIAN)))
       
    def split(s):
        return [char for char in s]
         
    if (len(city)>1):
        for cp, cn in zip(city[:-1], city[1:]):
            probabilities[POS[cp], POS[cn]] += 1
    
    city_probabilities[city] = probabilities


df = pd.DataFrame(city_probabilities["КРАСНОДАР"], index=(c for c in RUSSIAN), columns=(c for c in RUSSIAN))
plt.figure(figsize=(14,8))
sns.heatmap(df)
plt.show()

def stand(text):
    text = text.upper()
    text = re.sub(r'\(.+\)', '', text)
    
    text = text.replace("Г.", "")
    text = text.replace("Ё", "Е")
    
    text = re.sub("[.|,|!|?| |-]","",text)
    text = re.sub("[^А-Я]","",text)

    return text



find_dict = {}
with open('./prepare/distinct_cities.csv', encoding=('utf-8')) as file_obj:
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        input_word = stand(row[0])
        find_dict[(input_word, row[0])] = []
        input_length = len(input_word)
        for city in city_probabilities.keys():
            score = 0
            count_bigram = 0
            probabilities_ = city_probabilities[city]
            for i in range(0, input_length, 1):
                if (i+1<input_length):
                    count_bigram+=1
                    first = input_word[i]
                    second = input_word[i+1]
                    proba = probabilities_[POS[first], POS[second]]
                    if (proba>0):
                        score = score + proba
            if ((count_bigram==2 and score==2) or (score>=3)):
                find_dict[(input_word, row[0])].append((city,score))
                
     
with open('./watch_data/cities_probabilities.csv', 'w') as csv_file:
    for key in find_dict:
        csv_file.write(str(key))
        csv_file.write("\t-\t")
        csv_file.write(str(find_dict[key]))
        csv_file.write('\n')
        
     
def takeSecond(elem):
    return elem[1]

cities_eq_dict = {}
for (short, full) in find_dict.keys():
    length = len(short)
    cities_list = find_dict[(short, full)]
    if cities_list:
        cities_list.sort(key=takeSecond, reverse=True)
        cities_eq_dict[full] = cities_list[0][0]

with open('./dicts/cities_eq_dict.json', 'w') as json_file:
    json.dump(cities_eq_dict, json_file, indent=2, ensure_ascii=False)
      














    
