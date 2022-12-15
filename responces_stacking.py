import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
import bz2
from tqdm import tqdm
import json 
    
with open('./dicts/cities_eq_dict.json') as f:
    data = f.read()
cities_eq_dict = json.loads(data)

responses = []
with bz2.BZ2File('./input/banki_responses.json.bz2', 'r') as thefile:
    for row in tqdm(thefile):
        resp = json.loads(row)
        if not resp['rating_not_checked'] and (len(resp['text'].split()) > 0):
            responses.append(resp)
            
cities_banks_dict = {}
for resp in responses:
    f_name_city = resp['city']

    if f_name_city in cities_eq_dict:
        city = cities_eq_dict[f_name_city]
    else:
        city = "Unknown"
    
    bank_name = resp['bank_name']
    if (city in cities_banks_dict.keys()):
        if (bank_name in cities_banks_dict[city].keys()):
            cities_banks_dict[city][bank_name].append(resp)
        else:
            cities_banks_dict[city][bank_name] = [resp]
    else:
        banks_resp_dict = {}
        banks_resp_dict[bank_name] = [resp]
        cities_banks_dict[city] = banks_resp_dict
        
moscow_responces = cities_banks_dict["МОСКВА"]

city_set = set()
for bank_name in moscow_responces.keys():
    
    print(bank_name + " - " + str(len(moscow_responces[bank_name])))
    for bank_list in moscow_responces[bank_name]:
        city_set.add(bank_list['city'])
        
print(city_set)
          
          
          