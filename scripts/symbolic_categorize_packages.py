import os 
import pandas as pd
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
import json

#itertools
import itertools

#text pipeline
from transformers import pipeline, EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


#classifier = pipeline("text-classification", model="j-hartmann/symbolic_categorize_packages") #model="j-hartmann/symbolic_categorize_packages" 
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
#open installed_packages.json
with open('installed_packages.json') as json_file:
    p_ = json.load(json_file)

class_results =[]
sections = []
sec2pack = dict()
pack2result = dict()
agg_result = dict()
for package in p_:
    print ("[+] %s" % package, '\n' )
    print ("[%s] %s" %(package, p_[package].get('Section')), '\n' )
    
    if p_[package] is None:
        continue
    text = ''
    descriptions = ''
     
    #text = p_[package]
    package_name = package
    current_section = p_[package].get('Section')

    text = str(p_[package])  
    
    if 'Description' in p_[package].keys() and p_[package]['Description'] is not None:
        descriptions = p_[package]['Description']
    else:
        descriptions = text

    if text == '' or descriptions == '':
        print ('[+]skipping',package,len(text),len(descriptions),'\n')
        continue

    #extract symbolic categories and sequences from text
    candidate_labels = ["kernel","system","service","utility","graphics","database","network","development","security","library","resources","media","miscellaneous"]
    #measure classification time 
   

    class_results = classifier(str(text).replace('_',' '), candidate_labels)
    if class_results is not None:
        labels = class_results['labels']
        scores = class_results['scores'] 
        pack2result[package_name] = dict(zip(labels,scores)) 

        #pack2result[package_name] = class_results
        for label,score in zip(labels,scores):
            agg_result[label] = agg_result.get(label,0.0) + score


    print ('[+]class_results:',class_results if class_results is not None else 'None') 

#normalize the agg_result
for k,v in agg_result.items():
    agg_result[k] = v / len(class_results)

print ('[+]aggregated avg label scores:',agg_result)

#sort the results
#reorder labels and scores corresponding to a constant order 
labels = ['kernel','system','service','utility','graphics','database','network','development','security','library','resources','media','miscellaneous']
discrete_labels =[0,1,2,3,4,5,6,7,8,9,10,11,12]


columns =['package','kernel','system','service','utility','graphics','database','network','development','security','library','resources','media','miscellaneous']

#save the results as csv from dict
df = pd.DataFrame(columns=columns)


#save the score for each label into the dataframe

for package in pack2result.keys():
    for label,score in pack2result[package].items():
        df[package][label] = score

#save the results as csv



df.to_csv('symbolic_categorize_packages.csv')
print ('[+]saved symbolic_categorize_packages.csv')

print('[+]done')
