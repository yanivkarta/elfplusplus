''' takes output from elf_walk.cpp and visualizes the elfs
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import json
import re
import math


#for vectorization

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances 
#label encoder
from sklearn.preprocessing import LabelEncoder
#pca    
from sklearn.decomposition import PCA
def main():
    #iterate files in current directory
    df = pd.DataFrame()#column per file, row per feature. transform before pca 


    for file in os.listdir():
        if file.endswith("_features.txt"):
            #
    
            if file.endswith("sparse_features.txt"):
                with open(file) as f:
                    key = file.split("_sparse_features.")[0]
                    if key not in df:
                        df[key] = {'sparse':[],'discrete':[],'continuous':[],'string':[]} 
                    for line in f:
                       df[key]['sparse'] = line

            if file.endswith("discrete_features.txt"):
                with open(file) as f:
                    for line in f:
                        #break up line tokens separated by comma 
                        tokens =[]
                        key = file.split("_discrete_features.")[0]

                        if key not in df:
                            df[key] = {'discrete':[{}],'continuous':[],'string':[]}

                        for token in line.split(","):
                            tokens.append(token)
                        df[key]['discrete'].append({ 'features':tokens}) 
                        


            if file.endswith("continuous_features.txt"):
                with open(file) as f:
                    key = file.split("_continuous_features.")[0]
                    for line in f:
                        for token in line.split(","):
                            df[key]['continuous'].append(float(token) if token.replace('.','',1).isdigit() else  token)


            if file.endswith("string_features.txt"):
                #add to binary names 

                key = file.split("_string_features.")[0]
                binary_name = key
                if binary_name not in df:
                    df[binary_name] = {'discrete':[],'continuous':[],'string':[]} 
                
                df[binary_name]['string'] = []
                with open(file) as f:
                    for line in f:
                        df[binary_name]['string'].append(line)
                        



    
    #vectorize values , prepare df
    vectorize = TfidfVectorizer()
    
    #sparse = np.array(sparse).astype(np.float64)
    #encode labels for discrete set

    
    
    df = df.T   
    df = df.fillna(0)
    print(df)
    df.to_csv("test.csv")
    #plot continuous
    
    plt.show()
    

    

    
    #dataframe["binary_names"] = binary_names

if __name__ == "__main__":
    main()