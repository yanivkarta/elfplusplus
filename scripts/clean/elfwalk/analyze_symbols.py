#analyze the raw symbols collected from the elf files
import sys
import pandas as pd
import argparse
import os
import re   
#vectorizers 
from sklearn.feature_extraction.text import CountVectorizer
#for tfidf
from sklearn.feature_extraction.text import TfidfTransformer
#dictionary
from collections import defaultdict 
from sklearn.feature_extraction import DictVectorizer 

#feature exrtraction 
from sklearn.decomposition import TruncatedSVD 

#partitioning 
from sklearn.cluster import KMeans

#pca
from sklearn.decomposition import PCA

#t-SNE
from sklearn.manifold import TSNE

#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

#matplotlib.colors.ListedColormap
from matplotlib.colors import ListedColormap


#ranking,feature importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 

#label encoding
from sklearn.preprocessing import LabelEncoder


#for time sampling
from time import gmtime, strftime
low_memory=False


symbols_file = 'symbols.csv'

def main(args):
    '''iterate /bin/ and parse elf files ''' 
    print("[+]start")    
    #no arguments needed
    #check if the file exists
    if not os.path.exists(symbols_file):
        print("Error: file does not exist")
        sys.exit(0)
    #read the csv file
    df = pd.read_csv(symbols_file) 
    #remove columns:
    
    #convert text to vectoized form
    #set df.dtypes to str
    df = df.astype(str)
    vectorizer = CountVectorizer()
    
    tfidf_transformer = TfidfTransformer()
        
    #vectorize the values in the dataframe
    #df = vectorizer.fit(df).transform(df)
    df_reconstructed = pd.DataFrame()
    total = len(df.columns)
    n=0
    current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    last_time = current_time


    #vectorize the text in the dataframe
    for col in df.columns:
        
        n+=1
        #check if the column is text
        #print the progress
        #if it's 'type' use label encoding
        #else use bag of words

        if col=='type':
            print('[+] vectorizing %s (%u/%u)\n' % (col, n, total ) )
            #use label encoding
            le = LabelEncoder()
            encoded_labels = le.fit_transform(df[col].astype(str))
            df_reconstructed.loc[:, col] = encoded_labels

            print('[+] done vectorizing %s (%u/%u)\n' % (col, n, total ) )
            continue
        print('[+] vectorizing %s (%u/%u)\n' % (col, n, total ) )
        #get the text from df[col]:
        text = df[col]
        print('[1]vectorizing text len %u\n' % len(text))
        
        vectorized_text = tfidf_transformer.fit_transform(vectorizer.fit_transform(text))
        #use tfidf to vectorize the text

        print('[2]done vectorizing text\n')

        #3 reduce the dimensionality
        print('[3]reducing dimensionality')
        #reduce the dimensionality
        #truncated SVD to reduce the dimensionality to the selected number of components in the dataframe 

        svd = TruncatedSVD(n_components=df.columns.shape[0], n_iter=100, random_state=42)
        semantic_matrix = svd.fit_transform(vectorized_text)

        print('[3]done reducing dimensionality\n')

        #get the cosine similarity matrix
        #print('[4]cosine similarity %s (%u/%u)\n' % (col, n, total))
        #cosine_similarity_matrix = cosine_similarity(semantic_matrix)

        #create a dataframe from the cosine similarity matrix
        df_col = pd.DataFrame(semantic_matrix) 
        df_reconstructed = pd.concat([df_reconstructed, df_col], axis=1) 

        print('[4]done cosine similarity\n')

        #display total processing time per column.
        current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print('[+] %s (%u/%u) %s %s\n' % (col, n, total, current_time, last_time) ) 
        last_time = current_time




    #save the dataframe to a csv file
    df_reconstructed.to_csv('processed_' + symbols_file, index=False)


    #print the dataframe
    
    print(df)

    print("[+]finished")

    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv[1:])
