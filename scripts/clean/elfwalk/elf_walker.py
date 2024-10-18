''' Analyze an ELF file,collect information about the ELF file and its sections and symbols make pandas dataframe '''

import pandas as pd
import numpy as np
import os
import re
import subprocess
import struct
import sys  
import hashlib
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection, NoteSection
from elftools.elf.constants import SH_FLAGS

#for bag of words
from sklearn.feature_extraction.text import CountVectorizer 
#for tfidf
from sklearn.feature_extraction.text import TfidfTransformer

#dictionary
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer

#for latent semantic indexing
from sklearn.decomposition import TruncatedSVD 
#for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
#matplotlib.colors.ListedColormap
from matplotlib.colors import ListedColormap
#for partitioning:
#SMOTE, ADASYN, RandomOverSampler

#for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#sns    
import seaborn as sns

#t-SNE
from sklearn.manifold import TSNE

#label encoder
from sklearn.preprocessing import LabelEncoder

#for wordcloud of symbols
from wordcloud import WordCloud

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian 

from collections import defaultdict


 

#global variable
count = 0
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
global_symbols = []
sym_df = pd.DataFrame()
token_dict_vectorizer = DictVectorizer()
create_vectorized_symbols = False #flag to create vectorized symbols use analyze_symbols.py instead(faster)

def create_tfidf_matrix(dataframe):
    """
    Create a TF-IDF matrix from a given dataframe.
    
    Args:
        dataframe (pandas.DataFrame): The dataframe from which to create the matrix.
    
    Returns:
        scipy.sparse.csr_matrix: The TF-IDF matrix.
    """
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(dataframe)
    return tfidf_matrix


def get_latent_semantic_matrix(tfidf_matrix):
    """
    Transform a TF-IDF matrix into a latent semantic matrix using SVD.

    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix.

    Returns:
        scipy.sparse.csr_matrix: The latent semantic matrix.
    """
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
    latent_semantic_matrix = svd.fit_transform(tfidf_matrix)
    return latent_semantic_matrix


def calculate_cosine_similarity(semantic_matrix):
    """
    Calculate the cosine similarity matrix from a given latent semantic matrix.

    Args:
        semantic_matrix (scipy.sparse.csr_matrix): The latent semantic matrix.

    Returns:
        scipy.sparse.csr_matrix: The cosine similarity matrix.
    """
    return cosine_similarity(semantic_matrix)

#global variables

def process_elffile(elffile):
    '''Process an ELF file and return a pandas dataframe'''
    if elffile is None:
        return None
    
    df = pd.DataFrame()

    if elffile is None:
        return df

    sections = list(elffile.iter_sections())
    elffile_header = elffile.header 
    
    #iterate header attributes and add them to the dataframe 
    for key, value in elffile_header.items():
        df[key] = value

    
    #print(elffile_header['e_ident'])


    name_entries = []
    global global_symbols,create_vectorized_symbols

    df_sym = pd.DataFrame()
    global sym_df
    df['global_start']=len(global_symbols)
    for section in sections:
        try:
            # Skip empty sections
            if section['sh_size'] == 0 or section.name == None:
                print ('[+]empty '+section.name, section['sh_size'], "\n" )
                continue
            else :
                print ('[+]processing section '+section.name, section['sh_size'], "\n" )

            if isinstance(section, SymbolTableSection):
            
                #symbols: key = section.name+'_'+symbol.name 
                for symbol in section.iter_symbols():
                    if len(symbol.name) > 0:
                        try:
                            name = str(symbol.name) 
                            value = str(symbol.entry['st_value']) if symbol.entry['st_value'] != 0 else "0x0"
                            size = str(symbol.entry['st_size'] )if symbol.entry['st_size'] != 0 else "0x0" 
                            type = str(symbol.entry['st_info']['type']) if symbol.entry['st_info']['type'] != 0 else "0x0" 

                            print("[+]section "+section.name, value, size, type, "\n" )
                            key = section.name+'_'+name
                            global_symbols.append({key: [name, value, size, type]})
                            #add the symbol to the dict vectorizer 
                            try:
                                if create_vectorized_symbols: 
                                    
                                    vectorized_symbols = vectorizer.fit_transform([key,name, value, size, type]) #vectorized_symbols = vectorizer.fit_transform({key: [ name, value, size, type]}) 
                                    vectorized_key = vectorizer.fit_transform([key])

                                    #add the vectorized symbols to the dataframe
                                    #print ('[+]normalizing '+key)
                                    normalized = tfidf_transformer.fit_transform(vectorized_symbols)
                                    
                                    #add the normalized symbols to the dataframe
                                    #add the csr_matrix to the dataframe 
                                    sym_df = pd.concat([sym_df.T, pd.DataFrame(normalized.toarray())], axis=1).T
                                    
                                    print('[+]done ('+str(count)+') ' + str(sym_df.shape[0]), 'rows, ', str(sym_df.shape[1]), 'columns',  "\n" ) 

                            except Exception as e:
                                print('[=>]symbol parsing error',e)

                                
                           
                        except Exception as e:
                            print(e)
                    else :
                        continue             
            elif isinstance(section, NoteSection):
                for note in section.iter_notes():
                    #if note does not have a name attribute skip
                    if not hasattr(note, 'name'):
                        continue
                    print("[+]note section "+section.name, note.name, "\n" )
                    df[section.name+'_'+note.name].join([note.name, str(note.entry['n_namesz']), str(note.entry['n_descsz'])])
            
            else :
                try:

                    print(section.name, section['sh_size'], "\n" )
                    #take the data as a string and vectorize it per token split by \0x000 or \0 
                    df[section.name].join([str(section.data(), 'utf-8')])
                except Exception as e:
                    print(e)
                    continue

        except Exception as e:
            print(e)
            
    #prepare the df for concatenation,normalize the number of columns
    sym_df.fillna(0, inplace=True)
    #transpose the symbols dataframe
    #sym_df = sym_df.T
    #add meta-data to the dataframe 
    df['symbol_count'] = sym_df.shape[0]
    df['symbol_columns'] = sym_df.shape[1]
    df['global_end']=len(global_symbols)
    df['n_sections'] = len(sections)
    df['n_symbols'] = len(global_symbols) - df['global_start']


    
    

    print ("[+]df shape:", df.shape)
    return df


def process_elf(elf):
    ''' Process an ELF file and return a pandas dataframe'''
    df = pd.DataFrame()
    global count 
    
    with open(elf, 'rb') as f:
        count+=1
        out_csv =str(count)+'.csv'
        elffile = ELFFile(f) 
        df = process_elffile(elffile)

        #vectorize the values in the dataframe
        #df = vectorizer.fit(df).transform(df)

        if df is None:
            return df
        #save the dataframe to a csv file 
        df.to_csv( out_csv, index=False)
    return df
 
    

def main(args):
    '''iterate /bin/ and parse elf files ''' 
    print("[+]start")    
    global count, global_symbols, sym_df
    count = 0
    #parse arguments

    if len(args) < 1:
        print("Usage: elf_walker.py <path>")
        sys.exit(0)

    if not os.path.exists(args[0]):
        print("Error: path does not exist")
        sys.exit(0) 

    if not os.path.isdir(args[0]):
        print("Error: path is not a directory")
        sys.exit(0)

    if not os.access(args[0], os.R_OK):
        print("Error: path is not readable")
        sys.exit(0)


    #get the path
    path = args[0]
    if path[-1] != '/':
        path += '/'

    try :
        #iterate /bin/ and parse elf files
        #check for existence of .csv files in the current directory
        #if exists, skip process_elf function

         
        for root, dirs, files in os.walk(path):
            for file in files: #assume all files in /bin are elf files 
            #check if it's an elf32/64 file,validate magic number and process the file 
                print( "[+]processing "+os.path.join(root, file)    )
                #if os.path.exists('./'+str(count)+'.csv'):
                #    continue
                try:
                    process_elf(os.path.join(root, file))
                except Exception as e:
                    print(e)
                    continue
        
        #LOAD all the csv files and vectorize them
        df = pd.DataFrame()
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.csv'):
                    #check if it's symbols.csv, if so, skip 
                    if file == 'symbols.csv' or file == 'processed_symbols.csv':
                        continue
                    #skip 'data.csv'
                    if file == 'data.csv' or file == 'sim_data.csv':
                        continue


                    #load the csv file
                    

                    print( "[+]loading "+os.path.join(root, file)    )
                    df_ = pd.read_csv( os.path.join(root, file) )
                    #normalize the number of columns
                    
                    df_ = df_.fillna(0.0)
                    vectorizer.fit(df_)
                    df = pd.concat([df, df_])
                    df.columns = df.columns.astype(str)
                    #vectorize the values in the dataframe
                    

        try :
            vectorizedf = vectorizer.fit(df).transform(df)
        except Exception as e:
            print(e+"[=>]could not vectorize\n")            
        print ("vectorized df shape:", vectorizedf.shape)
        print ("df shape:", df.shape)
        #get the global symbols
        orig_df = pd.DataFrame(df)

#create a csv file for the symbols
        try:
            print ('[+]creating symbols.csv')
            print ('[+]global symbols:', len(global_symbols))
            
            #compressed = vectorizer.inverse_transform(vectorized)
            #save as file first 
            csv = 'symbols.csv' 

            with open(csv, 'w') as f:
                #write the header
                f.write('#key,name,value,size,type\n')
                for symbol in global_symbols:
                    for key, value in symbol.items():   
                        f.write(key+','+str(value[0])+','+str(value[1])+','+str(value[2])+','+str(value[3])+ '\n') 
            
                f.close()
            
           
        except Exception as e:
            print('symbols could not be created',   e)
        
        #
        print ('[+]creating data.csv')

        #get the tfidf matrix
        tfidf_matrix =  create_tfidf_matrix(vectorizedf)
        #get the latent semantic matrix
        svd_matrix =  get_latent_semantic_matrix(tfidf_matrix)
        #get the cosine similarity matrix
        cosine_similarity_matrix =  calculate_cosine_similarity(svd_matrix)

        #save the matrix as csv
        print ('[+]saving sim_data.csv')
        
        pd.DataFrame(cosine_similarity_matrix).to_csv( 'sim_data.csv', index=False) 

        orig_df.to_csv( 'data.csv', index=False)


        print ("[+] data.csv done")

        #do the same for symbols.csv
        #use DictVectorizer to vectorize all the symbols


        #
        #if file exists, skip tfidf_symbols.csv creation
        #if sym_df is not None and sym_df.shape[0] > 0:
        #    sym_df.to_csv( 'tfidf_symbols.csv', index=False)
        #    print ("[+] tfidf_symbols.csv done")
        #plot the matrix
        # 

        #plot the matrix
        #
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #use sns to plot
         
        #draw kde plot
        #sns.kdeplot(df['0'], df['1'], df['2'], shade=True, cmap="Reds")

        #draw scatter plot
        #
        markers = ('s', 'x', 'o', '^', 'v') 
        colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w') 

        #add contour regions :
        #
        #create meshgrid
        

        #plot the meshgrid
        #
        try:
            #create meshgrid
            #
            print ("[+]plotting surface") 
            
            #create meshgrid
            #
            linearly_spaced_steps = np.linspace(-1, 1, 200)
            xx, yy = np.meshgrid(linearly_spaced_steps, linearly_spaced_steps)

            #plot the meshgrid
            #
             
            Z=xx*yy
 
            
            #plot the meshgrid
            #
            ax.contourf(xx, yy, Z, alpha=0.4)

            print ("[+]surface done") 

            #plot the data points
            #
            #create 3 dimensional meshgrid 
            #use label_encoder to encode the labels on all string columns
            #
            print ("[+]plotting scatter")
            #remove empty columns
            #cosine_similarity_matrix = cosine_similarity_matrix.dropna(axis=1, how='all') 

            #df = pd.DataFrame(tfidf_matrix.todense())
            #transform all the strings in df into values :
            #
            if cosine_similarity_matrix is not None and cosine_similarity_matrix.shape[0] > 0: 
                df = pd.DataFrame(cosine_similarity_matrix)
            else:
                df = pd.DataFrame(vectorizedf).aggregate(lambda x: ' '.join(x.astype(str))) 

            #remove empty columns
            df = df.dropna(axis=1, how='all')

            
            for i in range(0,df.shape[0] ):
                marker = markers[i%len(markers)]
                for j in range(0,df.shape[1] ):
                    color = colors[j%len(colors)]
                    #debug:
                    #
                    
                    #if j>11:
                    #    print ("[+] ignoring "+str(i)+','+str(j))
                    #    continue
                    print ("[+]scatter "+str(i)+','+str(j))
                    try:
                        vectorizedx = df.iloc[i][j]
                        vectorizedy = df.iloc[i][j+1]
                        vectorizedz = df.iloc[i][j+2]
                        ax.scatter(vectorizedx, vectorizedy, vectorizedz, marker=marker, color=color) 

                    except Exception as e:
                        print("[=>]could not plot scatter  [%d,%d]"%(i,j),e )

            sns.despine(ax=ax, offset=10)
            print ("[+]scatter done") 
 
        except Exception as e:
            print('[=>]could not plot meshgrid plot ',e)

 
        ax.set_xlabel('0')
        ax.set_ylabel('1')
        ax.set_zlabel('2')  
 
        #draw LDA plot 
        #
        #save png
        fig.savefig('elf_walk_distribution.png', dpi=300)



        plt.show()


        #draw t-SNE plot of the matrix in df
        #
        try:
            print ("[+]plotting t-sne")
            tsne = TSNE(n_components=3, random_state=0, n_iter=10000, perplexity=df.shape[0]//2) 
            np.set_printoptions(suppress=True)
            #prepare df for t-sne
            #
            df = df.fillna(0)
            df.columns = df.columns.astype('str')
            df = df.astype('float64')
            Y = tsne.fit_transform(df) 
             
            fig = plt.figure()
            ax = fig.add_subplot(111)
            print ("[+]creating subspace\n")
            colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
            markers = ('s', 'x', 'o', '^', 'v')
            linearly_spaced_steps = np.linspace(Y.min().min(), Y.max().max(),2000)
            xx,yy = np.meshgrid(linearly_spaced_steps, linearly_spaced_steps)
            Z=xx*yy/(df.max().max()+df.min().min()*0.5)
            #spread Z over the meshgrid as background fro the tsne

            #plot the meshgrid,expand over the grid space 
            #
            W = np.c_ [xx.ravel(), yy.ravel(), Z.ravel()]
            #multiply by 500 to spread the meshgrid
            #W = W*-500.0
            #expand the meshgrid
            W*=W.sum(axis=1, keepdims=True)
            W = W[W[:, 2].argsort()]
            #apply Y region to the meshgrid
            #
            W = W[(W[:, 0] >= Y[:, 0].min()) & (W[:, 0] <= Y[:, 0].max()) & (W[:, 1] >= Y[:, 1].min()) & (W[:, 1] <= Y[:, 1].max())] 
            print ("[+]subspace displaying background\n")
            #show the background
            ax.imshow(W, cmap='RdBu', interpolation='bilinear', extent=(-1, 1, -1, 1), origin='lower', alpha=0.4, zorder=-1,aspect='auto')
            ax.contourf(xx, yy, Z, alpha=0.4)

            #surface = ax.contourf(X, Y, Z, alpha=0.4)
            #ax.scatter(Y[:, 0], Y[:, 1], c='r', marker='o', alpha=1, edgecolor='k')
            for i in range(len(Y)):
                ax.scatter(Y[i, 0], Y[i, 1], color = colors[i%len(colors)], marker='o', alpha=1) 

            fig.savefig('tsne.png', dpi=300)    


            plt.show()

            
        except Exception as e:
            print("[=>]TSNE error: ",e)
            pass

        #plot the sym_df
        #
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            #use wordcloud to plot the symbols
            sym_df = pd.read_csv('symbols.csv', index_col=1)
            sym_df = sym_df.fillna(0)
            sym_df=sym_df.drop(axis=1, columns=['#key','value','size','type']) 
            wordcloud = WordCloud().generate(sym_df.to_string())


            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')

            fig.savefig('symbols_wordcloud.png', dpi=300)
            plt.show()
        
        except Exception as e:
            print("[=>]symbols wordcloud error: ",e)
            pass
        print("[+]end")
            
    except Exception as e:
        print(e)
        pass

#main function
if __name__ == '__main__':
    main(sys.argv[1:])