

''' analyze categories of the files according to the package association and drill down granularity. ''' 

import pandas as pd
import os
import re
import subprocess
import sys
import hashlib
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection, NoteSection
from elftools.elf.constants import SH_FLAGS
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def get_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()  
    



def main():
    #read the json file with the files 
    df = pd.read_json('package_installed_files.json')

    #read the csv file
    classified_packages  = pd.read_csv('symbolic_categorize_packages.csv')


            
    #merge the two dataframes
    #df = df.merge(classified_packages, on='package', how='left') 

    #write the dataframe to a csv file
    #classified_packages.columns = ['package','kernel','system','service','utility','graphics','database','network','development','security','library','resources','media','miscellaneous'] 

    for package in df['package']:
        print(package)

    
    package_name =[]
    labels =[]
    scores =[]

    
    #classified_packages.to_csv('normalized.csv') 
    pass


if __name__ == "__main__":
    main()