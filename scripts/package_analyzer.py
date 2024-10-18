''' package analyzer '''
''' depends on dpkg_enumerator.py and symbolic_categorize_package.py '''
''' usage : python package_analyzer.py '''


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
import pandas as pd
import threading
import multiprocessing
import time
import psutil
import json
#for elf tools
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns

import dpkg_enumerator as dpkg_enumerator

#itertools
import itertools
#text pipeline

from transformers import pipeline, EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score




#create a job for each package parse each ELF file in the package and add to a dataframe


class analyze_job(dpkg_enumerator.job):
    def __init__(self, id):
        self.id = id
        self.result = None
        self.func = None
        self.args = None
        self.mutex = threading.Lock()
        self.threads_finished = 0
        self.threads_error = 0
        self.threads_result = dict()
        self.threads_error = 0
        self.package_name = None
        self.package_object = None
    def set(self, func, args):
        self.func = func
        self.args = args
        return
    def __call__(self):
        self.result = self.func(*self.args)
        return self
    #override job_func:

    def job_func(self, args):
        print(self.args)
        self.result = self.func(*self.args)
        return
    def set_job_func(self, func):
        self.func = func
        return


classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def hello_package(args):
    print(args.package_name + "running on thread:%d\n"%threading.current_thread().ident ) 
    if args.package_object is None:
        return
    args.package_object['package_name'] = args.package_name
    object_string = str(args.package_object).replace('_',' ')

    labels = ['kernel','system','service','utility','graphics','database','network','development','security','library','resources','media','miscellaneous']
    print('classifying '+ str(args.package_object['package_name']),'\n')
    result = classifier(object_string, labels)
    args.result = result

    
    print(result)
 
    print("[+] job for " + args.package_name + " done\n")
    
    return result 

def load_installed_packages():
    with open('installed_packages.json') as json_file:
        installed_package_files = json.load(json_file)
        return installed_package_files
    




def main():
    id =0
    installed_packages = load_installed_packages()
    #break the json into a dict(dict())
    if len(installed_packages) == 0:
        print("no installed packages")
        return 0

    dpkg_enumerator.set_python_options()

    jobs = []
    packages = list(installed_packages.keys())
    for i in range(0,len(installed_packages)):
        j = analyze_job(i)
        j.id = i
        j.args = [j]
        j.func = hello_package
        j.package_name = packages[i]
        j.package_object = installed_packages[packages[i]]
        jobs.append(j)
        
    #use dpkg_enumerator as a job manager

    en = dpkg_enumerator.dpkg_enumerator( len(jobs) )
    en.MAX_THREADS = 500

    for job in jobs:
        en.add_job(job)

    en.enumerate()
    en.wait()
    print(en.threads_result)
    #save the results : 
    df = pd.DataFrame(columns=['package_name','kernel','system','service','utility','graphics','database','network','development','security','library','resources','media','miscellaneous']) 
        
    for job in jobs:
        result = job.result
        package_name = job.package_name
        df[package_name] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        for label,score in zip(result['labels'],result['scores']):
            df[package_name][label] = score

    
    #save the results as csv

    df.to_csv('categorized_packages.csv', index=False)

    print('[+]saved categorized_packages.csv')
  
    return 0

if __name__ == "__main__":
    main()

    
