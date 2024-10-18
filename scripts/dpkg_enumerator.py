
''' dpkg enumerator for linux '''
''' index of all the files associated with the package. '''
''' when crawling the elf, this information can already be associated with the symbols. ''' 



import os
import re
import subprocess
import struct
import sys  
import hashlib
import binascii
import json
import resource
#for building database/dataframe
import pandas as pd
#for plotting
import matplotlib.pyplot as plt
import seaborn as sns 
#threads,multiprocessing,mutex and semaphores
import threading
import multiprocessing
import time
import psutil
import numpy as np
#for word cloud / category plot /tfidf and bag of words
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer



class job:
    def __init__(self, id):
        self.id = id
        self.result = None
        self.func = None
        self.args = None
        self.mutex = threading.Lock()
        self.threads_finished = 0
        self.threads_error = 0
        self.threads_result = [{'id':None,'result':None}]
        self.threads_error = 0
        self.package_name = None

    def set(self, func, args):
        self.func = func
        self.args = args
        return
    
    def __call__(self):
        self.result = self.func(*self.args)
        return

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)



class dpkg_enumerator:
    def __init__(self, threads_count):
        print ("[+]dpkg_enumerator\n")
        self.mutex = threading.Lock() 
        self.threads = []
        self.threads_count = threads_count
        self.threads_processed = 0
        self.threads_running = 0
        self.threads_finished = 0
        self.threads_failed = 0
        self.threads_success = 0
        self.threads_skipped = 0
        self.threads_error = 0
        self.threads_total = 0
        self.threads_result = dict()
        self.jobs = []
        self.MAX_THREADS = 1000

    def add_job(self, job):
        #lock the mutex
        with self.mutex:
            self.jobs.append(job)

    def start(self):
        for job in self.jobs:

            t = threading.Thread(target=job)
            t.start()
            self.threads.append(t)

        for t in self.threads:
            t.join()    

    def get_results(self):
        results = []
        with self.mutex:
            results = self.threads_result
        return results

    def print_results(self):
        with self.mutex:
            for result in self.threads_result:
                print(result['id'],result['result'])


    def print_stats(self):
        with self.mutex:
            print("threads_processed:", self.threads_processed)
            print("threads_running:", self.threads_running)
            print("threads_finished:", self.threads_finished)
            print("threads_failed:", self.threads_failed)
            print("threads_success:", self.threads_success)
            print("threads_skipped:", self.threads_skipped)
            print("threads_error:", self.threads_error)
            print("threads_total:", self.threads_total)
            print("threads_result:", self.threads_result)


    def run(self):
        threadcount =0
        for job in self.jobs:
            
            #update stats

            with self.mutex:
                self.threads_processed += 1
                self.threads_total += 1
                self.threads_running += 1
            t = threading.Thread(target=job)
            t.start()
            self.threads.append(t)
            threadcount=threadcount+1
            if threadcount > self.MAX_THREADS:
                self.wait()
                threadcount=0


        
    def wait(self):
        for t in self.threads:
            t.join()
            with self.mutex:
                self.threads_running -= 1
                self.threads_finished += 1


    def __del__(self):
        self.wait()

    def cancel(self):
        for t in self.threads:
            t.cancel()

    def install_signals(self):
        pass

    def uninstall_signals(self):
        pass

    def enumerate(self):
        self.run()
        self.wait()
        self.print_results()
        self.print_stats()

    def reset(self):
        with self.mutex:
            #kill all the threads
            for t in self.threads:
                t.cancel()
            #reset the stats

        self.oldthreads = self.threads    
        self.threads = []
        self.oldthreadcount = self.threads_count
        self.oldthreadsprocessed = self.threads_processed
        self.oldthreadsrunning = self.threads_running
        self.oldthreadsfinished = self.threads_finished
        self.oldthreadsfailed = self.threads_failed
        self.oldthreadssuccess = self.threads_success
        self.oldthreadsresult = self.threads_result
        self.oldthreadserror = self.threads_error
        self.oldthreadstotal = self.threads_total
        self.oldthreadjobs = self.jobs

        self.threads_count = 0
        self.threads_processed = 0
        self.threads_running = 0
        self.threads_finished = 0
        self.threads_failed = 0
        self.threads_success = 0
        self.threads_skipped = 0
        self.threads_error = 0
        self.threads_total = 0
        self.threads_result = dict()
        self.jobs = []      

    def enumerate_all(self):
        self.reset()
        #self.threads_count = 100
        self.enumerate()

    def enumerate_one(self):
        self.reset()
        #self.threads_count = 1
        self.enumerate()
    
    def enumerate(self):
        self.run()
        self.wait()
        self.print_results()
        self.print_stats()


global_mutex = threading.Lock()
global_results = dict()

def job_func(i):
    print ("[+]job_func running on thread:%d\n"%threading.current_thread().ident )
    #get the package name from the job
    package_name =i# i.package_name
    results = get_package_files(package_name)
    if results is None:
        return
    print ("[+]got %d results" % len(results))
    #set the result
    #with job_manager.mutex:
     #   job_manager.threads_result[i] = results
    with global_mutex:
        global_results[i] = results 
    print ("[+]job_func done\n")
    return

def add_package_files_information(package_name):
    print ("[+]adding package files information...\n")
    package_files = dict()
    
    return

def read_dpkg_installed_packages():
    print ("[+]reading dpkg installed packages...\n")
    installed_packages = dict().fromkeys('Package', dict())
    with open('/var/lib/dpkg/status', 'r') as f:
        package_name = None
        last_key = None

        #package_collection = dict().fromkeys('Package', {'key':None,'value':None}) 
        for line in f:
            print(line)
            #parse the line
            try : 
                 
                 firstoff = line.find(':')
                 key = line[0:firstoff].strip()
                 value = line[firstoff+1:].strip()
                 if key is not None:
                     key = key.replace(' ','_')
                     key.encode('utf-8')

                 if value is not None:
                     value.encode('utf-8')


                 
                 print ("[+]key %s value %s\n" % (key,value))


                 if len(key)<=1 or len(value)<=1 or key == '' or value == '':
                     continue
                 
                 if key == 'Package':
                         if value is not None:
                             package_name = value

                         continue
                 
                 if package_name not in installed_packages:
                     installed_packages[package_name] = dict()

                 installed_packages[package_name][key] = value
            
            except Exception as e:
                print("Error: read_dpkg_installed_packages ", e)
                continue
    
    print ("[+]read dpkg installed packages done\n")
    for package in installed_packages:
        for key in installed_packages[package]:
            print("[+] [%s]::[%s]: %s" % (package, key, installed_packages[package][key]))
            

   
    return installed_packages


def get_package_files(package_name):
    print ("[+]get_package_files %s\n" % package_name)
    key_val = dict()
    #get list of packages
    try:
        paths = subprocess.check_output('apt-file list ' + package_name, shell=True) 
        paths = paths.decode('utf-8')
        paths = paths.split('\n')

    except subprocess.CalledProcessError as e:
        print("Error: get_package_files ", e)
        #check if the error is because there are too many opened files
        if 'too many open files' in str(e):
            #retry 
            os.sleep(1)
            paths = subprocess.check_output('apt-file list ' + package_name, shell=True) 
            paths = paths.decode('utf-8')
            paths = paths.split('\n')

        else:
            return key_val#empty

    for i in range(0,len(paths)):
        if ':' not in paths[i]:
            continue
        key_val[paths[i].split(':')[0].strip().replace(' ','_')]  = paths[i].split(':')[1].strip()
    print ("[+]get_package_files %s done %d files\n" %( package_name, len(key_val))) 
    
    return key_val


p_ =None

def main():
    print ("[+]dpkg_enumerator\n")
    installed_packages = read_dpkg_installed_packages()
    
    print ("[+]total packages :", len(installed_packages))

    total,installed_total=0,0
    jobs = []
    for package in installed_packages:
        
        if 'Status' in installed_packages[package].keys() and installed_packages[package]['Status'] is not None:
            total = total + 1
            if 'installed' in installed_packages[package]['Status']:
                installed_total = installed_total + 1                
                #installed_packages[package]['Files'] = get_package_files(package)
                #add job:
                jobs.append(package)


    print ("[+]total size :%d,installed:%d" % (total,installed_total))

    print ("[+]jobs :%d" % len(jobs))

    #save installed_packages to json
    with open('installed_packages.json', 'w') as outfile:
        json.dump(installed_packages, outfile)


    en = dpkg_enumerator(len(jobs))
    package_names = list(installed_packages.keys())
    #create jobs
    for i in range(0,len(jobs)):
        j = job(jobs[i])
        j.id = i
        j.args = [jobs[i]]
        j.package_name = package_names[i]
        j.func = job_func
        en.add_job(j)
        
    
    
    en.enumerate() 
    en.wait()

    en.print_results()
    en.print_stats()

    #save global results files
    with open('package_installed_files.json', 'w') as outfile:
        json.dump(global_results, outfile)

    print ("[+]done\n")

    dataframe = pd.DataFrame.from_dict(global_results, orient='index')
    dataframe.to_csv('package_installed_files.csv', index=False)


    #show a word cloud of the descriptions from each package
    #create word cloud

    #exit(0)
    #show a word cloud of the descriptions from each package
    #        

def draw_word_cloud():
    print ("[+]draw_word_cloud\n")
    global p_
    if p_ is None:
        #read json
        with open('installed_packages.json') as json_file:
            p_ = json.load(json_file)

    print ("[+]total packages :", len(p_))

    
    descriptions = []
    for package in p_:
        if 'Description' in p_[package].keys() and p_[package]['Description'] is not None:
            print ("[+] %s" % p_[package]['Description'])
            descriptions.append(p_[package]['Description'])
    
    print ("[+]total descriptions :", len(descriptions))
    # Create and generate a word cloud image:
    wordcloud = WordCloud( background_color='black', height=2000, width=1000).generate( ' '.join(descriptions) )


    #save the image
    wordcloud.to_file("packages_word_cloud.png") 

    #aggregate the p_ values
    text = '' 
    for package in p_:
        for key in p_[package].keys():
            text = text + p_[package][key] + ' '
    print ("[+]total text :", len(text))

    wordcloud = WordCloud( background_color='black', height=2000, width=1000).generate( text ) 

    #save the image
    wordcloud.to_file("packages_word_cloud_all.png")
    print ("[+]done\n")



def set_python_options():
    print ("[+]set_python_options\n")
    #set the max recursion depth
    sys.setrecursionlimit(10**6)
    #set the max threads
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, 100000))
    resource.setrlimit(resource.RLIMIT_NPROC, (100000, 100000))
    #for IPC
    resource.setrlimit(resource.RLIMIT_MSGQUEUE, (100000, 100000))
    print ("[+]set_python_options done\n")


if __name__ == "__main__":
    
    set_python_options()
    main() 
    draw_word_cloud()
    print ("[+]draw_word_cloud done\n")
    exit(0)

