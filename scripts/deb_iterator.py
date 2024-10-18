''' debian package iterator '''
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

#for save queue
import queue 

global mutex

mutex = threading.Lock()
semaphore = threading.Semaphore(0)


'''job manager class for parallel processing'''
class JobManager:
    def __init__(self, jobs):
        self.jobs = jobs
        self.threads = []
        self.results = []

    def add_job(self, job):
        self.jobs.append(job)   

    def start(self):
        for job in self.jobs:
            t = threading.Thread(target=job)
            t.start()
            self.threads.append(t)
        for t in self.threads:
            t.join()    

    def get_results(self):
        return self.results
    
    def print_results(self):
        for result in self.results:
            print(result)
    
    def wait(self):
        for t in self.threads:
            t.join()    
    
    def __del__(self):
        self.wait() 

    def __str__(self):
        return str(self.results)    
    
    def __repr__(self):
        return str(self.results)    
    

class safe_dict(dict):
    '''safe dictionary for multithreading'''
    def __missing__(self, key):
        return None
    
    #override dictionary functions,use mutex to prevent race conditions
    
    def __init__(self, *args, **kwargs):
        #initialize mutex
        mutex = threading.Lock()
        with mutex:
            dict.__init__(self, *args, **kwargs)    

    def copy(self):
        with mutex:
            return dict.copy(self)

    def clear(self):
        with mutex:
            return dict.clear(self)

    def __getitem__(self, key):
        with mutex:
            return dict.__getitem__(self, key)

    def get(self, key, default=None):
        with mutex:
            return dict.get(self, key, default)

    def pop(self, key, default=None):
        with mutex:
            return dict.pop(self, key, default)

    def setdefault(self, key, default=None):
        with mutex:
            return dict.setdefault(self, key, default)

    def update(self, *args, **kwargs):
        with mutex:
            return dict.update(self, *args, **kwargs)

    def popitem(self):
        with mutex:
            return dict.popitem(self)

    def keys(self):
        with mutex:
            return dict.keys(self)

    def values(self):
        with mutex:
            return dict.values(self)

    def items(self):
        with mutex:
            return dict.items(self)

    def has_key(self, key):
        with mutex:
            return dict.has_key(self, key)

    def __cmp__(self, dict_):
        with mutex:
            return dict.__cmp__(self, dict_)

    def __eq__(self, dict_):
        with mutex:
            return dict.__eq__(self, dict_)

    def __ne__(self, dict_):
        with mutex:
            return dict.__ne__(self, dict_) 

    def __repr__(self):
        with mutex:
            return dict.__repr__(self)

    

    def __setitem__(self, key, value):
        with mutex:
            dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        with mutex:
            dict.__delitem__(self, key)

    def __contains__(self, key):
        with mutex:
            return dict.__contains__(self, key) 

    def __iter__(self):
        with mutex:
            return dict.__iter__(self)

    def __len__(self):
        with mutex:
            return dict.__len__(self)

    def __repr__(self):
        with mutex:
            return dict.__repr__(self)

    def __str__(self):
        with mutex:
            return dict.__str__(self)





''' parse and output the list of packages similar to using apt-cache pkgnames '''

def apt_cache_pkgnames():
    ''' parse and output the list of packages similar to using apt-cache pkgnames '''
    #get list of packages
    try:
        packages = subprocess.check_output('apt-cache pkgnames', shell=True) 
        packages = packages.decode('utf-8')
        packages = packages.split('\n') 
        #remove empty entries and packages that are not installed
        packages = list(filter(None, packages))
        return packages
    except subprocess.CalledProcessError:
        print("Error: apt-cache pkgnames")
        sys.exit(0)


def apt_file_show(package):
    ''' parse and output the list of packages similar to using apt-file show ''' 
    #get list of packages
    try:
        paths = subprocess.check_output('apt-file show ' + package, shell=True) 
        paths = paths.decode('utf-8')
        paths = paths.split('\n')
        return paths

    except subprocess.CalledProcessError:
        return None
    
def get_package_information(package):
    ''' parse and output the list of packages similar to using apt-file show ''' 
    #get list of packages
    paths = subprocess.check_output('apt-cache show ' + package, shell=True) 
    paths = paths.decode('utf-8')
    paths = paths.split('\n')
    return paths


def get_app_armor_information(file):
    '''  search for the file in the appArmor database '''

    result = subprocess.run(['apparmor_status', '--json'], stdout=subprocess.PIPE)
    data = result.stdout.decode('utf-8')
    aa_status  = json.loads(data)
    if file in aa_status:
        print(aa_status[file])
    else:
        print("file not found")

    return aa_status
    

def main(args):
    '''iterate all packages and parse elf files ''' 
    print("[+]start")
    dataframe = pd.DataFrame()
    packages = apt_cache_pkgnames()    
    
    coll = safe_dict.fromkeys(packages, {'info': None, 'files': {}, 'apparmor': None})
 
    for package in packages:
        print ("[+]processing package :", package)
        files = apt_file_show(package)
        if files is None:
            continue
        package_information = get_package_information(package) 
        coll[package]['info'] = package_information
        #coll[package]['files'] = files
        #run asynchronously
        
        for file in files:
            #print ("[+]additional information for file :", file)
            #get se_linux_version_from_file(file)
            #get file type
            #get file permissions
            #get file owner
            #get file group
            #get file size
            #get file modification time
            #filename is the suffix, description is the prefix
            #split file to '%s:%s' % (description, filename) 

            if ':' not in file:
                file_name = file
                description = ''
            else:
                file_name = file.split(':')[1]
                #trim any leading or trailing spaces
                file_name = file_name.strip()

                description = file.split(':')[0]
                
            if len(file_name) == 0:
                continue
            
            if os.path.exists(file_name) == False:
                continue

            try:

                stat = os.stat(file_name)
            #print(stat)
                
                print("[+]file_name: " + file_name, "[description: " + description + "]",'\n')

                #print(stat)
                #only add files we can stat

                coll[package]['files'][file_name] = stat

                os_access = stat.st_mode
                
                owner = stat.st_uid
                group = stat.st_gid
                size = stat.st_size
                modification_time = stat.st_mtime
                inode =stat.st_ino
                

                print('[+]os_access:', str(os_access),'\n')
                print('[+]owner:', str(owner),'\n')
                print('[+]group:', str(group),'\n')
                print('[+]size:', str(size),    '\n')
                print('[+]modification_time:', modification_time,'\n')
                print('[+]inode:', str(inode),'\n')
                


                
                #get apparmor information

            except FileNotFoundError:
                print("[==>]Error: file does not exist" + file_name)
                continue
                #sys.exit(0)
            except PermissionError:
                print("[==>] permission denied" + file_name)
                continue
                #sys.exit(0)
            except Exception as e:
                print("[==>] unknown error" + file_name, e)
                continue
                #sys.exit(0) 

    print('[+]collected all the information.\n')

    dataframe = pd.DataFrame.from_dict(coll, orient='index')
    dataframe.to_csv('packages.csv')


    print('[+]done dumping packages.csv\n')


if __name__ == "__main__":
    main(sys.argv[1:])
