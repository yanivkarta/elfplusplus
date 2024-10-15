
#include <elf.h>
#include <fcntl.h>
#include <unistd.h>

#include <filesystem>

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <fstream>
//mmap
#include <sys/mman.h>
#include <sys/stat.h>
//demange cpp function names
#include <cxxabi.h>
//disassemble instructions from elf file
#include <x86_64-linux-gnu/sys/ptrace.h>
#include <x86_64-linux-gnu/sys/user.h>
#include <x86_64-linux-gnu/sys/reg.h> 
#include <x86_64-linux-gnu/sys/wait.h>
#include <x86_64-linux-gnu/bits/pthreadtypes.h>
#include <x86_64-linux-gnu/bits/thread-shared-types.h>
#include <x86_64-linux-gnu/bits/pthreadtypes-arch.h>
#include <x86_64-linux-gnu/bits/stat.h>
#include <x86_64-linux-gnu/bits/types/struct_sigstack.h>
//filesystem:
#include <filesystem>
//for demangle cpp function names
#include <cxxabi.h>


#include "autoencoder.h"
#include "lstm.h"
#include "matrix.h"
#include "info_helper.h"
#include "optimizers.h"
#include "utils.h"
#include "sampling_helper.h"
#include "elfplusplus.h"


std::map<std::string, std::vector<std::string>> file_symbols_map; //key: filename, value: (features, string table) 




//same like elf_walk, iterate folders and dump features, use the features to train on different types of elf files 
//

std::vector<std::string> list_files(const std::string& path) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_directory()) {
                
                //make sure it's not a symlink
                if (std::filesystem::is_symlink(entry.path())) {
                    continue;
                }else if (!std::filesystem::is_directory(entry.path())) {
                    continue;
                }
                try {
                    auto listed = list_files(entry.path().string());
                    if (!listed.empty()) {
                        files.insert(files.end(), listed.begin(), listed.end());
                    }

                }
                catch (const std::filesystem::filesystem_error& ex) {
                    std::cerr << ex.what() << std::endl;
                    continue;
                } 
                catch (const std::length_error& ex) {
                    std::cerr << ex.what() << std::endl; 
                    continue;
                }
                
            }
            else if (entry.is_regular_file())
            {
                files.push_back(entry.path().string());
            }
            else {
                //ignore symlinks
                continue;
            }   
    }
    return files;
}


provallo::matrix<real_t> extract_elf_features(const std::string& elf_file) {

    //open elf file
    provallo::matrix<real_t> ret(0, 0);
    int fd = open(elf_file.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cout << "Cannot open file: " << elf_file << std::endl;
        return provallo::matrix<real_t>(0, 0);
    }

    //stat size
    struct stat st; 
    if (fstat(fd, &st) == -1) {
        std::cout << "Cannot stat file: " << elf_file << std::endl;
        return provallo::matrix<real_t>(0, 0);
    }

    //mmap
    void* addr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        std::cout << "Cannot mmap file: " << elf_file << std::endl;
        return provallo::matrix<real_t>(0, 0);
    }   
    //extract features
    try 
    {
        
        Elf64_Ehdr* ehdr = (Elf64_Ehdr*)addr;

        if (ehdr->e_ident[EI_MAG0] != 0x7f
            || ehdr->e_ident[EI_MAG1] != 'E'
            || ehdr->e_ident[EI_MAG2] != 'L'
            || ehdr->e_ident[EI_MAG3] != 'F') 
        {
            std::cout << "Not an ELF file: " << elf_file << std::endl;
            munmap(addr, st.st_size);
            close(fd);
            return provallo::matrix<real_t>(0, 0);
        }
        elfpp::ElfFile64 elf(ehdr);
        ret = elf.features();
        auto names = elf.get_section_names();
        std::cout <<"[+]"<<elf_file<< "::Names:(" << std::to_string(names.size()) << ")"  << std::endl; 
        std::cout <<"[+]"<<elf_file<< "::Features:(" << std::to_string(ret.size1()) <<","<< std::to_string(ret.size2()) << ")"  << std::endl;
        
        //add to map

        file_symbols_map.insert(std::pair<std::string, std::vector<std::string>>(elf_file, names)); 

        real_t ratio =ret.size1()*ret.size2() / names.size();
        size_t sample_size = ratio*names.size();

        std::cout <<"[+]"<<elf_file<< "::Sample size:(" << std::to_string(sample_size) << ")"  << std::endl; 

        std::cout <<"[+]"<<elf_file<< "::Ratio:(" << std::to_string(ratio) << ")"  << std::endl;

        //set purple text

        size_t sample_index = 0;

        for (auto& name : names) {

            if (name.length() > 0) {
                
                //print name
                std::cout << "[+]\033[35m" << elf_file << "::" << name << std::endl; 
                //print sample from the matrix corresponding to the name 
                //
                //print the portion of the matrix corresponding to the name 
                //using the sample_size, sample_index,ratio to select the portion 
                //
                //matrix data in orange italic
                std::cout << "\033[3m\033[38;5;208m" << std::endl;

                for (size_t i=sample_index*sample_size; i<sample_index*sample_size+sample_size&&i<ret.size1() ; i++) { 
                     std::cout<<"[+] rowsum["<<i<<"]"<<ret.row_sum(i)<<std::endl; 
                     

                }

                std::cout << "\033[0m" << std::endl;

            }
            sample_index++;
        }
        munmap(addr, st.st_size);
        close(fd);
    }
    catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        munmap(addr, st.st_size);
        close(fd);
    }


        
    
    return ret;
}
//add to the training set: 
//

provallo::matrix<real_t> global_training(0, 0);
provallo::matrix<real_t> global_test(0, 0);


void add_dataset(std::string file, provallo::matrix<real_t>& training,provallo::matrix<real_t>& global) {
    

    
    if (training.size1() == 0 || training.size2() == 0) {
        std::cerr << "add_training: training set is empty" << std::endl;
        return;
    }
    if(global.size1() == 0 || global.size2() == 0) {
        global = training;
        return;
    }

    if (training.cols() > global.cols()) {
        auto tmp = global; 
        auto old_cols = global.size2();
        global.resize(tmp.size1()+training.size1(), training.size2()); 
        //first copy tmp to global_training, then add training rows 
        for (size_t i = 0; i < tmp.size1(); i++) {
            for (size_t j = 0; j < old_cols; j++) {
                global(i, j) = tmp(i, j);
            }
            for (size_t j = old_cols; j < global.size2(); j++) {
                global(i, j) = 0.0;
            }
        }
        //append training data rows
        for (size_t i = 0; i < training.size1(); i++) {
            for (size_t j = 0; j < training.size2(); j++) {
                global(i+tmp.size1(), j) = training(i, j);
            } 

        } //done

    }//if
    else if (training.cols() == global.cols()) {
              //if it's equal to the global_training just add rows 
        auto tmp = global ;
        global.resize(tmp.size1()+training.size1(), training.size2()); 

        for (size_t i = 0; i < tmp.size1(); i++) {
            for (size_t j=0; j < tmp.size2(); j++) {
                global(i, j) = tmp(i, j);
            }
        }
        for (size_t i = 0; i < training.size1(); i++) {
            for (size_t j = 0; j < training.size2(); j++) {
                global(i+tmp.size1(), j) = training(i, j);
            }

        }

    }
    else {
        //add columns to the training set 
        auto tmp = training;
        auto old_cols = training.size2();
        training.resize(training.size1(), old_cols+global.size2()); 
        //first copy tmp to training, then add global_training cols 
        for (size_t i = 0; i < training.size1(); i++) {
            for (size_t j = 0; j < old_cols; j++) {
                training(i, j) = tmp(i, j);
            }
            for (size_t j = old_cols; j < training.size2(); j++) {
                training(i, j) = 0.0;
            }
        }
        //add rows to global_training
        auto tmp2 = global;
        global.resize(global.size1()+training.size1(), global.size2()); 
        //copy tmp2 to global_training
        for (size_t i = 0; i < tmp2.size1(); i++) {
            for (size_t j = 0; j < tmp2.size2(); j++) {
                global(i, j) = tmp2(i, j);
            }
        }
        //append training data rows
        for (size_t i = 0; i < training.size1(); i++) {
            for (size_t j = 0; j < training.size2(); j++) {
                global(i+tmp2.size1(), j) = training(i, j);
            }   
        }
        //done

        
        
    }    
}
void add_training(std::string file, provallo::matrix<real_t>& training) {

    add_dataset(file, training, global_training);
}
void add_testing(std::string file, provallo::matrix<real_t>& testing) {

    add_dataset(file, testing, global_test);
}
//main function, gets target folder for training/testing and test folder for validation 
//
int main(int argc, char **argv) {

    std::cout << "ELFFeatureExtractor++" << std::endl;
    //get the target folder for training/testing
    if (argc < 2) {
        //set red
        std::cout << "\033[1;31m[!]target folder not set\033[0m" << std::endl; 
        
        std::cout << "Usage: elf_feature_extractor <target folder> [test folder]" << std::endl;
        //reset
        std::cout << "\033[0m" << std::endl;
        return 0;
    }
    else if (argc < 3) {
        //set organe
        std::cout << "\033[1;33m[!]test folder not set\033[0m" << std::endl; 

        std::cout << "Usage: elf_feature_extractor <target folder> [test folder]" << std::endl; 
        //reset
        std::cout << "\033[0m" << std::endl;
        return 0;
    }
    std::string target_folder = argv[1];
    std::string test_folder = argv[2];

    //check if folders exist
    if (!std::filesystem::exists(target_folder) || !std::filesystem::is_directory(target_folder)) { 
        if (!std::filesystem::exists(test_folder))
        {
            //set red
            std::cout << "\033[1;31m[!]test folder not found\033[0m" << std::endl; 
            
            std::cout << "Usage: elf_feature_extractor <target folder> [test folder]" << std::endl;
            //reset
            std::cout << "\033[0m" << std::endl;
            return 0;
        }

        //set yellow
        std::cout << "\033[1;33m[!]target folder not found\033[0m" << std::endl;

        std::cout << "Usage: elf_feature_extractor <target folder> [test folder]" << std::endl;
        //reset
        std::cout << "\033[0m" << std::endl;
        return 0;

    }
    
    std::vector<std::string> target_files = list_files(target_folder);
    std::vector<std::string> test_files = list_files(test_folder);

    std::cout << "[+]target files: " << target_files.size() << std::endl;
    std::cout << "[+]test files: " << test_files.size() << std::endl;

    //extract features for each elf
    for (const auto& file : target_files) {
         //extract features for each elf
        //set green
        std::cout << "\033[1;32m[+]extracting features for: " << file << "\033[0m" << std::endl; 
        //reset
        
        auto ret = extract_elf_features(file);
        if(ret.size1()>0 && ret.size2()>0)
        {
         add_training(file, ret);
         std::cout <<"[+] global training matrix size: " << global_training.size1() << "," << global_training.size2() << std::endl; 
         


        }
        else {
            //set red
            std::cout << "\033[1;31m[!]failed to extract features for: " << file << "\033[0m" << std::endl; 
            
        }
        std::cout << "\033[0m" << std::endl;
    }   

    
    //set green
    std::cout << "\033[1;32m[+]training global matrix size: " << global_training.size1() << "," << global_training.size2() << "\033[0m" << std::endl; 

    //reset
    std::cout << "\033[0m" << std::endl;
    //train LSTM
    provallo::LSTM<real_t> lstm(global_training.size2(), global_training.size2(), global_training.size2()); 
    lstm.fit(global_training, global_training);
    real_t loss = lstm.get_loss();
    std::cout << "[+]training loss: " << loss << std::endl;
    std::cout << "[+]training complete" << std::endl;



    char x = getchar();

    for (const auto& file : test_files) {
        std::cout << "[+]extracting features for: " << file << std::endl;
        //extract features for each elf
        auto ret = extract_elf_features(file);
        if(ret.size1()>0 && ret.size2()>0)
        {
         add_testing(file, ret);
         std::cout <<"[+] global test matrix size: " << global_test.size1() << "," << global_test.size2() << std::endl; 
        }
        else {
            //set red
            std::cout << "\033[1;31m[!]failed to extract features for: " << file << "\033[0m" << std::endl; 
            
        }   

    }

    //set green
    std::cout << "\033[1;32m[+]test global matrix size: " << global_test.size1() << "," << global_test.size2() << "\033[0m" << std::endl; 

    auto ret = lstm.predict(global_test);

    //set green
    std::cout << "\033[1;32m[+]predicted matrix size: " << ret.size1() << "," << ret.size2() << "\033[0m" << std::endl; 

    //predictions vs. ground truth 
    std::cout << "[+]ground truth vs. predictions" << std::endl;
    auto rows = std::min(ret.size1(), global_test.size1());
    auto cols = std::min(ret.size2(), global_test.size2());

    //set green
    //print loss after testing
    std::cout << "\033[1;32m[+]loss: " << lstm.get_loss() << "\033[0m" << std::endl; 
    x = getchar();
    for (size_t i = 0; i <rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout <<  global_test(i, j) << " " << ret(i, j) << " ";  
        }
        std::cout << std::endl;
    }
    //set green
    std::cout << "\033[1;32m[+]done\033[0m" << std::endl; 
    //reset
    std::cout << "\033[0m" << std::endl;
    return 0;
}