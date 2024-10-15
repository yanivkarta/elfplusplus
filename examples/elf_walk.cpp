
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

//for CFG - control flow graph
#include <x86_64-linux-gnu/bits/pthreadtypes.h>
#include <x86_64-linux-gnu/bits/thread-shared-types.h>
#include <x86_64-linux-gnu/bits/pthreadtypes-arch.h>



/* 
   @author: Yaniv Karta
   @description: here we use elf.h structures, without our fast elf library first
   to benchmark the time it takes to load an elf file,parse the symbols and map the memory itself 
   to extract features and store them in a dataset for training or testing the model 
   elf++ library will reduce the amount of required code to extract features from an elf file 
   this code is taken as an example of parsing an elf file without our elf++ library 


*/

//demangle using c++ abi

inline std::string demangle(const char* name) {
  int status = -1;
  std::unique_ptr<char, void(*)(void*)> res {
    abi::__cxa_demangle(name, NULL, NULL, &status),
    std::free
  };
  return (status==0) ? res.get() : name ;
  
}



inline std::vector<std::string> list_files(const std::string& folder) {
    std::vector<std::string> files;
    //check if folder exists:
    if (!std::filesystem::exists(folder)) {
        std::cout << "Folder does not exist" << std::endl;
        return files;

    }

    //list files
    try { 
        std::filesystem::directory_iterator it(folder);

        for (const auto& entry : it) {
        //check if it is a file:
        if (std::filesystem::is_regular_file(entry)) {
            files.push_back(entry.path().string());
        }
        else if (std::filesystem::is_directory(entry)) {
            std::vector<std::string> sub_files = list_files(entry.path().string());
            files.insert(files.end(), sub_files.begin(), sub_files.end());
        }
        else if (std::filesystem::is_symlink(entry)) {
            std::cout << "Skipping symbolic link: " << entry.path().string() << std::endl;
        }
        

        }

    }
    catch (const std::filesystem::filesystem_error& ex) {
        std::cout << ex.what() << std::endl;
        return files;
    }

    
    return files;
}
 

int main(int argc, char *argv[]) {

    std::cout << "ELFWalk++" << std::endl;

    if (argc < 2) {
        std::cout << "Usage: elf_walk <folder>" << std::endl;
        return 1;
    }


    //open elf file and walk 
    //use elf.h structures, without the library first: 
    
    auto directory = argv[1];

    std::vector<std::string> files = list_files(directory);
    std::vector<int64_t> durations = {};
    //extracted features from each elf:
    std::map<std::string, std::vector<uint64_t>> discrete_features = {}; 
    std::map<std::string, std::vector<double>> continuous_features = {}; 
    std::map<std::string,std::vector<std::string>> string_features = {}; 
    //add tag sparse features here:
    std::map<std::string, std::vector<uint64_t>> sparse_features = {}; 

    size_t total_elfs =0;
    
    for (const auto& file : files) {
        //open elf file
        auto start = std::chrono::high_resolution_clock::now(); 
    
        std::cout << "[+]Processing: " << file << std::endl;
        //get fstat64

        struct stat64 st;

        int ret = stat64(file.c_str(), &st);

        if (ret < 0) {
            std::cout << "[-]Could not stat file: " << file << std::endl;
            continue;
        }
        else {
            //transform st_time to time_t
            time_t t = st.st_mtime, t2 = st.st_atime, t3 = st.st_ctime; 

            //get ctime of st_time
            std::string modification = std::ctime(&t);
            std::string access = std::ctime(&t2);
            std::string change = std::ctime(&t3);

            std::cout << "[+]File Size: " << st.st_size << " bytes" << std::endl; 
            std::cout << "[+]File Type: 0x" << std::oct << st.st_mode << std::dec << std::endl;
            std::cout << "[+]File Permissions: 0" << std::oct << (st.st_mode & 0777)   << std::dec << std::endl; 
            std::cout << "[+]File UID: " << st.st_uid << std::endl;
            std::cout << "[+]File GID: " << st.st_gid << std::endl;
            std::cout << "[+]File Size: " << st.st_size << " bytes" << std::endl; 
            std::cout << "[+]File Mode: " << st.st_mode << std::endl;
            std::cout << "[+]File Modification Time: "<< modification << std::endl;
            std::cout << "[+]File Access Time: " << access << std::endl;
            std::cout << "[+]File Change Time: " << change << std::endl;
            std::cout << "[+]File Inode: 0x" <<std::hex << st.st_ino  << std::endl;
            std::cout << "[+]File Hard Links: " << st.st_nlink << std::endl;
            std::cout << "[+]File Link: " << st.st_nlink << std::endl;
            std::cout << "[+]File Major: " << st.st_dev << std::endl;
            std::cout << "[+]File Minor: " << st.st_rdev << std::endl;
            std::cout << "[+]File Size: " << st.st_size << " bytes" << std::endl; 

            if (discrete_features.find(file) == discrete_features.end()) { 
                discrete_features.insert({file, {(uint64_t)st.st_size, st.st_mode, st.st_uid, st.st_gid, st.st_mode, st.st_nlink, st.st_dev, st.st_rdev}}); 

            }
            else {
                discrete_features[file].push_back(st.st_size);
                discrete_features[file].push_back(st.st_mode);
                discrete_features[file].push_back(st.st_uid);
                discrete_features[file].push_back(st.st_gid);
                discrete_features[file].push_back(st.st_mode);
                discrete_features[file].push_back(st.st_nlink);
                discrete_features[file].push_back(st.st_dev);
                discrete_features[file].push_back(st.st_rdev);

            }


        }


        
        int fd = open(file.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cout << "[-]Could not open file: " << file << std::endl;
            continue;
        }   

        //read elf  header
        Elf64_Ehdr* elf = NULL;
        
        //memory map file  to size of file . 
        elf = (Elf64_Ehdr*)mmap(&elf,st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        //check if it's a valid elf file

        if (elf == MAP_FAILED) {
            std::cout << "[-]Could not mmap file: " << file << std::endl;
            close(fd);
            continue;
        }
        //check elf header values if it's a valid elf file
        if (elf->e_ident[EI_MAG0] != 0x7f || elf->e_ident[EI_MAG1] != 'E' || elf->e_ident[EI_MAG2] != 'L' || elf->e_ident[EI_MAG3] != 'F') {
            std::cout << "[-]Invalid elf file: " << file << std::endl;
            close(fd);
            continue;
        }
        else if (elf->e_ident[EI_CLASS] != ELFCLASS64) {
            std::cout << "[-]Invalid elf file: " << file << std::endl;
            close(fd);
            continue;
        }
        else if (elf->e_ident[EI_DATA] != ELFDATA2LSB) {
            std::cout << "[-]Invalid elf file: " << file << std::endl;
            close(fd);
            continue;
        }    
        std::cout<< "\033[36m";

        std::cout<< "File: " << file << std::endl;
        std::cout<< "Magic: " << elf->e_ident[EI_MAG0] << elf->e_ident[EI_MAG1] << elf->e_ident[EI_MAG2] << elf->e_ident[EI_MAG3] << std::endl;
        std::cout<< "Class: " << elf->e_ident[EI_CLASS] << std::endl;
        std::cout<< "Data: " << elf->e_ident[EI_DATA] << std::endl;
        std::cout<< "Version: " << elf->e_ident[EI_VERSION] << std::endl;
        std::cout<< "OS/ABI: " << elf->e_ident[EI_OSABI] << std::endl;
        std::cout<< "ABI Version: " << elf->e_ident[EI_ABIVERSION] << std::endl;
        std::cout<< "\033[22m";
        std::cout<< "Type: " << elf->e_type << std::endl;
        std::cout<< "Machine: " << elf->e_machine << std::endl;
        std::cout<< "Version: " << elf->e_version << std::endl;
        std::cout<< "Entry point address: 0x" << std::hex << elf->e_entry << std::endl; // 
        std::cout<< "Start of program headers: 0x"<< std::hex << elf->e_phoff << std::endl;
        std::cout<< "Start of section headers: 0x" <<std::hex << elf->e_shoff << std::endl;  
        std::cout<< "[+]Program headers: " << std::dec << elf->e_phnum << std::endl;
        std::cout<< "[+]Section headers: " << std::dec << elf->e_shnum << std::endl;
        std::cout<< "[+]Section header string table index: " << std::dec << elf->e_shstrndx << std::endl;

        //add discrete features:

        discrete_features[file].push_back(elf->e_type);
        discrete_features[file].push_back(elf->e_machine);
        discrete_features[file].push_back(elf->e_version);
        discrete_features[file].push_back(elf->e_entry);
        discrete_features[file].push_back(elf->e_phoff);
        discrete_features[file].push_back(elf->e_shoff);
        discrete_features[file].push_back(elf->e_flags);
        discrete_features[file].push_back(elf->e_ehsize);
        discrete_features[file].push_back(elf->e_phentsize);
        discrete_features[file].push_back(elf->e_phnum);
        discrete_features[file].push_back(elf->e_shentsize);
        discrete_features[file].push_back(elf->e_shnum);
        discrete_features[file].push_back(elf->e_shstrndx);





        Elf64_Phdr* phdr = (Elf64_Phdr*)((char*)elf + elf->e_phoff);

        Elf64_Shdr* shdr = (Elf64_Shdr*)((char*)elf + elf->e_shoff);

        Elf64_Shdr* shstrtab = (Elf64_Shdr*)((char*)elf + shdr[elf->e_shstrndx].sh_offset);


        std::vector<std::string> strtab_strings;

        for (int i = 0; i < elf->e_phnum; i++) {
            
            std::cout<< "\033[32m";
            std::cout<< "[+][+]Program Header: " << i << std::endl;
            std::cout<< "[+][+]Type: " << std::hex << phdr[i].p_type << std::endl;
            std::cout<< "[+][+]Offset: " << std::hex << phdr[i].p_offset << std::endl;
            std::cout<< "[+][+]Virtual address: " << std::hex << phdr[i].p_vaddr << std::endl;
            std::cout<< "[+][+]Physical address: " << std::hex << phdr[i].p_paddr << std::endl;
            std::cout<< "[+][+]File offset: " << std::dec << phdr[i].p_offset << std::endl; // 
            std::cout<< "[+][+]Memory size: " << std::dec << phdr[i].p_memsz << std::endl;
            std::cout<< "[+][+]Flags: " << std::hex << phdr[i].p_flags << std::endl;
            std::cout<< "[+][+]Alignment: " << std::dec << phdr[i].p_align << std::endl;
            std::cout<< "\033[22m";

            //add sparse features here:
            if (phdr[i].p_type == PT_NULL) continue;
            else if (phdr[i].p_type == PT_LOAD) continue;
            else if (phdr[i].p_type == PT_DYNAMIC) continue;
            else if (phdr[i].p_type == PT_INTERP) continue;
            else if (phdr[i].p_type == PT_PHDR) continue;

            //add sparse features here:
            if( sparse_features.find(file) == sparse_features.end() ) {
                sparse_features[file] = {};
            }
            sparse_features[file].push_back(phdr[i].p_type);
            sparse_features[file].push_back(phdr[i].p_offset);
            sparse_features[file].push_back(phdr[i].p_vaddr);
            sparse_features[file].push_back(phdr[i].p_paddr);
            sparse_features[file].push_back(phdr[i].p_memsz);
            sparse_features[file].push_back(phdr[i].p_flags);
            sparse_features[file].push_back(phdr[i].p_align);
        }
        std::cout<< "[+]Section headers: " << elf->e_shnum << std::endl;
        //print section headers

        for (int i = 0; i < elf->e_shnum; i++) {
            // green
            std::cout<< "\033[32m";

            std::cout<< "[+][+]Section Header: " << std::to_string(i)  << std::endl;
            //std::cout<< "[+][+]Name: " << shstrtab[shdr[i].sh_name].sh_name << std::endl;
            std::cout<< "[+][+]Type: " << std::hex << shdr[i].sh_type << std::endl; 
            std::cout<< "[+][+]Flags: " << std::hex << shdr[i].sh_flags << std::endl;
            std::cout<< "[+][+]Address: " << std::hex << shdr[i].sh_addr << std::endl;
            std::cout<< "[+][+]Offset: " << std::dec << shdr[i].sh_offset << std::endl;
            std::cout<< "[+][+]Size: " << std::dec << shdr[i].sh_size << std::endl;
            std::cout<< "[+][+]Link: " << std::dec << shdr[i].sh_link << std::endl;
            std::cout<< "[+][+]Info: " << std::dec << shdr[i].sh_info << std::endl;
            std::cout<< "[+][+]Alignment: " << std::dec << shdr[i].sh_addralign << std::endl;
            std::cout<< "[+][+]Entry size: " << std::dec << shdr[i].sh_entsize << std::endl;
            if (shdr[i].sh_size==0) {
                std::cout << "[!]Empty section" << std::endl;
                
                
                continue;
            }
            //check type
            if (shdr[i].sh_type == SHT_STRTAB) {    
                std::cout << "[!]SHT_STRTAB" << std::endl;
                //print the string table
                char* strtab = (char*)elf + shdr[i].sh_offset; 
                
                std::string str;
                
                for (int j = 0; j < shdr[i].sh_size; j++) {
                    //std::cout << strtab[j];
                    str+=strtab[j];
                    if (strtab[j] == 0) {
                        strtab_strings.push_back(str);
                        str = "";
                    }
                }
                
            }
            else if (shdr[i].sh_type == SHT_PROGBITS) {
                std::cout << "[!]SHT_PROGBITS" << std::endl;
                //dissassemble the section
                //make sure shdr[i].sh_offset is valid and not out of bounds 
                if (shdr[i].sh_offset == 0 || shdr[i].sh_offset > shdr[i].sh_size) {
                    std::cout << "[!]Invalid section: " << file << std::endl;
                    continue;
                }
                else {
                    //dissassemble the section
                    //slice the section into 4KB chunks
                    //and dissassemble each chunk

                    std::cout<<"[+]Section is ready to be disassembled: " << file << std::endl; 

                }
            }
            else if (shdr[i].sh_type == SHT_NOBITS) {
                std::cout << "[!]SHT_NOBITS" << std::endl;
            }
            else if (shdr[i].sh_type == SHT_SYMTAB) {
                std::cout << "[!]SHT_SYMTAB" << std::endl;
            }
            else if (shdr[i].sh_type == SHT_DYNSYM) {
                std::cout << "[!]SHT_DYNSYM" << std::endl;
                //make sure shdr[i].sh_offset is valid and not out of bounds
                if (shdr[i].sh_offset == 0 || shdr[i].sh_offset > shdr[i].sh_size) { 
                    std::cout << "[!]Invalid dyn section: " << file << std::endl;
                    continue;
                }
                Elf64_Dyn* dyn = (Elf64_Dyn*)(elf + shdr[i].sh_offset); 
                //check dyn if it's valid
                if (dyn == nullptr || shdr[i].sh_size < sizeof(Elf64_Dyn)) {
                    std::cout << "[!]Invalid dyn section: " << file << std::endl;
                    continue;
                }
                for (int j = 0; j < shdr[i].sh_size/sizeof(Elf64_Dyn); j++) { 

                    if (dyn[j].d_tag == DT_NULL) {
                        std::cout << "[!]DT_NULL" << std::endl;
                        continue;
                    }

                    std::cout << "[!]Tag: " << dyn[j].d_tag << " " << dyn[j].d_un.d_val << std::endl; 

                    //add sparse features
                    if (sparse_features.find(file) == sparse_features.end()) {
                        sparse_features[file] = { uint64_t(dyn[j].d_tag) };
                    }
                    else
                    sparse_features[file].push_back(dyn[j].d_tag); 


                    if (dyn[j].d_tag == DT_NEEDED) {
                        std::cout << "[!]DT_NEEDED" << std::endl;
                    }
                    else if (dyn[j].d_tag == DT_STRTAB) {
                        std::cout << "[!]DT_STRTAB" << std::endl;
                    }
                    else if (dyn[j].d_tag == DT_SYMTAB) {
                        std::cout << "[!]DT_SYMTAB" << std::endl;
                    }
                    else if (dyn[j].d_tag == DT_PLTGOT) {
                        std::cout << "[!]DT_PLTGOT" << std::endl;
                    }
                    else if (dyn[j].d_tag == DT_JMPREL) {
                        std::cout << "[!]DT_JMPREL" << std::endl;
                    }

                }

            }
            else if (shdr[i].sh_type == SHT_DYNAMIC) {
                std::cout << "[!]SHT_DYNAMIC" << std::endl;
            }
            else if (shdr[i].sh_type == SHT_REL) {
                std::cout << "[!]SHT_REL" << std::endl;
            }
            else if (shdr[i].sh_type == SHT_RELA) {
                std::cout << "[!]SHT_RELA" << std::endl;
            }
        

            //add string features
        } 
        //map section headers
        total_elfs++;
        munmap(elf, st.st_size);
        close(fd);
        auto end = std::chrono::high_resolution_clock::now(); 
        //add times :
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); 
        std::cout << "Time taken: " << duration << " nanoseconds" << std::endl; 
        durations.push_back(duration);


        //symbols :
        std::cout << "[+]strings extracted: " << strtab_strings.size() << std::endl; 
        std::cout <<" ======================================================" << std::endl; 
        //pink
        std::cout << "\033[1;35m" ;

        for (auto s : strtab_strings) { 
            auto demangled = demangle(s.c_str());

            //std::cout << demangle(s.c_str()) << std::endl;
            //std::cout << s << std::endl;
            //add demangled string to feature string list
            string_features[file].push_back(demangled);

            //std::cout << demangled << std::endl;
        }
        
        std::cout << "\033[0m" ;
        std::cout <<" ======================================================" << std::endl; 
        //reset
        std::cout << "\033[0m" ;
        

    }//for files

    std::cout << "Total files: " << std::dec<<files.size() << std::endl;

    std::cout << "Average time taken per file: " << std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size() << " nanoseconds" << std::endl; 
    
    //save dataset
    std::cout << "Saving dataset" << std::endl;

    //save sparse features for each file
    for (auto file : files) {

        auto file_name = file.substr(file.find_last_of("/\\") + 1);

        std::cout << "Saving sparse features for file: " << file << std::endl;
        std::ofstream out_sparse_features;
        out_sparse_features.open(file_name + "_sparse_features.txt");
        for (auto s : sparse_features[file]) {
            out_sparse_features << std::hex << s << std::endl;
        }
        out_sparse_features.close();
    }   

    //save string features for each file
    for (auto file : files) {
        auto file_name = file.substr(file.find_last_of("/\\") + 1);
        std::cout << "Saving string features for file: " << file << std::endl;
        std::ofstream out_string_features;
        out_string_features.open(file_name + "_string_features.txt");
        for (auto s : string_features[file]) {
            out_string_features << s << std::endl;
        }
        out_string_features.close();
    }

    //save discrete features for each file 
    size_t file_id = 0;
    for (auto file : files) {
        auto file_name = file.substr(file.find_last_of("/\\") + 1);
        std::cout << "Saving discrete features for file: " << file << std::endl;
        std::ofstream out_discrete_features;
        out_discrete_features.open(file_name + "_discrete_features.txt");
        for (auto s : discrete_features[file]) {
            out_discrete_features << s << ',' ;
        }
        out_discrete_features << std::to_string(file_id) << std::endl;
        file_id++;

        out_discrete_features.close();
    }
    
    
    //done
    
    std::cout << "[+]Done" << std::endl;
    //reset term
    std::cout << "\033[0m" ;
    return 0;
}