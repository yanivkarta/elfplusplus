// C++x17 thread-safe ELF library
// Copyright (C) 2022 Yaniv Karta
// SPDX-License-Identifier: MIT



#ifndef ELFPLUSPLUS_H   
#define ELFPLUSPLUS_H

#include <elf.h>
//STB_DEBUG
#ifndef STB_DEBUG
#define STB_DEBUG 1
#endif //STB_DEBUG
//STB_GLOBAL
#ifndef STB_GLOBAL
#define STB_GLOBAL 2
#endif 

#ifndef STB_NONE
#define STB_NONE 0
#endif //STB_NONE

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <iomanip>
#include <iostream>
//constants for the elf++ library
//PF flags
#ifndef PF_X
#define PF_X 1
#define PF_W 2
#define PF_R 4
#endif //PF_X


//e-ident fields
#ifndef EI_MAG0

#define EI_MAG0 0
#define EI_MAG1 1
#define EI_MAG2 2

#define EI_MAG3 3
#define EI_CLASS 4
#define EI_DATA 5
#define EI_VERSION 6
#define EI_OSABI 7
#define EI_ABIVERSION 8
#define EI_PAD 9
#define EI_NIDENT 16    

#endif //EI_MAG0
//ELFMAG
#ifndef ELFMAG0
#define ELFMAG0 0x7f
#define ELFMAG1 'E'
#define ELFMAG2 'L'
#define ELFMAG3 'F'
#endif //ELFMAG0
//ELFCLASS
#define ELFCLASS32 1
#define ELFCLASS64 2
//ELFVERSION
#define EV_CURRENT 1

//ELFOSABI
#ifndef ELFOSABI_NONE
#define ELFOSABI_NONE 0
#define ELFOSABI_SYSV 0
#define ELFOSABI_LINUX 3
#define ELFOSABI_HPUX 8
#define ELFOSABI_SOLARIS 6
#define ELFOSABI_AIX 7
#define ELFOSABI_IRIX imirix\0
#define ELFOSABI_FREEBSD 9
#define ELFOSABI_TRU64 10
#define ELFOSABI_MODESTO 2
#define ELFOSABI_OPENBSD 1
#define ELFOSABI_ARM AOUT
#define ELFOSABI_STANDALONE 255 
#endif //ELFOSABI_NONE

//ELFABIVERSION
#ifndef ELFABIVERSION_CURRENT
#define ELFABIVERSION_CURRENT 1
#endif //ELFABIVERSION_CURRENT
#ifndef PT_NULL
//PT_NULL   

#define PT_NULL 0
//PT_LOAD
#define PT_LOAD 1
//PT_DYNAMIC
#define PT_DYNAMIC 2
//PT_INTERP
#define PT_INTERP 3
//PT_NOTE
#define PT_NOTE 4
//PT_SHLIB
#define PT_SHLIB 5
//PT_PHDR
#define PT_PHDR 6
#endif //PT_NULL

//SHT_NULL
#ifndef SHT_NULL
#define SHT_NULL 0
//SHT_PROGBITS
#define SHT_PROGBITS 1
//SHT_SYMTAB
#define SHT_SYMTAB 2
//SHT_STRTAB
#define SHT_STRTAB 3
//SHT_RELA
#define SHT_RELA 4
//SHT_HASH
#define SHT_HASH 5
//SHT_DYNAMIC
#define SHT_DYNAMIC 6
//SHT_NOTE
#define SHT_NOTE 7
//SHT_NOBITS
#define SHT_NOBITS 8
//SHT_REL
#define SHT_REL 9
//SHT_SHLIB
#define SHT_SHLIB 10
//SHT_DYNSYM
#define SHT_DYNSYM 11
//SHT_INIT_ARRAY
#define SHT_INIT_ARRAY 14
//SHT_FINI_ARRAY
#define SHT_FINI_ARRAY 15
//SHT_PREINIT_ARRAY
#define SHT_PREINIT_ARRAY 16
//SHT_GROUP
#define SHT_GROUP 17
//SHT_SYMTAB_SHNDX
#define SHT_SYMTAB_SHNDX 18
//SHT_LOPROC
#define SHT_LOPROC 0x70000000
//SHT_HIPROC
#define SHT_HIPROC 0x7fffffff
//SHT_LOUSER
#define SHT_LOUSER 0x80000000
//SHT_HIUSER
#define SHT_HIUSER 0xffffffff
#endif
//Dynamic section

#ifndef DT_NULL
#define DT_NULL 0
#define DT_NEEDED 1
#define DT_PLTRELSZ 2
#define DT_PLTGOT 3
#define DT_HASH 4
#define DT_STRTAB 5
#define DT_SYMTAB 6
#define DT_RELA 7
#define DT_RELASZ 8
#define DT_RELAENT 9
#define DT_STRSZ 10
#define DT_SYMENT 11
#define DT_INIT 12
#define DT_FINI 13
#define DT_SONAME 14
#define DT_RPATH 15
#define DT_SYMBOLIC 16
#define DT_REL 17
#define DT_RELSZ 18 
#define DT_RELENT 19
#define DT_PLTREL 20
#define DT_DEBUG 21
#define DT_TEXTREL 22
#define DT_JMPREL 23
#define DT_BIND_NOW 24
#define DT_INIT_ARRAY 25    
#define DT_FINI_ARRAY 26

#endif 

//printing flags
#ifndef ELF_PRINT_NONE
#define ELF_PRINT_NONE 0
#define ELF_PRINT_PROGRAM_HEADERS 1
#define ELF_PRINT_SECTION_HEADERS 2
#define ELF_PRINT_DYNAMIC_SECTION 4 
#define ELF_PRINT_ALL 7
#endif //ELF_PRINT_NONE
#include "matrix.h"

namespace elfpp {

        class ElfFile 
        {
        protected:
            std::string _path;
            //typedef std::map<uint64_t, std::tuple<std::string,provallo::matrix<double>,provallo::matrix<double> > > feature_map_t; 
            typedef provallo::matrix<double> feature_map_t;


            feature_map_t _features;


        public:
            ElfFile() {}
            ElfFile(const std::string& path) {
                _path = path;
                
            }

            ElfFile(const ElfFile& other) {
                _path = other._path;
                _features = other._features;
            }

            ElfFile& operator=(const ElfFile& other) {
                _path = other._path;
                _features = other._features;
                return *this;
            }

            ~ElfFile() {}


            const std::string& path() const {
                return _path;
            }

            feature_map_t& features() {
                return _features;
            }

            const feature_map_t& features() const {
                return _features;
            }
            void set_path(const std::string& path) {
                _path = path;
            }
            
            //pure virtual for subclasses
            virtual std::vector<std::string> get_section_names() const=0;
            virtual bool isValid() const = 0;
        };

        class ElfFile64 : public ElfFile {
            std::vector<Elf64_Shdr*> _sections;
            std::vector<Elf64_Phdr*> _segments;
            std::vector<Elf64_Sym*> _symbols;
            std::vector<Elf64_Addr*> _dynsyms;
            std::vector<Elf64_Dyn*> _dynamic; 
            Elf64_Ehdr* _baseaddr  = nullptr;
            //_gnu_hash 
            std::vector<Elf64_Word*> _gnu_hash; 
            //hash table 
            std::vector<std::pair<uint64_t, uint64_t>> _hash;
            std::vector<Elf64_Word*> _gnu_hash_extra;
            //verdef verneed 
            std::vector<Elf64_Verdef*> _verdef;
            std::vector<Elf64_Verneed*> _verneed;

            //gnu_verdef gnu_verneed
            std::vector<Elf64_Verdaux*> _gnu_verdef;
            std::vector<Elf64_Vernaux*> _gnu_verneed;
            //init_array,fini_array,preinit_array 
            std::vector<Elf64_Xword*> _init_array;
            std::vector<Elf64_Xword*> _fini_array;
            std::vector<Elf64_Xword*> _preinit_array;

            //bss,interp,etc
             std::vector<Elf64_Xword*> _relocations; 
             std::vector<std::tuple<uint64_t, uint64_t, Elf64_Addr*>> _relocations_info; 

             typedef std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> tuple_base;
             //load dynamic segments tuple(offset, vaddr, paddr, size, memsz, flags, align, type) 
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _load_segments_info; 
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _dynamic_segments_info;
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _stack_segments_info; 
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _bss_segments_info; 
             //phdr
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _phdr_segments_info; 
            //interp 
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _interp_segments_info; 
             //note segment 
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _note_segments_info; 
             //shlib 
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _shlib_segments_info; 
             //gnu_relro
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _gnu_relro_segments_info; 
             //properties
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _property_segments_info; 
             //relro
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _relro_segments_info; 
             //shstrtab
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _shstrtab_segments_info; 
             //ehframe
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _ehframe_segments_info; 
             //tls  
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _tls_segments_info; 
             //symbols_info
             std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _symbols_info; 


             //other

            std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> _other_segments_info;

             

             
             
            std::vector<std::string> _strtab;
            std::vector<std::string> _shstrtab; 
            std::vector<std::string> _dynstrtab;
            //note section
            std::vector<Elf64_Nhdr*> _notes;                
            inline std::string get_string(Elf64_Word index) const {
              
                if (index >= _shstrtab.size()) {
                    if (index >= _strtab.size()) {
                        return "[UNKNOWN]";
                    }
                    return _strtab[index];
                }
                return _shstrtab[index];
            }
            


        public:
            ElfFile64() : ElfFile() {}
            ElfFile64(Elf64_Ehdr* ehdr) : ElfFile(), _baseaddr(ehdr) { 

                if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
                    throw std::runtime_error("Not a 64-bit ELF file");
                }

                _sections = std::vector<Elf64_Shdr*>(ehdr->e_shnum); 
                _segments = std::vector<Elf64_Phdr*>(ehdr->e_phnum);
                _symbols = std::vector<Elf64_Sym*>();
                _dynamic = std::vector<Elf64_Dyn*>();

                _strtab = std::vector<std::string>(ehdr->e_shstrndx + 1);

                _relocations = std::vector<Elf64_Xword*>(ehdr->e_shnum);

                _dynsyms = std::vector<Elf64_Addr*>();

                _notes = std::vector<Elf64_Nhdr*>();

                Elf64_Shdr* shdr = (Elf64_Shdr*)((char*)ehdr + ehdr->e_shoff); 

                Elf64_Shdr* shstrtab = (Elf64_Shdr*)((char*)ehdr + shdr[ehdr->e_shstrndx].sh_offset);



                //Elf64_Shdr* shstrtab = (Elf64_Shdr*)((char*)ehdr + shdr[ehdr->e_shstrndx].sh_offset);
                if (!shstrtab || !shstrtab->sh_type!= SHT_STRTAB) {
                    
                    for (int i = 0; i < ehdr->e_shnum; ++i) {
                        if (shdr[i].sh_type == SHT_STRTAB) {
                            shstrtab = &shdr[i];
                            break;
                        }
                    }

                    
                }

                uint64_t index = ehdr->e_shstrndx;     
                const char* builtin = "[BUILTIN]";
                for (int i = 0; i < ehdr->e_shnum; ++i) {
                    
                    auto ptr = (char*)ehdr +ehdr->e_shstrndx + shstrtab->sh_offset;
                    ptr += shdr[i].sh_name;
                    if (ptr[0] != '\0')
                    _strtab.push_back(std::string(ptr));
                    else _strtab.push_back(std::string(builtin));
                } 


                //shstrtab
                for (int i = 0; i < ehdr->e_shnum; ++i) {
                        if (shdr[i].sh_type == SHT_STRTAB) {
                        
                        auto ptr = (char*)ehdr + shdr[i].sh_offset ;

                        std::string str = "";
                        for (int j = 0; j < shdr[i].sh_size; ++j) {
                            

                            
                            if ( ptr[j] == '\0' && str.size() > 0) {
                                std::cout<< "[+] adding SHT_STRTAB string: " << str << std::endl;
                                _shstrtab.push_back(str);
                                str = "";

                            }
                            else if (ptr[j] != '\0')
                                str += ptr[j];
                            else continue;
                        }   
                 
                    } //SHT_STRTAB 
                    else if (shdr[i].sh_type == SHT_SYMTAB) {

                        _symbols.push_back((Elf64_Sym*)((char*)ehdr + shdr[i].sh_offset)); 


                    }
                    else if (shdr[i].sh_type == SHT_DYNSYM) {
                        Elf64_Sym* ptr = (Elf64_Sym*)((char*)ehdr + shdr[i].sh_offset); 
                        for (int j = 0; j < shdr[i].sh_size / sizeof(Elf64_Sym); ++j) {
                            Elf64_Addr addr = ptr[j].st_value;
                            _dynsyms.push_back(&addr);
                            
                        }



                    }
                    else if (shdr[i].sh_type == SHT_DYNAMIC) {
                        Elf64_Dyn* ptr = (Elf64_Dyn*)((char*)ehdr + shdr[i].sh_offset); 

                        for (int j = 0; j < shdr[i].sh_size / sizeof(Elf64_Dyn); ++j) {
                            _dynamic.push_back(ptr);
                            ptr++;
                        }
                        

                    }
                    else if (shdr[i].sh_type == SHT_RELA) {
                        _relocations.clear();
                        Elf64_Addr* ptr = (Elf64_Addr*)((char*)ehdr + shdr[i].sh_offset); 

                        for (int j = 0; j < shdr[i].sh_size / sizeof(Elf64_Xword); ++j) {
                            _relocations.push_back(ptr);
                            ptr++;
                        }

                    }
                    else if (shdr[i].sh_type == SHT_NOTE) {
                        _notes.push_back((Elf64_Nhdr*)((char*)ehdr + shdr[i].sh_offset));
                    }
                    _sections[i] = &shdr[i];
                }
                //fill segments:
                Elf64_Phdr* phdr = (Elf64_Phdr*)((char*)ehdr + ehdr->e_phoff);

                for (int i = 0; i < ehdr->e_phnum; ++i) {
                    if (phdr[i].p_type == PT_NULL) {
                        std::cout << "[+]skipping [NULL] segment: " << i << " type: " << phdr[i].p_type << " offset: " << phdr[i].p_offset << " size: " << phdr[i].p_filesz << std::endl; 
                        continue;
                    }


                    _segments[i] = &phdr[i];
                    //print segment
                    std::cout << "[+] segment: " << i << " type: " << phdr[i].p_type << " offset: " << phdr[i].p_offset << " size: " << phdr[i].p_filesz << std::endl; 

                }

                _dynamic.clear();
                

                //fill dynamic:
                for (int i = 0; i < ehdr->e_shnum; ++i) {
                    
                    if (shdr[i].sh_type != SHT_DYNAMIC) {
                        continue;
                    }

                    //shdr[i] is the dynamic section
                    uint64_t number_of_entries = shdr[i].sh_size/shdr[i].sh_entsize; 

                    

                    std::cout << "[+]dynamic: " << i <<  " number of entries: " << number_of_entries <<  std::endl;
                    
                    //get dyn table
                    Elf64_Addr* dyn_addr = (Elf64_Addr*)((char*)ehdr + shdr[i].sh_offset); 

                    
                    Elf64_Dyn* dyn = (Elf64_Dyn*)(dyn_addr);
                    

                    for (int j = 0; j < number_of_entries; ++j) {
                        std::cout << "[+]dynamic: " << i << " dyn: " << j << std::endl;
                        if(!dyn || dyn->d_tag == DT_NULL ) {
                            std::cout<<"[dynamic] NULL"<<std::endl;
                            continue;
                        }

                        std::cout<<"[dynamic] type: "<<dyn->d_tag<<std::endl;
                        std::cout<<"[dynamic] value: "<<dyn->d_un.d_val<<std::endl;
                        std::string dyn_str = get_dyn_string(dyn);
                        std::cout<<"[dynamic] name: "<<dyn_str<<std::endl;
                        std::cout<<std::endl;
                        _dynamic.push_back(dyn);
                        
                     }//for j


                    //validate dyn

                    
                    
                    
                    //fill _dynstrtab
                    if (_dynamic[i]->d_tag == DT_STRTAB) {
                        //
                        std::cout<<"[+] adding DT_STRTAB: " <<std::hex<< _dynamic[i]->d_un.d_ptr << std::dec<<std::endl; 
                        //iterate and copy  
                        auto ptr = (char*)ehdr + _dynamic[i]->d_un.d_ptr;
                        size_t size = _dynamic[i]->d_un.d_val; 
                        std::string str = "";
                        for (size_t j = 0; j < size; ++j) {

                            if ( ptr[j] == '\0' && str.size() > 0) {
                                std::cout<< "[+] adding DT_STRTAB string: " << str << std::endl;
                                _dynstrtab.push_back(str);
                                str = "";
                            }
                            else if (ptr[j] != '\0')
                                str += ptr[j];
                            else continue;  

                            
                        } //for
                    }//if

                    std::cout<<std::endl;        

                }//for i

                std::cout << std::endl;
                std::cout<<"[+] dynstrtab size: " << _dynstrtab.size() << std::endl;
                //fill relocations:
                for (int i = 0; i < ehdr->e_shnum; ++i) {
                    if (shdr[i].sh_type != SHT_REL) {
                        continue;
                    }

                    std::cout<<"[+] relocations: " << i << std::endl; 
                    Elf64_Rel* rel = (Elf64_Rel*)((char*)ehdr + shdr[i].sh_offset); 

                    for (int j = 0; j < shdr[i].sh_size/sizeof(Elf64_Rel); ++j) {
                        std::cout << "[+]relocations: " << i << " rel: " << j << std::endl;
                        Elf64_Rel* rel = (Elf64_Rel*)((char*)ehdr + shdr[i].sh_offset + j*sizeof(Elf64_Rel)); 

                        _relocations.push_back((Elf64_Addr*)rel);

                        rel++;
                    }


                    _relocations[i] = (Elf64_Xword*)((char*)ehdr + shdr[i].sh_offset);
                }

                
                //fill notes:
                for (int i = 0; i < ehdr->e_shnum; ++i) {
                    
                    if (shdr[i].sh_type != SHT_NOTE) {
                        continue;
                    }
                    Elf64_Nhdr* note = (Elf64_Nhdr*)((char*)ehdr + shdr[i].sh_offset); 
                    //validate note 

                    if(!note || note->n_namesz < 4 || note->n_descsz < 20) {
                        continue;
                         
                    }

                    _notes.push_back(note)  ;

                }



                
                parse_segments();

                parse_sections() ;
                
                
                parse_symbols();
                parse_dynamic();
                parse_relocations();
                parse_dynsyms();

                for (auto &dyn : _dynamic) {
                    if(!dyn || dyn->d_tag == DT_NULL ) {
                        continue;
                    }
                    std::cout<<"[dynamic] type: "<<dyn->d_tag<<std::endl;
                    std::cout<<"[dynamic] value: "<<dyn->d_un.d_val<<std::endl;
                    std::string dyn_str = get_dyn_string(dyn);
                    std::cout<<"[dynamic] name: "<<dyn_str<<std::endl;
                    std::cout<<std::endl;
                }
                provallo::matrix<uint64_t> m(_sections.size(), _sections.size());
                for (int i = 0; i < _sections.size(); ++i) {
                    for (int j = 0; j < _sections.size(); ++j) {
                        if( _sections[i]==nullptr) {
                            m(i, j) = 0;
                            continue;
                        }
                        m(i, j) = _sections[i]->sh_offset;
                    }
                }


                
                provallo::matrix<uint64_t> d(_dynsyms.size(), _dynsyms.size()); 
                for (int i = 0; i < _dynsyms.size(); ++i) {
                    for (int j = 0; j < _dynsyms.size(); ++j) {
                        if(_dynsyms[i]==nullptr) {
                            d(i, j) = 0;
                            continue;
                        }
                        d(i, j) = ((Elf64_Dyn*)_dynsyms[i])->d_un.d_val; 
                    }
                }



                provallo::matrix<uint64_t> r(_relocations.size(), _relocations.size()); 
                for (int i = 0; i < _relocations.size(); ++i) {
                    for (int j = 0; j < _relocations.size(); ++j) {
                        auto rel = _relocations[i];
                        if(rel==nullptr) {
                            r(i, j) = 0;
                            continue;
                        }
                        r(i, j) = ((Elf64_Rel*)rel)->r_offset;
                    }
                }


                provallo::matrix<uint64_t> x(_dynamic.size(), _dynamic.size()); 
                for (int i = 0; i < _dynamic.size(); ++i) {
                    for (int j = 0; j < _dynamic.size(); ++j) {
                        if(_dynamic[i]==nullptr) {
                            x(i, j) = 0;
                            continue;
                        }
                        
                        else x(i, j) = ((Elf64_Dyn*)_dynamic[i])->d_un.d_val; 
                    }
                }
                

                //make a matrix of all the tuples from segment info
                //count the tuple columns and rows
                //make a matrix for the segment info tuples. 
                //they all have the same number of variables, so we can just add them from the beginning. 
                if(d.size1()==0||d.size2()==0) 
                    d.resize(1,1);
                if (d.sum()==0) 
                    d.fill(1);
                if(r.size1()==0||r.size2()==0) 
                    r.resize(1,1);

                if (r.sum()==0) 
                    r.fill(1);
                if (x.size1()==0||x.size2()==0)
                    x.resize(1,1);

                if (x.sum()==0) 
                    x.fill(1);
                if (m.size1()==0||m.size2()==0) 
                    m.resize(1,1);
                    
                if(m.sum()==0)
                    m.fill(1);
                

                size_t cols = std::max(m.size1(), std::max(d.size1(), std::max(r.size1(), x.size1()))); 
                size_t rows = std::max(m.size2(), std::max(d.size2(), std::max(r.size2(), x.size2()))); 
                uint64_t file_id=ehdr->e_shoff+ehdr->e_shnum*ehdr->e_shentsize;

                std::cout <<"file id: " << file_id << " cols:" << "cols: " << cols << " rows: " << rows << std::endl; 


                provallo::matrix<uint64_t> aggregate(rows, cols);

                for (int i = 0; i < aggregate.size1(); ++i) {
                    for (int j = 0; j < aggregate.size2(); ++j) {
                        aggregate(i, j) = m(i%m.size1(), j%m.size2()) + d(i%d.size1(), j%d.size2()) + r(i%r.size1(), j%r.size2()) + d(i%d.size1(), j%   d.size2()) + x(i%x.size1(), j%x.size2()); // + m(i%m.size1(), j);
                        //aggregate(i, j) %= 0x100000000;

                    }
                }   
                std::cout<<"adding tuples "<< std::endl; 
                std::vector<tuple_base> tuples_vec = {_load_segments_info,_dynamic_segments_info,_stack_segments_info,_bss_segments_info,
_phdr_segments_info,_interp_segments_info,_note_segments_info,_shlib_segments_info,
_gnu_relro_segments_info,_property_segments_info,_relro_segments_info,_shstrtab_segments_info
,_ehframe_segments_info,_tls_segments_info,_symbols_info,_other_segments_info};

                provallo::matrix<uint64_t> tuples(tuples_vec.size(), tuples_vec[0].size()); 

                for (int i = 0; i < tuples_vec.size(); ++i) {
                    //i is the index of the datarow 
                    for (int j = 0; j < tuples_vec[i].size(); ++j) {
                        for (int k = 0; k < tuples.size2(); ++k) {
                            //get k element from tuple at i,j

                        auto tuple =tuples_vec[i][j];
                        //get k element from tuple at i,j
                        //unpack tuple 
                        //use decltype to get the type of the tuple
                        //and use std::get to get the value of the tuple 
                        if ( k >= std::tuple_size<decltype(tuple)>::value) {
                            tuples(i, k) = 0;
                            continue;
                        }
                        
                        //assign tuple(i, k) 
                        tuples(i, k) = [&]( size_t k) -> uint64_t {
                        
                        const auto [element1, element2, element3, element4, element5, element6 , element7 , element8 ] = tuple;  

                        if (k == 0) {
                            return element1;
                        } else if (k == 1) {
                            return element2;
                        } else if (k == 2) {    
                            return element3;
                        } else if (k == 3) {
                            return element4;
                        } else if (k == 4) {
                            return element5;
                        } else if (k == 5) {
                            return element6;
                        } else if (k == 6) {
                            return element7;
                        } else if (k == 7) {
                            return element8;
                        }
                        else {
                            return 0;
                        }
                            
                            //parse tuple, get value of k element

                          
                            
                            
                        }(k);    



                        }

                    }
                }

#ifdef _DEBUG_MATRIX_OUTPUT
                //print the tuples matrix :
                std::cout<<"==============="<<std::endl;
                std::cout<<"Tuples Matrix"<<std::endl;
                std::cout << tuples << std::endl;
                std::cout<<"==============="<<std::endl;

                std::cout<<"Aggregate Matrix"<<std::endl;
                std::cout<<"==============="<<std::endl;
                
                std::cout << aggregate << std::endl; 
                std::cout<<"==============="<<std::endl;

                aggregate=aggregate*tuples;
                std::cout<<"Tuppled Aggregate Matrix"<<std::endl;
                std::cout<<"==============="<<std::endl;
                std::cout << aggregate << std::endl; 
                std::cout<<"==============="<<std::endl;
#endif //_DEBUG_MATRIX_OUTPUT

                
                
                //size_t nrot=4;
                //use tikhonov regularization
                //provallo::matrix<uint64_t> st = provallo::tikhonov(aggregate, tuples, aggregate.size2());

                //std::cout<<"Tikhonov Matrix"<<std::endl;
                //std::cout<<"==============="<<std::endl;
                //std::cout << st << std::endl; 
                //std::cout<<"==============="<<std::endl;

                provallo::matrix<double> conv = provallo::matrix<double>(aggregate.size1(), aggregate.size2()); 
                uint64_t min,max,mean;
                min = aggregate.minCoeff();
                max = aggregate.maxCoeff();
                mean = aggregate.mean();
                if (max == min) {
                    max = min + 1;
                }
                else if(max<min){
                    auto tmp = max;
                    max = min;
                    min = tmp;
                    }

                std::cout <<"[+] creating feature matrix"<<std::endl;
                for (int i = 0; i < aggregate.size1(); ++i) {
                    for (int j = 0; j < aggregate.size2(); ++j) {
                        //make the matrix between 0 and 1
                        conv(i, j) = double(aggregate(i, j) - min) / double(max - min) * 0.99 + 0.01; 
                    }
                }   
                //show the feature matrix
                
                this->_features = conv;

                //print features : 
                std::cout<<"Features Matrix"<<std::endl;
                std::cout<<"==================================================="<<std::endl; 
                std::cout << "||entropy: " << conv.entropy() << std::endl;
                //std::cout <<"||adjoint: "<<conv.adjoint() << "" << std::endl;
                std::cout <<"||mean: "<<conv.mean() << std::endl;
                
                std::cout<<"||min: "<<conv.minCoeff()  << std::endl;
                std::cout<<"||max: "<<conv.maxCoeff()  << std::endl;
                std::cout<<"||sum: "<<conv.sum()  << std::endl;
                std::cout<<"||divergence size: "<<conv.divergent().size() << "" << std::endl;
                std::cout<<"==================================================="<<std::endl; 
            }//constructor
            std::string get_dyn_string(Elf64_Dyn* dyn) {

                if (dyn->d_tag == DT_NULL || dyn->d_un.d_val >= _dynstrtab.size()) {  
                    return "[NULL]";
                }
                return _dynstrtab[dyn->d_un.d_val];

            }
            void disassemble_segment(Elf64_Phdr* segment) {
                
                //use concept of hexdump

                Elf64_Addr begin = segment->p_vaddr;
                Elf64_Addr end = begin + segment->p_memsz;
                if(end-begin > 0x100000) {
                    //too big, not supported
                    return;
                }

                std::cout << "begin: " << std::hex << begin << " end: " << end << std::dec << std::endl;

                std::cout << std::hex << begin << " - " << end << std::dec << std::endl;

                std::cout << "Segment: " << segment->p_type << std::endl;

                if (begin-end <= 0) {
                    return;
                }
                for (Elf64_Addr it = begin; it < end; ++it) {
                    
                    char* ptr = (char*)it;
                    std::cout << std::hex << ptr << " ";

                    if ((it - begin) % 16 == 15) {
                        std::cout << std::endl;
                    } 

                }


                


            }
            
            ElfFile64(const std::string& path) : ElfFile(path) {}   
            ElfFile64(const ElfFile64& other) : ElfFile(other) {}
            ElfFile64& operator=(const ElfFile64& other) {
                ElfFile::operator=(other);
                return *this;
            }
            bool isValid() const { return true; }
            virtual std::vector<std::string> get_section_names() const 
            {
                std::vector<std::string> names;
                std::string str;
                for (int i = 0; i < _sections.size(); ++i) {
                    str = get_string(_sections[i]->sh_name);
                    if (str != "") {
                        names.push_back(str);
                    } else {
                        names.push_back("[NULL]");  
                    }
                    //names.push_back(get_string(_sections[i]->sh_name));
                }
                return names;
            }
             

            virtual ~ElfFile64() {}
            void parse_sections();
            void parse_segments();
            void parse_symbols();
            
            void parse_relocations();
            void parse_dynsyms();
            void parse_dynamic();
            void print() const;

            

        };

        class ElfFile32 : public ElfFile {
            std::vector<Elf32_Shdr*> _sections;
            std::vector<Elf32_Phdr*> _segments;
            std::vector<Elf32_Sym*> _symbols;
            std::vector<Elf32_Dyn*> _dynamic;
            std::vector<Elf32_Xword> _relocations;
            std::vector<std::string> _strtab;
            std::vector<Elf32_Addr> _dynsyms;
            //notes
            std::vector<Elf32_Nhdr> _notes;
            //shstrtab
            std::vector<std::string> _shstrtab;
            //dynstrtab
            std::vector<std::string> _dynstrtab;
            //init_array
            std::vector<Elf32_Addr> _init_array;


            Elf32_Ehdr* _ehdr=nullptr;


        public:


            
            ElfFile32() : ElfFile() {}
            ElfFile32(const std::string& path) : ElfFile(path) {}
            ElfFile32(const ElfFile32& other) : ElfFile(other) {}
            ElfFile32(Elf32_Ehdr* ehdr) : ElfFile() {

                _ehdr = ehdr;
                _sections = std::vector<Elf32_Shdr*>(ehdr->e_shnum);
                _segments = std::vector<Elf32_Phdr*>(ehdr->e_phnum);
                //_symbols = std::vector<Elf32_Sym*>(ehdr->e_shnum);
                //_dynamic = std::vector<Elf32_Dyn*>(ehdr->e_shnum);  
                //strtab = std::vector<std::string>(ehdr->e_shstrndx);   

                //_relocations = std::vector<Elf32_Xword>(ehdr->e_shnum);

                //_dynsyms = std::vector<Elf32_Addr>(ehdr->e_shnum);

                uint32_t shstrndx = ehdr->e_shstrndx;
                uint32_t type = ehdr->e_type;

                std::cout<<"[sections] size: "<<std::to_string(_sections.size())<<std::endl;
                //fill sections :
                for (int i = 0; i < ehdr->e_shnum; ++i) {
                    std::cout << "section: " << i << std::endl;
                    Elf32_Shdr* shdr = (Elf32_Shdr*)((char*)ehdr + ehdr->e_shoff + i * ehdr->e_shentsize);
                    _sections[i] = shdr;
                    if(shdr->sh_type == SHT_NULL) {
                        continue;
                    }
                    if (shdr[i].sh_type == SHT_STRTAB) {
                        std::cout<< "[+] adding string table" << std::endl;
                        auto ptr = (char*)ehdr + shdr[i].sh_offset;
                        size_t size = shdr[i].sh_size;
                        std::string str = "";
                        for (int j = 0; j < size; ++j) {
                            
                            if ( ptr[j] == '\0' && str.size() > 0) {
                                std::cout<< "[+] adding string(SHT_STRTAB): " << str << std::endl;
                                _shstrtab.push_back(str);
                                str = "";

                            }
                            else if (ptr[j] != '\0')
                                str += ptr[j];
                            else continue;
                        }   //for
                    }//if
                    else if (shdr[i].sh_type == SHT_SYMTAB) {
                             if (shdr[i].sh_entsize < sizeof(Elf32_Sym)) 
                                 continue;

                             for (int j = 0; j < shdr[i].sh_size / shdr[i].sh_entsize; ++j) {
                                Elf32_Sym* sym = reinterpret_cast<Elf32_Sym*>(shdr[i].sh_offset + j * shdr[i].sh_entsize);
                                
                                _symbols.push_back(sym); // = sym; 
                                std::cout<< "[+] adding symbol: " << sym->st_name << " value: " << std::to_string(sym->st_value) << std::endl; 


                            }
                    }//else
                    else if (shdr[i].sh_type == SHT_DYNSYM) {

                        std::cout<< "[+] adding dynsym" << std::endl;
                        for (int j = 0; j < shdr[i].sh_size / shdr[i].sh_entsize; ++j) {
                            Elf32_Sym* sym = reinterpret_cast<Elf32_Sym*>(shdr[i].sh_offset + j * shdr[i].sh_entsize);

                            std::cout <<"[+] adding dynsym: " << sym->st_name << " value: " << std::to_string(sym->st_value) << std::endl;
                            _dynsyms.push_back((Elf32_Addr) sym->st_value); // sym->st_value;
                        }
                    }//else
                    else if (shdr[i].sh_type == SHT_REL) {
                        std::cout<< "[+] adding relocations" << std::endl;
                        if (shdr[i].sh_entsize < sizeof(Elf32_Xword)|| shdr[i].sh_size == 0) 
                            continue;

                        _relocations.resize(shdr[i].sh_size / shdr[i].sh_entsize); 

                        for (int j = 0; j < shdr[i].sh_size / shdr[i].sh_entsize; ++j) {
                            Elf32_Xword* rel = reinterpret_cast<Elf32_Xword*>(shdr[i].sh_offset + j * shdr[i].sh_entsize);
                            _relocations[j] =(Elf32_Xword) rel;
                        } 
                    }
                    else if (shdr[i].sh_type == SHT_DYNAMIC) {
                        std::cout<< "[+] adding dynamic" << std::endl;
                        if(shdr[i].sh_entsize < sizeof(Elf32_Dyn))
                            continue;   
                        _dynamic.resize(shdr[i].sh_size / shdr[i].sh_entsize);
                        for (int j = 0; j < shdr[i].sh_size / shdr[i].sh_entsize; ++j) {
                            Elf32_Dyn* dyn = reinterpret_cast<Elf32_Dyn*>(shdr[i].sh_offset + j * shdr[i].sh_entsize);
                            _dynamic[j] = dyn;
                        }
                    }   
                    else if (shdr[i].sh_type == SHT_NOBITS) {   
                            std::cout<< "[+] no bits" << std::endl; 
                    }
                    else if (shdr[i].sh_type == SHT_PROGBITS) {
                            std::cout<< "[+] prog bits" << std::endl;
                    }
                    else if (shdr[i].sh_type == SHT_NOTE) {
                        if (shdr[i].sh_entsize < sizeof(Elf32_Nhdr))
                            continue;
                        for (int j = 0; j < shdr[i].sh_size / shdr[i].sh_entsize; ++j) {
                            Elf32_Nhdr* note = reinterpret_cast<Elf32_Nhdr*>(shdr[i].sh_offset + j * shdr[i].sh_entsize);
                            _notes.push_back(*note);
                            std::cout<< "[+] adding note: " << note->n_namesz << " " << note->n_descsz << " " << note->n_type << std::endl;
                        }   
                    }
                    else if (shdr[i].sh_type == SHT_INIT_ARRAY) {
                        //get init array
                        if (shdr[i].sh_entsize < sizeof(Elf32_Addr))
                            continue;
                        std::vector<Elf32_Addr> init_array;
                        std::cout<< "[+] init array" << std::endl; 
                        for (int j = 0; j < shdr[i].sh_size / shdr[i].sh_entsize; ++j) {
                            Elf32_Dyn* dyn = reinterpret_cast<Elf32_Dyn*>(shdr[i].sh_offset + j * shdr[i].sh_entsize);
                            init_array.push_back(dyn->d_un.d_ptr);
                        }   

                        _init_array = init_array;

                    }
                    else if (shdr[i].sh_type == SHT_FINI_ARRAY) {
                        
                        size_t size = shdr[i].sh_size / (shdr[i].sh_entsize>0?shdr[i].sh_size / shdr[i].sh_entsize:1);

                        std::cout<< "[+] fini array" << " size: " << std::dec<< std::to_string(size) <<  std::endl;


                    }
                    else    std::cout<< "[+] unknown type: " << std::to_string(shdr[i].sh_type) << std::endl; 

                }//for

                
                //fill segments:
                for (int i = 0; i < ehdr->e_phnum; ++i) {
                    Elf32_Phdr* phdr = reinterpret_cast<Elf32_Phdr*>(ehdr->e_phoff + i * ehdr->e_phentsize);
                    _segments[i] = phdr;
                }

                
                

                
                //std::cout<<"[shstrtab] size: "<<std::to_string(_shstrtab.size())<<std::endl;
            }
            ElfFile32& operator=(const ElfFile32& other) {
                ElfFile::operator=(other);
                return *this;
            }
            inline std::string get_string(Elf32_Word index) const {
              
              
              
              
                if (index >= _shstrtab.size()) {
                    if (index >= _strtab.size()) {
                        return "[UNKNOWN]";
                    }
                    return _strtab[index];
                }
                return _shstrtab[index];
            }
            void print_ehdr()const;
            bool isValid() const { return true; }
            virtual void print() const ;
            virtual std::vector<std::string> get_section_names() const;
            ~ElfFile32() {}

        };
}    

extern "C" void on_sigint(int unused);
#endif  
