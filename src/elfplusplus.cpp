#include "elfplusplus.h"
#include "platform_helper.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <elf.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <mutex>    
#include <thread>
#include <chrono>
#include <algorithm>

//mmap  
#include <sys/mman.h>

bool is_elf(const std::string& path) {


    //check if it's an elf
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    Elf64_Ehdr ehdr;
    if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr)) {
        close(fd);
        return false;
    }
    close(fd);
    //check if it's an elf
    if (ehdr.e_ident[EI_MAG0] != 0x7f
            || ehdr.e_ident[EI_MAG1] != 'E'
            || ehdr.e_ident[EI_MAG2] != 'L'
            || ehdr.e_ident[EI_MAG3] != 'F') {

        return false;
    }

    return true;
}
  


namespace elfpp
{
    //implement ElfFile32
    void ElfFile64::parse_sections() {
        if (_baseaddr == nullptr) {
            std::cerr << "[!] no base address" << std::endl;
            return;
        }

        for (Elf64_Shdr* sec : _sections) {

            if (sec == nullptr) {
                std::cerr << "[!]NULL section ,skipping" << std::endl;
                continue;
            }
            //get section name from string table
            if (sec->sh_name >= _shstrtab.size()) {
                
                std::cerr << "[!]invalid section name" << std::endl;
                //check link to string table
                std::cerr<< "[+]link to string table: "<<sec->sh_link<<std::endl; 
                if(sec->sh_link >= _shstrtab.size()) {
                    std::cerr << "[!]invalid link name" << std::endl;
                }
                else {
                    std::cerr<< "[+]link name: "<<_shstrtab[sec->sh_link]<<std::endl; 
                } 
                
                
            }
            else {
                std::cout<<"[+]section name: "<<_shstrtab[sec->sh_name]<<std::endl; 
            }
            if (sec->sh_type == SHT_NULL) {
                continue;
            }
            else if (sec->sh_type == SHT_STRTAB) {
                continue;  // we already parsed that
            }
            else if (sec->sh_type == SHT_SYMTAB) {
                if (!_symbols.empty() && _symbols[0] != nullptr) {
                    std::cerr << "[!]multiple symbol tables" << std::endl;
                    continue;
                }
                else {
                    if (sec->sh_entsize < sizeof(Elf64_Sym)) {
                        std::cerr << "[!]invalid symbol table entry size" << std::endl;
                        continue;
                    }
                    Elf64_Sym** symbols = reinterpret_cast<Elf64_Sym**>(_baseaddr + sec->sh_offset);
                    for (int i = 0; i < sec->sh_size / sec->sh_entsize; ++i) {
                        _symbols.push_back(symbols[i]);
                    }
                }
                continue;
            }
            else if (sec->sh_type == SHT_PROGBITS) {
                Elf64_Phdr* phdr = reinterpret_cast<Elf64_Phdr*>(_baseaddr + sec->sh_offset);
                if (std::find(_segments.begin(), _segments.end(), phdr) != _segments.end()) {
                    std::cerr << "[!]multiple segments" << std::endl;
                    continue;
                }
                _segments.push_back(phdr);
            }
            else if (sec->sh_type == SHT_DYNAMIC) {
                if (!_dynamic_segments_info.empty()) {
                    std::cerr << "[!]multiple dynamic segments" << std::endl;
                    continue;
                }
                else {
                    _dynamic_segments_info.push_back(std::make_tuple(sec->sh_offset, sec->sh_offset, sec->sh_offset, sec->sh_offset, sec->sh_offset, sec->sh_offset, sec->sh_offset, sec->sh_offset));
                }
            }
            else if (sec->sh_type == SHT_REL) {
                Elf64_Xword* relocations = reinterpret_cast<Elf64_Xword*>(_baseaddr + sec->sh_offset);
                if (std::find(_relocations.begin(), _relocations.end(), relocations) != _relocations.end()) {
                    std::cerr << "[!]multiple relocations" << std::endl;
                    continue;
                }
                _relocations.push_back(relocations);
            }
            else if (sec->sh_type == SHT_DYNSYM) {
                if (!_symbols.empty()) {
                    std::cerr << "[!]multiple symbol tables" << std::endl;
                    continue;
                }
                _symbols.push_back(reinterpret_cast<Elf64_Sym*>(_baseaddr + sec->sh_offset));
            }
            else if (sec->sh_type == SHT_HASH) {
                auto hash = reinterpret_cast<Elf64_Word*>(_baseaddr + sec->sh_offset);
                if(hash==nullptr||sec->sh_offset<sizeof(Elf64_Shdr)) {
                    std::cerr << "[!]hash is null" << std::endl;
                    continue;
                }
                auto _hash_table = std::make_pair(hash[0],hash[1]);
                

                _hash.push_back(_hash_table);


            }
            else if (sec->sh_type == SHT_GNU_HASH) {
                _gnu_hash.push_back(reinterpret_cast<Elf64_Word*>(_baseaddr + sec->sh_offset));
            }
            else if (sec->sh_type == SHT_GNU_verdef) {
                _gnu_verdef.push_back(reinterpret_cast<Elf64_Verdaux*>(_baseaddr + sec->sh_offset));
            }
            else if (sec->sh_type == SHT_GNU_verneed) {
                _gnu_verneed.push_back(reinterpret_cast<Elf64_Vernaux*>(_baseaddr + sec->sh_offset));
            }
            else if (sec->sh_type == SHT_INIT_ARRAY) {
                _init_array.push_back(reinterpret_cast<Elf64_Addr*>(_baseaddr + sec->sh_offset));
            }
            else if (sec->sh_type == SHT_FINI_ARRAY) {
                _fini_array.push_back(reinterpret_cast<Elf64_Addr*>(_baseaddr + sec->sh_offset));
            }
            else if (sec->sh_type == SHT_PREINIT_ARRAY) {
                _preinit_array.push_back(reinterpret_cast<Elf64_Addr*>(_baseaddr + sec->sh_offset));
            }
            else if (sec->sh_type == SHT_NUM) {
                std::cerr << "[!]SHT_NUM unhandled section type " << std::to_string(sec->sh_type) << std::endl;
            }
            else {
                if (sec->sh_type != SHT_NULL) {
                    std::cerr << "[!] unhandled section type " << std::to_string(sec->sh_type) << std::endl;
                }
                else {
                    std::cerr << "[!]SHT_NULL unhandled section type " << std::to_string(sec->sh_type) << std::endl;
                }
            }
        }
    }

    //parse_relocations()

    void ElfFile64::parse_relocations() {
        //list of relocation information
               
         for ( auto reloc : _relocations) {   
            Elf64_Rel* rel =reinterpret_cast<Elf64_Rel*>(reloc);
            if (rel == nullptr) {
                std::cerr << "[!]relocation is null" << std::endl;
                continue;
            }
            auto info = rel->r_info ;
            auto offset = rel->r_offset; 
            auto type = rel->r_info >> 32;
            auto sym = rel->r_info & 0xffffffff;
            if (type == R_X86_64_RELATIVE) {
                std::cout << "R_X86_64_RELATIVE" << std::endl;
            }
            else if (type == R_X86_64_64) {
                std::cout << "R_X86_64_64" << std::endl;
            }
            else if (type == R_X86_64_GOTPCREL) {
                std::cout << "R_X86_64_GOTPCREL" << std::endl;
            }
            else if (type == R_X86_64_GOTPCRELX) {
                std::cout << "R_X86_64_GOTPCRELX" << std::endl;
            }
            else if (type == R_X86_64_REX_GOTPCRELX) {
                std::cout << "R_X86_64_REX_GOTPCRELX" << std::endl;
            }
            else if (type == R_X86_64_GOTPC32) {
                std::cout << "R_X86_64_GOTPC32" << std::endl;
            }
            
            else if (type == R_X86_64_GOTOFF64) {
                std::cout << "R_X86_64_GOTOFF64" << std::endl;
            }

 
        }

            std::sort(_relocations_info.begin(), _relocations_info.end()); 
    }//parse_relocations

    void ElfFile64::parse_symbols() {
        //list of symbols

        size_t n = _symbols.size(),m=0;
        if (n == 0) {
            return;
        }
        //get the symbols from the symbol table
        //FIXME:
        if(_symbols_info.size() > 0) {
            return;
        }
        if(n==1) {
            return;
        }
        for (Elf64_Sym* sym : _symbols) {
            if (sym == nullptr) {
                std::cerr << "[!]symbol is null" << std::endl;
                _symbols.erase(_symbols.begin() + m++);
                continue;
            }
            std::cout<<"[symbol] :"<<m<<"/"<<n<<std::endl;
            std::cout<<"[symbol] index: "<<sym->st_name<<std::endl; 
            if (sym->st_name == 0 || sym->st_name == 65535) {
                break;
            }
            if(sym->st_shndx == SHN_XINDEX) {
                break;
            }

            
            
            _symbols_info.push_back( std::make_tuple(sym->st_name, sym->st_value, sym->st_size, sym->st_info, sym->st_other, sym->st_shndx,n,m));  

            m++;
            if (m == n) {
                break;
            }

            std::cout<<std::endl;    
        }
        
    
    }//parse_symbols

    //parse_segments()

    void ElfFile64::parse_segments() {

        for (Elf64_Phdr* phdr : _segments) {
            if(!phdr || phdr->p_type == PT_NULL) {
                std::cerr << "[!]NULL segment" << std::endl;
                return;
            }
            

                auto offset = phdr->p_offset;
                auto vaddr = phdr->p_vaddr;
                auto paddr = phdr->p_paddr;
                auto size = phdr->p_filesz;
                auto memsz = phdr->p_memsz;
                auto flags = phdr->p_flags;
                auto align = phdr->p_align;
                auto type = phdr->p_type;
                std::string type_literal =std::to_string(type); 
                if (type == PT_LOAD) {
                    type_literal = "PT_LOAD";
                    _load_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type));
                }
                else if (type == PT_DYNAMIC) {
                    type_literal = "PT_DYNAMIC";
                    _dynamic_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type));
                }
                else if (type == PT_GNU_STACK) {
                    type_literal = "PT_GNU_STACK";
                    _stack_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type));
                }
                else  if (type ==PT_PHDR)  { 
                    type_literal = "PT_PHDR";
                    _phdr_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type));
                    
                }
                else if (type == PT_INTERP) {
                    type_literal = "PT_INTERP";
                    _interp_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type)); 
                }
                else if (type == PT_NOTE) {
                    type_literal = "PT_NOTE";
                    _note_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type)); 
                }
                else if (type == PT_SHLIB) {
                    type_literal = "PT_SHLIB";
                    _shlib_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type)); 
                }
                else if (type == PT_GNU_RELRO) {
                    type_literal = "PT_GNU_RELRO";
                    _relro_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type)); 
                }
                else if (type == PT_GNU_PROPERTY) 
                {
                    type_literal = "PT_GNU_PROPERTY";
                    _property_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type)); 
                }
                else if (type == PT_TLS) {
                    type_literal = "PT_TLS";
                    _tls_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type)); 
                }
                else if (type == PT_GNU_EH_FRAME) {
                    type_literal = "PT_GNU_EH_FRAME";
                    _ehframe_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type));
                }
                
                else {
                    type_literal = "other";
                    _other_segments_info.push_back(std::make_tuple(offset, vaddr, paddr, size, memsz, flags, align, type)); 

                }

            
            std::cout<<"[segment] type: "<<type_literal<<std::endl;
            std::cout<<"[segment] offset: "<<std::to_string(offset)<<std::endl; 
            std::cout<<"[segment] vaddr: "<<std::to_string(vaddr)<<std::endl;
            std::cout<<"[segment] paddr: "<<std::to_string(paddr)<<std::endl;
            std::cout<<"[segment] size: "<<std::to_string(size)<<std::endl;
            std::cout<<"[segment] memsz: "<<std::to_string(memsz)<<std::endl;
            std::cout<<"[segment] flags: "<<std::to_string(flags)<<std::endl;
            std::cout<<"[segment] align: "<<std::to_string(align)<<std::endl;


            if (phdr->p_flags & PF_X) {

                std::cout<<"[segment] executable"<<std::endl;


            }
            if (phdr->p_flags & PF_W) {
                std::cout<<"[segment] writable"<<std::endl;
            }
            if (phdr->p_flags & PF_R) {
                std::cout<<"[segment] readable"<<std::endl;
            }

            std::cout<<std::endl;


        }
    }//parse_segments

    //parse_dynsyms()

    void ElfFile64::parse_dynsyms() {

        for (Elf64_Addr* sym_addr : _dynsyms) {
            
            Elf64_Sym* sym = reinterpret_cast<Elf64_Sym*>(sym_addr);

            std::cout<<"[dynsym] index: "<<sym->st_name<<std::endl; 
            
            std::cout<<"[dynsym]symbol value: "<<sym->st_value<<std::endl; 
            std::cout<<"[dynsym]symbol size: "<<sym->st_size<<std::endl;
            std::cout<<"[dynsym]symbol type: "<<sym->st_info<<std::endl;
            std::cout<<"[dynsym]symbol other: "<<sym->st_other<<std::endl;
            std::cout<<"[dynsym]symbol shndx: "<<sym->st_shndx<<std::endl;
            std::cout<<std::endl;


        }

    }

    void ElfFile64::parse_dynamic() {
    for (Elf64_Dyn* dyn : _dynamic) {
        if(!dyn || dyn->d_tag == DT_NULL ) {
            std::cout<<"[dynamic] NULL"<<std::endl;
            return;
        }
        

        std::cout<<"[dynamic] type: "<<dyn->d_tag<<std::endl;
        std::cout<<"[dynamic] value: "<<dyn->d_un.d_val<<std::endl;
        std::string dyn_str = get_dyn_string(dyn);
        std::cout<<"[dynamic] name: "<<dyn_str<<std::endl;
        std::cout<<std::endl;        

    }
    
    }//parse_dynamic
    //print

    void ElfFile64::print() const{
        std::cout<<"[sections] size: "<<std::to_string(_sections.size())<<std::endl;
        //std::cout<<"[shstrtab] size: "<<std::to_string(_shstrtab.size())<<std::endl; 
        std::cout<<"[strtab] size: "<<std::to_string(_strtab.size())<<std::endl; 
        std::cout<<"[shstrtab] size: "<<std::to_string(_shstrtab.size())<<std::endl;

        
        std::cout<<"[reloc] size: "<<std::to_string(_relocations.size())<<std::endl;
        std::cout<<"[reloc_info] size: "<<_relocations_info.size()<<std::endl;  
        std::cout<<"[symbols] size: "<<_symbols.size()<<std::endl;
        std::cout<<"[segments] size: "<<_segments.size()<<std::endl;
        std::cout<<"[dynsyms] size: "<<_dynsyms.size()<<std::endl;
        std::cout<<"[dynamic] size: "<<_dynamic.size()<<std::endl;
        
        for (auto segment : _load_segments_info) {
            //get segment name from shstrtab 
            std::cout<<"[segment] name: "<<get_string(std::get<0>(segment))<<std::endl;
            std::cout<<"[segment] type: "<<std::get<7>(segment)<<std::endl;
            std::cout<<"[segment] offset: "<<std::get<0>(segment)<<std::endl; 
            std::cout<<"[segment] vaddr: "<<std::get<1>(segment)<<std::endl;
            std::cout<<"[segment] paddr: "<<std::get<2>(segment)<<std::endl;
            std::cout<<"[segment] size: "<<std::get<3>(segment)<<std::endl;
            std::cout<<"[segment] memsz: "<<std::get<4>(segment)<<std::endl;
            std::cout<<"[segment] flags: "<<std::get<5>(segment)<<std::endl;
            std::cout<<"[segment] align: "<<std::get<6>(segment)<<std::endl;
            std::cout<<std::endl;
        }
        for (auto segment : _other_segments_info) {
            std::cout<<"[segment] type: "<<std::get<7>(segment)<<std::endl;
            std::cout<<"[segment] offset: "<<std::get<0>(segment)<<std::endl;           
            std::cout<<"[segment] vaddr: "<<std::get<1>(segment)<<std::endl;
            std::cout<<"[segment] paddr: "<<std::get<2>(segment)<<std::endl;
            std::cout<<"[segment] size: "<<std::get<3>(segment)<<std::endl;
            std::cout<<"[segment] memsz: "<<std::get<4>(segment)<<std::endl;
            std::cout<<"[segment] flags: "<<std::get<5>(segment)<<std::endl;
            std::cout<<"[segment] align: "<<std::get<6>(segment)<<std::endl;
            std::cout<<std::endl;
        }
        for (auto str : _sections) {
            std::cout<<"[section] name: "<<str->sh_name<<std::endl; 
            std::cout<<"[section] type: "<<str->sh_type<<std::endl;
            std::cout<<"[section] flags: "<<str->sh_flags<<std::endl;
            std::cout<<"[section] addr: "<<str->sh_addr<<std::endl;
            std::cout<<"[section] offset: "<<str->sh_offset<<std::endl;
            std::cout<<"[section] size: "<<str->sh_size<<std::endl;
            std::cout<<"[section] link: "<<str->sh_link<<std::endl;
            std::cout<<"[section] info: "<<str->sh_info<<std::endl;
            std::cout<<"[section] addralign: "<<str->sh_addralign<<std::endl;
            std::cout<<"[section] entsize: "<<str->sh_entsize<<std::endl;
            std::cout<<std::endl;
        }

        //print injectable sections and relocations 
        for ( auto str : _sections) {
            if (str->sh_type == SHT_PROGBITS) {

                //check if section is injectable 
                if (str->sh_flags & SHF_EXECINSTR) {

                    std::cout<<"[injectable section] name: "<<get_string(str->sh_name)<<std::endl; 
                    std::cout<<"[injectable section] type: "<<str->sh_type<<std::endl;
                    std::cout<<"[injectable section] flags: "<<str->sh_flags<<std::endl;
                    std::cout<<"[injectable] addr: "<<str->sh_addr<<std::endl;
                    std::cout<<"[injectable] offset: "<<str->sh_offset<<std::endl;
                    std::cout<<"[injectable] size: "<<str->sh_size<<std::endl;
                    std::cout<<"[injectable] link: "<<str->sh_link<<std::endl;
                    std::cout<<"[injectable] info: "<<str->sh_info<<std::endl;
                    std::cout<<"[injectable] addralign: "<<str->sh_addralign<<std::endl;
                    std::cout<<"[injectable] entsize: "<<str->sh_entsize<<std::endl;
                    std::cout<<std::endl; 
                }
                else {

                    std::cout<<"[non-injectable section] name: "<<get_string(str->sh_name)<<std::endl;
                    std::cout<<"[non-injectable section] type: "<<str->sh_type<<std::endl;
                    std::cout<<"[non-injectable section] flags: "<<str->sh_flags<<std::endl;
                    std::cout<<"[non-injectable] addr: "<<str->sh_addr<<std::endl;
                    std::cout<<"[non-injectable] offset: "<<str->sh_offset<<std::endl;
                    std::cout<<"[non-injectable] size: "<<str->sh_size<<std::endl;
                    std::cout<<"[non-injectable] link: "<<str->sh_link<<std::endl;
                    std::cout<<"[non-injectable] info: "<<str->sh_info<<std::endl;
                    std::cout<<"[non-injectable] addralign: "<<str->sh_addralign<<std::endl;
                    std::cout<<"[non-injectable] entsize: "<<str->sh_entsize<<std::endl;
                    std::cout<<std::endl;
                }
        }

        }//for sections
        

    }//print info

    void ElfFile32::print_ehdr()const {

            
        if (!_ehdr) { 
            std::cout<<"[ehdr] _ehdr is null"<<std::endl;
            return;
        }

        std::cout<<"[elf32] e_ident: ["<<std::hex<<_ehdr->e_ident[0]<<std::hex<<_ehdr->e_ident[1]<<std::hex<<_ehdr->e_ident[2]<<std::hex<<_ehdr->e_ident[3]<<"]"<<std::endl; 
        std::cout<<"[elf32] e_type: "<<_ehdr->e_type<<std::endl;
        std::cout<<"[elf32] e_machine: "<<_ehdr->e_machine<<std::endl;
        std::cout<<"[elf32] e_version: "<<_ehdr->e_version<<std::endl;
        std::cout<<"[elf32] e_entry: "<<_ehdr->e_entry<<std::endl;
        std::cout<<"[elf32] e_phoff: "<<_ehdr->e_phoff<<std::endl;
        std::cout<<"[elf32] e_shoff: "<<_ehdr->e_shoff<<std::endl;
        std::cout<<"[elf32] e_flags: "<<_ehdr->e_flags<<std::endl;
        std::cout<<"[elf32] e_ehsize: "<<_ehdr->e_ehsize<<std::endl;
        std::cout<<"[elf32] e_phentsize: "<<_ehdr->e_phentsize<<std::endl;
        std::cout<<"[elf32] e_phnum: "<<_ehdr->e_phnum<<std::endl;
        std::cout<<"[elf32] e_shentsize: "<<_ehdr->e_shentsize<<std::endl;
        std::cout<<"[elf32] e_shnum: "<<_ehdr->e_shnum<<std::endl;
        std::cout<<"[elf32] e_shstrndx: "<<_ehdr->e_shstrndx<<std::endl;


        
    }
    //
    void ElfFile32::print()const {
            //print info
            //print base pointer
            print_ehdr();
            //print sections
            for (int i = 0; i < _sections.size(); ++i) {
                if (!_sections[i]) {
                    std::cout<<"[section] _sections["<<i<<"] is null"<<std::endl;
                    continue;
                }
                std::cout<<"[section] name: "<<get_string(_sections[i]->sh_name)<<std::endl;
                std::cout<<"[section] type: "<<_sections[i]->sh_type<<std::endl;
                std::cout<<"[section] flags: "<<_sections[i]->sh_flags<<std::endl;

                std::cout<<"[section] addr: "<<_sections[i]->sh_addr<<std::endl;
                std::cout<<"[section] offset: "<<_sections[i]->sh_offset<<std::endl;
                std::cout<<"[section] size: "<<_sections[i]->sh_size<<std::endl;
                std::cout<<"[section] link: "<<_sections[i]->sh_link<<std::endl;
                std::cout<<"[section] info: "<<_sections[i]->sh_info<<std::endl;
                std::cout<<"[section] addralign: "<<_sections[i]->sh_addralign<<std::endl;
                std::cout<<"[section] entsize: "<<_sections[i]->sh_entsize<<std::endl;
                std::cout<<std::endl;

            }//for sections
            std::cout<<"[segment] count: "<<std::dec<<_segments.size()<<std::endl;
            //print segments
            std::cout<<"[dynsym] count: "<<std::dec<<_dynsyms.size()<<std::endl;
            

            

    }
    std::vector<std::string> ElfFile32::get_section_names()const {

        std::vector<std::string> ret;
        for (int i = 0; i < _sections.size(); ++i) {
            ret.push_back(get_string(_sections[i]->sh_name));
        }
        return ret;
    }
    


}//namespace elfpp   


    
 