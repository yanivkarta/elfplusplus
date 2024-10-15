#include "../include/elfplusplus.h"

#include <iostream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <mutex>    
#include <thread>
#include <chrono>
#include <algorithm>
//stat and mmap
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

using namespace elfpp;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: elfdump <elf file>" << std::endl;
        return 0;
    }

    const char* file = argv[1];

    //fstat
    struct stat st;
    if (::stat(file, &st) == -1) {
        std::cout << "Error stat file: " << file << std::endl;
        return -1;
    }
    std::cout << "File: " << file << std::endl; 
    std::cout << "Mode: " << st.st_mode << std::endl;
    std::cout << "Size: " << st.st_size << std::endl;

    //open
    int fd = ::open(file, O_RDONLY);
    if (fd == -1) {
        std::cout << "Error open file: " << file << std::endl;
        return -1;
    }

    //mmap
    void* map = ::mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        std::cout << "Error mmap file: " << file << std::endl;
        return -1;
    }

    // get elf header
    Elf64_Ehdr* eh = (Elf64_Ehdr*)map;
    std::cout << "Magic: " << eh->e_ident[0] << eh->e_ident[1] << eh->e_ident[2] << eh->e_ident[3] << std::endl;
    std::cout << "Class: " << eh->e_ident[4] << std::endl; 
    std::cout << "Data: " << eh->e_ident[5] << std::endl;
    std::cout << "Version: " << eh->e_ident[6] << eh->e_ident[7] << std::endl;

    std::cout << "Type: " << eh->e_type << std::endl;
    std::cout << "Machine: " << eh->e_machine << std::endl;
    std::cout << "Version: " << eh->e_version << std::endl;

    std::cout << "Entry: " << eh->e_entry << std::endl;

    std::cout << "Flags: " << eh->e_flags << std::endl;

    std::cout << "Header size: " << eh->e_ehsize << std::endl;

    std::cout << "Program header offset: " << eh->e_phoff << std::endl;

    std::cout << "Section header offset: " << eh->e_shoff << std::endl;

    std::cout << "Flags: " << eh->e_flags << std::endl;
    std::cout << "Header size: " << eh->e_ehsize << std::endl;
    std::cout << "Program header entry size: " << eh->e_phentsize << std::endl;
    std::cout << "Program header entry count: " << eh->e_phnum << std::endl;
    std::cout << "Section header entry size: " << eh->e_shentsize << std::endl;
    std::cout << "Section header entry count: " << eh->e_shnum << std::endl;
    std::cout << "Section header string table index: " << eh->e_shstrndx << std::endl;

    //open elf and parse structures:
    try {
        ElfFile64 ef(eh);
        ef.print();
    } catch (std::exception& e) {
        
        std::cout << "[-] Error parse elf, trying 32-bit: " << e.what() << std::endl;

        ElfFile32 ef((Elf32_Ehdr*)eh);
        ef.print();
    }
    //close & munmap
    ::close(fd);
    ::munmap(map, st.st_size);
    //add all the features to the database
    
    return 0;    
}
