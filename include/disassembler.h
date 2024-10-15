#ifndef __DISASSEMBLER_H__
#define __DISASSEMBLER_H__


//linux elf disassembler
#include "elf.h"

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <iomanip>

#include <string_view>

using namespace std;

enum register_type {
    r_none = 0,
    r_8 = 1,
    r_16 = 2,
    r_32 = 3,
    r_64 = 4
};
enum x86_instruction_type {
    i_none = 0,
    i_8 = 1,
    i_16 = 2,
    i_32 = 3,
    i_64 = 4
};
enum x86_operand_type {
    o_none = 0,
    o_reg = 1,
    o_mem = 2,
    o_imm = 3
};
enum x86_operand_size {
    o8 = 1,
    o16 = 2,
    o32 = 4,
    o64 = 8
};

enum x86_reg {
    r_none = 0,
    r_ax = 1,
    r_bx = 2,
    r_cx = 3,
    r_dx = 4,
    r_sp = 5,
    r_bp = 6,
    r_si = 7,
    r_di = 8,
    r_8 = 9,
    r_9 = 10,
    r_10 = 11,
    r_11 = 12,
    r_12 = 13,
    r_13 = 14,
    r_14 = 15,
    r_15 = 16,  
    r_ip = 17

};
enum x86_mem {
    m_none = 0,
    m_ax = 1,
    m_bx = 2,
    m_cx = 3,
    m_dx = 4,
    m_sp = 5,
    m_bp = 6,
    m_si = 7,
    m_di = 8,
    m_8 = 9,
    m_9 = 10,
    m_10 = 11,
    m_11 = 12,
    m_12 = 13,
    m_13 = 14,
    m_14 = 15,
    m_15 = 16,  
    m_ip = 17
};      

enum x86_imm {
    i_none = 0, 
    i_8 = 1,
    i_16 = 2,
    i_32 = 3,
    i_64 = 4

};
enum x86_64_x86_instruction_type {
    i_none = 0,
    i_8 = 1,
    i_16 = 2,
    i_32 = 3,
    i_64 = 4,
    i_128 = 5,
    i_256 = 6
};
enum x86_64_x86_operand_type {
    o_none = 0,
    o_reg = 1,
    o_mem = 2,
    o_imm = 3

};
enum x86_64_x86_operand_size {
    o8 = 1,
    o16 = 2,
    o32 = 4,
    o64 = 8
};
enum x86_64_x86_reg {
    r_none = 0,
    r_ax = 1,
    r_bx = 2,
    r_cx = 3,
    r_dx = 4,
    r_sp = 5,
    r_bp = 6,
    r_si = 7,
    r_di = 8,
    r_8 = 9,
    r_9 = 10,
    r_10 = 11,
    r_11 = 12,
    r_12 = 13,
    r_13 = 14,
    r_14 = 15,
    r_15 = 16,
    r_ip = 17

};
enum x86_64_x86_mem {
    m_none = 0,
    m_ax = 1,
    m_bx = 2,
    m_cx = 3,
    m_dx = 4,
    m_sp = 5,
    m_bp = 6,
    m_si = 7,
    m_di = 8,
    m_8 = 9,
    m_9 = 10,
    m_10 = 11,
    m_11 = 12,
    m_12 = 13,
    m_13 = 14,
    m_14 = 15,
    m_15 = 16,
    m_ip = 17

};

enum x86_64_x86_imm {
    i_none = 0,
    i_8 = 1,
    i_16 = 2,
    i_32 = 3,
    i_64 = 4,
    i_128 = 5,
    i_256 = 6
};
enum x86_64_x86_reg_size {
    r8 = 1,
    r16 = 2,
    r32 = 4,
    r64 = 8,
    r128 = 16,
    r256 = 32,
    r512 = 64,
    r1024 = 128
};
enum x86_64_x86_mem_size {
    m8 = 1,
    m16 = 2,
    m32 = 4,
    m64 = 8,
    m128 = 16,
    m256 = 32,
    m512 = 64,
    m1024 = 128

};





template <class T>
class disassembler {

private:
    T* _chdr;
    T* _shdr;
    T* _symtab;
    T* _strtab;
    T* _reloc;
    T* _dyn;
    T* _plt;
    T* _got;
    T* _dynsym;
    T* _dynstr;
    T* _rela;
    T* _pltrel;

    T* _text;
    T* _data;
    T* _bss;
    T* _end;

    T* _baseaddr;
    
    T* _text_start;
    T* _text_end;

    T* _data_start;
    T* _data_end;

    T* _bss_start;
    T* _bss_end;

    T* _end_start;
    T* _end_end;

    T* _plt_start;
    T* _plt_end;

    T* _got_start;
    T* _got_end;

    T* _dynsym_start;
    T* _dynsym_end;

    T* _dynstr_start;
    T* _dynstr_end;

    T* _rela_start;
    T* _rela_end;

    T* _reloc_start;
    T* _reloc_end;

    T* _pltrel_start;
    T* _pltrel_end;
    enum x86_64_x86_operand_type _operand_type;
    enum x86_64_x86_operand_size _operand_size;
    enum x86_64_x86_reg _reg;
    enum x86_64_x86_mem _mem;
    enum x86_64_x86_imm _imm;
    enum x86_64_x86_reg_size _reg_size;
    enum x86_64_x86_mem_size _mem_size;

    void* _func = nullptr;

    T* _ptr = nullptr;
    T* _ptr_end = nullptr;
    T* _ptr_start = nullptr;
    T* _ptr2     = nullptr;
    T* _ptr2_end = nullptr;
    T* _ptr2_start  = nullptr;

    
public:
    void set_elf_header( Elf32_Chdr* chdr );
    void set_elf_header( Elf64_Chdr* chdr );
    std::vector<std::string> dissassemble_section( struct Elf64_Shdr* shdr );
    std::vector<std::string> dissassemble_section( struct Elf32_Shdr* shdr );
    std::vector<std::string> dissassemble_symbol( struct Elf64_Sym* sym );
    std::vector<std::string> dissassemble_symbol( struct Elf32_Sym* sym );

    std::vector<std::string> disassemble_segment( struct Elf64_Phdr* phdr ); 
    
    
};    
    
    



#endif //__DISASSEMBLER_H__