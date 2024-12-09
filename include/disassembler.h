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
enum instruction_type {
    i_none = 0,
    i_8 = 1,
    i_16 = 2,
    i_32 = 3,
    i_64 = 4,
    i_128 = 5,
    i_256 = 6,
    i_512 = 7
};


enum operand_type {
    o_none = 0,
    o_reg = 1,
    o_mem = 2,
    o_imm = 3,
};
enum x86_operand_size {
    o8 = 1,
    o16 = 2,
    o32 = 4,
    o64 = 8,
    o128 = 16,
    o256 = 32,
    o512 = 64
};
enum arm64_reg{
    r_none = 0,
    r_x0 = 1,
    r_x1 = 2,
    r_x2 = 3,
    r_x3 = 4,
    r_x4 = 5,
    r_x5 = 6,
    r_x6 = 7,
    r_x7 = 8,
    r_x8 = 9,
    r_x9 = 10,
    r_x10 = 11,
    r_x11 = 12,
    r_x12 = 13,
    r_x13 = 14,
    r_x14 = 15,
    r_x15 = 16,
    r_x16 = 17,
    r_x17 = 18,
    r_x18 = 19, 
    r_x19 = 20,
    r_x20 = 21,
    r_x21 = 22,
    r_x22 = 23,
    r_x23 = 24,
    r_x24 = 25,

    r_x25 = 26,
    r_x26 = 27,
    r_x27 = 28,
    r_x28 = 29,
    r_x29 = 30,
    r_x30 = 31,
    r_x31 = 32, 

    r_w0 = 33,
    r_w1 = 34,
    r_w2 = 35,
    r_w3 = 36,
    r_w4 = 37,
    r_w5 = 38,
    r_w6 = 39,
    r_w7 = 40,
    r_w8 = 41,
    r_w9 = 42,
    r_w10 = 43,
    r_w11 = 44,
    r_w12 = 45,
    r_w13 = 46,
    r_w14 = 47,
    r_w15 = 48,
    r_w16 = 49,
    r_w17 = 50,
    r_w18 = 51,
    r_w19 = 52,
    r_w20 = 53,
    r_w21 = 54,
    r_w22 = 55,
    r_w23 = 56,
    r_w24 = 57,
    r_w25 = 58,
    r_w26 = 59,
    r_w27 = 60,
    r_w28 = 61,
    r_w29 = 62,
    r_w30 = 63,
    r_w31 = 64, 
    //s,d,f :
    r_s0 = 65,
    r_s1 = 66,
    r_s2 = 67,
    r_s3 = 68,
    r_s4 = 69,
    r_s5 = 70,
    r_s6 = 71,
    r_s7 = 72,
    r_s8 = 73, 
    r_s9 = 74,
    r_s10 = 75,
    r_s11 = 76,
    r_s12 = 77,
    r_s13 = 78,
    r_s14 = 79,
    r_s15 = 80,
    r_s16 = 81,
    r_s17 = 82,
    r_s18 = 83,
    r_s19 = 84,
    r_s20 = 85,
    r_s21 = 86,
    r_s22 = 87,
    r_s23 = 88,
    r_s24 = 89,
    r_s25 = 90,
    r_s26 = 91,
    r_s27 = 92,
    r_s28 = 93, 
    r_s29 = 94,
    r_s30 = 95,
    r_s31 = 96,         

    r_d0 = 97,
    r_d1 = 98,
    r_d2 = 99,
    r_d3 = 100,
    r_d4 = 101,
    r_d5 = 102,
    r_d6 = 103,
    r_d7 = 104,
    r_d8 = 105, 
    r_d9 = 106,
    r_d10 = 107,
    r_d11 = 108,
    r_d12 = 109,
    r_d13 = 110,
    r_d14 = 111,
    r_d15 = 112,
    r_d16 = 113,
    r_d17 = 114,
    r_d18 = 115,
    r_d19 = 116,
    r_d20 = 117,
    r_d21 = 118,
    r_d22 = 119,
    r_d23 = 120,
    r_d24 = 121,
    r_d25 = 122,
    r_d26 = 123,
    r_d27 = 124,
    r_d28 = 125, 
    r_d29 = 126,
    r_d30 = 127,
    r_d31 = 128,

    r_f0 = 129,
    r_f1 = 130,
    r_f2 = 131,
    r_f3 = 132,
    r_f4 = 133,
    r_f5 = 134,
    r_f6 = 135,
    r_f7 = 136,
    r_f8 = 137, 
    r_f9 = 138,
    r_f10 = 139,
    r_f11 = 140,
    r_f12 = 141,
    r_f13 = 142,
    r_f14 = 143,
    r_f15 = 144,
    r_f16 = 145,
    r_f17 = 146,
    r_f18 = 147,
    r_f19 = 148,
    r_f20 = 149,
    r_f21 = 150,
    r_f22 = 151,
    r_f23 = 152,
    r_f24 = 153,
    r_f25 = 154,
    r_f26 = 155,
    r_f27 = 156,
    r_f28 = 157,
    r_f29 = 158,
    r_f30 = 159,
    r_f31 = 160,

    r_ip = 161,
    r_sp = 162,
    r_bp = 163,
    r_pc = 164,
    r_0 = 165,
    r_1 = 166,
    r_2 = 167,
    r_3 = 168,
    r_4 = 169,
    r_5 = 170,
    r_6 = 171,
    r_7 = 172,
    r_8 = 173,
    r_9 = 174,
    r_10 = 175,
    r_11 = 176,
    r_12 = 177,
    r_13 = 178,
    r_14 = 179,
    r_15 = 180,
    r_16 = 181,
    r_17 = 182,
    r_18 = 183,
    r_19 = 184,
    r_20 = 185,
    r_21 = 186,
    r_22 = 187,
    r_23 = 188,
    r_24 = 189,
    r_25 = 190,
    r_26 = 191,
    r_27 = 192,
    r_28 = 193,
    r_29 = 194,
    r_30 = 195,
    r_31 = 196

    
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


template <typename T = uint64_t> 
class disassembler {
protected:

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
public:
    disassembler() {}
    virtual ~disassembler() {}
   //set elf header :
    
    virtual void set_elf_header(Elf32_Chdr* chdr) = 0;
    virtual void set_elf_header(Elf64_Chdr* chdr) = 0;
    virtual std::vector<std::string> dissassemble_section(struct Elf64_Shdr* shdr) = 0;
    virtual std::vector<std::string> dissassemble_section(struct Elf32_Shdr* shdr) = 0;
    virtual std::vector<std::string> dissassemble_symbol(struct Elf64_Sym* sym) = 0;
    virtual std::vector<std::string> dissassemble_symbol(struct Elf32_Sym* sym) = 0;
    virtual std::vector<std::string> disassemble_segment(struct Elf64_Phdr* phdr) = 0;

};

template <class T>
class x86_disassembler : public disassembler<T> {

private:
   
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
    
    

template <class T>
class arm64_disassembler : public disassembler<T> {


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