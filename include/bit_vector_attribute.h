#ifndef __BIT_VECTOR_ATTRIBUTE_H__
#define __BIT_VECTOR_ATTRIBUTE_H__



#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>

 #include "matrix.h"
#include "optimizers.h"
//#include "dataset.h"
//#include "../util/csv_file.h"


//============================================================================== //
//--------------------------bit vector attribute-------------------------------- //
//             * this class is used to represent a bit vector attribute        * //
//--------------------------bit vector attribute-------------------------------- //
//============================================================================== // 
 
namespace provallo {

    #pragma pack(0)

    template <class T = uint8_t, size_t N = 8>
    class bit_type 
    {
        public :
        typedef T value_type;
        typedef T & reference;
        typedef const T & const_reference;
        typedef T * pointer;
        typedef const T * const_pointer;
        typedef T * iterator;

        static const size_t size = N;
        static const size_t npos = -1;
         

        private:
        value_type _bits; 
        public:
        bit_type() : _bits(0) {}
        bit_type(const bit_type &other) : _bits(other._bits) {}
        bit_type(bit_type &&other) : _bits(other._bits) {}
        bit_type(const T &other) : _bits(other) {}
        bit_type(T &&other) : _bits(other) {}
        bit_type(size_t i) : _bits(1 << i) {}


        bit_type & operator=(const bit_type &other) { 
            if( &other!=this) {
            _bits = other._bits;}
         return *this; 
         }
        bit_type & operator=(bit_type &&other) {
                if( &other!=this)
               { _bits = other._bits;} return *this; 
             }
        bit_type & operator=(const T &other) {
                if( &other!=this){
                _bits = other;} return *this;
             }
        bit_type & operator=(T &&other) { if(&other!=this){
         _bits = other;} return *this;
        }

        bool operator[](size_t i) const { return (_bits & (1 << i)) != 0; }
        bool operator[](size_t i) { return (_bits & (1 << i)) != 0; }

        bool operator[](int i) const { return (_bits & (1 << i)) != 0; }
        bool operator[](int i) { return (_bits & (1 << i)) != 0; }

        bool operator==(const bit_type &other) const { return _bits == other._bits; }
        bool operator!=(const bit_type &other) const { return _bits != other._bits; }
        bool operator<(const bit_type &other) const { return _bits < other._bits; }
        bool operator>(const bit_type &other) const { return _bits > other._bits; }
        bool operator<=(const bit_type &other) const { return _bits <= other._bits; }
        bool operator>=(const bit_type &other) const { return _bits >= other._bits; }
        bit_type operator&(const bit_type &other) const { return bit_type(_bits & other._bits); }
        bit_type operator|(const bit_type &other) const { return bit_type(_bits | other._bits); }
        bit_type operator^(const bit_type &other) const { return bit_type(_bits ^ other._bits); }
        bit_type operator~() const { return bit_type(~_bits); }
        bit_type & operator&=(const bit_type &other) { _bits &= other._bits; return *this; }
        bit_type & operator|=(const bit_type &other) { _bits |= other._bits; return *this; }
        bit_type & operator^=(const bit_type &other) { _bits ^= other._bits; return *this; }
        bit_type & operator<<=(const bit_type &other) { _bits <<= other._bits; return *this; }
        bit_type & operator>>=(const bit_type &other) { _bits >>= other._bits; return *this; }
        bit_type operator<<(const bit_type &other) const { return bit_type(_bits << other._bits); }
        bit_type operator>>(const bit_type &other) const { return bit_type(_bits >> other._bits); }
   //     bit_type operator<<(size_t i) const { return bit_type(_bits << i); }
   //     bit_type operator>>(size_t i) const { return bit_type(_bits >> i); }
        bit_type & operator<<=(size_t i) { _bits <<= i; return *this; }

        bit_type & operator>>=(size_t i) { _bits >>= i; return *this; }
        bit_type operator&(const T &other) const { return bit_type(_bits & other); }
        bit_type operator|(const T &other) const { return bit_type(_bits | other); }
        bit_type operator^(const T &other) const { return bit_type(_bits ^ other); }
        //bit_type operator~() const { return bit_type(~_bits); }
        bit_type & operator&=(const T &other) { _bits &= other; return *this; }
        bit_type & operator|=(const T &other) { _bits |= other; return *this; }
        bit_type & operator^=(const T &other) { _bits ^= other; return *this; }
        bit_type & operator<<=(const T &other) { _bits <<= other; return *this; }
        bit_type & operator>>=(const T &other) { _bits >>= other; return *this; }
        bit_type operator<<(const T &other) const { return bit_type(_bits << other); }
        bit_type operator>>(const T &other) const { return bit_type(_bits >> other); }
       //bit_type operator<<(size_t i) const { return bit_type(_bits << i); }
        //bit_type operator>>(size_t i) const { return bit_type(_bits >> i); }

        //bit_type & operator<<=(size_t i) { _bits <<= i; return *this; }
        //bit_type & operator>>=(size_t i) { _bits >>= i; return *this; }

        bit_type & set() { _bits = ~0; return *this; }
        bit_type & reset() { _bits = 0; return *this; }
        bit_type & flip() { _bits = ~_bits; return *this; }
        bit_type & set(size_t i) { _bits |= (1 << i); return *this; }
        bit_type & reset(size_t i) { _bits &= ~(1 << i); return *this; }
        bit_type & flip(size_t i) { _bits ^= (1 << i); return *this; }
        bit_type & set(size_t i, bool v) { if (v) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, bool v) { if (v) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, bool v) { if (v) flip(i); return *this; }
        bit_type & set(size_t i, const T &v) { if (v) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, const T &v) { if (v) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, const T &v) { if (v) flip(i); return *this; }
        bit_type & set(size_t i, const bit_type &v) { if (v[i]) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, const bit_type &v) { if (v[i]) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, const bit_type &v) { if (v[i]) flip(i); return *this; }
        bit_type & set(size_t i, const bit_type &v, bool b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, bool b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
        bit_type & flip(size_t i, const bit_type &v, bool b) { if (v[i]) flip(i, b); return *this; }
        bit_type & set(size_t i, const bit_type &v, const T &b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, const T &b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
        bit_type & flip(size_t i, const bit_type &v, const T &b) { if (v[i]) flip(i, b); return *this; }
        bit_type & set(size_t i, const bit_type &v, const bit_type &b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, const bit_type &b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
         
        friend bool operator==(const T &a, const bit_type &b) { return a == b._bits; }  
        friend bool operator!=(const T &a, const bit_type &b) { return a != b._bits; }
        friend bool operator<(const T &a, const bit_type &b) { return a < b._bits; }
        friend bool operator>(const T &a, const bit_type &b) { return a > b._bits; }
        friend bool operator<=(const T &a, const bit_type &b) { return a <= b._bits; }
        friend bool operator>=(const T &a, const bit_type &b) { return a >= b._bits; }

        friend bool operator==(const bit_type &a, const T &b) { return a._bits == b; }
        friend bool operator!=(const bit_type &a, const T &b) { return a._bits != b; }
        friend bool operator<(const bit_type &a, const T &b) { return a._bits < b; }
        friend bool operator>(const bit_type &a, const T &b) { return a._bits > b; }

        friend bool operator<=(const bit_type &a, const T &b) { return a._bits <= b; }
        friend bool operator>=(const bit_type &a, const T &b) { return a._bits >= b; }
        
        T & value() { return _bits; }
        const T & value() const { return _bits; }
        operator T() const { return _bits; }
        operator T&() { return _bits; }
        operator const T&() const { return _bits; }
        operator T*() { return &_bits; }
        operator const T*() const { return &_bits; }
        operator T&() const { return _bits; }
        operator const T&() { return _bits; }
        
        
     };

    typedef std::vector<bit_type<uint8_t,8>> u_bit_vector;
    typedef std::vector<bit_type<uint16_t,16>> u_bit_vector16;
    typedef std::vector<bit_type<uint32_t,32>> u_bit_vector32;
    typedef std::vector<bit_type<uint64_t,64>> u_bit_vector64;
    typedef std::vector<bit_type<uint8_t,8>> u_bit_vector8;
    typedef std::vector<bit_type<int8_t,8>> s_bit_vector8;
    typedef std::vector<bit_type<int16_t,16>> s_bit_vector16;
    typedef std::vector<bit_type<int32_t,32>> s_bit_vector32;
    typedef std::vector<bit_type<int64_t,64>> s_bit_vector64;
    typedef std::vector<bit_type<int8_t,8>> s_bit_vector;

    typedef std::vector<bit_type<float,32>> f_bit_vector32;
    typedef std::vector<bit_type<double,64>> f_bit_vector64;
    typedef std::vector<bit_type<float,32>> f_bit_vector;
    typedef std::vector<bit_type<bool,1>> b_bit_vector;

    template <class T,size_t N>
    std::ostream & operator<<(std::ostream &out, const bit_type<T,N> &b)
    {
        for (size_t i = 0; i < N; i++)
         out << b[i];
       
        return out;
    }   


    template <class T,size_t N>
    std::istream & operator>>(std::istream &in, bit_type<T,N> &b)
    {
        for (size_t i = 0; i < N; i++)
        {
            char c;
            in >> c;
            b[i] = c == '1';
        }
        return in;
    }   
    //to string
    template <class T,size_t N>
    std::string to_string(const bit_type<T,N> &b)
    {
        std::stringstream ss;
        ss << b;
        return ss.str();
    }   
    //n
    template <class T,size_t N> 
    std::string to_string(const bit_type<T,N> &b, size_t n)
    {
        std::stringstream ss;
        for (size_t i = 0; i < n; i++)
            ss << b[i];
        return ss.str();
    }   
    //separator     
    template <class T,size_t N>
    std::string to_string(const bit_type<T,N> &b, const std::string &sep)
    {
        std::stringstream ss;
        for (size_t i = 0; i < N; i++)
        {
            if (i > 0)
                ss << sep;
            ss << b[i];
        }
        return ss.str();
    } 
    template <class T,size_t N>
    bit_type<T,N>  unique_set_to_bit_type( const std::vector<T> &v)
    {
        bit_type<T,N> b;
        for (size_t i = 0; i < v.size(); i++)
            b.set(v[i]);
        return b;
    } 
    template <class T>
    std::vector<T>  
    unique_subset(const std::vector<T>& unique) 
    {

        std::vector<T> subset;
        for (size_t i = 0; i < unique.size(); i++)
            {
                //if not in subset, add it
                if (std::find(subset.begin(), subset.end(), unique[i]) == subset.end())
                    subset.push_back(unique[i]);
                    
            }
        return subset;
    }


    template <class T>
    std::vector<T>  
    unique_subset(  std::vector<T>& unique) 
    {
        //sort
        std::sort(unique.begin(), unique.end());
        //remove duplicates
        std::vector<T> subset;
        for (size_t i = 0; i < unique.size(); i++)
            {
                //if not in subset, add it
                if (std::find(subset.begin(), subset.end(), unique[i]) == subset.end())
                    subset.push_back(unique[i]);
                    
            }
        return subset;
    }
    //specialize unique subset for real_t
    template <>
    std::vector<real_t>
    unique_subset(  std::vector<real_t>& unique) 
    {
        //sort
        std::sort(unique.begin(), unique.end());
        real_t min=unique[0];
        real_t max=unique[unique.size()-1];
        
        //remove duplicates
        std::vector<real_t> subset;
        for (size_t i = 0; i < unique.size(); i++)
            {
                //if not in subset, add it
                if (std::find(subset.begin(), subset.end(), unique[i]) == subset.end())
                    subset.push_back(unique[i]);
                    
            } 
        size_t steps = subset.size();
        real_t step = (max-min)/steps;
        for (size_t i = 0; i < subset.size(); i++)
        {
                subset[i] = min + i*step;
        }
        return subset;
    }   


    //---------------------------------------------------------------------------------//   
    //bitwise operators:
    //---------------------------------------------------------------------------------//

    template <class T,size_t N> 
    bit_type<T,N> operator&(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a & b;
    }
    template <class T,size_t N> 
    bit_type<T,N> operator|(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a | b;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator^(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a ^ b;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator~(const bit_type<T,N> &a)
    {
        return ~a;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator<<(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a << b;
    }
    template <class T,size_t N>
    bit_type<T,N> operator>>(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a >> b;
    }
    template <class T,size_t N>
    bit_type<T,N> operator<<(const bit_type<T,N> &a, size_t i)
    {
        return a << i;
    } 
    template <class T,size_t N> 
    bit_type<T,N> operator>>(const bit_type<T,N> &a, size_t i)
    {
        return a >> i;
    } 

    //---------------------------------------------------------------------------------//
    //comparison operators:
    //---------------------------------------------------------------------------------//


    template <class T,size_t N>
    bool operator==(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a == b;
    }
    template <class T,size_t N>
    bool operator!=(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a != b;
    }
    template <class T,size_t N>
    bool operator<(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a < b;
    }
    template <class T,size_t N>
    bool operator>(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a > b;
    }
    template <class T,size_t N>
    bool operator<=(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a <= b;
    }
    template <class T,size_t N>
    bool operator>=(const bit_type<T,N> &a, const bit_type<T,N> &b)
    {
        return a >= b;
    }
    //---------------------------------------------------------------------------------//
    
    
    
    
    //dna sequence:
    typedef bit_type<uint8_t,2> dna_base; 
    typedef std::vector<dna_base> dna_sequence; 
    typedef matrix<dna_base> dna_matrix; 
    
    
    
    #pragma pack()

}

#endif 