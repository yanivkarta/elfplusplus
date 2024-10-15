#ifndef __CPU_MATRIX_H__
#define __CPU_MATRIX_H__

//mmx, sse, sse2, sse3, ssse3, sse4, avx, avx2, avx512  extensions implementation for our matrix. 
//use specialization for types and operations 

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <thread>
#include <mutex>
#include <immintrin.h>
#include <xmmintrin.h>
#include "matrix.h"


//create CPU vector for std::vector
//using AVX512,SIMD and mmx extensions for better performance 
//use aligned memory allocation for better performance 

namespace std{

template < typename T>
    class cpu_vector : public std::vector<T> { 
        //declare T *aligned_data; with linux aligned attribute
        T *aligned_data __attribute__((aligned(64))); 
        //declare data_ to poiont on vector<T>::data pointer
        T *data_ =  nullptr;

    public:
        //apply optimizations for the cpu_vector class 
        cpu_vector() : vector<T>() , aligned_data(nullptr){
            //allocate memory for the vector 
            //use aligned memory allocation for better performance 
            if(this->data_ != nullptr)
                delete [] this->data_; //parent allocated with new 
            aligned_data    = (T *)aligned_alloc(64, this->size() * sizeof(T)); 
            this->data_ = aligned_data;
        }   
        cpu_vector(size_t size) : vector<T>(size) {
            //allocate memory for the vector 
            //use aligned memory allocation for better performance 
            if(this->data_ != nullptr)
                delete [] this->data_; //parent allocated with new 
            aligned_data    = (T *)aligned_alloc(64, size * sizeof(T)); 
            this->data_ = aligned_data;
        }   
        cpu_vector(const cpu_vector<T> &other) : vector<T>(other.size()) {
            if( this->data_ != nullptr)
                free(this->data_); //
            aligned_data    = (T *)aligned_alloc(64, other.size() * sizeof(T)); 
            this->data_ = aligned_data;
            std::copy(other.data_, other.data + other.size(), this->data_);
        }   
        cpu_vector(cpu_vector<T> &&other) : vector<T>(other.size()) {
            if( this->data_ != nullptr)
                delete [] this->data_; //
            this->data_ = other.data_;
            other.data_ = nullptr;
        }   
        virtual ~cpu_vector() {
            if (this->data_ != nullptr) {
                //aligned destructor
                free(this->data_);
                this->data_ = nullptr;
            }
        }   
        const cpu_vector<T> &operator=(const cpu_vector<T> &other) {
            if (this->data != nullptr) {
                free(this->data);
            }
            this->size = other.size;
            this->data_ = (T *)aligned_alloc(64, other.size * sizeof(T));
            std::copy(other.data_, other.data_ + other.size(), this->data_);
            return *this;
        }   
        const cpu_vector<T> &operator=(cpu_vector<T> &&other) {
            if (this->data_ != nullptr) {
                free(this->data_);
            }
            this->size = other.size;
            this->data_ = other.data_;
            other.data_ = nullptr;
            other.size = 0;
            return *this;
        }   
        //default operator * for vector multiplication  
        cpu_vector<T> operator*(const cpu_vector<T> &other) {
            return vector<T>::operator*(other);
        }   
        //default operator + for vector addition    
        cpu_vector<T> operator+(const cpu_vector<T> &other) {
            return vector<T>::operator+(other);
        }   
        //default operator - for vector subtraction 
        cpu_vector<T> operator-(const cpu_vector<T> &other) {
            return vector<T>::operator-(other);
        }   
        //default operator / for vector division    
        cpu_vector<T> operator/(const cpu_vector<T> &other) {
            return vector<T>::operator/(other);
        }   
        void set(size_t index, T value) {
            this->data_[index] = value;
        }   
        T get(size_t index) const {
            return this->data_[index];
        }
        void fill(T value) {
            std::fill(this->data, this->data + this->size(), value);
        }
        cpu_vector<T> conjugate() const {
            *this = vector<T>::conjugate();
            return *this;
        }
        cpu_vector<T> operator *(T scalar) {
            cpu_vector<T> result(this->size());
            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = this->data_[i] * scalar;
            }
            return result;
        }
        cpu_vector<T> operator /(T scalar) {
            cpu_vector<T> result(this->size());
            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = this->data_[i] / scalar;
            }
            return result;
        }
        cpu_vector<T> operator +(T scalar) {
            cpu_vector<T> result(this->size());
            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = this->data_[i] + scalar;
            }
            return result;
        }
        cpu_vector<T> operator -(T scalar) {
            cpu_vector<T> result(this->size());
            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = this->data_[i] - scalar;
            }
            return result;
        }
        //negation operator
        cpu_vector<T> operator-() {
            cpu_vector<T> result(this->size());
            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = -this->data_[i];
            }
            return result;
        }

        //override std::vector begin and end to use the aligned data 
        T* begin() {
            return this->data_;
        }
        T* end() {
            return this->data_ + this->size();
        }
        
        
    };
    //specialize the operators for float, double, long double 
    template <> cpu_vector<float> cpu_vector<float>::operator*(const cpu_vector<float> &other) {
       
       
                        std::cpu_vector<float> result(this->size()); 
                        
                        //use avx512 for float
                        //use __asm__ __volatile__ to avoid inconsistent operand constraints in an ‘asm’ error

                        //load the vector ptrs and the size to the registers 
                        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (this->data_));
                        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (other.data_));
                        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (result.data_)); 
                        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (this->size())); 
                        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (other.size()));

                        //use vfmadd231ps to multiply the two vectors and add the result to the third vector
                        __asm__ __volatile__ ("movq %rax, %xmm0"); 
                        __asm__ __volatile__ ("movq %rbx, %xmm1"); 
                        __asm__ __volatile__ ("vfmadd231ps %xmm1, %xmm0, %xmm2"); 
                        __asm__ __volatile__ ("movq %xmm2, %rcx"); 
                        //store the result in the result vector data ptr
                        __asm__ __volatile__ ("movq %rcx, %rdx");
                        //return the result vector
                        return result;
    } 
       
    template <> cpu_vector<double> cpu_vector<double>::operator*(const cpu_vector<double> &other) {
        std::cpu_vector<double> result(this->size()); 
        //use __asm__ __volatile__ to avoid inconsistent operand constraints in an ‘asm’ error
        //load the vector ptrs and the size to the registers

        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (this->data_));
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (other.data_));

        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (result.data_));
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (this->size()));
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (other.size()));
        
        //use vfmadd231pd to multiply the two vectors and add the result to the third vector
        __asm__ __volatile__ ("movq %rax, %xmm0");
        __asm__ __volatile__ ("movq %rbx, %xmm1");
        __asm__ __volatile__ ("vfmadd231pd %xmm1, %xmm0, %xmm2");
        __asm__ __volatile__ ("movq %xmm2, %rcx");
        //store the result in the result vector data ptr
        __asm__ __volatile__ ("movq %rcx, %rdx");
        //return the result vector
        return result;
    }

    template <> cpu_vector<long double> cpu_vector<long double>::operator*(const cpu_vector<long double> &other) {
        
        std::cpu_vector<long double> result(this->size()); 
        //use __asm__ __volatile__ to avoid inconsistent operand constraints in an ‘asm’ error
        //load the vector ptrs and the size to the registers
        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (this->data_));
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (other.data_));
        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (result.data_));
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (this->size()));
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (other.size()));
        //use vfmadd231pd to multiply the two vectors and add the result to the third vector
        __asm__ __volatile__ ("movq %rax, %xmm0");
        __asm__ __volatile__ ("movq %rbx, %xmm1");
        __asm__ __volatile__ ("vfmadd231pd %xmm1, %xmm0, %xmm2");
        __asm__ __volatile__ ("movq %xmm2, %rcx");
        //store the result in the result vector data ptr
        __asm__ __volatile__ ("movq %rcx, %rdx");
        //return the result vector
        return result;  

    }   
    template <> cpu_vector<float> cpu_vector<float>::operator+(const cpu_vector<float> &other) {
        __m128 a, b, c;
        cpu_vector<float> result(this->size());
        for (size_t i = 0; i < this->size(); i += 4) {
            a = _mm_load_ps(&this->data_[i]);
            b = _mm_load_ps(&other.data_[i]);
            c = _mm_add_ps(a, b);
            _mm_store_ps(&result.data_[i], c);
        }
        return result;
    }   
    template <> cpu_vector<double> cpu_vector<double>::operator+(const cpu_vector<double> &other) {
        __m256d a, b, c;
        cpu_vector<double> result(this->size());
        for (size_t i = 0; i < this->size(); i += 4) {
            a = _mm256_load_pd(&this->data_[i]);
            b = _mm256_load_pd(&other.data_[i]);
            c = _mm256_add_pd(a, b);
            _mm256_store_pd(&result.data_[i], c);
        }
        return result;
    }   
    template <> cpu_vector<long double> cpu_vector<long double>::operator+(const cpu_vector<long double> &other) {
        __m256d a, b, c;
        cpu_vector<long double> result(this->size());
        for (size_t i = 0; i < this->size(); i += 4) {
            a = _mm256_load_pd((double*)&this->data_[i]);
            b = _mm256_load_pd((double*)&other.data_[i]);
            c = _mm256_add_pd(a, b);
            _mm256_store_pd((double*)&result.data_[i], c);
        }
        return result;
    }   
    template <> cpu_vector<float> cpu_vector<float>::operator-(const cpu_vector<float> &other) {
        __m128 a, b, c;
        cpu_vector<float> result(this->size());
        for (size_t i = 0; i < this->size(); i += 4) {
            a = _mm_load_ps(&this->data_[i]);
            b = _mm_load_ps(&other.data_[i]);
            c = _mm_sub_ps(a, b);
            _mm_store_ps(&result.data_[i], c);
        }
        return result;
    }  

} // namespace std

namespace provallo {
    
    template <typename T>
    class cpu_matrix : public matrix<T> {
    public:

        //apply optimizations for the cpu_matrix class 

        //declare T *aligned_data; with linux aligned attribute
        T *aligned_data __attribute__((aligned(64)));

          //declare data_ to poiont on matrix<T>::data pointer             
        cpu_matrix(size_t size1, size_t size2) : matrix<T>(size1, size2) {
            //allocate memory for the matrix 
            //use aligned memory allocation for better performance 
            if(this->data_ != nullptr)
                delete [] this->data_; //parent allocated with new 
            aligned_data    = (T *)aligned_alloc(64, size1 * size2 * sizeof(T)); 
            this->data_ = aligned_data;

        }

        cpu_matrix(const cpu_matrix<T> &other) : matrix<T>(other.size1_, other.size2_) {
            
            if( this->data_ != nullptr)
                free(this->data_); //

            aligned_data    = (T *)aligned_alloc(64, other.size1_ * other.size2_ * sizeof(T)); 
            this->data_ = aligned_data;
            
            
            std::copy(other.data_, other.data + other.size1_ * other.size2_, this->data_);
        }

        cpu_matrix(cpu_matrix<T> &&other) : matrix<T>(other.size1_, other.size2_) {
            if( this->data_ != nullptr)
                delete [] this->data_; //
            this->data_ = other.data_;
            other.data_ = nullptr;
        }

        virtual ~cpu_matrix() {
            if (this->data_ != nullptr) {
                //aligned destructor
                free(this->data_);
                this->data_ = nullptr;
            }
        }

        const cpu_matrix<T> &operator=(const cpu_matrix<T> &other) {
            if (this->data != nullptr) {
                free(this->data);
            }
            this->size1 = other.size1;
            this->size2_= other.size2_;
            
            this->data_ = (T *)aligned_alloc(64, other.size1_ * other.size2_ * sizeof(T));

            std::copy(other.data_, other.data_ + other.size1_ * other.size2_, this->data_);
            
            return *this;
        }

        const cpu_matrix<T> &operator=(cpu_matrix<T> &&other) {
            if (this->data_ != nullptr) {
                free(this->data_);
            }
            this->size1_ = other.size1_;
            this->size2_= other.size2_;
            this->data_ = other.data_;
            other.data_ = nullptr;
            other.size1_ = 0;
            other.size2_ = 0;

            return *this;

        }
        //default operator * for matrix multiplication 
        cpu_matrix<T> operator*(const cpu_matrix<T> &other) {
            return matrix<T>::operator*(other);
        } 
        //default operator + for matrix addition 
        cpu_matrix<T> operator+(const cpu_matrix<T> &other) {
            return matrix<T>::operator+(other);
        }   
        //default operator - for matrix subtraction
        cpu_matrix<T> operator-(const cpu_matrix<T> &other) {
            return matrix<T>::operator-(other);
        }   
        //default operator / for matrix division    
        cpu_matrix<T> operator/(const cpu_matrix<T> &other) {
            return matrix<T>::operator/(other);
            
        }   
        void set(size_t row, size_t col, T value) {
            this->data_[row * this->size2_ + col] = value;
        }

        T get(size_t row, size_t col) const {
            return this->data_[row * this->size2_ + col];
        }

        void fill(T value) {
            std::fill(this->data, this->data + this->size1_ * this->size2_, value);
        }
        cpu_matrix<T> conjugate() const {
            
            *this = matrix<T>::conjugate();
            return *this;
            
        }
        cpu_matrix<T> operator *(T scalar) {
            cpu_matrix<T> result(this->size1_, this->size2_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[i * this->size2_ + j] = this->data_[i * this->size2_ + j] * scalar;
                }
            }
            return result;
        }
        cpu_matrix<T> operator /(T scalar) {
            cpu_matrix<T> result(this->size1_, this->size2_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[i * this->size2_ + j] = this->data_[i * this->size2_ + j] / scalar;
                }
            }
            return result;
        }   

        cpu_matrix<T> operator +(T scalar) {
            cpu_matrix<T> result(this->size1_, this->size2_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[i * this->size2_ + j] = this->data_[i * this->size2_ + j] + scalar;
                }
            }
            return result;
        }   
        cpu_matrix<T> operator -(T scalar) {
            cpu_matrix<T> result(this->size1_, this->size2_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[i * this->size2_ + j] = this->data_[i * this->size2_ + j] - scalar;
                }
            }
            return result;
        }   
        //negation operator
        cpu_matrix<T> operator-() {
            cpu_matrix<T> result(this->size1_, this->size2_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[i * this->size2_ + j] = -this->data_[i * this->size2_ + j];
                }
            }
            return result;
        }
        //elementwise multiplication 
        cpu_matrix<T> elementwise(const cpu_matrix<T> &other, T (*op)(T, T)) {
            cpu_matrix<T> result(this->size1_, this->size2_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[i * this->size2_ + j] = op(this->data_[i * this->size2_ + j], other.data_[i * this->size2_ + j]);
                }
            }
            return result;
        }   
        //inverse, adjoint, transpose, determinant
        cpu_matrix<T> inverse() const {
            cpu_matrix<T> result(this->size1_, this->size2_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[i * this->size2_ + j] = 1.0 / this->data_[i * this->size2_ + j];
                }
            }
            return result;
        }
        cpu_matrix<T> adjoint() const {
           //implement adjoint with conjugate and transpose 
            return this->conjugate().transpose();
            
        }
        cpu_matrix<T> transpose() const {
            cpu_matrix<T> result(this->size2_, this->size1_);
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    result.data_[j * result.size2_ + i] = this->data_[i * this->size2_ + j];
                }
            }
            return result;
        }
        T determinant() const {
            T det = 0;
            size_t min=std::min(this->size1_,this->size2_); 
            for (size_t i = 0; i < min; i++) {
                det += this->data_[i * this->size2_ + i];
            }
            det = std::abs(det)  * (this->size1_ == this->size2_ ? 1 : 0); 

            return det;
            
        }

        void randomize(T min, T max) {
            //use rdseed for better randomization 

            __asm__ __volatile__ ("rdseed %0" : "=r" (min)); 
            __asm__ __volatile__ ("rdseed %0" : "=r" (max));

            //get the random device and use mersenne twister for better randomization 
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(min, max);
            for (size_t i = 0; i < this->size1_ * this->size2_; i   ++) {
                this->data_[i] = dis(gen);
            }   
        }

        void print() const {
            for (size_t i = 0; i < this->size1_; i++) {
                for (size_t j = 0; j < this->size2_; j++) {
                    std::cout << this->data_[i * this->size2_ + j] << " ";
                }
                std::cout << std::endl;
            }
        }
       
      
              
        T& operator()(size_t row, size_t col) {
            return this->data_[row * this->size2_ + col];
        }
        const T& operator()(size_t row, size_t col) const {
            return this->data_[row * this->size2_ + col];
        }
        
        //operators *=, +=, -=, /=
        cpu_matrix<T> &operator*=(const cpu_matrix<T> &other) {
            *this = *this * other;
            return *this;
        }
        cpu_matrix<T> &operator+=(const cpu_matrix<T> &other) {
            *this = *this + other;
            return *this;
        }
        cpu_matrix<T> &operator-=(const cpu_matrix<T> &other) {
            *this = *this - other;
            return *this;
        }
        cpu_matrix<T> &operator/=(const cpu_matrix<T> &other) {
            *this = *this / other;
            return *this;
        }

        cpu_matrix<T> &operator*=(T scalar) {
            *this = *this * scalar;
            return *this;
        }
        cpu_matrix<T> &operator/=(T scalar) {
            *this = *this / scalar;
            return *this;
        }
        cpu_matrix<T> &operator+=(T scalar) {
            *this = *this + scalar;
            return *this;
        }
        cpu_matrix<T> &operator-=(T scalar) {
            *this = *this - scalar;
            return *this;
        }
          //specialize the operators for float, double, long double       
         
        
    };
        
     template <> cpu_matrix<float> cpu_matrix<float>::operator*(const cpu_matrix<float> &other) {
      __m128 a, b, c;
        cpu_matrix<float> result(this->size1_, other.size2_); 
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < other.size2_; j++) {
                c = _mm_setzero_ps();
                for (size_t k = 0; k < this->size2_; k += 4) {
                    a = _mm_load_ps(&this->data_[i * this->size2_ + k]);
                    b = _mm_load_ps(&other.data_[k * other.size2_ + j]);
                    c = _mm_add_ps(c, _mm_mul_ps(a, b));
                }
                _mm_store_ps(&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;

    }   
    //double 
    template <> cpu_matrix<double> cpu_matrix<double>::operator*(const cpu_matrix<double> &other) {
     
        cpu_matrix<double> result(this->size1_, other.size2_); 
        //use __asm__ __volatile__ to avoid inconsistent operand constraints in an ‘asm’ error 

        //load the matrix ptrs and the size to the registers 
        //use vfmadd231pd to multiply the two vectors and add the result to the third vector 
        //store the result in the result vector data ptr. 
        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (this->data_)); 
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (other.data_)); 
        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (result.data_)); 
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (this->size1_)); 
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (this->size2_)); 
        __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (result.size2_)); 
        __asm__ __volatile__ ("movq %0, %%r8" : : "r" (other.size2_)); 
        __asm__ __volatile__ ("movq %0, %%r9" : : "r" (other.size1_)); 
        //use vfmadd231pd to multiply the two vectors and add the result to the third vector 
        __asm__ __volatile__ ("movq %rax, %xmm0"); 
        __asm__ __volatile__ ("movq %rbx, %xmm1"); 
        __asm__ __volatile__ ("vfmadd231pd %xmm1, %xmm0, %xmm2"); 
        __asm__ __volatile__ ("movq %xmm2, %rcx"); 
        //store the result in the result matrix data ptr
        __asm__ __volatile__ ("movq %rcx, %rdx"); 
        __asm__ __volatile__ ("movq %rsi, %rcx");

        //return
        return result;

    }   
    //long double
    template <> cpu_matrix<long double> cpu_matrix<long double>::operator*(const cpu_matrix<long double> &other) {
      __m256d a, b, c;
        cpu_matrix<long double> result(this->size1_, other.size2_); 
        //use avx512 for long double
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < other.size2_; j++) {
                c = _mm256_setzero_pd();
                for (size_t k = 0; k < this->size2_; k += 8) {
                    a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + k]);
                    b = _mm256_load_pd((double*)&other.data_[k * other.size2_ + j]);
                    c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
                    _mm256_store_pd((double*)&result.data_[i * result.size2_ + k], c);
                }
               

            }
        }   

        return result;

    }   
    //float (m128)
    template <> cpu_matrix<float> cpu_matrix<float>::operator+(const cpu_matrix<float> &other) {
        cpu_matrix<float> result(this->size1_, this->size2_);
        __m128 a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_load_ps(&other.data_[i * this->size2_ + j]);
                c = _mm_add_ps(a, b);
                _mm_store_ps(&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;
    }   
    //double (m256d)
    template <> cpu_matrix<double> cpu_matrix<double>::operator+(const cpu_matrix<double> &other) {
        cpu_matrix<double> result(this->size1_, this->size2_);
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 8) {
                a = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd(&other.data_[i * this->size2_ + j]);
                c = _mm256_add_pd(a, b);
                _mm256_store_pd(&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;
    }   
    //long double (m512d) is only supported with avx512 compiler flags
    //use m256d for long double 
    template <> cpu_matrix<long double> cpu_matrix<long double>::operator+(const cpu_matrix<long double> &other) {
        cpu_matrix<long double> result(this->size1_, this->size2_);
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 8) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd((double*)&other.data_[i * this->size2_ + j]);
                c = _mm256_add_pd(a, b);
                _mm256_store_pd((double*)&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;
    }   
    //float (m128)  
    template <> cpu_matrix<float> cpu_matrix<float>::operator-(const cpu_matrix<float> &other) {
        cpu_matrix<float> result(this->size1_, this->size2_);
        __m128 a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_load_ps(&other.data_[i * this->size2_ + j]);
                c = _mm_sub_ps(a, b);
                _mm_store_ps(&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;
    }   
   

    //double (m256d)
    template <> cpu_matrix<double> cpu_matrix<double>::operator-(const cpu_matrix<double> &other) {
        cpu_matrix<double> result(this->size1_, this->size2_);
        __m256d a, b, c;
         //do it with assembly instruction without iteration :
         //assign a, b, c to the registers 
            //use vsubpd to subtract the two vectors 
            //store the result in c 
            //store c in the result matrix 
            //return the result matrix 
            //avoid 'inconsistent operand constraints in an ‘asm’ by assigning the members to  a,b,c
        for (size_t i = 0; i < this->size1_; i++) { 
            for (size_t j = 0; j < this->size2_; j += 8) {
                a = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd(&other.data_[i * this->size2_ + j]);
                c = _mm256_sub_pd(a, b);
                _mm256_store_pd(&result.data_[i * result.size2_ + j], c);
            }   
        }
            return result;
    } 

 
    //long double (m512d) is only supported with avx512 compiler flags  
    //use m256d for long double
    template <> cpu_matrix<long double> cpu_matrix<long double>::operator-(const cpu_matrix<long double> &other) {
        cpu_matrix<long double> result(this->size1_, this->size2_);
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 8) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd((double*)&other.data_[i * this->size2_ + j]);
                c = _mm256_sub_pd(a, b);
                _mm256_store_pd((double*)&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;
    }
    //float (m128)
    template <> cpu_matrix<float> cpu_matrix<float>::operator/(const cpu_matrix<float> &other) {
        cpu_matrix<float> result(this->size1_, this->size2_);
        __m128 a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_load_ps(&other.data_[i * this->size2_ + j]);
                c = _mm_div_ps(a, b);
                _mm_store_ps(&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;
    }   
    //double (m256d)    
    template <> cpu_matrix<double> cpu_matrix<double>::operator/(const cpu_matrix<double> &other) {
        cpu_matrix<double> result(this->size1_, this->size2_);
        //load the matrix ptrs to the registers 
        //use vdivpd to divide the two vectors 
        //store the result in result vector data ptr.
        
        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (this->data_)); 
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (other.data_)); 
        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (result.data_)); 
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (this->size1_)); 
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (this->size2_)); 
        __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (result.size2_)); 
        __asm__ __volatile__ ("movq %0, %%r8" : : "r" (other.size2_)); 
        __asm__ __volatile__ ("movq %0, %%r9" : : "r" (other.size1_)); 
        
        //use vdivpd to divide the two vectors and assign the results on the result vector 
        __asm__ __volatile__ ("movq %rax, %xmm0"); 
        __asm__ __volatile__ ("movq %rbx, %xmm1"); 
        __asm__ __volatile__ ("vdivpd %xmm1, %xmm0, %xmm2"); 
        __asm__ __volatile__ ("movq %xmm2, %rcx"); 
        //store the result in the result matrix
        __asm__ __volatile__ ("movq %rcx, %rdx"); 
        __asm__ __volatile__ ("movq %rsi, %rcx");
        __asm__ __volatile__ ("movq %rdi, %rsi");
        __asm__ __volatile__ ("movq %r8, %rdi");
        __asm__ __volatile__ ("movq %r9, %r8");
        __asm__ __volatile__ ("movq %rdx, %r9");
        __asm__ __volatile__ ("movq %rcx, %rdx");
        __asm__ __volatile__ ("movq %rsi, %rcx");
        __asm__ __volatile__ ("movq %rdi, %rsi");
        __asm__ __volatile__ ("movq %r8, %rdi");
        __asm__ __volatile__ ("movq %r9, %r8");
        __asm__ __volatile__ ("movq %rdx, %r9"); 
        
        return result;
    }    
    //long double (m512d) is only supported with avx512 compiler flags 
    //use m256d for long double
    template <> cpu_matrix<long double> cpu_matrix<long double>::operator/(const cpu_matrix<long double> &other) {
        cpu_matrix<long double> result(this->size1_, this->size2_);
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 8) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd((double*)&other.data_[i * this->size2_ + j]);
                c = _mm256_div_pd(a, b);
                _mm256_store_pd((double*)&result.data_[i * result.size2_ + j], c);
            }
        }
        return result;
    }   
    //specialize transpose with sse for float 
    template <> cpu_matrix<float> cpu_matrix<float>::transpose() const {
        cpu_matrix<float> result(this->size2_, this->size1_);
        __m128 a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
                _mm_store_ps(&result.data_[j * result.size2_ + i], b);
            }
        }
        return result;
    }   
    //specialize transpose with sse for double
    template <> cpu_matrix<double> cpu_matrix<double>::transpose() const {
        cpu_matrix<double> result(this->size2_, this->size1_);
        //rotate the matrix with avx2 
        __m256d a, b;
        size_t size = this->size1_ * this->size2_; 
        for (size_t i = 0; i <  size; i+=8) {
            a = _mm256_load_pd(&this->data_[i]);
            b = _mm256_permute4x64_pd(a, _MM_SHUFFLE(0, 1, 2, 3));
            _mm256_store_pd(&result.data_[i], b);   
        }
        //assign the res    ult to the matrix

        return result;
    }   
    //specialize transpose with sse for long double 
    template <> cpu_matrix<long double> cpu_matrix<long double>::transpose() const {
        cpu_matrix<long double> result(this->size2_, this->size1_);
        __m256d a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this ->size2_; j += 4) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_permute4x64_pd(a, _MM_SHUFFLE(0, 1, 2, 3));
                _mm256_store_pd((double*)&result.data_[j * result.size2_ + i], b);
            }
        }
        return result;
    }
    
    //specialization for see for conjugate,adjoint and inverse 
    template <> cpu_matrix<float> cpu_matrix<float>::conjugate() const {
        cpu_matrix<float> result(this->size1_, this->size2_);
        __m128 a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
                
                _mm_store_ps(&result.data_[i * result.size2_ + j], b);
            }
        }
        return result;
    }   
    template <> cpu_matrix<double> cpu_matrix<double>::conjugate() const {
        cpu_matrix<double> result(this->size1_, this->size2_);
        __m256d a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                b = _mm256_permute4x64_pd(a, _MM_SHUFFLE(0, 1, 2, 3));
                _mm256_store_pd(&result.data_[i * result.size2_ + j], b);
            }
        }
        return result;
    }   
    template <> cpu_matrix<long double> cpu_matrix<long double>::conjugate() const {
        cpu_matrix<long double> result(this->size1_, this->size2_);
        __m256d a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_permute4x64_pd(a, _MM_SHUFFLE(0, 1, 2, 3));
                _mm256_store_pd((double*)&result.data_[i * result.size2_ + j], b);
            }
        }
        return result;
    }   
    template <> cpu_matrix<float> cpu_matrix<float>::inverse() const {
        cpu_matrix<float> result(this->size1_, this->size2_);
        __m128 a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_div_ps(_mm_set1_ps(1.0), a);
                _mm_store_ps(&result.data_[i * result.size2_ + j], b);
            }
        }
        return result;
    }   
    template <> cpu_matrix<double> cpu_matrix<double>::inverse() const {
        cpu_matrix<double> result(this->size1_, this->size2_);
        __m256d a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                b = _mm256_div_pd(_mm256_set1_pd(1.0), a);
                _mm256_store_pd(&result.data_[i * result.size2_ + j], b);
            }
        }
        return result;
    }   
    template <> cpu_matrix<long double> cpu_matrix<long double>::inverse() const {
        cpu_matrix<long double> result(this->size1_, this->size2_);
        __m256d a, b;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_div_pd(_mm256_set1_pd(1.0), a);
                _mm256_store_pd((double*)&result.data_[i * result.size2_ + j], b);
            }
        }
        return result;
    }   
    template <> cpu_matrix<float> cpu_matrix<float>::adjoint() const {
        return this->conjugate().transpose();
    }   
    template <> cpu_matrix<double> cpu_matrix<double>::adjoint() const {
        return this->conjugate().transpose();
    }   
    template <> cpu_matrix<long double> cpu_matrix<long double>::adjoint() const {
        return this->conjugate().transpose();
    }   
    //specialize for sse for determinant    
    template <> float cpu_matrix<float>::determinant() const {
        if (this->size1_ != this->size2_) {
            throw std::invalid_argument("Matrix must be square");
        }
        float det = 0;
        __m128 a;
        for (size_t i = 0; i < this->size1_; i++) {
            a = _mm_load_ps(&this->data_[i * this->size2_ + i]);
            det += _mm_cvtss_f32(a);
        }
        return det;
    }   
    //specialize for sse for determinant    
    template <> double cpu_matrix<double>::determinant() const {
        if (this->size1_ != this->size2_) {
            throw std::invalid_argument("Matrix must be square");
        }
        double det = 0;
        __m256d a;
        for (size_t i = 0; i < this->size1_; i++) {
            a = _mm256_load_pd(&this->data_[i * this->size2_ + i]);
            det += _mm256_cvtsd_f64(a);
        }
        return det;
    }   
    //specialize for sse for determinant    
    template <> long double cpu_matrix<long double>::determinant() const {
        if (this->size1_ != this->size2_) {
            throw std::invalid_argument("Matrix must be square");
        }
        long double det = 0;
        __m256d a;
        for (size_t i = 0; i < this->size1_; i++) {
            a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + i]);
            det += _mm256_cvtsd_f64(a);
        }
        return det;
    }   

    //scalar negation   
    template <> cpu_matrix<float> cpu_matrix<float>::operator-() {
        cpu_matrix<float> result(this->size1_, this->size2_);
        __m128 a, b;
        a = _mm_set1_ps(-1.0);
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                b = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                _mm_store_ps(&result.data_[i * result.size2_ + j], _mm_mul_ps(b, a));
            }
        }
        return result;
    }   
    //scalar negation   
    template <> cpu_matrix<double> cpu_matrix<double>::operator-() {
        cpu_matrix<double> result(this->size1_, this->size2_);
        __m256d a, b;
        a = _mm256_set1_pd(-1.0);
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                b = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                _mm256_store_pd(&result.data_[i * result.size2_ + j], _mm256_mul_pd(b, a));
            }
        }
        return result;
    }   
    //scalar negation   
    template <> cpu_matrix<long double> cpu_matrix<long double>::operator-() {
        cpu_matrix<long double> result(this->size1_, this->size2_);
        __m256d a, b;
        _m_empty();


        a = _mm256_set1_pd(-1.0);
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                b = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                _mm256_store_pd((double*)&result.data_[i * result.size2_ + j], _mm256_mul_pd(b, a));
            }
        }
        return result;
    }    
    //optimization for operator *=, +=, -=, /=: 
    //use assembly instructions to avoid iteration
    //use the same approach as the elementwise multiplication
    //assign a, b, c to the registers


    //first operator *=
    template <> cpu_matrix<float> &cpu_matrix<float>::operator*=(const cpu_matrix<float> &other) {
                //use sse to multiply the two matrices 
                __m128 a, b, c; 
                //get vectorized pointers to the data 
                auto* ptr1 = this->data_;
                auto* ptr2 = other.data_; 
                 //use sse to multiply the two matrices
                for (size_t i = 0; i < this->size1_; i++) {
                    for (size_t j = 0; j < other.size2_; j++) {
                        c = _mm_setzero_ps();
                        for (size_t k = 0; k < this->size2_; k += 4) {
                            a = _mm_load_ps(&ptr1[i * this->size2_ + k]);
                            b = _mm_load_ps(&ptr2[k * other.size2_ + j]);
                            c = _mm_add_ps(c, _mm_mul_ps(a, b));
                            //store on ptr3+i*size2+j
                            auto* ptr3 = &this->data_[i * this->size2_ + j]; 
                            _mm_store_ps(ptr3, c);
                        }
                        
                    }
                }   
                return *this;

                

    }   
    //operator *=
    template <> cpu_matrix<double> &cpu_matrix<double>::operator*=(const cpu_matrix<double> &other) {
                //use avx2 to multiply the two matrices 
                //__m256d a, b, c; 
                //use inline assembly to avoid iteration 
                __asm__ __volatile__ ("movq %0, %%rax" : : "r" (this->size1_)); 
                __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (other.size2_)); 
                __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (this->size2_)); 
                __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (this->data_)); 
                __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (other.data_)); 
                __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (this->data_)); 
                __asm__ __volatile__ ("movq %0, %%r8" : : "r" (other.data_)); 
                __asm__ __volatile__ ("movq %0, %%r9" : : "r" (this->data_)); 

                //use avx2 to multiply the two matrices 
                //set the avx vector to zero
                __asm__ __volatile__ ("vxorpd %ymm0, %ymm0, %ymm0"); 
                //load the data from the first matrix 
                __asm__ __volatile__ ("vmovapd (%rdx), %ymm1"); 
                //load the data from the second matrix
                __asm__ __volatile__ ("vmovapd (%rsi), %ymm2"); 
                //multiply the two matrices
                __asm__ __volatile__ ("vmulpd %ymm1, %ymm2, %ymm3"); 
                //store the result in the first matrix
                __asm__ __volatile__ ("vmovapd %ymm3, (%rdi)"); 
                //return the first matrix

                    return *this;

    }   
    //operator *=   
    template <> cpu_matrix<long double> &cpu_matrix<long double>::operator*=(const cpu_matrix<long double> &other) {
                //use avx512 to multiply the two matrices 
             //use avx2 to multiply the two matrices 
                //__m256d a, b, c; 
                //use inline assembly to avoid iteration 
                __asm__ __volatile__ ("movq %0, %%rax" : : "r" (this->size1_)); 
                __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (other.size2_)); 
                __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (this->size2_)); 
                __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (this->data_)); 
                __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (other.data_)); 
                __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (this->data_)); 
                __asm__ __volatile__ ("movq %0, %%r8" : : "r" (other.data_)); 
                __asm__ __volatile__ ("movq %0, %%r9" : : "r" (this->data_)); 

                //use avx2 to multiply the two matrices 
                //set the avx vector to zero
                __asm__ __volatile__ ("vxorpd %ymm0, %ymm0, %ymm0"); 
                //load the data from the first matrix 
                __asm__ __volatile__ ("vmovapd (%rdx), %ymm1"); 
                //load the data from the second matrix
                __asm__ __volatile__ ("vmovapd (%rsi), %ymm2"); 
                //multiply the two matrices
                __asm__ __volatile__ ("vmulpd %ymm1, %ymm2, %ymm3"); 
                //store the result in the first matrix
                __asm__ __volatile__ ("vmovapd %ymm3, (%rdi)"); 
                //return the first matrix
                return *this;              

    }   
    //operator -= <float>
    template <> cpu_matrix<float> &cpu_matrix<float>::operator-=(const cpu_matrix<float> &other) {
        __m128 a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_load_ps(&other.data_[i * this->size2_ + j]);
                c = _mm_sub_ps(a, b);
                _mm_store_ps(&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator -= <double>
    template <> cpu_matrix<double> &cpu_matrix<double>::operator-=(const cpu_matrix<double> &other) {
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd(&other.data_[i * this->size2_ + j]);
                c = _mm256_sub_pd(a, b);
                _mm256_store_pd(&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator -= <long double>
    template <> cpu_matrix<long double> &cpu_matrix<long double>::operator-=(const cpu_matrix<long double> &other) {
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd((double*)&other.data_[i * this->size2_ + j]);
                c = _mm256_sub_pd(a, b);
                _mm256_store_pd((double*)&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator += <float>   
    template <> cpu_matrix<float> &cpu_matrix<float>::operator+=(const cpu_matrix<float> &other) {
        __m128 a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_load_ps(&other.data_[i * this->size2_ + j]);
                c = _mm_add_ps(a, b);
                _mm_store_ps(&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator += <double>  
    template <> cpu_matrix<double> &cpu_matrix<double>::operator+=(const cpu_matrix<double> &other) {
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd(&other.data_[i * this->size2_ + j]);
                c = _mm256_add_pd(a, b);
                _mm256_store_pd(&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator += <long double> 
    template <> cpu_matrix<long double> &cpu_matrix<long double>::operator+=(const cpu_matrix<long double> &other) {
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd((double*)&other.data_[i * this->size2_ + j]);
                c = _mm256_add_pd(a, b);
                _mm256_store_pd((double*)&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator /= <float>   
    template <> cpu_matrix<float> &cpu_matrix<float>::operator/=(const cpu_matrix<float> &other) {
        __m128 a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm_load_ps(&this->data_[i * this->size2_ + j]);
                b = _mm_load_ps(&other.data_[i * this->size2_ + j]);
                c = _mm_div_ps(a, b);
                _mm_store_ps(&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator /= <double>  
    template <> cpu_matrix<double> &cpu_matrix<double>::operator/=(const cpu_matrix<double> &other) {
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd(&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd(&other.data_[i * this->size2_ + j]);
                c = _mm256_div_pd(a, b);
                _mm256_store_pd(&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    //operator /= <long double> 
    template <> cpu_matrix<long double> &cpu_matrix<long double>::operator/=(const cpu_matrix<long double> &other) {
        __m256d a, b, c;
        for (size_t i = 0; i < this->size1_; i++) {
            for (size_t j = 0; j < this->size2_; j += 4) {
                a = _mm256_load_pd((double*)&this->data_[i * this->size2_ + j]);
                b = _mm256_load_pd((double*)&other.data_[i * this->size2_ + j]);
                c = _mm256_div_pd(a, b);
                _mm256_store_pd((double*)&this->data_[i * this->size2_ + j], c);
            }
        }
        return *this;
    }   
    
    //global scalar operations
    template <class T> cpu_matrix<T> operator*(const cpu_matrix<T> &matrix, T scalar) {
        cpu_matrix<T> result(matrix.size1_, matrix.size2_);
        for (size_t i = 0; i < matrix.size1_; i++) {
            for (size_t j = 0; j < matrix.size2_; j++) {
                result(i, j) = matrix(i, j) * scalar;
            }
        }
        return result;
    } 
    //specialize for sse for float 
    template <> cpu_matrix<float> operator*(const cpu_matrix<float> &matrix, float scalar) {
        cpu_matrix<float> result(matrix.size1(), matrix.size2());
        __m128 a, b;
        a = _mm_set1_ps(scalar);
        for (size_t i = 0; i < matrix.size1(); i++) {
            for (size_t j = 0; j < matrix.size2(); j += 4) {
                b = _mm_load_ps(&matrix.data()[i * matrix.size2() + j]);
                _mm_store_ps(&matrix.data()[i * result.size2() + j], _mm_mul_ps(b, a));
            }
        }
        return result;
    }
    //specialize for sse for double
    template <> cpu_matrix<double> operator*(const cpu_matrix<double> &matrix, double scalar) {
        cpu_matrix<double> result(matrix.size1(), matrix.size2()); 
        //assign scalar to a _m256d
 

        //use inline assembly to avoid iteration 
        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (matrix.size1())); 
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (matrix.size2()));
        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (matrix.data()));
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (result.data()));
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (scalar));
        __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (matrix.data())); 
        __asm__ __volatile__ ("movq %0, %%r8" : : "r" (result.data())); 

        //use avx2 to multiply the two matrices
        //set the avx vector to zero
        __asm__ __volatile__ ("vxorpd %ymm0, %ymm0, %ymm0"); 
        //load the data from the   matrix
 
        //multiply the matrix with the scalar value 
        __asm__ __volatile__ ("vmulpd %ymm0, %ymm0, %ymm0"); 
        //store the result in the first matrix 
        __asm__ __volatile__ ("vmovapd %ymm0, (%rdi)"); 
        //return the first matrix
        return result;
    }
    
    //global scalar operations
    template <class T> cpu_matrix<T> operator*(T scalar, const cpu_matrix<T> &matrix) {
        return matrix * scalar;
    }
    //global scalar operations
    template <class T> cpu_matrix<T> operator+(const cpu_matrix<T> &matrix, T scalar) {
        cpu_matrix<T> result(matrix.size1_, matrix.size2_);
        for (size_t i = 0; i < matrix.size1_; i++) {
            for (size_t j = 0; j < matrix.size2_; j++) {
                result(i, j) = matrix(i, j) + scalar;
            }
        }
        return result;
    }
    //specialize for sse for float
    template <> cpu_matrix<float> operator+(const cpu_matrix<float> &matrix, float scalar) {
        cpu_matrix<float> result(matrix.size1(), matrix.size2());
        __m128 a, b;
        a = _mm_set1_ps(scalar);
        for (size_t i = 0; i < matrix.size1(); i++) {
            for (size_t j = 0; j < matrix.size2(); j += 4) {
                b = _mm_load_ps(&matrix.data()[i * matrix.size2() + j]);
                _mm_store_ps(&result.data()[i * result.size2() + j], _mm_add_ps(b, a));
            }
        }
        return result;
    }
    //specialize for sse for double
    template <> cpu_matrix<double> operator+(const cpu_matrix<double> &matrix, double scalar) {
        cpu_matrix<double> result(matrix.size1(), matrix.size2());
        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (matrix.size1())); 
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (matrix.size2()));
        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (matrix.data()));
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (result.data()));
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (scalar));
        __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (matrix.data())); 
        __asm__ __volatile__ ("movq %0, %%r8" : : "r" (result.data()));
        //use avx2 to multiply the two matrices
        //set the avx vector to zero
        __asm__ __volatile__ ("vxorpd %ymm0, %ymm0, %ymm0"); 
        //load the data from the   matrix 
        __asm__ __volatile__ ("vmovapd (%rcx), %ymm1"); 
        //load the data from the second matrix 
        __asm__ __volatile__ ("vmovapd (%rdx), %ymm2"); 
        // add the scalar to the othe matrix and store on this matrix
        __asm__ __volatile__ ("vaddpd %ymm1, %ymm2, %ymm3"); 
        //store the result in the first matrix 
        __asm__ __volatile__ ("vmovapd %ymm3, (%rdi)"); 

        //return the first matrix
        return result;
    }
    //global scalar operations
    template <class T> cpu_matrix<T> operator-(const cpu_matrix<T> &matrix, T scalar) {
        cpu_matrix<T> result(matrix.size1_, matrix.size2_);
        for (size_t i = 0; i < matrix.size1_; i++) {
            for (size_t j = 0; j < matrix.size2_; j++) {
                result(i, j) = matrix(i, j) - scalar;
            }
        }
        return result;
    }
    //specialize for sse for float
    template <> cpu_matrix<float> operator-(const cpu_matrix<float> &matrix, float scalar) {
        cpu_matrix<float> result(matrix.size1(), matrix.size2());
        __m128 a, b;
        a = _mm_set1_ps(scalar);
        for (size_t i = 0; i < matrix.size1(); i++) {
            for (size_t j = 0; j < matrix.size2(); j += 4) {
                b = _mm_load_ps(&matrix.data()[i * matrix.size2() + j]);
                _mm_store_ps(&result.data()[i * result.size2() + j], _mm_sub_ps(b, a));
            }
        }
        return result;
    }
    //specialize for sse for double
    template <> cpu_matrix<double> operator-(const cpu_matrix<double> &matrix, double scalar) {
        cpu_matrix<double> result(matrix.size1(), matrix.size2());
        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (matrix.size1()));
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (matrix.size2()));
        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (matrix.data()));
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (result.data()));
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (scalar));
        __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (matrix.data()));
        __asm__ __volatile__ ("movq %0, %%r8" : : "r" (result.data())); 
        //substract the matrix with the scalar value 
        __asm__ __volatile__ ("vsubpd %ymm1, %ymm2, %ymm3"); 
        //store the result in the first matrix
        __asm__ __volatile__ ("vmovapd %ymm3, (%rdi)"); 
        //return the first matrix
        return result;

    }

    //global scalar operations
    template <class T> cpu_matrix<T> operator/(const cpu_matrix<T> &matrix, T scalar) {
        cpu_matrix<T> result(matrix.size1_, matrix.size2_);
        for (size_t i = 0; i < matrix.size1_; i++) {
            for (size_t j = 0; j < matrix.size2_; j++) {
                result(i, j) = matrix(i, j) / scalar;
            }
        }
        return result;
    }
    //specialize for sse for float
    template <> cpu_matrix<float> operator/(const cpu_matrix<float> &matrix, float scalar) {
        cpu_matrix<float> result(matrix.size1(), matrix.size2());
        __m128 a, b;
        a = _mm_set1_ps(scalar);
        for (size_t i = 0; i < matrix.size1(); i++) {
            for (size_t j = 0; j < matrix.size2(); j += 4) {
                b = _mm_load_ps(&matrix.data()[i * matrix.size2() + j]);
                _mm_store_ps(&result.data()[i * result.size2() + j], _mm_div_ps(b, a));
            }
        }
        return result;
    }
    //specialize for sse for double
    template <> cpu_matrix<double> operator/(const cpu_matrix<double> &matrix, double scalar) {
        cpu_matrix<double> result(matrix.size1(), matrix.size2()); 
        //assign scalar to a _m256d with inline assembly. 
        __asm__ __volatile__ ("movq %0, %%rax" : : "r" (matrix.size1())); 
        __asm__ __volatile__ ("movq %0, %%rbx" : : "r" (matrix.size2()));
        __asm__ __volatile__ ("movq %0, %%rcx" : : "r" (matrix.data()));
        __asm__ __volatile__ ("movq %0, %%rdx" : : "r" (result.data()));
        __asm__ __volatile__ ("movq %0, %%rsi" : : "r" (scalar));
        __asm__ __volatile__ ("movq %0, %%rdi" : : "r" (matrix.data()));
        __asm__ __volatile__ ("movq %0, %%r8" : : "r" (result.data()));
        //use avx2 to divide the matrix with the scalar value
        //set the avx vector to zero
        __asm__ __volatile__ ("vxorpd %ymm0, %ymm0, %ymm0"); 
        //load the data from the   matrix   
        __asm__ __volatile__ ("vmovapd (%rcx), %ymm1"); 
        //divide and store on result matrix 
        __asm__ __volatile__ ("vdivpd %ymm1, %ymm0, %ymm2"); 
        //store the result in the first matrix
        __asm__ __volatile__ ("vmovapd %ymm2, (%rdi)");
        //return the first matrix

        return result;
    }
    //global scalar operations
    template <class T> cpu_matrix<T> operator+(T scalar, const cpu_matrix<T> &matrix) {
        return matrix + scalar;
    }
    //global scalar operations  
    template <class T> cpu_matrix<T> operator-(T scalar, const cpu_matrix<T> &matrix) {
        return matrix - scalar;
    }
    //global scalar operations
    //global scalar operations
    template <class T> cpu_matrix<T> operator/(T scalar, const cpu_matrix<T> &matrix) {
        return matrix / scalar;
    }

    //global matrix operations
    template <class T> cpu_matrix<T> operator*(const cpu_matrix<T> &matrix1, const cpu_matrix<T> &matrix2) {
        
        return matrix1 * matrix2;
    }   

    //global matrix operations
    template <class T> cpu_matrix<T> operator+(const cpu_matrix<T> &matrix1, const cpu_matrix<T> &matrix2) {
        return matrix1 + matrix2;
    }
    //global matrix operations
    template <class T> cpu_matrix<T> operator-(const cpu_matrix<T> &matrix1, const cpu_matrix<T> &matrix2) {
        return matrix1 - matrix2;
    }
    //global matrix operations
    template <class T> cpu_matrix<T> operator/(const cpu_matrix<T> &matrix1, const cpu_matrix<T> &matrix2) {
        return matrix1 / matrix2;
    }
    
    

}//namespace provallo
#endif 