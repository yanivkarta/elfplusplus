/*
 * matrix.h
 *
 *  Created on: May 11, 2021
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_MATRIX_H_
#define DECISION_ENGINE_MATRIX_H_

#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <random>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <functional>
#include <type_traits>
#include <iterator>
#include <initializer_list>
#include <utility>
#include <string>
#include <sstream>
#include <fstream>


#include "utils.h" //real_t
// simple matrix and NRC/NRCPP/eigen compat matrix.
  
namespace provallo
{
   
  // fixed Array performs as std::array:
  template <class T, std::size_t N>
  class fixed_array
  {
  public:
    T elems[N]; // fixed-size fixed_array of elements of type T

  public:
    // type definitions
    typedef T value_type;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef T &reference;
    typedef const T &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // iterator support
    iterator
    begin()
    {
      return elems;
    }
    const_iterator
    begin() const
    {
      return elems;
    }
    const_iterator
    cbegin() const
    {
      return elems;
    }

    iterator
    end()
    {
      return elems + N;
    }
    const_iterator
    end() const
    {
      return elems + N;
    }
    const_iterator
    cend() const
    {
      return elems + N;
    }

    // Default constructor
    fixed_array()
    {
      std::fill(elems, elems + N, T());
    }

    // reverse iterator support
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    reverse_iterator
    rbegin()
    {
      return reverse_iterator(end());
    }
    const_reverse_iterator
    rbegin() const
    {
      return const_reverse_iterator(end());
    }
    const_reverse_iterator
    crbegin() const
    {
      return const_reverse_iterator(end());
    }

    reverse_iterator
    rend()
    {
      return reverse_iterator(begin());
    }
    const_reverse_iterator
    rend() const
    {
      return const_reverse_iterator(begin());
    }
    const_reverse_iterator
    crend() const
    {
      return const_reverse_iterator(begin());
    }

    // operator[]
    reference
    operator[](size_type i)
    {
      assert(i < N);
      return elems[i];
    }

    const_reference
    operator[](size_type i) const
    {
      assert(i < N);
      return elems[i];
    }
    const fixed_array<T, N> &operator*(const T &value) const;
    const fixed_array<T, N> &operator/(const T &value) const;

    // at() with range check
    reference
    at(size_type i)
    {
      rangecheck(i);
      return elems[i];
    }
    const_reference
    at(size_type i) const
    {
      rangecheck(i);
      return elems[i];
    }

    // front() and back()
    reference
    front()
    {
      return elems[0];
    }

    const_reference
    front() const
    {
      return elems[0];
    }

    reference
    back()
    {
      return elems[N - 1];
    }

    const_reference
    back() const
    {
      return elems[N - 1];
    }

    // size is constant
    static size_type
    size()
    {
      return N;
    }
    static bool
    empty()
    {
      return false;
    }
    static size_type
    max_size()
    {
      return N;
    }
    enum
    {
      static_size = N
    };

    // swap (note: linear complexity)
    void
    swap(fixed_array<T, N> &y)
    {
      for (size_type i = 0; i < N; ++i)
        std::swap(elems[i], y.elems[i]);
    }

    // direct access to data (read-only)
    const T *
    data() const
    {
      return elems;
    }
    T *
    data()
    {
      return elems;
    }

    // use fixed_array as C fixed_array (direct read/write access to data)
    T *
    c_fixed_array()
    {
      return elems;
    }
    //
    

    // assignment with type conversion
    template <typename T2>
    fixed_array<T, N> &
    operator=(const fixed_array<T2, N> &rhs)
    {
      std::copy(rhs.begin(), rhs.end(), begin());
      return *this;
    }

    // assign one value to all elements
    void
    assign(const T &value)
    {
      fill(value); // A synonym for fill
    }

    void
    fill(const T &value)
    {
      std::fill_n(begin(), size(), value);
    }

  private:
    // check range (may be private because it is static)
    static void
    rangecheck(size_type i)
    {
      if (i >= size())
      {
        std::out_of_range e("fixed_array<>: index out of range");
        throw e;
      }
    }
  };

  // comparisons
  template <class T, std::size_t N>
  bool
  operator==(const fixed_array<T, N> &x, const fixed_array<T, N> &y)
  {
    return std::equal(x.begin(), x.end(), y.begin());
  }
  template <class T, std::size_t N>
  bool
  operator<(const fixed_array<T, N> &x, const fixed_array<T, N> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }
  template <class T, std::size_t N>
  bool
  operator!=(const fixed_array<T, N> &x, const fixed_array<T, N> &y)
  {
    return !(x == y);
  }
  template <class T, std::size_t N>
  bool
  operator>(const fixed_array<T, N> &x, const fixed_array<T, N> &y)
  {
    return y < x;
  }
  template <class T, std::size_t N>
  bool
  operator<=(const fixed_array<T, N> &x, const fixed_array<T, N> &y)
  {
    return !(y < x);
  }
  template <class T, std::size_t N>
  bool
  operator>=(const fixed_array<T, N> &x, const fixed_array<T, N> &y)
  {
    return !(x < y);
  }

  // global swap()
  template <class T, std::size_t N>
  inline void
  swap(fixed_array<T, N> &x, fixed_array<T, N> &y)
  {
    x.swap(y);
  }

  template <class T, std::size_t N>
  fixed_array<T, N>
  operator*(const T &scalar, const fixed_array<T, N> &b)
  {
    fixed_array<T, N> result;
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }

  template <class T, std::size_t N>
  fixed_array<T, N>
  operator*(const fixed_array<T, N> &a, const fixed_array<T, N> &b)
  {
    fixed_array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::multiplies<T>());
    return result;
  }

  template <class T, std::size_t N>
  fixed_array<T, N>
  operator+(const fixed_array<T, N> &a, const fixed_array<T, N> &b)
  {
    fixed_array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  template <class T, std::size_t N>
  fixed_array<T, N>
  operator-(const fixed_array<T, N> &a, const fixed_array<T, N> &b)
  {
    fixed_array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::minus<T>());
    return result;
  }

  template <class T, std::size_t N>
  T dot(const fixed_array<T, N> &a, const fixed_array<T, N> &b)
  {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
  }

  template <class T, std::size_t N>
  std::ostream &
  operator<<(std::ostream &out, const fixed_array<T, N> &a)
  {
    std::size_t i = 0;
    out << std::string("(");
    for (; i < N - 1; ++i)
      out << a[i] << ',';
    out << a[i] << std::string(")");
    return out;
  }

  // comparisons
  template <class T>
  bool
  operator==(const std::vector<T> &x, const std::vector<T> &y)
  {
    return std::equal(x.begin(), x.end(), y.begin());
  }
  template <class T>
  bool
  operator<(const std::vector<T> &x, const std::vector<T> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }
  template <class T>
  bool
  operator!=(const std::vector<T> &x, const std::vector<T> &y)
  {
    return !(x == y);
  }
  template <class T>
  bool
  operator>(const std::vector<T> &x, const std::vector<T> &y)
  {
    return y < x;
  }
  template <class T>
  bool
  operator<=(const std::vector<T> &x, const std::vector<T> &y)
  {
    return !(y < x);
  }
  template <class T>
  bool
  operator>=(const std::vector<T> &x, const std::vector<T> &y)
  {
    return !(x < y);
  }

  template <class T>
  std::vector<T>
  operator*(const T &scalar, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }

  template <class T>
  std::vector<T>
  operator*(const std::vector<T> &a, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::multiplies<T>());
    return result;
  }

  template <class T>
  std::vector<T>
  operator+(const std::vector<T> &a, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  template <class T>
  std::vector<T>
  operator-(const std::vector<T> &a, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::minus<T>());
    return result;
  }

  template <class T>
  T dot(const std::vector<T> &a, const std::vector<T> &b)
  {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
  }

  template <class T>
  std::ostream &
  operator<<(std::ostream &out, const std::vector<T> &a)
  {
    if(a.size()>0){

    std::size_t i = 0;
    out << std::string("(");
    for (; i < a.size() - 1; ++i)
      out << a[i] << std::string(",");
    out << a[i] << std::string(")");
    }
    else
    {
      out << std::string("()");
    } 
    return out;
  }

  class matrix_base;

  typedef std::shared_ptr<matrix_base> matrix_ptr;

  //matrix_base class for dataset oriented matrix operations 

  
  class matrix_base
  {
  private:
    double *data;

  public:
    size_t _rows;
    size_t _cols;
    double &
    pos(size_t row, size_t col)
    {
      return data[row * _cols + col];
    }
    const double & pos(size_t row,size_t col)const 
    {
      return data[row * _cols + col];
    }
    inline  size_t rows()const{return _rows;}
    inline  size_t cols()const{return _cols;}

    inline double & operator()(size_t row, size_t col)
    {
      return data[row * _cols + col];
    }

    inline double & operator()(size_t row, size_t col)const
    {
      return data[row * _cols + col];
    }
    matrix_base(size_t rows, size_t cols);
    matrix_base(const matrix_base &m);
    virtual ~matrix_base();
    virtual void
    clear();
  };
  //note : matrix is not derived from matrix_base, since matrix_base is  dataset oriented and matrix<T>is comutation oriented.
  // 
  template <class T>
  class matrix
  {
    typedef T *ptr;

  public:
    typedef T value_type;
    typedef T *fixed_array_type;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef T &reference;
    typedef const T &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;



    matrix() : size1_(0), size2_(0), data_(nullptr)
    { 
        //empty matrix
        
    }
    matrix(size_type size1, size_type size2) : size1_(size1), size2_(size2), data_(nullptr)
    {
      size_t a = size1 * size2;
      if (a>0)
        {
          
          data_ = new T[size1 * size2];
          std::fill(data_, data_ + size1 * size2, T());
          
        }
     
    }
    matrix(size_type size1, size_type size2, const T *value) : size1_(size1), size2_(size2), data_(nullptr)
    {
       size_t a_size = size1_ * size2_;
      if(a_size>0)
      {
           data_ = new T[a_size];
          std::copy(value, value +  a_size, data_);
      }
 
    }
    matrix(matrix_base &m) : size1_(m.rows()), size2_(m.cols()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          element(i, j) = m(i, j);


     }

    matrix(const matrix &m) : size1_(m.size1_), size2_(m.size2_), data_(nullptr)
    {
      size_t a_size = size1_ * size2_;
      if  (a_size>0)
      {
          data_ = new T[a_size];
          std::copy(m.data_, m.data_ + a_size, data_);
      }
     }
    matrix( matrix &&move_matrix) noexcept :size1_(std::move(move_matrix.size1_)), size2_(std::move(move_matrix.size2_)), data_(std::move(move_matrix.data_))
    {
     //clear the moved matrix-
      move_matrix.size1_ = 0;
      move_matrix.size2_ = 0;
      move_matrix.data_ = nullptr;
 
    }
    matrix(const std::initializer_list<T> &list) : size1_(list.size()), size2_(1), data_(nullptr)
    {
      data_ = new T[size1_];
      std::copy(list.begin(), list.end(), data_);
     }
    matrix(const std::initializer_list<std::initializer_list<T>> &list) : size1_(list.size()), size2_(list.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : list)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
     }
    matrix(const std::vector<T> &vec) : size1_(vec.size()), size2_(1), data_(nullptr)
    {
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
     }
    matrix(const std::vector<std::vector<T>> &vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
     }
    matrix(const std::vector<std::vector<T>> &&vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
     }
    matrix(const std::vector<T> &&vec) : size1_(vec.size()), size2_(1), data_(nullptr)
    {
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
     }
    matrix(const std::vector<std::initializer_list<T>> &vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
     }
    matrix(const std::vector<std::initializer_list<T>> &&vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
     }

    matrix(const T* dat_ptr,size_t size1,size_t size2):size1_(size1),size2_(size2),data_(nullptr)
    {
      size_t a_size=size1_*size2_;
      if(a_size>0)
      {
        data_ = new T[a_size];
        std::copy(dat_ptr,dat_ptr + a_size, data_);
      }
     }

    operator matrix_base() const
    {
      matrix_base m(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          m(i, j) = (real_t)element(i, j);
      return m;
    }


    #ifdef SIM_MATRIX_ITERATOR
     
    class const_iterator : public std::iterator<std::random_access_iterator_tag, T>
    {
      private:

      T *ptr_;
      size_t size1_;
      size_t size2_;
      size_t i_;
      size_t j_;

      public:
      const_iterator(T *ptr, size_t size1, size_t size2, size_t i, size_t j) : ptr_(ptr), size1_(size1), size2_(size2), i_(i), j_(j)
      {
      }
      const_iterator(const const_iterator &it) : ptr_(it.ptr_), size1_(it.size1_), size2_(it.size2_), i_(it.i_), j_(it.j_)
      {
      } 
      const_iterator(const_iterator &&it) : ptr_(it.ptr_), size1_(it.size1_), size2_(it.size2_), i_(it.i_), j_(it.j_)
      {
      }
      const_iterator &operator=(const const_iterator &it)
      {
        ptr_ = it.ptr_;
        size1_ = it.size1_;
        size2_ = it.size2_;
        i_ = it.i_;
        j_ = it.j_;
        return *this;
      }
      const_iterator &operator=(const_iterator &&it)
      {
        ptr_ = it.ptr_;
        size1_ = it.size1_;
        size2_ = it.size2_;
        i_ = it.i_;
        j_ = it.j_;
        return *this;
      }
      const_iterator &operator++()
      {
        if (j_ < size2_ - 1)
          ++j_;
        else
        {
          j_ = 0;
          ++i_;
        }
        return *this;
      }
      const_iterator operator++(int)
      {
        const_iterator tmp(*this);
        operator++();
        return tmp;
      }
      const_iterator &operator--()
      {
        if (j_ > 0)
          --j_;
        else
        {
          j_ = size2_ - 1;
          --i_;
        }
        return *this;
      }
      const_iterator operator--(int)
      {
        const_iterator tmp(*this);
        operator--();
        return tmp;
      }
      const_iterator &operator+=(size_t n)
      {
        size_t k = i_ * size2_ + j_ + n;
        i_ = k / size2_;
        j_ = k % size2_;
        return *this;
      }
      const_iterator &operator-=(size_t n)
      {
        size_t k = i_ * size2_ + j_ - n;
        i_ = k / size2_;
        j_ = k % size2_;
        return *this;
      }
      const_iterator operator+(size_t n) const
      {
        const_iterator tmp(*this);
        tmp += n;
        return tmp;
      }
      const_iterator operator-(size_t n) const
      {
        const_iterator tmp(*this);
        tmp -= n;
        return tmp;
      }
      size_t operator-(const const_iterator &it) const
      {
        return i_ * size2_ + j_ - it.i_ * size2_ - it.j_;
      }
      bool operator==(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ == it.j_;
      }
      bool operator!=(const const_iterator &it) const
      {
        return ptr_ != it.ptr_ || i_ != it.i_ || j_ != it.j_;
      }
      bool operator<(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ < it.j_;
      }
      bool operator>(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ > it.j_;
      }
      bool operator<=(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ <= it.j_;
      }
      bool operator>=(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ >= it.j_;
      }
      const T &operator*() const
      {
        return *(ptr_ + i_ * size2_ + j_);
      }
      const T *operator->() const
      {
        return ptr_ + i_ * size2_ + j_;
      }
      const T &operator[](size_t n) const
      {
        return *(ptr_ + i_ * size2_ + j_ + n);
      }

    };

    const_iterator begin() const
    {
      return const_iterator(data_, size1_, size2_, 0, 0);
    }
    const_iterator end() const
    {
      return const_iterator(data_, size1_, size2_, size1_, 0);
    }
    const_iterator cbegin() const
    {
      return const_iterator(data_, size1_, size2_, 0, 0);
    }
    const_iterator cend() const
    {
      return const_iterator(data_, size1_, size2_, size1_, 0);
    }
    #endif // ITERATOR

    //rowwise & colwise implementation
    template <typename F = real_t>
    matrix<T> rowwise(F f = 0.0) const
    {
      matrix<T> tmp(size1_, 1);
      for (size_t i = 0; i < size1_; i++)
        tmp(i, 0) = f(data_[i * size2_]);
      return tmp;
    }
    template <typename F>
    matrix<T> colwise(F f) const
    {
      matrix<T> tmp(1, size2_);
      for (size_t i = 0; i < size2_; i++)
        tmp(0, i) = f(data_[i]);
      return tmp;
    }

    //empty colwise() & rowwise() implementation
    matrix<T> rowwise() const
    {
      matrix<T> tmp(size1_, 1);
      for (size_t i = 0; i < size1_; i++)
        tmp(i, 0) = data_[i * size2_];
      return tmp;
    }
    matrix<T> colwise() const
    {
      matrix<T> tmp(1, size2_);
      for (size_t i = 0; i < size2_; i++)
        tmp(0, i) = data_[i];
      return tmp;
    }

    //cwiseAbs implementation for neural layers
    matrix<T> cwiseAbs() const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = std::abs(data_[i * size2_ + j]);
      return tmp;
    }
    //maxCoeff  
    T maxCoeff() const
    {
      T max = data_[0];
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          if (data_[i * size2_ + j] > max)
            max = data_[i * size2_ + j];
      return max;
    }
    //minCoeff
    T minCoeff() const
    {
      T min = data_[0];
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          if (data_[i * size2_ + j] < min)
            min = data_[i * size2_ + j];
      return min;
    }
    // max 
    matrix<T> max(const matrix<T> &m) const
    {
      assert(size1_ == m.size1_ && size2_ == m.size2_);
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
        {
          if (data_[i * size2_ + j] > m(i, j))
            tmp(i, j) = data_[i * size2_ + j];
          else
            tmp(i, j) = m(i, j);
        }
      return tmp;
    }
    // min
    matrix<T> min(const matrix<T> &m) const
    {
      assert(size1_ == m.size1_ && size2_ == m.size2_);
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
        {
          if (data_[i * size2_ + j] < m(i, j))
            tmp(i, j) = data_[i * size2_ + j];
          else
            tmp(i, j) = m(i, j);
        }
      return tmp;
    }
    T max()const 
    {
      T max = data_[0];
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          if (data_[i * size2_ + j] > max)
            max = data_[i * size2_ + j];
      return max;
    } 
    T min()const 
    {
      T min = data_[0];
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          if (data_[i * size2_ + j] < min)
            min = data_[i * size2_ + j];
      return min;
    }
    matrix<T> exp()const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = std::exp(data_[i * size2_ + j]);
      return tmp;
    }

    matrix<T> log()const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = std::log(data_[i * size2_ + j]);
      return tmp;
    }
    matrix<T> pow(const T &value)const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = std::pow(data_[i * size2_ + j], value);
      return tmp;
    }

    //cwiseProduct of two matrices
    //matrix<T> cwiseProduct implementation: calculates the columnwise product of two matrices
    matrix<T> cwiseProduct(const matrix<T> &m) const
    {
      size_t min_size1 = std::min(size1_, m.size1_);
      size_t min_size2 = std::min(size2_, m.size2_);
      matrix<T> tmp(min_size1, min_size2);
       for (size_t i = 0; i < min_size1; i++)
        for (size_t j = 0; j < min_size2; j++)
          tmp(i, j) = data_[i *min_size2 + j] * m(i, j);
      return tmp;
    }
    //matrix<T> cwiseQuotient implementation: calculates the columnwise quotient of two matrices
    matrix<T> cwiseQuotient(const matrix<T> &m) const
    {
      assert(size1_ == m.size1_ && size2_ == m.size2_);
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = data_[i * size2_ + j] / m(i, j);
      return tmp;
    }
    //matrix<T> cwiseProduct implementation: calculates the columnwise product of two matrices
    matrix<T> cwiseProduct(const matrix<T> &m, size_t row) const
    {
      assert(size1_ == m.size1_ && size2_ == m.size2_);
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = data_[i * size2_ + j] * m(row, j);
      return tmp;
    }
    //cwiseProduct of a matrix and T  
    matrix<T> cwiseProduct(const T &value) const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = data_[i * size2_ + j] * value;
      return tmp;
    }

    T asym_similarity(const matrix<T> &m) const
    {
     // size is not equal, use minimum size 
      size_t min_size1 = std::min(size1_, m.size1_); 
      size_t min_size2 = std::min(size2_, m.size2_); 
      T sim = 0; 
      for (size_t i = 0; i < min_size1; i++)
        for (size_t j = 0; j < min_size2; j++)
          sim += data_[i * size2_ + j] * m(i, j);
      return sim;
      
    }
    //similarity between two matrices 
    T sym_similarity(const matrix<T> &m) const
    {
      assert(size1_ == m.size1_ && size2_ == m.size2_);
      T sim = 0;
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          sim += data_[i * size2_ + j] * m(i, j);
      return sim;
    } 
    T similarity(const matrix<T> &m) const
    {
      if(size1_==m.size1_ && size2_==m.size2_)
        return sym_similarity(m);
      else
        return asym_similarity(m);
    } 
 
    T det()const
    {
      T det(T(0));
      //return determinant of the matrix :
      if(size1_!=size2_)
      {
        std::cerr<<"Matrix is not square, determinant is not defined"<<std::endl;
        return 0;
      }
      else
      {
        if(size1_==1)
          return data_[0];
        else if(size1_==2)
          return data_[0]*data_[3]-data_[1]*data_[2];
        else
        {
          
          size_t size1=size1_; 
          size_t size2=size2_; 
          for(size_t i=0;i<size1;i++)
          {
            matrix<T> tmp(size1-1,size2-1); 
            for(size_t j=1;j<size1;j++)
              for(size_t k=0;k<size2;k++)
              {
                if(k<i)
                  tmp(j-1,k)=data_[j*size2_+k];
                else if(k>i)
                  tmp(j-1,k-1)=data_[j*size2_+k];
              } 
            
            if(i%2==0)  
              det+=data_[i*size2]*tmp.det();
            else
              det-=data_[i*size2]*tmp.det();
            
          }
          
        }
      } 
      return det;
    }
    //inverse of a matrix (const version)
    matrix<T> inverse() const
    {
      
      if(size1_!=size2_)
      {
        //make it sqrt of total size 
        size_t size = std::sqrt(size1_ * size2_); 
        matrix<T> tmp(size, size); 
        matrix<T> adj(size, size); 
        T det = this->det(); 

        if (det == 0)
          return tmp; 
        adj = adjoint();
        tmp = adj.cwiseProduct(1 / det);

        return tmp;
      } 
      else
      {
        matrix<T> tmp(size1_, size2_);
        matrix<T> adj(size1_, size2_);
        T det = this->det();
        if (det == 0)
          return tmp;
        adj = adjoint();
        tmp = adj.cwiseProduct(1 / det);
        return tmp;
      } 
      
    } 
    //remove column - copy matrix twice, non const version of the function is more efficient

    matrix<T> remove_column(size_t col) const
    {
      if(col>=size2_)
        return *this;

      matrix<T> tmp(size1_, size2_ - 1);
      for (size_t i = 0; i < size1_; i++)
      {
        for (size_t j = 0; j < col; j++)
          tmp(i, j) = data_[i * size2_ + j];
        for (size_t j = col; j < size2_ - 1; j++)
          tmp(i, j) = data_[i * size2_ + j + 1];
      }
      return tmp;
    }

    //remove column non const : 
    void remove_column_nc(size_t col)
    {
      if(col>=size2_)
        return;

      matrix<T> tmp(size1_, size2_ - 1);
      for (size_t i = 0; i < size1_; i++)
      {
        for (size_t j = 0; j < col; j++)
          tmp(i, j) = data_[i * size2_ + j];
        for (size_t j = col; j < size2_ - 1; j++)
          tmp(i, j) = data_[i * size2_ + j + 1];
      }
      delete[] data_;
      data_ = tmp.data_;
      tmp.size2_ = 0; //avoid double free.
      size2_--;
    } 
    
    //remove row
    matrix<T> remove_row(size_t row) const
    {
      if(row>=size1_)
        return *this;

      matrix<T> tmp(size1_ - 1, size2_);
      for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = data_[i * size2_ + j];
      for (size_t i = row; i < size1_ - 1; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = data_[(i + 1) * size2_ + j];
      return tmp;
    }
    //remove row non const
    void remove_row_nc(size_t row)
    {
      if(row>=size1_)
        return;

      matrix<T> tmp(size1_ - 1, size2_);
      for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = data_[i * size2_ + j];
      for (size_t i = row; i < size1_ - 1; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = data_[(i + 1) * size2_ + j];
      delete[] data_;
      data_ = tmp.data_;
      tmp.size1_ = 0; //avoid double free.

      size1_--;
    }

    T log()
    {
      T sum = 0;
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          sum += std::log(data_[i * size2_ + j]);
      return sum;
    }
    
    //unaryExpr implementation
    template <typename F>
    matrix<T> unaryExpr(F f) const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = f(data_[i * size2_ + j]);
      return tmp;
    }   
    //unary expression of member function
    template <typename F , typename P>
    matrix<T> unaryExpr(F f , P* parent) const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = (parent->*f)(data_[i * size2_ + j]);
      return tmp;
    }


    T row_sum(size_t row)const
    {
      T sum = 0;
      for (size_t i = 0; i < size2_; i++)
        sum += data_[row * size2_ + i];
      return sum;
    }
    T col_sum(size_t col)const
    {
      T sum = 0;
      for (size_t i = 0; i < size1_; i++)
        sum += data_[i * size2_ + col];
      return sum;
    }
    T col_min(size_t col)const
    {
      T min = data_[0 * size2_ + col];
      for (size_t i = 0; i < size1_; i++)
        if (data_[i * size2_ + col] < min)
          min = data_[i * size2_ + col];
      return min;
    }
    T col_max(size_t col)const
    {
      T max = data_[0 * size2_ + col];
      for (size_t i = 0; i < size1_; i++)
        if (data_[i * size2_ + col] > max)
          max = data_[i * size2_ + col];
      return max;
    }
    T row_min(size_t row)const
    {
      T min = data_[row * size2_ + 0];
      for (size_t i = 0; i < size2_; i++)
        if (data_[row * size2_ + i] < min)
          min = data_[row * size2_ + i];
      return min;
    }
    T row_max(size_t row)const
    {
      T max = data_[row * size2_ + 0];
      for (size_t i = 0; i < size2_; i++)
        if (data_[row * size2_ + i] > max)
          max = data_[row * size2_ + i];
      return max;
    }

    T row_mean(size_t row)const
    {
      T sum = 0;
      for (size_t i = 0; i < size2_; i++)
        sum += data_[row * size2_ + i];
      return sum / size2_;
    }
    //invert 
    void invert() {
      //invert the matrix
      //this function is not optimized
      //it is used for small matrices
      //for large matrices use the invert function in the matrix class
      //this function is used for testing purposes
      //if sizes are different, transpose the matrix and invert it
      if (size1_ != size2_)
      {
        matrix<T> tmp = transpose();
        for(size_t i=0;i<size1_;i++)
          for(size_t j=0;j<size2_;j++)
            data_[i*size2_+j] = tmp(i,j);

        return;
      } 
      else
      {
        //sizes are equal:
        //invert the matrix positions
        matrix<T> tmp = *this;
        for (size_t i = 0; i < size1_; i++)
          for (size_t j = 0; j < size2_; j++)
            data_[i * size2_ + j] = tmp(j, i);


      }

    }
    T*& row_begin(size_t row)
    {
      
      return data_ + row * size2_;
    }
    T*& row_end(size_t row)
    {
      return data_ + (row + 1) * size2_;
    }
    T*& col_begin(size_t col)
    {
      return data_ + col;
    }
    T*& col_end(size_t col)
    {
      return data_ + size1_ * size2_ + col;
    }
    void remove_column(size_t col)
    {
      
      size_t new_col_size = size2_ - 1;
      //make sure col is not out of range 
      if(col>=size2_)
        return;
         
      T* tmp = new T[size1_ * new_col_size];
      for (size_t i = 0; i < size1_; i++)
      {
        for (size_t j = 0; j < col; j++)
          tmp[i * new_col_size + j] = data_[i * size2_ + j];
        for (size_t j = col; j < new_col_size; j++)
          tmp[i * new_col_size + j] = data_[i * size2_ + j + 1];
      }
      T* old_data= data_;
      data_ = tmp;
      
      delete[] old_data;
      size2_--;
    }
    void reduce_rows(size_t nRows,bool endian=true)
    {
      if(nRows>size1_)
        return;
      size_t new_row_size = size1_ - nRows;
      T* tmp = new T[new_row_size * size2_];
      if(endian)
      {
        for (size_t i = 0; i < new_row_size; i++)
          for (size_t j = 0; j < size2_; j++)
            tmp[i * size2_ + j] = data_[(i + nRows) * size2_ + j];
      }
      else
      {
        for (size_t i = 0; i < new_row_size; i++)
          for (size_t j = 0; j < size2_; j++)
            tmp[i * size2_ + j] = data_[(i + 1) * size2_ - j - 1];
      } 
      delete[] data_;
      data_ = tmp;
      size1_ = new_row_size;
    }
    std::vector<real_t> row_entropy()const 
    {
      //returns the entropy of each column of the matrix as a vector
      std::vector<real_t> entropy(size1_);
      for (size_t i = 0; i < size1_; i++)
      {
        entropy[i] = 0;
        for (size_t j = 0; j < size2_; j++)
          entropy[i] += data_[i * size2_ + j] * std::log(data_[i * size2_ + j]);
        entropy[i] = -entropy[i];
      }
      return entropy;

    }
    std::vector<real_t> col_entropy()const
    {
      //returns the entropy of each column of the matrix as a vector
      std::vector<real_t> entropy(size2_);
      for (size_t j = 0; j < size2_; j++)
      {
        entropy[j] = 0;
        for (size_t i = 0; i < size1_; i++)
          entropy[j] += data_[i * size2_ + j] * log(data_[i * size2_ + j]);
        entropy[j] = -entropy[j];
      }
      return entropy;

    } 

    real_t entropy()const
    {
      real_t entropy = 0.0;
      

      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          entropy += real_t( data_[i * size2_ + j] * std::log((T)data_[i * size2_ + j]));

      return -entropy;
    }
    void remove_row(size_t row)
    {
        
        size_t new_row_count = size1_ - 1;
        T* tmp = new T[new_row_count * size2_];
        for (size_t i = 0; i < row; i++)
          for (size_t j = 0; j < size2_; j++)
            tmp[i * size2_ + j] = data_[i * size2_ + j];
        for (size_t i = row; i < new_row_count; i++)
          for (size_t j = 0; j < size2_; j++)
            tmp[i * size2_ + j] = data_[(i + 1) * size2_ + j];
        delete[] data_;
        data_ = tmp;
        size1_--;
    }
 
    //tanh
    matrix<T> tanh() const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; ++j)
          tmp(i, j) = std::tanh(data_[i * size2_ + j]);
      return tmp;
    }

    //sigmoid
    matrix<T> sigmoid() const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; ++j)
          tmp(i, j) = 1.0 / (1.0 + std::exp(-data_[i * size2_ + j]));
      return tmp;
    } 
    //relu
    matrix<T> relu() const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; ++j)
          tmp(i, j) = std::max(0.0, data_[i * size2_ + j]);
      return tmp;
    } 
    //softmax
    matrix<T> softmax() const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
      {
        T sum = 0;
        for (size_t j = 0; j < size2_; j++)
          sum += std::exp(data_[i * size2_ + j]);
        for (size_t j = 0; j < size2_; j++)
          tmp(i, j) = std::exp(data_[i * size2_ + j]) / sum;
      }
      return tmp;
    }
    //identity
    matrix<T> identity() const
    {
      matrix<T> tmp(size1_, size2_);
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          if (i == j)
            tmp(i, j) = 1;
          else
            tmp(i, j) = 0;
      return tmp;
    }

    virtual ~matrix()
    {
      //avoid double free : 
      size_t size = size1_ * size2_;
      if (data_ != nullptr && size> 0 )
        {
          //DEBUG : 
          //std::cout<<"[!] debug  - matrix destructor data :"<<std::to_string(int64_t(data_)) <<std::endl;
          delete[] data_; 
          
        }
      size1_ = size2_ = 0;
      data_ = nullptr;
    }

    matrix<T> operator*(const T &value) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) * value;

      return ret;
    }
    matrix<T> operator/(const T &value) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) / value;

      return ret;
    }
    matrix<T> operator-(const T &value) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; ++j)
          ret(i, j) = ret(i, j) - value;

      return ret;
    }
    //for T-x
    matrix<T> operator -() const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; ++j)
          ret(i, j) = -ret(i, j);

      return ret;
    }
    //for T+x
    matrix<T> operator +() const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; ++j)
          ret(i, j) = +ret(i, j);

      return ret;
    }
     matrix<T> operator+(const T &value) const
    {
       matrix<T> ret = *this;
       for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) + value;

      return ret;
    }
    matrix<T> operator*(const matrix<T> &m) const
    {
      //returns a matrix with the same size as the first matrix 
      //and the same number of columns as the second matrix
      if(m.size1()!=size1_ || m.size2()!=size2_)
        {
         matrix<T> ret(size1_, m.size2_);

          for (size_t i = 0; i < size1_; i++) {
            for (size_t j = 0; j < ret.size2_; j++)  {
              ret(i, j) = element(i, j%size2_ ) * m(i%m.size1(), j);
            }
          }

          return ret;
        }
        else
        {
          matrix<T> ret(size1_, size2_);
          for (size_t i = 0; i < size1_; i++) {
            for (size_t j = 0; j < size2_; j++) {
              ret(i, j) = element(i, j) * m(i, j);
            }
          }
          return ret;
        }
        //never reached
        return matrix<T>();

    }
     matrix<T> operator/(const matrix<T> &m) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) / m(i, j);
       return ret;
    }
    const_reference
    operator()(size_type i, size_type j) const
    {
      size_t k = i * size2_ + j;
      if(k<size1_*size2_)
      return data_[k];
      else
      {
        return data_[0];
      }
      
    } 


    const_reference 
    element(size_type i, size_type j)const
    {
      size_t k = i * size2_ + j;
      if(k<size1_*size2_)
      return data_[k];
      else
      {
        return data_[0];
      }
      
    }

    reference
    element(size_type i, size_type j)
    {
      size_t k = i * size2_ + j;
      if(k<size1_*size2_)
      return data_[k];
      else
      {
        return data_[0];
      }
      
    }
    reference
    operator()(size_type i, size_type j)
    {
      size_t k = i * size2_ + j;
      size_t size = size1_ * size2_;
      if(data_==nullptr)
      {
        //try to allocate memory
        if(size1_*size2_>0)
        data_ = new T[size];

        return data_[k];

      }
      else       if(k<size) {
      return data_[k];
      }
      else
      {
        return data_[0];
      }
      
    }
    reference
    insert_element(size_type i, size_type j, const_reference t)
    {
      return (element(i, j) = t);
    }
    void
    erase_element(size_type i, size_type j)
    {
      element(i, j) = value_type();
    }
    void
    clear()
    {
      std::fill(data_, data_ + size1_ * size2_, value_type());      
    }
    matrix &
    operator=(const matrix &m)
    {
      size_t size = size1_ * size2_;
      
      if(m.data_!=this->data_){
      
      if(m.size1_==0 || m.size2_==0 || m.data_==nullptr   )
      {
        return *this;
      }
     
      if ( size < (m.size1_ * m.size2_))
      {
        if (data_ != nullptr && size > 0) {
            delete[] data_;
            data_ = new T[m.size1_ * m.size2_];
        }
        else if (data_ == nullptr)
        {
          data_ = new T[m.size1_ * m.size2_];
        }
      }
      

      size = m.size1_ * m.size2_;
      

      std::copy(m.data_, m.data_ + size, data_);

      size1_ = m.size1_;
      size2_ = m.size2_;

      }
      return *this;
      
    }
    matrix &operator=(const std::vector<T> &vec)
    {
      if (data_ != nullptr && size1_ * size2_ > 0)
        delete[]  data_ ;
      size1_ = vec.size();
      size2_ = 1;

      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
    }
    matrix &operator=(const std::vector<std::vector<T>> &vec)
    {
      if( vec.size()){
        if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;
      size1_ = vec.size();
      size2_ = vec.begin()->size();

      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);  
        ++i;
      }
      }//if vector is empty do nothing
      return *this;
    }
    matrix &operator=(const std::vector<std::initializer_list<T>> &vec)
    {
        if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;
      size1_ = vec.size();
      size2_ = vec.begin()->size();
  
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::vector<std::initializer_list<T>> &&vec)
    {
       if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;
      size1_ = vec.size();
      size2_ = vec.begin()->size();
    
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::initializer_list<std::initializer_list<T>> &vec)
    {
       if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;
      size1_ = vec.size();
      size2_ = vec.begin()->size();
  
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(  std::initializer_list<std::initializer_list<T>> &&vec)
    {
       if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;

      size1_ = vec.size();
      size2_ = vec.begin()->size();
  
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::initializer_list<T> &vec)
    {
         if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;
  
      size1_ = vec.size();
      size2_ = 1;
       data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
    }
    matrix &operator=(  std::initializer_list<T> &&vec)
    {
        if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;
      size1_ = vec.size();
      size2_ = 1;
   
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
    }
    matrix &operator=(  std::vector<T> &&vec)
    {
     //move data from vec 
     
      if (data_ != nullptr && size1_ * size2_ > 0)
        delete[] data_;
      size1_ = vec.size();
      size2_ = 1;
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
      vec.clear();

    }

    void
    resize(size_type dim, size_type dim1)
    {
      size_t size = size1_ * size2_;
      size_t new_size = dim * dim1;
      if(size==new_size)
      {
        if(size1_!=dim || size2_!=dim1)
        {
          size1_ = dim;
          size2_ = dim1;
        }
       // return;
      }
      else if ( size!= new_size)
        {
          if(size>0) {
            delete[] data_;
          }
          data_ = nullptr;
        }
        size1_ = dim;
        size2_ = dim1;
         if(data_==nullptr && size >0) //dont reallocate if size is the same, just clear the data 
          { 
            data_ = new T[new_size];
          }
          else if(data_==nullptr && size==0 && new_size>0)
          {
            data_ = new T[new_size];  
            return;
          }
          else if(data_!=nullptr && size!=new_size)
          {
            delete[] data_;
            data_ = new T[new_size];
          }
          else if(data_!=nullptr && size==new_size)
          {
            std::fill(data_, data_ + size, value_type());
          }
      
    }
    size_type
    size1() const
    {
      return size1_;
    }
    size_type
    size2() const
    {
      return size2_;
    }

    fixed_array_type row(size_t i)
    {
      fixed_array_type ret  = reinterpret_cast<fixed_array_type>( data_ + (i * cols()));    
      return ret;
    }
 
    fixed_array_type row(size_t i) const
    {
      fixed_array_type ret  = reinterpret_cast<fixed_array_type>(
          data() + (i * cols()));
      return ret;
    }
 
    template <size_t N> matrix<T> lpNorm() const 
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    }
    matrix<std::complex<T>> operator()()const
    {
      matrix<std::complex<T>> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = element(i, j);
      return ret;
    }
     matrix<T> lpNorm(double N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    }
    matrix<T> lpNorm(int N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 
    matrix<T> lpNorm(float N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 
    matrix<T> lpNorm(long double N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 
    matrix<T> lpNorm(long long int N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 
    matrix<T> slice(size_t row_start, size_t row_end) const
    {
      matrix<T> ret(row_end - row_start, size2());
      for (size_t i = row_start; i < row_end; i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i - row_start, j) = (*this)(i, j);
      return ret;
    }
    matrix<T> slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const
    {
      matrix<T> ret(row_end - row_start, col_end - col_start);
      for (size_t i = row_start; i < row_end; i++)
        for (size_t j = col_start; j < col_end; j++)
          ret(i - row_start, j - col_start) = (*this)(i, j);
      return ret;
    }

    //Nonlinear stability and bifurcation theory  Hans Troger,Alois Steindl 

    //Get divergent elements for f(x) : 
    std::vector<std::pair<size_t, size_t>> divergent(std::function<T(T)> f) const
    {
      std::complex<T> c;
      std::vector<std::pair<size_t, size_t>> ret; 
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
        {
          c = f((*this)(i, j));
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for 
      return ret;
    }   
    //get divergent elements for f(x,y) : 
    std::vector<std::pair<size_t, size_t>> divergent(std::function<T(T,T)> f , std::vector<T> y) const
    {
      std::complex<T> c;
      std::vector<std::pair<size_t, size_t>> ret; 
      //make sure y has the same size as the matrix
      if(y.size()!=size1())
        return ret;
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
        {
          c = f((*this)(i, j),y[i]);
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for
      return ret;

    }  

    //get fluttering elements for f(x) : 
    std::vector<std::pair<size_t, size_t>> flutter(std::function<T(T)> f) const
    {
      std::complex<T> c;
      std::vector<std::pair<size_t, size_t>> ret; 
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
        {
          c = f((*this)(i, j));
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for
      return ret; 

    }
    //get fluttering elements for f(x,y) : 
    std::vector<std::pair<size_t, size_t>> flutter(std::function<T(T,T)> f , std::vector<T> y) const
    {
      std::complex<T> c;
      std::vector<std::pair<size_t, size_t>> ret; 
      //make sure y has the same size as the matrix 
      if(y.size()!=size1())
        return ret;
      for (size_t i = 0; i < size1(); i++)  
        for (size_t j = 0; j < size2(); j++)
        {
          c = f((*this)(i, j),y[i]);
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for 
      return ret;
    } 
    //get stable elements for f(x) :
    std::vector<std::pair<size_t, size_t>> stable(std::function<T(T)> f) const
    {
      std::vector<std::pair<size_t, size_t>> ret;
      std::complex<T> c(T(0.0));
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++){
          c= f((*this)(i, j));
          if (!std::isinf(c.real()) && !std::isnan(c.real()) && !std::isinf(c.imag()) && !std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for

      return ret;

    } 
    //get stable elements for f(x,y) : 
    std::vector<std::pair<size_t, size_t>> stable(std::function<T(T,T)> f , std::vector<T> y) const
    {
      std::vector<std::pair<size_t, size_t>> ret; 
      std::complex<T> c;
      //make sure y has the same size as the matrix 
      if(y.size()!=size1())
        return ret; 

      for (size_t i = 0; i < size1(); i++)  
        for (size_t j = 0; j < size2(); j++)
        {
          c = f((*this)(i, j),y[i]);
          if (!std::isinf(c.real()) && !std::isnan(c.real()) && !std::isinf(c.imag()) && !std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for
      return ret;
    }     

    //get unstable elements for f(x) : 
    std::vector<std::pair<size_t, size_t>> unstable(std::function<T(T)> f) const
    {
      std::complex<T> c;
      std::vector<std::pair<size_t, size_t>> ret; 
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
        {
          c = f((*this)(i, j));
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for

      return ret;
    }   
    //get unstable elements for f(x,y) :  
    std::vector<std::pair<size_t, size_t>> unstable(std::function<T(T,T)> f , std::vector<T> y) const
    {
      std::vector<std::pair<size_t, size_t>> ret; 
      //make sure y has the same size as the matrix 
      if(y.size()!=size1())
        return ret; 
      //check if inf or -inf 
      std::complex<T> c; 
      for (size_t i = 0; i < size1(); i++){
        for (size_t j = 0; j < size2(); j++){
          c = f((*this)(i, j),y[i]);
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        
        } //for
      }//for
      
      return ret;
    } 
  
    //default stable: 
    std::vector<std::pair<size_t, size_t>> stable() const
    {
      std::vector<std::pair<size_t, size_t>> ret;
      std::complex<T> c;
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++){
          c = (*this)(i, j);
          if (!std::isinf(c.real()) && !std::isnan(c.real()) && !std::isinf(c.imag()) && !std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for

      return ret;

    } 
    //default unstable:
    std::vector<std::pair<size_t, size_t>> unstable() const
    {
      std::vector<std::pair<size_t, size_t>> ret;
      std::complex<T> c;
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
        {
          c = (*this)(i, j);
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for

      return ret;
    } 
    //default flutter:  
    std::vector<std::pair<size_t, size_t>> flutter() const
    {
      std::vector<std::pair<size_t, size_t>> ret;
      std::complex<T> c;
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
        {
          c = (*this)(i, j);
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for

      return ret;
    }   
    //default divergent:
    std::vector<std::pair<size_t, size_t>> divergent() const
    {
      std::vector<std::pair<size_t, size_t>> ret;
      std::complex<T> c;
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
        {
          c = (*this)(i, j);
          if (std::isinf(c.real()) || std::isnan(c.real()) || std::isinf(c.imag()) || std::isnan(c.imag()))
            ret.push_back(std::make_pair(i, j));
        } //for

      return ret;
    } 

    size_type
    rows() const
    {
      return size1();
    }
    size_type
    cols() const
    {
      return size2();
    }
    inline fixed_array_type &
    data()
    { 
      return data_;
    }


    inline const fixed_array_type &
    data()const
    {
       return data_;
    }
    void fill(const T &val)
    {
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          (*this)(i, j) = val;
    }
    inline matrix<real_t> covariance()  const
    { 
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = covariance(i,j); 
      return ret;
    }
    real_t covariance(size_t i,size_t j) const
    {
      return covariance(row(i),row(j));
    }
    real_t covariance(fixed_array_type a,fixed_array_type b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    real_t mean(const fixed_array_type &a) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += a[i];
      return ret/size1();
    } 
    T mean() const
    {
      T ret = 0;
      for(size_t i=0;i<size1();i++)
        for(size_t j=0;j<size2();j++)
          ret += data_[i*size2()+j];
      
      return ret/(size1()*size2()); 

    }
    T median()const 
    {
      std::vector<T> tmp(size1()*size2());
      for(size_t i=0;i<size1();i++)
        for(size_t j=0;j<size2();j++)
          tmp[i*size2()+j] = data_[i*size2()+j];
      std::sort(tmp.begin(),tmp.end());
      return tmp[tmp.size()/2]; 

    }
    inline matrix<real_t> correlation() const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = correlation(i,j); 
      return ret;
    } 
    real_t correlation(size_t i,size_t j) const
    {
      //returm correlation(row(i),row(j)); 
      return correlation(data_+i*size2(),data_+j*size2());

    }
    real_t correlation(const fixed_array_type &a,const  fixed_array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    //correlation_coefficient (x(i)-mean(x))*(y(i)-mean(y)) / ((x(i)-mean(x))2 * (y(i)-mean(y))2.
    inline
     matrix<real_t> correlation_coefficient()const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
         {
            ret(i,j) =std::sqrt( variance(i,j)/variance(i,i)*variance(j,j));  

         } 

      return ret;
    }
 
    inline matrix<real_t> variance()const 
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = variance(i,j); 
      return ret;
    }
    real_t variance(size_t i,size_t j)
    {
      return variance(row(i),row(j));
    }
    real_t variance(fixed_array_type &a,fixed_array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    } 
    inline matrix<real_t> std()   const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = std(i,j); 
      return ret;
    } 
    real_t std(size_t i,size_t j) const
    {
      return std(row(i),row(j));
    } 

    real_t std(const fixed_array_type &a,const fixed_array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    real_t col_mean(size_t i) const
    {
      real_t ret = 0.;
      for(size_t j=0;j<size1();j++)
        ret += data_[j*size2()+i ]; // (j,i);
      return ret/size1();
    }
    real_t col_std(size_t i) const
    {
      real_t ret = 0.;
      real_t mean = col_mean(i);
      for(size_t j=0;j<size1();j++)
        ret += (data_[j * size2_ + i]-mean)*(data_[j * size2_ + i]-mean);   // element(j,i)--> data_[j*size2()+i ] data_[j * size2_ + i]
      return std::sqrt(ret/(size1()-1));
    } 
    real_t col_variance(size_t col) const
    {
      real_t ret = 0.;
      real_t mean = col_mean(col);
      for(size_t i=0;i<size1();i++)
        ret += (data_[i * size2_ + col]-mean)*(data_[i * size2_ + col]-mean); // element(i,col)--> data_[i*size2()+col ] data_[i * size2_ + col]
      return ret/(size1()-1);
    }
 

    inline matrix<real_t> eigenvalues() const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = eigen_values(i,j); 
      return ret;
    }
    real_t eigen_values(size_t i,size_t j) const
    {
        real_t ret = 0;
        fixed_array_type a = row(i);
        fixed_array_type b = row(j);
        for(size_t k=0;k<size1_;k++)
          ret += (a[k]-mean(a))*(b[k]-mean(b)); 

        return (size1_>1)? ret/real_t(size1()-1) : ret; 
        
      //return eigen_values(row(i),row(j));
    }
    real_t eigen_values(const fixed_array_type &a,const fixed_array_type &b)  const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }



    inline matrix<real_t> eigenvectors() const 
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = eigen_vectors(i,j); 
      return ret;
    }
    real_t eigen_vectors(size_t i,size_t j) const
    {
      return eigen_vectors(row(i),row(j));
    }
    real_t eigen_vectors(const fixed_array_type &a,const fixed_array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    inline matrix<real_t> eigen_values_and_vectors() const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = eigen_values_and_vectors(i,j); 
      return ret;
    } 
    real_t eigen_values_and_vectors(size_t i,size_t j) const
    {
      return eigen_values_and_vectors(row(i),row(j));
    }
    real_t eigen_values_and_vectors(fixed_array_type &a,fixed_array_type &b)  const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }

    void get_eigen_values_and_vectors(std::vector<T> &eigen_values, matrix<T> &eigen_vectors) const
    {
      //calculate eigen values and vectors

      //eigen values
      eigen_values.resize(size1());
      eigen_vectors.resize(size1(), size1());

      //copy data
      std::copy(data_, data_+(size1_*size2_), eigen_vectors.data().begin());
      //calculate eigen values and vectors
      eigen(eigen_vectors, eigen_values);
      //sort eigen values and vectors
      std::vector<std::pair<T, size_t> > eigen_values_index(size1());
      for (size_t i = 0; i < size1(); ++i)
        eigen_values_index[i] = std::make_pair(eigen_values[i], i);
      std::sort(eigen_values_index.begin(), eigen_values_index.end(), std::greater<std::pair<T, size_t> >());
      //copy sorted eigen values and vectors
      for (size_t i = 0; i < size1(); ++i)
      {
        eigen_values[i] = eigen_values_index[i].first;
        for (size_t j = 0; j < size1(); ++j)
          eigen_vectors(j, i) = eigen_vectors(j, eigen_values_index[i].second);
      }
      //normalize eigen vectors
      for (size_t i = 0; i < size1(); ++i)
      {
        T norm = 0;
        for (size_t j = 0; j < size1(); ++j)
          norm += eigen_vectors(j, i) * eigen_vectors(j, i);
        norm = std::sqrt(norm);
        for (size_t j = 0; j < size1(); ++j)
          eigen_vectors(j, i) /= norm;
      } 
      
      //done  
      return;

    }      
    
    const fixed_array_type &
    as_diagonal()
    {

      static fixed_array_type diagonal_instance = nullptr;

      if (diagonal_instance)
        delete diagonal_instance;

      diagonal_instance = new T[size1_];

      for (size_t i = 0, j = 0; i < size1_ && j < size2_; ++i, ++j)
        diagonal_instance[i] = data_[i + j];

      return diagonal_instance;
    }

    //Let G=(V,E)
    //be a locally finite graph; this means that each vertex has finite degree. The Laplacian operator Δ acting on the space of functions f:V→R is given by
    //(Δf)(v)=∑v→w(f(w)−f(v))

    //where v→w
    //indicates an edge from the vertex v to the vertex w. When G=Z with edges between adjacent integers, this gives
    //(Δf)(n)=f(n+1)−2f(n)+f(n−1)

    //as expected. On the space of compactly supported functions f:V→R
    //(those that are zero except at finitely many positions) there is a natural inner product given by
    //⟨f,g⟩=∑vf(v)g(v)

    //and the Laplacian is Hermitian with respect to this inner product. Indeed,
    //⟨f,Δg⟩=∑v→w(f(v)g(w)−f(v)g(v))=∑v→w(f(w)g(v)−f(v)g(v))=⟨Δf,g⟩.
 
    matrix<T> & laplacian()const
    {
      matrix<T> ret(size1(), size2());
      
      for (size_t i = 0; i < size1(); ++i)
        for (size_t j = 0; j < size2(); ++j)
          ret(i, j) = laplacian(i, j);
      return ret;
    }

    T laplacian(size_t i, size_t j)const
    {
      T ret = 0;
      if (i == j)
      {
        ret = 0;
        for (size_t k = 0; k < size1(); ++k)
          ret += element(i, k);
        ret = -ret;
      }
      else
      {
        ret = element(i, j);
      }
      return ret;   
    }

    matrix<T> & laplacian_normalized()const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); ++i)
        for (size_t j = 0; j < size2(); ++j)
          ret(i, j) = laplacian(i, j) / element(i, i) / element(j, j) / size1() / size2() * 4 * M_PI * M_PI * 1e-7  * 1e-7      ;

      return ret; 
    }

    matrix<T> & laplacian_normalized(const matrix<T> & m)const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); ++i)
        for (size_t j = 0; j < size2(); ++j)
          ret(i, j) = laplacian(i, j) / element(i, i) / element(j, j) / size1() / size2() * 4 * M_PI * M_PI * 1e-7  * 1e-7 * m(i, j)     ;

      return ret;
    }

    matrix<bool> hermitian()const
    {
      matrix<bool> ret(size1(), size2());
      for (size_t i = 0; i < size1(); ++i)
        for (size_t j = 0; j < size2(); ++j)
          ret(i, j) = hermitian(i, j);
      return ret; 
    }

    bool hermitian(size_t i, size_t j)const
    {
      return element(i, j) == element(j, i);
    }

    matrix<bool> symmetric()const
    {
      matrix<bool> ret(size1(), size2());
      for (size_t i = 0; i < size1(); ++i)
        for (size_t j = 0; j < size2(); ++j)
          ret(i, j) = symmetric(i, j);
      return ret; 
    }
    bool symmetric(size_t i, size_t j)const
    {
      return element(i, j) == element(j, i);
    }
    
    matrix<bool> antisymmetric()const
    {
      matrix<bool> ret(size1(), size2());
      for (size_t i = 0; i < size1(); ++i)
        for (size_t j = 0; j < size2(); ++j)
          ret(i, j) = antisymmetric(i, j);
      return ret;
    }
    bool antisymmetric(size_t i, size_t j)const
    {
      return element(i, j) == -element(j, i);
    }

    matrix<T> sqrt()
    {

      matrix<T> sqrt_prod(size2(), size1());
      for (typename matrix<T>::size_type i = 0; i < size1(); i++)
        for (typename matrix<T>::size_type j = 0; j < size2(); j++)
          sqrt_prod(i, j) = std::sqrt(element(i, j));

      return sqrt_prod;
    }
    matrix<T>
    transpose() const
    {

      matrix<T> transp(size2(), size1());
      for (typename matrix<T>::size_type i = 0; i < size2(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size1(); ++j)
          transp(i, j) = element(j, i);
      return transp;
    }
    matrix<T> reverse() const
    {
      matrix<T> reverse(size1(), size2());
      for (typename matrix<T>::size_type i = 0; i < size2(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size1(); ++j)
          reverse(i, j) = element(size1() - i, size2() - j);
      return reverse;
    }
    //conjugate
    matrix<T> conjugate() const
    {
      matrix<T> conjugate(size1(), size2());
      for (typename matrix<T>::size_type i = 0; i < size1(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size2(); ++j)
          conjugate(i, j) = std::conj(element(i, j)).real();
      return conjugate;
    }
    //adjoint
    matrix<T> adjoint() const
    {
      matrix<T> adjoint(size2(), size1());
      for (typename matrix<T>::size_type i = 0; i < size2(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size1(); ++j)
          adjoint(i, j) = std::conj(element(j, i)).real();
      return adjoint;
    } 
    //conjugate transpose
    matrix<T> conjugate_transpose() const
    {
      matrix<T> conjugate_transpose(size2(), size1());
      for (typename matrix<T>::size_type i = 0; i < size2(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size1(); ++j)
          conjugate_transpose(i, j) = std::conj(element(j, i)).real() ;
      return conjugate_transpose;
    }


    iterator
    begin()
    {
      return data_;
    }
    const_iterator
    begin() const
    {
      return data_;
    }
    const_iterator
    cbegin() const
    {
      return data_;
    }

    iterator
    end()
    {
      return data_ + size1_ * size2_;
    }
    const_iterator
    end() const
    {
      return data_ + size1_ * size2_;
    }
    const_iterator
    cend() const
    {
      return data_ + size1_ * size2_;
    }
    T squaredNorm() const
    {

      T ret = 0.;
      for (typename matrix<T>::size_type i = 0; i < size1(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size2(); ++j)
          ret += element(i, j) * element(i, j);

      return ret;
    }

    T norm() const
    {
      return std::sqrt(squaredNorm());
    }

    T sum() const
    {
      T ret(0.);
      for (typename matrix<T>::size_type i = 0; i < size1(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size2(); ++j)
          ret += this->operator() (i, j);

      return ret;
    }

    T sum_row(size_t row) const
    {
      T ret(0.);
      for (size_t j = 0; j < size2_; j++)
        ret += data_[row * size2_ + j];
      return ret;
    }
    T dot_row(size_t row, const matrix<T> &other) const
    {
      T ret(0.);
      for (size_t j = 0; j < size2_; j++)
        ret += data_[row * size2_ + j] * other(row, j);
      return ret;
    }
    T dot_row(size_t row, const std::vector<T> &other) const
    {
      T ret(0.);
      for (size_t j = 0; j < size2_; j++)
        ret += data_[row * size2_ + j] * other[j];
      return ret;
    }
    // sets zero on all values
    void
    set_zero() { 
      this->clear(); 
      
      }
    const fixed_array_type &
    fixed_array() const
    {
      return data_;
    }
    // return 0 matrix

    static matrix
    Zero(const size_t &a, const size_t &b)
    {
      matrix ret(a, b);
      std::fill(ret.begin(), ret.end(), 0.0);
      return ret;
    }
    // fills a matrix with constant 1

    static matrix
    One(const size_t &a, const size_t &b)
    {
      matrix ret(a, b);
      std::fill(ret.begin(), ret.end(), 1.0);
      return ret;
    }

    // fills a matrix with constant value

    static matrix
    Constant(const size_t &a, const size_t &b, float pt)
    {

      matrix ret(a, b);
      for (auto &x : ret)
        x = pt;

      return ret;
    }

    // generates random values between -1 and 1

    static matrix
    Random(const size_t &a, const size_t &b)
    {
      std::random_device dev;
      std::mt19937 gen(dev());
      std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

      matrix ret(a, b);
      for (auto &x : ret)
        x = (T)1.0 - uniform_dist(gen);

      return ret;
    }

    inline T *
    operator[](const int i) // subscripting
    {
      return &data_[i * size2_];
    }
    inline const T *
    operator[](const int i) const // readonly
    {
      return &data_[i * size2_];
    }
    friend std::ofstream & operator << (std::ofstream &out, const matrix<T> &mat)
    {
      out << mat.size1() << " " << mat.size2() << std::endl;

      for (size_t i = 0; i < mat.size1(); i++)
      {
        for (size_t j = 0; j < mat.size2(); j++)
        {
          out << mat(i, j) << " ";
        }
        out << std::endl;
      }
      return out;
    }
    friend std::ifstream & operator >> (std::ifstream &in, matrix<T> &mat)
    {
      size_t size1, size2;
      in >> size1 >> size2;
      mat.resize(size1, size2);
      for (size_t i = 0; i < mat.size1(); i++)
      {
        for (size_t j = 0; j < mat.size2(); j++)
        {
          in >> mat(i, j);
        }
      }
      return in;
    } 

    //  
    // operators for matrix-matrix operations
    //
    matrix<T> &
    operator+=(const matrix<T> &rhs)
    {
      assert(size1_ == rhs.size1_ && size2_ == rhs.size2_);
      std::transform(data_.begin(), data_.end(), rhs.data_.begin(),
                     data_.begin(), std::plus<T>());
      return *this;
    }
    matrix<T> &
    operator-=(const T &rhs)
    {
      std::transform(&data_[0], &data_[size1_*size2_], &data_[0],
                     std::bind2nd(std::minus<T>(), rhs));
      return *this;
    }
    matrix<T> &
    operator-=(const matrix<T> &rhs)
    {
      size_t min_col = std::min(size2_, rhs.size2_); 
      size_t min_row = std::min(size1_, rhs.size1_); 
      for(size_t i=0;i<min_row;i++)
        for(size_t j=0;j<min_col;j++)
          data_[i*size2_+j] -= rhs(i,j);
    
      return *this;
    }

    matrix<T> &
    operator*=(const matrix<T> &rhs)
    {
      assert(size1_ == rhs.size1_ && size2_ == rhs.size2_);
      std::transform(&data_[0], &data_[size1_*size2_], &rhs.data_[0],
                      &data_[0], std::multiplies<T>());
      return *this;
    } 

    matrix<T> &
    operator/=(const matrix<T> &rhs)
    {
      assert(size1_ == rhs.size1_ && size2_ == rhs.size2_);
      std::transform(&data_[0], &data_[size1_*size2_], rhs.data_.begin(),
                      &data_[0], std::divides<T>());  
      return *this;
    }
    //  real determinant
    T real_det()const
    {
            T det = 0;
            size_t min=std::min(this->size1_,this->size2_); 
            for (size_t i = 0; i < min; i++) {
                det += this->data_[i * this->size2_ + i];
            }
            det = std::abs(det)  * (this->size1_ == this->size2_ ? 1 : 0); 
            
            return det;
      
    }
        //calculate the determinant of the smaller matrices allready with power -1 appied

    //serialize matrices to binary file 
    
    
    //  complex determinant
    std::complex<T> complex_det()const
    {
      std::complex<T> ret = 0;
      //calculate determinant of matrix
      if (size1_ == 1)
      {
        ret = data_[0];
      }
      else if (size1_ == 2)
      {
        ret = data_[0] * data_[3] - data_[1] * data_[2];
      }
      else
      {
        for (size_t i = 0; i < size1_; i++)
        {
          matrix<T> temp(size1_ - 1, size1_ - 1);
          for (size_t j = 1; j < size1_; j++)
          {
            for (size_t k = 0; k < size1_; k++)
            {
              if (k < i)
              {
                temp(j - 1, k) = data_[j * size1_ + k];
              }
              else if (k > i)
              {
                temp(j - 1, k - 1) = data_[j * size1_ + k];
              }
            }
          }
          ret += data_[i] * pow(-1, i) * temp.complex_det();
        }
      } 
      return ret;

    }      
    //print matrix
    void print() const
    {
      for (size_t i = 0; i < size1_; i++)
      {
        for (size_t j = 0; j < size2_; j++)
        {
          std::cout << data_[(i * size2_) + j] << " ";
        }
        std::cout << std::endl;
      }
    }
    //print matrix
    void print(std::ostream &out) const
    {
      for (size_t i = 0; i < size1_; i++)
      {
        for (size_t j = 0; j < size2_; j++)
        {
          out << data_[(i * size2_ )+ j] << " ";
        }
        out << std::endl;
      }
    } 
  protected:
    size_type size1_;
    size_type size2_;
    fixed_array_type data_;
  };
  template <class T>
  bool
  operator==(const matrix<T> &x, const matrix<T> &y)
  {

    if (  x.size1()==y.size1()&&x.size2()==y.size2()  )
     return std::equal(x.begin(), x.end(), y.begin());
    else
      return false;
  }

  template <class T>
  bool
  operator!=(const matrix<T> &x, const matrix<T> &y)
  {

    return !(x == y);
  }

  template <class T>
  bool
  operator<(const matrix<T> &x, const matrix<T> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }

  template <class T>
  bool
  operator>(const matrix<T> &x, const matrix<T> &y)
  {
    return y < x;
  }
  template <class T>
  bool
  operator<=(const matrix<T> &x, const matrix<T> &y)
  {
    return !(y < x);
  }
  template <class T>
  bool
  operator>=(const matrix<T> &x, const matrix<T> &y)
  {
    return !(x < y);
  }

  template <class T>
  matrix<T>
  operator*(const T &scalar, const matrix<T> &b)
  {
    matrix<T> result(b);
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }
  template <class T>
  matrix<T>
  operator*(const matrix<T> &a, const std::vector<T> &b)
  {
    //size_t a_size = a.size1()*a.size2();
    size_t b_size = b.size();
    matrix<T> result(a.size1(), b_size);
    for (size_t i = 0; i < a.size1(); ++i)
    {
      for (size_t k = 0; k < b.size(); ++k)
        result(i, k) = a(i, k) * b[k];
    }
    return result;

     
  }
  template <class T>
  matrix<T>
  operator*(const matrix<T> &a, const matrix<T> &b)
  {
    assert(a.size2() == b.size2());
    matrix<T> result(b.size1(), b.size2());
    for (size_t i = 0; i < b.size1(); ++i)
    {
      for (size_t k = 0; k < b.size2(); ++k)
        result(i, k) = b(i, k) * a(0, k);
    }
    return result;
  }
   
  template <class T, std::size_t N>
  const fixed_array<T, N> &
  fixed_array<T, N>::operator*(const T &a) const
  {
    static fixed_array<T, N> result = *this;

    for (size_t i = 0; i < result.size(); ++i)
      result[i] = result[i] * a;

    return *result;
  }

  template <class T, std::size_t N>
  const fixed_array<T, N> &
  fixed_array<T, N>::operator/(const T &a) const
  {
    static fixed_array<T, N> result = *this;

    for (size_t i = 0; i < result.size(); ++i)
      result[i] = result[i] / a;

    return *result;
  }

  template <class T, std::size_t N>
  fixed_array<T, N>
  operator*(const matrix<T> &a, const fixed_array<T, N> &b)
  {
    assert(a.size2() == N);
    fixed_array<T, N> result;
    for (size_t i = 0; i < result.size(); ++i)
    {
      T sum(0);
      for (size_t k = 0; k < a.size2(); ++k)
        sum += a(i, k) * b[k];
      result[i] = sum;
    }
    return result;
  }

  template <class T>
  matrix<T>
  operator+(const matrix<T> &a, const matrix<T> &b)
  {
  //  assert(a.size1() == b.size1() && a.size2() == b.size2());

    size_t m_col = std::min(a.size2(), b.size2()); 
    size_t m_row = std::min(a.size1(), b.size1());
    matrix<T> result(m_row, m_col);
    for(size_t i=0;i<m_row;i++)
      for(size_t j=0;j<m_col;j++)
        result(i,j) = a(i,j) + b(i,j);

    return result;
  }

  template <class T>
  matrix<T>
  operator-(const matrix<T> &a, const matrix<T> &b)
  {
    size_t m_col = std::min(a.size2(), b.size2());
    size_t m_row = std::min(a.size1(), b.size1());
    matrix<T> result(m_row, m_col);
    for(size_t i=0;i<m_row;i++)
      for(size_t j=0;j<m_col;j++)
        result(i,j) = a(i,j) - b(i,j);

    return result;
   }

  //for x-matrix  x = 1-x
  template <class T>
    const matrix<T> operator-(const T &lval,const matrix<T>& rval) 
    {
      const size_t size1 = rval.size1(), size2 = rval.size2();
      matrix<T> ret(rval);
      for (size_t i = 0; i < size1; i++)
        for (size_t j = 0; j <  size2 ; ++j)
          ret(i, j) = lval - ret(i, j) ;

      return ret;
    }

  template <class T>
  std::ostream &
  operator<<(std::ostream &out, const matrix<T> &a)
  {
    out << std::string("(");

    for (size_t j = 0; j < a.size1(); ++j)
    {
      size_t i = 0;
      out << std::string("(");
      for (; i < a.size2() - 1; ++i)
        out << a(j, i) << std::string(", ");
      out << a(j, i) << std::string(")");
      if (j < a.size1() - 1)
        out << std::string(", ");

    }
    out << std::string(")");
    return out;
  }
  // Dense matrix implementation with static size
  template <class T, std::size_t N, std::size_t M>
  class bounded_matrix
  {
    typedef T *pointer;

  public:
    // Type definitions
    typedef T value_type;
    typedef T *fixed_array_type;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef T &reference;
    typedef const T &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // ---- Construction and destruction

    // Default dense matrix constructor. Make a dense matrix of size (0,0)
    bounded_matrix()
    {
    }

    // Dense matrix constructor with defined size a initial value for all the matrix elements
    bounded_matrix(const value_type &init)
    {
      std::fill(data_, data_ + N * M, init);
    }

    // Dense matrix constructor with defined size and an initial data fixed_array
    bounded_matrix(const fixed_array_type &data)
    {
      std::copy(data, data + N * M, data_);
    }

    // Copy-constructor of a dense matrix
    bounded_matrix(const bounded_matrix<T, N, M> &m)
    {
      std::copy(m.data_, m.data_ + N * M, data_);
    }
    //move constructor of a dense matrix
    bounded_matrix(bounded_matrix<T, N, M> &&m) = delete;
 
    // iterator support
    iterator
    begin()
    {
      return data_;
    }
    const_iterator
    begin() const
    {
      return data_;
    }
    const_iterator
    cbegin() const
    {
      return data_;
    }

    iterator
    end()
    {
      return data_ + N * M;
    }
    const_iterator
    end() const
    {
      return data_ + N * M;
    }
    const_iterator
    cend() const
    {
      return data_ + N * M;
    }

    // reverse iterator support
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    reverse_iterator
    rbegin()
    {
      return reverse_iterator(end());
    }
    const_reverse_iterator
    rbegin() const
    {
      return const_reverse_iterator(end());
    }
    const_reverse_iterator
    crbegin() const
    {
      return const_reverse_iterator(end());
    }

    reverse_iterator
    rend()
    {
      return reverse_iterator(begin());
    }
    const_reverse_iterator
    rend() const
    {
      return const_reverse_iterator(begin());
    }
    const_reverse_iterator
    crend() const
    {
      return const_reverse_iterator(begin());
    }

    // ---- Accessors

    // Return the number of rows of the matrix
    static size_type
    size1()
    {
      return N;
    }

    // Return the number of colums of the matrix
    static size_type
    size2()
    {
      return M;
    }
    // Return a constant reference to the internal storage of a dense matrix, i.e. the raw data
    const fixed_array_type &
    data() const
    {
      return data_;
    }

    // Return a reference to the internal storage of a dense matrix, i.e. the raw data
    fixed_array_type &
    data()
    {
      return data_;
    }

    // Element access
    const_reference
    operator()(size_type i, size_type j) const
    {
      return data_[i * M + j];
    }

    reference
    at_element(size_type i, size_type j)
    {
      return data_[i * M + j];
    }

    reference
    operator()(size_type i, size_type j)
    {
      return data_[i * M + j];
    }

    // Element assignment
    reference
    insert_element(size_type i, size_type j, const_reference t)
    {
      return (at_element(i, j) = t);
    }

    void
    erase_element(size_type i, size_type j)
    {
      at_element(i, j) = value_type();
    }
    operator matrix<T>() const
    {
      matrix<T> ret(N, M);
      for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
          ret(i, j) = data_[i * M + j];
      return ret;
    }
    // Zeroing
    void
    clear()
    {
      std::fill(data_, data_ + N * M, value_type());
    }

    bounded_matrix<T, N, M> &
    operator=(const bounded_matrix<T, N, M> &m)
    {
      std::copy(m.data_, m.data_ + N * M, data_);
      return *this;
    }

  private:
    T data_[N * M];
  };

  

  // comparisons
  template <class T, std::size_t N, std::size_t M>
  bool
  operator==(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return std::equal(x.begin(), x.end(), y.begin());
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator<(const bounded_matrix<T, N, M> &x,
            const bounded_matrix<T, N, M> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator!=(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return !(x == y);
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator>(const bounded_matrix<T, N, M> &x,
            const bounded_matrix<T, N, M> &y)
  {
    return y < x;
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator<=(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return !(y < x);
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator>=(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return !(x < y);
  }

  template <class T, std::size_t N, std::size_t M>
  bounded_matrix<T, N, M>
  operator*(const T &scalar, const bounded_matrix<T, N, M> &b)
  {
    bounded_matrix<T, N, M> result(b);
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }

  // Naive matrix multiplication
  template <class T, std::size_t N1, std::size_t M1, std::size_t M2>
  bounded_matrix<T, N1, M2>
  operator*(const bounded_matrix<T, N1, M1> &a,
            const bounded_matrix<T, M1, M2> &b)
  {
    bounded_matrix<T, N1, M2> result;
    for (size_t i = 0; i < N1; ++i)
    {
      for (size_t j = 0; j < M2; ++j)
      {
        typename bounded_matrix<T, N1, M2>::value_type sum(0);
        for (size_t k = 0; k < M1; ++k)
          sum += a(i, k) * b(k, j);
        result(i, j) = sum;
      }
    }
    return result;
  }

  // Naive matrix multiplication
  template <class T, std::size_t N, std::size_t M>
  std::vector<T>
  operator*(const bounded_matrix<T, N, M> &a, const std::vector<T> &b)
  {
    // Sanity check (will only work in debug compilation)
    assert(M == b.size());
    std::vector<T> result(N);
    for (size_t i = 0; i < N; ++i)
    {
      T sum(0);
      for (size_t k = 0; k < M; ++k)
        sum += a(i, k) * b[k];
      result[i] = sum;
    }
    return result;
  }

  // Naive matrix multiplication
  template <class T, std::size_t N, std::size_t M>
  fixed_array<T, N>
  operator*(const bounded_matrix<T, N, M> &a, const fixed_array<T, M> &b)
  {
    fixed_array<T, N> result;
    for (size_t i = 0; i < N; ++i)
    {
      T sum(0);
      for (size_t k = 0; k < M; ++k)
        sum += a(i, k) * b[k];
      result[i] = sum;
    }
    return result;
  }

  template <class T, std::size_t N, std::size_t M>
  bounded_matrix<T, N, M>
  operator+(const bounded_matrix<T, N, M> &a,
            const bounded_matrix<T, N, M> &b)
  {
    // Sanity check (will only work in debug compilation)
    assert(a.size1() == b.size1());
    assert(a.size2() == b.size2());
    bounded_matrix<T, N, M> result(a);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  template <class T, std::size_t N, std::size_t M>
  bounded_matrix<T, N, M>
  operator-(const bounded_matrix<T, N, M> &a,
            const bounded_matrix<T, N, M> &b)
  {
    // Sanity check (will only work in debug compilation)
    assert(a.size1() == b.size1());
    assert(a.size2() == b.size2());
    bounded_matrix<T, N, M> result(a);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::minus<T>());
    return result;
  }

  // Initialize fixed_array
  template <class T, std::size_t N>
  void
  resizeArray(fixed_array<T, N> *values, size_t dim)
  {
  }
  

  template <class T>
  void
  resizeArray(std::vector<T> *values, size_t dim)
  {
    values->resize(dim);
  }

  template <typename Function, typename Array>
  static typename Array::value_type
  central_deriv(Function function, Array &point,
                const typename Array::value_type &h,
                typename Array::size_type i,
                typename Array::value_type *abserr_round,
                typename Array::value_type *abserr_trunc)
  {
    typedef typename Array::value_type value_type;

    // Save original value
    value_type orig = point[i];
    value_type eps = std::numeric_limits<value_type>::epsilon();

    // Compute the derivative using the 5-point rule (x-h, x-h/2, x,
    // x+h/2, x+h). Note that the central point is not used.
    // Compute the error using the difference between the 5-point and
    // the 3-point rule (x-h,x,x+h).
    point[i] = orig - h;
    value_type fm1 = function(point);
    point[i] = orig + h;
    value_type fp1 = function(point);

    point[i] = orig - h / 2;
    value_type fmh = function(point);
    point[i] = orig + h / 2;
    value_type fph = function(point);

    value_type r3 = 0.5 * (fp1 - fm1);
    value_type r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;

    value_type e3 = (fabs(fp1) + fabs(fm1)) * eps;
    value_type e5 = 2.0 * (fabs(fph) + fabs(fmh)) * eps + e3;

    value_type dy = std::max(fabs(r3 / h), fabs(r5 / h)) * (fabs(orig) / h) * eps;

    // The truncation error in the r5 approximation itself is O(h^4).
    // However, for safety, we estimate the error from r5-r3, which is
    // O(h^2).  By scaling h we will minimize this estimated error, not
    // the actual truncation error in r5.
    *abserr_trunc = fabs((r5 - r3) / h);
    *abserr_round = fabs(e5 / h) + dy;

    // Before leave put the original value into the point
    point[i] = orig;

    return r5 / h;
  }

  template <typename Function, typename Array>
  typename Array::value_type
  partial_derivative(Function function, Array &point,
                     const typename Array::value_type &h,
                     typename Array::size_type i)
  {
    typedef typename Array::value_type value_type;
    value_type round, trunc;
    value_type r_0 = central_deriv(function, point, h, i, &round, &trunc);
    value_type error = round + trunc;

    if (round < trunc && (round > 0 && trunc > 0))
    {
      value_type round_opt, trunc_opt;
      // Compute an optimized step-size to minimize the total error,
      // using the scaling of the truncation error (O(h^2)) and
      // rounding error (O(1/h)). */
      value_type h_opt = h * pow(round / (2.0 * trunc), 1.0 / 3.0);
      value_type r_opt = central_deriv(function, point, h_opt, i,
                                       &round_opt, &trunc_opt);
      value_type error_opt = round_opt + trunc_opt;

      // Check that the new error is smaller, and that the new derivative
      // is consistent with the error bounds of the original estimate.
      if (error_opt < error && fabs(r_opt - r_0) < 4.0 * error)
      {
        r_0 = r_opt;
        error = error_opt;
      }
    }

    return r_0;
  }
  // Gradient operator (partial derivative in all components)
  template <typename Function, typename Array>
  Array
  gradient(Function function, Array &point,
           const typename Array::value_type &h)
  {
    
    //typedef typename Array::value_type value_type;
    typedef typename Array::size_type size_type;
    // Initialize and copy fixed_array
    Array result(point);
    for (size_type i = 0; i < result.size(); ++i)
      result[i] = partial_derivative(function, point, h, i);
    return result;
  }

  // Partial derivative functor
  template <typename Function, typename Array>
  class PartialDeriv
  {
    typedef typename Array::value_type value_type;
    typedef typename Array::size_type size_type;

    // Internal function
    Function _function;
    // Step
    value_type _h;
    // Variable
    size_type _i;

  public:
    PartialDeriv(const Function &function, const value_type &h,
                 const size_type &i) : _function(function), _h(h), _i(i)
    {
    }

    // Calculate partial derivative
    value_type
    operator()(Array &point) const
    {
      return partial_derivative(_function, point, _h, _i);
    }

    ~PartialDeriv()
    {
    }
  };

  template <typename Function, typename Array>
  class Gradient
  {
    typedef typename Array::value_type value_type;
    typedef typename Array::size_type size_type;

    // Internal function
    Function _function;
    // Step
    value_type _h;

  public:
    Gradient(const Function &function, const value_type &h) : _function(function), _h(h)
    {
    }

    // Calculate partial derivative
    Array
    operator()(Array &point) const
    {
      return gradient(_function, point, _h);
    }

    ~Gradient()
    {
    }
  };

  // Calculate hessian matrix
  template <typename Function, typename Array>
  matrix<typename Array::value_type>
  hessian(Function function, Array &point,
          const typename Array::value_type &h)
  {
    // Hessian matrix
    matrix<typename Array::value_type> mhessian(point.size(),
                                                point.size());
    // Loop over each element of the matrix
    for (typename Array::size_type i = 0; i < point.size(); ++i)
    {
      PartialDeriv<Function, Array> first_deriv(function, h, i);
      for (typename Array::size_type j = 0; j < point.size(); ++j)
      {
        // Calculate second derivative (i,j)
        PartialDeriv<PartialDeriv<Function, Array>, Array> second_deriv(
            first_deriv, h, j);
        mhessian(i, j) = second_deriv(point);
      }
    }
    // Return result
    return mhessian;
  }

  // Calculate hessian matrix
  template <typename Function, typename FloatType, std::size_t N>
  bounded_matrix<FloatType, N, N>
  bounded_hessian(Function function, fixed_array<FloatType, N> &point,
                  const typename fixed_array<FloatType, N>::value_type &h)
  {
    // Hessian matrix
    bounded_matrix<FloatType, N, N> mhessian;
    // Loop over each element of the matrix
    for (typename fixed_array<FloatType, N>::size_type i = 0; i < point.size();
         ++i)
    {
      PartialDeriv<Function, fixed_array<FloatType, N>> first_deriv(function, h,
                                                              i);
      for (typename fixed_array<FloatType, N>::size_type j = 0; j < point.size();
           ++j)
      {
        // Calculate second derivative (i,j)
        PartialDeriv<PartialDeriv<Function, fixed_array<FloatType, N>>,
                     fixed_array<FloatType, N>>
            second_deriv(first_deriv, h, j);
        mhessian(i, j) = second_deriv(point);
      }
    }
    // Return result
    return mhessian;
  }

  // 1) calculate cblas zgemm :
  // 2) sort results
  // 3) find max
  // 4) return unary expression labmda of max value with 0.
  // 5) to eliminate complex factor of the number
  template <typename T>
  struct Quaternion;
  template <typename T>
  struct RollPitchYaw;
  template <typename T>
  struct AxisAngle;
  template <typename T>
  struct Vec3;

  template <typename T>
  struct RollPitchYaw
  {
    // Angles in radians
    T roll;
    T pitch;
    T yaw;

    // Coordinate system:
    // x forward, roll around x, positive rotation clockwise
    // y left, pitch around y, positive rotation down
    // z up, yaw around z, positive rotation to the left

    RollPitchYaw(const T roll_, const T pitch_, const T yaw_);
    RollPitchYaw();

    matrix<T> toRotationMatrix() const;
    Quaternion<T> toQuaternion() const;
    AxisAngle<T> toAxisAngle() const;
  };

  template <typename T>
  struct AxisAngle
  {
    T phi;
    T x;
    T y;
    T z;

    AxisAngle(const T phi_, const T x_, const T y_, const T z_);
    AxisAngle(const T x_, const T y_, const T z_);
    AxisAngle(const Vec3<T> &v);
    AxisAngle();

    AxisAngle<T> normalized() const;

    matrix<T> toRotationMatrix() const;
    Quaternion<T> toQuaternion() const;
    RollPitchYaw<T> toRollPitchYaw() const;
  };

  template <typename T>
  struct Quaternion
  {
    T w; // real
    T x; // imaginary
    T y; // imaginary
    T z; // imaginary

    Quaternion(const T w_, const T x_, const T y_, const T z_) : w(w_), x(x_), y(y_), z(z) {}
    Quaternion() = default;

    matrix<T> toRotationMatrix() const;
    AxisAngle<T> toAxisAngle() const;
    RollPitchYaw<T> toRollPitchYaw() const;

    T norm() const;
    T squaredNorm() const;
    Quaternion<T> normalized() const;
  };

  template <typename T>
  RollPitchYaw<T> Quaternion<T>::toRollPitchYaw() const
  {
    RollPitchYaw<T> rpy;
    // Roll
    T sinr_cosp = 2.0 * (w * x + y * z);
    T cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    rpy.roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch
    T sinp = 2.0 * (w * y - z * x);
    if (std::fabs(sinp) >= 1)
    {
      rpy.pitch = std::copysign(M_PI / 2.0, sinp); // Use 90 degrees if out of range
    }
    else
    {
      rpy.pitch = std::asin(sinp);
    }

    // Yaw
    T siny_cosp = 2.0 * (w * z + x * y);
    T cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    rpy.yaw = std::atan2(siny_cosp, cosy_cosp);
    return rpy;
  }

  template <typename T>
  matrix<T> Quaternion<T>::toRotationMatrix() const
  {
    matrix<T> m(3, 3);

    const Quaternion<T> qn = this->normalized();
    const T qr = qn.w;
    const T qi = qn.x;
    const T qj = qn.y;
    const T qk = qn.z;

    m(0, 0) = 1.0 - 2.0 * (qj * qj + qk * qk);
    m(0, 1) = 2.0 * (qi * qj - qk * qr);
    m(0, 2) = 2.0 * (qi * qk + qj * qr);
    m(1, 0) = 2.0 * (qi * qj + qk * qr);
    m(1, 1) = 1.0 - 2.0 * (qi * qi + qk * qk);
    m(1, 2) = 2.0 * (qj * qk - qi * qr);
    m(2, 0) = 2.0 * (qi * qk - qj * qr);
    m(2, 1) = 2.0 * (qj * qk + qi * qr);
    m(2, 2) = 1.0 - 2.0 * (qi * qi + qj * qj);

    return m;
  }
  template <typename T>
  T Quaternion<T>::norm() const
  {
    return std::sqrt(w * w + x * x + y * y + z * z);
  }
  template <typename T>
  T Quaternion<T>::squaredNorm() const
  {
    return w * w + x * x + y * y + z * z;
  }

  template <typename T>
  Quaternion<T> Quaternion<T>::normalized() const
  {
    const T d = this->norm();
    return Quaternion<T>(w / d, x / d, y / d, z / d);
  }

  template <typename T>
  Quaternion<T> operator*(const Quaternion<T> &q, const Quaternion<T> &p)
  {
    Vec3<T> qv = Vec3<T>(q.x, q.y, q.z);
    Vec3<T> pv = Vec3<T>(p.x, p.y, p.z);
    Vec3<T> intermediate_vector = qv.crossProduct(pv) + q.w * pv + p.w * qv;
    return Quaternion<T>(q.w * p.w - pv * qv, intermediate_vector.x, intermediate_vector.y, intermediate_vector.z);
  }

  template <typename T>
  Quaternion<T> rotationMatrixToQuaternion(const matrix<T> &m)
  {
    // Reference:
    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Quaternion<T> q;
    q.w = std::sqrt<T>(1.0 + m(0, 0) + m(1, 1) + m(2, 2)) / 2.0;
    q.x = (m(2, 1) - m(1, 2)) / (4.0 * q.w);
    q.y = (m(0, 2) - m(2, 0)) / (4.0 * q.w);
    q.z = (m(1, 0) - m(0, 1)) / (4.0 * q.w);

    return q;
  }

  template <typename T>
  Quaternion<T> RollPitchYaw<T>::toQuaternion() const
  {
    T cy = std::cos(yaw * 0.5);
    T sy = std::sin(yaw * 0.5);
    T cp = std::cos(pitch * 0.5);
    T sp = std::sin(pitch * 0.5);
    T cr = std::cos(roll * 0.5);
    T sr = std::sin(roll * 0.5);

    Quaternion<T> q;

    q.w = cy * cp * cr + sy * sp * sr;
    q.x = cy * cp * sr - sy * sp * cr;
    q.y = sy * cp * sr + cy * sp * cr;
    q.z = sy * cp * cr - cy * sp * sr;

    return q;
  }
  template <typename T>
  AxisAngle<T> RollPitchYaw<T>::toAxisAngle() const
  {
    Quaternion<T> q = toQuaternion();
    return q.toAxisAngle();
  }
  template <typename T>
  matrix<T> rotationMatrixFromYaw(const T yaw)
  {
    matrix<T> m(3, 3);
    const T ca = std::cos(yaw);
    const T sa = std::sin(yaw);

    m(0, 0) = ca;
    m(0, 1) = -sa;
    m(0, 2) = 0.0;

    m(1, 0) = sa;
    m(1, 1) = ca;
    m(1, 2) = 0.0;

    m(2, 0) = 0.0;
    m(2, 1) = 0.0;
    m(2, 2) = 1.0;

    return m;
  }

  template <typename T>
  matrix<T> rotationMatrixFromRoll(const T roll)
  {
    matrix<T> m(3, 3);
    const T ca = std::cos(roll);
    const T sa = std::sin(roll);

    m(0, 0) = 1.0;
    m(0, 1) = 0.0;
    m(0, 2) = 0.0;

    m(1, 0) = 0.0;
    m(1, 1) = ca;
    m(1, 2) = -sa;

    m(2, 0) = 0.0;
    m(2, 1) = sa;
    m(2, 2) = ca;

    return m;
  }
  template <typename T>
  matrix<T> rotationMatrixFromPitch(const T pitch)
  {
    matrix<T> m(3, 3);
    const T ca = std::cos(pitch);
    const T sa = std::sin(pitch);

    m(0, 0) = ca;
    m(0, 1) = 0.0;
    m(0, 2) = sa;

    m(1, 0) = 0.0;
    m(1, 1) = 1.0;
    m(1, 2) = 0.0;

    m(2, 0) = -sa;
    m(2, 1) = 0.0;
    m(2, 2) = ca;

    return m;
  }
  template <typename T>
  matrix<T> RollPitchYaw<T>::toRotationMatrix() const
  {
    return rotationMatrixFromYaw(yaw) * rotationMatrixFromPitch(pitch) * rotationMatrixFromRoll(roll);
  }

  template <typename T>
  RollPitchYaw<T> rotationMatrixToRollPitchYaw(const matrix<T> &m)
  {
    return RollPitchYaw<T>(std::atan2<T>(m(2, 1), m(2, 2)), std::asin<T>(-m(2, 0)), std::atan2<T>(m(1, 0), m(0, 0)));
  }

  template <typename std_type, typename value_type>
  struct blas_multiplier
  {

    size_t max_matrix_count;

  public:
    const std_type &operator()(const std_type &matrices, const matrix<value_type> &transpose)
    {
      const std::complex<value_type> alpha = 1.0;
      const std::complex<value_type> beta = 0.0;
      size_t npoints = transpose.rows();

      size_t count = matrices.size();
      static std_type vResult(count);
      for (size_t k = 0; k < count; ++k)
      {
        size_t CblasColMajor = 0, CblasNoTrans = 0;
        matrix<value_type> a(npoints, npoints);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, npoints, npoints, npoints, &alpha,
                    matrices[k].data(), npoints, transpose.data(), npoints, &beta, a.data(), npoints);

#ifdef _WIN32
        vResult[k] = MatrixXd(Npts, Npts);
        for (int i = 0; i < Npts; ++i)
          for (int j = 0; j < Npts; ++j)
            vResult[k](i, j) = max(a(i, j).real(), 0.0);
#else
        vResult[k] = a.unaryExpr([](std::complex<value_type> x)
                                 { return max(x.real(), 0.0); });
#endif
      }
      return vResult;
    }
  };

  // General quadratic form
  template <typename FloatType, std::size_t N>
  struct Quadratic
  {
    // Quadratic term
    bounded_matrix<FloatType, N, N> _a;
    // Linear term
    fixed_array<FloatType, N> _b;
    // Constant
    FloatType _c;

  public:
    Quadratic(const bounded_matrix<FloatType, N, N> &a,
              fixed_array<FloatType, N> b /*= fixed_array<FloatType,N>()*/,
              const FloatType &c = FloatType()) : _a(a), _b(b), _c(c)
    {
    }

    FloatType
    operator()(const fixed_array<FloatType, N> &point) const
    {
      FloatType quad = dot(point, (_a * point));
      return (0.5 * quad - dot(_b, point) + _c);
    }
    ~Quadratic()
    {
    }
  };

  // Transpose matrix without calling matrix<T>::transpose()
  template <typename T>
  matrix<T>
  transpose(const matrix<T> &mat)
  {
    matrix<T> transp(mat.size2(), mat.size1());
    for (typename matrix<T>::size_type i = 0; i < mat.size2(); ++i)
      for (typename matrix<T>::size_type j = 0; j < mat.size1(); ++j)
        transp(i, j) = mat(j, i);
    return transp;
  }

  // Transpose matrix without calling matrix<T>::transpose()
  template <typename T, std::size_t N, std::size_t M>
  bounded_matrix<T, M, N>
  transpose(const bounded_matrix<T, N, M> &mat)
  {
    bounded_matrix<T, M, N> transp;
    for (typename bounded_matrix<T, N, M>::size_type i = 0; i < M; ++i)
      for (typename bounded_matrix<T, N, M>::size_type j = 0; j < N; ++j)
        transp(i, j) = mat(j, i);
    return transp;
  }

  // Transpose a vector
  template <typename T>
  matrix<T>
  transpose(const std::vector<T> &vec)
  {
    matrix<T> transp(1, vec.size());
    for (typename matrix<T>::size_type i = 0; i < vec.size(); ++i)
      transp(0, i) = vec[i];
    return transp;
  }

  // Transpose a vector
  template <typename T, std::size_t N>
  matrix<T>
  transpose(const fixed_array<T, N> &vec)
  {
    matrix<T> transp(1, vec.size());
    for (typename matrix<T>::size_type i = 0; i < N; ++i)
      transp(0, i) = vec[i];
    return transp;
  }

  // Get identity matrix
  template <typename FloatType>
  matrix<FloatType>
  identity(std::size_t n)
  {
    matrix<FloatType> ident(n, n, FloatType());
    for (size_t i = 0; i < n; ++i)
      ident(i, i) = 1.0;
    return ident;
  }

  // Check if a vector (fixed_array) has nan values
  template <typename T>
  bool
  isNan(const std::vector<T> &vec)
  {
    for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++i)
      if (isnan(vec[i]))
        return true;

    return false;
  }

  template <typename T, std::size_t N>
  bool
  isNan(const fixed_array<T, N> &vec)
  {
    for (typename fixed_array<T, N>::size_type i = 0; i < N; ++i)
      if (isnan(vec[i]))
        return true;
    return false;
  }

  // utilities with matrices :

  //jacobi utitlity for  matrix rot
  template <typename T>
  inline void rot(matrix<T> &a, T s, T tau, const size_t i, const size_t j, const size_t k, const size_t l)
  {
    T g = a(i, j), h = a(k, l);
    a(i, j) = g - s * (h + g * tau);
    a(k, l) = h + s * (g - h * tau);
  }
  // jacobi 
  template <typename T>
  void slvsml(matrix<T> &out, matrix<T> &rhs)
  {
    // size_t i,j;
    T h(0.5);
    for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 3; ++j)
        out(i, j) = T(0.);

    out(1, 1) = -h * h * rhs(1, 1) / 4.;
  }
  //jacobi rstrct utility :

  template <typename T>
  void rstrct(matrix<T> &uc, matrix<T> &uf)
  {
    // coarse grid:
    size_t ic, iif, jc, jf, ncc;
    size_t nc = uc.rows();
    ncc = 2 * nc - 2;
    for (jf = 2, jc = 1; jc < nc - 1; jc++, jf += 2)
    {
      for (iif = 2, ic = 1; ic < nc - 1; ic++, iif += 2)
      {
        uc(ic, jc) = T(0.5) * uf(iif, jf - 1) + T(0.125) * uf(iif + 1, jf) + uf(iif - 1, jf) + uf(iif, jf + 1) + uf(iif, jf - 1);
      }
    }
    for (jc = 0, ic = 0; jc < nc; ic++, jc += 2)
    {
      uc(ic, 0) = uf(jc, 0);
      uc(ic, nc - 1) = uf(jc, ncc);
    }

    for (jc = 0, ic = 0; ic < nc; ic++, jc += 2)
    {
      uc(0, ic) = uf(0, jc);
      uc(nc - 1, ic) = uf(ncc, jc);
    }
  }
  //  interpolate from coarse to fine grid  :
  //  uc : coarse grid
  //  uf : fine grid

  template <typename T>
  void interp(matrix<T> &uf, matrix<T> &uc)
  {

    size_t ic, iif, jc, jf, nc;
    const size_t nf = uf.rows();
    nc = size_t(T(nf) / 2. + 1.);
    for (jc = 0; jc < nc; jc++)
      for (ic = 0; ic < nc; ic++)
        uf(2 * ic, 2 * jc) = uc(ic, jc);
    for (jf = 0; jf < nf; jf += 2)
      for (iif = 1; iif < nf - 1; iif++)
        uf(iif, jf) = T(.5) * (uf(iif + 1, jf) + uf(iif - 1, jf));
    for (jf = 1; jf < nf; jf += 2)
      for (iif = 0; iif < nf; iif++)
        uf(iif, jf) = T(.5) * (uf(iif, jf + 1) + uf(iif, jf - 1));
  }
  //    add interpolation to uf from uc
  template <typename T>
  void addint(matrix<T> &uf, matrix<T> &uc, matrix<T> &res)
  {
    size_t i, j;
    size_t nf = uf.rows();
    interp(res, uc);
    for (j = 0; j < nf; j++)
      for (i = 0; i < nf; i++)
        uf(i, j) += res(i, j);
  }
  //
  // compute residual r = b - A*x
  // A is the matrix, x is the solution, b is the right hand side
  // r is the residual
  //
  template <typename T>
  void mg(int j, matrix<T> &u, matrix<T> &rhs)
  {
    const size_t PRE = 1, NPOST = 1;
    size_t npost, jpre, nc, nf;
    nf = u.rows();
    nc = reinterpret_cast<size_t>(T((nf) + 1.0) / 2.);
    if (j == 0)
      slvsml(u, rhs);
    else
    {
      matrix<T> res(nc, nc), v(matrix<T>::Constant(nc, nc, T(0.))), temp(nf, nf);
      // pre relaxation
      for (jpre = 0; jpre < PRE; jpre++)
        relax(u, rhs);
      // residual computation
      resid(temp, u, rhs);
      // restriction
      rstrct(res, temp);
      // recursive call
      mg(j - 1, v, res);
      // interpolation
      addint(u, v, temp);
      // post relaxation
      for (size_t jpost = 0; jpost < NPOST; jpost++)
        relax(u, rhs);
    }
  }

  // Jacobi method
  // a1 : matrix
  // d : diagonal
  // v : eigenvectors
  // nrot : number of rotations
  template <typename T>
  void jacobi(const matrix<T>& a1, std::vector<T>& d, matrix<T>& v, size_t& nrot) {
    if (a1.rows() != a1.cols()) {
      throw std::invalid_argument("Matrix must be square");
    }
    size_t n = a1.rows();
    d.resize(n);
    v.resize(n, n);
    for (size_t i = 0; i < n; ++i) {
      v(i, i) = 1.;
    }
    jacobi_helper(a1, d, v, nrot);
  }

  template <typename T>
  void jacobi_helper(const matrix<T>& a, std::vector<T>& d, matrix<T>& v, size_t& nrot) {
    size_t n = d.size();
    matrix<T> b(n), z(n);
    jacobi_loop(a, d, b, z, v, nrot);
  }

  template <typename T>
  void jacobi_loop(const matrix<T>& a, std::vector<T>& d, matrix<T>& b, matrix<T>& z, matrix<T>& v, size_t& nrot) {
    size_t n = d.size();
    for (size_t ip = 0; ip < n; ++ip) {
      b[ip] = d[ip] = a(ip, ip);
      z[ip] = T(0.);
    }
    nrot = 0;
    for (size_t i = 0; i <= 50; ++i) {
      T sm = jacobi_sm(a, n);
      if (sm == 0.0) {
        return;
      }
      T thresh = (i < 4) ? .2 * sm / T(n * n) : 0.;
      //jacobi_thresh(a, d, b, z, v, nrot, thresh, ip, iq, n);
      jacobi_thresh(a, d, b, z, v, nrot, thresh, i, n);
    }
  }

  template <typename T>
  T jacobi_sm(const matrix<T>& a, size_t n) {
    T sm = 0.;
    for (size_t ip = 0; ip < n - 1; ++ip) {
      for (size_t iq = ip + 1; iq < n; ++iq) {
        sm += std::fabs(a(ip, iq));
      }
    }
    return sm;
  }

  template <typename T>
  void jacobi_thresh(const matrix<T>& a, std::vector<T>& d, matrix<T>& b, matrix<T>& z, matrix<T>& v, size_t& nrot, T thresh, size_t& ip, size_t& iq, size_t n) {
    for (ip = 0; ip < n; ++ip) {
      for (iq = ip + 1; iq < n; ++iq) {
        if (std::fabs(a(ip, iq)) > thresh) {
          jacobi_rot(a, d, b, z, v, nrot, ip, iq, n);
        }
      }
    }
  }

  template <typename T>
  void jacobi_rot(const matrix<T>& a, std::vector<T>& d, matrix<T>& b, matrix<T>& z, matrix<T>& v, size_t& nrot, size_t ip, size_t iq, size_t n) {
    T g, h, theta, t, sm, s, tau, c,thresh=0.0;
    g = 100. * std::fabs(a(ip, iq));
    h = d[iq] - d[ip];
    if (ip > 0 && iq + 1 < n && std::fabs(d[ip]) + g == std::fabs(d[ip]) && (std::fabs(d[iq]) + g == std::fabs(d[iq]))) {
      a(ip, iq) = 0.;
    } else if (std::fabs(a(ip, iq)) > thresh) {
      if ((std::fabs(h) + g) == std::fabs(h)) {
        t = (a(ip, iq) / h);
      } else {
        theta = .5 * h / a(ip, iq);
        t = 1. / (std::fabs(theta) + std::sqrt(1. + (theta * theta)));
        if (theta < 0.) {
          t = -t;
        }
      }
      c = 1. / std::sqrt(1. + (t * t));
      s = t * c;
      tau = s / (1. + c);
      h = t * a(ip, iq);
      z[ip] -= h;
      z[iq] += h;
      d[ip] -= h;
      d[iq] += h;
      a(ip, iq) = 0.;
      jacobi_rot_update(a, s, tau, ip, iq, n);
    }
    jacobi_update(b, z, d, n);
    jacobi_update(v, s, tau, ip, iq, n);
    nrot++;
  }

  template <typename T>
  void jacobi_rot_update(matrix<T>& a, T s, T tau, size_t ip, size_t iq, size_t n) {
    for (size_t j = 0; j < ip; ++j) {
      rot(a, s, tau, j, ip, j, iq);
    }
    for (size_t j = ip + 1; j < iq; ++j) {
      rot(a, s, tau, ip, j, j, iq);
    }
    for (size_t j = iq + 1; j < n; ++j) {
      rot(a, s, tau, ip, j, iq, j);
    }
  }

  template <typename T>
  void jacobi_update(matrix<T>& b, matrix<T>& z, std::vector<T>& d, size_t n) {
    for (size_t ip = 0; ip < n; ++ip) {
      b[ip] += z[ip];
      d[ip] += b[ip];
      z[ip] = 0.;
    }
  }

  // Jacobi method (return eigenvalues)
  template <typename T>
  matrix<T> jacobi(const matrix<T> &a, size_t &nrot)
  {
    std::vector<T> diagonal(a.rows());
    for (size_t i = 0; i < a.rows(); ++i)
      diagonal[i] = a(i, i);

    matrix<T> v(a.size1(), a.size2());
    jacobi<T>(a, diagonal, v, nrot);
    return v;
  }

  // Class to create gaussian points using Box-Muller
  template <typename Float>
  class Gaussian
  {
    // Mean
    Float _mean;
    // Standard deviation
    Float _stdev;

  public:
    Gaussian(Float mean, Float stdev) : _mean(mean), _stdev(stdev)
    {
    }

    // Generate point
    Float
    operator()() const
    {
      // Auxiliary parameters
      std::random_device rd  ;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> uniform(0, RAND_MAX);

      Float theta = 2.0 * M_PI * uniform(gen);
      Float rho = std::sqrt(-2.0 * std::log(1.0 - uniform(gen)));
      // Sample point (using Box-Muller)
      Float x = _mean + _stdev * rho * cos(theta);
      Float y = _mean + _stdev * rho * sin(theta);
      return std::sqrt(x * x + y * y);
    }
    //set mean
    void set_mean(Float mean)
    {
      _mean = mean;
    }
    //set standard deviation
    void set_stddev(Float stdev)
    {
      _stdev = stdev;
    }
    //set variance
    void set_variance(Float variance)
    {
      _stdev = std::sqrt(variance);
    }

    ~Gaussian()
    {
    }
  };
   // Spherical distribution
  template <class Array>
  class SphericalPoint
  {
    typedef typename Array::value_type float_type;
    // Gaussian functor
    Gaussian<float_type> _gauss;
    // Radius
    float_type _radius;
    // Dimension
    size_t _dim;

  public:
    SphericalPoint(size_t dim, float_type radius = 1.0) : _gauss(0, 1), _radius(radius), _dim(dim)
    {
    }

    Array
    operator()(std::random_device &r) const
    {
      // Buffer
      std::mt19937 gen(r());
      std::uniform_int_distribution<> uniform(0, RAND_MAX);

      Array _buffer;
      resizeArray(&_buffer, _dim);
      for (size_t i = 0; i < _buffer.size(); ++i)
        _buffer[i] = _gauss(uniform(gen));
      _buffer = (_radius / sqrt(dot(_buffer, _buffer))) * _buffer;
      return _buffer;
    }

    ~SphericalPoint()
    {
    }
  };

  // global operators for X/matrix
  template <typename T>
  matrix<T> operator/(const T &b, const matrix<T> &a)
  {
    matrix<T> ret(a.size1(), a.size2());
    for (typename matrix<T>::size_type i = 0; i < a.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < a.size2(); j++)
      {
        ret(i, j) = a(i, j) / b;
      }
    }

    return ret;
  }
  //    
  template <typename T>
  matrix<T> operator/(const matrix<T> &a, const T &b)
  {
    matrix<T> ret(a.size1(), a.size2());
    for (typename matrix<T>::size_type i = 0; i < a.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < a.size2(); j++)
      {
        ret(i, j) = a(i, j) / b;
      }
    }

    return ret;
  } // end operator/

  template <typename T>
  matrix<T> operator/(const matrix<T> &a, const matrix<T> &b)
  {
    matrix<T> ret(a.size1(), a.size2());
    for (typename matrix<T>::size_type i = 0; i < a.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < a.size2(); j++)
      {
        ret(i, j) = a(i, j) / b(i, j);
      }
    }

    return ret;
  } // end operator/     
  //template friend ofstream/ifstream operators 
  template <typename T>
  std::ofstream &operator<<(std::ofstream &ofs, const matrix<T> &m)
  {
    ofs << m.size1() << " " << m.size2() << std::endl;
    for (typename matrix<T>::size_type i = 0; i < m.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < m.size2(); j++)
      {
        ofs << m(i, j) << " ";
      }
      ofs << std::endl;
    }
    return ofs;
  } 
  template <typename T>
  std::ifstream &operator>>(std::ifstream &ifs, matrix<T> &m)
  {
    typename matrix<T>::size_type rows, cols;
    ifs >> rows >> cols;
    m.resize(rows, cols);
    for (typename matrix<T>::size_type i = 0; i < m.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < m.size2(); j++)
      {
        ifs >> m(i, j);
      }
    }
    return ifs;
  } 


  //Kolmorogov-Smirnov test 
  template <typename T>
  T kolmogorov_smirnov(const matrix<T> & data )
  {
    T d = 0.0;
    for (typename matrix<T>::size_type i = 0; i < data.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < data.size2(); j++)
      {
        T d1 = std::abs(data(i, j) - T(0.5));
        if (d1 > d)
          d = d1;
      }
    } // end for

    return d;
  } // end kolmogorov_smirnov
  //Tikhonov regularization for matrix a and b with lambda 

  template <typename T> 
  matrix<T> tikhonov(const matrix<T> &a, const matrix<T> &b, const T &lambda)
  {
    matrix<T> a_transpose = a.transpose();
    matrix<T> a_transpose_a = a_transpose * a; 
    matrix<T> a_transpose_b = a_transpose * b; 
    matrix<T> a_transpose_a_lambda = a_transpose_a + lambda 
    *a_transpose_a.identity(); 
    matrix<T> a_transpose_a_lambda_inv = a_transpose_a_lambda.inverse(); 
    matrix<T> x = a_transpose_a_lambda_inv * a_transpose_b; 
    return x;
  }
} /* namespace provallo */ ;
#endif /* DECISION_ENGINE_MATRIX_H_ */
