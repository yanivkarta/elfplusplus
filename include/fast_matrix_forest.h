#ifndef _FAST_MATRIX_FOREST_H_
#define _FAST_MATRIX_FOREST_H_

//fast_matrix_forest.h
//is a collection of matrices that can be used to replace trees,nodes,leafs and forests 
//in decision trees and random forests 

//super_tree is a matrix of indices that can be used to access the forest 
//super_tree_probabilities is a matrix of probabilities that can be used to access the forest
//super_tree_values is a matrix of values that can be used to access the forest

//super_tree_values_projection is a matrix of values that can be used to access the forest
//super_tree_hplane is a matrix of hyperplanes that can be used to access the forest


#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <set>
#include <cmath>
#include <numeric>
#include <functional>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <limits>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include "matrix.h" 
namespace provallo { 
    
    

//each matrix is a transition matrix from a tree structure
//the tree structure is a vector of vectors of discrete or continous values 
//the tree_matrix is a matrix of probabilities

struct  matrix_indices : public std::pair<size_t,size_t>
{
    matrix_indices():std::pair<size_t,size_t>(0,0){} 
    matrix_indices(size_t i,size_t j):std::pair<size_t,size_t>(i,j){}
    size_t i() const {return first;}
    size_t j() const {return second;}
    size_t & i() {return first;}    
    size_t & j() {return second;}

    size_t path_len() const {return (0-first) * (0-second);} 
    bool operator==(const matrix_indices & other) const
    {
        return (first==other.first && second==other.second);
    }
    bool operator!=(const matrix_indices & other) const
    {
        return !(*this==other);
    }
    bool operator<(const matrix_indices & other) const
    {
        return (first<other.first || (first==other.first && second<other.second));
    }
    bool operator>(const matrix_indices & other) const
    {
        return (first>other.first || (first==other.first && second>other.second));
    }
    bool operator<=(const matrix_indices & other) const
    {
        return (first<=other.first || (first==other.first && second<=other.second));
    }
    bool operator>=(const matrix_indices & other) const
    {
        return (first>=other.first || (first==other.first && second>=other.second));
    }
    matrix_indices operator+(const matrix_indices & other) const
    {
        return matrix_indices(first+other.first,second+other.second);
    }
    matrix_indices operator-(const matrix_indices & other) const
    {
        return matrix_indices(first-other.first,second-other.second);
    }
    matrix_indices operator*(const matrix_indices & other) const
    {
        return matrix_indices(first*other.first,second*other.second);
    }
    matrix_indices operator/(const matrix_indices & other) const
    {
        return matrix_indices(first/other.first,second/other.second);
    }
    matrix_indices operator%(const matrix_indices & other) const
    {
        return matrix_indices(first%other.first,second%other.second);
    }
    matrix_indices operator+(const size_t & other) const
    {
        return matrix_indices(first+other,second+other);
    }
    matrix_indices operator-(const size_t & other) const
    {
        return matrix_indices(first-other,second-other);
    }   
    matrix_indices operator*(const size_t & other) const
    {
        return matrix_indices(first*other,second*other);
    }   
    matrix_indices operator/(const size_t & other) const
    {
        return matrix_indices(first/other,second/other);
    }   
    //arithmetic operators
    matrix_indices & operator+=(const matrix_indices & other) 
    {
        first+=other.first;
        second+=other.second;
        return *this;
    }
    matrix_indices & operator-=(const matrix_indices & other) 
    {
        first-=other.first;
        second-=other.second;
        return *this;
    }   
    matrix_indices & operator*=(const matrix_indices & other) 
    {
        first*=other.first;
        second*=other.second;
        return *this;
    }
    matrix_indices & operator/=(const matrix_indices & other) 
    {
        first/=other.first;
        second/=other.second;
        return *this;
    }
    matrix_indices & operator%=(const matrix_indices & other) 
    {
        first%=other.first;
        second%=other.second;
        return *this;
    }
    matrix_indices & operator+=(const size_t & other) 
    {
        first+=other;
        second+=other;
        return *this;
    }
    matrix_indices & operator-=(const size_t & other) 
    {
        first-=other;
        second-=other;
        return *this;
    }
    matrix_indices & operator*=(const size_t & other) 
    {
        first*=other;
        second*=other;
        return *this;
    }
    matrix_indices & operator/=(const size_t & other) 
    {
        first/=other;
        second/=other;
        return *this;
    }
    //logical operators
    bool operator!() const
    {
        return (first==0 && second==0);
    }
    bool operator&&(const matrix_indices & other) const
    {
        return (first && other.first && second && other.second);
    }
    bool operator||(const matrix_indices & other) const
    {
        return (first || other.first || second || other.second);
    }
    bool operator&&(const size_t & other) const
    {
        return (first && other && second && other);
    }
    bool operator||(const size_t & other) const
    {
        return (first || other || second || other);
    }
    bool operator==(const size_t & other) const
    {
        return (first==other && second==other);
    }
    bool operator!=(const size_t & other) const
    {
        return (first!=other || second!=other);
    }
    friend std::ostream & operator<<(std::ostream & os,const matrix_indices & other)
    {
        os<<"("<<other.first<<","<<other.second<<")";
        return os;
    }   
    friend std::istream & operator>>(std::istream & is,matrix_indices & other)
    {
        is>>other.first>>other.second;
        return is;
    }
    static std::atomic_uint64_t matrix_indices_count;

};  
    //matrix indices replaces PathLength of IsoForest and PathLengths of ExtIsoForest 
    typedef struct tag_hyperplane
	{
        uint64_t hplane_id;
        matrix_indices hplane_indices; //indices of the hplane in the super_tree 
        size_t hplane_depth;
        size_t hplane_level;
        size_t hplane_parent;
        size_t hplane_left;
        size_t hplane_right;
        std::random_device rd;
        std::mt19937 gen;
        
        size_t hplane_dim;
        size_t hplane_feature;
        size_t hplane_feature_index;
        size_t hplane_feature_index_left;
        size_t hplane_feature_index_right;
        real_t hplane_feature_value;
        real_t hplane_feature_value_left;
        real_t hplane_feature_value_right;
        real_t hplane_feature_value_min;
        real_t hplane_feature_value_max;
        real_t hplane_feature_value_range;
        real_t weight;
        real_t score;
        std::normal_distribution<real_t> distribution;        
        //hplane constructor
        tag_hyperplane():hplane_id(++hplane_count),hplane_indices(matrix_indices(0,0)),hplane_depth(0),hplane_level(0),hplane_parent(0),hplane_left(0),hplane_right(0),hplane_dim(0),hplane_feature(0),hplane_feature_index(0),hplane_feature_index_left(0),hplane_feature_index_right(0),hplane_feature_value(0.0),hplane_feature_value_left(0.0),hplane_feature_value_right(0.0),hplane_feature_value_min(0.0),hplane_feature_value_max(0.0),hplane_feature_value_range(0.0),weight(0.0),score(0.0),distribution(0.0,1.0){}
        //copy constructor
        tag_hyperplane(const tag_hyperplane & other):hplane_id(other.hplane_id),hplane_indices(other.hplane_indices),hplane_depth(other.hplane_depth),hplane_level(other.hplane_level),hplane_parent(other.hplane_parent),hplane_left(other.hplane_left),hplane_right(other.hplane_right),hplane_dim(other.hplane_dim),hplane_feature(other.hplane_feature),hplane_feature_index(other.hplane_feature_index),hplane_feature_index_left(other.hplane_feature_index_left),hplane_feature_index_right(other.hplane_feature_index_right),hplane_feature_value(other.hplane_feature_value),hplane_feature_value_left(other.hplane_feature_value_left),hplane_feature_value_right(other.hplane_feature_value_right),hplane_feature_value_min(other.hplane_feature_value_min),hplane_feature_value_max(other.hplane_feature_value_max),hplane_feature_value_range(other.hplane_feature_value_range),weight(other.weight),score(other.score),distribution(other.distribution){}
        //move constructor
        tag_hyperplane(tag_hyperplane && other):hplane_id(other.hplane_id),hplane_indices(other.hplane_indices),hplane_depth(other.hplane_depth),hplane_level(other.hplane_level),hplane_parent(other.hplane_parent),hplane_left(other.hplane_left),hplane_right(other.hplane_right),hplane_dim(other.hplane_dim),hplane_feature(other.hplane_feature),hplane_feature_index(other.hplane_feature_index),hplane_feature_index_left(other.hplane_feature_index_left),hplane_feature_index_right(other.hplane_feature_index_right),hplane_feature_value(other.hplane_feature_value),hplane_feature_value_left(other.hplane_feature_value_left),hplane_feature_value_right(other.hplane_feature_value_right),hplane_feature_value_min(other.hplane_feature_value_min),hplane_feature_value_max(other.hplane_feature_value_max),hplane_feature_value_range(other.hplane_feature_value_range),weight(other.weight),score(other.score),distribution(other.distribution){}
        //copy assignment
        tag_hyperplane & operator=(const tag_hyperplane & other)
        {
            if(this!=&other)
            {
                //hplane_id=other.hplane_id;
                hplane_indices=other.hplane_indices;
                hplane_depth=other.hplane_depth;
                hplane_level=other.hplane_level;
                hplane_parent=other.hplane_parent;
                hplane_left=other.hplane_left;
                hplane_right=other.hplane_right;
                hplane_dim=other.hplane_dim;
                hplane_feature=other.hplane_feature;
                hplane_feature_index=other.hplane_feature_index;
                hplane_feature_index_left=other.hplane_feature_index_left;
                hplane_feature_index_right=other.hplane_feature_index_right;
                hplane_feature_value=other.hplane_feature_value;
                hplane_feature_value_left=other.hplane_feature_value_left;
                hplane_feature_value_right=other.hplane_feature_value_right;
                hplane_feature_value_min=other.hplane_feature_value_min;
                hplane_feature_value_max=other.hplane_feature_value_max;
                hplane_feature_value_range=other.hplane_feature_value_range;
                weight=other.weight;
                score=other.score;
                distribution=other.distribution;

            }
            return *this;
        }   
        //move assignment
        tag_hyperplane & operator=(tag_hyperplane && other)
        {
            if(this!=&other)
            {
                hplane_id=std::move(other.hplane_id);
                hplane_indices = std::move(other.hplane_indices);
                hplane_depth=std::move(other.hplane_depth);
                hplane_level=std::move(other.hplane_level);
                hplane_parent=std::move(other.hplane_parent);
                hplane_left=std::move(other.hplane_left);
                hplane_right=std::move(other.hplane_right);
                hplane_dim=std::move(other.hplane_dim);
                hplane_feature=std::move(other.hplane_feature);
                hplane_feature_index=std::move(other.hplane_feature_index);
                hplane_feature_index_left=std::move(other.hplane_feature_index_left);
                hplane_feature_index_right=std::move(other.hplane_feature_index_right); 
                hplane_feature_value=std::move(other.hplane_feature_value);
                hplane_feature_value_left=std::move(other.hplane_feature_value_left);
                hplane_feature_value_right=std::move(other.hplane_feature_value_right);
                hplane_feature_value_min=std::move(other.hplane_feature_value_min);

                hplane_feature_value_max=std::move(other.hplane_feature_value_max); 
                hplane_feature_value_range=std::move(other.hplane_feature_value_range);
                weight=std::move(other.weight);
                score=std::move(other.score);
                distribution=std::move(other.distribution);
                
                //other.hplane_id=0;
                //other.hplane_indices=matrix_indices(0,0);
                //other.hplane_depth=0;
                //other.hplane_level=0;
                //other.hplane_parent=0;
                
            }
            return *this;
        }   
        //destructor
        ~tag_hyperplane(){}
        //comparison operators
        bool operator==(const tag_hyperplane & other) const
        {
            return (hplane_id==other.hplane_id && hplane_indices==other.hplane_indices && hplane_depth==other.hplane_depth && hplane_level==other.hplane_level && hplane_parent==other.hplane_parent && hplane_left==other.hplane_left && hplane_right==other.hplane_right && hplane_dim==other.hplane_dim && hplane_feature==other.hplane_feature && hplane_feature_index==other.hplane_feature_index && hplane_feature_index_left==other.hplane_feature_index_left && hplane_feature_index_right==other.hplane_feature_index_right && hplane_feature_value==other.hplane_feature_value && hplane_feature_value_left==other.hplane_feature_value_left && hplane_feature_value_right==other.hplane_feature_value_right && hplane_feature_value_min==other.hplane_feature_value_min && hplane_feature_value_max==other.hplane_feature_value_max && hplane_feature_value_range==other.hplane_feature_value_range && weight==other.weight && score==other.score);
        }   
        bool operator!=(const tag_hyperplane & other) const
        {
            return !(*this==other);
        }
        //arithmetic operators
        tag_hyperplane& operator+(const tag_hyperplane & other)
        {
           this->hplane_indices=this->hplane_indices+other.hplane_indices;  

           return *this;
        }   
        tag_hyperplane& operator-(const tag_hyperplane& other)
        {
            this->hplane_indices-=other.hplane_indices;
            return *this;
        }
        tag_hyperplane& operator*(const tag_hyperplane& other)
        {
            this->hplane_indices*=other.hplane_indices;
            return *this;
        }
        tag_hyperplane& operator/(const tag_hyperplane& other)
        {
            this->hplane_indices/=other.hplane_indices;
            return *this;
        }
        //logical operators
        bool operator!() const
        {
            return (hplane_id==0 && hplane_indices==matrix_indices(0,0) && hplane_depth==0 && hplane_level==0 && hplane_parent==0 && hplane_left==0 && hplane_right==0 && hplane_dim==0 && hplane_feature==0 && hplane_feature_index==0 && hplane_feature_index_left==0 && hplane_feature_index_right==0 && hplane_feature_value==0.0 && hplane_feature_value_left==0.0 && hplane_feature_value_right==0.0 && hplane_feature_value_min==0.0 && hplane_feature_value_max==0.0 && hplane_feature_value_range==0.0 && weight==0.0 && score==0.0);
        }

        void print(std::ostream& os)const 
        {
            os<<"hplane_id="<<hplane_id<<std::endl;
            os<<"hplane_indices="<<hplane_indices<<std::endl;
            os<<"hplane_depth="<<hplane_depth<<std::endl;
            os<<"hplane_level="<<hplane_level<<std::endl;
            os<<"hplane_parent="<<hplane_parent<<std::endl;

            os<<"hplane_left="<<hplane_left<<std::endl;
            os<<"hplane_right="<<hplane_right<<std::endl;


            os<<"hplane_dim="<<hplane_dim<<std::endl;
            os<<"hplane_feature="<<hplane_feature<<std::endl;
            os<<"hplane_feature_index="<<hplane_feature_index<<std::endl;

            os<<"hplane_feature_index_left="<<hplane_feature_index_left<<std::endl;
            os<<"hplane_feature_index_right="<<hplane_feature_index_right<<std::endl;

 
            os<<"hplane_feature_value="<<hplane_feature_value<<std::endl;
            os<<"hplane_feature_value_left="<<hplane_feature_value_left<<std::endl;

            os<<"hplane_feature_value_right="<<hplane_feature_value_right<<std::endl;
            os<<"hplane_feature_value_min="<<hplane_feature_value_min<<std::endl;
            os<<"hplane_feature_value_max="<<hplane_feature_value_max<<std::endl;
            os<<"hplane_feature_value_range="<<hplane_feature_value_range<<std::endl;
            os<<"weight="<<weight<<std::endl;
            os<<"score="<<score<<std::endl;
            os<<"distribution="<<distribution<<std::endl;

            
        }
        //operator <<
        friend std::ostream & operator<<(std::ostream & os , const tag_hyperplane& other )
        {
            
            other.print(os);
            return os;
                        
       }

    
        static std::atomic_uint64_t hplane_count;
        //ifstream/ofstream
        friend std::ifstream & operator>>(std::ifstream & is,tag_hyperplane & other)
        {
            is>>other.hplane_id;
            is>>other.hplane_indices;
            is>>other.hplane_depth;
            is>>other.hplane_level;
            is>>other.hplane_parent;
            is>>other.hplane_left;
            is>>other.hplane_right;
            is>>other.hplane_dim;
            is>>other.hplane_feature;
            is>>other.hplane_feature_index;
            is>>other.hplane_feature_index_left;
            is>>other.hplane_feature_index_right;
            is>>other.hplane_feature_value;
            is>>other.hplane_feature_value_left;
            is>>other.hplane_feature_value_right;
            is>>other.hplane_feature_value_min;
            is>>other.hplane_feature_value_max;
            is>>other.hplane_feature_value_range;
            is>>other.weight;
            is>>other.score;
            //is>>other.distribution;
            return is;
        }   
        friend std::ofstream & operator<<(std::ofstream & os,const tag_hyperplane & other)
        {
            os<<other.hplane_id<<std::endl;
            os<<other.hplane_indices<<std::endl;
            os<<other.hplane_depth<<std::endl;
            os<<other.hplane_level<<std::endl;
            os<<other.hplane_parent<<std::endl;
            os<<other.hplane_left<<std::endl;
            os<<other.hplane_right<<std::endl;
            os<<other.hplane_dim<<std::endl;
            os<<other.hplane_feature<<std::endl;
            os<<other.hplane_feature_index<<std::endl;
            os<<other.hplane_feature_index_left<<std::endl;
            os<<other.hplane_feature_index_right<<std::endl;
            os<<other.hplane_feature_value<<std::endl;
            os<<other.hplane_feature_value_left<<std::endl;
            os<<other.hplane_feature_value_right<<std::endl;
            os<<other.hplane_feature_value_min<<std::endl;
            os<<other.hplane_feature_value_max<<std::endl;
            os<<other.hplane_feature_value_range<<std::endl;
            os<<other.weight<<std::endl;
            os<<other.score<<std::endl;
            //os<<other.distribution<<std::endl;
            return os;
        }   

        //support for matrix<hplane> : 
        // 1. matrix<hplane> m(10,10);
        // 2. m(0,0)=hplane(
        // 3. m(0,0).hplane_id=0;
        // 4. m(0,0).hplane_indices=matrix_indices(0,0);
        // 5. m(0,0).hplane_depth=0;
        // 6. m(0,0).hplane_level=0;
        // 7. m(0,0).hplane_parent=0;
        ///
        //operators:    

	} hplane; 
    //hyperplane  

template < typename T , typename U = std::vector<T> >   
class super_tree {
      private:
        std::vector<std::vector<U>> _forest; //path lengths of the forest 
        provallo::matrix<matrix_indices> _super_tree;
        provallo::matrix<real_t> _super_tree_probabilities;
        provallo::matrix<T> _super_tree_values;
        provallo::matrix<T> _super_tree_values_projection;
        provallo::matrix<provallo::hplane> _super_tree_hplane;
        std::random_device rd;
        //metrics

        public:
        //support for make_unique:
        
        //constructors
        
        
        //super_tree constructor
        super_tree( const matrix<T>& data,const std::vector<U>& labels ,size_t nfeatures,size_t nsamples ):
        _forest(1),
        _super_tree(nsamples,nfeatures),
        _super_tree_probabilities(nsamples,nfeatures),
        _super_tree_values(nsamples,nfeatures),
        _super_tree_values_projection(nsamples,nfeatures), 
        _super_tree_hplane(nsamples,nfeatures) 

        {
            real_t min = data.minCoeff();
            real_t max = data.maxCoeff();
            real_t sum = data.sum();
             //update labels
            for(size_t i=0;i<nsamples;i++) 
            {
                for(size_t j=0;j<nfeatures;++j)
                {

                    
                    size_t label = labels[i];


                    //use the hyperplane distribution to set the hyperplane values 

                    
                    
                    

                    //set the structure ( depth,level,dim,right....)

                    //std::cout<<"i="<<i<<" j="<<j<<" data(i,j)="<<data(i,j)<<std::endl; 
                    _super_tree(i,j)=matrix_indices(i,j); 
                    _super_tree_probabilities(i,j)=data.row_sum(i)*data.col_sum(j) / sum;  
                    _super_tree_values(i,j)=data(i,j);
                    _super_tree_values_projection(i,j)=data(i,j);
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=i+j; 
                    _super_tree_hplane(i,j).hplane_depth= i;
                    _super_tree_hplane(i,j).hplane_level= j;
                    _super_tree_hplane(i,j).hplane_parent=  i+j>0?i+j-1:0; 
                    _super_tree_hplane(i,j).hplane_left=  i+j>0?i-1+j-1:0;
                    _super_tree_hplane(i,j).hplane_right=   j+nfeatures-1-i>0?i+1+j-1:0; 
                    _super_tree_hplane(i,j).hplane_dim=  j; 
                    _super_tree_hplane(i,j).hplane_feature = j;
                    _super_tree_hplane(i,j).hplane_feature_index = j; 
                    _super_tree_hplane(i,j).hplane_feature_index_left = j-1; 
                    _super_tree_hplane(i,j).hplane_feature_index_right = j+1; 
                    _super_tree_hplane(i,j).hplane_feature_value = data(i,j); 
                    _super_tree_hplane(i,j).hplane_feature_value_left = j>0?data(i,j-1):0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right = j+1<nfeatures?data(i,j+1):0.0; 
                    _super_tree_hplane(i,j).hplane_feature_value_min =min; 
                    _super_tree_hplane(i,j).hplane_feature_value_max = max; 
                    _super_tree_hplane(i,j).hplane_feature_value_range = max-min; 
                    _super_tree_hplane(i,j).weight = min/max*data(i,j);  
                    _super_tree_hplane(i,j).score = data(i,j) - min/max*data(i,j)+eta; 

                    _super_tree_hplane(i,j).distribution=std::normal_distribution<real_t>(min,max); 

                    //update the labels
                    _super_tree_values_projection(i,j)=T(label); 
                    _super_tree_values(i,j)=T(label); 
                    _super_tree_probabilities(i,j)=data.row_sum(i)*data.col_sum(j) / sum; 
                    

                }
            }
            
        }
        //without counts (default from data) :
        super_tree( const matrix<T>& data,std::vector<U>& labels ) : super_tree(data,labels,data.cols(),data.rows()){} 
        
        super_tree(const std::vector<U> & forest):
        _forest(forest),
        _super_tree(forest.size(),forest[0].size()),
        _super_tree_probabilities(forest.size(),
        forest[0].size()),
        _super_tree_values(forest.size(),forest[0].size()),
        _super_tree_values_projection(forest.size(),forest[0].size()),
        _super_tree_hplane(forest.size(),forest[0].size())  {
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree(i,j)=matrix_indices(i,j);
                    _super_tree_probabilities(i,j)=0.0;
                    _super_tree_values(i,j)=forest[i][j];
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0); 
                     
                }
            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree):_forest(forest),_super_tree(super_tree)
        {
            _super_tree_probabilities.resize(forest.size(),forest[0].size());
            _super_tree_values.resize(forest.size(),forest[0].size());
            _super_tree_values_projection.resize(forest.size(),forest[0].size());
            _super_tree_hplane.resize(forest.size(),forest[0].size());
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree_probabilities(i,j)=0.0;
                    _super_tree_values(i,j)=forest[i][j];
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0);

                }

            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree,const provallo::matrix<real_t> & super_tree_probabilities):_forest(forest),_super_tree(super_tree),_super_tree_probabilities(super_tree_probabilities)
        {
            _super_tree_values.resize(forest.size(),forest[0].size());
            _super_tree_values_projection.resize(forest.size(),forest[0].size());
            _super_tree_hplane.resize(forest.size(),forest[0].size());
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree_values(i,j)=forest[i][j];
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0);
                }
            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree,const provallo::matrix<real_t> & super_tree_probabilities,const provallo::matrix<T> & super_tree_values):_forest(forest),_super_tree(super_tree),_super_tree_probabilities(super_tree_probabilities),_super_tree_values(super_tree_values)
        {
            _super_tree_values_projection.resize(forest.size(),forest[0].size());
            _super_tree_hplane.resize(forest.size(),forest[0].size());
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0);
                }
            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree,const provallo::matrix<real_t> & super_tree_probabilities,const provallo::matrix<T> & super_tree_values,const provallo::matrix<T> & super_tree_values_projection,const provallo::matrix<provallo::hplane> & super_tree_hplane):_forest(forest),_super_tree(super_tree),_super_tree_probabilities(super_tree_probabilities),_super_tree_values(super_tree_values),_super_tree_values_projection(super_tree_values_projection),_super_tree_hplane(super_tree_hplane){}
        //copy constructor
        super_tree(const super_tree & other):_forest(other._forest),_super_tree(other._super_tree),_super_tree_probabilities(other._super_tree_probabilities),_super_tree_values(other._super_tree_values),_super_tree_values_projection(other._super_tree_values_projection),_super_tree_hplane(other._super_tree_hplane){}
        //move constructor
        super_tree(super_tree && other):_forest(std::move(other._forest)),_super_tree(std::move(other._super_tree)),_super_tree_probabilities(std::move(other._super_tree_probabilities)),_super_tree_values(std::move(other._super_tree_values)),_super_tree_values_projection(std::move(other._super_tree_values_projection)),_super_tree_hplane(std::move(other._super_tree_hplane)){}
        //copy assignment

        super_tree & operator=(const super_tree & other)
        {
            if(this!=&other)
            {
                _forest=other._forest;
                _super_tree=other._super_tree;
                _super_tree_probabilities=other._super_tree_probabilities;
                _super_tree_values=other._super_tree_values;
                _super_tree_values_projection=other._super_tree_values_projection;
                _super_tree_hplane=other._super_tree_hplane;
            }
            return *this;
        }
        //move assignment
        super_tree & operator=(super_tree && other)
        {
            if(this!=&other)
            {
                _forest=std::move(other._forest);
                _super_tree=std::move(other._super_tree);
                _super_tree_probabilities=std::move(other._super_tree_probabilities);
                _super_tree_values=std::move(other._super_tree_values);
                _super_tree_values_projection=std::move(other._super_tree_values_projection);
                _super_tree_hplane=std::move(other._super_tree_hplane);
            }
            return *this;
        }

        //destructor
        virtual ~super_tree() = default;
        //getters
        const std::vector<U> & forest() const {return _forest;}
        std::vector<U> & forest() {return _forest;}
        const provallo::matrix<matrix_indices> & get_super_tree() const {return _super_tree;}
        provallo::matrix<matrix_indices> & get_super_tree() {return _super_tree;}
        const provallo::matrix<real_t> & super_tree_probabilities() const {return _super_tree_probabilities;}
        provallo::matrix<real_t> & super_tree_probabilities() {return _super_tree_probabilities;}
        const provallo::matrix<T> & super_tree_values() const {return _super_tree_values;}
        provallo::matrix<T> & super_tree_values() {return _super_tree_values;}
        const provallo::matrix<T> & super_tree_values_projection() const {return _super_tree_values_projection;}
        provallo::matrix<T> & super_tree_values_projection() {return _super_tree_values_projection;}
         
        const provallo::matrix<provallo::hplane> & super_tree_hplane() const {return _super_tree_hplane;}
        provallo::matrix<provallo::hplane> & super_tree_hplane() {return _super_tree_hplane;}
        //setters
        void set_forest(const std::vector<U> & forest) {_forest=forest;}
        void set_super_tree(const provallo::matrix<matrix_indices> & super_tree) {_super_tree=super_tree;}
        void set_super_tree_probabilities(const provallo::matrix<real_t> & super_tree_probabilities) {_super_tree_probabilities=super_tree_probabilities;}
        void set_super_tree_values(const provallo::matrix<T> & super_tree_values) {_super_tree_values=super_tree_values;}
        void set_super_tree_values_projection(const provallo::matrix<T> & super_tree_values_projection) {_super_tree_values_projection=super_tree_values_projection;}
        void set_super_tree_hplane(const provallo::hplane & super_tree_hplane) {_super_tree_hplane=super_tree_hplane;}
        //operators
        bool operator==(const super_tree & other) const
        {
            return (_forest==other._forest && _super_tree==other._super_tree && _super_tree_probabilities==other._super_tree_probabilities && _super_tree_values==other._super_tree_values && _super_tree_values_projection==other._super_tree_values_projection && _super_tree_hplane==other._super_tree_hplane);
        }       
        bool operator!=(const super_tree & other) const
        {
            return !(*this==other);
        }

        //methods
        void print(std::ostream & os=std::cout) const
        {
            os<<"forest:"<<std::endl;
            for(size_t i=0;i<_forest.size();i++)
            {
                os<<"tree "<<i<<":";
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    os<<_forest[i][j]<<" ";
                }
                os<<std::endl;
            }
            os<<"super_tree:"<<std::endl;
            _super_tree.print(os);
            os<<"super_tree_probabilities:"<<std::endl;
            _super_tree_probabilities.print(os);
            os<<"super_tree_values:"<<std::endl;
            _super_tree_values.print(os);
            os<<"super_tree_values_projection:"<<std::endl;
            _super_tree_values_projection.print(os);
            //os<<"super_tree_hplane:"<<std::endl;
           // _super_tree_hplane.print(os);
        }

        //transform the super_tree into a matrix of probabilities
        //and set the hplane values according to random projections 
        //of the super_tree values

       inline void initialize_hplanes()
       {
            //initialize hplanes
            //set hplane values according to random projections of the super_tree values
            //set hplane values according to random projections of the super_tree values


            //initialize hplanes
            this->gen = std::mt19937(this->rd()); 

            
            std::uniform_real_distribution<> dis(0.0, 1.0);
            std::uniform_int_distribution<> dis_rows(0, _super_tree_values.rows()-1);
            std::uniform_int_distribution<> dis_cols(0, _super_tree_values.cols()-1);   
            std::uniform_int_distribution<> dis_levels(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_depths(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_parents(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_lefts(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_rights(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_dims(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_features(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_min(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_max(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_range(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_weights(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_scores(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_probabilities(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_values(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_values_projection(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_hplanes(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_forest(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_super_tree(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_super_tree_probabilities(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_super_tree_values(0, _super_tree_values.cols()-1);


            //project the super_tree values into the hplane values 
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_values_projection(i,j)=_super_tree_values(i,j)*_super_tree_hplane(i,j).hplane_feature_value;
                }
            }
            //set hplane values according to random projections of the super_tree values
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    //initialize hplane distribution
                    _super_tree_hplane(i,j).distribution=std::normal_distribution<real_t>(_super_tree_values.minCoeff(),_super_tree_values.maxCoeff()); 
                    //update hplane values according to random projections of the super_tree values 

                    _super_tree_hplane(i,j).hplane_id=dis_hplanes(this->gen);
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(dis_rows(this->gen),dis_cols(this->gen));
                    _super_tree_hplane(i,j).hplane_depth=dis_depths(this->gen);
                    _super_tree_hplane(i,j).hplane_level=dis_levels(this->gen);
                    _super_tree_hplane(i,j).hplane_parent=dis_parents(this->gen);
                    _super_tree_hplane(i,j).hplane_left=dis_lefts(this->gen);
                    _super_tree_hplane(i,j).hplane_right=dis_rights(this->gen);
                    _super_tree_hplane(i,j).hplane_dim=dis_dims(this->gen);
                    _super_tree_hplane(i,j).hplane_feature=dis_features(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_index=dis_feature_indices(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_index_left=dis_feature_indices_left(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_index_right=dis_feature_indices_right(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_value=dis_feature_values(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_value_left=dis_feature_values_left(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_value_right=dis_feature_values_right(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_value_min=dis_feature_values_min(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_value_max=dis_feature_values_max(this->gen);
                    _super_tree_hplane(i,j).hplane_feature_value_range=dis_feature_values_range(this->gen);
                    _super_tree_hplane(i,j).weight=dis_weights(this->gen);
                    _super_tree_hplane(i,j).score=dis_scores(this->gen);
                }
            }
            
       }
        inline void initialize_probabilities()
        {
            //initialize probabilities
            //set probabilities according to random projections of the super_tree values
            //set probabilities according to random projections of the super_tree values    
            //initialize probabilities
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            std::uniform_real_distribution<> dis_scores(0.0, 1.0);
            std::uniform_real_distribution<> dis_weights(0.0, 1.0);
            std::uniform_int_distribution<> dis_rows(0, _super_tree_values.rows()-1);
            std::uniform_int_distribution<> dis_cols(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_levels(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_depths(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_parents(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_lefts(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_rights(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_dims(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_features(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_min(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_max(0, _super_tree_values.cols()-1); 
            real_t sum = 0.0;

            //  //project the super_tree values into the probabilities
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {   
                    //get the hplane of the super_tree
                    hplane& hplane=_super_tree_hplane(i,j);
                    //update hplane values according to random projections of the super_tree values 
                    hplane.hplane_feature_value=_super_tree_values(i,j)*dis_feature_values(gen); 
                    hplane.hplane_depth=dis_depths(gen);
                    hplane.hplane_level=dis_levels(gen);
                    hplane.hplane_parent=dis_parents(gen);
                    hplane.hplane_left=dis_lefts(gen);
                    hplane.hplane_right=dis_rights(gen);
                    hplane.hplane_dim=dis_dims(gen);
                    hplane.hplane_feature=dis_features(gen);
                    hplane.hplane_feature_index=dis_feature_indices(gen);
                    hplane.hplane_feature_index_left=dis_feature_indices_left(gen);
                    hplane.hplane_feature_index_right=dis_feature_indices_right(gen);
                    hplane.hplane_feature_value_left=dis_feature_values_left(gen);
                    hplane.hplane_feature_value_right=dis_feature_values_right(gen);
                    hplane.hplane_feature_value_min=dis_feature_values_min(gen);
                    hplane.hplane_feature_value_max=dis_feature_values_max(gen);
                     //update probabilities according to random projections of the super_tree values
                    _super_tree_probabilities(i,j)=_super_tree_values(i,j)*hplane.hplane_feature_value;
                    hplane.weight=dis_weights(gen);

                    hplane.score=dis_scores(gen)/_super_tree_values(i,j);
                    sum+=_super_tree_probabilities(i,j);

                 }
                 //normalize probabilities
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_probabilities(i,j)/=sum;
                }
            }       
             //normalize probabilities
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                real_t sum = 0.0;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    sum+=_super_tree_probabilities(i,j);
                }
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_probabilities(i,j)/=sum;
                }

            }

            //for each tree in the forest update the probabilities of the leaves 
            //according to the probabilities of the super_tree

            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    _forest[i][j]=_super_tree_probabilities(i,j);
                    sum+=_forest[i][j];

                }
            }   
            //for each tree in the forest normalize the probabilities of the leaves 
            //according to the probabilities of the super_tree  
            for(size_t i=0;i<_forest.size();i++)
            {
           
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    _forest[i][j]/=sum;
                }
            }   
           //normalize probabilities
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_probabilities(i,j)/=sum;
                }

            }
            //for each node in the super_tree update the probabilities of the leaves
            //according to the probabilities of the super_tree
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree(i,j).i() =_super_tree_probabilities(i,j) / _super_tree_probabilities.minCoeff(); 
                    _super_tree(i,j).j() =_super_tree_probabilities(i,j) / _super_tree_probabilities.maxCoeff();  
                
                }
            }
            //for each node in the super_tree normalize the probabilities of the leaves
            //according to the probabilities of the super_tree
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                real_t sum = 0.0;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    sum+=_super_tree_values(i,j);
                }
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_values(i,j)/=sum;
                }
            }             
        }
        real_t get_leaf_probability(const size_t & i,const size_t & j) const
        {
            return _super_tree_probabilities(i,j);
        }   
        real_t get_leaf_value(const size_t & i,const size_t & j) const
        {
            return _super_tree_values(i,j);
        }   
        real_t get_leaf_value_projection(const size_t & i,const size_t & j) const
        {
            return _super_tree_values_projection(i,j);
        }   
        real_t get_leaf_hplane(const size_t & i,const size_t & j) const
        {
            return _super_tree_hplane(i,j).hplane_feature_value;
        }   
        real_t get_leaf_forest(const size_t & i,const size_t & j) const
        {
            return _forest[i][j];
        }   
        real_t get_leaf_super_tree(const size_t & i,const size_t & j) const
        {
            return _super_tree_values(i,j);
        }
        //get trees,nodes,leaves
        std::vector<U> get_trees() const
        {
            return _forest;
        }   
        std::vector<T> get_nodes() const
        {
            std::vector<T> nodes;
            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    nodes.push_back(_forest[i][j]);
                }
            }
            return nodes;
        }   
        std::vector<T> get_leaves() const
        {
            std::vector<T> leaves;
            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    leaves.push_back(_forest[i][j]);
                }
            }
            return leaves;
        }   
        //get super_tree,nodes,leaves
        
        provallo::matrix<T> get_super_tree_nodes() const
        {
            return _super_tree_values;
        }
        provallo::matrix<T> get_super_tree_leaves() const
        {
            return _super_tree_values;
        }
        //get super_tree_probabilities,nodes,leaves
        //same probabilities for nodes and leaves
        provallo::matrix<real_t> get_super_tree_probabilities() const
        {
            return _super_tree_probabilities;
        }
        provallo::matrix<real_t> get_super_tree_nodes_probabilities() const
        {
            return _super_tree_probabilities;
        }   
        provallo::matrix<real_t> get_super_tree_leaves_probabilities() const
        {
            return _super_tree_probabilities;
        }   
        //get super_tree_values,nodes,leaves
        provallo::matrix<T> get_super_tree_values() const
        {
            return _super_tree_values;
        }   
        provallo::matrix<T> get_super_tree_nodes_values() const
        {
            return _super_tree_values;
        }
        provallo::matrix<T> get_super_tree_leaves_values() const
        {
            return _super_tree_values;
        }
        //get super_tree_values_projection,nodes,leaves
        provallo::matrix<T> get_super_tree_values_projection() const
        {
            return _super_tree_values_projection;
        }
        void process_hplanes()
        {
            //calculate hplane values for each node,leaf and tree:
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_hplane(i,j).hplane_feature_value=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_dim=i;
                    _super_tree_hplane(i,j).hplane_feature=j;
                    _super_tree_hplane(i,j).hplane_feature_index=i;
                    _super_tree_hplane(i,j).hplane_feature_index_left=i;
                    _super_tree_hplane(i,j).hplane_feature_index_right=i;
                    _super_tree_hplane(i,j).hplane_feature_value_left=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_right=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_min=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_max=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_range=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).weight=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).score= _super_tree_values_projection(i,j)/_super_tree_values(i,j);
                    real_t projected_value = _super_tree_values_projection(i,j); 
                    real_t hyperplane_projected_value= _super_tree_hplane(i,j).distribution(gen) * _super_tree_hplane(i,j).hplane_feature_value; 
 
                    real_t projection_intersection = _super_tree_hplane(i,j).distribution(gen) * _super_tree_values(i,j);
                    real_t hyperplane_projected_intersection = _super_tree_hplane(i,j).distribution(gen) * _super_tree_hplane(i,j).hplane_feature_value;     
 
                     //update the probabilities 
                    _super_tree_probabilities(i,j)=projected_value/_super_tree_values(i,j); 
                    //update the hplane values according to the probabilities
                    _super_tree_hplane(i,j).hplane_feature_value=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_left=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_right=projected_value;
                    //update the super_tree projections according to the hplane values 
                    _super_tree_values_projection(i,j)=projected_value; 
                    _super_tree_hplane(i,j).hplane_feature_value=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_left=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_right=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_min=projection_intersection; 
                    _super_tree_hplane(i,j).hplane_feature_value_max=hyperplane_projected_intersection;
                    _super_tree_hplane(i,j).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection; 
                    //update weight
                    matrix_indices hplane_indices=_super_tree_hplane(i,j).hplane_indices; 
                    //calculate 'path-lens' from the root to the leaf by comparing the indices of the hplanes 
                    real_t path = hplane_indices.i()+hplane_indices.j(); 
                    //calculate the weight of the leaf by comparing the path-lens of the leaf and the root 
                    _super_tree_hplane(i,j).weight=1.0/path; 
                    //update the score
                    _super_tree_hplane(i,j).score= _super_tree_values_projection(i,j)/_super_tree_values(i,j);
                    //update the hplane values according to the probabilities 
                    _super_tree_hplane(i,j).hplane_feature_value=projected_value; 
                    _super_tree_hplane(i,j).hplane_feature_value_left=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_right=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_min=projection_intersection; 
                    _super_tree_hplane(i,j).hplane_feature_value_max=hyperplane_projected_intersection;
                    _super_tree_hplane(i,j).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection;  
                    //update the hplane values according to the probabilities 
                    _super_tree_hplane(i,j).hplane_feature_value=projected_value; 
                    _super_tree_hplane(i,j).hplane_feature_value_left=projected_value - _super_tree_hplane(i,j).hplane_feature_value_range; 
                    _super_tree_hplane(i,j).hplane_feature_value_right=projected_value + _super_tree_hplane(i,j).hplane_feature_value_range;  
                    _super_tree_hplane(i,j).hplane_feature_value_min=hyperplane_projected_value - _super_tree_hplane(i,j).hplane_feature_value_range;  
                    _super_tree_hplane(i,j).hplane_feature_value_max=hyperplane_projected_value + _super_tree_hplane(i,j).hplane_feature_value_range;  
                    _super_tree_hplane(i,j).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection; 
                    
                }
            }   
         
        }//process_hplanes
        //get super_tree_hplane,nodes,leaves
        provallo::matrix<provallo::hplane> get_super_tree_hplane() const
        {
            return _super_tree_hplane;
        }
        provallo::matrix<provallo::hplane> get_super_tree_nodes_hplane() const
        {
            return _super_tree_hplane;
        }
        provallo::matrix<provallo::hplane> get_super_tree_leaves_hplane() const
        {
            return _super_tree_hplane;
        }
        //get super_tree_forest,nodes,leaves
        std::vector<std::vector<U>> get_super_tree_forest() const
        {
            std::vector<std::vector<U>> super_tree_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<U> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree(i,j));
                }
                super_tree_forest.push_back(tree);
            }
            return super_tree_forest;
        }
        std::vector<U> get_super_tree_nodes_forest() const
        {
            std::vector<U> super_tree_nodes_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    super_tree_nodes_forest.push_back(_super_tree(i,j));
                }
            }
            return super_tree_nodes_forest;
        }
        std::vector<U> get_super_tree_leaves_forest() const
        {
            std::vector<U> super_tree_leaves_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    super_tree_leaves_forest.push_back(_super_tree(i,j));
                }
            }
            return super_tree_leaves_forest;
        }
        //get super_tree_probabilities_forest,nodes,leaves
        std::vector<std::vector<real_t>> get_super_tree_probabilities_forest() const
        {
            std::vector<std::vector<real_t>> super_tree_probabilities_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<real_t> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_probabilities(i,j));
                }
                super_tree_probabilities_forest.push_back(tree);
            }
            return super_tree_probabilities_forest;
        }
        //get super_tree_values_forest,nodes,leaves
        std::vector<std::vector<T>> get_super_tree_values_forest() const
        {
            std::vector<std::vector<T>> super_tree_values_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<T> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_values(i,j));
                }
                super_tree_values_forest.push_back(tree);
            }
            return super_tree_values_forest;
        }
        //get super_tree_values_projection_forest,nodes,leaves
        std::vector<std::vector<T>> get_super_tree_values_projection_forest() const
        {
            std::vector<std::vector<T>> super_tree_values_projection_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<T> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_values_projection(i,j));
                }
                super_tree_values_projection_forest.push_back(tree);
            }
            return super_tree_values_projection_forest;
        }
        //get super_tree_hplane_forest,nodes,leaves
        std::vector<std::vector<provallo::hplane>> get_super_tree_hplane_forest() const
        {
            std::vector<std::vector<provallo::hplane>> super_tree_hplane_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<provallo::hplane> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_hplane(i,j));
                }
                super_tree_hplane_forest.push_back(tree);
            }
            return super_tree_hplane_forest;
        }
        //get super_tree_forest,nodes,leaves
        std::vector<std::vector<U>> get_super_tree_forest(const size_t & i) const
        {
            std::vector<std::vector<U>> super_tree_forest;
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                std::vector<U> tree;
                tree.push_back(_super_tree(i,j));
                super_tree_forest.push_back(tree);
            }
            return super_tree_forest;
        }
        std::vector<U> get_super_tree_nodes_forest(const size_t & i) const
        {
            std::vector<U> super_tree_nodes_forest;
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                super_tree_nodes_forest.push_back(_super_tree(i,j));
            }
            return super_tree_nodes_forest;
        }   
        std::vector<U> get_super_tree_leaves_forest(const size_t & i) const
        {
            std::vector<U> super_tree_leaves_forest;
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                super_tree_leaves_forest.push_back(_super_tree(i,j));
            }
            return super_tree_leaves_forest;
        }   
        //test hyperplane value projection quality
        //return the average error
        inline real_t test_projection_quality()
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            if(this->_forest.size()>0)
            {
                size_t tree_index=0;
                
                for(auto& tree : this->_forest)
                {
                    size_t node_index=0;
                    for(auto& node : tree)
                    {
                        
                        //get the hplane of the super_tree
                        hplane& hplane=this->_super_tree_hplane(tree_index,node_index);


                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree_index,node_index);  
                        //get the hyperplane projection of the super_tree value
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(this->gen) *projected_value; 
                        real_t hyperplane_projected_intersection = hplane.distribution(this->gen) *hyperplane_projected_value ;
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection;

                        this->eta+=error;
                        //update the error
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_min= this->_super_tree_values.col_min(node) -error/2.0;   
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_max= _super_tree_values.col_max(node) +error/2.0;
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_range= _super_tree_values.col_max(node)-_super_tree_values.col_min(node); 
                        //update the probabilities
                        this->_super_tree_probabilities(tree_index,node_index)=projected_value/this->_super_tree_values(tree_index,node_index);
                        //update the hplane values according to the probabilities
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value=projected_value;
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_left=projected_value;
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_right=projected_value;
                        //update the super_tree projections according to the hplane values
                        this->_super_tree_values_projection(tree_index,node_index)=projected_value;
                        node_index++;
                    }
                    tree_index++;

                }
            }
            else
            {
                for(size_t tree=0;tree<this->_super_tree_values.rows();tree++)
                {
                    for(size_t node=0;node<this->_super_tree_values.cols();node++)
                    {
                        size_t tree_index=tree;
                        size_t node_index=node;

                        //get the hplane of the super_tree
                        hplane& hplane=this->_super_tree_hplane(tree,node);

                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree,node);
                        //get the hyperplane projection of the super_tree value
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(gen) *projected_value;

                        real_t hyperplane_projected_intersection = hplane.distribution(gen) *hyperplane_projected_value; 
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection;
                        //update the error
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_min=projection_intersection -error/2.0; 
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_max=hyperplane_projected_intersection +error/2.0; 

                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection -error/2.0; 
                        //update the probabilities
                        this->_super_tree_probabilities(tree,node)=projected_value/this->_super_tree_values(tree,node) -error/2.0; 
                        //update the hplane values according to the probabilities
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value=projected_value;
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_left=projected_value;
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_right=projected_value;
                        //update the super_tree projections according to the hplane values
                        this->_super_tree_values_projection(tree,node)=projected_value;

                     }

                }

            }
            //calculate the average error
            real_t average_error=0.0;
            if(this->_forest.size()>0)
            {   size_t tree_index=0;

                for(auto& tree : this->_forest)
                {
                    size_t node_index=0;
                    for(auto& node : tree)
                    {
                        //get the hplane of the super_tree
                        
                        hplane& hplane=this->_super_tree_hplane(tree_index,node_index); 
                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree_index,node_index); 
                        //get the hyperplane projection of the super_tree value 
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(gen) *projected_value;

                        real_t hyperplane_projected_intersection = 1 - (1-hplane.distribution(gen)*hyperplane_projected_value); 
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection; 
                        //update the error
                        hplane.hplane_feature_value_min=projection_intersection -error/2.0; 
                        hplane.hplane_feature_value_max=hyperplane_projected_intersection +error/2.0;
                        hplane.hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection -error/2.0;
                        //update the probabilities (score and weight )
                        hplane.weight=projected_value/this->_super_tree_values(tree_index,node_index) -error/2.0; 
                        hplane.score=projected_value/this->_super_tree_values(tree_index,node_index) -error/2.0; 

                        //hplane.probability=projected_value/this->_super_tree_values(tree_index,node_index) -error/2.0; 
                        //update the hplane values according to the probabilities
                        hplane.hplane_feature_value=T(node);
                        hplane.hplane_feature_value_left=projected_value; 
                        hplane.hplane_feature_value_right=projected_value;
                        
                        //update the average error
                        average_error+=error / this->_super_tree_values.rows() / this->_super_tree_values.cols() / this->_forest.size(); 

                        node_index++;
                    }
                    tree_index++;
                }
            }
            else
            {
                for(size_t tree=0;tree<this->_super_tree_values.rows();tree++)
                {
                    for(size_t node=0;node<this->_super_tree_values.cols();node++)
                    {
                        size_t tree_index=tree; 
                        size_t node_index=node;
                        //get the hplane of the super_tree
                        hplane& hplane=this->_super_tree_hplane(tree_index,node_index);
                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree,node);
                        //get the hyperplane projection of the super_tree value
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(gen) *projected_value; 

                        real_t hyperplane_projected_intersection = hplane.distribution(gen) *hyperplane_projected_value; 
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection;
                        //update the error
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_min=projection_intersection;
                        this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_max=hyperplane_projected_intersection;
                        //update the average error
                        average_error+=error;
                        //update the probabilities
                        this->_super_tree_probabilities(tree,node)=projected_value/this->_super_tree_values(tree,node); 
                        //done.
                    }
                }
            }
            //calculate the average error
            average_error/=(this->_super_tree_values.rows()*this->_super_tree_values.cols());
            return average_error;
        }//test_projection_quality
        //test hyperplane value projection quality
        //return the average error
        inline real_t test_projection_quality(const size_t & tree,const size_t & node)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            //get the hplane of the super_tree
            size_t tree_index=tree;
            size_t node_index=node;

            hplane& hplane=this->_super_tree_hplane(tree_index,node_index);
            //get the projection of the super_tree value
            real_t projected_value = this->_super_tree_values_projection(tree,node);
            //get the hyperplane projection of the super_tree value
            real_t hyperplane_projected_value= hplane.hplane_feature_value;
            //get the intersection of the projection and the hyperplane projection
            real_t projection_intersection = hplane.distribution(projected_value);

            real_t hyperplane_projected_intersection = hplane.distribution(hyperplane_projected_value);
            //get the error
            real_t error = hyperplane_projected_intersection-projection_intersection;
            //update the error
            this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_min=projection_intersection;
            this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_max=hyperplane_projected_intersection;
            //update the probabilities
            this->_super_tree_probabilities(tree,node)=projected_value/this->_super_tree_values(tree,node);
            //update the hplane values according to the probabilities
            this->_super_tree_hplane(tree_index,node_index).hplane_feature_value=projected_value;
            this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_left=projected_value;
            this->_super_tree_hplane(tree_index,node_index).hplane_feature_value_right=projected_value;
            //update the super_tree projections according to the hplane values
            this->_super_tree_values_projection(tree,node)=projected_value;
            //calculate the average error
            return error;
        }//test_projection_quality

      void print(std::ostream& out)
      {
            //print super tree:
            out<<"super_tree:"<<std::endl;

            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    out<<std::to_string(i)<<","<<std::to_string(j)<<":"<<_super_tree(i,j).first << ","<<_super_tree(i,j).second  <<" ";  
                }
                out<<std::endl;
            }
            //print super tree values:
            out<<"super_tree_values:"<<std::endl;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    out<<_super_tree_values(i,j)<<" ";
                }
                out<<std::endl;
            }
            //print super tree values projection:
            out<<"super_tree_values_projection:"<<std::endl;
            for(size_t i=0;i<_super_tree_values_projection.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values_projection.cols();j++)
                {
                    out<<_super_tree_values_projection(i,j)<<" ";
                }
                out<<std::endl;
            }   
            //print super tree probabilities:
            out<<"super_tree_probabilities:"<<std::endl;
            for(size_t i=0;i<_super_tree_probabilities.rows();i++)
            {
                for(size_t j=0;j<_super_tree_probabilities.cols();j++)
                {
                    out<<_super_tree_probabilities(i,j)<<" ";
                }
                out<<std::endl;
            }
            //print super tree hplane:
            out<<"super_tree_hplane:"<<std::endl;
            for(size_t i=0;i<_super_tree_hplane.rows();i++)
            {
                for(size_t j=0;j<_super_tree_hplane.cols();j++)
                {
                    out<<_super_tree_hplane(i,j)<<" ";
                }
                out<<std::endl;
            }   
            //print forest:
            out<<"forest:"<<std::endl;
            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    out<<_forest[i][j]<<" ";
                }
                out<<std::endl;
            }   
            //print forest values:
                        
      }
      //algorithm descibed in .tex file :
        //fit the model

       void fit(const matrix<T>& X,std::vector<U> y)
        {
            //initialize the super_tree values 
            initialize_values(X,y); 
            //initialize the super_tree hplane values 
            initialize_hplanes(); 
            //initialize the super_tree probabilities 
            initialize_probabilities(); 
            //initialize the forest
            initialize_forest();
            //initialize the forest values
            initialize_forest_values(); 
            //check the prediction quality

            this->eta   = test_projection_quality(); 
            //update the super_tree values]
            //get y_pred from prediction plane :
            //update momentum:
            std::vector<U> y_pred(y.size(),T(0));
            this->momentum = this->eta * this->momentum + (1.0 - this->eta) * std::accumulate(y_pred.begin(), y_pred.end(),0) / y_pred.size() ;

            for(size_t i=0;i<y.size();i++)
            {
                //use the hplane distribution to get the hyperplane value 
                real_t hyperplane_value = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) * _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value;
                //use the hplane distribution to get the hyperplane value 
                real_t hyperplane_projected_value = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) * hyperplane_value; 
                //use the hplane distribution to get the hyperplane value
                real_t hyperplane_projected_value_left = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) * hyperplane_value;
                //use the hplane distribution to get the hyperplane value 
                real_t hyperplane_projected_value_right = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) * hyperplane_value; 
                //update the hplane values according to the probabilities 
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value=hyperplane_value; 
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_left=hyperplane_projected_value_left; 
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_right=hyperplane_projected_value_right; 
                //update the super_tree projections according to the hplane values 
                _super_tree_values_projection(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=hyperplane_projected_value; 
                //update the super_tree values according to the hplane values 
                _super_tree_values(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=hyperplane_value; 
                //update the super_tree probabilities according to the hplane values 
                _super_tree_probabilities(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=hyperplane_projected_value/hyperplane_value; 
                 //update the indices according to the hplane values 
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_indices.i()=i%_super_tree_hplane.rows(); 
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_indices.j()=i%_super_tree_hplane.cols(); 
                //update the momentum
                this->momentum = this->eta * this->momentum + (1.0 - this->eta) * std::accumulate(y_pred.begin(), y_pred.end(),0) / y_pred.size() ; 
                
                //add projection to y_pred and map to expected min-max labels
                y_pred[i]= hyperplane_projected_value; 
                //apply log(1+exp(x)) to y_pred to map to expected min-max labels 
                y_pred[i]=std::log(1+std::exp(y_pred[i])); 


            } 
            //if eta==0.0 then the super_tree values are not updated 
            if(this->eta>0.0)
            {
                //update the super_tree values
                update_values(X,y_pred); 
                //update the super_tree hplane values
                update_hplanes(); 
                //update the super_tree probabilities
                update_probabilities(); 
                //update the forest
                update_forest();
                //update the forest values
                update_forest_values(); 
            } 
            //check the prediction quality
            this->eta   = test_projection_quality();
                        
        }   
        //fit with double<double<real_t>> X  and extract y from X[len-1]
        void fit(const std::vector<std::vector<real_t>>& data)
        {
            std::vector<U> labels;
            matrix<T> X(data.size(),data[0].size()-1);
            for(size_t i=0;i<data.size();i++)
            {
                for(size_t j=0;j<data[0].size()-1;j++)
                {
                    X(i,j)=data[i][j];
                }
                labels.push_back(data[i][data[0].size()-1]);
            }   
            fit(X,labels);
            
        }
      
      //predict without yy: 
        std::vector<U> predict ( const matrix<T>& XX) 
        {
            //update the projection plane over the path lens of the new data: 
            //predict Y for each row of XX:
            auto predict_y  = this->_super_tree_values_projection * XX.transpose(); 

            //update the projection plane over the path lens of the new data: 
            //convert predict_y to std::vector<U> : 
            std::vector<U> predict_y_vec(predict_y.data(),predict_y.data()+predict_y.rows()*predict_y.cols()); 
           
            //get the prediction plane over the path lens of the new data:
            
            std::vector<U> y_pred(XX.rows(),U(0)); 
            for(size_t i=0;i<XX.rows();i++)
            {
                real_t hyperplane_value = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) * _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value; 
                
                //calculate the intersection of the projection and the hyperplane projection 
                real_t projection_intersection = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) *predict_y_vec[i]; 
                real_t hyperplane_projected_intersection = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) *hyperplane_value; 
                //calculate the error
                real_t error = hyperplane_projected_intersection-projection_intersection; 
                //update the error
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_min=projection_intersection -error/2.0; 
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_max=hyperplane_projected_intersection +error/2.0; 
                //update the probabilities
                _super_tree_probabilities(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y_vec[i]/_super_tree_values(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()); 
                //update the hplane values according to the probabilities
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value=predict_y_vec[i];
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_left=predict_y_vec[i];
                _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_right=predict_y_vec[i];
                //update the super_tree projections according to the hplane values
                _super_tree_values_projection(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y_vec[i];
                if ( error > 0.0 ) 
                {
                    //update the super_tree values according to the hplane values
                    _super_tree_values(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=hyperplane_value;
                    //update the super_tree hplane values
                    _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value=hyperplane_value;
                    _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_left=hyperplane_value;
                    _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_right=hyperplane_value;
                    _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_min=projection_intersection -error/2.0; 
                    _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_max=hyperplane_projected_intersection +error/2.0; 
                    _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection; 
                    y_pred[i]=  predict_y_vec[i]; 
                //apply log(1+exp(x)) to y_pred to map to expected min-max labels 

                }   
                else
                {
                    y_pred[i]=  hyperplane_projected_intersection; 
                    y_pred[i]=std::log(1+std::exp(y_pred[i])); 

                    
                }
            }      
            return y_pred;
    
        } 
        //predict with yy:
      
      std::vector<U> predict ( const matrix<T>& XX, std::vector<U> yy) 
      {
        //update the projection plane over the path lens of the new data: 
        //predict Y for each row of XX:
        auto predict_y  = this->_super_tree_values_projection * XX.transpose(); 

        //update the projection plane over the path lens of the new data: 
        //convert predict_y to std::vector<U> : 
        std::vector<U> predict_y_vec(predict_y.data(),predict_y.data()+predict_y.rows()*predict_y.cols());  
        //get the prediction plane over the path lens of the new data:
        std::vector<U> y_pred(XX.rows(),U(0));
        for(size_t i=0;i<XX.rows();i++)
        {
            real_t hyperplane_value = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) * _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value; 
            
            //calculate the intersection of the projection and the hyperplane projection 
            real_t projection_intersection = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) *predict_y_vec[i]; 
            real_t hyperplane_projected_intersection = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) *hyperplane_value; 
            //calculate the error
            real_t error = hyperplane_projected_intersection-projection_intersection; 
            //update the error
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_min=projection_intersection -error/2.0; 
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_max=hyperplane_projected_intersection +error/2.0; 
            //update the probabilities
            _super_tree_probabilities(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y_vec[i]/_super_tree_values(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()); 
            //update the hplane values according to the probabilities
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value=predict_y_vec[i];
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_left=predict_y_vec[i];
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_right=predict_y_vec[i];
            //update the super_tree projections according to the hplane values
            _super_tree_values_projection(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()); 
            _super_tree_probabilities(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()); 
             y_pred[i]= hyperplane_projected_intersection;


        }
        //calculate the score between y_pred and yy 
        real_t score=0.0;
        for(size_t i=0;i<yy.size();i++)
        {
            score+=1-std::log(yy[i]-y_pred[i]); 


        }
        score/=yy.size();
        

        //update the momentum
        this->momentum = this->eta * this->momentum + (1.0 - this->eta) * std::accumulate(y_pred.begin(), y_pred.end(),0) / y_pred.size() ;
        //update eta
        this->eta   = test_projection_quality(); 

        std::cout<<"eta:"<<std::to_string(this->eta)<<std::endl; 
        std::cout<<"score:"<<std::to_string(score)<<std::endl;

        return y_pred;
        
      } 
      std::vector<real_t> get_anomaly_score(const provallo::matrix<T>& data) 
      {
        //try to predict data :
        std::vector<uint32_t> y_pred = predict(data); 
        //calculate the anomaly score 
        std::vector<real_t> anomaly_score(data.rows(),real_t(0)); 
        for(size_t i=0;i<data.rows();i++)
        {
            anomaly_score[i]=std::abs(y_pred[i]-data(i,0)); 
        }
        return anomaly_score;

      }

     
      std::vector<real_t> predict ( const matrix<T>& XX, std::vector<real_t> yy) 
      {
        //update the projection plane over the path lens of the new data: 
        //predict Y for each row of XX:
        auto predict_y  = this->_super_tree_values_projection * XX.transpose(); 

        //update the projection plane over the path lens of the new data: 
        //convert predict_y to std::vector<U> : 
        std::vector<U> predict_y_vec(predict_y.data(),predict_y.data()+predict_y.rows()*predict_y.cols());  
        //get the prediction plane over the path lens of the new data:
        std::vector<U> y_pred(XX.rows(),U(0));
        std::vector<real_t> y_pred_real_t(XX.rows(),real_t(0));
        for(size_t i=0;i<XX.rows();i++)
        {
            real_t hyperplane_value = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) * _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value; 
            
            //calculate the intersection of the projection and the hyperplane projection 
            real_t projection_intersection = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) *predict_y_vec[i]; 
            real_t hyperplane_projected_intersection = _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).distribution(gen) *hyperplane_value; 
            //calculate the error
            real_t error = hyperplane_projected_intersection-projection_intersection; 
            //update the error
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_min=projection_intersection -error/2.0; 
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_max=hyperplane_projected_intersection +error/2.0; 
            //update the probabilities
            _super_tree_probabilities(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y_vec[i]/_super_tree_values(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()); 
            //update the hplane values according to the probabilities
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value=predict_y_vec[i];
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_left=predict_y_vec[i]; 
            _super_tree_hplane(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols()).hplane_feature_value_right=predict_y_vec[i];
            //update the super_tree projections according to the hplane values
            _super_tree_values_projection(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols());
            _super_tree_probabilities(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols())=predict_y(i%_super_tree_hplane.rows(),i%_super_tree_hplane.cols());
            y_pred[i]= hyperplane_projected_intersection;

            

        }
        //calculate the score between y_pred and yy
        real_t score=0.0;
        for(size_t i=0;i<yy.size();i++)
        {
            score+=1-std::log(yy[i]-y_pred[i]);
        }
        score/=yy.size();
        //update the momentum
        this->momentum = this->eta * this->momentum + (1.0 - this->eta) * std::accumulate(y_pred.begin(), y_pred.end(),0) / y_pred.size() ;
        //update eta
        this->eta   = test_projection_quality();
        std::cout<<"eta:"<<std::to_string(this->eta)<<std::endl;
        std::cout<<"score:"<<std::to_string(score)<<std::endl;
        for ( auto & y : y_pred)
        {
            y_pred_real_t.push_back(real_t(y));
        } 
        return y_pred_real_t;
      }
      void update_values (const matrix<T>& X, std::vector<U> y) 
      {
        
        //update value leafs from X and y 
        for(size_t i=0;i<y.size();i++)
        {
            this->_super_tree_values(i,0)=y[i];
        }
        for (size_t i = 0; i < X.rows(); i++) 
        {
            for (size_t j = 0; j < X.cols(); j++) 
            {
                this->_super_tree_values(i,0)+=X(i,j)/X.cols(); 

            }
        }
        //update hyperplane values
        process_hplanes();


      } 

      //get path length from the hyperplane values :
        inline real_t get_path_length(const size_t & tree,const size_t & node) 
        {
            //get the hplane of the super_tree
            hplane& hplane=this->_super_tree_hplane(tree,node);
            //get the projection of the super_tree value
            real_t projected_value = this->_super_tree_values_projection(tree,node);
            //get the hyperplane projection of the super_tree value
            real_t hyperplane_projected_value= hplane.hplane_feature_value;
            //get the intersection of the projection and the hyperplane projection
            real_t projection_intersection = hplane.distribution(gen) *projected_value;

            real_t hyperplane_projected_intersection = hplane.distribution(gen) *hyperplane_projected_value; 
            //get the error
            real_t error = hyperplane_projected_intersection-projection_intersection;
            //update the error
            this->_super_tree_hplane(tree,node).hplane_feature_value_min=projection_intersection -error/2.0; 
            this->_super_tree_hplane(tree,node).hplane_feature_value_max=hyperplane_projected_intersection +error/2.0; 
            //update the probabilities
            this->_super_tree_probabilities(tree,node)=projected_value/this->_super_tree_values(tree,node);
            //update the hplane values according to the probabilities
            this->_super_tree_hplane(tree,node).hplane_feature_value=projected_value;
            this->_super_tree_hplane(tree,node).hplane_feature_value_left=projected_value;
            this->_super_tree_hplane(tree,node).hplane_feature_value_right=projected_value;
            //update the super_tree projections according to the hplane values
            this->_super_tree_values_projection(tree,node)=projected_value;
            //calculate the path length
            return hyperplane_projected_intersection-projection_intersection;
        }   
        //get path length from input data :
        std::vector<std::vector<real_t>> get_path_length(const matrix<T>& X) 
        {
            //get the path length for each row of X
            std::vector<std::vector<real_t>> path_length(X.rows(),std::vector<real_t>(X.cols(),real_t(0))); 
            for(size_t i=0;i<X.rows();i++)
            {
                for(size_t j=0;j<X.cols();j++)
                {
                    path_length[i][j]=get_path_length(i,j); 
                }
            }
            return path_length;
        }       

        //get anomaly score and path length from data: 
        std::vector<std::vector<real_t>> get_anomaly_score_path_length(const matrix<T>& X) 
        {
            //get the anomaly score and path length for each row of X
            std::vector<std::vector<real_t>> anomaly_score_path_length(X.rows(),std::vector<real_t>(X.cols(),real_t(0))); 
            for(size_t i=0;i<X.rows();i++)
            {
                for(size_t j=0;j<X.cols();j++)
                {
                    //get the anomaly score for each row of X
                    std::vector<U> y_pred = predict(X.row(i)); 
                    //calculate the anomaly score for each row of X
                    for(size_t k=0;k<y_pred.size();k++)
                    {
                        anomaly_score_path_length[i][j]+=std::log(1+std::exp(y_pred[k])); 
                    }
                    anomaly_score_path_length[i][j]/=y_pred.size(); 
                }
            }
            return anomaly_score_path_length;
        }   

        /// get_anomaly_score_and_path_length 
 
        std::vector<std::vector<real_t>> get_anomaly_score_and_path_length(const std::vector<std::vector<T>>& X) 
        {
            //create a matrix from the input data
            matrix<T> data(X.size(),X[0].size()); 
            for(size_t i=0;i<X.size();i++)
            {
                for(size_t j=0;j<X[i].size();j++)
                {
                    data(i,j)=X[i][j]; 
                }
            }
            return get_anomaly_score_and_path_length(data); 

        }   
        std::vector<std::vector<real_t>> get_anomaly_score_and_path_length(const matrix<T>& X) 
        {
            //get the anomaly score and path length for each row of X
            std::vector<std::vector<real_t>> anomaly_score_and_path_length(X.rows(),std::vector<real_t>(X.cols(),real_t(0))); 
            //get the convergence score for each row of X 
            real_t path_len = 0.0;
            real_t path_len_sum = 1.0;
            auto results = predict(X); 
            size_t result_index = 0;
            //get the anomaly score and path length for each row of X 
            for ( auto res : results )
            {
                //get the path len from the hyperplane values corresponding to the input data pointed by the result:

                auto value = this->get_super_tree_hplane(result_index%_super_tree_hplane.rows()).distribution(gen) * this->get_super_tree_hplane(result_index%_super_tree_hplane.rows()).hplane_feature_value; 
                auto projected_value = this->get_super_tree_hplane(result_index%_super_tree_hplane.rows()).distribution(gen) * value; 
                auto hyperplane_projected_value = this->get_super_tree_hplane(result_index%_super_tree_hplane.rows()).distribution(gen) * value; 
                auto hyperplane_projected_value_left = this->get_super_tree_hplane(result_index%_super_tree_hplane.rows()).distribution(gen) * value;
                auto hyperplane_projected_value_right = this->get_super_tree_hplane(result_index%_super_tree_hplane.rows()).distribution(gen) * value;
                //add path length to the path_len_sum 
                path_len_sum += hyperplane_projected_value - projected_value; 
                //add path length to the path_len
                path_len += hyperplane_projected_value - projected_value; 
                //add path length to the anomaly_score_and_path_length
                anomaly_score_and_path_length[result_index%_super_tree_hplane.rows()][result_index%_super_tree_hplane.cols()] = path_len; 
                //update the result_index
                
                result_index++;
            }
            return anomaly_score_and_path_length;
        }
        void update_hplanes () 
        {
            //update the super_tree hplane values
            for(size_t i=0;i<this->_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<this->_super_tree_values.cols();j++)
                {
                    //get the hplane of the super_tree
                    hplane& hplane=this->_super_tree_hplane(i,j);
                    //get the projection of the super_tree value
                    real_t projected_value = this->_super_tree_values_projection(i,j);
                    //get the hyperplane projection of the super_tree value
                    real_t hyperplane_projected_value= hplane.hplane_feature_value;
                    //get the intersection of the projection and the hyperplane projection
                    real_t projection_intersection = hplane.distribution(gen) *projected_value;
    
                    real_t hyperplane_projected_intersection = hplane.distribution( gen) *hyperplane_projected_value; 
                    //get the error
                    real_t error = hyperplane_projected_intersection-projection_intersection;
                    //update the error
                    this->_super_tree_hplane(i,j).hplane_feature_value_min=projection_intersection -error/2.0; 
                    this->_super_tree_hplane(i,j).hplane_feature_value_max=hyperplane_projected_intersection +error/2.0; 
                    //update the probabilities
                    this->_super_tree_probabilities(i,j)=projected_value/this->_super_tree_values(i,j);
                    //update the hplane values according to the probabilities
                    this->_super_tree_hplane(i,j).hplane_feature_value=projected_value;
                    this->_super_tree_hplane(i,j).hplane_feature_value_left=projected_value;
                    this->_super_tree_hplane(i,j).hplane_feature_value_right=projected_value;
                    //update the super_tree projections according to the hplane values
                    this->_super_tree_values_projection(i,j)=projected_value;
                }
            }
        }   
        void update_probabilities () 
        {
            //update the super_tree probabilities
            for(size_t i=0;i<this->_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<this->_super_tree_values.cols();j++)
                {
                    this->_super_tree_probabilities(i,j)=this->_super_tree_values_projection(i,j)/this->_super_tree_values(i,j);
                }
            }
        }   
        void update_forest () 
        {
            //update the forest
            for(size_t i=0;i<this->_forest.size();i++)
            {
                for(size_t j=0;j<this->_forest[i].size();j++)
                {
                    this->_forest[i][j]=this->_super_tree_probabilities(i,j);
                }
            }
        }   
        void update_forest_values () 
        {
            //update the forest values
            for(size_t i=0;i<this->_forest.size();i++)
            {
                for(size_t j=0;j<this->_forest[i].size();j++)
                {
                    this->_forest[i][j]=this->_super_tree_values(i,j);
                }
            }
        }   
        void initialize_values (const matrix<T>& X, std::vector<U> y) 
        {
            //initialize the super_tree values 
            for(size_t i=0;i<y.size();i++)
            {
                this->_super_tree_values(i,0)=y[i];
            }
            for (size_t i = 0; i < X.rows(); i++) 
            {
                for (size_t j = 0; j < X.cols(); j++) 
                {
                    this->_super_tree_values(i,0)+=X(i,j)/X.cols(); 

                }
            }   
            //update hyperplane values
            process_hplanes();

        }   
         
        void initialize_forest () 
        {
            //initialize the forest
            for(size_t i=0;i<this->_forest.size();i++)
            {
                for(size_t j=0;j<this->_forest[i].size();j++)
                {
                    this->_forest[i][j]=this->_super_tree_probabilities(i,j);
                }
            }
        }   
        void initialize_forest_values () 
        {
            //initialize the forest values
            
            for(size_t i=0;i<this->_forest.size();i++)
            {
                for(size_t j=0;j<this->_forest[i].size();j++)
                {
                    this->_forest[i][j]=this->_super_tree_values(i,j);
                }
            }
        }   
        void initialize_forest_hplanes () 
        {
            //if we were intialized from a forest do this: 

            //initialize the forest hplane values 
            if( _forest.size()>0&&_forest[0].size()>1)
            {
            for(size_t i=0;i<this->_forest.size();i++)
            {
                for(size_t j=0;j<this->_forest[i].size();j++)
                {
                    //get the hplane of the super_tree
                    hplane& hplane=this->_super_tree_hplane(i,j);
                    //get the projection of the super_tree value
                    real_t projected_value = this->_super_tree_values_projection(i,j);
                    //get the hyperplane projection of the super_tree value
                    real_t hyperplane_projected_value= hplane.hplane_feature_value;
                    //get the intersection of the projection and the hyperplane projection
                    real_t projection_intersection = hplane.distribution(projected_value);

                    real_t hyperplane_projected_intersection = hplane.distribution(hyperplane_projected_value);
                    //get the error
                    real_t error = hyperplane_projected_intersection-projection_intersection;
                    //update the error
                    this->_super_tree_hplane(i,j).hplane_feature_value_min=projection_intersection;
                    this->_super_tree_hplane(i,j).hplane_feature_value_max=hyperplane_projected_intersection;
                    //update the probabilities
                    this->_super_tree_probabilities(i,j)=projected_value/this->_super_tree_values(i,j);
                    //update the hplane values according to the probabilities
                    this->_super_tree_hplane(i,j).hplane_feature_value=projected_value;
                    this->_super_tree_hplane(i,j).hplane_feature_value_left=projected_value;
                    this->_super_tree_hplane(i,j).hplane_feature_value_right=projected_value;
                    //update the super_tree projections according to the hplane values
                    this->_super_tree_values_projection(i,j)=projected_value;
                    //this will constraint and bound the decision boundaries of the 'forest'
                }
            }
            }
            else
            {
                //initialize the super trees,hplane,probabilities,values,values_projection 
                initialize_hplanes();
            }
        }

      std::vector<real_t> score (matrix<T> X, std::vector<U> y) 
      {
         //compare the prediction plane over the path lens of the new data with the actual values: 
            std::vector<U> y_pred=predict(X,y); 
            std::vector<T> y_score(y.size(),T(0)); 
            //score is 1/n sum (y_pred-y)log(y_pred/y) + sum(y_pred-y) 
            const real_t eps = 1e-15;
            for(size_t i=0;i<y.size();i++)
            {
                y_score[i] = (y_pred[i]-y[i])*std::log(y_pred[i]/(y[i]+eps)) + (y_pred[i]-y[i]); 
                if(y_score[i]!=y_score[i])
                {
                    //nan is 0, inf is 1 ,-inf is -1 ..

                    y_score[i]=0.0;
                }

                y_score[i]/=real_t(y.size()); 
              
            }
            
            return y_score;
      } 
      void print()
      {
        //print super tree:
        std::cout<<"super_tree:"<<std::endl; 
        for(size_t i=0;i<_super_tree_values.rows();i++)
        {
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                std::cout<<std::to_string(i)<<","<<std::to_string(j)<<":"<<_super_tree(i,j).first << ","<<_super_tree(i,j).second  <<" ";  
            }
            std::cout<<std::endl;
        }
        //print super tree values:
        std::cout<<"super_tree_values:"<<std::endl;
        for(size_t i=0;i<_super_tree_values.rows();i++)
        {
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                std::cout<<_super_tree_values(i,j)<<" ";
            }
            std::cout<<std::endl;
        }
        //print super tree values projection:
        std::cout<<"super_tree_values_projection:"<<std::endl;
        for(size_t i=0;i<_super_tree_values_projection.rows();i++)
        {
            for(size_t j=0;j<_super_tree_values_projection.cols();j++)
            {
                std::cout<<_super_tree_values_projection(i,j)<<" ";
            }
            std::cout<<std::endl;
        }
        //print super tree probabilities:
        std::cout<<"super_tree_probabilities:"<<std::endl;
        for(size_t i=0;i<_super_tree_probabilities.rows();i++)
        {
            for(size_t j=0;j<_super_tree_probabilities.cols();j++)
            {
                std::cout<<_super_tree_probabilities(i,j)<<" ";
            }
            std::cout<<std::endl;
        }
        //print eta and momentum:
        std::cout<<"eta:"<<std::to_string(eta)<<std::endl;
        std::cout<<"momentum:"<<std::to_string(momentum)<<std::endl; 
        //print super tree hplane:
        std::cout<<"super_tree_hplane:"<<std::endl; 
        for(size_t i=0;i<_super_tree_hplane.rows();i++)
        {
            for(size_t j=0;j<_super_tree_hplane.cols();j++)
            {
                std::cout<<_super_tree_hplane(i,j)<<" ";
            }
            std::cout<<std::endl;
        }
      }
      //get stability of the model 
        real_t stability()
        {
            auto values_stability = _super_tree_values.stable().size(); 
            auto values_projection_stability = _super_tree_values_projection.stable().size();
            auto probabilities_stability = _super_tree_probabilities.stable().size();
            
            auto hplane_stability = 0;// _super_tree_hplane.stable([](hplane& hplane){return hplane.hplane_feature_value;}   ).size();


            return (values_stability+values_projection_stability+probabilities_stability+hplane_stability)/4.0; 
            
        }   
        real_t divergent_stability()
        {
            auto values_stability = _super_tree_values.divergent();
            auto values_projection_stability = _super_tree_values_projection.divergent(); 
            auto probabilities_stability = _super_tree_probabilities.divergent(); 
            //auto hplane_stability = 0;// _super_tree_hplane.divergent([](hplane& hplane){return hplane.hplane_feature_value;}   ); 
            return (values_stability.size()+values_projection_stability.size()+probabilities_stability.size())/3.0;         

        }   

        real_t flutter_stability()
        {
            auto values_stability = _super_tree_values.flutter();
            auto values_projection_stability = _super_tree_values_projection.flutter(); 
            auto probabilities_stability = _super_tree_probabilities.flutter(); 
            //auto hplane_stability = _super_tree_hplane.flutter(); 
            return (values_stability.size()+values_projection_stability.size()+probabilities_stability.size())/3.0;

            //return (values_stability.size()+values_projection_stability.size()+probabilities_stability.size()+hplane_stability.size())/4.0;         
        } 
        real_t unstable_stability()
        {
            auto values_stability = _super_tree_values.stable();
            auto values_projection_stability = _super_tree_values_projection.stable(); 
            auto probabilities_stability = _super_tree_probabilities.stable(); 
            //auto hplane_stability = _super_tree_hplane.stable(); 
            return (values_stability.size()+values_projection_stability.size()+probabilities_stability.size())/3.0; 

             //return (values_stability.size()+values_projection_stability.size()+probabilities_stability.size()+hplane_stability.size())/4.0;     
        }
 
      real_t get_eta() const
      {
          return eta;
      }
      real_t get_momentum() const
      {
            return momentum;
      }
      //set eta and momentum 
        void set_eta(const real_t & eta)
        {
            this->eta=eta;
        }
        void set_momentum(const real_t & momentum)
        {
            this->momentum=momentum;
        }

      //set number of trees:
      void set_num_trees(const size_t & number_of_trees)
      {
          this->number_of_trees=number_of_trees;
      }
      void set_max_depth(const size_t & max_depth)
      {
          this->max_depth=max_depth;
      }
      void set_number_of_nodes(const size_t & number_of_nodes)
      {
            this->number_of_nodes=number_of_nodes;
      }
        void set_number_of_hplanes(const size_t & number_of_hplanes)
        {
                this->number_of_hplanes=number_of_hplanes;
        }
        void set_number_of_features(const size_t & number_of_features)
        {
                this->number_of_features=number_of_features;
        }
        void set_number_of_samples(const size_t & number_of_samples)
        {
                this->number_of_samples=number_of_samples;
        }
        void set_number_of_classes(const size_t & number_of_classes)
        {
                this->number_of_classes=number_of_classes;
        }
        void set_number_of_iterations(const size_t & number_of_iterations)
        {
                this->number_of_iterations=number_of_iterations;
        }
        void set_number_of_super_trees(const size_t & number_of_super_trees)
        {
                this->number_of_super_trees=number_of_super_trees;
        }
        void set_number_of_super_tree_nodes(const size_t & number_of_super_tree_nodes)
        {
                this->number_of_super_tree_nodes=number_of_super_tree_nodes;
        }
        void set_number_of_super_tree_hplanes(const size_t & number_of_super_tree_hplanes)
        {
                this->number_of_super_tree_hplanes=number_of_super_tree_hplanes;
        }
        void set_number_of_super_tree_features(const size_t & number_of_super_tree_features)
        {
                this->number_of_super_tree_features=number_of_super_tree_features;
        }
        void set_number_of_super_tree_samples(const size_t & number_of_super_tree_samples)
        {
                this->number_of_super_tree_samples=number_of_super_tree_samples;
        }
        void set_number_of_super_tree_classes(const size_t & number_of_super_tree_classes)
        {
                this->number_of_super_tree_classes=number_of_super_tree_classes;
        }
        void set_random_seed(const uint64_t & seed)
        {
                this->seed = uint64_t( double(seed) / std::numeric_limits<uint64_t>::max() * (std::numeric_limits<uint32_t>::max() - 1.0) );
        }
        void set_max_samples(const uint64_t & max_samples)
        {
                this->max_samples=max_samples;
        }
      
      real_t eta = 0.00001;
      real_t momentum=0.001;
      std::mt19937 gen;
      size_t number_of_trees=1;
      size_t number_of_nodes=1;
      size_t number_of_hplanes=1;
      size_t number_of_features=1;
      size_t number_of_samples=1;
      size_t number_of_classes=1;
      size_t number_of_iterations=1;
      size_t number_of_super_trees=1;
      size_t number_of_super_tree_nodes=1;
      size_t number_of_super_tree_hplanes=1;
      size_t number_of_super_tree_features=1;
      size_t number_of_super_tree_samples=1;
      size_t number_of_super_tree_classes=1;
      //max values : 
        size_t max_depth=1;
        //max samples:
        uint64_t max_samples=std::numeric_limits<uint64_t>::max();
        //max features:
        uint64_t max_features=std::numeric_limits<uint32_t>::max(); 
        //max classes:
        uint64_t max_classes=std::numeric_limits<uint32_t>::max();
        //max nodes:
        uint64_t max_nodes=std::numeric_limits<uint32_t>::max();
        //max hplanes:
        uint64_t max_hplanes=std::numeric_limits<uint32_t>::max();
        //max super trees:
        uint64_t max_super_trees=std::numeric_limits<uint32_t>::max();
        //max super tree nodes:
        uint64_t max_super_tree_nodes=std::numeric_limits<uint32_t>::max();
        //max super tree hplanes:
        uint64_t max_super_tree_hplanes=std::numeric_limits<uint32_t>::max();


      uint64_t seed=0;
      
 };

 
}//namespace provallo


#endif // _FAST_MATRIX_FOREST_H_