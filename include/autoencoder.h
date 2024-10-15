#ifndef PROVALLO_AUTO_ENCODER_H_
#define PROVALLO_AUTO_ENCODER_H_

//#include "neuralhelper.h"
#include "matrix.h"

#include <string>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <vector>

namespace provallo
{


  //utility class distribution class.
  class class_dist
  {
    // Histogram (class attribute tag is the index of the array)
    std::vector<real_t> _histogram;
    // Total sum of the histogram bins
    real_t _sum;
    // Print distribution
    friend std::ostream&
    operator<< (std::ostream &out, const class_dist &q);

    using attribute = size_t;
    using discrete_value = size_t;

  public:
     // Constructor
    class_dist (size_t nbins = 0, real_t  default_value = 0.0) : _histogram (nbins, default_value), _sum (nbins*default_value) 
    {
        
    }
    //copy constructor
    class_dist (const class_dist& other) :
        _histogram (other._histogram), _sum (other._sum)
    {
    } 
    //move constructor
    class_dist (class_dist&& other) :
        _histogram (std::move(other._histogram)), _sum (std::move(other._sum))
    {
    }
    //copy assignment

    const class_dist& operator = (const class_dist& other)
    {
      this->_histogram=other._histogram;
      this->_sum=other._sum;
      
      return *this;

    }
    const class_dist& operator = (class_dist&& other)
    {
      this->_histogram=std::move(other._histogram);
      this->_sum=std::move(other._sum);
      
      return *this;

    }
    bool
    operator!= (const class_dist &other) const
    {
      if (_sum != other._sum)
	        return true;
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      if (_histogram[i] != other._histogram[i])
	        return true;
      
      return false;
    }
    bool
    operator== (const class_dist &other) const
    {
      if (_sum != other._sum)
	        return false;
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      if (_histogram[i] != other._histogram[i])
	        return false;
      
      return true;
    }
    
    real_t* array() {return _histogram.data();}
    //mode and percentage
    std::pair<attribute,real_t> mode_and_percentage () const;
    std::pair<attribute,real_t> mode_and_percentage (const std::vector<discrete_value>& exclude) const;
    std::pair<attribute,real_t> mode_and_percentage (const std::vector<discrete_value>& exclude, const std::vector<discrete_value>& include) const;
 
    // Get size
    size_t
    size () const
    {
      return _histogram.size ();
    }
    // Accumulate a specific tag
    void
    accum (size_t tag, real_t weight = 1.0)
    {
      if(tag<_histogram.size()  ) {
        _histogram[tag] += weight;
      }
      else {
        //resize and add  
          throw std::runtime_error("class_dist::accum tag out of range"); 
        // assert(0);
      }
        _sum += weight;
     }

    // Accumulate a specific tag
    void accum (const class_dist& other)  { 
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other._histogram[i]; 
      _sum += other._sum;
    }
    // Accumulate a specific tag  
    void accum (class_dist&& other)  {    
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other._histogram[i];
      _sum += other._sum;
    }
    // Accumulate a specific tag
    void    
    accum (const std::vector<real_t> &other)
    { 
       for (size_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other[i];
      _sum += std::accumulate(other.begin(),other.end(),0.0);
    }
      // Set a specific tag
    void
    set (size_t tag, real_t weight)
    {
      _sum -= _histogram[tag];
      _histogram[tag] = weight;
      _sum += weight;
    }
  
    std::vector<real_t>::iterator begin() { return _histogram.begin(); } 
    std::vector<real_t>::iterator end() { return _histogram.end(); } 
    std::vector<real_t>::const_iterator begin() const { return _histogram.begin(); } 
    std::vector<real_t>::const_iterator end() const { return _histogram.end(); } 

    // Get a specific tag
    real_t
    get (size_t tag) const
    {
      return _histogram[tag];
    } 
    // Get a specific tag
    real_t&  
    get (size_t tag) 
    {
      return _histogram[tag];
    }
    
    void add (size_t tag, real_t weight)
    {
      if(tag<_histogram.size()) {
        _histogram[tag] += weight;
        _sum += weight;
      }else
      {
        //resize and add  
        _histogram.resize(tag+1,0.0);
        _histogram[tag] = weight;
        _sum += weight;
       }
      
   }

    // Get sum of the data
    real_t
    sum () const
    {
      return _sum;
    }
    // Get weight of a histogram bins
    real_t
    weight (size_t i) const
    {
      return _histogram[i];
    }
    // Get percentage of a histogram bin
    real_t
    percentage (size_t i) const
    {
      if (_sum != 0.0)
	      return _histogram[i] / _sum;
      return 0.0;
    }

    std::vector<real_t>
    cumulative () const
    {
      std::vector<real_t> values (size (), 0.0);
      for (size_t i = 0; i < size (); ++i)
      {
        auto f= percentage(i);
        // Cumulative probability
        if (i > 0)
          values[i] = f + values[i - 1];
        else
          values[i] = f;
      }
      values[size () - 1] = 1.0;
      return values;
    }
    // Get the probability of a histogram bin
    real_t
    probability (size_t i) const
    {
      if (_sum != 0.0)
        return _histogram[i] / _sum;
      return 0.0;
    }

    // Get the probability of a histogram bin
    real_t
    probability (size_t i, real_t sum) const
    {
      if (sum != 0.0)
        return _histogram[i] / sum;
      return 0.0;
    }
    // Get the probability of a histogram bin
    real_t
    probability (size_t i, real_t sum, real_t weight) const
    {
      if (sum != 0.0)
        return  ( _histogram[i] / sum ) * weight;
      return 0.0;
    } 
    // Get the probability of a histogram bin
    real_t
    probability (size_t i, const std::vector<real_t> &sum) const
    {
      if (sum[i] != 0.0)
        return _histogram[i] / sum[i];
      return 0.0;
    } 
    // Get the probability of a histogram bin
    real_t 
    probability (size_t i, const std::vector<real_t> &sum, real_t weight) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight;
      return 0.0;
    }   


    // Get the probability of a histogram bin 
    real_t
    probability (size_t i, const std::vector<real_t> &sum, const std::vector<real_t> &weight) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight[i];
      return 0.0;   
    }   
    // Get the probability of a histogram bin
    real_t

    probability (size_t i, const std::vector<real_t> &sum, const std::vector<real_t> &weight, real_t weight_sum) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight[i] * weight_sum;
      return 0.0;   
    }   

    void update(size_t tag, real_t weight) {
      _histogram[tag] += weight;
      _sum += weight;
    } 
    void update(size_t tag, real_t weight, real_t old_weight) {
      _histogram[tag] += weight- old_weight;
      _sum += weight - old_weight; 
    } 
 

    void setup(size_t nbins)
    {
      _histogram.clear();
      _histogram.resize(nbins,0.0);
      _sum=0.0; 
    } 
    // Get the mode of the distribution

    attribute
    mode () const;

    // Get the entropy of the distribution
    real_t
    entropy () const;

    // Get the gini index of the distribution
    real_t
    gini () const;

    
    
    virtual
    ~class_dist ()
    {
        //delete _histogram;
        _histogram.clear();
        _sum=0.0;

        
    }
  };

  std::ostream&
  operator<< (std::ostream &out, const class_dist &q);

    //simple shokoder autoencoder
    //
    // see
    //http://www.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
    //http://www.stanford.edu/class/cs294a/sparseAutoencoder_notes.pdf
    //vae : http://arxiv.org/pdf/1312.6114v10.pdf
    //variational autoencoder : http://arxiv.org/pdf/1312.6114v10.pdf
    //http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf
    
    

    template <typename T, typename real_x = real_t>
    class auto_encoder
    {
    protected:
        size_t inputDim;
        size_t hiddenDim;
        size_t outputDim;
        T *input = nullptr;
        T *hidden   = nullptr;
        T *output   = nullptr;
        T *weight1  = nullptr;
        T *weight2      = nullptr;
        T *bias1    =   nullptr;
        T *bias2    =   nullptr;
        T *weight1Grad  =   nullptr;
        T *weight2Grad  =   nullptr;
        T *bias1Grad    =      nullptr;
        T *bias2Grad    =   nullptr;
        T *weight1Momentum = nullptr;
        T *weight2Momentum =    nullptr;
        T *bias1Momentum    =   nullptr;
        T *bias2Momentum    =   nullptr;
        T *weight1Update    =   nullptr;
        T *weight2Update    =   nullptr;

        T *bias1Update  =   nullptr;
        T *bias2Update  =   nullptr;
        T *weight1Decay =   nullptr;
        T *weight2Decay =   nullptr;
        T *bias1Decay   =   nullptr;
        T *bias2Decay   =   nullptr;
        T *weight1Sparsity  =   nullptr;
        T *weight2Sparsity  =   nullptr;
        T *bias1Sparsity    =   nullptr;
        T *bias2Sparsity    =   nullptr;
        T *weight1SparsityHat   =   nullptr;
        T *weight2SparsityHat =  nullptr;
        T *bias1SparsityHat =   nullptr;
        T *bias2SparsityHat=        nullptr;
        T *weight1SparsityGrad  =       nullptr;
        T *weight2SparsityGrad =      nullptr;
        T *bias1SparsityGrad    =      nullptr;
        T *bias2SparsityGrad    =      nullptr;
        T *weight1SparsityGradHat   =   nullptr;
        T *weight2SparsityGradHat   =   nullptr;

        T* weight1Inc   =   nullptr;
        T* weight2Inc   =   nullptr;
        T* weight1GradPrev  =   nullptr;
        T* weight2GradPrev  =   nullptr;

        T* bias1Inc =   nullptr;
        T* bias2Inc =   nullptr;
        T* bias1GradPrev    =   nullptr;
        T* bias2GradPrev    =   nullptr;
        T* weight1Prev  =   nullptr;
        T* weight2Prev =     nullptr;



        real_x learningRate =   real_x(0.01);
        real_x momentum =   real_x(0.9);
        real_x weightDecay  =   real_x(0.0001);
        real_x sparsityParam    =   real_x(0.01);
        real_x beta =   real_x(3);
        real_x sparsityParamHat =   real_x(0.01);
        real_x sparsityPenalty  =   real_x(0.0);
        real_x sparsityGradient =   real_x(0.0);
        real_x sparsityGradientHat  =   real_x(0.0);

        T *bias1SparsityGradHat = nullptr;
        T *bias2SparsityGradHat = nullptr;
        
        T* bias1Prev = nullptr;
        T* bias2Prev =  nullptr;


        //prevprev
        T* weight1GradPrevPrev  =   nullptr;
        T* weight2GradPrevPrev  =   nullptr;
        T* bias1GradPrevPrev    =   nullptr;
        T* bias2GradPrevPrev    =   nullptr;
        real_x ce_loss =   real_x(0.0);
        real_x sparsity_loss =   real_x(0.0);
        real_x weight_decay_loss =   real_x(0.0);
        real_x total_loss =   real_x(0.0);
        real_x sparsity_loss_hat =   real_x(0.0);
        real_x sparsity_loss_grad =   real_x(0.0);
        real_x sparsity_loss_grad_hat =   real_x(0.0);
        real_x sparsity_loss_grad_hat_hat =   real_x(0.0);
        real_x sparsity_loss_grad_hat_hat_hat =   real_x(0.0);


        //override operator new T[] to allocate memory on the heap from memory mapped file 
        //use memory mapped file allocator to allocate memory on the heap 


        void initializeWeight();
        void initializeBias();
        void initializeWeight(T *weight, size_t size);
        void initializeBias(T *bias, size_t size);
        void initializeWeight(T *weight, size_t row, size_t col);
        void conjugateGradient();
        void initialize_autoencoder();
        void backprop();
        virtual void clear();
        virtual void initializeWeightGrad()
        {
            //initialize the gradients 
            initializeWeight1Grad();
            initializeWeight2Grad();
            //initialize bias grads
            initializeBias1Grad();
            initializeBias2Grad();

            //initialize weight prev
            initializeWeight1Prev();
            initializeWeight2Prev();
            //initialize bias prev
            initializeBias1Prev();
            initializeBias2Prev();

            //initialize weight momentum
            initializeWeight1Momentum();

            initializeWeight2Momentum();
            //initialize bias momentum
            initializeBias1Momentum();
            initializeBias2Momentum();


            //initialize weight update
            initializeWeight1Update();
            initializeWeight2Update();

            //initialize bias update
            initializeBias1Update();
            initializeBias2Update();


            //initialize weight decay
            initializeWeight1Decay();
            initializeWeight2Decay();


            //initialize bias decay 
            initializeBias1Decay();
            initializeBias2Decay();


            //initialize weight sparsity

            initializeWeight1Sparsity();
            initializeWeight2Sparsity();


            //initialize bias sparsity
            initializeBias1Sparsity();
            initializeBias2Sparsity();


            //initialize weight sparsity hat
            initializeWeight1SparsityHat();
            initializeWeight2SparsityHat();

            
            //initialize bias sparsity hat
            initializeBias1SparsityHat();
            initializeBias2SparsityHat();


            //initialize weight sparsity grad
            initializeWeight1SparsityGrad();
            initializeWeight2SparsityGrad();


            //initialize bias sparsity grad
            initializeBias1SparsityGrad();
            initializeBias2SparsityGrad();


            //initialize weight sparsity grad hat
            initializeWeight1SparsityGradHat();
            initializeWeight2SparsityGradHat();

            //initialize bias sparsity grad hat

            initializeBias1SparsityGradHat();
            initializeBias2SparsityGradHat();

            //initialize weight grad prev prev

            initializeWeight1GradPrevPrev();
            initializeWeight2GradPrevPrev();

            //initialize bias grad prev prev
            //
            initializeBias1GradPrevPrev();
            initializeBias2GradPrevPrev();

        }
       
        typedef T (auto_encoder<T,real_x>::*XactivationFunctionPtr)(T);

        


         XactivationFunctionPtr activationFunctionPtr; 
         XactivationFunctionPtr activationGradientFunctionPtr;
         XactivationFunctionPtr activationPrimeFunctionPtr;
         XactivationFunctionPtr activationPrimeGradientFunctionPtr;
         XactivationFunctionPtr activationPrimeGradientHatFunctionPtr;

        void initializeActivationFunction();    
        void forward();
        void forward(T*&);
        void forward(const matrix<T>& input);
        
        void backward();
        
        //single case backward
        void backward(T*& input,
                      T*& target);



        void backward(const matrix<T>& input,
                      const matrix<T>& target);

        void backward(const matrix<T>& input,
                      matrix<T>& output,
                      const matrix<T>& target,
                      matrix<T>& grad);

        void update();
        void initializeWeightsAndBiases();
        void initializeInput();
        void initializeHidden();
        void initializeOutput();
        void allocateWeightsAndBiases();
        //void initializeWeight();
        //void initializeBias();
        //void initializeActivationFunction();
        //void initializeWeightGrad();
    public:
        //constructor
        auto_encoder(size_t inputDim, size_t hiddenDim, size_t outputDim);
        //copy constructor
        auto_encoder(const auto_encoder<T,real_x>& other);
        //copy assignment
        auto_encoder<T,real_x>& operator=(const auto_encoder<T,real_x>& other);
        //move constructor
        auto_encoder(auto_encoder<T,real_x>&& other);
        //move assignment
        auto_encoder<T,real_x>& operator=(auto_encoder<T,real_x>&& other);
        //destructor
        virtual ~auto_encoder(); 
        //activation functions
        T sigmoid(T x);
        T sigmoidPrime(T x);
        T sigmoidGradient(T x);
        T relu(T x);
        T reluGradient(T x);
        T reLuLeaky(T x);
        T reLuLeakyGradient(T x);
        T leakyReluPrime(T x);
        T leakyRelu(T x);
        T leakyReluGradient(T x);

        T identityGradient(T x);
        T tanhPrime(T x);
        T reluPrime(T x);
        T softplusPrime(T x);
        T linearPrime(T x);
        T softmaxPrime(T x);
        T identity(T x);

        T tanh(T x);
        T tanhGradient(T x);
        T softplus(T x);
        T softplusGradient(T x);
        T linear(T x);
        T linearGradient(T x);
        T softmax(T x);
        T softmaxGradient(T x);

        T gaussian(T x);
        T gaussianGradient(T x);
        T gaussianPrime(T x);

        T sinusoid(T x);
        T sinusoidGradient(T x);
        T sinusoidPrime(T x);

        T softsign(T x);
        T softsignGradient(T x);
        T softsignPrime(T x);

        T sinc(T x);
        T sincGradient(T x);
        T sincPrime(T x);

        T bentIdentity(T x);
        T bentIdentityGradient(T x);
        T bentIdentityPrime(T x);

        T softExponentialPrime(T x);
        T softExponential(T x);
        T softExponentialGradient(T x);
        //copy helpers 
        void copy(const auto_encoder<T,real_x>& other);
        void copy(auto_encoder<T,real_x>&& other);
        //copy weights and biases
        void copyWeightsAndBiases(const auto_encoder<T,real_x>& other);
        void copyWeightsAndBiases(auto_encoder<T,real_x>&& other);
        //copy activation functions
        void copyActivationFunctions(const auto_encoder<T,real_x>& other);
        void copyActivationFunctions(auto_encoder<T,real_x>&& other);
        //copy input
        void copyInput(const auto_encoder<T,real_x>& other);
        void copyInput(auto_encoder<T,real_x>&& other);
        //copy hidden
        void copyHidden(const auto_encoder<T,real_x>& other);
        void copyHidden(auto_encoder<T,real_x>&& other);
        //copy output
        void copyOutput(const auto_encoder<T,real_x>& other);
        void copyOutput(auto_encoder<T,real_x>&& other);
        //copy input dim
        void copyInputDim(const auto_encoder<T,real_x>& other);
        void copyInputDim(auto_encoder<T,real_x>&& other);
        //copy hidden dim
        void copyHiddenDim(const auto_encoder<T,real_x>& other);
        void copyHiddenDim(auto_encoder<T,real_x>&& other);
        //copy output dim
        void copyOutputDim(const auto_encoder<T,real_x>& other);
        void copyOutputDim(auto_encoder<T,real_x>&& other);
        void copy_parameters(const auto_encoder<T,real_x>& other);
        void copy_parameters(auto_encoder<T,real_x>&& other);
        //copy weight grads 
        void copyWeightGrad(const auto_encoder<T,real_x>& other);
        
        //cost function

        inline T cost(T *i1, T *o1, size_t size)
        {

            T cost = 0;
 
            for (size_t i = 0; i < size; i++)
            {
                cost += (i1[i] -o1[i]) * (i1[i] -o1[i]);
            }
            return cost;

        }
        void train(T *input, T *output, size_t size);
        void train (matrix<T>& input,  class_dist& output);
 
        //predict
        void predict(T *input, size_t size, T *output, size_t outputSize)
        {

            

            forward(input, size);

            //copy the output
            
            //don't forget to update the gradients 
            backward(input, size, output, outputSize);
            update();
            



        }
        void predict (const matrix<T>& input,  matrix<T>& output)
        {
            forward(input);
            size_t size = output.size2();
            for ( size_t i=0;i<output.size1();i++)
            {
                for (size_t j=0;j<size;j++)
                {
                    output(i,j) = hidden[i];
                    output(i,j) = output(i,j) * weight2[i+j*size];
                    output(i,j) = output(i,j) + bias2[i+j*size  ]   ;

                    output(i,j) = (this->*activationFunctionPtr)(output(i,j)); 
                }
            }
            //update gradients

            backward(input, output);
            update();
            
            

        }

        void test(T *input, T *output, size_t size);
        void test (matrix<T>& input,  class_dist& output);


        void dump(  std::ostream &out = std::cout) const;  
        
        void save(std::string filename);
        void save_as_pt(std::string filename);

        void load(std::string filename);
        void load(std::istream &in);

        void feedforward(T *input, T *output, size_t size);
        
        void backprop(T *input, T *output, size_t size);

        T *getWeight1()
        {
            return weight1;
        }
        T *getWeight2()
        {
            return weight2;
        }
        void setInput(T *input);
        void setHidden(T *hidden);
        void setOutput(T *output);
        //set dimension
        void setInputDim(size_t inputDim);
        void setHiddenDim(size_t hiddenDim);
        void setOutputDim(size_t outputDim);


        T *getInput() const;
        T *getHidden() const;
        T *getOutput() const;
        size_t getInputDim() const;
        size_t getHiddenDim() const;
        size_t getOutputDim() const;
        void setLearningRate(real_x learningRate);
        void setMomentum(real_x momentum);
        void setWeightDecay(real_x weightDecay);
        void setSparsityParam(real_x sparsityParam);
        void setBeta(real_x beta);

        real_x getLearningRate() const;
        real_x getMomentum() const;
        real_x getWeightDecay() const;
        real_x getSparsityParam() const;
        real_x getBeta() const;

        T *getWeight1() const
        {
            return weight1;
        }
        T *getWeight2() const
        {
            return weight2;
        }
        void setWeight1(T *weight1);
        void setWeight2(T *weight2);
        void setBias1(T *bias1);
        void setBias2(T *bias2);

        T *getBias1() const;
        T *getBias2() const;


        void setSparsityParamHat(real_x sparsityParamHat);
        void setSparsityPenalty(real_x sparsityPenalty);
        void setSparsityGradient(real_x sparsityGradient);
        void setSparsityGradientHat(real_x sparsityGradientHat);
        
        real_x getSparsityParamHat() const;
        real_x getSparsityPenalty() const;


 

        void setWeight1Inc(T* weight1Inc);
        void setWeight2Inc(T* weight2Inc);
        void setBias1Inc(T* bias1Inc);
        void setBias2Inc(T* bias2Inc);

        T *getWeight1Inc() const
        {
            return weight1Inc;
        }
        T *getWeight2Inc() const
        {
            return weight2Inc;
        }

        T *getBias1Inc() const
        {
            return bias1Inc;
        }

        T *getBias2Inc() const
        {
            return bias2Inc;
        }


        void setWeight1Grad(T* setWeight1Grad); 
        void setWeight2Grad(T* setWeight2Grad);
        void setBias1Grad(T* setBias1Grad);
        void setBias2Grad(T* setBias2Grad);

        T *getWeight1Grad() const
        {
            return weight1Grad;
        }
        T *getWeight2Grad() const
        {
            return weight2Grad;
        }
        T *getBias1Grad() const
        {
            return bias1Grad;
        }
        T *getBias2Grad() const
        {
            return bias2Grad;
        }


        //previous values :
        void setWeight1Prev(T* weight1Prev);
        void setWeight2Prev(T* weight2Prev);
        void setBias1Prev(T* bias1Prev);
        void setBias2Prev(T* bias2Prev);

        T *getWeight1Prev() const;
        T *getWeight2Prev() const;
        T *getBias1Prev() const;
        T *getBias2Prev() const;  


        void updateWeight1GradPrev();
        void updateWeight2GradPrev();
        void updateBias1GradPrev();
        void updateBias2GradPrev();
        
        void updateWeight1Inc();
        void updateWeight2Inc();
        void updateBias1Inc();
        void updateBias2Inc();

        void updateWeight1();
        void updateWeight2();
        void updateBias1();
        void updateBias2();


        void updateWeight1Decay();
        void updateWeight2Decay();
        void updateBias1Decay();
        void updateBias2Decay();


        void updateWeight1Sparsity();
        void updateWeight2Sparsity();
        void updateBias1Sparsity();
        void updateBias2Sparsity();

        void updateWeight1SparsityHat();
        void updateWeight2SparsityHat();
        void updateBias1SparsityHat();
        void updateBias2SparsityHat();


        void updateBias1GradPrevPrev();
        void updateBias2GradPrevPrev();
        void updateWeight1GradPrevPrev();
        void updateWeight2GradPrevPrev();

        void updateWeight1SparsityGrad();
        void updateWeight2SparsityGrad();
        void updateBias1SparsityGrad();
        void updateBias2SparsityGrad();

        void updateWeight1SparsityGradHat();
        void updateWeight2SparsityGradHat();
        void updateBias1SparsityGradHat();
        void updateBias2SparsityGradHat();

        
        void initializeWeight1Grad()
        {
            initializeWeight(weight1Grad, inputDim, hiddenDim);
        }
        void initializeWeight2Grad()
        {
            initializeWeight(weight2Grad, hiddenDim, outputDim);
        }
        void initializeBias1Grad()
        {
            initializeBias(bias1Grad, hiddenDim);
        }
        void initializeBias2Grad()
        {
            initializeBias(bias2Grad, outputDim);
        }

        void initializeWeight1GradPrev()
        {
            initializeWeight(weight1GradPrev, inputDim, hiddenDim);
        }
        void initializeWeight2GradPrev()
        {
            initializeWeight(weight2GradPrev, hiddenDim, outputDim);
        }
        void initializeBias1GradPrev()  
        {
            initializeBias(bias1GradPrev, hiddenDim);
        }

        void initializeBias2GradPrev()
        {
            initializeBias(bias2GradPrev, outputDim);
        }
        void initializeWeight1Inc()
        {
            initializeWeight(weight1Inc, inputDim, hiddenDim);
        }
        void initializeWeight2Inc()
        {
            initializeWeight(weight2Inc, hiddenDim, outputDim);
        }
        void initializeBias1Inc()
        {
            initializeBias(bias1Inc, hiddenDim);
        }
        void initializeBias2Inc()
        {
            initializeBias(bias2Inc, outputDim);
        }
        void initializeWeight1SparsityGrad()
        {
            initializeWeight(weight1SparsityGrad, inputDim, hiddenDim);
        }
        void initializeWeight2SparsityGrad()
        {
            initializeWeight(weight2SparsityGrad, hiddenDim, outputDim);
        }
        void initializeBias1SparsityGrad()
        {
            initializeBias(bias1SparsityGrad, hiddenDim);
        }
        void initializeBias2SparsityGrad()
        {
            initializeBias(bias2SparsityGrad, outputDim);
        }
        void initializeWeight1SparsityGradHat()
        {
            initializeWeight(weight1SparsityGradHat, inputDim, hiddenDim);
        }
        void initializeWeight2SparsityGradHat()
        {
            initializeWeight(weight2SparsityGradHat, hiddenDim, outputDim);
        }
        void initializeBias1SparsityGradHat()
        {
            initializeBias(bias1SparsityGradHat, hiddenDim);
        }
        void initializeBias2SparsityGradHat()
        {
            initializeBias(bias2SparsityGradHat, outputDim);
        }
        void initializeWeight1SparsityHat()
        {
            initializeWeight(weight1SparsityHat, inputDim, hiddenDim);
        }
        void initializeWeight2SparsityHat()
        {
            initializeWeight(weight2SparsityHat, hiddenDim, outputDim);
        }
        void initializeBias1SparsityHat()
        {
            initializeBias(bias1SparsityHat, hiddenDim);
        }
        void initializeBias2SparsityHat()
        {
            initializeBias(bias2SparsityHat, outputDim);
        }
        void initializeWeight1Sparsity()
        {
            initializeWeight(weight1Sparsity, inputDim, hiddenDim);
        }
        void initializeWeight2Sparsity()
        {
            initializeWeight(weight2Sparsity, hiddenDim, outputDim);
        }
        void initializeBias1Sparsity()
        {
            initializeBias(bias1Sparsity, hiddenDim);
        }
        void initializeBias2Sparsity()
        {
            initializeBias(bias2Sparsity, outputDim);
        }
        void initializeWeight1Decay()
        {
            initializeWeight(weight1Decay, inputDim, hiddenDim);
        }
        void initializeWeight2Decay()
        {
            initializeWeight(weight2Decay, hiddenDim, outputDim);
        }
        void initializeBias1Decay()
        {
            if(bias1Decay == nullptr)
                bias1Decay = new T[hiddenDim];
            initializeBias(bias1Decay, hiddenDim);
        }
        void initializeBias2Decay()
        {
            if(bias2Decay == nullptr)
                bias2Decay = new T[outputDim]; 
            initializeBias(bias2Decay, outputDim);
        }
        void initializeWeight1Prev()
        {
            if(weight1Prev == nullptr)
                weight1Prev = new T[inputDim * hiddenDim];  

            initializeWeight(weight1Prev, inputDim, hiddenDim);

        }
        void initializeWeight2Prev()
        {
            if(weight2Prev == nullptr)
                weight2Prev = new T[hiddenDim * outputDim];
            initializeWeight(weight2Prev, hiddenDim, outputDim);
        }
        void initializeBias1Prev()
        {   
            if(bias1Prev == nullptr)
                bias1Prev = new T[hiddenDim];
            initializeBias(bias1Prev, hiddenDim);

        }
        void initializeBias2Prev()
        {
            if(bias2Prev == nullptr)
                bias2Prev = new T[outputDim];
            initializeBias(bias2Prev, outputDim);
        }
        void initializeWeight1Momentum()
        {
            if(weight1Momentum == nullptr)
                weight1Momentum = new T[inputDim * hiddenDim];  

            initializeWeight(weight1Momentum, inputDim, hiddenDim);
        }
        void initializeWeight2Momentum()
        {
            if(weight2Momentum == nullptr)
                weight2Momentum = new T[hiddenDim * outputDim]; 
            initializeWeight(weight2Momentum, hiddenDim, outputDim);
        }
        void initializeBias1Momentum()
        {
            if(bias1Momentum == nullptr)
                bias1Momentum = new T[hiddenDim];   
            initializeBias(bias1Momentum, hiddenDim);
        }
        void initializeBias2Momentum()
        {
            if(bias2Momentum == nullptr)
                bias2Momentum = new T[outputDim];   
            initializeBias(bias2Momentum, outputDim);
        }
        void initializeWeight1Update()
        {
            if(weight1Update == nullptr)
                weight1Update = new T[inputDim * hiddenDim];    
            initializeWeight(weight1Update, inputDim, hiddenDim);
        }
        void initializeWeight2Update()
        {
            if(weight2Update == nullptr)
                weight2Update = new T[hiddenDim * outputDim];   
            initializeWeight(weight2Update, hiddenDim, outputDim);
        }
        void initializeBias1Update()
        {
            if(bias1Update == nullptr)
                bias1Update = new T[hiddenDim]; 
            initializeBias(bias1Update, hiddenDim);
        }
        void initializeBias2Update()
        {
            if(bias2Update == nullptr)
                bias2Update = new T[outputDim];
            initializeBias(bias2Update, outputDim);
        }
        void initializeWeight1()
        {
            if(weight1 == nullptr)
                weight1 = new T[inputDim * hiddenDim];
            initializeWeight(weight1, inputDim, hiddenDim);
        }
        void initializeWeight2(){
            if(weight2 == nullptr)
                weight2 = new T[hiddenDim * outputDim];
            initializeWeight(weight2, hiddenDim, outputDim);
        }

        void initializeBias1()
        {
            if(bias1 == nullptr)
                bias1 = new T[hiddenDim];
            initializeBias(bias1, hiddenDim);
        }
        void initializeBias2()
        {
            if(bias2 == nullptr)
                bias2 = new T[outputDim];
            initializeBias(bias2, outputDim);
        }


        void initializeWeight1GradPrevPrev()
        {
            if(weight1GradPrevPrev == nullptr)
                weight1GradPrevPrev = new T[inputDim * hiddenDim];  
            initializeWeight(weight1GradPrevPrev, inputDim, hiddenDim);
        }
        void initializeWeight2GradPrevPrev()
        {
            if(weight1GradPrevPrev == nullptr)
                weight1GradPrevPrev = new T[hiddenDim * outputDim]; 
            initializeWeight(weight2GradPrevPrev, hiddenDim, outputDim);
        }
        void initializeBias1GradPrevPrev()
        {
            if(bias1GradPrevPrev == nullptr)
                bias1GradPrevPrev = new T[hiddenDim];
            initializeBias(bias1GradPrevPrev, hiddenDim);
        }
        void initializeBias2GradPrevPrev()
        {
            if(bias2GradPrevPrev == nullptr)
                bias2GradPrevPrev = new T[outputDim];
            initializeBias(bias2GradPrevPrev, outputDim);
        }




        //gnuplot 
        void gnuplot(const std::string filename)
        {
            std::ofstream out(filename); 
            size_t npos= filename.find_last_of('/',0) ;
            std::string title = filename.substr(npos+1, filename.find("."));
            
            out << "set terminal png" << std::endl;
            out << "set output \" "<<title<<"_autoencoder.png\"" << std::endl;
            out << "set title \""<<title<<"autoencoder\"" << std::endl;
            out << "set xlabel \"x\"" << std::endl;
            out << "set ylabel \"y\"" << std::endl;
            out << "plot \"autoencoder.dat\" using 1:2 title \"input\" with lines, \"autoencoder.dat\" using 1:3 title \"output\" with lines" << std::endl;
            out.close();
        }

        void updateWeight1Grad();   
        void updateWeight2Grad();
        void updateBias1Grad();
        void updateBias2Grad();
        //void updateWeight1();
        //void updateWeight2();
        //void updateBias1();
       // void updateBias2();
        void updateWeight1Prev();
        void updateWeight2Prev();
        void updateBias1Prev();
        void updateBias2Prev();
        void updateWeight1Momentum();
        void updateWeight2Momentum();
        void updateBias1Momentum();
        void updateBias2Momentum();
        void updateWeight1Update();
        void updateWeight2Update();

         
     };
    //variational auto encoder
    template <typename T, typename real_x = real_t>
    class variational_auto_encoder : public auto_encoder<T, real_x>
    {
        //variational auto encoder
        //variational auto encoder is a type of auto encoder that uses a variational bayesian approach to learning
        //additional variables for the variational auto encoder:
        //latentDim : the dimension of the latent space
        //latent : the latent space
        //latentMean : the mean of the latent space
        //latentLogVar : the log variance of the latent space
        //latentMeanGrad : the gradient of the latent mean
        //latentLogVarGrad : the gradient of the latent log variance

        //variational auto encoder uses the reparameterization trick to sample from the latent space
        //the reparameterization trick is used to sample from a distribution with a reparameterization of the distribution
         //variational auto encoder uses the kullback leibler divergence to measure the difference between the latent space and the prior distribution   
        protected:



        size_t latentDim;
        T *latent;
        T *latentMean;
        T *latentLogVar;
        T *latentMeanGrad;
        T *latentLogVarGrad;
        T *latentMeanGradPrev;
        T *latentLogVarGradPrev;
        T *latentMeanMomentum;
        T *latentLogVarMomentum;
        T *latentMeanUpdate;
        T *latentLogVarUpdate;
        T *latentMeanDecay;
        T *latentLogVarDecay;
        T *latentMeanSparsity;
        T *latentLogVarSparsity;
        T *latentMeanSparsityHat;
        T *latentLogVarSparsityHat;
        T *latentMeanSparsityGrad;
        T *latentLogVarSparsityGrad;
        T *latentMeanSparsityGradHat;
        T *latentLogVarSparsityGradHat;
        T *latentMeanGradPrevPrev;
        T *latentLogVarGradPrevPrev;
        T *latentMeanInc;
        T* latentMeanSparsityGradPrev;
        T* latentLogVarSparsityGradPrev;
        T* latentMeanSparsityGradHatPrev;
        T* latentLogVarSparsityGradHatPrev;
        T* latentMeanPrev;
        T* latentMeanSparsityGradPrevPrev;
        T* latentLogVarSparsityGradPrevPrev;
        T* latentMeanSparsityGradHatPrevPrev;
        //init helpers for the variational auto encoder
        void initializeLatent()
        {
            //reallocate the latent with the desired size:
            initialize(latent, latentDim);    

            //initialize the latent mean
            initializeLatentMean();
            //initialize the latent log var
            initializeLatentLogVar();
            //initialize the latent mean grad
            initializeLatentMeanGrad();
            //initialize the latent log var grad
            initializeLatentLogVarGrad();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize the latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize the latent mean momentum
            initializeLatentMeanMomentum();
            //initialize the latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize the latent mean update
            initializeLatentMeanUpdate();
            //initialize the latent log var update
            initializeLatentLogVarUpdate();
            //initialize the latent mean decay
            initializeLatentMeanDecay();
            //initialize the latent log var decay
            initializeLatentLogVarDecay();
            //initialize the latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize the latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize the latent mean sparsity hat
            initializeLatentMeanSparsityHat();
            //initialize the latent log var sparsity hat
            initializeLatentLogVarSparsityHat();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();



        }
        void initializeLatentMean()
        {
            initialize(latentMean, latentDim, T(0));
            //initialize the latent mean grad
            initializeLatentMeanGrad();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize the latent mean momentum
            initializeLatentMeanMomentum();
            //initialize the latent mean update
            initializeLatentMeanUpdate();
            //initialize the latent mean decay
            initializeLatentMeanDecay();
            //initialize the latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize the latent mean sparsity hat
            initializeLatentMeanSparsityHat();

             //done

        }
        void initializeLatentLogVar()
        {
            //reallocate the latent log var with the desired size:
            initialize(latentLogVar, latentDim, T(0));


            //initialize the latent log var grad
            initializeLatentLogVarGrad();
            //initialize the latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize the latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize the latent log var update
            initializeLatentLogVarUpdate();
            //initialize the latent log var decay
            initializeLatentLogVarDecay();
            //initialize the latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize the latent log var sparsity hat
            initializeLatentLogVarSparsityHat();

            //done
        }
        void initializeLatentMeanGrad()
        {
            //reallocate the latent mean grad with the desired size:
            initialize(latentMeanGrad, latentDim, T(0));    

            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize the latent mean momentum
            initializeLatentMeanMomentum();
            //initialize the latent mean update
            initializeLatentMeanUpdate();
            //initialize the latent mean decay
            initializeLatentMeanDecay();
            //initialize the latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize the latent mean sparsity hat
            initializeLatentMeanSparsityHat();

            //done
        }
        void initializeLatentLogVarGrad()
        {
            //reallocate the latent log var grad with the desired size:
          
            initialize(latentLogVarGrad, latentDim, T(0));
            //initialize the latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize the latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize the latent log var update
            initializeLatentLogVarUpdate();
            //initialize the latent log var decay
            initializeLatentLogVarDecay();
            //initialize the latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize the latent log var sparsity hat
            initializeLatentLogVarSparsityHat();

            //done

        }
        void initializeLatentMeanGradPrev()
        {
            initialize(latentMeanGradPrev, latentDim, T(0));

            //done
        }
        void initializeLatentLogVarGradPrev()
        {
            initialize(latentLogVarGradPrev, latentDim, T(0));

            //done
        }
        void initializeLatentMeanMomentum()
        {
            initialize(latentMeanMomentum, latentDim, T(0));    
            
            //done      
        }
        void initializeLatentLogVarMomentum()
        {
            initialize(latentLogVarMomentum, latentDim, T(0));
            //done
        }
        void initialize ( T* _init_member,size_t size, const T value_)
        {
            //reallocate the member with the desired size:
            if(_init_member != nullptr)
            {
                delete [] _init_member;
                _init_member=nullptr;

            }
            _init_member = new T[size];
            //initialize the member
            for (size_t i = 0; i < size; i++)
            {
                _init_member[i] = value_;
            }
        }
        void initializeLatentMeanUpdate()
        {
            //reallocate the latent mean update with the desired size:
            initialize(latentMeanUpdate, latentDim, T(0));  
        }
        void initializeLatentLogVarUpdate()
        {
            //reallocate the latent log var update with the desired size:
            initialize(latentLogVarUpdate, latentDim, T(0));  
        }
        void initializeLatentMeanDecay()
        {
            //reallocate the latent mean decay with the desired size:
            initialize(latentMeanDecay, latentDim, T(0));  
        }
        void initializeLatentLogVarDecay()
        {
            //reallocate the latent log var decay with the desired size:
            initialize(latentLogVarDecay, latentDim, T(0));  
        }
        void initializeLatentMeanSparsity()
        {
            //reallocate the latent mean sparsity with the desired size:
            initialize(latentMeanSparsity, latentDim, T(0));  
        }
        void initializeLatentLogVarSparsity()
        {
            //reallocate the latent log var sparsity with the desired size:
            initialize(latentLogVarSparsity, latentDim, T(0));  
        }
        void initializeLatentMeanSparsityHat()
        {
            //reallocate the latent mean sparsity hat with the desired size:
            initialize(latentMeanSparsityHat, latentDim, T(0));  
        }
        void initializeLatentLogVarSparsityHat()
        {
            //reallocate the latent log var sparsity hat with the desired size:
            initialize(latentLogVarSparsityHat, latentDim, T(0));  
        }
        //update helpers for the variational auto encoder
        void updateLatentMeanGrad()
        {
            //update the latent mean grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanGrad[i] = latentMeanGradPrev[i] + latentMeanGrad[i];
            }
        }
        void updateLatentLogVarGrad()
        {
            //update the latent log var grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarGrad[i] = latentLogVarGradPrev[i] + latentLogVarGrad[i];
            }
        }
        void updateLatentMeanGradPrev()
        {
            //update the latent mean grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanGradPrev[i] = latentMeanGrad[i];
            }
        }
        void updateLatentLogVarGradPrev()
        {
            //update the latent log var grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarGradPrev[i] = latentLogVarGrad[i];
            }
        }
        void updateLatentMeanMomentum()
        {
            //update the latent mean momentum
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanMomentum[i] = latentMeanMomentum[i] * this->momentum + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarMomentum()
        {
            //update the latent log var momentum
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarMomentum[i] = latentLogVarMomentum[i] * this->momentum + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanUpdate()
        {
            //update the latent mean update
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanUpdate[i] = latentMeanUpdate[i] * this->momentum + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarUpdate()
        {
            //update the latent log var update
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarUpdate[i] = latentLogVarUpdate[i] * this->momentum + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanDecay()
        {
            //update the latent mean decay
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanDecay[i] = latentMeanDecay[i] * this->weightDecay + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarDecay()
        {
            //update the latent log var decay
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarDecay[i] = latentLogVarDecay[i] * this->weightDecay + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanSparsity()
        {
            //update the latent mean sparsity
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsity[i] = latentMeanSparsity[i] * this->sparsityParam + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsity()
        {
            //update the latent log var sparsity
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsity[i] = latentLogVarSparsity[i] * this->sparsityParam + this->learningRate * latentLogVarGrad[i];
            }
        }   
        void updateLatentMeanSparsityHat()
        {
            //update the latent mean sparsity hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityHat[i] = latentMeanSparsityHat[i] * this->sparsityParamHat + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsityHat()
        {
            //update the latent log var sparsity hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityHat[i] = latentLogVarSparsityHat[i] * this->sparsityParamHat + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMean()
        {
            //update the latent mean
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMean[i] = latentMean[i] + latentMeanUpdate[i] + latentMeanMomentum[i] + latentMeanDecay[i] + latentMeanSparsity[i] + latentMeanSparsityHat[i];
            }
        }
        void updateLatentLogVar()
        {
            //update the latent log var
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVar[i] = latentLogVar[i] + latentLogVarUpdate[i] + latentLogVarMomentum[i] + latentLogVarDecay[i] + latentLogVarSparsity[i] + latentLogVarSparsityHat[i];
            }
        }
        void updateLatentMeanGradPrevPrev()
        {
            //update the latent mean grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanGradPrevPrev[i] = latentMeanGradPrev[i];
            }
        }
        void updateLatentLogVarGradPrevPrev()
        {
            //update the latent log var grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarGradPrevPrev[i] = latentLogVarGradPrev[i];
            }
        }
        void updateLatentMeanSparsityGrad()
        {
            //update the latent mean sparsity grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGrad[i] = latentMeanSparsityGrad[i] * this->sparsityGradient + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsityGrad()
        {
            //update the latent log var sparsity grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGrad[i] = latentLogVarSparsityGrad[i] * this->sparsityGradient + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanSparsityGradHat()
        {
            //update the latent mean sparsity grad hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradHat[i] = latentMeanSparsityGradHat[i] * this->sparsityGradientHat + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsityGradHat()
        {
            //update the latent log var sparsity grad hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradHat[i] = latentLogVarSparsityGradHat[i] * this->sparsityGradientHat + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanSparsityGradPrev()
        {
            //update the latent mean sparsity grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradPrev[i] = latentMeanSparsityGrad[i];
            }
        }
        void updateLatentLogVarSparsityGradPrev()
        {
            //update the latent log var sparsity grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradPrev[i] = latentLogVarSparsityGrad[i];
            }
        }
        void updateLatentMeanSparsityGradHatPrev()
        {
            //update the latent mean sparsity grad hat prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradHatPrev[i] = latentMeanSparsityGradHat[i];
            }
        }   
        void updateLatentLogVarSparsityGradHatPrev()
        {
            //update the latent log var sparsity grad hat prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradHatPrev[i] = latentLogVarSparsityGradHat[i];
            }
        }
        void updateLatentMeanSparsityGradPrevPrev()
        {
            //update the latent mean sparsity grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradPrevPrev[i] = latentMeanSparsityGradPrev[i];
            }
        }
        void updateLatentLogVarSparsityGradPrevPrev()
        {
            //update the latent log var sparsity grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradPrevPrev[i] = latentLogVarSparsityGradPrev[i];
            }
        }
        void updateLatentMeanSparsityGradHatPrevPrev()
        {
            //update the latent mean sparsity grad hat prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradHatPrevPrev[i] = latentMeanSparsityGradHatPrev[i];
            }
        }
        //

        public:

        
        variational_auto_encoder(size_t inputDim, size_t hiddenDim, size_t outputDim, size_t lDim) :
        auto_encoder<T,real_x>(inputDim,hiddenDim,outputDim) , latentDim(lDim)
        {
            //initialize the latent dim
            setLatentDim(latentDim);
            //initialize the latent space
            initializeLatent();
            //initialize the latent mean
            initializeLatentMean();
            //initialize the latent log var
            initializeLatentLogVar();
            //initialize the latent mean grad
            initializeLatentMeanGrad();
            //initialize the latent log var grad
            initializeLatentLogVarGrad();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //call the initialize weight grad function
            initializeWeightGrad();

        } 

        //destructor
        virtual ~variational_auto_encoder();
        //copy constructor
        variational_auto_encoder(const variational_auto_encoder &vae);
        //copy assignment operator
        variational_auto_encoder& operator=(const variational_auto_encoder &vae);
        //move constructor
        variational_auto_encoder(variational_auto_encoder &&vae);
        //move assignment operator
        variational_auto_encoder& operator=(variational_auto_encoder &&vae);

        variational_auto_encoder(matrix<T> input,matrix<T> target,std::vector<T> classes) : auto_encoder<T,real_x>(input,target,classes)
        {
            //call the constructor with the correct number of latent dimensions 
            //based on the input and target

            //initialize the latent dim
            //latent dim is the number of classes + 1 
            //count unique classes and add 1 for the bias
            size_t latentDim =  std::unique(classes.begin(),classes.end()) - classes.begin() + 1; 

            setLatentDim(latentDim);
            //set the input/output dim
            setInputDim(input.cols());
            setOutputDim(target.cols());

            //set the inputs and outputs from the matrices
            setInputs(input);
            setTargets(target);
            //set the classes
            setClasses(classes);
            setInputDim(input.cols());
            setOutputDim(target.cols()); 

            //initialize the latent space
            

            initializeLatent();
            //initialize the latent mean
            initializeLatentMean();
            //initialize the latent log var
            initializeLatentLogVar();
            //initialize the latent mean grad
            initializeLatentMeanGrad();
            //initialize the latent log var grad
            initializeLatentLogVarGrad();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //call the initialize weight grad function
            initializeWeightGrad();

        }
        //get the latent dim
        void setLatentDim(size_t latentDim);

        size_t getLatentDim() const;
        
        //override auto encoder functions
        void initializeWeightGrad() override
        {
            auto_encoder<T,real_x>::initializeWeightGrad();
            //initialize the gradients
            initializeWeight1Grad();
            initializeWeight2Grad();
            //initialize bias grads
            initializeBias1Grad();
            initializeBias2Grad();
            //initialize latent mean grad
            initializeLatentMeanGrad();
            //initialize latent log var grad
            initializeLatentLogVarGrad();
            //initialize latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize latent mean momentum
            initializeLatentMeanMomentum();
            //initialize latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize latent mean update
            initializeLatentMeanUpdate();
            //initialize latent log var update
            initializeLatentLogVarUpdate();
            //initialize latent mean decay
            initializeLatentMeanDecay();
            //initialize latent log var decay
            initializeLatentLogVarDecay();
            //initialize latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize latent mean sparsity hat

            initializeLatentMeanSparsityHat();
            //initialize latent log var sparsity hat
            initializeLatentLogVarSparsityHat();

            //initialize weight prev
            auto_encoder<T,real_x>::initializeWeight1Prev();
            auto_encoder<T,real_x>::initializeWeight2Prev();
            //initialize bias prev
            auto_encoder<T,real_x>::initializeBias1Prev();
            auto_encoder<T,real_x>::initializeBias2Prev();
            //initialize latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize latent mean momentum
            initializeLatentMeanMomentum();
            //initialize latent log var momentum
            initializeLatentLogVarMomentum();


            //initialize latent mean update
            initializeLatentMeanUpdate();

        }
        void initializeWeightGrad(T *weightGrad, size_t size)
        {
                initialize(weightGrad, size, T(0));

        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col)
        {
            initialize(weightGrad, row*col,T(0));

        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth)
        {
            initialize(weightGrad, row*col*depth,T(0));

        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height)
        {
            initialize(weightGrad, row*col*depth*height,T(0));
        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height, size_t width)
        {
            initialize(weightGrad, row*col*depth*height*width,T(0));
        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height, size_t width, size_t length)
        {
            initialize(weightGrad, row*col*depth*height*width*length,T(0));
        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height, size_t width, size_t length, size_t dimension)
        {
            initialize(weightGrad, row*col*depth*height*width*length*dimension,T(0));
        }
        //fit with latent space

        void fit(T *input, size_t inputSize, size_t batchSize, size_t epoch)
        {
            //fit the model
            for (size_t i = 0; i < epoch; i++)
            {
                //train the model
                train(input, inputSize, batchSize);
            }
        }
     
        class_dist predict(T *input, size_t inputSize)
        {
            //predict the class distribution
            class_dist dist;
            //get the output
            predict(input, inputSize, dist);
            //return the class distribution
            return dist;
        }
        void predict(T *input, size_t inputSize, T *output, size_t outputSize)
        {
            //predict the output
            //get the output
            predict(input, inputSize, output, outputSize);
        }
        
        void predict (  provallo::matrix <T> &input, provallo::matrix <T> &output)
        {
            //predict the output
            //get the output
            predict(input.data(), input.size1()*input.size2(), output.data(), output.size1()*output.size2());   
        }

        virtual void train(T *input, size_t inputSize, size_t batchSize)
        {
            //Train the model
            //get the batch
            T *batch = getBatch(input, inputSize, batchSize);
            //train the model
            train(batch, batchSize);

            //delete the batch ?
            delete [] batch;
            
        }

        //train with matrix input
        virtual void train(provallo::matrix <T> &input ,matrix<T> & out  )
        {
            //train the model
            //get the batch
            T *batch = getBatch(input.data(), input.size1()*input.size2(), out.size1()*out.size2());
            //train the model
            train(batch, out.data(), out.size1()*out.size2());

            for ( size_t i = 0; i < out.size1(); i++)
            {
                for (size_t j = 0; j < out.size2(); j++)
                {
                    out(i,j) = batch[i*out.size2()+j];
                }
                
            }   
            
            //done

        }   

  
        virtual class_dist test(T *input, size_t inputSize)
        {
            //test the model
            class_dist dist;
            dist.setup(this->outputDim);
            //get the output
            //test(input, inputSize, dist);
            T* output = new T[this->outputDim]; 
            if (output != nullptr) {
            test(input, inputSize, output, this->outputDim);
            //get the class distribution
                for(size_t i = 0; i < dist.size(); i++)
                {
                    dist.accum(i,output[i]);
                }
                delete [] output;
            }
            //return the class distribution
            return dist;
        }

        virtual void test(T *input, size_t inputSize, T *output, size_t outputSize)
        {
            //test the model
            //get the output
            predict(input, inputSize, output, outputSize);

            
        }
        virtual void initializeWeight();
        virtual void initializeBias();
        virtual void initializeActivationFunction();
        virtual void initializeWeight1Grad();
        virtual void initializeWeight2Grad();
        virtual void initializeBias1Grad();
        virtual void initializeBias2Grad();
        virtual void initializeWeight1Momentum();   
 
    };   


    template <typename T, typename real_x>
    inline auto_encoder<T, real_x>::auto_encoder(size_t inputD, size_t hiddenD, size_t outputD):inputDim(inputD),hiddenDim(hiddenD),outputDim(outputD),
    weight1(nullptr),weight2(nullptr),bias1(nullptr),bias2(nullptr),weight1Grad(nullptr),weight2Grad(nullptr),bias1Grad(nullptr),bias2Grad(nullptr),
    weight1Momentum(nullptr),weight2Momentum(nullptr),bias1Momentum(nullptr),bias2Momentum(nullptr),weight1Update(nullptr),weight2Update(nullptr),
    bias1Update(nullptr),bias2Update(nullptr),weight1Decay(nullptr),weight2Decay(nullptr),bias1Decay(nullptr),bias2Decay(nullptr),
    weight1Sparsity(nullptr),weight2Sparsity(nullptr),bias1Sparsity(nullptr),bias2Sparsity(nullptr),weight1SparsityHat(nullptr),weight2SparsityHat(nullptr),
    bias1SparsityHat(nullptr),bias2SparsityHat(nullptr),weight1GradPrev(nullptr),weight2GradPrev(nullptr),bias1GradPrev(nullptr),bias2GradPrev(nullptr),
    weight1GradPrevPrev(nullptr),weight2GradPrevPrev(nullptr),bias1GradPrevPrev(nullptr),bias2GradPrevPrev(nullptr)
    {
        //std::cout << "auto_encoder constructor" << std::endl;
        if(inputDim==0||outputDim==0||hiddenDim==0)
        {
            throw std::runtime_error("auto_encoder constructor: inputDim, hiddenDim, or outputDim is zero");
        }
        //initialize the auto encoder
        initialize_autoencoder();
        //

    }
      
    template <typename T, typename real_x>
    auto_encoder<T, real_x>::~auto_encoder()
    {
        if (input != nullptr)
        {
            delete[] input;
            input = nullptr;
        }
        if (hidden != nullptr)
        {
            delete[] hidden;
            hidden = nullptr;
        }
        if (output != nullptr)
        {
            delete[] output;
            output = nullptr;
        }
        if (weight1 != nullptr)
        {
            delete[] weight1;
            weight1 = nullptr;
        }
        if (weight2 != nullptr)
        {
            delete[] weight2;
            weight2 = nullptr;
        }
        if (bias1 != nullptr)
        {
            delete[] bias1;
            bias1 = nullptr;
        }
        if (bias2 != nullptr)
        {
            delete[] bias2;
            bias2 = nullptr;
        }
        if (weight1Grad != nullptr)
        {
            delete[] weight1Grad;
            weight1Grad = nullptr;
        }
        if (weight2Grad != nullptr)
        {
            delete[] weight2Grad;
            weight2Grad = nullptr;
        }
        if (bias1Grad != nullptr)
        {
            delete[] bias1Grad;
            bias1Grad = nullptr;
        }
        if (bias2Grad != nullptr)
        {
            delete[] bias2Grad;
            bias2Grad = nullptr;
        }
        if (weight1Momentum != nullptr)
        {
            delete[] weight1Momentum;
            weight1Momentum = nullptr;
        }
        if (weight2Momentum != nullptr)
        {
            delete[] weight2Momentum;
            weight2Momentum = nullptr;
        }

        if (bias1Momentum != nullptr)
        {
            delete[] bias1Momentum;
            bias1Momentum = nullptr;
        }   

        if (bias2Momentum != nullptr)
        {
            delete[] bias2Momentum;
            bias2Momentum = nullptr;
        }   

        if (weight1Update != nullptr)
        {
            delete[] weight1Update;
            weight1Update = nullptr;
        }   

        if (weight2Update != nullptr)
        {
            delete[] weight2Update;
            weight2Update = nullptr;
        }   

        if (bias1Update != nullptr)
        {
            delete[] bias1Update;
            bias1Update = nullptr;
        }   
        //bias2Update
        if (bias2Update != nullptr)
        {
            delete[] bias2Update;
            bias2Update = nullptr;
        }
        if (weight1Decay != nullptr)
        {
            delete[] weight1Decay;
            weight1Decay = nullptr;
        }
        if (weight2Decay != nullptr)
        {
            delete[] weight2Decay;
            weight2Decay = nullptr;
        }
        //bias1Decay
        if (bias1Decay != nullptr)
        {
            delete[] bias1Decay;
            bias1Decay = nullptr;
        }
        //bias2Decay
        if (bias2Decay != nullptr)
        {
            delete[] bias2Decay;
            bias2Decay = nullptr;
        }
        //weight1Sparsity
        if (weight1Sparsity != nullptr)
        {
            delete[] weight1Sparsity;
            weight1Sparsity = nullptr;
        }
        //weight2Sparsity
        if (weight2Sparsity != nullptr)
        {
            delete[] weight2Sparsity;
            weight2Sparsity = nullptr;
        }
        //bias1Sparsity
        if (bias1Sparsity != nullptr)
        {
            delete[] bias1Sparsity;
            bias1Sparsity = nullptr;
        }
        //bias2Sparsity
        if (bias2Sparsity != nullptr)
        {
            delete[] bias2Sparsity;
            bias2Sparsity = nullptr;
        }
        //weight1SparsityHat
        if (weight1SparsityHat != nullptr)
        {
            delete[] weight1SparsityHat;
            weight1SparsityHat = nullptr;
        }
        //weight2SparsityHat
        if (weight2SparsityHat != nullptr)
        {
            delete[] weight2SparsityHat;
            weight2SparsityHat = nullptr;
        }
        //bias1SparsityHat
        if (bias1SparsityHat != nullptr)
        {
            delete[] bias1SparsityHat;
            bias1SparsityHat = nullptr;
        }
        //bias2SparsityHat
        if (bias2SparsityHat != nullptr)
        {
            delete[] bias2SparsityHat;
            bias2SparsityHat = nullptr;
        }
        //weight1SparsityGrad
        if (weight1SparsityGrad != nullptr)
        {
            delete[] weight1SparsityGrad;
            weight1SparsityGrad = nullptr;
        }
        //weight2SparsityGrad
        if (weight2SparsityGrad != nullptr)
        {   

            delete[] weight2SparsityGrad;
            weight2SparsityGrad = nullptr;
        }
        //bias1SparsityGrad
        if (bias1SparsityGrad != nullptr)
        {
            delete[] bias1SparsityGrad;
            bias1SparsityGrad = nullptr;
        }
        //bias2SparsityGrad
        if (bias2SparsityGrad != nullptr)
        {
            delete[] bias2SparsityGrad;
            bias2SparsityGrad = nullptr;
        }
        //weight1SparsityGradHat
        if (weight1SparsityGradHat != nullptr)
        {
            delete[] weight1SparsityGradHat;
            weight1SparsityGradHat = nullptr;
        }
        //weight2SparsityGradHat
        if (weight2SparsityGradHat != nullptr)
        {
            delete[] weight2SparsityGradHat;
            weight2SparsityGradHat = nullptr;
        }
        //weight1Inc
        if (weight1Inc != nullptr)
        {
            delete[] weight1Inc;
            weight1Inc = nullptr;
        }
        //weight2Inc
        if (weight2Inc != nullptr)
        {
            delete[] weight2Inc;
            weight2Inc = nullptr;
        }
        //weight1GradPrev
        if (weight1GradPrev != nullptr)
        {
            delete[] weight1GradPrev;
            weight1GradPrev = nullptr;
        }
        //weight2GradPrev
        if (weight2GradPrev != nullptr)
        {
            delete[] weight2GradPrev;
            weight2GradPrev = nullptr;
        }
        //bias1Inc
        if (bias1Inc != nullptr)
        {
            delete[] bias1Inc;
            bias1Inc = nullptr;
        }
        //bias2Inc
        if (bias2Inc != nullptr)
        {
            delete[] bias2Inc;
            bias2Inc = nullptr;
        }
        //bias1GradPrev
        if (bias1GradPrev != nullptr)
        {
            delete[] bias1GradPrev;
            bias1GradPrev = nullptr;
        }
        //bias2GradPrev
        if (bias2GradPrev != nullptr)
        {
            delete[] bias2GradPrev;
            bias2GradPrev = nullptr;
        }

        //prevprev
        if (weight1GradPrevPrev != nullptr)
        {
            delete[] weight1GradPrevPrev;
            weight1GradPrevPrev = nullptr;
        }
        //prevprev
        if (weight2GradPrevPrev != nullptr)
        {
            delete[] weight2GradPrevPrev;
            weight2GradPrevPrev = nullptr;
        }
        
        //done
    }
    //initialize autoencoder
    template <typename T, typename real_x> 
    void auto_encoder<T,real_x>::initialize_autoencoder()
    {
        //initialize the autoencoder

        initializeInput();
        initializeHidden();
        initializeOutput();
        allocateWeightsAndBiases();
        initializeWeight();
        initializeBias();
        initializeActivationFunction();
        initializeWeightGrad();
        initializeWeight1Momentum();
        initializeWeight2Momentum();
        initializeBias1Momentum();

    } 
    template <typename T, typename real_x> 
    void auto_encoder<T,real_x>::allocateWeightsAndBiases()
    {
            //allocate the weights and biases
 
            if(weight1)
            {
                delete [] weight1;
                weight1 = nullptr;
            }
            if(weight2)
            {
                delete [] weight2;
                weight2 = nullptr;
            }
            if(bias1)
            {
                delete [] bias1;
                bias1 = nullptr;
            }
            if(bias2)
            {
                delete [] bias2;
                bias2 = nullptr;
            }
            //allocate the weights and biases
            weight1 = new T[inputDim * hiddenDim];
            weight2 = new T[hiddenDim * outputDim];
            bias1 = new  T[hiddenDim];
            bias2 = new  T[outputDim];
            //allocate gradients for the weights and biases 
            if(weight1Grad)
            {
                delete [] weight1Grad;
                weight1Grad = nullptr;
            }
            weight1Grad = new T[inputDim * hiddenDim];  
            if(weight2Grad)
            {
                delete [] weight2Grad;
                weight2Grad = nullptr;
            }
            weight2Grad = new T[hiddenDim * outputDim];
            if(bias1Grad)
            {
                delete [] bias1Grad;
                bias1Grad = nullptr;
            }
            bias1Grad = new T[hiddenDim];
            if(bias2Grad)
            {
                delete [] bias2Grad;
                bias2Grad = nullptr;
            }
            bias2Grad = new T[outputDim];
            //allocate the momentum for the weights and biases
            if(weight1Momentum) {
                delete [] weight1Momentum;
                weight1Momentum = nullptr;
            }

            weight1Momentum = new T[inputDim * hiddenDim];
            if (weight2Momentum) {
                delete [] weight2Momentum;
                weight2Momentum = nullptr;
            }   
            weight2Momentum = new T[hiddenDim * outputDim];
            if (bias1Momentum) {
                delete [] bias1Momentum;
                bias1Momentum = nullptr;
            }   
            bias1Momentum = new T[hiddenDim];
            if (bias2Momentum) {
                delete [] bias2Momentum;
                bias2Momentum = nullptr;
            }   
            bias2Momentum = new T[outputDim];
            //allocate the update for the weights and biases
            if(weight1Update) {
                delete [] weight1Update;
                weight1Update = nullptr;
            }   
            weight1Update = new T[inputDim * hiddenDim];
            if (weight2Update) {
                delete [] weight2Update;
                weight2Update = nullptr;
            }   
            weight2Update = new T[hiddenDim * outputDim];
            if (bias1Update) {
                delete [] bias1Update;
                bias1Update = nullptr;
            }   
            bias1Update = new T[hiddenDim];
            if (bias2Update) {
                delete [] bias2Update;
                bias2Update = nullptr;
            }
            bias2Update = new T[outputDim];

            //allocate the decay for the weights and biases
            if(weight1Decay) {
                delete [] weight1Decay;
                weight1Decay = nullptr;
            }   
            weight1Decay = new T[inputDim * hiddenDim];
            if (weight2Decay) {
                delete [] weight2Decay;
                weight2Decay = nullptr;
            }   
            weight2Decay = new T[hiddenDim * outputDim];
            if (bias1Decay) {
                delete [] bias1Decay;
                bias1Decay = nullptr;
            }       
            bias1Decay = new T[hiddenDim];
            if (bias2Decay) {
                delete [] bias2Decay;
                bias2Decay = nullptr;
            }   
            bias2Decay = new T[outputDim];

            //allocate the sparsity for the weights and biases
            if(weight1Sparsity) {
                delete [] weight1Sparsity;
                weight1Sparsity = nullptr;
            }   
            weight1Sparsity = new T[inputDim * hiddenDim];
            if (weight2Sparsity) {
                delete [] weight2Sparsity;
                weight2Sparsity = nullptr;
            }   
            weight2Sparsity = new T[hiddenDim * outputDim];
            if (bias1Sparsity) {
                delete [] bias1Sparsity;
                bias1Sparsity = nullptr;
            }   
            bias1Sparsity = new T[hiddenDim];
            if (bias2Sparsity) {
                delete [] bias2Sparsity;
                bias2Sparsity = nullptr;
            }   
            bias2Sparsity = new T[outputDim];

            //allocate the sparsity hat for the weights and biases
            if(weight1SparsityHat) {
                delete [] weight1SparsityHat;
                weight1SparsityHat = nullptr;
            }   
            weight1SparsityHat = new T[inputDim * hiddenDim];
            if (weight2SparsityHat) {
                delete [] weight2SparsityHat;
                weight2SparsityHat = nullptr;
            }   
            weight2SparsityHat = new T[hiddenDim * outputDim];
            if (bias1SparsityHat) {
                delete [] bias1SparsityHat;
                bias1SparsityHat = nullptr;
            }   
            bias1SparsityHat = new T[hiddenDim];

            if (bias2SparsityHat) {
                delete [] bias2SparsityHat;
                bias2SparsityHat = nullptr;
            }   
            bias2SparsityHat = new T[outputDim];

            //allocate the sparsity grad for the weights and biases
            if(weight1SparsityGrad) {
                delete [] weight1SparsityGrad;
                weight1SparsityGrad = nullptr;
            }   
            weight1SparsityGrad = new T[inputDim * hiddenDim];
            if (weight2SparsityGrad) {
                delete [] weight2SparsityGrad;
                weight2SparsityGrad = nullptr;
            }   
            weight2SparsityGrad = new T[hiddenDim * outputDim];
            if (bias1SparsityGrad) {
                delete [] bias1SparsityGrad;
                bias1SparsityGrad = nullptr;
            }
            bias1SparsityGrad = new T[hiddenDim];
            if (bias2SparsityGrad) {
                delete [] bias2SparsityGrad;
                bias2SparsityGrad = nullptr;
            }
            bias2SparsityGrad = new T[outputDim];

            //allocate the sparsity grad hat for the weights and biases
            if(weight1SparsityGradHat) {
                delete [] weight1SparsityGradHat;
                weight1SparsityGradHat = nullptr;
            }   
            weight1SparsityGradHat = new T[inputDim * hiddenDim];
            if (weight2SparsityGradHat) {
                delete [] weight2SparsityGradHat;
                weight2SparsityGradHat = nullptr;
            }   
            weight2SparsityGradHat = new T[hiddenDim * outputDim];
            if (bias1SparsityGradHat) {
                delete [] bias1SparsityGradHat;
                bias1SparsityGradHat = nullptr;
            }   
            bias1SparsityGradHat = new T[hiddenDim];
            if (bias2SparsityGradHat) {
                delete [] bias2SparsityGradHat;
                bias2SparsityGradHat = nullptr;
            }   
            bias2SparsityGradHat = new T[outputDim];

            //allocate the increment for the weights and biases
            if(weight1Inc) {
                delete [] weight1Inc;
                weight1Inc = nullptr;
            }   
            weight1Inc = new T[inputDim * hiddenDim];
            if (weight2Inc) {
                delete [] weight2Inc;
                weight2Inc = nullptr;
            }   
            weight2Inc = new T[hiddenDim * outputDim];
            if (bias1Inc) {
                delete [] bias1Inc;
                bias1Inc = nullptr;
            }   
            bias1Inc = new T[hiddenDim];
            if (bias2Inc) {
                delete [] bias2Inc;
                bias2Inc = nullptr;
            }   
            bias2Inc = new T[outputDim];

            //allocate the previous gradient for the weights and biases
            if(weight1GradPrev) {
                delete [] weight1GradPrev;
                weight1GradPrev = nullptr;
            }   
            weight1GradPrev = new T[inputDim * hiddenDim];
            if (weight2GradPrev) {
                delete [] weight2GradPrev;
                weight2GradPrev = nullptr;
            }
            weight2GradPrev = new T[hiddenDim * outputDim];
            if (bias1GradPrev) {
                delete [] bias1GradPrev;
                bias1GradPrev = nullptr;
            }
            bias1GradPrev = new T[hiddenDim];
            if (bias2GradPrev) {
                delete [] bias2GradPrev;
                bias2GradPrev = nullptr;
            }
            bias2GradPrev = new T[outputDim];

            //allocate the previous gradient for the weights and biases
            if(weight1GradPrevPrev  != nullptr) {
                delete [] weight1GradPrevPrev;
                weight1GradPrevPrev = nullptr;
            }   
            weight1GradPrevPrev = new T[inputDim * hiddenDim];
            if (weight2GradPrevPrev) {
                delete [] weight2GradPrevPrev;
                weight2GradPrevPrev = nullptr;
            }   
            weight2GradPrevPrev = new T[hiddenDim * outputDim];
            if (bias1GradPrevPrev) {
                delete [] bias1GradPrevPrev;
                bias1GradPrevPrev = nullptr;
            }
            bias1GradPrevPrev = new T[hiddenDim];
            if (bias2GradPrevPrev) {
                delete [] bias2GradPrevPrev;
                bias2GradPrevPrev = nullptr;
            }
            bias2GradPrevPrev = new T[outputDim];

            //done

            size_t total_memory =  outputDim*(inputDim+hiddenDim+1) + hiddenDim*(outputDim+1) + 2*inputDim*hiddenDim + 2*hiddenDim*outputDim + 2*hiddenDim + 2*outputDim;   

            std::cout << "[+] autoencoder total memory allocated: " << std::to_string((total_memory*sizeof(real_t)/1024.0)) << std::endl; 

    }   

    //initialize input
    template <typename T, typename real_x> 
    void auto_encoder<T,real_x>::initializeInput()
    {
        //initialize the input
        input = new T[inputDim];
        if (input == nullptr)
        {
            std::cout << "[-] autoencoder - error in initializeInput - input is null." << std::endl;
            return;
        }
        //done
    }   
    //initialize hidden
    template < typename T, typename real_x> 
    void auto_encoder<T,real_x>::initializeHidden()
    {
        //initialize the hidden
        hidden = new T[hiddenDim];
        if (hidden == nullptr)
        {
            std::cout << "[-] autoencoder - error in initializeHidden - hidden is null." << std::endl;
            return;
        }
        //done
    }   
    //initialize output
    template <typename T, typename real_x>  
    void auto_encoder<T,real_x>::initializeOutput()
    {
        //initialize the output
        output = new T[outputDim];
        if (output == nullptr)
        {
            std::cout << "[-] autoencoder - error in initializeOutput - output is null." << std::endl;
            return;
        }
        //done
    }       


    //copy constructor:
    template <typename T, typename real_x>
    auto_encoder<T,real_x>::auto_encoder(const auto_encoder<T,real_x> &rhs):inputDim(rhs.inputDim),hiddenDim(rhs.hiddenDim),outputDim(rhs.outputDim),
    weight1(nullptr),weight2(nullptr),bias1(nullptr),bias2(nullptr),weight1Grad(nullptr),weight2Grad(nullptr),bias1Grad(nullptr),bias2Grad(nullptr),
    weight1Momentum(nullptr),weight2Momentum(nullptr),bias1Momentum(nullptr),bias2Momentum(nullptr),weight1Update(nullptr),weight2Update(nullptr),
    bias1Update(nullptr),bias2Update(nullptr),weight1Decay(nullptr),weight2Decay(nullptr),bias1Decay(nullptr),bias2Decay(nullptr),
    weight1Sparsity(nullptr),weight2Sparsity(nullptr),bias1Sparsity(nullptr),bias2Sparsity(nullptr),weight1SparsityHat(nullptr),weight2SparsityHat(nullptr),
    bias1SparsityHat(nullptr),bias2SparsityHat(nullptr),weight1SparsityGrad(nullptr),weight2SparsityGrad(nullptr),bias1SparsityGrad(nullptr),bias2SparsityGrad(nullptr),
    weight1SparsityGradHat(nullptr),weight2SparsityGradHat(nullptr),weight1Inc(nullptr),weight2Inc(nullptr),weight1GradPrev(nullptr),weight2GradPrev(nullptr),
    bias1Inc(nullptr),bias2Inc(nullptr),bias1GradPrev(nullptr),bias2GradPrev(nullptr),weight1GradPrevPrev(nullptr),weight2GradPrevPrev(nullptr),
    bias1GradPrevPrev(nullptr),bias2GradPrevPrev(nullptr)
    {
        //std::cout << "auto_encoder copy constructor" << std::endl;
        //copy constructor
        //copy the autoencoder
        copy(rhs);
        //done
    }
    //copy assignment operator
    template <typename T, typename real_x>
    auto_encoder<T,real_x> &auto_encoder<T,real_x>::operator=(const auto_encoder<T,real_x> &rhs)
    {
        //std::cout << "auto_encoder copy assignment operator" << std::endl;
        //copy assignment operator
        //copy the autoencoder
        copy(rhs);
        //return this
        return *this;
        //done
    }   
    //copy activation functions
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copyActivationFunctions(const auto_encoder<T,real_x>& rhs)
    {
        this->activationFunctionPtr = rhs.activationFunctionPtr;
        this->activationGradientFunctionPtr = rhs.activationGradientFunctionPtr;
        this->activationPrimeFunctionPtr = rhs.activationPrimeFunctionPtr;
        this->activationPrimeGradientFunctionPtr = rhs.activationPrimeGradientFunctionPtr;
        this->activationPrimeGradientHatFunctionPtr = rhs.activationPrimeGradientHatFunctionPtr;
        //done  
    }
    //copy parameters :
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copy_parameters(const auto_encoder<T,real_x> &rhs)
    {
        //copy the parameters
        //copy the inputDim
        inputDim = rhs.inputDim;
        //copy the hiddenDim
        hiddenDim = rhs.hiddenDim;
        //copy the outputDim
        outputDim = rhs.outputDim;
        //done
        this->beta = rhs.beta;
         this->total_loss = rhs.total_loss;
        this->ce_loss = rhs.ce_loss;
        this->learningRate = rhs.learningRate;
        this->momentum = rhs.momentum;
  
    }

    //copy
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copy(const auto_encoder<T,real_x> &rhs)
    {   

        //make sure inputdim/outputdim and hidden are equal.
        if (inputDim != rhs.inputDim || outputDim != rhs.outputDim || hiddenDim != rhs.hiddenDim)
        {
            std::cout << "[-] autoencoder - error in copy - inputDim/outputDim/hiddenDim are not equal." << std::endl;
            return;
        }
        copy_parameters(rhs);
        //copy the autoencoder
        //copy the input
        copyInput(rhs);
        //copy the hidden
        copyHidden(rhs);
        //copy the output
        copyOutput(rhs);
        //copy the weights and biases
        copyWeightsAndBiases(rhs);
        //copy the activation function
        copyActivationFunctions(rhs);
        //copy the weight grad
        copyWeightGrad(rhs);
        //done
    }
    //copy weight grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copyWeightGrad(const auto_encoder<T,real_x>& rhs)
    {
        //copy the weight grad
        //copy the weight1 grad
        if (weight1Grad != nullptr)
        {
            delete[] weight1Grad;
            weight1Grad = nullptr;
        }
        weight1Grad = new T[inputDim * hiddenDim];
        if (weight1Grad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightGrad - weight1Grad is null." << std::endl;
            return;
        }
        //copy the weight1 grad
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Grad[i] = rhs.weight1Grad[i];
        }
        //copy the weight2 grad
        if (weight2Grad != nullptr)
        {
            delete[] weight2Grad;
            weight2Grad = nullptr;
        }
        weight2Grad = new T[hiddenDim * outputDim];
        if (weight2Grad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightGrad - weight2Grad is null." << std::endl;
            return;
        }
        //copy the weight2 grad
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Grad[i] = rhs.weight2Grad[i];
        }
        //copy the bias1 grad
        if (bias1Grad != nullptr)
        {
            delete[] bias1Grad;
            bias1Grad = nullptr;
        }
        bias1Grad = new T[hiddenDim];
        if (bias1Grad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightGrad - bias1Grad is null." << std::endl;
            return;
        }
        //copy the bias1 grad
        for(size_t i = 0; i < hiddenDim; i++)
        {
            bias1Grad[i] = rhs.bias1Grad[i];
        }
        //copy the bias2 grad
        if (bias2Grad != nullptr)
        {
            delete[] bias2Grad;
            bias2Grad = nullptr;
        }
        bias2Grad = new T[outputDim];
        if (bias2Grad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightGrad - bias2Grad is null." << std::endl;
            return;
        }
        //copy the bias2 grad
        for(size_t i = 0; i < outputDim; i++)
        {
            bias2Grad[i] = rhs.bias2Grad[i];
        }
        //done

    }
    //copy input
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copyInput(const auto_encoder<T,real_x> &rhs)
    {   
        //copy the input
        if (input != nullptr)
        {
            delete[] input;
            input = nullptr;
        }
        input = new T[inputDim];
        if (input == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyInput - input is null." << std::endl;
            return;
        }
        //copy the input
        for(size_t i = 0; i < inputDim; i++)
        {
            input[i] = rhs.input[i];
        }
        //done
    }
    //copy hidden
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copyHidden(const auto_encoder<T,real_x> &rhs)
    {
        //copy the hidden
        this->hiddenDim = rhs.hiddenDim;
        if (hidden != nullptr)
        {
            delete[] hidden;
            hidden = nullptr;
        }
        hidden = new T[hiddenDim];
        if (hidden == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyHidden - hidden is null." << std::endl;
            return;
        }
        //copy the hidden
        for(size_t i = 0; i < hiddenDim; i++)
        {
            hidden[i] = rhs.hidden[i];
        }   

        //done
    }

    //copy output
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copyOutput(const auto_encoder<T,real_x> &rhs)
    {
        //copy the output
        this->outputDim = rhs.outputDim;
        if (output != nullptr)
        {
            delete[] output;
            output = nullptr;
        }
        output = new T[outputDim];
        if (output == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyOutput - output is null." << std::endl;
            return;
        }
        //copy the output
        for(size_t i = 0; i < outputDim; i++)
        {
            output[i] = rhs.output[i];
        }   
        //done
    }   
    //copy weights and biases
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::copyWeightsAndBiases(const auto_encoder<T,real_x> &rhs)
    {
        //copy the weights and biases
        //copy the weight1
        if(this->weight1 != nullptr)
        {
            delete [] this->weight1;
            this->weight1 = nullptr;
        }
        this->weight1 = new T[inputDim * hiddenDim];
        if(this->weight1 == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1 is null." << std::endl;
            return;
        }
        //copy the weight1
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1[i] = rhs.weight1[i];
        }   

        //copy the weight2
        if (this->weight2 != nullptr)
        {
            delete[] this->weight2;
            this->weight2 = nullptr;
        }
        this->weight2 = new T[hiddenDim * outputDim];
        if (this->weight2 == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2 is null." << std::endl;
            return;
        }
        //copy the weight2
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2[i] = rhs.weight2[i];
        }       
        //copy the bias1
        if (this->bias1 != nullptr)
        {
            delete[] this->bias1;
            this->bias1 = nullptr;
        }       
        this->bias1 = new T[hiddenDim];
        if (this->bias1 == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1 is null." << std::endl;
            return;
        }   
        //copy the bias1    
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1[i] = rhs.bias1[i];
        }       
        if (this->bias2!=nullptr)
        {
            delete[] this->bias2;
            this->bias2 = nullptr;
        }
        this->bias2 = new T[outputDim];
        //copy the bias2
        for(size_t i=0;i<outputDim;i++)
        {
            this->bias2[i] = rhs.bias2[i];
        }   
        
        //copy previous gradient
        if (this->weight1GradPrev != nullptr)
        {
            delete[] this->weight1GradPrev;
            this->weight1GradPrev = nullptr;
        }
        this->weight1GradPrev = new T[inputDim * hiddenDim];
        if (this->weight1GradPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1GradPrev is null." << std::endl;
            return;
        }
        //copy the weight1GradPrev
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1GradPrev[i] = rhs.weight1GradPrev[i];
        }   
        //copy the weight2GradPrev
        if (this->weight2GradPrev != nullptr)
        {
            delete[] this->weight2GradPrev;
            this->weight2GradPrev = nullptr;
        }   
        this->weight2GradPrev = new T[hiddenDim * outputDim];
        if (this->weight2GradPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2GradPrev is null." << std::endl;
            return;
        }   
        //copy the weight2GradPrev  
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2GradPrev[i] = rhs.weight2GradPrev[i];
        }   
        //copy the bias1GradPrev    
        if (this->bias1GradPrev != nullptr)
        {
            delete[] this->bias1GradPrev;
            this->bias1GradPrev = nullptr;
        }       
        this->bias1GradPrev = new T[hiddenDim]; 
        if (this->bias1GradPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1GradPrev is null." << std::endl;
            return;
        }
        //copy the bias1GradPrev
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1GradPrev[i] = rhs.bias1GradPrev[i];
        }
        //copy the bias2GradPrev
        if (this->bias2GradPrev != nullptr)
        {
            delete[] this->bias2GradPrev;
            this->bias2GradPrev = nullptr;
        }
        this->bias2GradPrev = new T[outputDim];
        if (this->bias2GradPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2GradPrev is null." << std::endl;
            return;
        }
        //copy the bias2GradPrev
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2GradPrev[i] = rhs.bias2GradPrev[i];
        }
        //copy the prevprev 
        if (this->weight1GradPrevPrev != nullptr)
        {
            delete[] this->weight1GradPrevPrev;
            this->weight1GradPrevPrev = nullptr;
        }   
        this->weight1GradPrevPrev = new T[inputDim * hiddenDim];
        if (this->weight1GradPrevPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1GradPrevPrev is null." << std::endl;
            return;
        }   
        //copy the weight1GradPrevPrev  
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1GradPrevPrev[i] = rhs.weight1GradPrevPrev[i];
        }   
        //copy the weight2GradPrevPrev  
        if (this->weight2GradPrevPrev != nullptr)
        {
            delete[] this->weight2GradPrevPrev;
            this->weight2GradPrevPrev = nullptr;
        }   
        this->weight2GradPrevPrev = new T[hiddenDim * outputDim];
        if (this->weight2GradPrevPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2GradPrevPrev is null." << std::endl;
            return;
        }
        //copy the weight2GradPrevPrev
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2GradPrevPrev[i] = rhs.weight2GradPrevPrev[i];
        }
        //copy the bias1GradPrevPrev
        if (this->bias1GradPrevPrev != nullptr)
        {
            delete[] this->bias1GradPrevPrev;
            this->bias1GradPrevPrev = nullptr;
        }
        this->bias1GradPrevPrev = new T[hiddenDim];
        if (this->bias1GradPrevPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1GradPrevPrev is null." << std::endl;
            return;
        }

        //copy the bias1GradPrevPrev
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1GradPrevPrev[i] = rhs.bias1GradPrevPrev[i];
        }
        //copy the bias2GradPrevPrev
        if (this->bias2GradPrevPrev != nullptr)
        {
            delete[] this->bias2GradPrevPrev;
            this->bias2GradPrevPrev = nullptr;
        }       
        this->bias2GradPrevPrev = new T[outputDim]; 

        if (this->bias2GradPrevPrev == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2GradPrevPrev is null." << std::endl;
            return;
        }
        //copy the bias2GradPrevPrev
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2GradPrevPrev[i] = rhs.bias2GradPrevPrev[i];
        }
        //copy sparsity
        if (this->weight1Sparsity != nullptr)
        {
            delete[] this->weight1Sparsity;
            this->weight1Sparsity = nullptr;
        }   
        this->weight1Sparsity = new T[inputDim * hiddenDim];
        if (this->weight1Sparsity == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1Sparsity is null." << std::endl;
            return;
        }   
        //copy the weight1Sparsity  
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1Sparsity[i] = rhs.weight1Sparsity[i];
        }   
        //copy the weight2Sparsity  
        if (this->weight2Sparsity != nullptr)
        {
            delete[] this->weight2Sparsity;
            this->weight2Sparsity = nullptr;
        }   
        this->weight2Sparsity = new T[hiddenDim * outputDim];   
        if (this->weight2Sparsity == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2Sparsity is null." << std::endl;
            return;
        }   
        //copy the weight2Sparsity
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2Sparsity[i] = rhs.weight2Sparsity[i];
        }   
        //copy the bias1Sparsity    
        if (this->bias1Sparsity != nullptr)
        {
            delete[] this->bias1Sparsity;
            this->bias1Sparsity = nullptr;
        }   
        this->bias1Sparsity = new T[hiddenDim]; 
        if (this->bias1Sparsity == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1Sparsity is null." << std::endl;
            return;
        }   
        //copy the bias1Sparsity    
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1Sparsity[i] = rhs.bias1Sparsity[i];
        }   
        //copy the bias2Sparsity
        if (this->bias2Sparsity != nullptr)
        {
            delete[] this->bias2Sparsity;
            this->bias2Sparsity = nullptr;
        }           
        this->bias2Sparsity = new T[outputDim]; 
        if (this->bias2Sparsity == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2Sparsity is null." << std::endl;
            return;
        }   
        //copy the bias2Sparsity
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2Sparsity[i] = rhs.bias2Sparsity[i];
        }
        //copy sparsity hat
        if (this->weight1SparsityHat != nullptr)
        {
            delete[] this->weight1SparsityHat;
            this->weight1SparsityHat = nullptr;
        }   
        this->weight1SparsityHat = new T[inputDim * hiddenDim]; 
        if (this->weight1SparsityHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1SparsityHat is null." << std::endl;
            return;
        }   
        //copy the weight1SparsityHat
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1SparsityHat[i] = rhs.weight1SparsityHat[i];
        }   
        //copy the weight2SparsityHat   
        if (this->weight2SparsityHat != nullptr)
        {
            delete[] this->weight2SparsityHat;
            this->weight2SparsityHat = nullptr;
        }   
        this->weight2SparsityHat = new T[hiddenDim * outputDim];
        if (this->weight2SparsityHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2SparsityHat is null." << std::endl;
            return;
        }       
        //copy the weight2SparsityHat   
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2SparsityHat[i] = rhs.weight2SparsityHat[i];
        }   
        //copy the bias1SparsityHat 
        if (this->bias1SparsityHat != nullptr)
        {
            delete[] this->bias1SparsityHat;
            this->bias1SparsityHat = nullptr;
        }   
        this->bias1SparsityHat = new T[hiddenDim];  
        if (this->bias1SparsityHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1SparsityHat is null." << std::endl;
            return;
        }   
        //copy the bias1SparsityHat
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1SparsityHat[i] = rhs.bias1SparsityHat[i];
        }   
        //copy the bias2SparsityHat
        if (this->bias2SparsityHat != nullptr)
        {
            delete[] this->bias2SparsityHat;
            this->bias2SparsityHat = nullptr;
        }
        this->bias2SparsityHat = new T[outputDim];
        if (this->bias2SparsityHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2SparsityHat is null." << std::endl;
            return;
        }
        //copy the bias2SparsityHat
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2SparsityHat[i] = rhs.bias2SparsityHat[i];
        }
        //copy sparsity grad
        if (this->weight1SparsityGrad != nullptr)
        {
            delete[] this->weight1SparsityGrad;
            this->weight1SparsityGrad = nullptr;
        }
        this->weight1SparsityGrad = new T[inputDim * hiddenDim];
        if (this->weight1SparsityGrad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1SparsityGrad is null." << std::endl;
            return;
        }


        //copy the weight1SparsityGrad
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1SparsityGrad[i] = rhs.weight1SparsityGrad[i];
        }   
        //copy the weight2SparsityGrad
        if (this->weight2SparsityGrad != nullptr)
        {
            delete[] this->weight2SparsityGrad;
            this->weight2SparsityGrad = nullptr;
        }       
        this->weight2SparsityGrad = new T[hiddenDim * outputDim];   
        if (this->weight2SparsityGrad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2SparsityGrad is null." << std::endl;
            return;
        }   
        //copy the weight2SparsityGrad  
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2SparsityGrad[i] = rhs.weight2SparsityGrad[i];
        }   
        //copy the bias1SparsityGrad    
        if (this->bias1SparsityGrad != nullptr)
        {
            delete[] this->bias1SparsityGrad;
            this->bias1SparsityGrad = nullptr;
        }   
        this->bias1SparsityGrad = new T[hiddenDim]; 
        if (this->bias1SparsityGrad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1SparsityGrad is null." << std::endl;
            return;
        }   
        //copy the bias1SparsityGrad
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1SparsityGrad[i] = rhs.bias1SparsityGrad[i];
        }
        //copy the bias2SparsityGrad
        if (this->bias2SparsityGrad != nullptr)
        {
            delete[] this->bias2SparsityGrad;
            this->bias2SparsityGrad = nullptr;
        }
        this->bias2SparsityGrad = new T[outputDim];
        if (this->bias2SparsityGrad == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2SparsityGrad is null." << std::endl;
            return;
        }
        //copy the bias2SparsityGrad
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2SparsityGrad[i] = rhs.bias2SparsityGrad[i];
        }
        //copy sparsity grad hat
        if (this->weight1SparsityGradHat != nullptr)
        {
            delete[] this->weight1SparsityGradHat;
            this->weight1SparsityGradHat = nullptr;
        }
        this->weight1SparsityGradHat = new T[inputDim * hiddenDim];
        if (this->weight1SparsityGradHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1SparsityGradHat is null." << std::endl;
            return;
        }
        //copy the weight1SparsityGradHat
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1SparsityGradHat[i] = rhs.weight1SparsityGradHat[i];
        }
        //copy the weight2SparsityGradHat
        if (this->weight2SparsityGradHat != nullptr)
        {
            delete[] this->weight2SparsityGradHat;
            this->weight2SparsityGradHat = nullptr;
        }
        this->weight2SparsityGradHat = new T[hiddenDim * outputDim];
        if (this->weight2SparsityGradHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2SparsityGradHat is null." << std::endl;
            return;
        }
        //copy the weight2SparsityGradHat
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2SparsityGradHat[i] = rhs.weight2SparsityGradHat[i];
        }
        //copy the bias1SparsityGradHat
        if (this->bias1SparsityGradHat != nullptr)
        {
            delete[] this->bias1SparsityGradHat;
            this->bias1SparsityGradHat = nullptr;
        }
        this->bias1SparsityGradHat = new T[hiddenDim];
        if (this->bias1SparsityGradHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1SparsityGradHat is null." << std::endl;
            return;
        }
        //copy the bias1SparsityGradHat
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1SparsityGradHat[i] = rhs.bias1SparsityGradHat[i];
        }
        //copy the bias2SparsityGradHat
        if (this->bias2SparsityGradHat != nullptr)
        {
            delete[] this->bias2SparsityGradHat;
            this->bias2SparsityGradHat = nullptr;
        }
        this->bias2SparsityGradHat = new T[outputDim];
        if (this->bias2SparsityGradHat == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2SparsityGradHat is null." << std::endl;
            return;
        }
        //copy the bias2SparsityGradHat
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2SparsityGradHat[i] = rhs.bias2SparsityGradHat[i];
        }
        //copy the weight1Inc
        if (this->weight1Inc != nullptr)
        {
            delete[] this->weight1Inc;
            this->weight1Inc = nullptr;
        }
        this->weight1Inc = new T[inputDim * hiddenDim];
        if (this->weight1Inc == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1Inc is null." << std::endl;
            return;
        }
        //copy the weight1Inc
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1Inc[i] = rhs.weight1Inc[i];
        }
        //copy the weight2Inc
        if (this->weight2Inc != nullptr)
        {
            delete[] this->weight2Inc;
            this->weight2Inc = nullptr;
        }
        this->weight2Inc = new T[hiddenDim * outputDim];
        if (this->weight2Inc == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2Inc is null." << std::endl;
            return;
        }
        //copy the weight2Inc
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2Inc[i] = rhs.weight2Inc[i];
        }
        //copy the bias1Inc
        if (this->bias1Inc != nullptr)
        {
            delete[] this->bias1Inc;
            this->bias1Inc = nullptr;
        }
        this->bias1Inc = new T[hiddenDim];
        if (this->bias1Inc == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1Inc is null." << std::endl;
            return;
        }
        //copy the bias1Inc
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1Inc[i] = rhs.bias1Inc[i];
        }
        //copy the bias2Inc
        if (this->bias2Inc != nullptr)
        {
            delete[] this->bias2Inc;
            this->bias2Inc = nullptr;
        }
        this->bias2Inc = new T[outputDim];

        if (this->bias2Inc == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2Inc is null." << std::endl;
            return;
        }
        //copy the bias2Inc
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2Inc[i] = rhs.bias2Inc[i];
        }
        //copy the weight1Update
        if (this->weight1Update != nullptr)
        {
            delete[] this->weight1Update;
            this->weight1Update = nullptr;
        }
        this->weight1Update = new T[inputDim * hiddenDim];
        if (this->weight1Update == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1Update is null." << std::endl;
            return;
        }
        //copy the weight1Update
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1Update[i] = rhs.weight1Update[i];
        }
        //copy the weight2Update
        if (this->weight2Update != nullptr)
        {
            delete[] this->weight2Update;
            this->weight2Update = nullptr;
        }
        this->weight2Update = new T[hiddenDim * outputDim];
        if (this->weight2Update == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2Update is null." << std::endl;
            return;
        }   
        //copy the weight2Update
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2Update[i] = rhs.weight2Update[i];
        }   
        //copy the bias1Update
        if (this->bias1Update != nullptr)
        {
            delete[] this->bias1Update;
            this->bias1Update = nullptr;
        }   
        this->bias1Update = new T[hiddenDim];
        if (this->bias1Update == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1Update is null." << std::endl;
            return;
        }
        //copy the bias1Update
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1Update[i] = rhs.bias1Update[i];
        }
        //copy the bias2Update
        if (this->bias2Update != nullptr)
        {
            delete[] this->bias2Update;
            this->bias2Update = nullptr;
        }
        this->bias2Update = new T[outputDim];
        if (this->bias2Update == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2Update is null." << std::endl;
            return;
        }
        //copy the bias2Update
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2Update[i] = rhs.bias2Update[i];
        }
        //copy the weight1Decay
        if (this->weight1Decay != nullptr)
        {
            delete[] this->weight1Decay;
            this->weight1Decay = nullptr;
        }
        this->weight1Decay = new T[inputDim * hiddenDim];
        if (this->weight1Decay == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight1Decay is null." << std::endl;
            return;
        }
        //copy the weight1Decay
        for(size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            this->weight1Decay[i] = rhs.weight1Decay[i];
        }
        //copy the weight2Decay
        if (this->weight2Decay != nullptr)
        {
            delete[] this->weight2Decay;
            this->weight2Decay = nullptr;
        }
        this->weight2Decay = new T[hiddenDim * outputDim];
        if (this->weight2Decay == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - weight2Decay is null." << std::endl;
            return;
        }
        //copy the weight2Decay
        for(size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            this->weight2Decay[i] = rhs.weight2Decay[i];
        }
        //copy the bias1Decay
        if (this->bias1Decay != nullptr)
        {
            delete[] this->bias1Decay;
            this->bias1Decay = nullptr;
        }
        this->bias1Decay = new T[hiddenDim];
        if (this->bias1Decay == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias1Decay is null." << std::endl;
            return;
        }
        //copy the bias1Decay
        for(size_t i = 0; i < hiddenDim; i++)
        {
            this->bias1Decay[i] = rhs.bias1Decay[i];
        }
        //copy the bias2Decay
        if (this->bias2Decay != nullptr)
        {
            delete[] this->bias2Decay;
            this->bias2Decay = nullptr;
        }           
        this->bias2Decay = new T[outputDim];    
        if (this->bias2Decay == nullptr)
        {
            std::cout << "[-] autoencoder - error in copyWeightsAndBiases - bias2Decay is null." << std::endl;
            return;
        }   
        //copy the bias2Decay
        for(size_t i = 0; i < outputDim; i++)
        {
            this->bias2Decay[i] = rhs.bias2Decay[i];
        }
        
        //done

        
        //done

    }   








    //initialize weight


    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeWeight()
    {
        
        //initialize the weight
        

        initializeWeight(weight1, inputDim, hiddenDim);
        initializeWeight(weight2, hiddenDim, outputDim);
        initializeWeight(weight1Grad, inputDim, hiddenDim);
        initializeWeight(weight2Grad, hiddenDim, outputDim);
        initializeWeight(weight1Momentum, inputDim, hiddenDim); 
        initializeWeight(weight2Momentum, hiddenDim, outputDim);

        initializeWeight(weight1Update, inputDim, hiddenDim);
        initializeWeight(weight2Update, hiddenDim, outputDim);
        initializeWeight(weight1Decay, inputDim, hiddenDim);
        initializeWeight(weight2Decay, hiddenDim, outputDim);
        initializeWeight(weight1Sparsity, inputDim, hiddenDim);
        initializeWeight(weight2Sparsity, hiddenDim, outputDim);
        initializeWeight(weight1SparsityHat, inputDim, hiddenDim);
        initializeWeight(weight2SparsityHat, hiddenDim, outputDim);
        initializeWeight(weight1SparsityGrad, inputDim, hiddenDim);
        initializeWeight(weight2SparsityGrad, hiddenDim, outputDim);
        initializeWeight(weight1SparsityGradHat, inputDim, hiddenDim);
        initializeWeight(weight2SparsityGradHat, hiddenDim, outputDim);
        initializeWeight(weight1Inc, inputDim, hiddenDim);
        initializeWeight(weight2Inc, hiddenDim, outputDim);
        initializeWeight(weight1GradPrev, inputDim, hiddenDim);
        initializeWeight(weight2GradPrev, hiddenDim, outputDim);

        initializeWeight(weight1GradPrevPrev, inputDim, hiddenDim);
        initializeWeight(weight2GradPrevPrev, hiddenDim, outputDim);

        //done


    }

    //initialize bias
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeBias()
    {
        initializeBias(input, inputDim);
        initializeBias(hidden, hiddenDim);
        initializeBias(output, outputDim);

        initializeBias(bias1, hiddenDim);
        initializeBias(bias2, outputDim);
        initializeBias(bias1Grad, hiddenDim);
        initializeBias(bias2Grad, outputDim);
        initializeBias(bias1Momentum, hiddenDim);
        initializeBias(bias2Momentum, outputDim);
        initializeBias(bias1Update, hiddenDim);
        initializeBias(bias2Update, outputDim);
        initializeBias(bias1Decay, hiddenDim);
        initializeBias(bias2Decay, outputDim);
        initializeBias(bias1Sparsity, hiddenDim);
        initializeBias(bias2Sparsity, outputDim);
        initializeBias(bias1SparsityHat, hiddenDim);
        initializeBias(bias2SparsityHat, outputDim);
        initializeBias(bias1SparsityGrad, hiddenDim);
        initializeBias(bias2SparsityGrad, outputDim);
        initializeBias(bias1SparsityGradHat, hiddenDim);
        initializeBias(bias2SparsityGradHat, outputDim);
        initializeBias(bias1Inc, hiddenDim);
        initializeBias(bias2Inc, outputDim);
        initializeBias(bias1GradPrev, hiddenDim);
        initializeBias(bias2GradPrev, outputDim);
        initializeBias(bias1GradPrevPrev, hiddenDim);
        initializeBias(bias2GradPrevPrev, outputDim);

        //done

    }
    //initialize activation function
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeWeight(T *weight, size_t size)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (size_t i = 0; i < size; i++)
        {
            weight[i] = dis(gen);
        }
        //done
    }
    //initialize weight grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeBias(T *bias, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            bias[i] = T(0.0);
            
        }

        //done

    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeWeight(T *weight, size_t row, size_t col)
    {
        static std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0., 1.0);
        for (size_t i = 0; i < row; i++)
        {
            for (size_t j = 0; j < col; j++)
            {
                weight[i * col + j] = dis(gen);
            }
        
        }//for
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::train(T *input, T *output, size_t size)
    {
        if ( input==nullptr || output==nullptr)
        {
            std::cout << "[-] autoencoder - error in training - no input." << std::endl;
            return;
        }
        //std::cout<<"[+]DEBUG  autoencoder - training - feedforward"<<std::endl;
        feedforward(input, output, size);
        //std::cout<<"[+]DEBUG  autoencoder - training - backprop"<<std::endl;
        backprop(input, output, size);
        //std::cout<<"[+]DEBUG  autoencoder - training - update"<<std::endl; 
        update();
        //std::cout<<"[+]DEBUG  autoencoder - training - done"<<std::endl;

        //done
    }

    //train with matrix and fill classdist 
    template <typename T, typename real_x> 
    void auto_encoder<T,real_x>::train( matrix<T>& input , class_dist& output)
    {
        output.setup(outputDim);//setup output

        T* inputarray  =  new T[input.cols()];
        T* outputarray =  new T[outputDim];
        if ( inputarray==nullptr || outputarray==nullptr)
        {
            std::cout << "error in train matrix" << std::endl;
            return;
        }

        for (size_t i = 0; i < input.rows(); i++)
        {
            for (size_t j = 0; j < input.cols(); j++)
            {
                inputarray[j]=input(i,j);


            }

            train(inputarray,outputarray, 1);
            //update error/loss
           
         }
        
        for (size_t j = 0; j < outputDim && j<output.size(); j++)
        {
                    output.set(j,outputarray[j] );
        }

        if (inputarray!=nullptr)
        {
            delete[] inputarray;
            inputarray=nullptr;
        }   
        if (outputarray!=nullptr)
        {
            delete[] outputarray;
            outputarray=nullptr;
        }
 
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::test(T *input, T *output, size_t size)
    {
        
        feedforward(input, output, size);

        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::test(matrix<T>& input, class_dist& output)
    {
        T* inputarray  =  new T[input.cols()];
        T* outputarray =  new T[outputDim];
        
        if ( inputarray==nullptr || outputarray==nullptr)
        {
            std::cout << "error in test matrix" << std::endl;
            return;
        }

        bzero(outputarray,sizeof(T)*outputDim);
        bzero(inputarray,sizeof(T)*input.cols());
        
        for (size_t i = 0; i < input.rows(); i++)
        {
            for (size_t j = 0; j < input.cols(); j++)
            {
                inputarray[j]=input(i,j);       
            }
            test(inputarray,outputarray,1);
            for (size_t j = 0; j < outputDim && j<output.size(); j++)
            {
                output.accum(j,outputarray[j] / outputDim);
            }   
        }
        if (inputarray!=nullptr)
        {
            delete[] inputarray;
            inputarray=nullptr;
        }
        if (outputarray!=nullptr)
        {
            delete[] outputarray;
            outputarray=nullptr;
        }
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::feedforward(T *input, T *output, size_t size)
    {
        //std::cout << "feedforward" << std::endl;
        //use activation function
        //call forward()

        //assume input is inputDim size and output is size,which is not necessirly
        //outDim size
        if(!input || !output)
        {
            std::cout << "[-] autoencoder - error in feedforward - no input or output." << std::endl;
            return;
        }
        if(!this->input) {
            this->input = new T[inputDim];
        }
        for(size_t i=0;i<inputDim;i++)
        {
            this->input[i]=input[i];
        }
        if(!this->output) {
            this->output = new T[outputDim];
            
        }
        forward();
        //we updated the input and outputs, now we can update the weights and backpropagate
        //the error
        //update weights:

        this->updateWeight1();
        this->updateWeight2();
        this->updateBias1();
        this->updateBias2();
        //conjugategradient 
        //call backprop
        backprop();
        //done 

        //update output

        for(size_t i=0;i<outputDim&&i<size;i++)
        {
            output[i]=this->output[i];
        }

        //std::cout << "[+] autoencoder - feedforward done." << std::endl;

        //done

    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backprop()
    {
        //std::cout << "backprop" << std::endl;
        //use activation function
        //call forward to update input()
        forward();
        // call conjugategradient()
        // then call backward()
         
        this->updateWeight1();
        this->updateWeight2();
        this->updateBias1();
        this->updateBias2();
        //call backward
        backward();
        //done

    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backprop(T *input_param, T *output_param, size_t size)
    {
        real_x loss = 0.0;
        real_x MSE=0.0;
        static const size_t DEBUG_ITER = 1000;
        static size_t iter = 0;

        //assume input_param is inputDim size and output_param is size,which is not necessirly
        //outDim size
        if(!input_param || !output_param)
        {
            std::cout << "[-] autoencoder - error in backprop - no input or output." << std::endl;
            return;
        }
        if(size < outputDim)
        {
            size = outputDim;
        }
        //assume input_param is inputDim size and output_param is size,which is not necessirly 
        //outDim size

        for(size_t i=0;i<inputDim;i++)
        {
            input[i]=input_param[i];
        }
        for(size_t i=0;i<outputDim;i++)
        {
            output[i]=output_param[i];
        }
        backprop();
      
        //done

        //update error/loss
        for(size_t i=0;i<outputDim&&i<size;i++)
        {
            loss += (output[i] - output_param[i]) * (output[i] - output_param[i]);
        }
        loss /= 2.0;
        MSE = loss / real_t(size);
         //done
        //update output_param:
        for(size_t i=0;i<outputDim&&i<size;i++)
        {
            output_param[i]=output[i];
        }
        if(iter++ % DEBUG_ITER == 0)
        {
            std::cout<<"[+] DEBUG autoencoder - backprop done, MSE="<<MSE<<std::endl; 
        }
        //done

 
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sigmoid(T x)
    {
        return T(1.) / (T(1.) + exp(-x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sigmoidPrime(T x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::tanh(T x)
    {
        return T(exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::tanhPrime(T x)
    {
        return T(1 - tanh(x) * tanh(x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::relu(T x)
    {
        return T(x > 0 ? x : 0);
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::reluPrime(T x)
    {
        return T(x > 0 ? 1 : 0);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::leakyRelu(T x)
    {
        return x > 0 ? x : 0.01 * x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::leakyReluPrime(T x)
    {
        return x > 0 ? 1 : 0.01;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softplus(T x)
    {
        return std::log(1 + std::exp(x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softplusPrime(T x)
    {
        return sigmoid(x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::linear(T x)
    {
        return x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::linearGradient(T x)
    {
        return 1;
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::gaussian(T x)
    {
        return exp(-x * x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::gaussianPrime(T x)
    {
        return -2 * x * exp(-x * x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sinusoid(T x)
    {
        return sin(x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sinusoidPrime(T x)
    {
        return cos(x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sinc(T x)
    {
        return x == 0 ? 1 : cos(x) / x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sincPrime(T x)
    {
        return x == 0 ? 0 : (x * sin(x) - cos(x)) / (x * x);
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::bentIdentity(T x)
    {
        return (sqrt(x * x + 1) - 1) / 2 + x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::bentIdentityPrime(T x)
    {
        return x / (2 * sqrt(x * x + 1)) + 1;
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softExponential(T x)
    {
        return x < 0 ? log(1 + exp(x)) : x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softExponentialPrime(T x)
    {
        return x < 0 ? exp(x) / (1 + exp(x)) : 1;
    }

    template <typename T, typename real_x>
    size_t auto_encoder<T,real_x>::getInputDim() const
    {
        return inputDim;
    }
    template <typename T, typename real_x>
    size_t auto_encoder<T,real_x>::getHiddenDim() const
    {
        return hiddenDim;
    }
    template <typename T, typename real_x>
    size_t auto_encoder<T,real_x>::getOutputDim() const
    {
        return outputDim;
    }

    template <typename T, typename real_x>
    real_x auto_encoder<T,real_x>::getLearningRate() const
    {
        return learningRate;
    }

    template <typename T, typename real_x>
    real_x auto_encoder<T,real_x>::getMomentum() const
    {
        return momentum;
    }

    template <typename T, typename real_x>
    T *auto_encoder<T,real_x>::getBias1() const
    {
        return bias1;
    }
    template <typename T, typename real_x>
    T *auto_encoder<T,real_x>::getBias2() const
    {
        return bias2;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setInputDim(size_t inputDim)
    {
        this->inputDim = inputDim;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setHiddenDim(size_t hiddenDim)
    {
        this->hiddenDim = hiddenDim;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setOutputDim(size_t outputDim)
    {
        this->outputDim = outputDim;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setLearningRate(real_x learningRate)
    {
        this->learningRate = learningRate;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setMomentum(real_x momentum)
    {
        this->momentum = momentum;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight1(T *weight1)
    {
        if (this->weight1 != nullptr)
        {
            delete[] this->weight1;
        }
        this->weight1 = weight1;

        // initialize weight1Inc
        if (weight1Inc != nullptr)
        {
            delete[] weight1Inc;
        }
        weight1Inc = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Inc[i] = T(0.);
        }

        // initialize weight1Grad
        if (weight1Grad != nullptr)
        {
            delete[] weight1Grad;
        }
        weight1Grad = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Grad[i] = T(0.);
        }

        // initialize weight1GradPrev
        if (weight1GradPrev != nullptr)
        {
            delete[] weight1GradPrev;
        }
        weight1GradPrev = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrev[i] = T(0.);
        }
    }

    template <typename T, typename real_x>

    void auto_encoder<T,real_x>::setWeight2(T *weight2)
    {

        if (this->weight2 != nullptr)
        {
            delete[] this->weight2;
        }

        this->weight2 = weight2;

        // initialize weight2Inc
        if (weight2Inc != nullptr)
        {
            delete[] weight2Inc;
        }
        weight2Inc = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Inc[i] = T(0.);
        }

        // initialize weight2Grad
        if (weight2Grad != nullptr)
        {
            delete[] weight2Grad;
        }
        weight2Grad = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Grad[i] = T(0.);
        }

        // initialize weight2GradPrev
        if (weight2GradPrev != nullptr)
        {
            delete[] weight2GradPrev;
        }
        weight2GradPrev = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrev[i] = T(0.);
        }
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias1(T *bias1)
    {
        if (this->bias1 != nullptr)
        {
            delete[] this->bias1;
        }
        this->bias1 = bias1;

        // initialize bias1Inc
        if (bias1Inc != nullptr)
        {
            delete[] bias1Inc;
        }
        bias1Inc = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Inc[i] = T(0.);
        }

        // initialize bias1Grad
        if (bias1Grad != nullptr)
        {
            delete[] bias1Grad;
        }
        bias1Grad = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Grad[i] = T(0.);
        }

        // initialize bias1GradPrev
        if (bias1GradPrev != nullptr)
        {
            delete[] bias1GradPrev;
        }
        bias1GradPrev = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrev[i] = T(0.);
        }
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias2(T *bias2)
    {
        if (this->bias2 != nullptr)
        {
            delete[] this->bias2;
        }
        this->bias2 = bias2;

        // initialize bias2Inc
        if (bias2Inc != nullptr)
        {
            delete[] bias2Inc;
        }
        bias2Inc = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Inc[i] = T(0.);
        }

        // initialize bias2Grad
        if (bias2Grad != nullptr)
        {
            delete[] bias2Grad;
        }
        bias2Grad = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Grad[i] = T(0.);
        }

        // initialize bias2GradPrev
        if (bias2GradPrev != nullptr)
        {
            delete[] bias2GradPrev;
        }
        bias2GradPrev = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrev[i] = T(0.);
        }
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight1Inc(T *weight1Inc)
    {
        if (this->weight1Inc != nullptr)
        {
            delete[] this->weight1Inc;
        }
        this->weight1Inc = weight1Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight2Inc(T *weight2Inc)
    {
        if (this->weight2Inc != nullptr)
        {
            delete[] this->weight2Inc;
        }
        this->weight2Inc = weight2Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias1Inc(T *bias1Inc)
    {
        if (this->bias1Inc != nullptr)
        {
            delete[] this->bias1Inc;
        }
        this->bias1Inc = bias1Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias2Inc(T *bias2Inc)
    {
        if (this->bias2Inc != nullptr)
        {
            delete[] this->bias2Inc;
        }
        this->bias2Inc = bias2Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight1Grad(T *weight1Grad)
    {
        if (this->weight1Grad != nullptr)
        {
            delete[] this->weight1Grad;
        }
        this->weight1Grad = weight1Grad;
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight2Grad(T *weight2Grad)
    {
        if (this->weight2Grad != nullptr)
        {
            delete[] this->weight2Grad;
        }
        this->weight2Grad = weight2Grad;
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias1Grad(T *bias1Grad)
    {
        if (this->bias1Grad != nullptr)
        {
            delete[] this->bias1Grad;
        }
        this->bias1Grad = bias1Grad;
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias2Grad(T *bias2Grad)
    {
        if (this->bias2Grad != nullptr)
        {
            delete[] this->bias2Grad;
        }
        this->bias2Grad = bias2Grad;
    }  


    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::dump(  std::ostream &out) const

    {
        out << std::string("inputDim: ") << std::to_string ( inputDim ) << std::endl;
        out << std::string("hiddenDim: ")<<std::to_string ( hiddenDim )  << std::endl;
        out << std::string("outputDim: ") << std::to_string(outputDim)<< std::endl;
        out << std::string("weight1: ") << std::endl;
 

        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << std::to_string( weight1[i] ) << char(' ');
        }
        out << std::endl;
        out << std::string("weight2: ") << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << std::to_string( weight2[i] ) << char(' ')    ;

        }
        out << std::endl;
        out << "bias1: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out<< std::to_string( bias1[i] ) << char(' ')    ;
        }
        out<< std::endl;
        
        out<< std::string("bias2: ") << std::endl;
        
        for (size_t i = 0; i < outputDim; i++)
        {
            out <<std::to_string( bias2[i] )<< char(' ')    ;

        }
        out << std::endl;

        out << std::string("weight1Inc: ") << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {

            out << std::to_string(weight1Inc[i]) << char(' ')    ;  
        }
        out << std::endl;
        out << std::string("weight2Inc: ") << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << std::to_string(weight2Inc[i] )<< char(' ')    ;
        }
        out << std::endl;
        out << "bias1Inc: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << std::to_string(bias1Inc[i]) << char(' ')    ;
        }
        out << std::endl;
        out << "bias2Inc: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2Inc[i] << " ";
        }
        out << std::endl;



        out << "weight1Grad: " << std::endl;

        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << weight1Grad[i] << " ";
        }
        out << std::endl;
        out << "weight2Grad: " << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << weight2Grad[i] << " ";
        }
        out << std::endl;
        out << "bias1Grad: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << bias1Grad[i] << " ";
        }
        out << std::endl;
        out << "bias2Grad: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2Grad[i] << " ";
        }
        out << std::endl;
        
        out << "weight1GradPrev: " << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << weight1GradPrev[i] << " ";
        }
        out << std::endl;
        out << "weight2GradPrev: " << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << weight2GradPrev[i] << " ";
        }
        out << std::endl;
        out << "bias1GradPrev: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << bias1GradPrev[i] << " ";
        }
        out << std::endl;
        out << "bias2GradPrev: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2GradPrev[i] << " ";
        }
        out << std::endl;



        out << "weight1GradPrevPrev: " << std::endl;    


        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << weight1GradPrevPrev[i] << " ";
        }   
        out << std::endl;

        out << "weight2GradPrevPrev: " << std::endl;            


        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << weight2GradPrevPrev[i] << " ";
        }
        out << std::endl;
        out << "bias1GradPrevPrev: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << bias1GradPrevPrev[i] << " ";
        }


        out << std::endl;
        out << "bias2GradPrevPrev: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2GradPrevPrev[i] << " ";
        }
        out << std::endl;

        //SAVE PARAMETERS (learning rate, momentum, etc.)
        out << "learningRate: " << learningRate << std::endl;
        out << "momentum: " << momentum << std::endl;
        out << "weightDecay: " << weightDecay << std::endl;
        out << "sparsityParam: " << sparsityParam << std::endl;
        out << "beta: " << beta << std::endl;
         //SAVE SPARSITY
        out << "weight1Sparsity: " << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << weight1Sparsity[i] << " ";
        }
        out << std::endl;
        out << "weight2Sparsity: " << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << weight2Sparsity[i] << " ";
        }   
        out << std::endl;
        out << "bias1Sparsity: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << bias1Sparsity[i] << " ";
        }
        out << std::endl;
        
        
        //all members are saved

        
    }
    //load & save 
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::load(std::istream &in)
    {
        std::cout<<"[+]auto_encoder load"<<std::endl;
        clear();
        std::string line;
        std::string name;
        std::string value;
        
        //load at the same order of the dump(ofstream) 
        //load inputDim
        std::getline(in, line);
        std::istringstream iss(line);
        iss >> name >> value;
        inputDim = std::stoi(value);
        //load hiddenDim
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name >> value;
        hiddenDim = std::stoi(value);
        //load outputDim
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name >> value;
        outputDim = std::stoi(value);
        //load weight1
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight1 = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            iss >> value;
            weight1[i] = std::stod(value);
        }   
        //load weight2
        std::getline(in, line); 
        iss = std::istringstream(line);
        iss >> name;
        weight2 = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            iss >> value;
            weight2[i] = std::stod(value);
        }   
        //load bias1
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias1 = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            iss >> value;
            bias1[i] = std::stod(value);
        }   
        //load bias2
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias2 = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            iss >> value;
            bias2[i] = std::stod(value);
        }
        //load weight1Inc
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight1Inc = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            iss >> value;
            weight1Inc[i] = std::stod(value);
        }
        //load weight2Inc
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight2Inc = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            iss >> value;
            weight2Inc[i] = std::stod(value);
        }   
        //load bias1Inc
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias1Inc = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            iss >> value;
            bias1Inc[i] = std::stod(value);
        }
        //load bias2Inc
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias2Inc = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            iss >> value;
            bias2Inc[i] = std::stod(value);
        }   
        //load weight1Grad
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight1Grad = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            iss >> value;
            weight1Grad[i] = std::stod(value);
        }       
        //load weight2Grad  
        std::getline(in, line);     
        iss = std::istringstream(line);
        iss >> name;
        weight2Grad = new T[hiddenDim * outputDim]; 
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            iss >> value;
            weight2Grad[i] = std::stod(value);
        }   
        //load bias1Grad
        std::getline(in, line); 
        iss = std::istringstream(line);
        iss >> name;
        bias1Grad = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            iss >> value;
            bias1Grad[i] = std::stod(value);
        }
        //load bias2Grad
        std::getline(in, line);
        iss = std::istringstream(line); 
        iss >> name;
        bias2Grad = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            iss >> value;
            bias2Grad[i] = std::stod(value);
        }   
        //load weight1GradPrev
        std::getline(in, line); 
        iss = std::istringstream(line);
        iss >> name;
        weight1GradPrev = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            iss >> value;
            weight1GradPrev[i] = std::stod(value);
        }
        //load weight2GradPrev
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight2GradPrev = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            iss >> value;
            weight2GradPrev[i] = std::stod(value);
        }
        //load bias1GradPrev
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias1GradPrev = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            iss >> value;
            bias1GradPrev[i] = std::stod(value);
        }   
        //load bias2GradPrev
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias2GradPrev = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            iss >> value;
            bias2GradPrev[i] = std::stod(value);
        }   
        //load weight1GradPrevPrev
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight1GradPrevPrev = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            iss >> value;
            weight1GradPrevPrev[i] = std::stod(value);
        }   
        //load weight2GradPrevPrev
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight2GradPrevPrev = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            iss >> value;
            weight2GradPrevPrev[i] = std::stod(value);
        }   
        //load bias1GradPrevPrev
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias1GradPrevPrev = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            iss >> value;
            bias1GradPrevPrev[i] = std::stod(value);
        }
        //load bias2GradPrevPrev
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias2GradPrevPrev = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            iss >> value;
            bias2GradPrevPrev[i] = std::stod(value);
        }
        //load learningRate
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name >> value;
        learningRate = std::stod(value);
        //load momentum
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name >> value;
        momentum = std::stod(value);
        //load weightDecay
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name >> value;
        weightDecay = std::stod(value);
        //load sparsityParam
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name >> value;
        sparsityParam = std::stod(value);
        //load beta
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name >> value;
        beta = std::stod(value);
        //load weight1Sparsity
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        weight1Sparsity = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            iss >> value;
            weight1Sparsity[i] = std::stod(value);
        }   
        //load weight2Sparsity
        std::getline(in, line); 
        iss = std::istringstream(line);
        iss >> name;
        weight2Sparsity = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            iss >> value;
            weight2Sparsity[i] = std::stod(value);
        }   
        //load bias1Sparsity
        std::getline(in, line);
        iss = std::istringstream(line);
        iss >> name;
        bias1Sparsity = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            iss >> value;
            bias1Sparsity[i] = std::stod(value);
        }
        std::cout<<"[-]auto_encoder load"<<std::endl;


    }





    template<typename T, typename real_x>
    void auto_encoder<T,real_x>::load(std::string filename)
    {
        std::ifstream in(filename);
        load(in);
        in.close();

    }
    template<typename T, typename real_x>
    void  auto_encoder<T,real_x>::save(std::string filename)
    {
        try
        {
        //delete file if exists 
        std::remove(filename.c_str()); 
        //create file
        std::ofstream out(filename , std::ios::out   | std::ios::trunc);
        dump(out);
        out.close();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }



    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeActivationFunction()
    {
        XactivationFunctionPtr f(&auto_encoder<T,real_x>::sigmoid);
        XactivationFunctionPtr fPrime(&auto_encoder<T,real_x>::sigmoidPrime);

        activationFunctionPtr = f;
        activationPrimeFunctionPtr = fPrime;
        activationGradientFunctionPtr = fPrime;
        activationPrimeGradientFunctionPtr = f;
        activationPrimeGradientHatFunctionPtr = fPrime;
 
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::forward(const matrix<T>& input )
    {
        //std::cout << "auto_encoder forward" << std::endl;
        //check input size  :
        //update input dimensions
        for( size_t i=0 ; i < input.size1(); i++  ) {
            //update input from matrix to array
            for ( size_t j=0; j< input.size2(); j++ )
                this->input[(i*input.size2()+j)%inputDim] = input(i,j);
            //forward updates hidden and output.
            forward();

         }
        //update everything else
        //std::cout << "auto_encoder forward end" << std::endl;
        
       
    }
    //forward: 
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::forward(T*& in)
    {
        //copy input to this->input
        for (size_t i = 0; i < inputDim; i++)
        {
            this->input[i] = in[i];
        }   
        //forward updates hidden and output.
        forward();
        //done
    }
    //forward: 
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::forward()
    {
        //std::cout << "auto_encoder forward" << std::endl;
        //update hidden dimensions

        //update previous weight1GradPrevPrev 
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrevPrev[i] = weight1GradPrev[i];
        }
        //update previous weight2GradPrevPrev
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrevPrev[i] = weight2GradPrev[i];
        }
        //update previous bias1GradPrevPrev
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrevPrev[i] = bias1GradPrev[i];
        }
        //update previous bias2GradPrevPrev
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrevPrev[i] = bias2GradPrev[i];
        }
        //update previous weight1GradPrev
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrev[i] = weight1Grad[i];
        }
        //update previous weight2GradPrev
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrev[i] = weight2Grad[i];
        }   
        //update previous bias1GradPrev
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrev[i] = bias1Grad[i];
        }
        //update previous bias2GradPrev
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrev[i] = bias2Grad[i];
        }
        
        //if not using sparsity constraint 
        


        for (size_t i = 0; i < hiddenDim; i++)
        {
            hidden[i] = 0;
            for (size_t j = 0; j < inputDim; j++)
            {
                hidden[i] += input[j] * weight1[j * hiddenDim + i];
            }
            hidden[i] = (this->*activationFunctionPtr)(hidden[i]);
        }
        //update output dimensions
        for (size_t i = 0; i < outputDim; i++)
        {
            output[i] = 0;
            for (size_t j = 0; j < hiddenDim; j++)
            {
                output[i] += hidden[j] * weight2[j * outputDim + i];
            }
            output[i] = (this->*activationFunctionPtr)(output[i]);
        }
        //update everything else 

        //std::cout << "auto_encoder forward end" << std::endl; 
        


    }  
    //backward:
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backward(const matrix<T>& input,  matrix<T>& output,const matrix<T>& target,matrix<T>& grad)
    {
        //update input from matrix to array
        output.resize(input.size1(),1);
        grad.resize(input.size1(),1);
        for( size_t i=0 ; i < input.size1(); i++  )
        {   for ( size_t j=0; j< input.size2(); j++ )
            {
                                this->input[(i*input.size2()+j)%inputDim] = input(i,j); 
                                output(i,0) = 0.0;
                            grad(i,0) = 0.0;

            }
            //input is updated: call backward()
            backward(); 
            //update output
            for (size_t i = 0; i < outputDim; i++)
            {
                output(i,0)  += this->output[i] + target(i,0) * (this->output[i] - input(i,0)) * (this->*activationPrimeFunctionPtr)(this->output[i]) * (this->*activationPrimeFunctionPtr)(this->output[i])   ;

            }
            //update grad
            for (size_t i = 0; i < inputDim; i++)
            {
                grad(i,0) = this->input[i];
            }


        } //end for
        //update output
        for (size_t i = 0; i < outputDim; i++)
        {
            output(i,0) /= input.size1();
        }   
        //update grad
        for (size_t i = 0; i < inputDim; i++)
        {
            grad(i,0) /= input.size1();
        }   

        //update everything else


    }
    //backward with a single case :
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backward(  T*& input, T*& output )
    {
        for (size_t i = 0; i < inputDim; i++)
        {
            this->input[i] = input[i];
        }
        //input is updated: call backward()
        backward();
        //update output
        for (size_t i = 0; i < outputDim; i++)
        {
            output[i] = this->output[i];
        }
        //update everything else
        update();
        

        
    }
    //backward with input and target :
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backward(const matrix<T>& input,const matrix<T>& target)
    {
        //std::cout << "auto_encoder backward" << std::endl;
        //update weight2Grad
        //update input from matrix to array
        for( size_t i=0 ; i < input.size1(); i++  )
        {
            for ( size_t j=0; j< input.size2(); j++ )
            {
                    this->input[(i*input.size2()+j)%inputDim] = input(i,j);
            }
            for ( size_t j=0; j< target.size2(); j++ )
            {
                    this->output[(i*target.size2()+j)%outputDim] = target(i,j);
            }

            //input is updated: call backward()
            backward();
            //update weight2Grad
            for (size_t i = 0; i < outputDim; i++)
            {
                for (size_t j = 0; j < hiddenDim; j++)
                {
                    weight2Grad[j * outputDim + i] += hidden[j] * (this->output[i] - this->input[i]) * (this->*activationPrimeFunctionPtr)(this->output[i]);
                }
            }   

            //update bias2Grad
            for (size_t i = 0; i < outputDim; i++)
            {
                bias2Grad[i] += (this->output[i] - this->input[i]) * (this->*activationPrimeFunctionPtr)(output[i]);
            }
            //update weight1Grad   :
            for (size_t i = 0; i < hiddenDim; i++)
            {
                for (size_t j = 0; j < inputDim; j++)
                {
                    T sum = 0;
                    for (size_t k = 0; k < outputDim; k++)
                    {
                        sum += (output[k] - this->input[k]) * (this->*activationPrimeFunctionPtr)(output[k]) * weight2[i * outputDim + k];
                    }
                    weight1Grad[j * hiddenDim + i] += this->input[j] * sum * (this->*activationPrimeFunctionPtr)(hidden[i]);
                }
            }
            //update bias1Grad
            for (size_t i = 0; i < hiddenDim; i++)
            {
                T sum = 0;
                for (size_t j = 0; j < outputDim; j++)
                {
                    sum += (output[j] - this->input[j]) * (this->*activationPrimeFunctionPtr)(output[j]) * weight2[i * outputDim + j];
                }
                bias1Grad[i] += sum * (this->*activationPrimeFunctionPtr)(hidden[i]);
            }
            //update everything else :

            //std::cout << "auto_encoder backward end" << std::endl;

        }
            
        
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backward()
    {
       
        //std::cout << "auto_encoder backward" << std::endl;
        //update hidden dimensions
        for(size_t i=0;i<hiddenDim;i++)
        {
            hidden[i] = 0;
            for (size_t j = 0; j < inputDim; j++)
            {
                hidden[i] += input[j] * weight1[j * hiddenDim + i];
            }
            hidden[i] = (this->*activationFunctionPtr)(hidden[i]);
         }
       //update weights,gradients and biases:
     
        //debug 
       // std::cout<<"[+] DEBUG : auto_encoder backward  hidden done. updating gradients"<<std::endl; 
       // update();
       // std::cout<<"[+] DEBUG : auto_encoder backward  gradients done"<<std::endl; 
        //

    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::update()
    {
        //std::cout << "auto_encoder update" << std::endl;
        //update weight1
     //weight1 is nan:
        if(weight1Prev==NULL)
        initializeWeight1Prev();
        if(weight1GradPrev==NULL)
        initializeWeight1GradPrev();
        if(weight1GradPrevPrev==NULL)
        initializeWeight1GradPrevPrev();
        if(weight1Inc==NULL)
        initializeWeight1Inc();
        if(weight1Sparsity==NULL)
        initializeWeight1Sparsity();
        if(weight1Grad==NULL)
        initializeWeight1Grad();
        if(weight1GradPrev==NULL)
        initializeWeight1GradPrev();
        
        if(weight2Prev==NULL)
        initializeWeight2Prev();
        if(weight2GradPrev==NULL)
        initializeWeight2GradPrev();
        if(weight2GradPrevPrev==NULL)
        initializeWeight2GradPrevPrev();
        if(weight2Inc==NULL)
        initializeWeight2Inc();
        if(weight2Sparsity==NULL)
        initializeWeight2Sparsity();
        if(weight2Grad==NULL)
        initializeWeight2Grad();

        if(weight2GradPrev==NULL)
        initializeWeight2GradPrev();
        if(bias1Prev==NULL)
        initializeBias1Prev();
        if(bias1GradPrev==NULL)
        initializeBias1GradPrev();
        if(bias1GradPrevPrev==NULL)
        initializeBias1GradPrevPrev();
        if(bias1Inc==NULL)
        initializeBias1Inc();
        if(bias1Sparsity==NULL)
        initializeBias1Sparsity();
        if(bias1Grad==NULL)
        initializeBias1Grad();
        if(bias1GradPrev==NULL)
        initializeBias1GradPrev();
        if(bias2Prev==NULL)
        initializeBias2Prev();
        if(bias2GradPrev==NULL)
        initializeBias2GradPrev();
        if(bias2GradPrevPrev==NULL)
        initializeBias2GradPrevPrev();
        if(bias2Inc==NULL)
        initializeBias2Inc();
        if(bias2Sparsity==NULL)
        initializeBias2Sparsity();
        if(bias2Grad==NULL)
        initializeBias2Grad();

        //update weight2GradPrev
        updateWeight2GradPrev();
        
        //update bias1GradPrev
        updateBias1GradPrev();
        
        //update bias2GradPrev
        updateBias2GradPrev();
        
        
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            if(weight1[i]==weight1[i])
                if(weight1Grad[i]==weight1Grad[i])
                    weight1[i] -= learningRate * weight1Grad[i];
                else
                {
                    //weight1Grad is nan:
                    weight1Grad[i]=0.01*weight1[i];
                    weight1[i] -= learningRate * weight1Grad[i];

                }
            else
            {
                //weight1 is nan:
                if(weight1Grad[i]==weight1Grad[i])
                    weight1[i]=0.01*weight1Grad[i];
                else
                {
                    //weight1 and weight1Grad are nan:
                    weight1[i]=hidden[i%hiddenDim]*input[i%inputDim]*0.01;
                    weight1Grad[i]=0.01;
                }   
                

            }   
        }
        //update weight2
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {   
            if(weight2[i]==weight2[i]&&weight2Grad[i]==weight2Grad[i])
                weight2[i] -= learningRate * weight2Grad[i];
            else
            {
                //weight2 or weight2Grad is nan:
                if(weight2[i]==weight2[i])
                    weight2Grad[i]=0.01*weight2[i];
                else
                {
                    //weight2 and weight2Grad are nan:
                    weight2[i]=hidden[i%hiddenDim]*output[i%outputDim]*0.01;
                    weight2Grad[i]=0.01;
                }   
            }   
        }
        //update bias1
        for (size_t i = 0; i < hiddenDim; i++)
        {
            if(bias1[i]==bias1[i]&&bias1Grad[i]==bias1Grad[i])
                bias1[i] -= learningRate * bias1Grad[i];
            else
            {
                //bias1 or bias1Grad is nan:
                if(bias1[i]==bias1[i])
                    bias1Grad[i]=0.01*bias1[i];
                else
                {
                    //bias1 and bias1Grad are nan:
                    bias1[i]=hidden[i%hiddenDim]*0.01;
                    bias1Grad[i]=0.01;
                }   
            }   
        }
         
        //update bias2
        for (size_t i = 0; i < outputDim; i++)
        {
            if(bias2[i]==bias2[i]&&bias2Grad[i]==bias2Grad[i])
                bias2[i] -= learningRate * bias2Grad[i];
            else
            {
                //bias2 or bias2Grad is nan:
                if(bias2[i]==bias2[i])
                    bias2Grad[i]=0.01*bias2[i];
                else
                {
                    //bias2 and bias2Grad are nan:
                    bias2[i]=output[i%outputDim]*0.01;
                    bias2Grad[i]=0.01;
                }   
            }   
            
        }

        //update weight1GradPrevPrev
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrevPrev[i] = weight1GradPrev[i];
        }
        //update weight2GradPrevPrev
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrevPrev[i] = weight2GradPrev[i];
        }
        //update bias1GradPrevPrev
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrevPrev[i] = bias1GradPrev[i];
        }
        //update bias2GradPrevPrev
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrevPrev[i] = bias2GradPrev[i];
        }
        //update weight1GradPrev
        updateWeight1GradPrev();
        //update weight2GradPrev
        updateWeight2GradPrev();
        
        //update bias1GradPrev
        updateBias1GradPrev();
        
        //update bias2GradPrev
        updateBias2GradPrev();
        
        //update weight1Inc

        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            if(weight1Inc==weight1Inc)
                weight1Inc[i] = momentum * weight1Inc[i] + learningRate * weight1Grad[i];
            else
            {
                //weight1Inc is nan:
                weight1Inc[i]=0.01*weight1Grad[i];
            }

            weight1Inc[i] = momentum * weight1Inc[i] + learningRate * weight1Grad[i];
        }
        //update weight2Inc
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            if(weight2Inc==weight2Inc)
                weight2Inc[i] = momentum * weight2Inc[i] + learningRate * weight2Grad[i];
            else
            {
                //weight2Inc is nan:
                weight2Inc[i]=0.01*weight2Grad[i];
            }

        }
        //update bias1Inc
        
        for (size_t i = 0; i < hiddenDim; i++)
        {
            if(bias1Inc==bias1Inc)
                bias1Inc[i] = momentum * bias1Inc[i] + learningRate * bias1Grad[i];
            else
            {
                //bias1Inc is nan:
                bias1Inc[i]=0.01*bias1Grad[i];
            }
         }
        //update bias2Inc
        for (size_t i = 0; i < outputDim; i++)
        {
          
            if(bias2Inc==bias2Inc)
                bias2Inc[i] = momentum * bias2Inc[i] + learningRate * bias2Grad[i];
            else
            {
                //bias2Inc is nan:
                bias2Inc[i]=0.01*bias2Grad[i];
            }
         }
        //update weight1
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            if(weight1==weight1) {
                if(weight1Prev==NULL)
                    initializeWeight1Prev();

                weight1Prev[i] = weight1[i];

                weight1[i] -= weight1Inc[i];

            }
            else
            {
           

                weight1Prev[i]=0.01*weight1Inc[i];  
                weight1[i] -= weight1Inc[i];
                
            }
        }
        //update weight2
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            if(weight2==weight2) {
                weight2Prev[i] = weight2[i];

                weight2[i] -= weight2Inc[i];

            }
            else
            {
                //weight2 is nan:
                weight2Prev[i]=0.01*weight2Inc[i];  
                weight2[i] -= weight2Inc[i];
                
            }
        }
        //update bias1
        for (size_t i = 0; i < hiddenDim; i++)
        {
            if(bias1==bias1) {
                bias1Prev[i] = bias1[i];

                bias1[i] -= bias1Inc[i];

            }
            else
            {
                //bias1 is nan:
                bias1Prev[i]=0.01*bias1Inc[i];  
                bias1[i] -= bias1Inc[i];
                
            }
        }
        //update bias2
        for (size_t i = 0; i < outputDim; i++)
        {
            if(bias2==bias2) {
                bias2Prev[i] = bias2[i];

                bias2[i] -= bias2Inc[i];

            }
            else
            {
                //bias2 is nan:
                bias2Prev[i]=0.01*bias2Inc[i];  
                bias2[i] -= bias2Inc[i];
                
            }
        }
        //update everything else
        
        //std::cout << "auto_encoder update end" << std::endl;
        //std::cout << "auto_encoder update end" << std::endl;

    }
    //update weight1GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1GradPrev()
    {
        //std::cout << "auto_encoder updateWeight1GradPrev" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            if(weight1Grad[i]==weight1Grad[i])
                weight1GradPrev[i] = weight1Grad[i];
            else
            {
                //weight1GradPrev is nan:
                weight1GradPrev[i]=0.0001;
            }
         }
        //std::cout << "auto_encoder updateWeight1GradPrev end" << std::endl;

    }
    //update weight2GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2GradPrev()
    {
        //std::cout << "auto_encoder updateWeight2GradPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            if (weight2Grad[i]==weight2Grad[i])
            {
              weight2GradPrev[i] = weight2Grad[i];
            }
            else
            {
                //weight2GradPrev is nan:
                weight2Grad[i]=0.0001*weight2[i]*2.0;
                weight2GradPrev[i]=0.0001;
            }
            
        }
        //std::cout << "auto_encoder updateWeight2GradPrev end" << std::endl;

    }
    //update bias1GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1GradPrev()
    {
        //std::cout << "auto_encoder updateBias1GradPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrev[i] = bias1Grad[i];
        }
        //std::cout << "auto_encoder updateBias1GradPrev end" << std::endl;

    }
    //update bias2GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2GradPrev()
    {
        //std::cout << "auto_encoder updateBias2GradPrev" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrev[i] = bias2Grad[i];
        }
        //std::cout << "auto_encoder updateBias2GradPrev end" << std::endl;

    } 


    //update weight1Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1Inc()
    {
        //std::cout << "auto_encoder updateWeight1Inc" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Inc[i] = momentum * weight1Inc[i] + learningRate * weight1Grad[i];
        }
        //std::cout << "auto_encoder updateWeight1Inc end" << std::endl;

    }
    //update weight2Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2Inc()
    {
        //std::cout << "auto_encoder updateWeight2Inc" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Inc[i] = momentum * weight2Inc[i] + learningRate * weight2Grad[i];
        }
        //std::cout << "auto_encoder updateWeight2Inc end" << std::endl;

    }
    //update bias1Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1Inc()
    {
        //std::cout << "auto_encoder updateBias1Inc" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Inc[i] = momentum * bias1Inc[i] + learningRate * bias1Grad[i];
        }
        //std::cout << "auto_encoder updateBias1Inc end" << std::endl;

    }
    //update bias2Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2Inc()
    {
        //std::cout << "auto_encoder updateBias2Inc" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Inc[i] = momentum * bias2Inc[i] + learningRate * bias2Grad[i];
        }
        //std::cout << "auto_encoder updateBias2Inc end" << std::endl;

    }

    //update weight1GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1GradPrevPrev()
    {
        //std::cout << "auto_encoder updateWeight1GradPrevPrev" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrevPrev[i] = weight1GradPrev[i];
        }
        //std::cout << "auto_encoder updateWeight1GradPrevPrev end" << std::endl;

    }

    //update weight2GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2GradPrevPrev()
    {
        //std::cout << "auto_encoder updateWeight2GradPrevPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrevPrev[i] = weight2GradPrev[i];
        }
        //std::cout << "auto_encoder updateWeight2GradPrevPrev end" << std::endl;

    }
    //update bias1GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1GradPrevPrev()
    {
        //std::cout << "auto_encoder updateBias1GradPrevPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrevPrev[i] = bias1GradPrev[i];
        }
        //std::cout << "auto_encoder updateBias1GradPrevPrev end" << std::endl;

    }
    //update bias2GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2GradPrevPrev()
    {
        //std::cout << "auto_encoder updateBias2GradPrevPrev" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrevPrev[i] = bias2GradPrev[i];
        }
        //std::cout << "auto_encoder updateBias2GradPrevPrev end" << std::endl;

    }

    //update weight1Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1Grad()
    {
        //std::cout << "auto_encoder updateWeight1Grad" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Grad[i] = weight1GradPrev[i] + momentum * weight1GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateWeight1Grad end" << std::endl;

    }
    //update weight2Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2Grad()
    {
        //std::cout << "auto_encoder updateWeight2Grad" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Grad[i] = weight2GradPrev[i] + momentum * weight2GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateWeight2Grad end" << std::endl;

    }
    //update bias1Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1Grad()
    {
        //std::cout << "auto_encoder updateBias1Grad" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Grad[i] = bias1GradPrev[i] + momentum * bias1GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateBias1Grad end" << std::endl;

    }
    //update bias2Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2Grad()
    {
        //std::cout << "auto_encoder updateBias2Grad" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Grad[i] = bias2GradPrev[i] + momentum * bias2GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateBias2Grad end" << std::endl;

    }
    
    //update weight1
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1()
    {
        //std::cout << "auto_encoder updateWeight1" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            if(weight1Grad[i]==weight1Grad[i])
                weight1[i] -= learningRate * weight1Grad[i];
            else
            {
                //weight1Grad is nan:
                weight1Grad[i]=0.01*weight1[i];
                weight1[i] -= learningRate * weight1Grad[i];

            }
        }
        //std::cout << "auto_encoder updateWeight1 end" << std::endl;

    }
    //update weight2
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::updateWeight2()
    {
        //std::cout << "auto_encoder updateWeight2" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            if(weight2Grad[i]==weight2Grad[i])
                weight2[i] -= learningRate * weight2Grad[i];
            else
            {
                //weight2Grad is nan:
                weight2Grad[i]=0.01*weight2[i];
                weight2[i] -= learningRate * weight2Grad[i];

            }   
        }
        //std::cout << "auto_encoder updateWeight2 end" << std::endl;

    }
    //update bias1
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::updateBias1()
    {
        //std::cout << "auto_encoder updateBias1" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            if(bias1Grad[i]==bias1Grad[i])
                bias1[i] -= learningRate * bias1Grad[i];
            else
            {
                //bias1Grad is nan:
                bias1Grad[i]=0.01*bias1[i];
                bias1[i] -= learningRate * bias1Grad[i];

            }
        }
        //std::cout << "auto_encoder updateBias1 end" << std::endl;

    }
    //update bias2
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::updateBias2()
    {
        //std::cout << "auto_encoder updateBias2" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            if(bias2Grad[i]==bias2Grad[i])
                bias2[i] -= learningRate * bias2Grad[i];
            else
            {
                //bias2Grad is nan:
                bias2Grad[i]=0.01*bias2[i];
                bias2[i] -= learningRate * bias2Grad[i];

            }
            
        }
        //std::cout << "auto_encoder updateBias2 end" << std::endl;

    } 
    //conjugate gradient  
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::conjugateGradient()
    {
        //std::cout << "auto_encoder conjugateGradient " << std::endl;
        //update weight1Grad
        updateWeight1Grad();
        //update weight2Grad
        updateWeight2Grad();
        //update bias1Grad
        updateBias1Grad();
        //update bias2Grad
        updateBias2Grad();
        //update weight1
        updateWeight1();
        //update weight2
        updateWeight2();
        //update bias1
        updateBias1();
        //update bias2
        updateBias2();
        //update weight1GradPrevPrev
        updateWeight1GradPrevPrev();
        //update weight2GradPrevPrev
        updateWeight2GradPrevPrev();
        //update bias1GradPrevPrev
        updateBias1GradPrevPrev();
        //update bias2GradPrevPrev
        updateBias2GradPrevPrev();
        //update weight1Inc
        updateWeight1Inc();
        //update weight2Inc
        updateWeight2Inc();
        //update bias1Inc
        updateBias1Inc();
        //update bias2Inc
        updateBias2Inc();
        
        //std::cout << "auto_encoder conjugateGradient end" << std::endl;
    }

    //clear()
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::clear()
    {
        //delete everything and set to zero
        //std::cout << "auto_encoder clear" << std::endl;
        auto check = [](T* ptr) {if (ptr != nullptr) { delete[] ptr; ptr = nullptr; } };    
        check(input);
        check(hidden);
        check(output);
        check(weight1);
        check(weight2);
        check(bias1);
        check(bias2);
        check(weight1Grad);
        check(weight2Grad);
        check(bias1Grad);
        check(bias2Grad);
        check(weight1GradPrev);
        check(weight2GradPrev);
        check(bias1GradPrev);
        check(bias2GradPrev);
        check(weight1GradPrevPrev);
        check(weight2GradPrevPrev);
        check(bias1GradPrevPrev);
        check(bias2GradPrevPrev);
        check(weight1Inc);
        check(weight2Inc);
        check(bias1Inc);
        check(bias2Inc);
        check(weight1Sparsity);
        check(weight2Sparsity);
        //std::cout << "auto_encoder clear end" << std::endl;
        
    }
    struct tf_auto_encoder
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.conjugateGradient();
            ae.getOutput(output);
        }
    };  

    struct  tf_auto_encoder_grad
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight1Grad(output);
        }
    };  

    struct  tf_auto_encoder_grad2
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight2Grad(output);
        }
    };     

    struct  tf_auto_encoder_grad3
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias1Grad(output);
        }
    };

    struct  tf_auto_encoder_grad4
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias2Grad(output);
        }
    };

    struct  tf_auto_encoder_grad5
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight1(output);
        }
    };

    struct  tf_auto_encoder_grad6
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight2(output);
        }
    };

    struct  tf_auto_encoder_grad7
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias1(output);
        }
    };
     
    struct  tf_auto_encoder_grad8
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias2(output);
        }
    };

    struct  tf_auto_encoder_grad9
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            output[0] = 0;
        }
    };

    //autoencoder<>::save_as_pt - save autoencoder as tensorflow pre trained file
    //
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::save_as_pt ( const std::string filename ) {

        std::ofstream out(filename, std::ios::binary); 
        if (!out.is_open()) {
            std::cout << "Cannot open file to write: " << filename << std::endl;
            return;
        }
        //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
        //save weights and biases
        out.write((char*)weight1, sizeof(real_x) * inputDim * hiddenDim);
        out.write((char*)weight2, sizeof(real_x) * hiddenDim * outputDim);
        out.write((char*)bias1, sizeof(real_x) * hiddenDim);
        out.write((char*)bias2, sizeof(real_x) * outputDim);
        out.close();
    }

    //convert autoencoder json to tensorflow pt file:
    //src: autoencoder json file
    //dst: tensorflow pt file
    template <typename T, typename real_x >
    struct file_converter_pt 
    {
        void operator()(const std::string src,const std::string dst)
        {

            auto_encoder<T,real_x> ae;
            ae.load(src);
            ae.save_as_pt(dst); 

        }
    };
 
    template <class T,class real_x>
    class softmax_classifier : public auto_encoder<T,real_x>
    {
    protected:
       size_t n_classes;
       size_t n_dimensions;

       matrix<real_t> weight;
       real_t alpha=1.0;
       real_t lambda=0.01;
       real_t cost=0.;
       real_t accuracy=0.;
       real_t loss=0.;//softmax loss.
       size_t total_samples=0;
       real_t correct_samples=0.;
 
    
       friend class auto_encoder<T,real_x>;
       
       public:
        softmax_classifier(size_t n_classes,size_t n_dimensions,real_t alpha=1.0,real_t lambda=0.025):auto_encoder<T,real_x>(n_dimensions,n_classes * n_dimensions,n_classes),weight(n_classes,n_dimensions),alpha(alpha),lambda(lambda) 
        {
            this->n_classes = n_classes;
            this->n_dimensions = n_dimensions;
            this->alpha = alpha;
            this->lambda = lambda;
            //init random weight
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
        

            weight.resize(n_classes,n_dimensions);
            //init weight from autoencoders weights and biases
            //update weight from autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    weight(j,i) = auto_encoder<T,real_x>::weight1[i * n_classes + j];
                    if(weight(j,i)!=weight(j,i))
                    {
                        //replace nan with random value
                        weight(j,i) = dis(gen);

                    }
                }
            }       
            //update bias from autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                weight(i,0) =   this->softplus( auto_encoder<T,real_x>::bias1[i] ) ;
            }
            //done.
            
        }//constructor      

        softmax_classifier(size_t n_classes,size_t n_dimensions,real_t alpha,real_t lambda,real_t* weight1,real_t* weight2,real_t* bias1,real_t* bias2):auto_encoder<T,real_x>(n_dimensions,n_classes * n_dimensions,n_classes),weight(n_classes,n_dimensions),alpha(alpha),lambda(lambda) 
        {
            this->n_classes = n_classes;
            this->n_dimensions = n_dimensions;
            this->alpha = alpha;
            this->lambda = lambda;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
        

            //init random weight
            weight.resize(n_classes,n_dimensions); 
            //init weight from autoencoders weights and biases
            //update weight from autoencoder
            if(weight1)
            {
                for (size_t i = 0; i < n_dimensions; i++)
                {
                    for (size_t j = 0; j < n_classes; j++)
                    {
                        if(weight1[i * n_classes + j]==weight1[i * n_classes + j])
                            weight(j,i) = weight1[i * n_classes + j];
                            else
                            {
                                weight(j,i) =  dis(gen);
                                weight1[i * n_classes + j] = weight(j,i);

                            }

                    }
                }
            }
            if(weight2)
            {
                for (size_t i = 0; i < n_classes; i++)
                {
                    for (size_t j = 0; j < n_dimensions; j++)
                    {
                        if(weight2[j * n_classes + i]==weight2[j * n_classes + i ])
                        weight(i,j) = weight2[j * n_classes + i];
                        else{
                            weight(i,j) = dis(gen);
                            weight2[j * n_classes + i] = weight(i,j);

                        }
                    }
                }
            }
            //update bias from autoencoder
            if(bias1)
            {
                for (size_t i = 0; i < n_classes; i++)
                {
                    if(bias1[i]==bias1[i])
                     weight(i,0) = bias1[i];
                     else
                     bias1[i] = 0.0;
                }
            }
            if(bias2)
            {
                for (size_t i = 0; i < n_dimensions; i++)
                {
                    if(bias2[i]==bias2[i])
                     weight(0,i) =        bias2[i];
                     else
                        bias2[i] = 0.0;
                }
            }
            //done.


        }//constructor 

        //copy constructor
        softmax_classifier(const softmax_classifier& other):auto_encoder<T,real_x>(other),weight(other.n_classes,other.n_dimensions),alpha(other.alpha),lambda(other.lambda)
        {
            this->n_classes = other.n_classes;
            this->n_dimensions = other.n_dimensions;
            this->alpha = other.alpha;
            this->lambda = other.lambda;
            //init random weight
            weight.resize(n_classes,n_dimensions);
            //update weight from autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    weight(j,i) = other.weight(j,i);
                }
            }
            //update bias from autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                weight(i,0) = other.weight(i,0);
            }
        }       
        //base copy constructor   
        softmax_classifier(const auto_encoder<T,real_x>& other) :   auto_encoder<T,real_x>(other) ,weight(auto_encoder<T,real_x>::inputDim,auto_encoder<T,real_x>::inputDim*auto_encoder<T,real_x>::hiddenDim),alpha(1.0),lambda(0.01)
        {
            this->n_classes =  this->inputDim;
            this->n_dimensions = this->inputDim*this->hiddenDim;
            //assuming weight1 and bias were copied from autoencoder
            //update weight from autoencoder

            weight.resize(n_classes,n_dimensions);
            //update weight from autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    weight(j,i) =  auto_encoder<T,real_x>::weight1[i * n_classes + j];
                }
            }
            //update bias from autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                weight(i,0) =  auto_encoder<T,real_x>::bias1[i];
            }
            //std::cout << "softmax weight: " << weight << std::endl;
            //std::cout << "softmax weight1: " << other.weight1 << std::endl;
            //std::cout << "softmax bias1: " << other.bias1 << std::endl;

        }   
        

        //copy assignment operator
        softmax_classifier& operator=(const softmax_classifier& other)
        {
            if(this != &other)
            {
                this->n_classes = other.n_classes;
                this->n_dimensions = other.n_dimensions;
                this->alpha = other.alpha;
                this->lambda = other.lambda;
                //init random weight
                weight = matrix<real_t>::Random(n_classes,n_dimensions);
                //update weight from autoencoder
                for (size_t i = 0; i < n_dimensions; i++)
                {
                    for (size_t j = 0; j < n_classes; j++)
                    {
                        weight(j,i) = other.weight(j,i);
                    }
                }
                //update bias from autoencoder
                for (size_t i = 0; i < n_classes; i++)
                {
                    weight(i,0) = other.weight(i,0);
                }
            }
            return *this;
        }   

        

        void balance_labels()
        {
            for(size_t i=0;i<n_classes;++i)
            {
                for(size_t j=0;j<n_dimensions;++j)
                {
                    if(weight(i,j)!=weight(i,j))
                    weight(i,j) = 1.0;
                }
                //balance bias
                if(weight(i,0)!=weight(i,0))
                weight(i,0) = 1.0;
            }
        }

        //forward
        void forward(const matrix<real_t>& input,matrix<real_t>& output)
        {
            //autoencoder :
            //feedforward on autoencoder was already called on train

            auto_encoder<T,real_x>::forward(input);
            //update sofmax weight from autoencoder
            if(weight.size1()!= n_classes || weight.size2() != n_dimensions)
                weight.resize(n_classes,n_dimensions);
            //update weight from autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    //update weight from autoencoder
                    weight(j,i) = auto_encoder<T,real_x>::weight1[i * n_classes + j];
                    // update weight from autoencoder
                    if(weight(j,i)!=weight(j,i))
                    {
                        //replace nan with random value
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_real_distribution<> dis(0.0, 1.0);
                        weight(j,i) = dis(gen);

                    }
                     
                }
            }
            //update bias from autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                if(auto_encoder<T,real_x>::bias1[i] == auto_encoder<T,real_x>::bias1[i])
                weight(i,0) = auto_encoder<T,real_x>::bias1[i];
                else
                {
                    //replace nan with random value
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<> dis(0.0, 1.0);
                    weight(i,0) = dis(gen);
                    //update autoencoder bias1 with random value
                    auto_encoder<T,real_x>::bias1[i] = weight(i,0);
                }   
            }   

            //cross entropy:

            //update output from softmax. 
            for(size_t i=0;i<output.size1();++i)
            {
                for(size_t j=0;j<output.size2();++j)
                {
                    output(i,j) = this->softplus(output(i,j)) ;
                }
            }
             
            //done.
        }
        //backward
        void backward(const matrix<real_t>& input, matrix<real_t>& output,const matrix<real_t>& target,matrix<real_t>& grad)
        {
          
            

            //autoencoder :
            auto_encoder<T,real_x>::backward(input,output,target,grad);
            //softmax :
            
            //update weight from autoencoder:
            if(weight.size1()!= n_classes || weight.size2() != n_dimensions)
                weight.resize(n_classes,n_dimensions);
            //update weight from autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    //update weight from autoencoder
                    weight(j,i) = auto_encoder<T,real_x>::weight1[i * n_classes + j];
                    // update weight from autoencoder
                    if(weight(j,i)!=weight(j,i))
                    {
                        //replace nan with random value
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_real_distribution<> dis(0.0, 1.0);
                        weight(j,i) = dis(gen);

                    }
                     
                }
            }   
            //update bias from autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                weight(i,0) = auto_encoder<T,real_x>::bias1[i];
            }   
            //update output from softmax:
            for(size_t i=0;i<output.size1();++i)
            {
                for(size_t j=0;j<output.size2();++j)
                {
                    output(i,j) = this->softplus(output(i,j)) ;
                }
            }
            //update grad from softmax:
            for(size_t i=0;i<grad.size1();++i)
            {
                for(size_t j=0;j<grad.size2();++j)
                {
                    grad(i,j) = this->softplus(grad(i,j)) ;
                }
            }
            //done.

        }   
        //update
        void update(const matrix<real_t>& grad)
        {
            //autoencoder :
            auto_encoder<T,real_x>::update( );
            //softmax
            if(grad.size1()&&grad.size2())
                weight = weight - ( grad * alpha  );
            //update weight on autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    auto_encoder<T,real_x>::weight1[i * n_classes + j] = weight(j,i);
                }
            }
            //update bias on autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                auto_encoder<T,real_x>::bias1[i] = weight(i,0);
            }
            //update weight on autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    auto_encoder<T,real_x>::weight2[i * n_classes + j] = weight(j,i);
                }
            }
            //update bias on autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                auto_encoder<T,real_x>::bias2[i] = weight(i,0);
            }
        

            //done.
            
        }
        
        //cross entropy for softmax
        real_t cross_entropy(const matrix<real_t>& target,const matrix<real_t>& output)
        {
            //cross entropy
            real_t cross_entropy = 0.;
            size_t cols = std::min(target.size2(),output.size2());
            size_t rows = std::min(target.size1(),output.size1());
            for (size_t i = 0; i < rows; i++)
            {
                for (size_t j = 0; j < cols; j++)
                {
                    cross_entropy += target(i,j) * std::log(output(i,j));
                }
            } 
            return cross_entropy;
        }   
        //softmax loss
        real_t softmax_loss(const matrix<real_t>& target,const matrix<real_t>& output)
        {
            //softmax loss
            real_t softmax_loss = 0.;
            for (size_t i = 0; i < target.size1(); i++)
            {
                for (size_t j = 0; j < target.size2(); j++)
                {
                    softmax_loss += target(i,j) * std::log(output(i,j));
                }
            }
            return softmax_loss;
        }   
        //softmax accuracy
        real_t softmax_accuracy(const matrix<real_t>& target,const matrix<real_t>& output)
        {
            //softmax accuracy
            real_t softmax_accuracy = 0.;
            for (size_t i = 0; i < target.size1(); i++)
            {
                for (size_t j = 0; j < target.size2(); j++)
                {
                    if(target(i,j)==output(i,j))
                        softmax_accuracy += 1.;
                }
            }
            return softmax_accuracy;
        }
        //softmax cost
        real_t softmax_cost(const matrix<real_t>& target,const matrix<real_t>& output)
        {
            //softmax cost
            real_t softmax_cost = 0.;
            for (size_t i = 0; i < target.size1(); i++)
            {
                for (size_t j = 0; j < target.size2(); j++)
                {
                    softmax_cost += target(i,j) * std::log(output(i,j));
                }
            }
            return softmax_cost;
        }   

        
        //train

        

        void train(const matrix<T>& input, std::vector<size_t>& label_indices) 
        {
            //reset total samples and correct samples
            std::cout<<"[+] DEBUG : softmax_classifier::train cases# "<<std::to_string(input.size1())<<" ,"
            <<std::to_string(input.size2())<<" features, "<<std::to_string(label_indices.size())<<" labels"
            <<std::endl;
            this->total_samples = 0;
            this->correct_samples = 0;
            this->loss = 0.;
            this->accuracy = 1.;
            this->cost = 0.;
            const real_t epsilon = 0.0000001;
            //autoencoder :
            std::vector<real_t> target_indices;
            for (size_t i = 0; i < label_indices.size(); i++)
            {
                target_indices.push_back(real_t(label_indices[i]));
            }
            //auto_encoder<T,real_x>::train(input.data(),target_indices.data(),input.size1()*input.size2());
            //softmax:
            //target:
            std::map<  size_t  , double > class_distribution ;
            for(size_t n=0;n<n_classes;++n)
            {
                class_distribution.insert(std::make_pair(n,0.0));
            }   

            double total_dist = 0.;
            matrix<real_t> target(input.rows(),n_classes);
            for (size_t i = 0; i < input.rows(); i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    
                    target(i,j) = target_indices[i]==j?1.:0.;
                    if(target(i,j)==1.)
                    class_distribution[j]=class_distribution[j]+1.;
                }
                total_dist += 1.;
                //target(i,label_indices[i]) = target_indices[i];
            }
            //debug class distribution : 

            for( auto& dist : class_distribution) 
            {
                //update weight with the row sum of target distribution to balance the labels. 
                
                for (size_t i = 0; i < n_dimensions; i++)
                {
                    weight(dist.first,i) = weight(dist.first,i) * (dist.second/total_dist);
                }
 
                std::cout << "[+] DEBUG : softmax_classifier::train distribution : "<<std::to_string(dist.first)<<" , "<<std::to_string(dist.second/total_dist)<<" ( label prob )" << std::endl; 
            }

            //balance distribution of labels:
            //balance_labels();
            //train:
            train(input,target);

            //update labels from target :
            for (size_t i = 0; i < input.rows(); i++)
            {
                //get max from target indices:
                size_t max_index = 0;
                real_t sum=0.;
                real_t prob=0.;//probability
                real_t row_sum = input.row_sum(i);
                
                for (size_t j = 0; j < n_classes; j++)
                {   
                    //check nan
                    if( target(i,j)!=target(i,j) )
                        {
                            if(row_sum!=row_sum) 
                                target(i,j) =epsilon;
                            else if (row_sum==0.0)
                            {
                                target(i,j) =epsilon;
                            }
                            else
                            target(i,j) = j*epsilon;
                        }
                    //check max
                    if(target(i,j)>target(i,max_index))
                      { 
                        prob = target(i,j)==target(i,j)?target(i,j):epsilon  ;
                        max_index = j;

                        if(prob==prob)
                        {
                            sum+=prob;
                            max_index = j;
                        }//else : prob is nan 
                        else{
                            //backpropogate nan/-nan :
                            prob =  target(i,j) = epsilon;
                        } 

                      }
                    
                }
                label_indices[i] = max_index;
                
                std::cout<<"[+] DEBUG : softmax_classifier::train : "<<std::to_string(i)<<" case," 
                <<std::to_string(label_indices[i])<<" label, "<<std::to_string(prob/sum)<<" probability" 
                << ", row sum : " << std::to_string(target.row_sum(i))<<    std::endl; 

            } 
            //done.
         }
        
        //train a single case
        void train ( T*& input, T*& expected_output )
        {
 
            if(!input||!expected_output)
                return;
            //autoencoder :
            auto_encoder<T,real_x>::train(input,expected_output,n_classes);
            size_t output_size = auto_encoder<T,real_x>::outputDim; 

           
            //train a single case with feedforward and backpropogation 
            //over autoencoder and softmax
            auto_encoder<T,real_x>::forward(input); 
            auto_encoder<T,real_x>::backward(input,expected_output); 
            //update weight from autoencoder 
            
            //update weight from autoencoder
            if(weight.size1()!= n_classes || weight.size2() != n_dimensions)
                weight.resize(n_classes,n_dimensions);
            //update weight from autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    //update weight from autoencoder
                    weight(j,i) = auto_encoder<T,real_x>::weight1[i * n_classes + j];
                    // update weight from autoencoder
                    if(weight(j,i)!=weight(j,i))
                    {
                        //replace nan with random value
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_real_distribution<> dis(0.0, 1.0);
                        weight(j,i) = dis(gen);

                    }
                     
                }
            }   
            //update bias from autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                weight(i,0) = auto_encoder<T,real_x>::bias1[i];
            }
            
             //update labels from target :
            //get max from target indices: 
            
            //update output from softmax.
            for(size_t i=0;i<output_size;++i)
            {
                expected_output[i] = this->output[i];
            }

            size_t max_index = 0;
            for (size_t j = 0; j < n_classes; j++)
            {
                if(expected_output[j] >expected_output[max_index])
                    max_index = j;
            }

            //update label:
            for(size_t i=0;i<output_size;++i)
            {
                i==max_index?expected_output[i]=1.:expected_output[i]=0.;
                
            }
            

            //update total samples and correct samples 
            this->total_samples += 1;
            this->correct_samples +=  1.0 - (expected_output[max_index] - this->output[max_index]); 
            this->loss = this->total_loss / real_t(this->total_samples);
            this->accuracy = this->correct_samples / real_t(this->total_samples);
              //done.
        }

        //train

        void train(const matrix<T>& input,  matrix<T>& target)
        {
            //reset total samples and correct samples
            this->total_samples = 0;
            this->correct_samples = 0;
            this->loss = 0.;
            this->accuracy = 1.;
            this->cost = 0.;


            matrix<real_t> output(target);

            //autoencoder :
            auto_encoder<T,real_x>::train(input.data(),output.data(),output.size1()*output.size2());

            //softmax:
            std::cout << "[+] DEBUG : softmax FF  " << std::endl;            //feedforward
            forward(input,output);
            std::cout << "[+] DEBUG : softmax BP  " << std::endl;
            //apply gradient on backpropogation

            static matrix<real_t> grad(n_classes,n_dimensions);
            //equals to matrix<>::Zero
            
            
            backward(input,output,target,grad);
            //backward updates grad?

            //update
            //update(grad) debug:
            //std::cout << "softmax weight: " << weight << std::endl;


            /*std::cout << "softmax weight: " << weight << std::endl;
            std::cout << "softmax bias: " << weight.row(0) << std::endl;
            std::cout << "softmax grad: " << grad << std::endl;
            std::cout << "softmax grad bias: " << grad.row(0) << std::endl;
            std::cout << "softmax output: " << output << std::endl;
            std::cout << "softmax target: " << target << std::endl;
            std::cout << "softmax error: " << (output - target)/(output.size1()*target.size2()) << std::endl;
            */
            update(grad);    
            //calculate loss:
            this->total_samples += input.rows();
            
            real_t correct  = 0.;
            real_t loss = 0. , loss_d = 0.;
            for (size_t i = 0; i < input.rows(); i++)
            {
                for (size_t j = 0; j < input.cols(); j++)
                {
                    if(output(i,j)!=output(i,j))
                        output(i,j) = 0.0;
                    correct += target(i,j) == output(i,j)?1.:0;
                    
                    loss_d = target(i,j) * std::log(output(i,j));
                    if(loss_d == loss_d)
                        loss += loss_d;
                    else
                        loss += 0.;

                }
            }   
            loss = -loss;
            correct_samples += correct;
            if(loss!=-NAN&&loss!=NAN)
                this->total_loss += loss;
            


            this->loss = this->total_loss / real_t(this->total_samples);

            this->accuracy = correct / real_t(input.rows());
            this->cost =  this->loss + this->lambda * this->auto_encoder<T,real_x>::cost(output.data(),target.data(),output.size1()*output.size2()  );


            std::cout << "[+] softmax total loss: " << this->total_loss << std::endl;
            std::cout << "[+] softmax total samples: " << this->total_samples << std::endl;
            std::cout << "[+] softmax loss: " << this->loss << std::endl;
            std::cout << "[+] softmax accuracy: " <<(100.0 - this->accuracy ) << std::endl;
            std::cout << "[+] softmax cost: " << this->cost << std::endl;

             
            //update target:
            for (size_t i = 0; i < target.rows(); i++)
            {
                for (size_t j = 0; j < target.cols(); j++)
                {
                     
                        target(i,j) =  output(i,j); 
                }
            }
        }
        //predict
        void predict(  provallo::matrix<T>& input,provallo::matrix<T>& output)
        {
            
            //autoencoder :
            //

            auto_encoder<T,real_x>::predict(  input,output); 
            
            //softmax
            forward(input,output);
            //std::cout << "softmax output: " << output << std::endl;
            //loss
            real_t loss = 0.,loss_d=0.;
            for (size_t i = 0; i < input.rows(); i++)
            {
                for (size_t j = 0; j < input.cols(); j++)
                {
                    loss_d = output(i,j) * std::log(output(i,j));
                    if(loss_d!=-NAN&&loss_d!=NAN)
                        loss += loss_d;
                    else
                        loss += 0.;
                }
            }   
            loss = -loss;

            this->loss = loss;
            this->accuracy = 1.0 - loss;
                  //done.

             //this->cost =  this->loss + this->lambda * this->auto_encoder<T,real_x>::cost(output.data(),output.data(),output.size1()*output.size2()  );


            
        }
        void test(const matrix<T>& input ,std::vector<size_t>& target_)
        {
            std::vector<real_t> target(  target_.size() );
            for (size_t i = 0; i < target_.size(); i++)
            {
                target[i] = real_t(target_[i]);
            }
            
            test(input,target);
            
            for ( size_t i = 0; i < target.size(); i++)
            {
                target_[i] = size_t(target[i]);
            }

        }
        //test with labels indices
        void test(const matrix<T>& input ,std::vector<real_t>& target)
        {
            //reset total samples and correct samples
            this->total_samples = 0;
            this->correct_samples = 0.;
            //autoencoder :
            matrix<T> output(input);  
            auto_encoder<T,real_x>::predict(input,output);
            //softmax :
            //target:
            matrix<T> target_matrix(input.rows(),n_classes);
            for (size_t i = 0; i < input.rows(); i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    target_matrix(i,j) = target[i]==j?1.:0.;
                }
               // target_matrix(i,target[i]) = target[i];
            }
            //test
            test(input,target_matrix);
            //update target:
            for (size_t i = 0; i < target_matrix.rows(); i++)
            {
                size_t max_index = 0;
                for (size_t j = 0; j < target_matrix.cols(); j++)
                {
                    if(target_matrix(i,j)==target_matrix(i,j)&&target_matrix(i,j)>target_matrix(i,max_index))
                        max_index = j;
                }
                target[i] = max_index;
                
            }    
            target_matrix.clear();
            output.clear();
            //done.

        }
        //test
        void test(const matrix<T>& input, matrix<T>& target)
        {

            //reset total samples and correct samples
            this->total_samples = 0;
            this->correct_samples = 0.;
            //autoencoder :
            matrix<T> output(target);  
            forward(input,output);
            real_t error = 0.0;
            total_samples += input.rows();
            real_t loss=0.0 , loss_d=0.;
            for (size_t i = 0; i < target.rows(); i++)
            {

                //check each target class for each row
                for (size_t j = 0; j < target.cols(); j++)
                {
                    error += target(i,j) == output(i,j)?1.:0;
                    loss_d = target(i,j) * std::log(output(i,j));
                    if(loss_d!=-NAN&&loss_d!=NAN)
                        loss += loss_d;
                    else
                        loss += 0.;
                }
                
            }   
            
            error = -error;
            std::cout << "[+] softmax error: " << std::to_string(error) << std::endl;

            this->total_loss += error;
            this->loss = this->total_loss / this->total_samples;
            this->accuracy = correct_samples / real_t(this->total_samples);
            this->cost =  this->loss + this->lambda * this->auto_encoder<T,real_x>::cost(output.data(),target.data(),output.size1()*output.size2()  );
            //update target:
            for (size_t i = 0; i < target.rows(); i++)
            {
                for (size_t j = 0; j < target.cols(); j++)
                {
                    target(i,j) = output(i,j) ;
                }
            }
             
            //done.

        }
        //probabilities accessor
        matrix<real_t> probabilities()const
        {
            return weight;
        }
        void backpropogate_error(const matrix<T>& input,const matrix<T>& target)
        {
            matrix<real_t> output(target.rows(),target.cols()); 
            forward(input,output);
            matrix<real_t> error = output - target;
            //calculate loss:
            real_t loss = error.squaredNorm();
            std::cout << "[+] softmax loss: " << loss << std::endl;
        }
        //backpropogate error
        void backpropogate_error(const matrix<T>& input,const matrix<T>& target,matrix<real_x>& error)
        {
            matrix<real_t> output(target.rows(),target.cols()); 
            forward(input,output);
            error = output - target;
            //calculate loss:
            real_t loss = error.squaredNorm();
            std::cout << "[+] softmax loss: " << loss << std::endl;
            //update target:
            for (size_t i = 0; i < target.rows(); i++)
            {
                for (size_t j = 0; j < target.cols(); j++)
                {
                    target(i,j) = target(i,j)==1?j:0;
                }
            }
            //update output:
            for (size_t i = 0; i < output.rows(); i++)
            {
                for (size_t j = 0; j < output.cols(); j++)
                {
                    output(i,j) = output(i,j)==1?j:0;
                }
            }
        }


        void gnuplot(const std::string filename)
        {
            //gnuplot
            //script file :
            std::ofstream out(filename.c_str());
            //data file :
            std::ofstream out2((filename+std::string(".dat")).c_str());

            if (!out.is_open()) {
                std::cout << "Cannot open file to write: " << filename << std::endl;
                return;
            }
             for (size_t i = 0; i < n_classes; i++)
            {
                for (size_t j = 0; j < n_dimensions; j++)
                {
                    out2 << std::to_string(this->weight(i,j) )<< " ";
                }
                out2 << i << std::endl;
            } 
             //close file
             
            out2.close();
            //write the graph to file
            out << "#!/usr/bin/gnuplot -persist" << std::endl;
            out << "set title \"softmax classifier\"" << std::endl;
            out << "set terminal png size 800,600 enhanced font \"Helvetica,20\"" << std::endl;
            out << "set xlabel \"x\"" << std::endl;
            out << "set ylabel \"y\"" << std::endl;
            out << "set zlabel \"z\"" << std::endl;
            out <<"set xrange [-1:1]" << std::endl;
            out <<"set yrange [-1:1]" << std::endl;
            out <<"set zrange [-1:1]" << std::endl;
            out <<"unset colorbox"<< std::endl;
            out <<"set palette defined (-1 \"blue\", 0 \"white\", 1 \"red\")"<< std::endl;
            out <<"set pm3d at b"<< std::endl;
            out <<"set view 60,30"<< std::endl;
            out <<"set isosamples 50"<< std::endl;
            out <<"set hidden3d back offset 1 trianglepattern 3 undefined 1 altdiagonal bentover"<< std::endl;
            out <<"set style data lines"<< std::endl;
            out <<"set style function lines"<< std::endl;
            out <<"set style line 1 lt 1 lw 1 lc rgb \"red\""<< std::endl;
            out <<"set style line 2 lt 1 lw 1 lc rgb \"blue\""<< std::endl;
            out <<"set style line 3 lt 1 lw 1 lc rgb \"green\""<< std::endl;
            out <<"set style line 4 lt 1 lw 1 lc rgb \"yellow\""<< std::endl;
            out <<"set style line 5 lt 1 lw 1 lc rgb \"black\""<< std::endl;
            out <<"set style line 6 lt 1 lw 1 lc rgb \"cyan\""<< std::endl;

            out <<"set style line 7 lt 1 lw 1 lc rgb \"orange\""<< std::endl;
            out <<"set style line 8 lt 1 lw 1 lc rgb \"purple\""<< std::endl;
            out <<"set style line 9 lt 1 lw 1 lc rgb \"brown\""<< std::endl;
            out <<"set style line 10 lt 1 lw 1 lc rgb \"pink\""<< std::endl;
            out <<"set style line 11 lt 1 lw 1 lc rgb \"gray\""<< std::endl;
            out <<"set style line 12 lt 1 lw 1 lc rgb \"violet\""<< std::endl;
            out <<"set style line 13 lt 1 lw 1 lc rgb \"magenta\""<< std::endl;
            out <<"set style line 14 lt 1 lw 1 lc rgb \"gold\""<< std::endl;
            out <<"set style line 15 lt 1 lw 1 lc rgb \"silver\""<< std::endl;
            out <<"set style line 16 lt 1 lw 1 lc rgb \"dark-gray\""<< std::endl;
            //set parallel view 
            out <<"set view equal xyz"<< std::endl; 
            out <<"set contour base"<< std::endl;
            out <<"set cntrparam levels 20"<< std::endl;
            out <<"set style data lines"<< std::endl;
            out <<"set style function lines"<< std::endl;
            out <<"set style line 1 lt 1 lw 1 lc rgb \"red\""<< std::endl;
            out <<"set style line 2 lt 1 lw 1 lc rgb \"blue\""<< std::endl;
            out <<"set style line 3 lt 1 lw 1 lc rgb \"green\""<< std::endl;
            out <<"set style line 4 lt 1 lw 1 lc rgb \"yellow\""<< std::endl;
            out <<"set style line 5 lt 1 lw 1 lc rgb \"black\""<< std::endl;
            out <<"set style line 6 lt 1 lw 1 lc rgb \"cyan\""<< std::endl;
            
            //plot autoencoder weights 
            out << "splot \"" << filename << ".dat\" using 1:2:3:4 with points palette pt 7 ps 1 title \"softmax\"" << std::endl;
            //plot autoencoder
            out.close();
            //done.
        }
         //save
        void save(const std::string filename)
        {
             //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
            //save weights and biases

            //delete file if exists 
            try {
            std::remove((std::string("softmax_")+std::string(filename).c_str()).c_str());

            std::ofstream out(( std::string("softmax_")+std::string(filename) ).c_str());            
            if (!out.is_open()) {
                std::cout << "Cannot open file to write: " << filename << std::endl;
                return;
            }
            //write the path to autoencoder
            out << filename << std::endl;
            //write weights and biases
            out.write((char*)weight.data(), sizeof(real_x) * n_classes * n_dimensions);
            out.close();
            }
            catch (const std::exception& e) {
                std::cout << e.what() << std::endl;
            }   
            //save autoencoder
            auto_encoder<T,real_x>::save(filename);
        }
        //load
        void load(const std::string filename)
        {
            //load autoencoder
            auto_encoder<T,real_x>::load(filename);
            //load softmax
            std::ifstream in( (std::string("softmax_")+std::string(filename)).c_str() );
            if (!in.is_open()) {
                std::cout << "Cannot open file to read: " << filename << std::endl;
                return;
            }
            //read the path to autoencoder
            std::string ae_filename;
            std::getline(in,ae_filename);
            //read weights and biases
            in.read((char*)weight.data(), sizeof(real_x) * n_classes * n_dimensions);
            in.close();

            //update weight from autoencoder
            for (size_t i = 0; i < n_dimensions; i++)
            {
                for (size_t j = 0; j < n_classes; j++)
                {
                    weight(j,i) = auto_encoder<T,real_x>::weight1[i * n_classes + j];
                }
            }   
            //update bias from autoencoder
            for (size_t i = 0; i < n_classes; i++)
            {
                weight(i,0) = auto_encoder<T,real_x>::bias1[i];
            }   
            
            
        }

        //get class labels  from the argmax of the sofmax 
        void get_class_labels(const matrix<real_t>& input,matrix<real_t>& output)
        {
            matrix<real_t> softmax;
            forward(input,softmax);
            output (input.rows(),1);
            for (size_t i = 0; i < input.rows(); i++)
            {
                real_t max =  softmax.maxCoeff();
                for (size_t j = 0; j < input.cols(); j++)
                {
                    if(softmax(i,j) == max)
                    {
                        output(i,0) = j;
                        break;
                    }
                }
                
            }
        }
        virtual ~softmax_classifier() override
        {
         if( !auto_encoder<real_t>::inputDim || !auto_encoder<real_t>::hiddenDim || !auto_encoder<real_t>::outputDim )
                return; 
            
            if(this->input!=nullptr)
            {   
                delete[] this->input;
            }
            else
            return;
            this->input = nullptr;
            if(this->hidden)
                delete[] this->hidden;
            else 
            return;
            this->hidden = nullptr;
            if(this->output)
                delete[] this->output;
            else
            return;

            this->output = nullptr;
            
            if(this->weight1)
                delete[] this->weight1;

            this->weight1 = nullptr;
            if(this->weight2)
                delete[] this->weight2;

            this->weight2 = nullptr;
            if(this->bias1)
                delete[] this->bias1;

            this->bias1 = nullptr;
            if(this->bias2)
                delete[] this->bias2;

            this->bias2 = nullptr;
            if(this->weight1Grad)
                delete[] this->weight1Grad;

            this->weight1Grad = nullptr;
            if(this->weight2Grad)
                delete[] this->weight2Grad;

            this->weight2Grad = nullptr;
            if(this->bias1Grad)
                delete[] this->bias1Grad;

            this->bias1Grad = nullptr;
            if(this->bias2Grad)
                delete[] this->bias2Grad;

            this->bias2Grad = nullptr;
            if(this->weight1GradPrev)
                delete[] this->weight1GradPrev;

            this->weight1GradPrev = nullptr;
            if(this->weight2GradPrev)
                delete[] this->weight2GradPrev;

            this->weight2GradPrev = nullptr;
            if(this->bias1GradPrev)
                delete[] this->bias1GradPrev;

            this->bias1GradPrev = nullptr;
            if(this->bias2GradPrev)
                delete[] this->bias2GradPrev;

            this->bias2GradPrev = nullptr;
            if(this->weight1GradPrevPrev)
                delete[] this->weight1GradPrevPrev;

            this->weight1GradPrevPrev = nullptr;
            if(this->weight2GradPrevPrev)
                delete[] this->weight2GradPrevPrev;

            this->weight2GradPrevPrev = nullptr;
            if(this->bias1GradPrevPrev)
                delete[] this->bias1GradPrevPrev;

            this->bias1GradPrevPrev = nullptr;
            if(this->bias2GradPrevPrev)
                delete[] this->bias2GradPrevPrev;

            this->bias2GradPrevPrev = nullptr;
            if(this->weight1Inc)
                delete[] this->weight1Inc;

            this->weight1Inc = nullptr;
            if(this->weight2Inc)
                delete[] this->weight2Inc;

            this->weight2Inc = nullptr;

            if(this->bias1Inc)
                delete[] this->bias1Inc;
            this->bias1Inc = nullptr;
            if(this->bias2Inc)
                delete[] this->bias2Inc;
            this->bias2Inc = nullptr;
            if(this->weight1Sparsity)
                delete[] this->weight1Sparsity;
            this->weight1Sparsity = nullptr;

            if(this->weight2Sparsity)
                delete[] this->weight2Sparsity;
            this->weight2Sparsity = nullptr;

            if(this->bias1Sparsity)
                delete[] this->bias1Sparsity;
            this->bias1Sparsity = nullptr;
            if(this->bias2Sparsity)
                delete[] this->bias2Sparsity;   
            this->bias2Sparsity = nullptr;

 
 
        }


    }; 
    template <class T,class real_x>
    class variational_softmax : public softmax_classifier<T,real_x> 
    {
        //variational softmax
        //https://arxiv.org/pdf/1511.06038.pdf 
        //
        //variational softmax is a softmax classifier with a gaussian prior on the weights
        //the gaussian prior is learned by the autoencoder
        //the autoencoder is trained to minimize the reconstruction error and the KL divergence between the gaussian prior and the posterior
        //the autoencoder is trained with the conjugate gradient method  
        //the autoencoder is trained with the variational softmax as the loss function


        //autoencoder
        auto_encoder<T,real_x> ae;
        
        //classifier input
        matrix<real_t> input;
        //train labels:
        matrix<real_t> target;

        //weight data:
        matrix<real_t> weight;

        public: 

        //conversion constructor from softmax<real_t,real_x> 
        variational_softmax (const softmax_classifier<T,real_x>& conversion ) : softmax_classifier<T,real_x>(conversion),weight(conversion->n_classes,conversion->n_dimensions)
        {
            //init random weight
            this->weight = matrix<real_t>::Random(this->n_classes,this->n_dimensions);
            //init autoencoder
            ae = auto_encoder<T,real_x>(this->n_dimensions,this->n_dimensions,0.01,0.01);
            input = matrix<real_t>::Zero(1,this->n_dimensions);
            target = matrix<real_t>::Zero(1,this->n_classes);

            
        } 
        //constructor
        variational_softmax(size_t n_classes,size_t n_dimensions,real_t alpha,real_t lambda)
        {
            this->n_classes = n_classes;
            this->n_dimensions = n_dimensions;
            this->alpha = alpha;
            this->lambda = lambda;
            //init random weight
            this->weight = matrix<real_t>::Random(n_classes,n_dimensions);
            //init autoencoder
            ae = auto_encoder<T,real_x>(n_dimensions,n_dimensions,0.01,0.01);
        }   
        //train
        void train(const matrix<real_t>& input,const matrix<real_t>& target)
        {
            //set input
            this->input = input;
            //set target
            this->target = target;
            //train autoencoder
            ae.train(input,input);
            //train classifier
            softmax_classifier<T,real_x>::train(input,target);
        }
        //predict
        void predict(const matrix<real_t>& input,matrix<real_t>& output)
        {
            softmax_classifier<T,real_x>::predict(input,output);
        }
        //save
        void save(const std::string filename)
        {
            std::ofstream out(filename, std::ios::binary); 
            if (!out.is_open()) {
                std::cout << "Cannot open file to write: " << filename << std::endl;
                return;
            }
            //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
            //save weights and biases
            out.write((char*)this->weight.data(), sizeof(real_x) * this->n_classes * this->n_dimensions);
            out.close();
        }
        //load
        void load(const std::string filename)
        {
            std::ifstream in(filename, std::ios::binary); 
            if (!in.is_open()) {
                std::cout << "Cannot open file to read: " << filename << std::endl;
                return;
            }
            //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
            //save weights and biases
            in.read((char*)this->weight.data(), sizeof(real_x) * this->n_classes * this->n_dimensions);
            in.close();
        }
        //get weights
        void getWeight(matrix<real_t>& weight)
        {
            weight = this->weight;
        } 
        //forward
        void forward(const matrix<real_t>& input,matrix<real_t>& output)
        {
            //softmax
            output = input * this->weight.transpose();
            output = output.unaryExpr([](real_t x) { return std::exp(x); });
            output = output.rowwise([](real_t x) {return x;}) / output.sum();
        }   
        //backward
        void backward(const matrix<real_t>& input,const matrix<real_t>& output,const matrix<real_t>& target,matrix<real_t>& grad)
        {
            //softmax
            grad = output - target;
            
            this->weight = this->weight - this->alpha * grad.transpose() * input; 

            //autoencoder
            //set input
            for ( size_t i=0;i<input.rows();++i)
            {
                //copy input row into input:
                for(size_t j=0;j<this->inputDim;++j)
                {
                  auto_encoder<T,real_x>::input[j]=input(i,j); 
                  //update auto encoder on the sample:
                  forward();                    
                }
                
            }
            
            grad = grad.transpose() * input;
            //regularization
            grad = grad + this->lambda * this->weight;
            
            //update weight on autoencoder
            for (size_t i = 0; i < this->n_dimensions; i++)
            {
                for (size_t j = 0; j < this->n_classes; j++)
                {
                    auto_encoder<T,real_x>::weight1[i * this->n_classes + j] = this->weight(j,i);
                }
            }
            //update bias on autoencoder
            for (size_t i = 0; i < this->n_classes; i++)
            {
                auto_encoder<T,real_x>::bias1[i] = this->weight(i,0);
            }
            //return output projections:
            output = input * this->weight.transpose();
            output = output.unaryExpr([](real_t x) { return std::exp(x); });
            output = output.rowwise([](real_t x) {return x;}) / output.sum();
            //upodate output label:
            for (size_t i = 0; i < output.rows(); i++)
            {
                for (size_t j = 0; j < output.cols(); j++)
                {
                    output(i,j) = output(i,j)==1?j:0;
                }
            }
            //update target:
            for (size_t i = 0; i < target.rows(); i++)
            {
                for (size_t j = 0; j < target.cols(); j++)
                {
                    target(i,j) = target(i,j)==1?j:0;
                }
            }
            
            //done  

        }
        //update
        void update(const matrix<real_t>& grad)
        {
            this->weight = this->weight - this->alpha * grad;
        }
        //get weight
        matrix<real_t> get_weight()const
        {
            return this->weight;
        }


     };//variational_softmax
    
     //softmax with dropout
        template <class T,class real_x> 
        class dropout_softmax : public softmax_classifier<T,real_x> 
        {
            //dropout softmax
            //https://arxiv.org/pdf/1511.06038.pdf 
            //
            //dropout softmax is a softmax classifier with a gaussian prior on the weights
            //the gaussian prior is learned by the autoencoder
            //the autoencoder is trained to minimize the reconstruction error and the KL divergence between the gaussian prior and the posterior
            //the autoencoder is trained with the conjugate gradient method  
            //the autoencoder is trained with the variational softmax as the loss function      
            //dropout softmax is a softmax classifier with dropout on the input and hidden layers

            //autoencoder
            typedef softmax_classifier<T,real_x> parent;
            typedef auto_encoder<T,real_x> grandparent;

            //dropout
            real_t dropout_rate;
            //dropout mask
            matrix<real_t> dropout_mask;
            //dropout input
            matrix<real_t> dropout_input;
            //dropout hidden
            matrix<real_t> dropout_hidden;
            //dropout output
            matrix<real_t> dropout_output;
            //dropout weight
            matrix<real_t> dropout_weight;
            //dropout bias
            matrix<real_t> dropout_bias;
            //dropout weight grad
            matrix<real_t> dropout_weight_grad;
            //dropout bias grad
            matrix<real_t> dropout_bias_grad;
            //dropout weight grad prev
            matrix<real_t> dropout_weight_grad_prev;
            //dropout bias grad prev
            matrix<real_t> dropout_bias_grad_prev;
            //dropout weight grad prev prev
            matrix<real_t> dropout_weight_grad_prev_prev;
            //dropout bias grad prev prev
            matrix<real_t> dropout_bias_grad_prev_prev;
            //dropout weight inc
            matrix<real_t> dropout_weight_inc;
            //dropout bias inc
            matrix<real_t> dropout_bias_inc;
            //dropout weight sparsity
            matrix<real_t> dropout_weight_sparsity;
            //dropout bias sparsity
            matrix<real_t> dropout_bias_sparsity;
            //dropout weight sparsity
            matrix<real_t> dropout_weight_sparsity_prev;
            //dropout bias sparsity
            matrix<real_t> dropout_bias_sparsity_prev;
            //dropout weight sparsity
            matrix<real_t> dropout_weight_sparsity_prev_prev;
            //dropout bias sparsity
            matrix<real_t> dropout_bias_sparsity_prev_prev;
            //dropout weight sparsity

            void init_dropout()
            {
             
                //matrices size n_classes x n_dimensions:
                std::vector<matrix<real_t>*> dropout_matrices = {&dropout_weight,&dropout_weight_grad,&dropout_weight_grad_prev,&dropout_weight_grad_prev_prev,&dropout_weight_inc,&dropout_weight_sparsity,&dropout_weight_sparsity_prev,&dropout_weight_sparsity_prev_prev};
                //matrices size n_classes x 1:
                std::vector<matrix<real_t>*> dropout_matrices2 = {&dropout_bias,&dropout_bias_grad,&dropout_bias_grad_prev,&dropout_bias_grad_prev_prev,&dropout_bias_inc,&dropout_bias_sparsity,&dropout_bias_sparsity_prev,&dropout_bias_sparsity_prev_prev};
                //other matrices:
                std::vector<matrix<real_t>*> dropout_matrices3 = {&dropout_mask,&dropout_input,&dropout_hidden,&dropout_output}; 



                

                //seed random uniform distribution
                std::random_device rd;
                if(dropout_rate<0.1)
                    dropout_rate = 0.5;

                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0, this->n_classes);
                //init dropout matrices
                for (size_t i = 0; i < dropout_matrices.size(); i++)
                {
                    dropout_matrices[i]->resize(this->n_classes,this->n_dimensions);
                    //set dropout values on dropout matrices
                    for (size_t j = 0; j < dropout_matrices[i]->size1(); j++)
                    {
                        for (size_t k = 0; k < dropout_matrices[i]->size2(); k++)
                        {
                            dropout_matrices[i]->operator()(j,k) = dis(gen) < dropout_rate?0.:1.;

                        }
                    }
                }
                //init dropout matrices2
                for (size_t i = 0; i < dropout_matrices2.size(); i++)
                {
                    dropout_matrices2[i]->resize(this->n_classes,1);
                    //set dropout values on dropout matrices
                    for (size_t j = 0; j < dropout_matrices2[i]->size1(); j++)
                    {
                        for (size_t k = 0; k < dropout_matrices2[i]->size2(); k++)
                        {
                            dropout_matrices2[i]->operator()(j,k) = dis(gen) < dropout_rate?0.:1.;

                        }
                    }   

                }
                //init dropout matrices3
                for (size_t i = 0; i < dropout_matrices3.size(); i++)
                {
                    dropout_matrices3[i]->resize(1,this->n_dimensions);
                    //set dropout values on dropout matrices
                    for (size_t j = 0; j < dropout_matrices3[i]->size1(); j++)
                    {
                        for (size_t k = 0; k < dropout_matrices3[i]->size2(); k++)
                        {
                            dropout_matrices3[i]->operator()(j,k) = dis(gen) < dropout_rate?0.:1.;

                        }
                    }   

                }   
                //done
                
              }

            void update_dropout_mask(const matrix<real_t>& target)
            {
                dropout_output.resize(target.rows(),target.cols());
                //update target 
                for (size_t i = 0; i < target.rows(); i++)
                {
                    for (size_t j = 0; j < target.cols(); j++)
                    {
                        dropout_output(i,j) = target(i,j)==1?1.:0.;
                    }
                }
                forward();
                
                //update dropout mask
                
                
                for (size_t i = 0; i < dropout_mask.size1(); i++)
                {
                    for (size_t j = 0; j < dropout_mask.size2(); j++)
                    {
                        dropout_mask(i,j) = dropout_rate < dropout_output(i,j)?0.:1.;
                    }
                }
                //update dropout weight
                dropout_weight = dropout_weight.cwiseProduct(dropout_mask);
                //update dropout weight grad
                dropout_weight_grad = dropout_weight_grad.cwiseProduct(dropout_mask);
                //update dropout weight grad prev
                dropout_weight_grad_prev = dropout_weight_grad_prev.cwiseProduct(dropout_mask);
                //update dropout weight grad prev prev
                dropout_weight_grad_prev_prev = dropout_weight_grad_prev_prev.cwiseProduct(dropout_mask);
                //update dropout weight inc
                dropout_weight_inc = dropout_weight_inc.cwiseProduct(dropout_mask);
                //update dropout weight sparsity
                dropout_weight_sparsity = dropout_weight_sparsity.cwiseProduct(dropout_mask);
                //update dropout weight sparsity prev
                dropout_weight_sparsity_prev = dropout_weight_sparsity_prev.cwiseProduct(dropout_mask);
                //update dropout weight sparsity prev prev
                dropout_weight_sparsity_prev_prev = dropout_weight_sparsity_prev_prev.cwiseProduct(dropout_mask);
                //update dropout bias
                dropout_bias = dropout_bias.cwiseProduct(dropout_mask);
                //update dropout bias grad
                dropout_bias_grad = dropout_bias_grad.cwiseProduct(dropout_mask);
                //update dropout bias grad prev
                dropout_bias_grad_prev = dropout_bias_grad_prev.cwiseProduct(dropout_mask);
                //update dropout bias grad prev prev
                dropout_bias_grad_prev_prev = dropout_bias_grad_prev_prev.cwiseProduct(dropout_mask);
                //update dropout bias inc
                dropout_bias_inc = dropout_bias_inc.cwiseProduct(dropout_mask);
                //update dropout bias sparsity
                dropout_bias_sparsity = dropout_bias_sparsity.cwiseProduct(dropout_mask);
                //update dropout bias sparsity prev
                dropout_bias_sparsity_prev = dropout_bias_sparsity_prev.cwiseProduct(dropout_mask);
                
                


                //done

            }

            public:
            //conversion constructor from softmax<real_t,real_x>
            dropout_softmax (const softmax_classifier<T,real_x>& conversion ) : softmax_classifier<T,real_x>(conversion)
            {
                dropout_rate = 0.5;
                init_dropout();
            }   

            //constructor
            dropout_softmax(size_t n_classes,size_t n_dimensions,real_t alpha,real_t lambda) : softmax_classifier<T,real_x>(n_classes,n_dimensions,alpha,lambda)
            {
                dropout_rate = 0.5;
                init_dropout();
            }   
            //train
            void train(const matrix<real_t>&input ,std::vector<size_t>& labels )
            {
                std::vector<real_t> target(  labels.size() );
                for (size_t i = 0; i < labels.size(); i++)
                {
                    target[i] = real_t(labels[i]);
                    
                }
                
                train(input,target);
                
                for ( size_t i = 0; i < labels.size(); i++)
                {
                    labels[i] = size_t(target[i]);
                }
            }
            //train
            void train (const matrix<real_t>& input,std::vector<real_t>& labels )
            {
                //forward input:

                for ( size_t i=0 ; i< input.size1()/this->getInputDim();++i )
                {
                    for ( size_t j=0 ; j< this->getInputDim();++j )
                    {
                        auto_encoder<T,real_x>::input[j] = input(i,j);
                    }
                    //update autoencoder on the sample:
                    auto_encoder<T,real_x>::forward();
                    

                }
                //forward softmax
                forward();
                
                //set target
                this->dropout_output.resize(labels.size(),this->n_classes);
                //set target
                for (size_t i = 0; i < labels.size(); i++)
                {
                    for (size_t j = 0; j < this->n_classes; j++)
                    {
                        this->dropout_output(i,j) = labels[i]==j?1.:0.;
                    }
                }
                //train classifier,autoencoder:
                softmax_classifier<T,real_x>::train(input,dropout_output);
                //update dropout mask
                update_dropout_mask(dropout_output);

                //update labels from output:
                for (size_t i = 0; i < this->dropout_output.rows(); i++)
                {
                    size_t max_index = 0;
                    for (size_t j = 0; j < dropout_output.cols(); j++)
                    {
                        if(this->dropout_output(i,j)>dropout_output(i,max_index))
                            max_index = j;
                    }
                    labels[i] = max_index;
                    
                }   
                //done

            }
            void train(const matrix<real_t>& input,  matrix<real_t>& target)
            {
                //set input
                this->input = input;
                //set target
                this->target = target;
                //train classifier,autoencoder:
                softmax_classifier<T,real_x>::train(input,this->target);
                //update dropout mask
                update_dropout_mask(target);
                //done
            }
            //predict
            void predict(const matrix<real_t>& input,matrix<real_t>& output)
            {
                softmax_classifier<T,real_x>::predict(input,output);

            }
            //save
            void save(const std::string filename)
            {
                std::ofstream out(filename, std::ios::binary); 
                if (!out.is_open()) {
                    std::cout << "Cannot open file to write: " << filename << std::endl;
                    return;
                }
                //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
                //save weights and biases
                out.write((char*)this->weight.data(), sizeof(real_x) * this->n_classes * this->n_dimensions);
                out.close();
            }
            //load
            void load(const std::string filename)
            {
                std::ifstream in(filename, std::ios::binary); 
                if (!in.is_open()) {
                    std::cout << "Cannot open file to read: " << filename << std::endl;
                    return;
                }
                //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
                //save weights and biases
                in.read((char*)this->weight.data(), sizeof(real_x) * this->n_classes * this->n_dimensions);
                in.close();
            }
            //get weights
            void getWeight(matrix<real_t>& weight)
            {
                weight = this->weight;
            }
            void forward()
            {
                //forward autoencoder
                auto_encoder<T,real_x>::forward();
                //forward softmax
                softmax_classifier<T,real_x>::forward(this->dropout_input,this->dropout_output);
            
            }   

            
            //forward

            void forward(const matrix<real_t>& input,matrix<real_t>& output)
            {
                //softmax
                output = input * this->weight.transpose();
                output = output.unaryExpr([](real_t x) { return std::exp(x); });
                output = output.rowwise([](real_t x) {return x;}) / output.sum();
            }   
            //backward
            void backward(const matrix<real_t>& input,const matrix<real_t>& output,const matrix<real_t>& target,matrix<real_t>& grad)
            {
                //softmax
                grad = output - target;
                
                this->weight = this->weight - this->alpha * grad.transpose() * input; 

                //autoencoder
                //set input
                for ( size_t i=0;i<input.rows();++i)
                {
                    //copy input row into input:
                    for(size_t j=0;j<this->inputDim;++j)
                    {
                      auto_encoder<T,real_x>::input[j]=input(i,j); 
                      //update auto encoder on the sample:
                      forward();                    
                    }
                    
                }
                
                grad = grad.transpose() * input;
                //regularization
                grad = grad + this->lambda * this->weight;
                
                //update weight on autoencoder
                for (size_t i = 0; i < this->n_dimensions; i++)
                {
                    for (size_t j = 0; j < this->n_classes; j++)
                    {
                        auto_encoder<T,real_x>::weight1[i * this->n_classes + j] = this->weight(j,i);
                    }
                }
                //update bias on autoencoder
                for (size_t i = 0; i < this->n_classes; i++)
                {
                    auto_encoder<T,real_x>::bias1[i] = this->weight(i,0);
                }
                //return output projections:
                output = input * this->weight.transpose();
                output = output.unaryExpr([](real_t x) { return std::exp(x); });
                output = output.rowwise([](real_t x) {return x;}) / output.sum();
                    
            }   
            //update
            void update(const matrix<real_t>& grad)
            {
                this->weight = this->weight - this->alpha * grad;
            }           
            //get weight
            matrix<real_t> get_weight()const
            {
                return this->weight;
            }
            //get dropout rate
            real_t get_dropout_rate()const
            {
                return dropout_rate;
            }
            //set dropout rate
            void set_dropout_rate(real_t dropout_rate)
            {
                this->dropout_rate = dropout_rate;
                init_dropout();
            }
            //get dropout mask
            matrix<real_t> get_dropout_mask()const
            {
                return dropout_mask;
            }
            //get dropout input
            matrix<real_t> get_dropout_input()const
            {
                return dropout_input;
            }
            //get dropout hidden
            matrix<real_t> get_dropout_hidden()const
            {
                return dropout_hidden;
            }
            //get dropout output
            matrix<real_t> get_dropout_output()const
            {
                return dropout_output;
            }
            //get dropout weight
            matrix<real_t> get_dropout_weight()const
            {
                return dropout_weight;
            }
            //get dropout bias
            matrix<real_t> get_dropout_bias()const
            {
                return dropout_bias;
            }
        }; //dropout_softmax
        

        //fix softmax multi class classifier
        template <class T,class real_x>
        class multiclass_softmax : public   softmax_classifier<T,real_x>
        {
            private :
            //number of classes
            size_t n_classes;
            //number of dimensions
            size_t n_dimensions;
            //learning rate
            real_t alpha;
            //regularization rate
            real_t lambda;
            //total samples
            size_t total_samples;
            //correct samples

            size_t correct_samples;
            //total loss
            real_t total_loss;
            //loss
            real_t loss;
            //accuracy
            real_t accuracy;
            //cost
            real_t cost;
            //weight
            matrix<real_t> weight;
            //bias
            matrix<real_t> bias;
            //weight grad
            matrix<real_t> weight_grad;
            //bias grad
            matrix<real_t> bias_grad;
            //weight grad prev
            matrix<real_t> weight_grad_prev;
            //bias grad prev
            matrix<real_t> bias_grad_prev;
            //weight grad prev prev
            matrix<real_t> weight_grad_prev_prev;
            //bias grad prev prev
            matrix<real_t> bias_grad_prev_prev;
            //weight inc
            matrix<real_t> weight_inc;
            //bias inc
            matrix<real_t> bias_inc;
            //weight sparsity
            matrix<real_t> weight_sparsity;
            //bias sparsity
            matrix<real_t> bias_sparsity;
            //weight sparsity
            matrix<real_t> weight_sparsity_prev;
            //bias sparsity
            matrix<real_t> bias_sparsity_prev;
            //weight sparsity
            matrix<real_t> weight_sparsity_prev_prev;
            //bias sparsity
            matrix<real_t> bias_sparsity_prev_prev;
            

            public:
            //constructor
            multiclass_softmax(size_t n_classes,size_t n_dimensions,real_t alpha,real_t lambda) : softmax_classifier<T,real_x>(n_classes,n_dimensions,  alpha ,lambda ), n_classes(n_classes),n_dimensions(n_dimensions),alpha(alpha),lambda(lambda)   
            {
                //init random weight
                weight = matrix<real_t>::Random(n_classes,n_dimensions);
                //init random bias
                bias = matrix<real_t>::Random(n_classes,1);
                //init weight grad
                weight_grad = matrix<real_t>::Zero(n_classes,n_dimensions);
                //init bias grad
                bias_grad.resize(n_classes,1);
                //init weight grad prev
                weight_grad_prev.resize(n_classes,n_dimensions);
                //init bias grad prev
                bias_grad_prev.resize(n_classes,1);
                //init weight grad prev prev
                weight_grad_prev_prev.resize(n_classes,n_dimensions);
                //init bias grad prev prev
                bias_grad_prev_prev.resize(n_classes,1);
                //init weight inc
                weight_inc.resize(n_classes,n_dimensions);
                //init bias inc
                bias_inc.resize(n_classes,1);
                //init weight sparsity
                weight_sparsity.resize(n_classes,n_dimensions);
                //init bias sparsity
                bias_sparsity.resize(n_classes,1);
                //init weight sparsity
                weight_sparsity_prev.resize(n_classes,n_dimensions);
                //init bias sparsity
                bias_sparsity_prev.resize(n_classes,1);
                //init weight sparsity
                weight_sparsity_prev_prev.resize(n_classes,n_dimensions);
                //init bias sparsity
                bias_sparsity_prev_prev.resize(n_classes,1);
                //init total samples
                total_samples = 0;
                //init correct samples
                correct_samples = 0;
                //init total loss
                total_loss = 0;
                //init loss
                loss = 0;
                //init accuracy
                accuracy = 0;
                //init cost
                cost = 0;
            }
            //train with labels
            void train(const matrix<real_t>& input,std::vector<size_t>& labels)
            {
                std::vector<real_t> target(  labels.size() );
                for (size_t i = 0; i < labels.size(); i++)
                {
                    target[i] = real_t(labels[i]);
                    
                }
                
                train(input,target);
                
                for ( size_t i = 0; i < labels.size(); i++)
                {
                    labels[i] = size_t(target[i]);
                }
            }   
            //train with vector<real_t> target :
            void train(const matrix<real_t>& input,std::vector<real_t>& target)
            {
                matrix<real_t> target_matrix(target.size(),1);
                for (size_t i = 0; i < target.size(); i++)
                {
                    target_matrix(i,0) = target[i];
                }
                train(input,target_matrix);
                //update target
                for (size_t i = 0; i < target.size(); i++)
                {
                    target[i] = target_matrix(i,0);
                }

            }   
            //train 
            void train(const matrix<real_t>& input,  matrix<real_t>& target)
            {
                //forward
                matrix<real_t> output;
                forward(input,output);
                //backward
                matrix<real_t> grad;
                backward(input,output,target,grad);
                //update
                update(grad);
                //update total samples
                total_samples += input.rows();
                //update correct samples
                for (size_t i = 0; i < input.rows(); i++)
                {
                    if(output(i,0) == target(i,0))
                        correct_samples++;
                }
                //update total loss
                total_loss += loss;
                //update loss
                loss = 0;
                //update accuracy
                accuracy = real_t(correct_samples)/real_t(total_samples);
                //update cost
                cost = total_loss/real_t(total_samples); 
                //return output on target matrix
                target = output;

            }   
            //predict
            void predict(const matrix<real_t>& input,matrix<real_t>& output)
            {
                forward(input,output);
            }   

            //forward
            void forward(const matrix<real_t>& input,matrix<real_t>& output)
            {
                //softmax
                output = input * weight.transpose();
                output = output.unaryExpr([](real_t x) { return std::exp(x); });
                output = output.rowwise() / output.sum();
            }
            //backward
            void backward(const matrix<real_t>& input,const matrix<real_t>& output,const matrix<real_t>& target,matrix<real_t>& grad)
            {
                //softmax
                grad = output - target;
                //weight grad
                weight_grad = grad.transpose() * input;
                //bias grad
                bias_grad = grad.colwise(   ) * bias_inc.sum(); 
                //regularization
                weight_grad = weight_grad + lambda * weight;
                //loss
                loss = -target.cwiseProduct(output.unaryExpr([](real_t x) { return std::log(x); })).sum();
            }
            //update
            void update(const matrix<real_t>& grad)
            {
                //weight inc
                weight_inc = -alpha * grad;
                //bias inc colwise sum
                bias_inc = -alpha * grad.colwise() * bias_inc.sum() ;

                //weight sparsity
                weight_sparsity = weight_sparsity.cwiseProduct(grad);
                //bias sparsity
                bias_sparsity = bias_sparsity.cwiseProduct(grad.colwise()*grad.sum());
                //weight sparsity
                weight_sparsity_prev = weight_sparsity_prev.cwiseProduct(grad);
                //bias sparsity
                bias_sparsity_prev = bias_sparsity_prev.cwiseProduct(grad.colwise()*grad.sum());
                //weight sparsity
                weight_sparsity_prev_prev = weight_sparsity_prev_prev.cwiseProduct(grad);
                //bias sparsity
                bias_sparsity_prev_prev = bias_sparsity_prev_prev.cwiseProduct(grad.colwise()*grad.sum());
                //weight grad prev prev
                weight_grad_prev_prev = weight_grad_prev;
                //bias grad prev prev
                bias_grad_prev_prev = bias_grad_prev;
                //weight grad prev
                weight_grad_prev = weight_grad;
                //bias grad prev
                bias_grad_prev = bias_grad;
                //weight
                weight = weight + weight_inc;
                //bias
                bias = bias + bias_inc;
            }   

            //save
            void save(const std::string filename)
            {
                std::ofstream out(filename, std::ios::binary); 
                if (!out.is_open()) {
                    std::cout << "Cannot open file to write: " << filename << std::endl;
                    return;
                }
                //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
                //save weights and biases
                out.write((char*)weight.data(), sizeof(real_x) * n_classes * n_dimensions);
                out.write((char*)bias.data(), sizeof(real_x) * n_classes * 1);
                out.close();
            }   
            //load
            void load(const std::string filename)
            {
                std::ifstream in(filename, std::ios::binary); 
                if (!in.is_open()) {
                    std::cout << "Cannot open file to read: " << filename << std::endl;
                    return;
                }
                //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
                //save weights and biases
                in.read((char*)weight.data(), sizeof(real_x) * n_classes * n_dimensions);
                in.read((char*)bias.data(), sizeof(real_x) * n_classes * 1);
                in.close();
            }
            //get weight
            matrix<real_t> get_weight()const
            {
                return weight;
            }
            //get bias
            matrix<real_t> get_bias()const
            {
                return bias;
            }
            //get weight grad
            matrix<real_t> get_weight_grad()const
            {
                return weight_grad;
            }
            //get bias grad
            matrix<real_t> get_bias_grad()const
            {
                return bias_grad;
            }
            //get weight grad prev
            matrix<real_t> get_weight_grad_prev()const
            {
                return weight_grad_prev;
            }
            //get bias grad prev
            matrix<real_t> get_bias_grad_prev()const
            {
                return bias_grad_prev;
            }
            // prevprev are not accesible publicly.
            //get weight inc
            matrix<real_t> get_weight_inc()const
            {
                return weight_inc;
            }
            //get bias inc
            matrix<real_t> get_bias_inc()const
            {
                return bias_inc;
            }
            //get weight sparsity
            matrix<real_t> get_weight_sparsity()const
            {
                return weight_sparsity;
            }
            //get bias sparsity
            matrix<real_t> get_bias_sparsity()const
            {
                return bias_sparsity;
            }
            //get weight sparsity prev
            matrix<real_t> get_weight_sparsity_prev()const
            {
                return weight_sparsity_prev;
            }   
            //get bias sparsity prev
            matrix<real_t> get_bias_sparsity_prev()const
            {
                return bias_sparsity_prev;
            }
            //get weight sparsity prev prev
            matrix<real_t> get_weight_sparsity_prev_prev()const
            {
                return weight_sparsity_prev_prev;
            }   
            //get bias sparsity prev prev
            matrix<real_t> get_bias_sparsity_prev_prev()const
            {
                return bias_sparsity_prev_prev;
            }
            //get total samples
            size_t get_total_samples()const
            {
                return total_samples;
            }
            //get correct samples
            size_t get_correct_samples()const
            {
                return correct_samples;
            }
            //get total loss
            real_t get_total_loss()const
            {
                return total_loss;
            }
            //get loss
            real_t get_loss()const
            {
                return loss;
            }
            //get accuracy
            real_t get_accuracy()const
            {
                return accuracy;
            }
            //get cost
            real_t get_cost()const
            {
                return cost;
            }
            //get number of classes
            size_t get_n_classes()const
            {
                return n_classes;
            }
            //get number of dimensions

            size_t get_n_dimensions()const
            {
                return n_dimensions;
            }
            //get learning rate
            real_t get_alpha()const
            {
                return alpha;
            }
            //get regularization rate
            real_t get_lambda()const
            {
                return lambda;
            }
            //set weight
            void set_weight(const matrix<real_t>& weight)
            {
                this->weight = weight;
            }
            //set bias
            void set_bias(const matrix<real_t>& bias)
            {
                this->bias = bias;
            }
            //set weight grad
            void set_weight_grad(const matrix<real_t>& weight_grad)
            {
                this->weight_grad = weight_grad;
            }
            //set bias grad
            void set_bias_grad(const matrix<real_t>& bias_grad)
            {
                this->bias_grad = bias_grad;
            }



        };

} // namespace provallo

#endif /* PROVALLO_AUTO_ENCODER_H_ */