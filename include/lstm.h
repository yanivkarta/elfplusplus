#ifndef __LSTM_H__
#define __LSTM_H__

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>

#include "matrix.h"
#include "utils.h"
#include "sampling_helper.h" //for sampling helpers
#include "info_helper.h"



#define WEIGHTS_SCALE (1.f/256)

namespace provallo
{
    

    template <typename T> 
    class LSTM
    {
        using Matrix = matrix<T>;
        //cross entropy loss function
        provallo::ce_loss<T> cross_entropy_loss;
        provallo::kl_loss<T> kl_loss;
        //sampling helpers
        fft_sampling_helper<T> _sampler;
        //activation functions 
        std::vector<provallo::activation_helper<T>> activations;
        
        //input size
        size_t _input_size;
        //output size
        size_t _output_size;
        //hidden size
        size_t _hidden_size;

        //weights
        Matrix _Wxh;
        Matrix _Whh;
        Matrix _Why;
        //biases
        Matrix _bh;
        Matrix _by;
        //hidden state
        Matrix _hprev;
        //memory state
        Matrix _cprev;
        //input
        Matrix _x;
        //output
        Matrix _y;
        //target
        Matrix _target;
        //loss
        real_t _loss;
        //learning rate
        real_t _learning_rate;
        //gradient clipping
        real_t _clip;
        //number of iterations
        size_t _n;
        //number of epochs
        size_t _epochs;
        //batch size
        size_t _batch_size;
        //number of layers
        size_t _num_layers;
        //number of threads
        size_t _num_threads = 4;
        //number of samples
        size_t _num_samples;
        //number of samples
        size_t _num_samples_per_epoch;
        //number of samples
        size_t _num_batches_per_epoch;
        //number of samples
        size_t _num_batches;
        //random number generator    
        
        std::mutex _mutex; //general barrier mutex
        
        std::mutex _grad_mutex;//gradient barrier mutex

        std::mutex _input_mutex;//input barrier mutex

        std::mutex _output_mutex;//output barrier mutex

        std::mutex _loss_mutex;//loss barrier mutex
        
        std::condition_variable _cv;
        
        //helper matrices
        Matrix _Wfh;
        Matrix _Wih;
        Matrix _Wgh;
        Matrix _Woh;
        Matrix _Wfx;
        Matrix _Wix;
        Matrix _Wgx;
        Matrix _Wox;
        Matrix _Wfy;

        Matrix _Wiy;
        Matrix _Wgy;
        Matrix _Woy;
        
        Matrix _bf;
        Matrix _bi;
        Matrix _bg;
        
        Matrix _bo;
        Matrix _bcf;
        Matrix _bci;
        Matrix _bcg;
        Matrix _bco;
        //derivatives
        Matrix _dWfh;
        Matrix _dWih;
        Matrix _dWgh;
        Matrix _dWoh;
        Matrix _dWfx;
        Matrix _dWix;
        Matrix _dWgx;
        Matrix _dWox;
        Matrix _dWfy;
        Matrix _dWhy;
        Matrix _dWhh;
        Matrix _dWxh;
        Matrix _dWiy;
        Matrix _dWgy;
        Matrix _dWoy;

        Matrix _dbf;
        Matrix _dbi;
        Matrix _dbg;

        Matrix _dbo;
        Matrix _dbc;
        Matrix _dbh;
        Matrix _dby;
        Matrix _dloss;
        //derivatives


        real_t loss_val = 0.0;
        real_t ce_loss_val = 0.0;
        real_t kl_loss_val = 0.0;
        real_t l2_reg = 0.0;
        real_t momentum = 0.0;
        real_t weight_decay = 0.0;
        real_t dropout = 0.0;
        
        gaussian_spike_train_generator<T> _gaussian_spike_train_generator;
        boltzman_base<T> _boltzman_base;
        helmholtz_machine<T> _helmholtz_machine;

        
        public:
        //constructor 
        //minimize construction of spike trains according to random_train_type selection. 

        LSTM(size_t input_size, size_t hidden_size, size_t output_size, size_t seq_length, real_t learning_rate, size_t epochs, size_t batch_size, size_t num_layers, size_t num_threads,size_t random_train_type=0) :
            _input_size(input_size),
            _output_size(output_size),
            _hidden_size(hidden_size),
            _Wxh(input_size,hidden_size),
            _Whh(hidden_size,hidden_size),
            _Why(hidden_size,output_size),
            _bh(1,hidden_size),
            _by(1,output_size),
            _hprev(1,hidden_size),
            _cprev(1,hidden_size),
            _x(1,input_size),
            _y(1,output_size),
            _target(1,output_size),
            _loss(0),
            _learning_rate(learning_rate),
            _clip(5),
            _n(0),
            _epochs(epochs),
            _batch_size(batch_size),
            _num_layers(num_layers),
            _num_threads(num_threads),
            _num_samples(seq_length),
            _num_samples_per_epoch(seq_length*batch_size),
            _num_batches_per_epoch(batch_size),
            _num_batches(epochs*batch_size)
        {
            //initialize the spike engine according to the random train type 

            //initialize weights
            randomize_weights(random_train_type);
            this->_num_threads = 1; 
            //initialize activations
            activations.resize(4);
            activations[0] = activation_helper<T>( activation_helper_activations::sigmoid_default); 
            activations[1] = activation_helper<T>( activation_helper_activations::sigmoid_default); 
            activations[2] = activation_helper<T>( activation_helper_activations::sigmoid_default);
            activations[3] = activation_helper<T>( activation_helper_activations::softmax_default);

            //initialize sampler
            _sampler.init(seq_length);

            //set up default momentum
            momentum = 0.9;
            //set up default weight decay
            weight_decay = 0.0005;
            //set up default dropout
            dropout = 0.5;
            //set up default l2 regularization
            l2_reg = 0.0001;
         
        }
        //simplified constructor: 
        LSTM(size_t input_size,size_t hidden_size,size_t output_size ):LSTM(input_size,hidden_size,output_size,1,0.01,1,1,1,1) {} 
        //randomize weights
        void randomize_weights(size_t random_train_type) {
 
            //initialize the selected random train type 
            if(random_train_type==0) 
            {
                _gaussian_spike_train_generator.init(_input_size,_hidden_size);
            }
            else if(random_train_type==1)
            {
                _boltzman_base.init(_input_size,_hidden_size);
            }
            else if(random_train_type==2)
            {
                _helmholtz_machine.init(_input_size,_hidden_size);
            }
            else
            {
                std::cout<<"invalid random train type"<<std::endl;
                exit(1);
            }
            //initialize weights, set matrix size for each layer 
            std::vector<size_t> sizes;
            sizes.push_back(_input_size);
            for(size_t i=0; i<_num_layers; ++i) {
                sizes.push_back(_hidden_size);
            }
            sizes.push_back(_output_size);
            //initialize weights
            std::vector<T> tmp_weights(_input_size*_hidden_size);
            std::vector<T> tmp_biases(_hidden_size);

            std::vector<T> output_wf, output_wi, output_wg, output_wo; 

            for(size_t i=0; i<_num_layers; ++i) {
            
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<> dis(-1, 1);
                    for(size_t j=0; j<_input_size*_hidden_size; ++j) {
                        tmp_weights[j] = dis(gen);
                    }
                    for(size_t j=0; j<_hidden_size; ++j) {
                        tmp_biases[j] = dis(gen);
                    }
                //choose gaussian spike train generator or boltzman base or helmholtz machine 
                if(random_train_type==0)
                {
                    _gaussian_spike_train_generator.generate();
                    _gaussian_spike_train_generator.refine();
                    //get the output weights
                    output_wf = _gaussian_spike_train_generator.get_output();
                    _gaussian_spike_train_generator.refine();

                    output_wi = _gaussian_spike_train_generator.get_output();
                    _gaussian_spike_train_generator.refine();

                    output_wg = _gaussian_spike_train_generator.get_output();
                    _gaussian_spike_train_generator.refine();

                    output_wo = _gaussian_spike_train_generator.get_output();
                    _gaussian_spike_train_generator.refine();

                     
                }
                else if(random_train_type==1)
                {
                     _boltzman_base.generate();
                     //refine the weights
                     _boltzman_base.refine();
                        //get the output weights
                    output_wf = _boltzman_base.get_output();
                    //refine the weights
                    _boltzman_base.refine();
                    output_wi = _boltzman_base.get_output();
                    //refine the weights
                    _boltzman_base.refine();
                    output_wg = _boltzman_base.get_output();
                    //refine the weights
                    _boltzman_base.refine();
                    output_wo = _boltzman_base.get_output();
                    //refine the weights


                }
                else if(random_train_type==2)
                {
                    _helmholtz_machine.generate();
                    //refine the weights
                    _helmholtz_machine.refine();
                    //get the output weights
                    output_wf = _helmholtz_machine.get_output();
                    //refine the weights
                    _helmholtz_machine.refine();
                    output_wi = _helmholtz_machine.get_output();
                    //refine the weights
                    _helmholtz_machine.refine();
                    output_wg = _helmholtz_machine.get_output();
                    //refine the weights
                    _helmholtz_machine.refine();
                    output_wo = _helmholtz_machine.get_output();
                    //refine the weights

                }
        

                }//end for
                
                //use the generated weights to initialize the LSTM  layers
                _Wxh.resize(_input_size,_hidden_size);
                _Whh.resize(_hidden_size,_hidden_size);
                _Why.resize(_hidden_size,_output_size);
                _bh.resize(1,_hidden_size);
                _by.resize(1,_output_size);
                //initialize weights from tmp_weights or from 
                //
                for(size_t i=0; i<_input_size; ++i) {
                    for(size_t j=0; j<_hidden_size; ++j) {
                        _Wxh(i,j) = tmp_weights[i*_hidden_size+j];
                    }
                }
                for(size_t i=0; i<_hidden_size; ++i) {
                    for(size_t j=0; j<_hidden_size; ++j) {
                        _Whh(i,j) = tmp_weights[(_input_size*_hidden_size+i*_hidden_size+j )%tmp_weights.size()]; 
                    }
                }
                for(size_t i=0; i<_hidden_size; ++i) {
                    for(size_t j=0; j<_output_size; ++j) {
                        _Why(i,j) = tmp_weights[(_input_size*_hidden_size+_hidden_size*_hidden_size+i*_output_size+j)%tmp_weights.size()]; 
                    }
                }
                for(size_t i=0; i<_hidden_size; ++i) {
                    _bh(0,i) = tmp_biases[i];
                }
                for(size_t i=0; i<_output_size; ++i) {
                    _by(0,i) = tmp_biases[_hidden_size+i];
                }
                //initialize output weights
                _Wfh.resize(_input_size,_hidden_size);
                _Wih.resize(_input_size,_hidden_size);
                _Wgh.resize(_input_size,_hidden_size);
                _Woh.resize(_input_size,_hidden_size);
                _Wfx.resize(_input_size,_hidden_size);
                _Wix.resize(_input_size,_hidden_size);
                _Wgx.resize(_input_size,_hidden_size);
                _Wox.resize(_input_size,_hidden_size);  
                //initialize weights
                for(size_t i=0; i<_input_size; ++i) {
                    for(size_t j=0; j<_hidden_size; ++j) {
                        _Wfh(i,j) = output_wf[i*_hidden_size+j];
                        _Wih(i,j) = output_wi[i*_hidden_size+j];
                        _Wgh(i,j) = output_wg[i*_hidden_size+j];
                        _Woh(i,j) = output_wo[i*_hidden_size+j];
                        _Wfx(i,j) = output_wf[i*_hidden_size+j];
                        _Wix(i,j) = output_wi[i*_hidden_size+j];
                        _Wgx(i,j) = output_wg[i*_hidden_size+j];
                        _Wox(i,j) = output_wo[i*_hidden_size+j];
                    }
                }   
                //initialize output biases
                _bf.resize(1,_hidden_size);
                _bi.resize(1,_hidden_size);
                _bg.resize(1,_hidden_size);
                _bo.resize(1,_hidden_size);
                //initialize biases
                for(size_t i=0; i<_hidden_size; ++i) {
                    _bf(0,i) = tmp_biases[i];
                    _bi(0,i) = tmp_biases[i];
                    _bg(0,i) = tmp_biases[i];
                    _bo(0,i) = tmp_biases[i];
                }
                //initialize output weights
                _Wfy.resize(_hidden_size,_output_size);
                _Wiy.resize(_hidden_size,_output_size);
                _Wgy.resize(_hidden_size,_output_size);
                _Woy.resize(_hidden_size,_output_size);
                //initialize weights
                for(size_t i=0; i<_hidden_size; ++i) {
                    for(size_t j=0; j<_output_size; ++j) {
                        _Wfy(i,j) = output_wf[i*_output_size+j];
                        _Wiy(i,j) = output_wi[i*_output_size+j];
                        _Wgy(i,j) = output_wg[i*_output_size+j];
                        _Woy(i,j) = output_wo[i*_output_size+j];
                    }//for
                }//for
                
                //done

        }//end randomize_weights


        //destructor
        ~LSTM() {}
        //fit
        void fit(const Matrix& X, const Matrix& Y) {
            //initialize hidden state
            _hprev.fill(0);
            //initialize memory state
            _cprev.fill(0);
            //initialize loss
            _loss = 0;
            _dloss.resize(1,_output_size);
            //initialize number of iterations
            _n = 0;
            //use multiple threads
            std::vector<std::thread> threads;
            size_t quanta =_num_samples/real_t(_num_threads);
            if (quanta < 1) quanta = _num_threads;
            //divide the data into batches
            //_ce_loss.reset();
            //_kl_loss.reset();

            std::cout<<"fitting with "<<_num_threads<<" threads"<<std::endl;    
            for(size_t i=0; i<_num_threads; ++i) {
                //slice X into quantized slices
                size_t start = i*quanta;
                size_t end = (i+1)*quanta;
                matrix<T> Xq = X.slice(start,end);
                matrix<T> Yq = Y.slice(start,end);
                threads.push_back(std::thread(&LSTM::fit_batch,this,Xq,Yq,i));
                //slice Y into quantized slices
                
             }
            //join threads
            for(auto& t: threads) {
                t.join();
            }

            //done
            //set output and loss
            update_loss();

        }
       
        //forward
        void forward() {
            //iterate forward on the input
            for(size_t i=0; i<_num_samples; ++i) {
                //get the input
                _x = _x.slice(i,i+1);
                //get the target
                _target = _target.slice(i,i+1);
                //forward pass - single sample 
                forward_sample();

                //compute loss
                for (size_t j=0; j<_output_size; ++j) {
                    _loss += -_target(0,j) * log<2>(_y(0,j)); 
                    _dloss(0,j) = -_target(0,j) / _y(0,j);
                    
                }
                //increment number of iterations
                _n++;
                //print loss
                if(_n%100==0) {
                    std::cout << "loss: " << get_ce_loss() << " kl_loss: " <<get_kl_loss() << std::endl;
                }
               
            }   

        }

        //forward_sample
        void forward_sample()
        {
            //update weights and biases thread safe:
            //weights:
            {
            //lock the general mutex for weights
            std::lock_guard<std::mutex> lock(_mutex);
            //update weights
            _Wxh -= (_learning_rate * _dWxh);
            _Whh -= (_learning_rate * _dWhh);
            _Why -= (_learning_rate * _dWhy);
            _bh -= (_learning_rate * _dbh);
            _by -= (_learning_rate * _dby);
            }
            //lock gradients mutex
            {
            //lock the mutex
            std::lock_guard<std::mutex> lock(_grad_mutex);
             //update weights
            _Wfh -=  (_learning_rate * _dWfh);
            _Wih -= (_learning_rate * _dWih);
            _Wgh -= (_learning_rate * _dWgh);
            _Woh -= (_learning_rate * _dWoh);
            }

            //update biases 
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //update biases
            _bf -= _learning_rate * _dbf;
            _bi -= _learning_rate * _dbi;
            _bg -= _learning_rate * _dbg;

            _bo -= _learning_rate * _dbo;
            //update conv biases
            _bcf -= _learning_rate * _dbc;
            _bci -= _learning_rate * _dbc;
            _bcg -= _learning_rate * _dbc;
            _bco -= _learning_rate * _dbc;
            
            }
            //update loss
            {
            //lock the loss mutex
            std::lock_guard<std::mutex> lock(_loss_mutex);
            //update loss
            this->_loss -= (_learning_rate * _dloss).sum();
            
            }
            
            //update the output
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //update output
            _y = _y * _Why + _by;
            }

            //update hidden state
            _hprev = _Whh * _hprev + _Wxh * _x + _bh; 
            //update memory state
            _cprev = _cprev * _Wgh + _Wfh * _x + _Wih * _hprev + _Wgh * _cprev + _bh; 

            //done
 
        }
         
        
        //update
        matrix<T> _clip_gradient(matrix<T>& cr)
        {
            //clip the gradient
            for(size_t i=0; i<cr.rows(); ++i) {
                for(size_t j=0; j<cr.cols(); ++j) {
                    if(cr(i,j) > _clip) {
                        cr(i,j) = _clip;
                    }
                    else if(cr(i,j) < -_clip) {
                        cr(i,j) = -_clip;
                    }
                }
            }
            return cr;
        }
        //update
        void update() {
            //refactor: add l2 regularization and momentum 
            //update weights and biases thread safe:
            //l2 regularization
            //weights:
            {
            //lock the general mutex for weights
            std::lock_guard<std::mutex> lock(_mutex);
            //update weights
            _Wxh -= _learning_rate * l2_reg * _Wxh; 
            _Whh -= _learning_rate * l2_reg * _Whh;
            _Why -= _learning_rate * l2_reg * _Why;
            _bh -= _learning_rate * l2_reg * _bh;
            _by -= _learning_rate * l2_reg * _by;
             }
            //lock gradients mutex
            {
            //lock the mutex
            std::lock_guard<std::mutex> lock(_grad_mutex);
             //update weights with momentum 
            _Wfh -=  _learning_rate * l2_reg * _Wfh;
            _Wih -= _learning_rate * l2_reg * _Wih;
            _Wgh -= _learning_rate * l2_reg * _Wgh;
            _Woh -= _learning_rate * l2_reg * _Woh;
            }
            //update biases 
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //update biases
            _bf -= _learning_rate * l2_reg * _bf;
            _bi -= _learning_rate * l2_reg * _bi;
            _bg -= _learning_rate * l2_reg * _bg;
            }
            //update loss
            {
            //lock the loss mutex
            std::lock_guard<std::mutex> lock(_loss_mutex);
            //update loss
            this->_loss -= _learning_rate * l2_reg * this->_loss;
            }
            //update the output
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //update output
            _y = _y * _Why + _by;
            }
            //update hidden state
            _hprev = _Whh * _hprev + _Wxh * _x + _bh;
            //update memory state
            _cprev = _cprev * _Wgh + _Wfh * _x + _Wih * _hprev + _Wgh * _cprev + _bh;
            //done
        }
        void update_loss()
        {
            //update cross entropy loss
            //apply loss function
            auto ce =  cross_entropy_loss.apply(_y,_target);
            //update kl loss
            auto kl = kl_loss.apply(_y,_target);
            //update loss
            
            
            ce_loss_val = ce.sum();
            kl_loss_val = kl.sum();
            loss_val = ce_loss_val + kl_loss_val;
            

        }
         //fit batch threaded function
        void fit_batch(const Matrix& X, const Matrix& Y, size_t batch) {

           //each thread has its own random number generator
           std::random_device rd;
           std::mt19937 gen(rd());
           std::uniform_int_distribution<> dis(0, _num_samples-1);
            //copy the input
           _x = X;
           //copy the target
           _target = Y;
           //reset cross entropy loss
   
           
              //forward pass
                for(size_t i=0; i<_num_samples; ++i) {
                    //get the input
                    _x = X.slice(i,i+1);
                    //get the target
                    _target = Y.slice(i,i+1);
                    //forward pass
                    forward();
                    //compute loss
                    for (size_t j=0; j<_output_size; ++j) {
                        _loss += -_target(0,j) * log<2>(_y(0,j)); 
                        _dloss(0,j) = -_target(0,j) / _y(0,j);
                        
                    }
                    //increment number of iterations
                    _n++;
                    //print loss
                    if(_n%100==0) {
                        std::cout << "batch: " << batch << " loss: " << get_ce_loss() << " kl_loss: " <<get_kl_loss() << std::endl;
                    }
                   
                }
                //backward pass
                backward();
                //update weights
                update();
                //increment number of iterations
                _n++;
                //print loss
                if(_n%100==0) {
                    std::cout << "batch: " << batch << " loss: " << get_ce_loss() << " kl_loss: " <<get_kl_loss() << std::endl;
                }

        }//end fit_batch
        //backward - single threaded
        void backward() {
            //initialize gradients
            Matrix X(_input_size,_num_samples);
            Matrix Y(_output_size,_num_samples);

            //copy the input
            X = _x;
            //copy the target
            Y = _target;
            //initialize gradients


            Matrix dWxh(_input_size,_hidden_size);
            Matrix dWhh(_hidden_size,_hidden_size);
            Matrix dWhy(_hidden_size,_output_size);
            Matrix dbh(1,_hidden_size);
            Matrix dby(1,_output_size);
            //initialize hidden state
            Matrix dhnext(1,_hidden_size);
            //initialize memory state
            Matrix dcnext(1,_hidden_size);
            //initialize input
            Matrix dx(1,_input_size);
            //initialize output
            Matrix dy(1,_output_size);
            //initialize target
            Matrix dt(1,_output_size);
            

            //calculate the gradients
            for(size_t i=0; i<_num_samples; ++i) {
                //get the input
                _x = X.slice(i,i+1);
                //get the target
                _target = Y.slice(i,i+1);
                //forward pass
                forward();
                //backward pass
                backward_sample();
                //increment number of iterations
                _n++;
                //print loss
                if(_n%100==0) {
                    std::cout << "loss: " << get_ce_loss() << " kl_loss: " <<get_kl_loss() << std::endl;
                }
            }   
             
        }
        //backward_sample
        void backward_sample()
        {
            //update weights and biases thread safe:
            //weights: 
            //use the selected activation function 
           auto sigmoid = activations[0];
           auto tanh = activations[1];
           auto relu = activations[2];
           auto softmax = activations[3];
            //initialize gradients
            {
            //lock the general mutex for weights
            std::lock_guard<std::mutex> lock(_mutex);
            //update weights with gradients 

            _Wxh -= (_learning_rate * _dWxh);
            _Whh -= (_learning_rate * _dWhh);
            _Why -= (_learning_rate * _dWhy);
            _bh -= (_learning_rate * _dbh);
            _by -= (_learning_rate * _dby);
            
            //activate gradients
            sigmoid(_dWxh);
            sigmoid(_dWhh);
            sigmoid(_dWhy);
            sigmoid(_dbh);
            sigmoid(_dby);
            }
            

            //use activation function on the gradients derivatives 

            //lock gradients mutex
            {
            //lock the mutex
            std::lock_guard<std::mutex> lock(_grad_mutex);
             //update weights
            _Wfh -=  (_learning_rate * _dWfh);
            _Wih -= (_learning_rate * _dWih);
            _Wgh -= (_learning_rate * _dWgh);
            _Woh -= (_learning_rate * _dWoh);
            //activate gradients
            sigmoid(_dWfh);
            sigmoid(_dWih);
            sigmoid(_dWgh);
            sigmoid(_dWoh);
            }
            

            //update biases 
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //update biases
            _bf -= _learning_rate * _dbf;
            _bi -= _learning_rate * _dbi;
            _bg -= _learning_rate * _dbg;

            _bo -= _learning_rate * _dbo;
            //update conv biases
            _bcf -= _learning_rate * _dbc;
            _bci -= _learning_rate * _dbc;
            _bcg -= _learning_rate * _dbc;
            _bco -= _learning_rate * _dbc;

            //activate gradients
            sigmoid(_dbf);
            sigmoid(_dbi);
            sigmoid(_dbg);
            sigmoid(_dbo);
            sigmoid(_dbc);

                        
            }
            //update loss
            {
            //lock the loss mutex
            std::lock_guard<std::mutex> lock(_loss_mutex);
            //update loss
            this->_loss -= (_learning_rate * _dloss).sum();

            }
            //update the output
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //update output
            _y = _y * _Why + _by;
            //activate gradients
            softmax(_y);
            }


            //update hidden state
            _hprev =  _Whh * _hprev + _Wxh * _x + _bh;
            //update memory state
            _cprev =    _cprev * _Wgh + _Wfh * _x + _Wih * _hprev + _Wgh * _cprev + _bh; 
            //done
 
        }
        //backward batch threaded function

        //backpropagation through time
        void bptt() {
            //initialize gradients
            Matrix dWxh(_input_size,_hidden_size);
            Matrix dWhh(_hidden_size,_hidden_size);
            Matrix dWhy(_hidden_size,_output_size);
            Matrix dbh(1,_hidden_size);
            Matrix dby(1,_output_size);
            //initialize hidden state
            Matrix dhnext(1,_hidden_size);
            //initialize memory state
            Matrix dcnext(1,_hidden_size);
            //initialize input
            Matrix dx(1,_input_size);
            //initialize output
            Matrix dy(1,_output_size);
            //initialize target
            Matrix dt(1,_output_size);
            //initialize loss
            real_t loss = 1e-7;

            //initialize number of iterations
            size_t n = 0;
            //use multiple threads
            std::vector<std::thread> threads;
            size_t quanta = _num_batches/real_t(_num_threads);
            if (quanta < 1) quanta = 1;
            //divide the data into batches
            for(size_t i=0; i<_num_batches; ++i) {
                //slice X into quantized slices
                size_t start = i*quanta;
                size_t end = (i+1)*quanta;
                matrix<T> Xq = _x.slice(start,end);
                matrix<T> Yq = _target.slice(start,end);
                threads.push_back(std::thread(&LSTM::backward_batch,this,Xq,Yq,i));
                //slice Y into quantized slices
             }
            //join threads
            for(auto& t: threads) {
                t.join();
            }
            //update output
            _y = _y.slice(0,1);

        }   
        //backward batch threaded function
        void backward_batch(const Matrix& X, const Matrix& Y, size_t batch) {
            //initialize gradients
            size_t nn = batch;
            Matrix dWxh(_input_size,_hidden_size);
            Matrix dWhh(_hidden_size,_hidden_size);
            Matrix dWhy(_hidden_size,_output_size);
            Matrix dbh(1,_hidden_size);
            Matrix dby(1,_output_size);
            //initialize hidden state
            Matrix dhnext(1,_hidden_size);
            //initialize memory state
            Matrix dcnext(1,_hidden_size);
            //initialize input
            Matrix dx(1,_input_size);
            //initialize output
            Matrix dy(1,_output_size);
            //initialize target
            Matrix dt(1,_output_size);
            //initialize loss
            //real_t loss = 0;
            //initialize number of iterations
             //use multiple threads
            std::vector<std::thread> threads;
            size_t quanta = _num_batches/real_t(_num_threads);
            if (quanta < 1) quanta = 1;
            //divide the data into batches
            for(size_t i=0; i<_num_batches; ++i) {
                //slice X into quantized slices
                size_t start = i*quanta;
                size_t end = (i+1)*quanta;
                matrix<T> Xq = X.slice(start,end);
                matrix<T> Yq = Y.slice(start,end);
                threads.push_back(std::thread(&LSTM::backward_batch,this,Xq,Yq,i));
                nn--;
                //slice Y into quantized slices
             }
            //join threads
            for(auto& t: threads) {
                t.join();
            }
            
            

        }   
        Matrix predict(const Matrix& X) {

           auto sigmoid = activations[0];
           auto tanh = activations[1];
           auto relu = activations[2];
           auto softmax = activations[3];

            //lock the input mutex
            std::lock_guard<std::mutex> lock(_input_mutex);
            //copy the input
            _x = X;
            //update weights and biases thread safe:
            //weights:
            {
            //lock the general mutex for weights
            std::lock_guard<std::mutex> lock(_mutex);
            //activate gradients
            sigmoid(_Wxh);
            sigmoid(_Whh);
            sigmoid(_Why);
            sigmoid(_bh);
            sigmoid(_by);
            //update weights

            _Wxh -= _learning_rate * _dWxh;
            _Whh -= _learning_rate * _dWhh;
            _Why -= _learning_rate * _dWhy;
            _bh -= _learning_rate * _dbh;
            _by -= _learning_rate * _dby;

            
            }    
            //lock gradients mutex
            {
            //lock the mutex
            std::lock_guard<std::mutex> lock(_grad_mutex);
            //activate gradients
            sigmoid(_Wfh);
            sigmoid(_Wih);

            sigmoid(_Wgh);
            sigmoid(_Woh);
            //update weights

            _Wfh -= _learning_rate * _dWfh;
            _Wih -= _learning_rate * _dWih;

            _Wgh -= _learning_rate * _dWgh;
            _Woh -= _learning_rate * _dWoh;

            }

            //update biases
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //activate gradients
            sigmoid(_Wfh);
            sigmoid(_Wih);

            sigmoid(_Wgh);
            sigmoid(_Woh);
            //update weights

            _Wfh -= _learning_rate * _dWfh;
            _Wih -= _learning_rate * _dWih;

            _Wgh -= _learning_rate * _dWgh;
            _Woh -= _learning_rate * _dWoh;

            }

            //predict
            sigmoid(_Wxh);
            sigmoid(_Whh);
            sigmoid(_Why);
            sigmoid(_bh);
            sigmoid(_by);

            //update output
            _y = _y.slice(0,1);
            
            return _y;

        }
        
        //predict  
        Matrix predict(const Matrix& X, size_t batch) {
             predict_batch(X,batch);
                return _y;
            //done
        }
        //predict_batch threaded function
        void predict_batch(const Matrix& X, size_t batch )
        {
           auto sigmoid = activations[0];
           auto tanh = activations[1];
           auto relu = activations[2];
           auto softmax = activations[3];


            //lock the input mutex
            std::lock_guard<std::mutex> lock(_input_mutex);
            //copy the input
            _x = X;
            //update weights and biases thread safe:
            //weights:
            {
            //lock the general mutex for weights
            std::lock_guard<std::mutex> lock(_mutex);
            //activate gradients
            sigmoid(_Wxh);
            sigmoid(_Whh);
            sigmoid(_Why);
            sigmoid(_bh);
            sigmoid(_by);
            
            //update weights
            
            _Wxh -= _learning_rate * _dWxh;
            _Whh -= _learning_rate * _dWhh;
            _Why -= _learning_rate * _dWhy;
            _bh -= _learning_rate * _dbh;
            _by -= _learning_rate * _dby;
            
            
            }
            //lock gradients mutex
            {
            //lock the mutex
            std::lock_guard<std::mutex> lock(_grad_mutex);
             //update weights
            _Wxh -= _learning_rate * _dWxh;

            _Whh -= _learning_rate * _dWhh;
            _Why -= _learning_rate * _dWhy;
            _bh -= _learning_rate * _dbh;
            _by -= _learning_rate * _dby;
            }
            //lock the output mutex
            {
            //lock the output mutex
            std::lock_guard<std::mutex> lock(_output_mutex);
            //update biases
            _bf -= (_learning_rate * _dbf);
            _bi -= (_learning_rate * _dbi);
            _bg -= (_learning_rate * _dbg);
            _bo -= (_learning_rate * _dbo);
            //update conv biases
            _bcf -= (_learning_rate * _dbc);
            _bci -= (_learning_rate * _dbc);
            _bcg -= (_learning_rate * _dbc);
            _bco -= (_learning_rate * _dbc);
            }
            //lock the loss mutex
            {
            //lock the loss mutex
            std::lock_guard<std::mutex> lock(_loss_mutex);
            //update loss
            _loss -= _learning_rate * _dloss.sum();
            }
            //increment number of iterations
            _n++;
            //print loss
            if(_n%100==0) {
                std::cout << "batch: " << batch << " loss: " << _loss << std::endl;
            }
            
             //done

        }
        //predict threaded function

        //get the output
        Matrix& get_output() {
            return _y;
        }
        //get the loss
        real_t get_loss() {
            return _loss;
        }
        //get the loss
        real_t get_ce_loss() {
            return ce_loss_val;
        }
        //get the loss
        real_t get_kl_loss() {
            return kl_loss_val;
        }
        //get the number of iterations
        size_t get_iterations() {
            return _n;
        }
        //get the number of iterations
        size_t get_epochs() {
            return _epochs;
        }

    };//end class LSTM
}//end namespace provallo


#endif // __LSTM_H__